# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic 2-bit QAD (Bolt recipe) on ONE GPU, with the ternary bonsai27 memory recipe.

--phase cache   run the frozen teacher once over the training stream and the eval rows; store tokens +
top-K teacher log-probs (the teacher is never on the GPU during training);
--phase evalset teacher top-K of a held-out role eval set (.roles.jsonl) beside a finished cache (--eval-set N).
--phase train   student latents (fp16) fake-quantized per the layout (omk_qad.layout: llama.cpp's plan,
e.g. IQ2_M with experts fixed at IQ2_XXS), loss = Jaccard over top-K + rest (Bolt), updates
applied inside backward (omk_qad.fakequant), small fp tensors on AdamW;
--phase export  latents -> GGUF at the plan with the encoders training served (omk_qad.qad_export).
Without --load this exports the untrained base: the PTQ floor at the same plan.

--mode bf16 (omk_qad.recovery): no quantizer and no layout. The student's own bf16 weights in
--train-scope (e.g. gemma4) train against the cache's teacher, which may be another model of the same
tokenizer; export writes the source HF checkpoint back with only the trained tensors replaced (--out-hf).

python -m omk_qad.train --phase cache  --model M --cache C --data a.jsonl=3,b.parquet=1 --eval-data e.txt
python -m omk_qad.train --phase train  --model M --layout L.pt --cache C --out O [--load O/state.pt]
python -m omk_qad.train --phase export --model M --layout L.pt [--load O/state.pt] --gguf X.gguf
"""

import argparse
import copy
import hashlib
import json
import math
import netrc
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .fakequant import (
    Control,
    Latent,
    adafactor_state,
    build_segments,
    cosine_lr,
    install,
    load_adafactor_state,
)
from .losses import jaccard_topk_rest, kl_topk_rest, rkl_topk_rest
from .ptq import load_text_causal_lm
from .recovery import softcap, softcap_of
from .runctl import Control as RunControl
from .runctl import (
    RunLock,
    State,
    atomic_json,
    atomic_torch_save,
    check_identity,
    file_id,
    parse_allow,
)

LOG = [None]


def say(*x) -> None:
    """Print and append to the run log."""
    s = " ".join(str(v) for v in x)
    print(s, flush=True)
    if LOG[0]:
        LOG[0].write(s + "\n")
        LOG[0].flush()


# ===================================================================================================== data
def _texts(path: str, tok):
    """Documents of one source: (token ids, add_eos). .txt = one document, .jsonl/.parquet rows = documents."""
    p = Path(path)
    if p.suffix == ".txt":
        yield tok(p.read_text()).input_ids, True
        return

    def one(row):
        if row.get("messages"):
            text = tok.apply_chat_template(row["messages"], tokenize=False)
            return tok(text, add_special_tokens=False).input_ids, False
        return tok(row["text"]).input_ids, True

    if p.suffix == ".jsonl":
        with p.open() as f:
            for line in f:
                if line.strip():
                    yield one(json.loads(line))
    elif p.suffix == ".parquet":
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(p)
        cols = [c for c in ("text", "messages") if c in pf.schema_arrow.names]
        for b in pf.iter_batches(batch_size=256, columns=cols):
            yield from (one(r) for r in b.to_pylist())
    else:
        raise SystemExit(f"unsupported data file {p} (.txt | .jsonl | .parquet)")


def _packed(path: str, tok, seq: int, eos: int, passes: list):
    """Endless stream of ``seq``-token windows from one source (restarts at the end; ``passes`` counts)."""
    buf: list[int] = []
    while True:
        n_docs = 0
        for ids, add_eos in _texts(path, tok):
            n_docs += 1
            buf.extend(ids + ([eos] if add_eos and eos is not None else []))
            while len(buf) >= seq:
                yield torch.tensor(buf[:seq])
                buf = buf[seq:]
        passes.append(path)
        if n_docs == 0:
            raise SystemExit(f"{path}: no documents")


def _role_rows(path: str, seq: int, passes: list):
    """Endless stream of (ids, role) rows of a ``.roles.jsonl`` source (omk_qad.rollouts windows, ``seq`` long).

    The first pass is checked against the builder's ``.stats.json`` window count: a short (torn) file is refused.
    """
    side = Path(path).with_suffix(".stats.json")
    want = json.loads(side.read_text()).get("windows") if side.exists() else None
    first = True
    while True:
        n = 0
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if len(row["ids"]) != seq or len(row["role"]) != seq:
                    raise SystemExit(f"{path}: a row of {len(row['ids'])} tokens, the cache runs --seq {seq}")
                n += 1
                yield torch.tensor(row["ids"]), torch.tensor(row["role"], dtype=torch.int8)
        passes.append(path)
        if n == 0:
            raise SystemExit(f"{path}: no rows")
        if first and want is not None and n != want:
            raise SystemExit(f"{path}: {n} rows, its {side.name} says {want} (a torn or replaced file)")
        first = False


def _stream(path: str, tok, seq: int, eos: int, passes: list):
    """(ids, role or None) rows of one training source: role windows, or packed text (no role)."""
    if path.endswith(".roles.jsonl"):
        return _role_rows(path, seq, passes)
    return ((x, None) for x in _packed(path, tok, seq, eos, passes))


# ===================================================================================================== telemetry
# The API key IS the switch (omnimergekit's gepo_brevity / train_grpo_efficiency pattern): WANDB_API_KEY, else the
# api.wandb.ai entry of ~/.netrc. A key anywhere on the host means the run logs to W&B with nothing to remember on
# the command line. A key with no wandb package is an incomplete omk-modelopt env (wandb is pinned there): refused.
# Asking for W&B explicitly (any --wandb-* value) without a key or a reachable W&B refuses too; on the automatic
# path a W&B outage only warns, so it can never kill a training run.
WANDB_HOST = "api.wandb.ai"


def _wandb_key() -> tuple[str, str]:
    """(key, source), or ("", "") when no credential exists."""
    key = os.environ.get("WANDB_API_KEY", "").strip()
    if key:
        return key, "env:WANDB_API_KEY"
    try:
        auth = netrc.netrc().authenticators(WANDB_HOST)
        if auth and auth[2]:
            return auth[2].strip(), f"netrc:{WANDB_HOST}"
    except (OSError, netrc.NetrcParseError):
        pass
    return "", ""


class Telemetry:
    """One W&B run per training run dir; its id comes from the dir, so a relaunch (resume) reattaches to it."""

    def __init__(self, a, run: Path, config: dict, resume_step: int | None = None):
        """Start (or reattach to) the run, or stay off, per the rules above.

        ``resume_step``: the checkpoint's step. The W&B run is REWOUND to it (``resume_from``), so the rows a killed
        run logged after its last checkpoint are replaced by the resumed trajectory, not kept beside it; where rewind
        is unavailable (offline, older server) it reattaches and skips the steps it already holds (said once).
        """
        self.wb, self.skip_to, self.said_skip = None, -1, False
        explicit = bool(a.wandb_project or a.wandb_name or a.wandb_entity)
        if a.no_wandb or os.environ.get("WANDB_MODE") == "disabled":
            say(f"WANDB off ({'--no-wandb' if a.no_wandb else 'WANDB_MODE=disabled'})")
            return
        key, source = _wandb_key()
        if not key:
            msg = f"WANDB_API_KEY unset and no {WANDB_HOST} entry in ~/.netrc"
            if explicit:
                raise SystemExit(f"--wandb-* asks for W&B, but {msg}")
            say(f"WANDB off ({msg})")
            return
        try:
            import wandb
        except ImportError:
            raise SystemExit("a W&B key is present but the wandb package is not: the omk-modelopt env is incomplete "
                             "(omk_qad/conda_env/setup_conda_env.sh --update)") from None
        os.environ["WANDB_API_KEY"] = key
        project = a.wandb_project or os.environ.get("WANDB_PROJECT") or "omk-qad"
        stage = "stage1" if a.stage == 1 else f"stage2-{a.branch}"
        name = a.wandb_name or f"{Path(a.out).parent.name}-{Path(a.out).name}-{stage}"
        rid = hashlib.sha1(str(run.resolve()).encode()).hexdigest()[:16]
        entity = a.wandb_entity or os.environ.get("WANDB_ENTITY") or None
        if resume_step is not None:
            try:
                self.wb = wandb.init(project=project, entity=entity, name=name, config=config, dir=str(run),
                                     resume_from=f"{rid}?_step={resume_step}")
                say(f"WANDB rewound {rid} to step {resume_step} (later rows of the killed run are archived by W&B)")
            except Exception as e:
                say(f"WANDB rewind unavailable ({type(e).__name__}: {e}); reattaching instead")
                self.wb = None
                # the failed init leaves its stream registered under the run id: without a teardown the reattach
                # below is refused with "run ID ... is in use" (wandb 0.26.1, rewind in private preview; bm-889)
                wandb.teardown()
        try:
            if self.wb is None:
                self.wb = wandb.init(project=project, entity=entity, name=name, id=rid, resume="allow", config=config,
                                     dir=str(run))
                if resume_step is not None:
                    self.skip_to = int(getattr(self.wb, "step", 0) or 0) - 1  # rows W&B already holds
        except Exception as e:  # any W&B failure: refuse when asked for, else train without it
            if explicit:
                raise SystemExit(f"wandb.init failed: {type(e).__name__}: {e}") from None
            say(f"WANDB off -- wandb.init failed: {type(e).__name__}: {e}")
            return
        say(f"WANDB on key={source} project={project} name={name} id={rid} url={self.wb.url}")

    def log(self, data: dict, step: int) -> None:
        """Log numeric values at a training step (a failed upload is reported, never fatal)."""
        if self.wb is None:
            return
        if step <= self.skip_to:  # logged before the kill; W&B would drop it anyway
            if not self.said_skip:
                say(f"WANDB holds steps up to {self.skip_to} from before the restart (no rewind): those rows are "
                    f"the killed run's, the resumed run logs from step {self.skip_to + 1}")
                self.said_skip = True
            return
        try:
            self.wb.log({k: v for k, v in data.items() if isinstance(v, int | float)}, step=step)
        except Exception as e:  # telemetry must not stop training
            say(f"WANDB log failed at step {step}: {type(e).__name__}: {e}")

    def eval(self, r: dict, step: int) -> None:
        """One evaluate() result: overall metrics plus every eval family."""
        flat = {f"eval/{k}": v for k, v in r.items() if k not in ("families", "sets", "tag")}
        for fam, m in (r.get("families") or {}).items():
            flat.update({f"eval/{fam}/{k}": v for k, v in m.items()})
        for nm, s in (r.get("sets") or {}).items():  # --eval-set: eval/<set>/<metric>, eval/<set>/<family>/<metric>
            flat.update({f"eval/{nm}/{k}": v for k, v in s.items() if k != "families"})
            for fam, m in (s.get("families") or {}).items():
                flat.update({f"eval/{nm}/{fam}/{k}": v for k, v in m.items()})
        self.log(flat, step)

    def finish(self) -> None:
        """Close the run (a paused run is finished too; its relaunch resumes the same id)."""
        if self.wb is not None:
            self.wb.finish()
            self.wb = None


# Cache settings that decide what the cached targets ARE (n_seq is not one: raising it extends the cache).
CACHE_IDENTITY = ("topk", "seq", "shard", "seed", "n_eval", "mode")


def cache_identity(a) -> dict:
    """Identity of a teacher cache: the teacher (GGUF or HF weights), tokenizer, data and the target settings."""
    ident = {k: getattr(a, k, None) for k in CACHE_IDENTITY}
    ident["teacher"] = {"gguf": file_id(a.teacher_gguf)} if a.teacher_gguf else _model_id(a.model)
    ident["tokenizer"] = _model_id(a.model) if a.teacher_gguf else "teacher"
    ident["data"] = [[q.split("=")[0], q.split("=")[1] if "=" in q else "1", file_id(q.split("=")[0])]
                     for q in a.data.split(",") if q]
    ident["eval_data"] = [[q, file_id(q)] for q in a.eval_data.split(",") if q]
    return ident


def _ident_sha(ident: dict) -> str:
    return hashlib.sha256(json.dumps(ident, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _x_sha(x: torch.Tensor) -> str:
    return hashlib.sha256(x.to(torch.int32).contiguous().numpy().tobytes()).hexdigest()[:16]


@torch.no_grad()
def head_topk(model, h, x, cap, k: int, chunk: int):
    """Top-``k`` soft-capped log-probs of next tokens over ``h`` [B, T, D], ``chunk`` positions at a time.

    The whole head at once is B x T x vocab in fp32 (4 x 16384 x 262144 = 68 GB for Gemma-4 at --seq 16384,
    bm-889); per chunk it is chunk x vocab. Returns int32 / fp16 [B, T-1, k] on the CPU and the summed NLL of ``x``.
    """
    head, ii, vv, nll = model.get_output_embeddings(), [], [], 0.0
    for c0 in range(0, h.shape[1] - 1, chunk):
        c1 = min(c0 + chunk, h.shape[1] - 1)
        lp = F.log_softmax(softcap(head(h[:, c0:c1]).float(), cap), -1)
        v, i = lp.topk(k, -1)
        nll -= lp.gather(-1, x[:, c0 + 1 : c1 + 1, None]).sum().item()
        ii.append(i.to(torch.int32).cpu())
        vv.append(v.to(torch.float16).cpu())
    return torch.cat(ii, 1), torch.cat(vv, 1), nll


def phase_cache(a) -> None:
    """Teacher top-K cache over the mixed training stream and the eval rows.

    One process per cache dir (``cache.lock``); ``identity.json`` (teacher, tokenizer, data, topk, seq, shard, seed)
    refuses a restart with other settings unless ``--allow-change``; every shard has a ``train_NNNN.json`` sidecar
    (identity hash, rows, sha of its token rows) checked before reuse; drain / pause between shards via
    ``<cache>/control`` or signals; ``state.json``; ``cache.json`` (atomic) only when complete.
    """
    os.makedirs(a.cache, exist_ok=True)
    LOG[0] = open(os.path.join(a.cache, "cache.log"), "a")
    say("args", json.dumps(vars(a)))
    lock = RunLock(Path(a.cache) / "cache.lock", f"cache {a.cache}")
    st = State(Path(a.cache) / "state.json", cache=a.cache, step="anchor" if getattr(a, "anchor", False) else "cache")
    ctl_run = RunControl(Path(a.cache) / "control", name=f"cache {Path(a.cache).name}", log=say)
    try:
        dtype = torch.float32 if a.device == "cpu" else torch.bfloat16
        if getattr(a, "anchor", False):
            anchor_pass(a, dtype, st, ctl_run)
        else:
            _phase_cache(a, dtype, st, ctl_run)
    except BaseException as e:
        st.refused_or_failed(e)
        raise
    finally:
        ctl_run.restore()
        lock.release()


def _phase_cache(a, dtype, st: State, control: RunControl) -> None:
    cache = Path(a.cache)
    ident = cache_identity(a)
    if (cache / "cache.json").exists() and not (cache / "identity.json").exists():  # a cache from before identity
        old = json.loads((cache / "cache.json").read_text()).get("args", {})
        bad = {k: (old.get(k), getattr(a, k, None)) for k in (*CACHE_IDENTITY, "model", "teacher_gguf", "data")
               if k in old and old.get(k) != getattr(a, k, None)}
        if bad:
            raise SystemExit(f"RUN_REFUSED cache {cache}: cache.json was made with {bad}")
        say("identity.json created for an existing cache (its args matched)")
    check_identity(cache / "identity.json", ident, parse_allow(getattr(a, "allow_change", [])),
                   what=f"cache {cache}", log=say)
    isha = _ident_sha(ident)
    st.update(status="running", identity=isha, n_seq=a.n_seq)
    imat = {"collector": None, "seqs": 0}
    if a.teacher_gguf:  # libllama teacher (omk_qad.gguf_teacher); --model gives the tokenizer only
        from transformers import AutoTokenizer

        from .gguf_teacher import GGUFTeacher

        if a.imatrix_seqs:
            say("--teacher-gguf: no imatrix (it needs the torch teacher's activations); --imatrix-seqs set to 0")
            a.imatrix_seqs = 0
        tok = AutoTokenizer.from_pretrained(a.model)
        gguf = GGUFTeacher(a.teacher_gguf, a.seq, a.topk, len(tok), a.teacher_gguf_bin, a.teacher_ngl,
                           a.teacher_ubatch, log=say, device=a.device)
        import atexit

        atexit.register(gguf.close)  # the llama_topk child never outlives a failed or drained cache pass

        def topk_of(x, want=None):
            return gguf(x, want)
    else:
        teacher, tok = load_text_causal_lm(a.model, a.device, dtype)
        teacher.requires_grad_(False)
        cap = softcap_of(teacher)  # Gemma-4: the teacher's distribution is the soft-capped one
        from .mtp import attach, checkpoint_mtp_keys, load_weights, mtp_layers

        if a.imatrix_seqs and mtp_layers(teacher.config) and checkpoint_mtp_keys(Path(a.model)):
            attach(teacher)  # the MTP head's inputs go into the imatrix too (no MTP-head imatrix gap)
            load_weights(teacher, Path(a.model))

    @torch.no_grad()
    def topk_of_torch(x, want=None):
        x = x.to(a.device)
        h = teacher.get_decoder()(input_ids=x).last_hidden_state
        if imat["collector"] is not None and getattr(teacher, "mtp", None) is not None:
            teacher.mtp(teacher.get_input_embeddings()(x[:, 1:]), h[:, :-1], teacher.get_decoder().rotary_emb)
        i, v, nll = head_topk(teacher, h, x, cap, a.topk, a.loss_chunk)
        if imat["collector"] is not None:
            imat["seqs"] += x.shape[0]
            if imat["seqs"] >= a.imatrix_seqs:
                save_imatrix()
        return i, v, nll

    if not a.teacher_gguf:
        topk_of = topk_of_torch

    def save_imatrix():
        from .imatrix import save, save_folded

        col, imat["collector"] = imat["collector"], None
        col.close()
        if not imat["seqs"]:
            raise SystemExit("imatrix: no sequence reached the collector; refusing to save an empty imatrix")
        layout = torch.load(a.layout, weights_only=False) if a.layout else None
        datasets = [p for p, _ in srcs]
        paths = []
        if not (Path(a.cache) / "imatrix.pt").exists():  # never overwrite an imatrix (it is part of provenance)
            paths += save(col.state(), layout, Path(a.cache), datasets, a.seq, imat["seqs"], Path(a.llama_cpp))
        if col.fsums:
            paths.append(save_folded(col.folded_state(), Path(a.cache), datasets, a.seq, imat["seqs"]))
        say(f"IMATRIX from {imat['seqs']} x {a.seq} tokens ({len(col.sums)} weights) -> {[str(q) for q in paths]}")

    passes: list = []
    fams = [q for q in a.eval_data.split(",") if q]  # one eval family per file (name = file stem)
    names = [Path(q).stem for q in fams]
    if len(set(names)) != len(names):
        raise SystemExit(f"eval families need distinct file stems: {names}")
    ex, fam = [], []
    for k, q in enumerate(fams):
        stream = _packed(q, tok, a.seq, tok.eos_token_id, passes)
        ex += [next(stream) for _ in range(a.n_eval)]
        fam += [k] * a.n_eval
    ex = torch.stack(ex)
    ev_p = cache / "eval.pt"
    old_ev = torch.load(ev_p) if ev_p.exists() else None
    same_rows = old_ev is not None and torch.equal(old_ev["x"], ex.to(torch.int32))
    if same_rows and old_ev.get("identity") in (None, isha):
        if old_ev.get("identity") is None:  # an eval.pt from before identity: same rows, so tag it
            atomic_torch_save({**old_ev, "identity": isha}, ev_p)
        say(f"eval.pt kept: same identity and the same {len(ex)} rows (finished runs keep their reference)")
    else:
        if old_ev is not None:  # never silently replace the reference of runs that already used it
            raise SystemExit(f"RUN_REFUSED cache {cache}: eval.pt holds other rows or another identity; move it "
                             "aside (or use a new cache dir) to rebuild it")
        parts = [topk_of(ex[i : i + 4]) for i in range(0, len(ex), 4)]
        ev = {"x": ex.to(torch.int32), "idx": torch.cat([q[0] for q in parts]),
              "val": torch.cat([q[1] for q in parts]),
              "teacher_ppl": math.exp(sum(q[2] for q in parts) / (len(ex) * (a.seq - 1))),
              "family": torch.tensor(fam, dtype=torch.int32), "families": names, "identity": isha}
        atomic_torch_save(ev, ev_p)
        say(f"eval: {len(ex)} x {a.seq} tokens ({a.n_eval} per family {names}); teacher ppl "
            f"{ev['teacher_ppl']:.3f}; top-{a.topk} mass {ev['val'].float().exp().sum(-1).mean().item():.4f}")

    srcs = [(q.split("=")[0], float(q.split("=")[1]) if "=" in q else 1.0) for q in a.data.split(",") if q]
    folded, block = (None, None)
    fpath = Path(a.cache) / "imatrix.folded.pt"
    if a.layout:
        from .imatrix import folded_candidates

        folded, block = folded_candidates(torch.load(a.layout, weights_only=False))
        if fpath.exists():  # computed under another Hadamard block (e.g. before the model-wide block): set aside
            old = {e.get("block") for e in torch.load(fpath, weights_only=False)["entries"].values()}
            if old != {block}:
                stale = fpath.with_name(f"imatrix.folded.block{'-'.join(map(str, sorted(old)))}.stale.pt")
                fpath.replace(stale)
                say(f"imatrix.folded.pt had blocks {sorted(old)}, this layout folds with {block}: kept as {stale.name}")
    want_folded = bool(folded) and not fpath.exists()
    if a.imatrix_seqs and (not (Path(a.cache) / "imatrix.pt").exists() or want_folded):
        from .imatrix import Collector

        imat["collector"] = Collector(teacher, folded=folded if want_folded else None, block=block)
        if not a.layout:
            say("WARNING: no --layout: imatrix.pt only (no imatrix.gguf, no folded statistic for ternary students)")
    streams = [_stream(p, tok, a.seq, tok.eos_token_id, passes) for p, _ in srcs]
    rng, n_src = random.Random(a.seed), [0] * len(srcs)
    t0 = time.time()
    for s0 in range(0, a.n_seq, a.shard):
        fn = os.path.join(a.cache, f"train_{s0 // a.shard:04d}.pt")
        picks = rng.choices(range(len(srcs)), [w for _, w in srcs], k=min(a.shard, a.n_seq - s0))
        rows = [next(streams[k]) for k in picks]  # always drawn, so a resume stays aligned
        x = torch.stack([q[0] for q in rows])
        role = None
        if any(q[1] is not None for q in rows):  # role data: text rows are teacher-trained everywhere
            role = torch.stack([q[1] if q[1] is not None else torch.ones(a.seq, dtype=torch.int8) for q in rows])
        for k in picks:
            n_src[k] += 1
        side = Path(fn).with_suffix(".json")
        want_side = {"identity": isha, "rows": len(x), "x_sha": _x_sha(x), "s0": s0}
        if os.path.exists(fn) and not side.exists():  # a shard from before sidecars: verify its rows once
            old_x = torch.load(fn)["x"]
            if old_x.shape[0] == len(x) and torch.equal(old_x, x.to(torch.int32)):
                atomic_json(side, want_side)
            else:
                raise SystemExit(f"RUN_REFUSED cache {cache}: {fn} holds other rows than this stream draws")
        if os.path.exists(fn):
            have = json.loads(side.read_text())
            if have.get("identity") != isha or have.get("x_sha") != want_side["x_sha"]:
                raise SystemExit(f"RUN_REFUSED cache {cache}: {fn} was made with another identity or other rows "
                                 f"({have} vs {want_side})")
            if have.get("rows") == len(x):
                # a resumed cache still owes its imatrix: run the teacher on the cached rows for the imatrix only
                for i in range(0, len(x), a.cache_bs):
                    if imat["collector"] is None:
                        break
                    topk_of(x[i : i + a.cache_bs])
                continue
        if os.path.exists(fn):  # a short last shard from a smaller --n-seq: recompute it, keep the old one
            os.replace(fn, f"{fn}.rows{have.get('rows')}.old")
            say(f"{fn}: had {have.get('rows')} rows, this run needs {len(x)}: recomputed (old kept as .old)")
        want = None if role is None else role[:, 1:] > 0  # a GGUF teacher skips unscored positions
        parts = [topk_of(x[i : i + a.cache_bs], None if want is None else want[i : i + a.cache_bs])
                 for i in range(0, len(x), a.cache_bs)]
        atomic_torch_save({"x": x.to(torch.int32), "idx": torch.cat([q[0] for q in parts]),
                           "val": torch.cat([q[1] for q in parts]), **({"role": role} if role is not None else {})},
                          fn)
        atomic_json(side, want_side)  # written after the shard: a shard without its sidecar is re-verified
        say(f"cached {s0 + len(x)}/{a.n_seq} sequences [{time.time() - t0:.0f}s] rows per source {n_src}")
        st.update(cached=s0 + len(x))
        last = s0 + len(x) >= a.n_seq  # a drain on the last shard still finishes (imatrix + cache.json)
        if not last and control.boundary(on_pause=lambda: st.update(status="paused")) == "drain":
            st.update(status="drained")
            say(f"CACHE_DRAINED after {s0 + len(x)}/{a.n_seq} sequences; relaunch the same command to continue")
            if a.teacher_gguf:
                gguf.close()
            return
        st.update(status="running")
    if imat["collector"] is not None:  # fewer sequences than --imatrix-seqs
        save_imatrix()
    meta = {"args": vars(a), "n_seq": a.n_seq, "shard": a.shard, "topk": a.topk, "seq": a.seq, "identity": isha,
            "rows_per_source": dict(zip([p for p, _ in srcs], n_src)), "source_passes": passes}
    if a.teacher_gguf:
        gguf.close()
        side = Path(a.teacher_gguf + ".sha256")  # the download's verified digest (hashing 60 GB here is not)
        meta["teacher"] = {"gguf": a.teacher_gguf, "sha256": side.read_text().split()[0] if side.exists() else None}
    old = json.loads((cache / "cache.json").read_text()) if (cache / "cache.json").exists() else {}
    for k in ("anchor_model",):  # an anchor pass of the same cache survives an extension of it
        if k in old:
            meta[k] = old[k]
    atomic_json(cache / "cache.json", meta)
    st.update(status="done", cached=a.n_seq)
    say("CACHE_DONE")


def cache_ref(cache) -> str | dict:
    """What identifies a finished cache to its eval sets: its identity hash, or (before identity) its cache.json."""
    meta = json.loads((Path(cache) / "cache.json").read_text())
    return meta["identity"] if "identity" in meta else file_id(Path(cache) / "cache.json")


def evalset_path(cache, name: str) -> Path:
    """Where ``--phase evalset`` keeps the named eval set of a cache."""
    return Path(cache) / f"evalset_{name}.pt"


def phase_evalset(a) -> None:
    """Teacher top-K of a held-out role eval set, kept beside a finished cache as ``evalset_<name>.pt``.

    The cache itself is not touched (its identity names its own --eval-data; a new family there would refuse every
    shard). The set comes from the cache's teacher, tokenizer, seq and topk (checked against its identity.json);
    of each ``.roles.jsonl`` file ``--n-eval`` rows spread evenly over it are kept (none twice), and only
    teacher-scored positions are computed and scored.
    Use rollouts the cache never trained on.
    """
    cache = Path(a.cache)
    fams = [q for q in a.eval_data.split(",") if q]
    if not a.eval_set or not fams or not all(q.endswith(".roles.jsonl") for q in fams):
        raise SystemExit("--phase evalset needs --eval-set NAME and --eval-data f.roles.jsonl[,g.roles.jsonl]")
    names = [Path(q).name.removesuffix(".roles.jsonl") for q in fams]
    if len(set(names)) != len(names):
        raise SystemExit(f"eval families need distinct names: {names}")
    if not (cache / "cache.json").exists():
        raise SystemExit(f"{cache}: no finished cache (cache.json) to put an eval set beside")
    if (cache / "identity.json").exists():
        have = json.loads((cache / "identity.json").read_text())["identity"]
        mine = json.loads(json.dumps(cache_identity(a), default=str))
        keys = ("teacher", "tokenizer", "seq", "topk")
    else:  # a cache from before identity.json: its recorded args
        have, mine = json.loads((cache / "cache.json").read_text())["args"], vars(a)
        keys = ("model", "teacher_gguf", "seq", "topk")
    bad = {k: (have.get(k), mine.get(k)) for k in keys if (have.get(k) or "") != (mine.get(k) or "")}  # unset = ""
    if bad:
        raise SystemExit(f"RUN_REFUSED evalset: cache {cache} was made with other settings {bad}")
    ex, rs, fam = [], [], []
    for k, q in enumerate(fams):
        passes: list = []
        stream, rows = _role_rows(q, a.seq, passes), []
        while True:  # one whole pass (its window count is checked against the .stats.json); no row is used twice
            row = next(stream)
            if passes:
                break
            rows.append(row)
        # --n-eval rows spread evenly over the file (rows come grouped by rollout: the first n would be one or two)
        pick = range(len(rows)) if len(rows) <= a.n_eval else [i * len(rows) // a.n_eval for i in range(a.n_eval)]
        for i in pick:
            ex.append(rows[i][0])
            rs.append(rows[i][1])
        fam += [k] * len(pick)
        say(f"evalset {a.eval_set}: {names[k]} {len(pick)} of {len(rows)} rows")
    x, role = torch.stack(ex).to(torch.int32), torch.stack(rs)
    want = teacher_scored(role[:, 1:])
    ident = {"cache": cache_ref(cache), "seq": a.seq, "topk": a.topk,
             "eval_data": [[q, file_id(q)] for q in fams]}
    out = evalset_path(cache, a.eval_set)
    if out.exists():
        old = torch.load(out)
        if old.get("identity") == ident and torch.equal(old["x"], x):
            say(f"{out} kept: same rows and identity")
            return
        raise SystemExit(f"RUN_REFUSED evalset: {out} holds other rows or another identity; use another --eval-set")
    topk_of, close, gguf = _eval_teacher(a)
    try:
        parts = [topk_of(x[i : i + 4], want[i : i + 4]) for i in range(0, len(x), 4)]
    finally:
        close()
    n = int(want.sum())
    ev = {"x": x, "idx": torch.cat([q[0] for q in parts]), "val": torch.cat([q[1] for q in parts]), "role": role,
          # a GGUF teacher sums its NLL over the wanted positions only; the torch head sums all of them
          "teacher_ppl": math.exp(sum(q[2] for q in parts) / n) if gguf else None,
          "family": torch.tensor(fam, dtype=torch.int32), "families": names, "identity": ident}
    atomic_torch_save(ev, out)
    say(f"EVALSET_DONE {out}: {len(x)} x {a.seq} tokens, {n} teacher-scored ({n / want.numel():.1%}); teacher ppl "
        f"{ev['teacher_ppl'] if ev['teacher_ppl'] is None else round(ev['teacher_ppl'], 3)}")


def _eval_teacher(a):
    """``(topk(x, want) -> (idx, val, nll), close, is_gguf)``: the cache's teacher without the cache pass's hooks."""
    if a.teacher_gguf:
        from transformers import AutoTokenizer

        from .gguf_teacher import GGUFTeacher

        tok = AutoTokenizer.from_pretrained(a.model)
        g = GGUFTeacher(a.teacher_gguf, a.seq, a.topk, len(tok), a.teacher_gguf_bin, a.teacher_ngl, a.teacher_ubatch,
                        log=say, device=a.device)
        return g, g.close, True
    teacher, _ = load_text_causal_lm(a.model, a.device, torch.float32 if a.device == "cpu" else torch.bfloat16)
    teacher.requires_grad_(False)
    cap = softcap_of(teacher)

    @torch.no_grad()
    def topk(x, want=None):
        x = x.to(a.device)
        return head_topk(teacher, teacher.get_decoder()(input_ids=x).last_hidden_state, x, cap, a.topk, a.loss_chunk)

    return topk, lambda: None, False


def anchor_pass(a, dtype, st: State, control: RunControl) -> None:
    """Add the ANCHOR model's top-K (``aidx``/``aval``) to every cached shard holding anchor-role positions.

    Runs over the very rows the teacher pass cached (read back from the shards), so the two targets can never
    misalign. ``--model`` is the anchor (bf16 recovery: the original student, whose coding is kept).
    """
    from .rollouts import ANCHOR

    meta_p = Path(a.cache) / "cache.json"
    if not meta_p.exists():
        raise SystemExit(f"--anchor needs a finished teacher cache in {a.cache}")
    meta = json.loads(meta_p.read_text())
    if meta.get("anchor_model") not in (None, a.model):
        raise SystemExit(f"{a.cache} already holds anchor targets of {meta['anchor_model']}")
    # the anchor model is recorded BEFORE the first shard, so an interrupted pass cannot be finished by another
    check_identity(Path(a.cache) / "anchor_identity.json", {"anchor_model": a.model, "id": _model_id(a.model)},
                   parse_allow(getattr(a, "allow_change", [])), what=f"anchor {a.cache}", log=say)
    st.update(status="running")
    model, _ = load_text_causal_lm(a.model, a.device, dtype)
    model.requires_grad_(False)
    cap = softcap_of(model)
    n_sh, done = math.ceil(meta["n_seq"] / meta["shard"]), 0
    for s in range(n_sh):
        fn = Path(a.cache) / f"train_{s:04d}.pt"
        sh = torch.load(fn)
        if "role" not in sh:
            raise SystemExit(f"{fn}: no role data (the anchor targets only positions of role {ANCHOR})")
        if "aidx" in sh and sh.get("anchor_model", a.model) != a.model:
            raise SystemExit(f"{fn}: holds anchor targets of {sh['anchor_model']}, not {a.model}")
        if "aidx" in sh or not (sh["role"] == ANCHOR).any():
            continue
        ai, av = [], []
        with torch.no_grad():
            for i in range(0, len(sh["x"]), a.cache_bs):
                x = sh["x"][i : i + a.cache_bs].long().to(a.device)
                h = model.get_decoder()(input_ids=x).last_hidden_state
                k, v, _ = head_topk(model, h, x, cap, meta["topk"], a.loss_chunk)
                ai.append(k)
                av.append(v)
        sh.update(aidx=torch.cat(ai), aval=torch.cat(av), anchor_model=a.model)
        atomic_torch_save(sh, fn)
        done += 1
        say(f"anchor {fn.name}: {int((sh['role'] == ANCHOR).sum())} anchor positions")
        st.update(anchor_shard=s + 1, of=n_sh)
        if s + 1 < n_sh and control.boundary(on_pause=lambda: st.update(status="paused")) == "drain":
            st.update(status="drained")
            say(f"ANCHOR_DRAINED after shard {s + 1}/{n_sh}; relaunch the same command to continue")
            return
        st.update(status="running")
    meta = json.loads(meta_p.read_text())  # re-read: never write back a stale copy of the master metadata
    meta["anchor_model"] = a.model
    atomic_json(meta_p, meta)
    st.update(status="done")
    say(f"ANCHOR_DONE {done} shards from {a.model}")


# ===================================================================================================== student
def build_student(a, layout: dict):
    """Student model with latents installed; returns (model, latents, fp_params, ctl, head_name)."""
    cuda = a.device != "cpu"
    student, _ = load_text_causal_lm(a.model, a.device, torch.bfloat16 if cuda else torch.float32)
    if layout.get("mtp"):  # the layout carries the MTP head: rebuild it and load the checkpoint's mtp.* tensors
        from .mtp import attach, load_weights

        attach(student)
        say(f"MTP head: {load_weights(student, Path(a.model))} tensors loaded from {a.model} (mode {a.mtp})")
    named = dict(student.named_parameters())
    if {n: list(p.shape) for n, p in named.items()} != layout["params"]:
        raise SystemExit("student parameters do not match the layout (names or shapes); an MTP-less layout needs "
                         "a checkpoint without MTP or a layout built with --no-mtp")
    from .fakequant import Codec

    ctl = Control(compute_dtype=torch.bfloat16 if cuda else torch.float32, beta2=a.beta2, clip=a.clip,
                  ggml_refresh=a.ggml_refresh, refresh=getattr(a, "refresh", 1),
                  dummy=torch.zeros((), requires_grad=True, device=a.device),
                  hadamard_block=layout.get("hadamard_block"))
    codec = Codec(Path(layout["llama_cpp"]), ternary_method=getattr(a, "ternary_method", "top:86"))
    latent_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[a.latent_dtype]
    export_only = set(a.export_only.split(",")) if a.export_only else set()
    head_param = student.get_output_embeddings().weight
    head_name = next(n for n, p in named.items() if p is head_param)
    latents, fp_params = {}, []
    for n, p in named.items():
        segs = build_segments(layout, n, train_types=layout["classes"].get(n) not in export_only)
        if segs or (p.ndim >= 2 and p.numel() >= a.latent_min):
            latents[n] = Latent(n, p, segs, ctl, codec, latent_dtype)
        else:
            p.data = p.data.float()
            p.requires_grad_(True)
            fp_params.append(p)
    if getattr(a, "imatrix", "") and a.imatrix != "none":
        from .imatrix import load as load_imatrix
        from .imatrix import mean_sq

        rec = load_imatrix(Path(a.imatrix))
        missing = [n for n, lat in latents.items() if lat.segments and n not in rec["entries"]]
        for n, lat in latents.items():
            if lat.segments and n in rec["entries"]:
                lat.set_imatrix(mean_sq(rec["entries"][n]).to(a.device))
        folded = [n for n, lat in latents.items() if lat.basis is not None]
        fpath = Path(a.imatrix).with_name("imatrix.folded.pt")
        weighted_f = 0
        if folded and fpath.exists():
            rec_f = load_imatrix(fpath)
            if rec_f.get("basis") != "seeded":
                raise SystemExit(f"{fpath}: basis {rec_f.get('basis')!r}, this trainer folds with seeded signs")
            for n in folded:
                e = rec_f["entries"].get(n)
                if e is None:
                    continue
                if e["block"] != latents[n].basis.block:
                    raise SystemExit(f"{fpath}: {n} block {e['block']} != the student's {latents[n].basis.block}")
                latents[n].set_imatrix(mean_sq(e).to(a.device), folded=True)
                weighted_f += 1
        say(f"imatrix {a.imatrix}: {len(latents) - len(missing) - len(folded)} model-basis latents weighted; "
            f"without data: {missing[:6]}; ternary: {weighted_f}/{len(folded)} weighted from "
            f"{fpath.name if weighted_f else '- (no folded statistic)'}")
    tied = head_param is student.get_input_embeddings().weight
    install(student, latents, skip=set() if tied else {head_name})
    a.softcap = softcap_of(student)
    student.config.use_cache = False
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    n_q = sum(lat.param.numel() for lat in latents.values() if lat.segments)
    n_t = sum(lat.param.numel() for lat in latents.values() if lat.basis is not None)
    say(f"student: {len(latents)} latents ({sum(lat.param.numel() for lat in latents.values()) / 1e9:.3f}B as "
        f"{a.latent_dtype}; {n_q / 1e9:.3f}B fake-quantized), fp params {sum(p.numel() for p in fp_params) / 1e6:.2f}M,"
        f" head {head_name} ({'tied' if tied else 'untied'}); ternary {n_t / 1e9:.3f}B"
        f" ({getattr(a, 'ternary_method', 'top:86') if n_t else '-'})")
    return student, latents, fp_params, ctl, head_name


def hidden_of(student, x, cuda: bool):
    """Final hidden states of the decoder body (the head is applied per chunk by the caller)."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
        return student.get_decoder()(input_ids=x).last_hidden_state


def has_mtp(student) -> bool:
    """Whether the student carries an MTP head (omk_qad.mtp)."""
    return getattr(student, "mtp", None) is not None


def mtp_hidden(student, x, h, cuda: bool):
    """MTP output z [B, T-1, H] from trunk h (detached: MTP never pushes the trunk) and x_{t+1} embeddings."""
    with torch.no_grad():  # the embedding is not trained through the MTP loss either (no hook use counted)
        e = student.get_input_embeddings()(x[:, 1:])
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
        return student.mtp(e, h[:, :-1].detach(), student.get_decoder().rotary_emb)


def token_terms(logits, idx, val, x_next):
    """Per-token Jaccard, KL (top-K + rest), CE and top-1 agreement."""
    logits = logits.float()
    return (jaccard_topk_rest(logits, idx.long(), val.float()), kl_topk_rest(logits, idx.long(), val.float()),
            F.cross_entropy(logits, x_next, reduction="none"), (logits.argmax(-1) == idx[:, 0].long()).float())


DISTILL = {"jaccard": jaccard_topk_rest, "kl": kl_topk_rest, "rkl": rkl_topk_rest}  # --loss: per-token top-K + rest


def distill_loss(logits, idx, val, a) -> torch.Tensor:
    """The per-token distillation loss ``--loss`` names (jaccard | kl = KL(t||s) | rkl = KL(s||t))."""
    return DISTILL[a.loss](logits.float(), idx.long(), val.float())


def teacher_scored(role: torch.Tensor) -> torch.Tensor:
    """Positions whose loss is distillation from the teacher (roles 1 and 3; 2 = anchor, 0 = not trained)."""
    return (role == 1) | (role == 3)


@torch.no_grad()
def evaluate(student, head, ev, a, tag, extra=None, sets=None) -> dict:
    """Jaccard / KL(top-K) / top-1 / ppl of the student against the cached teacher on the eval rows.

    With an MTP head also ``*_mtp``: its prediction of x_{t+2} vs the teacher's distribution at t+1
    (``top1_mtp`` is the draft-acceptance proxy). ``sets`` (--eval-set): named role eval sets, scored only where
    training distils from the teacher, reported under ``sets`` beside the cache's own eval.
    """
    student.eval()
    res = {"tag": tag, **_score(student, head, ev, a)}
    if sets:
        res["sets"] = {nm: _score(student, head, s, a) for nm, s in sets.items()}
    student.train()
    res.update(extra or {})
    say("EVAL", json.dumps(res))
    return res


def _score(student, head, ev, a) -> dict:
    """The metrics of one eval set: every position, or with a ``role`` only the teacher-scored ones (+ ``scored``)."""
    cuda = a.device != "cpu"
    wh = head.served()
    mtp = has_mtp(student)
    tot = dict.fromkeys(("jac", "klk", "rkl", "nll", "top1", "jac_m", "klk_m", "top1_m"), 0.0)
    n = n_m = 0
    names = ev.get("families") or []
    fam = ev["family"].tolist() if "family" in ev else [0] * len(ev["x"])
    per = [dict.fromkeys(("jac", "klk", "rkl", "nll", "top1", "n"), 0.0) for _ in names]
    role = ev.get("role")

    def mask(i, t0, t1):  # None: every position counts
        return None if role is None else teacher_scored(role[i, t0:t1]).to(a.device)

    for i0 in range(0, len(ev["x"]), 4):
        x = ev["x"][i0 : i0 + 4].long().to(a.device)
        h = hidden_of(student, x, cuda)
        z = mtp_hidden(student, x, h, cuda) if mtp else None
        for i in range(len(x)):
            for c0 in range(0, x.shape[1] - 1, a.loss_chunk):
                c1 = min(c0 + a.loss_chunk, x.shape[1] - 1)
                m = mask(i0 + i, c0 + 1, c1 + 1)  # the role of the predicted token, as in training
                if m is not None and not m.any():
                    continue
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
                    lg = softcap(F.linear(h[i, c0:c1], wh), getattr(a, "softcap", None))
                ei, ev_ = ev["idx"][i0 + i, c0:c1].to(a.device), ev["val"][i0 + i, c0:c1].to(a.device)
                j, k, ce, t1 = token_terms(lg, ei, ev_, x[i, c0 + 1 : c1 + 1])
                rk = rkl_topk_rest(lg.float(), ei.long(), ev_.float())
                if m is not None:
                    j, k, ce, t1, rk = j[m], k[m], ce[m], t1[m], rk[m]
                terms = {"jac": j.sum().item(), "klk": k.sum().item(), "rkl": rk.sum().item(), "nll": ce.sum().item(),
                         "top1": t1.sum().item()}
                cnt = c1 - c0 if m is None else int(m.sum())
                for key, v in terms.items():
                    tot[key] += v
                    if per:
                        per[fam[i0 + i]][key] += v
                if per:
                    per[fam[i0 + i]]["n"] += cnt
                n += cnt
            if mtp:  # z_t (t <= T-3) predicts x_{t+2}; teacher distribution at t+1
                for c0 in range(0, x.shape[1] - 2, a.loss_chunk):
                    c1 = min(c0 + a.loss_chunk, x.shape[1] - 2)
                    m = mask(i0 + i, c0 + 2, c1 + 2)
                    if m is not None and not m.any():
                        continue
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
                        lg = softcap(F.linear(z[i, c0:c1], wh), getattr(a, "softcap", None))
                    j, k, _, t1 = token_terms(lg, ev["idx"][i0 + i, c0 + 1 : c1 + 1].to(a.device),
                                              ev["val"][i0 + i, c0 + 1 : c1 + 1].to(a.device), x[i, c0 + 2 : c1 + 2])
                    if m is not None:
                        j, k, t1 = j[m], k[m], t1[m]
                    tot["jac_m"] += j.sum().item()
                    tot["klk_m"] += k.sum().item()
                    tot["top1_m"] += t1.sum().item()
                    n_m += c1 - c0 if m is None else int(m.sum())
    if not n:
        raise SystemExit("eval set without a teacher-scored position (roles 1 / 3)")
    res = {"jaccard": tot["jac"] / n, "klk": tot["klk"] / n, "rkl": tot["rkl"] / n,
           "top1": tot["top1"] / n, "ppl": math.exp(tot["nll"] / n), "teacher_ppl": ev.get("teacher_ppl")}
    if role is not None:
        res["scored"] = n
    if mtp:
        res.update(jaccard_mtp=tot["jac_m"] / n_m, klk_mtp=tot["klk_m"] / n_m, top1_mtp=tot["top1_m"] / n_m)
    if len(per) > 1:  # the same metrics per eval family (rows of each family are disjoint)
        res["families"] = {nm: {"jaccard": f["jac"] / f["n"], "klk": f["klk"] / f["n"], "rkl": f["rkl"] / f["n"],
                                "top1": f["top1"] / f["n"], "ppl": math.exp(f["nll"] / f["n"])}
                           for nm, f in zip(names, per, strict=True)}
    return res


def role_loss(lg, x_next, r, idx, val, aidx, aval, a) -> torch.Tensor:
    """Per-token loss of one chunk under roles (omk_qad.rollouts): 1 teacher, 2 anchor, 3 teacher + unlikelihood."""
    per = distill_loss(lg, idx, val, a) * ((r == 1) | (r == 3))
    if (r == 2).any():
        if aidx is None:
            raise SystemExit("anchor-role positions without anchor targets (run --phase cache --anchor)")
        per = per + a.anchor_weight * distill_loss(lg, aidx, aval, a) * (r == 2)
    if (r == 3).any():  # push down the repeated token the student produced: -log(1 - p_s(x_next))
        p = F.log_softmax(lg.float(), -1).gather(-1, x_next[:, None]).squeeze(-1).exp()
        per = per + a.ul_weight * -torch.log((1 - p).clamp_min(1e-6)) * (r == 3)
    return per


def train_step(student, latents, head, fp_params, opt, x, idx, val, a, role=None, aidx=None, aval=None) -> float:
    """One update: body forward, head + loss per chunk, head gradient, body backward (latents update inside).

    With ``role`` (role data, omk_qad.rollouts) only scored positions count: teacher / anchor / loop losses per
    the target token's role, normalised by the number of scored positions.
    """
    cuda = a.device != "cpu"
    h = hidden_of(student, x, cuda)
    hd = h.detach().requires_grad_()
    frozen = getattr(head, "frozen", False)  # bf16 mode: a head outside the scope gets no weight gradient
    head.add_use()
    wh = head.served() if frozen else head.served().requires_grad_()
    n_tok = x.shape[0] * (x.shape[1] - 1) if role is None else max(1, int((role[:, 1:] > 0).sum()))
    tot = 0.0
    for i in range(x.shape[0]):
        for c0 in range(0, x.shape[1] - 1, a.loss_chunk):
            c1 = min(c0 + a.loss_chunk, x.shape[1] - 1)
            if role is not None and not (role[i, c0 + 1 : c1 + 1] > 0).any():
                continue  # nothing scored in this chunk (prompt / tool output / padding)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
                lg = softcap(F.linear(hd[i, c0:c1], wh), getattr(a, "softcap", None))
            if role is not None:
                r = role[i, c0 + 1 : c1 + 1].long()
                ce = F.cross_entropy(lg.float(), x[i, c0 + 1 : c1 + 1], reduction="none") * (r > 0)
                if a.loss == "ce":  # SFT: every scored position is plain next-token CE (no teacher targets)
                    loss = ce.sum() / n_tok
                    loss.backward()
                    tot += loss.item()
                    continue
                per = role_loss(lg, x[i, c0 + 1 : c1 + 1], r, idx[i, c0:c1], val[i, c0:c1],
                                None if aidx is None else aidx[i, c0:c1], None if aval is None else aval[i, c0:c1], a)
                loss = (per.sum() + a.ce_weight * ce.sum()) / n_tok
                loss.backward()
                tot += loss.item()
                continue
            d = distill_loss(lg, idx[i, c0:c1], val[i, c0:c1], a)
            ce = F.cross_entropy(lg.float(), x[i, c0 + 1 : c1 + 1], reduction="none")
            loss = (d.sum() + a.ce_weight * ce.sum()) / n_tok
            loss.backward()
            tot += loss.item()
    if a.mtp == "train" and has_mtp(student):  # MTP distillation: only mtp.* learns (trunk/embed/head detached)
        z = mtp_hidden(student, x, h, cuda)
        zd = z.detach().requires_grad_()
        n_m = x.shape[0] * (x.shape[1] - 2)
        whd = wh.detach()
        for i in range(x.shape[0]):
            for c0 in range(0, x.shape[1] - 2, a.loss_chunk):
                c1 = min(c0 + a.loss_chunk, x.shape[1] - 2)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
                    lg = softcap(F.linear(zd[i, c0:c1], whd), getattr(a, "softcap", None))
                d = distill_loss(lg, idx[i, c0 + 1 : c1 + 1], val[i, c0 + 1 : c1 + 1], a)
                ce = F.cross_entropy(lg.float(), x[i, c0 + 2 : c1 + 2], reduction="none")
                loss = a.mtp_weight * (d.sum() + a.ce_weight * ce.sum()) / n_m
                loss.backward()
                tot += loss.item()
        z.backward(zd.grad)
    if not frozen:
        head.on_grad(wh.grad)
    h.backward(hd.grad)
    for lat in latents.values():
        lat.flush()
    if fp_params and opt is not None:  # opt None: gradients are left in .grad (gradient checks)
        torch.nn.utils.clip_grad_norm_(fp_params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    return tot


def _rng_state() -> dict:
    st = {"torch": torch.get_rng_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st: dict) -> None:
    torch.set_rng_state(st["torch"])
    random.setstate(st["python"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def save_state(path: Path, step: int, latents, student, opt, a, ctl=None, results=None, extra=None) -> None:
    """Atomic full checkpoint: latents, Adafactor + AdamW state, fp params, step, RNG, stats, eval history."""
    named = dict(student.named_parameters())
    sd = {"step": step, "args": vars(a), "latents": {n: lat.param.data.cpu() for n, lat in latents.items()},
          "adafactor": adafactor_state(latents), "adamw": opt.state_dict() if opt else None,
          # trainable non-latents only (bf16 mode freezes everything outside its scope: never written here)
          "fp": {n: p.detach().cpu() for n, p in named.items() if n not in latents and p.requires_grad},
          "rng": _rng_state(), "stats": dict(ctl.stats) if ctl else {}, "results": results or [],
          "folded": {n: lat.basis.block for n, lat in latents.items() if lat.basis is not None},
          # a refresh window serves payloads older than the latent: a resume must serve the same ones
          "payloads": {n: lat.cache_state() for n, lat in latents.items()}
          if max(getattr(a, "refresh", 1), getattr(a, "ggml_refresh", 1)) > 1 else {},
          "saved_at": time.time(), **(extra or {})}
    path = Path(path)
    atomic_torch_save(sd, path)  # per-process temp name: two writers can never rename each other's file
    atomic_json(path.with_suffix(".json"),
                {"step": step, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"), "stage": getattr(a, "stage", 1),
                 "branch": getattr(a, "branch", ""),
                 **{k: v for k, v in (extra or {}).items() if isinstance(v, str | int)}})
    del sd


def load_state(path: str, latents, student, opt, ctl=None, weights_only: bool = False) -> dict:
    """Restore a checkpoint written by :func:`save_state`; ``weights_only`` keeps optimizer/RNG/step fresh."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if set(ck["latents"]) != set(latents):
        raise SystemExit(f"{path}: latent set differs from this student's")
    was = ck.get("folded", {})  # latents stored in the ternary (Hadamard-folded) basis, with their block
    for n, lat in latents.items():
        now = lat.basis is not None
        if (n in was) == now and (not now or was[n] == lat.basis.block):
            lat.param.data.copy_(ck["latents"][n])
            lat.touch()
            if not weights_only and n in ck.get("payloads", {}):
                lat.load_cache(ck["payloads"][n])
        elif now and n not in was and weights_only:  # e.g. a stage-2 ternary branch from a model-basis stage 1
            lat.set_model_weight(ck["latents"][n])
        else:
            raise SystemExit(f"{path}: {n} is stored {'folded' if n in was else 'in the model basis'} but this "
                             f"student holds it {'folded' if now else 'in the model basis'}"
                             f"{'' if weights_only else ' (a resume must use the same layout)'}")
    named = dict(student.named_parameters())
    for n, v in ck["fp"].items():
        named[n].data.copy_(v)
    if not weights_only:
        load_adafactor_state(latents, ck["adafactor"])
        if opt is not None and ck.get("adamw"):
            opt.load_state_dict(ck["adamw"])
        if ck.get("rng"):
            _set_rng_state(ck["rng"])
        if ctl is not None and ck.get("stats"):
            ctl.stats.update(ck["stats"])
    say(f"loaded {path} (step {ck['step']}{', weights only' if weights_only else ''})")
    return ck


# Settings that do not change what a run computes: they may differ between the segments of one run.
RUNTIME_ARGS = {"phase", "device", "stop_at", "log_every", "eval_every", "save_every", "save_minutes", "snapshot_every",
                "wandb_project", "wandb_name", "wandb_entity", "no_wandb", "load", "allow_change", "llama_cpp",
                "teacher_gguf_bin", "teacher_ngl", "teacher_ubatch", "cache_bs", "gguf", "out_hf", "loss_chunk",
                "eval_set"}  # an eval set only measures: a run may gain one on resume


def _model_id(model: str) -> dict:
    """Identity of an HF model dir: config + weight index (or the single weight file)."""
    d, out = Path(model), {}
    for f in ("config.json", "model.safetensors.index.json", "model.safetensors"):
        if (d / f).exists():
            out[f] = file_id(d / f)
    return out or {"path": model}


def sft_files(a) -> list[str]:
    """``--sft-data`` files of a ``--loss ce`` run (refused with any other loss, or a ``ce`` run without them)."""
    files = [f for f in getattr(a, "sft_data", "").split(",") if f]
    if (a.loss == "ce") != bool(files):
        raise SystemExit("--loss ce trains on --sft-data (and --sft-data needs --loss ce)")
    if files and not all(f.endswith(".roles.jsonl") for f in files):
        raise SystemExit("--sft-data takes .roles.jsonl files (omk_qad.rollouts)")
    if files and a.ce_weight:
        raise SystemExit("--loss ce is the CE: --ce-weight would count it twice")
    if files and a.mtp == "train":
        raise SystemExit("--loss ce has no teacher targets for an MTP head: pass --mtp freeze")
    return files


def train_identity(a) -> dict:
    """What a training run's outputs depend on: every non-runtime arg plus the content of its inputs."""
    ident = {k: v for k, v in sorted(vars(a).items()) if k not in RUNTIME_ARGS}
    if ident.get("data_order") == "cache":  # the only order before --data-order: older identities stay valid
        del ident["data_order"]
    if not ident.get("sft_data"):  # likewise every run before --sft-data
        ident.pop("sft_data", None)
    ident["id_model"] = _model_id(a.model)
    cmeta = json.loads((Path(a.cache) / "cache.json").read_text())
    ident["id_cache"] = ({k: cmeta.get(k) for k in ("identity", "n_seq", "anchor_model")} if "identity" in cmeta
                         else file_id(Path(a.cache) / "cache.json"))  # what the targets ARE, not how it was launched
    for k in ("layout", "imatrix", "init_from"):
        v = getattr(a, k, "")
        if v and v not in ("none", "auto") and Path(v).exists():
            ident[f"id_{k}"] = file_id(v)
    if sft_files(a):
        ident["id_sft_data"] = {f: file_id(f) for f in sft_files(a)}
    return ident


def check_resume_args(saved: dict, a, allow: dict, src: str) -> None:
    """Refuse to resume a checkpoint written with different run-critical args (legacy runs have no identity.json)."""
    cur = vars(a)
    diff = {k: (saved.get(k), cur.get(k)) for k in sorted(set(saved) | set(cur))
            if k not in RUNTIME_ARGS and saved.get(k) != cur.get(k) and k in saved}
    bad = {k: v for k, v in diff.items() if k not in allow or (allow[k] is not None and str(v[1]) != allow[k])}
    if bad:
        raise SystemExit(f"RUN_REFUSED resume from {src}: written with different settings ("
                         + "; ".join(f"{k}: {x!r} -> {y!r}" for k, (x, y) in bad.items())
                         + f"). Relaunch with the original values, or pass --allow-change {' '.join(bad)}.")
    for k, (x, y) in diff.items():
        say(f"ALLOWED CHANGE on resume {k}: {x!r} -> {y!r}")


class Guard:
    """Divergence guard (bug-825): roll back to the last healthy state and cut the LR when the loss blows up.

    A loss spike shows up AFTER the updates that caused it, so skipping a step cannot repair it. The guard
    keeps a CPU snapshot of the whole training state (latents, Adafactor and AdamW state, fp params), taken
    every ``every`` healthy steps; ``patience`` consecutive steps with a non-finite loss or a loss above
    ``ratio`` x the EMA of healthy losses restore it and multiply the LR scale by ``cut``. More than ``max``
    rollbacks end the run. ``ratio`` 0 turns it off (no snapshot, no cost).
    """

    def __init__(self, a, latents, student, opt):
        """Read the --guard-* options; nothing is snapshotted until :meth:`take`."""
        self.ratio, self.patience, self.every = a.guard_ratio, a.guard_patience, a.guard_every
        self.cut, self.max = a.guard_cut, a.guard_max
        self.latents, self.opt = latents, opt
        self.fp = {n: p for n, p in student.named_parameters() if n not in latents and p.requires_grad}
        self.scale, self.ema, self.bad, self.rollbacks, self.healthy = 1.0, None, 0, 0, 0
        self.snap: dict | None = None

    @property
    def on(self) -> bool:
        """Whether the guard is enabled."""
        return self.ratio > 0

    def state(self) -> dict:
        """What a checkpoint keeps (the snapshot itself is not persisted: a resume snapshots the resumed state)."""
        return {"scale": self.scale, "ema": self.ema, "rollbacks": self.rollbacks, "healthy": self.healthy}

    def load(self, st: dict | None) -> None:
        """Inverse of :meth:`state`."""
        if st:
            self.scale, self.ema = st["scale"], st["ema"]
            self.rollbacks, self.healthy = st["rollbacks"], st["healthy"]

    @torch.no_grad()
    def take(self, step: int) -> None:
        """Snapshot the training state to CPU."""
        if not self.on:
            return
        self.snap = {"step": step,
                     "latents": {n: lat.param.data.to("cpu", copy=True) for n, lat in self.latents.items()},
                     "adafactor": {n: tuple(t.to("cpu", copy=True) for t in st)
                                   for n, st in adafactor_state(self.latents).items()},
                     "fp": {n: p.detach().to("cpu", copy=True) for n, p in self.fp.items()},
                     "adamw": copy.deepcopy(self.opt.state_dict()) if self.opt is not None else None}

    @torch.no_grad()
    def _restore(self) -> None:
        s = self.snap
        assert s is not None
        for n, lat in self.latents.items():
            lat.param.data.copy_(s["latents"][n])
            lat.touch()
        load_adafactor_state(self.latents, s["adafactor"])
        for n, p in self.fp.items():
            p.data.copy_(s["fp"][n])
            p.grad = None
        if self.opt is not None:
            self.opt.load_state_dict(copy.deepcopy(s["adamw"]))

    def observe(self, step: int, loss: float) -> bool:
        """Account one step's loss; True when it rolled the state back (the caller then skips this step's saves)."""
        if not self.on:
            return False
        if not math.isfinite(loss) or (self.ema is not None and self.healthy >= self.every
                                       and loss > self.ratio * self.ema):
            self.bad += 1
            say(f"GUARD step {step}: loss {loss:.5f} > {self.ratio} x ema {self.ema:.5f} ({self.bad}/{self.patience})")
            if self.bad < self.patience:
                return False
            if self.snap is None or self.rollbacks >= self.max:
                raise SystemExit(f"GUARD_FAIL step {step}: diverged after {self.rollbacks} rollbacks "
                                 f"(LR scale {self.scale:g})")
            self._restore()
            self.rollbacks += 1
            self.scale *= self.cut
            self.bad = 0
            say(f"GUARD_ROLLBACK step {step} -> state of step {self.snap['step']}, LR scale {self.scale:g} "
                f"(rollback {self.rollbacks}/{self.max})")
            return True
        self.bad = 0
        self.ema = loss if self.ema is None else 0.9 * self.ema + 0.1 * loss
        self.healthy += 1
        if step % self.every == 0:
            self.take(step)
        return False


def same_model(p: str, q: str) -> bool:
    """Whether two model dirs are the same checkpoint (same path, or identical config + weight index)."""
    a, b = Path(p).resolve(), Path(q).resolve()
    if a == b:
        return True

    def sig(d: Path) -> tuple:
        files = [d / "config.json", d / "model.safetensors.index.json"]
        return tuple(f.read_bytes() if f.exists() else b"" for f in files)

    return sig(a) == sig(b) and (a / "model.safetensors.index.json").exists()


def run_dir_of(a) -> Path:
    """Stage 1: <out>/stage1 ; stage 2: <out>/stage2/<branch>."""
    if a.stage == 1:
        return Path(a.out) / "stage1"
    if not a.branch:
        raise SystemExit("--stage 2 needs --branch NAME (each stage-2 variant is its own branch)")
    return Path(a.out) / "stage2" / a.branch


def phase_train(a) -> None:
    """QAD training loop with rolling checkpoints, pause/resume and stage-1 / stage-2 runs.

    One process per run dir (``run.lock``); ``identity.json`` + the checkpoint's args refuse a resume with other
    run-critical settings unless ``--allow-change`` names them; ``state.json`` running / paused / drained / done /
    failed. Control: ``touch <run>/PAUSE`` or SIGTERM/SIGINT/SIGUSR1 = checkpoint at the step boundary and exit
    (PAUSED; a 2nd SIGTERM exits at once, last.pt intact); ``echo pause > <run>/control`` (or SIGUSR2) = checkpoint
    and wait in-process until ``run``.
    """
    run = run_dir_of(a)
    run.mkdir(parents=True, exist_ok=True)
    lock = RunLock(run / "run.lock", f"train {run}")
    st = State(run / "state.json", run=str(run))
    ctl_run = RunControl(run / "control", name=f"train {run.name}", log=say)
    try:
        _phase_train(a, run, st, ctl_run)
    except BaseException as e:
        st.refused_or_failed(e)
        raise
    finally:
        ctl_run.restore()
        lock.release()


class CacheRows:
    """The cached training rows in the order a run consumes them, one batch at a time.

    ``shuffle`` (default): one permutation of every cached row, seeded by ``--seed`` alone, so a resume continues it
    and a batch mixes sources and documents. Cache rows follow their source files' order, and a rollout source is
    sorted by run: in that order bm-889 trained one rollout at a time (steps 201-300 were a single run). Shards are
    memory-mapped, so a batch reads only its rows. ``cache``: the cache's own order, one shard in memory at a time
    (the order of runs made before --data-order).
    """

    KEYS = ("x", "idx", "val", "role", "aidx", "aval")

    def __init__(self, cache: str, n_seq: int, shard: int, order: str, seed: int):
        """Fix the order of all ``n_seq`` cached rows (``shard`` rows per file)."""
        self.cache, self.shard, self.order = cache, shard, order
        self.rows = list(range(n_seq))
        if order == "shuffle":
            random.Random(seed).shuffle(self.rows)
        self.open: dict[int, dict] = {}

    def _shard(self, k: int) -> dict:
        if k not in self.open:
            if self.order == "cache":  # sequential: the previous shard is done
                self.open.clear()
            fn = os.path.join(self.cache, f"train_{k:04d}.pt")
            self.open[k] = torch.load(fn, mmap=self.order == "shuffle")
        return self.open[k]

    def batch(self, s0: int, bs: int) -> dict:
        """Rows ``s0 .. s0+bs`` of the order as batch tensors (``role``/``aidx``/``aval`` None when no row has them)."""
        got = [(self._shard(r // self.shard), r % self.shard) for r in self.rows[s0 : s0 + bs]]
        out = {}
        for key in self.KEYS:
            if not any(key in sh for sh, _ in got):
                out[key] = None
                continue
            vals = []
            for sh, j in got:
                if key in sh:
                    vals.append(sh[key][j])
                elif key == "role":  # a shard of text rows only: text rows are teacher-trained everywhere
                    vals.append(torch.ones(sh["x"].shape[1], dtype=torch.int8))
                else:  # a shard without anchor positions: its own teacher targets stand in, masked by role != 2
                    vals.append(sh["idx" if key == "aidx" else "val"][j])
            out[key] = torch.stack(vals)
        return out


class SftRows:
    """``--loss ce`` training rows: the windows of ``--sft-data`` (.roles.jsonl), in memory, in the run's order.

    The same contract as ``CacheRows`` (``shuffle``: one permutation seeded by ``--seed``; ``cache``: file order),
    with no teacher targets: ``idx`` / ``val`` / ``aidx`` / ``aval`` are None. Each file is read once and checked
    against its ``.stats.json`` window count (``_role_rows``).
    """

    def __init__(self, files: list[str], seq: int, order: str, seed: int):
        """Read every row of ``files`` (``seq`` tokens each) and fix their order."""
        xs, rs = [], []
        for f in files:
            done: list = []
            for x, r in _role_rows(f, seq, done):
                if done:
                    break
                xs.append(x.to(torch.int32))
                rs.append(r)
        if not xs:
            raise SystemExit("--sft-data: no rows")
        self.x, self.role = torch.stack(xs), torch.stack(rs)
        self.rows = list(range(len(xs)))
        if order == "shuffle":
            random.Random(seed).shuffle(self.rows)

    def __len__(self) -> int:
        """Number of rows."""
        return len(self.rows)

    def batch(self, s0: int, bs: int) -> dict:
        """Rows ``s0 .. s0+bs`` of the order (as ``CacheRows.batch``, teacher keys None)."""
        sel = torch.tensor(self.rows[s0 : s0 + bs])
        return {"x": self.x[sel], "role": self.role[sel], "idx": None, "val": None, "aidx": None, "aval": None}


def legacy_data_order(run: Path, order: str, log=print) -> str:
    """A run started before --data-order (its identity has no data_order) continues in the cache order."""
    idp = run / "identity.json"
    ident = json.loads(idp.read_text()).get("identity", {}) if idp.exists() else {}
    if order != "cache" and ((run / "last.pt").exists() or idp.exists()) and "data_order" not in ident:
        log("DATA order cache: this run started before --data-order, it resumes in the order it was trained in")
        return "cache"
    return order


def _phase_train(a, run: Path, st: State, control: RunControl) -> None:
    LOG[0] = open(run / "log.txt", "a")
    say("args", json.dumps(vars(a)))
    a.data_order = legacy_data_order(run, a.data_order, say)
    torch.manual_seed(a.seed)
    meta = json.load(open(os.path.join(a.cache, "cache.json")))
    sh = meta["shard"]
    last = a.stop_at or a.steps
    sft = sft_files(a)
    if sft:
        rows = SftRows(sft, meta["seq"], a.data_order, a.seed)
        if len(rows) < last * a.bs:
            raise SystemExit(f"--sft-data holds {len(rows)} rows, the run needs {last * a.bs}")
    elif meta["n_seq"] < last * a.bs:
        raise SystemExit(f"cache holds {meta['n_seq']} sequences, the run needs {last * a.bs}")
    if not sft and sh % a.bs:
        raise SystemExit("--bs must divide the cache shard size")
    final = run / "final.pt"
    if final.exists():
        raise SystemExit(f"{final} exists: this stage/branch is finished and immutable (use a new --branch)")
    ev = torch.load(os.path.join(a.cache, "eval.pt"))
    sets = {}
    for nm in (q for q in a.eval_set.split(",") if q):  # role eval sets (--phase evalset) of this very cache
        sets[nm] = torch.load(evalset_path(a.cache, nm))
        if sets[nm]["identity"]["cache"] != cache_ref(a.cache):
            raise SystemExit(f"eval set {nm} was made beside another cache ({sets[nm]['identity']['cache']} vs "
                             f"{cache_ref(a.cache)})")
        say(f"eval set {nm}: {len(sets[nm]['x'])} rows, {int(teacher_scored(sets[nm]['role'][:, 1:]).sum())} "
            "teacher-scored positions")
    bf16 = a.mode == "bf16"
    if a.imatrix == "auto":  # bf16 mode encodes nothing: no imatrix
        a.imatrix = str(Path(a.cache) / "imatrix.pt") if (Path(a.cache) / "imatrix.pt").exists() and not bf16 \
            else "none"
    if (run / "last.pt").exists() and not (run / "identity.json").exists():
        say("identity.json created for an existing run: earlier segments are checked by their saved args only")
    check_identity(run / "identity.json", train_identity(a), parse_allow(a.allow_change), what=f"train {run}",
                   log=say)
    if bf16:
        from .recovery import build_student_bf16

        student, latents, fp_params, ctl, head = build_student_bf16(a, load_text_causal_lm, say)
    else:
        layout = torch.load(a.layout, weights_only=False)
        student, latents, fp_params, ctl, head_name = build_student(a, layout)
        head = latents[head_name]
    allow = parse_allow(a.allow_change)
    opt = torch.optim.AdamW(fp_params, lr=a.fp_lr, betas=(0.9, 0.95), weight_decay=0.0) if fp_params else None
    last_ck = run / "last.pt"
    origin = {"cache": a.cache, "layout": a.layout, "imatrix": a.imatrix}
    fpath = Path(a.imatrix).with_name("imatrix.folded.pt") if a.imatrix != "none" else None
    if fpath is not None and fpath.exists():
        origin["imatrix_folded"] = str(fpath)
    results: list = []
    start = 1
    guard = Guard(a, latents, student, opt)
    if a.load and last_ck.exists() and Path(a.load).resolve() != last_ck.resolve() and "load" not in allow:
        raise SystemExit(f"RUN_REFUSED --load {a.load}: {last_ck} exists and would be overwritten by a resume from "
                         "another checkpoint (use a new --branch, or --allow-change load)")
    resume = a.load or (str(last_ck) if last_ck.exists() else "")
    if resume:  # full state: continue exactly where it stopped
        ck = load_state(resume, latents, student, opt, ctl)
        check_resume_args(ck.get("args", {}), a, allow, resume)
        start, results = int(ck["step"]) + 1, ck.get("results", [])
        origin["init_from"] = ck.get("init_from", "")
        guard.load(ck.get("guard"))
        say(f"RESUME from step {ck['step']} ({resume})")
    else:
        if a.stage == 2:  # weights of the finished stage 1; optimizer, schedule and RNG start fresh
            src = a.init_from or str(Path(a.out) / "stage1" / "final.pt")
            if not Path(src).exists():
                raise SystemExit(f"stage 2 needs a finished stage 1: {src} not found")
            load_state(src, latents, student, None, weights_only=True)
            origin["init_from"] = src
        elif bf16 and (meta["args"].get("teacher_gguf") or not same_model(meta["args"]["model"], a.model)):
            # another teacher: KL is the gap, not ~0. With --teacher-gguf the cache's --model only gave the tokenizer
            # (bm-889: the 31B GGUF cache carries the 26B student's dir there and would have failed the gate)
            teacher = meta["args"].get("teacher_gguf") or meta["args"]["model"]
            say(f"LOSSLESS_GATE skipped: cache teacher {teacher} is not the student {a.model}")
        else:  # lossless gate: with the quantizer bypassed the student must BE the teacher
            ctl.bypass = True
            g = evaluate(student, head, ev, a, "lossless")
            ctl.bypass = False
            say(f"LOSSLESS_GATE klk={g['klk']:.6f} (max {a.gate})")
            if g["klk"] > a.gate:
                raise SystemExit("unquantized student does not reproduce the teacher cache")
        results.append(evaluate(student, head, ev, a, "init", dict(ctl.stats), sets))
    tel = Telemetry(a, run, {**vars(a), "origin": origin, "cache_meta": {k: meta[k] for k in ("n_seq", "seq", "topk")
                                                                         if k in meta}},
                    resume_step=start - 1 if resume else None)
    if not resume:  # a resumed run's W&B run already holds its init eval
        tel.eval(results[-1], start - 1)
    pause_file = run / "PAUSE"
    st.update(status="running", step=start - 1, steps=a.steps, last=last)
    if not sft:
        rows = CacheRows(a.cache, meta["n_seq"], sh, a.data_order, a.seed)
    src = f"sft {len(rows)} rows, " if sft else ""
    how = f" (seed {a.seed}): first rows {rows.rows[:8]}" if a.data_order == "shuffle" else ": cache rows in sequence"
    say(f"DATA {src}order {a.data_order}{how}")
    t0, t_save = time.time(), time.time()

    def checkpoint(step, path=last_ck):
        save_state(path, step, latents, student, opt, a, ctl, results,
                   {**origin, "guard": guard.state()} if guard.on else origin)
        say(f"CHECKPOINT step {step} -> {path}")

    def peak() -> str:  # read by omk_qad.multi preflight: what this student needs on its GPU
        if a.device == "cpu":
            return "PEAK_MEM allocated=0.0GiB reserved=0.0GiB (cpu)"
        return (f"PEAK_MEM allocated={torch.cuda.max_memory_allocated(a.device) / 2**30:.2f}GiB "
                f"reserved={torch.cuda.max_memory_reserved(a.device) / 2**30:.2f}GiB")

    step = start - 1
    guard.take(step)
    # bm-888: from_pretrained returns an eval-mode model and HF gradient checkpointing only runs while training; the
    # init eval's trailing train() is skipped on resume, which left a resumed run without checkpointing (25 -> 80 GiB)
    student.train()
    for step in range(start, last + 1):
        ctl.step = step
        ctl.lr = cosine_lr(step - 1, a.steps, a.lr, a.min_lr, a.warmup) * guard.scale
        if opt is not None:
            for gq in opt.param_groups:
                gq["lr"] = a.fp_lr * ctl.lr / a.lr
        b = rows.batch((step - 1) * a.bs, a.bs)
        x = b["x"].long().to(a.device)
        idx, val = (None, None) if b["idx"] is None else (b["idx"].to(a.device), b["val"].to(a.device))
        roles = {k: None if b[k] is None else b[k].to(a.device) for k in ("role", "aidx", "aval")}
        loss = train_step(student, latents, head, fp_params, opt, x, idx, val, a, **roles)
        rolled = guard.observe(step, loss)
        row = {"train/loss": loss, "train/lr": ctl.lr, **({"train/guard_rollback": 1} if rolled else {})}
        if step % a.log_every == 0 or step <= 3:
            mem = torch.cuda.max_memory_allocated(a.device) / 2**30 if a.device != "cpu" else 0.0
            sps = (time.time() - t0) / (step - start + 1)
            say(f"step {step} loss={loss:.5f} lr={ctl.lr:.2e} {sps:.2f}s/step "
                f"{mem:.1f}GiB {json.dumps(ctl.stats)}")
            row.update({"train/s_per_step": sps, "train/peak_mem_gib": mem,
                        **{f"quant/{k}": v for k, v in ctl.stats.items()}})
        tel.log(row, step)
        if step % a.eval_every == 0 or step == last:
            results.append(evaluate(student, head, ev, a, f"step{step}", dict(ctl.stats), sets))
            tel.eval(results[-1], step)
            atomic_json(run / "summary.json", {"args": vars(a), "origin": origin, "results": results})
        if a.snapshot_every and step % a.snapshot_every == 0:
            checkpoint(step, run / f"ckpt_step{step:07d}.pt")
        st.update(step=step, loss=loss)
        mode = control.mode()
        why = f"pause file {pause_file}" if pause_file.exists() else ("drain" if mode == "drain" else "")
        if mode == "pause" and not why and not guard.bad:  # in-process pause: checkpoint first, keep the model loaded
            checkpoint(step)
            t_save = time.time()
            st.update(status="paused", step=step)
            if control.boundary() == "drain":
                why = "drain while paused"
            st.update(status="running")
        if guard.bad and not why:  # never roll the saved state forward onto a step the guard suspects
            continue
        finishing = step == a.steps and not why  # final.pt is written right after the loop: no second copy
        if not finishing and (why or rolled or step % a.save_every == 0 or step == last
                              or time.time() - t_save > a.save_minutes * 60):
            checkpoint(step)
            t_save = time.time()
        if why:
            pause_file.unlink(missing_ok=True)
            st.update(status="drained", step=step)
            say(f"PAUSED at step {step} ({why}); relaunch the same command to resume from {last_ck}")
            say(peak())
            tel.finish()
            return
    if step == a.steps and step >= start:
        save_state(final, step, latents, student, opt, a, ctl, results, origin)
        for f in (last_ck, last_ck.with_suffix(".json")):  # a finished stage never resumes (final.pt is immutable)
            f.unlink(missing_ok=True)
        say(f"STAGE_DONE stage {a.stage}{' branch ' + a.branch if a.branch else ''}: {final}")
    say(peak())
    st.update(status="done", step=step)
    say(f"QAD_DONE at step {step} in {time.time() - t0:.0f}s")
    tel.finish()


def phase_export(a) -> None:
    """GGUF of the trained (or, without --load, the untrained = PTQ floor) student at the layout's plan.

    ``--mode bf16``: the source HF checkpoint with the trained tensors of ``--load`` written in (``--out-hf``).
    """
    if a.mode == "bf16":
        from .recovery import export_bf16

        if not (a.load and a.out_hf):
            raise SystemExit("--mode bf16 export needs --load CKPT and --out-hf DIR")
        export_bf16(Path(a.model), a.load, Path(a.out_hf), say)
        return
    from .qad_export import export_gguf

    layout = torch.load(a.layout, weights_only=False)
    if a.load:  # the imatrix training encoded with is part of the run: export must use the same one
        trained_with = torch.load(a.load, map_location="cpu", weights_only=False).get("imatrix", "none") or "none"
        if a.imatrix in ("", "auto"):
            a.imatrix = trained_with
        elif a.imatrix != trained_with:
            raise SystemExit(f"--imatrix {a.imatrix} differs from the one this run trained with ({trained_with})")
    elif a.imatrix in ("", "auto"):
        a.imatrix = "none"
    student, latents, fp_params, _, _ = build_student(a, layout)
    if a.load:
        load_state(a.load, latents, student, None, weights_only=True)
    from .imatrix import load as load_imatrix

    imatrix = load_imatrix(Path(a.imatrix)) if a.imatrix != "none" else None
    fpath = Path(a.imatrix).with_name("imatrix.folded.pt") if imatrix else None
    imatrix_folded = load_imatrix(fpath) if fpath is not None and fpath.exists() else None
    report = export_gguf(student, layout, Path(a.gguf), device=a.device, imatrix=imatrix, latents=latents,
                         imatrix_folded=imatrix_folded)
    if imatrix_folded is not None and report.get("prism_hadamard"):  # provenance of the ternary encodings
        import shutil

        shutil.copyfile(fpath, Path(a.gguf).with_suffix(".imatrix.folded.pt"))
    keys = ("type_counts", "verified", "verify_failures", "type_mismatches", "export_only", "with_imatrix",
            "size_bytes", "seconds")
    print(json.dumps({k: report[k] for k in keys}, indent=1))


def main(argv=None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["cache", "evalset", "train", "export"], required=True)
    ap.add_argument("--mode", choices=["qad", "bf16"], default="qad",
                    help="qad: fake-quantized latents per --layout; bf16: train the model's own bf16 weights in "
                    "--train-scope, no quantizer (omk_qad.recovery)")
    ap.add_argument("--train-scope", default="gemma4", help="bf16 mode: gemma4 | classes:<a,b,...>")
    ap.add_argument("--train-router", action="store_true", help="bf16 mode, gemma4 scope: also train router.proj")
    ap.add_argument("--out-hf", default="", help="bf16 export: output HF checkpoint dir (must be new/empty)")
    ap.add_argument("--anchor", action="store_true", help="cache: add --model's top-K as the ANCHOR targets of the "
                    "anchor-role positions of an existing role cache (bf16 recovery: the original student)")
    ap.add_argument("--anchor-weight", type=float, default=1.0, help="train: weight of the anchor-role loss")
    ap.add_argument("--ul-weight", type=float, default=1.0, help="train: unlikelihood weight on loop-role tokens")
    ap.add_argument("--model", required=True, help="HF model dir (weights) of the teacher / student base")
    ap.add_argument("--teacher-gguf", default="", help="cache: the teacher as a GGUF, run by libllama "
                    "(omk_qad.gguf_teacher); --model then gives the tokenizer only. No imatrix")
    ap.add_argument("--teacher-gguf-bin", default="", help="llama_topk binary (default omk_qad/llama_topk/llama_topk)")
    ap.add_argument("--teacher-ngl", type=int, default=999, help="--teacher-gguf: layers on the GPU")
    ap.add_argument("--teacher-ubatch", type=int, default=512, help="--teacher-gguf: tokens per decode")
    ap.add_argument("--layout", default="", help="omk_qad.layout output (.pt)")
    ap.add_argument("--cache", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--gguf", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-order", choices=("shuffle", "cache"), default="shuffle",
                    help="order of the cached training rows: one seeded permutation (default), or the cache's own "
                    "order (the order of runs made before this flag; such runs resume in it)")
    # cache
    ap.add_argument("--data", default="", help="comma list of path[=weight] (.txt | .jsonl text/messages | .parquet)")
    ap.add_argument("--eval-data", default="", help="comma list of eval files, one family each (name = file stem)")
    ap.add_argument("--n-seq", type=int, default=16000)
    ap.add_argument("--n-eval", type=int, default=32, help="eval rows per eval family")
    ap.add_argument("--eval-set", default="", help="evalset: the name of the role eval set to build beside --cache; "
                    "train: comma list of such sets, scored at their teacher-scored positions on every eval")
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--shard", type=int, default=1000)
    ap.add_argument("--cache-bs", type=int, default=4)
    ap.add_argument("--imatrix-seqs", type=int, default=256, help="cache: training sequences the imatrix is "
                    "collected on (0 = none); saved as <cache>/imatrix.pt (+ imatrix.gguf with --layout)")
    ap.add_argument("--imatrix", default="auto", help="train: imatrix.pt weighting the encoders (auto = "
                    "<cache>/imatrix.pt if present; none = off); export: auto = the one the run trained with")
    ap.add_argument("--llama-cpp", default="/opt/llama.cpp")
    # train
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--stop-at", type=int, default=0, help="end after this step, schedule unchanged")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5, help="Bolt QAD peak LR")
    ap.add_argument("--min-lr", type=float, default=1e-7)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--fp-lr", type=float, default=1e-5)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--clip", type=float, default=1.0, help="max RMS of one tensor's (expert's) normalized update")
    ap.add_argument("--loss", choices=sorted([*DISTILL, "ce"]), default="jaccard",
                    help="distillation loss over the teacher top-K + rest: jaccard | kl = KL(t||s) | rkl = KL(s||t); "
                    "ce = SFT: cross-entropy on the scored positions of --sft-data, no teacher targets")
    ap.add_argument("--sft-data", default="", help="--loss ce: comma list of .roles.jsonl (omk_qad.rollouts) whose "
                    "rows are the training data; --cache then only gives the eval rows and eval sets")
    ap.add_argument("--ce-weight", type=float, default=0.0, help="Bolt: no CE term")
    ap.add_argument("--loss-chunk", type=int, default=1024, help="tokens per head/loss chunk (bounds logits memory)")
    ap.add_argument("--mtp", choices=["train", "freeze"], default="train",
                    help="MTP head (when the layout carries one): distill it (trunk detached) or keep it frozen")
    ap.add_argument("--mtp-weight", type=float, default=0.3, help="weight of the MTP distillation loss")
    ap.add_argument("--latent-dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--latent-min", type=int, default=1 << 20, help="2-D+ tensors at least this big are latents")
    ap.add_argument("--ternary-method", default="top:86", help="ternarizer of PQ2_0 / PTQ1_0 tensors: top:K | opt")
    ap.add_argument("--export-only", default="embedding", help="classes quantized at export only, not in training")
    ap.add_argument("--ggml-refresh", type=int, default=1, help="re-encode libggml-typed segments every N steps")
    ap.add_argument("--refresh", type=int, default=1,
                    help="re-encode ModelOpt-typed segments (IQ1/IQ2/Q8_0) every N steps instead of every update")
    ap.add_argument("--gate", type=float, default=0.01, help="max top-K KL of the unquantized student (lossless gate)")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--save-every", type=int, default=100, help="rolling last.pt every N steps")
    ap.add_argument("--save-minutes", type=float, default=20.0, help="... and at least every M minutes")
    ap.add_argument("--snapshot-every", type=int, default=0, help="also keep numbered ckpt_stepN.pt (0 = off)")
    ap.add_argument("--guard-ratio", type=float, default=0.0, help="divergence guard: loss above this x the EMA "
                    "of healthy losses counts as diverging (0 = off); see Guard")
    ap.add_argument("--guard-patience", type=int, default=3, help="consecutive diverging steps before a rollback")
    ap.add_argument("--guard-every", type=int, default=25, help="healthy steps between in-memory rollback snapshots")
    ap.add_argument("--guard-cut", type=float, default=0.5, help="LR scale factor applied at each rollback")
    ap.add_argument("--guard-max", type=int, default=4, help="rollbacks before the run fails (GUARD_FAIL)")
    ap.add_argument("--stage", type=int, choices=[1, 2], default=1)
    ap.add_argument("--branch", default="", help="stage-2 branch name (each variant of stage 2 is its own branch)")
    ap.add_argument("--init-from", default="", help="stage 2: stage-1 final.pt (default <out>/stage1/final.pt)")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--wandb-project", default="", help="W&B project (default $WANDB_PROJECT or omk-qad). W&B is on "
                    "by itself when a key exists (WANDB_API_KEY or ~/.netrc); naming any --wandb-* makes it required")
    ap.add_argument("--wandb-name", default="", help="W&B run name (default <run>-<student>-<stage>)")
    ap.add_argument("--wandb-entity", default="", help="W&B entity (default $WANDB_ENTITY)")
    ap.add_argument("--no-wandb", action="store_true", help="no W&B even when a key exists")
    ap.add_argument("--allow-change", nargs="*", default=[], metavar="KEY[=VALUE]",
                    help="train: run-critical settings allowed to differ from the run's identity.json / checkpoint")
    ap.add_argument("--load", default="", help="train: resume from this full checkpoint (default: <run>/last.pt "
                    "if present); export: checkpoint whose weights to export")
    a = ap.parse_args(argv)
    {"cache": phase_cache, "evalset": phase_evalset, "train": phase_train, "export": phase_export}[a.phase](a)


if __name__ == "__main__":
    main()
