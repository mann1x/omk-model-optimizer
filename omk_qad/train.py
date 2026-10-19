# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic 2-bit QAD (Bolt recipe) on ONE GPU, with the ternary bonsai27 memory recipe.

--phase cache   run the frozen teacher once over the training stream and the eval rows; store tokens +
top-K teacher log-probs (the teacher is never on the GPU during training);
--phase train   student latents (fp16) fake-quantized per the layout (omk_qad.layout: llama.cpp's plan,
e.g. IQ2_M with experts fixed at IQ2_XXS), loss = Jaccard over top-K + rest (Bolt), updates
applied inside backward (omk_qad.fakequant), small fp tensors on AdamW;
--phase export  latents -> GGUF at the plan with the encoders training served (omk_qad.qad_export).
Without --load this exports the untrained base: the PTQ floor at the same plan.

python -m omk_qad.train --phase cache  --model M --cache C --data a.jsonl=3,b.parquet=1 --eval-data e.txt
python -m omk_qad.train --phase train  --model M --layout L.pt --cache C --out O [--load O/state.pt]
python -m omk_qad.train --phase export --model M --layout L.pt [--load O/state.pt] --gguf X.gguf
"""

import argparse
import json
import math
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
from .losses import jaccard_topk_rest, kl_topk_rest
from .ptq import load_text_causal_lm

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


def phase_cache(a) -> None:
    """Teacher top-K cache over the mixed training stream and the eval rows."""
    os.makedirs(a.cache, exist_ok=True)
    LOG[0] = open(os.path.join(a.cache, "cache.log"), "a")
    say("args", json.dumps(vars(a)))
    dtype = torch.float32 if a.device == "cpu" else torch.bfloat16
    teacher, tok = load_text_causal_lm(a.model, a.device, dtype)
    teacher.requires_grad_(False)
    from .mtp import attach, checkpoint_mtp_keys, load_weights, mtp_layers

    if a.imatrix_seqs and mtp_layers(teacher.config) and checkpoint_mtp_keys(Path(a.model)):
        attach(teacher)  # the MTP head's inputs go into the imatrix too (no MTP-head imatrix gap)
        load_weights(teacher, Path(a.model))
    imat = {"collector": None, "seqs": 0}

    @torch.no_grad()
    def topk_of(x):
        x = x.to(a.device)
        h = teacher.get_decoder()(input_ids=x).last_hidden_state
        if imat["collector"] is not None and getattr(teacher, "mtp", None) is not None:
            teacher.mtp(teacher.get_input_embeddings()(x[:, 1:]), h[:, :-1], teacher.get_decoder().rotary_emb)
        lp = F.log_softmax(teacher.get_output_embeddings()(h[:, :-1]).float(), -1)
        v, i = lp.topk(a.topk, -1)
        nll = -lp.gather(-1, x[:, 1:, None]).sum().item()
        if imat["collector"] is not None:
            imat["seqs"] += x.shape[0]
            if imat["seqs"] >= a.imatrix_seqs:
                save_imatrix()
        return i.to(torch.int32).cpu(), v.to(torch.float16).cpu(), nll

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
    ev_stream = _packed(a.eval_data, tok, a.seq, tok.eos_token_id, passes)
    ex = torch.stack([next(ev_stream) for _ in range(a.n_eval)])
    parts = [topk_of(ex[i : i + 4]) for i in range(0, len(ex), 4)]
    ev = {"x": ex.to(torch.int32), "idx": torch.cat([q[0] for q in parts]), "val": torch.cat([q[1] for q in parts]),
          "teacher_ppl": math.exp(sum(q[2] for q in parts) / (len(ex) * (a.seq - 1)))}
    torch.save(ev, os.path.join(a.cache, "eval.pt"))
    say(f"eval: {len(ex)} x {a.seq} tokens; teacher ppl {ev['teacher_ppl']:.3f}; top-{a.topk} mass "
        f"{ev['val'].float().exp().sum(-1).mean().item():.4f}")

    srcs = [(q.split("=")[0], float(q.split("=")[1]) if "=" in q else 1.0) for q in a.data.split(",") if q]
    want_folded = bool(a.layout) and not (Path(a.cache) / "imatrix.folded.pt").exists()
    if a.imatrix_seqs and (not (Path(a.cache) / "imatrix.pt").exists() or want_folded):
        from .imatrix import Collector, folded_candidates

        folded = folded_candidates(torch.load(a.layout, weights_only=False)) if want_folded else None
        imat["collector"] = Collector(teacher, folded=folded)  # training-stream rows only (eval rows done above)
        if not a.layout:
            say("WARNING: no --layout: imatrix.pt only (no imatrix.gguf, no folded statistic for ternary students)")
    streams = [_packed(p, tok, a.seq, tok.eos_token_id, passes) for p, _ in srcs]
    rng, n_src = random.Random(a.seed), [0] * len(srcs)
    t0 = time.time()
    for s0 in range(0, a.n_seq, a.shard):
        fn = os.path.join(a.cache, f"train_{s0 // a.shard:04d}.pt")
        picks = rng.choices(range(len(srcs)), [w for _, w in srcs], k=min(a.shard, a.n_seq - s0))
        x = torch.stack([next(streams[k]) for k in picks])  # always drawn, so a resume stays aligned
        for k in picks:
            n_src[k] += 1
        if os.path.exists(fn):
            # a resumed cache still owes its imatrix: run the teacher on the cached rows for the imatrix only
            for i in range(0, len(x), a.cache_bs):
                if imat["collector"] is None:
                    break
                topk_of(x[i : i + a.cache_bs])
            continue
        parts = [topk_of(x[i : i + a.cache_bs]) for i in range(0, len(x), a.cache_bs)]
        torch.save({"x": x.to(torch.int32), "idx": torch.cat([q[0] for q in parts]),
                    "val": torch.cat([q[1] for q in parts])}, fn + ".tmp")
        os.replace(fn + ".tmp", fn)
        say(f"cached {s0 + len(x)}/{a.n_seq} sequences [{time.time() - t0:.0f}s] rows per source {n_src}")
    if imat["collector"] is not None:  # fewer sequences than --imatrix-seqs
        save_imatrix()
    meta = {"args": vars(a), "n_seq": a.n_seq, "shard": a.shard, "topk": a.topk, "seq": a.seq,
            "rows_per_source": dict(zip([p for p, _ in srcs], n_src)), "source_passes": passes}
    json.dump(meta, open(os.path.join(a.cache, "cache.json"), "w"), indent=1)
    say("CACHE_DONE")


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
                  ggml_refresh=a.ggml_refresh, dummy=torch.zeros((), requires_grad=True, device=a.device))
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


@torch.no_grad()
def evaluate(student, head, ev, a, tag, extra=None) -> dict:
    """Jaccard / KL(top-K) / top-1 / ppl of the student against the cached teacher on the eval rows.

    With an MTP head also ``*_mtp``: its prediction of x_{t+2} vs the teacher's distribution at t+1
    (``top1_mtp`` is the draft-acceptance proxy).
    """
    cuda = a.device != "cpu"
    student.eval()
    wh = head.served()
    mtp = has_mtp(student)
    tot = dict.fromkeys(("jac", "klk", "nll", "top1", "jac_m", "klk_m", "top1_m"), 0.0)
    n = n_m = 0
    for i0 in range(0, len(ev["x"]), 4):
        x = ev["x"][i0 : i0 + 4].long().to(a.device)
        h = hidden_of(student, x, cuda)
        z = mtp_hidden(student, x, h, cuda) if mtp else None
        for i in range(len(x)):
            for c0 in range(0, x.shape[1] - 1, a.loss_chunk):
                c1 = min(c0 + a.loss_chunk, x.shape[1] - 1)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
                    lg = F.linear(h[i, c0:c1], wh)
                j, k, ce, t1 = token_terms(lg, ev["idx"][i0 + i, c0:c1].to(a.device),
                                           ev["val"][i0 + i, c0:c1].to(a.device), x[i, c0 + 1 : c1 + 1])
                tot["jac"] += j.sum().item()
                tot["klk"] += k.sum().item()
                tot["nll"] += ce.sum().item()
                tot["top1"] += t1.sum().item()
                n += c1 - c0
            if mtp:  # z_t (t <= T-3) predicts x_{t+2}; teacher distribution at t+1
                for c0 in range(0, x.shape[1] - 2, a.loss_chunk):
                    c1 = min(c0 + a.loss_chunk, x.shape[1] - 2)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
                        lg = F.linear(z[i, c0:c1], wh)
                    j, k, _, t1 = token_terms(lg, ev["idx"][i0 + i, c0 + 1 : c1 + 1].to(a.device),
                                              ev["val"][i0 + i, c0 + 1 : c1 + 1].to(a.device), x[i, c0 + 2 : c1 + 2])
                    tot["jac_m"] += j.sum().item()
                    tot["klk_m"] += k.sum().item()
                    tot["top1_m"] += t1.sum().item()
                    n_m += c1 - c0
    student.train()
    res = {"tag": tag, "jaccard": tot["jac"] / n, "klk": tot["klk"] / n, "top1": tot["top1"] / n,
           "ppl": math.exp(tot["nll"] / n), "teacher_ppl": ev.get("teacher_ppl")}
    if mtp:
        res.update(jaccard_mtp=tot["jac_m"] / n_m, klk_mtp=tot["klk_m"] / n_m, top1_mtp=tot["top1_m"] / n_m)
    res.update(extra or {})
    say("EVAL", json.dumps(res))
    return res


def train_step(student, latents, head, fp_params, opt, x, idx, val, a) -> float:
    """One update: body forward, head + loss per chunk, head gradient, body backward (latents update inside)."""
    cuda = a.device != "cpu"
    h = hidden_of(student, x, cuda)
    hd = h.detach().requires_grad_()
    head.add_use()
    wh = head.served().requires_grad_()
    n_tok = x.shape[0] * (x.shape[1] - 1)
    tot = 0.0
    for i in range(x.shape[0]):
        for c0 in range(0, x.shape[1] - 1, a.loss_chunk):
            c1 = min(c0 + a.loss_chunk, x.shape[1] - 1)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
                lg = F.linear(hd[i, c0:c1], wh)
            j, k, ce, _ = token_terms(lg, idx[i, c0:c1], val[i, c0:c1], x[i, c0 + 1 : c1 + 1])
            loss = ((j if a.loss == "jaccard" else k).sum() + a.ce_weight * ce.sum()) / n_tok
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
                    lg = F.linear(zd[i, c0:c1], whd)
                j, k, ce, _ = token_terms(lg, idx[i, c0 + 1 : c1 + 1], val[i, c0 + 1 : c1 + 1], x[i, c0 + 2 : c1 + 2])
                loss = a.mtp_weight * ((j if a.loss == "jaccard" else k).sum() + a.ce_weight * ce.sum()) / n_m
                loss.backward()
                tot += loss.item()
        z.backward(zd.grad)
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
          "fp": {n: p.detach().cpu() for n, p in named.items() if n not in latents},
          "rng": _rng_state(), "stats": dict(ctl.stats) if ctl else {}, "results": results or [],
          "folded": {n: lat.basis.block for n, lat in latents.items() if lat.basis is not None},
          "saved_at": time.time(), **(extra or {})}
    path = Path(path)
    torch.save(sd, str(path) + ".tmp")
    os.replace(str(path) + ".tmp", path)
    path.with_suffix(".json").write_text(json.dumps(
        {"step": step, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"), "stage": getattr(a, "stage", 1),
         "branch": getattr(a, "branch", ""), **{k: v for k, v in (extra or {}).items() if isinstance(v, str | int)}},
        indent=1))
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


class Pauser:
    """SIGINT / SIGTERM or a PAUSE file in the run dir: finish the step, checkpoint, exit cleanly."""

    def __init__(self, run_dir: Path):
        """Install the signal handlers."""
        import signal

        self.file, self.why = run_dir / "PAUSE", ""
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._on_signal)

    def _on_signal(self, signum, frame):
        del frame
        self.why = f"signal {signum}"

    def requested(self) -> str:
        """Reason to pause now, or ''."""
        if not self.why and self.file.exists():
            self.why = f"pause file {self.file}"
        return self.why


def run_dir_of(a) -> Path:
    """Stage 1: <out>/stage1 ; stage 2: <out>/stage2/<branch>."""
    if a.stage == 1:
        return Path(a.out) / "stage1"
    if not a.branch:
        raise SystemExit("--stage 2 needs --branch NAME (each stage-2 variant is its own branch)")
    return Path(a.out) / "stage2" / a.branch


def phase_train(a) -> None:
    """QAD training loop with rolling checkpoints, pause/resume and stage-1 / stage-2 runs."""
    run = run_dir_of(a)
    run.mkdir(parents=True, exist_ok=True)
    LOG[0] = open(run / "log.txt", "a")
    say("args", json.dumps(vars(a)))
    torch.manual_seed(a.seed)
    meta = json.load(open(os.path.join(a.cache, "cache.json")))
    sh = meta["shard"]
    last = a.stop_at or a.steps
    if meta["n_seq"] < last * a.bs:
        raise SystemExit(f"cache holds {meta['n_seq']} sequences, the run needs {last * a.bs}")
    if sh % a.bs:
        raise SystemExit("--bs must divide the cache shard size")
    final = run / "final.pt"
    if final.exists():
        raise SystemExit(f"{final} exists: this stage/branch is finished and immutable (use a new --branch)")
    ev = torch.load(os.path.join(a.cache, "eval.pt"))
    if a.imatrix == "auto":
        a.imatrix = str(Path(a.cache) / "imatrix.pt") if (Path(a.cache) / "imatrix.pt").exists() else "none"
    layout = torch.load(a.layout, weights_only=False)
    student, latents, fp_params, ctl, head_name = build_student(a, layout)
    head = latents[head_name]
    opt = torch.optim.AdamW(fp_params, lr=a.fp_lr, betas=(0.9, 0.95), weight_decay=0.0) if fp_params else None
    last_ck = run / "last.pt"
    origin = {"cache": a.cache, "layout": a.layout, "imatrix": a.imatrix}
    fpath = Path(a.imatrix).with_name("imatrix.folded.pt") if a.imatrix != "none" else None
    if fpath is not None and fpath.exists():
        origin["imatrix_folded"] = str(fpath)
    results: list = []
    start = 1
    resume = a.load or (str(last_ck) if last_ck.exists() else "")
    if resume:  # full state: continue exactly where it stopped
        ck = load_state(resume, latents, student, opt, ctl)
        start, results = int(ck["step"]) + 1, ck.get("results", [])
        origin["init_from"] = ck.get("init_from", "")
        say(f"RESUME from step {ck['step']} ({resume})")
    else:
        if a.stage == 2:  # weights of the finished stage 1; optimizer, schedule and RNG start fresh
            src = a.init_from or str(Path(a.out) / "stage1" / "final.pt")
            if not Path(src).exists():
                raise SystemExit(f"stage 2 needs a finished stage 1: {src} not found")
            load_state(src, latents, student, None, weights_only=True)
            origin["init_from"] = src
        else:  # lossless gate: with the quantizer bypassed the student must BE the teacher
            ctl.bypass = True
            g = evaluate(student, head, ev, a, "lossless")
            ctl.bypass = False
            say(f"LOSSLESS_GATE klk={g['klk']:.6f} (max {a.gate})")
            if g["klk"] > a.gate:
                raise SystemExit("unquantized student does not reproduce the teacher cache")
        results.append(evaluate(student, head, ev, a, "init", dict(ctl.stats)))
    pauser = Pauser(run)
    shard = (-1, None)
    t0, t_save = time.time(), time.time()

    def checkpoint(step, path=last_ck):
        save_state(path, step, latents, student, opt, a, ctl, results, origin)
        say(f"CHECKPOINT step {step} -> {path}")

    step = start - 1
    for step in range(start, last + 1):
        ctl.step = step
        ctl.lr = cosine_lr(step - 1, a.steps, a.lr, a.min_lr, a.warmup)
        if opt is not None:
            for gq in opt.param_groups:
                gq["lr"] = a.fp_lr * ctl.lr / a.lr
        s0 = (step - 1) * a.bs
        if s0 // sh != shard[0]:
            shard = (s0 // sh, torch.load(os.path.join(a.cache, f"train_{s0 // sh:04d}.pt")))
        j = s0 % sh
        x = shard[1]["x"][j : j + a.bs].long().to(a.device)
        idx, val = shard[1]["idx"][j : j + a.bs].to(a.device), shard[1]["val"][j : j + a.bs].to(a.device)
        loss = train_step(student, latents, head, fp_params, opt, x, idx, val, a)
        if step % a.log_every == 0 or step <= 3:
            mem = torch.cuda.max_memory_allocated(a.device) / 2**30 if a.device != "cpu" else 0.0
            say(f"step {step} loss={loss:.5f} lr={ctl.lr:.2e} {(time.time() - t0) / (step - start + 1):.2f}s/step "
                f"{mem:.1f}GiB {json.dumps(ctl.stats)}")
        if step % a.eval_every == 0 or step == last:
            results.append(evaluate(student, head, ev, a, f"step{step}", dict(ctl.stats)))
            (run / "summary.json").write_text(json.dumps({"args": vars(a), "origin": origin, "results": results},
                                                          indent=1))
        if a.snapshot_every and step % a.snapshot_every == 0:
            checkpoint(step, run / f"ckpt_step{step:07d}.pt")
        why = pauser.requested()
        if why or step % a.save_every == 0 or step == last or time.time() - t_save > a.save_minutes * 60:
            checkpoint(step)
            t_save = time.time()
        if why:
            pauser.file.unlink(missing_ok=True)
            say(f"PAUSED at step {step} ({why}); relaunch the same command to resume from {last_ck}")
            return
    if step == a.steps and step >= start:
        save_state(final, step, latents, student, opt, a, ctl, results, origin)
        say(f"STAGE_DONE stage {a.stage}{' branch ' + a.branch if a.branch else ''}: {final}")
    say(f"QAD_DONE at step {step} in {time.time() - t0:.0f}s")


def phase_export(a) -> None:
    """GGUF of the trained (or, without --load, the untrained = PTQ floor) student at the layout's plan."""
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
    if any(lat.basis is not None for lat in latents.values()):
        raise SystemExit("ternary (PQ2_0 / PTQ1_0) GGUF export is not implemented yet (plan M4); training works")
    if a.load:
        load_state(a.load, latents, student, None, weights_only=True)
    from .imatrix import load as load_imatrix

    imatrix = load_imatrix(Path(a.imatrix)) if a.imatrix != "none" else None
    report = export_gguf(student, layout, Path(a.gguf), device=a.device, imatrix=imatrix)
    keys = ("type_counts", "verified", "verify_failures", "type_mismatches", "export_only", "with_imatrix",
            "size_bytes", "seconds")
    print(json.dumps({k: report[k] for k in keys}, indent=1))


def main(argv=None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["cache", "train", "export"], required=True)
    ap.add_argument("--model", required=True, help="HF model dir (weights) of the teacher / student base")
    ap.add_argument("--layout", default="", help="omk_qad.layout output (.pt)")
    ap.add_argument("--cache", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--gguf", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    # cache
    ap.add_argument("--data", default="", help="comma list of path[=weight] (.txt | .jsonl text/messages | .parquet)")
    ap.add_argument("--eval-data", default="")
    ap.add_argument("--n-seq", type=int, default=16000)
    ap.add_argument("--n-eval", type=int, default=32)
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
    ap.add_argument("--loss", choices=["jaccard", "kl"], default="jaccard")
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
    ap.add_argument("--gate", type=float, default=0.01, help="max top-K KL of the unquantized student (lossless gate)")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--save-every", type=int, default=100, help="rolling last.pt every N steps")
    ap.add_argument("--save-minutes", type=float, default=20.0, help="... and at least every M minutes")
    ap.add_argument("--snapshot-every", type=int, default=0, help="also keep numbered ckpt_stepN.pt (0 = off)")
    ap.add_argument("--stage", type=int, choices=[1, 2], default=1)
    ap.add_argument("--branch", default="", help="stage-2 branch name (each variant of stage 2 is its own branch)")
    ap.add_argument("--init-from", default="", help="stage 2: stage-1 final.pt (default <out>/stage1/final.pt)")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--load", default="", help="train: resume from this full checkpoint (default: <run>/last.pt "
                    "if present); export: checkpoint whose weights to export")
    a = ap.parse_args(argv)
    {"cache": phase_cache, "train": phase_train, "export": phase_export}[a.phase](a)


if __name__ == "__main__":
    main()
