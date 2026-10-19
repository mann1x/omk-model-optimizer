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

    @torch.no_grad()
    def topk_of(x):
        lp = F.log_softmax(teacher(input_ids=x.to(a.device)).logits[:, :-1].float(), -1)
        v, i = lp.topk(a.topk, -1)
        nll = -lp.gather(-1, x[:, 1:, None].to(a.device)).sum().item()
        return i.to(torch.int32).cpu(), v.to(torch.float16).cpu(), nll

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
            continue
        parts = [topk_of(x[i : i + a.cache_bs]) for i in range(0, len(x), a.cache_bs)]
        torch.save({"x": x.to(torch.int32), "idx": torch.cat([q[0] for q in parts]),
                    "val": torch.cat([q[1] for q in parts])}, fn + ".tmp")
        os.replace(fn + ".tmp", fn)
        say(f"cached {s0 + len(x)}/{a.n_seq} sequences [{time.time() - t0:.0f}s] rows per source {n_src}")
    meta = {"args": vars(a), "n_seq": a.n_seq, "shard": a.shard, "topk": a.topk, "seq": a.seq,
            "rows_per_source": dict(zip([p for p, _ in srcs], n_src)), "source_passes": passes}
    json.dump(meta, open(os.path.join(a.cache, "cache.json"), "w"), indent=1)
    say("CACHE_DONE")


# ===================================================================================================== student
def build_student(a, layout: dict):
    """Student model with latents installed; returns (model, latents, fp_params, ctl, head_name)."""
    cuda = a.device != "cpu"
    student, _ = load_text_causal_lm(a.model, a.device, torch.bfloat16 if cuda else torch.float32)
    named = dict(student.named_parameters())
    if {n: list(p.shape) for n, p in named.items()} != layout["params"]:
        raise SystemExit("student parameters do not match the layout (names or shapes)")
    from .fakequant import Codec

    ctl = Control(compute_dtype=torch.bfloat16 if cuda else torch.float32, beta2=a.beta2, clip=a.clip,
                  ggml_refresh=a.ggml_refresh, dummy=torch.zeros((), requires_grad=True, device=a.device))
    codec = Codec(Path(layout["llama_cpp"]))
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
    tied = head_param is student.get_input_embeddings().weight
    install(student, latents, skip=set() if tied else {head_name})
    student.config.use_cache = False
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    n_q = sum(lat.param.numel() for lat in latents.values() if lat.segments)
    say(f"student: {len(latents)} latents ({sum(lat.param.numel() for lat in latents.values()) / 1e9:.3f}B as "
        f"{a.latent_dtype}; {n_q / 1e9:.3f}B fake-quantized), fp params {sum(p.numel() for p in fp_params) / 1e6:.2f}M,"
        f" head {head_name} ({'tied' if tied else 'untied'})")
    return student, latents, fp_params, ctl, head_name


def hidden_of(student, x, cuda: bool):
    """Final hidden states of the decoder body (the head is applied per chunk by the caller)."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
        return student.get_decoder()(input_ids=x).last_hidden_state


def token_terms(logits, idx, val, x_next):
    """Per-token Jaccard, KL (top-K + rest), CE and top-1 agreement."""
    logits = logits.float()
    return (jaccard_topk_rest(logits, idx.long(), val.float()), kl_topk_rest(logits, idx.long(), val.float()),
            F.cross_entropy(logits, x_next, reduction="none"), (logits.argmax(-1) == idx[:, 0].long()).float())


@torch.no_grad()
def evaluate(student, head, ev, a, tag, extra=None) -> dict:
    """Jaccard / KL(top-K) / top-1 / ppl of the student against the cached teacher on the eval rows."""
    cuda = a.device != "cpu"
    student.eval()
    wh = head.served()
    tot = {"jac": 0.0, "klk": 0.0, "nll": 0.0, "top1": 0.0}
    n = 0
    for i0 in range(0, len(ev["x"]), 4):
        x = ev["x"][i0 : i0 + 4].long().to(a.device)
        h = hidden_of(student, x, cuda)
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
    student.train()
    res = {"tag": tag, "jaccard": tot["jac"] / n, "klk": tot["klk"] / n, "top1": tot["top1"] / n,
           "ppl": math.exp(tot["nll"] / n), "teacher_ppl": ev.get("teacher_ppl")}
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
    head.on_grad(wh.grad)
    h.backward(hd.grad)
    for lat in latents.values():
        lat.flush()
    if fp_params and opt is not None:  # opt None: gradients are left in .grad (gradient checks)
        torch.nn.utils.clip_grad_norm_(fp_params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    return tot


def save_state(path: Path, step: int, latents, student, opt, a) -> None:
    """Atomic checkpoint: latents, Adafactor + AdamW state, fp params, step."""
    named = dict(student.named_parameters())
    sd = {"step": step, "args": vars(a), "latents": {n: lat.param.data.cpu() for n, lat in latents.items()},
          "adafactor": adafactor_state(latents), "adamw": opt.state_dict() if opt else None,
          "fp": {n: p.detach().cpu() for n, p in named.items() if n not in latents}}
    torch.save(sd, str(path) + ".tmp")
    os.replace(str(path) + ".tmp", path)


def load_state(path: str, latents, student, opt) -> int:
    """Restore a checkpoint written by :func:`save_state`; returns its step."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if set(ck["latents"]) != set(latents):
        raise SystemExit(f"{path}: latent set differs from this student's")
    for n, lat in latents.items():
        lat.param.data.copy_(ck["latents"][n])
        lat.touch()
    load_adafactor_state(latents, ck["adafactor"])
    named = dict(student.named_parameters())
    for n, v in ck["fp"].items():
        named[n].data.copy_(v)
    if opt is not None and ck.get("adamw"):
        opt.load_state_dict(ck["adamw"])
    say(f"loaded {path} (step {ck['step']})")
    return int(ck["step"])


def phase_train(a) -> None:
    """QAD training loop (see module docstring)."""
    os.makedirs(a.out, exist_ok=True)
    LOG[0] = open(os.path.join(a.out, "log.txt"), "a")
    say("args", json.dumps(vars(a)))
    torch.manual_seed(a.seed)
    meta = json.load(open(os.path.join(a.cache, "cache.json")))
    sh = meta["shard"]
    last = a.stop_at or a.steps
    if meta["n_seq"] < last * a.bs:
        raise SystemExit(f"cache holds {meta['n_seq']} sequences, the run needs {last * a.bs}")
    if sh % a.bs:
        raise SystemExit("--bs must divide the cache shard size")
    ev = torch.load(os.path.join(a.cache, "eval.pt"))
    layout = torch.load(a.layout, weights_only=False)
    student, latents, fp_params, ctl, head_name = build_student(a, layout)
    head = latents[head_name]
    opt = torch.optim.AdamW(fp_params, lr=a.fp_lr, betas=(0.9, 0.95), weight_decay=0.0) if fp_params else None
    start = 1
    if a.load:
        start = load_state(a.load, latents, student, opt) + 1
    else:  # lossless gate: with the quantizer bypassed the student must BE the teacher
        ctl.bypass = True
        g = evaluate(student, head, ev, a, "lossless")
        ctl.bypass = False
        say(f"LOSSLESS_GATE klk={g['klk']:.6f} (max {a.gate})")
        if g["klk"] > a.gate:
            raise SystemExit("unquantized student does not reproduce the teacher cache")
    results = [evaluate(student, head, ev, a, "init", dict(ctl.stats))]
    shard = (-1, None)
    t0 = time.time()
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
            json.dump({"args": vars(a), "results": results}, open(os.path.join(a.out, "summary.json"), "w"),
                      indent=1)
        if step % a.save_every == 0 or step == last:
            save_state(Path(a.out) / "state.pt", step, latents, student, opt, a)
    say(f"QAD_DONE in {time.time() - t0:.0f}s")


def phase_export(a) -> None:
    """GGUF of the trained (or, without --load, the untrained = PTQ floor) student at the layout's plan."""
    from .qad_export import export_gguf

    layout = torch.load(a.layout, weights_only=False)
    student, latents, fp_params, _, _ = build_student(a, layout)
    if a.load:
        load_state(a.load, latents, student, None)
    report = export_gguf(student, layout, Path(a.gguf), device=a.device)
    keys = ("type_counts", "verified", "verify_failures", "type_mismatches", "export_only", "size_bytes", "seconds")
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
    ap.add_argument("--latent-dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--latent-min", type=int, default=1 << 20, help="2-D+ tensors at least this big are latents")
    ap.add_argument("--export-only", default="embedding", help="classes quantized at export only, not in training")
    ap.add_argument("--ggml-refresh", type=int, default=1, help="re-encode libggml-typed segments every N steps")
    ap.add_argument("--gate", type=float, default=0.01, help="max top-K KL of the unquantized student (lossless gate)")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--load", default="")
    a = ap.parse_args(argv)
    {"cache": phase_cache, "train": phase_train, "export": phase_export}[a.phase](a)


if __name__ == "__main__":
    main()
