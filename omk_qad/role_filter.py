# SPDX-License-Identifier: Apache-2.0
"""Re-role an existing ``.roles.jsonl`` (omk_qad.rollouts windows) without rebuilding it from its captures.

    python -m omk_qad.role_filter --in pool.roles.jsonl --keep-roles 3,4 --min-tokens 3:16
        --out loops.train.roles.jsonl [--eval-out loops.eval.roles.jsonl --eval-sessions 6 --seed 0]

Roles outside ``--keep-roles`` become 0 (context: read, never scored); a window with fewer than N tokens of role R
(``--min-tokens R:N``) is dropped. The use: an SFT run (``--loss ce``) takes only the loop labels of on-policy
rollouts (role 3 = unlikelihood of the repeated token), not the rollouts' own text as cross-entropy targets
(those runs are the looping model's output, failed ones included).

``--loop-head N``: only the first N tokens of each loop span (a run of role 3) stay loop, the rest of the span
becomes context: the loss sits on the turn into the loop. Deep in a loop any model continues it by copying its own
context (bm-898: p(repeat) 0.997 on the pruned 20B), and unlikelihood there trains against copying in general.

``--sample N``: keep N of the surviving windows, drawn with ``--seed`` (file order kept): a replay subset of an SFT
source for a short second stage.

``--eval-out``: the windows of ``--eval-sessions`` whole sessions (the id up to the first ``/``; drawn with
``--seed`` among sessions that keep a window) go there instead: a held-out split with no session on both sides.
Each output gets the ``.stats.json`` the trainer checks (window count, sha256).
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

N_ROLES = 5


def loop_head(role: list[int], n: int) -> list[int]:
    """Role 3 only on the first ``n`` tokens of each run of 3; the rest of the run becomes 0."""
    out, run = list(role), 0
    for i, x in enumerate(role):
        run = run + 1 if x == 3 else 0
        if run > n:
            out[i] = 0
    return out


def refilter(rows, keep: set[int], min_tokens: dict[int, int], head: int = 0):
    """(kept rows with roles outside ``keep`` zeroed, tokens_by_role of the kept rows, windows dropped)."""
    out, tot, dropped = [], [0] * N_ROLES, 0
    for r in rows:
        role = [x if x in keep else 0 for x in (loop_head(r["role"], head) if head else r["role"])]
        cnt = [0] * N_ROLES
        for x in role:
            cnt[x] += 1
        if any(cnt[k] < n for k, n in min_tokens.items()):
            dropped += 1
            continue
        out.append({**r, "role": role})
        tot = [a + b for a, b in zip(tot, cnt, strict=True)]
    return out, tot, dropped


def split_sessions(rows, n: int, seed: int):
    """(train rows, eval rows): ``n`` whole sessions (id prefix before ``/``) drawn with ``seed`` are the eval side."""
    sess = sorted({r["id"].split("/")[0] for r in rows})
    if not 0 < n < len(sess):
        raise SystemExit(f"--eval-sessions {n}: the kept windows span {len(sess)} sessions")
    held = set(random.Random(seed).sample(sess, n))
    return ([r for r in rows if r["id"].split("/")[0] not in held], [r for r in rows if r["id"].split("/")[0] in held],
            sorted(held))


def write(path: str, rows, extra: dict) -> dict:
    """The rows and their ``.stats.json`` (data first), atomically; refuses an existing output."""
    from .runctl import atomic_json, atomic_write

    if Path(path).exists():
        raise SystemExit(f"{path} exists, refusing to overwrite")
    data = "".join(json.dumps(w) + "\n" for w in rows)
    tot = [0] * N_ROLES
    for r in rows:
        for x in r["role"]:
            tot[x] += 1
    stats = {"windows": len(rows), "tokens_by_role": tot, "sessions": len({r["id"].split("/")[0] for r in rows}),
             "sha256": hashlib.sha256(data.encode()).hexdigest(), "complete": True, **extra}
    atomic_write(path, data)
    atomic_json(Path(path).with_suffix(".stats.json"), stats)
    return stats


def main() -> None:
    """CLI: see the module docstring."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="source .roles.jsonl")
    ap.add_argument("--out", required=True, help="output .roles.jsonl")
    ap.add_argument("--keep-roles", required=True, help="comma list of roles to keep, e.g. 3,4")
    ap.add_argument("--min-tokens", default="", help="comma list ROLE:N, drop windows with fewer than N tokens of ROLE")
    ap.add_argument("--loop-head", type=int, default=0, help="keep only the first N tokens of each loop span as "
                    "loop (0 = whole spans)")
    ap.add_argument("--sample", type=int, default=0, help="keep N of the surviving windows, drawn with --seed")
    ap.add_argument("--eval-out", help="held-out .roles.jsonl of --eval-sessions whole sessions")
    ap.add_argument("--eval-sessions", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for f in (a.out, a.eval_out):
        if f and not f.endswith(".roles.jsonl"):
            raise SystemExit("outputs must end in .roles.jsonl (the trainer recognises role data by that suffix)")
    if bool(a.eval_out) != bool(a.eval_sessions):
        raise SystemExit("--eval-out and --eval-sessions go together")
    keep = {int(x) for x in a.keep_roles.split(",")}
    mins = {int(k): int(v) for k, v in (kv.split(":") for kv in a.min_tokens.split(",") if kv)}
    raw = Path(a.inp).read_bytes()
    rows = [json.loads(ln) for ln in raw.decode().splitlines() if ln.strip()]
    side = Path(a.inp).with_suffix(".stats.json")
    want = json.loads(side.read_text()).get("windows") if side.exists() else None
    if want is not None and want != len(rows):
        raise SystemExit(f"{a.inp}: {len(rows)} rows, its {side.name} says {want} (a torn or replaced file)")
    kept, _, dropped = refilter(rows, keep, mins, a.loop_head)
    if a.sample:
        if a.sample > len(kept):
            raise SystemExit(f"--sample {a.sample}: only {len(kept)} windows survive the filter")
        pick = set(random.Random(a.seed).sample(range(len(kept)), a.sample))
        kept = [r for i, r in enumerate(kept) if i in pick]
    extra = {"source": a.inp, "source_sha256": hashlib.sha256(raw).hexdigest(), "source_windows": len(rows),
             "dropped": dropped, "args": vars(a)}
    if a.eval_out:
        kept, held, names = split_sessions(kept, a.eval_sessions, a.seed)
        print(json.dumps({"eval": write(a.eval_out, held, {**extra, "eval_session_ids": names})}))
        extra = {**extra, "eval_session_ids": names}
    print(json.dumps({"train": write(a.out, kept, extra)}))


if __name__ == "__main__":
    main()
