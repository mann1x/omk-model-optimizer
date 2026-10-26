#!/usr/bin/env python3
"""Build mix_v1 (bm-877): small code-heavy sources for the omk_qad Qwen3.5-4B smoke, from data already on bs2.

Each Nemotron source is a seeded random sample of whole rows (line offsets from a full scan, so long rows are not
over-picked), rendered with the student's own chat template (tools included) into {"text": ...}, capped at
--cap tokens per document, and filled until it holds >= budget * --margin tokens, so omk_qad's packer never
restarts (repeats) a source within one run. Eval rows are drawn first and are disjoint from the train rows.

Out: ROOT/data/mix_v1/<source>.train.jsonl, <source>.eval.jsonl, manifest.json (budgets, real token counts, row
indices, sha256 of every output, upstream repo sha from the stager's manifest).
"""
import argparse
import hashlib
import json
import mmap
import random
from pathlib import Path

NEM = "data/nemotron"
OC = "SFT-OpenCode-v1"
CP = "SFT-Competitive-Programming-v2/data"
# OpenCode rows open with the same ~9.9k-char opencode agent prompt (only the <env> path differs) and placeholder
# tool schemas: rendered whole, every document is mostly that boilerplate, so the system turn and tools are dropped.
DROP_SYSTEM = {"opencode"}
SOURCES = {  # name -> (files under data/nemotron, train token budget)
    "opencode": ([f"{OC}/{d}/data.jsonl" for d in ("general", "bash_only_tool", "bash_only_tool_skills",
                                                   "agent_skills", "agent_skills_question_tool", "question_tool")],
                 3_000_000),
    "swe_agentless": (["SFT-SWE-v2/data/agentless.jsonl"], 1_500_000),
    "cp_reasoning": ([f"{CP}/competitive_programming_{x}.jsonl.windows.jsonl"
                      for x in ("python_00", "python_01", "cpp_00", "cpp_01")], 1_500_000),
    "exercism_sql": ([f"{CP}/exercism.jsonl", f"{CP}/text_to_sql.jsonl"], 1_000_000),
    # opt-in (not in mix_v1): agentic tool calling, for the families stage 1 fitted worst (bm-885)
    "agentic": ([f"SFT-Agentic-v2/data/{x}.jsonl" for x in ("tool_calling", "interactive_agent")], 3_000_000),
}
MIX_V1 = "opencode,swe_agentless,cp_reasoning,exercism_sql"


def excluded(paths: list[str]) -> tuple[set, set]:
    """Rows of earlier mixes, never drawn again.

    (file, byte offset) of every train AND eval row their manifests record, plus the sha256 of every document they
    wrote: upstream sets repeat a conversation at another offset (bm-885: one such OpenCode duplicate).
    """
    rows, texts = set(), set()
    for p in paths:
        for name, s in json.loads(Path(p).read_text())["sources"].items():
            for split in ("train", "eval"):
                rows.update((s["files"][fi], off) for fi, off in s.get(split, {}).get("rows", []))
                out = Path(p).parent / f"{name}.{split}.jsonl"
                if out.exists():
                    with out.open() as f:
                        texts.update(text_sha(json.loads(ln)["text"]) for ln in f if ln.strip())
    return rows, texts


def text_sha(text: str) -> str:
    """Identity of a rendered document."""
    return hashlib.sha256(text.encode()).hexdigest()


def offsets(path: Path) -> list[int]:
    out, pos = [], 0
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
        n = len(m)
        while pos < n:
            out.append(pos)
            j = m.find(b"\n", pos)
            pos = n if j < 0 else j + 1
    return out


def row_at(handles, fi, off):
    h = handles[fi]
    h.seek(off)
    return json.loads(h.readline())


def tool_defs(row) -> list | None:
    """OpenCode ships MCP-style {id, description, inputSchema.jsonSchema}; the template wants OpenAI functions."""
    tools = row.get("tools") or []
    out = []
    for t in tools:
        if "function" in t:
            out.append(t)
        elif "id" in t:
            out.append({"type": "function", "function": {
                "name": t["id"], "description": t.get("description", ""),
                "parameters": (t.get("inputSchema") or {}).get("jsonSchema") or {"type": "object"}}})
    return out or None


DEMO_USER = "Change the greeting to be more casual"


def norm_messages(msgs):
    """Tool-call arguments as dicts, reasoning carried once, and the fixed exercism demo dropped."""
    if any(m["role"] == "user" and m.get("content") == DEMO_USER for m in msgs):
        # exercism: every row repeats one system prompt + a demo user/assistant pair (~1k tokens of boilerplate)
        i = next(i for i, m in enumerate(msgs) if m["role"] == "user" and m.get("content") == DEMO_USER)
        msgs = msgs[i + 2:]
    out = []
    for m in msgs:
        m = dict(m)
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", tc)
            if isinstance(fn.get("arguments"), str):
                try:
                    fn["arguments"] = json.loads(fn["arguments"])
                except json.JSONDecodeError:
                    fn["arguments"] = {"raw": fn["arguments"]}
        c = m.get("content")
        if isinstance(c, list):  # OpenCode tool results: AI-SDK parts [{type: tool-result, output: {type, value}}]
            parts = []
            for it in c:
                if isinstance(it, dict) and it.get("type") == "tool-result":
                    v = (it.get("output") or {}).get("value")
                    parts.append(v if isinstance(v, str) else json.dumps(v))
                elif isinstance(it, dict) and it.get("type") == "text":
                    parts.append(it.get("text", ""))
                else:
                    raise ValueError(f"unhandled content part {str(it)[:80]}")
            c = m["content"] = "\n".join(parts)
        elif c is None:
            c = m["content"] = ""
        if m["role"] == "assistant" and isinstance(c, str) and "</think>" in c and m.get("reasoning_content"):
            m["content"] = c.split("</think>", 1)[1].lstrip("\n")  # reasoning is carried once, by reasoning_content
        out.append(m)
    return out


class PromptTooLong(ValueError):
    pass


def render(row, tok, cap: int, drop_system: bool = False):
    msgs = norm_messages(row["messages"])
    if drop_system:
        msgs = [m for m in msgs if m["role"] != "system"]
    text = tok.apply_chat_template(msgs, tools=None if drop_system else tool_defs(row), tokenize=False)
    ids = tok(text, add_special_tokens=False).input_ids
    first = text.find("<|im_start|>assistant")
    if first < 0 or len(tok(text[:first], add_special_tokens=False).input_ids) > cap // 2:
        # a document that is (almost) all prompt, e.g. a repository tree, carries no response to distil
        raise PromptTooLong(f"first assistant turn after {cap // 2} tokens")
    if len(ids) > cap:
        ids = ids[:cap]
        text = tok.decode(ids)
    return text, len(ids)


# ===================================================================================== upstream rows (--fetch-rows)
# The raw Nemotron files were deleted after the mixes were built (207 GB). A mix manifest still names every row it
# used as (file index, byte offset) plus the upstream repo sha, so each row can be fetched again by HTTP range at that
# revision, and proved to be the same row: rendered as build_mix renders it, it must give the mix's own text.
WINDOWS = ".windows.jsonl"


def upstream(f: str) -> tuple[str, str, bool]:
    """(Nemotron set, file path inside its repo, windowed) of a file name under data/nemotron."""
    s, _, path = f.partition("/")
    return s, path.removesuffix(WINDOWS), path.endswith(WINDOWS)


def http_range(u: str, start: int, end: int, tries: int = 12) -> bytes:
    """Bytes [start, end] of ``u`` (inclusive), retried; HTTP 429 waits out HF's rate limit (Retry-After)."""
    import time
    import urllib.error
    import urllib.request

    for t in range(tries):
        try:
            req = urllib.request.Request(u, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                b = r.read()
            if r.status != 206 and start > 0:
                raise OSError(f"range not honoured (HTTP {r.status})")
            return b
        except OSError as e:  # noqa: PERF203 (a transient network error is retried)
            if t == tries - 1:
                raise
            wait = 2 + 4 * t
            if isinstance(e, urllib.error.HTTPError) and e.code == 429:
                wait = max(int(e.headers.get("Retry-After") or 0), 60 * (t + 1))
            time.sleep(wait)
    raise AssertionError


def line_at(get, off: int, size: int, chunk: int = 1 << 20) -> bytes:
    """The whole line starting at ``off`` (without its newline); ``get(start, end)`` reads bytes."""
    buf = b""
    while True:
        end = min(size, off + len(buf) + chunk) - 1
        buf += get(off + len(buf), end)
        j = buf.find(b"\n")
        if j >= 0:
            return buf[:j]
        if end + 1 >= size:
            return buf
        chunk *= 2


def window_map(get, size: int, nwin: int, wbytes: int, chunk: int = 1 << 20) -> list[tuple[int, int, int]]:
    """(local start, upstream start, kept bytes) of each window of omk/datasets/builders/stage_assets.window().

    A window is ``wbytes`` at ``(size - wbytes) * k // (nwin - 1)`` with its partial first and last lines dropped;
    the .windows.jsonl file is the windows back to back, so a local offset maps into exactly one of them.
    """
    out, local = [], 0
    for k in range(nwin):
        start = 0 if nwin == 1 else (size - wbytes) * k // (nwin - 1)
        end = min(size, start + wbytes) - 1
        first = start
        if start > 0:  # the partial first line ends at the first newline at or after start
            first = start + len(line_at(get, start, size, chunk)) + 1
        last = end + 1
        if end + 1 < size:  # the partial last line starts after the last newline at or before end
            c, tail = chunk, b""
            while b"\n" not in tail:
                c = min(c * 2, end + 1 - first)
                tail = get(end + 1 - c, end)
            last = end + 1 - c + tail.rindex(b"\n") + 1
        out.append((local, first, last - first))
        local += last - first
    return out


def to_upstream(wmap: list[tuple[int, int, int]], off: int) -> int:
    """Upstream byte offset of local offset ``off`` in a .windows.jsonl file."""
    for local, first, kept in wmap:
        if local <= off < local + kept:
            return first + off - local
    raise ValueError(f"offset {off} is outside every window")


def conversation(row: dict, drop_system: bool) -> dict:
    """The row as omk_qad.rollouts ``--in`` reads it: the messages build_mix rendered, reasoning in its own field."""
    msgs = norm_messages(row["messages"])
    if drop_system:
        msgs = [m for m in msgs if m["role"] != "system"]
    for m in msgs:
        c = m.get("content")
        if m["role"] == "assistant" and isinstance(c, str) and "</think>" in c and not m.get("reasoning_content"):
            r, _, rest = c.partition("</think>")
            m["reasoning_content"] = r.split("<think>", 1)[-1].strip()
            m["content"] = rest.lstrip("\n")
    return {"messages": msgs, "tools": None if drop_system else tool_defs(row)}


def fetch_rows(a, tok) -> None:
    """Fetch, verify and map every train/eval row of mix ``a.name``; write the row map and conversations."""
    from concurrent.futures import ThreadPoolExecutor

    from stage_assets import HF, WINDOWED, api, url

    mix = a.root / "data" / a.name
    man = json.loads((mix / "manifest.json").read_text())
    if Path(man["tokenizer"]).resolve() != Path(a.tokenizer).resolve():
        raise SystemExit(f"{a.name} was rendered with {man['tokenizer']}, not --tokenizer {a.tokenizer}")
    sizes, wmaps = {}, {}

    def size_of(s, rev, path):
        if (s, rev) not in sizes:
            d = api("datasets", f"nvidia/Nemotron-{s}", rev)
            sizes[(s, rev)] = {x["rfilename"]: x.get("size") for x in d["siblings"]}
        return sizes[(s, rev)][path]

    (mix / "rows").mkdir(exist_ok=True)
    rowmap = []
    for name, src in man["sources"].items():
        if a.only != MIX_V1 and name not in a.only.split(","):
            continue
        for split in ("eval", "train"):
            jobs = []
            for i, (fi, off) in enumerate(src[split]["rows"]):
                s, path, windowed = upstream(src["files"][fi])
                rev = src["upstream_sha"][s]
                u = url("datasets", f"nvidia/Nemotron-{s}", path, rev)
                size = size_of(s, rev, path)
                if windowed:
                    if u not in wmaps:
                        wmaps[u] = window_map(lambda x, y, u=u: http_range(u, x, y), size, *WINDOWED[s])
                    off = to_upstream(wmaps[u], off)
                jobs.append((i, s, rev, path, u, off, size))
            raw = mix / "rows" / f"{name}.{split}.raw.jsonl"
            old = raw.read_bytes().split(b"\n")[:-1] if raw.exists() else []
            if len(old) == len(jobs):  # fetched by an earlier run: proved again below, not fetched again
                lines = old
            else:
                with ThreadPoolExecutor(4) as ex:  # HF rate-limits range reads (429 at 16 in flight)
                    lines = list(ex.map(lambda j: line_at(lambda x, y, u=j[4]: http_range(u, x, y), j[5], j[6]),
                                        jobs))
            texts = [json.loads(ln)["text"] for ln in (mix / f"{name}.{split}.jsonl").open() if ln.strip()]
            if len(texts) != len(lines):
                raise SystemExit(f"{name}.{split}: {len(texts)} documents, the manifest names {len(lines)} rows")
            conv = mix / f"{name}.{split}.conv.jsonl"
            with raw.open("wb") as fr, conv.open("w") as fc:
                for (i, s, rev, path, _, off, _), ln, text in zip(jobs, lines, texts, strict=True):
                    row = json.loads(ln)
                    got, _ = render(row, tok, man["cap"], src["drop_system"])
                    if got != text:
                        raise SystemExit(f"{name}.{split} row {i} ({s}/{path}@{off}): the fetched row does not "
                                         "render to the mix's document")
                    fr.write(ln + b"\n")
                    rid = f"{a.name}/{name}/{split}/{i}"
                    fc.write(json.dumps({"id": rid, **conversation(row, src["drop_system"])}) + "\n")
                    rowmap.append({"id": rid, "repo": f"nvidia/Nemotron-{s}", "rev": rev, "path": path,
                                   "offset": off, "length": len(ln), "sha256": hashlib.sha256(ln).hexdigest(),
                                   "url": f"{HF}/datasets/nvidia/Nemotron-{s}/resolve/{rev}/{path}"})
            print(f"{name}.{split}: {len(lines)} rows fetched and verified", flush=True)
    out = mix / "rowmap.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rowmap))
    print(f"ROWMAP_DONE {out}: {len(rowmap)} rows, sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("/mnt/sdc/omk-model-optimizer"))
    ap.add_argument("--tokenizer", default="/mnt/sdc/omk-model-optimizer/models/Qwen3.5-4B")
    ap.add_argument("--cap", type=int, default=4096)
    ap.add_argument("--margin", type=float, default=1.05)
    ap.add_argument("--n-eval", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--only", default=MIX_V1)
    ap.add_argument("--name", default="mix_v1", help="output mix: ROOT/data/<name>")
    ap.add_argument("--exclude", default="", help="comma list of earlier mix manifests whose rows must not be reused")
    ap.add_argument("--budget", default="", help="train token budget overrides: name=tokens,...")
    ap.add_argument("--preview", type=int, default=0, help="print N rendered docs per source and stop")
    ap.add_argument("--fetch-rows", action="store_true", help="re-fetch the rows mix --name used from HF by byte "
                    "range (raw files deleted), prove each against the mix's text, write rows/, rowmap.jsonl and "
                    "<source>.<split>.conv.jsonl (omk_qad.rollouts --in); --only limits the sources")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    if a.fetch_rows:
        fetch_rows(a, tok)
        return
    skip, seen_text = excluded([p for p in a.exclude.split(",") if p])
    budgets = {k: int(float(v)) for k, v in (kv.split("=") for kv in a.budget.split(",") if kv)}
    out_dir = a.root / "data" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    up = json.loads((a.root / NEM / "manifest.json").read_text())
    man_p = out_dir / "manifest.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else {"sources": {}}
    man.update(tokenizer=a.tokenizer, cap=a.cap, margin=a.margin, seed=a.seed,
               **({"excludes": a.exclude.split(",")} if skip else {}))
    for name in a.only.split(","):
        files, budget = SOURCES[name]
        budget = budgets.get(name, budget)
        rng = random.Random(f"{a.seed}:{name}")
        pool = []
        for fi, f in enumerate(files):
            p = a.root / NEM / f
            rows = offsets(p)
            pool += [(fi, o) for o in rows if (f, o) not in skip]
            print(f"{name}: {f} rows={len(rows)} usable={len(pool)}", flush=True)
        rng.shuffle(pool)
        handles = [open(a.root / NEM / f, "rb") for f in files]
        if a.preview:
            for fi, o in pool[: a.preview]:
                try:
                    t, n = render(row_at(handles, fi, o), tok, a.cap, name in DROP_SYSTEM)
                except PromptTooLong as e:
                    print(f"--- {name} {Path(files[fi]).name}@{o} SKIP {e}")
                    continue
                marks = {k: t.count(k) for k in ("<|im_start|>system", "<|im_start|>user", "<|im_start|>assistant",
                                                  "<think>", "</think>", "<tools>", "<tool_call>", "<tool_response>")}
                print(f"--- {name} {Path(files[fi]).name}@{o} tokens={n} {marks}")
                print("    head:", repr(t[:160]))
            continue
        stats = {}
        it = iter(pool)
        for split, target in (("eval", None), ("train", int(budget * a.margin))):
            path = out_dir / f"{name}.{split}.jsonl"
            toks = docs = bad = long_prompt = dup = 0
            used = []
            with open(path, "w") as o:
                for fi, off in it:
                    try:
                        t, n = render(row_at(handles, fi, off), tok, a.cap, name in DROP_SYSTEM)
                    except PromptTooLong:
                        long_prompt += 1
                        continue
                    except Exception as e:
                        bad += 1
                        if bad <= 3:
                            print(f"   skip {files[fi]}@{off}: {type(e).__name__}: {str(e)[:120]}", flush=True)
                        continue
                    if seen_text and text_sha(t) in seen_text:  # an earlier mix wrote this same document
                        dup += 1
                        continue
                    o.write(json.dumps({"text": t}) + "\n")
                    toks += n
                    docs += 1
                    used.append([fi, off])
                    if (target is None and docs >= a.n_eval) or (target is not None and toks >= target):
                        break
            if target is not None and toks < target:
                raise SystemExit(f"{name}: only {toks} tokens < {target}")
            stats[split] = {"docs": docs, "tokens": toks, "skipped": bad, "skipped_long_prompt": long_prompt,
                            **({"skipped_duplicate": dup} if seen_text else {}),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": used}
            print(f"{name}.{split}: docs={docs} tokens={toks} skipped={bad} long_prompt={long_prompt} duplicate={dup}",
                  flush=True)
        upstream = {s: up[s.split("/")[0]]["sha"] for s in {f.split("/")[0] for f in files} if s in up}
        man["sources"][name] = {"files": files, "budget": budget, "drop_system": name in DROP_SYSTEM,
                                "upstream_sha": upstream, **stats,
                                **({"excluded_rows": sum(f in files for f, _ in skip)} if skip else {})}
        man_p.write_text(json.dumps(man, indent=1))
        for h in handles:
            h.close()
    print("MIX_DONE", flush=True)


if __name__ == "__main__":
    main()
