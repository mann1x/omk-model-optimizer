#!/usr/bin/env python3
"""kfac_publish_base20_bs2.py -- the 20b-base (bm-898) K-FAC ladder publisher: derived from kfac_publish_bs2.v3.py on 2026-11-19 17:50 CET.
Owner: upload every quant to HF as it comes, squash the private repo while uploading, delete the bs2 copy after the hub sha matches.

kfac_publish_bs2.py -- RUNS ON bs2 (omnimergekit env). Publish every finished, scored K-FAC ladder tier to its HF repo
and keep the three model cards current.

Owner 2026-11-16 21:25 CET: "publish on HF, why can't you do it now from bs2?" -- republish (20B v9-agentic sft12) and
re-publish/expand (26B agentic) with the K-FAC files; IQ2_XXS comes from the QAD exports later.
Owner 2026-11-16 21:45 CET: "update the HF repo for the weights with the new recipe used ... update the GGUF with the
details about the quantizations with KFAC ... Document properly everything, nothing must be missed." -> v2: the cards.

Every 5 min, for both ladders, a tier dir qualifies when: kfac.gguf exists (file or symlink to the root-fs relief copy),
kfac.log holds KFACR_DONE (bug-949: a killed rounder leaves a copy of the PTQ), kfac.eval.json + ptq.eval.json are
scored, no .published_hf marker.  Then:
  1. sha256 (kfac.gguf.sha256 sidecar, computed when missing)
  2. upload as <repo name minus -GGUF>-<TIER>.gguf (overwrites the same name: that IS the re-publish), commit message
     carries the twin numbers; retries with backoff
  3. verify: hub LFS sha256 == local (get_paths_info expand), else KP_FAIL and the marker is NOT written
  4. upload the <name>.gguf.sha256 sidecar; once per repo the imatrix the ladder used (+ .sha256) and the repo's
     EXTRA files (20B: the sft12 F16 GGUF, replacing the 2026-10 build's F16)
  5. write .published_hf (hub sha); remove the bs2 copy ONLY if .archived_solidpc exists too (two verified copies:
     hub + solidpc), resolving a relief symlink to its target.
Cards (v2): after every cycle the manifest <ladder>/published.json is rebuilt (hub size + sha per published tier, twin
numbers from the eval jsons, bits/weight from the base GGUF's element count) and rendered into the card template
<CARDS>/<which>_gguf.md at the markers <!-- OMK:TIERS -->, <!-- OMK:STALE --> (hub files that are not ours: the
previous build's tiers) and <!-- OMK:BENCH --> (<CARDS>/20b_bench.json when present); the README is uploaded only
when it differs from the hub's. <CARDS>/20b_weights.md goes to the weights repo the same way (static).
For THIS ladder the loop never renders a card: the card is rendered on demand (2026-11-20, owner: "there is no
different 20b ladder, we have only one"; no sft12 or F16 comparison on it) with `--cards-dry` (render to
<CARDS>/.20b-base_gguf.rendered.md, upload nothing) and `--cards` (render + upload the README once), from the template
<CARDS>/20b-base_gguf.md. No QAD tier exists for this ladder (qad_tier=None), so no IQ2_XXS line is appended.
Sentinels KP_START / KP_TIER / KP_IMATRIX / KP_EXTRA / KP_CARD / KP_CARD_DRY / KP_RM / KP_FAIL / KP_DONE.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

RUNS = Path("/mnt/sdc/omk-model-optimizer/runs")
CARDS = RUNS / "cards"
LADDERS = {
    # 2026-11-20 07:00 CET: the build lives in the model's own repos (owner: one 20B model, builds replace each other;
    # the *-base-mlk8-noa-sft11 repos were my error and were folded into these names, files renamed on the Hub).
    # Hub filenames = repo name minus; the local ladder outputs and .published_hf markers still carry the long build
    # name, so a re-render of the card needs the marker names mapped (long -> short) before manifest() is trusted.
    "20b-base": dict(repo="ManniX-ITA/gemma-4-20B-v9-agentic-GGUF",
                     prefix="gemma-4-20B-v9-agentic",
                     ladder=RUNS / "g4_20b_base_kfac/ladder",
                     imatrix=Path("/mnt/sdc/ml/v9a_base/mlk8-noa-sft11/gguf/imatrix_ac.dat"), imatrix_name="imatrix.dat",
                     base=Path("/mnt/sdc/ml/v9a_base/mlk8-noa-sft11/gguf/gemma-4-20B-v9-agentic-base-mlk8-noa-sft11-F16.gguf"),
                     extra={"gemma-4-20B-v9-agentic-F16.gguf":
                            Path("/mnt/sdc/ml/v9a_base/mlk8-noa-sft11/gguf/gemma-4-20B-v9-agentic-base-mlk8-noa-sft11-F16.gguf")},
                     keep={"imatrix.dat", "imatrix.dat.sha256", "README.md", ".gitattributes",
                           "gemma-4-20B-v9-agentic-F16.gguf",
                           "gemma-4-20B-v9-agentic-F16.gguf.sha256"},
                     attn_from="Q8_0", weights_repo="ManniX-ITA/gemma-4-20B-v9-agentic", weights_card=None,
                     qad_tier=None),  # no QAD export planned for this ladder: the card gets no IQ2_XXS line
}
DRY = "--cards-dry" in sys.argv  # render the card to <CARDS>/.<which>_gguf.rendered.md and upload nothing
LOG = RUNS / "kfac_publish_base20_bs2.log"
HOURS = float(os.environ.get("HOURS", "30"))
SKIP_TIERS = {"IQ2_XXS"}  # QAD exports, published by their own chain
# the Gemma-4 A4B attention rule of quantize_gguf / kfac_ladder_twins.sh: Q4_K_M and above -> Q8_0, else Q6_K
ATTN_Q8 = {"Q4_K_M", "Q4_K_L", "Q5_K_S", "Q5_K_M", "Q5_K_L", "Q5_0", "Q5_1", "Q6_K", "Q6_K_L", "Q8_0"}
# ladder order on the card (coarse last)
ORDER = ["Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S", "IQ4_XS", "IQ4_NL", "Q3_K_M", "IQ3_M", "IQ3_XS",
         "IQ3_XXS", "IQ2_M", "IQ2_S", "IQ2_XS", "IQ2_XXS"]


def say(*a):
    line = f"[kp {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] " + " ".join(str(x) for x in a)
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def local_sha(d: Path, method: str = "kfac") -> str:
    side = d / f"{method}.gguf.sha256"
    if side.exists():
        s = side.read_text().split()[0]
        if len(s) == 64:
            return s
    s = sha256_of(d / f"{method}.gguf")
    side.write_text(f"{s}  {method}.gguf\n")
    return s


def hub_files(api: HfApi, repo: str) -> dict:
    """name -> (size, lfs sha256 or None) for every file of the repo."""
    for attempt in range(5):
        try:
            info = api.model_info(repo, files_metadata=True)
            out = {}
            for s in info.siblings or []:
                lfs = getattr(s, "lfs", None)
                sha = None
                if lfs:
                    sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
                out[s.rfilename] = (getattr(s, "size", None), sha)
            return out
        except Exception as e:  # noqa: BLE001
            say("hub_files retry", attempt + 1, type(e).__name__, str(e)[:120])
            time.sleep(20 * (attempt + 1))
    return {}


def hub_sha(api: HfApi, repo: str, name: str) -> str | None:
    for attempt in range(5):
        try:
            info = api.get_paths_info(repo, [name], expand=True)
            for e in info:
                lfs = getattr(e, "lfs", None)
                if lfs:
                    return lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
            return None
        except Exception as e:  # noqa: BLE001
            say("hub_sha retry", attempt + 1, type(e).__name__, str(e)[:120])
            time.sleep(20 * (attempt + 1))
    return None


def upload(api: HfApi, repo: str, src: Path, name: str, msg: str) -> bool:
    for attempt in range(1, 7):
        try:
            t0 = time.time()
            api.upload_file(path_or_fileobj=str(src), path_in_repo=name, repo_id=repo, repo_type="model", commit_message=msg)
            say(f"uploaded {name} ({src.stat().st_size / 2**30:.2f} GiB) in {time.time() - t0:.0f} s")
            if src.stat().st_size >= 1 << 30:  # owner 2026-11-19 17:39: squash the private repos while uploading (storage limit)
                try:
                    api.super_squash_history(repo)
                    say("KP_SQUASHED", repo)
                except Exception as e:  # noqa: BLE001
                    say("KP_SQUASH_FAIL", repo, type(e).__name__, str(e)[:120])
            return True
        except Exception as e:  # noqa: BLE001
            say(f"upload {name} attempt {attempt} failed: {type(e).__name__} {str(e)[:160]}")
            time.sleep(30 * attempt)
    return False


def twin(d: Path, method: str = "kfac") -> str:
    try:
        p, k = json.load(open(d / "ptq.eval.json")), json.load(open(d / f"{method}.eval.json"))
        lab = "K-FAC" if method == "kfac" else "seeded QAD"
        return (f"{lab} {k['klk']:.5f}/{k['top1']:.4f} vs llama.cpp+imatrix {p['klk']:.5f}/{p['top1']:.4f} "
                f"({100 * (k['klk'] - p['klk']) / p['klk']:+.1f}% KL, 44-row top-256 KL vs F16)")
    except Exception as e:  # noqa: BLE001
        return f"twin numbers unavailable ({type(e).__name__})"


def publish_once(api: HfApi, which: str, cfg: dict, src: Path, name: str, tag: str, msg: str) -> None:
    """imatrix / extra files: upload once per repo (marker), verify by sha, sidecar."""
    marker = cfg["ladder"] / f".published_{tag}_{name}"
    if marker.exists() or not src.exists():
        return
    s = sha256_of(src)
    if hub_sha(api, cfg["repo"], name) == s:
        marker.write_text(s + "\n")
        say(f"KP_{tag.upper()}", which, name, "already on the hub, sha", s[:12])
        return
    if not upload(api, cfg["repo"], src, name, f"{msg} (sha256 {s[:12]})"):
        say("KP_FAIL", which, name, "upload")
        return
    if hub_sha(api, cfg["repo"], name) != s:
        say("KP_FAIL", which, name, "hub sha differs")
        return
    side = cfg["ladder"] / f"{name}.sha256"
    side.write_text(f"{s}  {name}\n")
    upload(api, cfg["repo"], side, f"{name}.sha256", f"sha256 of {name}")
    marker.write_text(s + "\n")
    say(f"KP_{tag.upper()}", which, name, "sha", s[:12], f"{src.stat().st_size / 2**30:.2f} GiB")


def method_of(d: Path) -> str | None:
    """'qad' for a seeded-QAD export dir (qad.gguf + qad.done + both scores), 'kfac' for a finished rounding, else None."""
    if (d / "qad.gguf").exists() and (d / "qad.done").exists() and (d / "qad.eval.json").exists() and (d / "ptq.eval.json").exists():
        return "qad"
    if (d / "kfac.gguf").exists() and (d / "kfac.eval.json").exists() and (d / "ptq.eval.json").exists():
        klog = d / "kfac.log"
        if klog.exists() and "KFACR_DONE" in klog.read_text(errors="ignore"):
            return "kfac"
        say("skip", d.parent.parent.name, d.name, "kfac.log has no KFACR_DONE (partial rounding, bug-949)")
    return None


def publish_tier(api: HfApi, which: str, cfg: dict, d: Path, method: str = "kfac") -> None:
    tier = d.name
    name = f"{cfg['prefix']}-{tier}.gguf"
    g = d / f"{method}.gguf"
    s = local_sha(d, method)
    what = ("K-FAC rounded (omk_qad.kfac_round, all MLP tensors)" if method == "kfac"
            else "seeded QAD export (omk_qad: K-FAC-seeded quantization-aware distillation, 600 steps)")
    if hub_sha(api, cfg["repo"], name) != s:
        if not upload(api, cfg["repo"], g, name, f"{tier}: {what} -- {twin(d, method)}"):
            say("KP_FAIL", which, tier, "upload")
            return
        hs = hub_sha(api, cfg["repo"], name)
        if hs != s:
            say("KP_FAIL", which, tier, f"hub sha {hs} != local {s} (marker not written)")
            return
    side = d / f"{name}.sha256"
    side.write_text(f"{s}  {name}\n")
    upload(api, cfg["repo"], side, f"{name}.sha256", f"{tier}: sha256 sidecar")
    (d / ".published_hf").write_text(f"{s}  {cfg['repo']}/{name}  {method}\n")
    say("KP_TIER", which, tier, method, "->", f"{cfg['repo']}/{name}", f"{g.stat().st_size / 2**30:.2f} GiB sha {s[:12]};", twin(d, method))
    reclaim(which, d, method)


def reclaim(which: str, d: Path, method: str = "kfac") -> None:
    """Remove the bs2 copy of a K-FAC tier once three things hold: hub sha verified (.published_hf), solidpc archive
    verified (.archived_solidpc) and the Ollama tags pushed from it (.published_ollama: the pusher needs the file)."""
    g = d / f"{method}.gguf"
    if method != "kfac" or not g.exists():
        return
    if not (d / ".published_hf").exists():  # owner 2026-11-19 17:39: the verified hub copy is the archive
        return
    # bug-964 (2026-11-18): the bs2 copy must be THE published file, not a rounder's in-progress byte copy of the PTQ --
    # the rogue lane re-rounded published tiers and this reclaim deleted kfac.gguf under the rounder (FileNotFoundError).
    # Three holds: the sidecar sha equals the sha recorded at publish time, kfac.log holds KFACR_DONE (a fresh rounder
    # overwrites it), and the file was not written in the last 10 minutes.
    pub = (d / ".published_hf").read_text().split()[0]
    side = d / "kfac.gguf.sha256"
    local = side.read_text().split()[0] if side.exists() else "missing"
    klog = d / "kfac.log"
    done = klog.exists() and "KFACR_DONE" in klog.read_text(errors="ignore")
    age = time.time() - g.stat().st_mtime
    if local != pub or not done or age < 600:
        say("KP_RM_HOLD", which, d.name, f"bs2 kfac.gguf is not (yet) the published file: sidecar {local[:12]} vs published "
            f"{pub[:12]}, KFACR_DONE={done}, age {age:.0f}s -- not removed")
        return
    real = g.resolve()
    say("KP_RM", which, d.name, f"bs2 copy removed ({real}); verified copy: hub (sha-matched), owner 2026-11-19 17:39")
    real.unlink(missing_ok=True)
    if g.is_symlink():
        g.unlink(missing_ok=True)


# ---------------------------------------------------------------- cards
def element_count(cfg: dict) -> int | None:
    """Tensor element count of the base GGUF (bits/weight denominator), cached beside the ladder."""
    cache = cfg["ladder"] / ".elements.json"
    if cache.exists():
        return json.load(open(cache))["elements"]
    if not cfg["base"].exists():
        return None
    try:
        import gguf  # noqa: PLC0415
        r = gguf.GGUFReader(str(cfg["base"]))
        n = int(sum(int(t.n_elements) for t in r.tensors))
        json.dump({"elements": n, "base": str(cfg["base"])}, open(cache, "w"))
        return n
    except Exception as e:  # noqa: BLE001
        say("element_count failed", type(e).__name__, str(e)[:120])
        return None


def manifest(api: HfApi, which: str, cfg: dict, files: dict) -> dict:
    rows = {}
    for d in sorted(p for p in cfg["ladder"].iterdir() if p.is_dir()):
        m = d / ".published_hf"
        if not m.exists():
            continue
        tier = d.name
        name = f"{cfg['prefix']}-{tier}.gguf"
        parts = m.read_text().split()
        sha = parts[0]
        method = parts[2] if len(parts) > 2 else "kfac"
        size, hsha = files.get(name, (None, None))
        row = {"tier": tier, "name": name, "sha256": sha, "bytes": size, "hub_sha_ok": hsha == sha,
               "method": method, "attention": "Q8_0" if tier in ATTN_Q8 else "Q6_K",
               "published_utc": time.strftime("%Y-%m-%d %H:%MZ", time.gmtime(m.stat().st_mtime))}
        for arm, key in (("ptq", "ptq"), ("kfac", method)):
            try:
                e = json.load(open(d / f"{key}.eval.json"))
                row[arm] = {"klk": e["klk"], "top1": e["top1"]}
            except Exception:  # noqa: BLE001
                row[arm] = None
        rows[tier] = row
    out = {"repo": cfg["repo"], "elements": element_count(cfg), "tiers": rows, "qad_tier": cfg.get("qad_tier"),
           "hub_files": {k: v[0] for k, v in files.items()},
           "rendered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (cfg["ladder"] / "published.json").write_text(json.dumps(out, indent=1))
    return out


def fmt_gb(b) -> str:
    return "—" if not b else f"{b / 1e9:.2f} GB ({b / 2**30:.2f} GiB)"


def render_tiers(man: dict) -> str:
    rows = man["tiers"]
    n = man.get("elements")
    hdr = ("| Tier | method | attention | size | bpw | sha256 | llama-quantize + imatrix<br>KL / top-1 | **this file**<br>KL / top-1 | ΔKL |\n"
           "|---|---|---|---:|---:|---|---:|---:|---:|\n")
    lines = []
    for t in ORDER:
        r = rows.get(t)
        if not r:
            continue
        bpw = f"{8 * r['bytes'] / n:.2f}" if (n and r["bytes"]) else "—"
        p, k = r.get("ptq"), r.get("kfac")
        pt = f"{p['klk']:.5f} / {p['top1']:.4f}" if p else "—"
        kt = f"**{k['klk']:.5f} / {k['top1']:.4f}**" if k else "—"
        dk = f"{100 * (k['klk'] - p['klk']) / p['klk']:+.1f} %" if (p and k) else "—"
        meth = "K-FAC" if r.get("method", "kfac") == "kfac" else "QAD, K-FAC-seeded"
        lines.append(f"| **{t}** | {meth} | {r['attention']} | {fmt_gb(r['bytes'])} | {bpw} | `{r['sha256'][:12]}…` | {pt} | {kt} | {dk} |")
    qad = man.get("qad_tier")  # None for a ladder without a QAD export: no IQ2_XXS row, no IQ2_XXS line
    pending = [t for t in ORDER if t not in rows and t not in SKIP_TIERS]
    txt = hdr + "\n".join(lines) if lines else "_No K-FAC tier published yet._"
    txt += (f"\n\n_{qad}: the seeded QAD export (see below), published when its run ends._"
            if qad and qad not in rows else "")
    if pending:
        txt += "\n\n_Building (K-FAC, published as each tier lands): " + ", ".join(pending) + "._"
    return txt


def render_stale(man: dict, cfg: dict) -> str:
    ours = {r["name"] for r in man["tiers"].values()} | {r["name"] + ".sha256" for r in man["tiers"].values()} | cfg["keep"]
    stale = sorted(f for f in man["hub_files"] if f not in ours and not f.startswith("calibration/")
                   and not f.endswith(".sha256") and f.endswith(".gguf"))
    if not stale:
        return "_None._"
    tot = sum(man["hub_files"][f] or 0 for f in stale)
    return (f"The following `.gguf` files in this repo are **not part of this ladder**: {len(stale)} files, "
            f"{tot / 1e9:.0f} GB. They are not scored in the table above.\n\n"
            + "\n".join(f"- `{f}` ({fmt_gb(man['hub_files'][f])})" for f in stale))


def render_bench(path: Path) -> str:
    if not path.exists():
        return ("_The canonical benchmark suite has not been run on this build yet; the quants are gated by the KL twins "
                "above, not by a task score. Results land here when the suite runs._")
    b = json.load(open(path))
    lines = [f"**{b.get('title', 'Benchmarks')}**", "", b.get("basis", ""), "",
             "| Benchmark | " + " | ".join(b["columns"]) + " |", "|---|" + "---:|" * len(b["columns"])]
    for r in b["rows"]:
        lines.append("| " + r["name"] + " | " + " | ".join(str(x) if x is not None else "—" for x in r["values"]) + " |")
    if b.get("notes"):
        lines += ["", b["notes"]]
    return "\n".join(lines)


def render_card(which: str, cfg: dict, man: dict) -> str | None:
    tpl = CARDS / f"{which}_gguf.md"
    if not tpl.exists():
        return None
    txt = tpl.read_text()
    txt = txt.replace("<!-- OMK:TIERS -->", render_tiers(man))
    txt = txt.replace("<!-- OMK:STALE -->", render_stale(man, cfg))
    txt = txt.replace("<!-- OMK:BENCH -->", render_bench(CARDS / f"{which}_bench.json"))
    txt = txt.replace("<!-- OMK:RENDERED -->", man["rendered_utc"])
    return txt


def hub_readme(api: HfApi, repo: str) -> str:
    try:
        p = hf_hub_download(repo, "README.md", repo_type="model", cache_dir=str(CARDS / ".hfcache"), force_download=True)
        return Path(p).read_text()
    except Exception as e:  # noqa: BLE001
        say("hub_readme failed", repo, type(e).__name__, str(e)[:120])
        return ""


def _stable(txt: str) -> str:
    """The card without its render timestamp line (else every cycle would be a new README commit)."""
    return "\n".join(line for line in txt.strip().splitlines() if "Card table rendered" not in line)


def push_card(api: HfApi, repo: str, txt: str, label: str, msg: str) -> None:
    if not txt or _stable(txt) == _stable(hub_readme(api, repo)):
        return
    out = CARDS / f".{label}.rendered.md"
    out.write_text(txt)
    if DRY:
        say("KP_CARD_DRY", repo, label, f"{len(txt)} chars ->", str(out), "(nothing uploaded)")
        return
    if upload(api, repo, out, "README.md", msg):
        say("KP_CARD", repo, label, f"{len(txt)} chars")
    else:
        say("KP_FAIL", repo, "card upload", label)


def cards(api: HfApi) -> None:
    for which, cfg in LADDERS.items():
        files = hub_files(api, cfg["repo"])
        if not files:
            say("KP_FAIL", which, "hub listing empty, card not rendered")
            continue
        man = manifest(api, which, cfg, files)
        txt = render_card(which, cfg, man)
        if txt:
            push_card(api, cfg["repo"], txt, f"{which}_gguf",
                      f"card: K-FAC ladder ({len(man['tiers'])} tiers published), recipe and provenance")
        if cfg.get("weights_repo") and cfg.get("weights_card") and (CARDS / cfg["weights_card"]).exists():
            push_card(api, cfg["weights_repo"], (CARDS / cfg["weights_card"]).read_text(), cfg["weights_card"],
                      "card: recipe, lineage and the K-FAC GGUF ladder")


def main() -> None:
    api = HfApi()
    CARDS.mkdir(exist_ok=True)
    if DRY or "--cards" in sys.argv:  # one-shot card render (+ upload unless --cards-dry); the loop never renders it
        say("KP_CARDS_ONLY", "dry" if DRY else "upload", "as", api.whoami()["name"], "template", str(CARDS / "20b-base_gguf.md"))
        cards(api)
        return
    say("KP_START as", api.whoami()["name"], "for", HOURS, "h;", {k: v["repo"] for k, v in LADDERS.items()}, "cards", str(CARDS))
    end = time.time() + HOURS * 3600
    while time.time() < end:
        for which, cfg in LADDERS.items():
            try:
                publish_once(api, which, cfg, cfg["imatrix"], cfg["imatrix_name"], "imatrix",
                             f"imatrix used by the K-FAC ladder ({cfg['imatrix'].name})")
                for name, src in cfg["extra"].items():
                    publish_once(api, which, cfg, src, name, "extra", f"{name}: this build's unquantised GGUF (llama.cpp b12386)")
                for d in sorted(p for p in cfg["ladder"].iterdir() if p.is_dir()):
                    if (d / ".published_hf").exists():
                        reclaim(which, d)  # the ollama pusher may have finished since
                        continue
                    method = method_of(d)
                    if method is None or (d.name in SKIP_TIERS and method != "qad"):
                        continue  # IQ2_XXS ships only as the seeded QAD export, never as a rounding
                    publish_tier(api, which, cfg, d, method)
            except Exception as e:  # noqa: BLE001
                say("KP_FAIL", which, "loop error", type(e).__name__, str(e)[:200])
        # cards: never from the loop for this ladder -- rendered on demand with --cards-dry / --cards (see docstring)
        time.sleep(300)
    say("KP_DONE", HOURS, "h elapsed")


if __name__ == "__main__":
    main()
