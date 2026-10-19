# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D103
"""End-to-end CPU tests: tiny models of both QAD targets -> census -> PTQ -> GGUF -> verify -> llama.cpp load.

Artifacts are ~1 GB, so they go to $OMK_QAD_WORK (persistent disk), never to pytest's tmp_path (/tmp is tmpfs
on solidpc). Needs the HF cache (or network) for the reference configs/tokenizers and a built llama.cpp.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_DEFAULT_WORK = "/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models/omk_qad_work/pytest"
WORK = Path(os.environ.get("OMK_QAD_WORK", _DEFAULT_WORK))
LLAMA_CPP = Path(os.environ.get("OMK_LLAMA_CPP", "/opt/llama.cpp"))
CPU_ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}


def _run(*args: str) -> str:
    proc = subprocess.run([sys.executable, "-m", *args], capture_output=True, text=True, env=CPU_ENV, check=False)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return proc.stdout


@pytest.fixture(scope="module")
def tiny_models():
    from omk_qad.tiny import make_tiny

    WORK.mkdir(parents=True, exist_ok=True)
    out = {}
    for kind in ("moe", "dense"):
        d = WORK / f"tiny_{kind}"
        if not (d / "config.json").exists():
            make_tiny(kind, d)
        out[kind] = d
    return out


@pytest.mark.parametrize("kind", ["moe", "dense"])
def test_census_covers_every_parameter(tiny_models, kind):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from omk_qad.census import build_quant_cfg, run_census

    config = AutoConfig.from_pretrained(tiny_models[kind]).get_text_config()
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    census = run_census(model)
    assert sum(e.params for e in census.entries) == census.model_params
    classes = census.by_class()
    two_bit = "routed_experts" if kind == "moe" else "dense_mlp"
    assert {e.fmt for e in classes[two_bit]} == {"iq2_xxs"}
    assert {e.fmt for e in classes["attention"]} == {"q8_0"}
    assert {e.fmt for e in classes["linear_attn"]} == {"q8_0"}
    if kind == "moe":
        assert len(classes["router"]) == config.num_hidden_layers
        assert {e.fmt for e in classes["shared_experts"]} == {"iq2_xxs"}
    assert not [e for e in census.entries if e.cls == "other" and e.kind != "conv"]
    cfg = build_quant_cfg(census)
    assert cfg["quant_cfg"][0] == {"quantizer_name": "*", "enable": False}


@pytest.mark.parametrize(
    ("kind", "policy", "two_bit_type"),
    [
        ("moe", "", "IQ2_XXS"),
        ("dense", "", "IQ2_XXS"),
        ("moe", "routed_experts=iq2_s,shared_experts=iq2_s", "IQ2_S"),
    ],
)
def test_ptq_export_verify_and_load(tiny_models, kind, policy, two_bit_type):
    tag = f"e2e_{kind}_{two_bit_type.lower()}"
    packed, gguf_path = WORK / tag, WORK / f"{tag}.gguf"
    args = ["omk_qad.ptq", "--model", str(tiny_models[kind]), "--out", str(packed), "--calib-seqs", "2", "--seq", "64"]
    _run(*args, *(["--policy", policy] if policy else []))
    _run("omk_qad.gguf_export", "--src", str(packed), "--out", str(gguf_path), "--llama-cpp", str(LLAMA_CPP))
    report = json.loads(gguf_path.with_suffix(".export.json").read_text())
    assert report["verify_failures"] == []
    n_layers = 4
    n_two_bit = n_layers * (6 if kind == "moe" else 3)  # moe: gate/up/down exps + shared; dense: gate/up/down
    assert report["type_counts"].get(two_bit_type) == n_two_bit
    assert report["verified"] == sum(v for k, v in report["type_counts"].items() if k != "F32") - 1  # minus token_embd
    unmatched = report["unmatched_float"]
    expected_float = ("ssm_conv1d", "ssm_alpha", "ssm_beta", "ffn_gate_inp")
    assert all(any(s in n for s in expected_float) for n in unmatched), unmatched

    text = WORK / "ppl_text.txt"
    if not text.exists():
        text.write_text((LLAMA_CPP / "README.md").read_text()[:60000])
    proc = subprocess.run(
        [str(LLAMA_CPP / "build/bin/llama-perplexity"), "-m", str(gguf_path), "-f", str(text), "-c", "128",
         "--chunks", "2", "-t", "8", "-ngl", "0"],
        capture_output=True, text=True, env=CPU_ENV, check=False,
    )
    assert proc.returncode == 0 and "Final estimate" in proc.stdout + proc.stderr, proc.stderr[-2000:]
