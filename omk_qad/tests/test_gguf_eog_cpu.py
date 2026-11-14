# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D103
"""CPU tests of the GGUF export's Gemma-4 completeness: checkpoint buffers and end-of-generation metadata.

The bm-892 IQ2 export of gemma-4-26B-A4B (2026-11-14) generated garbage in llama.cpp although every tensor it wrote
verified: the exporter handed the converter named_parameters() only, so the 30 ``layer_scalar`` buffers
(blk.<n>.layer_output_scale) never reached the GGUF, which llama.cpp loads as optional. It also kept the converter's
eos 1 with no eot (Gemma 4 ends a turn with <turn|>; ollama's gemma4 parser reads eot).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_DEFAULT_WORK = "/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models/omk_qad_work/pytest"
WORK = Path(os.environ.get("OMK_QAD_WORK", _DEFAULT_WORK)) / "gguf_eog"
LLAMA_CPP = Path(os.environ.get("OMK_LLAMA_CPP", "/opt/llama.cpp"))
CPU_ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "WANDB_MODE": "disabled"}


def _gguf():
    for p in (str(LLAMA_CPP / "gguf-py"), str(LLAMA_CPP)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import gguf

    return gguf


@pytest.fixture(scope="module")
def g4():
    """Tiny gemma4_moe with DISTINCT per-layer layer_scalar values (1.0 everywhere would hide a dropped scale)."""
    from safetensors.torch import load_file, save_file

    from omk_qad.layout import build_layout
    from omk_qad.tiny import make_tiny

    WORK.mkdir(parents=True, exist_ok=True)
    model = WORK / "tiny_g4_scaled"
    if not (model / "config.json").exists():
        make_tiny("gemma4_moe", model, seed=0)
        for shard in sorted(model.glob("*.safetensors")):
            sd = load_file(str(shard))
            for k in sd:
                if k.endswith("layer_scalar"):
                    layer = int(k.split(".layers.")[1].split(".")[0])
                    sd[k] = torch.full_like(sd[k], 0.25 + 0.1 * layer)
            save_file(sd, str(shard), metadata={"format": "pt"})
    lay = WORK / "layout_g4.pt"
    if not lay.exists():
        torch.save(build_layout(model, LLAMA_CPP, "IQ2_M", work=WORK / "lay_g4"), lay)
    scalars = {}
    for shard in sorted(model.glob("*.safetensors")):
        for k, v in load_file(str(shard)).items():
            if k.endswith("layer_scalar"):
                scalars[int(k.split(".layers.")[1].split(".")[0])] = v.float().reshape(-1).numpy()
    assert scalars and len({float(v[0]) for v in scalars.values()}) == len(scalars)
    return model, lay, scalars


def test_gemma4_export_carries_layer_scalars_and_eog(g4):
    model, lay, scalars = g4
    out = WORK / "ptq_g4.gguf"
    for f in (out, out.with_suffix(".export.json")):
        f.unlink(missing_ok=True)
    proc = subprocess.run([sys.executable, "-m", "omk_qad.train", "--phase", "export", "--model", str(model),
                           "--layout", str(lay), "--device", "cpu", "--gguf", str(out), "--latent-min", "65536"],
                          capture_output=True, text=True, env=CPU_ENV, check=False)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    rep = json.loads(out.with_suffix(".export.json").read_text())
    assert rep["verify_failures"] == [] and rep["type_mismatches"] == []
    assert rep["source_not_exported"] == []
    assert len(rep["buffers"]) == len(scalars)
    eog = rep["eog"]
    assert eog["ok"] and eog["checked"], eog
    assert (eog["eos_text"], eog["eot_text"]) == ("<turn|>", "<eos>")
    assert eog["converter"]["eos"] != eog["eos"]  # the converter's own pick was the defect being corrected

    gguf = _gguf()
    # general.file_type is the layout's ftype, not the converter's ALL_F32 (0) that xollama showed as "F32"
    want_ft = int(gguf.LlamaFileType[f"MOSTLY_{rep['ftype'].upper()}"])
    assert want_ft != 0 and rep["file_type"] == {"want": want_ft, "read": want_ft}, rep["file_type"]
    tensors = {t.name: t for t in gguf.GGUFReader(out).tensors}
    for layer, want in scalars.items():
        got = np.asarray(tensors[f"blk.{layer}.layer_output_scale.weight"].data, dtype=np.float32).reshape(-1)
        assert np.array_equal(got, want), layer


def test_eog_check_rejects_converter_defaults(g4, tmp_path):
    """check() fails a Gemma-4 GGUF with the converter's eos 1 and no eot, and passes it once stamped."""
    from omk_qad import gguf_eog

    gguf = _gguf()
    src = WORK / "ptq_g4.gguf"
    if not src.exists():
        pytest.skip("needs the export test's GGUF")
    bad = tmp_path / "eos1.gguf"
    subprocess.run([sys.executable, str(LLAMA_CPP / "gguf-py/gguf/scripts/gguf_new_metadata.py"),
                    "--remove-metadata", "tokenizer.ggml.eot_token_id", "--special-token-by-id", "eos", "1",
                    "--force", str(src), str(bad)], check=True, capture_output=True, env=CPU_ENV)
    rep = gguf_eog.check(bad, gguf)
    assert not rep["ok"]
    assert any("eos 1 is '<eos>'" in p for p in rep["problems"]) and "no eot_token_id" in rep["problems"]
    assert gguf_eog.check(src, gguf)["ok"]
