# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

import modelopt.torch.quantization.ggml.q8_0 as q8_0_module
from modelopt.torch.quantization.extensions import get_cuda_ext_ggml
from modelopt.torch.quantization.ggml.q8_0 import dequantize_q8_0, quantize_q8_0


def _extension():
    extension = get_cuda_ext_ggml(raise_if_failed=True)
    assert extension is not None
    return extension


def test_q8_0_cuda_pack_matches_pytorch_encoder_and_is_decodable(monkeypatch):
    generator = torch.Generator(device="cuda").manual_seed(1234)
    weight = torch.randn((8, 64), generator=generator, device="cuda", dtype=torch.bfloat16)

    packed = _extension().q8_0_pack(weight).reshape(8, 2, 34)
    packed_again = _extension().q8_0_pack(weight).reshape(8, 2, 34)
    monkeypatch.setattr(q8_0_module, "get_cuda_ext_ggml", lambda: None)
    reference, shape = quantize_q8_0(weight)
    reconstructed = dequantize_q8_0(packed, shape)

    assert packed.shape == (8, 2, 34)
    assert torch.equal(packed, packed_again)
    assert torch.equal(packed, reference)
    assert shape.device.type == "cpu"
    normalized_mse = (
        reconstructed.float() - weight.float()
    ).square().mean() / weight.float().square().mean()
    assert normalized_mse < 1e-4


def test_q8_0_cuda_scale_keeps_its_low_byte(monkeypatch):
    """The fp16 block scale must be stored whole (bm-892 / bug-945).

    ptxas 12.6 and 12.8 compile a byte store of half bits (``payload[0] = (uint8_t)bits``) to F2I.U8.F16, the
    half's value truncated to an integer: every scale below 1 lost its low byte and the dequantised weights shrank
    by 0-20 %. With random scales the low byte is zero in about 1 block of 256, so a run of all-zero low bytes
    over 1024 blocks is the miscompile, not chance.
    """
    generator = torch.Generator(device="cuda").manual_seed(99)
    weight = torch.randn((64, 512), generator=generator, device="cuda", dtype=torch.bfloat16) * 0.02
    packed = _extension().q8_0_pack(weight).reshape(-1, 34)
    low = packed[:, 0]
    assert (low != 0).float().mean() > 0.9, "Q8_0 scale low byte lost (ptxas F2I.U8.F16 byte-store miscompile)"
    # The scale bytes must agree with the PyTorch encoder block for block (the quants may differ in rare
    # rounding ties between the two float paths; the byte-exact reference is libggml, checked below when present).
    monkeypatch.setattr(q8_0_module, "get_cuda_ext_ggml", lambda: None)
    reference, _ = quantize_q8_0(weight)
    assert torch.equal(packed[:, :2], reference.reshape(-1, 34)[:, :2])
    try:
        from pathlib import Path

        from omk_qad.ggml_lib import GGMLLib
    except ImportError:
        return
    import os

    llama = next((p for p in (os.environ.get("OMK_LLAMA_CPP"), "/opt/llama.cpp", "/mnt/sdc/ml/llama.cpp-b12386")
                  if p and (Path(p) / "build/bin/libggml-base.so").is_file()), None)
    if llama is None:
        return
    ref = GGMLLib(Path(llama)).quantize(8, weight.float().cpu().numpy())  # GGML_TYPE_Q8_0 = 8
    assert torch.equal(packed.cpu(), torch.from_numpy(ref).reshape(-1, 34)), "CUDA Q8_0 pack != llama-quantize bytes"


def test_q8_0_cuda_extension_has_no_f2i_u8_f16():
    """SASS gate: the built extension must not contain the miscompiled byte store (needs cuobjdump; else skipped)."""
    import os
    import shutil
    import subprocess

    extension = _extension()
    so = getattr(extension, "__file__", None)
    if not so or not os.path.isfile(so):
        pytest.skip("extension module has no .so path")
    cuobjdump = shutil.which("cuobjdump") or os.path.join(
        os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin", "cuobjdump"
    )
    if not os.path.isfile(cuobjdump):
        pytest.skip("cuobjdump not available")
    sass = subprocess.run([cuobjdump, "-sass", so], capture_output=True, text=True, check=False).stdout
    assert sass, "cuobjdump produced no SASS"
    assert "F2I.U8.F16" not in sass, "fp16 byte store miscompiled to F2I.U8.F16 in the GGML CUDA extension"


def test_q8_0_cuda_zero_and_nonfinite_encoding_match_reference(monkeypatch):
    weight = torch.zeros((2, 32), device="cuda", dtype=torch.bfloat16)
    weight[1, :3] = torch.tensor([torch.nan, torch.inf, -torch.inf], device="cuda")

    packed = _extension().q8_0_pack(weight).reshape(2, 1, 34)
    monkeypatch.setattr(q8_0_module, "get_cuda_ext_ggml", lambda: None)
    reference, shape = quantize_q8_0(weight)

    assert not packed.any()
    assert torch.equal(packed, reference)
    assert torch.equal(dequantize_q8_0(packed, shape), torch.zeros_like(weight))


def test_q8_0_cuda_falls_back_to_pytorch_encoder(monkeypatch):
    monkeypatch.setattr(q8_0_module, "get_cuda_ext_ggml", lambda: None)
    generator = torch.Generator(device="cuda").manual_seed(1234)
    weight = torch.randn((2, 32), generator=generator, device="cuda", dtype=torch.bfloat16)

    packed, shape = quantize_q8_0(weight)
    reconstructed = dequantize_q8_0(packed, shape)

    assert packed.shape == (2, 1, 34)
    assert torch.isfinite(reconstructed).all()


def test_q8_0_cuda_float64_matches_pytorch_encoder():
    weight = torch.randn(4, 32, dtype=torch.float64, generator=torch.Generator().manual_seed(7))

    reference, _ = quantize_q8_0(weight)
    packed, _ = quantize_q8_0(weight.cuda())

    assert torch.equal(reference, packed.cpu())


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([5207.0, 20.5, -20.5], [127, 0, 0]),
        ([1.0, 3.4999 / 127, -3.4999 / 127], [127, 3, -3]),
    ],
    ids=["reciprocal-rounding", "unrounded-scale"],
)
def test_q8_0_cuda_reciprocal_rounding_boundary_matches_pytorch_encoder(
    monkeypatch, values, expected
):
    weight = torch.zeros((1, 32), device="cuda", dtype=torch.float32)
    weight[0, :3] = torch.tensor(values, device="cuda")

    packed = _extension().q8_0_pack(weight).reshape(1, 1, 34)
    monkeypatch.setattr(q8_0_module, "get_cuda_ext_ggml", lambda: None)
    reference, _ = quantize_q8_0(weight)

    assert torch.equal(packed, reference)
    assert packed[0, 0, 2:5].view(torch.int8).tolist() == expected
