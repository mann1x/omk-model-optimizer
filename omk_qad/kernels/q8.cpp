// SPDX-License-Identifier: Apache-2.0
// Python binding of omk_qad's Q8_0 CUDA encoder (q8.cu; omk_qad/q8.py loads it).

#include <torch/extension.h>

at::Tensor q8_0_pack_cuda(at::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("q8_0_pack", &q8_0_pack_cuda,
             "Encode a contiguous float32 CUDA [rows, cols] tensor (cols a multiple of 32) to Q8_0: uint8 "
             "[rows, cols / 32, 34], byte-identical to llama.cpp's quantize_row_q8_0_ref (llama-quantize ignores the "
             "imatrix for Q8_0).",
             pybind11::arg("input"));
}
