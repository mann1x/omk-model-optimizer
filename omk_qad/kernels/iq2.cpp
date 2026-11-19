// SPDX-License-Identifier: Apache-2.0
// Python binding of omk_qad's IQ2_XXS / IQ2_XS / IQ2_S CUDA encoder (iq2.cu; omk_qad/iq2.py loads it with its tables).

#include <torch/extension.h>

#include <optional>

at::Tensor iq2_pack_cuda(at::Tensor input, std::optional<at::Tensor> importance, int64_t kind, at::Tensor grid,
                         at::Tensor kmap, at::Tensor neigh);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("iq2_pack", &iq2_pack_cuda,
             "Encode a contiguous float32 CUDA [rows, cols] tensor (cols a multiple of 256) to IQ2_XXS (kind=0: uint8 "
             "[rows, cols / 256, 66]), IQ2_XS (kind=1: [.., 74]) or IQ2_S (kind=2: [.., 82]), byte-identical to "
             "llama.cpp's quantize_iq2_xxs / quantize_iq2_xs / quantize_iq2_s. importance: float32 [cols] E[x^2] "
             "(mandatory for IQ2_XXS and IQ2_XS); grid / kmap / neigh: omk_qad.iq2 tables of that type.",
             pybind11::arg("input"), pybind11::arg("importance"), pybind11::arg("kind"), pybind11::arg("grid"),
             pybind11::arg("kmap"), pybind11::arg("neigh"));
}
