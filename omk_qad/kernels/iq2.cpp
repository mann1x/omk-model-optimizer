// SPDX-License-Identifier: Apache-2.0
// Python binding of omk_qad's IQ2_XXS / IQ2_S CUDA encoder (iq2.cu; omk_qad/iq2.py loads it with its tables).

#include <torch/extension.h>

#include <optional>

at::Tensor iq2_pack_cuda(at::Tensor input, std::optional<at::Tensor> importance, bool s, at::Tensor grid,
                         at::Tensor kmap, at::Tensor neigh);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("iq2_pack", &iq2_pack_cuda,
             "Encode a contiguous float32 CUDA [rows, cols] tensor (cols a multiple of 256) to IQ2_XXS (s=False: uint8 "
             "[rows, cols / 256, 66]) or IQ2_S (s=True: [.., 82]), byte-identical to llama.cpp's quantize_iq2_xxs / "
             "quantize_iq2_s. importance: float32 [cols] E[x^2] (mandatory for IQ2_XXS); grid / kmap / neigh: "
             "omk_qad.iq2 tables.",
             pybind11::arg("input"), pybind11::arg("importance"), pybind11::arg("s"), pybind11::arg("grid"),
             pybind11::arg("kmap"), pybind11::arg("neigh"));
}
