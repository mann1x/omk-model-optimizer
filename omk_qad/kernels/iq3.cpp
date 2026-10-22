// SPDX-License-Identifier: Apache-2.0
// Python binding of omk_qad's IQ3_XXS / IQ3_S CUDA encoder (iq3.cu; omk_qad/iq3.py loads it with its tables).

#include <torch/extension.h>

#include <optional>

at::Tensor iq3_pack_cuda(at::Tensor input, std::optional<at::Tensor> importance, bool s, at::Tensor grid,
                         at::Tensor kmap, at::Tensor neigh);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("iq3_pack", &iq3_pack_cuda,
             "Encode a contiguous float32 CUDA [rows, cols] tensor (cols a multiple of 256) to IQ3_XXS (s=False: uint8 "
             "[rows, cols / 256, 98]) or IQ3_S (s=True: [.., 110]), byte-identical to llama.cpp's quantize_iq3_xxs / "
             "quantize_iq3_s. importance: optional float32 [cols] E[x^2]; grid / kmap / neigh: omk_qad.iq3 tables.",
             pybind11::arg("input"), pybind11::arg("importance"), pybind11::arg("s"), pybind11::arg("grid"),
             pybind11::arg("kmap"), pybind11::arg("neigh"));
}
