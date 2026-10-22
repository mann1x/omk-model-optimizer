// SPDX-License-Identifier: Apache-2.0
// Python binding of omk_qad's IQ4_XS / IQ4_NL CUDA encoder (iq4.cu; omk_qad/iq4.py loads it).

#include <torch/extension.h>

#include <optional>

at::Tensor iq4_xs_pack_cuda(at::Tensor input, std::optional<at::Tensor> importance);
at::Tensor iq4_nl_pack_cuda(at::Tensor input, std::optional<at::Tensor> importance);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("iq4_xs_pack", &iq4_xs_pack_cuda,
             "Encode a contiguous float32 CUDA [rows, cols] tensor (cols a multiple of 256) to IQ4_XS: uint8 "
             "[rows, cols / 256, 136], byte-identical to llama.cpp's quantize_iq4_xs. importance: optional float32 "
             "[cols] E[x^2] (llama-quantize's imatrix convention).",
             pybind11::arg("input"), pybind11::arg("importance") = pybind11::none());
  module.def("iq4_nl_pack", &iq4_nl_pack_cuda,
             "Encode a contiguous float32 CUDA [rows, cols] tensor (cols a multiple of 32) to IQ4_NL: uint8 "
             "[rows, cols / 32, 18], byte-identical to llama.cpp's quantize_iq4_nl. importance as for iq4_xs_pack.",
             pybind11::arg("input"), pybind11::arg("importance") = pybind11::none());
}
