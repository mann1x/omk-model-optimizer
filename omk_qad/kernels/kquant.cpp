// SPDX-License-Identifier: Apache-2.0
// Python binding of omk_qad's Q2_K..Q6_K and Q4_0/Q4_1/Q5_0/Q5_1 CUDA encoders (kquant.cu; omk_qad/kquant.py loads it).

#include <torch/extension.h>

#include <optional>

at::Tensor q2_k_pack_cuda(at::Tensor input, std::optional<at::Tensor> importance);
at::Tensor q3_k_pack_cuda(at::Tensor input, std::optional<at::Tensor> importance);
at::Tensor q4_k_pack_cuda(at::Tensor input, std::optional<at::Tensor> importance);
at::Tensor q5_k_pack_cuda(at::Tensor input, std::optional<at::Tensor> importance);
at::Tensor q6_k_pack_cuda(at::Tensor input, std::optional<at::Tensor> importance);
at::Tensor q4_0_pack_cuda(at::Tensor input, std::optional<at::Tensor> importance);
at::Tensor q4_1_pack_cuda(at::Tensor input, std::optional<at::Tensor> importance);
at::Tensor q5_0_pack_cuda(at::Tensor input, std::optional<at::Tensor> importance);
at::Tensor q5_1_pack_cuda(at::Tensor input, std::optional<at::Tensor> importance);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("q4_k_pack", &q4_k_pack_cuda,
             "Encode a contiguous float32 CUDA [rows, cols] tensor (cols a multiple of 256) to Q4_K: uint8 "
             "[rows, cols / 256, 144], byte-identical to llama.cpp's quantize_q4_K. importance: optional float32 "
             "[cols] E[x^2] (llama-quantize's imatrix convention; without it the quantize_row_q4_K_ref path).",
             pybind11::arg("input"), pybind11::arg("importance") = pybind11::none());
  module.def("q5_k_pack", &q5_k_pack_cuda,
             "Encode to Q5_K: uint8 [rows, cols / 256, 176], byte-identical to llama.cpp's quantize_q5_K. "
             "importance as for q4_k_pack.",
             pybind11::arg("input"), pybind11::arg("importance") = pybind11::none());
  module.def("q6_k_pack", &q6_k_pack_cuda,
             "Encode to Q6_K: uint8 [rows, cols / 256, 210], byte-identical to llama.cpp's quantize_q6_K. "
             "importance as for q4_k_pack.",
             pybind11::arg("input"), pybind11::arg("importance") = pybind11::none());
  module.def("q2_k_pack", &q2_k_pack_cuda, "Encode to Q2_K: uint8 [rows, cols / 256, 84] (quantize_q2_K).",
             pybind11::arg("input"), pybind11::arg("importance") = pybind11::none());
  module.def("q3_k_pack", &q3_k_pack_cuda, "Encode to Q3_K: uint8 [rows, cols / 256, 110] (quantize_q3_K).",
             pybind11::arg("input"), pybind11::arg("importance") = pybind11::none());
  module.def("q4_0_pack", &q4_0_pack_cuda, "Encode to Q4_0: uint8 [rows, cols / 32, 18] (quantize_q4_0).",
             pybind11::arg("input"), pybind11::arg("importance") = pybind11::none());
  module.def("q4_1_pack", &q4_1_pack_cuda, "Encode to Q4_1: uint8 [rows, cols / 32, 20] (quantize_q4_1).",
             pybind11::arg("input"), pybind11::arg("importance") = pybind11::none());
  module.def("q5_0_pack", &q5_0_pack_cuda, "Encode to Q5_0: uint8 [rows, cols / 32, 22] (quantize_q5_0).",
             pybind11::arg("input"), pybind11::arg("importance") = pybind11::none());
  module.def("q5_1_pack", &q5_1_pack_cuda, "Encode to Q5_1: uint8 [rows, cols / 32, 24] (quantize_q5_1).",
             pybind11::arg("input"), pybind11::arg("importance") = pybind11::none());
}
