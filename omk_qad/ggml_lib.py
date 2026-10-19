# SPDX-License-Identifier: Apache-2.0
"""libggml's own block encoder/decoder via ctypes: the exact code llama-quantize runs.

ModelOpt encodes IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ2_S/Q8_0 (torch + CUDA). Every other GGML type an llama.cpp
mix assigns (IQ3_S, Q2_K..Q6_K, IQ4_XS, ...) is encoded here with ``ggml_quantize_chunk`` and decoded with
the type's ``to_float``, so a tensor fake-quantized in training and the GGUF that ships hold the same blocks.
Rows are independent in every GGML format, so chunks of rows run in parallel threads (ctypes drops the GIL).
"""

import ctypes
import os
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path

import numpy as np

_TO_FLOAT = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64)


class _TypeTraits(ctypes.Structure):
    # mirrors struct ggml_type_traits in ggml/include/ggml.h
    _fields_ = (
        ("type_name", ctypes.c_char_p),
        ("blck_size", ctypes.c_int64),
        ("blck_size_interleave", ctypes.c_int64),
        ("type_size", ctypes.c_size_t),
        ("is_quantized", ctypes.c_bool),
        ("to_float", ctypes.c_void_p),
        ("from_float_ref", ctypes.c_void_p),
    )


class GGMLLib:
    """Thin binding of libggml-base: quantize/dequantize whole 2-D float32 tensors row-parallel."""

    def __init__(self, llama_cpp: Path, threads: int | None = None):
        """Load ``libggml-base.so`` from ``<llama_cpp>/build/bin``."""
        path = Path(llama_cpp) / "build" / "bin" / "libggml-base.so"
        self.lib = ctypes.CDLL(str(path))
        self.lib.ggml_quantize_chunk.restype = ctypes.c_size_t
        self.lib.ggml_quantize_chunk.argtypes = (
            ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_float),
        )
        self.lib.ggml_row_size.restype = ctypes.c_size_t
        self.lib.ggml_row_size.argtypes = (ctypes.c_int, ctypes.c_int64)
        self.lib.ggml_quantize_requires_imatrix.restype = ctypes.c_bool
        self.lib.ggml_quantize_requires_imatrix.argtypes = (ctypes.c_int,)
        self.lib.ggml_get_type_traits.restype = ctypes.POINTER(_TypeTraits)
        self.lib.ggml_get_type_traits.argtypes = (ctypes.c_int,)
        self.threads = threads or max(1, (os.cpu_count() or 2) // 2)
        self.path = path

    @cache  # one binding per process; the cache only holds small decoder handles
    def _traits(self, qtype: int) -> tuple[int, int, object]:
        t = self.lib.ggml_get_type_traits(qtype).contents
        if not t.is_quantized or not t.to_float:
            raise ValueError(f"ggml type {qtype} ({t.type_name}) is not a quantized type with a decoder")
        return int(t.blck_size), int(t.type_size), _TO_FLOAT(t.to_float)

    def row_size(self, qtype: int, n_per_row: int) -> int:
        """Bytes of one encoded row."""
        return int(self.lib.ggml_row_size(qtype, n_per_row))

    def requires_imatrix(self, qtype: int) -> bool:
        """Whether libggml refuses to encode ``qtype`` without importance weights."""
        return bool(self.lib.ggml_quantize_requires_imatrix(qtype))

    def _chunks(self, rows: int) -> list[tuple[int, int]]:
        step = max(1, -(-rows // (self.threads * 4)))
        return [(r, min(r + step, rows)) for r in range(0, rows, step)]

    def quantize(self, qtype: int, x: np.ndarray, imatrix: np.ndarray | None = None) -> np.ndarray:
        """Encode float32 ``x`` [rows, n] to uint8 [rows, row_size]; ``imatrix`` is [n] float32 or None."""
        blck, _, _ = self._traits(qtype)
        x = np.ascontiguousarray(x, dtype=np.float32)
        rows, n = x.shape
        if n % blck:
            raise ValueError(f"row width {n} is not a multiple of the block size {blck}")
        if imatrix is None and self.requires_imatrix(qtype):
            raise ValueError(f"ggml type {qtype} requires an importance matrix")
        imat = None if imatrix is None else np.ascontiguousarray(imatrix, dtype=np.float32)
        if imat is not None and imat.shape != (n,):
            raise ValueError(f"imatrix shape {imat.shape} != ({n},)")
        rs = self.row_size(qtype, n)
        out = np.empty((rows, rs), dtype=np.uint8)
        fp = ctypes.POINTER(ctypes.c_float)
        imat_p = None if imat is None else imat.ctypes.data_as(fp)

        def run(r0_r1):
            r0, r1 = r0_r1
            # src/dst point at the chunk's first row, so start = 0 (start must be a whole row anyway)
            written = self.lib.ggml_quantize_chunk(
                qtype, x[r0:r1].ctypes.data_as(fp), out[r0:r1].ctypes.data, 0, r1 - r0, n, imat_p
            )
            if written != (r1 - r0) * rs:
                raise RuntimeError(f"ggml_quantize_chunk wrote {written} bytes, expected {(r1 - r0) * rs}")

        with ThreadPoolExecutor(self.threads) as ex:
            list(ex.map(run, self._chunks(rows)))
        return out

    def dequantize(self, qtype: int, q: np.ndarray, n: int) -> np.ndarray:
        """Decode uint8 [rows, row_size] back to float32 [rows, n]."""
        _, _, to_float = self._traits(qtype)
        q = np.ascontiguousarray(q, dtype=np.uint8)
        rows = q.shape[0]
        if q.shape[1] != self.row_size(qtype, n):
            raise ValueError(f"row bytes {q.shape[1]} != row_size {self.row_size(qtype, n)}")
        out = np.empty((rows, n), dtype=np.float32)
        fp = ctypes.POINTER(ctypes.c_float)

        def run(r0_r1):
            r0, r1 = r0_r1
            to_float(q[r0:r1].ctypes.data, out[r0:r1].ctypes.data_as(fp), (r1 - r0) * n)

        with ThreadPoolExecutor(self.threads) as ex:
            list(ex.map(run, self._chunks(rows)))
        return out
