# SPDX-License-Identifier: Apache-2.0
r"""A GGUF teacher for ``--phase cache`` (``--teacher-gguf``): top-K log-probs from libllama, not torch.

``llama_topk`` (omk_qad/llama_topk/llama_topk.cpp) holds the model and answers one sequence at a time over a pipe;
positions nobody trains on (role 0: prompt, tool output, padding) are not computed. Its log-probs are the model's
own (llama.cpp applies Gemma-4's final-logit soft cap in the graph), and the tokenizer is ``--model``'s, which must
be the GGUF's vocabulary (checked: same size).

Build against a CUDA build of llama.cpp (``LLAMA`` = its source tree, built in ``$LLAMA/build``):

    g++ -O3 -fopenmp -std=c++17 omk_qad/llama_topk/llama_topk.cpp -I$LLAMA/include -I$LLAMA/ggml/include \\
        -L$LLAMA/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$LLAMA/build/bin -o omk_qad/llama_topk/llama_topk
"""

import subprocess
from pathlib import Path

import numpy as np
import torch

BIN = Path(__file__).parent / "llama_topk" / "llama_topk"


class GGUFTeacher:
    """``teacher(x, want) -> (idx int32 [B, S-1, K], val fp16 [B, S-1, K], nll of the wanted positions)``."""

    def __init__(self, gguf: str, n_ctx: int, topk: int, n_vocab: int, binary: str = "", ngl: int = 999,
                 ubatch: int = 512, threads: int = 16, log=print):
        """Start ``llama_topk`` on ``gguf`` and wait for its ready line (vocabulary checked)."""
        exe = Path(binary) if binary else BIN
        if not exe.exists():
            raise SystemExit(f"{exe}: not built (see omk_qad/gguf_teacher.py)")
        self.k = topk
        self.p = subprocess.Popen([str(exe), "--model", gguf, "--ctx", str(n_ctx), "--topk", str(topk),
                                   "--ngl", str(ngl), "--ubatch", str(ubatch), "--threads", str(threads)],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for ln in self.p.stderr:  # the load log, up to the ready line
            s = ln.decode(errors="replace").rstrip()
            if s.startswith("LLAMA_TOPK_READY"):
                log(f"GGUF teacher {gguf}: {s}")
                got = int(s.split("n_vocab=")[1].split()[0])
                if got != n_vocab:
                    raise SystemExit(f"GGUF teacher vocabulary {got} != the tokenizer's {n_vocab}")
                break
        else:
            raise SystemExit(f"llama_topk exited before it was ready (rc {self.p.wait()})")
        # later stderr (llama.cpp's per-decode chatter) is drained so the pipe never fills
        import threading

        threading.Thread(target=lambda: [None for _ in self.p.stderr], daemon=True).start()

    def __call__(self, x: torch.Tensor, want: torch.Tensor | None = None):
        """Top-K of every position of every row (``want`` [B, S-1] bool: positions to compute, default all)."""
        b, s = x.shape
        k = self.k
        idx = np.empty((b, s - 1, k), np.int32)
        val = np.empty((b, s - 1, k), np.float16)
        nll = 0.0
        rec = np.dtype([("idx", np.int32, k), ("val", np.float32, k), ("tgt", np.float32)])
        for r in range(b):
            w = np.ones(s - 1, np.uint8) if want is None else want[r].numpy().astype(np.uint8)
            self.p.stdin.write(np.int32(s).tobytes() + x[r].numpy().astype(np.int32).tobytes() + w.tobytes())
            self.p.stdin.flush()
            buf = self.p.stdout.read(rec.itemsize * (s - 1))
            if len(buf) != rec.itemsize * (s - 1):
                raise SystemExit(f"llama_topk died mid-sequence (rc {self.p.poll()})")
            out = np.frombuffer(buf, rec)
            idx[r], val[r] = out["idx"], out["val"].astype(np.float16)
            nll -= float(out["tgt"][w.astype(bool)].sum())
        return torch.from_numpy(idx), torch.from_numpy(val), nll

    def close(self) -> None:
        """End the process (n = 0)."""
        if self.p.poll() is None:
            self.p.stdin.write(np.int32(0).tobytes())
            self.p.stdin.close()
            self.p.wait(timeout=120)
