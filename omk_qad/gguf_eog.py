# SPDX-License-Identifier: Apache-2.0
"""End-of-generation metadata of the GGUFs omk_qad writes (Gemma 4).

llama.cpp's converter reads ``eos_token_id`` from config/generation_config. For Gemma 4 that is a list
(``[1, 106]`` / ``[1, 106, 50]``) and the converter keeps the first element, so a converted Gemma-4 GGUF says
``eos = 1 (<eos>)`` and has no ``eot`` at all. The turn terminator is ``<turn|>`` (106). llama.cpp hides the
defect (it re-types control tokens at load), ollama's ``PARSER gemma4`` does not: it reads ``eot_token_id``
and, without one, never sees the turn end. omnimergekit's quantize_gguf.py stamps eos 106 / eot 1 on its
builds; omk_qad's exporters wrote the converter's values (bm-892 iq2.gguf: eos 1, no eot).

``install(conv, gguf)``  before the converter writes: on a Gemma-4 converter, set eos = ``<turn|>`` and
                         eot = ``<eos>`` in the writer's KV table, by token TEXT from the vocab being written.
``check(path, gguf)``    after: re-read the file; on Gemma 4 the EOG tokens must be exactly those, CONTROL
                         typed, no eom, and the chat template must end turns with ``<turn|>``.
Other architectures are left as the converter wrote them and reported as not checked.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

GEMMA4_ARCH = "gemma4"
GEMMA4_EOS = "<turn|>"  # 106: ends every turn, the stop token
GEMMA4_EOT = "<eos>"  # 1: what ollama's gemma4 parser reads as end of turn
CONTROL = 3  # gguf TokenType.CONTROL


def _arch_of(conv, gguf) -> str:
    return gguf.MODEL_ARCH_NAMES[conv.model_arch]


def stamp_kv(kv: dict, gguf) -> dict:
    """Set the Gemma-4 eos/eot in a writer KV table (``GGUFWriter.kv_data[0]``); returns {eos, eot} ids."""
    keys = gguf.Keys.Tokenizer
    tokens = kv.get(keys.LIST)
    if tokens is None:
        raise SystemExit("gemma4 EOG: the writer has no vocab (tokenizer.ggml.tokens) to find <turn|>/<eos> in")
    tokens = [t.decode("utf-8", "replace") if isinstance(t, (bytes, bytearray)) else str(t) for t in tokens.value]
    ids = {}
    for role, text in (("eos", GEMMA4_EOS), ("eot", GEMMA4_EOT)):
        hits = [i for i, t in enumerate(tokens) if t == text]
        if len(hits) != 1:
            raise SystemExit(f"gemma4 EOG: {text!r} occurs {len(hits)} times in the vocab (expected once)")
        ids[role] = hits[0]
    kv[keys.EOS_ID] = gguf.GGUFValue(ids["eos"], gguf.GGUFValueType.UINT32)
    kv[keys.EOT_ID] = gguf.GGUFValue(ids["eot"], gguf.GGUFValueType.UINT32)
    return ids


def install(conv, gguf) -> dict | None:
    """Make ``conv`` (a llama.cpp converter instance) write Gemma-4 EOG ids; None for other architectures.

    The KV table is complete only when the converter calls ``write_header_to_file`` (after set_vocab and
    prepare_metadata), so the ids are stamped there, before the header reaches the file.
    """
    if _arch_of(conv, gguf) != GEMMA4_ARCH:
        return None
    writer = conv.gguf_writer
    real = writer.write_header_to_file
    stamped: dict = {}

    def write_header_to_file(*args, **kwargs):
        before = {r: (writer.kv_data[0][k].value if k in writer.kv_data[0] else None)
                  for r, k in (("eos", gguf.Keys.Tokenizer.EOS_ID), ("eot", gguf.Keys.Tokenizer.EOT_ID))}
        stamped.update(stamp_kv(writer.kv_data[0], gguf), converter=before)
        return real(*args, **kwargs)

    writer.write_header_to_file = write_header_to_file
    return stamped


def check(path: Path, gguf) -> dict:
    """EOG report of a written GGUF; ``ok`` False lists the problems (Gemma 4 only, others ``checked`` False)."""
    r = gguf.GGUFReader(str(path))
    f = r.fields

    def scalar(key):
        v = f.get(key)
        return None if v is None else int(v.parts[v.data[0]][0])

    def string(key):
        v = f.get(key)
        return None if v is None else bytes(v.parts[v.data[0]]).decode("utf-8", "replace")

    arch = string("general.architecture")
    if arch != GEMMA4_ARCH:
        return {"arch": arch, "checked": False, "ok": True}
    keys = gguf.Keys.Tokenizer
    tok, typ = f.get(keys.LIST), f.get(keys.TOKEN_TYPE)
    eos, eot, eom = scalar(keys.EOS_ID), scalar(keys.EOT_ID), scalar(keys.EOM_ID)
    rep = {"arch": arch, "checked": True, "eos": eos, "eot": eot, "eom": eom, "problems": []}
    if tok is None or typ is None:
        rep["problems"].append("no vocab / token types")
    else:
        n = len(tok.data)

        def text(i):
            return bytes(tok.parts[tok.data[i]]).decode("utf-8", "replace")

        def ttype(i):
            return int(typ.parts[typ.data[i]][0])

        for role, i, want in (("eos", eos, GEMMA4_EOS), ("eot", eot, GEMMA4_EOT)):
            if i is None:
                rep["problems"].append(f"no {role}_token_id")
            elif not 0 <= i < n:
                rep["problems"].append(f"{role} {i} outside the vocab ({n})")
            else:
                rep[f"{role}_text"], rep[f"{role}_type"] = text(i), ttype(i)
                if text(i) != want:
                    rep["problems"].append(f"{role} {i} is {text(i)!r}, expected {want!r}")
                if ttype(i) != CONTROL:
                    rep["problems"].append(f"{role} {i} ({text(i)!r}) is token type {ttype(i)}, expected CONTROL")
    if eom is not None:
        rep["problems"].append(f"eom_token_id {eom} set (Gemma 4 has no eom)")
    tpl = string(keys.CHAT_TEMPLATE)
    if not tpl:
        rep["problems"].append("no chat template")
    elif GEMMA4_EOS not in tpl:
        rep["problems"].append(f"chat template never emits {GEMMA4_EOS!r}")
    rep["ok"] = not rep["problems"]
    return rep
