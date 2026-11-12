"""bm-889 sft2: vLLM 0.20.2 gemma4 tool parser with the array hang fixed (loaded by --tool-parser-plugin).

vLLM's ``_parse_gemma4_array`` loops forever when an array element starts with a stray ``]``: the bare-value scan
stops on it without advancing, appends an empty value and retries the same position, growing the list until the
API server runs out of memory. The parser runs on the server's event loop, so one malformed generated tool call
blocks every request (2026-10-30 09:59Z: sft2 generation stalled, py-spy at gemma4_tool_parser.py:278).

This module replaces the module-level function with a copy whose bare-value branch skips a character it cannot
consume. Every input that terminated before parses exactly as before; only the hanging case changes (the stray
character is dropped). The parser is registered as ``gemma4_safe`` so the served command line names the fix;
the replacement is module-wide, so ``gemma4`` is fixed in this process too.
"""
from vllm.tool_parsers import gemma4_tool_parser as g4
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

STRING_DELIM = g4.STRING_DELIM


def _parse_gemma4_array(arr_str: str, *, partial: bool = False) -> list:
    """vLLM 0.20.2's ``_parse_gemma4_array`` plus a progress guard in the bare-value branch."""
    items: list = []
    i = 0
    n = len(arr_str)

    while i < n:
        while i < n and arr_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n:
            break

        if arr_str[i:].startswith(STRING_DELIM):
            i += len(STRING_DELIM)
            end_pos = arr_str.find(STRING_DELIM, i)
            if end_pos == -1:
                items.append(arr_str[i:])
                break
            items.append(arr_str[i:end_pos])
            i = end_pos + len(STRING_DELIM)

        elif arr_str[i] == "{":
            depth = 1
            obj_start = i + 1
            i += 1
            while i < n and depth > 0:
                if arr_str[i:].startswith(STRING_DELIM):
                    i += len(STRING_DELIM)
                    nd = arr_str.find(STRING_DELIM, i)
                    i = nd + len(STRING_DELIM) if nd != -1 else n
                    continue
                if arr_str[i] == "{":
                    depth += 1
                elif arr_str[i] == "}":
                    depth -= 1
                i += 1
            if depth > 0:
                items.append(g4._parse_gemma4_args(arr_str[obj_start:i], partial=True))
            else:
                items.append(g4._parse_gemma4_args(arr_str[obj_start : i - 1]))

        elif arr_str[i] == "[":
            depth = 1
            sub_start = i + 1
            i += 1
            while i < n and depth > 0:
                if arr_str[i] == "[":
                    depth += 1
                elif arr_str[i] == "]":
                    depth -= 1
                i += 1
            if depth > 0:
                items.append(_parse_gemma4_array(arr_str[sub_start:i], partial=True))
            else:
                items.append(_parse_gemma4_array(arr_str[sub_start : i - 1]))

        else:
            val_start = i
            while i < n and arr_str[i] not in (",", "]"):
                i += 1
            if i == val_start:  # a stray "]" the scan cannot consume: skip it (stock vLLM loops here forever)
                i += 1
                continue
            if partial and i >= n:
                break
            items.append(g4._parse_gemma4_value(arr_str[val_start:i]))

    return items


g4._parse_gemma4_array = _parse_gemma4_array


class Gemma4SafeToolParser(g4.Gemma4ToolParser):
    """Gemma4ToolParser running on the fixed ``_parse_gemma4_array``."""


ToolParserManager.register_module(name="gemma4_safe", module=Gemma4SafeToolParser)
