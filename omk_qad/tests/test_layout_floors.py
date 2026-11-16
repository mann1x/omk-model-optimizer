"""FAMILY_FLOORS: Gemma-4 attention is never planned or exported under 4 bits unless explicitly allowed (bm-892)."""

import json

import pytest

from omk_qad.layout import FAMILY_FLOORS, family_of, floor_type, low_bit_violations


def _layout(model_type, attn_type, tmp_path=None, record_type=True):
    lay = {
        "model_dir": str(tmp_path) if tmp_path else "/nonexistent",
        "targets": {
            "blk.0.attn_q.weight": {"type": attn_type, "cls": "attention"},
            "blk.0.attn_output.weight": {"type": attn_type, "cls": "attention"},
            "blk.0.ffn_gate_up_exps.weight": {"type": "IQ2_XXS", "cls": "routed_experts"},
            "blk.0.ffn_gate.weight": {"type": "Q4_K", "cls": "dense_mlp"},
            "blk.0.attn_norm.weight": {"type": "F32", "cls": "vectors"},
        },
    }
    if record_type:
        lay["model_type"] = model_type
    return lay


def test_family_key_merges_text_config():
    """``gemma4_text`` (the text config) and ``gemma4`` (the top-level config) are one floor family."""
    assert family_of("gemma4_text") == "gemma4" == family_of("Gemma4")
    assert "gemma4" in FAMILY_FLOORS and "attention" in FAMILY_FLOORS["gemma4"]
    assert family_of(None) == ""


@pytest.mark.parametrize("rest", ["IQ2_M", "IQ2_XXS", "IQ3_XXS", "IQ3_S", "Q2_K", "Q3_K_S", "Q3_K_M", "Q4_K_S",
                                  "IQ4_XS"])
def test_gemma4_attention_floor_is_q6k_below_q4_k_m(rest):
    """Owner's rule: every tier below Q4_K_M carries all attention at Q6_K."""
    assert floor_type("gemma4", "attention", rest) == "Q6_K"


@pytest.mark.parametrize("rest", ["Q4_K_M", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0", "q8_0"])
def test_gemma4_attention_floor_is_q8_0_from_q4_k_m(rest):
    """Owner's rule: Q4_K_M and every tier above carry all attention at Q8_0 (``bolt`` resolves to Q8_0 upstream)."""
    assert floor_type("gemma4", "attention", rest) == "Q8_0"


def test_unfloored_class_or_family_has_no_floor():
    """Only the floored classes of a floored family get a default; everything else follows the rest rules."""
    assert floor_type("gemma4", "routed_experts", "IQ2_M") is None
    assert floor_type("qwen3_moe", "attention", "IQ2_M") is None


@pytest.mark.parametrize("t", ["IQ2_S", "IQ3_S", "IQ2_XXS", "Q3_K", "Q2_K", "TQ1_0"])
def test_gemma4_low_bit_attention_is_refused(t):
    """Every sub-4-bit type on a Gemma-4 attention tensor is a SystemExit, with the reason in the message."""
    with pytest.raises(SystemExit, match="attention at 2-3 bits"):
        low_bit_violations(_layout("gemma4", t))


@pytest.mark.parametrize("t", ["Q4_K", "Q6_K", "Q8_0", "IQ4_XS", "IQ4_NL", "F16"])
def test_gemma4_lifted_attention_passes(t):
    """4 bits and up (and floats) pass: the floor is a minimum, not a fixed type."""
    assert low_bit_violations(_layout("gemma4", t)) == []


def test_allow_returns_the_list_instead_of_raising():
    """With permission the violations are reported (for the planner's warning), never silently dropped."""
    bad = low_bit_violations(_layout("gemma4", "IQ2_S"), allow=True)
    assert bad == ["blk.0.attn_q.weight=IQ2_S", "blk.0.attn_output.weight=IQ2_S"]


def test_other_families_are_not_floored():
    """The rule is per family: a Qwen layout with 2-bit attention is the caller's business."""
    assert low_bit_violations(_layout("qwen3_5_moe", "IQ2_S")) == []


def test_old_layout_without_model_type_reads_config_json(tmp_path):
    """A layout.pt built before ``model_type`` was recorded is still checked, from the model's config.json."""
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gemma4"}))
    with pytest.raises(SystemExit):
        low_bit_violations(_layout("gemma4", "IQ3_S", tmp_path, record_type=False))
    assert low_bit_violations(_layout("gemma4", "Q8_0", tmp_path, record_type=False)) == []
