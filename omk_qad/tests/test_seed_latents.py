"""omk_qad.train.seed_latents: K-FAC W_adj files land in the right routed latents, partial seeds are refused."""

import pytest
import torch

from omk_qad.train import seed_latents


class Lat:
    """The three Latent attributes seed_latents touches."""

    def __init__(self, shape):
        """A zero fp16 latent of ``shape`` in the model basis."""
        self.shape, self.basis = tuple(shape), None
        self.param = torch.nn.Parameter(torch.zeros(shape, dtype=torch.float16), requires_grad=False)

    def set_model_weight(self, w):
        """Copy a model-basis weight in (what Latent.set_model_weight does without a Hadamard basis)."""
        self.param.data.copy_(w.reshape(self.shape).to(self.param.dtype))


def _files(tmp, layers, shape=(4, 6, 8)):
    for j in layers:
        torch.save({"stem": "ffn_gate_up_exps", "layer": j, "qtype": "IQ2_XXS", "method": "gptq", "ref": "",
                    "latent": torch.full(shape, float(j + 1), dtype=torch.float16)}, tmp / f"layer_{j:02d}.pt")


def test_seeds_every_routed_layer(tmp_path):
    """Each layer file lands in that layer's gate_up latent; down_proj and attention are untouched."""
    lats = {f"model.language_model.layers.{j}.experts.gate_up_proj": Lat((4, 6, 8)) for j in range(3)}
    lats["model.language_model.layers.0.experts.down_proj"] = Lat((4, 8, 3))
    lats["model.language_model.layers.0.self_attn.q_proj.weight"] = Lat((8, 8))
    _files(tmp_path, range(3))
    msgs = []
    what = seed_latents(str(tmp_path), lats, msgs.append)
    assert "3 gate_up latents" in what and msgs and msgs[0].startswith("SEED_LATENTS")
    for j in range(3):
        assert torch.all(lats[f"model.language_model.layers.{j}.experts.gate_up_proj"].param == j + 1)
    assert torch.all(lats["model.language_model.layers.0.experts.down_proj"].param == 0)


def test_partial_seed_is_refused(tmp_path):
    """A routed layer without a seed file is a silent mix of two roundings: refused by name."""
    lats = {f"model.language_model.layers.{j}.experts.gate_up_proj": Lat((4, 6, 8)) for j in range(3)}
    _files(tmp_path, [0, 2])
    with pytest.raises(SystemExit, match=r"SEED_FAIL.*\[1\]"):
        seed_latents(str(tmp_path), lats, print)


def test_shape_mismatch_is_refused(tmp_path):
    """A file whose W_adj does not match the latent's shape is refused."""
    lats = {"model.language_model.layers.0.experts.gate_up_proj": Lat((4, 6, 9))}
    _files(tmp_path, [0])
    with pytest.raises(SystemExit, match="SEED_FAIL"):
        seed_latents(str(tmp_path), lats, print)
