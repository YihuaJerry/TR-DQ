import torch

from trdq.quantization.models.utils import get_rot
from trdq.quantization.quantizer.base_quantizer import WeightQuantizer


class Config(dict):
    """Minimal attribute-access mapping used to test the quantizer in isolation."""

    __getattr__ = dict.__getitem__


def test_generated_rotation_is_orthogonal() -> None:
    torch.manual_seed(0)
    rotation = get_rot(8)
    identity = torch.eye(8)
    assert torch.allclose(rotation.T @ rotation, identity, atol=1e-5)
    assert torch.allclose(
        rotation[:, 0].abs(), torch.full((8,), 8**-0.5), atol=1e-5
    )


def test_weight_quantizer_runs_on_cpu() -> None:
    config = Config(
        n_bits=4,
        mixed_precision=None,
        timestep_wise=False,
        per_group="channel",
        channel_dim=0,
        scale_method="min_max",
        round_mode="nearest",
        sym=False,
        running_stat=False,
    )
    quantizer = WeightQuantizer(config)
    source = torch.linspace(-1, 1, 32).reshape(4, 8)
    result = quantizer(source)
    assert result.shape == source.shape
    assert torch.isfinite(result).all()
