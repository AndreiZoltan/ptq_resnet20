import torch
from torch import nn
import pytest
from lbtorch import ConvRelu
from lbtorch import quant, dequant
from lbtorch import LBObserver


@pytest.mark.parametrize(
    "fweight, n_bits, eps, qscheme",
    [
        # [5*torch.randn(64, 3, 3, 3), 4, 30, "quantize_per_tensor"],
        [5 * torch.randn(64, 3, 3, 3), 2, 120, "quantize_per_tensor"]
    ],
)
def lbquantize_test(fweight, n_bits, eps, qscheme):
    obs = LBObserver(n_bits, qscheme)
    obs(fweight)
    scale, zero_point = obs.calculate_qparams()
    weight = quant(fweight, scale, zero_point, n_bits, qscheme)
    de_fweight = dequant(weight, scale, zero_point, n_bits, fweight.shape[-1])
    assert de_fweight.dtype == torch.float32
    assert de_fweight.shape == fweight.shape
    assert torch.linalg.norm(de_fweight - fweight) < eps
    assert 1.5 * torch.linalg.norm(de_fweight - fweight) < torch.linalg.norm(fweight)
