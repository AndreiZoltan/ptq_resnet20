import torch
import pytest
from ..lbtorch import LBObserver, quant, dequant


@pytest.mark.parametrize(
    "tensor, n_bits, eps, qscheme",
    [
        [torch.randn(10, 10, 10, 10), 16, 0.01, "quantize_per_tensor"],
        [torch.randn(10, 10, 10, 10), 8, 1, "quantize_per_tensor"],
        [torch.randn(10, 10, 10, 10), 4, 150, "quantize_per_tensor"],
        [torch.randn(10, 10, 10, 10), 2, 200, "quantize_per_tensor"],
        [torch.randn(10, 10, 10, 10), 16, 0.01, "quantize_per_channel"],
        [torch.randn(10, 10, 10, 10), 8, 1, "quantize_per_channel"],
        [torch.randn(10, 10, 10, 10), 4, 150, "quantize_per_channel"],
        [torch.randn(10, 10, 10, 10), 2, 200, "quantize_per_channel"],
    ],
)
def qstub_test(tensor, n_bits, eps, qscheme):
    stub = LBObserver(n_bits, qscheme)
    stub(tensor)
    scale, zero_point = stub.calculate_qparams()
    qtensor = quant(tensor, scale, zero_point, n_bits, qscheme)
    dtensor = dequant(qtensor, scale, zero_point, n_bits, tensor.shape[-1])
    assert torch.linalg.norm(tensor - dtensor) < eps
