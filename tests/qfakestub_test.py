import torch
import pytest
from ..lbtorch import QFakeStub, quant, dequant


@pytest.mark.parametrize(
    "tensor, n_bits, eps",
    [
        [torch.randn(10, 10, 10, 10), 16, 0.01],
        [torch.randn(10, 10, 10, 10), 8, 1],
        [torch.randn(10, 10, 10, 10), 4, 150],
        [torch.randn(10, 10, 10, 10), 2, 200],
    ],
)
def qfakestub_test(tensor, n_bits, eps):
    stub = QFakeStub(n_bits)
    stub(tensor)
    scale, zero_point = stub.calculate_qparams()
    qtensor = quant(tensor, scale, zero_point, n_bits, "quantize_per_tensor")
    dtensor = dequant(qtensor, scale, zero_point, n_bits, tensor.shape[-1])
    assert torch.linalg.norm(tensor - dtensor) < eps
