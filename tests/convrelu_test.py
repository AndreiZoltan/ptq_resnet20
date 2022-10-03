import torch
import pytest
from lbtorch import ConvRelu
from lbtorch import quant


@pytest.mark.parametrize(
    "cin, cout, k, tensor, scale, zero_point",
    [[3, 64, (3, 3), torch.randn(42, 3, 32, 32), torch.ones(1), torch.zeros(1)]],
)
def convrelu_test(cin, cout, k, tensor, scale, zero_point):
    model = ConvRelu(cin, cout, k)
    # tensor = quant(tensor, scale, zero_point, 8, "quantize_per_tensor")
    out = model(tensor)  # , scale, zero_point)
    assert out.shape[0] == tensor.shape[0]
    assert out.shape[1] == cout
    assert (out >= 0).all()
