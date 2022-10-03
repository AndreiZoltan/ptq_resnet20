from ..src.qmodel import qresnet20
import pytest
import torch


@pytest.mark.parametrize(
    "tensor, n_bits",
    [
        # [torch.randn(2, 3, 32, 32), 2],
        # [torch.randn(2, 3, 32, 32), 4],
        [torch.randn(2, 3, 32, 32), 8],
        [torch.randn(2, 3, 32, 32), 16],
    ],
)
def qmodel_test(tensor, n_bits):
    qmodel = qresnet20(n_bits)
    qmodel.quantize()
    output = qmodel(tensor)
    assert output.shape[0] == tensor.shape[0]
    assert output.shape[1] == 10
