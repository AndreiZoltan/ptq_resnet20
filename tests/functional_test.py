import torch
import pytest
from ..lbtorch import quant, dequant, to_int4, to_int2


@pytest.mark.parametrize(
    "tensor, scale, zero_point, n_bits, qscheme, true_value",
    [
        [
            2 * torch.ones(42, 2, 3, 4),
            torch.ones(42),
            torch.zeros(42),
            8,
            "quantize_per_channel",
            2 * torch.ones(42, 2, 3, 4).to(torch.int8),
        ]
    ],
)
def quant_test(tensor, scale, zero_point, n_bits, qscheme, true_value):
    out_value = quant(tensor, scale, zero_point, n_bits, qscheme)
    is_equal = out_value == true_value
    assert is_equal.all()
    assert out_value.dtype == torch.int8  # isinstance doesn't work with torch


def dequant_channel_test():
    to_dequant = quant(
        2 * torch.ones(42, 2, 3, 4),
        torch.ones(42),
        torch.zeros(42),
        8,
        "quantize_per_channel",
    )
    out_value = dequant(to_dequant, torch.ones(42), torch.zeros(42), 8, 4)
    true_value = 2 * torch.ones(42, 2, 3, 4).to(torch.float)
    is_equal = out_value == true_value
    assert is_equal.all()
    assert out_value.dtype == torch.float32  # isinstance doesn't work with torch


@pytest.mark.parametrize(
    "tensor, scale, zero_point, n_bits, qscheme, true_value",
    [
        [
            2 * torch.ones(42, 2, 3, 4),
            torch.ones(42),
            torch.zeros(42),
            8,
            "quantize_per_channel",
            2 * torch.ones(42, 2, 3, 4),
        ],
        [
            7.01 * torch.ones(42, 2, 3, 4),
            torch.ones(42),
            torch.zeros(42),
            8,
            "quantize_per_channel",
            7 * torch.ones(42, 2, 3, 4),
        ],
        [
            -10.2 * torch.ones(42, 2, 3, 4),
            torch.ones(42),
            torch.zeros(42),
            8,
            "quantize_per_channel",
            -10 * torch.ones(42, 2, 3, 4),
        ],
        [
            -128 * torch.ones(42, 2, 3, 4),
            torch.ones(42),
            torch.zeros(42),
            8,
            "quantize_per_channel",
            -128 * torch.ones(42, 2, 3, 4),
        ],
        [
            1000 * torch.ones(42, 2, 3, 4),
            torch.ones(42),
            torch.zeros(42),
            8,
            "quantize_per_channel",
            127 * torch.ones(42, 2, 3, 4),
        ],
        [
            -1000 * torch.ones(42, 2, 3, 4),
            torch.ones(42),
            torch.zeros(42),
            8,
            "quantize_per_channel",
            -128 * torch.ones(42, 2, 3, 4),
        ],
        [
            24 * torch.ones(42, 2, 3, 4),
            2 * torch.ones(42),
            10 * torch.zeros(42),
            8,
            "quantize_per_channel",
            24 * torch.ones(42, 2, 3, 4),
        ],
        [
            24 * torch.ones(42, 2, 3, 4),
            torch.ones(1),
            torch.zeros(1),
            8,
            "quantize_per_tensor",
            24 * torch.ones(42, 2, 3, 4),
        ],
        [
            2**17 * torch.ones(42, 2, 3, 4),
            torch.ones(42),
            torch.zeros(42),
            16,
            "quantize_per_channel",
            (2**15 - 1) * torch.ones(42, 2, 3, 4),
        ],
        [
            -(2**17) * torch.ones(42, 2, 3, 4),
            torch.ones(42),
            torch.zeros(42),
            16,
            "quantize_per_channel",
            -(2**15) * torch.ones(42, 2, 3, 4),
        ],
        [
            2**17 * torch.ones(42, 2, 3, 4),
            torch.ones(1),
            torch.zeros(1),
            16,
            "quantize_per_tensor",
            (2**15 - 1) * torch.ones(42, 2, 3, 4),
        ],
        [
            7 * torch.ones(42, 2, 3, 4),
            torch.ones(42),
            torch.zeros(42),
            4,
            "quantize_per_channel",
            7 * torch.ones(42, 2, 3, 4),
        ],
        [
            1000 * torch.ones(42, 2, 3, 4),
            torch.ones(42),
            torch.zeros(42),
            4,
            "quantize_per_channel",
            7 * torch.ones(42, 2, 3, 4),
        ],
        [
            -1000 * torch.ones(42, 2, 3, 4),
            torch.ones(42),
            torch.zeros(42),
            4,
            "quantize_per_channel",
            -8 * torch.ones(42, 2, 3, 4),
        ],
        [
            -1000 * torch.ones(42, 2, 3, 4),
            torch.ones(42),
            torch.zeros(42),
            2,
            "quantize_per_channel",
            -2 * torch.ones(42, 2, 3, 4),
        ],
        [
            2 * torch.ones(42, 2, 3, 4),
            torch.ones(1),
            torch.zeros(1),
            2,
            "quantize_per_tensor",
            1 * torch.ones(42, 2, 3, 4),
        ],
    ],
)
def quant_dequant_test(tensor, scale, zero_point, n_bits, qscheme, true_value):
    to_dequant = quant(tensor, scale, zero_point, n_bits, qscheme)
    out_value = dequant(to_dequant, scale, zero_point, n_bits, tensor.shape[-1])
    # print(out_value.shape, true_value.shape)
    is_equal = out_value == true_value
    assert is_equal.all()
    assert out_value.dtype == torch.float32  # isinstance doesn't work with torch


@pytest.mark.parametrize(
    "tensor, scale, zero_point, n_bits, qscheme, eps",
    [
        [
            0.25 * torch.randn(42, 2, 3, 4),
            0.01 * torch.ones(42),
            0 * torch.zeros(42),
            8,
            "quantize_per_channel",
            1,
        ],
        [
            0.25 * torch.ones(42, 2, 3, 4),
            0.01 * torch.ones(42),
            10 * torch.ones(42),
            16,
            "quantize_per_channel",
            0.01,
        ],
        [
            0.25 * torch.ones(42, 2, 3, 4),
            0.01 * torch.ones(1),
            10 * torch.ones(1),
            16,
            "quantize_per_tensor",
            0.01,
        ],
        [
            0.25 * torch.ones(42, 2, 3, 4),
            0.043 * torch.ones(42),
            -1 * torch.ones(42),
            4,
            "quantize_per_channel",
            0.5,
        ],
        [
            0.25 * torch.ones(42, 2, 3, 4),
            0.043 * torch.ones(1),
            -1 * torch.ones(1),
            4,
            "quantize_per_tensor",
            0.5,
        ],
        [
            0.25 * torch.ones(42, 2, 3, 4),
            0.3 * torch.ones(1),
            0 * torch.ones(1),
            2,
            "quantize_per_tensor",
            2,
        ],
        [
            0.25 * torch.ones(42, 2, 3, 4),
            0.3 * torch.ones(42),
            0 * torch.ones(42),
            2,
            "quantize_per_channel",
            2,
        ],
        # [
        #     0.25 * torch.ones(2, 3, 32, 32),
        #     0.1 * torch.ones(1),
        #     0 * torch.ones(1),
        #     4,
        #     "quantize_per_tensor",
        #     2,
        # ],
    ],
)
def quant_dequant_float_test(tensor, scale, zero_point, n_bits, qscheme, eps):
    to_dequant = quant(tensor, scale, zero_point, n_bits, qscheme)
    out_value = dequant(to_dequant, scale, zero_point, n_bits, tensor.shape[-1])
    assert tensor.shape == out_value.shape
    assert torch.linalg.norm(tensor - out_value) < eps
    assert out_value.dtype == torch.float32  # isinstance doesn't work with torch


@pytest.mark.parametrize(
    "tensor, true_value",
    [
        [
            -7 * torch.ones(11),
            -103 * torch.ones(6),
        ],
        [
            -7 * torch.ones(10),
            -103 * torch.ones(5),
        ],
    ],
)
def to_int4_test(tensor, true_value):
    value = to_int4(tensor)
    is_equal = value == true_value
    assert is_equal.all()


@pytest.mark.parametrize(
    "tensor, true_value",
    [
        [
            -1 * torch.ones(12),
            -1 * torch.ones(3),
        ],
        [
            -1 * torch.ones(13),
            -1 * torch.ones(4),
        ],
    ],
)
def to_int2_test(tensor, true_value):
    value = to_int2(tensor)
    is_equal = value == true_value
    assert is_equal.all()
