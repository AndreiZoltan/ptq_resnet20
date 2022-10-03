import torch
from torch import nn
import torch.nn.functional as F

import os

from .functional import quant, dequant
from .lbobserver import LBObserver

from typing import Tuple


class ConvRelu(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: int = 1,
        padding: int = 0,
        device=None,
        n_bits: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride: int = stride
        self.padding: int = padding
        self.device = device

        assert n_bits in {2, 4, 8, 16}
        self.n_bits = n_bits
        self.w = kernel_size[1]
        if n_bits == 2:
            self.register_parameter(
                name="weight",
                param=nn.Parameter(
                    torch.ones(
                        out_channels,
                        in_channels,
                        kernel_size[0],
                        kernel_size[1] // 4 + 1,
                        dtype=torch.int8,
                    ),
                    requires_grad=False,
                ),
            )
        elif n_bits == 4:
            self.register_parameter(
                name="weight",
                param=nn.Parameter(
                    torch.ones(
                        out_channels,
                        in_channels,
                        kernel_size[0],
                        kernel_size[1] // 2 + 1,
                        dtype=torch.int8,
                    ),
                    requires_grad=False,
                ),
            )
        elif n_bits == 8:
            self.register_parameter(
                name="weight",
                param=nn.Parameter(
                    torch.ones(
                        out_channels,
                        in_channels,
                        kernel_size[0],
                        kernel_size[1],
                        dtype=torch.int8,
                    ),
                    requires_grad=False,
                ),
            )
        else:
            self.register_parameter(
                name="weight",
                param=nn.Parameter(
                    torch.ones(
                        out_channels,
                        in_channels,
                        kernel_size[0],
                        kernel_size[1],
                        dtype=torch.int16,
                    ),
                    requires_grad=False,
                ),
            )

        self.register_parameter(
            name="bias", param=nn.Parameter(torch.zeros(out_channels))
        )

        self.register_parameter(
            name="scale", param=nn.Parameter(torch.ones(out_channels))
        )
        self.register_parameter(
            name="zero_point", param=nn.Parameter(torch.zeros(out_channels))
        )

        self.fweight = torch.tensor([False])

    def conv(self, tensor: torch.Tensor):
        if not self.fweight.any():
            self.fweight = dequant(
                self.weight, self.scale, self.zero_point, self.n_bits, self.w
            )
        return F.conv2d(
            tensor,
            self.fweight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

    def set_weight_bias(self, weight: torch.Tensor, bias: torch.Tensor):
        self.fweight = weight
        self.bias = bias

    def prepare(self, model: nn.Module):
        if not isinstance(model, nn.Module):
            raise TypeError
        raise NotImplementedError

    def quantize(self):
        obs = LBObserver(self.n_bits, "quantize_per_channel")
        obs(self.fweight)
        scale, zero_point = obs.calculate_qparams()
        self.weight = nn.Parameter(
            quant(self.fweight, scale, zero_point, self.n_bits, "quantize_per_channel"),
            requires_grad=False,
        )
        self.fweight = dequant(self.weight, scale, zero_point, self.n_bits, self.w)

    def forward(self, x):
        x = F.relu(self.conv(x))
        return x

    def size(self):
        torch.save(self.state_dict(), "temp.p")
        size = os.path.getsize("temp.p") / 1e6
        os.remove("temp.p")
        return size


class LBConv(ConvRelu):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int = 1,
        padding: int = 0,
        device=None,
        n_bits: int = 8,
    ):
        super(LBConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, device, n_bits
        )

    def forward(self, x):
        x = self.conv(x)
        return x
