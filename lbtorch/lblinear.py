import torch
from torch import nn
import torch.nn.functional as F

import os

from .functional import quant, dequant
from .lbobserver import LBObserver


class LBLinear(nn.Module):
    def __init__(self, in_features, out_features, device=None, n_bits=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device

        assert n_bits in {2, 4, 8, 16}
        self.n_bits = n_bits
        self.w = in_features
        if n_bits == 2:
            self.register_parameter(
                name="weight",
                param=nn.Parameter(
                    torch.ones(
                        out_features,
                        in_features // 4 + 1,
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
                        out_features,
                        in_features // 2 + 1,
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
                        out_features,
                        in_features,
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
                        out_features,
                        in_features,
                        dtype=torch.int16,
                    ),
                    requires_grad=False,
                ),
            )

        self.register_parameter(
            name="bias", param=nn.Parameter(torch.zeros(out_features))
        )

        self.register_parameter(name="scale", param=nn.Parameter(torch.ones(1)))
        self.register_parameter(name="zero_point", param=nn.Parameter(torch.zeros(1)))

        self.fweight = torch.tensor([False])

    def linear(self, tensor: torch.Tensor):
        if not self.fweight.any():
            self.fweight = dequant(
                self.weight, self.scale, self.zero_point, self.n_bits, self.w
            )
        return F.linear(
            tensor.to(torch.float32), self.fweight.to(torch.float32), self.bias
        )

    def set_weight_bias(
        self, weight: torch.Tensor, bias: torch.Tensor = torch.tensor([False])
    ):
        self.fweight = weight
        if bias.any():
            self.bias = bias

    def prepare(self, model: nn.Module):
        if not isinstance(model, nn.Module):
            raise TypeError
        raise NotImplementedError

    def quantize(self):
        obs = LBObserver(self.n_bits, "quantize_per_tensor")
        obs(self.fweight)
        scale, zero_point = obs.calculate_qparams()
        self.weight = nn.Parameter(
            quant(self.fweight, scale, zero_point, self.n_bits, "quantize_per_channel"),
            requires_grad=False,
        )
        self.fweight = dequant(self.weight, scale, zero_point, self.n_bits, self.w)

    def forward(self, x):
        x = self.linear(x)
        return x

    def size(self):
        torch.save(self.state_dict(), "temp.p")
        size = os.path.getsize("temp.p") / 1e6
        os.remove("temp.p")
        return size
