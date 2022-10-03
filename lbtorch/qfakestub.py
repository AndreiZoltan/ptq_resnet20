import torch
from torch import nn
from torch.quantization import observer
from lbtorch import quant, dequant


class QFakeStub(nn.Module):
    def __init__(self, n_bits: int = 8, eps: torch.Tensor = torch.tensor([0.00001])):
        super(QFakeStub, self).__init__()
        self.observe: bool = True
        self.observer: observer = observer.default_histogram_observer()

        self.n_bits: int = n_bits
        self.quant_max = 2 ** (n_bits - 1)
        self.quant_min = -(2 ** (n_bits - 1))
        self.eps = eps
        self.scale: torch.Tensor = torch.ones(1)
        self.zero_point: torch.Tensor = torch.zeros(1)

    def update_qparams(self):
        max_val_pos = torch.max(self.observer.max_val, torch.zeros(1))
        min_val_neg = torch.min(self.observer.min_val, torch.zeros(1))
        scale = (max_val_pos - min_val_neg) / float(self.quant_max - self.quant_min)
        self.scale = torch.max(scale, self.eps)
        zero_point = self.quant_min - torch.round(min_val_neg / scale).to(torch.int)
        self.zero_point = torch.clamp(zero_point, self.quant_min, self.quant_max)

    def calculate_qparams(self):
        self.update_qparams()
        return self.scale, self.zero_point

    def forward(self, x: torch.Tensor):
        if self.observe:
            self.observer(x)
            self.update_qparams()
            return x
        else:
            out = quant(
                x, self.scale, self.zero_point, self.n_bits, "quantize_per_tensor"
            )
            return dequant(out, self.scale, self.zero_point, self.n_bits, x.shape[-1])
