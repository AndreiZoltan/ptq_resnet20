import torch
from torch import nn
from torch.quantization import observer


class LBObserver(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        qscheme: str = "quantize_per_tensor",
        eps: torch.Tensor = torch.tensor([0.00001]),
    ):
        super(LBObserver, self).__init__()
        assert qscheme in {"quantize_per_channel", "quantize_per_tensor"}
        self.qscheme = qscheme
        if qscheme == "quantize_per_tensor":
            self.observer: observer = observer.default_histogram_observer()
        else:
            self.observer: observer = observer.default_per_channel_weight_observer()

        self.n_bits: int = n_bits
        self.quant_max = 2 ** (n_bits - 1)
        self.quant_min = -(2 ** (n_bits - 1))
        self.eps = eps
        self.scale: torch.Tensor = torch.ones(1)
        self.zero_point: torch.Tensor = torch.zeros(1)

    def update_qparams(self):
        max_val_pos = torch.max(self.observer.max_val, torch.zeros(1))
        min_val_neg = torch.min(self.observer.min_val, torch.zeros(1))
        if self.qscheme == "quantize_per_tensor":
            scale = (max_val_pos - min_val_neg) / float(self.quant_max - self.quant_min)
            self.scale = torch.max(scale, self.eps)
            zero_point = self.quant_min - torch.round(min_val_neg / scale).to(torch.int)
            self.zero_point = torch.clamp(zero_point, self.quant_min, self.quant_max)
        else:
            max_val_pos = torch.max(-min_val_neg, max_val_pos)
            scale = max_val_pos / (float(self.quant_max - self.quant_min) / 2)
            self.scale = torch.max(scale, self.eps)
            self.zero_point = torch.zeros(max_val_pos.shape[0])

    def calculate_qparams(self):
        self.update_qparams()
        return self.scale, self.zero_point

    def forward(self, x: torch.Tensor):
        self.observer(x)
        self.update_qparams()
        return x
