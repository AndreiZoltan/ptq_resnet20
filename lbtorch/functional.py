import torch


def set_tensor_attr(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    n_bits: int,
    qscheme: str,
):
    setattr(tensor, "scale", scale)
    setattr(tensor, "zero_point", zero_point)
    setattr(tensor, "n_bits", n_bits)
    setattr(tensor, "qscheme", qscheme)


def to_int4(tensor: torch.Tensor):
    missing_dims = 4 - len(tensor.shape)
    for _ in range(missing_dims):
        tensor = torch.unsqueeze(tensor, 0)
    tensor = tensor.to(torch.int8)
    width = tensor.shape[3] - 1
    for w in range(tensor.shape[3]):
        tensor[:, :, :, w // 2] &= (tensor[:, :, :, w] & 0xF) << (
            4 * ((w + 1) % 2)
        ) | 0xF << (4 * (w % 2))

    tensor = tensor[:, :, :, : width // 2 + 1]
    for i in range(missing_dims):
        tensor = torch.squeeze(tensor, 0)
    setattr(tensor, "w", width + 1)
    return tensor


def to_int2(tensor: torch.Tensor):
    missing_dims = 4 - len(tensor.shape)
    for _ in range(missing_dims):
        tensor = torch.unsqueeze(tensor, 0)
    tensor = tensor.to(torch.int8)
    shift = {0: 3, 1: 2, 2: 1, 3: 0}
    fill = {0: 0x3F, 1: 0xCF, 2: 0xF3, 3: 0xFC}
    width = tensor.shape[3] - 1
    for w in range(tensor.shape[3]):
        tensor[:, :, :, w // 4] &= (
            (tensor[:, :, :, w] & 0x3) << (2 * (shift[w % 4]))
        ) | fill[w % 4]

    tensor = tensor[:, :, :, : width // 4 + 1]
    for i in range(missing_dims):
        tensor = torch.squeeze(tensor, 0)
    setattr(tensor, "w", width + 1)
    return tensor


def quant(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    n_bits: int,
    qscheme: str,
) -> torch.Tensor:
    """
    https://github.com/pytorch/pytorch/issues/74540

    good paper to try https://arxiv.org/pdf/1909.13144.pdf
    """
    assert qscheme in {"quantize_per_channel", "quantize_per_tensor"}
    assert n_bits in {2, 4, 8, 16}
    if qscheme == "quantize_per_channel":
        assert len(tensor.shape) == 4
        C_out, C_int, H, W = tensor.shape
        assert zero_point.shape[0] == C_out
        assert scale.shape[0] == C_out
        tensor = torch.round(
            tensor / scale.reshape(C_out, 1, 1, 1) + zero_point.reshape(C_out, 1, 1, 1)
        )
        if n_bits == 2:
            tensor = torch.clip(tensor, -(2 ** (n_bits - 1)), 2 ** (n_bits - 1) - 1)
            tensor = tensor.to(torch.int8)
            tensor = to_int2(tensor)
            set_tensor_attr(tensor, scale, zero_point, n_bits, qscheme)
            return tensor
        elif n_bits == 4:
            tensor = torch.clip(tensor, -(2 ** (n_bits - 1)), 2 ** (n_bits - 1) - 1)
            tensor = tensor.to(torch.int8)
            tensor = to_int4(tensor)
            set_tensor_attr(tensor, scale, zero_point, n_bits, qscheme)
            return tensor
        elif n_bits == 8:
            tensor = torch.clip(tensor, -(2 ** (n_bits - 1)), 2 ** (n_bits - 1) - 1)
            tensor = tensor.to(torch.int8)
            set_tensor_attr(tensor, scale, zero_point, n_bits, qscheme)
            return tensor
        else:
            tensor = torch.clip(tensor, -(2 ** (n_bits - 1)), 2 ** (n_bits - 1) - 1)
            tensor = tensor.to(torch.int16)
            set_tensor_attr(tensor, scale, zero_point, n_bits, qscheme)
            return tensor

    else:
        tensor = torch.round(tensor / scale + zero_point)
        assert zero_point.shape[0] == 1
        assert scale.shape[0] == 1
        if n_bits == 2:
            tensor = torch.clip(tensor, -(2 ** (n_bits - 1)), 2 ** (n_bits - 1) - 1)
            tensor = tensor.to(torch.int8)
            tensor = to_int2(tensor)
            set_tensor_attr(tensor, scale, zero_point, n_bits, qscheme)
            return tensor
        elif n_bits == 4:
            tensor = torch.clip(tensor, -(2 ** (n_bits - 1)), 2 ** (n_bits - 1) - 1)
            tensor = tensor.to(torch.int8)
            tensor = to_int4(tensor)
            set_tensor_attr(tensor, scale, zero_point, n_bits, qscheme)
            return tensor
        elif n_bits == 8:
            tensor = torch.clip(tensor, -(2 ** (n_bits - 1)), 2 ** (n_bits - 1) - 1)
            tensor = tensor.to(torch.int8)
            set_tensor_attr(tensor, scale, zero_point, n_bits, qscheme)
            return tensor
        else:
            tensor = torch.clip(tensor, -(2 ** (n_bits - 1)), 2 ** (n_bits - 1) - 1)
            tensor = tensor.to(torch.int16)
            set_tensor_attr(tensor, scale, zero_point, n_bits, qscheme)
            return tensor


def dequant(tensor: torch.Tensor, scale, zero_point, n_bits, w):
    if scale.shape[0] == 1:
        qscheme = "quantize_per_tensor"
    else:
        qscheme = "quantize_per_channel"
    assert qscheme in {"quantize_per_channel", "quantize_per_tensor"}

    if n_bits == 2:
        missing_dims = 4 - len(tensor.shape)
        for _ in range(missing_dims):
            tensor = torch.unsqueeze(tensor, 0)
        tensor = torch.cat(
            (
                tensor,
                torch.zeros(*tensor.shape[:-1], w - tensor.shape[-1], dtype=torch.int8),
            ),
            dim=-1,
        )
        assert tensor.dtype == torch.int8  # isinstance doesn't work with torch
        shift = {0: 3, 1: 2, 2: 1, 3: 0}
        fill = {0: 0xC0, 1: 0x30, 2: 0xC, 3: 0x3}
        for w in range(tensor.shape[3] - 1, -1, -1):
            val = (
                tensor[:, :, :, w // 4]
                & (fill[w % 4]) >> 2 * shift[w % 4]
            )
            mask = torch.ge(val & (1 << 1), 1)
            val[mask] |= 0xFC
            tensor[:, :, :, w] = val
        for i in range(missing_dims):
            tensor = torch.squeeze(tensor, 0)

    elif n_bits == 4:
        missing_dims = 4 - len(tensor.shape)
        for _ in range(missing_dims):
            tensor = torch.unsqueeze(tensor, 0)
        tensor = torch.cat(
            (
                tensor,
                torch.zeros(*tensor.shape[:-1], w - tensor.shape[-1], dtype=torch.int8),
            ),
            dim=-1,
        )
        assert tensor.dtype == torch.int8  # isinstance doesn't work with torch
        for w in range(tensor.shape[3] - 1, -1, -1):
            val = tensor[:, :, :, w // 2] & (
                0xF << 4 * ((w + 1) % 2)
            ) >> 4 * ((w + 1) % 2)
            mask = torch.ge(val & (1 << 3), 1)
            val[mask] |= 0xF << 4
            tensor[:, :, :, w] = val
        for i in range(missing_dims):
            tensor = torch.squeeze(tensor, 0)

    if qscheme == "quantize_per_channel":
        C_out, C_int, H, W = tensor.shape
        tensor = tensor.to(torch.float32)
        tensor = (tensor - zero_point.reshape(C_out, 1, 1, 1)) * scale.reshape(
            C_out, 1, 1, 1
        )
        return tensor
    else:
        tensor = tensor.to(torch.float32)
        tensor = (tensor - zero_point) * scale
        return tensor
