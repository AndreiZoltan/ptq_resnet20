import torch
from torch import nn

# THIS FILE WILL BE REFACTORED
def fuse_conv_bn(model):
    previous_name = None

    for module_name in model._modules:
        previous_name = (
            module_name if previous_name is None else previous_name
        )  # Initialization

        conv_fused = fuse_single_conv_bn_pair(
            model._modules[module_name], model._modules[previous_name]
        )
        if conv_fused:
            model._modules[previous_name] = conv_fused
            model._modules[module_name] = nn.Identity()

        if len(model._modules[module_name]._modules) > 0:
            fuse_conv_bn(model._modules[module_name])

        previous_name = module_name

    return model


def fuse_single_conv_bn_pair(block1, block2):
    if isinstance(block1, nn.BatchNorm2d) and isinstance(block2, nn.Conv2d):
        m = block1
        conv = block2

        bn_st_dict = m.state_dict()
        conv_st_dict = conv.state_dict()

        # BatchNorm params
        eps = m.eps
        mu = bn_st_dict["running_mean"]
        var = bn_st_dict["running_var"]
        gamma = bn_st_dict["weight"]

        if "bias" in bn_st_dict:
            beta = bn_st_dict["bias"]
        else:
            beta = torch.zeros(gamma.size(0)).float().to(gamma.device)

        # Conv params
        W = conv_st_dict["weight"]
        if "bias" in conv_st_dict:
            bias = conv_st_dict["bias"]
        else:
            bias = torch.zeros(W.size(0)).float().to(gamma.device)

        denom = torch.sqrt(var + eps)
        b = beta - gamma.mul(mu).div(denom)
        A = gamma.div(denom)
        bias *= A
        A = A.expand_as(W.transpose(0, -1)).transpose(0, -1)

        W.mul_(A)
        bias.add_(b)

        conv.weight.data.copy_(W)

        if conv.bias is None:
            conv.bias = torch.nn.Parameter(bias)
        else:
            conv.bias.data.copy_(bias)

        return conv

    else:
        return False


def prepare(model, qmodel):
    previous_name = None

    for module_name in model._modules:
        if isinstance(model._modules[module_name], torch.nn.Conv2d):
            weight = getattr(model, module_name).weight.data
            bias = nn.Parameter(getattr(model, module_name).bias.data)
            getattr(qmodel, module_name).set_weight_bias(weight, bias)
            # print(getattr(model, module_name), getattr(qmodel, module_name))

        if isinstance(model._modules[module_name], torch.nn.Linear):
            weight = getattr(model, module_name).weight.data
            bias = nn.Parameter(getattr(model, module_name).bias.data)
            getattr(qmodel, module_name).set_weight_bias(weight, bias)

        if len(model._modules[module_name]._modules) > 0:
            prepare(model._modules[module_name], qmodel._modules[module_name])

    return model
