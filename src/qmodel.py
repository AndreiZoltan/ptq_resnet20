import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

from pathlib import Path

PARENT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(r"{}".format(PARENT_DIR))
from lbtorch import ConvRelu, QFakeStub, LBConv, LBLinear

import os


class QBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, n_bits: int = 8):
        super(QBasicBlock, self).__init__()
        self.conv1 = ConvRelu(
            in_planes,
            planes,
            kernel_size=(3, 3),
            stride=stride,
            padding=1,
            n_bits=n_bits,
        )
        self.conv2 = LBConv(
            planes, planes, kernel_size=(3, 3), stride=1, padding=1, n_bits=n_bits
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                LBConv(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=(1, 1),
                    stride=stride,
                    n_bits=n_bits,
                )
            )

        self.fakestub = QFakeStub(n_bits)
        self.fakestub2 = QFakeStub(n_bits)

    def quantize(self):
        setattr(self.fakestub, "observe", False)
        setattr(self.fakestub2, "observe", False)
        self.conv1.quantize()
        self.conv2.quantize()

    def forward(self, x):
        x = self.fakestub(x)
        out = self.conv1(x)
        out = self.fakestub2(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, n_bits: int = 8, num_classes: int = 10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = ConvRelu(
            3, 64, kernel_size=(3, 3), stride=1, padding=0, n_bits=n_bits
        )
        self.layer1 = self._make_layer(
            block, 64, num_blocks[0], stride=1, n_bits=n_bits
        )
        self.layer2 = self._make_layer(
            block, 128, num_blocks[1], stride=2, n_bits=n_bits
        )
        self.layer3 = self._make_layer(
            block, 256, num_blocks[2], stride=2, n_bits=n_bits
        )
        self.layer4 = self._make_layer(
            block, 512, num_blocks[3], stride=2, n_bits=n_bits
        )
        self.linear = LBLinear(512 * block.expansion, num_classes)

        self.fakestub = QFakeStub(n_bits)
        self.fakestub2 = QFakeStub(n_bits)

    def _make_layer(self, block, planes, num_blocks, stride, n_bits):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, n_bits))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def quantize(self):
        setattr(self.fakestub, "observe", False)
        setattr(self.fakestub2, "observe", False)
        self.conv1.quantize()
        for child in list(self.named_children()):
            if "layer" in child[0]:
                for block in list(child[1].children()):
                    block.quantize()

    def forward(self, x):
        x = self.fakestub(x)
        out = self.conv1(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fakestub2(out)
        out = self.linear(out)
        return out

    def print_size_of_model(self):
        torch.save(self.state_dict(), "temp.p")
        print("Size (MB):", os.path.getsize("temp.p") / 1e6)
        os.remove("temp.p")


def qresnet20(n_bits):
    return ResNet(QBasicBlock, [2, 2, 3, 2], n_bits=n_bits)
