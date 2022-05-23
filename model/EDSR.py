import torch
import torch.nn as nn
import math
from common import Upsampler, default_conv


class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()

        self.conv1 = default_conv(256, 256, 3)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = default_conv(256, 256, 3)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.conv1(x))
        output = self.conv2(output)
        output *= 0.1
        output = torch.add(output, identity_data)
        return output


class EDSR(nn.Module):
    def __init__(self):
        super(EDSR, self).__init__()

        self.conv_input = default_conv(3, 256, 3)

        self.residual = self.make_layer(_Residual_Block, 32)

        self.conv_mid = default_conv(256, 256, 3)

        self.upscale = nn.Sequential(
            Upsampler(4, 256),
            default_conv(256, 3, 3)
        )

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_input(x)
        residual = out
        out = self.conv_mid(self.residual(out))
        out = torch.add(out, residual)
        out = self.upscale(out)
        return out
