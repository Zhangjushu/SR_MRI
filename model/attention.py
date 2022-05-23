from torch.nn import functional as F
import torch.nn as nn
import torch
import math
from torchsummary import summary


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


def get_1d_dct(i, freq, L):
    result = math.cos(math.pi * freq * (i + 0.5) / L) / math.sqrt(L)
    if freq == 0:
        return result
    else:
        return result * math.sqrt(2)


def get_dct_weights(width, height, channel):
    dct_weights = torch.zeros(channel, width, height)
    for t_x in range(width):
        for t_y in range(height):
            dct_weights[:, t_x, t_y] = get_1d_dct(t_x, t_x, width) * get_1d_dct(t_y, t_y, height)
    return dct_weights

class Freq_CALayer(nn.Module):
    def __init__(self, channel):
        super(Freq_CALayer, self).__init__()

        self.width = 48
        self.height = 48
        self.register_buffer('pre_computed_dct_weights', get_dct_weights(self.width, self.height, channel))
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // 16, 1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(channel // 16, channel, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        x_pooled = x
        if h != self.height or w != self.width:
            x_pooled = F.adaptive_avg_pool2d(x, (self.height, self.width))
        y = torch.sum(x_pooled * self.pre_computed_dct_weights, dim=(2, 3), keepdim=True)  # Cx1x1
        y = self.conv_du(y)
        return y.expand_as(x) * x


class Pixel_Self_Attention(nn.Module):
    def __init__(self):
        super(Pixel_Self_Attention, self).__init__()

        self.soft = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.size()
        proj_query = x.view(b, c, -1)  # C X HW
        proj_key = x.view(b, c, -1).permute(0, 2, 1)  # HW X C
        # energy = torch.bmm(proj_query, proj_key)  # C X C
        energy = torch.bmm(proj_key, proj_query)  # HW X HW
        # attention = self.soft(energy)  # C x C
        attention = self.soft(energy)  # HW x HW
        # out = torch.bmm(attention, proj_query)  # C x HW
        out = torch.bmm(proj_query, attention)  # C x HW
        out = out.view(b, c, h, w)  # C x H x W
        return self.gamma * out + x   #

class Channel_Self_Attention(nn.Module):
    def __init__(self):
        super(Channel_Self_Attention, self).__init__()

        self.soft = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.size()
        proj_query = x.view(b, c, -1)  # C X HW
        proj_key = x.view(b, c, -1).permute(0, 2, 1)  # HW X C
        energy = torch.bmm(proj_query, proj_key)  # C X C
        attention = self.soft(energy)  # C x C
        out = torch.bmm(attention, proj_query)  # C x HW
        out = out.view(b, c, h, w)  # C x H x W
        return self.gamma * out + x


class Layer_Self_Attention(nn.Module):
    def __init__(self):
        super(Layer_Self_Attention, self).__init__()

        self.psa = Pixel_Self_Attention()
        self.csa = Channel_Self_Attention()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        lsa = self.psa(x) + self.csa(x)
        return self.gamma * lsa


if __name__ == '__main__':
    model = Pixel_Self_Attention()
    summary(model=model, input_size=[(64, 32, 32)], batch_size=16, device='cpu')
