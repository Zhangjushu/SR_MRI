import torch.nn as nn
from torchsummary import summary
from utils import Upsampler, default_conv
from model.attention import Freq_CALayer, Layer_Self_Attention
import torch


class RCAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bn=False):
        super(RCAB, self).__init__()

        modules_body = []
        for i in range(2):
            modules_body.append(default_conv(n_feat, n_feat, kernel_size))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(nn.ReLU(True))
        modules_body.append(Freq_CALayer(n_feat))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class ResidualGroup(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, n_resblocks):
        super(ResidualGroup, self).__init__()

        modules_body = [
            RCAB(n_feat, kernel_size, reduction, bn=False) \
            for _ in range(n_resblocks)
        ]
        modules_body.append(default_conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class RFCAN(nn.Module):
    def __init__(self):
        super(RFCAN, self).__init__()

        n_resgroups = 10
        n_resblocks = 20
        n_feats = 64
        kernel_size = 3
        reduction = 16
        scale = 8
        n_colors = 3

        # define head module
        modules_head = [default_conv(n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                n_feats, kernel_size, reduction, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)
        ]

        # define tail module
        modules_tail = [
            Upsampler(scale, n_feats),
            default_conv(n_feats, n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)
        self.lsa = Layer_Self_Attention()
        self.layer_catconv = default_conv(n_feats * 10, n_feats, 1)
        self.last_catconv = default_conv(n_feats * 2, n_feats, 1)
        self.last_conv = default_conv(n_feats, n_feats, kernel_size)

    def forward(self, x):
        x = self.head(x)
        res = x

        group_feature = []
        for name, midlayer in self.body._modules.items():
            res = midlayer(res)
            group_feature.append(res)
        lsa = self.lsa(torch.cat(group_feature, dim=1))
        lsa = self.layer_catconv(lsa)

        res = self.last_conv(res)
        out = torch.cat([res, lsa], dim=1)
        y = self.last_catconv(out) + x
        y = self.tail(y)
        return y


if __name__ == '__main__':
    model = RFCAN()
    summary(model=model, input_size=[(3, 48, 48)], batch_size=16, device='cpu')


# 16,039,011
# Total params: 15,645,795