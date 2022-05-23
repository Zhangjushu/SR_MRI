import torch.nn as nn
from torchsummary import summary
from utils import Upsampler, default_conv
from model.attention import CALayer


class RCAB(nn.Module):
    def __init__(self, n_feat, kernel_size, bn=False):
        super(RCAB, self).__init__()

        modules_body = []
        for i in range(2):
            modules_body.append(default_conv(n_feat, n_feat, kernel_size))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(nn.ReLU(True))
        modules_body.append(CALayer(n_feat))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class ResidualGroup(nn.Module):
    def __init__(self, n_feat, kernel_size, n_resblocks):
        super(ResidualGroup, self).__init__()

        modules_body = [
            RCAB(n_feat, kernel_size, bn=False) \
            for _ in range(n_resblocks)
        ]
        modules_body.append(default_conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class RCAN(nn.Module):
    def __init__(self):
        super(RCAN, self).__init__()

        n_resgroups = 10
        n_resblocks = 20
        n_feats = 64
        kernel_size = 3
        scale = 4
        n_colors = 3

        # define head module
        modules_head = [default_conv(n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                n_feats, kernel_size, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)
        ]
        modules_body.append(default_conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            Upsampler(scale, n_feats),
            default_conv(n_feats, n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x


if __name__ == '__main__':
    model = RCAN()
    summary(model=model, input_size=[(3, 64, 64)], batch_size=16, device='cpu')
