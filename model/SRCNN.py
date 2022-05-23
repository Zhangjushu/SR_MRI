from torch import nn
from common import default_conv


class SRCNN(nn.Module):
    def __init__(self, num_channels=3):
        super(SRCNN, self).__init__()
        self.conv1 = default_conv(num_channels, 64, 9)
        self.conv2 = default_conv(64, 32, 5)
        self.conv3 = default_conv(32, num_channels, 5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
