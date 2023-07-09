import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2

class ResidualUpsample(nn.Module):
    def __init__(self, input_channels, output_channels, skip_channels):
        super(ResidualUpsample, self).__init__()

        self.conv1x1 = nn.Conv2d(skip_channels, output_channels, kernel_size=1)
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, skip_connection=None):
        x = self.upsample(x)
        x = self.conv(x)
        if skip_connection is not None:
            skip_connection = self.conv1x1(skip_connection)
            x += F.interpolate(skip_connection, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return x



class MobileNetV2UNetInverter(nn.Module):
    def __init__(self):
        super().__init__()
        self.mobilenet = mobilenet_v2(pretrained=True).features

        self.up1 = ResidualUpsample(320, 64, skip_channels=96)
        self.up2 = ResidualUpsample(64, 32, skip_channels=32)
        self.up3 = ResidualUpsample(32, 24, skip_channels=24)
        self.up4 = ResidualUpsample(24, 16, skip_channels=16)
        self.up5 = nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2)

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.mobilenet):
            x = layer(x)
            if i in [3, 6, 13, 17]:
                features.append(x)
        features.reverse()

        out = self.up1(features[0], features[1])
        out = self.up2(out, features[2])
        out = self.up3(out, features[3])
        out = self.up4(out, None)
        out = self.up5(out)

        return out