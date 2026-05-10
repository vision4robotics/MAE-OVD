"""
Lightweight Encoder: Multi-scale outputs F3(64), F4(128), F5(256).
Simple 5-stage CNN: input H×W → F3 ~1/8, F4 1/16, F5 1/32 downsampling, channels 64/128/256, suitable for pretraining and ONNX.
"""
import torch
import torch.nn as nn


def _conv_bn_relu(in_c, out_c, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class LightweightEncoder(nn.Module):
    """
    Multi-scale backbone: 5 stride-2 stages, taking outputs of last 3 stages as F3, F4, F5.
    Example: input 320×320 → F3 40×40×64, F4 20×20×128, F5 10×10×256.
    """

    def __init__(self, in_channels=3, out_channels=(64, 128, 256)):
        super().__init__()
        self.out_channels = list(out_channels)
        # F3=64, F4=128, F5=256
        c0, c1 = 32, 64
        c3, c4, c5 = out_channels[0], out_channels[1], out_channels[2]  # 64, 128, 256

        self.stem = _conv_bn_relu(in_channels, c0, stride=2)       # /2
        self.stage1 = _conv_bn_relu(c0, c1, stride=2)             # /4
        self.stage2 = _conv_bn_relu(c1, c3, stride=2)             # /8  -> F3
        self.stage3 = _conv_bn_relu(c3, c4, stride=2)              # /16 -> F4
        self.stage4 = _conv_bn_relu(c4, c5, stride=2)              # /32 -> F5

    def forward(self, x):
        x = self.stem(x)      # /2
        x = self.stage1(x)   # /4
        f3 = self.stage2(x)   # /8  (B, 64, H/8, W/8)
        f4 = self.stage3(f3)  # /16 (B, 128, H/16, W/16)
        f5 = self.stage4(f4)  # /32 (B, 256, H/32, W/32)
        return [f3, f4, f5]
