"""The one network both pipelines use: a small 4-level UNet (base 16, ~1.9M params).

Architecture was never the lever in this project -- labels and readout were.
384 px input, base 24 and deeper variants were all measured no better.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 3, base: int = 16, depth: int = 4):
        super().__init__()
        self.depth = depth
        chans = [base * (2 ** i) for i in range(depth + 1)]
        self.downs = nn.ModuleList()
        prev = in_ch
        for c in chans[:-1]:
            self.downs.append(DoubleConv(prev, c))
            prev = c
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(chans[-2], chans[-1])
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in range(depth):
            self.ups.append(nn.ConvTranspose2d(chans[-1 - i], chans[-2 - i], 2, stride=2))
            self.up_convs.append(DoubleConv(chans[-1 - i], chans[-2 - i]))
        self.head = nn.Conv2d(chans[0], out_ch, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for i in range(self.depth):
            x = self.ups[i](x)
            skip = skips[-1 - i]
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = self.up_convs[i](torch.cat([x, skip], dim=1))
        return self.head(x)


def load_checkpoint(path: str, device: str = "cpu") -> UNet:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"]
    net = UNet(out_ch=sd["head.weight"].shape[0], base=sd["head.weight"].shape[1])
    net.load_state_dict(sd)
    return net.to(device).eval()


def save_checkpoint(path: str, net: UNet, **extra) -> None:
    torch.save({"model": net.state_dict(),
                "cfg": {"base": net.head.weight.shape[1], "out_ch": net.head.weight.shape[0]},
                **extra}, path)
