"""Lightweight single-output U-Net for XY-only Neural-SST.

The network does not directly predict controls or a complete trajectory. Instead,
it outputs an unnormalized logit at each location on the same pixel grid as the map,
indicating whether that location belongs to a feasible path. At runtime, ``sigmoid``
converts these logits to ``P_xy``, which is used as a spatial bias for SST state
sampling. OMPL remains responsible for collision checking and vehicle-dynamics
feasibility.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two ``Conv → GroupNorm → SiLU`` blocks that preserve spatial dimensions.

    A ``3×3`` convolution with ``padding=1`` preserves H/W, allowing U-Net skip
    connections to align pixel by pixel. Convolution bias is disabled because the
    following GroupNorm already provides a learnable offset. Eight GroupNorm groups
    do not depend on batch size, which suits the single-task inference common in this
    demo.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Transform ``[B, C_in, H, W]`` through the layers and return ``[B, C_out, H, W]``."""
        return self.layers(inputs)


class XYProbabilityUNet(nn.Module):
    """Predict one XY path-probability logit map from five conditioning maps.

    Input channels are occupancy, start heatmap, goal heatmap, sin(start yaw),
    and cos(start yaw).  The only output channel is the XY path logit map.
    """

    def __init__(self, input_channels: int = 5, base_channels: int = 32) -> None:
        super().__init__()
        # Double the channel count at each downsampling level. Four encoder stages
        # reduce a 224×224 input to 14×14.
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.encoders = nn.ModuleList([
            DoubleConv(input_channels, channels[0]),
            DoubleConv(channels[0], channels[1]),
            DoubleConv(channels[1], channels[2]),
            DoubleConv(channels[2], channels[3]),
        ])
        # MaxPool only reduces resolution. Save each stage's features as a skip
        # connection so the decoder can recover detail.
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(channels[3], channels[3] * 2)
        # Decoder input channels = upsampled features + same-scale encoder skip features.
        self.decoders = nn.ModuleList([
            DoubleConv(channels[3] * 2 + channels[3], channels[3]),
            DoubleConv(channels[3] + channels[2], channels[2]),
            DoubleConv(channels[2] + channels[1], channels[1]),
            DoubleConv(channels[1] + channels[0], channels[0]),
        ])
        # A 1×1 convolution mixes channels only, preserving each pixel's spatial
        # location while producing a single path logit.
        self.output = nn.Conv2d(channels[0], 1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return one unnormalized XY logit map with shape [B, 1, H, W]."""
        # Collect encoder features at every scale and use them in reverse to form
        # standard U-Net long skip connections.
        skips: list[torch.Tensor] = []
        features = inputs
        for encoder in self.encoders:
            features = encoder(features)
            skips.append(features)
            features = self.pool(features)
        features = self.bottleneck(features)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            # Interpolate explicitly to the skip's dimensions to prevent concatenation
            # misalignment if input sizes change or are odd.
            features = F.interpolate(features, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            # Concatenation provides the decoder with both high-level semantics and
            # low-level geometric edges at matching pixel locations.
            features = decoder(torch.cat((features, skip), dim=1))
        return self.output(features)
