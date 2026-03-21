import os

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.ARF import arf
import torch_dct as dct

class ConvBlock(nn.Module):

    def __init__(self, input_channel, output_channel):
        super().__init__()
        # 用 nn.Sequential 定义一个顺序的层容器，其中包含多个层，按顺序执行。
        self.layers = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channel))  # 定义一个批量归一化层，归一化输出通道数为 output_channel

    # 前向传播
    def forward(self, inp):
        return self.layers(inp)  # 将输入 inp 传递给 self.layers，返回卷积块的输出

class ConvBlock1(nn.Module):

    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.conv1 = arf(
            input_channel,
            output_channel,
            3,
            1,
            1
        )
        self.bn = nn.BatchNorm2d(output_channel)

    # 前向传播
    def forward(self, inp, epoch, hw_range):
        inp = self.conv1(inp, epoch, hw_range)
        # print('hhhhhhh')
        # inp = self.conv(inp)
        inp = self.bn(inp)
        return inp  # 将输入 inp 传递给 self.layers，返回卷积块的输出

class BackBone(nn.Module):

    def __init__(self, num_channel=64):
        super().__init__()

        self.layers = nn.Sequential(
            ConvBlock1(3, num_channel),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            ConvBlock1(num_channel, num_channel),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            ConvBlock1(num_channel, num_channel),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            ConvBlock1(num_channel, num_channel),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2)
        )

    def forward_all_layers(self, x, epoch, hw_range):           # , epoch, hw_range
        for layer in self.layers:
            if isinstance(layer, ConvBlock1):
                x = layer(x, epoch, hw_range)
            else:
                x = layer(x)
        return x

    def forward(self, inp, epoch, hw_range):
        x = self.forward_all_layers(inp, epoch, hw_range)
        return x

class BackBone1(nn.Module):

    def __init__(self, num_channel=64):
        super().__init__()

        self.layers = nn.Sequential(
            ConvBlock1(3, num_channel),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            ConvBlock1(num_channel, num_channel),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            ConvBlock1(num_channel, num_channel),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            ConvBlock1(num_channel, num_channel),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2)
        )

    def forward_all_layers(self, x, epoch, hw_range):           # , epoch, hw_range
        for layer in self.layers:
            if isinstance(layer, ConvBlock1):
                x = layer(x, epoch, hw_range)
            else:
                x = layer(x)
        return x

    def forward(self, inp, epoch, hw_range):
        x = self.forward_all_layers(inp, epoch, hw_range)
        return x


class SpectralSoftMask(nn.Module):
    def __init__(self, channels=3, mask_size=16):
        super().__init__()
        self.mask = nn.Parameter(torch.ones(1, channels, mask_size, mask_size))
    def forward(self, freq_feat):
        B, C, H, W = freq_feat.shape
        mask = F.interpolate(self.mask, size=(H, W), mode='bilinear', align_corners=False)
        return freq_feat * mask


class dctBackbone(nn.Module):
    def __init__(self, num_channel=64, mask_size=16):
        super().__init__()
        self.spatial_backbone = BackBone(num_channel)
        self.freq_backbone = BackBone1(num_channel)
        self.spectral_mask = SpectralSoftMask(channels=3, mask_size=mask_size)

        self.fuse = SpatialAttentionFusion(in_channels=num_channel, num_branches=2, reduction_ratio=4)

    def forward(self, inp, epoch, hw_range):
        # 空域分支
        spatial_feat = self.spatial_backbone(inp, epoch, hw_range)
        # 频域分支
        freq_inp = dct.dct_2d(inp)
        # 在频域突出细粒度特征
        freq_inp = self.spectral_mask(freq_inp)       # [B, 3, H, W]
        freq_feat = dct.idct_2d(freq_inp)
        freq_feat = self.freq_backbone(freq_feat, epoch, hw_range)

        feat = self.fuse(freq_feat,spatial_feat)
        return feat

class SpatialAttentionFusion(nn.Module):
    def __init__(self, in_channels=64, num_branches=4, reduction_ratio=4):
        super().__init__()
        self.mask_generator_conv = nn.Sequential(
            nn.Conv2d(in_channels * num_branches, in_channels // reduction_ratio, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels // reduction_ratio, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, num_branches, kernel_size=1)
        )
        self.softmax = nn.Softmax(dim=1)
    def forward(self, *features):
        concatenated_features = torch.cat(features, dim=1)
        spatial_masks = self.mask_generator_conv(concatenated_features)
        spatial_masks = self.softmax(spatial_masks)
        fused_feat = torch.zeros_like(features[0])
        for i, feat in enumerate(features):
            mask = spatial_masks[:, i:i+1, :, :]
            fused_feat += feat * mask
        return fused_feat


if __name__ == '__main__':
    img = torch.randn(4, 3, 84, 84)
    epoch = 0
    hw_range = (84, 84)
    backbone = dctBackbone(num_channel=64)
    # backbone = dctBackbone(
    #     num_channel=64, fusion='cat', cond_dim=64,
    #     K=3, band_tau=1.0, learnable_bands=True, hard_gate=True
    # )
    feat = backbone(img, epoch, hw_range)
    print(feat.shape)


