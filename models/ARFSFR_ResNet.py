from models.ARFS import resnet12 as resnet
from models.ARFF import resnet12 as dctresnet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as dct

class dctBackbone(nn.Module):
    def __init__(self, mask_size=16):
        super().__init__()
        self.spatial_backbone = resnet()
        self.freq_backbone = dctresnet()

        self.fuse = SpatialAttentionFusion(in_channels=640, num_branches=2, reduction_ratio=4)
        self.spectral_mask = SpectralSoftMask(channels=3, mask_size=mask_size)

    def forward(self, inp, epoch, hw_range):
        # 空域分支
        spatial_feat = self.spatial_backbone(inp, epoch, hw_range)
        # 频域分支
        x = dct.dct_2d(inp)
        x = self.spectral_mask(x)
        inp = dct.idct_2d(x)
        freq_feat = self.freq_backbone(inp, epoch, hw_range)

        feat = self.fuse(freq_feat,spatial_feat)
        return feat

class SpatialAttentionFusion(nn.Module):
    def __init__(self, in_channels=64, num_branches=4, reduction_ratio=8):
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

class SpectralSoftMask(nn.Module):
    def __init__(self, channels=3, mask_size=16):
        super().__init__()
        self.mask = nn.Parameter(torch.ones(1, channels, mask_size, mask_size))
    def forward(self, freq_feat):
        B, C, H, W = freq_feat.shape
        mask = F.interpolate(self.mask, size=(H, W), mode='bilinear', align_corners=False)
        return freq_feat * mask

if __name__ == '__main__':
    img = torch.randn(4, 3, 84, 84)

    backbone = dctBackbone()

    feat = backbone(img, epoch=0, hw_range=[1,9])
    print(feat.shape)


