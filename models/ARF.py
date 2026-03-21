import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleSpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=7, padding=3)
    def forward(self, x):
        attn = torch.sigmoid(self.conv(x))
        return x * attn  # 注意力特征

class arf(nn.Module):

    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.inc = inc
        self.outc = outc
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.i_list = [33, 35, 37, 73, 53, 55, 57, 75, 77]
        self.convs = nn.ModuleList([
            nn.Conv2d(inc, outc, kernel_size=(i // 10, i % 10), stride=(i // 10, i % 10), padding=0)
            for i in self.i_list
        ])
        self.spatial_attn = SimpleSpatialAttention(inc)
        self.m_conv = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(outc, outc, kernel_size=3, padding=1),
            nn.Tanh()
        )
        self.b_conv = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(outc, outc, kernel_size=3, padding=1)
        )
        self.l_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inc, 1, 1),
            nn.Sigmoid()
        )
        self.w_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inc, 1, 1),
            nn.Sigmoid()
        )
        self.dropout2 = nn.Dropout2d(0.2)

    def forward(self, x, epoch, hw_range):
        scale = hw_range[1] // 9
        x_attn = self.spatial_attn(x) # 空间注意力
        m = self.m_conv(x)
        bias = self.b_conv(x)
        l = self.l_conv(x_attn) * (hw_range[1] - 1) + 1
        w = self.w_conv(x_attn) * (hw_range[1] - 1) + 1
        mean_l = l.mean().item()
        mean_w = w.mean().item()
        N_X = int(mean_l // scale)
        N_Y = int(mean_w // scale)
        N_X = N_X - 1 if N_X % 2 == 0 else N_X
        N_Y = N_Y - 1 if N_Y % 2 == 0 else N_Y
        N_X, N_Y = max(N_X, 3), max(N_Y, 3)
        N_X, N_Y = min(N_X, 7), min(N_Y, 7)
        N = N_X * N_Y
        self.last_kernel = (N_X, N_Y)

        # print(f"[DEBUG] kernel size selected = ({N_X}, {N_Y})")

        # 采样点全局共享，只生成一次 offset
        # 构造自适应核的采样格局中心点
        center_x = torch.linspace(-1, 1, N_X, device=x.device)
        center_y = torch.linspace(-1, 1, N_Y, device=x.device)
        grid_y, grid_x = torch.meshgrid(center_x, center_y, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1).view(1, N_X, N_Y, 2)  # 归一化到[-1,1]

        # 扩展到全图
        grid = F.interpolate(grid.permute(0, 3, 1, 2), size=x.shape[2:], mode='bilinear', align_corners=True)
        grid = grid.permute(0, 2, 3, 1)  # B x H x W x 2
        B = x.size(0)
        if grid.size(0) == 1:
            grid = grid.expand(B, -1, -1, -1)  # [B, H, W, 2]

        # grid_sample采样
        sampled = F.grid_sample(x, grid, mode='bilinear', align_corners=False)

        # 卷积分支
        idx = self.i_list.index(N_X * 10 + N_Y)
        x_offset = self.dropout2(sampled)
        x_offset = self.convs[idx](x_offset)

        # 保证尺寸对齐
        if x_offset.shape[-2:] != m.shape[-2:]:
            x_offset = F.interpolate(x_offset, size=m.shape[-2:], mode='bilinear', align_corners=False)
        out = x_offset * m + bias
        return out



if __name__ == '__main__':
    x = torch.randn(4, 32, 24, 24)
    arconv = arf(32, 64)
    y = arconv(x, epoch=1, hw_range=[1, 9])
    print(y.shape)



