# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class SimpleSpatialAttention(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.conv = nn.Conv2d(channels, 1, kernel_size=7, padding=3)
#     def forward(self, x):
#         attn = torch.sigmoid(self.conv(x))
#         return x * attn  # 注意力特征
#
# class arf(nn.Module):
#
#     def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1):
#         super().__init__()
#         self.inc = inc
#         self.outc = outc
#         self.kernel_size = kernel_size
#         self.padding = padding
#         self.stride = stride
#
#         self.i_list = [33, 35, 37, 73, 53, 55, 57, 75, 77]
#         self.convs = nn.ModuleList([
#             nn.Conv2d(inc, outc, kernel_size=(i // 10, i % 10), stride=(i // 10, i % 10), padding=0)
#             for i in self.i_list
#         ])
#         self.spatial_attn = SimpleSpatialAttention(inc)
#         self.m_conv = nn.Sequential(
#             nn.Conv2d(inc, outc, kernel_size=3, padding=1),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(outc, outc, kernel_size=3, padding=1),
#             nn.Tanh()
#         )
#         self.b_conv = nn.Sequential(
#             nn.Conv2d(inc, outc, kernel_size=3, padding=1),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(outc, outc, kernel_size=3, padding=1)
#         )
#         self.l_conv = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(inc, 1, 1),
#             nn.Sigmoid()
#         )
#         self.w_conv = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(inc, 1, 1),
#             nn.Sigmoid()
#         )
#         self.dropout2 = nn.Dropout2d(0.2)
#
#     def forward(self, x, epoch, hw_range):
#         scale = hw_range[1] // 9
#         x_attn = self.spatial_attn(x) # 空间注意力
#         m = self.m_conv(x)
#         bias = self.b_conv(x)
#         l = self.l_conv(x_attn) * (hw_range[1] - 1) + 1
#         w = self.w_conv(x_attn) * (hw_range[1] - 1) + 1
#         mean_l = l.mean().item()
#         mean_w = w.mean().item()
#         N_X = int(mean_l // scale)
#         N_Y = int(mean_w // scale)
#         N_X = N_X - 1 if N_X % 2 == 0 else N_X
#         N_Y = N_Y - 1 if N_Y % 2 == 0 else N_Y
#         N_X, N_Y = max(N_X, 3), max(N_Y, 3)
#         N_X, N_Y = min(N_X, 7), min(N_Y, 7)
#         N = N_X * N_Y
#         self.last_kernel = (N_X, N_Y)
#
#         # print(f"[DEBUG] kernel size selected = ({N_X}, {N_Y})")
#
#         # 采样点全局共享，只生成一次 offset
#         # 构造自适应核的采样格局中心点
#         center_x = torch.linspace(-1, 1, N_X, device=x.device)
#         center_y = torch.linspace(-1, 1, N_Y, device=x.device)
#         grid_y, grid_x = torch.meshgrid(center_x, center_y, indexing='ij')
#         grid = torch.stack([grid_x, grid_y], dim=-1).view(1, N_X, N_Y, 2)  # 归一化到[-1,1]
#
#         # 扩展到全图
#         grid = F.interpolate(grid.permute(0, 3, 1, 2), size=x.shape[2:], mode='bilinear', align_corners=True)
#         grid = grid.permute(0, 2, 3, 1)  # B x H x W x 2
#         B = x.size(0)
#         if grid.size(0) == 1:
#             grid = grid.expand(B, -1, -1, -1)  # [B, H, W, 2]
#
#         # grid_sample采样
#         sampled = F.grid_sample(x, grid, mode='bilinear', align_corners=False)
#
#         # 卷积分支
#         idx = self.i_list.index(N_X * 10 + N_Y)
#         x_offset = self.dropout2(sampled)
#         x_offset = self.convs[idx](x_offset)
#
#         # 保证尺寸对齐
#         if x_offset.shape[-2:] != m.shape[-2:]:
#             x_offset = F.interpolate(x_offset, size=m.shape[-2:], mode='bilinear', align_corners=False)
#         out = x_offset * m + bias
#         return out
#
#
#
# if __name__ == '__main__':
#     x = torch.randn(4, 32, 24, 24)
#     arconv = arf(32, 64)
#     y = arconv(x, epoch=1, hw_range=[1, 9])
#     print(y.shape)
#
#
#
import torch
import torch.nn as nn
import torch.nn.functional as F


class arf(nn.Module):
    """
    改造版：     ****** 到9
    - 保留：空间注意力 + 预测 N_X/N_Y 决定核尺寸 + grid_sample + 对应 conv + m/bias 调制
    - 改进：l, w 不再只是算 N_X/N_Y，而是生成一个连续的 offset 场去形变 grid，
            这样 l_conv / w_conv 在“采样结构”上是可导、可学习的
    """
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.inc = inc
        self.outc = outc
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        # if Conv4
        # self.i_list = [33, 35, 37, 73, 53, 55, 57, 75, 77, 39,59,79, 93, 95, 97, 99]
        # self.convs = nn.ModuleList([
        #     nn.Conv2d(inc, outc,
        #               kernel_size=(i // 10, i % 10),
        #               stride=(i // 10, i % 10),
        #               padding=0)
        #     for i in self.i_list
        # ])

        # ResNet12
        self.i_list = [33, 35, 37, 73, 53, 55, 57, 75, 77, 39, 59, 79, 93, 95, 97, 99]

        self.convs = nn.ModuleList([
            nn.Conv2d(
                inc,
                outc,
                kernel_size=(i // 10, i % 10),
                stride=1,
                padding=((i // 10) // 2, (i % 10) // 2),
                bias=False
            )
            for i in self.i_list
        ])

        # self.spatial_attn = SimpleSpatialAttention(inc)

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

        # ★ 关键改动：l_conv / w_conv 变成“空间场”，不再 AdaptiveAvgPool2d(1)
        mid = max(inc // 4, 8)
        self.l_conv = nn.Sequential(
            nn.Conv2d(inc, mid, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid, 1, kernel_size=3, padding=1),
            nn.Sigmoid()   # [B,1,H,W] -> 再映射到 [1, hw_range] 用
        )
        self.w_conv = nn.Sequential(
            nn.Conv2d(inc, mid, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.dropout2 = nn.Dropout2d(0.1)

        # 记录选中的 kernel 尺寸，方便你 debug
        self.last_kernel = None

        self.post_attn = ECABlock(outc, k_size=3)

    def forward(self, x, epoch, hw_range):
        """
        x: [B,C,H,W]
        epoch, hw_range: 保留原接口，hw_range 用来控制 l/w 的范围
        """
        B, C, H, W = x.shape
        assert isinstance(hw_range, list) and len(hw_range) == 2
        hmin, hmax = hw_range
        scale = hmax // 9
        if hmin == 1 and hmax == 3:
            scale = 1

        # 1. 空间注意力
        # x_attn = self.spatial_attn(x)  # [B,C,H,W]

        # 2. multiplicative / bias
        m = self.m_conv(x)            # [B,outc,Hm,Wm]
        bias = self.b_conv(x)
        Hm, Wm = m.shape[-2:]

        # 3. l, w 空间场（连续值），同时也用全局平均粗略决定 N_X/N_Y
        #    l_raw, w_raw ∈ [0,1] -> 映射到 [1, hmax]
        l_raw = self.l_conv(x)                # [B,1,H,W]
        w_raw = self.w_conv(x)                # [B,1,H,W]
        l = l_raw * (hmax - 1) + 1
        w = w_raw * (hmax - 1) + 1

        # --- 3.1 旧思路：用全局均值粗略选一个离散核尺寸（不可导，作为 heuristic 就好） ---
        with torch.no_grad():
            mean_l = l.mean().item()
            mean_w = w.mean().item()
            N_X = int(mean_l // scale)
            N_Y = int(mean_w // scale)
            if N_X % 2 == 0:
                N_X -= 1
            if N_Y % 2 == 0:
                N_Y -= 1
            N_X, N_Y = max(N_X, 3), max(N_Y, 3)
            N_X, N_Y = min(N_X, 9), min(N_Y, 9)
            self.last_kernel = (N_X, N_Y)

        N = N_X * N_Y

        # 4. 构造基础 grid（与你原来一致），只依赖 N_X/N_Y（离散）
        center_x = torch.linspace(-1, 1, N_X, device=x.device)
        center_y = torch.linspace(-1, 1, N_Y, device=x.device)
        grid_y, grid_x = torch.meshgrid(center_x, center_y, indexing='ij')
        base_grid = torch.stack([grid_x, grid_y], dim=-1).view(1, N_X, N_Y, 2)  # [1,NX,NY,2]

        base_grid = F.interpolate(
            base_grid.permute(0, 3, 1, 2),
            size=x.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        base_grid = base_grid.permute(0, 2, 3, 1)   # [1,H,W,2]
        if B > 1:
            base_grid = base_grid.expand(B, -1, -1, -1)  # [B,H,W,2]

        # 5. ★ 关键改动：用 l, w 生成一个 “连续 offset 场” 去形变 grid
        #    l, w 当前范围大约是 [1, hmax]，先归一化到 [-1, 1] 再缩放为小的偏移
        #    注意：grid[...,0] 是 x (width)，grid[...,1] 是 y (height)
        hw_mid = (hmax + 1) / 2.0
        l_norm = (l - hw_mid) / hw_mid         # [B,1,H,W] in [-1,1] roughly
        w_norm = (w - hw_mid) / hw_mid

        # 把 l/w 映射到小范围偏移
        offset_scale = 0.06  #0.3 if Conv4
        # dy 用 l_norm，dx 用 w_norm
        dy = (l_norm * offset_scale).squeeze(1)   # [B,H,W]
        dx = (w_norm * offset_scale).squeeze(1)   # [B,H,W]


        offset_grid = torch.stack([dx, dy], dim=-1)  # [B,H,W,2]

        # 最终可导的 grid
        grid = base_grid + offset_grid
        grid = torch.clamp(grid, -1, 1)          # 防止跑出图外

        # 6. grid_sample 采样（现在 grid 对 l_conv / w_conv 是连着梯度的）
        sampled = F.grid_sample(x, grid, mode='bilinear', align_corners=False)

        # 7. 卷积分支：仍然只用一个 conv（由 N_X/N_Y 决定），保持原有高效思想不变
        idx = self.i_list.index(N_X * 10 + N_Y)
        x_offset = self.dropout2(sampled)
        x_offset = self.convs[idx](x_offset)     # [B,outc,H',W']

        # 保证尺寸对齐
        if x_offset.shape[-2:] != (Hm, Wm):
            x_offset = F.interpolate(
                x_offset,
                size=(Hm, Wm),
                mode='bilinear',
                align_corners=False
            )

        out = x_offset * m + bias
        out = self.post_attn(out)
        return out


class ECABlock(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)                  # [B,C,1,1]
        y = y.squeeze(-1).transpose(-1, -2)  # [B,1,C]
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.transpose(-1, -2).unsqueeze(-1)  # [B,C,1,1]
        return x * y





if __name__ == '__main__':
    x = torch.randn(4, 32, 24, 24)
    arconv = arARConv(32, 64)
    y = arconv(x, epoch=1, hw_range=[1, 9])
    print(y.shape)
