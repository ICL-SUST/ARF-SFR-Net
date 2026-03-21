import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli

from models.ARF import arf
import torch_dct as dct

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class DropBlock(nn.Module):
    def __init__(self, block_size):
        super(DropBlock, self).__init__()

        self.block_size = block_size

    def forward(self, x, gamma):
        # shape: (bsize, channels, height, width)
        # 随机遮掉连续的block区域来正则化卷积神经网络
        if self.training:
            batch_size, channels, height, width = x.shape

            bernoulli = Bernoulli(gamma)  # 伯努利分布  gamma控制每个中心像素点被丢弃的概率
            mask = bernoulli.sample(
                (batch_size, channels, height - (self.block_size - 1), width - (self.block_size - 1))).cuda()
            block_mask = self._compute_block_mask(mask)
            countM = block_mask.size()[0] * block_mask.size()[1] * block_mask.size()[2] * block_mask.size()[3]
            count_ones = block_mask.sum()

            return block_mask * x * (countM / count_ones)
        else:
            return x

    def _compute_block_mask(self, mask):
        left_padding = int((self.block_size - 1) / 2)
        right_padding = int(self.block_size / 2)

        batch_size, channels, height, width = mask.shape
        non_zero_idxs = mask.nonzero()
        nr_blocks = non_zero_idxs.shape[0]

        offsets = torch.stack(
            [
                torch.arange(self.block_size).view(-1, 1).expand(self.block_size, self.block_size).reshape(-1),
                # - left_padding,
                torch.arange(self.block_size).repeat(self.block_size),  # - left_padding
            ]
        ).t().cuda()
        offsets = torch.cat((torch.zeros(self.block_size ** 2, 2).cuda().long(), offsets.long()), 1)

        if nr_blocks > 0:
            non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()

            block_idxs = non_zero_idxs + offsets
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            padded_mask[block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]] = 1.
        else:
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))

        block_mask = 1 - padded_mask  # [:height, :width]
        return block_mask


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False,
                 block_size=1, max_pool=True):
        super(BasicBlock, self).__init__()

        # self.conv1 = conv3x3(inplanes, planes)
        if planes == 4:
        # if planes in (64, 160):
            self.conv1 = ConvBlock1(inplanes, planes)
            self.use_arconv1 = True
        else:
            self.conv1 = conv3x3(inplanes, planes)
            self.use_arconv1 = False
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)

        # self.conv2 = ConvBlock1(planes, planes)
        # self.conv2 = conv3x3(planes, planes)
        if planes in (320, 640):
        # if planes == 640:
            self.conv2 = ConvBlock1(planes, planes)
            self.use_arconv2 = True
        else:
            self.conv2 = conv3x3(planes, planes)
            self.use_arconv2 = False
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        # if planes == 64:
        #     self.conv3 = ConvBlock1(planes, planes)
        #     self.use_arconv3 = True
        # else:
        #     self.conv3 = conv3x3(planes, planes)
        #     self.use_arconv3 = False
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)
        self.max_pool = max_pool

    def forward(self, x, epoch=None, hw_range=None):
        self.num_batches_tracked += 1

        residual = x

        # out = self.conv1(x)
        if self.use_arconv1:
            out = self.conv1(x, epoch, hw_range)
        else:
            out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.conv2(out, epoch, hw_range)
        # out = self.conv2(out)
        if self.use_arconv2:
            out = self.conv2(out, epoch, hw_range)
        else:
            out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        # if self.use_arconv3:
        #     out = self.conv2(out, epoch, hw_range)
        # else:
        #     out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        # print(f"block: {self.__class__.__name__}, out: {out.shape}, residual: {residual.shape}")

        out += residual
        out = self.relu(out)

        if self.max_pool:
            out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]  # 特征图空间分辨率
                keep_rate = max(1.0 - self.drop_rate / (20 * 2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size ** 2 * feat_size ** 2 / (feat_size - self.block_size + 1) ** 2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out

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
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(output_channel)

    # 前向传播
    def forward(self, inp, epoch, hw_range):
        inp = self.conv1(inp, epoch, hw_range)
        inp = self.bn(inp)
        return inp  # 将输入 inp 传递给 self.layers，返回卷积块的输出


class ResNet(nn.Module):

    def __init__(self, block, n_blocks, drop_rate=0.0, dropblock_size=5, max_pool=True, mask_size=16):
        super(ResNet, self).__init__()

        self.inplanes = 3
        self.layer1 = self._make_layer(block, n_blocks[0], 64,
                                       stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, n_blocks[1], 160,
                                       stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, n_blocks[2], 320,
                                       stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer4 = self._make_layer(block, n_blocks[3], 640,
                                       stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size,
                                       max_pool=max_pool)

        self.drop_rate = drop_rate
        self.spectral_mask = SpectralSoftMask(channels=3, mask_size=mask_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, n_block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1,
                    max_pool=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if n_block == 1:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size,
                          max_pool=max_pool)
        else:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate)
        layers.append(layer)
        self.inplanes = planes * block.expansion

        for i in range(1, n_block):
            if i == n_block - 1:
                layer = block(self.inplanes, planes, drop_rate=drop_rate, drop_block=drop_block,
                              block_size=block_size)
            else:
                layer = block(self.inplanes, planes, drop_rate=drop_rate)
            layers.append(layer)

        # return nn.ModuleList(layers)
        return nn.Sequential(*layers)

    def forward_layer(self, layer, x, epoch, hw_range):
        # layer: ModuleList
        for block in layer:
            x = block(x, epoch, hw_range)
        return x

    def forward(self, x, epoch=None, hw_range=None):
        # x =dct.dct_2d(x)
        # x = self.spectral_mask(x)
        # x = dct.idct_2d(x)
        x = self.forward_layer(self.layer1, x, epoch, hw_range)
        x = self.forward_layer(self.layer2, x, epoch, hw_range)
        x = self.forward_layer(self.layer3, x, epoch, hw_range)
        x = self.forward_layer(self.layer4, x, epoch, hw_range)
        return x

class SpectralSoftMask(nn.Module):
    def __init__(self, channels=3, mask_size=16):
        super().__init__()
        self.mask = nn.Parameter(torch.ones(1, channels, mask_size, mask_size))
    def forward(self, freq_feat):
        B, C, H, W = freq_feat.shape
        mask = F.interpolate(self.mask, size=(H, W), mode='bilinear', align_corners=False)
        return freq_feat * mask

# class SpectralSoftMask(nn.Module):
#     def __init__(self, channels=3, mask_size=16, m_min=0.5, m_max=1.5):
#         super().__init__()
#         self.m_min, self.m_max = m_min, m_max
#         self.mask_logit = nn.Parameter(torch.zeros(1, channels, mask_size, mask_size))
#         # 可选：初始化成“高左上、低右下”的低通先验
#         with torch.no_grad():
#             H = W = mask_size
#             yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
#             r = (yy + xx).float() / (H + W)
#             prior = 1.0 - r  # 低频大
#             self.mask_logit.copy_(torch.logit(torch.clamp((prior - m_min)/(m_max - m_min), 1e-4, 1-1e-4))[None, None])
#     def forward(self, freq_feat):
#         B, C, H, W = freq_feat.shape
#         logit = F.interpolate(self.mask_logit, size=(H, W), mode='bilinear', align_corners=False)
#         mask = torch.sigmoid(logit) * (self.m_max - self.m_min) + self.m_min
#         return freq_feat * mask


def resnet12(drop_rate=0.0, max_pool=True, **kwargs):
    """Constructs a ResNet-12 model.
    """

    model = ResNet(BasicBlock, [1, 1, 1, 1], drop_rate=drop_rate, max_pool=max_pool, **kwargs)

    return model



if __name__ == '__main__':
    model = resnet12()
    data = torch.randn(2, 3, 84, 84)
    H, W = data.shape[-2:]          # (84, 84)
    x = model(data, epoch=0, hw_range=(H, W))
    print(x.shape)
