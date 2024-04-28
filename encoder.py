# -*- coding:utf-8 -*-
# Author:Ding
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias = False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias = False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class ChannelAttention_1(nn.Module):
    def __init__(self, in_planes, ratio = 16):
        super(ChannelAttention_1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes ** 2, in_planes ** 2 // ratio, 1, bias = False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes ** 2 // ratio, in_planes, 1, bias = False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        bi_out = torch.bmm(self.avg_pool(x).squeeze(-1), torch.transpose(self.max_pool(x).squeeze(-1), 1, 2)) / x.shape[
            1] ** 2
        bi_out = bi_out.view(-1, x.shape[1] ** 2).unsqueeze(-1).unsqueeze(-1)
        out = self.fc(bi_out)
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size = 7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding = kernel_size // 2, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim = 1, keepdim = True)
        max_out, _ = torch.max(x, dim = 1, keepdim = True)
        x = torch.cat([avg_out, max_out], dim = 1)
        x = self.conv1(x)
        return self.sigmoid(x)


# 定义一个基本卷积块，包括两个3x3的卷积层和一个BatchNorm层
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channels),
            )

        self.ca = ChannelAttention_1(out_channels, ratio = 16)
        self.sa = SpatialAttention()

    def forward(self, x):
        identity = x
        # 第一层卷积和BN
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # 第二层卷积和BN
        out = self.conv2(out)
        out = self.bn2(out)
        # CBAM
        out = self.ca(out) * out
        out = self.sa(out) * out
        # 如果输入的x与输出的out的尺寸不同，需要进行下采样
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity  # 残差连接
        out = self.relu(out)
        return out


# 定义Encoder
class Encoder(nn.Module):
    def __init__(self, patch_size, band_size, dim):
        super(Encoder, self).__init__()
        self.in_channels = 64
        self.patch_sizes = [2 * i + 1 for i in range(1, int(patch_size / 2 + 1))]
        in_bands = [len([i for i in range(0, band_size, j + 1)]) for j in range(len(self.patch_sizes))]
        self.initial_layers = nn.ModuleList([])

        self.layer1 = nn.ModuleList([])
        self.layer2 = nn.ModuleList([])
        self.layer3 = nn.ModuleList([])
        self.feat_embed = nn.ModuleList([])

        for i in range(len(self.patch_sizes)):
            size = self.patch_sizes[i]
            in_band = in_bands[i]
            self.initial_layers.append(nn.Sequential(
                nn.Conv2d(in_band, 64, kernel_size = size, stride = 1, padding = int(size / 2), bias = False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace = True),
            ))
            self.layer1.append(self._make_layer(BasicBlock, 64, 1, stride = 1))
            self.layer2.append(self._make_layer(BasicBlock, 64, 1, stride = 1))
            self.layer3.append(self._make_layer(BasicBlock, 64, 1, stride = 1))
            self.feat_embed.append(nn.Conv2d(64, dim, kernel_size = size, groups = min(64, dim)))

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # 第一个块的步幅为stride，后面的块步幅为1
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        feat_seq_scale1 = []
        feat_seq_scale2 = []
        feat_seq_scale3 = []
        for i in range(len(x)):
            out = self.initial_layers[i](x[i])
            out = self.layer1[i](out)
            feat_seq_scale1.append(self.feat_embed[i](out).squeeze())
            out = self.layer2[i](out)
            feat_seq_scale2.append(self.feat_embed[i](out).squeeze())
            out = self.layer3[i](out)
            feat_seq_scale3.append(self.feat_embed[i](out).squeeze())

        feat_seq_1 = torch.stack(feat_seq_scale1, dim = 1)
        feat_seq_2 = torch.stack(feat_seq_scale2, dim = 1)
        feat_seq_3 = torch.stack(feat_seq_scale3, dim = 1)

        return feat_seq_1, feat_seq_2, feat_seq_3

# Example
# net = Encoder(9, 154, 64)
# x = [torch.zeros([8, 154, 3, 3]), torch.zeros([8, 77, 5, 5]), torch.zeros([8, 52, 7, 7]), torch.zeros([8, 39, 9, 9])]
# out = net(x)
