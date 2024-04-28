# -*- coding:utf-8 -*-
# Author:Ding
import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder


# 在产生各个sub patches时进行波段采样
def band_sample(batch_size, band_size, ratio, device):
    a = torch.tensor([i for i in range(0, band_size, ratio)], device = device)
    a = a.repeat(batch_size, 1)
    b = torch.randint(0, ratio, a.shape, device = device)
    c = a + b
    c[c >= band_size] = band_size - 1

    return c


class FinalModel(nn.Module):
    def __init__(self, seq_len, band_size, patch_size, dim, depth, heads, mlp_dim, dim_head,
                 dropout = 0.1, emb_dropout = 0.1):
        super().__init__()
        self.encoder = Encoder(patch_size, band_size, dim)
        self.decoder1 = Decoder(seq_len, dim, depth, heads, mlp_dim, dim_head, dropout, emb_dropout)
        self.decoder2 = Decoder(seq_len, dim, depth, heads, mlp_dim, dim_head, dropout, emb_dropout)
        self.decoder3 = Decoder(seq_len, dim, depth, heads, mlp_dim, dim_head, dropout, emb_dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(3 * dim),
            nn.Linear(3 * dim, 2, bias = True)
        )
        self.mlp_head2 = nn.Sequential(
            nn.LayerNorm(3 * dim),
            nn.Linear(3 * dim, 2, bias = True)
        )
        self.mlp_head3 = nn.Sequential(
            nn.LayerNorm(3 * dim),
            nn.Linear(3 * dim, 2, bias = True)
        )

    @staticmethod
    def get_patch_set(x):
        batch_size, band_size, patch_size, _ = x.shape
        patch_set_len = int(patch_size / 2)
        patch_set = []
        for i in range(1, patch_set_len + 1):
            band_index = band_sample(batch_size, band_size, i, x.device)
            band_index = band_index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 2 * i + 1, 2 * i + 1)
            sub_patch = torch.gather(x, dim = 1, index = band_index)
            patch_set.append(sub_patch)

        return patch_set

    def forward(self, x):
        # x: [batch, 2 * channel, patch, patch]
        # TODO 两个时相随机选择的波段不一致
        x1, x2 = torch.split(x, int(x.shape[1] / 2), 1)
        x1 = self.get_patch_set(x1)
        x2 = self.get_patch_set(x2)
        feat_seq_11, feat_seq_12, feat_seq_13 = self.encoder(x1)
        feat_seq_21, feat_seq_22, feat_seq_23 = self.encoder(x2)

        # TODO: 拼接 or 相减
        # feat_seq_1 = torch.cat((feat_seq_11, feat_seq_21), dim = 1)
        # feat_seq_2 = torch.cat((feat_seq_12, feat_seq_22), dim = 1)
        # feat_seq_3 = torch.cat((feat_seq_13, feat_seq_23), dim = 1)
        #
        # out1 = self.decoder1(feat_seq_1)
        # out2 = self.decoder2(feat_seq_2)
        # out3 = self.decoder3(feat_seq_3)

        out1 = torch.cat((feat_seq_21 - feat_seq_11).unbind(dim = 1), 1)
        out2 = torch.cat((feat_seq_22 - feat_seq_12).unbind(dim = 1), 1)
        out3 = torch.cat((feat_seq_23 - feat_seq_13).unbind(dim = 1), 1)

        # out = torch.cat((out1, out2, out3), dim = -1)

        return self.mlp_head(out1) + self.mlp_head2(out2) + self.mlp_head3(out3)

# Example
# seq_len, band_size, patch_size, dim, depth, heads, mlp_dim, dim_head = 4, 154, 9, 64, 4, 2, 8, 16
# model = FinalModel(seq_len, band_size, patch_size, dim, depth, heads, mlp_dim, dim_head)
# x = torch.zeros([8, 308, 9, 9])
# # x = [torch.zeros([8, 154, 3, 3]), torch.zeros([8, 77, 5, 5]), torch.zeros([8, 52, 7, 7]), torch.zeros([8, 39, 9, 9])]
# out = model(x, 'test')
