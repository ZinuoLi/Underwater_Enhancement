import torch
import numpy as np
from torch import nn, matmul, dot
import torch.nn.functional as F
import os
import math

from timm.models.layers import trunc_normal_

from models.FFCNet_3C import FFCNet3C
from models.FFCNet import FFCNet
from models.blocks import CBlock_ln, SwinTransformerBlock
from models.LPFormer import LPViT
from models.LPFormer import U_LPViT
from models.LPnafnet import LPNAFNet


class Local_pred(nn.Module):
    def __init__(self, dim=16, number=4, type='ccc'):
        super(Local_pred, self).__init__()
        # initial convolution
        self.conv1 = nn.Conv2d(3, dim, 3, padding=1, groups=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # main blocks
        block = CBlock_ln(dim)
        block_t = SwinTransformerBlock(dim)  # head number
        if type == 'ccc':
            blocks1 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
            blocks2 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
        elif type == 'ttt':
            blocks1, blocks2 = [block_t for _ in range(number)], [block_t for _ in range(number)]
        elif type == 'cct':
            blocks1, blocks2 = [block, block, block_t], [block, block, block_t]
        self.mul_blocks = nn.Sequential(*blocks1, nn.Conv2d(dim, 3, 3, 1, 1), nn.ReLU())
        self.add_blocks = nn.Sequential(*blocks2, nn.Conv2d(dim, 3, 3, 1, 1), nn.Tanh())

    def forward(self, img):
        img1 = self.relu(self.conv1(img))
        mul = self.mul_blocks(img1)
        add = self.add_blocks(img1)

        return mul, add


class IAT(nn.Module):
    def __init__(self, in_dim=3):
        super(IAT, self).__init__()
        self.local_net = FFCNet3C()
        self.global_net = LPNAFNet()

    def forward(self, img_low):
        local_res = self.local_net(img_low)
        global_res = self.global_net(img_low)

        out = local_res * global_res

        return out


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    img = torch.Tensor(1, 3, 460, 640)
    net = IAT()
    print('total parameters:', sum(param.numel() for param in net.parameters()))
    res = net(img)
    print(res.shape)
