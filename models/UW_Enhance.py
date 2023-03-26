# from models.High_FFC import BasicBlock
from models.nafnet import NAFNet
from pytorch_wavelets import DWTForward, DWTInverse
from loss import Perceptual
import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.High_FFC import *
from models.unet import UNET
from models.FFCNet import FFCNet


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_features, in_features, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class Trans_low(nn.Module):
    def __init__(self, num_residual_blocks=5):
        super(Trans_low, self).__init__()

        model = [nn.Conv2d(9, 16, 3, padding=1),
                 nn.InstanceNorm2d(16),
                 nn.LeakyReLU(),
                 nn.Conv2d(16, 64, 3, padding=1),
                 nn.LeakyReLU()]

        for _ in range(num_residual_blocks):
            model += [ResidualBlock(64)]

        model += [nn.Conv2d(64, 16, 3, padding=1),
                  nn.LeakyReLU(),
                  nn.Conv2d(16, 9, 3, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = x + self.model(x)
        out = torch.tanh(out)
        return out


class UWEnhancer(nn.Module):

    def __init__(self):
        super(UWEnhancer, self).__init__()
        self.dwt = DWTForward(J=1, mode='zero', wave='haar')
        self.idwt = DWTInverse(mode='zero', wave='haar')
        self.ll_layer_module = FFCNet()
        # self.h_layer = FFCResNet(BasicBlock, [1, 1, 1, 1], out_h=64, out_w=64)
        # self.h_layer = UNET()
        # self.h_layer = FFCNet()
        self.h_layer = Trans_low()
        self.criterion_l1 = torch.nn.SmoothL1Loss()

    def forward(self, inp):
        inp_ll, inp_hf = self.dwt(inp)

        inp_hl = inp_hf[0][:, :, 0, :, :]
        inp_lh = inp_hf[0][:, :, 1, :, :]
        inp_hh = inp_hf[0][:, :, 2, :, :]

        inp_ll_hat = self.ll_layer_module(inp_ll)
        inp_H = torch.cat((inp_hl, inp_lh, inp_hh), dim=1)
        # print(inp_H.shape)

        out_H = self.h_layer(inp_H)
        out_hl = out_H[:, 0:3, :, :]
        out_lh = out_H[:, 3:6, :, :]
        out_hh = out_H[:, 6:9, :, :]

        recon_hl = out_hl.unsqueeze(2)
        recon_lh = out_lh.unsqueeze(2)
        recon_hh = out_hh.unsqueeze(2)

        recon_hf = [torch.cat((recon_hl, recon_lh, recon_hh), dim=2)]

        result = self.idwt((inp_ll_hat, recon_hf))

        return result


if __name__ == '__main__':
    tensor = torch.randn(1, 3, 128, 128).cuda()
    model = UWEnhancer().cuda()
    print('total parameters:', sum(param.numel() for param in model.parameters()))
    res = model(tensor)
    print(res.shape)
