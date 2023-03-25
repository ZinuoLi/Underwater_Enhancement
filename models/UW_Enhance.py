from models.High_FFC import BasicBlock
from models.nafnet import NAFNet
from pytorch_wavelets import DWTForward, DWTInverse
from loss import Perceptual
import torch
import torch.nn as nn
from models.IAT_ import IAT
import torch.nn.functional as F
from models.High_FFC import *
from models.unet import UNET


class UWEnhancer(nn.Module):

    def __init__(self):
        super(UWEnhancer, self).__init__()
        self.dwt = DWTForward(J=1, mode='zero', wave='haar')
        self.idwt = DWTInverse(mode='zero', wave='haar')
        self.ll_layer_module = IAT()
        self.h_layer = FFCResNet(BasicBlock, [1, 1, 1, 1], out_h=64, out_w=64)
        # self.h_layer = UNET()
        self.criterion_l1 = torch.nn.SmoothL1Loss()

    def hdr_loss(self, gt, pred):
        b, c, h, w = gt.shape
        gt_reflect_view = gt.view(b, c, h * w).permute(0, 2, 1)
        pred_reflect_view = pred.view(b, c, h * w).permute(0, 2, 1)
        gt_reflect_norm = torch.nn.functional.normalize(gt_reflect_view, dim=-1)
        pred_reflect_norm = torch.nn.functional.normalize(pred_reflect_view, dim=-1)
        cose_value = gt_reflect_norm * pred_reflect_norm
        cose_value = torch.sum(cose_value, dim=-1)
        color_loss = torch.mean(1 - cose_value)

        total_loss_value = 0.2 * color_loss

        return total_loss_value

    def ll_forward(self, inp_ll):
        return self.ll_layer_module(inp_ll)

    def forward(self, inp):
        inp_ll, inp_hf = self.dwt(inp)

        inp_hl = inp_hf[0][:, :, 0, :, :]
        inp_lh = inp_hf[0][:, :, 1, :, :]
        inp_hh = inp_hf[0][:, :, 2, :, :]

        inp_ll_hat = self.ll_forward(inp_ll)
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

        result = self.idwt((inp_ll_hat[2], recon_hf))

        return result


if __name__ == '__main__':
    tensor = torch.randn(1, 3, 360, 640).cuda()
    model = UWEnhancer().cuda()
    print('total parameters:', sum(param.numel() for param in model.parameters()))
    res, loss = model(tensor, tensor)
    print(res.shape)
    print(loss.item())
