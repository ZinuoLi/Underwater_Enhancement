import torch
import torch.nn as nn
from models.unet import UNET
from models.transformer import Restormer
from models.nafnet import NAFNet
from pytorch_wavelets import DWTForward, DWTInverse
from loss import Perceptual


class Enhancer(nn.Module):

    def __init__(self, params, device, ll_layer='Transformer', enhance='CSR'):
        super(Enhancer, self).__init__()
        self.dwt = DWTForward(J=1, mode='zero', wave='haar')
        self.idwt = DWTInverse(mode='zero', wave='haar')
        self.device = device
        self.ll_layer = ll_layer
        self.enhance = enhance
        if ll_layer == 'Transformer':
            self.ll_layer_module = Restormer()
        elif ll_layer == 'NAF':
            self.ll_layer_module = NAFNet()
        else:
            self.ll_layer_module = UNET()
        self.h_layer = UNET()

        self.enhance_module = UNET()

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
        if self.ll_layer == 'Refine':
            return self.ll_layer_module(self.backbone(inp_ll))
        return self.ll_layer_module(inp_ll)

    def h_forward(self, hl, lh, hh):
        return self.h_layer(hl), self.h_layer(lh), self.h_layer(hh)

    # def enhance_forward(self, ll, inp):
    #     if self.enhance == 'HDR':
    #         return self.enhance_module(ll, inp)
    #     return self.enhance_module(inp)

    def forward(self, inp, tar):
        inp_ll, inp_hf = self.dwt(inp)

        inp_hl = inp_hf[0][:, :, 0, :, :]
        inp_lh = inp_hf[0][:, :, 1, :, :]
        inp_hh = inp_hf[0][:, :, 2, :, :]

        tar_ll, tar_hf = self.dwt(tar)

        tar_hl = tar_hf[0][:, :, 0, :, :]
        tar_lh = tar_hf[0][:, :, 1, :, :]
        tar_hh = tar_hf[0][:, :, 2, :, :]

        inp_ll_hat = self.ll_forward(inp_ll)
        inp_hl_hat, inp_lh_hat, inp_hh_hat = self.h_forward(inp_hl, inp_lh, inp_hh)

        loss_ll = self.hdr_loss(tar_ll, inp_ll_hat)
        loss_hf = self.criterion_l1(inp_hl_hat, tar_hl) + self.criterion_l1(inp_lh_hat, tar_lh) + self.criterion_l1(
            inp_hh_hat, tar_hh)

        recon_hl = inp_hl_hat.unsqueeze(2)
        recon_lh = inp_lh_hat.unsqueeze(2)
        recon_hh = inp_hh_hat.unsqueeze(2)

        recon_hf = [torch.cat((recon_hl, recon_lh, recon_hh), dim=2)]
        # out = self.idwt((inp_ll_hat, recon_hf))

        out_enhanced = self.idwt((inp_ll_hat, recon_hf))

        criterion_prec = Perceptual()

        # loss_recon = self.criterion_l1(out, tar) + criterion_prec(out, tar)

        loss_recon = self.criterion_l1(out_enhanced, tar) + criterion_prec(out_enhanced, tar)
        # out_enhanced = self.enhance_forward(inp_ll_hat, out)

        # loss_csr = self.hdr_loss(out_enhanced, tar)

        # total_loss = loss_ll + loss_hf + loss_recon + loss_csr

        total_loss = loss_ll + loss_hf + loss_recon

        if self.training:
            return out_enhanced, total_loss
        else:
            return out_enhanced


if __name__ == '__main__':
    tensor = torch.randn(1, 3, 512, 512).cuda()
    params = lambda: None
    params.luma_bins = 8
    params.channel_multiplier = 1
    params.spatial_bin = 8
    params.batch_norm = False
    params.low_size = 256
    params.full_size = 512
    params.eps_value = 1e-4
    model = Enhancer(params, 'cuda').cuda()
    # 57200896
    print('total parameters:', sum(param.numel() for param in model.parameters()))
    res, loss = model(tensor, tensor)
    print(res.shape)
    print(loss.item())
    # res = model(tensor, None, training=False)
    # print(res.shape)