import math
import numbers
import os

import torch
from torch import nn
from timm.models.layers import trunc_normal_, DropPath
from pytorch_wavelets import DWTForward, DWTInverse
import torch.nn.functional as F
from torchvision.utils import save_image
from einops import rearrange, einops
from models.nafnet import NAFNet


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


# Axis-based Multi-head Self-Attention

class NextAttentionImplZ(nn.Module):
    def __init__(self, num_dims, num_heads, bias) -> None:
        super().__init__()
        self.num_dims = num_dims
        self.num_heads = num_heads
        self.q1 = nn.Conv2d(num_dims, num_dims * 3, kernel_size=1, bias=bias)
        self.q2 = nn.Conv2d(num_dims * 3, num_dims * 3, kernel_size=3, padding=1, groups=num_dims * 3, bias=bias)
        self.q3 = nn.Conv2d(num_dims * 3, num_dims * 3, kernel_size=3, padding=1, groups=num_dims * 3, bias=bias)

        self.fac = nn.Parameter(torch.ones(1))
        self.fin = nn.Conv2d(num_dims, num_dims, kernel_size=1, bias=bias)
        return

    def forward(self, x):
        # x: [n, c, h, w]
        n, c, h, w = x.size()
        n_heads, dim_head = self.num_heads, c // self.num_heads
        reshape = lambda x: einops.rearrange(x, "n (nh dh) h w -> (n nh h) w dh", nh=n_heads, dh=dim_head)

        qkv = self.q3(self.q2(self.q1(x)))
        q, k, v = map(reshape, qkv.chunk(3, dim=1))
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # fac = dim_head ** -0.5
        res = k.transpose(-2, -1)
        res = torch.matmul(q, res) * self.fac
        res = torch.softmax(res, dim=-1)

        res = torch.matmul(res, v)
        res = einops.rearrange(res, "(n nh h) w dh -> n (nh dh) h w", nh=n_heads, dh=dim_head, n=n, h=h)
        res = self.fin(res)

        return res


# Axis-based Multi-head Self-Attention (row and col attention)
class NextAttentionZ(nn.Module):
    def __init__(self, num_dims, num_heads=1, bias=True) -> None:
        super().__init__()
        assert num_dims % num_heads == 0
        self.num_dims = num_dims
        self.num_heads = num_heads
        self.row_att = NextAttentionImplZ(num_dims, num_heads, bias)
        self.col_att = NextAttentionImplZ(num_dims, num_heads, bias)
        return

    def forward(self, x: torch.Tensor):
        assert len(x.size()) == 4

        x = self.row_att(x)
        x = x.transpose(-2, -1)
        x = self.col_att(x)
        x = x.transpose(-2, -1)
        # print(x.shape)

        return x


class LPAttention(nn.Module):
    def __init__(self, depth=2, num_dims=3, bias=True):
        super().__init__()
        self.lap_pyramid = LapPyramidConv(depth)
        # k conv
        self.conv1 = nn.Conv2d(num_dims, num_dims * 3, kernel_size=1, bias=bias)
        self.conv2 = nn.Conv2d(num_dims * 3, num_dims * 3, kernel_size=3, padding=1, groups=num_dims * 3, bias=bias)
        self.conv3 = nn.Conv2d(num_dims * 6, num_dims * 3, kernel_size=3, padding=1, bias=bias)
        self.conv4 = nn.Conv2d(num_dims * 6, num_dims * 3, kernel_size=3, padding=1, bias=bias)
        # v conv
        self.conv1_ = nn.Conv2d(num_dims, num_dims * 3, kernel_size=1, bias=bias)
        self.conv2_ = nn.Conv2d(num_dims * 3, num_dims * 3, kernel_size=3, padding=1, groups=num_dims * 3, bias=bias)
        self.conv3_ = nn.Conv2d(num_dims * 6, num_dims * 3, kernel_size=3, padding=1, bias=bias)
        self.conv4_ = nn.Conv2d(num_dims * 6, num_dims * 3, kernel_size=3, padding=1, bias=bias)

        self.conv5 = nn.Conv2d(num_dims * 3, num_dims, kernel_size=3, padding=1, bias=bias)

    def forward(self, x):
        q = self.conv2(self.conv1(x))
        pyr_inp = self.lap_pyramid.pyramid_decom(img=x)

        k1 = self.conv2(self.conv1(pyr_inp[-1]))
        k2 = self.conv2(self.conv1(pyr_inp[-2]))
        k3 = self.conv2(self.conv1(pyr_inp[-3]))
        # k1->k2
        k1 = nn.functional.interpolate(k1, size=(pyr_inp[-2].shape[2], pyr_inp[-2].shape[3]))
        k1k2 = torch.cat([k1, k2], dim=1)
        k1k2 = self.conv3(k1k2)
        # k1k2(k2)->k3
        k1k2 = nn.functional.interpolate(k1k2, size=(pyr_inp[-3].shape[2], pyr_inp[-3].shape[3]))
        k = torch.cat([k1k2, k3], dim=1)
        k = self.conv4(k)

        v1 = self.conv2_(self.conv1_(pyr_inp[-1]))
        v2 = self.conv2_(self.conv1_(pyr_inp[-2]))
        v3 = self.conv2_(self.conv1_(pyr_inp[-3]))
        v1 = nn.functional.interpolate(v1, size=(pyr_inp[-2].shape[2], pyr_inp[-2].shape[3]))
        v1v2 = torch.cat([v1, v2], dim=1)
        v1v2 = self.conv3_(v1v2)
        v1v2 = nn.functional.interpolate(v1v2, size=(pyr_inp[-3].shape[2], pyr_inp[-3].shape[3]))
        v = torch.cat([v1v2, v3], dim=1)
        v = self.conv4_(v)

        qk = q @ k.transpose(2, 3)
        qkv = qk @ v
        qkv = self.conv5(qkv)
        return qkv


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


# Dual Gated Feed-Forward Networ
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x2) * x1 + F.gelu(x1) * x2
        x = self.project_out(x)
        return x


# in_channels=c, LayerNorm2d(c)
class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=1, ffn_expansion_factor=2.66, bias=True):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm2d(dim)
        self.lpattn = LPAttention()
        self.norm2 = LayerNorm2d(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.lpattn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


def gauss_kernel(channels=3):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    return kernel


class LapPyramidConv(nn.Module):
    def __init__(self, num_high=4):
        super(LapPyramidConv, self).__init__()

        self.num_high = num_high
        self.kernel = gauss_kernel()

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel.to(img.device), groups=img.shape[1])
        return out

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            up = self.upsample(down)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[-1]
        for level in reversed(pyr[:-1]):
            up = self.upsample(image)
            if up.shape[2] != level.shape[2] or up.shape[3] != level.shape[3]:
                up = nn.functional.interpolate(up, size=(level.shape[2], level.shape[3]))
            image = up + level
        return image


# Downsample Block
class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.conv(x).contiguous()
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H / 2 * W / 2 * self.in_channel * self.out_channel * 4 * 4
        print("Downsample:{%.2f}" % (flops / 1e9))
        return flops


# Upsample Block
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).contiguous()
        return out

    def flops(self, H, W):
        flops = 0
        flops += H * 2 * W * 2 * self.in_channel * self.out_channel * 2 * 2
        print("Upsample:{%.2f}" % (flops / 1e9))
        return flops


class U_LPViT(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=3,
                 num_blocks=[4, 8, 12, 20],
                 num_refinement_blocks=2,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 attention=True,
                 dowsample=Downsample,
                 upsample=Upsample
                 ):
        super().__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.conv1 = nn.Conv2d(6, 3, kernel_size=1, bias=bias)
        self.conv2 = nn.Conv2d(6, 3, kernel_size=1, bias=bias)
        self.conv3 = nn.Conv2d(6, 3, kernel_size=1, bias=bias)
        self.conv4 = nn.Conv2d(6, 3, kernel_size=1, bias=bias)

        self.encoder_1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for _ in
            range(num_blocks[0])])

        self.dowsample_1 = dowsample(dim, dim)

        self.encoder_2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for
            _ in range(num_blocks[0])])

        self.dowsample_2 = dowsample(dim, dim)

        self.encoder_3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for
            _ in range(num_blocks[0])])

        self.dowsample_3 = dowsample(dim, dim)

        self.encoder_4 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for
            _ in range(num_blocks[0])])

        self.dowsample_4 = dowsample(dim, dim)

        # BottleNeck
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for
            _ in range(num_blocks[0])])

        self.decoder_1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for
            _ in range(num_refinement_blocks)])

        self.upsample_1 = upsample(dim, dim)

        self.decoder_2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for
            _ in range(num_refinement_blocks)])

        self.upsample_2 = upsample(dim, dim)

        self.decoder_3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for
            _ in range(num_refinement_blocks)])

        self.upsample_3 = upsample(dim, dim)

        self.decoder_4 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for
            _ in range(num_refinement_blocks)])

        self.upsample_4 = upsample(dim, dim)

    def forward(self, x):
        inp_enc_encoder1 = self.patch_embed(x)

        # Encoder
        out_enc_encoder1 = self.encoder_1(inp_enc_encoder1)
        down1 = self.dowsample_1(out_enc_encoder1)

        out_enc_encoder2 = self.encoder_2(down1)
        down2 = self.dowsample_2(out_enc_encoder2)

        out_enc_encoder3 = self.encoder_3(down2)
        down3 = self.dowsample_3(out_enc_encoder3)

        out_enc_encoder4 = self.encoder_4(down3)
        down4 = self.dowsample_3(out_enc_encoder4)
        # print(down4.shape)

        # Bottleneck
        bottleneck = self.latent(down4)

        # Decoder
        up1 = self.upsample_1(bottleneck)
        if up1.shape != out_enc_encoder4:
            up1 = nn.functional.interpolate(up1, size=(out_enc_encoder4.shape[2], out_enc_encoder4.shape[3]))
        sc1 = torch.cat([up1, out_enc_encoder4], 1)
        sc1 = self.conv1(sc1)
        out_dec_encoder1 = self.decoder_1(sc1)

        up2 = self.upsample_2(out_dec_encoder1)
        if up1.shape != out_enc_encoder3:
            up2 = nn.functional.interpolate(up2, size=(out_enc_encoder3.shape[2], out_enc_encoder3.shape[3]))
        sc2 = torch.cat([up2, out_enc_encoder3], 1)
        sc2 = self.conv2(sc2)
        out_dec_encoder2 = self.decoder_2(sc2)

        up3 = self.upsample_3(out_dec_encoder2)
        if up3.shape != out_enc_encoder2:
            up3 = nn.functional.interpolate(up3, size=(out_enc_encoder2.shape[2], out_enc_encoder2.shape[3]))
        sc3 = torch.cat([up3, out_enc_encoder2], 1)
        sc3 = self.conv3(sc3)
        out_dec_encoder3 = self.decoder_3(sc3)

        up4 = self.upsample_4(out_dec_encoder3)
        if up4.shape != out_enc_encoder2:
            up4 = nn.functional.interpolate(up4, size=(out_enc_encoder1.shape[2], out_enc_encoder1.shape[3]))
        sc4 = torch.cat([up4, out_enc_encoder1], 1)
        sc4 = self.conv4(sc4)
        out_dec_encoder4 = self.decoder_4(sc4)

        return out_dec_encoder4


# Cross-layer Attention Fusion Block
class LAM_Module_v2(nn.Module):
    """ Layer attention module"""

    def __init__(self, in_dim, bias=True):
        super(LAM_Module_v2, self).__init__()
        self.chanel_in = in_dim

        self.temperature = nn.Parameter(torch.ones(1))

        self.qkv = nn.Conv2d(self.chanel_in, self.chanel_in * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(self.chanel_in * 3, self.chanel_in * 3, kernel_size=3, stride=1, padding=1,
                                    groups=self.chanel_in * 3, bias=bias)
        self.project_out = nn.Conv2d(self.chanel_in, self.chanel_in, kernel_size=1, bias=bias)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, N, C, height, width = x.size()

        x_input = x.view(m_batchsize, N * C, height, width)
        qkv = self.qkv_dwconv(self.qkv(x_input))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(m_batchsize, N, -1)
        k = k.view(m_batchsize, N, -1)
        v = v.view(m_batchsize, N, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out_1 = (attn @ v)
        out_1 = out_1.view(m_batchsize, -1, height, width)

        out_1 = self.project_out(out_1)
        out_1 = out_1.view(m_batchsize, N, C, height, width)

        out = out_1 + x
        out = out.view(m_batchsize, -1, height, width)
        return out


class LPViT(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=3,
                 num_blocks=[4, 8, 12, 20],
                 num_refinement_blocks=2,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 attention=True
                 ):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for _ in
            range(num_blocks[0])])

        self.encoder_2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for
            _ in range(num_blocks[0])])

        self.encoder_3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for
            _ in range(num_blocks[0])])

        self.layer_fussion = LAM_Module_v2(in_dim=int(dim * 3))
        self.conv_fuss = nn.Conv2d(int(dim * 3), int(dim), kernel_size=1, bias=bias)

        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for
            _ in range(num_blocks[0])])

        self.trans_low = NAFNet()

        self.coefficient_1_0 = nn.Parameter(torch.ones((2, int(int(dim)))), requires_grad=attention)

        self.refinement_1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for
            _ in range(num_refinement_blocks)])
        self.refinement_2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for
            _ in range(num_refinement_blocks)])
        self.refinement_3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for
            _ in range(num_refinement_blocks)])

        self.layer_fussion_2 = LAM_Module_v2(in_dim=int(dim * 3))
        self.conv_fuss_2 = nn.Conv2d(int(dim * 3), int(dim), kernel_size=1, bias=bias)

        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp):
        inp_enc_encoder1 = self.patch_embed(inp)
        # print(inp_enc_encoder1.shape)
        out_enc_encoder1 = self.encoder_1(inp_enc_encoder1)
        # print(out_enc_encoder1.shape)
        out_enc_encoder2 = self.encoder_2(out_enc_encoder1)
        # print(out_enc_encoder2.shape)
        out_enc_encoder3 = self.encoder_3(out_enc_encoder2)
        # print(out_enc_encoder3.shape)

        inp_fusion_123 = torch.cat(
            [out_enc_encoder1.unsqueeze(1), out_enc_encoder2.unsqueeze(1), out_enc_encoder3.unsqueeze(1)], dim=1)

        out_fusion_123 = self.layer_fussion(inp_fusion_123)
        out_fusion_123 = self.conv_fuss(out_fusion_123)

        out_enc = self.trans_low(out_fusion_123)

        out_fusion_123 = self.latent(out_fusion_123)

        out = self.coefficient_1_0[0, :][None, :, None, None] * out_fusion_123 + self.coefficient_1_0[1, :][None, :,
                                                                                 None, None] * out_enc

        out_1 = self.refinement_1(out)
        out_2 = self.refinement_2(out_1)
        out_3 = self.refinement_3(out_2)

        inp_fusion = torch.cat([out_1.unsqueeze(1), out_2.unsqueeze(1), out_3.unsqueeze(1)], dim=1)
        out_fusion_123 = self.layer_fussion_2(inp_fusion)
        out = self.conv_fuss_2(out_fusion_123)
        result = self.output(out)

        return result


if __name__ == '__main__':
    model = LPViT().cuda()
    tensor = torch.randn(4, 3, 230, 320).cuda()
    res = model(tensor)
    print(res.shape)
    from thop import profile, clever_format

    macs, params = profile(model, inputs=(tensor,))
    macs, params = clever_format([macs, params], "%.3f")
    print("params :", params)
