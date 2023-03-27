import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

from models.nn_utils import get_padding


class FourierUnit(torch.nn.Module):
    """Implements Fourier Unit block.

    Applies FFT to tensor and performs convolution in spectral domain.
    After that return to time domain with Inverse FFT.

    Attributes:
        inter_conv: conv-bn-relu block that performs conv in spectral domain

    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            fu_kernel: int = 1,
            padding_type: str = "reflect",
            fft_norm: str = "ortho",
            use_only_freq: bool = True,
            norm_layer=nn.BatchNorm2d,
            bias: bool = True,
    ):
        super().__init__()
        self.fft_norm = fft_norm
        self.use_only_freq = use_only_freq

        self.inter_conv = nn.Sequential(
            nn.Conv2d(
                in_channels * 2,
                out_channels * 2,
                kernel_size=fu_kernel,
                stride=1,
                padding=get_padding(fu_kernel),
                padding_mode=padding_type,
                bias=bias,
            ),
            norm_layer(out_channels * 2),
            nn.ReLU(True),
        )

    def forward(self, x):
        batch_size, ch, freq_dim, embed_dim = x.size()

        dims_to_fft = (-2,) if self.use_only_freq else (-2, -1)
        recover_length = (freq_dim,) if self.use_only_freq else (freq_dim, embed_dim)

        fft_representation = torch.fft.rfftn(x, dim=dims_to_fft, norm=self.fft_norm)

        # (B, Ch, 2, FFT_freq, FFT_embed)
        fft_representation = torch.stack(
            (fft_representation.real, fft_representation.imag), dim=2
        )  # .view(batch_size, ch * 2, -1, embed_dim)

        ffted_dims = fft_representation.size()[-2:]
        fft_representation = fft_representation.view(
            (
                batch_size,
                ch * 2,
            )
            + ffted_dims
        )

        fft_representation = (
            self.inter_conv(fft_representation)
            .view(
                (
                    batch_size,
                    ch,
                    2,
                )
                + ffted_dims
            )
            .permute(0, 1, 3, 4, 2)
        )

        fft_representation = torch.complex(
            fft_representation[..., 0], fft_representation[..., 1]
        )

        reconstructed_x = torch.fft.irfftn(
            fft_representation, dim=dims_to_fft, s=recover_length, norm=self.fft_norm
        )

        assert reconstructed_x.size() == x.size()

        return reconstructed_x


class SpectralTransform(torch.nn.Module):
    """Implements Spectrals Transform block.

    Residual Block containing Fourier Unit with convolutions before and after.

    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            fu_kernel: int = 1,
            padding_type: str = "reflect",
            fft_norm: str = "ortho",
            use_only_freq: bool = True,
            norm_layer=nn.BatchNorm2d,
            bias: bool = False,
    ):
        super().__init__()
        halved_out_ch = out_channels // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, halved_out_ch, kernel_size=1, stride=1, bias=bias),
            norm_layer(halved_out_ch),
            nn.ReLU(True),
        )

        self.fu = FourierUnit(
            halved_out_ch,
            halved_out_ch,
            fu_kernel=fu_kernel,
            use_only_freq=use_only_freq,
            fft_norm=fft_norm,
            padding_type=padding_type,
            norm_layer=norm_layer,
        )

        self.conv2 = nn.Conv2d(
            halved_out_ch, out_channels, kernel_size=1, stride=1, bias=bias
        )

    def forward(self, x):
        residual = self.conv1(x)
        x = self.fu(residual)
        x += residual
        x = self.conv2(x)

        return x


class FastFourierConvolution(torch.nn.Module):
    """Implements FFC block.

    Divides Tensor in two branches: local and global. Local branch performs
    convolutions and global branch applies Spectral Transform layer.
    After performing transforms in local and global branches outputs are passed through BatchNorm + ReLU
    and eventually concatenated. Based on proportion of input and output global channels if the number is equal
    to zero respective blocks are replaced by Identity Transform.
    For clarity refer to original paper.

    Attributes:
        local_in_channels: # input channels for l2l and l2g convs
        local_out_channels: # output channels for l2l and g2l convs
        global_in_channels: # input channels for g2l and g2g convs
        global_out_channels: # output_channels for l2g and g2g convs
        l2l_layer: local to local Convolution
        l2g_layer: local to global Convolution
        g2l_layer: global to local Convolution
        g2g_layer: global to global Spectral Transform

    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            alpha_in: float = 0.5,
            alpha_out: float = 0.5,
            kernel_size: int = 3,
            padding_type: str = "reflect",
            fu_kernel: int = 1,
            fft_norm: str = "ortho",
            bias: bool = True,
            norm_layer=nn.BatchNorm2d,
            activation=nn.ReLU(True),
            use_only_freq: bool = True,
    ):
        """Inits FFC module.

        Args:
            in_channels: total channels of tensor before dividing into local and global
            alpha_in:
                proportion of global channels as input
            alpha_out:
                proportion of global channels as output
            use_only_freq:
                controls dimensionality of fft in Fourier Unit. If false uses 2D fft in Fourier Unit affecting both
                frequency and time dimensions, otherwise applies 1D FFT only to frequency dimension

        """
        super().__init__()
        self.global_in_channels = int(in_channels * alpha_in)
        self.local_in_channels = in_channels - self.global_in_channels
        self.global_out_channels = int(out_channels * alpha_out)
        self.local_out_channels = out_channels - self.global_out_channels

        padding = get_padding(kernel_size)

        tmp_module = self._get_module_on_true_predicate(
            self.local_in_channels > 0 and self.local_out_channels > 0,
            nn.Conv2d,
            nn.Identity,
        )
        self.l2l_layer = tmp_module(
            self.local_in_channels,
            self.local_out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            padding_mode=padding_type,
            bias=bias,
        )

        tmp_module = self._get_module_on_true_predicate(
            self.local_in_channels > 0 and self.global_out_channels > 0,
            nn.Conv2d,
            nn.Identity,
        )
        self.l2g_layer = tmp_module(
            self.local_in_channels,
            self.global_out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            padding_mode=padding_type,
            bias=bias,
        )

        tmp_module = self._get_module_on_true_predicate(
            self.global_in_channels > 0 and self.local_out_channels > 0,
            nn.Conv2d,
            nn.Identity,
        )
        self.g2l_layer = tmp_module(
            self.global_in_channels,
            self.local_out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            padding_mode=padding_type,
            bias=bias,
        )

        tmp_module = self._get_module_on_true_predicate(
            self.global_in_channels > 0 and self.global_out_channels > 0,
            SpectralTransform,
            nn.Identity,
        )
        self.g2g_layer = tmp_module(
            self.global_in_channels,
            self.global_out_channels,
            fu_kernel=fu_kernel,
            fft_norm=fft_norm,
            padding_type=padding_type,
            bias=bias,
            norm_layer=norm_layer,
            use_only_freq=use_only_freq,
        )

        self.local_bn_relu = (
            nn.Sequential(norm_layer(self.local_out_channels), activation)
            if self.local_out_channels != 0
            else nn.Identity()
        )

        self.global_bn_relu = (
            nn.Sequential(norm_layer(self.global_out_channels), activation)
            if self.global_out_channels != 0
            else nn.Identity()
        )

    @staticmethod
    def _get_module_on_true_predicate(
            condition: bool, true_module=nn.Identity, false_module=nn.Identity
    ):
        if condition:
            return true_module
        else:
            return false_module

    def forward(self, x):

        #  chunk into local and global channels
        x_l, x_g = (
            x[:, : self.local_in_channels, ...],
            x[:, self.local_in_channels:, ...],
        )
        x_l = 0 if x_l.size()[1] == 0 else x_l
        x_g = 0 if x_g.size()[1] == 0 else x_g

        out_local, out_global = torch.Tensor(0).to(x.device), torch.Tensor(0).to(
            x.device
        )

        if self.local_out_channels != 0:
            out_local = self.l2l_layer(x_l) + self.g2l_layer(x_g)
            out_local = self.local_bn_relu(out_local)

        if self.global_out_channels != 0:
            out_global = self.l2g_layer(x_l) + self.g2g_layer(x_g)
            out_global = self.global_bn_relu(out_global)

        #  (B, out_ch, F, T)
        output = torch.cat((out_local, out_global), dim=1)

        return output


class FFCResNetBlock(torch.nn.Module):
    """Implements Residual FFC block.

    Contains two FFC blocks with residual connection.

    Wraps around FFC arguments.

    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            alpha_in: float = 0.5,
            alpha_out: float = 0.5,
            kernel_size: int = 3,
            padding_type: str = "reflect",
            bias: bool = True,
            fu_kernel: int = 1,
            fft_norm: str = "ortho",
            use_only_freq: bool = True,
            norm_layer=nn.BatchNorm2d,
            activation=nn.ReLU(True),
    ):
        super().__init__()
        self.ffc1 = FastFourierConvolution(
            in_channels,
            out_channels,
            alpha_in=alpha_in,
            alpha_out=alpha_out,
            kernel_size=kernel_size,
            padding_type=padding_type,
            fu_kernel=fu_kernel,
            fft_norm=fft_norm,
            use_only_freq=use_only_freq,
            bias=bias,
            norm_layer=norm_layer,
            activation=activation,
        )

        # self.ffc2 = FastFourierConvolution(
        #     in_channels,
        #     out_channels,
        #     alpha_in=alpha_in,
        #     alpha_out=alpha_out,
        #     kernel_size=kernel_size,
        #     padding_type=padding_type,
        #     fu_kernel=fu_kernel,
        #     fft_norm=fft_norm,
        #     use_only_freq=use_only_freq,
        #     bias=bias,
        #     norm_layer=norm_layer,
        #     activation=activation,
        # )

    def forward(self, x):
        out = self.ffc1(x)
        # out = self.ffc2(out)
        return x + out


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


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        # self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
        #                        groups=dw_channel,
        #                        bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.ffconv = self._makeLayer_(FFCResNetBlock, dw_channel, dw_channel, 1)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def _makeLayer_(self, block, inChannals, outChannals, blocks):
        layers = []
        layers.append(block(inChannals, outChannals))

        for i in range(1, blocks):
            layers.append(block(outChannals, outChannals))

        return nn.Sequential(*layers)

    def forward(self, inp):
        x = inp
        # print("inp", x.shape)
        x = self.norm1(x)

        x = self.conv1(x)
        # print(x.shape)
        # x = self.conv2(x)
        # print(x.shape)
        x = self.ffconv(x)
        # print("test", x.shape)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        # print(x.shape)
        # #
        x = self.dropout1(x)
        #
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class FFCNAFNet(nn.Module):

    def __init__(self, img_channel=3, width=32, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


if __name__ == "__main__":
    model = FFCNAFNet().cuda()

    model.eval()
    input = torch.randn(1, 3, 36, 64).cuda()
    # input = torch.randn(1, 3, 32, 32)
    y = model(input)
    # print(y.size())

    print('total parameters:', sum(param.numel() for param in model.parameters()))

    from thop import profile

    flops, params = profile(model=model, inputs=(input,))
    # print('Model:{:.2f} GFLOPs and {:.2f}M parameters'.format(flops / 1e9, params / 1e6))
