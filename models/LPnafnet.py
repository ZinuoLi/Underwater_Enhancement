import torch
import torch.nn as nn
import torch.nn.functional as F


def gauss_kernel(channels=32):
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


class LPAttention(nn.Module):
    def __init__(self, depth=2, num_dims=32, bias=True):
        super().__init__()
        self.lap_pyramid = LapPyramidConv(depth)
        self.relu = nn.LeakyReLU()
        # q conv
        self.conv1_q = nn.Conv2d(num_dims, num_dims * 3, kernel_size=1, bias=bias)
        self.conv2_q = nn.Conv2d(num_dims * 3, num_dims * 3, kernel_size=3, padding=1, groups=num_dims * 3, bias=bias)

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
        q = self.conv2_q(self.conv1_q(x))
        q = self.relu(q)
        pyr_inp = self.lap_pyramid.pyramid_decom(img=x)

        k1 = self.conv2(self.conv1(pyr_inp[-1]))
        k2 = self.conv2(self.conv1(pyr_inp[-2]))
        k3 = self.conv2(self.conv1(pyr_inp[-3]))
        # k1->k2
        k1 = nn.functional.interpolate(k1, size=(pyr_inp[-2].shape[2], pyr_inp[-2].shape[3]))
        k1k2 = torch.cat([k1, k2], dim=1)
        k1k2 = self.conv3(k1k2)
        k1k2 = self.relu(k1k2)
        # k1k2(k2)->k3
        k1k2 = nn.functional.interpolate(k1k2, size=(pyr_inp[-3].shape[2], pyr_inp[-3].shape[3]))
        k = torch.cat([k1k2, k3], dim=1)
        k = self.conv4(k)
        k = self.relu(k)

        v1 = self.conv2_(self.conv1_(pyr_inp[-1]))
        v2 = self.conv2_(self.conv1_(pyr_inp[-2]))
        v3 = self.conv2_(self.conv1_(pyr_inp[-3]))
        v1 = nn.functional.interpolate(v1, size=(pyr_inp[-2].shape[2], pyr_inp[-2].shape[3]))
        v1v2 = torch.cat([v1, v2], dim=1)
        v1v2 = self.conv3_(v1v2)
        v1v2 = self.relu(v1v2)
        v1v2 = nn.functional.interpolate(v1v2, size=(pyr_inp[-3].shape[2], pyr_inp[-3].shape[3]))
        v = torch.cat([v1v2, v3], dim=1)
        v = self.conv4_(v)
        v = self.relu(v)

        qk = q @ k.transpose(2, 3)
        qkv = qk @ v
        qkv = self.conv5(qkv)
        qkv = self.relu(qkv)
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
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

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
        self.lpattn = LPAttention()

    def forward(self, inp):
        x = inp
        x = self.lpattn(x)
        # x = self.norm1(x)
        #
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.sg(x)
        # x = x * self.sca(x)
        # x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class LPNAFNet(nn.Module):

    def __init__(self, img_channel=3, width=32, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 10],
                 dec_blk_nums=[10, 4, 2, 2]):
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
                nn.Conv2d(chan, chan, 2, 2)
            )

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 4, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan
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
    img_channel = 3
    width = 32

    enc_blks = [2, 2, 4, 8]
    middle_blk_num = 12
    dec_blks = [2, 2, 2, 2]

    print('enc blks', enc_blks, 'middle blk num', middle_blk_num, 'dec blks', dec_blks, 'width', width)

    # using('start . ')
    model = LPNAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                     enc_blk_nums=enc_blks, dec_blk_nums=dec_blks).cuda()

    model.eval()
    input = torch.randn(1, 3, 460, 640).cuda()
    # input = torch.randn(1, 3, 32, 32)
    y = model(input)
    print(y.size())
