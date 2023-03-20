import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_layer(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, activation=nn.ReLU,
               batch_norm=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)]
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    if activation:
        layers.append(activation())
    return nn.Sequential(*layers)


def fc_layer(in_channels, out_channels, bias=True, activation=nn.ReLU, batch_norm=False):
    layers = [nn.Linear(int(in_channels), int(out_channels), bias=bias)]
    if batch_norm:
        layers.append(nn.BatchNorm1d(out_channels))
    if activation:
        layers.append(activation())
    return nn.Sequential(*layers)


def slicing(grid, guide):
    N, C, H, W = guide.shape
    device = grid.get_device()
    if device >= 0:
        hh, ww = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device))  # H, W
    else:
        hh, ww = torch.meshgrid(torch.arange(H), torch.arange(W))  # H, W
    # To [-1, 1] range for grid_sample
    hh = hh / (H - 1) * 2 - 1
    ww = ww / (W - 1) * 2 - 1
    guide = guide * 2 - 1
    hh = hh[None, :, :, None].repeat(N, 1, 1, 1)  # N, H, W, C=1
    ww = ww[None, :, :, None].repeat(N, 1, 1, 1)  # N, H, W, C=1
    guide = guide.permute(0, 2, 3, 1)  # N, H, W, C=1

    guide_coords = torch.cat([ww, hh, guide], dim=3)  # N, H, W, 3
    # unsqueeze because extra D dimension
    guide_coords = guide_coords.unsqueeze(1)  # N, Dout=1, H, W, 3
    sliced = F.grid_sample(grid, guide_coords, align_corners=False, padding_mode="border")  # N, C=12, Dout=1, H, W
    sliced = sliced.squeeze(2)  # N, C=12, H, W

    return sliced


def apply(sliced, fullres):
    # r' = w1*r + w2*g + w3*b + w4
    rr = fullres * sliced[:, 0:3, :, :]  # N, C=3, H, W
    gg = fullres * sliced[:, 4:7, :, :]  # N, C=3, H, W
    bb = fullres * sliced[:, 8:11, :, :]  # N, C=3, H, W
    rr = torch.sum(rr, dim=1) + sliced[:, 3, :, :]  # N, H, W
    gg = torch.sum(gg, dim=1) + sliced[:, 7, :, :]  # N, H, W
    bb = torch.sum(bb, dim=1) + sliced[:, 11, :, :]  # N, H, W
    output = torch.stack([rr, gg, bb], dim=1)  # N, C=3, H, W
    return output


class Coefficients(nn.Module):
    def __init__(self, params, c_in=3):
        super(Coefficients, self).__init__()
        self.params = params
        self.relu = nn.ReLU()

        # ===========================Splat===========================
        self.splat1 = conv_layer(c_in, 8, kernel_size=3, stride=2, padding=1, batch_norm=False)
        self.splat2 = conv_layer(8, 16, kernel_size=3, stride=2, padding=1, batch_norm=params['batch_norm'])
        self.splat3 = conv_layer(16, 32, kernel_size=3, stride=2, padding=1, batch_norm=params['batch_norm'])
        self.splat4 = conv_layer(32, 64, kernel_size=3, stride=2, padding=1, batch_norm=params['batch_norm'])

        # ===========================Global===========================
        # Conv until 4x4
        self.global1 = conv_layer(64, 64, kernel_size=3, stride=2, padding=1, batch_norm=params['batch_norm'])
        self.global2 = conv_layer(64, 64, kernel_size=3, stride=2, padding=1, batch_norm=params['batch_norm'])
        # Caculate size after flatten for fc layers
        flatten_size = 4 * 4 * 64  # 4x4 * nchans
        self.global3 = fc_layer(flatten_size, 256, batch_norm=params['batch_norm'])
        self.global4 = fc_layer(256, 128, batch_norm=params['batch_norm'])
        self.global5 = fc_layer(128, 64, activation=None)

        # ===========================Local===========================
        self.local1 = conv_layer(64, 64, kernel_size=3, padding=1, batch_norm=params['batch_norm'])
        self.local2 = conv_layer(64, 64, kernel_size=3, padding=1, bias=False, activation=None)

        # ===========================predicton===========================
        self.pred = conv_layer(64, 96, kernel_size=1, activation=None)  # 64 -> 96

    def forward(self, x):
        N = x.shape[0]
        # ===========================Splat===========================
        x = self.splat1(x)  # N, C=8,  H=128, W=128
        x = self.splat2(x)  # N, C=16, H=64,  W=64
        x = self.splat3(x)  # N, C=32, H=32,  W=32
        x = self.splat4(x)  # N, C=64, H=16,  W=16
        splat_out = x  # N, C=64, H=16,  W=16

        # ===========================Global===========================
        # convs
        x = self.global1(x)  # N, C=64, H=8, W=8
        x = self.global2(x)  # N, C=64, H=4, W=4
        # flatten
        x = x.view(N, -1)  # N, C=64, H=4, W=4 -> N, 1024
        # fcs
        x = self.global3(x)  # N, 256
        x = self.global4(x)  # N, 128
        x = self.global5(x)  # N, 64
        global_out = x  # N, 64

        # ===========================Local===========================
        x = splat_out
        x = self.local1(x)  # N, C=64, H=16,  W=16
        x = self.local2(x)  # N, C=64, H=16,  W=16
        local_out = x  # N, C=64, H=16, W=16

        # ===========================Fusion===========================
        global_out = global_out[:, :, None, None]  # N, 64， 1， 1
        fusion = self.relu(local_out + global_out)  # N, C=64, H=16, W=16

        # ===========================Prediction===========================
        x = self.pred(fusion)  # N, C=96, H=16, W=16
        x = x.view(N, 12, 8, 16, 16)  # N, C=12, D=8, H=16, W=16

        return x


class Guide(nn.Module):
    def __init__(self, params, c_in=3):
        super(Guide, self).__init__()
        self.params = params
        # Number of relus/control points for the curve
        self.nrelus = 16
        self.c_in = c_in
        self.M = nn.Parameter(
            torch.eye(c_in, dtype=torch.float32) + torch.randn(1, dtype=torch.float32) * 1e-4)  # (c_in, c_in)
        self.M_bias = nn.Parameter(torch.zeros(c_in, dtype=torch.float32))  # (c_in,)
        # The shifts/thresholds in x of relus
        thresholds = np.linspace(0, 1, self.nrelus, endpoint=False, dtype=np.float32)  # (nrelus,)
        thresholds = torch.tensor(thresholds)  # (nrelus,)
        thresholds = thresholds[None, None, None, :]  # (1, 1, 1, nrelus)
        thresholds = thresholds.repeat(1, 1, c_in, 1)  # (1, 1, c_in, nrelus)
        self.thresholds = nn.Parameter(thresholds)  # (1, 1, c_in, nrelus)
        # The slopes of relus
        slopes = torch.zeros(1, 1, 1, c_in, self.nrelus, dtype=torch.float32)  # (1, 1, 1, c_in, nrelus)
        slopes[:, :, :, :, 0] = 1.0
        self.slopes = nn.Parameter(slopes)

        self.relu = nn.ReLU()
        self.bias = nn.Parameter(torch.tensor(0, dtype=torch.float32))

    def forward(self, x):
        # Permute from (N, C=3, H, W) to (N, H, W, C=3)
        x = x.permute(0, 2, 3, 1)  # N, H, W, C=3
        old_shape = x.shape  # (N, H, W, C=3)

        x = torch.matmul(x.reshape(-1, self.c_in), self.M)  # N*H*W, C=3
        x = x + self.M_bias
        x = x.reshape(old_shape)  # N, H, W, C=3
        x = x.unsqueeze(4)  # N, H, W, C=3, 1

        x = torch.sum(self.slopes * self.relu(x - self.thresholds), dim=4)  # N, H, W, C=3

        x = x.permute(0, 3, 1, 2)  # N, C=3, H, W
        x = torch.sum(x, dim=1, keepdim=True) / self.c_in  # N, C=1, H, W
        x = x + self.bias  # N, C=1, H, W
        x = torch.clamp(x, 0, 1)  # N, C=1, H, W

        return x


class HDRNet(nn.Module):
    def __init__(self, params):
        super(HDRNet, self).__init__()
        self.coefficients = Coefficients(params)
        self.guide = Guide(params)

    def forward(self, lowres, fullres):
        grid = self.coefficients(lowres)
        guide = self.guide(fullres)
        sliced = slicing(grid, guide)
        output = apply(sliced, fullres)
        return output


if __name__ == '__main__':
    tensor = torch.randn(1, 3, 512, 512).cuda()
    low = torch.randn(1, 3, 256, 256).cuda()
    params = {'batch_norm': False}
    model = HDRNet(params).cuda()
    res = model(low, tensor)
    print(res.shape)
