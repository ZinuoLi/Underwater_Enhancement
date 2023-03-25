import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse


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


class Model(nn.Module):
    def __init__(self, depth=2):
        super(Model, self).__init__()
        self.dwt = DWTForward(J=1, mode='zero', wave='haar')
        self.idwt = DWTInverse(mode='zero', wave='haar')
        self.high = Trans_low()

    def forward(self, inp, tar):
        inp_ll, inp_hf = self.dwt(inp)

        inp_hl = inp_hf[0][:, :, 0, :, :]
        inp_lh = inp_hf[0][:, :, 1, :, :]
        inp_hh = inp_hf[0][:, :, 2, :, :]

        inp_H = torch.cat((inp_hl, inp_lh, inp_hh), dim=1)
        print(inp_H.shape)
        out_H = self.high(inp_H)
        print(out_H.shape)

        out_hl = out_H[:, 0:3, :, :]
        out_lh = out_H[:, 3:6, :, :]
        out_hh = out_H[:, 6:9, :, :]
        print(out_hl.shape)
        print(out_lh.shape)
        print(out_hh.shape)
        pass


if __name__ == '__main__':
    test = torch.randn(1, 3, 256, 256).cuda()
    model = Model().cuda()
    out = model(test, test)
    # print(out.shape)
