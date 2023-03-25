import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.color import rgb_to_lab
from torchvision.models.vgg import vgg16, VGG16_Weights


class Perceptual(torch.nn.Module):
    def __init__(self):
        super(Perceptual, self).__init__()
        vgg_model = vgg16(weights=VGG16_Weights.DEFAULT).features[:16]
        for param in vgg_model.parameters():
            param.requires_grad = False
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            module.to(x.device)
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, pred_im, gt):
        loss = []
        pred_im_features = self.output_features(pred_im)
        gt_features = self.output_features(gt)
        for pred_im_feature, gt_feature in zip(pred_im_features, gt_features):
            loss.append(F.mse_loss(pred_im_feature, gt_feature))
        return sum(loss) / len(loss)


class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()

    def forward(self, inp, tar):
        lab_inp, lab_tar = rgb_to_lab(inp), rgb_to_lab(tar)
        l1, a1, b1 = torch.moveaxis(lab_inp, 1, 0)[:3]
        l2, a2, b2 = torch.moveaxis(lab_tar, 1, 0)[:3]
        return torch.mean(torch.sqrt((l2 - l1) ** 2 + (a2 - a1) ** 2 + (b2 - b1) ** 2))


if __name__ == '__main__':
    tensor1 = torch.randn(1, 3, 360, 540)
    tensor2 = torch.randn(1, 3, 360, 540)
    loss = ColorLoss()
    l = loss(tensor1, tensor2)
    im_orig = tensor1.squeeze(0).permute(1, 2, 0).numpy()
    im_edit = tensor2.squeeze(0).permute(1, 2, 0).numpy()
    from skimage import color

    lab_orig = color.rgb2lab(im_orig)
    lab_edit = color.rgb2lab(im_edit)
    de_diff = color.deltaE_cie76(lab_orig, lab_edit)
    print(np.mean(de_diff))
    print(l)
