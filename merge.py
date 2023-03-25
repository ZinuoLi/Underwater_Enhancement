# coding:utf-8
import torch.nn as nn
import torch

import os, torchvision
from PIL import Image
from torchvision import transforms as trans


def wavelet():
    from pytorch_wavelets import DWTForward, DWTInverse  # (or import DWT, IDWT)
    # J为分解的层次数,wave表示使用的变换方法
    xfm = DWTForward(J=1, mode='zero', wave='haar')  # Accepts all wave types available to PyWavelets
    ifm = DWTInverse(mode='zero', wave='haar')

    img = Image.open('./4.png')
    transform = trans.Compose([
        trans.ToTensor()
    ])
    img = transform(img).unsqueeze(0)
    inp_ll, inp_hf = xfm(img)

    inp_hl = inp_hf[0][:, :, 0, :, :]
    inp_lh = inp_hf[0][:, :, 1, :, :]
    inp_hh = inp_hf[0][:, :, 2, :, :]

    torchvision.utils.save_image(inp_ll, 'll-.png')
    torchvision.utils.save_image(inp_hl, 'hl-.png')
    torchvision.utils.save_image(inp_lh, 'lh-.png')
    torchvision.utils.save_image(inp_hh, 'hh-.png')


if __name__ == '__main__':
    # pass
    # import torchvision.transforms.functional as F
    # import torchvision.transforms as TF
    #
    # img1 = cv2.imread('hh.png')
    # img2 = cv2.imread('hl.png')
    # img3 = cv2.imread('lh.png')
    # res = cv2.merge([img1, img2, img3])
    # res_ = F.to_tensor(res)
    # print(res_.shape)
    #
    # # 将九通道图片分离成三个三通道图片
    # img4 = res_[:,0:3,:,:]
    # img5 = res_[:,3:6,:,:]
    # img6 = res_[:,6:9,:,:]
    #
    # # 将三张图片保存
    # cv2.imwrite('image1.png', img4)
    # cv2.imwrite('image2.png', img5)
    # cv2.imwrite('image3.png', img6)

    wavelet()
