import os
import random

import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import albumentations as A


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, film_class='target', img_options=None):
        super(DataLoaderTrain, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, film_class)))

        self.inp_filenames = [os.path.join(rgb_dir, 'input', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, film_class, x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target

        self.ps_w = self.img_options['w']
        self.ps_h = self.img_options['h']

        self.transform = A.Compose([
            A.RandomResizedCrop(width=img_options['w'], height=img_options['h']),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3), ],
            additional_targets={
                'target': 'image',
                # 'mask': 'image'
            }
        )

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')

        inp_img = np.array(inp_img)
        tar_img = np.array(tar_img)

        transformed = self.transform(image=inp_img, target=tar_img)

        inp_img = F.to_tensor(transformed['image'])
        tar_img = F.to_tensor(transformed['target'])

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, filename
    
    
    def get_labels(self):
        labels = []
        for f_name in self.inp_filenames:
            if 'test' in f_name:
                label = 0
            else:
                label = 1
            labels.append(label)
        return labels


class DataLoaderTest(Dataset):
    def __init__(self, rgb_dir, film_class='target', img_options=None):
        super(DataLoaderTest, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, film_class)))

        self.inp_filenames = [os.path.join(rgb_dir, 'input', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, film_class, x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target

        self.ps_w = self.img_options['w']
        self.ps_h = self.img_options['h']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')

        inp_img = np.array(inp_img)
        tar_img = np.array(tar_img)

        inp_img = F.to_tensor(inp_img)
        tar_img = F.to_tensor(tar_img)

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, filename
