import torch.utils.data as data
from PIL import Image
import glob
import numpy as np
from torchvision import transforms


def default_loader(path):
    return Image.open(path)


class LoLDataSet(data.Dataset):
    def __init__(self, loader=default_loader):
        self.low_images = sorted(glob.glob('/home/xuhang/dataset/LOLdataset/our485/low/*.png'))
        self.gt_images = [im.replace('/low', '/high') for im in self.low_images]
        self.loader = loader

    def __getitem__(self, index):
        low_image = self.low_images[index]
        gt_image = self.gt_images[index]
        low_image = self.loader(low_image)
        gt_image = self.loader(gt_image)

        low_image = np.array(low_image).astype('float32') / 255.0
        gt_image = np.array(gt_image).astype('float32') / 255.0
        low_image = transforms.ToTensor()(low_image)
        gt_image = transforms.ToTensor()(gt_image)

        return low_image, gt_image

    def __len__(self):
        return len(self.low_images)
