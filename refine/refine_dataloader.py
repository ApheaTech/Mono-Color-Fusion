import torch
import torch.utils.data as data
from PIL import Image
import os
import glob
import numpy as np
from torchvision import transforms


def default_loader(path):
    return Image.open(path).convert('YCbCr')


IMAGENET_STATS = {'mean': [0.485, 0.456, 0.406],
                  'std': [0.229, 0.224, 0.225]}


def get_transform(std=0., normalize=IMAGENET_STATS):
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize(**normalize)]
    return transforms.Compose(transform_list)


class myImageFolder(data.Dataset):
    def __init__(self, root, loader=default_loader):
        # TODO 确认两边都有文件
        left_shift_images = sorted(glob.glob(os.path.join(root, 'frames_finalpass', '*/*/*/left/*_left_shift.png')))
        right_images = [im.replace('/left', '/right') for im in left_shift_images]
        right_images = [im.replace('_left_shift.png', '.webp') for im in right_images]
        self.loader = loader
        self.transform = get_transform()
        self.left_shift_images = []
        self.right_images = []
        # 减少数量以方便调试
        count = 0
        for left_shift_image, right_image in zip(left_shift_images, right_images):
            # if count < 4399:
            #     self.left_shift_images += [left_shift_image]
            #     self.right_images += [right_image]
            #     count = count + 1
            self.left_shift_images += [left_shift_image]
            self.right_images += [right_image]

    def __getitem__(self, index):
        left_shift_image = self.left_shift_images[index]
        right_image = self.right_images[index]
        left_shift_image = self.loader(left_shift_image)
        right_image = self.loader(right_image)

        # left_shift_image = np.array(left_shift_image).astype('uint8')
        # right_image = np.array(right_image).astype('uint8')
        # left_shift_image = self.transform(left_shift_image)
        # right_image = self.transform(right_image)
        left_shift_image = np.array(left_shift_image).astype('float32') / 255.0
        right_image = np.array(right_image).astype('float32') / 255.0
        left_shift_image = transforms.ToTensor()(left_shift_image)
        right_image = transforms.ToTensor()(right_image)
        # left_shift_image = torch.from_numpy(left_shift_image).permute(2, 0, 1).float()
        # right_image = torch.from_numpy(right_image).permute(2, 0, 1).float()

        return left_shift_image, right_image

    def __len__(self):
        return len(self.right_images)
