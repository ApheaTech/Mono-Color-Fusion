import os

import torch
import cv2
import argparse
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import transforms

from pre_deal.LIME import LIME
from core.raft_stereo import RAFTStereo
from core.utils.utils import InputPadder

# 配置
DEVICE = 'cuda:0'


def pre_process(left_img_path, right_img_path):
    # Lime
    # lime = LIME()
    # lime.load(left_img_path)
    # lime.enhance()
    # left_color_img = Image.fromarray(lime.R)
    # left_gray_img = left_color_img.convert(mode='L').convert(mode='RGB')
    #
    # lime.load(right_img_path)
    # lime.enhance()
    # right_mono_img = Image.fromarray(lime.R)

    # Gamma cv2.decolor
    # left_img = cv2.imread(left_img_path).astype('float32') / 255.0
    # right_img = cv2.imread(right_img_path).astype('float32') / 255.0
    # right_img_sum = np.sum(right_img)
    # left_color_imgs = []
    # min_diff = 10000000
    # min_index = 0
    # for gamma in range(1, 11):
    #     left_color_img = np.power(left_img, gamma/10)
    #     left_color_imgs.append(left_color_img)
    #     left_gray_img = np.repeat(cv2.decolor((left_color_img*255).astype('uint8'))[0][:, :, np.newaxis], 3, axis=2)
    #     left_gray_sum = np.sum(np.array(left_gray_img).astype('float32') / 255.0)
    #     sum_diff = abs(right_img_sum-left_gray_sum)
    #     if sum_diff <= min_diff:
    #         min_diff = sum_diff
    #         min_index = gamma
    # left_color_img = (left_color_imgs[min_index] * 255.0).astype('uint8')
    # left_gray_img = np.repeat(cv2.decolor(left_color_img)[0][:, :, np.newaxis], 3, axis=2)
    # right_mono_img = cv2.imread(right_img_path).astype('float32')

    left_img = np.array(Image.open(left_img_path)).astype('float32') / 255.0
    right_img = np.array(Image.open(right_img_path)).astype('float32') / 255.0
    # mono图像的强度值
    right_img_sum = np.sum(right_img)

    left_color_imgs = []
    min_diff = 10000000
    min_index = 0

    # 遍历gamma
    for gamma in range(1, 11):
        # color图gamma变换
        left_color_img = np.power(left_img, gamma / 10)
        left_color_imgs.append(left_color_img)
        # 得到color图灰度图的强度值
        left_gray_img = Image.fromarray((left_color_img * 255).astype('uint8'), mode='RGB').convert(mode='L').convert(
            mode='RGB')
        left_gray_sum = np.sum(np.array(left_gray_img).astype('float32') / 255.0)
        # 求强度差值
        sum_diff = abs(right_img_sum - left_gray_sum)

        if sum_diff <= min_diff:
            min_diff = sum_diff
            min_index = gamma
    # 得到灰度图强度值与mono图强度值最接近的彩色图
    left_color_img = (left_color_imgs[min_index] * 255.0).astype('uint8')
    left_color_img = Image.fromarray(left_color_img)
    left_gray_img = left_color_img.convert(mode='L').convert(mode='RGB')
    # 暂时不对mono图像增强
    right_mono_img = Image.open(right_img_path)

    return left_gray_img, right_mono_img, left_color_img


def stereo_matching(left_gray_img, right_mono_img, args):
    start = time.time()
    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))
    print("load model use time --> ", time.time() - start)

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        start = time.time()
        image1 = transforms.ToTensor()(left_gray_img)[None].to(DEVICE)
        image2 = transforms.ToTensor()(right_mono_img)[None].to(DEVICE)

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        _, flow_up = model(image1, image2, iters=32, test_mode=True)
        disp = -flow_up.cpu().numpy().squeeze().astype('uint16')
        jet = -flow_up.cpu().numpy().squeeze()
        print("calculate disp use time --> ", time.time() - start)

    img_width = left_gray_img.size[0]
    img_height = left_gray_img.size[1]
    divis_by = 32
    pad_ht = (((img_height // divis_by) + 1) * divis_by - img_height) % divis_by
    pad_wd = (((img_width // divis_by) + 1) * divis_by - img_width) % divis_by

    return disp[pad_ht:img_height + pad_ht, pad_wd:pad_wd + img_width], jet


def fusion(disparity, left_color_img, right_mono_img):
    left_img_ycbcr = np.array(left_color_img.convert(mode='YCbCr'))
    right_img_ycbcr = np.array(right_mono_img.convert(mode='YCbCr'))

    w = disparity.shape[-1]
    coord = np.linspace(0, w - 1, w)[None,]  # 1xW
    right_shifted = coord - disparity
    occ_mask_l = right_shifted <= 0
    right_shifted[occ_mask_l] = 0  # set negative locations to 0
    right_shifted = right_shifted.astype('uint16')

    # 扭曲的彩色图像
    left_shift = np.zeros_like(left_img_ycbcr)
    # 融合图像
    fusion_img = right_img_ycbcr
    start = time.time()
    for i in range(fusion_img.shape[0]):
        left_shift[i, right_shifted[i, :], :] = left_img_ycbcr[i, :, :]
    # 此处若有缺失，则会用到left的值，需要fix
    a = left_shift[:, :, 1:3]
    a[a == 0] = 128
    fusion_img[:, :, 1:3] = a
    print("fusion img use time --> ", time.time() - start)

    left_shift = Image.fromarray(left_shift, mode='YCbCr').convert('RGB')
    fusion_img = Image.fromarray(fusion_img, mode='YCbCr').convert('RGB')

    return fusion_img, left_shift


def main(args):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    save_dir = './temp_output/' + now
    os.mkdir(save_dir)

    left_img_path = '/home/xuhang/dataset/mono+color/left/3#_4lux_color.bmp'
    right_img_path = '/home/xuhang/dataset/mono+color/right/3#_4lux_mono.bmp'

    # 预处理
    left_gray_img, right_mono_img, left_color_img = pre_process(left_img_path, right_img_path)

    # 立体匹配
    disp, jet = stereo_matching(left_gray_img, right_mono_img, args)

    # 空间融合
    fusion_img, left_shift = fusion(disp, left_color_img, right_mono_img)

    # 颜色修正

    # 保存
    left_gray_img.save(save_dir + '/left_gray_img.png')
    right_mono_img.save(save_dir + '/right_mono_img.png')
    left_color_img.save(save_dir + '/left_color_img.png')
    plt.imsave(save_dir + '/jet.png', jet, cmap='jet')
    left_shift.save(save_dir + '/left_shift.png')
    fusion_img.save(save_dir + '/fusion_enhance.png')
    Image.open(left_img_path).save(save_dir + '/origin_left.png')
    Image.open(right_img_path).save(save_dir + '/origin_right.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--restore_ckpt',
                        default="/home/xuhang/stereo_matching/RAFT_Stereo/raftstereo-middlebury.pth",
                        help="restore checkpoint")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3,
                        help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg",
                        help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true',
                        help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    args = parser.parse_args()

    main(args)
