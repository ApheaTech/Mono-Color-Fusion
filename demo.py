import sys

# sys.path.append('core')

import argparse
import glob
import numpy as np
import torch
import time
from tqdm import tqdm
from pathlib import Path
from core.raft_stereo import RAFTStereo
from core.utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt

DEVICE = 'cuda'


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def demo(args):
    start = time.time()
    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))
    print("load model use time --> ", time.time() - start)

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            start = time.time()
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            _, flow_up = model(image1, image2, iters=args.valid_iters, test_mode=True)
            file_stem = imfile1.split('/')[-2]
            if args.save_numpy:
                np.save(output_directory / f"{file_stem}.npy", flow_up.cpu().numpy().squeeze())
            plt.imsave(output_directory / f"{file_stem}.png", -flow_up.cpu().numpy().squeeze(), cmap='jet')
            disp = -flow_up.cpu().numpy().squeeze().astype('uint16')
            print("calculate disp use time --> ", time.time() - start)

    return disp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt',
                        default="/home/xuhang/stereo_matching/RAFT_Stereo/raftstereo-middlebury.pth",
                        help="restore checkpoint")
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames",
                        default="/home/xuhang/stereo_matching/stereo-transformer-sttr-light/mydata/left/0001.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames",
                        default="/home/xuhang/stereo_matching/stereo-transformer-sttr-light/mydata/right/0001.png")
    parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
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

    left_img = '/home/xuhang/dataset/mono+color/left/16#_10lux_color.bmp'
    right_img = '/home/xuhang/dataset/mono+color/right/16#_10lux_mono.bmp'
    args.left_imgs = left_img
    args.right_imgs = right_img
    left_img_ycbcr = Image.open(left_img).convert('YCbCr')
    right_img_ycbcr = Image.open(right_img).convert('YCbCr')
    left_img_ycbcr = np.array(left_img_ycbcr)
    right_img_ycbcr = np.array(right_img_ycbcr)
    img_height = left_img_ycbcr.shape[0]
    img_width = left_img_ycbcr.shape[1]

    divis_by = 32
    pad_ht = (((img_height // divis_by) + 1) * divis_by - img_height) % divis_by
    pad_wd = (((img_width // divis_by) + 1) * divis_by - img_width) % divis_by

    disparity = demo(args)

    disparity = disparity[pad_ht:img_height + pad_ht, pad_wd:pad_wd + img_width]
    np.save('disparity_ndarray', disparity)
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
    # for i in range(disparity.shape[0]):
    #     for j in range(disparity.shape[1]):
    #         target = j - disparity[i][j]
    #         if (target > 0) and (target < disparity.shape[1]):
    #             left_shift[i, target, 0] = left_img_ycbcr[i, j, 0]
    #             left_shift[i, target, 1] = left_img_ycbcr[i, j, 1]
    #             left_shift[i, target, 2] = left_img_ycbcr[i, j, 2]
    #             fusion_img[i, target, 1] = left_img_ycbcr[i, j, 1]
    #             fusion_img[i, target, 2] = left_img_ycbcr[i, j, 2]
    start = time.time()
    for i in range(fusion_img.shape[0]):
        left_shift[i, right_shifted[i, :], :] = left_img_ycbcr[i, :, :]
    # 此处若有缺失，则会用到left的值，需要fix
    fusion_img[:, :, 1:3] = left_shift[:, :, 1:3]
    print("fusion img use time --> ", time.time() - start)

    left_shift = Image.fromarray(left_shift, mode='YCbCr').convert('RGB')
    left_shift.save('left_shift.png')
    fusion_img = Image.fromarray(fusion_img, mode='YCbCr').convert('RGB')
    fusion_img.save('fusion.png')

    # for i in range(disparity.shape[0]):
    #     for j in range(disparity.shape[1]):
    #         target = j - disparity[i][j]
    #         if (target > 0) and (target < disparity.shape[1]):
    #             right_img_ycbcr[i, target, 1] = left_img_ycbcr[i, j, 1]
    #             right_img_ycbcr[i, target, 2] = left_img_ycbcr[i, j, 2]
    # right_img = Image.fromarray(right_img_ycbcr, mode='YCbCr').convert('RGB')
    # right_img.save('fusion.png')

    print('hello')
