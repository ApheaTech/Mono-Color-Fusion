import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import torchvision.transforms as transforms

from unet.unet_model import UNet
from refine_dataloader import myImageFolder
from core.utils.utils import InputPadder
from color_resnet import ColorResnet

# model = UNet(n_channels=3, n_classes=3, bilinear=False).to('cuda')
device = 'cuda:0'
model = ColorResnet().to(device)
# load checkpoint
checkpoint = torch.load('refine_epoch0.pth')
model.load_state_dict(checkpoint, strict=True)
print("Done loading checkpoint")

learning_rate = 0.001
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
scaler = GradScaler()

train_loader = torch.utils.data.DataLoader(
    myImageFolder('/home/xuhang/dataset/SceneFlow/Driving'),
    batch_size=1, shuffle=True, num_workers=1, drop_last=False)

writer = SummaryWriter("log")


# writer.add_graph(model, (torch.zeros(1, 3, 544, 960).to('cuda'), torch.zeros(1, 3, 544, 960).to('cuda')))


def train(left_shift_image, right_image):
    padder = InputPadder(right_image.shape, divis_by=32)
    right_image = padder.pad(right_image)
    right_image = right_image[0]

    model.train()

    optimizer.zero_grad()

    with autocast():
        Cb = model(left_shift_image[:, 1:2], right_image[:, 0][None])
        Cb_loss = F.mse_loss(Cb, right_image[:, 1:2])
        Cr = model(left_shift_image[:, 2:3], right_image[:, 0][None])
        Cr_loss = F.mse_loss(Cr, right_image[:, 2:3])

    scaler.scale(Cb_loss).backward()
    Cb_loss = Cb_loss.item()

    scaler.scale(Cr_loss).backward()
    Cr_loss = Cr_loss.item()

    loss = Cb_loss + Cr_loss

    scaler.step(optimizer)
    scaler.update()
    # loss.backward()
    # optimizer.step()

    return loss, torch.cat([right_image[:, 0][None], Cb, Cr], dim=1)


def main():
    # test
    if True:
        model.eval()

        # left_shift_image = Image.open('/home/xuhang/stereo_matching/RAFT_Stereo/left_shift.png').convert('YCbCr')
        left_shift_image = Image.open('/home/xuhang/dataset/SceneFlow/Driving/frames_finalpass/15mm_focallength/scene_backwards/fast/left/0001_left_shift.png').convert('YCbCr')
        left_shift_image = np.array(left_shift_image).astype('float32') / 255.0
        left_shift_image = transforms.ToTensor()(left_shift_image).to(device)

        # right_image = Image.open('/home/xuhang/dataset/mono+color/right/16#_10lux_mono.bmp').convert('YCbCr')
        right_image = Image.open('/home/xuhang/dataset/SceneFlow/Driving/frames_finalpass/15mm_focallength/scene_backwards/fast/right/0001.webp').convert('YCbCr')
        right_image = np.array(right_image).astype('float32') / 255.0
        right_image = transforms.ToTensor()(right_image).to(device)

        Cb = model(left_shift_image[1:2][None], right_image[0][None, None])
        Cr = model(left_shift_image[2:3][None], right_image[0][None, None])
        pred = torch.cat([right_image[0][None, None], Cb, Cr], dim=1)[0]
        pred = transforms.ToPILImage(mode='YCbCr')(pred).convert('RGB')
        pred.save('pred.png')
        # torchvision.utils.save_image(pred, 'pred.png')
        print('hello')
    # train
    else:
        for epoch in range(100):
            total_train_loss = 0
            count = 0
            for batch_idx, (left_shift_image, right_image) in enumerate(tqdm(train_loader)):
                left_shift_image = left_shift_image.to(device)
                right_image = right_image.to(device)

                loss, pred = train(left_shift_image, right_image)
                if loss < 10000:
                    total_train_loss += loss

                iter_idx = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Loss/Train', loss, iter_idx)

                if (batch_idx+1) % 100 == 0:
                    print('loss=%.3f' % total_train_loss)
                    total_train_loss = 0
                    writer.add_image('pred',
                                     transforms.ToTensor()(
                                         transforms.ToPILImage(mode='YCbCr')(pred.squeeze()).convert('RGB')),
                                     iter_idx)
                    writer.add_image('left_shift',
                                     transforms.ToTensor()(
                                         transforms.ToPILImage(mode='YCbCr')(left_shift_image.squeeze()).convert('RGB')),
                                     iter_idx)
                    writer.add_image('right',
                                     transforms.ToTensor()(
                                         transforms.ToPILImage(mode='YCbCr')(right_image.squeeze()).convert('RGB')),
                                     iter_idx)
            # print('loss=%.3f' % (loss/len(train_loader)))
            # print('epoch %d average loss=%.3f' % (epoch, total_train_loss / 4399))

            save_path = Path('refine_epoch%d.pth' % epoch)
            print(f"Saving file {save_path.absolute()}")
            torch.save(model.state_dict(), save_path)
        print('train success')
        writer.close()


if __name__ == '__main__':
    main()
