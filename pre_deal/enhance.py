import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import optim
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from unet.unet_model import UNet
from enhance_dataloader import LoLDataSet
from LIME import LIME

device = 'cuda:0'
model = UNet(3, 3).to(device)
learning_rate = 0.001
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
scaler = GradScaler()

writer = SummaryWriter("log")


def main():
    # 数据集准备
    train_loader = torch.utils.data.DataLoader(LoLDataSet(), batch_size=6, shuffle=True, num_workers=1, drop_last=False)
    # 训练
    for epoch in range(30):
        for batch_idx, (low_image, gt_image) in enumerate(tqdm(train_loader)):
            low_image = low_image.to(device)
            gt_image = gt_image.to(device)
            with autocast():
                enhance_image = model(low_image)

            loss = F.l1_loss(enhance_image, gt_image)
            scaler.scale(loss).backward()
            loss = loss.item()

            scaler.step(optimizer)
            scaler.update()

            batch_iter = epoch * len(train_loader) + batch_idx
            writer.add_image('enhance img', enhance_image[0], batch_iter)
            writer.add_scalar('l1 loss', loss, batch_iter)

        save_path = Path('enhance_epoch%d.pth' % epoch)
        print(f"Saving file {save_path.absolute()}")
        torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    main()
