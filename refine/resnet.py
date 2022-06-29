import torch
import torch.nn as nn

class BaicsBlock(nn.Module):
    # 主分支的卷积个数的倍数
    def expansion(self):
        expansion = 1
        return expansion

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BaicsBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel,
                               out_channels=out_channel,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)  # 不使用偏置,bias=False
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu =nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channel,
                               out_channels=out_channel,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample    # 下采样参数，虚线的残差结构

    def forward(self, x):
        # 捷径分支下采样参数保存变量
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x +=identity
        x = self.relu(x)

        return x

class Bottleneck(nn.Module):
    def expansion(self):
        expansion = 4
        return expansion

    def __bool__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel,
                               out_channels=out_channel,
                               kernel_size=1,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channel,
                               out_channels=out_channel,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(in_channels=out_channel,
                               out_channels=out_channel,
                               kernel_size=1,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion())
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x += identity
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, block, block_list, num_classes=1000, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=self.in_channel,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.layer_1 = self.make_layer(block, 64, block_list[0])
        self.layer_2 = self.make_layer(block, 128, block_list[1], stride=2)
        self.layer_3 = self.make_layer(block, 256, block_list[2], stride=2)
        self.layer_4 = self.make_layer(block, 512, block_list[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool1d((1,1))
            self.fc = nn.Linear(512 * block.expansion(self), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def make_layer(self, block, channel, block_list, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion(self):
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion(self), kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion(self)))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion(self)

        for _ in range(1, block_list):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def ResNet18(num_classes=1000, include_top=True):
    return ResNet(BaicsBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)


def ResNet34(num_classes=1000, include_top=True):
    return ResNet(BaicsBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def ResNet50(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def ResNet101(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def ResNet152(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, include_top=include_top)
