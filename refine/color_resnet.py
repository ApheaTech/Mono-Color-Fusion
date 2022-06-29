import torch
import torch.nn as nn


class ResNet(nn.Module):
    r"""Resnet layers implementation summarized in the paper (layer 2~3)"""

    def __init__(self, in_feature, hid_feature, out_feature, repeat=8, k_size=5, stride=1, pad=2):
        super(ResNet, self).__init__()
        # layer 1
        self.layer_top = nn.Sequential(
            nn.Conv2d(in_feature, hid_feature, kernel_size=k_size, stride=stride, padding=pad, bias=False),
            nn.BatchNorm2d(hid_feature),
            nn.ReLU(inplace=True), )
        # layer 2~17
        self.layer_middles = nn.ModuleList()
        for i in range(repeat):
            self.layer_middles.append(nn.Sequential(
                # layer 2
                nn.Conv2d(hid_feature, hid_feature, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(hid_feature),
                nn.ReLU(inplace=True),
                # layer 3
                nn.Conv2d(hid_feature, hid_feature, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(hid_feature),
                nn.ReLU(inplace=True), ))
        # layer 18
        self.layer_bottom = nn.Conv2d(hid_feature, out_feature, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.layer_top(x)
        for layer in self.layer_middles:
            x = layer(x) + x
        x = self.layer_bottom(x)
        return x


class ColorResnet(nn.Module):
    def __init__(self, n=32):
        super(ColorResnet, self).__init__()

        # self.resnet2 = ResNet(1, n, n)
        self.resnet3 = ResNet(1, n, n)
        self.resnet4 = ResNet(1+n, n, 1)

    def forward(self, C_P, mono_img):
        # G_CP = self.resnet2(C_P)
        # G_Y = self.resnet3(mono_img)
        # G = torch.cat([G_CP, G_Y], dim=1)
        # C = self.resnet4(G)
        # return C + C_P

        G_CP = C_P
        G_Y = self.resnet3(mono_img)
        G = torch.cat([G_CP, G_Y], dim=1)
        C = self.resnet4(G)
        return C + C_P
