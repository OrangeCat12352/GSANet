import torch
from torch import nn
from model.Decoder.attention_Decoder import Decoder
import torch.nn.functional as F
from model.Encoder.uniformer import uniformer_base_ls


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SDEM(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.convs = nn.ModuleList(
            [nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)] * 4)

    def forward(self, xs, anchor):
        ans = torch.ones_like(anchor)
        target_size = anchor.shape[-1]

        for i, x in enumerate(xs):
            if x.shape[-1] > target_size:
                x = F.adaptive_avg_pool2d(x, (target_size, target_size))
            elif x.shape[-1] < target_size:
                x = F.interpolate(x, size=(target_size, target_size),
                                  mode='bilinear', align_corners=True)

            ans = ans * self.convs[i](x)

        return ans


class MyNet(nn.Module):
    def __init__(self,
                 channel=32):
        super(MyNet, self).__init__()

        # uniformer_base
        backbone = uniformer_base_ls()  # [64, 128, 320, 512]
        path = 'pretrain/uniformer_base_ls_in1k.pth'
        save_model = torch.load(path, map_location='cpu')
        model_dict = backbone.state_dict()
        state_dict = {k: v for k, v in save_model['model'].items() if k in model_dict.keys()}
        backbone.load_state_dict(state_dict)
        self.backbone = backbone

        # neck模块
        self.ca_1 = ChannelAttention(64)
        self.sa_1 = SpatialAttention()

        self.ca_2 = ChannelAttention(128)
        self.sa_2 = SpatialAttention()

        self.ca_3 = ChannelAttention(320)
        self.sa_3 = SpatialAttention()

        self.ca_4 = ChannelAttention(512)
        self.sa_4 = SpatialAttention()

        self.Translayer_1 = BasicConv2d(64, channel, 1)
        self.Translayer_2 = BasicConv2d(128, channel, 1)
        self.Translayer_3 = BasicConv2d(320, channel, 1)
        self.Translayer_4 = BasicConv2d(512, channel, 1)

        self.sdem_1 = SDEM(channel)
        self.sdem_2 = SDEM(channel)
        self.sdem_3 = SDEM(channel)
        self.sdem_4 = SDEM(channel)

        # Decoder模块
        self.Decoder = Decoder(in_channel_List=[32, 32, 32, 32])

    def upsample(self, x, input):
        return F.interpolate(x, size=input.shape[2:], mode='bilinear', align_corners=True)

    def forward(self, x):
        # backbone
        encoder = self.backbone(x)
        x1 = encoder[0]  # 64x88x88
        x2 = encoder[1]  # 128x44x44
        x3 = encoder[2]  # 320x22x22
        x4 = encoder[3]  # 512x11x11

        # neck
        f1 = self.ca_1(x1) * x1
        f1 = self.sa_1(f1) * f1
        f1 = self.Translayer_1(f1)

        f2 = self.ca_2(x2) * x2
        f2 = self.sa_2(f2) * f2
        f2 = self.Translayer_2(f2)

        f3 = self.ca_3(x3) * x3
        f3 = self.sa_3(f3) * f3
        f3 = self.Translayer_3(f3)

        f4 = self.ca_4(x4) * x4
        f4 = self.sa_4(f4) * f4
        f4 = self.Translayer_4(f4)

        f11 = self.sdem_1([f1, f2, f3, f4], f1)
        f21 = self.sdem_2([f1, f2, f3, f4], f2)
        f31 = self.sdem_3([f1, f2, f3, f4], f3)
        f41 = self.sdem_4([f1, f2, f3, f4], f4)

        # Decoder
        sal, sig_sal = self.Decoder(f11, f21, f31, f41)

        return sal, sig_sal
