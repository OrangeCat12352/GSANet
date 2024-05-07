import torch
import torch.nn as nn
import torch.nn.functional as F


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


# FGM 模块
class SSFM(nn.Module):
    def __init__(self, in_dim):
        super(SSFM, self).__init__()

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, guiding_map0):
        m_batchsize, C, height, width = x.size()
        guiding_map0 = F.interpolate(guiding_map0, x.size()[2:], mode='bilinear', align_corners=True)

        guiding_map = F.sigmoid(guiding_map0)

        query = self.query_conv(x) * (1 + guiding_map)
        proj_query = query.view(m_batchsize, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x) * (1 + guiding_map)
        proj_key = key.view(m_batchsize, -1, width * height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        self.energy = energy
        self.attention = attention

        value = self.value_conv(x) * (1 + guiding_map)
        proj_value = value.view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x

        return out


class Decoder(nn.Module):
    def __init__(self, in_channel_List=None):
        super(Decoder, self).__init__()

        if in_channel_List is None:
            in_channel_List = [64, 128, 320, 512]



        self.feature_fuse = nn.Sequential(
            BasicConv2d(in_channel_List[3], 1, 3, 1, 1),
        )

        # self.cgm_4 = FeatureGuideModule(in_channel_List[3])
        # self.cgm_3 = FeatureGuideModule(in_channel_List[2])
        # self.cgm_2 = FeatureGuideModule(in_channel_List[1])

        self.ssfm_4 = SSFM(in_channel_List[3])
        self.ssfm_3 = SSFM(in_channel_List[2])
        self.ssfm_2 = SSFM(in_channel_List[1])

        self.decoder_module3 = BasicConv2d(in_channel_List[3] + in_channel_List[2], in_channel_List[2], 3, 1, 1)
        self.decoder_module2 = BasicConv2d(in_channel_List[2] + in_channel_List[1], in_channel_List[1], 3, 1, 1)
        self.decoder_module1 = BasicConv2d(in_channel_List[1] + in_channel_List[0], in_channel_List[0], 3, 1, 1)

        self.decoder_final = nn.Conv2d(in_channel_List[0], 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input1, input2, input3, input4):
        size = input1.size()[2:]
        layer4_1 = F.interpolate(input4, size, mode='bilinear', align_corners=True)
        feature_map = self.feature_fuse(layer4_1)

        layer4 = self.ssfm_4(input4, feature_map)
        feature3 = self.decoder_module3(
            torch.cat([F.interpolate(layer4, scale_factor=2, mode='bilinear', align_corners=True), input3], 1))

        layer3 = self.ssfm_3(feature3, feature_map)
        feature2 = self.decoder_module2(
            torch.cat([F.interpolate(layer3, scale_factor=2, mode='bilinear', align_corners=True), input2], 1))

        layer2 = self.ssfm_2(feature2, feature_map)
        feature1 = self.decoder_module1(
            torch.cat([F.interpolate(layer2, scale_factor=2, mode='bilinear', align_corners=True), input1], 1))
        # return feature1
        final_map = F.interpolate(self.decoder_final(feature1), scale_factor=4, mode='bilinear', align_corners=True)
        return final_map, self.sigmoid(final_map)
        # return feature1,layer2,layer3,layer4


if __name__ == '__main__':
    input1 = torch.rand(8, 64, 88, 88)
    input2 = torch.rand(8, 128, 44, 44)
    input3 = torch.rand(8, 320, 22, 22)
    input4 = torch.rand(8, 512, 11, 11)

    cg = Decoder()
    out1, out2 = cg(input1, input2, input3, input4)
    print(out1.shape)
    print(out2.shape)
