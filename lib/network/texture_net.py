import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleLayerTexture(nn.Module):
    def __init__(self, W, H):
        super(SingleLayerTexture, self).__init__()
        self.layer1 = nn.Parameter(torch.FloatTensor(1, 1, W, H).zero_())

    def forward(self, x):
        batch = x.shape[0]
        x = x * 2.0 - 1.0
        y = F.grid_sample(self.layer1.repeat(batch, 1, 1, 1), x)
        return y


class LaplacianPyramid(nn.Module):
    def __init__(self, W, H):
        super(LaplacianPyramid, self).__init__()
        self.layer1 = nn.Parameter(torch.FloatTensor(1, 1, W, H).zero_())
        self.layer2 = nn.Parameter(torch.FloatTensor(1, 1, W // 2, H // 2).zero_())
        self.layer3 = nn.Parameter(torch.FloatTensor(1, 1, W // 4, H // 4).zero_())
        self.layer4 = nn.Parameter(torch.FloatTensor(1, 1, W // 8, H // 8).zero_())

    def forward(self, x):
        batch = x.shape[0]
        x = x * 2.0 - 1.0
        y1 = F.grid_sample(self.layer1.repeat(batch,1,1,1), x, align_corners=True)
        y2 = F.grid_sample(self.layer2.repeat(batch,1,1,1), x, align_corners=True)
        y3 = F.grid_sample(self.layer3.repeat(batch,1,1,1), x, align_corners=True)
        y4 = F.grid_sample(self.layer4.repeat(batch,1,1,1), x, align_corners=True)
        y = y1 + y2 + y3 + y4
        return y


class Texture(nn.Module):
    def __init__(self, W, H, feature_num, use_pyramid=False):
        super(Texture, self).__init__()
        self.feature_num = feature_num
        self.use_pyramid = use_pyramid
        self.layer1 = nn.ParameterList()
        self.layer2 = nn.ParameterList()
        self.layer3 = nn.ParameterList()
        self.layer4 = nn.ParameterList()
        if self.use_pyramid:
            self.textures = nn.ModuleList([LaplacianPyramid(W, H) for i in range(feature_num)])
            for i in range(self.feature_num):
                self.layer1.append(self.textures[i].layer1)
                self.layer2.append(self.textures[i].layer2)
                self.layer3.append(self.textures[i].layer3)
                self.layer4.append(self.textures[i].layer4)
        else:
            self.textures = nn.ModuleList([SingleLayerTexture(W, H) for i in range(feature_num)])
            for i in range(self.feature_num):
                self.layer1.append(self.textures[i].layer1)

    def setZero(self):
        for i in range(self.feature_num):
            if self.use_pyramid:
                self.textures[i].layer1[:, :, 0, 0].data.zero_()
                self.textures[i].layer2[:, :, 0, 0].data.zero_()
                self.textures[i].layer3[:, :, 0, 0].data.zero_()
                self.textures[i].layer4[:, :, 0, 0].data.zero_()
            else:
                pass

    def forward(self, x):
        y_i = []
        for i in range(self.feature_num):
            y = self.textures[i](x)
            y_i.append(y)
        y = torch.cat(tuple(y_i), dim=1)
        return y

class TextureNet(nn.Module):
    def __init__(self, W, H, feature_channels, use_pyramid):
        super(TextureNet, self).__init__()
        self.feature_channels = feature_channels
        self.use_pyramid = use_pyramid
        self.texture = Texture(W, H, feature_channels, use_pyramid)

        self.weightInit()

    def weightInit(self):
        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

    def setZero(self):
        self.texture.setZero()

    def forward(self, **kwargs):
        uv_map = kwargs["tgt_uv_map"]
        # print("uv map", uv_map.shape)
        x = self.texture(uv_map)
        return x