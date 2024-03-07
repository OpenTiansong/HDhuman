import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import logging
from icecream import ic

class ImNormalizer(object):
    def __init__(self, in_fmt="-11"):
        self.in_fmt = in_fmt
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def apply(self, x):
        # ic(x.device)
        self.mean = self.mean.to(x.device)
        self.std = self.std.to(x.device)
        if self.in_fmt == "-11":
            x = (x + 1) / 2
        elif self.in_fmt != "01":
            raise Exception("invalid input format")
        return (x - self.mean) / self.std

class INNormalizer(object):
    def __init__(self, in_fmt="-11"):
        self.in_fmt = in_fmt
        self.mean = torch.Tensor([0.485, 0.456, 0.406, 0.5, 0.5, 0.5]).view(1, 6, 1, 1)
        self.std = torch.Tensor([0.229, 0.224, 0.225, 0.5, 0.5, 0.5]).view(1, 6, 1, 1)

    def apply(self, x):
        self.mean = self.mean.to(x.device)
        self.std = self.std.to(x.device)
        if self.in_fmt == "-11":
            x = (x + 1) / 2
        elif self.in_fmt != "01":
            raise Exception("invalid input format")
        return (x - self.mean) / self.std
# Encoder net
class ResUNet(nn.Module):
    def __init__(
        self, out_channels_0=64, out_channels=-1, depth=5, resnet="resnet18"
    ):
        super().__init__()

        if resnet == "resnet18":
            resnet = torchvision.models.resnet18(pretrained=True)
        else:
            raise Exception("invalid resnet model")

        self.normalizer = ImNormalizer()

        if depth < 1 or depth > 5:
            raise Exception("invalid depth of UNet")

        encs = nn.ModuleList()
        enc_translates = nn.ModuleList()
        decs = nn.ModuleList()
        enc_channels = 0
        if depth == 5:
            encs.append(resnet.layer4)
            enc_translates.append(self.convrelu(512, 512, 1))
            enc_channels = 512
        if depth >= 4:
            encs.append(resnet.layer3)
            enc_translates.append(self.convrelu(256, 256, 1))
            if enc_channels > 0:
                decs.append(self.convrelu(enc_channels + 256, 256))
            enc_channels = 256
        if depth >= 3:
            encs.append(resnet.layer2)
            enc_translates.append(self.convrelu(128, 128, 1))
            if enc_channels > 0:
                decs.append(self.convrelu(enc_channels + 128, 128))
            enc_channels = 128
        if depth >= 2:
            encs.append(nn.Sequential(resnet.maxpool, resnet.layer1))
            enc_translates.append(self.convrelu(64, 64, 1))
            if enc_channels > 0:
                decs.append(self.convrelu(enc_channels + 64, 64))
            enc_channels = 64
        if depth >= 1:
            encs.append(nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu))
            enc_translates.append(self.convrelu(64, 64, 1))
            if enc_channels > 0:
                decs.append(self.convrelu(enc_channels + 64, 64))
            enc_channels = 64
        enc_translates.append(
            nn.Sequential(self.convrelu(3, 64), self.convrelu(64, 64))
        )
        decs.append(self.convrelu(enc_channels + 64, out_channels_0))

        self.encs = nn.ModuleList(reversed(encs))
        self.enc_translates = nn.ModuleList(reversed(enc_translates))
        self.decs = nn.ModuleList(decs)

        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        if out_channels <= 0:
            self.out_conv = None
        else:
            self.out_conv = nn.Conv2d(
                out_channels_0, out_channels, kernel_size=1, padding=0
            )

    def convrelu(self, in_channels, out_channels, kernel_size=3, padding=None):
        if padding is None:
            padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )

    # disable batchnorm learning in self.encs
    def train(self, mode=True):
        super().train(mode=mode)
        if not mode:
            return
        for mod in self.encs.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.eval()
                for param in mod.parameters():
                    param.requires_grad_(False)

    def forward(self, x):
        # x = self.normalizer.apply(x)
        # print("x\n", x[0:100])
        # assert 0

        x = (x+1)/2
        x[:, 0] = (x[:, 0] - 0.485) / 0.229
        x[:, 1] = (x[:, 1] - 0.456) / 0.224
        x[:, 2] = (x[:, 2] - 0.406) / 0.225

        outs = [self.enc_translates[0](x)]
        # print("outs0\n", outs) # same
        for enc, enc_translates in zip(self.encs, self.enc_translates[1:]):
            # print("x0\n", x)
            x = enc(x)
            # print("x1\n", x)
            outs.append(enc_translates(x))

        # print("outs1\n", outs)
        # assert 0

        for dec in self.decs:
            x0, x1 = outs.pop(), outs.pop()
            x = torch.cat((self.upsample(x0), x1), dim=1)
            x = dec(x)
            outs.append(x)
        x = outs.pop()
        # print("x\n", x[0:100])
        # assert 0

        if self.out_conv:
            x = self.out_conv(x)
        # print("x\n", x[0:100])
        # assert 0
        return x

class EncodeNet(nn.Module):
    def __init__(
            self,
            res_unet,
    ):
        super(EncodeNet, self).__init__()
        self.enc_net = res_unet

    # def forward(self, **kwargs):
    #     if "src_ims" in kwargs:
    #         src_ims = kwargs["src_ims"]
    #     else:
    #         logging.info(f"Eocode Net needs src imgs as input")
    #         assert 0
    #     B, V, C, H, W = src_ims.shape
    #     src_ims = src_ims.view(B * V, C, H, W)
    #     features = self.enc_net(src_ims)
    #     return features.view(B, V, *features.shape[-3:])

    def forward(self, src_img):
        """

        @param src_img: [B, 3, H, W]
        @return:
        """
        # ic(type(self.enc_net))
        feature = self.enc_net(src_img) # [B, C, H, W]
        # return feature.unsqueeze(1) # [B, 1, C, H, W]
        return feature # [B, C, H, W]

    def forward_image(self, src_img):
        # print("get feature0")
        feature = self.enc_net(src_img) # [B, C, H, W]
        # print("get feature1")
        return feature.unsqueeze(1) # [B, 1, C, H, W]

    # for training texture net
    def forward_tgt(self, tgt_img):
        feature = self.enc_net(tgt_img)
        return feature

class Encoder(nn.Module):
    def __init__(
            self,
            depth,
            out_channels,
    ):
        super(Encoder, self).__init__()
        self.enc_net = ResUNet(depth = depth, out_channels_0=out_channels)

    def forward(self, **kwargs):
        if "src_ims" in kwargs:
            src_ims = kwargs["src_ims"]
        else:
            logging.info(f"Eocode Net needs src imgs as input")
            assert 0
        B, V, C, H, W = src_ims.shape
        src_ims = src_ims.view(B * V, C, H, W)
        features = self.enc_net(src_ims)
        return features.view(B, V, *features.shape[-3:])

    def forward_image(self, src_img):
        # print("get feature0")
        feature = self.enc_net(src_img) # [B, C, H, W]
        # print("get feature1")
        return feature.unsqueeze(1) # [B, 1, C, H, W]

    # for training texture net
    def forward_tgt(self, tgt_img):
        feature = self.enc_net(tgt_img)
        return feature

# Encoder net
class ResUNetIN(nn.Module):
    def __init__(
        self, out_channels_0=64, out_channels=-1, depth=5, resnet="resnet18"
    ):
        super().__init__()

        if resnet == "resnet18":
            resnet = torchvision.models.resnet18(pretrained=True)
        else:
            raise Exception("invalid resnet model")

        self.normalizer = INNormalizer()

        if depth < 1 or depth > 5:
            raise Exception("invalid depth of UNet")

        encs = nn.ModuleList()
        enc_translates = nn.ModuleList()
        decs = nn.ModuleList()
        enc_channels = 0
        if depth == 5:
            encs.append(resnet.layer4)
            enc_translates.append(self.convrelu(512, 512, 1))
            enc_channels = 512
        if depth >= 4:
            encs.append(resnet.layer3)
            enc_translates.append(self.convrelu(256, 256, 1))
            if enc_channels > 0:
                decs.append(self.convrelu(enc_channels + 256, 256))
            enc_channels = 256
        if depth >= 3:
            encs.append(resnet.layer2)
            enc_translates.append(self.convrelu(128, 128, 1))
            if enc_channels > 0:
                decs.append(self.convrelu(enc_channels + 128, 128))
            enc_channels = 128
        if depth >= 2:
            encs.append(nn.Sequential(resnet.maxpool, resnet.layer1))
            enc_translates.append(self.convrelu(64, 64, 1))
            if enc_channels > 0:
                decs.append(self.convrelu(enc_channels + 64, 64))
            enc_channels = 64
        if depth >= 1:
            encs.append(nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu))
            enc_translates.append(self.convrelu(64, 64, 1))
            if enc_channels > 0:
                decs.append(self.convrelu(enc_channels + 64, 64))
            enc_channels = 64
        enc_translates.append(
            nn.Sequential(self.convrelu(6, 64), self.convrelu(64, 64))
        )
        decs.append(self.convrelu(enc_channels + 64, out_channels_0))

        self.encs = nn.ModuleList(reversed(encs))
        self.enc_translates = nn.ModuleList(reversed(enc_translates))
        self.decs = nn.ModuleList(decs)

        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        if out_channels <= 0:
            self.out_conv = None
        else:
            self.out_conv = nn.Conv2d(
                out_channels_0, out_channels, kernel_size=1, padding=0
            )

    def convrelu(self, in_channels, out_channels, kernel_size=3, padding=None):
        if padding is None:
            padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )

    # disable batchnorm learning in self.encs
    def train(self, mode=True):
        super().train(mode=mode)
        if not mode:
            return
        for mod in self.encs.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.eval()
                for param in mod.parameters():
                    param.requires_grad_(False)

    def forward(self, x):
        x = self.normalizer.apply(x)
        # print("x\n", x[0:100])
        # assert 0

        outs = [self.enc_translates[0](x)]
        # print("outs0\n", outs) # same
        for enc, enc_translates in zip(self.encs, self.enc_translates[1:]):
            # print("x0\n", x)
            x = enc(x)
            # print("x1\n", x)
            outs.append(enc_translates(x))

        # print("outs1\n", outs)
        # assert 0

        for dec in self.decs:
            x0, x1 = outs.pop(), outs.pop()
            x = torch.cat((self.upsample(x0), x1), dim=1)
            x = dec(x)
            outs.append(x)
        x = outs.pop()
        # print("x\n", x[0:100])
        # assert 0

        if self.out_conv:
            x = self.out_conv(x)
        # print("x\n", x[0:100])
        # assert 0
        return x

class EncoderIN(nn.Module):
    def __init__(
            self,
            depth=3,
            out_channels=32,
    ):
        super(EncoderIN, self).__init__()
        self.enc_net = ResUNetIN(depth = depth, out_channels_0=out_channels)

    def forward(self, **kwargs):
        if "src_ims" in kwargs:
            src_ims = kwargs["src_ims"]
        else:
            logging.info(f"Eocode Net needs src imgs as input")
            assert 0
        B, V, C, H, W = src_ims.shape
        src_ims = src_ims.view(B * V, C, H, W)
        features = self.enc_net(src_ims)
        return features.view(B, V, *features.shape[-3:])

    def forward_in(self, src_img, src_normal):
        # print("get feature0")
        src_in = torch.cat([src_img, src_normal], dim=1) # [B, 6, H, W]
        feature = self.enc_net(src_in) # [B, 6, H, W]
        # print("get feature1")
        return feature.unsqueeze(1) # [B, 1, 32, H, W]

    # for training texture net
    def forward_tgt(self, tgt_img):
        feature = self.enc_net(tgt_img)
        return feature

# class GraphAttention(nn.)