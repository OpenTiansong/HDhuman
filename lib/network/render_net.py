import torch
import torch.nn as nn
import torch_scatter
import logging

import ext
import lib.network.net_config as net_config

# Decoder net
class UNet(nn.Module):
    def __init__(
        self,
        in_channels,
        enc_channels=[64, 128, 256],
        dec_channels=[128, 64],
        out_channels=3,
        n_enc_convs=2,
        n_dec_convs=2,
    ):
        super().__init__()

        self.encs = nn.ModuleList()
        self.enc_translates = nn.ModuleList()
        pool = False
        for enc_channel in enc_channels:
            stage = self.create_stage(
                in_channels, enc_channel, n_enc_convs, pool
            )
            self.encs.append(stage)
            translate = nn.Conv2d(enc_channel, enc_channel, kernel_size=1)
            self.enc_translates.append(translate)
            in_channels, pool = enc_channel, True

        self.decs = nn.ModuleList()
        for idx, dec_channel in enumerate(dec_channels):
            in_channels = enc_channels[-idx - 1] + enc_channels[-idx - 2]
            stage = self.create_stage(
                in_channels, dec_channel, n_dec_convs, False
            )
            self.decs.append(stage)

        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        if out_channels <= 0:
            self.out_conv = None
        else:
            self.out_conv = nn.Conv2d(
                dec_channels[-1], out_channels, kernel_size=1, padding=0
            )

    def convrelu(self, in_channels, out_channels, kernel_size=3, padding=None):
        if padding is None:
            padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )

    def create_stage(self, in_channels, out_channels, n_convs, pool):
        mods = []
        if pool:
            mods.append(nn.AvgPool2d(kernel_size=2))
        for _ in range(n_convs):
            mods.append(self.convrelu(in_channels, out_channels))
            in_channels = out_channels
        return nn.Sequential(*mods)

    def forward(self, x):
        outs = []
        # print(f"self.encs {next(self.encs).is_cuda}")
        for enc, enc_translates in zip(self.encs, self.enc_translates):
            x = enc(x)
            outs.append(enc_translates(x))

        for dec in self.decs:
            x0, x1 = outs.pop(), outs.pop()
            x = torch.cat((self.upsample(x0), x1), dim=1)
            x = dec(x)
            outs.append(x)

        x = outs.pop()
        if self.out_conv:
            x = self.out_conv(x)
        return x

class RenderNet(nn.Module):
    def __init__(
        self,
        nets,
        out_conv,
        nets_residual,
        # point_avg_mode="mean",
        # gnns=None,

    ):
        super().__init__()
        # self.point_avg_mode = point_avg_mode # avg
        self.nets_residual = nets_residual
        self.nets = nets # unet
        # if gnns is not None and len(gnns) > len(nets):
        #     raise Exception("invalid number of gnns")
        # self.gnns = gnns
        self.out_conv = out_conv

    # def get_tgt_feature_map(self, ptx, **kwargs):
    #     B, V, C, H, W = ptx.shape
    #     ptx = ext.mytorch.map_to_list_bl(  # [N, 16]
    #         ptx,
    #         kwargs["m2l_src_idx"],
    #         kwargs["m2l_src_pos"],
    #         B,
    #         V,
    #         H,
    #         W
    #     )
    #
    #     bs = B
    #     height = H
    #     width = W
    #
    #     point_key = kwargs["point_key"]
    #     pixel_tgt_idx = kwargs["pixel_tgt_idx"]
    #
    #     # average per point
    #     if self.point_avg_mode == "avg":  # this
    #         x = torch_scatter.segment_csr(ptx, point_key, reduce="mean")  # [160746, 16]
    #         # print("x", x.shape) # [160746, 16]
    #         # assert 0
    #         # print("point_key", point_key.shape) # [537021]
    #         # print("ptx", ptx.shape) # [2676923, 16]
    #         # print("x", x.shape) # [537020, 16]
    #         # print("x0", x[1:5, :])
    #         #
    #         # with torch.no_grad():
    #         #     print("diravg")
    #         #     point_tgt_dirs = kwargs["point_tgt_dirs"][pixel_tgt_idx.long()]
    #         #     point_tgt_dirs = torch_scatter.gather_csr(
    #         #         point_tgt_dirs, point_key
    #         #     )
    #         #     weight = (kwargs["point_src_dirs"] * point_tgt_dirs).sum(dim=1)
    #         #     weight = torch.clamp(weight, 0.01, 1)
    #         #     weight_sum = torch_scatter.segment_csr(
    #         #         weight, point_key, reduce="sum"
    #         #     )
    #         #     weight /= torch_scatter.gather_csr(weight_sum, point_key)
    #         #
    #         # x = weight.view(-1, 1) * ptx
    #         # x = torch_scatter.segment_csr(x, point_key, reduce="sum")
    #         # print("x1", x[1:5, :])
    #         # assert 0
    #
    #
    #     elif self.point_avg_mode == "diravg":
    #         with torch.no_grad():
    #             # print("diravg")
    #             point_tgt_dirs = kwargs["point_tgt_dirs"][pixel_tgt_idx.long()]
    #             point_tgt_dirs = torch_scatter.gather_csr(
    #                 point_tgt_dirs, point_key
    #             )
    #             weight = (kwargs["point_src_dirs"] * point_tgt_dirs).sum(dim=1)
    #             weight = torch.clamp(weight, 0.01, 1)
    #             weight_sum = torch_scatter.segment_csr(
    #                 weight, point_key, reduce="sum"
    #             )
    #             weight /= torch_scatter.gather_csr(weight_sum, point_key)
    #
    #         x = weight.view(-1, 1) * ptx
    #         x = torch_scatter.segment_csr(x, point_key, reduce="sum")
    #     else:
    #         raise Exception("invalid avg_mode")
    #
    #     # print("x", type(x), x.shape)
    #     # print("pixel_tgt_idx", type(pixel_tgt_idx), pixel_tgt_idx.shape)
    #     # print(bs, height, width)
    #     # assert 0
    #
    #     # project to target
    #     x, mask = ext.mytorch.list_to_map(x, pixel_tgt_idx, bs, height, width)  # [B, C, H, W]
    #     # print("x", x.shape) # [1, 16, 576, 992]
    #     return x, mask

    def forward(self, encoded_src_feature_maps, **kwargs): # point features
        x, mask = self.get_tgt_feature_map(encoded_src_feature_maps, **kwargs) # [B, C, H, W]
        # logging.info(f"mask {mask.shape}") # [B, 1, H, W]

        B, V, C, H, W = encoded_src_feature_maps.shape
        bs = B
        height = H
        width = W

        # mv fusion & refinement net
        for sidx in range(len(self.nets)):
            # process per 3D point
            if self.gnns is not None and sidx < len(self.gnns):
                gnn = self.gnns[sidx]
                x, ptx = gnn(x, encoded_src_feature_maps, bs, height, width, **kwargs) # [1, 16, 576, 992] mv fusion scheme [MLPDIR]
                # print("x0", x.shape)

            unet = self.nets[sidx]
            if self.nets_residual: # this 1
                x = x + unet(x) # [1, 16, 576, 992]
            else:
                x = unet(x)

        # logging.info(f"x {x.device}")
        # logging.info(f"out conv {next(self.out_conv.parameters()).device}")
        if self.out_conv:
            x = self.out_conv(x) # [1, 3, 576, 992]
            # print("x2", x.shape)

        tgt_dm = kwargs["tgt_dm"]
        mask_from_tgt_dm = (tgt_dm == 0).unsqueeze(1).repeat(1, 3, 1, 1) # [B, 3, H, W]
        if net_config.background_black:
            x[mask_from_tgt_dm] = -1
        if net_config.background_white:
            x[mask_from_tgt_dm] = 1


        # x *= mask_from_tgt_dm

        # print("tgt_dm", tgt_dm.shape) # [B, H, W]
        # print("mask", mask.shape) # [B, 1, H, W]
        # print("out", x.shape) # [B, C, H, W]
        # assert 0

        return {"out": x, "mask": mask}

    def forward_feature_map(self, tgt_feature_map, **kwargs):
        # bs, C, height, width = tgt_feature_map.shape
        x = tgt_feature_map

        # mv fusion & refinement net
        for sidx in range(len(self.nets)):

            unet = self.nets[sidx]
            if self.nets_residual:  # this 1
                x = x + unet(x)  # [1, 16, 576, 992]
                # print("x1", x.shape) # [1, 16, 576, 992]
            else:
                x = unet(x)

        # logging.info(f"x {x.device}")
        # logging.info(f"out conv {next(self.out_conv.parameters()).device}")
        if self.out_conv:
            x = self.out_conv(x)  # [1, 3, 576, 992]
            # print("x2", x.shape)

        tgt_dm = kwargs["tgt_dm"]
        # tgt_dm = kwargs["tgt_depth"]
        mask_from_tgt_dm = (tgt_dm == 0).unsqueeze(1).repeat(1, 3, 1, 1)  # [B, 3, H, W]
        if net_config.background_black:
            x[mask_from_tgt_dm] = -1
        if net_config.background_white:
            x[mask_from_tgt_dm] = 1

        # x *= mask_from_tgt_dm

        # print("tgt_dm", tgt_dm.shape) # [B, H, W]
        # print("mask", mask.shape) # [B, 1, H, W]
        # print("out", x.shape) # [B, C, H, W]
        # assert 0

        return {"out": x}

    def test_tex_feature_map(self, tgt_feature_map, **kwargs):
        # bs, C, height, width = tgt_feature_map.shape
        x = tgt_feature_map

        # mv fusion & refinement net
        for sidx in range(len(self.nets)):

            unet = self.nets[sidx]
            if self.nets_residual:  # this 1
                x = x + unet(x)  # [1, 16, 576, 992]
                # print("x1", x.shape) # [1, 16, 576, 992]
            else:
                x = unet(x)

        # logging.info(f"x {x.device}")
        # logging.info(f"out conv {next(self.out_conv.parameters()).device}")
        if self.out_conv:
            x = self.out_conv(x)  # [1, 3, 576, 992]
            # print("x2", x.shape)

        # tgt_uv_map = kwargs["tgt_uv_map"]
        # B, H, W, C = tgt_uv_map.shape
        tgt_uv = kwargs["tgt_uv_map"].permute(0, 3, 1, 2) # [B, 2, H, W]
        mask_from_tgt_dm = (tgt_uv[:, 0:1] == 0).repeat(1, 3, 1, 1)  # [B, 3, H, W]
        if net_config.background_black:
            x[mask_from_tgt_dm] = -1
        if net_config.background_white:
            x[mask_from_tgt_dm] = 1

        # x *= mask_from_tgt_dm

        # print("tgt_dm", tgt_dm.shape) # [B, H, W]
        # print("mask", mask.shape) # [B, 1, H, W]
        # print("out", x.shape) # [B, C, H, W]
        # assert 0

        return {"out": x}