import torch
import torch_scatter
from icecream import ic

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

import ext.mytorch as svs_ext_mytorch
import lib.network.net_config as net_config
from lib.network.encode_net import EncodeNet, ResUNet
from lib.network.render_net import RenderNet, UNet

def get_encode_net(encode_net_name):
    if encode_net_name in ["id", "identity"]:
        pass

        # logging.info(f"[NET][EncNet] identity")
        # enc_net = Identity()
        # enc_channels = 3
    elif encode_net_name == "iddummy":
        pass
        # logging.info(f"[NET][EncNet] iddummy")
        # enc_net = nn.Conv2d(3, 3, kernel_size=1, padding=0, bias=False)
        # enc_net.weight.data.fill_(0)
        # enc_net.weight.data[0, 0, 0, 0] = 1
        # enc_net.weight.data[1, 1, 0, 0] = 1
        # enc_net.weight.data[2, 2, 0, 0] = 1
        # enc_channels = 3
    elif encode_net_name == "resunet3":
        pass
        # logging.info(f"[NET][EncNet] resunet3[.64]")
        # enc_net = ResUNet(depth=3)
        # enc_channels = 64
    elif encode_net_name == "resunet3.32":
        logging.info(f"[NET][EncNet] resunet3.32")
        res_unet = ResUNet(out_channels_0=32, depth=3)
        # enc_channels = 32
    elif encode_net_name == "resunet3.16": # this
        logging.info(f"[NET][EncNet] resunet3.16")
        res_unet = ResUNet(out_channels_0=16, depth=3)
        # enc_channels = 16
    elif encode_net_name == "resunet3.64":
        logging.info(f"[NET][EncNet] resunet3.64")
        res_unet = ResUNet(out_channels_0=64, depth=3)
    elif encode_net_name == "vggunet16.3":
        pass

        # logging.info(f"[NET][EncNet] vggunet16.3")
        # enc_net = VGGUNet(net="vgg16", n_encoder_stages=3)
        # enc_channels = 64
    else:
        raise Exception("invalid enc_net")

    enc_net = EncodeNet(res_unet)

    return enc_net

# this for create unet of render net
def get_refnet_net(net_name, params, in_channels):
    if net_name == "id":
        pass

        # logging.info(f"[NET][RefNet]   Identity()")
        # return Identity(), in_channels
    elif net_name == "unet": # this
        depth, n_conv, channels = list(map(int, params)) # 5 2 16
        enc_channels = [channels * (2 ** idx) for idx in range(depth - 1)]
        enc_channels.append(enc_channels[-1])
        dec_channels = enc_channels[::-1][1:]
        logging.info(
            f"[NET][UNet]   Unet(in_channels={in_channels}, enc_channels={enc_channels}, dec_channels={dec_channels}, n_conv={n_conv})"
        )
        unet = UNet(
            in_channels=in_channels,
            enc_channels=enc_channels,
            dec_channels=dec_channels,
            out_channels=-1,
            n_enc_convs=n_conv,
            n_dec_convs=n_conv,
        )
        return unet, dec_channels[-1]
    else:
        raise Exception(f"invalid net_name {net_name}")

def get_refnet_net_new(net_name, params, unet_channels, in_channels):
    if net_name == "id":
        pass

        # logging.info(f"[NET][RefNet]   Identity()")
        # return Identity(), in_channels
    elif net_name == "unet": # this
        depth, n_conv = list(map(int, params)) # 5 2
        # enc_channels = [channels * (2 ** idx) for idx in range(depth - 1)]
        unet_channels.append(unet_channels[-1])
        # ic(unet_channels, depth)
        assert len(unet_channels) == depth, "enc channels should equal to depth"
        dec_channels = unet_channels[::-1][1:]
        logging.info(
            f"[NET][UNet]   Unet(in_channels={in_channels}, enc_channels={unet_channels}, dec_channels={dec_channels}, n_conv={n_conv})"
        )
        unet = UNet(
            in_channels=in_channels,
            enc_channels=unet_channels,
            dec_channels=dec_channels,
            out_channels=-1,
            n_enc_convs=n_conv,
            n_dec_convs=n_conv,
        )
        return unet, dec_channels[-1]
    else:
        raise Exception(f"invalid net_name {net_name}")

# def get_render_net(name, in_channels):
#     # ref_net format
#     # point_edges_mode . point+aux+data . point_avg_mode .
#     #   type+n_seq+residual+net [ . type+gnn]
#     #
#     # net=id  ... id
#     # net=unet ... unet + depth + n_conv + channels
#
#     splits = name.split(".")
#     point_edges_mode = splits[0] # peone
#     point_aux_data = splits[1] # dirs
#     point_avg_mode = splits[2] # avg
#     logging.info(f"[NET][RefNet] point_edges_mode={point_edges_mode}") # penone
#     logging.info(f"[NET][RefNet] point_aux_data={point_aux_data}") # dirs
#     logging.info(f"[NET][RefNet] point_avg_mode={point_avg_mode}") # avg
#
#     net_params = splits[3].split("+") # seq+9+1+unet+5+2+19
#     ref_type = net_params[0] # seq
#     n_seq = int(net_params[1]) # 9
#     nets_residual = int(net_params[2]) != 0 # 1
#     net_name = net_params[3] # unet
#     net_params = net_params[4:] # 5 2 19
#
#     nets = nn.ModuleList()
#     if ref_type == "shared":
#         logging.info(
#             f"[NET][RefNet] Shared {n_seq} nets, nets_residual={nets_residual}"
#         )
#         net, in_channels = get_refnet_net(net_name, net_params, in_channels)
#         for _ in range(n_seq):
#             nets.append(net)
#     elif ref_type == "seq": # this
#         logging.info(
#             f"[NET][RefNet] Seq {n_seq} nets, nets_residual={nets_residual}" # n_seq:9, nets_residual:True
#         )
#         for _ in range(n_seq):
#             net, in_channels = get_refnet_net(net_name, net_params, in_channels)
#             nets.append(net)
#     else:
#         raise Exception(f"invalid ref_type {ref_type}")
#
#     if len(splits) == 5:
#         pass
#
#         # gnn = splits[4].split("+")
#         # gnn_type = gnn[0] # single
#         # gnn_name = gnn[1] # mlpdir
#         # gnn_params = gnn[2:] # mean+3+64+16
#         # gnns = nn.ModuleList()
#         # if gnn_type == "single": # this
#         #     logging.info("[NET][RefNet] Single gnn")
#         #     gnn, in_channels = get_gnn(gnn_name, gnn_params, in_channels)
#         #     gnns.append(gnn)
#         # elif gnn_type == "shared":
#         #     logging.info(f"[NET][RefNet] Shared {n_seq} gnns")
#         #     gnn, in_channels = get_gnn(gnn_name, gnn_params, in_channels)
#         #     for _ in range(n_seq):
#         #         gnns.append(gnn)
#         # elif gnn_type == "seq":
#         #     logging.info(f"[NET][RefNet] Seq {n_seq} gnns")
#         #     for gnnidx in range(n_seq):
#         #         gnn, in_channels = get_gnn(gnn_name, gnn_params, in_channels)
#         #         gnns.append(gnn)
#         # else:
#         #     raise Exception(f"invalid gnn type {gnn_type}")
#     else:
#         gnns = None
#
#     if net_name.startswith("id"):
#         logging.info(f"[NET][RefNet] no out_conv")
#         out_conv = None
#     else: # this
#         logging.info(f"[NET][RefNet] out_conv({in_channels}, 3)")
#         out_conv = nn.Conv2d(in_channels, 3, kernel_size=1, padding=0)
#
#     render_net = RenderNet(
#         point_avg_mode=point_avg_mode,
#         nets=nets,
#         nets_residual=nets_residual,
#         gnns=gnns,
#         out_conv=out_conv,
#     )
#     # print(ref_net)
#     return render_net

def get_render_net(render_net_name, encode_net_name):
    # ref_net format
    # point_edges_mode . point+aux+data . point_avg_mode .
    #   type+n_seq+residual+net [ . type+gnn]
    #
    # net=id  ... id
    # net=unet ... unet + depth + n_conv + channels

    in_channels = int(encode_net_name.split(".")[1])
    splits = render_net_name.split(".")
    # point_edges_mode = splits[0] # peone
    # point_aux_data = splits[1] # dirs
    # point_avg_mode = splits[2] # avg
    # logging.info(f"[NET][RefNet] point_edges_mode={point_edges_mode}") # penone
    # logging.info(f"[NET][RefNet] point_aux_data={point_aux_data}") # dirs
    # logging.info(f"[NET][RefNet] point_avg_mode={point_avg_mode}") # avg

    net_params = splits[3].split("+") # seq+9+1+unet+5+2+19
    ref_type = net_params[0] # seq
    n_seq = int(net_params[1]) # 9
    nets_residual = int(net_params[2]) != 0 # 1
    net_name = net_params[3] # unet
    net_params = net_params[4:] # 5 2 19

    nets = nn.ModuleList()
    if ref_type == "shared":
        # logging.info(
        #     f"[NET][RefNet] Shared {n_seq} nets, nets_residual={nets_residual}"
        # )
        # net, in_channels = get_refnet_net(net_name, net_params, in_channels)
        # for _ in range(n_seq):
        #     nets.append(net)
        pass
    elif ref_type == "seq": # this
        logging.info(
            f"[NET][RefNet] Seq {n_seq} nets, nets_residual={nets_residual}" # n_seq:9, nets_residual:True
        )
        for _ in range(n_seq):
            net, in_channels = get_refnet_net(net_name, net_params, in_channels)
            nets.append(net)
    else:
        raise Exception(f"invalid ref_type {ref_type}")

    if net_name.startswith("id"):
        logging.info(f"[NET][RefNet] no out_conv")
        out_conv = None
    else: # this
        logging.info(f"[NET][RefNet] out_conv({in_channels}, 3)")
        out_conv = nn.Conv2d(in_channels, 3, kernel_size=1, padding=0)

    render_net = RenderNet(
        nets=nets,
        nets_residual=nets_residual,
        out_conv=out_conv,
    )
    # print(ref_net)
    return render_net

def get_render_net_new(in_channels, num_seq, nets_residual, unet_channels):


    nets = nn.ModuleList()
    logging.info(
        f"[NET][RefNet] Seq {num_seq} nets, nets_residual={nets_residual}"  # n_seq:9, nets_residual:True
    )
    for _ in range(num_seq):
        net, in_channels = get_refnet_net_new(
            net_name="unet", 
            params=[5, 2], 
            unet_channels = unet_channels.copy(),
            in_channels=in_channels
        )
        nets.append(net)

    logging.info(f"[NET][RefNet] out_conv({in_channels}, 3)")
    out_conv = nn.Conv2d(in_channels, 3, kernel_size=1, padding=0)

    render_net = RenderNet(
        nets=nets,
        nets_residual=nets_residual,
        out_conv=out_conv,
    )
    # print(ref_net)
    return render_net

def fuse_mv_features_to_tgt(encoded_src_features, **kwargs):
    """

    @param encoded_src_features: [B, V, C, H, W]
    @param kwargs:
    @return:
    """
    B, V, C, H, W = encoded_src_features.shape
    tgt_feature_list = svs_ext_mytorch.map_to_list_bl(  # [sum_of_every_valid_tgt_pix's_valid_src_view_num, enc_channels(16)] in cuda 0
        encoded_src_features,
        kwargs["m2l_src_idx"],
        kwargs["m2l_src_pos"],
        B,
        V,
        H,
        W
    )
    # print(tgt_feature_list.shape) # [tgt_pixel_num, enc_channels(16)]
    # assert 0

    bs = B
    height = H
    width = W

    point_key = kwargs["point_key"] # [N0 + 1, 16]
    pixel_tgt_idx = kwargs["pixel_tgt_idx"] # [N0]


    with torch.no_grad():
        # print("diravg")
        point_tgt_dirs = kwargs["point_tgt_dirs"][pixel_tgt_idx.long()]
        # ic(point_tgt_dirs, point_tgt_dirs.shape)
        point_tgt_dirs = torch_scatter.gather_csr(
            point_tgt_dirs, point_key
        )
        weight = (kwargs["point_src_dirs"] * point_tgt_dirs).sum(dim=1)
        weight = torch.clamp(weight, 0.01, 1)
        weight_sum = torch_scatter.segment_csr(
            weight, point_key, reduce="sum"
        )
        weight /= torch_scatter.gather_csr(weight_sum, point_key)

    tgt_feature_list = weight.view(-1, 1) * tgt_feature_list
    tgt_feature_list = torch_scatter.segment_csr(tgt_feature_list, point_key, reduce="sum")

    # project to target
    tgt_feature_map, mask = svs_ext_mytorch.list_to_map(tgt_feature_list, pixel_tgt_idx, bs, height, width)  # [B, C, H, W]
    # print("x", x.shape) # [1, 16, 576, 992]
    return tgt_feature_map

def fuse_mv_features_to_tgt_depth_testing(depth_mlp_net, encoded_src_features, **kwargs):
    """

    @param encoded_src_features: [B, V, C, H, W]
    @param kwargs:
    @return:
    """
    B, V, C, H, W = encoded_src_features.shape
    tgt_feature_list = svs_ext_mytorch.map_to_list_bl(  
        encoded_src_features,
        kwargs["m2l_src_idx"],
        kwargs["m2l_src_pos"],
        B,
        V,
        H,
        W
    )

    bs = B
    height = H
    width = W

    point_key = kwargs["point_key"] # [N0 + 1, 16]
    pixel_tgt_idx = kwargs["pixel_tgt_idx"] # [N0]

    mlp_input = torch.cat([tgt_feature_list, kwargs["depth_pairs"]], dim=1).unsqueeze(2) # [N, 18, 1]
    # print(mlp_input.shape)

    weights = depth_mlp_net(mlp_input)[1] # [N, 1, 1]
    # print(weights.shape)
    weights = weights.squeeze(2)
    weights += 1e-4

    weights_sum = torch_scatter.segment_csr(weights, point_key, reduce="sum")
    weights_sum = torch_scatter.gather_csr(weights_sum, point_key)
    weights /= weights_sum
    # print(torch_scatter.segment_csr(weights, point_key, reduce="sum"))
    # assert 0
    tgt_feature_list = weights.view(-1, 1) * tgt_feature_list
    tgt_feature_list = torch_scatter.segment_csr(tgt_feature_list, point_key, reduce="sum")


    # project to target
    tgt_feature_map, mask = svs_ext_mytorch.list_to_map(tgt_feature_list, pixel_tgt_idx, bs, height, width)  # [B, C, H, W]
    # print("x", x.shape) # [1, 16, 576, 992]
    return tgt_feature_map

def render_surface(model_net, encoded_src_features, data, query_xyztdir, weight_fuse, embed):
    """

    @param encoded_src_features: [B, V, C, H, W]
    @param kwargs:
    @return:
    """
    V, C, H, W = encoded_src_features.shape
    B = 1
    # [sum_of_every_valid_tgt_pix's_valid_src_view_num, enc_channels(16)] in cuda 0
    # [src_pix_num, 16]
    src_feature = svs_ext_mytorch.map_to_list_bl(
        encoded_src_features.unsqueeze(0),
        data["src_idx"],
        data["src_pos"],
        B, V, H, W
    )

    tgt_pix_num = data["tgt_pos"].shape[0]
    xyzt = torch.cat([data["tgt_pos"], data["timestep"].unsqueeze(0).repeat(tgt_pix_num, 1)], dim=1)  # [tgt_pix_num, 4]
    embedxyzt = model_net.embedxyzt(xyzt)  # [tgt_pix_num, 84]
    embeddir = model_net.embeddir(data["tgt_dirs"])  # [tgt_pix_num, 27]
    # ic(xyzt[:10, :], embedxyzt[:10, :], data["tgt_dirs"][:10, :], embeddir[:10, :])

    xyzt_gather = torch_scatter.gather_csr(xyzt, data["point_key"])
    embedxyzt_gather = torch_scatter.gather_csr(embedxyzt, data["point_key"])  # [src_pix_num, 84]
    embeddir_gather = torch_scatter.gather_csr(embeddir, data["point_key"])  # [src_pix_num, 27]
    
    if weight_fuse:
        if embed:
            weights = model_net.mlp_weight_net(torch.cat([embedxyzt_gather, embeddir_gather, src_feature], dim=1)) # [src_pix_num, 1]
        else:
            weights = model_net.mlp_weight_net(torch.cat([xyzt_gather, src_feature], dim=1)) # [src_pix_num, 1]

        weights_exp = torch.exp(weights) # [src_pix_num, 1]
        weights_sum = torch_scatter.segment_csr(weights_exp, data["point_key"], reduce="sum") # [tgt_pix_num, 1]
        weights = weights_exp / torch_scatter.gather_csr(weights_sum, data["point_key"]) # [src_pix_num, 1]
        # ic(torch_scatter.segment_csr(weights, data["point_key"], reduce="sum"))
        # ic(weights.shape)

        tgt_feature = torch_scatter.segment_csr(weights*src_feature, data["point_key"], reduce="sum") # [tgt_pix_num, 16]
    else:
        with torch.no_grad():
            # print("diravg")
            point_tgt_dirs = data["tgt_dirs"]
            point_tgt_dirs = torch_scatter.gather_csr(point_tgt_dirs, data["point_key"])
            weight = (data["src_dirs"] * point_tgt_dirs).sum(dim=1)
            weight = torch.clamp(weight, 0.01, 1)
            weight_sum = torch_scatter.segment_csr(weight, data["point_key"], reduce="sum")
            weight /= torch_scatter.gather_csr(weight_sum, data["point_key"])

        tgt_feature = weight.view(-1, 1) * src_feature
        tgt_feature = torch_scatter.segment_csr(tgt_feature, data["point_key"], reduce="sum") # [tgt_pix_num, 3]
    # logging.info(src_feature[10000, :])


    if query_xyztdir:
        if embed:
            tgt_rgb = model_net.mlp_render_net(torch.cat([embedxyzt, embeddir, tgt_feature], dim=1)) # [tgt_pix_num, 3]
        else:
            tgt_rgb = model_net.mlp_render_net(torch.cat([xyzt, tgt_feature], dim=1))  # [tgt_pix_num, 3]
    else:
        tgt_rgb = model_net.mlp_render_net(tgt_feature) # [tgt_pix_num, 3]
    # ic(tgt_rgb.shape)
    # assert 0

    tgt_rgb_map, mask = svs_ext_mytorch.list_to_map(tgt_rgb, data["tgt_idx"], B, H, W)  # [B, C, H, W]

    return tgt_rgb_map


def tgt_feature_seq_to_map(tgt_feature_seq, **kwargs):
    """

    @param tgt_feature_seq: [N, 16]
    @param kwargs:
    @return:
    """

    bs, height, width = kwargs["tgt_dm"].shape


    point_key = kwargs["point_key"] # [N0 + 1, 16]
    pixel_tgt_idx = kwargs["pixel_tgt_idx"] # [N0]

    # logging.info(f"tgt feature list {tgt_feature_list.shape}")
    # logging.info(f"point key {point_key.shape}")
    # logging.info(f"pixel tgt idx {pixel_tgt_idx.shape}")
    # assert 0

    # average per point
    if net_config.point_avg_mode == "diravg":
        with torch.no_grad():
            # print("diravg")
            point_tgt_dirs = kwargs["point_tgt_dirs"][pixel_tgt_idx.long()]
            point_tgt_dirs = torch_scatter.gather_csr(
                point_tgt_dirs, point_key
            )
            weight = (kwargs["point_src_dirs"] * point_tgt_dirs).sum(dim=1)
            weight = torch.clamp(weight, 0.01, 1)
            weight_sum = torch_scatter.segment_csr(
                weight, point_key, reduce="sum"
            )
            weight /= torch_scatter.gather_csr(weight_sum, point_key)

        tgt_feature_list = weight.view(-1, 1) * tgt_feature_seq
        tgt_feature_list = torch_scatter.segment_csr(tgt_feature_list, point_key, reduce="sum")
    else:
        raise Exception("invalid avg_mode")

    # print("x", type(x), x.shape)
    # print("pixel_tgt_idx", type(pixel_tgt_idx), pixel_tgt_idx.shape)
    # print(bs, height, width)
    # assert 0

    # project to target
    tgt_feature_map, mask = svs_ext_mytorch.list_to_map(tgt_feature_list, pixel_tgt_idx, bs, height, width)  # [B, C, H, W]
    # print("x", x.shape) # [1, 16, 576, 992]
    return tgt_feature_map

def fuse_mv_features_gatdir(encoded_src_features, graph_attention_net, **kwargs):
    """

        @param encoded_src_features: [B, V, C, H, W]
        @param kwargs:
        @return:
        """
    # Bilinear sample features from src imgs
    B, V, C, H, W = encoded_src_features.shape
    tgt_feature_list = svs_ext_mytorch.map_to_list_bl( # [sum_of_every_valid_tgt_pix's_valid_src_view_num, enc_channels(16)] in cuda 0
        encoded_src_features,
        kwargs["m2l_src_idx"],
        kwargs["m2l_src_pos"],
        B,
        V,
        H,
        W
    )
    # print(tgt_feature_list.shape) # [tgt_pixel_num, enc_channels(16)]
    # assert 0

    bs = B
    height = H
    width = W

    # point_key = kwargs["point_key"]  # [valid_tgt_pix_num + 1] # the first element is 0; for tgt valid pix i, point_key[i+1] - point_key[i] = tgt pix i's valid src views num
    # pixel_tgt_idx = kwargs["pixel_tgt_idx"]  # [valid_tgt_pix_num] all valid tgt img pix idx


    tgt_feature_map, _ = graph_attention_net(None, tgt_feature_list, bs, height, width, **kwargs)



    return tgt_feature_map

def fuse_mv_features_gatimg(encoded_src_features, gatimg_net, **kwargs):
    """

        @param encoded_src_features: [B, V, C, H, W]
        @param kwargs:
        @return:
        """
    # Bilinear sample features from src imgs
    B, V, C, H, W = encoded_src_features.shape
    # [sum_of_every_valid_tgt_pix's_valid_src_view_num, enc_channels(16)] in cuda 0
    tgt_feature_list = svs_ext_mytorch.map_to_list_bl(
        encoded_src_features,
        kwargs["m2l_src_idx"],
        kwargs["m2l_src_pos"],
        B,
        V,
        H,
        W
    )

    bs = B
    height = H
    width = W


    tgt_feature_map = gatimg_net(tgt_feature_list, bs, height, width, **kwargs)

    return tgt_feature_map

def fuse_mv_features_gatif(encoded_src_features, gatif_net, fcodes_net, **kwargs):
    """

    @param encoded_src_features: [B, V, C, H, W]
    @param kwargs:
    @return:
    """
    # Bilinear sample features from src imgs
    B, V, C, H, W = encoded_src_features.shape
    # [sum_of_every_valid_tgt_pix's_valid_src_view_num, enc_channels(16)] in cuda 0
    tgt_feature_list = svs_ext_mytorch.map_to_list_bl(
        encoded_src_features,
        kwargs["m2l_src_idx"],
        kwargs["m2l_src_pos"],
        B,
        V,
        H,
        W
    )
    # a = torch.empty(*tgt_feature_list.shape, dtype=tgt_feature_list.dtype, device="cuda:1")
    # a = tgt_feature_list.numpy()
    # print(a.device)
    # a.cpu()
    # tgt_feature_list.detach().to("cpu")
    # print(tgt_feature_list.shape) # [tgt_pixel_num, enc_channels(16)]
    # assert 0

    bs = B
    height = H
    width = W
    # print(kwargs.keys())
    # print(kwargs["frame_idx"].shape)
    # print(kwargs["frame_idx"])
    # print(type(kwargs["frame_idx"]))
    # print(kwargs["frame_idx"].shape)
    fcode = fcodes_net.get_frame_code(kwargs["frame_idx"])

    # point_key = kwargs["point_key"]  # [valid_tgt_pix_num + 1] # the first element is 0; for tgt valid pix i, point_key[i+1] - point_key[i] = tgt pix i's valid src views num
    # pixel_tgt_idx = kwargs["pixel_tgt_idx"]  # [valid_tgt_pix_num] all valid tgt img pix idx


    # tgt_feature_map = gatimg_net(tgt_feature_list, bs, height, width, **kwargs)
    tgt_feature_map = gatif_net(tgt_feature_list, fcode, bs, height, width, **kwargs)

    return tgt_feature_map

def fuse_mv_features_nerf(enci_src_features, encn_src_features, pile_num, gatind_net, mlpind_net, gatid_net, **data):
    """

    @param enci_src_features: [B, V, C, H, W]
    @param kwargs:
    @return:
    """
    # Bilinear sample features from src imgs
    B, V, C, H, W = enci_src_features.shape
    ic(enci_src_features.shape)
    # # [sum_of_every_valid_tgt_pix's_valid_src_view_num, enc_channels(16)] in cuda 0
    # tgt_featurei_list = svs_ext_mytorch.map_to_list_bl(
    #     enci_src_features,
    #     data["m2l_src_idx"],
    #     data["m2l_src_pos"],
    #     B, V, H, W
    # )
    #
    # tgt_featuren_list = svs_ext_mytorch.map_to_list_bl(
    #     encn_src_features,
    #     data["m2l_src_idx"],
    #     data["m2l_src_pos"],
    #     B, V, H, W
    # )
    #
    # tgt_dirs_list = torch_scatter.gather_csr(data["point_tgt_dirs"], data["point_key"])
    # src_dirs_list = data["point_src_dirs"]
    # tgt_featureind_list = torch.cat([tgt_featurei_list, tgt_featuren_list, tgt_dirs_list, src_dirs_list], dim=1)
    # tgt_att_featureind_list = gatind_net.forward(tgt_featureind_list, data["point_edges"], data["point_key"])
    # tgt_sigma_list = mlpind_net.forward(tgt_att_featureind_list)
    # sigma2alpha = lambda sigma: (1. - torch.exp(sigma))
    # tgt_alpha_list = sigma2alpha(tgt_sigma_list)


    # tgt_featurei_pile_list = []
    # tgt_featuren_pile_list = []
    alpha = torch.zeros((H * W, pile_num + 1), dtype=enci_src_features.dtype).to(data["device"])
    featurei = torch.zeros((H * W, pile_num + 1, C), dtype=enci_src_features.dtype).to(data["device"])
    # sigma2alpha = lambda sigma: (1. - torch.exp(sigma))
    sigma2alpha = lambda sigma: torch.exp(sigma)
    for p in range(pile_num + 1):
        if p == pile_num:
            with torch.no_grad():
                tgt_featurei_list = svs_ext_mytorch.map_to_list_bl(
                    enci_src_features,
                    data["m2l_src_idx"],
                    data["m2l_src_pos"],
                    B, V, H, W
                )

                tgt_featuren_list = svs_ext_mytorch.map_to_list_bl(
                    encn_src_features,
                    data["m2l_src_idx"],
                    data["m2l_src_pos"],
                    B, V, H, W
                )

            tgt_dirs_list = torch_scatter.gather_csr(data["point_tgt_dirs"], data["point_key"])
            src_dirs_list = data["point_src_dirs"]
            tgt_featureind_list = torch.cat([tgt_featurei_list, tgt_featuren_list, tgt_dirs_list, src_dirs_list], dim=1)
            tgt_att_featureind_list = gatind_net.forward(tgt_featureind_list, data["point_edges"], data["point_key"])
            tgt_sigma_list = mlpind_net.forward(tgt_att_featureind_list)
            tgt_alpha_list = sigma2alpha(tgt_sigma_list)
            alpha[data["pixel_tgt_idx"].long(), p // 2:p//2+1] = tgt_alpha_list

            tgt_featureid_list = torch.cat([tgt_featurei_list, tgt_dirs_list, src_dirs_list], dim=1)
            tgt_featurei = gatid_net.forward(tgt_featureid_list, data["point_edges"], data["point_key"]) # [N, C]
            featurei[data["pixel_tgt_idx"].long(), p // 2] = tgt_featurei # [N, C]

        if p < pile_num:
            # ic(data["src_idx_pile"][p].shape)
            # ic(data["src_pos_pile"][p].shape)
            with torch.no_grad():
                tgt_featurei_list = svs_ext_mytorch.map_to_list_bl(
                    enci_src_features,
                    data["src_idx_pile"][p],
                    data["src_pos_pile"][p],
                    B, V, H, W
                )
                # tgt_featurei_pile_list.append(featurei_list)

                tgt_featuren_list = svs_ext_mytorch.map_to_list_bl(
                    encn_src_features,
                    data["src_idx_pile"][p],
                    data["src_pos_pile"][p],
                    B, V, H, W
                )
            # ic("333333333333333333")
            # tgt_featuren_pile_list.append(featuren_list)

            tgt_dirs_list = torch_scatter.gather_csr(data["tgt_dirs_pile"][p], data["point_key_pile"][p])
            src_dirs_list = data["src_dirs_pile"][p]
            tgt_featureind_list = torch.cat([tgt_featurei_list, tgt_featuren_list, tgt_dirs_list, src_dirs_list], dim=1)
            tgt_att_featureind_list = gatind_net.forward(tgt_featureind_list, data["edges_pile"][p],
                                                         data["point_key_pile"][p])
            tgt_sigma_list = mlpind_net.forward(tgt_att_featureind_list)
            tgt_alpha_list = sigma2alpha(tgt_sigma_list)

            tgt_featureid_list = torch.cat([tgt_featurei_list, tgt_dirs_list, src_dirs_list], dim=1)
            tgt_featurei = gatid_net.forward(tgt_featureid_list, data["edges_pile"][p],
                                             data["point_key_pile"][p])  # [N, C]
            if p < pile_num // 2:
                alpha[data["tgt_idx_pile"][p].long(), p:p+1] = tgt_alpha_list
                featurei[data["tgt_idx_pile"][p].long(), p] = tgt_featurei  # [N, C]
            if (p >= pile_num // 2) and (p < pile_num):
                alpha[data["tgt_idx_pile"][p].long(), p+1:p+2] = tgt_alpha_list
                featurei[data["tgt_idx_pile"][p].long(), p+1] = tgt_featurei

    alpha_sum = torch.sum(alpha, dim=1, keepdim=True)
    alpha /= (alpha_sum + 1e-5)
    alpha = alpha.unsqueeze(2).repeat(1, 1, C)



    featurei = torch.sum(featurei * alpha, dim=1) # [H*W, C]

    featurei = featurei[data["pixel_tgt_idx"].long()] # [valid_pix_num, C]
    weights_list = alpha[data["pixel_tgt_idx"].long(), :, 0] # [valid_pix_num, pile+1]
    # ic("222222222222")

    tgt_feature_map, _ = svs_ext_mytorch.list_to_map(featurei, data["pixel_tgt_idx"], B, H, W) # [B, C, H, W]
    weights_map, _ = svs_ext_mytorch.list_to_map(weights_list, data["pixel_tgt_idx"], B, H, W) # [B, pile_num+1, H, W]
    assert weights_map.shape[1] == pile_num+1, "wrong weights num"

    return tgt_feature_map, weights_map



def blend_feature_with_visibility(tgt_encoded_feature, tgt_texture_feature, visibility_map):
    blend_feature = tgt_encoded_feature.clone()

    feature_channels = tgt_encoded_feature.shape[1]
    visibility_map = visibility_map.unsqueeze(1).repeat(1, feature_channels, 1, 1) # [B, 16, H, W]
    visibility_mask = (visibility_map == 0)

    blend_feature[visibility_mask] = tgt_texture_feature[visibility_mask]
    return blend_feature