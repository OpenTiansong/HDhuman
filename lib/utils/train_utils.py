import numpy as np
import logging
from PIL import Image
import torch

import os
import co.utils as utils
from utils.net_utils import *
import random
import time
from icecream import ic
import cv2

import lib.network.net_config as net_config


def format_err_str(errs, div=1):
    err_list = []
    for v in errs.values():
        if isinstance(v, (list, np.ndarray)):
            err_list.extend(v.ravel())
        else:
            err_list.append(v)
    err = sum(err_list)
    if len(err_list) > 1:
        err_str = f"{err / div:0.4f}=" + "+".join(
            [f"{e / div:0.4f}" for e in err_list]
        )
    else:
        err_str = f"{err / div:0.4f}"
    return err_str

def save_img(input_data_dict, output, img_path, total_iteration, phase):
    input_write = {}
    for k, v in input_data_dict.items():
        if k in ["src_ims", "tgt"]:
            input_write[k] = v.cpu()[0] # [V, C, H, W]

    input_syns_img_list = []
    for i in range(input_write["src_ims"].shape[0]):
        img = np.clip(np.transpose(input_write["src_ims"][i].detach().numpy(), (1, 2, 0)), -1, 1) # [H, W, C]
        img = (img * 0.5 + 0.5) * 255.0
        input_syns_img_list.append(img)

    output_img = np.clip(output["out"].cpu()[0].detach().numpy(), -1, 1)
    output_img = np.transpose(output_img, (1, 2, 0))
    output_img = (output_img * 0.5 + 0.5) * 255.0
    input_syns_img_list.append(output_img)
    input_syns_img = np.concatenate(input_syns_img_list, axis=1)

    if phase == "train":
        Image.fromarray(np.uint8(input_syns_img)).save(f"{img_path}/iter_{total_iteration}_train_input_syns.png")
    if phase == "eval":
        Image.fromarray(np.uint8(input_syns_img)).save(f"{img_path}/iter_{total_iteration}_eval_input_syns.png")

    gt_tgt_img = np.clip(input_write["tgt"].cpu().detach().numpy(), -1, 1)
    gt_tgt_img = np.transpose(gt_tgt_img, (1, 2, 0))
    gt_tgt_img = (gt_tgt_img * 0.5 + 0.5) * 255.0

    output_img_cut = output_img[:gt_tgt_img.shape[0], :gt_tgt_img.shape[1], :]

    gt_syn_img_list = [gt_tgt_img, output_img_cut]
    gt_syn_img = np.concatenate(gt_syn_img_list, axis=1)

    if phase == "train":
        Image.fromarray(np.uint8(gt_syn_img)).save(f"{img_path}/iter_{total_iteration}_train_gt_syn.png")
    if phase == "eval":
        Image.fromarray(np.uint8(gt_syn_img)).save(f"{img_path}/iter_{total_iteration}_eval_gt_syn.png")

def save_gt_render_visibility(input_data_dict, output, save_path):
    output_img = np.clip(output["out"].cpu()[0].detach().numpy(), -0.9999, 0.9999)
    output_img = np.transpose(output_img, (1, 2, 0))
    output_img = (output_img * 0.5 + 0.5) * 255.0

    tgt_gt_img = input_data_dict["tgt"].cpu()[0] # [C, H, W]
    tgt_gt_img = np.clip(np.transpose(tgt_gt_img.detach().numpy(), (1, 2, 0)), -0.9999, 0.9999) # [H, W, 3]
    tgt_gt_img = (tgt_gt_img * 0.5 + 0.5) * 255

    visibility_map = input_data_dict["tgt_visibility_map"].cpu()[0].numpy()[..., None] # [H, W, 1]
    visibility_map = visibility_map.repeat(3, axis=2) # [H, W, 3]

    output_gt_img_list = [tgt_gt_img, output_img, visibility_map]
    output_gt_img = np.concatenate(output_gt_img_list, axis=1)

    Image.fromarray(np.uint8(output_gt_img)).save(save_path)

    # output_img = output["out"].cpu()[0]

def save_gt_tex(input_data_dict, output, save_path):
    output_img = np.clip(output["out"].cpu()[0].detach().numpy(), -0.9999, 0.9999)
    output_img = np.transpose(output_img, (1, 2, 0))
    output_img = (output_img * 0.5 + 0.5) * 255.0

    tgt_gt_img = input_data_dict["tgt"].cpu()[0] # [C, H, W]
    tgt_gt_img = np.clip(np.transpose(tgt_gt_img.detach().numpy(), (1, 2, 0)), -0.9999, 0.9999) # [H, W, 3]
    tgt_gt_img = (tgt_gt_img * 0.5 + 0.5) * 255

    output_gt_img_list = [tgt_gt_img, output_img]
    output_gt_img = np.concatenate(output_gt_img_list, axis=1)

    Image.fromarray(np.uint8(output_gt_img)).save(save_path)

def save_gt_tex_encode_blend_vis(input_data_dict, render_tex, render_encode, render_blend, save_path):
    output_img_list = []

    tgt_gt_img = input_data_dict["tgt"].cpu()[0]  # [C, H, W]
    tgt_gt_img = np.clip(np.transpose(tgt_gt_img.detach().numpy(), (1, 2, 0)), -0.9999, 0.9999)  # [H, W, 3]
    tgt_gt_img = (tgt_gt_img * 0.5 + 0.5) * 255

    output_img_list.append(tgt_gt_img)

    for output in [render_tex, render_encode, render_blend]:
        output_img = np.clip(output["out"].cpu()[0].detach().numpy(), -0.9999, 0.9999)
        output_img = np.transpose(output_img, (1, 2, 0))
        output_img = (output_img * 0.5 + 0.5) * 255.0
        output_img_list.append(output_img)

    visibility_map = input_data_dict["tgt_visibility_map"].cpu()[0].numpy()[..., None]  # [H, W, 1]
    visibility_map = visibility_map.repeat(3, axis=2)  # [H, W, 3]
    output_img_list.append(visibility_map)

    output_img_cat = np.concatenate(output_img_list, axis=1)

    Image.fromarray(np.uint8(output_img_cat)).save(save_path)

def save_input_gt_encode_vis(input_data_dict, render_encode, save_path):
    output_img_list = []

    # print(input_data_dict["src_ims"].shape) # [B, V, C, H, W]
    # assert 0
    for i in range(input_data_dict["src_ims"].shape[1]):
        src_img = input_data_dict["src_ims"].cpu()[0][i]
        src_img = np.clip(np.transpose(src_img.detach().numpy(), (1, 2, 0)), -0.9999, 0.9999)  # [H, W, 3]
        src_img = (src_img * 0.5 + 0.5) * 255
        output_img_list.append(src_img)

    tgt_gt_img = input_data_dict["tgt"].cpu()[0]  # [C, H, W]
    tgt_gt_img = np.clip(np.transpose(tgt_gt_img.detach().numpy(), (1, 2, 0)), -0.9999, 0.9999)  # [H, W, 3]
    tgt_gt_img = (tgt_gt_img * 0.5 + 0.5) * 255

    output_img_list.append(tgt_gt_img)

    for output in [render_encode]:
        output_img = np.clip(output["out"].cpu()[0].detach().numpy(), -0.9999, 0.9999)
        output_img = np.transpose(output_img, (1, 2, 0))
        output_img = (output_img * 0.5 + 0.5) * 255.0
        output_img_list.append(output_img)

    visibility_map = input_data_dict["tgt_visibility_map"].cpu()[0].numpy()[..., None]  # [H, W, 1]
    visibility_map = visibility_map.repeat(3, axis=2)  # [H, W, 3]
    output_img_list.append(visibility_map)

    output_img_cat = np.concatenate(output_img_list, axis=1)

    Image.fromarray(np.uint8(output_img_cat)).save(save_path)


def save_input_normal_gt_encode_vis(input_data_dict, render_encode, save_path):
    output_img_list = []

    # print(input_data_dict["src_ims"].shape) # [B, V, C, H, W]
    # assert 0
    for i in range(input_data_dict["src_ims"].shape[1]):
        src_img = input_data_dict["src_ims"].cpu()[0][i]
        src_img = np.clip(np.transpose(src_img.detach().numpy(), (1, 2, 0)), -0.9999, 0.9999)  # [H, W, 3]
        src_img = (src_img * 0.5 + 0.5) * 255

        src_normal = input_data_dict["src_normals"].cpu()[0][i]
        src_normal = np.clip(np.transpose(src_normal.detach().numpy(), (1, 2, 0)), -0.9999, 0.9999)  # [H, W, 3]
        src_normal = (src_normal * 0.5 + 0.5) * 255
        output_img_list.append(src_img)
        output_img_list.append(src_normal)

    tgt_gt_img = input_data_dict["tgt"].cpu()[0]  # [C, H, W]
    tgt_gt_img = np.clip(np.transpose(tgt_gt_img.detach().numpy(), (1, 2, 0)), -0.9999, 0.9999)  # [H, W, 3]
    tgt_gt_img = (tgt_gt_img * 0.5 + 0.5) * 255

    output_img_list.append(tgt_gt_img)

    for output in [render_encode]:
        output_img = np.clip(output["out"].cpu()[0].detach().numpy(), -0.9999, 0.9999)
        output_img = np.transpose(output_img, (1, 2, 0))
        output_img = (output_img * 0.5 + 0.5) * 255.0
        output_img_list.append(output_img)

    visibility_map = input_data_dict["tgt_visibility_map"].cpu()[0].numpy()[..., None]  # [H, W, 1]
    visibility_map = visibility_map.repeat(3, axis=2)  # [H, W, 3]
    output_img_list.append(visibility_map)

    output_img_cat = np.concatenate(output_img_list, axis=1)

    Image.fromarray(np.uint8(output_img_cat)).save(save_path)

def save_weights_map(weights_map, save_path):
    weight_num = weights_map.shape[1]
    weights_list = []
    for w in range(weight_num):
        weight = np.clip(weights_map[0, w, :, :].detach().cpu().numpy(), -0.9999, 0.9999) * 255 # [H, W]
        weights_list.append(weight)
    weights_cat_map = np.concatenate(weights_list, axis=1)
    Image.fromarray(np.uint8(weights_cat_map)).save(save_path)

def fix_output_img_size(output, train_data):
    est = output["out"]
    tgt = train_data["tgt"]

    # fix size
    if est.shape[-1] > tgt.shape[-1]:
        est = est[..., : tgt.shape[-1]]
    # if tgt.shape[-1] > est.shape[-1]:
    #     tgt = tgt[..., : est.shape[-1]]
    if est.shape[-2] > tgt.shape[-2]:
        est = est[..., : tgt.shape[-2], :]
    # if tgt.shape[-2] > est.shape[-2]:
    #     tgt = tgt[..., : est.shape[-2], :]

    return est

# def mask_feature_with_visibility(feature, visibility_map):
#     """
#
#     @param feature: [B, C, H, W]
#     @param visibility_map: [B, H, W]
#     @return:
#     """
#     feature_copy = feature.copy()
#
#     feature_channels = feature.shape[1]
#     visibility_map = visibility_map.unsqueeze(1) # [B, 1, H, W]
#     visibility_map = visibility_map.repeat(1, feature_channels, 1, 1) # [B, C, H, W]
#     # logging.info(f"visibility map {visibility_map.shape}")
#     # assert 0
#
#     visibility_mask = (visibility_map == 0)
#     feature_copy[visibility_mask] = 0
#     return feature_copy

def collate_fn(batch):
    # def collate_cat(batch, k):
    #     # return torch.cat([torch.from_numpy(b[k]) for b in batch])
    #     return torch.from_numpy(batch[0][k])

    # def collate_num(batch, k):
    #     # print(len(batch)) # 1
    #     # print(k, batch[0][k], type(batch[0][k]))
    #     return torch.from_numpy(batch[0][k])

    # def collate_entangle_batch_idx(batch, k):
    #     batch_size = len(batch)
    #     for bidx, b in enumerate(batch):
    #         b[k] = b[k] * batch_size + bidx
    #     return collate_cat(batch, k)

    # def collate_entanlge_height_width(batch, k, height, width):
    #     for bidx, b in enumerate(batch):
    #         b[k] = bidx * height * width + b[k]
    #     return collate_cat(batch, k)

    # def collate_continue(batch, k):
    #     offset = 0
    #     for bidx, b in enumerate(batch):
    #         b[k] = b[k] + offset
    #         offset = b[k][-1] + 1
    #     ret = collate_cat(batch, k)
    #     return ret

    # def collate_prefix(batch, k):
    #     offset = 0
    #     for bidx, b in enumerate(batch):
    #         if bidx > 0:
    #             b[k] = b[k][1:]
    #         b[k] = b[k] + offset
    #         offset = b[k][-1]
    #     ret = collate_cat(batch, k)
    #     return ret

    # def collate_edges(batch, k, key_k):
    #     prefix = [0] + [b[key_k].shape[0] for b in batch]
    #     prefix = np.cumsum(prefix)
    #     # ic(prefix)
    #     # ic(len(batch), len(prefix))
    #     for b, p in zip(batch, prefix):
    #         # ic(k)
    #         # ic(b[k].shape)
    #         b[k] = b[k] + p
    #         # ic(b[k].shape)
    #     ret = collate_cat(batch, k)
    #     # assert 0
    #     return ret.transpose(1, 0)

    def collate_list(batch, k):
        ret = []
        if k.startswith("edges_"):
            for edge in batch[0][k]:
                ret.append(torch.from_numpy(edge).transpose(1, 0))#.to(batch[0]["device"]))
            return ret
        else:
            for data in batch[0][k]:
                ret.append(torch.from_numpy(data))#.to(batch[0]["device"]))
            return ret

    ret = {}
    keys = list(batch[0].keys())
    for k in keys:
        if k in ["tgt", "src_ims", "src_masks", "src_ind", "tgt_dm", "tgt_ma", "src_dms", "src_Ks", "src_Rs", "src_ts",
                 "tgt_uv_map", "tgt_visibility_map", "src_normals", "src_extrinsics_w2c", "tgt_depth_hd",
                 "tgt_border"]:
            if type(batch[0][k]) is np.ndarray:
                ret[k] = torch.stack([torch.from_numpy(b[k]) for b in batch])

            # ret[k] = torch.stack([torch.from_numpy(batch[0][k])], dim=0)#.to(batch[0]["device"]) # add batch dim
        # TODO: does only work for batch size = 1
        elif k in ["src_imgs512", "src_masks512", "src_intrinsics512", "src_extrinsics_c2w",
                    "tgt_intrinsic", "tgt_intrinsic512", "tgt_extrinsic_c2w", "tgt_img512",
                   "far_plane", "near_plane", "human_center", "human_length",
                   "src_imgs", "src_depths", "tgt_img", "tgt_depth"]:
            if type(batch[0][k]) is np.ndarray:
                ret[k] = torch.stack([torch.from_numpy(b[k]) for b in batch], dim=0)
            # if batch[0][k] != None:
            #     ret[k] = torch.stack([torch.from_numpy(b[k]) for b in batch], dim=0)
        elif k in [
            "point_src_edge_bins",
            "point_src_edge_weights",
            "point_tgt_edge_bins",
            "point_tgt_edge_weights",
        ]:
            raise Exception(f"invalid k (={k}) in batch collate")
            # ret[k] = torch.stack([torch.from_numpy(b[k]) for b in batch])
        elif k in [
            "m2l_src_pos",
            "src_pos", "tgt_pos",
            "depth_pairs",
            "tgt_dirs",
            "point_dirs",
            "point_src_dirs", "src_dirs",
            "point_tgt_dirs",
            "pixel_nb_dists",
            "pixel_nb_weights",
            "tgt_px"
        ]:
            # ret[k] = collate_cat(batch, k)
            if type(batch[0][k]) is np.ndarray:
                ret[k] = torch.from_numpy(batch[0][k])#.to(batch[0]["device"])
        elif k in ["m2l_src_idx", "src_idx"]:
            # ret[k] = collate_entangle_batch_idx(batch, k)
            # ic(type(batch[0]["src_idx"]))

            if type(batch[0][k]) is np.ndarray:
                ret[k] = torch.from_numpy(batch[0][k])#.to(batch[0]["device"])
        elif k in ["pixel_tgt_idx", "tgt_idx"]:
            # height, width = batch[0]["tgt"].shape[-2:]
            # ret[k] = collate_entanlge_height_width(batch, k, height, width)
            if type(batch[0][k]) is np.ndarray:
                # ic(k)
                ret[k] = torch.from_numpy(batch[0][k])#.to(batch[0]["device"])
        # elif k in [
        #     "pixel_nb_key",
        #     # "m2l_tgt_idx",  # TODO: m2l_tgt_idx wrong
        # ]:
        #     ret[k] = collate_continue(batch, k)
        elif k in [
            "point_key",
            # "pixel_tgt_key",
            # "m2l_prefix",  # m2l_prefix is wrong, should be added
        ]:
            # ret[k] = collate_prefix(batch, k)
            if type(batch[0][k]) is np.ndarray:
                ret[k] = torch.from_numpy(batch[0][k])#.to(batch[0]["device"])
        # TODO: does only work for batch size = 1
        elif k in ["point_edges",
                   # 'point_tgt_edges',
                   # 'point_src_edges'
                   ]:
            # ret[k] = collate_edges(batch, k, "point_key")
            if type(batch[0][k]) is np.ndarray:
                ret[k] = torch.from_numpy(batch[0][k]).transpose(1, 0)#.to(batch[0]["device"])
        elif k in ["frame_idx", "frame", "timestep"]:
            # ret[k] = collate_num(batch, k)
            if type(batch[0][k]) is np.ndarray:
                ret[k] = torch.from_numpy(batch[0][k])#.to(batch[0]["device"])
        elif k in ["name", "src_gray_maps", "device", "avg_src", "start_x", "start_y"]:
            ret[k] = batch[0][k]
        elif k.endswith("_pile"):
            # if k == "edges_pile":
            ret[k] = collate_list(batch, k)
            # pass
        else:
            raise Exception(f"invalid k (={k}) in batch collate")
    return ret


def set_manual_seed():
    ## Set manual seed
    np.random.seed(int(time.time()))
    random.seed(int(time.time()))
    torch.manual_seed(int(time.time()))
    torch.cuda.manual_seed(int(time.time()))

def get_encode_render_net(enc_net, ref_net):
    ## Net
    # enc_net = "resunet3.16"
    encode_net = get_encode_net(enc_net)

    encoded_feature_channels = int(enc_net.split(".")[1])
    render_net = get_render_net(ref_net, encoded_feature_channels)
    return encode_net, render_net

# def load_pre_encode_render_model(encode_net, render_net, pre_encode_model_path, pre_render_model_path, train_device):
#     if os.path.exists(pre_encode_model_path):
#         # logging.info("Error: Not pre model!")
#         # assert 0
#         logging.info(f"Load pre encode model from {pre_encode_model_path}")
#         pre_encode_state_dict = torch.load(pre_encode_model_path, map_location=train_device)
#
#         # print("pre_encode_state_dict")
#         # print(pre_encode_state_dict.keys())
#         # print("encode net state dict")
#         # print(encode_net.state_dict().keys())
#
#         state_dict_encode = {}
#         for key in encode_net.state_dict().keys():
#             if key in pre_encode_state_dict.keys():
#                 if encode_net.state_dict()[key].shape == pre_encode_state_dict[key].shape:
#                     state_dict_encode[key] = pre_encode_state_dict[key]
#                     # logging.info(f"Load {key}")
#                 else:
#                     logging.info(f"Eocode Net: Not load {key} | Shape mismatch!")
#             else:
#                 logging.info(f"Eocode Net: Not load {key} | Not in pre model!")
#
#         encode_net.load_state_dict(state_dict_encode, strict=False)
#     else:
#         logging.info("Not load pre encode model!")
#     encode_net = encode_net.to(train_device)
#
#     if os.path.exists(pre_render_model_path):
#         logging.info(f"Load pre render model from {pre_render_model_path}")
#         pre_render_state_dict = torch.load(pre_render_model_path, map_location=train_device)
#
#         # print("pre_render_state_dict")
#         # print(pre_render_state_dict.keys())
#         # print("render net state dict")
#         # print(render_net.state_dict().keys())
#         # assert 0
#
#         state_dict_render = {}
#         for key in render_net.state_dict().keys():
#             if key in pre_render_state_dict.keys():
#                 if render_net.state_dict()[key].shape == pre_render_state_dict[key].shape:
#                     state_dict_render[key] = pre_render_state_dict[key]
#                     # logging.info(f"Load {key}")
#                 else:
#                     logging.info(f"Render Net: Not load {key} | Shape mismatch!")
#             else:
#                 logging.info(f"Render Net: Not load {key} | Not in pre model!")
#
#         render_net.load_state_dict(state_dict_render, strict=False)
#     else:
#         logging.info("Not load pre render model!")
#     render_net = render_net.to(train_device)
#     return encode_net, render_net

def load_pretrained_encode_model(encode_net, pretrained_encode_model_path, train_device):
    if os.path.exists(pretrained_encode_model_path):
        # logging.info("Error: Not pre model!")
        # assert 0
        logging.info(f"Load pre encode model from {pretrained_encode_model_path}")
        pre_encode_state_dict = torch.load(pretrained_encode_model_path, map_location=train_device)

        # print("pre_encode_state_dict")
        # print(pre_encode_state_dict.keys())
        # print("encode net state dict")
        # print(encode_net.state_dict().keys())

        state_dict_encode = {}
        for key in encode_net.state_dict().keys():
            if key in pre_encode_state_dict.keys():
                if encode_net.state_dict()[key].shape == pre_encode_state_dict[key].shape:
                    state_dict_encode[key] = pre_encode_state_dict[key]
                    # logging.info(f"Load {key}")
                else:
                    logging.info(f"Eocode Net: Not load {key} | Shape mismatch!")
            else:
                logging.info(f"Eocode Net: Not load {key} | Not in pre model!")

        encode_net.load_state_dict(state_dict_encode, strict=False)
    else:
        logging.info("Not load pre encode model!")
    encode_net = encode_net.to(train_device)

    return encode_net

def load_pretrained_render_model(render_net, pretrained_render_model_path, train_device):
    if os.path.exists(pretrained_render_model_path):
        logging.info(f"Load pre render model from {pretrained_render_model_path}")
        pre_render_state_dict = torch.load(pretrained_render_model_path, map_location=train_device)

        # print("pre_render_state_dict")
        # print(pre_render_state_dict.keys())
        # print("render net state dict")
        # print(render_net.state_dict().keys())
        # assert 0

        state_dict_render = {}
        for key in render_net.state_dict().keys():
            if key in pre_render_state_dict.keys():
                if render_net.state_dict()[key].shape == pre_render_state_dict[key].shape:
                    state_dict_render[key] = pre_render_state_dict[key]
                    # logging.info(f"Load {key}")
                else:
                    logging.info(f"Render Net: Not load {key} | Shape mismatch!")
            else:
                logging.info(f"Render Net: Not load {key} | Not in pre model!")

        render_net.load_state_dict(state_dict_render, strict=False)
    else:
        logging.info("Not load pre render model!")
    render_net = render_net.to(train_device)
    return render_net

def load_pretrained_tex_model(texture_net, pretrained_tex_model_path, device):
    if os.path.exists(pretrained_tex_model_path):
        logging.info(f"Load pre texture model from {pretrained_tex_model_path}")
        pre_texture_state_dict = torch.load(pretrained_tex_model_path, map_location=device)

        # print("pre_render_state_dict")
        # print(pre_render_state_dict.keys())
        # print("render net state dict")
        # print(render_net.state_dict().keys())
        # assert 0

        state_dict_texture = {}
        for key in texture_net.state_dict().keys():
            if key in pre_texture_state_dict.keys():
                if texture_net.state_dict()[key].shape == pre_texture_state_dict[key].shape:
                    state_dict_texture[key] = pre_texture_state_dict[key]
                    # logging.info(f"Load {key}")
                else:
                    logging.info(f"Texture Net: Not load {key} | Shape mismatch!")
            else:
                logging.info(f"Texture Net: Not load {key} | Not in pre model!")

        texture_net.load_state_dict(state_dict_texture, strict=False)
    else:
        logging.info("Not load pre texture model!")
    texture_net = texture_net.to(net_config.train_device)
    return texture_net

# def load_encode_render_optimizer(encode_net, render_net, encode_optim_path, render_optim_path, learning_rate,
#                                  train_device):
#     encode_optimizer = torch.optim.Adam(encode_net.parameters(), learning_rate)
#     if os.path.exists(encode_optim_path):
#         logging.info(f"Load pre encode optim from {encode_optim_path}")
#         encode_optimizer.load_state_dict(
#             torch.load(encode_optim_path, map_location=train_device))
#     else:
#         logging.info(f"Not load encode optim!")
#     encode_optimizer.zero_grad()
#
#     render_optimizer = torch.optim.Adam(render_net.parameters(), learning_rate)
#     if os.path.exists(render_optim_path):
#         logging.info(f"Load pre render optim from {render_optim_path}")
#         render_optimizer.load_state_dict(
#             torch.load(render_optim_path, map_location=train_device))
#     else:
#         logging.info(f"Not load render optim")
#     render_optimizer.zero_grad()
#     return encode_optimizer, render_optimizer

def get_encode_optimizer(encode_net, encode_optim_path, learning_rate, train_device):
    encode_optimizer = torch.optim.Adam(encode_net.parameters(), learning_rate)
    if os.path.exists(encode_optim_path):
        logging.info(f"Load pre encode optim from {encode_optim_path}")
        encode_optimizer.load_state_dict(
            torch.load(encode_optim_path, map_location=train_device))
    else:
        logging.info(f"Not load encode optim!")
    encode_optimizer.zero_grad()
    return encode_optimizer

def get_render_optimizer(render_net, render_optim_path, learning_rate, train_device):
    render_optimizer = torch.optim.Adam(render_net.parameters(), learning_rate)
    if os.path.exists(render_optim_path):
        logging.info(f"Load pre render optim from {render_optim_path}")
        render_optimizer.load_state_dict(
            torch.load(render_optim_path, map_location=train_device))
    else:
        logging.info(f"Not load render optim")
    render_optimizer.zero_grad()
    return render_optimizer

def get_tex_optimizer(texture_net, tex_optim_path, learning_rate, device):
    texture_optimizer = torch.optim.Adam(texture_net.parameters(), learning_rate)
    if os.path.exists(tex_optim_path):
        logging.info(f"Load pre texture optim from {tex_optim_path}")
        texture_optimizer.load_state_dict(torch.load(tex_optim_path, map_location=device))
    else:
        logging.info(f"Not load texture optim")
    texture_optimizer.zero_grad()
    return texture_optimizer

def save_src_tgt_render_rgb_vis(data, render_rgb, save_path):
    save_imgs = []
    V = data["src_ims"][0].shape[0]
    for i in range(V):
        src_img = data["src_ims"][0][i].permute(1, 2, 0).cpu().numpy() # [H, W, C]
        src_img = np.clip(src_img, -0.9999, 0.9999)
        src_img = (src_img + 1.) / 2 * 255
        save_imgs.append(src_img)
    tgt_img = data["tgt"][0].permute(1, 2, 0).cpu().numpy()
    tgt_img = np.clip(tgt_img, -0.9999, 0.9999)
    tgt_img = (tgt_img + 1) / 2 * 255
    save_imgs.append(tgt_img)

    render_rgb = render_rgb.cpu().numpy()
    render_rgb = np.clip(render_rgb, -0.9999, 0.9999)
    render_rgb = (render_rgb + 1) / 2 * 255
    save_imgs.append(render_rgb)

    visibility_map = data["tgt_visibility_map"].cpu()[0].numpy()[..., None]  # [H, W, 1]
    visibility_map = visibility_map.repeat(3, axis=2)  # [H, W, 3]
    save_imgs.append(visibility_map)

    save_imgs = np.concatenate(save_imgs, axis=1)
    Image.fromarray(np.uint8(save_imgs)).save(save_path)

def save_src_gt_render_vis(data, render_img, save_path):
    save_imgs = []
    V = data["src_imgs"][0].shape[0]
    for i in range(V):
        src_img = data["src_imgs"][0][i].permute(1, 2, 0).cpu().numpy()  # [H, W, C]
        src_img = np.clip(src_img, -0.9999, 0.9999)
        src_img = (src_img + 1.) / 2 * 255
        save_imgs.append(src_img)
    tgt_img = data["tgt_img"][0].permute(1, 2, 0).cpu().numpy()
    tgt_img = np.clip(tgt_img, -0.9999, 0.9999)
    tgt_img = (tgt_img + 1) / 2 * 255
    save_imgs.append(tgt_img)
    # ic(tgt_img.shape)

    render_rgb = render_img[0].cpu().detach().permute(1, 2, 0).numpy()
    render_rgb = np.clip(render_rgb, -0.9999, 0.9999)
    render_rgb = (render_rgb + 1) / 2 * 255
    save_imgs.append(render_rgb)
    # ic(render_rgb.shape)

    visibility_map = data["tgt_visibility_map"].cpu()[0].numpy()[..., None]  # [H, W, 1]
    visibility_map = visibility_map.repeat(3, axis=2)  # [H, W, 3]
    save_imgs.append(visibility_map)
    # ic(visibility_map.shape)

    save_imgs = np.concatenate(save_imgs, axis=1)
    Image.fromarray(np.uint8(save_imgs)).save(save_path)

def save_src_gt_render_conf_vis(data, render_img, confidence_map, save_path):
    save_imgs = []
    V = data["src_imgs"][0].shape[0]
    for i in range(V):
        src_img = data["src_imgs"][0][i].permute(1, 2, 0).cpu().numpy()  # [H, W, C]
        src_img = np.clip(src_img, -0.9999, 0.9999)
        src_img = (src_img + 1.) / 2 * 255
        save_imgs.append(src_img)
    tgt_img = data["tgt_img"][0].permute(1, 2, 0).cpu().numpy()
    tgt_img = np.clip(tgt_img, -0.9999, 0.9999)
    tgt_img = (tgt_img + 1) / 2 * 255
    save_imgs.append(tgt_img)
    # ic(tgt_img.shape)

    render_rgb = render_img[0].cpu().detach().permute(1, 2, 0).numpy()
    render_rgb = np.clip(render_rgb, -0.9999, 0.9999)
    render_rgb = (render_rgb + 1) / 2 * 255
    save_imgs.append(render_rgb)
    # ic(render_rgb.shape)

    # ic(confidence_map.shape)
    confidence_map = confidence_map[0].cpu().detach().permute(1, 2, 0).numpy().repeat(3, axis=2)

    confidence_map *= 255
    save_imgs.append(confidence_map)

    visibility_map = data["tgt_visibility_map"].cpu()[0].numpy()[..., None]  # [H, W, 1]
    visibility_map = visibility_map.repeat(3, axis=2)  # [H, W, 3]
    save_imgs.append(visibility_map)
    # ic(visibility_map.shape)

    save_imgs = np.concatenate(save_imgs, axis=1)
    Image.fromarray(np.uint8(save_imgs)).save(save_path)

def save_src_tgt_render_rgb(data, render_rgb, save_path):
    save_imgs = []
    V = data["src_ims"][0].shape[0]
    for i in range(V):
        src_img = data["src_ims"][0][i].permute(1, 2, 0).cpu().numpy() # [H, W, C]
        src_img = np.clip(src_img, -0.9999, 0.9999)
        src_img = (src_img + 1.) / 2 * 255
        save_imgs.append(src_img)
    tgt_img = data["tgt"][0].permute(1, 2, 0).cpu().numpy()
    tgt_img = np.clip(tgt_img, -0.9999, 0.9999)
    tgt_img = (tgt_img + 1) / 2 * 255
    save_imgs.append(tgt_img)

    render_rgb = render_rgb.cpu().numpy()
    render_rgb = np.clip(render_rgb, -0.9999, 0.9999)
    render_rgb = (render_rgb + 1) / 2 * 255
    save_imgs.append(render_rgb)

    save_imgs = np.concatenate(save_imgs, axis=1)
    Image.fromarray(np.uint8(save_imgs)).save(save_path)

def save_tgt_render_rgb(data, render_rgb, save_path):
    save_imgs = []
    tgt_img = data["tgt"][0].permute(1, 2, 0).cpu().numpy()
    tgt_img = np.clip(tgt_img, -0.9999, 0.9999)
    tgt_img = (tgt_img + 1) / 2 * 255
    save_imgs.append(tgt_img)

    render_rgb = render_rgb.cpu().numpy()
    render_rgb = np.clip(render_rgb, -0.9999, 0.9999)
    render_rgb = (render_rgb + 1) / 2 * 255
    save_imgs.append(render_rgb)

    save_imgs = np.concatenate(save_imgs, axis=1)
    Image.fromarray(np.uint8(save_imgs)).save(save_path)

def save_depth_render_depth(data, render_depth, save_path):
    save_depths = np.concatenate([data["tgt_dm"][0].cpu().numpy(), render_depth.cpu().numpy()], axis=1)
    save_depths *= 180
    Image.fromarray(np.uint8(save_depths)).save(save_path)

def save_render_depth(render_depth, save_path):
    save_depths = render_depth.cpu().numpy()
    save_depths *= 180
    save_depths = cv2.applyColorMap(np.uint8(save_depths), cv2.COLORMAP_JET)
    Image.fromarray(save_depths).save(save_path)

def get_normals(data, normal_net):
    with torch.no_grad():
        # logging.info(train_data["src_imgs512"].dtype)
        normal_net_deivce = next(normal_net.module.parameters()).device
        normals = normal_net(data["src_imgs512"].squeeze(0).to(normal_net_deivce)).to(data["src_masks512"].device)
        normals = normals * data["src_masks512"].squeeze(0)  # [V, C, H, W]
        data["src_normals512"] = normals.unsqueeze(0)  # [1, V, C, H, W]
        # [1, V, C(6), H, W]
        data["src_imgnormals512"] = torch.cat([data["src_imgs512"], data["src_normals512"]],
                                                   dim=2)


# def get_ZZR_img_params(frames_list, views_list, )