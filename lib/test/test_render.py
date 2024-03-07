




from typing import List, Tuple, Dict, Union, Optional

import numpy as np
import time

import numpy as np
import cv2 as cv
from PIL import Image

from icecream import ic

import taichi as ti
import torch

import lib.taichi_three as t3

import lib.utils.eval_utils as eval_utils
from lib.network.encode_net import EncodeNet
from lib.network.render_net import RenderNet
from lib.utils.img_utils import crop_img, pad_img, pad_depth, uncrop_rgb_img
from lib.utils.net_utils import tgt_feature_seq_to_map, fuse_mv_features_to_tgt

#should import path of StableViewSynthesis in calling script
import ext

def call_map_source_points_version_0(tgt_depth, tgt_K, tgt_R, tgt_t, src_dms, src_gray_maps, src_Ks, src_Rs, src_ts, n_src_views, bwd_thresh):
    (
        src_pos,  # [173394, 2] 343963    in tgt pixel order
        src_idx,  # [173394] in tgt pixel order
        src_dirs,  # [173394, 3] in tgt pixel order
        tgt_dirs,  # [1048576, 3] 1024*1024
        point_key,  # [160747]
        tgt_idx  # [160746]
    ) = ext.preprocess.map_source_points(
        tgt_depth,
        tgt_K,
        tgt_R,
        tgt_t,
        src_dms,
        src_gray_maps,
        src_Ks,
        src_Rs,
        src_ts,
        src_gray_maps.shape[0],
        bwd_depth_thresh=bwd_thresh,
        n_max_sources=n_src_views,
        rank_mode="pointdir"
    )
    return src_pos, src_idx, src_dirs, tgt_dirs, point_key, tgt_idx
    
def call_map_source_points_version_1(tgt_depth, tgt_K, tgt_R, tgt_t, src_dms, src_gray_maps, src_Ks, src_Rs, src_ts, n_src_views, bwd_thresh):
    (
        src_pos,  # [173394, 2] 343963    in tgt pixel order
        src_idx,  # [173394] in tgt pixel order
        src_dirs,  # [173394, 3] in tgt pixel order
        tgt_dirs,  # [1048576, 3] 1024*1024
        point_key,  # [160747]
        tgt_idx  # [160746]
    ) = ext.preprocess.map_source_points(
        tgt_dm=tgt_depth,
        tgt_K=tgt_K,
        tgt_R=tgt_R,
        tgt_t=tgt_t,
        tgt_count=np.array(list(range(0, src_dms.shape[0])), dtype=np.int32),
        src_dms=src_dms,
        src_Ks=src_Ks,
        src_Rs=src_Rs,
        src_ts=src_ts,
        bwd_depth_thresh=bwd_thresh,
        n_max_sources=n_src_views,
        rank_mode="pointdir"
    )
    return src_pos, src_idx, src_dirs, tgt_dirs, point_key, tgt_idx

def render_one_subject(
        encode_net:EncodeNet, 
        render_net:RenderNet, 
        src_cameras:List[Dict[str, np.ndarray]],
        render_cameras:List[Dict[str, np.ndarray]],
        mesh:Dict[str, np.ndarray], 
        src_img_list,
        ori_width:int, 
        ori_height:int,
        scale:float, 
        device:torch.device,
        flag_map_to_list_bl_seq:bool=True, 
        flag_map_to_list_bl:bool=False, 
        list_path_save:Optional[List[str]]=None
    ):
    '''
    @param encode_net: EncodeNet
    @param render_net: RenderNet
    @param src_cameras: list of dict with keys "intrinsic" and "extrinsic",
        shapes are [3, 3] and [3, 4] respectively
    @param render_cameras: list of dict with keys "intrinsic" and "extrinsic",
        shapes are [3, 3] and [3, 4] respectively
    @param mesh: dict with keys "v", "vn", "f", load by t3.readobj
    @param src_img_list: list of np.ndarray with shape [H, W, 3], dtype np.uint8
    @param ori_width: int
    @param ori_height: int
    @param scale: float
    @param device: torch.device
    @param flag_map_to_list_bl_seq: bool
    @param flag_map_to_list_bl: bool
    @param list_path_save: list of str
        if list_path_save is None, return list of Image
        if not None, save in these paths and return None
    '''

    assert isinstance(encode_net, EncodeNet), type(encode_net)
    assert isinstance(render_net, RenderNet), type(render_net)
    
    scale_width, scale_height = int(ori_width * scale), int(ori_height * scale)
    render_res = max(scale_width, scale_height)
    scale_x = float(scale_width) / float(ori_width)
    scale_y = float(scale_height) / float(ori_height)
    res_crop = (int(scale_width / 2), scale_height)

    ## Setup Taichi
    ti.init(ti.gpu)
    # obj = t3.readobj(str(mesh_path), scale=1)
    model = t3.Model(obj=mesh)
    scene = t3.Scene()
    scene.add_model(model)
    camera_ori = t3.Camera(res=(render_res, render_res))
    camera_crop = t3.Camera(res=res_crop)
    scene.add_camera(camera_crop)
    scene.add_camera(camera_ori)
    light = t3.Light([0, 0, 1])
    scene.add_light(light)
    # print("Setup taichi finished!")

    # Encode ref views
    enc_feat_list = []
    src_Ks_list = []
    src_Rs_list = []
    src_ts_list = []
    depth_list = []
    gray_map_list = []
    ref_view_num = len(src_img_list)

    #return 
    list_image_render_uncrop = []

    for view in range(ref_view_num):
        print(f"Encoded ref view {view}")
        img = src_img_list[view]
        if scale < 1:
            img = cv.resize(img, (scale_width, scale_height), interpolation=cv.INTER_AREA)

        min_x, min_y, max_x, max_y, center_x, center_y = eval_utils.find_img_border(img)
        start_x = int(center_x - res_crop[0] / 2)
        start_y = int(center_y - res_crop[1] / 2)
        img = crop_img(img, start_x, start_y, res_crop[0], res_crop[1])

        gray_map = cv.cvtColor(img, cv.COLOR_RGB2GRAY).astype(np.float32)
        gray_map = pad_img(gray_map)
        img = (img.astype(np.float32) / 255) * 2 - 1
        img = img.transpose(2, 0, 1) # [C, H, W]
        # img = pad_img(img)[None][None] # [B, 1, C, H, W]
        img = pad_img(img)[None] # [B, C, H, W]
        img = torch.from_numpy(img).to(device)

        with torch.no_grad():
            enc_feat = encode_net.forward_image(img) # [1, 1, 16, 1024, 1024]

        enc_feat_list.append(enc_feat)

        intrinsic = src_cameras[view]["intrinsic"].copy()
        extrinsic = src_cameras[view]["extrinsic"].copy()

        intrinsic[0, :] *= scale_x
        intrinsic[1, :] *= scale_y

        intrinsic[0, 2] -= start_x
        intrinsic[1, 2] -= start_y

        # Render depth
        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        trans_py = np.transpose(extrinsic[:,0:3])
        pos_py = -trans_py @ extrinsic[:,3]
        camera_crop.set_intrinsic(fx, fy, cx, cy)
        camera_crop.set_extrinsic(trans_py, pos_py)
        camera_crop._init()
        scene.render()
        disparity = camera_crop.zbuf.to_numpy().swapaxes(0, 1)  # [::-1, :]
        mask = (disparity != 0)
        depth = 1 / disparity
        depth[~mask] = 0
        # print(depth.shape) # [1024, 1024]
        # assert 0
        depth = pad_depth(depth)

        depth_list.append(depth)
        src_Ks_list.append(intrinsic)
        src_Rs_list.append(extrinsic[:,0:3])
        src_ts_list.append(extrinsic[:,3])
        gray_map_list.append(gray_map)
    # assert 0

    feature_channels = enc_feat_list[0].shape[-3] # 16
    src_Ks = np.stack(src_Ks_list, axis=0) # [V, 3, 3]
    src_Rs = np.stack(src_Rs_list, axis=0) # [V, 3]
    src_ts = np.stack(src_ts_list, axis=0) # [V, 3]
    src_dms = np.stack(depth_list, axis=0) # [V, C, H]
    # print(f"src dms {src_dms.shape}")
    src_gray_maps = np.stack(gray_map_list, axis=0) # [V, C, H]
    print("Encoding ref views finished!")

    # Prepare tgt data
    # free_view_camera_path = os.path.join(data_root, "free_view_camera")

    for count_target_view, camera in enumerate(render_cameras):
        time_start = time.time()

        ret = {}

        intrinsic = camera["intrinsic"].copy()
        extrinsic = camera["extrinsic"].copy()

        intrinsic[0, :] *= scale_x
        intrinsic[1, :] *= scale_y

        tgt_K = intrinsic
        tgt_extrinsic = extrinsic
        tgt_R = tgt_extrinsic[:, 0:3]
        tgt_t = tgt_extrinsic[:, 3]

        ## Render target view depth
        fx, fy, cx, cy = tgt_K[0, 0], tgt_K[1, 1], tgt_K[0, 2], tgt_K[1, 2]
        trans_py = np.transpose(tgt_R)
        pos_py = -trans_py @ tgt_t
        camera_ori.set_intrinsic(fx, fy, cx, cy)
        camera_ori.set_extrinsic(trans_py, pos_py)
        camera_ori._init()
        scene.render()
        tgt_disparity = camera_ori.zbuf.to_numpy().swapaxes(0, 1)  # [::-1, :]
        # print("tgt dis", tgt_disparity.shape)
        mask = (tgt_disparity != 0)
        tgt_depth = 1 / tgt_disparity
        tgt_depth[~mask] = 0
        # tgt_depth_tmp = cv2.resize(mask.astype(np.uint8)*255, (1024, 768), interpolation=cv2.INTER_NEAREST)
        # cv2.imwrite("./test/0.png", tgt_depth)
        # cv2.imshow("", tgt_depth_tmp)
        # cv2.waitKey(0)
        # print(f"Render costs {(time.time() - time_start):.2f}s")

        ## Crop tgt depth and update tgt intrinsic
        min_x, min_y, max_x, max_y, center_x, center_y = eval_utils.find_img_border(tgt_depth)
        # print("min_x", min_x, "max_x", max_x, "center_x", center_x)
        # assert 0
        start_x = int(center_x - res_crop[0] / 2)
        start_y = int(center_y - res_crop[1] / 2)
        tgt_depth = crop_img(tgt_depth, start_x, start_y, res_crop[0], res_crop[1])
        tgt_depth = pad_depth(tgt_depth)
        # cv2.imshow("", tgt_depth)
        # cv2.waitKey(0)
        tgt_K[0, 2] -= start_x
        tgt_K[1, 2] -= start_y
        ret["tgt_dm"] = tgt_depth # [H, W]

        ## Projection & Back projection on target view
        bwd_thresh = 0.01

        n_src_views = src_gray_maps.shape[0]
        (
            src_pos,  # [173394, 2] 343963    in tgt pixel order
            src_idx,  # [173394] in tgt pixel order
            src_dirs,  # [173394, 3] in tgt pixel order
            tgt_dirs,  # [1048576, 3] 1024*1024
            point_key,  # [160747]
            tgt_idx  # [160746]
        ) = ext.preprocess.map_source_points(
            tgt_dm=tgt_depth,
            tgt_K=tgt_K,
            tgt_R=tgt_R,
            tgt_t=tgt_t,
            tgt_count=np.array(list(range(0, src_dms.shape[0])), dtype=np.int32),
            src_dms=src_dms,
            src_Ks=src_Ks,
            src_Rs=src_Rs,
            src_ts=src_ts,
            bwd_depth_thresh=bwd_thresh,
            n_max_sources=n_src_views,
            rank_mode="pointdir"
        )
        #print(src_pos.shape[0] / tgt_idx.shape[0])
        tgt_visibility_map = eval_utils.get_visibility_map(tgt_depth, tgt_idx) # [H, W]

        point_key = point_key.astype(np.int64)
        ret["point_key"] = point_key
        ret["pixel_tgt_idx"] = tgt_idx
        ret["point_src_dirs"] = src_dirs
        ret["point_tgt_dirs"] = tgt_dirs

        ret["m2l_src_idx"] = src_idx
        ret["m2l_src_pos"] = src_pos
        ret["tgt_visibility_map"] = tgt_visibility_map # [H, W]


        for key in ret.keys():
            if not torch.is_tensor(ret[key]):
                ret[key] = torch.from_numpy(ret[key])
            if key in ["tgt_visibility_map", "tgt_dm", "tgt_uv_map"]:
                ret[key] = ret[key].unsqueeze(0).to(device) # to get mask of tgt view
            if key.startswith(("point_", "pixel_", "m2l_")):
                ret[key] = ret[key].to(device)
        # print("no zero pix num", get_pixel_num(tgt_depth))
        # print("tgt_idx", tgt_idx.shape)
        # print("src_pos", src_pos.shape)

        # print("src_idx", src_idx.shape)
        # print("src_dirs", src_dirs.shape)
        # print("tgt_dirs", tgt_dirs.shape)
        # print("point_key", point_key.shape)

        ## sample src encoded features to get tgt feature  [N, 16] in tgt pixel order
        if flag_map_to_list_bl_seq: # False
            (
                m2l_prefix, # [9] times of every map being sampled
                m2l_tgt_idx, # [173394] in src map order
                m2l_src_pos # [173394, 2] in src map order
            ) = ext.preprocess.inverse_map_to_list(src_idx, src_pos, src_dms.shape[0])
            # print("m2l_prefix", m2l_prefix.shape)
            # print("m2l_tgt_idx", m2l_tgt_idx.shape)
            # print("src_pos", src_pos.shape)
            # print("m2l_prefix", m2l_prefix)
            ret["m2l_prefix"] = m2l_prefix
            ret["m2l_tgt_idx"] = m2l_tgt_idx
            ret["m2l_src_pos"] = m2l_src_pos

            for key in ret.keys():
                if not torch.is_tensor(ret[key]):
                    ret[key] = torch.from_numpy(ret[key])

            m2l_prefix_cum = np.hstack((np.zeros((1,), dtype=m2l_prefix.dtype), np.cumsum(ret["m2l_prefix"].numpy())))
            # print(m2l_prefix_cum)
            output = torch.empty((m2l_prefix_cum[-1], feature_channels))
            # print("output", output.shape)
            for view in np.argsort(m2l_prefix)[::-1]:
                if m2l_prefix[view] == 0:
                    continue
                feature = enc_feat_list[view][0, 0].cpu()
                # torch.save(feature, "./tmp/tmp.pt")
                # feature1 = torch.load("./tmp/tmp.pt", map_location="cpu")
                # print("feature", feature.shape)
                m2l_from = m2l_prefix_cum[view]
                m2l_end = m2l_prefix_cum[view+1]
                m2l_tgt_idx_view = ret["m2l_tgt_idx"][m2l_from:m2l_end]
                m2l_src_pos_view = ret["m2l_src_pos"][m2l_from:m2l_end]
                ext.mytorch.flag_map_to_list_bl_seq(
                    feature, m2l_tgt_idx_view, m2l_src_pos_view, output
                )
                output = output.to(device)
                tgt_encode_feature = tgt_feature_seq_to_map(output, **ret)
                # print(f"feature {feature.shape} output {output.shape}")

        if flag_map_to_list_bl:
            # for i in range(len(enc_feat_list)):
            #     enc_feat_list[i] = enc_feat_list[i].unsqueeze(0).unsqueeze(0) # [1, 1, 16, H, W] * V
            enc_src_feats = torch.cat(enc_feat_list, dim=1) # [1, V, 16, H, W]
            # B, V, C, H, W = enc_src_feats.shape

            tgt_encode_feature = fuse_mv_features_to_tgt(enc_src_feats, **ret)

        # with torch.no_grad():
        #     tgt_texture_feature = texture_net.forward(**ret)
        #
        # tgt_blend_feature = blend_feature_with_visibility(tgt_encode_feature, tgt_texture_feature,
        #                                                   ret["tgt_visibility_map"])
        with torch.no_grad():
            render_encode = render_net.forward_feature_map(tgt_encode_feature, **ret)
            
        img_render = render_encode["out"].detach().cpu().numpy()
        img_render = (np.clip(img_render, -0.9999, 0.9999) + 1) / 2
        img_render = img_render.transpose(0, 2, 3, 1)[0]
        img_render = (img_render * 255).astype(np.uint8)
        img_render_uncrop = uncrop_rgb_img(img_render, start_x, start_y, scale_width, scale_height)

        time_end = time.time()

        # print(f"Render img saves to {img_render_visibility_save_path}")
        print(f"View {count_target_view:06d} costs {(time_end - time_start):.2f}s!")

        img_render_uncrop = Image.fromarray(img_render_uncrop)

        #write image
        if list_path_save is not None:
            path_save = list_path_save[count_target_view]
            img_render_uncrop.save(path_save)
        else:
            list_image_render_uncrop.append(img_render_uncrop)

    if list_path_save is None:
        assert len(list_image_render_uncrop) == len(render_cameras), (len(list_image_render_uncrop), len(render_cameras))
        return list_image_render_uncrop
    else:
        assert len(list_image_render_uncrop) == 0, len(list_image_render_uncrop)
        return None