# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import torch.nn.functional as F
import time

from icecream import ic


rng = np.random.RandomState(int(time.time()))

########################################################################################################################
# ray batch sampling
########################################################################################################################


def parse_camera(params):
    H = params[:, 0]
    W = params[:, 1]
    intrinsics = params[:, 2:18].reshape((-1, 4, 4))
    c2w = params[:, 18:34].reshape((-1, 4, 4))
    return W, H, intrinsics, c2w


def dilate_img(img, kernel_size=20):
    import cv2
    assert img.dtype == np.uint8
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilation = cv2.dilate(img / 255, kernel, iterations=1) * 255
    return dilation


class RaySamplerSingleImage(object):
    # def __init__(self, data, device, resize_factor=1, render_stride=1):
    #     super().__init__()
    #     self.render_stride = render_stride # 1
    #     self.rgb = data['rgb'] if 'rgb' in data.keys() else None
    #     self.camera = data['camera']
    #     self.rgb_path = data['rgb_path']
    #     self.depth_range = data['depth_range']
    #     self.device = device
    #     W, H, self.intrinsics, self.c2w_mat = parse_camera(self.camera)
    #     self.batch_size = len(self.camera)
    #     # ic(self.intrinsics) # [1, 4, 4] first[3, 3] is intrinsic
    #     # ic(self.c2w_mat) # [1, 4, 4]
    #
    #     self.H = int(H[0])
    #     self.W = int(W[0])
    #
    #     # half-resolution output
    #     if resize_factor != 1: # False
    #         self.W = int(self.W * resize_factor)
    #         self.H = int(self.H * resize_factor)
    #         self.intrinsics[:, :2, :3] *= resize_factor
    #         if self.rgb is not None:
    #             self.rgb = F.interpolate(self.rgb.permute(0, 3, 1, 2), scale_factor=resize_factor).permute(0, 2, 3, 1)
    #
    #     self.rays_o, self.rays_d = self.get_rays_single_image(self.H, self.W, self.intrinsics, self.c2w_mat)
    #     if self.rgb is not None:
    #         self.rgb = self.rgb.reshape(-1, 3)
    #
    #     if 'src_rgbs' in data.keys():
    #         self.src_rgbs = data['src_rgbs']
    #     else:
    #         self.src_rgbs = None
    #     if 'src_cameras' in data.keys():
    #         self.src_cameras = data['src_cameras']
    #     else:
    #         self.src_cameras = None

    # def __init__(self, tgt_img_hd, tgt_intrinsic_hd, tgt_extrinsic_c2w, far_plane, near_plane, device,
    #              resize_factor=1, render_stride=1):
    #     super().__init__()
    #     self.render_stride = render_stride # 1
    #     # self.rgb = data['rgb'] if 'rgb' in data.keys() else None
    #     self.img_hd = tgt_img_hd # [B, C, H, W]
    #     B, _, H, W = self.img_hd.shape
    #     self.far_plane = far_plane
    #     self.near_plane = near_plane
    #     self.far_plane = far_plane
    #     self.tgt_extrinsic_c2w = np.identity(4, dtype=np.float32)[None, ...].repeat(B, axis=0)
    #     self.tgt_extrinsic_c2w = torch.from_numpy(self.tgt_extrinsic_c2w).to(device)
    #     self.tgt_extrinsic_c2w[:, :3] = tgt_extrinsic_c2w
    #     # self.tgt_extrinsic_c2w = torch.from_numpy(self.tgt_extrinsic_c2w)
    #     self.tgt_intrinsic = np.identity(4, dtype=np.float32)[None, ...].repeat(B, axis=0) # [1, 4, 4]
    #     self.tgt_intrinsic = torch.from_numpy(self.tgt_intrinsic).to(device)
    #     self.tgt_intrinsic[:, :3, :3] = tgt_intrinsic_hd
    #     # self.tgt_intrinsic = torch.from_numpy(self.tgt_intrinsic)
    #     self.batch_size = B
    #     self.H, self.W = int(H), int(W)
    #
    #
    #
    #
    #     # self.camera = data['camera']
    #     # self.rgb_path = data['rgb_path']
    #     # self.depth_range = data['depth_range']
    #     self.device = device
    #     # W, H, self.intrinsics, self.c2w_mat = parse_camera(self.camera)
    #     # self.batch_size = len(self.camera)
    #     # ic(self.intrinsics) # [1, 4, 4] first[3, 3] is intrinsic
    #     # ic(self.c2w_mat) # [1, 4, 4]
    #
    #     # self.H = int(H[0])
    #     # self.W = int(W[0])
    #
    #     # self.rays_o, self.rays_d = self.get_rays_single_image(self.H, self.W, self.intrinsics, self.c2w_mat)
    #     self.rays_o, self.rays_d = self.get_rays_single_image(self.H, self.W, self.tgt_intrinsic,
    #                                                           self.tgt_extrinsic_c2w)
    #
    #
    #     self.rgb = None
    #     self.src_rgbs = None
    #     # self.src_cameras = None
    #
    #     # # half-resolution output
    #     # if resize_factor != 1: # False
    #     #     self.W = int(self.W * resize_factor)
    #     #     self.H = int(self.H * resize_factor)
    #     #     self.intrinsics[:, :2, :3] *= resize_factor
    #     #     if self.rgb is not None:
    #     #         self.rgb = F.interpolate(self.rgb.permute(0, 3, 1, 2), scale_factor=resize_factor).permute(0, 2, 3, 1)
    #     #
    #     #
    #     # if self.rgb is not None:
    #     #     self.rgb = self.rgb.reshape(-1, 3)
    #     #
    #     # if 'src_rgbs' in data.keys():
    #     #     self.src_rgbs = data['src_rgbs']
    #     # else:
    #     #     self.src_rgbs = None
    #     # if 'src_cameras' in data.keys():
    #     #     self.src_cameras = data['src_cameras']
    #     # else:
    #     #     self.src_cameras = None

    def __init__(self, data, device):
        super().__init__()

        tgt_img_hd = data["tgt"]
        tgt_intrinsic_hd = data["tgt_intrinsic"]
        # tgt_depth_hd = data["tgt_depth_hd"]
        tgt_depth_hd = data["tgt_dm"]
        tgt_extrinsic_c2w = data["tgt_extrinsic_c2w"]
        far_plane = data["far_plane"]
        near_plane = data["near_plane"]
        self.tgt_border = list(data["tgt_border"].squeeze(0).cpu().numpy()) # [4]
        # ic(self.tgt_border)

        self.render_stride = 1 # 1
        # self.rgb = data['rgb'] if 'rgb' in data.keys() else None
        self.img_hd = tgt_img_hd # [B, C, H, W]
        B, _, H, W = self.img_hd.shape
        self.far_plane = far_plane
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.tgt_extrinsic_c2w = np.identity(4, dtype=np.float32)[None, ...].repeat(B, axis=0)
        self.tgt_extrinsic_c2w = torch.from_numpy(self.tgt_extrinsic_c2w).to(device)
        self.tgt_extrinsic_c2w[:, :3] = tgt_extrinsic_c2w
        # self.tgt_extrinsic_c2w = torch.from_numpy(self.tgt_extrinsic_c2w)
        self.tgt_intrinsic = np.identity(4, dtype=np.float32)[None, ...].repeat(B, axis=0) # [1, 4, 4]
        self.tgt_intrinsic = torch.from_numpy(self.tgt_intrinsic).to(device)
        self.tgt_intrinsic[:, :3, :3] = tgt_intrinsic_hd
        # self.tgt_intrinsic = torch.from_numpy(self.tgt_intrinsic)
        self.batch_size = B
        self.H, self.W = int(H), int(W)
        self.tgt_img_hd = self.img_hd.permute(0, 2, 3, 1).reshape(H*W, 3)
        self.tgt_depth_hd = tgt_depth_hd.squeeze(0).reshape(H*W) # [H*W,]


        # g_channel = self.tgt_img_hd[:, 0]
        # self.fore_idx = np.argwhere(g_channel.cpu().numpy() != 0)[:, 0] # foreground
        # self.back_idx = np.argwhere(g_channel.cpu().numpy() == 0)[:, 0] # background

        # ic(self.fore_idx.shape)





        # self.camera = data['camera']
        # self.rgb_path = data['rgb_path']
        # self.depth_range = data['depth_range']
        self.device = device
        # W, H, self.intrinsics, self.c2w_mat = parse_camera(self.camera)
        # self.batch_size = len(self.camera)
        # ic(self.intrinsics) # [1, 4, 4] first[3, 3] is intrinsic
        # ic(self.c2w_mat) # [1, 4, 4]

        # self.H = int(H[0])
        # self.W = int(W[0])

        # self.rays_o, self.rays_d = self.get_rays_single_image(self.H, self.W, self.intrinsics, self.c2w_mat)
        self.rays_o, self.rays_d = self.get_rays_single_image(self.H, self.W, self.tgt_intrinsic,
                                                              self.tgt_extrinsic_c2w)


        self.rgb = None
        self.src_rgbs = None
        self.src_cameras = None

        # # half-resolution output
        # if resize_factor != 1: # False
        #     self.W = int(self.W * resize_factor)
        #     self.H = int(self.H * resize_factor)
        #     self.intrinsics[:, :2, :3] *= resize_factor
        #     if self.rgb is not None:
        #         self.rgb = F.interpolate(self.rgb.permute(0, 3, 1, 2), scale_factor=resize_factor).permute(0, 2, 3, 1)
        #
        #
        # if self.rgb is not None:
        #     self.rgb = self.rgb.reshape(-1, 3)
        #
        # if 'src_rgbs' in data.keys():
        #     self.src_rgbs = data['src_rgbs']
        # else:
        #     self.src_rgbs = None
        # if 'src_cameras' in data.keys():
        #     self.src_cameras = data['src_cameras']
        # else:
        #     self.src_cameras = None

    # def __init__(self, tgt_img_hd, tgt_intrinsic_hd, tgt_depth_hd, tgt_extrinsic_c2w, far_plane, near_plane, device):
    #     super().__init__()
    #     self.render_stride = 1 # 1
    #     # self.rgb = data['rgb'] if 'rgb' in data.keys() else None
    #     self.img_hd = tgt_img_hd # [B, C, H, W]
    #     B, _, H, W = self.img_hd.shape
    #     self.far_plane = far_plane
    #     self.near_plane = near_plane
    #     self.far_plane = far_plane
    #     self.tgt_extrinsic_c2w = np.identity(4, dtype=np.float32)[None, ...].repeat(B, axis=0)
    #     self.tgt_extrinsic_c2w = torch.from_numpy(self.tgt_extrinsic_c2w).to(device)
    #     self.tgt_extrinsic_c2w[:, :3] = tgt_extrinsic_c2w
    #     # self.tgt_extrinsic_c2w = torch.from_numpy(self.tgt_extrinsic_c2w)
    #     self.tgt_intrinsic = np.identity(4, dtype=np.float32)[None, ...].repeat(B, axis=0) # [1, 4, 4]
    #     self.tgt_intrinsic = torch.from_numpy(self.tgt_intrinsic).to(device)
    #     self.tgt_intrinsic[:, :3, :3] = tgt_intrinsic_hd
    #     # self.tgt_intrinsic = torch.from_numpy(self.tgt_intrinsic)
    #     self.batch_size = B
    #     self.H, self.W = int(H), int(W)
    #     self.tgt_img_hd = self.img_hd.permute(0, 2, 3, 1).reshape(H*W, 3)
    #     self.tgt_depth_hd = tgt_depth_hd.squeeze(0).reshape(H*W) # [H*W,]
    #
    #     # g_channel = self.tgt_img_hd[:, 0]
    #     # self.fore_idx = np.argwhere(g_channel.cpu().numpy() != 0)[:, 0] # foreground
    #     # self.back_idx = np.argwhere(g_channel.cpu().numpy() == 0)[:, 0] # background
    #     self.fore_idx = np.argwhere(self.tgt_depth_hd.cpu().numpy() > 0)[:, 0] # foreground
    #     self.back_idx = np.argwhere(self.tgt_depth_hd.cpu().numpy() == 0)[:, 0] # background
    #     # ic(self.fore_idx.shape)
    #
    #
    #
    #
    #
    #     # self.camera = data['camera']
    #     # self.rgb_path = data['rgb_path']
    #     # self.depth_range = data['depth_range']
    #     self.device = device
    #     # W, H, self.intrinsics, self.c2w_mat = parse_camera(self.camera)
    #     # self.batch_size = len(self.camera)
    #     # ic(self.intrinsics) # [1, 4, 4] first[3, 3] is intrinsic
    #     # ic(self.c2w_mat) # [1, 4, 4]
    #
    #     # self.H = int(H[0])
    #     # self.W = int(W[0])
    #
    #     # self.rays_o, self.rays_d = self.get_rays_single_image(self.H, self.W, self.intrinsics, self.c2w_mat)
    #     self.rays_o, self.rays_d = self.get_rays_single_image(self.H, self.W, self.tgt_intrinsic,
    #                                                           self.tgt_extrinsic_c2w)
    #
    #
    #     self.rgb = None
    #     self.src_rgbs = None
    #     self.src_cameras = None
    #
    #     # # half-resolution output
    #     # if resize_factor != 1: # False
    #     #     self.W = int(self.W * resize_factor)
    #     #     self.H = int(self.H * resize_factor)
    #     #     self.intrinsics[:, :2, :3] *= resize_factor
    #     #     if self.rgb is not None:
    #     #         self.rgb = F.interpolate(self.rgb.permute(0, 3, 1, 2), scale_factor=resize_factor).permute(0, 2, 3, 1)
    #     #
    #     #
    #     # if self.rgb is not None:
    #     #     self.rgb = self.rgb.reshape(-1, 3)
    #     #
    #     # if 'src_rgbs' in data.keys():
    #     #     self.src_rgbs = data['src_rgbs']
    #     # else:
    #     #     self.src_rgbs = None
    #     # if 'src_cameras' in data.keys():
    #     #     self.src_cameras = data['src_cameras']
    #     # else:
    #     #     self.src_cameras = None

    def get_rays_single_image(self, H, W, intrinsics, c2w):
        '''
        :param H: image height
        :param W: image width
        :param intrinsics: 4 by 4 intrinsic matrix
        :param c2w: 4 by 4 camera to world extrinsic matrix
        :return:
        '''
        u, v = np.meshgrid(np.arange(W)[::self.render_stride], np.arange(H)[::self.render_stride])
        # ic(u.shape, u) # [756, 1008] [H, W]
        # ic(v.shape, v) # [756, 1008] [H, W]
        u = u.reshape(-1).astype(dtype=np.float32)  # + 0.5    # add half pixel
        v = v.reshape(-1).astype(dtype=np.float32)  # + 0.5
        # ic(u.shape) # [762048]
        # ic(v.shape) # [762048]
        pixels = np.stack((u, v, np.ones_like(u)), axis=0)  # (3, H*W)
        # ic(pixels.shape, pixels) # [3, 762048]
        pixels = torch.from_numpy(pixels).to(self.device)
        batched_pixels = pixels.unsqueeze(0).repeat(self.batch_size, 1, 1) # [1, 3, 762048]
        # ic(batched_pixels) # u, v, 1
        # ic(intrinsics.shape, c2w.shape)
        # [B, H*W, 3]
        rays_d = (c2w[:, :3, :3].bmm(torch.inverse(intrinsics[:, :3, :3])).bmm(batched_pixels)).transpose(1, 2)
        # ic(rays_d.shape) # [1, 762048, 3]
        rays_d = rays_d.reshape(-1, 3) # [762048, 3]
        rays_o = c2w[:, :3, 3].unsqueeze(1).repeat(1, rays_d.shape[0], 1).reshape(-1, 3)  # [762048, 3]B x HW x 3
        # ic(rays_d.shape, rays_d)
        # ic(rays_o.shape, rays_o[:200]) # [H*W, 3]
        # assert 0
        return rays_o, rays_d

    def get_all(self):
        # ic(self.rays_o.shape) # [762048, 3] 762048=752*1008
        # ic(self.rays_d.shape) # [762048, 3]
        # ic(self.depth_range.shape) # [1, 2]
        # ic(self.camera.shape) # [1, 34]
        # # ic(self.rgb.shape) # None
        # ic(self.src_rgbs.shape) # [1, 10, 756, 1008, 3]
        # ic(self.src_cameras.shape) # [1, 10, 34]
        ret = {'ray_o': self.rays_o,
               'ray_d': self.rays_d,
               # 'depth_range': self.depth_range.cuda(), # [1, 2]
               # 'camera': self.camera.cuda(),
               # 'rgb': self.rgb.cuda() if self.rgb is not None else None,
               # 'src_rgbs': self.src_rgbs.cuda() if self.src_rgbs is not None else None,
               # 'src_cameras': self.src_cameras.cuda() if self.src_cameras is not None else None,
        }
        return ret

    def sample_random_pixel(self, N_rand, sample_mode, center_ratio, fore_ratio):
        if sample_mode == "fore_back":
            fore_pix_num = int(N_rand * fore_ratio)
            back_pix_num = N_rand - fore_pix_num
            min_x, min_y, max_x, max_y = self.tgt_border
            u, v = np.meshgrid(np.arange(min_x, max_x), np.arange(min_y, max_y))
            u = u.reshape(-1)
            v = v.reshape(-1)
            select_fore_inds = rng.choice(u.shape[0], size=(fore_pix_num,), replace=False)
            select_fore_inds = v[select_fore_inds] + self.W * u[select_fore_inds]

            select_back_inds = rng.choice(self.H*self.W, size=(back_pix_num,), replace=False)
            select_inds = np.concatenate([select_fore_inds, select_back_inds], axis=0)




            # fore_pix_num = int(N_rand * fore_ratio)
            # back_pix_num = N_rand - fore_pix_num
            # fore_idx = np.argwhere(self.tgt_depth_hd.cpu().numpy() > 0)[:, 0]  # foreground
            # back_idx = np.argwhere(self.tgt_depth_hd.cpu().numpy() == 0)[:, 0]  # background
            # fore_inds = rng.choice(fore_idx.shape[0], size=(fore_pix_num,), replace=False)
            # back_inds = rng.choice(back_idx.shape[0], size=(back_pix_num,), replace=False)
            # select_fore_inds = fore_idx[fore_inds]
            # select_back_inds = back_idx[back_inds]
            # select_inds = np.concatenate([select_fore_inds, select_back_inds], axis=0)
            # ic(select_inds.shape)

        if sample_mode == 'center':
            # min_x, min_y, max_x, max_y = self.tgt_border
            # u, v = np.meshgrid(np.arange(min_x, max_x), np.arange(min_y, max_y))
            # u = u.reshape(-1)
            # v = v.reshape(-1)
            #
            # select_inds = rng.choice(u.shape[0], size=(N_rand,), replace=False)
            # select_inds = v[select_inds] + self.W * u[select_inds]

            # border_H = int(self.H * (1 - center_ratio) / 2.)
            # border_W = int(self.W * (1 - center_ratio) / 2.)
            #
            # # pixel coordinates
            # u, v = np.meshgrid(np.arange(border_H, self.H - border_H),
            #                    np.arange(border_W, self.W - border_W))
            # u = u.reshape(-1)
            # v = v.reshape(-1)
            #
            # select_inds = rng.choice(u.shape[0], size=(N_rand,), replace=False)
            # select_inds = v[select_inds] + self.W * u[select_inds]

            pass

        if sample_mode == 'uniform': # True
            # Random from one image
            select_inds = rng.choice(self.H*self.W, size=(N_rand,), replace=False)
            # ic(select_inds.shape)

        return select_inds

    def random_sample(self, N_rand, sample_mode, center_ratio, fore_ratio):
        '''
        :param N_rand: number of rays to be casted
        :return:
        '''

        select_inds = self.sample_random_pixel(N_rand, sample_mode, center_ratio, fore_ratio=fore_ratio)
        # ic(type(select_inds), select_inds)
        # assert 0

        rays_o = self.rays_o[select_inds] # [R, 3]
        # ic(rays_o.shape)
        rays_d = self.rays_d[select_inds] # [R, 3]
        rgb = self.tgt_img_hd[select_inds] # [R, 3]
        depth = self.tgt_depth_hd[select_inds] # [R,]
        # ic(depth[:400] < self.far_plane.squeeze(0), depth[:400] > self.near_plane.squeeze(0))
        # ic(depth[:400][depth[:400] <= self.near_plane.squeeze(0)], depth[:400])
        # assert 0

        # if self.rgb is not None:
        #     rgb = self.rgb[select_inds]
        # else:
        #     rgb = None

        # valid_depth = (depth > self.near_plane.squeeze(0)) & (depth < self.far_plane.squeeze(0))

        ret = {'ray_o': rays_o,
               'ray_d': rays_d,
               # 'camera': self.camera.cuda(),
               # 'depth_range': self.depth_range.cuda(),
               # "far_plane": self.far_plane, # [1, 1]
               # "near_plane": self.near_plane,
               "rgb": rgb,
               "depth": depth, # [R,]
               # 'src_rgbs': self.src_rgbs.cuda() if self.src_rgbs is not None else None,
               # 'src_cameras': self.src_cameras.cuda() if self.src_cameras is not None else None,
               "selected_inds": select_inds,
               # "valid_depth": valid_depth
        }
        return ret
