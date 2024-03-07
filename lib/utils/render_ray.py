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


import torch
import torch.nn.functional as F
from collections import OrderedDict
from icecream import ic
import time

########################################################################################################################
# helper functions for nerf ray rendering
########################################################################################################################


def sample_pdf(bins, weights, N_samples, det=False):
    '''
    :param bins: tensor of shape [N_rays, M+1], M is the number of bins
    :param weights: tensor of shape [N_rays, M]
    :param N_samples: number of samples along each ray
    :param det: if True, will perform deterministic sampling
    :return: [N_rays, N_samples]
    '''

    M = weights.shape[1]
    weights += 1e-5
    # Get pdf
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)    # [N_rays, M]
    cdf = torch.cumsum(pdf, dim=-1)  # [N_rays, M]
    cdf = torch.cat([torch.zeros_like(cdf[:, 0:1]), cdf], dim=-1) # [N_rays, M+1]

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., N_samples, device=bins.device)
        u = u.unsqueeze(0).repeat(bins.shape[0], 1)       # [N_rays, N_samples]
    else:
        u = torch.rand(bins.shape[0], N_samples, device=bins.device)

    # Invert CDF
    above_inds = torch.zeros_like(u, dtype=torch.long)       # [N_rays, N_samples]
    for i in range(M):
        above_inds += (u >= cdf[:, i:i+1]).long()

    # random sample inside each bin
    below_inds = torch.clamp(above_inds-1, min=0)
    inds_g = torch.stack((below_inds, above_inds), dim=2)     # [N_rays, N_samples, 2]

    cdf = cdf.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples, M+1]
    cdf_g = torch.gather(input=cdf, dim=-1, index=inds_g)  # [N_rays, N_samples, 2]

    bins = bins.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples, M+1]
    bins_g = torch.gather(input=bins, dim=-1, index=inds_g)  # [N_rays, N_samples, 2]

    # t = (u-cdf_g[:, :, 0]) / (cdf_g[:, :, 1] - cdf_g[:, :, 0] + TINY_NUMBER)  # [N_rays, N_samples]
    # fix numeric issue
    denom = cdf_g[:, :, 1] - cdf_g[:, :, 0]      # [N_rays, N_samples]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[:, :, 0]) / denom

    samples = bins_g[:, :, 0] + t * (bins_g[:, :, 1]-bins_g[:, :, 0])

    return samples


def sample_along_camera_ray(ray_o,
                            ray_d, # [4096, 3]
                            # depth_range,
                            far_plane,
                            near_plane,
                            N_samples,
                            inv_uniform,
                            det):
    '''
    :param ray_o: origin of the ray in scene coordinate system; tensor of shape [N_rays, 3]
    :param ray_d: homogeneous ray direction vectors in scene coordinate system; tensor of shape [N_rays, 3]
    :param depth_range: [near_depth, far_depth]
    :param inv_uniform: if True, uniformly sampling inverse depth
    :param det: if True, will perform deterministic sampling
    :return: tensor of shape [N_rays, N_samples, 3]
    '''
    # will sample inside [near_depth, far_depth]
    # assume the nearest possible depth is at least (min_ratio * depth)
    # near_depth_value = depth_range[0, 0]
    # far_depth_value = depth_range[0, 1]
    near_depth_value = near_plane[0]
    far_depth_value = far_plane[0]

    # ic(near_depth_value, far_depth_value) # 1.2 9.42
    # assert 0
    assert near_depth_value > 0 and far_depth_value > 0 and far_depth_value > near_depth_value

    near_depth = near_depth_value * torch.ones_like(ray_d[..., 0])
    # ic(ray_d.shape, ray_d) # [4096, 3]

    far_depth = far_depth_value * torch.ones_like(ray_d[..., 0])
    if inv_uniform:
        start = 1. / near_depth     # [N_rays,]
        step = (1. / far_depth - start) / (N_samples-1)
        inv_z_vals = torch.stack([start+i*step for i in range(N_samples)], dim=1)  # [N_rays, N_samples] [4096, 64]
        z_vals = 1. / inv_z_vals # [4096, 64]
    else:
        start = near_depth
        step = (far_depth - near_depth) / (N_samples-1)
        # [4096, 64] [N_rays, N_samples]
        z_vals = torch.stack([start+i*step for i in range(N_samples)], dim=1)

        # ic(z_vals.shape, z_vals[0, :], near_depth_value, far_depth_value) # [4096, 64]
        # assert 0
    # assert 0

    # ic(near_depth_value, far_depth_value, z_vals.shape, z_vals)
    # if not det: # False
    #     # get intervals between samples
    #     mids = .5 * (z_vals[:, 1:] + z_vals[:, :-1])
    #     # ic(mids.shape)
    #     upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
    #     lower = torch.cat([z_vals[:, 0:1], mids], dim=-1)
    #     # uniform samples in those intervals
    #     t_rand = torch.rand_like(z_vals)
    #     z_vals = lower + (upper - lower) * t_rand   # [N_rays, N_samples]

    ray_d = ray_d.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples, 3] [4096, 64, 3]
    ray_o = ray_o.unsqueeze(1).repeat(1, N_samples, 1) # [4096, 64, 3]
    pts = z_vals.unsqueeze(2) * ray_d + ray_o       # [N_rays, N_samples, 3] [4096, 64, 3]
    return pts, z_vals


########################################################################################################################
# ray rendering of nerf
########################################################################################################################

def raw2outputs(raw, z_vals, mask, white_bkgd=False):
    '''
    :param raw: raw network output; tensor of shape [N_rays, N_samples, 4]
    :param z_vals: depth of point samples along rays; tensor of shape [N_rays, N_samples]
    :param ray_d: [N_rays, 3]
    :return: {'rgb': [N_rays, 3], 'depth': [N_rays,], 'weights': [N_rays,], 'depth_std': [N_rays,]}
    '''
    rgb = raw[:, :, :3]     # [N_rays, N_samples, 3]
    sigma = raw[:, :, 3]    # [N_rays, N_samples]

    # note: we did not use the intervals here, because in practice different scenes from COLMAP can have
    # very different scales, and using interval can affect the model's generalization ability.
    # Therefore we don't use the intervals for both training and evaluation.
    sigma2alpha = lambda sigma, dists: 1. - torch.exp(-sigma)

    # point samples are ordered with increasing depth
    # interval between samples
    dists = z_vals[:, 1:] - z_vals[:, :-1]
    dists = torch.cat((dists, dists[:, -1:]), dim=-1)  # [N_rays, N_samples]

    alpha = sigma2alpha(sigma, dists)  # [N_rays, N_samples]

    # Eq. (3): T
    T = torch.cumprod(1. - alpha + 1e-10, dim=-1)[:, :-1]   # [N_rays, N_samples-1]
    T = torch.cat((torch.ones_like(T[:, 0:1]), T), dim=-1)  # [N_rays, N_samples]

    # maths show weights, and summation of weights along a ray, are always inside [0, 1]
    weights = alpha * T     # [N_rays, N_samples]
    rgb_map = torch.sum(weights.unsqueeze(2) * rgb, dim=1)  # [N_rays, 3]

    if white_bkgd:
        rgb_map = rgb_map + (1. - torch.sum(weights, dim=-1, keepdim=True))

    mask = mask.float().sum(dim=1) > 8  # should at least have 8 valid observation on the ray, otherwise don't consider its loss
    depth_map = torch.sum(weights * z_vals, dim=-1)     # [N_rays,]

    ret = OrderedDict([('rgb', rgb_map),
                       ('depth', depth_map),
                       ('weights', weights),                # used for importance sampling of fine samples
                       ('mask', mask),
                       ('alpha', alpha),
                       ('z_vals', z_vals)
                       ])

    return ret

def sigma2alpha(sigma):
    return 1. - torch.exp(-sigma)

def calc_weights(projector, data, pts, geo_featmaps, model_net, query_z, mlp_parallel):
    num_src_views = data["src_Ks"].shape[1]
    N_rays, N_samples = pts.shape[:2]
    # [N_views, N_rays, N_samples, 3] [N_views, N_rays, N_samples, 2]

    pts_camera_space, pixel_locations512, mask_in_front512 = projector.calc_projections(pts, data["src_intrinsics512"],
                                                                                        data["src_extrinsics_w2c"])

    # [4, 4096, 64, 2] [V, R, S, 2]
    normalized_pixel_locations512 = projector.normalize(pixel_locations512, data["src_imgs512"].shape[3],
                                                        data["src_imgs512"].shape[4])
    # for i in range(20, 40):
    #     ic(i, (normalized_pixel_locations512[1, :500, 31, :] + 1) / 2 * 511)
    #     ic(i, pixel_locations512[1, :500, 31, :])
    # assert 0
    # ic(normalized_pixel_locations.shape, normalized_pixel_locations512.shape)
    # print(time.time()-start, "s")
    # start = time.time()

    # geo net
    geofeat = F.grid_sample(geo_featmaps, normalized_pixel_locations512, align_corners=True)  # [4, 256, 4096, 64]
    # print(time.time()-start, "s")
    # start = time.time()
    geo_dim = geofeat.shape[1]
    geofeat = geofeat.permute(2, 3, 0, 1).reshape(N_rays * N_samples, num_src_views, geo_dim)  # [R*S, 4, 256]
    geo_att_feat = model_net.geo_attention_net(geofeat)  # [R*S, 4, 256] -> [R*S, 4, 256]
    # print(time.time()-start, "s")
    # start = time.time()

    if query_z:
        pts_z_camera_space = pts_camera_space[:, :, :, 2:].reshape(num_src_views, N_rays * N_samples, 1)  # [V, R*S, 1]

        # [V, 4, 1]
        centers_world_space = torch.cat([data["human_center"].squeeze(0), torch.ones((1)).cuda()], dim=0)[
            None, ..., None]. \
            repeat(num_src_views, 1, 1)
        # [V, 3, 1]
        centers_camera_space = (data["src_extrinsics_w2c"].squeeze(0)).bmm(centers_world_space)
        centers_z_camera_space = centers_camera_space[:, 2:, :].repeat(1, N_rays * N_samples, 1)  # [V, R*S, 1]
        z_feat = (pts_z_camera_space - centers_z_camera_space) / (
                    (data["human_length"].squeeze(0)) * 1.7321)  # [V, R*S, 1]
        z_feat = z_feat.permute(1, 0, 2)  # [R*S, V, 1]

        geo_att_z_feat = torch.cat([geo_att_feat, z_feat], dim=2).permute(1, 2, 0)  # [R*S, V, 257] -> [V, 257, R*S]
        sigma, _ = model_net.geo_mlp_net(geo_att_z_feat)  # [V, 1, R*S]
        sigma = torch.mean(sigma, dim=0).permute(1, 0).reshape(N_rays, N_samples, 1).squeeze(2)  # [R, S]
    else:
        # geo_attavg_feat = torch.mean(geo_att_feat, dim=1).unsqueeze(2) # [R*S, 256, 1]
        # sigma, _ = model_net.geo_mlp_net(geo_attavg_feat) # [R*S, 1, 1]
        # sigma = sigma[:, 0, 0].reshape(N_rays, N_samples) # [R, S]

        if mlp_parallel:
            geo_attavg_feat = torch.mean(geo_att_feat, dim=1) # [R*S, 256]
            # [2, R*S/2, geo_dim] -> [2, geo_dim, R*S/2]
            geo_attavg_feat = geo_attavg_feat.reshape(2, N_rays*N_samples//2, geo_dim).permute(0, 2, 1)
            # [2, 1, R*S/2] -> [R*S]
            sigma, phi = model_net.geo_mlp_net(geo_attavg_feat).squeeze(1).reshape(N_rays*N_samples)
            sigma = sigma.reshape(N_rays, N_samples)  # [R, S]
        else:
            # geo_attavg_feat = torch.mean(geo_att_feat, dim=1).unsqueeze(2).permute(2, 1, 0)  # [1, 256, R*S]
            # sigma, phi = model_net.geo_mlp_net(geo_attavg_feat)  # [1, 1, R*S] [1, 128, R*S]
            # # ic(sigma.shape, phi.shape)
            # sigma = sigma[0, 0, :].reshape(N_rays, N_samples)  # [R, S]

            geo_attavg_feat = torch.mean(geo_att_feat, dim=1).unsqueeze(2) # [R*S, 256, 1]
            sigma, phi = model_net.geo_mlp_net(geo_attavg_feat) # [R*S, 1, 1] [R*S, 128, 1]
            sigma = sigma[:, 0, 0].reshape(N_rays, N_samples) # [R, S]


    # print(time.time()-start, "s")
    # start = time.time()

    # ic(sigma[:, 31])
    # ic(sigma[0], sigma[20], sigma[31])
    # assert 0

    # sigma2alpha = lambda sigma: 1. - torch.exp(-sigma)

    alpha = sigma2alpha(sigma)
    T = torch.cumprod(1. - alpha + 1e-10, dim=-1)[:, :-1]  # [N_rays, N_samples-1]
    T = torch.cat((torch.ones_like(T[:, 0:1]), T), dim=-1)  # [N_rays, N_samples]
    weights = alpha * T  # [N_rays, N_samples]

    return weights, phi

def render_rays(ray_batch,
                data,
                # model,
                model_net,
                # geo_fusion_net,
                # img_attention_net,
                # img_mlp_net,
                # geo_mlp_net,
                # featmaps,
                geo_featmaps,
                img_featmaps,
                projector,
                N_samples,
                inv_uniform,
                N_importance,
                det,
                white_bkgd,
                train_only_depth,
                train_only_rgb,
                train_rgb_depth,
                query_z,
                mlp_parallel,
                geo_cond_weights, add_img_att):
    '''
    :param ray_batch: {'ray_o': [N_rays, 3] , 'ray_d': [N_rays, 3], 'view_dir': [N_rays, 2]}
    :param model:  {'net_coarse':  , 'net_fine': }
    :param N_samples: samples along each ray (for both coarse and fine model)
    :param inv_uniform: if True, uniformly sample inverse depth for coarse model
    :param N_importance: additional samples along each ray produced by importance sampling (for fine model)
    :param det: if True, will deterministicly sample depths
    :return: {'outputs_coarse': {}, 'outputs_fine': {}}
    '''
    # start = time.time()
    num_src_views = data["src_ims"].shape[1]




    # ret = {'outputs_coarse': None,
    #        'outputs_fine': None}
    ret = {}

    # pts: [N_rays, N_samples, 3] [4096, 64, 3]   z_vals: [N_rays, N_samples] [4096, 64]
    pts, z_vals = sample_along_camera_ray(ray_o=ray_batch['ray_o'],
                                          ray_d=ray_batch['ray_d'],
                                          # depth_range=ray_batch['depth_range'],
                                          far_plane=data["far_plane"],
                                          near_plane=data["near_plane"],
                                          N_samples=N_samples, inv_uniform=inv_uniform, det=det)
    N_rays, N_samples = pts.shape[:2]

    with torch.no_grad():
        weights, phi = calc_weights(projector=projector, data=data, pts=pts, geo_featmaps=geo_featmaps, model_net=model_net,
                               query_z=query_z, mlp_parallel=mlp_parallel)

    if N_importance > 0:
        # detach since we would like to decouple the coarse and fine networks


        # take mid-points of depth samples
        z_vals_mid = .5 * (z_vals[:, 1:] + z_vals[:, :-1])   # [N_rays, N_samples-1]
        weights = weights[:, 1:-1]      # [N_rays, N_samples-2]
        z_samples = sample_pdf(bins=z_vals_mid, weights=weights,
                               N_samples=N_importance, det=det)  # [N_rays, N_importance]

        z_vals = torch.cat((z_vals, z_samples), dim=-1)  # [N_rays, N_samples + N_importance]

        # samples are sorted with increasing depth
        z_vals, _ = torch.sort(z_vals, dim=-1)
        N_total_samples = N_samples + N_importance

        ray_d = ray_batch["ray_d"].unsqueeze(1).repeat(1, N_total_samples, 1)  # [N_rays, N_samples, 3] [4096, 128, 3]
        ray_o = ray_batch["ray_o"].unsqueeze(1).repeat(1, N_total_samples, 1)  # [4096, 128, 3]
        pts = z_vals.unsqueeze(2) * ray_d + ray_o  # [N_rays, N_samples, 3] [4096, 128, 3]
        # weights: [R*S, 1, 1] phi: [R*S, 128, 1]
        weights, phi = calc_weights(projector=projector, data=data, pts=pts, geo_featmaps=geo_featmaps, model_net=model_net,
                               query_z=query_z, mlp_parallel=mlp_parallel)

    img_depth = torch.sum(weights * z_vals, dim=1) # [R,]
    # ic(sigma[0], alpha[0], T[0], weights[0])
    # ic(img_depth)
    ret["render_depth"] = img_depth
    # print(time.time()-start, "s")

    N_rays, N_samples = pts.shape[:2] # update Sample Number!!!!
    if train_rgb_depth or train_only_rgb:
        _, pixel_locations, mask_in_front = projector.calc_projections(pts, data["src_Ks"], data["src_extrinsics_w2c"])
        normalized_pixel_locations = projector.normalize(pixel_locations, data["src_ims"].shape[3],
                                                         data["src_ims"].shape[4]) # [4, 4096, 64, 2]
        imgfeat = F.grid_sample(img_featmaps, normalized_pixel_locations, align_corners=True) # [V, img_dim, R, S]
        # src_rgb_feat = F.grid_sample(data["src_ims"][0], normalized_pixel_locations, align_corners=True) # [V, 3, R, S]
        img_dim = imgfeat.shape[1]

        if geo_cond_weights:
            geo_cond_feat = phi.permute(2, 1, 0).repeat(num_src_views, 1, 1) # [V, 128, R*S]
            if add_img_att:
                imgfeat = imgfeat.permute(2, 3, 0, 1).reshape(N_rays * N_samples, num_src_views,
                                                              img_dim)  # [R*S, V, img_dim]
                img_att_feat = model_net.img_attention_net(imgfeat)  # [R*S, V, img_dim]
                # [V, img_dim, R*S]
                img_attavg_feat = torch.mean(img_att_feat, dim=1).permute(1, 0).unsqueeze(0).repeat(num_src_views, 1, 1)
                imgfeat = imgfeat.permute(1, 2, 0)  # [V, img_dim, R*S]
                # ic(imgfeat.shape, img_attavg_feat.shape, geo_cond_feat.shape)
                geo_cond_img_feat = torch.cat([imgfeat, img_attavg_feat, geo_cond_feat], dim=1) # [V, 128+img_dim*2, R*S]
                # [img_dim*2+128, V, R*S] -> [1, img_dim*2+128, V*R*S] -> [V*R*S, img_dim*2+128, 1]

                geo_cond_img_feat = geo_cond_img_feat.permute(1, 0, 2).reshape(img_dim*2 + 128, -1).unsqueeze(0).\
                    permute(2, 1, 0)
            else:
                imgfeat = imgfeat.reshape(num_src_views, img_dim, N_rays * N_samples)  # [V, img_dim, R*S]
                geo_cond_img_feat = torch.cat([imgfeat, geo_cond_feat], dim=1) # [V, img_dim+128, R*S]
                # [img_dim+128, V, R*S] -> [1, img_dim+128, V*R*S]
                geo_cond_img_feat = geo_cond_img_feat.permute(1, 0, 2).reshape(img_dim+128, -1).unsqueeze(0)

                geo_cond_img_feat = geo_cond_img_feat.permute(2, 1, 0)
            img_feat_blending_weights = model_net.geo_cond_mlp_net(geo_cond_img_feat)[0] # [V*R*S, 1, 1]
            # [V, R*S]
            img_feat_blending_weights = img_feat_blending_weights[:, 0, 0].reshape(num_src_views, N_rays*N_samples)
            img_feat_blending_weights = F.softmax(img_feat_blending_weights, dim=0).unsqueeze(1) # [V, 1, R*S]
            img_blend_feat = torch.sum(img_feat_blending_weights * imgfeat, dim=0).unsqueeze(0) # [1, img_dim, R*S]
            img_all_rgb = model_net.img_mlp_net(img_blend_feat)[0] # [1, 3, R*S]
            img_all_rgb = img_all_rgb.permute(2, 1, 0).unsqueeze(2).reshape(N_rays, N_samples, 3)  # [R, S, 3]
            img_rgb = torch.sum(weights.unsqueeze(2) * img_all_rgb, dim=1)  # [R, 3]
        else:
            imgfeat = imgfeat.permute(2, 3, 0, 1).reshape(N_rays*N_samples, num_src_views, img_dim) # [R*S, V, img_dim]
            img_att_feat = model_net.img_attention_net(imgfeat) # [R*S, V, img_dim]
            img_attavg_feat = torch.mean(img_att_feat, dim=1) # [R*S, img_dim]



            if mlp_parallel:
                # [2, R*S/2, img_dim] -> [2, img_dim, R*S/2]
                img_attavg_feat = img_attavg_feat.reshape(2, N_rays*N_samples//2, img_dim).permute(0, 2, 1)
                img_all_rgb = model_net.img_mlp_net(img_attavg_feat)[0]  # [2, 3, R*S/2]
                img_all_rgb = img_all_rgb.permute(0, 2, 1).reshape(N_rays*N_samples, 3) # [R*S, 3]
                img_all_rgb = img_all_rgb.reshape(N_rays, N_samples, 3)  # [R, S, 3]
                img_rgb = torch.sum(weights.unsqueeze(2) * img_all_rgb, dim=1)  # [R, 3]
            else:
                img_attavg_feat = img_attavg_feat.unsqueeze(2).permute(2, 1, 0) # [1, img_dim, R*S]
                img_all_rgb = model_net.img_mlp_net(img_attavg_feat)[0] # [1, 3, R*S]
                img_all_rgb = img_all_rgb.permute(2, 1, 0).unsqueeze(2).reshape(N_rays, N_samples, 3) # [R, S, 3]
                img_rgb = torch.sum(weights.unsqueeze(2) * img_all_rgb, dim=1) # [R, 3]

        # img_attavg_feat = img_attavg_feat.reshape(N_rays, N_samples, img_dim)  # [R, S, img_dim]
        # img_sum_feat = torch.sum(weights.unsqueeze(2) * img_attavg_feat, dim=1).unsqueeze(2) # [R, img_dim, 1]
        # img_sum_feat = img_sum_feat.permute(2, 1, 0) # [1, img_dim, R]
        # img_rgb, _ = model_net.img_mlp_net(img_sum_feat) # [1, 3, R]
        # img_rgb = img_rgb.permute(2, 1, 0).squeeze(2) # [R, 3]

        ret["render_rgb"] = img_rgb # [R, 3]
        # print("infer rgb")
    else:
        # print("no infer rgb")
        pass

    return ret







    # intensity = intensity.masked_fill(mask_in_front512 )
    # for view  in range(num_src_views):
    #     geo_sv_feat = geo_mlp_net()
    # geo_z_att_feat = geo_fusion_net(geo_z_feat)
    # geo_z_fuse_feat = torch.mean(geo_z_att_feat, dim=1) # [R*S, 257]

    # geo_embed_feat = geo_mlp_net(geo_z_feat.permute(1, 2, 0)) # [V, 257, R*S] -> [V, 512, R*S]
    # ic(geo_embed_feat.shape)
    #
    # imgfeat = F.grid_sample(img_featmaps, normalized_pixel_locations, align_corners=True) # [4, 16, 4096, 64]
    # imgfeat = imgfeat.permute(2, 3, 0, 1) # [N_rays, N_samples, V, 16]

    # N_rays, N_samples, num_src_views, geo_dim = geofeat.shape
    # geofeat = geofeat.reshape(N_rays*N_samples, num_src_views, geo_dim)
    # geo_att_feat = geo_fusion_net(geofeat)
    # geo_att_feat = geo_att_feat.reshape(N_rays, N_samples, num_src_views, geo_dim)
    # geo_fuse_feat = torch.mean(geo_att_feat, dim=2) # [4096, 64, 256]
    # ic(geo_fuse_feat.shape)
    # ic(geofeat.shape, imgfeat.shape)
    # assert 0


    # ic(pixel_locations, pixel_locations512)



    # pixel_mask = mask[..., 0].sum(dim=2) > 1   # [N_rays, N_samples], should at least have 2 observations
    # # [4096, 64, 4] [N_rays, N_samples, 4]
    # raw_coarse = model.net_coarse(rgb_feat, ray_diff, mask)
    # outputs_coarse = raw2outputs(raw_coarse, z_vals, pixel_mask,
    #                              white_bkgd=white_bkgd)
    # ret['outputs_coarse'] = outputs_coarse
    #
    # if N_importance > 0:
    #     assert model.net_fine is not None
    #     # detach since we would like to decouple the coarse and fine networks
    #     weights = outputs_coarse['weights'].clone().detach()            # [N_rays, N_samples]
    #     if inv_uniform:
    #         inv_z_vals = 1. / z_vals
    #         inv_z_vals_mid = .5 * (inv_z_vals[:, 1:] + inv_z_vals[:, :-1])   # [N_rays, N_samples-1]
    #         weights = weights[:, 1:-1]      # [N_rays, N_samples-2]
    #         inv_z_vals = sample_pdf(bins=torch.flip(inv_z_vals_mid, dims=[1]),
    #                                 weights=torch.flip(weights, dims=[1]),
    #                                 N_samples=N_importance, det=det)  # [N_rays, N_importance]
    #         z_samples = 1. / inv_z_vals
    #     else:
    #         # take mid-points of depth samples
    #         z_vals_mid = .5 * (z_vals[:, 1:] + z_vals[:, :-1])   # [N_rays, N_samples-1]
    #         weights = weights[:, 1:-1]      # [N_rays, N_samples-2]
    #         z_samples = sample_pdf(bins=z_vals_mid, weights=weights,
    #                                N_samples=N_importance, det=det)  # [N_rays, N_importance]
    #
    #     z_vals = torch.cat((z_vals, z_samples), dim=-1)  # [N_rays, N_samples + N_importance]
    #
    #     # samples are sorted with increasing depth
    #     z_vals, _ = torch.sort(z_vals, dim=-1)
    #     N_total_samples = N_samples + N_importance
    #
    #     viewdirs = ray_batch['ray_d'].unsqueeze(1).repeat(1, N_total_samples, 1)
    #     ray_o = ray_batch['ray_o'].unsqueeze(1).repeat(1, N_total_samples, 1)
    #     pts = z_vals.unsqueeze(2) * viewdirs + ray_o  # [N_rays, N_samples + N_importance, 3]
    #
    #     rgb_feat_sampled, ray_diff, mask = projector.compute(pts, ray_batch['camera'],
    #                                                          ray_batch['src_rgbs'],
    #                                                          ray_batch['src_cameras'],
    #                                                          featmaps=featmaps[1])
    #
    #     pixel_mask = mask[..., 0].sum(dim=2) > 1  # [N_rays, N_samples]. should at least have 2 observations
    #     raw_fine = model.net_fine(rgb_feat_sampled, ray_diff, mask)
    #     outputs_fine = raw2outputs(raw_fine, z_vals, pixel_mask,
    #                                white_bkgd=white_bkgd)
    #     ret['outputs_fine'] = outputs_fine


