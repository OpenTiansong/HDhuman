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
from collections import OrderedDict
from utils.render_ray import render_rays
from icecream import ic

def render_single_image(data,
                        # ray_sampler,
                        ray_batch,
                        model_net,
                        projector,
                        chunk_size,
                        N_samples,
                        query_z,
                        encode_geo_img,
                        train_only_depth,
                        train_only_rgb,
                        train_rgb_depth,
                        # inv_uniform=False,
                        N_importance,
                        mlp_parallel,
                        geo_cond_weights, add_img_att,
                        ):
    '''
    :param ray_sampler: RaySamplingSingleImage for this view
    :param model:  {'net_coarse': , 'net_fine': , ...}
    :param chunk_size: number of rays in a chunk
    :param N_samples: samples along each ray (for both coarse and fine model)
    :param inv_uniform: if True, uniformly sample inverse depth for coarse model
    :param N_importance: additional samples along each ray produced by importance sampling (for fine model)
    :return: {'outputs_coarse': {'rgb': numpy, 'depth': numpy, ...}, 'outputs_fine': {}}
    '''

    # all_ret = OrderedDict([('outputs_coarse', OrderedDict()),
    #                        ('outputs_fine', OrderedDict())])

    all_ret = {"render_depth": [],
               "render_rgb": []}

    N_rays = ray_batch['ray_o'].shape[0] # 762048
    # ic(N_rays)
    H, W = data["tgt"].shape[-2:]
    geometry_featmaps, imghd_featmaps = encode_geo_img(data=data, model_net=model_net)
    render_rgb, render_depth = None, None
    for i in range(0, N_rays, chunk_size):
        chunk = {}
        # for k in ray_batch:
            # if k in ['camera', 'depth_range', 'src_rgbs', 'src_cameras']:
            #     chunk[k] = ray_batch[k]
            # elif ray_batch[k] is not None:
            #     chunk[k] = ray_batch[k][i:i+chunk_size]
            # else:
            #     chunk[k] = None
        chunk["ray_o"] = ray_batch["ray_o"][i:i + chunk_size]
        chunk["ray_d"] = ray_batch["ray_d"][i:i + chunk_size]


        ret = render_rays(ray_batch=chunk,
                          data=data,
                          model_net=model_net,
                          # geo_fusion_net=geo_fusion_net,
                          # geo_mlp_net=geo_mlp_net,
                          # img_attention_net=img_attention_net,
                          # img_mlp_net=img_mlp_net,
                          projector=projector,
                          # featmaps=None,
                          geo_featmaps=geometry_featmaps,
                          img_featmaps=imghd_featmaps,
                          N_samples=N_samples,
                          inv_uniform=False,
                          N_importance=N_importance,
                          det=False,
                          white_bkgd=False,
                          train_only_depth=train_only_depth,
                          train_only_rgb=train_only_rgb,
                          train_rgb_depth=train_rgb_depth,
                          query_z=query_z,
                          mlp_parallel=mlp_parallel, geo_cond_weights=geo_cond_weights, add_img_att=add_img_att)
        # print(i)
        if not train_only_depth:
            all_ret["render_rgb"].append(ret["render_rgb"])
        all_ret["render_depth"].append(ret["render_depth"])


        # handle both coarse and fine outputs
        # cache chunk results on cpu
        # if i == 0:
        #     for k in ret['outputs_coarse']:
        #         all_ret['outputs_coarse'][k] = []
        #
        #     if ret['outputs_fine'] is None:
        #         all_ret['outputs_fine'] = None
        #     else:
        #         for k in ret['outputs_fine']:
        #             all_ret['outputs_fine'][k] = []
        #
        # for k in ret['outputs_coarse']:
        #     all_ret['outputs_coarse'][k].append(ret['outputs_coarse'][k].cpu())
        #
        # if ret['outputs_fine'] is not None:
        #     for k in ret['outputs_fine']:
        #         all_ret['outputs_fine'][k].append(ret['outputs_fine'][k].cpu())
    if not train_only_depth:
        render_rgb = torch.cat(all_ret["render_rgb"], dim=0).reshape(H, W, 3)
    render_depth = torch.cat(all_ret["render_depth"], dim=0).reshape(H, W)
    return render_rgb, render_depth

    # rgb_strided = torch.ones(ray_sampler.H, ray_sampler.W, 3)[::render_stride, ::render_stride, :]
    # # merge chunk results and reshape
    # for k in all_ret['outputs_coarse']:
    #     if k == 'random_sigma':
    #         continue
    #     tmp = torch.cat(all_ret['outputs_coarse'][k], dim=0).reshape((rgb_strided.shape[0],
    #                                                                   rgb_strided.shape[1], -1))
    #     all_ret['outputs_coarse'][k] = tmp.squeeze()
    #
    # all_ret['outputs_coarse']['rgb'][all_ret['outputs_coarse']['mask'] == 0] = 1.
    # if all_ret['outputs_fine'] is not None:
    #     for k in all_ret['outputs_fine']:
    #         if k == 'random_sigma':
    #             continue
    #         tmp = torch.cat(all_ret['outputs_fine'][k], dim=0).reshape((rgb_strided.shape[0],
    #                                                                     rgb_strided.shape[1], -1))
    #
    #         all_ret['outputs_fine'][k] = tmp.squeeze()
    #
    # return all_ret



