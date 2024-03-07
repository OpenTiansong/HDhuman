from typing import List, Tuple, Dict, Union, Optional
#for import "lib" of hdhuman
import script_config

import sys
import os
import os.path as osp
import time

#for import "ext" of StableViewSynthesis
path_svs_repo = script_config.path_folder_svs_repo
assert osp.isdir(path_svs_repo), path_svs_repo
sys.path.append(path_svs_repo)

import torch
from lib.data.example_data import Example_Data_Manager
from lib.test.test_render_api import render_one_subject_from_path_and_write_to_folder
from lib.test.test_resource_manager import Test_Resource_Manager

if __name__ == '__main__':
    os.environ['TI_ENABLE_CUDA'] = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    DEVICE = 'cuda:0'
    NUM_VIEW_INPUT = 6
    NUM_VIEW_OUTPUT = 180 #can be factors of 360, there are 360 pre-defined views
    path_ckpt_encode = osp.join(script_config.path_folder_checkpoints, "encode_net_iter_351000")
    path_ckpt_render = osp.join(script_config.path_folder_checkpoints, "render_net_iter_351000")
    assert osp.isfile(path_ckpt_encode), path_ckpt_encode
    assert osp.isfile(path_ckpt_render), path_ckpt_render

    test_rm = Test_Resource_Manager(
        path_ckpt_encode=path_ckpt_encode,
        path_ckpt_render=path_ckpt_render,
        device=torch.device(DEVICE)
    )

    path_folder_example_data = script_config.path_folder_example_data
    example_dm = Example_Data_Manager(path_folder_example_data)
    path_folder_render_output = osp.join(path_folder_example_data, 'render_output')
    os.makedirs(path_folder_render_output, exist_ok=True)

    for name_subject in example_dm.get_name_list():
        path_mesh, list_path_image, list_cameras_in_src, list_cameras_ex_src = example_dm.get_data_n_view_camera_and_paths_for_rendering(
            name_subject, 
            num_view=NUM_VIEW_INPUT
        )

        path_folder_output_per_subject = osp.join(path_folder_render_output, name_subject)
        os.makedirs(path_folder_output_per_subject, exist_ok=True)

        ____, ____, list_cameras_in_dst, list_cameras_ex_dst = example_dm.get_data_n_view_camera_and_paths_for_rendering(
            name_subject, 
            num_view=NUM_VIEW_OUTPUT
        )

        render_one_subject_from_path_and_write_to_folder(
            path_mesh=path_mesh,
            list_path_image_src=list_path_image,
            list_camera_in_src=list_cameras_in_src,
            list_camera_ex_src=list_cameras_ex_src,
            list_camera_in_dst=list_cameras_in_dst,
            list_camera_ex_dst=list_cameras_ex_dst,
            test_rm=test_rm,
            path_folder_output=path_folder_output_per_subject
        )

        
    

