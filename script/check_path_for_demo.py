
import script_config

import os
import os.path as osp

import numpy as np
import trimesh
import cv2 as cv

from lib.data.example_data import Example_Data_Manager


def check_path_of_exmaple_data():
    path_data_root = script_config.path_folder_example_data
    assert osp.isdir(path_data_root), path_data_root

    dm_example = Example_Data_Manager(path_root=path_data_root)

    NUM_VIEW = 360
    for name_subject in dm_example.get_name_list():
        path_mesh, list_path_image, list_cameras_in, list_cameras_ex = dm_example.get_data_n_view_camera_and_paths_for_rendering(name_subject, NUM_VIEW)

        #check all paths
        assert osp.isfile(path_mesh), path_mesh
        for path_image in list_path_image:
            assert osp.isfile(path_image), path_image

def check_path_of_checkpoints():
    path_folder_checkpoints = script_config.path_folder_checkpoints
    assert osp.isdir(path_folder_checkpoints), path_folder_checkpoints

    list_filename_checkpoint = [
        'encode_net_iter_351000',
        'encode_optim_iter_351000',
        'render_net_iter_351000',
        'render_optim_iter_351000'
    ]

    for filename in list_filename_checkpoint:
        path_checkpoint = osp.join(path_folder_checkpoints, filename)
        assert osp.isfile(path_checkpoint), path_checkpoint


if __name__ == '__main__':
    check_path_of_exmaple_data()
    check_path_of_checkpoints()