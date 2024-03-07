
from typing import List, Dict, Tuple, Union, Optional

import os
import os.path as osp

import numpy as np
import cv2 as cv

import trimesh

import json

class Example_Data_Manager():
    def __init__(self, path_root:str) -> None:
        self.path_root = path_root
        self.list_name = Example_Data_Manager.init_list_name(self.path_root)

    def get_name_list(self) -> (List[str]):
        return self.list_name

    def get_data_n_view_camera_and_paths_for_rendering(self, name_subject:str, num_view:int):
        path_mesh = Example_Data_Manager.get_path_mesh(self.path_root, name_subject)
        list_path_image = Example_Data_Manager.get_list_path_images(self.path_root, name_subject, num_view)
        list_cameras_in, list_cameras_ex = Example_Data_Manager.get_list_cameras(self.path_root, name_subject, num_view)

        assert num_view == len(list_path_image), len(list_path_image)
        assert num_view == len(list_cameras_in), len(list_cameras_in)
        assert num_view == len(list_cameras_ex), len(list_cameras_ex)

        return path_mesh, list_path_image, list_cameras_in, list_cameras_ex

    @staticmethod
    def get_list_index_view_by_num_view(num_view:int) -> (List[int]):
        assert num_view <= 360
        list_index_view = list(range(0, 360, 360//num_view))
        return list_index_view

    @staticmethod
    def get_list_path_images(path_root:str, name_subject:str, num_view:int) -> (List[str]):
        path_folder_image = osp.join(path_root, name_subject, 'rendering_as_gt')
        assert osp.isdir(path_folder_image), path_folder_image

        list_index_view = Example_Data_Manager.get_list_index_view_by_num_view(num_view)

        list_path = []
        for index_view in list_index_view:
            path_image = osp.join(path_folder_image, '%03d.png' % index_view)
            list_path.append(path_image)

        return list_path
    
    @staticmethod
    def get_list_cameras(path_root:str, name_subject:str, num_view:int) -> (Tuple[List[np.ndarray], List[np.ndarray]]):
        path_camera = osp.join(path_root, name_subject, 'cameras.json')
        assert osp.isfile(path_camera), path_camera
        with open(path_camera, 'r') as f:
            dict_camera_param_as_nested_list = json.load(f)

        list_index_view = Example_Data_Manager.get_list_index_view_by_num_view(num_view)
        list_camera_in = []
        list_camera_ex = []

        assert set(dict_camera_param_as_nested_list.keys()) == {'Ks', 'Rs', 'Ts'}, dict_camera_param_as_nested_list.keys()
        assert len(dict_camera_param_as_nested_list['Ks']) == 360, len(dict_camera_param_as_nested_list['Ks'])
        assert len(dict_camera_param_as_nested_list['Rs']) == 360, len(dict_camera_param_as_nested_list['Rs'])
        assert len(dict_camera_param_as_nested_list['Ts']) == 360, len(dict_camera_param_as_nested_list['Ts'])

        for index_view in list_index_view:
            K:np.ndarray = np.array(dict_camera_param_as_nested_list['Ks'][index_view])
            R:np.ndarray = np.array(dict_camera_param_as_nested_list['Rs'][index_view])
            T:np.ndarray = np.array(dict_camera_param_as_nested_list['Ts'][index_view])

            assert K.shape == (3, 3), K.shape
            assert R.shape == (3, 3), T.shape
            assert T.shape == (3,), T.shape

            RT_44 = np.identity(4)
            RT_44[0:3, 0:3] = R
            RT_44[0:3, 3] = T

            list_camera_in.append(K)
            list_camera_ex.append(RT_44)

        return list_camera_in, list_camera_ex

    @staticmethod
    def get_path_mesh(path_root:str, name_subject:str, type_mesh:str='recon') -> (str):
        path_mesh = osp.join(path_root, name_subject, '%s.obj' % type_mesh)
        return path_mesh

    @staticmethod
    def init_list_name(path_base) -> (List[str]):
        list_foldername:List[str] = sorted(os.listdir(path_base))
        list_name:List[str] = []
        for foldername in list_foldername:
            if foldername.isdigit():
                name_as_integer = int(foldername)
                if 100000000000000 < name_as_integer and name_as_integer < 900000000000000:
                    list_name.append(foldername)
        assert len(list_name) > 0, list_name
        return list_name