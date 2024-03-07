
from typing import List, Tuple, Dict, Union, Optional

import numpy as np
import trimesh
import os.path as osp

from PIL import Image
import lib.taichi_three as t3
from lib.test.test_render import render_one_subject
from lib.test.test_resource_manager import Test_Resource_Manager


def check_list_camera_parameters(
        list_camera_in:List[np.ndarray],
        list_camera_ex:List[np.ndarray],
        num_view:int
    ) -> (None):
    '''
    check the format of the camera parameters
    '''
    #check the format of the camera parameters
    for count_view in range(num_view):
        assert list_camera_in[count_view].shape == (3, 3), list_camera_in[count_view].shape
        assert list_camera_ex[count_view].shape == (3, 4) or list_camera_ex[count_view].shape == (4, 4), list_camera_ex[count_view].shape

def convert_camera_in(list_camera_in:List[np.ndarray], height:int) -> List[np.ndarray]:
    '''
    source format: check in HDhuman/script/check_projection.py, as the input of render_one_subject_from_path_and_write_to_folder(test_render_api.py)
    target format: used in render_one_subject (test_render.py)
    '''
    #convert the camera intrinsic parameters
    list_camera_in_ret = []
    for camera_in in list_camera_in:
        camera_in_ret = camera_in.copy()
        camera_in_ret[1, 1] = camera_in[1, 1] * -1.0
        camera_in_ret[1, 2] = camera_in[1, 2] * -1.0 + height
        list_camera_in_ret.append(camera_in_ret)
    return list_camera_in_ret

def combine_in_and_ex_to_dict(list_camera_in:List[np.ndarray], list_camera_ex:List[np.ndarray]) -> List[Dict[str, np.ndarray]]:
    '''
    combine the camera intrinsic and extrinsic parameters to a dictionary
    make the shape of camera extrinsic parameters to (3, 4)
    '''
    assert len(list_camera_in) == len(list_camera_ex), (len(list_camera_in), len(list_camera_ex))

    list_camera = []
    for count_view in range(len(list_camera_in)):
        assert list_camera_in[count_view].shape == (3, 3), list_camera_in[count_view].shape
        assert list_camera_ex[count_view].shape == (3, 4) or \
                list_camera_ex[count_view].shape == (4, 4), list_camera_ex[count_view].shape
        camera = {
            "extrinsic": list_camera_ex[count_view][0:3, 0:4],
            "intrinsic": list_camera_in[count_view]
        }
        list_camera.append(camera)
    return list_camera

def render_one_subject_from_path_and_write_to_folder(
        path_mesh:str,
        list_path_image_src:List[str],
        list_camera_in_src:List[np.ndarray],
        list_camera_ex_src:List[np.ndarray],
        list_camera_in_dst:List[np.ndarray],
        list_camera_ex_dst:List[np.ndarray],
        test_rm:Test_Resource_Manager,
        path_folder_output:str,
    ):
    '''
    take
    * path_mesh
    * list_path_image_src
    * list_camera_in_src
    * list_camera_ex_src
    as input
    render the mesh with the given camera parameters
    * list_camera_in_dst
    * list_camera_ex_dst
    and write the rendered images to the given folder
    * path_folder_output_per_subject
    '''

    #check the format of the source camera parameters
    NUM_VIEW_INPUT = len(list_path_image_src)
    check_list_camera_parameters(list_camera_in_src, list_camera_ex_src, NUM_VIEW_INPUT)
        
    #check the format of the destination camera parameters
    NUM_VIEW_OUTPUT = len(list_camera_in_dst)
    check_list_camera_parameters(list_camera_in_dst, list_camera_ex_dst, NUM_VIEW_OUTPUT)

    #load the mesh
    mesh:Dict[np.ndarray] = t3.readobj(path_mesh)
    
    #load the source images as RGB arrays
    list_image_src = [np.array(Image.open(path_image)) for path_image in list_path_image_src]

    H, W, ___ = list_image_src[0].shape
    for image_src in list_image_src:
        assert image_src.shape == (H, W, 3), image_src.shape
    
    #convert the format of the camera intrinsics
    list_camera_in_dst_converted = convert_camera_in(list_camera_in_dst, H)
    list_camera_in_src_converted = convert_camera_in(list_camera_in_src, H)

    list_dict_camera_src = combine_in_and_ex_to_dict(list_camera_in_src_converted, list_camera_ex_src)
    list_dict_camera_dst = combine_in_and_ex_to_dict(list_camera_in_dst_converted, list_camera_ex_dst)

    list_path_save = [osp.join(path_folder_output, f'rendered_{count_view}.png') for count_view in range(NUM_VIEW_OUTPUT)]

    render_one_subject(
        encode_net=test_rm.get_encoder_net(),
        render_net=test_rm.get_render_net(),
        src_cameras=list_dict_camera_src,
        render_cameras=list_dict_camera_dst,
        mesh=mesh,
        src_img_list=list_image_src,
        list_path_save=list_path_save,
        flag_map_to_list_bl=True,
        flag_map_to_list_bl_seq=False,
        ori_height=H,
        ori_width=W,
        scale=1.0,
        device=test_rm.get_device()
    )
    


        

    

    