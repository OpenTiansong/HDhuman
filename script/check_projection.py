
import script_config

import os
import os.path as osp

import numpy as np
import trimesh
import cv2 as cv

from lib.data.example_data import Example_Data_Manager

def project_to_image_space(
        mesh:trimesh.Trimesh,
        cam_in:np.ndarray,
        cam_ex:np.ndarray,
        image_bg:np.ndarray
    ) -> (np.ndarray):

    assert isinstance(mesh, trimesh.Trimesh)
    assert cam_in.shape == (3, 3), cam_in.shape
    assert cam_ex.shape == (3, 4) or cam_ex.shape == (4, 4), cam_ex.shape
    
    R = cam_ex[0:3, 0:3]
    T = cam_ex[0:3, 3]
    if cam_ex.shape == (4, 4):
        assert np.array_equal(cam_ex[3, 0:4], np.array([0, 0, 0, 1], dtype=cam_ex.dtype)), cam_ex[3, 0:4]
        
    verts_cam = np.matmul(np.matmul(mesh.vertices, R.T) + T, cam_in.T)
    verts_image = verts_cam[:, 0:2] / verts_cam[:, 2:3]

    image_vis = image_bg.copy()
    for vert_image in verts_image:
        if 0 <= vert_image[0] and vert_image[0] < W and 0 <= vert_image[1] and vert_image[1] < H:
            image_vis[H - 1 - int(vert_image[1]), int(vert_image[0])] = np.array([255, 255, 255], dtype=np.uint8)
    return image_vis


if __name__ == '__main__':
    path_data_root = script_config.path_folder_example_data
    assert osp.isdir(path_data_root), path_data_root

    path_output_check_camera_param = osp.join(path_data_root, 'check_camera')
    os.makedirs(path_output_check_camera_param, exist_ok=True)

    dm_example = Example_Data_Manager(path_root=path_data_root)

    list_name_subject = dm_example.get_name_list()

    NUM_VIEW = 6

    for name_subject in list_name_subject:
        path_mesh, list_path_image, list_cameras_in, list_cameras_ex = dm_example.get_data_n_view_camera_and_paths_for_rendering(name_subject, NUM_VIEW)

        path_folder_output_per_subject = osp.join(path_output_check_camera_param, name_subject)
        os.makedirs(path_folder_output_per_subject, exist_ok=True)

        mesh = trimesh.load_mesh(path_mesh)
        num_view = len(list_path_image)
        assert num_view == len(list_path_image), (num_view, len(list_path_image))
        assert num_view == len(list_cameras_in), (num_view, len(list_cameras_in))
        assert num_view == len(list_cameras_ex), (num_view, len(list_cameras_ex))

        for count_view in range(num_view):
            path_image = list_path_image[count_view]
            cam_in = list_cameras_in[count_view]
            cam_ex = list_cameras_ex[count_view]

            image_bg = cv.imread(path_image)
            H, W, ____ = image_bg.shape

            image_projection_vis = project_to_image_space(mesh, cam_in, cam_ex, image_bg)
            path_image_projection_vis = osp.join(path_folder_output_per_subject, '%02d.png' % count_view)
            cv.imwrite(path_image_projection_vis, image_projection_vis)