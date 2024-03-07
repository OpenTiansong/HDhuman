import numpy as np
from PIL.Image import Image



def get_visibility_map(tgt_depth, tgt_idx):
    height, width = tgt_depth.shape
    visibility_map = np.full((height, width), 127, dtype=np.uint8)

    depth_mask = (tgt_depth != 0)
    visibility_map[depth_mask] = 0
    for pix in tgt_idx:
        pix_height = pix // width
        pix_width = pix % width
        visibility_map[pix_height, pix_width] = 255
    # visibility_map = visibility_map[..., None]
    # print(visibility_map.shape)
    # visibility_map = visibility_map.repeat(3, axis=2)
    return visibility_map # [H, W]

def cat_render_visibility(img_render, visibility_map):
    img_render_visiblility = np.concatenate([img_render, visibility_map], axis=1)
    img_render_visiblility = Image.fromarray(img_render_visiblility)

def find_img_border(img):
    if len(img.shape) == 3: # rgb img
        img = np.sum(img, axis=2)

    height, width = img.shape

    min_x = 0
    for i in range(0, width):
        col_sum = np.sum(img[:, i])
        if col_sum != 0:
            min_x = i
            break

    max_x = width - 1
    for i in range(width - 1, 0, -1):
        col_sum = np.sum(img[:, i])
        if col_sum != 0:
            max_x = i
            break

    min_y = 0
    for i in range(0, height):
        row_sum = np.sum(img[i, :])
        if row_sum != 0:
            min_y = i
            break

    max_y = height - 1
    for i in range(height-1, 0, -1):
        row_sum = np.sum(img[i, :])
        if row_sum != 0:
            max_y = i
            break

    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2

    return min_x, min_y, max_x, max_y, center_x, center_y

# def crop_img(img, start_x, start_y, length_x, length_y):
#     height, width = img.shape[0], img.shape[1]
#
#     if len(img.shape) == 3: # rgb img
#         img_pad = cv2.copyMakeBorder(img, height, height, width, width, cv2.BORDER_CONSTANT, value=[0, 0, 0])
#     if len(img.shape) == 2:
#         img_pad = cv2.copyMakeBorder(img, height, height, width, width, cv2.BORDER_CONSTANT, value=[0])
#
#     start_x_pad = start_x + width
#     start_y_pad = start_y + height
#
#     img_ret = img_pad[start_y_pad:start_y_pad+length_y, start_x_pad:start_x_pad+length_x, ...]
#
#     return img_ret