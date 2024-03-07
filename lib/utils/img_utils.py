import numpy as np
import cv2

def find_img_border(img):
    if len(img.shape) == 3: # rgb img
        img = np.sum(img, axis=2)

    height, width = img.shape

    for i in range(0, width):
        col_sum = np.sum(img[:, i])
        if col_sum != 0:
            min_x = i
            break

    for i in range(width - 1, 0, -1):
        col_sum = np.sum(img[:, i])
        if col_sum != 0:
            max_x = i
            break

    for i in range(0, height):
        row_sum = np.sum(img[i, :])
        if row_sum != 0:
            min_y = i
            break

    for i in range(height-1, 0, -1):
        row_sum = np.sum(img[i, :])
        if row_sum != 0:
            max_y = i
            break

    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2

    return min_x, min_y, max_x, max_y, center_x, center_y

def crop_img(img, start_x, start_y, length_x, length_y):
    """

    @param img: [H, W, 3] or [H, W]
    @param start_x:
    @param start_y:
    @param length_x:
    @param length_y:
    @return:
    """
    height, width = img.shape[0], img.shape[1]

    if len(img.shape) == 3: # rgb img
        img_pad = cv2.copyMakeBorder(img, height, height, width, width, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    if len(img.shape) == 2:
        img_pad = cv2.copyMakeBorder(img, height, height, width, width, cv2.BORDER_CONSTANT, value=[0])

    start_x_pad = start_x + width
    start_y_pad = start_y + height

    img_ret = img_pad[start_y_pad:start_y_pad+length_y, start_x_pad:start_x_pad+length_x, ...]

    return img_ret

def uncrop_rgb_img(img_crop, start_x, start_y, ori_width, ori_height):
    crop_height, crop_width, _ = img_crop.shape
    img_uncrop_pad = np.zeros((ori_height * 3, ori_width * 3, 3), dtype=np.uint8)
    start_x_pad = start_x + ori_width
    start_y_pad = start_y + ori_height
    img_uncrop_pad[start_y_pad:start_y_pad+crop_height, start_x_pad:start_x_pad+crop_width, :] = img_crop
    img_uncrop = img_uncrop_pad[ori_height:ori_height+ori_height, ori_width:ori_width+ori_width, :]
    return img_uncrop

def trans_img(img_ori):
    img = img_ori.copy()
    img = img.transpose(2, 0, 1) # [H, W, C] -> [C, H, W]
    img = (img / 255.) * 2. - 1.
    return img

def inv_trans_img(img_ori):
    img = img_ori.copy()

def pad_depth(img, pad_width=32):
    h, w = img.shape[-2:]
    # print("h, w", h, w)
    mh = h % pad_width
    ph = 0 if mh == 0 else pad_width - mh
    mw = w % pad_width
    pw = 0 if mw == 0 else pad_width - mw
    shape = [s for s in img.shape]
    shape[-2] += ph
    shape[-1] += pw
    im_p = np.zeros(shape, dtype=img.dtype)
    im_p[..., :h, :w] = img
    im = im_p
    return im

def pad_img(img, pad_width=32, flag_background_black:bool=True):
    '''
    flag_background_black: True: black, False: white
    '''
    h, w = img.shape[-2:]
    # print("h, w", h, w)
    mh = h % pad_width
    ph = 0 if mh == 0 else pad_width - mh
    mw = w % pad_width
    pw = 0 if mw == 0 else pad_width - mw
    shape = [s for s in img.shape]
    shape[-2] += ph
    shape[-1] += pw
    if flag_background_black:
        im_p = np.zeros(shape, dtype=img.dtype)
    else:
        im_p = np.full(shape, 255, dtype=img.dtype)
    im_p[..., :h, :w] = img
    im = im_p
    return im