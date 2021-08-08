import os
import numpy as np
from imageio import imread, imsave
import cv2


def save_uint8(name, img):
    if img.dtype != np.uint8:
        img = (img.clip(0, 1) * 255).astype(np.uint8)
    imsave(name, img)


def save_uint16(img_name, img):
    """img in [0, 1]"""
    img = img.clip(0, 1) * 65535
    img = img[:,:,[2,1,0]].astype(np.uint16)
    cv2.imwrite(img_name, img)


def save_hdr(name, hdr):
    #print(name)
    hdr = hdr[:, :, [2, 1, 0]].astype(np.float32)
    cv2.imwrite(name, hdr)


def mulog_transform(in_tensor, mu=5000.0):
    denom = np.log(1.0 + mu)
    out_tensor = np.log(1.0 + mu * in_tensor) / denom 
    return out_tensor


def crop_img_border(img, border=10):
    h, w, c = img.shape
    img = img[border: h - border, border: w - border, :]
    return img


def read_hdr(filename, use_cv2=True):
    ext = os.path.splitext(filename)[1]
    if use_cv2:
        hdr = cv2.imread(filename, -1)[:,:,::-1].clip(0)
    elif ext == '.hdr':
        hdr = cv2.imread(filename, -1)
    elif ext == '.npy':
        hdr = np.load(filenmae) 
    else:
        raise_not_defined()
    return hdr


def ldr_to_hdr(img, expo, gamma=2.2):
    img = img.clip(0, 1)
    img = np.power(img, gamma) # linearize
    img /= expo
    return img


def hdr_to_ldr(img, expo, gamma=2.2):
    img = np.power(img * expo, 1.0 / gamma)
    img = img.clip(0, 1)
    return img

