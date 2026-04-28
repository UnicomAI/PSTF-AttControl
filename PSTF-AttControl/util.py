import os
import sys
import torch
import numpy as np
import PIL
import PIL.Image
from PIL import Image
import scipy
import scipy.ndimage
import argparse

# 获取当前文件的父目录路径并加入环境变量
module_path = os.path.abspath(os.path.join('..', 'PreciseControl'))
if module_path not in sys.path:
    sys.path.append(module_path)

from ldm.modules.e4e.psp import pSp

def init_e4e_encoder(checkpoint_path, device='cuda:0'):
    """初始化 e4e_encoder 并固化权重"""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts["test_batch_size"] = 1
    opts['checkpoint_path'] = checkpoint_path
    opts['device'] = device
    opts = argparse.Namespace(**opts)

    e4e_encoder = pSp(opts)
    e4e_encoder.eval()
    e4e_encoder.to(device)
    for param in e4e_encoder.parameters():
        param.requires_grad = False
        
    latent_avg = ckpt['latent_avg'].to(device)
    return e4e_encoder, latent_avg

def get_wlatents(x, e4e_encoder, is_cars=False):
    with torch.no_grad():
        e4e_encoder.eval()
        codes = e4e_encoder.encoder(x)
        e4e_encoder.latent_avg = e4e_encoder.latent_avg.to(codes.device)
        if e4e_encoder.opts.start_from_latent_avg:
            if codes.ndim == 2:
                codes = codes + e4e_encoder.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
            else:
                codes = codes + e4e_encoder.latent_avg.repeat(codes.shape[0], 1, 1)
        if codes.shape[1] == 18 and is_cars:
            codes = codes[:, :16, :]
    return codes

def resize_img(input_image, target_size=1024, pad_to_max_side=True, mode=Image.BILINEAR, base_pixel_number=64):
    w, h = input_image.size
    # Resize so that the longest side is resized to target_size
    ratio = target_size / max(w, h)
    w_resize_new = round(w * ratio)
    h_resize_new = round(h * ratio)
    
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    # Adjust the new width and height to be divisible by base_pixel_number
    w_resize_new = (w_resize_new // base_pixel_number) * base_pixel_number
    h_resize_new = (h_resize_new // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    # Padding to center the image if needed
    if pad_to_max_side:
        max_side = target_size
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255  # White background
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    
    return input_image

def align_face(img, lm):
    """
    :param img: PIL Image
    :param lm: landmarks
    :return: PIL Image
    """
    lm_eye_left = lm[[33, 35, 36, 37, 39, 40, 41, 42],:]
    lm_eye_right = lm[[87, 89, 90, 91, 93, 94, 95, 96],:]
    lm_mouth_outer = lm[[56,57,54,60,59,62,69,68,72,65,53,64],:]
    
    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2
    
    output_size = 512
    transform_size = 512
    enable_padding = True

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        # Used ANTIALIAS to keep compatibility with original logic
        img = img.resize(rsize, getattr(PIL.Image, 'ANTIALIAS', PIL.Image.Resampling.LANCZOS))
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), getattr(PIL.Image, 'ANTIALIAS', PIL.Image.Resampling.LANCZOS))

    return img

def convert_ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(i) for i in obj]
    else:
        return obj