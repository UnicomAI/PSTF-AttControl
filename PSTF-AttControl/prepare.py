import os
import cv2
import glob
import json
import torch
import numpy as np
from PIL import Image
from diffusers.utils import load_image
import torchvision.transforms as transforms
from insightface.app import FaceAnalysis

# --------- 引用公共库 ---------
from util import init_e4e_encoder, get_wlatents, align_face, convert_ndarray_to_list

device = 'cuda:0'
checkpoint_path = '../PreciseControl/weights/encoder/e4e_ffhq_encode.pt'

# 初始化 E4E Encoder
e4e_encoder, latent_avg = init_e4e_encoder(checkpoint_path, device=device)

# prepare 'antelopev2' under ./models
app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

transform = transforms.Compose([
    transforms.Resize(1024),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

info = []
errorlist = []
iii = 0

for img in glob.glob("./ffhq-dataset/*.png"):
    face_image = load_image(img)
    assert face_image.size == (1024, 1024)
    resized_image = face_image

    # 创建一个 2048x2048 的白色背景
    background = Image.new('RGB', (2048, 2048), (255, 255, 255))
    x_offset = (2048 - 1024) // 2
    y_offset = (2048 - 1024) // 2

    # 将缩放后的图片粘贴到白色背景的中央
    background.paste(resized_image, (x_offset, y_offset))
    
    face_info = app.get(cv2.cvtColor(np.array(background), cv2.COLOR_RGB2BGR))
    if len(face_info) > 0:
        # only use the maximum face
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] 
        
        face_info['bbox'] = face_info['bbox'] - x_offset
        face_info['kps'] = face_info['kps'] - [x_offset, y_offset]
        
        face_emb = face_info['embedding']
        oneface = {
            "file_name": img,
            "additional_feature": "a person",
            "bbox": face_info['bbox'],
            "landmarks": face_info['kps'] - [x_offset, y_offset]
        }
        
        ldm = face_info['landmark_2d_106'] - [x_offset, y_offset]
        alignimg = align_face(face_image, ldm)
        
        aligned_face = transform(alignimg).permute(1,2,0)
        aligned_face = transforms.Resize((256, 256))(aligned_face.unsqueeze(0).permute(0, 3, 1, 2)).to(device)

        codes = get_wlatents(aligned_face, e4e_encoder, is_cars=False)
        
        np.save(img.replace('.png','.npy'), face_emb)
        np.save(img.replace('.png','_style.npy'), codes.cpu().numpy().squeeze(0))
        
        oneface["insightface_feature_file"] = img.replace('.png','.npy')
        oneface["stylegan_feature_file"] = img.replace('.png','_style.npy')
        
        info.append(convert_ndarray_to_list(oneface))
    else:
        errorlist.append(img)
    
    if iii % 10000 == 0:
        print(f"Processed: {iii}")
    iii += 1
        
np.save('errorlist.npy', errorlist)
with open('output_data.json', 'w') as json_file:
    json.dump(info, json_file, indent=4)