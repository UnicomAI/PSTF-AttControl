import os
import cv2
import json
import torch
import random
import numpy as np
from PIL import Image

import diffusers
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from diffusers import DPMSolverMultistepScheduler

import torchvision.transforms as transforms
from insightface.app import FaceAnalysis

import ip_adapter.config as config
from pipeline_stable_diffusion_xl_instantid_style import StableDiffusionXLInstantIDPipeline, draw_kps

from util import init_e4e_encoder, get_wlatents, resize_img, align_face

device_pipe = 'cuda:0'
device_e4e = 'cuda:1'

# prepare models under ./checkpoints
weights = "./output/ffhq_style_finetune_attn_modify_v2/ip_adapter-checkpoint-230000/pytorch_model.bin"
controlnet_path = './models/models--InstantX--InstantID/ControlNetModel'

# Load IdentityNet & Pipe
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
     "./models/models--stabilityai--stable-diffusion-xl-base-1.0", 
     controlnet=controlnet, torch_dtype=torch.float16
)
pipe.to(device_pipe)
pipe.load_ip_adapter_instantid(weights)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.scheduler.config.solver_order = 2  # '2' for DPM++ 2M
pipe.scheduler.config.algorithm_type = "dpmsolver++"

# --------- 初始化 e4e 及 decoder ---------
checkpoint_path = '../PreciseControl/weights/encoder/e4e_ffhq_encode.pt'
e4e_encoder, latent_avg = init_e4e_encoder(checkpoint_path, device=device_e4e)

stylegan_decoder = e4e_encoder.decoder
stylegan_decoder.eval()
stylegan_decoder.to(device_e4e)
for param in stylegan_decoder.parameters():
    param.requires_grad = False
    
def reconstruct(codes_m):
    with torch.no_grad():
        decoded_img, _ = stylegan_decoder([codes_m], input_is_latent=True, randomize_noise=False, return_latents=True)
        de_img = decoded_img.permute(0, 2, 3, 1).squeeze(0)
        de_img = (torch.clamp((de_img+1)/2, 0, 1)*255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        return de_img

# prepare 'antelopev2' under ./models
app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

transform = transforms.Compose([
    transforms.Resize(1024),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# 全局设定
config.attention_map_save = None 
seed = 889469
print(f"Seed: {seed}")
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

delta_w_dict = json.load(open("../PreciseControl/all_delta_w_dict.json"))

def parse(file):
    from facexlib.parsing import init_parsing_model
    from facexlib.utils.face_restoration_helper import FaceRestoreHelper
    from torchvision.transforms.functional import normalize

    device_parse = 'cuda:0'
    face_helper = FaceRestoreHelper(
                upscale_factor=1, face_size=512, crop_ratio=(1, 1),
                det_model='retinaface_resnet50', save_ext='png', device=device_parse)
    face_helper.face_parse = init_parsing_model(model_name='bisenet', device=device_parse)
    face_helper.clean_all()

    def img2tensor(imgs, bgr2rgb=True, float32=True):
        def _totensor(img, bgr2rgb, float32):
            if img.shape[2] == 3 and bgr2rgb:
                if img.dtype == 'float64':
                    img = img.astype('float32')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img.transpose(2, 0, 1))
            if float32:
                img = img.float()
            return img
        if isinstance(imgs, list):
            return [_totensor(img, bgr2rgb, float32) for img in imgs]
        return _totensor(imgs, bgr2rgb, float32)

    def to_gray(img):
        x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        x = x.repeat(1, 3, 1, 1)
        return x

    id_image = file
    image = Image.open(id_image).convert('RGB')
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    input_img = img2tensor(image_bgr, bgr2rgb=True).unsqueeze(0) / 255.0
    input_img = input_img.to(device_parse)
    parsing_out = face_helper.face_parse(normalize(input_img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
    parsing_out = parsing_out.argmax(dim=1).squeeze(0)

    num_labels = parsing_out.max().item() + 1
    color_map = np.array([[int(255 * idx / (num_labels - 1))] * 3 for idx in range(num_labels)], dtype=np.uint8)

    parsing_colored = np.zeros((parsing_out.shape[0], parsing_out.shape[1], 3), dtype=np.uint8)
    for label in range(num_labels):
        if (parsing_out.cpu().numpy() == label).sum() > 0:
            print(label, color_map[label])
        parsing_colored[parsing_out.cpu().numpy() == label] = color_map[label]

    colored_image = Image.fromarray(parsing_colored)
    colored_image.save("region_colored_image.png")

    from PIL import ImageDraw
    draw = ImageDraw.Draw(colored_image)
    for label in range(num_labels):
        mask = (parsing_out.cpu().numpy() == label)
        if mask.any():
            y, x = np.mean(np.argwhere(mask), axis=0).astype(int)
            draw.text((x, y), str(label), fill=(255, 255, 255))

    colored_image.save("region_colored_with_labels.png")

    bg_label = [0, 16, 18, 8, 9, 14, 15, 17]
    bg = sum(parsing_out == i for i in bg_label).bool()
    
    white_image = torch.ones_like(input_img)
    face_features_image = torch.where(bg, white_image, to_gray(input_img))

    bg_inverted = ~bg 
    bg_np = bg_inverted.squeeze().cpu().numpy().astype('uint8') * 255
    bg_image = Image.fromarray(bg_np)
    bg_image.save(file.replace('.png','_mask.png'))


def face_gen(file, attkeylist):
    prompt = 'A person'
    face_image = load_image(file)
    name = file.split('/')[-1].split('.')[0]
    savepath = "./images-mask-v6-lady-output/"
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    resized_image = resize_img(face_image)
    
    background = Image.new('RGB', (2048, 2048), (255, 255, 255))
    x_offset = (2048 - 1024) // 2
    y_offset = (2048 - 1024) // 2
    background.paste(resized_image, (x_offset, y_offset))

    face_info = app.get(cv2.cvtColor(np.array(background), cv2.COLOR_RGB2BGR))
    print(len(face_info))
    if len(face_info) > 0:
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]

        face_info['bbox'] = face_info['bbox'] - x_offset
        face_info['kps'] = face_info['kps'] - [x_offset, y_offset]

        face_emb = face_info['embedding']
        ldm = face_info['landmark_2d_106'] - [x_offset, y_offset]
        
        alignimg = align_face(resized_image, ldm)
        aligned_face = transform(alignimg).permute(1,2,0)
        aligned_face = transforms.Resize((256, 256))(aligned_face.unsqueeze(0).permute(0, 3, 1, 2)).to(device_e4e)

        codes = get_wlatents(aligned_face, e4e_encoder, is_cars=False)

    resized_image = resize_img(alignimg)
    alignimg.save(savepath+'/compare_att_'+name+"_codesimg.png")
    parse(savepath+'/compare_att_'+name+"_codesimg.png")
    
    bg_image = Image.open(savepath+'/compare_att_'+name+"_codesimg_mask_v2.png").convert('L')
    bg_np = np.array(bg_image)
    config.bg = torch.tensor(bg_np / 255.0)
    
    face_info2 = app.get(cv2.cvtColor(np.array(resized_image), cv2.COLOR_RGB2BGR))
    print(len(face_info2))
    if len(face_info2) > 0:
        face_info2 = sorted(face_info2, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
        kps_image = draw_kps(resized_image.convert("RGB"), face_info2['kps'])
        kps_image.save(savepath+'/compare_att_'+name+"_kps.png")

    negative_prompt = "ugly, deformed, noisy, blurry, low contrast"
    print(prompt)

    for attkey in attkeylist:
        config.attention_map_save = None 
        
        image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            image_embeds=face_emb,
            image_style_embeds=codes+torch.from_numpy(np.array(delta_w_dict['white'],dtype=np.float32)).to(device_pipe)*0.1,
            image=kps_image,
            controlnet_conditioning_scale=0.8,
            ip_adapter_scale=1,
            style_adapter_scale=0.6,
        ).images[0]
        image.save(savepath+'/compare_att_'+name+'_'+attkey+str(round(0.0, 2))+".png")

        for ii in np.arange(0.2, 3, 0.2):
            image = pipe(
                prompt,
                negative_prompt=negative_prompt,
                image_embeds=face_emb,
                image_style_embeds=codes+torch.from_numpy(np.array(delta_w_dict[attkey],dtype=np.float32)).to(device_pipe)*ii,
                image=kps_image,
                controlnet_conditioning_scale=0.8,
                ip_adapter_scale=1,
                style_adapter_scale=0.6,
            ).images[0]
            image.save(savepath+'/compare_att_'+name+'_'+attkey+str(round(ii, 2))+".png")


if __name__ == "__main__":
    attkeylist=['eyeglasses','gender','suprise','white']
    imgfile = "./examples/lifeifei.jpg"
    face_gen(imgfile, attkeylist)