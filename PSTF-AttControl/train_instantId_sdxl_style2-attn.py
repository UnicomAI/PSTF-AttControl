import os
import re
import random
import argparse
from pathlib import Path
import json
import itertools
import time
from datetime import datetime
import shutil
import torch
import torch.nn.functional as F
import numpy as np
import math
import cv2
from torchvision import transforms
from PIL import Image
import PIL
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, ControlNetModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection

from ip_adapter.resampler import Resampler
from ip_adapter.utils import is_torch2_available

if is_torch2_available():
    from ip_adapter.attention_processor_style import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor_style import IPAttnProcessor, AttnProcessor
	
from pipeline_stable_diffusion_xl_instantid_style import  draw_kps

import insightface
import onnxruntime
import numpy as np
from PIL import Image
from typing import List, Union, Dict, Set, Tuple

import sys
import os
import torch

# 获取当前文件的父目录路径
module_path = os.path.abspath(os.path.join('..', 'PreciseControl'))
if module_path not in sys.path:
    sys.path.append(module_path)

from ldm.modules.e4e.psp import pSp
import argparse


def getFaceSwapModel(model_path: str):
    model = insightface.model_zoo.get_model(model_path)
    return model


def getFaceAnalyser(det_size=(320, 320),ctx_id=0):
    face_analyser = insightface.app.FaceAnalysis(name="buffalo_l", root="./checkpoints", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_analyser.prepare(ctx_id=ctx_id, det_size=det_size)
    return face_analyser



# Process the dataset by loading info from a JSON file, which includes image files, image labels, feature files, keypoint coordinates.
class Prepare():
    def __init__(self,accelerator,size=1024):
        print('###',accelerator.process_index)
        self.device=accelerator.device
        self.size = size

        checkpoint_path='../PreciseControl/weights/encoder/e4e_ffhq_encode.pt'

        ckpt = torch.load(checkpoint_path, map_location='cpu')
        opts = ckpt['opts']
        opts["test_batch_size"] = 1
        # print(opts)

        opts['checkpoint_path'] = checkpoint_path
        opts['device'] = self.device
        opts = argparse.Namespace(**opts)

        e4e_encoder = pSp(opts)
        self.stylegan_decoder = e4e_encoder.decoder
        e4e_encoder.eval()
        # e4e_encoder.to(device)
        for param in e4e_encoder.parameters():
            param.requires_grad = False

        self.stylegan_decoder.eval()
        self.stylegan_decoder.to(self.device)
        for param in self.stylegan_decoder.parameters():
            param.requires_grad = False



# #         # load face_analyser
        self.face_analyser = getFaceAnalyser(ctx_id=accelerator.process_index)

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )
    
    def reconstruct(self,codes_m):
        with torch.no_grad():
            decoded_img, _ = self.stylegan_decoder([codes_m], input_is_latent=True, randomize_noise=False, return_latents=True)
            de_img = decoded_img.permute(0, 2, 3, 1).squeeze(0)
            

            de_img = (torch.clamp((de_img+1)/2, 0, 1)*255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            
            target_img =  cv2.cvtColor(de_img, cv2.COLOR_RGB2BGR)
            target_face = self.face_analyser.get(target_img)
            kps_image_tensor=None
            if len(target_face)>0:
                face_info = sorted(target_face, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
        
                kps = face_info['kps']
                raw_image = Image.fromarray(de_img)
                kps_image = draw_kps(raw_image, kps)
                # kps_image.save("kps_image.png")

                kps_image_tensor = self.image_transforms(kps_image)
            
            
            
            return de_img,kps_image_tensor



class MyDataset(torch.utils.data.Dataset):

    def __init__(self, json_file, tokenizer, tokenizer_2, size=1024, center_crop=True,
                 t_drop_rate=0.05, i_drop_rate=0.05, s_drop_rate=0.05, ti_drop_rate=0.05, image_root_path=""):
        super().__init__()

        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.size = size
        self.center_crop = center_crop
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.s_drop_rate = s_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path

        self.data = []
        with open(json_file, 'r') as json_content:
            self.data = json.load(json_content)

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                # transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.conditioning_image_transforms = transforms.Compose(
            [
                transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )

        # self.clip_image_processor = CLIPImageProcessor()
        
        
            
        self.delta_w_dict = json.load(open("../PreciseControl/all_delta_w_dict.json"))
        self.keyset = ['pose','angry', 'black', 'eyesclose', 'bald', 'bang', 'white', 'lights', 'yellow', 'sad', 'beard', 'gender', 'smile', 'suprise', 'eyeglasses', 'eyeg']
        


    def __getitem__(self, idx):
        item = self.data[idx]
        image_file = item["file_name"]
        text = item["additional_feature"]
        bbox = item['bbox']
        landmarks = item['landmarks']
        feature_file = item["insightface_feature_file"]
        style_feature_file = item["stylegan_feature_file"]
        
        # load face feature
        # face_id_embed = torch.load(os.path.join(self.image_root_path, feature_file), map_location="cpu")
        face_id_embed = np.load(feature_file)
        face_id_embed = torch.from_numpy(face_id_embed)
        face_id_embed = face_id_embed.reshape(1, -1)
        
        face_style_embed = np.expand_dims(np.load(style_feature_file), axis=0)
        face_style_embed_tensor = torch.from_numpy(face_style_embed)
        # face_style_embed_tensor = face_style_embed_tensor#.reshape(1, -1)

        # read image
        raw_image = Image.open(os.path.join(self.image_root_path, image_file))
        
        # set cfg drop rate
        drop_feature_embed = 0
        drop_text_embed = 0
        drop_style_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_feature_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            drop_text_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate+ self.s_drop_rate):
            drop_style_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.s_drop_rate + self.ti_drop_rate):
            drop_text_embed = 1
            drop_feature_embed = 1
            drop_style_embed = 1
        
        modify_num = random.random()
        
        
        
        # draw keypoints
        kps_image = draw_kps(raw_image.convert("RGB"), landmarks)

        # original size
        original_width, original_height = raw_image.size
        original_size = torch.tensor([original_height, original_width])

        # transform raw_image and kps_image
        image_tensor = self.image_transforms(raw_image.convert("RGB"))
        kps_image_tensor = self.conditioning_image_transforms(kps_image)

        # random crop
        delta_h = image_tensor.shape[1] - self.size
        delta_w = image_tensor.shape[2] - self.size
        assert not all([delta_h, delta_w])

        if self.center_crop:
            top = delta_h // 2
            left = delta_w // 2
        else:
            top = np.random.randint(0, delta_h // 2 + 1)  # random top crop
            # top = np.random.randint(0, delta_h + 1)  # random crop
            left = np.random.randint(0, delta_w + 1)  # random crop

        # The image and kps_image must follow the same cropping to ensure that the facial coordinates correspond correctly.
        image = transforms.functional.crop(
            image_tensor, top=top, left=left, height=self.size, width=self.size
        )
        kps_image = transforms.functional.crop(
            kps_image_tensor, top=top, left=left, height=self.size, width=self.size
        )

        crop_coords_top_left = torch.tensor([top, left])


        # CFG process
        if drop_text_embed:
            text = ""
        if drop_feature_embed:
            face_id_embed = torch.zeros_like(face_id_embed)
        if drop_style_embed:
            face_style_embed_tensor = torch.zeros_like(face_style_embed_tensor)
        
        flag=0
        if modify_num<0.3 and not drop_style_embed:
            flag = 1
            key = random.choice(self.keyset)
            codes_m = face_style_embed+np.array(self.delta_w_dict[key],dtype=np.float32)*np.random.uniform(0.1, 2.5)

            face_style_embed_tensor = torch.from_numpy(codes_m)
            # face_style_embed_tensor = face_style_embed_tensor#.reshape(1, -1)
        flag_tensor = torch.tensor([flag])
        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        text_input_ids_2 = self.tokenizer_2(
            text,
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        
                

        return {
            "image": image,
            "kps_image": kps_image,
            "text_input_ids": text_input_ids,
            "text_input_ids_2": text_input_ids_2,
            "face_id_embed": face_id_embed,
            "face_style_embed": face_style_embed_tensor,
            "original_size": original_size,
            "crop_coords_top_left": crop_coords_top_left,
            "target_size": torch.tensor([self.size, self.size]),
            "flag_tensor":flag_tensor,
        }

    def __len__(self):
        return len(self.data)


def collate_fn(data):
    # print(data)
    images = torch.stack([example["image"] for example in data])
    kps_images = torch.stack([example["kps_image"] for example in data])

    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    text_input_ids_2 = torch.cat([example["text_input_ids_2"] for example in data], dim=0)
    face_id_embed = torch.stack([example["face_id_embed"] for example in data])
    face_style_embed = torch.stack([example["face_style_embed"] for example in data])
    original_size = torch.stack([example["original_size"] for example in data])
    crop_coords_top_left = torch.stack([example["crop_coords_top_left"] for example in data])
    target_size = torch.stack([example["target_size"] for example in data])
    flag_tensor = torch.stack([example["flag_tensor"] for example in data])
    return {
        "images": images,
        "kps_images": kps_images,
        "text_input_ids": text_input_ids,
        "text_input_ids_2": text_input_ids_2,
        "face_id_embed": face_id_embed,
        "face_style_embed":face_style_embed,
        "original_size": original_size,
        "crop_coords_top_left": crop_coords_top_left,
        "target_size": target_size,
        "flag_tensor": flag_tensor,
    }


class InstantIDAdapter(torch.nn.Module):
    """InstantIDAdapter"""
    def __init__(self, unet, controlnet, feature_proj_model, style_feature_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        self.feature_proj_model = feature_proj_model
        self.style_feature_proj_model =  style_feature_proj_model
        self.adapter_modules = adapter_modules
        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)
        # self.prepare=prepare
        

    def forward(self,noisy_latents, timesteps, encoder_hidden_states, unet_added_cond_kwargs, feature_embeds, style_feature_embeds, controlnet_image):
        

        face_embedding = self.feature_proj_model(feature_embeds)
        style_embedding = self.style_feature_proj_model(style_feature_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, face_embedding, style_embedding], dim=1)
        # ControlNet conditioning.
        # print(noisy_latents.shape,timesteps.shape,face_embedding.shape,style_embedding.shape,controlnet_image.shape)
        # print("unet_added_cond_kwargs:",unet_added_cond_kwargs)
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=face_embedding,  # Insightface feature
            added_cond_kwargs=unet_added_cond_kwargs,
            controlnet_cond=controlnet_image,  # keypoints image
            return_dict=False,
        )
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=unet_added_cond_kwargs,
            down_block_additional_residuals=[sample for sample in down_block_res_samples],
            mid_block_additional_residual=mid_block_res_sample,
        ).sample

        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.feature_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Check if 'latents' exists in both the saved state_dict and the current model's state_dict
        strict_load_feature_proj_model = True

        # Load state dict for feature_proj_model and adapter_modules
        self.feature_proj_model.load_state_dict(state_dict["image_proj"], strict=strict_load_feature_proj_model)
        # self.style_feature_proj_model.load_state_dict(state_dict["image_style_proj"], strict=strict_load_feature_proj_model)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=False)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.feature_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of feature_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default=None,
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model. If not specified weights are initialized from unet.",
    )

    parser.add_argument(
        "--num_tokens",
        type=int,
        default=16,
        help="Number of tokens to query from the CLIP image encoding.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=1,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default=None,
        required=True,
        help="Training data",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="",
        required=True,
        help="Training data root path",
    )
    parser.add_argument('--clip_proc_mode',
                        choices=["seg_align", "seg_crop", "orig_align", "orig_crop", "seg_align_pad",
                                 "orig_align_pad"],
                        default="orig_crop",
                        help='The mode to preprocess clip image encoder input.')

    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to CLIP image encoder",
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-ip_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--noise_offset", type=float, default=None, help="noise offset")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    num_devices = accelerator.num_processes

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    if args.controlnet_model_name_or_path:
        print("Loading existing controlnet weights")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    else:
        print("Initializing controlnet weights from unet")
        controlnet = ControlNetModel.from_unet(unet)
        
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        


    unet.to(accelerator.device, dtype=weight_dtype)  # error
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    # image_encoder.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)  # error

    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    # image_encoder.requires_grad_(False)
    controlnet.requires_grad_(False)
    # controlnet.train()

    # ip-adapter: insightface feature
    num_tokens = 16

    feature_proj_model = Resampler(
        dim=1280,
        depth=4,
        dim_head=64,
        heads=20,
        num_queries=num_tokens,
        embedding_dim=512,
        output_dim=unet.config.cross_attention_dim,
        ff_mult=4,
    )

    feature_proj_model.requires_grad_(False)

    style_feature_proj_model = Resampler(
        dim=1280,
        depth=4,
        dim_head=64,
        heads=20,
        num_queries=num_tokens,
        embedding_dim=512*18,
        output_dim=unet.config.cross_attention_dim,
        ff_mult=4,
    )
    style_feature_proj_model.requires_grad_(True)
    style_feature_proj_model.train()

    # init adapter modules
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                "to_k_style.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_style.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=num_tokens,dtype=weight_dtype)
            attn_procs[name].load_state_dict(weights)
            for name, param in attn_procs[name].named_parameters():
                if "to_k_ip" in name or "to_v_ip" in name:
                    param.requires_grad = False 
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    
    
    prepare = Prepare(accelerator)

    # Instantiate InstantIDAdapter from pretrained model or from scratch.
    ip_adapter = InstantIDAdapter(unet, controlnet, feature_proj_model, style_feature_proj_model, adapter_modules, args.pretrained_ip_adapter_path)
    ip_adapter.feature_proj_model.to(dtype=weight_dtype)
    ip_adapter.style_feature_proj_model.to(dtype=weight_dtype)
    # Register a hook function to process the state of a specific module before saving.
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # find instance of InstantIDAdapter Model.
            for i, model_instance in enumerate(models):
                if isinstance(model_instance, InstantIDAdapter):
                    # When saving a checkpoint, only save the ip-adapter and image_proj, do not save the unet.
                    ip_adapter_state = {
                        'image_proj': model_instance.feature_proj_model.state_dict(),
                        'image_style_proj': model_instance.style_feature_proj_model.state_dict(),
                        'ip_adapter': model_instance.adapter_modules.state_dict(),
                    }
                    torch.save(ip_adapter_state, os.path.join(output_dir, 'pytorch_model.bin'))
                    print(f"IP-Adapter Model weights saved in {os.path.join(output_dir, 'pytorch_model.bin')}")
                    # Save controlnet separately.
                    sub_dir = "controlnet"
                    model_instance.controlnet.save_pretrained(os.path.join(output_dir, sub_dir))
                    print(f"Controlnet weights saved in {os.path.join(output_dir, controlnet)}")

                    weights.pop(i)
                    break

    def load_model_hook(models, input_dir):
        # find instance of InstantIDAdapter Model.
        while len(models) > 0:
            model_instance = models.pop()
            if isinstance(model_instance, InstantIDAdapter):
                ip_adapter_path = os.path.join(input_dir, 'pytorch_model.bin')
                if os.path.exists(ip_adapter_path):
                    ip_adapter_state = torch.load(ip_adapter_path)
                    model_instance.feature_proj_model.load_state_dict(ip_adapter_state['image_proj'])
                    model_instance.adapter_modules.load_state_dict(ip_adapter_state['ip_adapter'])
                    sub_dir = "controlnet"
                    model_instance.controlnet.from_pretrained(os.path.join(input_dir, sub_dir))
                    print(f"Model weights loaded from {ip_adapter_path}")
                else:
                    print(f"No saved weights found at {ip_adapter_path}")




    # trainable params
    params_to_opt = itertools.chain(#ip_adapter.feature_proj_model.parameters(),
                                    ip_adapter.style_feature_proj_model.parameters(),
                                    ip_adapter.adapter_modules.parameters(),)
                                    #ip_adapter.controlnet.parameters())
    params_to_opt = filter(lambda p: p.requires_grad, params_to_opt)


    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)

    # dataloader
    train_dataset = MyDataset(args.data_json_file, tokenizer=tokenizer, tokenizer_2=tokenizer_2, size=args.resolution,
                              center_crop=args.center_crop, image_root_path=args.data_root_path)
    total_data_size = len(train_dataset)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Prepare everything with our `accelerator`.
    ip_adapter, optimizer, train_dataloader = accelerator.prepare(ip_adapter, optimizer, train_dataloader)


    global_step = 0

    normalize = transforms.Normalize([0.5], [0.5])

    # Training loop
    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(ip_adapter):

                origin = batch["images"].clone()
                origin = normalize(origin)
                
                images = batch["images"]
                flag_tensor = batch["flag_tensor"]
                face_style_embed = batch["face_style_embed"]
                kps_images = batch["kps_images"]
                
                
                
                with torch.no_grad():
                    for ii in range(images.shape[0]):
                        if flag_tensor[ii,0]==1:
                            codes_m = face_style_embed[ii,:,:]
                            target,kps_image_tensor = prepare.reconstruct(codes_m)

                            images[ii,:,:] = prepare.image_transforms(Image.fromarray(target))
                            if kps_image_tensor is not None:
                                kps_images[ii,:,:] =kps_image_tensor 

                images = normalize(images)
                
                # continue

                # Convert images to latent space
                with torch.no_grad():
                    # vae of sdxl should use fp32
                    latents = vae.encode(
                        images.to(accelerator.device, dtype=torch.float32)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    latents = latents.to(accelerator.device, dtype=weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1)).to(
                        accelerator.device, dtype=weight_dtype)

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                feat_embeds = batch["face_id_embed"].to(accelerator.device, dtype=weight_dtype)
                face_style_embeds = batch["face_style_embed"].reshape(bsz,1,-1).to(accelerator.device, dtype=weight_dtype)
                kps_images = kps_images.to(accelerator.device, dtype=weight_dtype)

                with accelerator.autocast():                
                    with torch.no_grad():
                        encoder_output = text_encoder(batch['text_input_ids'].to(accelerator.device), output_hidden_states=True)
                        text_embeds = encoder_output.hidden_states[-2]
                        encoder_output_2 = text_encoder_2(batch['text_input_ids_2'].to(accelerator.device), output_hidden_states=True)
                        pooled_text_embeds = encoder_output_2[0]
                        text_embeds_2 = encoder_output_2.hidden_states[-2]
                        text_embeds = torch.concat([text_embeds, text_embeds_2], dim=-1)  # concat

                    # add cond
                    add_time_ids = [
                        batch["original_size"].to(accelerator.device),
                        batch["crop_coords_top_left"].to(accelerator.device),
                        batch["target_size"].to(accelerator.device),
                    ]
                    add_time_ids = torch.cat(add_time_ids, dim=1).to(accelerator.device, dtype=weight_dtype)
                    unet_added_cond_kwargs = {"text_embeds": pooled_text_embeds, "time_ids": add_time_ids}

                    noise_pred = ip_adapter(noisy_latents, timesteps, text_embeds, unet_added_cond_kwargs, feat_embeds, face_style_embeds, kps_images)

                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                

                # Backpropagate
                accelerator.backward(loss)#+id_loss)
                if accelerator.sync_gradients:
                    params_to_clip = optimizer.param_groups[0]['params']
                    # params_to_clip = params_to_opt
                    accelerator.clip_grad_norm_(params_to_clip, 1.0)
                optimizer.step()
                optimizer.zero_grad()

                now = datetime.now()
                formatted_time = now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                if accelerator.is_main_process and step % 10 == 0:
                    print("[{}]: Epoch {}, global_step {}, step {}, data_time: {}, time: {}, step_loss: {}".format(
                        formatted_time, epoch, global_step, step, load_data_time, time.perf_counter() - begin,
                        avg_loss))
                    if True:
                        
                        with torch.no_grad():
                            alphas = noise_scheduler.alphas_cumprod[timesteps].sqrt().view(-1, 1, 1, 1).to(accelerator.device)
                            sigmas = (1 - noise_scheduler.alphas_cumprod[timesteps]).sqrt().view(-1, 1, 1, 1).to(accelerator.device)

                            # 通过 DDPM 蒸馏公式从噪声预测恢复图像
                            pred_x_0 = alphas * noisy_latents - sigmas * noise_pred

                            pred_x_0 = (noisy_latents - sigmas * noise_pred)/alphas
                            decoded_images = vae.decode(pred_x_0 / vae.config.scaling_factor,return_dict=False)[0]#.sample()
                        save_dir = "./images_output/decoded_images"+str(step)
                        os.makedirs(save_dir, exist_ok=True)
                        decoded_images = (decoded_images.clamp(-1, 1) + 1) / 2  # 将图像范围从 [-1, 1] 调整到 [0, 1]
                        decoded_images = (decoded_images * 255).byte().cpu().numpy()  # 转换为 [0, 255]

                        for i, img_array in enumerate(decoded_images):
                            img = Image.fromarray(img_array.transpose(1, 2, 0))  # 转换为 HWC 格式
                            img.save(os.path.join(save_dir, f"decoded_image_{i}.png"))
                        origin = (origin.clamp(-1, 1) + 1) / 2  # 将图像范围从 [-1, 1] 调整到 [0, 1]
                        origin = (origin * 255).byte().cpu().numpy()  # 转换为 [0, 255]

                        for i, img_array in enumerate(origin):
                            img = Image.fromarray(img_array.transpose(1, 2, 0))  # 转换为 HWC 格式
                            img.save(os.path.join(save_dir, f"origin_{i}.png"))
                        images = (images.clamp(-1, 1) + 1) / 2  # 将图像范围从 [-1, 1] 调整到 [0, 1]
                        images = (images * 255).byte().cpu().numpy()  # 转换为 [0, 255]

                        for i, img_array in enumerate(images):
                            if flag_tensor[i,0]==1:
                                img = Image.fromarray(img_array.transpose(1, 2, 0))  # 转换为 HWC 格式
                                img.save(os.path.join(save_dir, f"images_edit_{i}.png"))  
                            else:
                                img = Image.fromarray(img_array.transpose(1, 2, 0))  # 转换为 HWC 格式
                                img.save(os.path.join(save_dir, f"images_{i}.png"))  

            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process and global_step % args.save_steps == 0:
                    # before saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]
                            print(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                            print(f"removing checkpoints: {', '.join(removing_checkpoints)}")
    
                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)
    
                    # Instead of accelerator.save_state(save_path), use the following:

                    # Define the save path
                    save_path = os.path.join(args.output_dir, f"ip_adapter-checkpoint-{global_step}")
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                    # # Save the model state
                    # torch.save(ip_adapter.state_dict(), save_path)


                    unwrapped_model = accelerator.unwrap_model(ip_adapter)
                    ip_adapter_state = {
                        'image_proj': unwrapped_model.feature_proj_model.state_dict(),
                        'image_style_proj': unwrapped_model.style_feature_proj_model.state_dict(),
                        'ip_adapter': unwrapped_model.adapter_modules.state_dict(),
                    }
                    torch.save(ip_adapter_state, os.path.join(save_path, 'pytorch_model.bin'))
                    # Save controlnet separately.
                    sub_dir = "controlnet"
                    unwrapped_model.controlnet.save_pretrained(os.path.join(save_path, sub_dir))

                    # Optionally print confirmation
                    if accelerator.is_main_process:
                        print(f"Checkpoint saved to {save_path}")

                    

            begin = time.perf_counter()


if __name__ == "__main__":
    main()
