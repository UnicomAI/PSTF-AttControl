# SDXL Model
export MODEL_NAME="models/models--stabilityai--stable-diffusion-xl-base-1.0"

# CLIP Model
export ENCODER_NAME="IP-Adapter/sdxl_models/image_encoder"
# pretrained InstantID model
export ADAPTOR_NAME='models/models--InstantX--InstantID/ip-adapter.bin'
export CONTROLNET_NAME='models/models--InstantX--InstantID/ControlNetModel'

# Dataset
export ROOT_DATA_DIR="/"
# This json file ' format:
# {"file_name": "./ffhq-dataset/1111.png","additional_feature": "a person", 
# "bbox": [-31.329412311315536, 160.6865997314453, 496.19240215420723, 688.1674156188965],
# "landmarks": [[133.046875, 318], [319.3125, 318], [221.0625, 422], [153.515625, 535], [298.84375, 537]],
# "insightface_feature_file": "./ffhq-dataset/1111.npy",
# "stylegan_feature_file": "./ffhq-dataset/1111_style.npy"
export JSON_FILE="./output_data_modify.json"


# Output
export OUTPUT_DIR="./output/ffhq_style_finetune_attn_modify"


echo "OUTPUT_DIR: $OUTPUT_DIR"
#accelerate launch --num_processes 8 --multi_gpu --mixed_precision "fp16" \
#CUDA_VISIBLE_DEVICES=0 \
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config_file "acc.yaml" --main_process_port 29611 --mixed_precision="fp16" train_instantId_sdxl_style2-attn.py \
  --pretrained_model_name_or_path $MODEL_NAME \
  --controlnet_model_name_or_path $CONTROLNET_NAME \
  --image_encoder_path $ENCODER_NAME \
  --pretrained_ip_adapter_path $ADAPTOR_NAME \
  --data_root_path $ROOT_DATA_DIR \
  --data_json_file $JSON_FILE \
  --output_dir $OUTPUT_DIR \
  --clip_proc_mode orig_crop \
  --mixed_precision="fp16" \
  --resolution 1024 \
  --learning_rate 1e-5 \
  --weight_decay=0.01 \
  --num_train_epochs 400 \
  --train_batch_size 2 \
  --dataloader_num_workers=2 \
  --checkpoints_total_limit 10 \
  --save_steps 5000 


