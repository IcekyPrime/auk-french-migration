#!/usr/bin/env bash
set -e

# ===================== 资源路径 =====================
train_jsonl=data/train.jsonl
val_jsonl=data/val.jsonl
config=ckpts/AuK/config.yaml
init_ckpt=ckpts/AuK/auk_base.safetensors
output_dir=ckpts/auk_lora_speaker
num_gpus=7
# 指定物理卡 0,1,2,3,4,6,7（跳过卡 5），映射为逻辑卡 0-6
gpu_ids=0,1,2,3,4,6,7

# ===================== LoRA =====================
lora_r=32
lora_alpha=32
lora_dropout=0.0

# ===================== TrainConfig =====================
learning_rate=1e-4
num_train_epochs=20
gradient_accumulation_steps=1
warmup_steps=50
frames_threshold=2700
max_samples=4
save_per_updates=500
last_per_updates=500
val_per_updates=100
logging_steps=10
dataloader_num_workers=4
seed=666

export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=${gpu_ids}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONWARNINGS="ignore"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

mkdir -p "$output_dir"

# 入口改为 data/lora_train.py（自带 LoRA 注入 + ZeRO-3 分片）
accelerate launch --num_processes ${num_gpus} --num_machines 1 --use_deepspeed --deepspeed_config_file data/ds_zero3.json \
  data/lora_train.py \
  --train_jsonl ${train_jsonl} \
  --val_jsonl ${val_jsonl} \
  --config ${config} \
  --init_ckpt ${init_ckpt} \
  --output_dir ${output_dir} \
  --lora_r ${lora_r} \
  --lora_alpha ${lora_alpha} \
  --lora_dropout ${lora_dropout} \
  --learning_rate ${learning_rate} \
  --num_train_epochs ${num_train_epochs} \
  --gradient_accumulation_steps ${gradient_accumulation_steps} \
  --warmup_steps ${warmup_steps} \
  --frames_threshold ${frames_threshold} \
  --max_samples ${max_samples} \
  --save_per_updates ${save_per_updates} \
  --last_per_updates ${last_per_updates} \
  --val_per_updates ${val_per_updates} \
  --logging_steps ${logging_steps} \
  --dataloader_num_workers ${dataloader_num_workers} \
  --seed ${seed}
