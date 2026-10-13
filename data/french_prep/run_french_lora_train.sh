#!/usr/bin/env bash
set -e

# ====================================================================
# AuK French TTS LoRA 微调启动脚本
# ====================================================================
# 用于在集群上对法语语料进行跨语言适配（language transfer），
# 让 base AuK 生成高质量法语语音。
#
# 前置条件：
#   1. conda activate auk
#   2. pip install openai-whisper  （ASR 需要）
#   3. data/french_prep/ 下已有 train_fr.jsonl / val_fr.jsonl
# ====================================================================

# ------------------- 路径配置 -------------------
export PYTHONPATH=src

# JSONL 数据集（VoxPopuli + MLS 合并）
train_jsonl=data/french_prep/train_fr_all.jsonl
val_jsonl=data/french_prep/val_fr_small.jsonl

# 基线模型与配置文件
# config.yaml 已随 merged ckpt 一起迁移到 ckpts/auk_lora_french_v3/；
# 若你的 config.yaml 放在 ckpts/AuK/，改下面这一行即可。
config=ckpts/auk_lora_french_v3/config.yaml

# 冷启动从上一轮「已合并」的法语 checkpoint（v3）继续，再注入扩展 LoRA 目标层。
# model_last.pt 的 base 权重是冻结原值 + 独立 lora_*，需先 merge 成干净 safetensors。
prev_ckpt_dir=ckpts/auk_lora_french_v3
prev_merged=${prev_ckpt_dir}/auk_lora_french_v3_merged.safetensors
if [ ! -f "${prev_merged}" ]; then
    echo "[merge] 合并上一轮 LoRA: ${prev_ckpt_dir}/model_last.pt -> ${prev_merged}"
    python data/merge_lora.py --ckpt ${prev_ckpt_dir}/model_last.pt --out ${prev_merged}
fi
init_ckpt=${prev_merged}
output_dir=ckpts/auk_lora_french_v4

# GPU 资源（新集群 H200，8 卡；按实际申请数改 num_gpus / gpu_ids）
num_gpus=8
gpu_ids=0,1,2,3,4,5,6,7
# 注意：data/ds_zero3.json 的 train_batch_size 需等于 num_gpus × max_samples（当前 8×4=32）

# ZeRO-3 DeepSpeed 配置
ds_config_file=data/ds_zero3.json

# ------------------- LoRA 超参 -------------------
lora_r=32
lora_alpha=32
lora_dropout=0.0
# LoRA 目标层：留空 = 使用扩展集合（lora.py 的 DEFAULT_TARGETS，含 txt_proj/proj_out/
# audio_embed/adaLN/to_out/time_mlp）。如只想用旧集合，可设为 LEGACY 后缀列表。
lora_targets=""

# ------------------- 训练超参 -------------------
learning_rate=5e-5             # 继续训练用小 lr，避免扰动已收敛模型
num_train_epochs=1             # 联合数据量大（VoxPopuli+MLS），1 epoch 足够
gradient_accumulation_steps=1
warmup_steps=100               # 占 total_updates(~4930) 约 2%，对低 lr 足够
frames_threshold=2700
max_samples=4                  # per-GPU batch size
save_per_updates=500
last_per_updates=500
val_per_updates=200
logging_steps=10
dataloader_num_workers=4
seed=666

# ------------------- 环境变量 -------------------
export CUDA_VISIBLE_DEVICES=${gpu_ids}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONWARNINGS="ignore"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

mkdir -p "$output_dir"

# ------------------- 校验数据 -------------------
if [ ! -f "$train_jsonl" ]; then
    echo "[ERROR] 训练集不存在: $train_jsonl"
    echo "        请先运行: cd <AuK_root> && python data/french_prep/preprocess.py"
    exit 1
fi

if [ ! -f "$val_jsonl" ]; then
    echo "[ERROR] 验证集不存在: $val_jsonl"
    exit 1
fi

TRAIN_N=$(wc -l < "$train_jsonl")
VAL_N=$(wc -l < "$val_jsonl")
echo "[info] train=$TRAIN_N val=$VAL_N"
echo "[info] output_dir=$output_dir"
echo "[info] config=$config"
echo ""

# 入口: data/lora_train.py（共享自中文微调，LoRA + ZeRO-3 修复版全已就绪）
# PYTHONPATH 需要包含项目根目录以便 import lora
accelerate launch --num_processes ${num_gpus} --num_machines 1 --use_deepspeed --deepspeed_config_file ${ds_config_file} \
  data/lora_train.py \
  --train_jsonl ${train_jsonl} \
  --val_jsonl ${val_jsonl} \
  --config ${config} \
  --init_ckpt ${init_ckpt} \
  --output_dir ${output_dir} \
  --lora_r ${lora_r} \
  --lora_alpha ${lora_alpha} \
  --lora_dropout ${lora_dropout} \
  $( [ -n "${lora_targets}" ] && echo "--lora_targets ${lora_targets}" ) \
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
