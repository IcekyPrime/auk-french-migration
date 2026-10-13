#!/usr/bin/env bash
set -e
# ============================================================
# 新集群数据准备：VoxPopuli + MLS → train/val jsonl（全部相对路径）
# ============================================================
# 关键点：jsonl 里 audio_url 用「相对 AuK 项目根的路径」，训练脚本从项目根启动即可
# 正确解析，不依赖任何绝对挂载路径，天然适配新集群迁移。
#
# 用法（在 AuK 项目根，conda activate auk）:
#   bash data/french_prep/setup_new_cluster.sh
#
# 前置依赖（auk 环境）:
#   pip install soundfile pyarrow openai-whisper
#   （VoxPopuli 下载需 hf-mirror，已内置 HF_ENDPOINT）
# ============================================================

cd "$(dirname "$0")/../.."      # 切到 AuK 项目根，保证所有相对路径一致
# 新集群 HF 直连正常，默认直连 huggingface.co；如需镜像，取消下面注释
# export HF_ENDPOINT=https://hf-mirror.com

# ---------- 可调参数 ----------
MLS_MAX_SAMPLES=${MLS_MAX_SAMPLES:-0}   # 0=全量；>0 采样上限（MLS 法语 train 约 25 万条，可设大值先试跑）

# ---------- 1) VoxPopuli 法语：下载 + jsonl（相对路径）----------
echo "[1/4] VoxPopuli 法语下载"
python data/french_prep/preprocess.py --stage download

echo "[2/4] VoxPopuli jsonl（相对路径）"
python data/french_prep/preprocess.py --stage jsonl \
    --audio-base data/french_prep/french_data

# ---------- 2) MLS 法语：下载 + jsonl（相对路径）----------
echo "[3/4] MLS 法语下载 + jsonl"
bash data/french_prep/download_mls.sh
python data/french_prep/preprocess_mls.py \
    --mls_root data/french_prep/mls_french \
    --split train \
    --out data/french_prep/mls_fr.jsonl \
    --audio-base data/french_prep/mls_french \
    --max-samples ${MLS_MAX_SAMPLES}

# ---------- 3) 合并 train / val ----------
echo "[4/4] 合并 train_fr_all.jsonl / val_fr_small.jsonl"
cat data/french_prep/train_fr.jsonl data/french_prep/mls_fr.jsonl > data/french_prep/train_fr_all.jsonl
cp data/french_prep/val_fr.jsonl data/french_prep/val_fr_small.jsonl

echo ""
echo "[done] 数据就绪（相对路径）:"
echo "   train: data/french_prep/train_fr_all.jsonl"
echo "   val:   data/french_prep/val_fr_small.jsonl"
echo "下一步: bash data/french_prep/run_french_lora_train.sh"
