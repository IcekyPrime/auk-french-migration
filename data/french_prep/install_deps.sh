#!/usr/bin/env bash
set -e
# ============================================================
# 新集群环境安装（在 AuK 项目根执行一次，conda 环境为 python 3.10）
#   - torch/torchaudio/torchvision（CUDA 12.8，适配 H200/Hopper）
#   - AuK 项目依赖 + train extra（读取 pyproject.toml）
#   - 训练/数据准备的额外依赖（pyproject 未收录）
# ============================================================
# 用法:
#   conda create -n auk python=3.10 -y && conda activate auk
#   bash data/french_prep/install_deps.sh
# ============================================================

cd "$(dirname "$0")/../.."      # 切到 AuK 项目根

# 1) torch 家族（CUDA 12.8；H200 是 Hopper，需 cu12x；其它 CUDA 版本改 cu 号）
echo "[1/3] 安装 torch/torchaudio/torchvision (cu128)"
pip install torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128

# 2) AuK 依赖 + train extra（ema-pytorch/tensorboard/matplotlib/tqdm）
echo "[2/3] 安装 AuK 依赖 + [train] extra"
pip install -e ".[train]"

# 3) 训练/数据准备的额外依赖（pyproject 未收录）
echo "[3/3] 安装训练/数据准备额外依赖"
pip install deepspeed soundfile pyarrow openai-whisper huggingface_hub

echo ""
echo "[done] 环境就绪。下一步: bash data/french_prep/setup_new_cluster.sh"
