# AuK 法语跨语言适配 LoRA 微调 — 完整操作手册

## 1. 架构概述

```
┌─────────────────────────────────────────────────────┐
│                    你的集群 (SSH via VSCode)           │
│                                                     │
│  /path/to/AuK/                                       │
│  ├── data/french_prep/                               │
│  │   ├── preprocess.py          ← 数据预处理 (download→filter→asr→jsonl) │
│  │   ├── run_french_lora_train.sh ← 训练启动脚本    │
│  │   ├── README.md                   ← 本文件       │
│  │   └── french_data/             ← 下载后的原始音频存放处 │
│  │                                              │
│  ├── data/lora.py               ← LoRA 注入 (中英通用，不改) │
│  ├── data/lora_train.py         ← 训练入口 (中英通用，不改) │
│  ├── data/merge_lora.py         ← 权重合并 (中英通用，不改) │
│  ├── data/ds_zero3.json        ← ZeRO-3 配置         │
│  └── ckpts/AuK/                ← 基线权重与配置文件     │
│                                                      │
│  训练产出:                                             │
│  └── ckpts/auk_lora_french/model_last.pt              │
│      → merge → ckpts/auk_lora_french/auk_merged.safetensors
└─────────────────────────────────────────────────────┘
```

## 2. 文件说明

| 文件 | 作用 | 是否语言敏感 |
|------|------|-------------|
| `french_prep/preprocess.py` | 数据预处理全流程（4 stage） | 否 — 指令模板可改 |
| `french_prep/run_french_lora_train.sh` | 训练启动脚本 | 否 — 参数化配置 |
| `data/lora.py` | LoRA 类定义 + apply + freeze | **不** — LoRA 层与语言无关 |
| `data/lora_train.py` | 训练循环（ZeRO-3 修复版） | **不** — JSONL 读入的文本即指令 |
| `data/merge_lora.py` | LoRA→权重折叠导出 safetensors | **不** — 只操作 lora_* 键 |
| `data/ds_zero3.json` | DeepSpeed ZeRO-3 配置 | **不** |

## 3. 与中文音色克隆方案的差异

| 维度 | 音色克隆（中文） | 语言适配（法语） |
|------|-----------------|-----------------|
| 目标 | 让模型用指定说话人的音色说任意话 | 让模型掌握法语发音规则，用参考音频的音色说法语 |
| 数据集 | 单说话人 ~376 条 | 多说话人 ~2000+ 条 Common Voice FR |
| ASR | SenseVoiceSmall (自动/zh) | Whisper large-v3 (fr) |
| 指令模板 | `"Say the following with the same voice: {text}"` | `"Dis le texte suivant dans la même voix : {text}"` |
| Learning rate | 1e-4 | 8e-5 |
| Epochs | 20 | 30 |
| Warmup | 50 steps | 100 steps |
| 预计训练时间 | ~14h | ~14-21h |
| 是否破坏原能力 | 几乎不 | 理论上不（但需实测验证中文仍可用） |

## 4. 前置安装

在集群 conda 环境中执行：

```bash
conda activate auk

# 1. 确保 PyTorch + CUDA 正常
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, count: {torch.cuda.device_count()}')"

# 2. 安装 OpenAI Whisper（仅预处理需要）
pip install openai-whisper
# 注意：large-v3 需要 ≥10GB VRAM。如显存不足改为 medium:
# pip install openai-whisper --force-reinstall  # 然后修改 preprocess.py WHISPER_MODEL="medium"

# 3. 确认已有依赖（应已存在于 au k环境）
pip list | grep -iE "accelerate|deepspeed|omegaconf|ema-pytorch"
```

## 5. Step-by-step 完整命令

### 5.1 Step 1: 获取 Common Voice 法语数据

```bash
cd ~/deeplearning_course_sjtu/AuK

# 方案 A: HuggingFace Datasets（推荐）
pip install datasets

python << 'PYEOF'
from datasets import load_dataset
import shutil, os

print("Downloading Common Voice FR ...")
ds = load_dataset('mozilla-foundation/common_voice_19_0', 'fr',
                  split='validated+unverified')
os.makedirs('data/french_prep/french_data', exist_ok=True)
n = 0
for row in ds:
    src = row['path']
    fn = os.path.basename(src)
    dst = f'data/french_prep/french_data/{fn}'
    if os.path.exists(dst):
        continue
    shutil.copy2(src, dst)
    n += 1
    if n % 200 == 0:
        print(f'  copied {n}')
print(f'Done: {n} files downloaded.')
PYEOF

# 预期：~40000 个 MP3 文件，耗时取决于网络 (~10-30min)
```

### 5.2 Step 2: 过滤 & ASR & JSONL

```bash
cd ~/deeplearning_course_sjtu/AuK

# 一键执行（filter + asr + jsonl）
python data/french_prep/preprocess.py

# 分步执行：
# python data/french_prep/preprocess.py --stage filter  # <1min
# python data/french_prep/preprocess.py --stage asr     # ~2-3h (large-v3 on 3090)
# python data/french_prep/preprocess.py --stage jsonl   # <1min
```

**ASR 完成后检查缓存：**

```bash
cat data/french_prep/asr_results_fr.json | python -c "
import json, sys
d = json.load(sys.stdin)
total = len(d)
empty = sum(1 for v in d.values() if not v.get('text'))
print(f'Total: {total}, Empty texts: {empty} ({100*empty/max(total,1):.1f}%)')
"
```

如果空文本率 > 5%，可能需要调整 Whisper 参数或换用 medium 模型重试。

### 5.3 Step 3: 启动训练

```bash
cd ~/deeplearning_course_sjtu/AuK

# 后台训练（tmux 推荐）
tmux new -s french_train

# 启动
chmod +x data/french_prep/run_french_lora_train.sh
bash data/french_prep/run_french_lora_train.sh

# 实时观察日志
tail -f ckpts/auk_lora_french/tensorboard/events.out.tfevents.* 2>/dev/null || \
  tail -f ckpts/auk_lora_french/*.log 2>/dev/null || true

# 按 Ctrl+A D 离开 tmux，训练继续运行
# 重新连接: tmux attach -t french_train
```

**预计时间表**（以 2000 条训练数据计）：

| 指标 | 值 |
|------|---|
| train updates/epoch | ceil(2000/4/7) = 72 |
| total updates | 72 × 30 = 2160 |
| per-update time | ~60-90s |
| epoch time | ~2.5-3.5h |
| total training time | ~75-105h（含 val 采样开销） |

实际更快，因为 `save_per_updates=200` 且 val 每 50 updates 触发。估算 **约 50-70 小时**。

### 5.4 Step 4: 合并权重

```bash
cd ~/deeplearning_course_sjtu/AuK

python data/merge_lora.py \
  --ckpt ckpts/auk_lora_french/model_last.pt \
  --out ckpts/auk_lora_french/auk_lora_french_merged.safetensors
```

预期输出：`saved ... | N tensors | merged lora layers: 40`

### 5.5 Step 5: 推理验证

```bash
cd ~/deeplearning_course_sjtu/AuK

# 选一条 Common Voice FR 音频当音色参考
REF=data/french_prep/french_data/$(ls data/french_prep/french_data/*.mp3 | shuf -n1 | xargs basename)
echo "Using reference audio: $REF"

# 生成法语 TTS
auk-infer \
  --ckpt ckpts/auk_lora_french/auk_lora_french_merged.safetensors \
  --audio "$REF" \
  --instruction 'Dis le texte suivant dans la même voix : "Le ciel était d'un bleu magnifique et les oiseaux chantaient joyeusement dans les arbres."' \
  --gen_seconds 8.0 \
  --nfe 64 --cfg 1.0 --cpu_offload \
  -o output_french_test.wav

# 生成第二句测试
auk-infer \
  --ckpt ckpts/auk_lora_french/auk_lora_french_merged.safetensors \
  --audio "$REF" \
  --instruction 'Dis le texte suivant dans la même voix : "Bonjour à tous, bienvenue dans cette émission spéciale que nous présentons depuis Paris."' \
  --gen_seconds 10.0 \
  --nfe 64 --cfg 1.0 --cpu_offload \
  -o output_french_test2.wav
```

## 6. 常见错误排查

| 错误 | 原因 | 解决 |
|------|------|------|
| `ModuleNotFoundError: no module named 'whisper'` | ASR 未装 whisper | `pip install openai-whisper` |
| `CUDA out of memory during ASR` | large-v3 超出单卡显存 | 改用 `WHISPER_MODEL="medium"` |
| `train/fr.jsonl not found` | preprocessing 未完成 | 先跑 `python data/french_prep/preprocess.py` |
| `Invalid mt19937 state` | Accelerate RNG bug（已修） | 确认用了修复版 `lora_train.py` |
| `_ALLGATHER_BASE timeout` | ZeRO-3 collective 死锁（已修） | 同上 |
| `auk-infer OOM` | 单卡装不下 fp32 全量模型 | 加 `--cpu_offload` |

## 7. 多语言扩展

该框架支持无限扩展其他语言：

| 语言 | 数据源 | 指令模板 | ASR 语言码 |
|------|--------|---------|-----------|
| Spanish | CV ES / LibriLight ES | `"Di el texto siguiente..."` | `es` |
| German | CV DE / VOA German | `"Sage den folgenden Text..."` | `de` |
| Japanese | CV JP / JW800 | `"以下のテキストを発話してください"` | `ja` |
| Italian | CV IT | `"Di il testo seguente..."` | `it` |

只需：
1. 复制 `preprocess.py`，修改 `INSTRUCTION_TPL`, `WHISPER_LANG`
2. 调整 `run_french_lora_train.sh` 中的路径和超参
3. 其余 `lora_train.py`、`merge_lora.py` 完全复用
