# AuK 法语（French）跨语言适配 LoRA 微调全流程

## 前置说明

### M-AILABS French190h 不适用法语 TTS

M-MAILABS "French190h" **不包含真实法语音频**。它聚合了 LJSpeech（英语）、VCTK（英语）、THCHS-30（中英混杂）等已有语料。如需法语 TTS，必须使用真实法语音频。

### 推荐数据源

| 数据源 | 许可证 | 语种 | 规模 | 下载方式 |
|---|---|---|---|---|
| [Mozilla Common Voice FR](https://commonvoice.mozilla.org/fr/datasets) | CC-BY-SA | 纯法语 | 190h+ (v19.0) | HuggingFace `datasets` 库 |
| [VoxPopuli FR](https://github.com/facebookresearch/voxpopuli) | CC BY 4.0 | 多语种含法语 | ~50k utterances | Facebook DL 直链 |
| [LibriLight Large French](https://arxiv.org/abs/2012.15208) | CC BY 4.0 | 纯法语 | 60h | FAIR DL 直链 |

**推荐**: Mozilla Common Voice — 人工标注质量最高、文本干净、说话人多样化。

---

## 快速开始

### Step 0: 环境准备

```bash
# conda 环境
conda activate auk

# Whisper ASR（仅预处理需要，训练不需要）
pip install openai-whisper torch torchaudio
# 注意: whisper large-v3 需要 ~10GB VRAM
# 如显存不足，修改 preprocess.py 中的 WHISPER_MODEL="medium"

# PyTorch 确保有 CUDA 支持（accelerate + DeepSpeed 依赖）
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Step 1: 获取法语音频

#### 方案 A: HuggingFace Datasets（推荐，最简单）

```bash
cd /path/to/AuK
pip install datasets

python -c "
from datasets import load_dataset
import shutil, os

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
    if n % 100 == 0:
        print(f'  copied {n}')
print(f'Done: {n} files')
"
```

这会下载所有已验证+未验证的法语 clip（约 40000+ 条）。过滤步骤会自动筛选出时长合适的样本。

#### 方案 B: 已有本地目录

如果你有其他方式获得法语音频（从 VOA French、TED Talks 等），直接放入：

```bash
cp /path/to/your_french_audio/*.mp3 data/french_prep/french_data/
```

格式要求：`.mp3`, `.flac`, `.wav`, `.opus` 均可。

### Step 2: 预处理

```bash
cd /path/to/AuK

# 一键执行全部阶段（下载→过滤→ASR→JSONL）
python data/fense_prep/preprocess.py

# 或分步执行便于断点续跑
python data/french_prep/preprocess.py --stage filter   # 按时长过滤
python data/french_prep/preprocess.py --stage asr       # Whisper 转写
python data/french_prep/preprocess.py --stage jsonl     # 生成 train/val JSONL
```

各阶段耗时估算（以 ~40000 clips → 目标 ~2000 条计）：

| 阶段 | GPU | 预期耗时 |
|---|---|---|
| filter | CPU | <1 min |
| asr (large-v3) | 1×3090 | ~2-3h |
| asr (medium) | 1×3090 | ~1h |
| jsonl | CPU | <1 min |

### Step 3: 启动训练

```bash
cd /path/to/AuK

# 赋予执行权限
chmod +x data/french_prep/run_french_lora_train.sh

# tmux / nohup 后台运行（推荐）
tmux new -s french_train
bash data/french_prep/run_french_lora_train.sh

# 观察训练日志
tail -f ckpts/auk_lora_french/train_fr.jsonl
```

训练预计耗时（7×3090, batch_per_gpu=4, ~2000 条数据）：
- ~280 updates / epoch × 30 epochs = 8400 total
- 每个 update ~60-90s → 约 **14-21 小时**

### Step 4: 合并权重 & 推理

```bash
cd /path/to/AuK

# 合并 LoRA 回基线
python data/merge_lora.py \
  --ckpt ckpts/auk_lora_french/model_last.pt \
  --out ckpts/auk_lora_french/auk_lora_french_merged.safetensors

# 推理测试
auk-infer \
  --ckpt ckpts/auk_lora_french/auk_lora_french_merged.safetensors \
  --audio data/french_prep/french_data/0001.mp3 \
  --instruction 'Dis le texte suivant dans la même voix : "Bonjour, comment allez-vous aujourd'hui ?"' \
  --gen_seconds 5.0 \
  --nfe 64 --cfg 1.0 --cpu_offload \
  -o output_french.wav
```

---

## 超参数说明

| 参数 | 值 | 与中文微调对比 | 理由 |
|---|---|---|---|
| `learning_rate` | 8e-5 | 中文 1e-4 | 跨语言迁移需更保守，避免破坏原有语言知识 |
| `num_train_epochs` | 30 | 中文 20 | 语言模式学习比音色克隆更慢 |
| `warmup_steps` | 100 | 中文 50 | 更高比例 warmup（~36% of total）保证初期稳定 |
| `lora_r` / `alpha` | 32 / 32 | 同中文 | LoRA 维度对语言不敏感 |
| `save_per_updates` | 200 | 中文 500 | 更多检查点便于调试和选择最佳 checkpoint |
| `max_samples` | 4 | 同中文 | 由显存决定 |

## 训练数据特征（典型 Common Voice FR 输出）

以 Common Voice FR v19.0 为例：

| 统计项 | 值 |
|---|---|
| 原始 clip 数 | ~40,000+ (validated + unverified) |
| 过滤后保留 | ~15,000-20,000 (时长 0.5s-30s) |
| 总时长 | ~100-150 小时 |
| 说话人数 | 100+ |
| 平均单条时长 | ~7-8 秒 |
| 最终训练集 | ~17,000 条 |
| 最终验证集 | ~1,900 条 |

> ⚠️ 如果数据量过大（>15000 条），可在 preprocessing 中设置 `MAX_SAMPLES=2000` 进行采样截断。

## 常见问题

### Q: Whisper 在法国语上 WER 有多高？

A: Whisper large-v3 在法语上的字符错误率（WER）约为 3-5%，远低于 SenseVoiceSmall（不支持法语）。对于 TTS 训练来说，3-5% 的 ASR 误差是可接受的——模型已经见过各种发音变化，少量转写错误不会显著影响生成质量。

### Q: 训练后会丢失中文能力吗？

A: LoRA 是低秩适配器，只更新注意力层和 FFN 的增量矩阵。**理论上不会完全遗忘**——base 模型的中文能力仍然完整保留在冻结权重中。但实际表现取决于：
- LoRA rank 大小（rank 32 较小，遗忘风险低）
- 训练步数（步数越多，LoRA 覆盖越多，可能更偏向法语风格）

建议推理时分别用中文指令和法语指令测试，评估双语切换效果。

### Q: 能同时做多语言适配吗？

A: 可以！训练不同语言的 LoRA 互不冲突。推理时只需加载对应的合并权重：
- `auk_lora_speaker` = 音色克隆（中英混合数据）
- `auk_lora_french` = 法语适配（法语数据）
- `auk_lora_japanese` = 日语适配（日语数据）
...以此类推。
