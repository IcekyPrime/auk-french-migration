#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AuK 微调数据预处理脚本（场景 B：特定说话人音色克隆）。

流程分三步，可单独执行便于断点续跑：
  1. filter  — 读取每条音频时长，把 <0.3s 或 >30s 的移入 _discarded，其余按序重命名为四位 ID（0001.ext, 0002.ext, ...）
  2. asr     — 用本地 SenseVoiceSmall 批量转写，结果缓存到 asr_results.json
  3. jsonl   — 依据 ASR 结果生成 train.jsonl / val.jsonl（9:1），格式遵循 AuK 零样本 TTS 官方模板

用法（在 auk_prep 环境内，PYTHONPATH 无需）:
    python preprocess.py --stage filter
    python preprocess.py --stage asr
    python preprocess.py --stage jsonl
    python preprocess.py               # 全部顺序执行
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "训练数据"          # 原始音频目录
DISCARD_DIR = RAW_DIR / "_discarded"    # 区间外音频移到这里（不物理删除，可追溯）
ASR_CACHE = DATA_DIR / "asr_results.json"
TRAIN_JSONL = DATA_DIR / "train.jsonl"
VAL_JSONL = DATA_DIR / "val.jsonl"

AUDIO_EXTS = {".flac", ".mp3", ".wav"}
MIN_SEC, MAX_SEC = 0.3, 30.0
# 零样本 TTS 官方指令模板（中英文统一用英文，见 docs/COOKBOOK.md 1.1）
INSTRUCTION_TPL = 'Say the following with the same voice: "{text}"'
SEED = 0
VAL_RATIO = 0.1
ASR_LANGUAGE = "auto"  # SenseVoice 自动检测语种（zh/en/yue/ja/ko）
# audio_url 使用的前缀（集群绝对路径）。本地生成 jsonl 时直接写入，避免上传后做字符串替换破坏 JSON 转义。
AUDIO_BASE = "/hpc_stor03/sjtu_home/haotian.xie/deeplearning_course_sjtu/AuK/data/训练数据"


def natural_key(p: Path):
    m = re.search(r"\((\d+)\)", p.stem)
    return (int(m.group(1)) if m else 0, p.name)


def iter_raw_audio():
    """扫描原始命名（"N (M).ext"）的音频，跳过已重命名的四位 ID 文件。"""
    files = []
    for p in RAW_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            if re.fullmatch(r"\d{4}\.[^.]+", p.name):
                continue
            files.append(p)
    return sorted(files, key=natural_key)


def iter_renamed_audio():
    """扫描已重命名（四位 ID）的音频。"""
    files = []
    for p in RAW_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS and re.fullmatch(r"\d{4}\.[^.]+", p.name):
            files.append(p)
    return sorted(files)


def duration_of(p: Path) -> float:
    """读音频时长（秒）。libsndfile 1.2.2 支持 flac/wav/mp3。"""
    import soundfile as sf
    info = sf.info(str(p))
    return info.frames / info.samplerate


def stage_filter():
    DISCARD_DIR.mkdir(exist_ok=True)
    keep, discarded = [], []
    for p in iter_raw_audio():
        try:
            d = duration_of(p)
        except Exception as e:
            print(f"[skip] 读取时长失败 {p.name}: {e}")
            discarded.append(p)
            continue
        if MIN_SEC <= d <= MAX_SEC:
            keep.append((p, d))
        else:
            discarded.append(p)
            print(f"[discard] {p.name} 时长={d:.2f}s（超出 {MIN_SEC}~{MAX_SEC}s）")

    for p in discarded:
        shutil.move(str(p), str(DISCARD_DIR / p.name))

    for idx, (p, d) in enumerate(keep, start=1):
        new = RAW_DIR / f"{idx:04d}{p.suffix.lower()}"
        os.rename(p, new)
        print(f"[rename] {p.name} -> {new.name}（{d:.2f}s）")
    print(f"filter 完成：保留 {len(keep)} 条，移出 {len(discarded)} 条")


def clean_asr_text(raw: str) -> str:
    return re.sub(r"<\|[^|]+\|>", "", raw or "").replace("</s>", "").strip()


def load_asr():
    from funasr import AutoModel
    return AutoModel(model="iic/SenseVoiceSmall", device="cpu", disable_update=True, disable_pbar=True)


def stage_asr():
    model = load_asr()
    cache = {}
    if ASR_CACHE.exists():
        cache = json.loads(ASR_CACHE.read_text(encoding="utf-8"))
    files = iter_renamed_audio()
    for p in files:
        key = p.name
        if cache.get(key, {}).get("text"):
            print(f"[cached] {key}")
            continue
        res = model.generate(input=str(p), cache={}, batch_size=1, language=ASR_LANGUAGE, use_itn=True)
        raw = res[0].get("text", "") if res else ""
        text = clean_asr_text(raw)
        cache[key] = {"text": text, "raw": raw}
        print(f"[asr] {key}: {text}")
        ASR_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    missing = [p.name for p in files if not cache.get(p.name, {}).get("text")]
    print(f"ASR 完成：{len(files)} 条，空文本 {len(missing)} 条")
    if missing:
        print("空文本样本：", missing)


def stage_jsonl(audio_base: str = AUDIO_BASE):
    cache = json.loads(ASR_CACHE.read_text(encoding="utf-8"))
    records = []
    for p in iter_renamed_audio():
        text = cache.get(p.name, {}).get("text", "")
        if not text:
            print(f"[skip jsonl] {p.name} 无转写")
            continue
        try:
            d = duration_of(p)
        except Exception as e:
            print(f"[skip jsonl] {p.name} 读时长失败: {e}")
            continue
        records.append((p, text, d))

    rng = random.Random(SEED)
    rng.shuffle(records)
    n_train = max(1, int(len(records) * (1 - VAL_RATIO)))
    splits = [("train", records[:n_train], TRAIN_JSONL), ("val", records[n_train:], VAL_JSONL)]

    for name, subset, out in splits:
        if not subset:
            print(f"[warn] {name} 集为空，跳过")
            continue
        lines = []
        for i, (p, text, d) in enumerate(subset):
            ref = subset[(i + 1) % len(subset)][0]  # 环形配对：参考音频取同说话人另一条
            obj = {
                "duration": round(d, 3),
                "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": INSTRUCTION_TPL.format(text=text)},
                        {"type": "audio", "audio_url": f"{audio_base}/{ref.name}"},
                    ]},
                    {"role": "assistant", "content": [
                        {"type": "audio", "audio_url": f"{audio_base}/{p.name}"},
                    ]},
                ],
            }
            lines.append(json.dumps(obj, ensure_ascii=False))
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{name}: {len(lines)} 条 -> {out.name}")


def main():
    ap = argparse.ArgumentParser(description="AuK 微调数据预处理")
    ap.add_argument("--stage", choices=["filter", "asr", "jsonl"], default=None,
                    help="只执行某一阶段；缺省则顺序执行全部")
    ap.add_argument("--audio-base", default=AUDIO_BASE,
                    help="jsonl 中 audio_url 的路径前缀（集群绝对路径）")
    args = ap.parse_args()

    if args.stage == "filter":
        stage_filter()
    elif args.stage == "asr":
        stage_asr()
    elif args.stage == "jsonl":
        stage_jsonl(args.audio_base)
    else:
        stage_filter()
        stage_asr()
        stage_jsonl()


if __name__ == "__main__":
    main()
