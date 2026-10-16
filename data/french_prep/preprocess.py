#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AuK 法语（French）跨语言适配 LoRA 微调 —— 数据预处理脚本。

数据源：Mozilla Common Voice 17.0 法语子集（config "fr"，CC0 许可，非 gated）。
自带 sentence 转写字段，无需 ASR。多说话人，适合语言适配。

流程分两步，可单独执行：
  1. download — 通过 hf-mirror 下载法语子集，保存音频到 french_data/，生成 records 缓存
  2. jsonl    — 读缓存，按说话人分组环形配对，时长过滤，生成 train_fr.jsonl / val_fr.jsonl

用法（在 AuK 项目根，conda 环境 auk）:
    python data/french_prep/preprocess.py --stage download
    python data/french_prep/preprocess.py --stage jsonl \
        --audio-base /hpc_stor03/sjtu_home/haotian.xie/deeplearning_course_sjtu/AuK/data/french_prep/french_data
    # 或一键执行（不指定 audio-base 时用本地路径）
    python data/french_prep/preprocess.py
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path


# ===================== 配置区 =====================

DATA_DIR = Path(__file__).resolve().parent          # data/french_prep/
AUDIO_DIR = Path(os.environ.get("FRENCH_DATA_DIR", str(DATA_DIR / "french_data")))  # 下载音频存放处；可用 FRENCH_DATA_DIR 重定向到无配额盘
RECORDS_CACHE = DATA_DIR / "asr_results_fr.json"    # records 缓存（复用文件名）
TRAIN_JSONL = DATA_DIR / "train_fr.jsonl"
VAL_JSONL = DATA_DIR / "val_fr.jsonl"

# VoxPopuli 数据集（CC0，欧洲议会多语言语音；config "fr" = 法语，约 100h+）
# 注：Common Voice 17.0 的 parquet 在 hf-mirror 上未同步，VoxPopuli 完整同步（已验证）。
DATASET_NAME = "facebook/voxpopuli"
CONFIG = "fr"
SPLITS = "train"                     # VoxPopuli 的 train split 已含充足数据

# HF 端点：默认走 hf-mirror（国内集群更稳）；如需直连，设环境变量 HF_ENDPOINT=https://huggingface.co 覆盖
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

# 过滤参数
MIN_SEC, MAX_SEC = 0.5, 30.0

# 指令模板（法语零样本 TTS；必须与推理时逐字一致）
INSTRUCTION_TPL = 'Dis le texte suivant dans la même voix : "{text}"'

SEED = 42
VAL_RATIO = 0.1
MAX_SAMPLES = 0          # 0 = 不限制；>0 时采样截断，用于快速试跑

# audio_url 前缀环境变量名
AUDIO_BASE_ENV = "FRENCH_AUDIO_BASE"


def clean_text(raw: str) -> str:
    """清洗转录文本。Common Voice sentence 通常已干净，这里只做兜底。"""
    t = (raw or "").strip()
    t = re.sub(r"\s+", " ", t)
    return t


# ===================== Stage 1: Download =====================

def stage_download(max_samples: int = 0):
    """通过 hf-mirror 直接下载 VoxPopuli 法语 parquet，提取音频字节并生成 records 缓存。

    绕开 datasets 的 Audio feature 解码（其依赖 torchcodec，集群缺 libnvrtc.so.13 会崩），
    直接用 pyarrow 读 parquet，从 audio 字段提取原始字节写文件。
    """
    os.environ["HF_ENDPOINT"] = HF_ENDPOINT

    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    import requests

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # 用 requests 直接请求 tree API，绕开 huggingface_hub 新版 list_repo_files 忽略 endpoint 的 bug
    r = requests.get(
        f"{HF_ENDPOINT}/api/datasets/{DATASET_NAME}/tree/main",
        params={"recursive": "true", "expand": "false"}, timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    items = data if isinstance(data, list) else data.get("items", [])
    files = [x.get("path") if isinstance(x, dict) else str(x) for x in items]
    print(f"[download] 总共 {len(files)} 个文件")
    # 打印语言列表供诊断
    lang_dirs = sorted(set(f.split("/")[0] for f in files if "/" in f and f.endswith(".parquet")))
    print(f"[download] 语言子目录: {lang_dirs}")
    # 匹配所有 /fr/ 路径下的 parquet
    parquet_files = sorted(
        f for f in files
        if f.endswith(".parquet") and f.startswith("fr/")
    )
    print(f"[download] 找到 {len(parquet_files)} 个法语 parquet 文件")
    if not parquet_files:
        print("[download] 未找到法语文件，请确认数据集结构")
        sys.exit(1)

    records = []
    n = 0
    uidx = 0
    skipped = 0
    struct_printed = False

    import shutil
    import tempfile

    for pf in parquet_files:
        tmp = Path(tempfile.mkdtemp(prefix="voxpopuli_parquet_"))
        try:
            local = hf_hub_download(DATASET_NAME, pf, repo_type="dataset", local_dir=str(tmp))
            table = pq.read_table(local)
            df = table.to_pandas()
        finally:
            # 读完即删 parquet，避免每份 ~3GB、共 ~40GB 堆积在缓存
            shutil.rmtree(tmp, ignore_errors=True)

        for i in range(len(df)):
            row = df.iloc[i]

            # 文本字段：优先 raw_text（保留大小写/标点），回退 normalized_text
            text = None
            for tcol in ("raw_text", "normalized_text", "sentence", "transcription"):
                if tcol in df.columns:
                    v = row.get(tcol)
                    if v is not None and str(v).strip():
                        text = clean_text(str(v))
                        break
            if not text:
                skipped += 1
                continue

            # 说话人字段
            speaker = "unknown"
            for scol in ("speaker_id", "client_id", "speaker"):
                if scol in df.columns:
                    v = row.get(scol)
                    if v is not None:
                        speaker = str(v)
                        break

            # 音频字节
            audio_bytes = None
            ext = ".mp3"
            if "audio" in df.columns:
                a = row.get("audio")
                if isinstance(a, dict):
                    # 首次打印结构，便于诊断
                    if not struct_printed:
                        print(f"[download] audio 字段 keys: {list(a.keys())}")
                        struct_printed = True
                    if a.get("bytes"):
                        audio_bytes = a["bytes"]
                    p = a.get("path")
                    if p:
                        ext = os.path.splitext(str(p))[1] or ".mp3"
                elif isinstance(a, (bytes, bytearray)):
                    audio_bytes = bytes(a)

            if not audio_bytes:
                skipped += 1
                continue

            dst_name = f"{speaker}_{uidx:06d}{ext}"
            dst_path = AUDIO_DIR / dst_name
            if not dst_path.exists():
                dst_path.write_bytes(audio_bytes)

            records.append({"wav": str(dst_path), "text": text, "speaker": speaker})
            n += 1
            uidx += 1
            if n % 500 == 0:
                print(f"[download] 已处理 {n} 条 ...")
            if max_samples > 0 and n >= max_samples:
                break

        if max_samples > 0 and n >= max_samples:
            break

    print(f"[download] 完成：{n} 条音频，跳过 {skipped} 条")
    print(f"[download] 说话人数：{len(set(r['speaker'] for r in records))}")

    cache = {"records": records, "dataset": DATASET_NAME, "config": CONFIG}
    RECORDS_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[download] records 已缓存到 {RECORDS_CACHE}")


# ===================== Stage 2: JSONL =====================

def duration_of(p: Path) -> float:
    """读取音频时长（秒）。优先 soundfile，失败回退 ffprobe。"""
    try:
        import soundfile as sf
        info = sf.info(str(p))
        return info.frames / info.samplerate
    except Exception:
        import subprocess
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(p)],
            capture_output=True, text=True, timeout=20,
        )
        return float(out.stdout.strip())


def stage_jsonl(audio_base: str = None):
    cache = json.loads(RECORDS_CACHE.read_text(encoding="utf-8"))
    records = cache.get("records", [])
    if not records:
        print("[jsonl] 缓存为空，请先运行 --stage download")
        sys.exit(1)

    # 时长过滤
    filtered = []
    for r in records:
        try:
            d = duration_of(Path(r["wav"]))
        except Exception as e:
            print(f"[jsonl skip] 读时长失败 {r['wav']}: {e}")
            continue
        if MIN_SEC <= d <= MAX_SEC:
            r["dur"] = d
            filtered.append(r)

    print(f"[jsonl] 时长过滤后 {len(filtered)} 条（原 {len(records)} 条）")

    # 可选采样截断
    if MAX_SAMPLES > 0 and len(filtered) > MAX_SAMPLES:
        rng = random.Random(SEED)
        rng.shuffle(filtered)
        filtered = filtered[:MAX_SAMPLES]
        print(f"[jsonl] 采样截断至 {MAX_SAMPLES} 条")

    # 按说话人分组
    by_speaker: dict[str, list[dict]] = {}
    for r in filtered:
        by_speaker.setdefault(r["speaker"], []).append(r)

    # audio_base 确定
    if audio_base is None:
        audio_base = os.environ.get(AUDIO_BASE_ENV, None)
        if audio_base is None:
            audio_base = str(AUDIO_DIR.resolve()).replace("\\", "/")
            print(f"[jsonl] 未指定 --audio-base，使用本地路径: {audio_base}")
            print(f"         提示: 若用于集群，请用 --audio-base 指定集群绝对路径")

    # 同说话人环形配对
    rng = random.Random(SEED)
    all_samples = []
    for spk, spk_records in by_speaker.items():
        if len(spk_records) < 2:
            continue
        rng.shuffle(spk_records)
        for i, rec in enumerate(spk_records):
            ref = spk_records[(i + 1) % len(spk_records)]
            all_samples.append({
                "duration": round(rec["dur"], 3),
                "target_wav": rec["wav"],
                "ref_wav": ref["wav"],
                "text": rec["text"],
            })

    rng.shuffle(all_samples)
    n_train = max(1, int(len(all_samples) * (1 - VAL_RATIO)))
    splits = [("train", all_samples[:n_train], TRAIN_JSONL),
              ("val", all_samples[n_train:], VAL_JSONL)]

    for name, subset, out_path in splits:
        if not subset:
            print(f"[warn] {name} 集为空，跳过")
            continue
        lines = []
        for s in subset:
            target_rel = os.path.relpath(s["target_wav"], AUDIO_DIR).replace("\\", "/")
            ref_rel = os.path.relpath(s["ref_wav"], AUDIO_DIR).replace("\\", "/")
            obj = {
                "duration": s["duration"],
                "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": INSTRUCTION_TPL.format(text=s["text"])},
                        {"type": "audio", "audio_url": f"{audio_base}/{ref_rel}"},
                    ]},
                    {"role": "assistant", "content": [
                        {"type": "audio", "audio_url": f"{audio_base}/{target_rel}"},
                    ]},
                ],
            }
            lines.append(json.dumps(obj, ensure_ascii=False))
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[jsonl] {name}: {len(lines)} 条 -> {out_path}")


# ===================== Main =====================

def main():
    ap = argparse.ArgumentParser(description="AuK 法语微调数据预处理（Common Voice 17.0 fr）")
    ap.add_argument("--stage", choices=["download", "jsonl"], default=None,
                    help="只执行某一阶段；缺省则顺序执行全部")
    ap.add_argument("--audio-base", default=None,
                    help="JSONL 中 audio_url 的路径前缀（集群绝对路径）")
    ap.add_argument("--max-samples", type=int, default=MAX_SAMPLES,
                    help="下载阶段采样上限（0=不限制）")
    args = ap.parse_args()

    if args.stage == "download":
        stage_download(args.max_samples)
    elif args.stage == "jsonl":
        stage_jsonl(args.audio_base)
    else:
        stage_download(args.max_samples)
        stage_jsonl(args.audio_base)

    print("\n[done] 全部完成。下一步: bash data/french_prep/run_french_lora_train.sh")


if __name__ == "__main__":
    main()
