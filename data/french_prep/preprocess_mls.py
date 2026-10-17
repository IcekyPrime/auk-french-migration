#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLS French 适配脚本：读 transcripts.txt + flac，生成与 VoxPopuli 完全同格式的 jsonl。

MLS 目录结构（已确认）:
    mls_french/{split}/audio/<speaker_id>/<book_id>/<speaker_id>_<book_id>_<utt>.flac
    mls_french/{split}/transcripts.txt    # 每行: <utt_id>\t<text>

音频 16kHz 单声道 flac；transcripts 自带转写（有声书），无需 ASR。

用法（conda auk，在 AuK 项目根）:
    python data/french_prep/preprocess_mls.py \
        --mls_root /mnt/gpfs_dataset_hdd/sjtu/data/asr/rawdata/MLS/mls_french \
        --split train \
        --out data/french_prep/mls_fr.jsonl \
        --max-samples 100000          # 采样上限，0=全量（全量约 25 万条，读时长约 40min）
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

# 与 VoxPopuli / 推理端逐字一致的指令模板
INSTRUCTION_TPL = 'Dis le texte suivant dans la même voix : "{text}"'

MIN_SEC, MAX_SEC = 0.5, 30.0
SEED = 42


def clean_text(raw: str) -> str:
    t = (raw or "").strip()
    t = re.sub(r"\s+", " ", t)
    return t


def duration_of(path: Path) -> float:
    # opus 走 ffprobe（soundfile/libsndfile 不支持 opus）；flac 优先 soundfile
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return info.frames / info.samplerate
    except Exception:
        import subprocess
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        return float(out.stdout.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mls_root", required=True, help="MLS 语言根目录（含 train/dev/test）")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True, help="输出 jsonl 路径")
    ap.add_argument("--max-samples", type=int, default=0, help="采样上限（0=全量）")
    ap.add_argument("--audio-base", default=None, help="JSONL 中 audio_url 前缀，默认用 mls_root")
    args = ap.parse_args()

    mls_root = Path(args.mls_root)
    split_dir = mls_root / args.split
    transcripts = split_dir / "transcripts.txt"
    audio_dir = split_dir / "audio"

    # 1) 读全量 transcripts（纯文本，快，可整读进内存）
    with open(transcripts, encoding="utf-8") as f:
        raw_lines = [l for l in f.read().splitlines() if l.strip()]
    print(f"[mls] transcripts 共 {len(raw_lines)} 行")

    # 2) 采样（随机，避免顺序采样偏向前几个 speaker）
    rng = random.Random(SEED)
    if args.max_samples > 0 and len(raw_lines) > args.max_samples:
        rng.shuffle(raw_lines)
        raw_lines = raw_lines[: args.max_samples]
        print(f"[mls] 随机采样 {args.max_samples} 行")

    # 3) 逐条解析 + 时长过滤
    records = []
    n = 0
    skipped = 0
    for line in raw_lines:
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            skipped += 1
            continue
        utt_id, text = parts[0], clean_text(parts[1])
        spk, book = utt_id.split("_")[0], utt_id.split("_")[1]
        wav = audio_dir / spk / book / f"{utt_id}.opus"
        if not wav.exists():
            skipped += 1
            continue
        try:
            d = duration_of(wav)
        except Exception:
            skipped += 1
            continue
        if not (MIN_SEC <= d <= MAX_SEC):
            skipped += 1
            continue
        records.append({"wav": str(wav), "text": text, "speaker": spk, "dur": d})
        n += 1
        if n % 20000 == 0:
            print(f"[mls] 已处理 {n} 条 ...")

    print(f"[mls] 有效 {n} 条，跳过 {skipped} 条，说话人 {len(set(r['speaker'] for r in records))}")

    # 4) 按 speaker 分组环形配对（与 VoxPopuli preprocess 一致）
    by_speaker: dict[str, list] = {}
    for r in records:
        by_speaker.setdefault(r["speaker"], []).append(r)

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

    # 5) 写 jsonl（audio_url 用 MLS 绝对路径，训练端直接读 gpfs）
    audio_base = (args.audio_base or str(mls_root)).rstrip("/")
    lines_out = []
    for s in all_samples:
        target_rel = os.path.relpath(s["target_wav"], mls_root).replace("\\", "/")
        ref_rel = os.path.relpath(s["ref_wav"], mls_root).replace("\\", "/")
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
        lines_out.append(json.dumps(obj, ensure_ascii=False))

    Path(args.out).write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    print(f"[mls] 输出 {len(lines_out)} 条 -> {args.out}")


if __name__ == "__main__":
    main()
