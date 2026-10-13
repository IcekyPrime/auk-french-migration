#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AuK 法语微调模型 -> X-Voice-Testset fr 子集 批量生成脚本。

读 X-Voice-Testset 的 zero_shot/fr/{text, prompt_wav.scp}，用 AuK 逐条生成，
输出对齐 X-Voice 评测的 uttid 命名（decode_dir/fr/wavs/uttid.wav）。

X-Voice 评测（WER/SIM/UTMOS）由 src/x_voice/eval 下的脚本承担，本脚本只负责生成。

用法（conda auk，在 AuK 项目根）:
    python data/french_prep/gen_xvoice_fr_wavs.py \
        --text      /path/to/x_voice_eval/zero_shot/fr/text \
        --prompt_scp /path/to/x_voice_eval/zero_shot/fr/prompt_wav.scp \
        --dump_dir  /path/to/x_voice_eval \
        --out_dir   /path/to/decode_dir/fr/wavs \
        --ckpt      ckpts/auk_lora_french_v3/auk_lora_french_v3_merged.safetensors \
        --config    ckpts/auk_lora_french_v3/config.yaml \
        --cpu_offload
"""
from __future__ import annotations

import argparse
import os
import sys

AUK_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(AUK_ROOT, "src"))

from auk.infer.infer_auk import AukInfer, save_audio

# 法语零样本 TTS 指令模板（必须与训练时逐字一致）
INSTRUCTION_TPL = 'Dis le texte suivant dans la même voix : "{text}"'


def load_text(text_file):
    items = []
    for line in open(text_file, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            items.append((parts[0], parts[1]))
    return items


def load_prompt_scp(scp_file):
    mapping = {}
    for line in open(scp_file, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            mapping[parts[0]] = parts[1]
    return mapping


def est_seconds(text, cps=13.0, min_sec=2.0):
    return max(min_sec, len(text) / cps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--prompt_scp", required=True)
    ap.add_argument("--dump_dir", required=True, help="x_voice_eval 根（prompt 相对路径的基准）")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpt", default=os.path.join(AUK_ROOT, "ckpts/auk_lora_french/auk_lora_french_merged.safetensors"))
    ap.add_argument("--config", default=os.path.join(AUK_ROOT, "ckpts/auk_lora_french/config.yaml"))
    ap.add_argument("--cpu_offload", action="store_true")
    ap.add_argument("--nfe", type=int, default=32)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    items = load_text(args.text)
    prompt_scp = load_prompt_scp(args.prompt_scp)

    # 只保留 prompt_wav.scp 里存在的 uttid
    items = [(u, t) for u, t in items if u in prompt_scp]
    if args.max_samples > 0:
        items = items[: args.max_samples]
    print(f"[gen] text={len(items)} 条，prompt={len(prompt_scp)} 个，生成 {len(items)} 条")

    if not items:
        raise SystemExit("[gen] 无可用数据，请检查 --text / --prompt_scp")

    engine = AukInfer(config_path=args.config, ckpt_path=args.ckpt, device=args.device, cpu_offload=args.cpu_offload)
    os.makedirs(args.out_dir, exist_ok=True)

    done = 0
    for i, (uttid, text) in enumerate(items):
        rel = prompt_scp[uttid]
        prompt_path = rel if os.path.isabs(rel) else os.path.join(args.dump_dir, rel)
        gen_seconds = est_seconds(text)
        instruction = INSTRUCTION_TPL.format(text=text)
        messages = [{"role": "user", "content": [
            {"type": "text", "text": instruction},
            {"type": "audio", "audio": prompt_path},
        ]}]
        out_path = os.path.join(args.out_dir, f"{uttid}.wav")
        print(f"[gen {i + 1}/{len(items)}] {uttid} <- {rel} | ~{gen_seconds:.1f}s")
        try:
            audio, sr = engine.generate(
                messages, audio=prompt_path, gen_seconds=gen_seconds,
                nfe=args.nfe, cfg_strength=args.cfg, sway_sampling_coef=-1.0, seed=args.seed,
            )
            save_audio(audio, sr, out_path)
            done += 1
        except Exception as e:
            print(f"[skip] {uttid}: {e}")

    print(f"[done] 生成完成 {done}/{len(items)} 条 -> {args.out_dir}")


if __name__ == "__main__":
    main()
