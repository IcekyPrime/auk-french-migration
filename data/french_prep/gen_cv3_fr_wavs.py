#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AuK 法语微调模型 -> CV3-Eval 零样本法语评测集 批量生成脚本。

读 CV3-Eval 的 fr 目标文本（text 文件），用 AuK french_data 里的法语 wav 作参考音色，
逐条生成，输出对齐 CV3-Eval 的 uttid 命名（uttid_N.wav），并生成用于说话人相似度评测
的 prompt_wav.scp（指向 french_data 参考音频）。

用法（conda auk，在 AuK 项目根）:
    python data/french_prep/gen_cv3_fr_wavs.py \
        --text      /hpc_stor03/sjtu_home/haotian.xie/deeplearning_course_sjtu/CV3-Eval/data/zero_shot/fr/text \
        --ref_dir   /hpc_stor03/sjtu_home/haotian.xie/deeplearning_course_sjtu/AuK/data/french_prep/french_data \
        --out_dir   /hpc_stor03/sjtu_home/haotian.xie/deeplearning_course_sjtu/CV3-Eval/results/zero_shot/fr/wavs \
        --prompt_scp_out /hpc_stor03/sjtu_home/haotian.xie/deeplearning_course_sjtu/CV3-Eval/results/zero_shot/fr/prompt_wav.scp \
        --cpu_offload
"""
from __future__ import annotations

import argparse
import os
import sys

# 把 AuK 的 src 加入 sys.path，便于 import auk（无需依赖外部 PYTHONPATH）
AUK_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(AUK_ROOT, "src"))

from auk.infer.infer_auk import AukInfer, save_audio

# 法语零样本 TTS 指令模板（必须与训练时逐字一致）
INSTRUCTION_TPL = 'Dis le texte suivant dans la même voix : "{text}"'


def load_text(text_file: str):
    items = []
    for line in open(text_file, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        items.append((parts[0], parts[1]))
    return items


def scan_ref_audios(ref_dir: str):
    """按文件名排序扫描参考音频，排除 speaker_id 缺失（None_）的样本。"""
    audios = []
    for name in sorted(os.listdir(ref_dir)):
        if name.lower().endswith(".wav") and not name.startswith("None_"):
            audios.append(name)
    return audios


def est_seconds(text: str, cps: float = 13.0, min_sec: float = 2.0):
    """按字符数粗估目标语音时长（法语约 13 字符/秒）。"""
    return max(min_sec, len(text) / cps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="CV3-Eval fr/text（uttid + 目标文本）")
    ap.add_argument("--ref_dir", required=True, help="AuK french_data 参考音频目录")
    ap.add_argument("--out_dir", required=True, help="生成音频输出目录（decode_dir/fr/wavs）")
    ap.add_argument("--prompt_scp_out", required=True, help="输出的 prompt_wav.scp（用于说话人相似度）")
    ap.add_argument("--ckpt", default=os.path.join(AUK_ROOT, "ckpts/auk_lora_french/auk_lora_french_merged.safetensors"))
    ap.add_argument("--config", default=os.path.join(AUK_ROOT, "ckpts/auk_lora_french/config.yaml"))
    ap.add_argument("--cpu_offload", action="store_true", help="单卡显存不足时开启（Qwen/DiT 空闲卸到 CPU）")
    ap.add_argument("--nfe", type=int, default=32)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0, help="固定随机种子，保证 base 与微调后可复现对比")
    ap.add_argument("--max_samples", type=int, default=0, help=">0 时只生成前 N 条（快速试跑）")
    ap.add_argument("--device", default=None, help="cuda / cuda:0 / cpu")
    args = ap.parse_args()

    items = load_text(args.text)
    ref_audios = scan_ref_audios(args.ref_dir)
    n = min(len(items), len(ref_audios))
    if args.max_samples > 0:
        n = min(n, args.max_samples)
    print(f"[gen] text={len(items)} ref_audios={len(ref_audios)} -> 生成 {n} 条")

    if n == 0:
        raise SystemExit("[gen] 无可用数据，请检查 --text 与 --ref_dir")

    engine = AukInfer(
        config_path=args.config,
        ckpt_path=args.ckpt,
        device=args.device,
        cpu_offload=args.cpu_offload,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    scp_lines = []
    for i in range(n):
        uttid, text = items[i]
        ref_name = ref_audios[i]
        ref_path = os.path.join(args.ref_dir, ref_name)
        gen_seconds = est_seconds(text)
        instruction = INSTRUCTION_TPL.format(text=text)
        messages = [{"role": "user", "content": [
            {"type": "text", "text": instruction},
            {"type": "audio", "audio": ref_path},
        ]}]
        out_path = os.path.join(args.out_dir, f"{uttid}.wav")
        print(f"[gen {i + 1}/{n}] {uttid} <- {ref_name} | ~{gen_seconds:.1f}s")
        try:
            audio, sr = engine.generate(
                messages,
                audio=ref_path,
                gen_seconds=gen_seconds,
                nfe=args.nfe,
                cfg_strength=args.cfg,
                sway_sampling_coef=-1.0,
                seed=args.seed,
            )
            save_audio(audio, sr, out_path)
            scp_lines.append(f"{uttid} {ref_path}\n")
        except Exception as e:
            print(f"[skip] {uttid}: {e}")

    with open(args.prompt_scp_out, "w", encoding="utf-8") as f:
        f.writelines(scp_lines)
    print(f"[done] 生成完成 {len(scp_lines)} 条 | prompt_wav.scp -> {args.prompt_scp_out}")


if __name__ == "__main__":
    main()
