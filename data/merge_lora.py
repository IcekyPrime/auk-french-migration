# -*- coding: utf-8 -*-
"""
合并 LoRA 权重回骨干，导出与 auk_base.safetensors 同格式的部署档。

官方推理入口 AukInfer 构建的是不带 LoRA 的普通 Flux2Edit，直接指向训练产出的
model_last.pt 会把 lora_* 键当 unexpected 忽略、base 仍是冻结原值（LoRA 效果丢失）。
因此需把 LoRA 折叠回权重（W' = W + B@A * scaling）后导出干净 safetensors，再喂给
auk-infer / AukInfer。
"""
from __future__ import annotations

import argparse

import torch
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="训练产出的 model_last.pt（含 ema_model_state_dict）")
    ap.add_argument("--out", required=True, help="输出合并后的 safetensors 路径")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ema = ckpt["ema_model_state_dict"]
    sd = {k.replace("ema_model.", ""): v for k, v in ema.items() if k not in ("initted", "step")}

    # 收集 lora 键：lora_A / lora_B / lora_scaling，前缀为其所属线性层
    lora = {}
    for k in list(sd.keys()):
        for suf in ("lora_A", "lora_B", "lora_scaling"):
            if k.endswith("." + suf):
                prefix = k[: -len(suf) - 1]
                lora.setdefault(prefix, {})[suf] = sd.pop(k)
                break

    # 折叠：W' = W + (B @ A) * scaling
    for prefix, m in lora.items():
        A = m["lora_A"]          # [r, in]
        B = m["lora_B"]          # [out, r]
        scaling = m.get("lora_scaling", torch.tensor(1.0))  # 缺省 alpha=r
        delta = (B @ A) * scaling.to(B.dtype)               # [out, in]
        w = sd[prefix + ".weight"]
        sd[prefix + ".weight"] = (w + delta.to(w.dtype)).contiguous()

    # 去掉 text_encoder.*（推理时单独从 Qwen snapshot 加载）
    sd = {k: v for k, v in sd.items() if not k.startswith("text_encoder.")}
    save_file(sd, args.out)
    print(f"saved {args.out} | {len(sd)} tensors | merged lora layers: {len(lora)}")


if __name__ == "__main__":
    main()
