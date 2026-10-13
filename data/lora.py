# -*- coding: utf-8 -*-
"""
AuK Flux2Edit 骨干的极简 LoRA 实现（不依赖 peft）。

在原始实现（只适配注意力 QKV 投影与 FFN 线性层）基础上，扩展可 LoRA 的目标层集合，
用于需要更大容量的跨语言适配（如法语）微调。新增目标层（均为 nn.Linear）：

  - txt_proj                文本投影 Linear(text_hidden_dim, dim)
  - proj_out                最终输出投影 Linear(dim, latent_dim)
  - audio_embed.linear      音频 prompt 嵌入 Linear(latent_dim, dim)
  - attn_norm(_x/_c).linear / norm_out.linear   adaLN 调制层 Linear(dim, dim*6 或 dim*2)
  - attn.to_out.0           注意力输出投影（原实现因 ModuleList 内层名 "0" 未命中）
  - time_embed.time_mlp.0/.2 timestep MLP 两层 Linear

仍保持冻结：各类 norm、卷积位置编码、time_embed 除 MLP 外的部分、以及全部 text_encoder。

匹配方式从「直接子模块名」改为「完整路径后缀」（named_modules() + get_submodule），
从而能命中 ModuleList / Sequential 内部、以及多个语义不同但都叫 "linear" 的层。
"""
from __future__ import annotations

import math

import torch
from torch import nn

# ---------------------------------------------------------------------------
# 可 LoRA 的目标层（相对 Flux2Edit 根的完整路径后缀，用 endswith 匹配）
# ---------------------------------------------------------------------------
DEFAULT_TARGETS = [
    # 注意力：音频流 QKV / 上下文 QKV / 音频输出投影 / 上下文输出投影
    ".attn.to_qkv",
    ".attn.to_qkv_c",
    ".attn.to_out.0",
    ".attn.to_out_c",
    # FFN（DiT 单流 ff；MMDiT 双流 ff_c / ff_x）
    ".ff.linear_in",
    ".ff.linear_out",
    ".ff_c.linear_in",
    ".ff_c.linear_out",
    ".ff_x.linear_in",
    ".ff_x.linear_out",
    # 文本 / 音频 / 输出投影
    ".txt_proj",
    ".proj_out",
    ".audio_embed.linear",
    # adaLN 调制层
    ".attn_norm.linear",
    ".attn_norm_c.linear",
    ".attn_norm_x.linear",
    ".norm_out.linear",
    # timestep MLP
    ".time_embed.time_mlp.0",
    ".time_embed.time_mlp.2",
]

# 旧版目标集合（只适配注意力 QKV 与 FF 线性），保留用于对比 / 回退
LEGACY_TARGETS = [
    ".attn.to_qkv",
    ".attn.to_qkv_c",
    ".attn.to_out_c",
    ".ff.linear_in",
    ".ff.linear_out",
    ".ff_c.linear_in",
    ".ff_c.linear_out",
    ".ff_x.linear_in",
    ".ff_x.linear_out",
]


class LoRALinear(nn.Linear):
    def __init__(self, in_features, out_features, r=32, alpha=32, dropout=0.0, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.r = int(r)
        self.alpha = int(alpha)
        # 存成 buffer，便于合并/导出时直接从 ckpt 恢复 scaling
        self.register_buffer("lora_scaling", torch.tensor(alpha / r, dtype=torch.float32))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        # 冻结原始权重与 bias，只训练 LoRA
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)  # B=0 -> 初始等效原权重

    def forward(self, x):
        out = super().forward(x)  # 冻结的 W x + b
        if self.r > 0:
            x = self.lora_dropout(x)
            out = out + (x @ self.lora_A.t() @ self.lora_B.t()) * self.lora_scaling
        return out


def _parse_targets(targets):
    """把 targets 规整为后缀列表。支持 None（默认）/ 字符串（逗号分隔）/ 列表。"""
    if targets is None:
        return list(DEFAULT_TARGETS)
    if isinstance(targets, str):
        return [t.strip() for t in targets.split(",") if t.strip()]
    return [t for t in targets]


def apply_lora(model: nn.Module, r=32, alpha=32, dropout=0.0, targets=None) -> int:
    """把 model 中匹配 targets（完整路径后缀）的 Linear 替换为 LoRALinear，返回替换数。"""
    targets = _parse_targets(targets)
    count = 0

    # 先物化 named_modules()，避免替换过程中迭代器失效
    for full_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(full_name.endswith(suf) for suf in targets):
            continue

        parent_name, _, leaf = full_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model

        new = LoRALinear(
            module.in_features, module.out_features,
            r=r, alpha=alpha, dropout=dropout, bias=module.bias is not None,
        )
        new.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            new.bias.data.copy_(module.bias.data)

        # setattr 会经 nn.Module.__setattr__ 注册进 _modules，对 ModuleList("0") / Sequential("0") 同样生效
        setattr(parent, leaf, new)
        count += 1

    return count


def freeze_for_lora(model: nn.Module) -> int:
    """冻结 model 全部参数，仅保留 lora_A / lora_B 可训；返回可训参数个数。"""
    for p in model.parameters():
        p.requires_grad_(False)
    n = 0
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad_(True)
            n += 1
    return n
