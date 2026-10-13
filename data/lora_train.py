# -*- coding: utf-8 -*-
"""
AuK 的 LoRA 微调入口（不改官方 src/auk/train/train.py）。

复用官方 train_loop 之外的组件，但重写 train_loop 以让 EMA 只包裹生成骨干
（transformer + layer fusion 参数），不复制冻结的 Qwen text encoder——官方 EMA
会 deepcopy 整个 model（含 3B Qwen，约 6GB），在 24GB 卡上会 OOM。

checkpoint 的 key 结构与官方完全一致（ema_model.transformer.* / ema_model.layer_weights），
因此 save_checkpoint、推理端 AukInfer、以及 merge_lora.py 均无需改动。

用法：在 AuK 项目根，PYTHONPATH=src，accelerate launch data/lora_train.py ...
"""
from __future__ import annotations

import math
import os
import random
import shutil
from dataclasses import dataclass, field

import torch
import torchaudio
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from ema_pytorch import EMA
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import HfArgumentParser, Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

from auk.model import CFMEdit, Flux2Edit
from auk.model.vae import load_vae_model
from auk.model.vae.bigvgan_flow_vae import BigVGANFlowVAEConfig
from auk.train.train import (
    AukJsonlDataset, DynamicBatchSampler, ScriptArgs, TrainConfig,
    evaluate, log_val_curve, logger, prepare_batch,
)

from lora import apply_lora, freeze_for_lora


def save_checkpoint(accelerator, model, optimizer, scheduler, update, output_dir, last=False):
    """与官方 save_checkpoint 一致，但关闭 EMA：部署权重直接取模型当前权重。

    为保持推理端 AukInfer 与 merge_lora.py 兼容，把模型权重以 ema_model.* 前缀写入
    ema_model_state_dict 字段。
    """
    # ZeRO-3 下 get_state_dict 会触发 all-gather 收集分片权重，这是一个跨 rank 的 collective，
    # 必须所有 rank 都执行（不能放在 is_main_process 之后，否则只有 rank 0 参与 gather，
    # 其余 rank 已 return → rank 0 卡在 _ALLGATHER_BASE 直到 600s 超时）。
    model_state = accelerator.get_state_dict(model)
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    model_state = {k: v for k, v in model_state.items() if not k.startswith("text_encoder.")}
    ema_state = {("ema_model." + k): v for k, v in model_state.items()}
    checkpoint = {
        "model_state_dict": model_state,
        "ema_model_state_dict": ema_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "update": update,
    }
    os.makedirs(output_dir, exist_ok=True)
    if last:
        accelerator.save(checkpoint, f"{output_dir}/model_last.pt")
        logger.info(f"[update {update}] saved model_last.pt")
        return
    accelerator.save(checkpoint, f"{output_dir}/model_{update}.pt")
    logger.info(f"[update {update}] saved model_{update}.pt")


@torch.no_grad()
def synthesize_and_save(accelerator, model, batch, vae_model, update, samples_dir):
    """ZeRO-3 安全版（官方 train.py 的原版只让 main process 跑 sample()）。

    官方版本开头 `if not accelerator.is_main_process: return` 在 DDP 下没问题（每 rank
    都有完整权重，前向非 collective）；但 ZeRO-3 权重是分片的，sample() 的 forward 会触发
    参数 _ALLGATHER_BASE 这个跨 rank collective。若只有 rank 0 跑前向，其余 rank 已回到
    训练循环卡在下一个 ALLREDUCE，两边等不同 collective → NCCL 600s 超时。
    因此这里让所有 rank 都执行前向，仅 main process 负责保存音频。
    """
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.eval()

    ref_audio = batch["ref_audio"][:1].to(accelerator.device)
    ref_audio_lengths = batch["ref_audio_lengths"][:1].to(accelerator.device)
    has_ref_audio = ref_audio.size(-1) > 0 and ref_audio_lengths[0].item() > 0
    if has_ref_audio:
        ref_latent, ref_lens = vae_model.encoding_and_normalization(ref_audio, sample_lengths=ref_audio_lengths)
        ref_len = ref_lens[0].item()
    else:
        ref_latent = torch.zeros(1, 0, vae_model.h.latent_dim, device=accelerator.device, dtype=torch.float32)
        ref_lens = torch.zeros(1, dtype=torch.long, device=accelerator.device)
        ref_len = 0

    target_audio = batch["audio"][:1].to(accelerator.device)
    target_lengths = batch["audio_lengths"][:1].to(accelerator.device)
    target_latent, target_latent_lens = vae_model.encoding_and_normalization(target_audio, sample_lengths=target_lengths)
    target_len = target_latent_lens[0].item()

    cond_inputs = unwrapped.build_cond_inputs(batch["messages"][:1], unwrapped.text_processor)

    mp = accelerator.mixed_precision
    autocast_dtype = torch.bfloat16 if mp == "bf16" else torch.float16 if mp == "fp16" else torch.float32
    with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_dtype != torch.float32):
        # 所有 rank 都必须走到这里，保证 ZeRO-3 参数 gather 的 collective 数量对齐。
        generated, _ = unwrapped.sample(
            cond=ref_latent,
            text=cond_inputs,
            duration=ref_len + target_len,
            lens=ref_lens,
            steps=32,
            cfg_strength=2.0,
            sway_sampling_coef=-1.0,
        )

    # 所有 rank 恢复训练模式（前面的 unwrapped.eval() 会关闭 dropout，非 main rank 若不恢复，
    # 后续训练前向会和 main rank 不一致）。
    model.train()

    if not accelerator.is_main_process:
        return

    generated = generated.to(torch.float32)

    os.makedirs(samples_dir, exist_ok=True)
    for name, latent in [
        ("gen", generated[:, ref_len : ref_len + target_len, :]),
        ("tgt", target_latent[:1, :target_len, :]),
    ]:
        latent = vae_model.denormalize(latent.to(accelerator.device))
        audio = vae_model.inference_from_latents(latent.permute(0, 2, 1)).cpu().to(torch.float32)
        if audio.ndim == 3:
            audio = audio.squeeze(0)
        if torch.isnan(audio).any() or torch.isinf(audio).any():
            logger.warning(f"[update {update}] {name} audio has NaN/Inf, skipping")
            continue
        torchaudio.save(f"{samples_dir}/update_{update}_{name}.wav", audio, 24000)
    logger.info(f"[update {update}] wrote synthesis samples to {samples_dir}")


class BackboneForEMA(nn.Module):
    """只含生成骨干（transformer + layer fusion 参数），供 EMA 包裹，排除冻结的 text_encoder。"""

    def __init__(self, cfm_model):
        super().__init__()
        self.transformer = cfm_model.transformer
        self.layer_weights = cfm_model.layer_weights
        self.layer_scale = cfm_model.layer_scale


@dataclass
class LoraArgs:
    lora_r: int = field(default=32, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha（scaling = alpha/r）"})
    lora_dropout: float = field(default=0.0, metadata={"help": "LoRA dropout"})
    lora_targets: str = field(
        default=None,
        metadata={"help": "逗号分隔的 LoRA 目标层后缀；缺省用 lora.DEFAULT_TARGETS（扩展集合）"},
    )


def train_loop(model, vae_model, train_dataset, val_dataset, train_config, output_dir):
    # 与官方 train_loop 一致，唯一区别：EMA 只包 BackboneForEMA，不复制冻结的 Qwen text encoder。
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # DeepSpeed 模式下 gradient_accumulation_steps / mixed_precision 等由 ds_config 控制，
    # 不能在这里传，否则 accelerate 会报配置冲突。
    # rng_types=[]：关闭 accelerate 每个 epoch 重新迭代 DataLoader 时做的 RNG 状态同步
    # （synchronize_rng_states）。多卡 + ZeRO-3 下该广播偶尔会得到非法 mt19937 状态，
    # 抛 "Invalid mt19937 state"，使部分 rank 崩掉、其余 rank 卡在 NCCL collective 超时
    # （watchdog 600s 后 SIGABRT 拉垮整个进程组）。RNG 同步只影响跨 rank 的 shuffle
    # 可复现性，与收敛无关，关闭即可。
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], rng_types=[])

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=train_config.learning_rate, betas=(0.9, 0.95))
    model, optimizer = accelerator.prepare(model, optimizer)

    vae_model = vae_model.to(accelerator.device)
    vae_model.requires_grad_(False)

    # ZeRO-3 会分片模型权重，无需再 CPU offload Qwen；关闭 EMA（省显存）。
    # LoRA 音色微调对 EMA 不敏感，部署权重直接取模型当前权重（见 save_checkpoint）。
    ema_model = None

    writer = SummaryWriter(f"{output_dir}/tensorboard") if accelerator.is_main_process else None

    # 多卡 ZeRO-3 下，所有 rank 每个 epoch 的 batch 数必须完全一致，否则某个 rank 会
    # 多跑一个 step：领先的 rank 进入下一个 step 的前向权重 all-gather（_ALLGATHER_BASE），
    # 而其余 rank 还停在梯度 ALLREDUCE，两者在不同 collective 上互相等待 → NCCL 600s 超时。
    # even_batches=True（accelerate 默认）保证 batch 数对齐；之前设 False 会因
    # len(dataset) 不能被 num_processes*bsz 整除而产生 batch 数错位。
    accelerator.even_batches = True
    # ZeRO-3 需要固定 micro batch size，改用固定 batch（去掉动态 batch sampler）。
    # batch_size 取 max_samples（默认 8；24GB 卡建议设小，如 4）
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=train_dataset.collate_fn,
        num_workers=train_config.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=False,
        batch_size=train_config.max_samples,
        shuffle=True,
    )

    # 注意：prepare 前的 len(train_dataloader) 是「单机」batch 数（如 94）。多卡 ZeRO-3 下
    # accelerate 会把它分片到 num_processes 个 rank，每个 rank 每 epoch 只处理
    # ceil(len/num_processes) 个 batch（94/7→14）。global_update 是各 rank 同步递增的，
    # 因此总 update 数必须除以 num_processes，否则分母虚高（1880 vs 实际 280），
    # lr schedule 也会按错误的步数规划。warmup_steps 同样按「每 rank 步数」计，不再乘 num_processes。
    total_updates = math.ceil(
        len(train_dataloader) / train_config.gradient_accumulation_steps / accelerator.num_processes
    ) * train_config.num_train_epochs
    warmup_updates = train_config.warmup_steps
    decay_updates = max(total_updates - warmup_updates, 1)
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_updates)
    decay_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_updates)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_updates])
    train_dataloader, scheduler = accelerator.prepare(train_dataloader, scheduler)

    val_loader = None
    val_dummy_input = None
    if val_dataset is not None:
        saved_even_batches = accelerator.even_batches
        accelerator.even_batches = True
        val_loader = DataLoader(
            val_dataset,
            collate_fn=val_dataset.collate_fn,
            num_workers=train_config.dataloader_num_workers,
            batch_size=4,
            shuffle=False,
        )
        val_loader = accelerator.prepare(val_loader)
        accelerator.even_batches = saved_even_batches
        val_dummy_input = next(iter(val_loader))

    samples_dir = f"{output_dir}/samples"
    val_curves_dir = f"{output_dir}/val_curves"
    updates_per_epoch = math.ceil(len(train_dataloader) / train_config.gradient_accumulation_steps)

    start_epoch = 0
    global_update = 0
    updates_done_in_epoch = 0
    resume_ckpt = os.path.join(output_dir, "model_last.pt")
    if os.path.exists(resume_ckpt):
        state = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        accelerator.unwrap_model(model).load_state_dict(state["model_state_dict"], strict=False)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        global_update = int(state["update"])
        start_epoch = global_update // updates_per_epoch
        updates_done_in_epoch = global_update % updates_per_epoch
        logger.info(
            f"resumed from {resume_ckpt} | update={global_update} "
            f"start_epoch={start_epoch} skip_batches={updates_done_in_epoch * train_config.gradient_accumulation_steps}"
        )

    for epoch in range(start_epoch, train_config.num_train_epochs):
        model.train()
        epoch_loader = train_dataloader
        if epoch == start_epoch and updates_done_in_epoch > 0:
            epoch_loader = accelerator.skip_first_batches(
                train_dataloader, updates_done_in_epoch * train_config.gradient_accumulation_steps
            )
        for batch in epoch_loader:
            with accelerator.accumulate(model):
                latent, latent_lens, cond_inputs, extras = prepare_batch(batch, vae_model)
                with accelerator.autocast():
                    loss, _, _ = model(latent, text=cond_inputs, lens=latent_lens, **extras)
                accelerator.backward(loss)
                if train_config.max_grad_norm > 0 and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_update += 1

                if accelerator.is_main_process and global_update % train_config.logging_steps == 0:
                    batch_size = latent.shape[0]
                    noise_len = latent.shape[1]
                    ref_len = extras["ref_latent"].shape[1]
                    text_len = cond_inputs["input_ids"].shape[1]
                    logger.info(
                        f"[epoch {epoch + 1}/{train_config.num_train_epochs} | update {global_update}/{total_updates}] "
                        f"loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e} "
                        f"bsz={batch_size} seq_len=noise:{noise_len}/ref:{ref_len}/text:{text_len}"
                    )

                if global_update % train_config.last_per_updates == 0:
                    save_checkpoint(accelerator, model, optimizer, scheduler, global_update, output_dir, last=True)
                if global_update % train_config.save_per_updates == 0:
                    save_checkpoint(accelerator, model, optimizer, scheduler, global_update, output_dir)
                if val_loader is not None and global_update % train_config.val_per_updates == 0:
                    t_grid, val_losses = evaluate(accelerator, model, val_loader, vae_model)
                    if accelerator.is_main_process:
                        log_val_curve(writer, val_curves_dir, t_grid, val_losses, global_update)
                    synthesize_and_save(accelerator, model, val_dummy_input, vae_model, global_update, samples_dir)

    save_checkpoint(accelerator, model, optimizer, scheduler, global_update, output_dir, last=True)
    if writer is not None:
        writer.close()
    accelerator.end_training()


def main():
    parser = HfArgumentParser((ScriptArgs, TrainConfig, LoraArgs))
    args, train_config, lora_args = parser.parse_args_into_dataclasses()

    random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)

    model_config = OmegaConf.load(args.config).model
    text_encoder_config = model_config.text_encoder
    vae_config = model_config.vae

    saved_config = os.path.join(args.output_dir, "config.yaml")
    if os.environ.get("RANK", "0") == "0" and not os.path.exists(saved_config):
        os.makedirs(args.output_dir, exist_ok=True)
        shutil.copy(args.config, saved_config)

    logger.info(f"Loading Qwen text encoder from {text_encoder_config.text_encoder_path} ...")
    thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        text_encoder_config.text_encoder_path, torch_dtype=torch.bfloat16
    )
    if thinker.visual is not None:
        del thinker.visual
        thinker.visual = None
    text_encoder = thinker
    text_processor = Qwen2_5OmniProcessor.from_pretrained(text_encoder_config.text_encoder_path)

    logger.info(f"Loading VAE from {vae_config.vae_model_path} ...")
    model_init_kwargs = OmegaConf.to_container(vae_config.get("model_init_kwargs", OmegaConf.create({})), resolve=True)
    vae_model = load_vae_model(
        vae_name=vae_config.vae_name,
        vae_cfg=BigVGANFlowVAEConfig.from_dict(model_init_kwargs),
        vae_ckpt=vae_config.vae_model_path,
        map_location="cpu",
    )
    vae_model.eval()
    vae_model.requires_grad_(False)

    logger.info("Building CFMEdit model ...")
    model_arc = OmegaConf.to_container(model_config.arch, resolve=True)
    model_arc["attn_backend"] = "torch"  # 训练不依赖 flash_attn，用 torch SDPA，免去编译 flash-attn
    # ZeRO-3 只分片模型权重，不分片激活；flux2_edit 默认不开启梯度 checkpointing，
    # 32 层 transformer 的前向激活会直接占满 24GB 显存。强制开启（每层重算）以省激活。
    model_arc["checkpoint_activations"] = True
    model_arc["checkpoint_every_n_layers"] = 1
    schedule_config = OmegaConf.to_container(model_config.get("schedule", OmegaConf.create({})), resolve=True)
    model = CFMEdit(
        transformer=Flux2Edit(**model_arc, latent_dim=vae_config.latent_dim),
        text_encoder=text_encoder,
        text_processor=text_processor,
        num_channels=vae_config.latent_dim,
        **schedule_config,
    )

    # 冷启动：先加载 init 权重（不含 lora），再注入 LoRA（B=0 等效原权重）。
    # resume 时 model_last.pt 已含 lora 键，由 train_loop 恢复，这里跳过冷启动加载。
    if os.path.exists(os.path.join(args.output_dir, "model_last.pt")):
        logger.info(f"found {args.output_dir}/model_last.pt | will resume, skipping init_ckpt cold-start load")
    else:
        logger.info(f"Loading init weights from {args.init_ckpt} ...")
        if args.init_ckpt.endswith(".safetensors"):
            state_dict = load_file(args.init_ckpt, device="cpu")
        else:
            checkpoint = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
            ema = checkpoint["ema_model_state_dict"]
            state_dict = {k.replace("ema_model.", ""): v for k, v in ema.items() if k not in ("initted", "step")}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        non_te_missing = [k for k in missing if not k.startswith("text_encoder.")]
        logger.info(f"init weights loaded | missing(non-text-encoder)={len(non_te_missing)} unexpected={len(unexpected)}")
        if non_te_missing:
            logger.warning(f"missing non-text-encoder keys, examples: {non_te_missing[:10]}")

    # 无条件注入 LoRA：冷启动 B=0 等效原权重；resume 时 train_loop 会从 model_last.pt
    # 把 lora_A/lora_B 权重恢复到已注入的结构上。targets=None 时用扩展集合。
    n_adapted = apply_lora(
        model.transformer,
        r=lora_args.lora_r,
        alpha=lora_args.lora_alpha,
        dropout=lora_args.lora_dropout,
        targets=lora_args.lora_targets,
    )
    n_trainable = freeze_for_lora(model.transformer)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"LoRA: adapted={n_adapted} layers | trainable={trainable:,}/{total:,} "
        f"({100 * trainable / total:.2f}%) | trainable_params={n_trainable}"
    )

    train_dataset = AukJsonlDataset(args.train_jsonl, text_processor)
    val_dataset = None
    if args.val_jsonl:
        val_dataset = AukJsonlDataset(args.val_jsonl, text_processor)

    logger.info(f"train={len(train_dataset)} val={len(val_dataset) if val_dataset else 0}")
    train_loop(model, vae_model, train_dataset, val_dataset, train_config, args.output_dir)


if __name__ == "__main__":
    main()
