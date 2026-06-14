import os
import math
import time
import draccus

# 防止 PyTorch 显存碎片化导致 OOM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import wandb
import torch
import torch.nn as nn
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from collections import deque
from datetime import datetime, timedelta
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from accelerate import PartialState
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, AutoConfig, AutoImageProcessor, AutoModelForVision2Seq
from huggingface_hub import snapshot_download
import tqdm

from core.exp_parking_dataset import ClassicVLADataset, classic_collate_fn
from core.modeling_prismatic import ClassicVLAForActionPrediction
from core.configuration_prismatic import OpenVLAConfig
from core.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from core.utils import model_is_on_hf_hub, visualize_train_expvla


@dataclass
class ClassicVLATrainConfig:
    vla_path: str = "/root/autodl-tmp/codes/OmniVLA/openvla-7b"
    data_root: str = "/root/autodl-tmp/codes/OmniVLA/datasets/ParkingVLA"
    run_root_dir: Path = Path("runs/runs_classicvla_smart_history")

    # 训练超参数（batch_size 为每张卡）
    batch_size: int = 2
    learning_rate: float = 1e-4
    grad_accumulation_steps: int = 16
    epochs: int = 15
    save_freq: int = 512          # 单位：优化器步
    resume: bool = False
    resume_dir: str = ""
    num_workers: int = 2          # 每进程

    # 学习率调度（按总优化器步数的比例）：
    # 前 warmup_ratio 线性从 10% 预热到 100%，之后恒定，
    # 到 decay_ratio 处再 x0.1。
    warmup_ratio: float = 0.05
    decay_ratio: float = 0.85

    # 可视化
    visualize_traj: bool = True
    visualize_dir: str = "vis/vis_classicvla_smart_history_train"

    # 历史轨迹配置
    history_mode: str = 'smart'
    max_history: int = 8
    future_steps: int = 8
    distance_interval: float = 0.5
    turn_yaw_thresh: float = 5.0
    turn_dense_interval: float = 0.1

    # LoRA 配置
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.05

    # Logging
    wandb_dir: str = "wandb/wandb_classicvla_smart_history"
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "ClassicVLA-Parking"
    wandb_log_freq: int = 16      # 单位：优化器步

    # generate 时最多生成的新 token 数
    max_new_tokens: int = 512


def wrap_ddp(module, device_id, use_distributed):
    if not use_distributed:
        return module
    return DDP(module, device_ids=[device_id], find_unused_parameters=True)


def unwrap(module):
    return module.module if isinstance(module, DDP) else module


def save_checkpoint(cfg, run_dir, log_step, vla, processor, loss=None):
    run_dir = Path(run_dir)
    suffix = f"step_{log_step}_loss_{loss:.4f}_ckpt" if loss is not None else f"step_{log_step}_ckpt"
    chkpt_dir = run_dir / suffix
    os.makedirs(chkpt_dir, exist_ok=True)
    processor.save_pretrained(chkpt_dir)
    unwrap(vla).save_pretrained(chkpt_dir / "lora_adapter")
    print(f"✅ Checkpoint saved at {chkpt_dir}")


def run_visualization(cfg, vla, tokenizer, batch, epoch, optim_step, visualize_dir, device_id):
    """用 generate_answer 跑一条样本可视化（只取 batch 第 0 条）。仅主进程调用。"""
    import re, json as _json

    _vis_batch = {k: (v[:1] if isinstance(v, torch.Tensor) else v[:1])
                  for k, v in batch.items()
                  if k != 'output_ids' and k != 'eos_ids' and not isinstance(v, int)}
    _vis_batch['pad_token_id'] = batch['pad_token_id']
    with torch.no_grad():
        gen_ids = unwrap(vla).generate_answer(_vis_batch, max_new_tokens=cfg.max_new_tokens)
    gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    print(f"Generated text:\n {gen_text[0]}")

    pred_actions_list = []
    pred_decisions_str = []
    gt_actions_list = []
    gt_decisions_str = []
    # 从 output_ids 解码 GT 文本，再解析出轨迹浮点值用于可视化
    gt_texts = tokenizer.batch_decode(
        [batch['output_ids'][i] for i in range(len(gen_text))],
        skip_special_tokens=True
    )
    for i, text in enumerate(gen_text):
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                obj = _json.loads(m.group())
                print(f"Generated json:\n {obj}")
                pred_traj = torch.tensor(obj['trajectory'], dtype=torch.float32)
                pred_actions_list.append(pred_traj)
                pred_decisions_str.append(str(obj.get('decision', '?')))
            except Exception:
                pred_actions_list.append(torch.zeros(8, 4))
                pred_decisions_str.append('parse_err')
                print(f"❌ Failed to parse JSON from generated text:\n{text}")
        else:
            pred_actions_list.append(torch.zeros(8, 4))
            pred_decisions_str.append('no_json')
        gt_m = re.search(r'\{.*\}', gt_texts[i], re.DOTALL)
        if gt_m:
            try:
                gt_obj = _json.loads(gt_m.group())
                gt_actions_list.append(torch.tensor(gt_obj['trajectory'], dtype=torch.float32))
                gt_decisions_str.append(str(gt_obj.get('decision', '?')))
            except Exception:
                gt_actions_list.append(torch.zeros(8, 4))
                gt_decisions_str.append('parse_err')
        else:
            gt_actions_list.append(torch.zeros(8, 4))
            gt_decisions_str.append('no_json')
        print(f"Pred decision: {pred_decisions_str[-1]}, Pred actions:\n {pred_actions_list[-1]}")

    visualize_train_expvla(
        project_folder=visualize_dir,
        pred_actions=torch.stack(pred_actions_list),
        gt_actions=torch.stack(gt_actions_list),
        pred_decisions=pred_decisions_str,
        gt_decisions=gt_decisions_str,
        instructions=batch.get('instruction', [''] * len(gen_text)),
        images_front=batch['pixel_values_front'],
        images_rear=batch['pixel_values_rear'],
        images_left=batch['pixel_values_left'],
        images_right=batch['pixel_values_right'],
        epoch=epoch,
        step=optim_step,
    )


@draccus.wrap()
def train(cfg: ClassicVLATrainConfig):
    distributed_state = PartialState()
    use_distributed = distributed_state.use_distributed
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)

    if distributed_state.is_main_process:
        os.makedirs(cfg.wandb_dir, exist_ok=True)
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name="classicvla_training", dir=cfg.wandb_dir)
        os.makedirs(cfg.run_root_dir, exist_ok=True)

    # === 1. 处理器与模型加载 ===
    Load_hf = model_is_on_hf_hub(cfg.vla_path)
    if Load_hf:
        cfg.vla_path = snapshot_download(repo_id=cfg.vla_path)
    else:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, ClassicVLAForActionPrediction)

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id else 0

    vla = ClassicVLAForActionPrediction.from_pretrained(
        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device_id)

    if cfg.use_lora:
        if cfg.resume and cfg.resume_dir:
            from peft import PeftModel
            vla = PeftModel.from_pretrained(vla, os.path.join(cfg.resume_dir, "lora_adapter"), is_trainable=True)
            if distributed_state.is_main_process:
                print(f"✅ Resumed LoRA adapter from {cfg.resume_dir}")
        else:
            target_modules = [name for name, module in vla.named_modules() if isinstance(module, nn.Linear)]
            lora_config = LoraConfig(
                r=cfg.lora_rank, lora_alpha=16, lora_dropout=cfg.lora_dropout,
                target_modules=target_modules, init_lora_weights="gaussian"
            )
            vla = get_peft_model(vla, lora_config)

    vla = wrap_ddp(vla, device_id, use_distributed)

    # === 2. 数据加载 ===
    dataset = ClassicVLADataset(
        cfg.data_root, tokenizer,
        processor.image_processor.apply_transform,
        future_steps=cfg.future_steps,
        max_history=cfg.max_history,
        history_mode=cfg.history_mode,
        distance_interval=cfg.distance_interval,
        turn_yaw_thresh=cfg.turn_yaw_thresh,
        turn_dense_interval=cfg.turn_dense_interval,
    )
    sampler = DistributedSampler(dataset, shuffle=True) if use_distributed else None
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=lambda b: classic_collate_fn(b, pad_token_id),
    )

    # === 3. 优化器与调度器 ===
    trainable_params = [p for p in vla.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    steps_per_epoch = math.ceil(len(dataloader) / cfg.grad_accumulation_steps)
    total_optim_steps = cfg.epochs * steps_per_epoch
    warmup_steps = max(1, int(total_optim_steps * cfg.warmup_ratio))
    decay_step = int(total_optim_steps * cfg.decay_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return 0.1 + 0.9 * (step + 1) / warmup_steps
        return 0.1 if step >= decay_step else 1.0

    scheduler = LambdaLR(optimizer, lr_lambda)

    if distributed_state.is_main_process:
        print(f"LR schedule: {total_optim_steps} optimizer steps total, "
              f"warmup {warmup_steps}, decay x0.1 at {decay_step}")

    recent_losses = deque(maxlen=50)

    # === 4. 训练循环 ===
    vla.train()
    optimizer.zero_grad()

    global_step = 0   # micro-step（每 rank 的 batch）
    optim_step = 0    # 优化器更新次数
    total_steps = cfg.epochs * len(dataloader)
    train_start_time = time.time()
    step_times = deque(maxlen=50)

    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    save_ckpt_dir = os.path.join(cfg.run_root_dir, date_str)
    visualize_dir = os.path.join(cfg.visualize_dir, date_str)
    if distributed_state.is_main_process:
        os.makedirs(save_ckpt_dir, exist_ok=True)
        os.makedirs(visualize_dir, exist_ok=True)

    for epoch in range(cfg.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        progress = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.epochs}", leave=False)
        for batch_idx, batch in enumerate(progress):
            step_start = time.time()

            is_update_step = (
                (batch_idx + 1) % cfg.grad_accumulation_steps == 0
                or (batch_idx + 1) == len(dataloader)
            )

            # move pixel values to device
            for k in ['pixel_values_front', 'pixel_values_rear', 'pixel_values_left', 'pixel_values_right']:
                batch[k] = batch[k].to(device_id)

            # 累积步跳过 DDP 梯度同步
            with ExitStack() as sync_ctx:
                if use_distributed and not is_update_step:
                    sync_ctx.enter_context(vla.no_sync())

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = vla(batch)
                    total_loss = outputs.loss
                    normalized_loss = total_loss / cfg.grad_accumulation_steps

                normalized_loss.backward()

            recent_losses.append(total_loss.item())

            if is_update_step:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optim_step += 1
                # 缓解各种变长序列带来的显存碎片问题
                torch.cuda.empty_cache()

            global_step += 1
            step_elapsed = time.time() - step_start
            step_times.append(step_elapsed)

            if distributed_state.is_main_process:
                avg_step_time = sum(step_times) / len(step_times)
                eta_seconds = avg_step_time * (total_steps - global_step)
                eta_str = str(timedelta(seconds=int(eta_seconds)))
                elapsed_str = str(timedelta(seconds=int(time.time() - train_start_time)))

                progress.set_postfix({
                    "Loss": f"{total_loss.item():.4f}",
                    "Elapsed": elapsed_str,
                    "ETA": eta_str,
                })

                if is_update_step and optim_step % cfg.wandb_log_freq == 0:
                    wandb.log({
                        "Loss/Total": sum(recent_losses) / len(recent_losses),
                        "Learning Rate": scheduler.get_last_lr()[0],
                    }, step=optim_step)

            # checkpoint + 可视化：主进程做完后全员 barrier，避免其它 rank 冲进下个 backward 触发 NCCL 超时
            if is_update_step and optim_step > 0 and optim_step % cfg.save_freq == 0:
                if distributed_state.is_main_process:
                    save_checkpoint(
                        cfg=cfg, run_dir=save_ckpt_dir, log_step=optim_step,
                        vla=vla, processor=processor, loss=total_loss.item(),
                    )
                    if cfg.visualize_traj:
                        run_visualization(cfg, vla, tokenizer, batch, epoch, optim_step, visualize_dir, device_id)
                distributed_state.wait_for_everyone()

    if distributed_state.is_main_process:
        save_checkpoint(
            cfg=cfg, run_dir=save_ckpt_dir, log_step=optim_step,
            vla=vla, processor=processor, loss=total_loss.item(),
        )
    distributed_state.wait_for_everyone()

    if distributed_state.is_main_process:
        print("🎉 Training completed!")
        wandb.finish()


if __name__ == "__main__":
    train()
