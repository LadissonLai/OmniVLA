import os
import time
import draccus

# 防止 PyTorch 显存碎片化导致 OOM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import wandb
import torch
import torch.nn as nn
from dataclasses import dataclass
from pathlib import Path
from collections import deque
from datetime import datetime, timedelta
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from accelerate import PartialState
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, AutoConfig, AutoImageProcessor, AutoModelForVision2Seq
from huggingface_hub import snapshot_download
import tqdm

from core.exp_parking_dataset import ClassicVLADataset, BalancedBatchSampler, classic_collate_fn
from core.modeling_prismatic import ClassicVLAForActionPrediction
from core.configuration_prismatic import OpenVLAConfig
from core.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from core.utils import model_is_on_hf_hub, visualize_train_expvla


@dataclass
class ClassicVLATrainConfig:
    vla_path: str = "/public/home/lqq_202430131053/codes/OmniVLA/openvla-7b"
    data_root: str = "/public/home/lqq_202430131053/codes/datasets/ParkingVLA"
    run_root_dir: Path = Path("runs_classicvla_smart_history")

    # 训练超参数
    batch_size: int = 2
    learning_rate: float = 1e-4
    grad_accumulation_steps: int = 16
    epochs: int = 15
    save_freq: int = 512
    resume: bool = False
    resume_dir: str = ""
    num_workers: int = 2
    lr_warmup_steps: int = 500
    num_steps_before_decay: int = 10000

    # 可视化
    visualize_traj: bool = True
    visualize_dir: str = "vis_classicvla_smart_history_train"

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
    wandb_dir: str = "wandb_classicvla_smart_history"
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "ClassicVLA-Parking"
    wandb_log_freq: int = 64

    

    # generate 时最多生成的新 token 数
    max_new_tokens: int = 512


def wrap_ddp(module, device_id):
    from torch.nn.parallel import DistributedDataParallel as DDP
    return DDP(module, device_ids=[device_id], find_unused_parameters=True)


def save_checkpoint(cfg, run_dir, log_step, vla, processor, loss=None):
    run_dir = Path(run_dir)
    suffix = f"step_{log_step}_loss_{loss:.4f}_ckpt" if loss is not None else f"step_{log_step}_ckpt"
    chkpt_dir = run_dir / suffix
    os.makedirs(chkpt_dir, exist_ok=True)
    processor.save_pretrained(chkpt_dir)
    vla.module.save_pretrained(chkpt_dir / "lora_adapter")
    print(f"✅ Checkpoint saved at {chkpt_dir}")


@draccus.wrap()
def train(cfg: ClassicVLATrainConfig):
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)

    if distributed_state.is_main_process:
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

    vla = wrap_ddp(vla, device_id)

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
    sampler = BalancedBatchSampler(dataset.pos_indices, dataset.neg_indices, num_neg_per_batch=cfg.batch_size)
    dataloader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=cfg.num_workers,
        collate_fn=lambda b: classic_collate_fn(b, pad_token_id),
    )

    # === 3. 优化器 ===
    trainable_params = [p for p in vla.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # 学习率预热与衰减
    original_lr = optimizer.param_groups[0]["lr"]
    scheduler = MultiStepLR(
        optimizer,
        milestones=[cfg.num_steps_before_decay],
        gamma=0.1,
    )

    recent_losses = deque(maxlen=50)

    # === 4. 训练循环 ===
    vla.train()
    optimizer.zero_grad()

    global_step = 0
    total_steps = cfg.epochs * len(dataloader)
    train_start_time = time.time()
    step_times = deque(maxlen=50)

    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    save_ckpt_dir = os.path.join(cfg.run_root_dir, date_str)
    os.makedirs(save_ckpt_dir, exist_ok=True)
    visualize_dir = os.path.join(cfg.visualize_dir, date_str)
    os.makedirs(visualize_dir, exist_ok=True)

    for epoch in range(cfg.epochs):
        progress = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.epochs}", leave=False)
        for batch_idx, batch in enumerate(progress):
            step_start = time.time()

            # move pixel values to device
            for k in ['pixel_values_front', 'pixel_values_rear', 'pixel_values_left', 'pixel_values_right']:
                batch[k] = batch[k].to(device_id)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = vla(batch)
                total_loss = outputs.loss
                normalized_loss = total_loss / cfg.grad_accumulation_steps

            normalized_loss.backward()
            recent_losses.append(total_loss.item())

            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                gradient_step_idx = global_step // cfg.grad_accumulation_steps
                
                # [If applicable] Linearly warm up learning rate from 10% to 100% of original
                if cfg.lr_warmup_steps > 0:
                    lr_progress = min((gradient_step_idx + 1) / cfg.lr_warmup_steps, 1.0)
                    current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = current_lr

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
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

                if global_step % cfg.wandb_log_freq == 0:
                    wandb.log({
                        "Loss/Total": sum(recent_losses) / len(recent_losses),
                        "Learning Rate": scheduler.get_last_lr()[0],
                    }, step=global_step)

                if global_step > 0 and global_step % cfg.save_freq == 0:
                    save_checkpoint(
                        cfg=cfg, run_dir=save_ckpt_dir, log_step=global_step,
                        vla=vla, processor=processor, loss=total_loss.item(),
                    )
                    if cfg.visualize_traj:
                        # 用 generate_answer 跑一条样本可视化（只取 batch 第 0 条）
                        _vis_batch = {k: (v[:1] if isinstance(v, torch.Tensor) else v[:1])
                                      for k, v in batch.items()
                                      if k != 'output_ids' and k != 'eos_ids' and not isinstance(v, int)}
                        _vis_batch['pad_token_id'] = batch['pad_token_id']
                        with torch.no_grad():
                            gen_ids = vla.module.generate_answer(_vis_batch, max_new_tokens=cfg.max_new_tokens)
                        gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                        print(f"Generated text:\n {gen_text[0]}")

                        import re, json as _json
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
                            step=global_step,
                        )
    
    if distributed_state.is_main_process:
        save_checkpoint(
                        cfg=cfg, run_dir=save_ckpt_dir, log_step=global_step,
                        vla=vla, processor=processor, loss=total_loss.item(),
                    )
        print("🎉 Training completed!")
        wandb.finish()


if __name__ == "__main__":
    train()
