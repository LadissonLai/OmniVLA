import os
import math
import time
import draccus
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
from transformers import AutoProcessor, AutoModelForVision2Seq

# 导入上面两个脚本的数据与模型
from core.exp_parking_dataset import ExpParkingDataset, BalancedBatchSampler, custom_collate_fn
from core.modeling_prismatic import ExpVLAForActionPrediction

from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
)
from core.utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map,
)
from huggingface_hub import HfApi, snapshot_download
from core.configuration_prismatic import OpenVLAConfig
from core.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
import tqdm

from core.constants import FUTURE_ACTION_WAYPOINTS, ACTION_DIM
from core.utils import visualize_train_expvla

@dataclass
class ExpVLAConfig:
    vla_path: str = "/public/home/lqq_202430131053/codes/OmniVLA/merged/openvla-7b-classic-merged"
    data_root: str = "/public/home/lqq_202430131053/codes/datasets/ParkingVLA"
    run_root_dir: Path = Path("runs_expvla_2_decode_smart_history_OFT2")
    
    # 学习率动态调节
    batch_size: int = 2
    learning_rate: float = 1e-4
    grad_accumulation_steps: int = 8
    epochs: int = 16
    save_freq: int = 512
    resume: bool = False
    resume_dir: str = ""
    num_workers: int = 2
    lr_warmup_steps: int = 500
    num_steps_before_decay: int = 2000
    
    # visulization
    visualize_traj: bool = True
    visualize_dir: str = "vis_exp_smart_history_OFT2_train"
    
    # History Trajectory 配置
    history_mode: str = 'smart'   # fixed_count: 采取固定数量的历史轨迹，如果大于8个则等间隔采用，如果少于8个则取少于8个，如果为0，则添加1个全0轨迹；
                                        # distance_interval: 采取固定距离间隔的历史轨迹，每隔一定距离采样一个轨迹点，数量动态变化。
                                        # smart: 如果历史为0，则添加1个全0轨迹； 动态数量历史轨迹，直线稀疏采样，弯道密集采样。
    distance_interval: float = 0.5      # for fixed_distance mode, in meters
    turn_yaw_thresh: float = 5.0        # for smart mode, minimum yaw change in degrees to consider it a turn, in degrees
    turn_dense_interval: float = 0.1    # for smart mode, minimum distance interval in meters when a turn is detected
    
    # LoRA 配置
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.05
    
    # Logging
    wandb_dir: str = "wandb_expvla_smart_history_OFT2"
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "ExpVLA-Parking"
    wandb_log_freq: int = 64
    
    # 损失权重
    W_ACT = 1.5
    W_SMOOTH = 0.5
    W_FORWARD = 0.5
    W_DEC = 1.0
    W_OBJ = 0.5

def wrap_ddp(module, device_id):
    from torch.nn.parallel import DistributedDataParallel as DDP
    return DDP(module, device_ids=[device_id], find_unused_parameters=True)

def save_training_checkpoint(
    cfg,
    run_dir,
    log_step,
    vla,
    processor,
    past_traj_projector,
    action_head,
    decision_head,
    loss=None,
) -> None:
    run_dir = Path(run_dir)
    if loss is not None:
        chkpt_dir = run_dir / f"step_{log_step}_loss_{loss:.4f}_ckpt"
    else:
        chkpt_dir = run_dir / f"step_{log_step}_ckpt"
    os.makedirs(chkpt_dir, exist_ok=True)
    
    # 保存 processor (包含 tokenizer, image processor 等周边配置文件)
    processor.save_pretrained(chkpt_dir)
    # 仅保存 LoRA adapter 权重，避免保存庞大的 LLM/VLM 基础网络权重
    vla.module.save_pretrained(chkpt_dir / "lora_adapter")
    torch.save(past_traj_projector.module.state_dict(), chkpt_dir / "past_traj_projector.pt")
    torch.save(action_head.module.state_dict(), chkpt_dir / "action_head.pt")
    torch.save(decision_head.module.state_dict(), chkpt_dir / "decision_head.pt")
    print(f"✅ Checkpoint saved at {chkpt_dir}")

@draccus.wrap()
def train_expvla(cfg: ExpVLAConfig):
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name="expvla_training", dir=cfg.wandb_dir)
        os.makedirs(cfg.run_root_dir, exist_ok=True)
        
    # === 1. 处理器与模型加载 ===
    print("model_is_on_hf_hub(cfg.vla_path)", model_is_on_hf_hub(cfg.vla_path))
    Load_hf = model_is_on_hf_hub(cfg.vla_path)
    if Load_hf:
        # Download model directly from Hugging Face Hub
        vla_download_path = snapshot_download(repo_id=cfg.vla_path)
        # Overwrite VLA path
        cfg.vla_path = vla_download_path
    else:
        # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, ExpVLAForActionPrediction)
        # print("*************************************")
        # print("Custom OpenVLA classes registered to Hugging Face Auto Classes.")
    
    
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id else 0
    
    vla = ExpVLAForActionPrediction.from_pretrained(
        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device_id)
    
    # print("=======================================")
    # print("vla class", type(vla))
    # print("llm class", type(vla.language_model))
    
    if cfg.use_lora:
        if cfg.resume and getattr(cfg, "resume_dir", ""):
            from peft import PeftModel
            vla = PeftModel.from_pretrained(vla, os.path.join(cfg.resume_dir, "lora_adapter"), is_trainable=True)
            if distributed_state.is_main_process:
                print(f"✅ Resumed LoRA adapter from {cfg.resume_dir}")
        else:
            target_modules =[name for name, module in vla.named_modules() if isinstance(module, nn.Linear)]
            lora_config = LoraConfig(
                r=cfg.lora_rank, lora_alpha=16, lora_dropout=cfg.lora_dropout,
                target_modules=target_modules, init_lora_weights="gaussian"
            )
            vla = get_peft_model(vla, lora_config)
    vla = wrap_ddp(vla, device_id)
    
    # === 2. 外部投影层与预测头初始化 ===
    llm_dim = vla.module.config.text_config.hidden_size
    vocab_size = vla.module.config.text_config.vocab_size
    
    past_traj_projector = nn.Sequential(
        nn.Linear(4, llm_dim // 2), nn.GELU(), nn.Linear(llm_dim // 2, llm_dim)
    ).to(device_id).to(torch.bfloat16)
    
    action_head = nn.Sequential(
        nn.Linear(llm_dim, llm_dim), nn.GELU(), nn.Linear(llm_dim, 1)  # 每个token输出一维，共32维
    ).to(device_id).to(torch.bfloat16)
    
    decision_head = nn.Linear(llm_dim, vocab_size).to(device_id).to(torch.bfloat16)

    if cfg.resume and getattr(cfg, "resume_dir", ""):
        def load_ckpt(model, _path):
            state_dict = torch.load(_path, map_location=f"cuda:{device_id}")
            # 兼容加载旧的检查点（包含 'module.' 前缀）
            new_state_dict = {k.replace("module.", "") if k.startswith("module.") else k: v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)

        load_ckpt(past_traj_projector, os.path.join(cfg.resume_dir, "past_traj_projector.pt"))
        load_ckpt(action_head, os.path.join(cfg.resume_dir, "action_head.pt"))
        load_ckpt(decision_head, os.path.join(cfg.resume_dir, "decision_head.pt"))
        if distributed_state.is_main_process:
            print(f"✅ Resumed custom heads from {cfg.resume_dir}")

    past_traj_projector = wrap_ddp(past_traj_projector, device_id)
    action_head = wrap_ddp(action_head, device_id)
    decision_head = wrap_ddp(decision_head, device_id)

    # === 3. 数据加载 ===
    dataset = ExpParkingDataset(
        cfg.data_root, 
        tokenizer, 
        processor.image_processor.apply_transform, 
        future_steps=FUTURE_ACTION_WAYPOINTS,
        history_mode=cfg.history_mode,
        distance_interval=cfg.distance_interval,
        turn_yaw_thresh=cfg.turn_yaw_thresh,
        turn_dense_interval=cfg.turn_dense_interval
    )
    # 这里的 cfg.batch_size 取其作为"负样本数量"的语义
    sampler = BalancedBatchSampler(dataset.pos_indices, dataset.neg_indices, num_neg_per_batch=cfg.batch_size)
    dataloader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=cfg.num_workers,
        collate_fn=lambda b: custom_collate_fn(b, pad_token_id)
    )
    
    # === 4. 优化器与损失函数 ===
    trainable_params =[p for p in vla.parameters() if p.requires_grad] + \
                       list(past_traj_projector.parameters()) + \
                       list(action_head.parameters()) + \
                       list(decision_head.parameters())
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # 学习率预热与衰减
    original_lr = optimizer.param_groups[0]["lr"]
    scheduler = MultiStepLR(
        optimizer,
        milestones=[cfg.num_steps_before_decay],
        gamma=0.1,
    )
    
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    
    
    recent_losses = deque(maxlen=50)

    # === 5. 训练循环 ===
    vla.train()
    optimizer.zero_grad()
    
    global_step = 0
    total_steps = cfg.epochs * len(dataloader)
    train_start_time = time.time()
    step_times = deque(maxlen=50)  # 用最近50步的耗时估算剩余时间
    
    # 创建本次训练的唯一保存目录
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    save_ckpt_dir = os.path.join(cfg.run_root_dir, f"{date_str}")
    os.makedirs(save_ckpt_dir, exist_ok=True)
    visualize_dir = os.path.join(cfg.visualize_dir, f"{date_str}")
    os.makedirs(visualize_dir, exist_ok=True)

    for epoch in range(cfg.epochs):
        progress = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.epochs}", leave=False)
        for batch_idx, batch in enumerate(progress):
            step_start = time.time()

            gt_action = batch['action_gt'].to(device_id).to(torch.bfloat16)  # [B, 8, 4]
            gt_decision = batch['decision_gt'].to(device_id)
            
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = vla(batch, past_traj_projector=past_traj_projector)
                last_hidden_states = outputs.hidden_states[-1] # [B, Seq_Len, llm_dim]
                
                num_act_tokens = FUTURE_ACTION_WAYPOINTS * ACTION_DIM
                decision_hidden = last_hidden_states[:, -(num_act_tokens + 3), :]   # [B, llm_dim], # 1 dec + 8*4 action tokens + 1 eos + 1 shift token
                actions_hidden = last_hidden_states[:, -(num_act_tokens + 2):-2, :] # [B, num_act_tokens, llm_dim], # 8*4 action tokens + 1 eos + 1 shift token

                pred_decision_logits = decision_head(decision_hidden)
                pred_actions_flat = action_head(actions_hidden).squeeze(-1)
                pred_actions = pred_actions_flat.view(-1, FUTURE_ACTION_WAYPOINTS, ACTION_DIM)
                
                # === Loss 计算 ===
                l_action = mse_loss(pred_actions, gt_action)
                l_obj = mse_loss(pred_actions[:, -1, 0:2], gt_action[:, -1, 0:2])
                
                diff_pred = pred_actions[:, 1:, :] - pred_actions[:, :-1, :]
                diff_gt = gt_action[:, 1:, :] - gt_action[:, :-1, :]
                l_smooth = mse_loss(diff_pred, diff_gt)
                
                # 增加前朝向loss (Forward Loss)
                # 自车方向向量固定为 (1, 0)，与位移向量做内积也就是计算 dx，如果 dx 为负（倒车）即为 loss
                diff_pred_xy = pred_actions[:, 1:, 0:2] - pred_actions[:, :-1, 0:2]
                inner_product = diff_pred_xy[:, :, 0] * 1.0 + diff_pred_xy[:, :, 1] * 0.0  # 实际上就是 dx
                l_forward = torch.relu(-inner_product).mean()
                
                l_decision = ce_loss(pred_decision_logits, gt_decision)
                total_loss = cfg.W_ACT * l_action + cfg.W_OBJ * l_obj + cfg.W_SMOOTH * l_smooth + cfg.W_DEC * l_decision + cfg.W_FORWARD * l_forward
                
                # Normalize loss to account for gradient accumulation
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
                
            global_step += 1

            # 记录本步耗时，用于估算剩余时间
            step_elapsed = time.time() - step_start
            step_times.append(step_elapsed)
            
            if distributed_state.is_main_process:
                # 计算剩余时间更新进度条 ETA
                avg_step_time = sum(step_times) / len(step_times)
                steps_remaining = total_steps - global_step
                eta_seconds = avg_step_time * steps_remaining
                eta_str = str(timedelta(seconds=int(eta_seconds)))
                elapsed_str = str(timedelta(seconds=int(time.time() - train_start_time)))

                progress.set_postfix({
                    "Loss": f"{total_loss.item():.4f}",
                    "Act": f"{l_action.item():.4f}",
                    "Dec": f"{l_decision.item():.4f}",
                    "Forward": f"{l_forward.item():.4f}",
                    "Elapsed": elapsed_str,
                    "ETA": eta_str,
                })
                
                # Logging & visualization
                if global_step % cfg.wandb_log_freq == 0:
                    wandb.log({
                        "Loss/Total": sum(recent_losses)/len(recent_losses),
                        "Loss/Action": l_action.item(),
                        "Loss/Smooth": l_smooth.item(),
                        "Loss/Obj (Endpoint)": l_obj.item(),
                        "Loss/Decision": l_decision.item(),
                        "Loss/Forward": l_forward.item(),
                        "Learning Rate": scheduler.get_last_lr()[0],
                    }, step=global_step)
                    
                    
                
            # save Checkpoint (all ranks wait)
            if global_step > 0 and global_step % cfg.save_freq == 0:
                if distributed_state.is_main_process:
                    save_training_checkpoint(
                        cfg=cfg,
                        run_dir=save_ckpt_dir,
                        log_step=global_step,
                        vla=vla,
                        processor=processor,
                        past_traj_projector=past_traj_projector,
                        action_head=action_head,
                        decision_head=decision_head,
                        loss=total_loss.item(),
                    )
                distributed_state.wait_for_everyone()
                
                if getattr(cfg, "visualize_traj", False):
                    pred_decisions = pred_decision_logits.argmax(dim=-1)
                    pred_decisions_str = [tokenizer.decode(idx.item()) for idx in pred_decisions]
                    gt_decisions_str = [tokenizer.decode(idx.item()) if idx.item() >= 0 else "IGNORE" for idx in gt_decision]
                    visualize_train_expvla(
                        project_folder=visualize_dir,
                        pred_actions=pred_actions,
                        gt_actions=gt_action,
                        pred_decisions=pred_decisions_str,
                        gt_decisions=gt_decisions_str,
                        instructions=batch.get('instruction', ["No instruction"] * pred_actions.shape[0]),
                        images_front=batch['pixel_values_front'],
                        images_rear=batch['pixel_values_rear'],
                        images_left=batch['pixel_values_left'],
                        images_right=batch['pixel_values_right'],
                        epoch=epoch,
                        step=global_step,
                    )
                
    if distributed_state.is_main_process:
        save_training_checkpoint(
                        cfg=cfg,
                        run_dir=save_ckpt_dir,
                        log_step=global_step,
                        vla=vla,
                        processor=processor,
                        past_traj_projector=past_traj_projector,
                        action_head=action_head,
                        decision_head=decision_head,
                        loss=total_loss.item(),
                    )
        distributed_state.wait_for_everyone()
        print("🎉 Training completed!")
        wandb.finish()

if __name__ == "__main__":
    train_expvla()
