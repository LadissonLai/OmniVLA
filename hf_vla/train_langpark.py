import os
import time
import draccus
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from pathlib import Path
from collections import deque
from datetime import datetime, timedelta
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from accelerate import PartialState
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
)
from huggingface_hub import snapshot_download
import tqdm

from core.langpark_dataset import LangParkDataset, langpark_collate_fn
from core.langpark_modules import MemoryEnhancementModule, InstructionAlignmentHead
from core.modeling_langpark import LangParkVLAForActionPrediction
from core.configuration_prismatic import OpenVLAConfig
from core.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from core.utils import model_is_on_hf_hub, visualize_train_expvla
from core.constants import ACTION_DIM, FUTURE_ACTION_WAYPOINTS


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LangParkVLAConfig:
    # Paths
    vla_path:     str  = "/public/home/lqq_202430131053/codes/OmniVLA/openvla-7b"
    data_root:    str  = "/public/home/lqq_202430131053/codes/OmniVLA/datasets/ParkingVLA2"
    run_root_dir: Path = Path("runs_langpark")

    # Training
    batch_size:              int   = 3
    learning_rate:           float = 1e-4
    grad_accumulation_steps: int   = 4
    epochs:                  int   = 16
    save_freq:               int   = 512
    resume:                  bool  = False
    resume_dir:              str   = ""
    num_workers:             int   = 2
    lr_warmup_steps:         int   = 500
    num_steps_before_decay:  int   = 2000

    # Visualisation
    visualize_traj: bool = True
    visualize_dir:  str  = "vis_langpark_train"

    # History config (must match dataset)
    history_mode:        str   = 'smart'
    distance_interval:   float = 0.5
    turn_yaw_thresh:     float = 5.0
    turn_dense_interval: float = 0.1

    # LoRA
    use_lora:     bool  = True
    lora_rank:    int   = 32
    lora_dropout: float = 0.05

    # IAM module
    num_mem_tokens: int = 16
    mem_num_heads:  int = 8
    align_num_heads: int = 8

    # Logging
    wandb_dir:      str = "wandb_langpark"
    wandb_entity:   str = "your-wandb-entity"
    wandb_project:  str = "LangPark-VLA"
    wandb_log_freq: int = 64

    # Loss weights
    W_ACT:   float = 1.0
    W_OBJ:   float = 0.5
    W_YAW:   float = 0.5
    W_XY1:   float = 0.5
    W_XY2:   float = 0.3
    W_YAW1:  float = 0.3
    W_DEC:   float = 1.0
    W_ALIGN: float = 0.1

    # Smooth L1 (Huber) transition point for position regression, in metres.
    # |error| < beta behaves like L2 (precise), |error| > beta like L1 (robust).
    huber_beta: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def wrap_ddp(module: nn.Module, device_id: int) -> nn.Module:
    from torch.nn.parallel import DistributedDataParallel as DDP
    return DDP(module, device_ids=[device_id], find_unused_parameters=True)


def save_training_checkpoint(
    cfg,
    run_dir,
    log_step: int,
    vla,
    processor,
    past_traj_projector,
    full_hist_projector,
    action_head,
    decision_head,
    mem_module,
    align_head,
    loss: float = None,
):
    run_dir = Path(run_dir)
    suffix  = f"step_{log_step}_loss_{loss:.4f}_ckpt" if loss is not None else f"step_{log_step}_ckpt"
    chkpt_dir = run_dir / suffix
    os.makedirs(chkpt_dir, exist_ok=True)

    processor.save_pretrained(chkpt_dir)
    vla.module.save_pretrained(chkpt_dir / "lora_adapter")

    torch.save(past_traj_projector.module.state_dict(), chkpt_dir / "past_traj_projector.pt")
    torch.save(full_hist_projector.module.state_dict(), chkpt_dir / "full_hist_projector.pt")
    torch.save(action_head.module.state_dict(),         chkpt_dir / "action_head.pt")
    torch.save(decision_head.module.state_dict(),       chkpt_dir / "decision_head.pt")
    torch.save(mem_module.module.state_dict(),          chkpt_dir / "mem_module.pt")
    torch.save(align_head.module.state_dict(),          chkpt_dir / "align_head.pt")

    print(f"Checkpoint saved at {chkpt_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

@draccus.wrap()
def train_langpark(cfg: LangParkVLAConfig):
    distributed_state = PartialState()
    device_id         = distributed_state.local_process_index
    torch.cuda.set_device(device_id)

    if distributed_state.is_main_process:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name="langpark_training",
            dir=cfg.wandb_dir,
        )
        os.makedirs(cfg.run_root_dir, exist_ok=True)

    # ── 1. Load processor and base VLA model ─────────────────────────────────

    if model_is_on_hf_hub(cfg.vla_path):
        cfg.vla_path = snapshot_download(repo_id=cfg.vla_path)
    else:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, LangParkVLAForActionPrediction)

    processor     = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    tokenizer     = processor.tokenizer
    pad_token_id  = tokenizer.pad_token_id if tokenizer.pad_token_id else 0

    vla = LangParkVLAForActionPrediction.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device_id)

    # Apply LoRA
    if cfg.use_lora:
        if cfg.resume and cfg.resume_dir:
            from peft import PeftModel
            vla = PeftModel.from_pretrained(
                vla, os.path.join(cfg.resume_dir, "lora_adapter"), is_trainable=True
            )
            if distributed_state.is_main_process:
                print(f"Resumed LoRA adapter from {cfg.resume_dir}")
        else:
            target_modules = [
                name for name, module in vla.named_modules()
                if isinstance(module, nn.Linear) and 'lm_head' not in name
            ]
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=16,
                lora_dropout=cfg.lora_dropout,
                target_modules=target_modules,
                init_lora_weights="gaussian",
            )
            vla = get_peft_model(vla, lora_config)

    vla = wrap_ddp(vla, device_id)

    # ── 2. External modules ───────────────────────────────────────────────────

    llm_dim    = vla.module.config.text_config.hidden_size
    vocab_size = vla.module.config.text_config.vocab_size

    def make_projector():
        return nn.Sequential(
            nn.Linear(4, llm_dim // 2),
            nn.GELU(),
            nn.Linear(llm_dim // 2, llm_dim),
        ).to(device_id).to(torch.bfloat16)

    past_traj_projector = make_projector()
    full_hist_projector = make_projector()

    action_head = nn.Sequential(
        nn.Linear(llm_dim, llm_dim),
        nn.GELU(),
        nn.Linear(llm_dim, 1),
    ).to(device_id).to(torch.bfloat16)

    decision_head = nn.Linear(llm_dim, vocab_size).to(device_id).to(torch.bfloat16)

    mem_module = MemoryEnhancementModule(
        llm_dim, cfg.num_mem_tokens, cfg.mem_num_heads
    ).to(device_id).to(torch.bfloat16)

    align_head = InstructionAlignmentHead(
        llm_dim, cfg.align_num_heads
    ).to(device_id).to(torch.bfloat16)

    # Resume external modules
    if cfg.resume and cfg.resume_dir:
        def load_ckpt(model, path):
            state_dict = torch.load(path, map_location=f"cuda:{device_id}")
            cleaned    = {
                (k[len("module."):] if k.startswith("module.") else k): v
                for k, v in state_dict.items()
            }
            model.load_state_dict(cleaned)

        load_ckpt(past_traj_projector, os.path.join(cfg.resume_dir, "past_traj_projector.pt"))
        load_ckpt(full_hist_projector, os.path.join(cfg.resume_dir, "full_hist_projector.pt"))
        load_ckpt(action_head,         os.path.join(cfg.resume_dir, "action_head.pt"))
        load_ckpt(decision_head,       os.path.join(cfg.resume_dir, "decision_head.pt"))
        load_ckpt(mem_module,          os.path.join(cfg.resume_dir, "mem_module.pt"))
        load_ckpt(align_head,          os.path.join(cfg.resume_dir, "align_head.pt"))
        if distributed_state.is_main_process:
            print(f"Resumed all external modules from {cfg.resume_dir}")

    # Wrap all external modules with DDP
    past_traj_projector = wrap_ddp(past_traj_projector, device_id)
    full_hist_projector = wrap_ddp(full_hist_projector, device_id)
    action_head         = wrap_ddp(action_head,         device_id)
    decision_head       = wrap_ddp(decision_head,       device_id)
    mem_module          = wrap_ddp(mem_module,          device_id)
    align_head          = wrap_ddp(align_head,          device_id)

    # ── 3. Dataset and DataLoader ─────────────────────────────────────────────

    dataset = LangParkDataset(
        data_root=cfg.data_root,
        tokenizer=tokenizer,
        image_transform=processor.image_processor.apply_transform,
        history_mode=cfg.history_mode,
        distance_interval=cfg.distance_interval,
        turn_yaw_thresh=cfg.turn_yaw_thresh,
        turn_dense_interval=cfg.turn_dense_interval,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=lambda b: langpark_collate_fn(b, pad_token_id),
    )

    # ── 4. Optimizer and scheduler ────────────────────────────────────────────

    trainable_params = (
        [p for p in vla.parameters() if p.requires_grad]
        + list(past_traj_projector.parameters())
        + list(full_hist_projector.parameters())
        + list(action_head.parameters())
        + list(decision_head.parameters())
        + list(mem_module.parameters())
        + list(align_head.parameters())
    )
    optimizer   = AdamW(trainable_params, lr=cfg.learning_rate)
    original_lr = optimizer.param_groups[0]["lr"]
    scheduler   = MultiStepLR(optimizer, milestones=[cfg.num_steps_before_decay], gamma=0.1)

    huber_loss = nn.SmoothL1Loss(beta=cfg.huber_beta)
    ce_loss    = nn.CrossEntropyLoss()

    recent_losses = deque(maxlen=50)

    # ── 5. Training setup ─────────────────────────────────────────────────────

    for module in (vla, past_traj_projector, full_hist_projector,
                   action_head, decision_head, mem_module, align_head):
        module.train()
    optimizer.zero_grad()

    global_step      = 0
    total_steps      = cfg.epochs * len(dataloader)
    train_start_time = time.time()
    step_times       = deque(maxlen=50)

    date_str      = datetime.now().strftime("%Y-%m-%d_%H-%M")
    save_ckpt_dir = os.path.join(cfg.run_root_dir, date_str)
    visualize_dir = os.path.join(cfg.visualize_dir, date_str)
    os.makedirs(save_ckpt_dir, exist_ok=True)
    os.makedirs(visualize_dir, exist_ok=True)

    NUM_ACT = FUTURE_ACTION_WAYPOINTS * ACTION_DIM   # 32
    NUM_MEM = cfg.num_mem_tokens                      # 16

    # ── 6. Training loop ──────────────────────────────────────────────────────

    for epoch in range(cfg.epochs):
        progress = tqdm.tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.epochs}", leave=False)

        for batch_idx, batch in enumerate(progress):
            step_start = time.time()

            gt_action   = batch['action_gt'].to(device_id).to(torch.bfloat16)   # [B, 8, 4]
            gt_decision = batch['decision_gt'].to(device_id)                     # [B]

            with torch.autocast("cuda", dtype=torch.bfloat16):
                # Forward
                outputs = vla(
                    batch,
                    past_traj_projector=past_traj_projector,
                    full_hist_projector=full_hist_projector,
                    mem_module=mem_module,
                )
                last_hidden = outputs.hidden_states[-1]   # [B, Seq, D]

                # ── dec / act hidden states (same indexing as ExpVLA) ──────────
                # Tail layout: ... MEM(16) | dec(1) | act(32) | EOS(1)
                # Hidden at pos -(NUM_ACT+3) = MEM[15]: predicts dec (autoregressive shift)
                # Hidden at pos -(NUM_ACT+2):-2 = dec..act[30]: predicts act[0]..act[31]
                decision_hidden = last_hidden[:, -(NUM_ACT + 3), :]
                actions_hidden  = last_hidden[:, -(NUM_ACT + 2):-2, :]

                # ── MEM hidden states ──────────────────────────────────────────
                # MEM[0..15] at positions -(NUM_ACT+2+16) to -(NUM_ACT+2) exclusive
                mem_hidden = last_hidden[:, -(NUM_ACT + 2 + NUM_MEM):-(NUM_ACT + 2), :]  # [B, 16, D]

                # ── Predictions ────────────────────────────────────────────────
                pred_decision_logits = decision_head(decision_hidden)             # [B, vocab]
                pred_actions_flat    = action_head(actions_hidden).squeeze(-1)    # [B, 32]
                pred_actions         = pred_actions_flat.view(-1, FUTURE_ACTION_WAYPOINTS, ACTION_DIM)  # [B,8,4]

                # ── Loss computation ───────────────────────────────────────────
                pred_xy = pred_actions[:, :, 0:2]              # [B,8,2]
                gt_xy   = gt_action[:, :, 0:2]

                # Position regression (xy ONLY)
                l_action = huber_loss(pred_xy, gt_xy)

                # Endpoint emphasis (xy only)
                l_obj = huber_loss(pred_xy[:, -1, :], gt_xy[:, -1, :])

                # Yaw absolute error: geodesic loss on SO(2), no atan2
                cos_pred  = pred_actions[:, :, 2]
                sin_pred  = pred_actions[:, :, 3]
                cos_gt    = gt_action[:, :, 2]
                sin_gt    = gt_action[:, :, 3]
                # Normalize predictions onto unit circle to avoid atan2 gradient explosion
                norm_pred  = (cos_pred**2 + sin_pred**2 + 1e-8).sqrt()
                cos_pred_n = cos_pred / norm_pred
                sin_pred_n = sin_pred / norm_pred
                cos_delta  = cos_pred_n * cos_gt + sin_pred_n * sin_gt   # cos(θ_pred - θ_gt)
                l_yaw      = (1.0 - cos_delta).mean()

                # XY 1st-order smoothness: consecutive displacement matches GT
                diff_pred_xy = pred_xy[:, 1:, :] - pred_xy[:, :-1, :]  # [B,7,2]
                diff_gt_xy   = gt_xy[:, 1:, :]   - gt_xy[:, :-1, :]
                l_xy_1st     = huber_loss(diff_pred_xy, diff_gt_xy)

                # XY 2nd-order smoothness: consecutive acceleration matches GT
                diff2_pred_xy = diff_pred_xy[:, 1:, :] - diff_pred_xy[:, :-1, :]  # [B,6,2]
                diff2_gt_xy   = diff_gt_xy[:, 1:, :]   - diff_gt_xy[:, :-1, :]
                l_xy_2nd      = huber_loss(diff2_pred_xy, diff2_gt_xy)

                # Yaw 1st-order smoothness: turning-rate geodesic loss, no atan2
                cos_turn_pred = cos_pred_n[:, 1:] * cos_pred_n[:, :-1] + sin_pred_n[:, 1:] * sin_pred_n[:, :-1]  # [B,7] θ_pre cos(θ_t+1 - θ_t)
                sin_turn_pred = sin_pred_n[:, 1:] * cos_pred_n[:, :-1] - cos_pred_n[:, 1:] * sin_pred_n[:, :-1]  # sin(θ_t+1 - θ_t)
                cos_turn_gt   = cos_gt[:, 1:] * cos_gt[:, :-1] + sin_gt[:, 1:] * sin_gt[:, :-1]
                sin_turn_gt   = sin_gt[:, 1:] * cos_gt[:, :-1] - cos_gt[:, 1:] * sin_gt[:, :-1]
                l_yaw_1st     = (1.0 - (cos_turn_pred * cos_turn_gt + sin_turn_pred * sin_turn_gt)).mean() # cos(delta_theta_{pre} - delta_theta_{gt})

                # Decision (slot id) classification
                l_decision = ce_loss(pred_decision_logits, gt_decision)

                # Instruction alignment
                align_logits = align_head(
                    outputs.instruct_emb, mem_hidden, outputs.instruct_mask
                )   # [B, L_inst, 3]
                align_labels = batch['align_label'].to(device_id)   # [B, L_inst], -100=ignore
                l_align = F.cross_entropy(
                    align_logits.reshape(-1, 3),
                    align_labels.reshape(-1),
                    ignore_index=-100,
                )

                # Weighted total
                total_loss = (
                    cfg.W_ACT   * l_action
                    + cfg.W_OBJ   * l_obj
                    + cfg.W_YAW   * l_yaw
                    + cfg.W_XY1   * l_xy_1st
                    + cfg.W_XY2   * l_xy_2nd
                    + cfg.W_YAW1  * l_yaw_1st
                    + cfg.W_DEC   * l_decision
                    + cfg.W_ALIGN * l_align
                )

                normalized_loss = total_loss / cfg.grad_accumulation_steps

            normalized_loss.backward()
            recent_losses.append(total_loss.item())

            # Gradient accumulation step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                gradient_step_idx = global_step // cfg.grad_accumulation_steps

                # Linear LR warm-up from 10 % to 100 %
                if cfg.lr_warmup_steps > 0:
                    lr_progress = min((gradient_step_idx + 1) / cfg.lr_warmup_steps, 1.0)
                    current_lr  = original_lr * (0.1 + 0.9 * lr_progress)
                    for pg in optimizer.param_groups:
                        pg["lr"] = current_lr

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            step_elapsed = time.time() - step_start
            step_times.append(step_elapsed)

            # ── Logging (main process only) ────────────────────────────────
            if distributed_state.is_main_process:
                avg_t       = sum(step_times) / len(step_times)
                eta_str     = str(timedelta(seconds=int(avg_t * (total_steps - global_step))))
                elapsed_str = str(timedelta(seconds=int(time.time() - train_start_time)))

                progress.set_postfix({
                    "Loss":    f"{total_loss.item():.4f}",
                    "Act":     f"{l_action.item():.4f}",
                    "Yaw":     f"{l_yaw.item():.4f}",
                    "XY1":     f"{l_xy_1st.item():.4f}",
                    "XY2":     f"{l_xy_2nd.item():.4f}",
                    "Dec":     f"{l_decision.item():.4f}",
                    "Elapsed": elapsed_str,
                    "ETA":     eta_str,
                })

                if global_step % cfg.wandb_log_freq == 0:
                    wandb.log(
                        {
                            "Loss/Total":    sum(recent_losses) / len(recent_losses),
                            "Loss/Action":   l_action.item(),
                            "Loss/Obj":      l_obj.item(),
                            "Loss/Yaw":      l_yaw.item(),
                            "Loss/XY_1st":   l_xy_1st.item(),
                            "Loss/XY_2nd":   l_xy_2nd.item(),
                            "Loss/Yaw_1st":  l_yaw_1st.item(),
                            "Loss/Decision": l_decision.item(),
                            "Loss/Align":    l_align.item(),
                            "LR":            scheduler.get_last_lr()[0],
                        },
                        step=global_step,
                    )

            # ── Checkpoint ────────────────────────────────────────────────
            if global_step > 0 and global_step % cfg.save_freq == 0:
                if distributed_state.is_main_process:
                    save_training_checkpoint(
                        cfg=cfg,
                        run_dir=save_ckpt_dir,
                        log_step=global_step,
                        vla=vla,
                        processor=processor,
                        past_traj_projector=past_traj_projector,
                        full_hist_projector=full_hist_projector,
                        action_head=action_head,
                        decision_head=decision_head,
                        mem_module=mem_module,
                        align_head=align_head,
                        loss=total_loss.item(),
                    )
                distributed_state.wait_for_everyone()

                # Trajectory visualisation
                if cfg.visualize_traj and distributed_state.is_main_process:
                    pred_decisions     = pred_decision_logits.argmax(dim=-1)
                    pred_decisions_str = [tokenizer.decode(idx.item()) for idx in pred_decisions]
                    gt_decisions_str   = [
                        tokenizer.decode(idx.item()) if idx.item() >= 0 else "IGNORE"
                        for idx in gt_decision
                    ]
                    visualize_train_expvla(
                        project_folder=visualize_dir,
                        pred_actions=pred_actions,
                        gt_actions=gt_action,
                        pred_decisions=pred_decisions_str,
                        gt_decisions=gt_decisions_str,
                        instructions=batch.get('instruction', [""] * pred_actions.shape[0]),
                        images_front=batch['pixel_values_front'],
                        images_rear=batch['pixel_values_rear'],
                        images_left=batch['pixel_values_left'],
                        images_right=batch['pixel_values_right'],
                        epoch=epoch,
                        step=global_step,
                    )

    # ── Final checkpoint ──────────────────────────────────────────────────────
    if distributed_state.is_main_process:
        save_training_checkpoint(
            cfg=cfg,
            run_dir=save_ckpt_dir,
            log_step=global_step,
            vla=vla,
            processor=processor,
            past_traj_projector=past_traj_projector,
            full_hist_projector=full_hist_projector,
            action_head=action_head,
            decision_head=decision_head,
            mem_module=mem_module,
            align_head=align_head,
            loss=total_loss.item(),
        )
        distributed_state.wait_for_everyone()
        print("Training completed!")
        wandb.finish()


if __name__ == "__main__":
    train_langpark()
