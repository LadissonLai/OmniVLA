"""
test_langpark_ablation_mem_only.py

Evaluation for Ablation #2: KEEP MemoryEnhancementModule, REMOVE
InstructionAlignmentHead.

Mirrors test_langpark.py but:
  - Keeps the MEM module / full_hist_projector (model class unchanged).
  - Drops the alignment head and the Language Progress Accuracy metric.
  - Visualization drops the instruction-alignment panel.
"""

import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.distributed as dist
import draccus
from collections import defaultdict
from dataclasses import dataclass
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from accelerate import PartialState
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel

from core.langpark_dataset import LangParkDataset, langpark_collate_fn
from core.langpark_modules import MemoryEnhancementModule
from core.modeling_langpark import LangParkVLAForActionPrediction
from core.configuration_prismatic import OpenVLAConfig
from core.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from core.utils import model_is_on_hf_hub, visualize_langpark
from core.constants import ACTION_DIM, FUTURE_ACTION_WAYPOINTS


@dataclass
class TestLangParkMemOnlyConfig:
    # Paths
    vla_path:    str = "/root/autodl-tmp/codes/OmniVLA/openvla-7b"
    resume_dir:  str = ""   # set to the ablation_mem_only checkpoint dir
    data_root:   str = "/root/autodl-tmp/codes/OmniVLA/datasets/ParkingVLA_testsets/"
    output_file: str = "metrics/test_langpark_ablation_mem_only.txt"

    # DataLoader
    batch_size:  int = 24
    num_workers: int = 4

    # History config (must match training)
    history_mode:        str   = 'smart'
    distance_interval:   float = 0.5
    turn_yaw_thresh:     float = 5.0
    turn_dense_interval: float = 0.1

    # MEM module config (must match training). Alignment head removed.
    num_mem_tokens: int = 16
    mem_num_heads:  int = 8

    # Visualization (mutually exclusive with metric evaluation)
    save_vis: bool = False
    vis_dir:  str  = "vis_langpark_ablation_mem_only_result"


def calc_l2(pred: torch.Tensor, gt: torch.Tensor) -> float:
    return torch.sqrt((pred[0] - gt[0]) ** 2 + (pred[1] - gt[1]) ** 2).item()


def calc_yaw_diff(pred: torch.Tensor, gt: torch.Tensor) -> float:
    diff = torch.atan2(pred[3], pred[2]) - torch.atan2(gt[3], gt[2])
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    return torch.abs(diff).item()


def gather_lists(local_list: list, use_distributed: bool, world_size: int) -> list:
    """Gather per-sample metric lists from all ranks into a single flat list.

    On single GPU (use_distributed=False) this is a no-op returning local_list.
    """
    if not use_distributed:
        return local_list
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_list)
    merged = []
    for part in gathered:
        merged.extend(part)
    return merged


@draccus.wrap()
def evaluate_langpark_ablation_mem_only(cfg: TestLangParkMemOnlyConfig):
    # ── 0. Distributed state (single-GPU: use_distributed=False, degrades to original) ──
    # Launch single GPU : python hf_vla/test_langpark_ablation_mem_only.py
    # Launch multi  GPU : torchrun --nproc_per_node=4 hf_vla/test_langpark_ablation_mem_only.py
    distributed_state = PartialState()
    use_distributed   = distributed_state.use_distributed
    device_id         = distributed_state.local_process_index
    world_size        = distributed_state.num_processes
    rank              = distributed_state.process_index
    is_main           = distributed_state.is_main_process
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    # ── 1. Register and load base model ───────────────────────────────────────
    if not model_is_on_hf_hub(cfg.vla_path):
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, LangParkVLAForActionPrediction)

    if is_main:
        print("Loading processor and base model...")
    processor    = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    tokenizer    = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id else 0

    vla = LangParkVLAForActionPrediction.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)

    if is_main:
        print(f"Loading LoRA adapter from {cfg.resume_dir} ...")
    vla = PeftModel.from_pretrained(
        vla, os.path.join(cfg.resume_dir, "lora_adapter"), is_trainable=False
    )

    # ── 2. External modules (MEM kept, alignment head removed) ────────────────
    llm_dim    = vla.config.text_config.hidden_size
    vocab_size = vla.config.text_config.vocab_size

    def make_projector() -> nn.Module:
        return nn.Sequential(
            nn.Linear(4, llm_dim // 2),
            nn.GELU(),
            nn.Linear(llm_dim // 2, llm_dim),
        ).to(device).to(torch.bfloat16)

    past_traj_projector = make_projector()
    full_hist_projector = make_projector()
    action_head   = nn.Sequential(
        nn.Linear(llm_dim, llm_dim), nn.GELU(), nn.Linear(llm_dim, 1)
    ).to(device).to(torch.bfloat16)
    decision_head = nn.Linear(llm_dim, vocab_size).to(device).to(torch.bfloat16)
    mem_module    = MemoryEnhancementModule(
        llm_dim, cfg.num_mem_tokens, cfg.mem_num_heads
    ).to(device).to(torch.bfloat16)

    def load_ckpt(model: nn.Module, path: str) -> None:
        state_dict = torch.load(path, map_location=device)
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

    for m in (vla, past_traj_projector, full_hist_projector,
              action_head, decision_head, mem_module):
        m.eval()

    # ── 3. Dataset & DataLoader ───────────────────────────────────────────────
    dataset = LangParkDataset(
        data_root=cfg.data_root,
        tokenizer=tokenizer,
        image_transform=processor.image_processor.apply_transform,
        history_mode=cfg.history_mode,
        distance_interval=cfg.distance_interval,
        turn_yaw_thresh=cfg.turn_yaw_thresh,
        turn_dense_interval=cfg.turn_dense_interval,
    )

    # Multi-GPU: shard by stride (no padding ⇒ no duplicate/missing samples,
    # so gathered metrics are identical to single-GPU). Single-GPU: full set.
    if use_distributed:
        eval_dataset = Subset(dataset, list(range(rank, len(dataset), world_size)))
    else:
        eval_dataset = dataset

    dataloader = DataLoader(
        eval_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=lambda b: langpark_collate_fn(b, pad_token_id),
    )

    # ── 4. Evaluation loop ────────────────────────────────────────────────────
    NUM_ACT = FUTURE_ACTION_WAYPOINTS * ACTION_DIM  # 32

    # ── Visualization branch (mutually exclusive with metric loop) ────────────
    if cfg.save_vis:
        traj_idx_map = defaultdict(list)
        for idx, s in enumerate(dataset.samples):
            traj_idx_map[s['traj_dir']].append(idx)

        # Multi-GPU: shard whole trajectories across ranks; each rank writes its
        # own PNGs independently (no gather needed). Single-GPU: all trajectories.
        traj_keys = sorted(traj_idx_map.keys())
        if use_distributed:
            traj_keys = traj_keys[rank::world_size]

        for traj_dir in tqdm(traj_keys, desc="Visualizing Trajectories", disable=not is_main):
            traj_name = os.path.basename(traj_dir)
            indices   = sorted(traj_idx_map[traj_dir], key=lambda i: dataset.samples[i]['t'])

            for idx in indices:
                sample  = dataset[idx]
                batch   = langpark_collate_fn([sample], pad_token_id)
                frame_t = dataset.samples[idx]['t']

                gt_action   = batch['action_gt'].to(device).to(torch.bfloat16)
                gt_decision = batch['decision_gt'].to(device)

                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = vla(
                        batch,
                        past_traj_projector=past_traj_projector,
                        full_hist_projector=full_hist_projector,
                        mem_module=mem_module,
                    )
                    last_hidden = outputs.hidden_states[-1]

                    decision_hidden = last_hidden[:, -(NUM_ACT + 3), :]
                    actions_hidden  = last_hidden[:, -(NUM_ACT + 2):-2, :]

                    pred_decision_logits = decision_head(decision_hidden)
                    pred_actions = action_head(actions_hidden).squeeze(-1).view(
                        -1, FUTURE_ACTION_WAYPOINTS, ACTION_DIM
                    )

                pred_dec_str = tokenizer.decode(
                    pred_decision_logits.argmax(dim=-1)[0].item()
                ).replace("<pad>", "").strip()
                gt_dec_str = tokenizer.decode(gt_decision[0].item()).replace("<pad>", "").strip()

                past_traj_vis = batch['history_traj'][0] if batch['history_traj'] else None

                # No instruction-alignment head in this ablation: pass empty
                # token/label lists so visualize_langpark renders no align panel.
                vis_path = os.path.join(cfg.vis_dir, traj_name, f"step_{frame_t:06d}.png")
                visualize_langpark(
                    save_path=vis_path,
                    pred_actions=pred_actions[0],
                    gt_actions=gt_action[0],
                    past_traj=past_traj_vis,
                    pred_decision=pred_dec_str,
                    gt_decision=gt_dec_str,
                    instruction=batch['instruction'][0],
                    image_front=batch['pixel_values_front'][0],
                    image_rear=batch['pixel_values_rear'][0],
                    image_left=batch['pixel_values_left'][0],
                    image_right=batch['pixel_values_right'][0],
                    token_texts=[],
                    gt_labels=[],
                    pred_labels=[],
                )
        distributed_state.wait_for_everyone()
        return  # skip metric evaluation

    l2_1s, l2_2s, l2_3s       = [], [], []
    yaw_1s, yaw_2s, yaw_3s    = [], [], []
    dec_correct                = []
    inf_time                   = []

    for batch in tqdm(dataloader, desc="Evaluating", disable=not is_main):
        gt_action    = batch['action_gt'].to(device).to(torch.bfloat16)  # [B, 8, 4]
        gt_decision  = batch['decision_gt'].to(device)                    # [B]
        B = gt_action.shape[0]

        t0 = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = vla(
                batch,
                past_traj_projector=past_traj_projector,
                full_hist_projector=full_hist_projector,
                mem_module=mem_module,
            )
            last_hidden = outputs.hidden_states[-1]

            # Tail layout: ... MEM(16) | dec(1) | act(32) | EOS(1)
            decision_hidden = last_hidden[:, -(NUM_ACT + 3), :]
            actions_hidden  = last_hidden[:, -(NUM_ACT + 2):-2, :]

            pred_decision_logits = decision_head(decision_hidden)             # [B, vocab]
            pred_actions         = action_head(actions_hidden).squeeze(-1)    # [B, 32]
            pred_actions         = pred_actions.view(-1, FUTURE_ACTION_WAYPOINTS, ACTION_DIM)

        per_sample_time = (time.time() - t0) / B

        pred_decisions = pred_decision_logits.argmax(dim=-1)  # [B]

        for i in range(B):
            p = pred_actions[i].cpu().float()
            g = gt_action[i].cpu().float()

            l2_1s.append(calc_l2(p[1], g[1]))
            l2_2s.append(calc_l2(p[3], g[3]))
            l2_3s.append(calc_l2(p[5], g[5]))

            yaw_1s.append(calc_yaw_diff(p[1], g[1]))
            yaw_2s.append(calc_yaw_diff(p[3], g[3]))
            yaw_3s.append(calc_yaw_diff(p[5], g[5]))

            dec_correct.append(int(pred_decisions[i].item() == gt_decision[i].item()))

            inf_time.append(per_sample_time)

    # ── 5. Gather metrics across ranks (no-op on single GPU) ──────────────────
    l2_1s       = gather_lists(l2_1s,       use_distributed, world_size)
    l2_2s       = gather_lists(l2_2s,       use_distributed, world_size)
    l2_3s       = gather_lists(l2_3s,       use_distributed, world_size)
    yaw_1s      = gather_lists(yaw_1s,      use_distributed, world_size)
    yaw_2s      = gather_lists(yaw_2s,      use_distributed, world_size)
    yaw_3s      = gather_lists(yaw_3s,      use_distributed, world_size)
    dec_correct = gather_lists(dec_correct, use_distributed, world_size)
    inf_time    = gather_lists(inf_time,    use_distributed, world_size)

    # Only the main process computes / prints / writes the summary.
    if not is_main:
        distributed_state.wait_for_everyone()
        return

    # ── 6. Summary ────────────────────────────────────────────────────────────
    total = len(l2_1s)
    summary = (
        "===== LangPark VLA Evaluation (Ablation #2: MEM only, no IAM head) =====\n"
        f"Checkpoint:                          {cfg.resume_dir}\n"
        f"Data root:                           {cfg.data_root}\n"
        f"Total Samples Evaluated:             {total}\n\n"
        f"Average L2 Dist 1s (2nd pt):         {np.mean(l2_1s):.4f} m\n"
        f"Average L2 Dist 2s (4th pt):         {np.mean(l2_2s):.4f} m\n"
        f"Average L2 Dist 3s (6th pt):         {np.mean(l2_3s):.4f} m\n"
        f"Average Yaw Err 1s (2nd pt):         {np.mean(yaw_1s):.4f} rad\n"
        f"Average Yaw Err 2s (4th pt):         {np.mean(yaw_2s):.4f} rad\n"
        f"Average Yaw Err 3s (6th pt):         {np.mean(yaw_3s):.4f} rad\n"
        f"Average Decision Accuracy:           {np.mean(dec_correct) * 100:.2f}%\n"
        f"Average Inference Time:              {np.mean(inf_time):.4f} s/sample\n"
    )

    print(summary)
    out_dir = os.path.dirname(cfg.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(cfg.output_file, 'w') as f:
        f.write(summary)

    distributed_state.wait_for_everyone()


if __name__ == "__main__":
    evaluate_langpark_ablation_mem_only()
