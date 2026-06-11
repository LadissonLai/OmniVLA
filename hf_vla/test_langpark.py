import os
import time
import torch
import numpy as np
import torch.nn as nn
import draccus
from collections import defaultdict
from dataclasses import dataclass
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel

from core.langpark_dataset import LangParkDataset, langpark_collate_fn
from core.langpark_modules import MemoryEnhancementModule, InstructionAlignmentHead
from core.modeling_langpark import LangParkVLAForActionPrediction
from core.configuration_prismatic import OpenVLAConfig
from core.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from core.utils import model_is_on_hf_hub, visualize_langpark
from core.constants import ACTION_DIM, FUTURE_ACTION_WAYPOINTS


@dataclass
class TestLangParkConfig:
    # Paths
    vla_path:    str = "/public/home/lqq_202430131053/codes/OmniVLA/openvla-7b"
    resume_dir:  str = "/public/home/lqq_202430131053/codes/OmniVLA/runs_langpark/2026-06-03_15-59/step_33376_loss_0.0498_ckpt"
    data_root:   str = "/public/home/lqq_202430131053/codes/OmniVLA/datasets/ParkingVLA2_val"
    output_file: str = "test_langpark2.txt"

    # DataLoader
    batch_size:  int = 4
    num_workers: int = 4

    # History config (must match training)
    history_mode:        str   = 'smart'
    distance_interval:   float = 0.5
    turn_yaw_thresh:     float = 5.0
    turn_dense_interval: float = 0.1

    # IAM module config (must match training)
    num_mem_tokens:  int = 16
    mem_num_heads:   int = 8
    align_num_heads: int = 8

    # Visualization (mutually exclusive with metric evaluation)
    save_vis: bool = True
    vis_dir:  str  = "vis_langpark_result2"


def calc_l2(pred: torch.Tensor, gt: torch.Tensor) -> float:
    return torch.sqrt((pred[0] - gt[0]) ** 2 + (pred[1] - gt[1]) ** 2).item()


def calc_yaw_diff(pred: torch.Tensor, gt: torch.Tensor) -> float:
    diff = torch.atan2(pred[3], pred[2]) - torch.atan2(gt[3], gt[2])
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    return torch.abs(diff).item()


@draccus.wrap()
def evaluate_langpark(cfg: TestLangParkConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. Register and load base model ───────────────────────────────────────
    if not model_is_on_hf_hub(cfg.vla_path):
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, LangParkVLAForActionPrediction)

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

    print(f"Loading LoRA adapter from {cfg.resume_dir} ...")
    vla = PeftModel.from_pretrained(
        vla, os.path.join(cfg.resume_dir, "lora_adapter"), is_trainable=False
    )

    # ── 2. External modules ───────────────────────────────────────────────────
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
    align_head    = InstructionAlignmentHead(
        llm_dim, cfg.align_num_heads
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
    load_ckpt(align_head,          os.path.join(cfg.resume_dir, "align_head.pt"))

    for m in (vla, past_traj_projector, full_hist_projector,
              action_head, decision_head, mem_module, align_head):
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

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=lambda b: langpark_collate_fn(b, pad_token_id),
    )

    # ── 4. Evaluation loop ────────────────────────────────────────────────────
    NUM_ACT = FUTURE_ACTION_WAYPOINTS * ACTION_DIM  # 32
    NUM_MEM = cfg.num_mem_tokens                     # 16

    # ── Visualization branch (mutually exclusive with metric loop) ────────────
    if cfg.save_vis:
        traj_idx_map = defaultdict(list)
        for idx, s in enumerate(dataset.samples):
            traj_idx_map[s['traj_dir']].append(idx)

        for traj_dir in tqdm(sorted(traj_idx_map.keys()), desc="Visualizing Trajectories"):
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
                    mem_hidden      = last_hidden[:, -(NUM_ACT + 2 + NUM_MEM):-(NUM_ACT + 2), :]

                    pred_decision_logits = decision_head(decision_hidden)
                    pred_actions = action_head(actions_hidden).squeeze(-1).view(
                        -1, FUTURE_ACTION_WAYPOINTS, ACTION_DIM
                    )
                    align_logits = align_head(
                        outputs.instruct_emb, mem_hidden, outputs.instruct_mask
                    )

                pred_dec_str = tokenizer.decode(
                    pred_decision_logits.argmax(dim=-1)[0].item()
                ).replace("<pad>", "").strip()
                gt_dec_str = tokenizer.decode(gt_decision[0].item()).replace("<pad>", "").strip()

                instruct_ids = batch['instruct_ids'][0]        # [L_inst]
                align_label  = batch['align_label'][0]         # [L_inst]
                align_pred   = align_logits.argmax(dim=-1)[0].cpu()  # [L_inst]

                # Align lengths defensively
                min_len     = min(instruct_ids.shape[0], align_pred.shape[0])
                token_texts = [tokenizer.convert_ids_to_tokens(instruct_ids[i].item()) for i in range(min_len)]
                gt_labels   = align_label[:min_len].tolist()
                pred_labels = align_pred[:min_len].tolist()

                past_traj_vis = batch['history_traj'][0] if batch['history_traj'] else None

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
                    token_texts=token_texts,
                    gt_labels=gt_labels,
                    pred_labels=pred_labels,
                )
        return  # skip metric evaluation

    l2_1s, l2_2s, l2_3s       = [], [], []
    yaw_1s, yaw_2s, yaw_3s    = [], [], []
    dec_correct                = []
    align_acc                  = []
    inf_time                   = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        gt_action    = batch['action_gt'].to(device).to(torch.bfloat16)  # [B, 8, 4]
        gt_decision  = batch['decision_gt'].to(device)                    # [B]
        align_labels = batch['align_label'].to(device)                    # [B, L_inst]
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
            mem_hidden      = last_hidden[:, -(NUM_ACT + 2 + NUM_MEM):-(NUM_ACT + 2), :]

            pred_decision_logits = decision_head(decision_hidden)             # [B, vocab]
            pred_actions         = action_head(actions_hidden).squeeze(-1)    # [B, 32]
            pred_actions         = pred_actions.view(-1, FUTURE_ACTION_WAYPOINTS, ACTION_DIM)

            align_logits = align_head(
                outputs.instruct_emb, mem_hidden, outputs.instruct_mask
            )  # [B, L_inst, 3]

        per_sample_time = (time.time() - t0) / B

        pred_decisions = pred_decision_logits.argmax(dim=-1)  # [B]
        align_preds    = align_logits.argmax(dim=-1)          # [B, L_inst]

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

            # Language progress accuracy: ignore -100 positions
            valid = (align_labels[i] != -100)
            if valid.sum() > 0:
                align_acc.append(
                    (align_preds[i][valid] == align_labels[i][valid]).float().mean().item()
                )

            inf_time.append(per_sample_time)

    # ── 5. Summary ────────────────────────────────────────────────────────────
    total = len(l2_1s)
    summary = (
        "========== LangPark VLA Evaluation Results ==========\n"
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
        f"Average Language Progress Accuracy:  {np.mean(align_acc) * 100:.2f}%\n"
        f"Average Inference Time:              {np.mean(inf_time):.4f} s/sample\n"
    )

    print(summary)
    out_dir = os.path.dirname(cfg.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(cfg.output_file, 'w') as f:
        f.write(summary)


if __name__ == "__main__":
    evaluate_langpark()
