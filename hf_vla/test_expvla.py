import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from tqdm import tqdm
from transformers import AutoProcessor, AutoConfig, AutoImageProcessor, AutoModelForVision2Seq
from peft import PeftModel
import torch.nn as nn
import draccus

# 从原有模块导入
from core.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from core.configuration_prismatic import OpenVLAConfig
from core.exp_parking_dataset import to_local_coords, custom_collate_fn, SingleTrajTestDataset
from core.modeling_prismatic import ExpVLAForActionPrediction
from core.utils import model_is_on_hf_hub
from core.constants import FUTURE_ACTION_WAYPOINTS, ACTION_DIM

@dataclass
class TestConfig:
    vla_path: str = "/public/home/lqq_202430131053/codes/OmniVLA/omnivla-original"
    resume_dir: str = "/public/home/lqq_202430131053/codes/OmniVLA/runs_expvla-good/2026-04-01_20-08/step_78592_checkpoint"
    data_root: str = "/public/home/lqq_202430131053/codes/datasets/ParkingVLA_Val" # 测试集路径
    output_file: str = "test_metrics_result.txt"
    
    # History Trajectory 配置，与训练一致
    history_mode: str = 'fixed_count'
    max_history: int = 8
    future_steps: int = 8
    distance_interval: float = 0.5
    turn_yaw_thresh: float = 5.0
    turn_dense_interval: float = 0.1

def calc_l2_distance(pred, gt):
    """计算单个时间点的 (x,y) L2 欧氏距离, 单位 m"""
    return torch.sqrt((pred[0] - gt[0])**2 + (pred[1] - gt[1])**2).item()

def calc_yaw_diff(pred, gt):
    """计算单个时间点的朝向角度差异（弧度）"""
    # 根据 dataset 的 to_local_coords，输出为 [x, y, cos(yaw), sin(yaw)]
    # 索引 2 为 cos(yaw), 索引 3 为 sin(yaw)
    pred_yaw = torch.atan2(pred[3], pred[2])
    gt_yaw = torch.atan2(gt[3], gt[2])
    
    diff_rad = pred_yaw - gt_yaw
    # 保证差值在 [-pi, pi] 之间
    diff_rad = torch.atan2(torch.sin(diff_rad), torch.cos(diff_rad))
    return torch.abs(diff_rad).item()

@draccus.wrap()
def evaluate_model(cfg: TestConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 注册组件
    if not model_is_on_hf_hub(cfg.vla_path):
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, ExpVLAForActionPrediction)

    print("Loading processor and base model...")
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id else 0
    
    vla = ExpVLAForActionPrediction.from_pretrained(
        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device)

    # 载入 LoRA
    print(f"Loading LoRA weights from {cfg.resume_dir} ...")
    vla = PeftModel.from_pretrained(vla, os.path.join(cfg.resume_dir, "lora_adapter"), is_trainable=False)
    
    llm_dim = vla.config.text_config.hidden_size
    vocab_size = vla.config.text_config.vocab_size
    
    # 初始化外挂网络
    past_traj_projector = nn.Sequential(nn.Linear(4, llm_dim // 2), nn.GELU(), nn.Linear(llm_dim // 2, llm_dim)).to(device).to(torch.bfloat16)
    action_head = nn.Sequential(nn.Linear(llm_dim, llm_dim), nn.GELU(), nn.Linear(llm_dim, 1)).to(device).to(torch.bfloat16)
    decision_head = nn.Linear(llm_dim, vocab_size).to(device).to(torch.bfloat16)

    # 加载外挂网络权重
    def load_ckpt(model, _path):
        state_dict = torch.load(_path, map_location=device)
        new_state_dict = {k.replace("module.", "") if k.startswith("module.") else k: v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)

    load_ckpt(past_traj_projector, os.path.join(cfg.resume_dir, "past_traj_projector.pt"))
    load_ckpt(action_head, os.path.join(cfg.resume_dir, "action_head.pt"))
    load_ckpt(decision_head, os.path.join(cfg.resume_dir, "decision_head.pt"))
    
    vla.eval()
    past_traj_projector.eval()
    action_head.eval()
    decision_head.eval()
    
    traj_dirs = [os.path.join(cfg.data_root, d) for d in os.listdir(cfg.data_root) if os.path.isdir(os.path.join(cfg.data_root, d))]
    
    global_results = {
        "l2_1s": [],
        "l2_2s": [],
        "l2_3s": [],
        "yaw_1s": [],
        "yaw_2s": [],
        "yaw_3s": [],
        "dec_acc": [],
        "parking_success": []
    }
    
    with open(cfg.output_file, 'w') as f:
        f.write("========== OmniVLA Evaluation Results ==========\n\n")
    
    for traj_dir in tqdm(traj_dirs, desc="Testing Trajectories"):
        dataset = SingleTrajTestDataset(
            traj_dir=traj_dir,
            tokenizer=tokenizer,
            image_transform=processor.image_processor.apply_transform,
            max_history=cfg.max_history,
            future_steps=cfg.future_steps,
            history_mode=cfg.history_mode,
            distance_interval=cfg.distance_interval,
            turn_yaw_thresh=cfg.turn_yaw_thresh,
            turn_dense_interval=cfg.turn_dense_interval
        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda b: custom_collate_fn(b, pad_token_id))
        
        traj_l2_1s = []
        traj_l2_2s = []
        traj_l2_3s = []
        traj_yaw_1s = []
        traj_yaw_2s = []
        traj_yaw_3s = []
        correct_decision = 0
        total_steps = len(dataset)
        
        last_step_success = False

        for step, batch in enumerate(dataloader):
            gt_action = batch['action_gt'].to(device).to(torch.bfloat16)
            gt_decision = batch['decision_gt'].to(device)
            
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = vla(batch, past_traj_projector=past_traj_projector)
                    last_hidden = outputs.hidden_states[-1]
                    
                    num_act_tokens = FUTURE_ACTION_WAYPOINTS * ACTION_DIM
                    decision_hidden = last_hidden[:, -(num_act_tokens + 3), :]      # 1 dec + 8*4 action tokens + 1 eos + 1 shift token
                    actions_hidden = last_hidden[:, -(num_act_tokens + 2):-2, :]    # 8*4 action tokens + 1 eos + 1 shift token

                    pred_decision_logits = decision_head(decision_hidden)
                    pred_actions_flat = action_head(actions_hidden).squeeze(-1)
                    pred_actions = pred_actions_flat.view(-1, FUTURE_ACTION_WAYPOINTS, ACTION_DIM)
            
            pred_point_1s = pred_actions[0, 1, 0:4].cpu().float() # 1秒 (第2个点，idx 1)
            pred_point_2s = pred_actions[0, 3, 0:4].cpu().float() # 2秒 (第4个点，idx 3)
            pred_point_3s = pred_actions[0, 5, 0:4].cpu().float() # 3秒 (第6个点，idx 5)
            
            gt_point_1s = gt_action[0, 1, 0:4].cpu().float()
            gt_point_2s = gt_action[0, 3, 0:4].cpu().float()
            gt_point_3s = gt_action[0, 5, 0:4].cpu().float()
            
            traj_l2_1s.append(calc_l2_distance(pred_point_1s, gt_point_1s))
            traj_l2_2s.append(calc_l2_distance(pred_point_2s, gt_point_2s))
            traj_l2_3s.append(calc_l2_distance(pred_point_3s, gt_point_3s))
            
            traj_yaw_1s.append(calc_yaw_diff(pred_point_1s, gt_point_1s))
            traj_yaw_2s.append(calc_yaw_diff(pred_point_2s, gt_point_2s))
            traj_yaw_3s.append(calc_yaw_diff(pred_point_3s, gt_point_3s))
            
            # Decision accuracy
            pred_dec = pred_decision_logits.argmax(dim=-1)[0]
            if pred_dec.item() == gt_decision[0].item():
                correct_decision += 1
                
            # Parking Success check (last step)
            if step == total_steps - 1:
                last_step_success = (pred_dec.item() == gt_decision[0].item())
                
        # 统计本条轨迹结果
        avg_l2_1s = np.mean(traj_l2_1s)
        avg_l2_2s = np.mean(traj_l2_2s)
        avg_l2_3s = np.mean(traj_l2_3s)
        avg_yaw_1s = np.mean(traj_yaw_1s)
        avg_yaw_2s = np.mean(traj_yaw_2s)
        avg_yaw_3s = np.mean(traj_yaw_3s)
        dec_acc = correct_decision / total_steps
        
        global_results['l2_1s'].append(avg_l2_1s)
        global_results['l2_2s'].append(avg_l2_2s)
        global_results['l2_3s'].append(avg_l2_3s)
        global_results['yaw_1s'].append(avg_yaw_1s)
        global_results['yaw_2s'].append(avg_yaw_2s)
        global_results['yaw_3s'].append(avg_yaw_3s)
        global_results['dec_acc'].append(dec_acc)
        global_results['parking_success'].append(1 if last_step_success else 0)
        
        traj_name = os.path.basename(traj_dir)
        with open(cfg.output_file, 'a') as f:
            f.write(f"Traj: {traj_name}\n")
            f.write(f"  L2 Dist 1s: {avg_l2_1s:.4f} m | L2 Dist 2s: {avg_l2_2s:.4f} m | L2 Dist 3s: {avg_l2_3s:.4f} m\n")
            f.write(f"  Yaw Error 1s: {avg_yaw_1s:.4f} rad | Yaw Error 2s: {avg_yaw_2s:.4f} rad | Yaw Error 3s: {avg_yaw_3s:.4f} rad\n")
            f.write(f"  Decision Acc: {dec_acc*100:.2f}%\n")
            f.write(f"  Parking Success: {'Yes' if last_step_success else 'No'}\n\n")

    # 全局统计
    final_l2_1s = np.mean(global_results['l2_1s'])
    final_l2_2s = np.mean(global_results['l2_2s'])
    final_l2_3s = np.mean(global_results['l2_3s'])
    final_yaw_1s = np.mean(global_results['yaw_1s'])
    final_yaw_2s = np.mean(global_results['yaw_2s'])
    final_yaw_3s = np.mean(global_results['yaw_3s'])
    final_dec_acc = np.mean(global_results['dec_acc'])
    final_parking_success = np.mean(global_results['parking_success'])
    
    summary = (
        "========== Global Summary ==========\n"
        f"Total Trajectories Evaluated: {len(traj_dirs)}\n"
        f"Average L2 Dist 1s (2nd pt):   {final_l2_1s:.4f} m\n"
        f"Average L2 Dist 2s (4th pt):   {final_l2_2s:.4f} m\n"
        f"Average L2 Dist 3s (6th pt):   {final_l2_3s:.4f} m\n"
        f"Average Yaw Err 1s (2nd pt):   {final_yaw_1s:.4f} rad\n"
        f"Average Yaw Err 2s (4th pt):   {final_yaw_2s:.4f} rad\n"
        f"Average Yaw Err 3s (6th pt):   {final_yaw_3s:.4f} rad\n"
        f"Average Decision Accuracy: {final_dec_acc*100:.2f}%\n"
        f"Parking Success Rate:      {final_parking_success*100:.2f}%\n"
    )
    
    print(summary)
    with open(cfg.output_file, 'a') as f:
        f.write(summary)

if __name__ == "__main__":
    evaluate_model()
