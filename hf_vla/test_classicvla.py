import os
import re
import json
import time
import torch
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoConfig, AutoImageProcessor, AutoModelForVision2Seq
from peft import PeftModel
import draccus

from core.exp_parking_dataset import SingleTrajClassicTestDataset, classic_collate_fn
from core.modeling_prismatic import ClassicVLAForActionPrediction
from core.configuration_prismatic import OpenVLAConfig
from core.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from core.utils import model_is_on_hf_hub, visualize_test_expvla
from core.constants import FUTURE_ACTION_WAYPOINTS, ACTION_DIM


@dataclass
class ClassicVLATestConfig:
    vla_path: str = "/public/home/lqq_202430131053/codes/OmniVLA/openvla-7b"
    resume_dir: str = "/public/home/lqq_202430131053/codes/OmniVLA/runs_classicvla_smart_history/2026-05-04_14-42/step_56010_loss_0.1578_ckpt"
    data_root: str = "/public/home/lqq_202430131053/codes/datasets/ParkingVLA_Val"
    output_file: str = "test_classicvla_smart_history_16epoch.txt"

    history_mode: str = 'smart'
    max_history: int = 8
    future_steps: int = 8
    distance_interval: float = 0.5
    turn_yaw_thresh: float = 5.0
    turn_dense_interval: float = 0.1

    max_new_tokens: int = 512
    
    # 可视化配置
    save_vis: bool = True
    vis_dir: str = "vis_classicvla_smart_history_result"


def calc_l2_distance(pred, gt):
    return torch.sqrt((pred[0] - gt[0]) ** 2 + (pred[1] - gt[1]) ** 2).item()


def calc_yaw_diff(pred, gt):
    pred_yaw = torch.atan2(pred[3], pred[2])
    gt_yaw = torch.atan2(gt[3], gt[2])
    diff_rad = torch.atan2(torch.sin(pred_yaw - gt_yaw), torch.cos(pred_yaw - gt_yaw))
    return torch.abs(diff_rad).item()


def parse_output(text: str):
    """从生成文本中提取 decision 和 trajectory，失败返回 None。"""
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group())
        print(f"Parsed JSON object: {obj}")
        decision = int(obj['decision'])
        traj = torch.tensor(obj['trajectory'], dtype=torch.float32)  # [8, 4]
        if traj.shape != (FUTURE_ACTION_WAYPOINTS, ACTION_DIM):
            return None
        return decision, traj
    except Exception:
        return None


@draccus.wrap()
def evaluate(cfg: ClassicVLATestConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not model_is_on_hf_hub(cfg.vla_path):
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, ClassicVLAForActionPrediction)

    print("Loading processor and base model...")
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id else 0

    vla = ClassicVLAForActionPrediction.from_pretrained(
        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device)

    if cfg.resume_dir:
        print(f"Loading LoRA weights from {cfg.resume_dir} ...")
        vla = PeftModel.from_pretrained(vla, os.path.join(cfg.resume_dir, "lora_adapter"), is_trainable=False)

    vla.eval()

    traj_dirs = sorted([
        os.path.join(cfg.data_root, d)
        for d in os.listdir(cfg.data_root)
        if os.path.isdir(os.path.join(cfg.data_root, d))
    ])

    global_results = {
        "l2_1s": [], "l2_2s": [], "l2_3s": [],
        "yaw_1s": [], "yaw_2s": [], "yaw_3s": [],
        "dec_acc": [], "parking_success": [],
        "parse_fail_rate": [], "inf_time": [],
    }

    with open(cfg.output_file, 'w') as f:
        f.write("========== ClassicVLA Evaluation Results ==========\n\n")

    for traj_dir in tqdm(traj_dirs, desc="Testing Trajectories"):
        dataset = SingleTrajClassicTestDataset(
            traj_dir=traj_dir,
            tokenizer=tokenizer,
            image_transform=processor.image_processor.apply_transform,
            max_history=cfg.max_history,
            future_steps=cfg.future_steps,
            history_mode=cfg.history_mode,
            distance_interval=cfg.distance_interval,
            turn_yaw_thresh=cfg.turn_yaw_thresh,
            turn_dense_interval=cfg.turn_dense_interval,
        )
        dataloader = DataLoader(
            dataset, batch_size=1, shuffle=False,
            collate_fn=lambda b: classic_collate_fn(b, pad_token_id),
        )

        traj_l2_1s, traj_l2_2s, traj_l2_3s = [], [], []
        traj_yaw_1s, traj_yaw_2s, traj_yaw_3s = [], [], []
        traj_inf_time = []
        correct_decision = 0
        parse_fails = 0
        total_steps = len(dataset)
        last_step_success = False

        for step, batch in enumerate(dataloader):
            for k in ['pixel_values_front', 'pixel_values_rear', 'pixel_values_left', 'pixel_values_right']:
                batch[k] = batch[k].to(device)

            gt_action = batch['action_gt'].to(device).to(torch.float32)  # [1, 8, 4]
            gt_decision = batch['decision_gt'][0].item()

            start_time = time.time()
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    gen_ids = vla.generate_answer(batch, max_new_tokens=cfg.max_new_tokens)

            gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            print(f"Step {step+1}/{total_steps} - Generated Text:\n{gen_text}\n")
            end_time = time.time()
            traj_inf_time.append(end_time - start_time)
            
            parsed = parse_output(gen_text)

            if parsed is None:
                parse_fails += 1
                # 解析失败时用全零轨迹填充，不影响其他步骤统计
                pred_actions = torch.zeros(8, 4)
                pred_decision = -1
            else:
                pred_decision, pred_actions = parsed
                pred_actions = pred_actions.float()

            pred_point_1s = pred_actions[1, 0:4]
            pred_point_2s = pred_actions[3, 0:4]
            pred_point_3s = pred_actions[5, 0:4]
            gt_point_1s = gt_action[0, 1, 0:4].cpu().float()
            gt_point_2s = gt_action[0, 3, 0:4].cpu().float()
            gt_point_3s = gt_action[0, 5, 0:4].cpu().float()

            traj_l2_1s.append(calc_l2_distance(pred_point_1s, gt_point_1s))
            traj_l2_2s.append(calc_l2_distance(pred_point_2s, gt_point_2s))
            traj_l2_3s.append(calc_l2_distance(pred_point_3s, gt_point_3s))
            traj_yaw_1s.append(calc_yaw_diff(pred_point_1s, gt_point_1s))
            traj_yaw_2s.append(calc_yaw_diff(pred_point_2s, gt_point_2s))
            traj_yaw_3s.append(calc_yaw_diff(pred_point_3s, gt_point_3s))

            if pred_decision == gt_decision:
                correct_decision += 1

            if step == total_steps - 1:
                last_step_success = (pred_decision == gt_decision)

            # Use visualization
            if cfg.save_vis:
                v_dir = cfg.vis_dir
                if not os.path.isabs(v_dir):
                    out_dir = os.path.dirname(cfg.output_file)
                    v_dir = os.path.join(out_dir, v_dir) if out_dir else v_dir
                    
                traj_name = os.path.basename(traj_dir)
                vis_save_path = os.path.join(v_dir, traj_name, f"step_{step}.png")
                instruction = batch.get('instruction', [""])[0] if 'instruction' in batch else "No instruction"
                
                history_traj = None
                p_hist_ids = batch.get('p_hist_ids')
                if p_hist_ids is not None:
                    hist_text = tokenizer.decode(p_hist_ids[0], skip_special_tokens=True)
                    matches = re.findall(r'\[([^\]]+)\]', hist_text.replace("History trajectory:", ""))
                    pts = []
                    for m in matches:
                        try:
                            pts.append([float(x) for x in m.split(',')])
                        except ValueError:
                            pass
                    if pts:
                        history_traj = torch.tensor(pts, dtype=torch.float32)
                
                visualize_test_expvla(
                    vis_save_path,
                    pred_actions,
                    gt_action[0],
                    history_traj,
                    str(pred_decision) if pred_decision != -1 else "Parse Fail",
                    str(gt_decision),
                    instruction,
                    batch['pixel_values_front'][0] if 'pixel_values_front' in batch else torch.zeros(3, 224, 224),
                    batch['pixel_values_rear'][0] if 'pixel_values_rear' in batch else torch.zeros(3, 224, 224),
                    batch['pixel_values_left'][0] if 'pixel_values_left' in batch else torch.zeros(3, 224, 224),
                    batch['pixel_values_right'][0] if 'pixel_values_right' in batch else torch.zeros(3, 224, 224)
                )

        avg_l2_1s = np.mean(traj_l2_1s)
        avg_l2_2s = np.mean(traj_l2_2s)
        avg_l2_3s = np.mean(traj_l2_3s)
        avg_yaw_1s = np.mean(traj_yaw_1s)
        avg_yaw_2s = np.mean(traj_yaw_2s)
        avg_yaw_3s = np.mean(traj_yaw_3s)
        avg_inf_time = np.mean(traj_inf_time)
        dec_acc = correct_decision / total_steps
        parse_fail_rate = parse_fails / total_steps

        global_results['l2_1s'].append(avg_l2_1s)
        global_results['l2_2s'].append(avg_l2_2s)
        global_results['l2_3s'].append(avg_l2_3s)
        global_results['yaw_1s'].append(avg_yaw_1s)
        global_results['yaw_2s'].append(avg_yaw_2s)
        global_results['yaw_3s'].append(avg_yaw_3s)
        global_results['dec_acc'].append(dec_acc)
        global_results['parking_success'].append(1 if last_step_success else 0)
        global_results['parse_fail_rate'].append(parse_fail_rate)
        global_results['inf_time'].append(avg_inf_time)

        traj_name = os.path.basename(traj_dir)
        with open(cfg.output_file, 'a') as f:
            f.write(f"Traj: {traj_name}\n")
            f.write(f"  L2 Dist 1s: {avg_l2_1s:.4f} m | 2s: {avg_l2_2s:.4f} m | 3s: {avg_l2_3s:.4f} m\n")
            f.write(f"  Yaw Err 1s: {avg_yaw_1s:.4f} rad | 2s: {avg_yaw_2s:.4f} rad | 3s: {avg_yaw_3s:.4f} rad\n")
            f.write(f"  Decision Acc: {dec_acc*100:.2f}%\n")
            f.write(f"  Avg Inference Time: {avg_inf_time:.4f} s/step\n")
            f.write(f"  Parking Success: {'Yes' if last_step_success else 'No'}\n")
            f.write(f"  Parse Fail Rate: {parse_fail_rate*100:.2f}%\n\n")

    summary = (
        "========== Global Summary ==========\n"
        f"Total Trajectories Evaluated: {len(traj_dirs)}\n"
        f"Average L2 Dist 1s (2nd pt):   {np.mean(global_results['l2_1s']):.4f} m\n"
        f"Average L2 Dist 2s (4th pt):   {np.mean(global_results['l2_2s']):.4f} m\n"
        f"Average L2 Dist 3s (6th pt):   {np.mean(global_results['l2_3s']):.4f} m\n"
        f"Average Yaw Err 1s (2nd pt):   {np.mean(global_results['yaw_1s']):.4f} rad\n"
        f"Average Yaw Err 2s (4th pt):   {np.mean(global_results['yaw_2s']):.4f} rad\n"
        f"Average Yaw Err 3s (6th pt):   {np.mean(global_results['yaw_3s']):.4f} rad\n"
        f"Average Decision Accuracy:     {np.mean(global_results['dec_acc'])*100:.2f}%\n"
        f"Global Average Inference Time: {np.mean(global_results['inf_time']):.4f} s/step\n"
        f"Parking Success Rate:          {np.mean(global_results['parking_success'])*100:.2f}%\n"
        f"Average Parse Fail Rate:       {np.mean(global_results['parse_fail_rate'])*100:.2f}%\n"
    )
    print(summary)
    with open(cfg.output_file, 'a') as f:
        f.write(summary)


if __name__ == "__main__":
    evaluate()
