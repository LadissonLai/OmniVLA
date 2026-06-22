import os
import re
import json
import time
import torch
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor, AutoConfig, AutoImageProcessor, AutoModelForVision2Seq
from peft import PeftModel
from accelerate import PartialState
from accelerate.utils import gather_object
import draccus

from core.exp_parking_dataset import MultiTrajClassicTestDataset, classic_collate_fn
from core.modeling_prismatic import ClassicVLAForActionPrediction
from core.configuration_prismatic import OpenVLAConfig
from core.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from core.utils import model_is_on_hf_hub
from core.constants import FUTURE_ACTION_WAYPOINTS, ACTION_DIM


@dataclass
class ClassicVLATestConfig:
    vla_path: str = "/root/autodl-tmp/codes/OmniVLA/openvla-7b"
    resume_dir: str = "/root/autodl-tmp/codes/OmniVLA/runs/runs_classicvla_smart_history/2026-06-15_17-23/step_12852_loss_0.3333_ckpt"
    data_root: str = "/root/autodl-tmp/codes/OmniVLA/datasets/ParkingVLA_testsets"
    output_file: str = "metrics/test_classicvla_smart_history_H20_6epoch.txt"

    history_mode: str = 'smart'
    max_history: int = 8
    future_steps: int = 8
    distance_interval: float = 0.5
    turn_yaw_thresh: float = 5.0
    turn_dense_interval: float = 0.1

    max_new_tokens: int = 512

    # 并行测试配置
    eval_batch_size: int = 16      # 每张卡的 batch size
    num_workers: int = 4           # 每进程 DataLoader worker 数


def calc_l2_distance(pred, gt):
    return torch.sqrt((pred[0] - gt[0]) ** 2 + (pred[1] - gt[1]) ** 2).item()


def calc_yaw_diff(pred, gt):
    pred_yaw = torch.atan2(pred[3], pred[2])
    gt_yaw = torch.atan2(gt[3], gt[2])
    diff_rad = torch.atan2(torch.sin(pred_yaw - gt_yaw), torch.cos(pred_yaw - gt_yaw))
    return torch.abs(diff_rad).item()


def parse_output(text: str):
    """从生成文本中提取 decision 和 trajectory，失败返回 None。

    处理 VLM 文本输出的边界情况：
      - 去除 markdown ``` 围栏；
      - 非贪婪匹配第一个完整 {...}（trajectory 内仅数字数组、无嵌套花括号）；
      - JSON 解析、键缺失、类型错误统一兜底为 None；
      - 形状必须为 (FUTURE_ACTION_WAYPOINTS, ACTION_DIM)；
      - 数值必须全部有限（拦截 NaN/Inf，避免污染 np.mean）。
    """
    if not text:
        return None
    # 去掉 ```json / ``` 等 markdown 围栏
    text = text.replace("```json", " ").replace("```", " ")
    m = re.search(r'\{.*?\}', text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group())
        decision = int(obj['decision'])
        traj = torch.tensor(obj['trajectory'], dtype=torch.float32)  # [8, 4]
        if traj.shape != (FUTURE_ACTION_WAYPOINTS, ACTION_DIM):
            return None
        if not torch.isfinite(traj).all():
            return None
        return decision, traj
    except Exception:
        return None


def evaluate_sample(gen_text, gt_action, gt_decision):
    """对单个样本的生成文本计算指标，返回 per-sample 指标字典。"""
    parsed = parse_output(gen_text)
    if parsed is None:
        parse_fail = 1
        pred_actions = torch.zeros(8, 4)
        pred_decision = -1
    else:
        parse_fail = 0
        pred_decision, pred_actions = parsed
        pred_actions = pred_actions.float()

    pred_point_1s = pred_actions[1, 0:4]
    pred_point_2s = pred_actions[3, 0:4]
    pred_point_3s = pred_actions[5, 0:4]
    gt_point_1s = gt_action[1, 0:4].cpu().float()
    gt_point_2s = gt_action[3, 0:4].cpu().float()
    gt_point_3s = gt_action[5, 0:4].cpu().float()

    return {
        "l2_1s": calc_l2_distance(pred_point_1s, gt_point_1s),
        "l2_2s": calc_l2_distance(pred_point_2s, gt_point_2s),
        "l2_3s": calc_l2_distance(pred_point_3s, gt_point_3s),
        "yaw_1s": calc_yaw_diff(pred_point_1s, gt_point_1s),
        "yaw_2s": calc_yaw_diff(pred_point_2s, gt_point_2s),
        "yaw_3s": calc_yaw_diff(pred_point_3s, gt_point_3s),
        "decision_correct": 1 if pred_decision == gt_decision else 0,
        "parse_fail": parse_fail,
    }


def aggregate_traj(traj_name, records):
    """将某条轨迹的所有 per-sample record 聚合成该轨迹的指标块。"""
    records = sorted(records, key=lambda r: r["step_idx"])
    total_steps = len(records)

    def mean(key):
        return float(np.mean([r[key] for r in records]))

    avg_l2_1s, avg_l2_2s, avg_l2_3s = mean("l2_1s"), mean("l2_2s"), mean("l2_3s")
    avg_yaw_1s, avg_yaw_2s, avg_yaw_3s = mean("yaw_1s"), mean("yaw_2s"), mean("yaw_3s")
    avg_inf_time = mean("inf_time")
    dec_acc = mean("decision_correct")
    parse_fail_rate = mean("parse_fail")

    # Parking Success：该轨迹末帧 decision 是否正确
    last_records = [r for r in records if r["is_last"]]
    last_step_success = bool(last_records[0]["decision_correct"]) if last_records else False

    text_block = (
        f"Traj: {traj_name}\n"
        f"  L2 Dist 1s: {avg_l2_1s:.4f} m | 2s: {avg_l2_2s:.4f} m | 3s: {avg_l2_3s:.4f} m\n"
        f"  Yaw Err 1s: {avg_yaw_1s:.4f} rad | 2s: {avg_yaw_2s:.4f} rad | 3s: {avg_yaw_3s:.4f} rad\n"
        f"  Decision Acc: {dec_acc*100:.2f}%\n"
        f"  Avg Inference Time: {avg_inf_time:.4f} s/step\n"
        f"  Parking Success: {'Yes' if last_step_success else 'No'}\n"
        f"  Parse Fail Rate: {parse_fail_rate*100:.2f}%\n\n"
    )

    return {
        "name": traj_name,
        "text": text_block,
        "l2_1s": avg_l2_1s, "l2_2s": avg_l2_2s, "l2_3s": avg_l2_3s,
        "yaw_1s": avg_yaw_1s, "yaw_2s": avg_yaw_2s, "yaw_3s": avg_yaw_3s,
        "dec_acc": dec_acc,
        "parking_success": 1 if last_step_success else 0,
        "parse_fail_rate": parse_fail_rate,
        "inf_time": avg_inf_time,
    }


@draccus.wrap()
def evaluate(cfg: ClassicVLATestConfig):
    distributed_state = PartialState()
    device = torch.device(f"cuda:{distributed_state.local_process_index}"
                          if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    if not model_is_on_hf_hub(cfg.vla_path):
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, ClassicVLAForActionPrediction)

    if distributed_state.is_main_process:
        print("Loading processor and base model...")
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id else 0
    image_transform = processor.image_processor.apply_transform

    vla = ClassicVLAForActionPrediction.from_pretrained(
        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device)

    if cfg.resume_dir:
        if distributed_state.is_main_process:
            print(f"Loading LoRA weights from {cfg.resume_dir} ...")
        vla = PeftModel.from_pretrained(vla, os.path.join(cfg.resume_dir, "lora_adapter"), is_trainable=False)

    vla.eval()

    # === 全轨迹测试集：所有轨迹的所有帧展开为统一样本集 ===
    dataset = MultiTrajClassicTestDataset(
        data_root=cfg.data_root,
        tokenizer=tokenizer,
        image_transform=image_transform,
        max_history=cfg.max_history,
        future_steps=cfg.future_steps,
        history_mode=cfg.history_mode,
        distance_interval=cfg.distance_interval,
        turn_yaw_thresh=cfg.turn_yaw_thresh,
        turn_dense_interval=cfg.turn_dense_interval,
    )

    # 样本级数据并行：按 rank round-robin 切分（单卡时 num_processes=1 → 全量）
    my_indices = list(range(distributed_state.process_index, len(dataset), distributed_state.num_processes))
    subset = Subset(dataset, my_indices)
    dataloader = DataLoader(
        subset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=lambda b: classic_collate_fn(b, pad_token_id),
    )

    local_records = []
    for batch in tqdm(
        dataloader,
        desc=f"[rank {distributed_state.process_index}] Testing",
        disable=not distributed_state.is_main_process,
    ):
        for k in ['pixel_values_front', 'pixel_values_rear', 'pixel_values_left', 'pixel_values_right']:
            batch[k] = batch[k].to(device)

        gt_action = batch['action_gt'].to(torch.float32)          # [B, 8, 4]
        gt_decision = batch['decision_gt']                        # [B]
        traj_names = batch['traj_name']                           # list[str]
        step_idxs = batch['step_idx']                             # [B]
        is_last = batch['is_last']                                # [B]
        bsz = gt_action.shape[0]

        start_time = time.time()
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                gen_ids = vla.generate_answer(batch, max_new_tokens=cfg.max_new_tokens)
        per_sample_time = (time.time() - start_time) / bsz

        gen_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        for i in range(bsz):
            metrics = evaluate_sample(gen_texts[i], gt_action[i], gt_decision[i].item())
            metrics.update({
                "traj_name": traj_names[i],
                "step_idx": int(step_idxs[i].item()),
                "is_last": bool(is_last[i].item()),
                "inf_time": per_sample_time,
            })
            local_records.append(metrics)

    # 汇总所有 rank 的 per-sample record（单卡时原样返回）
    distributed_state.wait_for_everyone()
    all_records = gather_object(local_records)

    if not distributed_state.is_main_process:
        return

    # 按轨迹分组聚合
    grouped = defaultdict(list)
    for r in all_records:
        grouped[r["traj_name"]].append(r)

    traj_results = [aggregate_traj(name, recs) for name, recs in grouped.items()]
    traj_results.sort(key=lambda r: r["name"])

    def col(key):
        return [r[key] for r in traj_results]

    out_dir = os.path.dirname(cfg.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(cfg.output_file, 'w') as f:
        f.write("========== ClassicVLA Evaluation Results ==========\n\n")
        for r in traj_results:
            f.write(r["text"])

        summary = (
            "========== Global Summary ==========\n"
            f"Total Trajectories Evaluated: {len(traj_results)}\n"
            f"Average L2 Dist 1s (2nd pt):   {np.mean(col('l2_1s')):.4f} m\n"
            f"Average L2 Dist 2s (4th pt):   {np.mean(col('l2_2s')):.4f} m\n"
            f"Average L2 Dist 3s (6th pt):   {np.mean(col('l2_3s')):.4f} m\n"
            f"Average Yaw Err 1s (2nd pt):   {np.mean(col('yaw_1s')):.4f} rad\n"
            f"Average Yaw Err 2s (4th pt):   {np.mean(col('yaw_2s')):.4f} rad\n"
            f"Average Yaw Err 3s (6th pt):   {np.mean(col('yaw_3s')):.4f} rad\n"
            f"Average Decision Accuracy:     {np.mean(col('dec_acc'))*100:.2f}%\n"
            f"Global Average Inference Time: {np.mean(col('inf_time')):.4f} s/step\n"
            f"Parking Success Rate:          {np.mean(col('parking_success'))*100:.2f}%\n"
            f"Average Parse Fail Rate:       {np.mean(col('parse_fail_rate'))*100:.2f}%\n"
        )
        f.write(summary)

    print(summary)


if __name__ == "__main__":
    evaluate()
