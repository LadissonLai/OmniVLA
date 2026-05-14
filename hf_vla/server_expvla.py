"""
server_expvla.py
FastAPI inference server for ExpVLA model.

启动方式（使用 draccus CLI）:
    python server_expvla.py --vla_path /path/to/model --resume_dir /path/to/ckpt
    python server_expvla.py --port 8001 --history_mode fixed_count
"""
import os
import sys
import time
import base64
import io
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import List

import draccus
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from transformers import AutoProcessor, AutoConfig, AutoImageProcessor, AutoModelForVision2Seq
from peft import PeftModel

from core.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from core.configuration_prismatic import OpenVLAConfig
from core.modeling_prismatic import ExpVLAForActionPrediction
from core.utils import model_is_on_hf_hub
from core.constants import FUTURE_ACTION_WAYPOINTS, ACTION_DIM


# ── 服务端配置（与 test_expvla.py 风格一致）────────────────────────────────────

@dataclass
class ServerConfig:
    vla_path: str = "/public/home/lqq_202430131053/codes/OmniVLA/merged/openvla-7b-classic-merged"
    resume_dir: str = "/public/home/lqq_202430131053/codes/OmniVLA/runs_expvla_2_decode_smart_history_OFT2/2026-05-10_19-13/step_59744_loss_0.0748_ckpt"
    host: str = "0.0.0.0"
    port: int = 9999

    # History Trajectory 配置，与训练保持一致
    history_mode: str = "smart"          # 'smart' | 'fixed_count' | 'fixed_distance'
    max_history: int = 8
    distance_interval: float = 0.5
    turn_yaw_thresh: float = 5.0
    turn_dense_interval: float = 0.1


# ── 全局状态（模型 + 配置）───────────────────────────────────────────────────

_MODEL: dict = {}
_CFG: ServerConfig = None  # 由 main() 写入


# ── 模型加载 ─────────────────────────────────────────────────────────────────

def _load_models(cfg: ServerConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not model_is_on_hf_hub(cfg.vla_path):
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, ExpVLAForActionPrediction)

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id else 0

    print("Loading base VLA model...")
    vla = ExpVLAForActionPrediction.from_pretrained(
        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device)

    print(f"Loading LoRA adapter from {cfg.resume_dir}...")
    vla = PeftModel.from_pretrained(
        vla, os.path.join(cfg.resume_dir, "lora_adapter"), is_trainable=False
    )

    llm_dim    = vla.config.text_config.hidden_size
    vocab_size = vla.config.text_config.vocab_size

    past_traj_projector = nn.Sequential(
        nn.Linear(4, llm_dim // 2), nn.GELU(), nn.Linear(llm_dim // 2, llm_dim)
    ).to(device).to(torch.bfloat16)

    action_head = nn.Sequential(
        nn.Linear(llm_dim, llm_dim), nn.GELU(), nn.Linear(llm_dim, 1)
    ).to(device).to(torch.bfloat16)

    decision_head = nn.Linear(llm_dim, vocab_size).to(device).to(torch.bfloat16)

    def _load_ckpt(model, path):
        state_dict = torch.load(path, map_location=device)
        cleaned = {(k[len("module."):] if k.startswith("module.") else k): v
                   for k, v in state_dict.items()}
        model.load_state_dict(cleaned)

    _load_ckpt(past_traj_projector, os.path.join(cfg.resume_dir, "past_traj_projector.pt"))
    _load_ckpt(action_head,         os.path.join(cfg.resume_dir, "action_head.pt"))
    _load_ckpt(decision_head,       os.path.join(cfg.resume_dir, "decision_head.pt"))

    for m in (vla, past_traj_projector, action_head, decision_head):
        m.eval()

    _MODEL.update({
        "device":              device,
        "tokenizer":           tokenizer,
        "pad_token_id":        pad_token_id,
        "image_transform":     processor.image_processor.apply_transform,
        "vla":                 vla,
        "past_traj_projector": past_traj_projector,
        "action_head":         action_head,
        "decision_head":       decision_head,
    })
    print("All models loaded and ready.")


# ── 历史轨迹采样（复刻 SingleTrajTestDataset 逻辑）──────────────────────────

def _to_local_coords(tx, ty, tyaw_deg, cx, cy, cyaw_deg) -> np.ndarray:
    """全局坐标 → 自车局部坐标 [x, y, cos(yaw), sin(yaw)]"""
    cyaw = np.deg2rad(cyaw_deg)
    tyaw = np.deg2rad(tyaw_deg)
    dx = tx - cx
    dy = ty - cy
    local_x   = dx * np.cos(cyaw) + dy * np.sin(cyaw)
    local_y   = -dx * np.sin(cyaw) + dy * np.cos(cyaw)
    local_yaw = tyaw - cyaw
    return np.array([local_x, local_y, np.cos(local_yaw), np.sin(local_yaw)], dtype=np.float32)


def _compute_history(
    history_poses: list,   # list of {"x", "y", "yaw"} dicts (global, oldest first)
    curr_pose: dict,       # {"x", "y", "yaw"} global
    cfg: ServerConfig,
) -> torch.Tensor:
    """
    根据 cfg.history_mode 从全局历史轨迹计算局部坐标 Tensor [N, 4]。
    对应 SingleTrajTestDataset 中的历史轨迹采样逻辑。
    """
    cx, cy, cyaw = curr_pose["x"], curr_pose["y"], curr_pose["yaw"]
    t = len(history_poses)

    if t == 0:
        return torch.zeros((1, 4), dtype=torch.float32)

    def lc(row):
        return _to_local_coords(row["x"], row["y"], row["yaw"], cx, cy, cyaw)

    history_traj = []

    if cfg.history_mode == "fixed_count":
        if t <= cfg.max_history:
            history_traj = [lc(row) for row in history_poses]
        else:
            indices = np.linspace(0, t - 1, cfg.max_history, dtype=int)
            history_traj = [lc(history_poses[i]) for i in indices]

    elif cfg.history_mode == "fixed_distance":
        last_idx = -1
        for i in range(t):
            if i == 0 or i == t - 1:
                history_traj.append(lc(history_poses[i]))
                last_idx = i
            else:
                dist = np.hypot(
                    history_poses[i]["x"] - history_poses[last_idx]["x"],
                    history_poses[i]["y"] - history_poses[last_idx]["y"],
                )
                if dist >= cfg.distance_interval:
                    history_traj.append(lc(history_poses[i]))
                    last_idx = i

    elif cfg.history_mode == "smart":
        # 判断每一帧是直行还是转弯
        states = []
        for i in range(t):
            start = max(0, i - 2)
            end   = min(t - 1, i + 2)
            yaw_diff = abs(history_poses[end]["yaw"] - history_poses[start]["yaw"]) % 360
            if yaw_diff > 180:
                yaw_diff = 360 - yaw_diff
            states.append("Turn" if yaw_diff > cfg.turn_yaw_thresh else "Straight")

        last_idx = -1
        for i in range(t):
            pt = lc(history_poses[i])
            if i == 0 or i == t - 1:
                history_traj.append(pt)
                last_idx = i
            else:
                if states[i] == "Straight":
                    if states[i] != states[i - 1] or (i + 1 < t and states[i] != states[i + 1]):
                        history_traj.append(pt)
                        last_idx = i
                else:
                    dist = np.hypot(
                        history_poses[i]["x"] - history_poses[last_idx]["x"],
                        history_poses[i]["y"] - history_poses[last_idx]["y"],
                    )
                    if dist >= cfg.turn_dense_interval:
                        history_traj.append(pt)
                        last_idx = i
    else:
        raise ValueError(f"Unknown history_mode: {cfg.history_mode}")

    if len(history_traj) == 0:
        return torch.zeros((1, 4), dtype=torch.float32)

    return torch.tensor(np.array(history_traj), dtype=torch.float32)


# ── Pydantic 请求/响应体 ──────────────────────────────────────────────────────

class GlobalPose(BaseModel):
    """全局坐标系下的一个位姿（yaw 单位：度）"""
    x:   float
    y:   float
    yaw: float


class ParkingSlot(BaseModel):
    id: int
    x:  float
    y:  float


class PredictRequest(BaseModel):
    instruction:     str               # 人类泊车自然语言指令
    image_front_b64: str               # 前视图像 base64（JPEG/PNG）
    image_rear_b64:  str
    image_left_b64:  str
    image_right_b64: str
    history_poses:   List[GlobalPose]  # 历史全局位姿序列（最旧在前，不含当前帧）
    curr_pose:       GlobalPose        # 当前帧全局位姿
    parking_slots:   List[ParkingSlot] # 当前帧检测到的停车位（可为空）


class Waypoint(BaseModel):
    x:       float
    y:       float
    cos_yaw: float
    sin_yaw: float


class PredictResponse(BaseModel):
    trajectory:       List[Waypoint]  # 未来 8 个轨迹点（自车局部坐标）
    decision_id:      int             # 预测的停车位 ID
    decision_str:     str             # 解码后的决策字符串
    inference_time_s: float           # 纯模型推理耗时（秒）


# ── 预处理 ────────────────────────────────────────────────────────────────────

def _decode_image(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")


def _build_batch(req: PredictRequest) -> dict:
    tokenizer    = _MODEL["tokenizer"]
    image_tf     = _MODEL["image_transform"]
    pad_token_id = _MODEL["pad_token_id"]
    device       = _MODEL["device"]

    def tokenize(text: str) -> torch.Tensor:
        return torch.tensor(
            tokenizer.encode(text, add_special_tokens=False), dtype=torch.long
        )

    # ── 图像预处理 ──────────────────────────────────────────────────────────
    pixel_values = {
        view: image_tf(_decode_image(b64)).unsqueeze(0).to(device)
        for view, b64 in (
            ("front", req.image_front_b64),
            ("rear",  req.image_rear_b64),
            ("left",  req.image_left_b64),
            ("right", req.image_right_b64),
        )
    }

    # ── 历史轨迹（服务端按配置 mode 采样，转换为局部坐标）─────────────────
    history_poses_dicts = [{"x": p.x, "y": p.y, "yaw": p.yaw} for p in req.history_poses]
    curr_pose_dict      = {"x": req.curr_pose.x, "y": req.curr_pose.y, "yaw": req.curr_pose.yaw}
    history_tensor = _compute_history(history_poses_dicts, curr_pose_dict, _CFG)

    # ── 停车位文本 ──────────────────────────────────────────────────────────
    if len(req.parking_slots) == 0:
        slots_str = "[]"
    else:
        slots_str = "Detected parking slots:[" + ",".join(
            f"(id={s.id},x={s.x:.2f},y={s.y:.2f})" for s in req.parking_slots
        ) + "]"

    # ── 组装 batch dict（batch_size=1）─────────────────────────────────────
    batch = {
        "pixel_values_front": pixel_values["front"],
        "pixel_values_rear":  pixel_values["rear"],
        "pixel_values_left":  pixel_values["left"],
        "pixel_values_right": pixel_values["right"],
        "history_traj":       [history_tensor],   # 模型内部做 padding/mask
        "instruction":        [req.instruction],
        "pad_token_id":       pad_token_id,
        "sys_prompt_ids": tokenize(
            "Please predict the future trajectory and select the parking slot id "
            "based on the human instruction, the current four observation images, "
            "the history trajectory, and the detected parking slots."
        ).unsqueeze(0).to(device),
        "instruct_ids": tokenize(f"Instruction: {req.instruction}").unsqueeze(0).to(device),
        "p_front_ids":  tokenize("Front view:").unsqueeze(0).to(device),
        "p_rear_ids":   tokenize("Rear view:").unsqueeze(0).to(device),
        "p_left_ids":   tokenize("Left view:").unsqueeze(0).to(device),
        "p_right_ids":  tokenize("Right view:").unsqueeze(0).to(device),
        "p_hist_ids":   tokenize("History trajectory:").unsqueeze(0).to(device),
        "p_slots_ids":  tokenize(
            f"Detected parking slot information: {slots_str}"
        ).unsqueeze(0).to(device),
    }
    return batch


# ── FastAPI App ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models(_CFG)
    yield
    _MODEL.clear()


app = FastAPI(title="ExpVLA Inference Server", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": bool(_MODEL),
        "history_mode": _CFG.history_mode if _CFG else "unknown",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not _MODEL:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    batch = _build_batch(req)

    vla                 = _MODEL["vla"]
    past_traj_projector = _MODEL["past_traj_projector"]
    action_head         = _MODEL["action_head"]
    decision_head       = _MODEL["decision_head"]
    tokenizer           = _MODEL["tokenizer"]

    t0 = time.time()
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs     = vla(batch, past_traj_projector=past_traj_projector)
            last_hidden = outputs.hidden_states[-1]  # [1, seq_len, llm_dim]

            num_act_tokens = FUTURE_ACTION_WAYPOINTS * ACTION_DIM
            # 从末尾往前：EOS | act_tokens(32) | dec_token | ...
            decision_hidden = last_hidden[:, -(num_act_tokens + 3), :]     # [1, llm_dim]
            actions_hidden  = last_hidden[:, -(num_act_tokens + 2):-2, :]  # [1, 32, llm_dim]

            pred_decision_logits = decision_head(decision_hidden)           # [1, vocab]
            pred_actions_flat    = action_head(actions_hidden).squeeze(-1)  # [1, 32]
            pred_actions = pred_actions_flat.view(-1, FUTURE_ACTION_WAYPOINTS, ACTION_DIM)  # [1,8,4]
    elapsed = time.time() - t0

    pred_dec_token = pred_decision_logits.argmax(dim=-1)[0].item()
    pred_dec_str   = tokenizer.decode(pred_dec_token).replace("<pad>", "").strip()
    try:
        decision_id = int(pred_dec_str)
    except ValueError:
        decision_id = pred_dec_token

    traj = pred_actions[0].cpu().float().tolist()  # [8, 4]

    return PredictResponse(
        trajectory=[Waypoint(x=p[0], y=p[1], cos_yaw=p[2], sin_yaw=p[3]) for p in traj],
        decision_id=decision_id,
        decision_str=pred_dec_str,
        inference_time_s=elapsed,
    )


# ── 入口（使用 draccus，与 test_expvla.py 保持一致）────────────────────────────

@draccus.wrap()
def main(cfg: ServerConfig):
    global _CFG
    _CFG = cfg
    print(f"[Config] vla_path    : {cfg.vla_path}")
    print(f"[Config] resume_dir  : {cfg.resume_dir}")
    print(f"[Config] history_mode: {cfg.history_mode}")
    print(f"[Config] host:port   : {cfg.host}:{cfg.port}")
    uvicorn.run(app, host=cfg.host, port=cfg.port)


if __name__ == "__main__":
    main()
