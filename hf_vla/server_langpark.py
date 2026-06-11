"""
server_langpark.py
FastAPI inference server for LangPark VLA (VLN) model.

封装 test_langpark.py 的推理逻辑，供边缘设备闭环访问。

输入: 语言指令 / 四张环视图 / 历史轨迹(全局位姿) / 当前位姿 / 停车位列表
      (历史采样模式由服务端启动时配置, 不暴露给客户端)
输出: 预测轨迹(8 点) / 停车位 id / 语言指令进度(逐 token) / 推理时间

启动方式（使用 draccus CLI）:
    python server_langpark.py
    python server_langpark.py --resume_dir /path/to/ckpt --port 9999
    python server_langpark.py --history_mode fixed_count
"""
import os
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
from core.langpark_modules import MemoryEnhancementModule, InstructionAlignmentHead
from core.modeling_langpark import LangParkVLAForActionPrediction
from core.utils import model_is_on_hf_hub
from core.constants import FUTURE_ACTION_WAYPOINTS, ACTION_DIM


# ── 服务端配置（与 test_langpark.py 风格一致）────────────────────────────────

@dataclass
class ServerConfig:
    vla_path:   str = "/public/home/lqq_202430131053/codes/OmniVLA/openvla-7b"
    resume_dir: str = "/public/home/lqq_202430131053/codes/OmniVLA/runs_langpark/2026-06-03_15-59/step_33376_loss_0.0498_ckpt"
    host: str = "0.0.0.0"
    port: int = 9999

    # History Trajectory 默认配置，与训练保持一致；可被单次请求覆盖
    history_mode: str = "smart"          # 'smart' | 'fixed_count' | 'fixed_distance'
    max_history: int = 8
    distance_interval: float = 0.5
    turn_yaw_thresh: float = 5.0
    turn_dense_interval: float = 0.1

    # IAM 模块配置（必须与训练一致）
    num_mem_tokens:  int = 16
    mem_num_heads:   int = 8
    align_num_heads: int = 8


# ── 全局状态（模型 + 配置）───────────────────────────────────────────────────

_MODEL: dict = {}
_CFG: ServerConfig = None  # 由 main() 写入

# 语言进度三分类语义（对应 lang_label_frames.csv 的 token_mask_012）
_PHASE_STR = {0: "done", 1: "active", 2: "upcoming"}


# ── 模型加载 ─────────────────────────────────────────────────────────────────

def _load_models(cfg: ServerConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not model_is_on_hf_hub(cfg.vla_path):
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, LangParkVLAForActionPrediction)

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id else 0

    print("Loading base VLA model...")
    vla = LangParkVLAForActionPrediction.from_pretrained(
        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device)

    print(f"Loading LoRA adapter from {cfg.resume_dir}...")
    vla = PeftModel.from_pretrained(
        vla, os.path.join(cfg.resume_dir, "lora_adapter"), is_trainable=False
    )

    llm_dim    = vla.config.text_config.hidden_size
    vocab_size = vla.config.text_config.vocab_size

    def make_projector() -> nn.Module:
        return nn.Sequential(
            nn.Linear(4, llm_dim // 2), nn.GELU(), nn.Linear(llm_dim // 2, llm_dim)
        ).to(device).to(torch.bfloat16)

    past_traj_projector = make_projector()
    full_hist_projector = make_projector()
    action_head = nn.Sequential(
        nn.Linear(llm_dim, llm_dim), nn.GELU(), nn.Linear(llm_dim, 1)
    ).to(device).to(torch.bfloat16)
    decision_head = nn.Linear(llm_dim, vocab_size).to(device).to(torch.bfloat16)
    mem_module = MemoryEnhancementModule(
        llm_dim, cfg.num_mem_tokens, cfg.mem_num_heads
    ).to(device).to(torch.bfloat16)
    align_head = InstructionAlignmentHead(
        llm_dim, cfg.align_num_heads
    ).to(device).to(torch.bfloat16)

    def _load_ckpt(model, path):
        state_dict = torch.load(path, map_location=device)
        cleaned = {(k[len("module."):] if k.startswith("module.") else k): v
                   for k, v in state_dict.items()}
        model.load_state_dict(cleaned)

    _load_ckpt(past_traj_projector, os.path.join(cfg.resume_dir, "past_traj_projector.pt"))
    _load_ckpt(full_hist_projector, os.path.join(cfg.resume_dir, "full_hist_projector.pt"))
    _load_ckpt(action_head,         os.path.join(cfg.resume_dir, "action_head.pt"))
    _load_ckpt(decision_head,       os.path.join(cfg.resume_dir, "decision_head.pt"))
    _load_ckpt(mem_module,          os.path.join(cfg.resume_dir, "mem_module.pt"))
    _load_ckpt(align_head,          os.path.join(cfg.resume_dir, "align_head.pt"))

    for m in (vla, past_traj_projector, full_hist_projector,
              action_head, decision_head, mem_module, align_head):
        m.eval()

    _MODEL.update({
        "device":              device,
        "tokenizer":           tokenizer,
        "pad_token_id":        pad_token_id,
        "image_transform":     processor.image_processor.apply_transform,
        "vla":                 vla,
        "past_traj_projector": past_traj_projector,
        "full_hist_projector": full_hist_projector,
        "action_head":         action_head,
        "decision_head":       decision_head,
        "mem_module":          mem_module,
        "align_head":          align_head,
    })
    print("All models loaded and ready.")


# ── 历史轨迹采样（复刻 LangParkDataset 逻辑）──────────────────────────────────

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


def _compute_history(history_poses: list, curr_pose: dict, mode: str, cfg: ServerConfig) -> torch.Tensor:
    """
    按 mode 从全局历史轨迹采样为局部坐标 Tensor [N, 4]。
    对应 LangParkDataset._get_smart_history 的采样逻辑。
    """
    cx, cy, cyaw = curr_pose["x"], curr_pose["y"], curr_pose["yaw"]
    t = len(history_poses)

    if t == 0:
        return torch.zeros((1, 4), dtype=torch.float32)

    def lc(row):
        return _to_local_coords(row["x"], row["y"], row["yaw"], cx, cy, cyaw)

    history_traj = []

    if mode == "fixed_count":
        if t <= cfg.max_history:
            history_traj = [lc(row) for row in history_poses]
        else:
            indices = np.linspace(0, t - 1, cfg.max_history, dtype=int)
            history_traj = [lc(history_poses[i]) for i in indices]

    elif mode == "fixed_distance":
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

    elif mode == "smart":
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
        raise ValueError(f"Unknown history_mode: {mode}")

    if len(history_traj) == 0:
        return torch.zeros((1, 4), dtype=torch.float32)

    return torch.tensor(np.array(history_traj), dtype=torch.float32)


def _compute_full_history(history_poses: list, curr_pose: dict) -> torch.Tensor:
    """全部历史帧（不抽稀）转局部坐标，供 mem_module 使用。对应 LangParkDataset 的 full_history_traj。"""
    if len(history_poses) == 0:
        return torch.zeros((1, 4), dtype=torch.float32)
    cx, cy, cyaw = curr_pose["x"], curr_pose["y"], curr_pose["yaw"]
    full = [_to_local_coords(p["x"], p["y"], p["yaw"], cx, cy, cyaw) for p in history_poses]
    return torch.tensor(np.array(full), dtype=torch.float32)


# ── Pydantic 请求/响应体 ──────────────────────────────────────────────────────

class GlobalPose(BaseModel):
    """全局坐标系下的一个位姿（yaw 单位：度）"""
    x:   float
    y:   float
    yaw: float


class ParkingSlot(BaseModel):
    """单个检测到的停车位（服务端自行拼接成模型输入字符串）"""
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
    parking_slots:   List[ParkingSlot] = []  # 检测到的停车位列表（空列表 → "[]"）


class Waypoint(BaseModel):
    x:       float
    y:       float
    cos_yaw: float
    sin_yaw: float


class TokenProgress(BaseModel):
    token:     str   # 指令 token 文本
    phase:     int   # 0=done / 1=active / 2=upcoming
    phase_str: str


class PredictResponse(BaseModel):
    trajectory:       List[Waypoint]       # 未来 8 个轨迹点（自车局部坐标）
    decision_id:      int                  # 预测的停车位 ID
    decision_str:     str                  # 解码后的决策字符串
    progress:         List[TokenProgress]  # 语言指令进度（逐 token）
    inference_time_s: float                # 纯模型推理耗时（秒）


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

    # ── 历史轨迹（局部坐标）──────────────────────────────────────────────────
    history_poses_dicts = [{"x": p.x, "y": p.y, "yaw": p.yaw} for p in req.history_poses]
    curr_pose_dict      = {"x": req.curr_pose.x, "y": req.curr_pose.y, "yaw": req.curr_pose.yaw}
    mode = _CFG.history_mode  # 历史采样模式由服务端启动配置决定

    history_tensor      = _compute_history(history_poses_dicts, curr_pose_dict, mode, _CFG)  # smart 采样
    full_history_tensor = _compute_full_history(history_poses_dicts, curr_pose_dict)         # 全量（MEM）

    # ── 停车位文本（服务端拼接，与 LangParkDataset 严格一致，含 .2f 精度）───────
    if len(req.parking_slots) > 0:
        slots_str = "[" + ",".join(
            f"(id={s.id},x={s.x:.2f},y={s.y:.2f})" for s in req.parking_slots
        ) + "]"
    else:
        slots_str = "[]"

    # ── 组装 batch dict（batch_size=1）─────────────────────────────────────
    batch = {
        "pixel_values_front": pixel_values["front"],
        "pixel_values_rear":  pixel_values["rear"],
        "pixel_values_left":  pixel_values["left"],
        "pixel_values_right": pixel_values["right"],
        "history_traj":       [history_tensor],        # 模型内部做 padding/mask
        "full_history_traj":  [full_history_tensor],   # 模型内部做 padding/mask
        "instruction":        [req.instruction],
        "pad_token_id":       pad_token_id,
        # sys_prompt 末尾带 "Instruction:"，instruct_ids 为原始指令（无前缀），
        # 与 LangParkDataset 严格一致，保证 progress 与 instruct token 1:1 对齐。
        "sys_prompt_ids": tokenize(
            "Please predict the future trajectory and select the parking slot id based on "
            "the human instruction, the current four observation images, the history "
            "trajectory, and the detected parking slots. Instruction:"
        ).unsqueeze(0).to(device),
        "instruct_ids": tokenize(req.instruction).unsqueeze(0).to(device),
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


app = FastAPI(title="LangPark VLA Inference Server", version="1.0", lifespan=lifespan)


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
    full_hist_projector = _MODEL["full_hist_projector"]
    action_head         = _MODEL["action_head"]
    decision_head       = _MODEL["decision_head"]
    mem_module          = _MODEL["mem_module"]
    align_head          = _MODEL["align_head"]
    tokenizer           = _MODEL["tokenizer"]

    NUM_ACT = FUTURE_ACTION_WAYPOINTS * ACTION_DIM   # 32
    NUM_MEM = _CFG.num_mem_tokens                     # 16

    t0 = time.time()
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = vla(
                batch,
                past_traj_projector=past_traj_projector,
                full_hist_projector=full_hist_projector,
                mem_module=mem_module,
            )
            last_hidden = outputs.hidden_states[-1]  # [1, seq_len, llm_dim]

            # Tail layout: ... MEM(16) | dec(1) | act(32) | EOS(1)
            decision_hidden = last_hidden[:, -(NUM_ACT + 3), :]
            actions_hidden  = last_hidden[:, -(NUM_ACT + 2):-2, :]
            mem_hidden      = last_hidden[:, -(NUM_ACT + 2 + NUM_MEM):-(NUM_ACT + 2), :]

            pred_decision_logits = decision_head(decision_hidden)           # [1, vocab]
            pred_actions_flat    = action_head(actions_hidden).squeeze(-1)  # [1, 32]
            pred_actions = pred_actions_flat.view(-1, FUTURE_ACTION_WAYPOINTS, ACTION_DIM)  # [1,8,4]

            align_logits = align_head(
                outputs.instruct_emb, mem_hidden, outputs.instruct_mask
            )  # [1, L_inst, 3]
    elapsed = time.time() - t0

    # ── 决策（停车位 id）─────────────────────────────────────────────────────
    pred_dec_token = pred_decision_logits.argmax(dim=-1)[0].item()
    pred_dec_str   = tokenizer.decode(pred_dec_token).replace("<pad>", "").strip()
    try:
        decision_id = int(pred_dec_str)
    except ValueError:
        decision_id = pred_dec_token

    # ── 轨迹 ────────────────────────────────────────────────────────────────
    traj = pred_actions[0].cpu().float().tolist()  # [8, 4]

    # ── 语言进度（逐 token）──────────────────────────────────────────────────
    instruct_ids = batch["instruct_ids"][0]              # [L_inst]
    align_pred   = align_logits.argmax(dim=-1)[0].cpu()  # [L_inst]
    min_len      = min(instruct_ids.shape[0], align_pred.shape[0])
    progress = []
    for i in range(min_len):
        tok   = tokenizer.convert_ids_to_tokens(instruct_ids[i].item())
        phase = int(align_pred[i].item())
        progress.append(TokenProgress(token=tok, phase=phase, phase_str=_PHASE_STR.get(phase, "unknown")))

    return PredictResponse(
        trajectory=[Waypoint(x=p[0], y=p[1], cos_yaw=p[2], sin_yaw=p[3]) for p in traj],
        decision_id=decision_id,
        decision_str=pred_dec_str,
        progress=progress,
        inference_time_s=elapsed,
    )


# ── 入口（使用 draccus，与 test_langpark.py 保持一致）──────────────────────────

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
