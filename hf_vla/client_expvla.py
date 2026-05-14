"""
client_expvla.py
ExpVLA FastAPI 客户端 —— 使用 dummy data 验证服务端联通性。

用法:
    python client_expvla.py                          # 连接 localhost:9999
    python client_expvla.py --server_url http://10.0.0.1:9999
"""
import argparse
import base64
import io
import json
import sys
import time

import numpy as np
import requests
from PIL import Image


# ── Dummy 数据生成 ────────────────────────────────────────────────────────────

def _make_dummy_image_b64(h: int = 224, w: int = 224) -> str:
    """生成随机 RGB 图像并编码为 base64 PNG 字符串"""
    arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _make_dummy_payload() -> dict:
    """构造一个完整的 dummy 推理请求 payload"""
    # 4 张随机图像
    images = {
        "image_front_b64": _make_dummy_image_b64(),
        "image_rear_b64":  _make_dummy_image_b64(),
        "image_left_b64":  _make_dummy_image_b64(),
        "image_right_b64": _make_dummy_image_b64(),
    }

    # 当前位姿（全局坐标）
    curr_pose = {"x": 10.0, "y": 5.0, "yaw": 90.0}

    # 5 个历史全局位姿（模拟车辆直行过来）
    history_poses = [
        {"x": 10.0 - i * 0.5, "y": 5.0, "yaw": 90.0}
        for i in range(5, 0, -1)
    ]

    # 3 个停车位
    parking_slots = [
        {"id": 1, "x": 3.0,  "y": 2.0},
        {"id": 2, "x": 3.0,  "y": 4.5},
        {"id": 3, "x": 3.0,  "y": 7.0},
    ]

    return {
        "instruction":    "Please park the vehicle in slot 2 on the left side.",
        **images,
        "curr_pose":      curr_pose,
        "history_poses":  history_poses,
        "parking_slots":  parking_slots,
    }


# ── 通信函数 ──────────────────────────────────────────────────────────────────

def check_health(server_url: str) -> bool:
    url = f"{server_url.rstrip('/')}/health"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        print(f"[Health] {data}")
        if not data.get("model_loaded"):
            print("[Health] WARNING: server reports model_loaded=False")
            return False
        return True
    except requests.exceptions.ConnectionError:
        print(f"[Health] Cannot reach server at {server_url}. Is it running?")
        return False


def send_predict(server_url: str, payload: dict) -> dict | None:
    url = f"{server_url.rstrip('/')}/predict"
    print(f"\n[Client] POST {url}")
    print(f"[Client] instruction : {payload['instruction']}")
    print(f"[Client] history pts : {len(payload['history_poses'])}")
    print(f"[Client] curr_pose   : {payload['curr_pose']}")
    print(f"[Client] slots       : {[s['id'] for s in payload['parking_slots']]}")

    t0 = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=120)
    except requests.exceptions.ConnectionError as e:
        print(f"[Client] Connection error: {e}")
        return None
    round_trip = time.time() - t0

    if resp.status_code != 200:
        print(f"[Client] ERROR {resp.status_code}: {resp.text}")
        return None

    result = resp.json()
    print(f"\n[Client] ===== Response =====")
    print(f"  HTTP round-trip time : {round_trip:.3f} s")
    print(f"  Server inference time: {result['inference_time_s']:.4f} s")
    print(f"  Decision ID  : {result['decision_id']}")
    print(f"  Decision Str : '{result['decision_str']}'")
    print(f"  Trajectory ({len(result['trajectory'])} waypoints):")
    for i, wp in enumerate(result["trajectory"]):
        print(
            f"    [{i}] x={wp['x']:+.4f}  y={wp['y']:+.4f}  "
            f"cos={wp['cos_yaw']:+.4f}  sin={wp['sin_yaw']:+.4f}"
        )
    return result


# ── 主入口 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ExpVLA FastAPI Client (dummy data test)")
    parser.add_argument(
        "--server_url", type=str, default="http://localhost:9999",
        help="服务端地址，默认 http://localhost:9999"
    )
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="重复推理次数（用于测试吞吐量）"
    )
    args = parser.parse_args()

    # 1. 健康检查
    if not check_health(args.server_url):
        sys.exit(1)

    # 2. 构造 dummy payload
    payload = _make_dummy_payload()

    # 3. 发送推理请求
    for i in range(args.repeat):
        if args.repeat > 1:
            print(f"\n{'='*40} Run {i+1}/{args.repeat} {'='*40}")
        result = send_predict(args.server_url, payload)
        if result is None:
            sys.exit(1)

    print("\n[Client] Done.")


if __name__ == "__main__":
    main()
