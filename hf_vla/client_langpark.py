"""
client_langpark.py
LangPark VLA 服务端的测试客户端：构造假数据，验证服务端与客户端能正常通信、推理。

用法:
    python client_langpark.py
    python client_langpark.py --url http://127.0.0.1:9999
"""
import io
import base64
import argparse
import numpy as np
from PIL import Image

import requests


def _make_fake_image_b64(color: tuple, size: int = 224) -> str:
    """生成一张纯色 + 噪声的假图片，编码为 base64 JPEG。"""
    arr = np.random.randint(0, 40, (size, size, 3), dtype=np.uint8)
    arr[:, :, 0] = np.clip(arr[:, :, 0] + color[0], 0, 255)
    arr[:, :, 1] = np.clip(arr[:, :, 1] + color[1], 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] + color[2], 0, 255)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _make_fake_history(n: int = 20) -> list:
    """生成一段假的全局历史轨迹：先直行后转弯。"""
    poses = []
    x, y, yaw = 0.0, 0.0, 0.0
    for i in range(n):
        # 前半段直行，后半段缓慢转弯
        if i < n // 2:
            x += 0.4
            yaw += 0.0
        else:
            x += 0.3
            y += 0.1
            yaw += 6.0  # 度
        poses.append({"x": round(x, 3), "y": round(y, 3), "yaw": round(yaw, 3)})
    return poses


def build_fake_request() -> dict:
    history = _make_fake_history(20)
    # 当前位姿 = 历史最后一帧再往前一点
    curr = {"x": history[-1]["x"] + 0.3,
            "y": history[-1]["y"] + 0.1,
            "yaw": history[-1]["yaw"] + 6.0}

    # 停车位列表：结构化数据，服务端自行拼接成模型输入字符串
    parking_slots = [
        {"id": 1, "x": 5.20, "y": 2.30},
        {"id": 2, "x": 6.10, "y": -1.50},
    ]

    return {
        "instruction": "Park into the empty slot on your right.",
        "image_front_b64": _make_fake_image_b64((180, 60, 60)),
        "image_rear_b64":  _make_fake_image_b64((60, 180, 60)),
        "image_left_b64":  _make_fake_image_b64((60, 60, 180)),
        "image_right_b64": _make_fake_image_b64((180, 180, 60)),
        "history_poses": history,
        "curr_pose": curr,
        "parking_slots": parking_slots,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://127.0.0.1:9999")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    # ── 1. 健康检查 ──────────────────────────────────────────────────────────
    print(f"[1/2] GET {args.url}/health")
    try:
        h = requests.get(f"{args.url}/health", timeout=args.timeout)
        h.raise_for_status()
        print("      health:", h.json())
    except Exception as e:
        print("      health check failed:", e)
        return

    # ── 2. 推理请求 ──────────────────────────────────────────────────────────
    payload = build_fake_request()
    print(f"\n[2/2] POST {args.url}/predict")
    print(f"      instruction : {payload['instruction']}")
    print(f"      history len : {len(payload['history_poses'])}")
    print(f"      parking_slots: {payload['parking_slots']}")

    try:
        r = requests.post(f"{args.url}/predict", json=payload, timeout=args.timeout)
        r.raise_for_status()
    except Exception as e:
        print("      predict failed:", e)
        if "r" in dir() and hasattr(r, "text"):
            print("      response:", r.text)
        return

    resp = r.json()

    # ── 结果展示 ─────────────────────────────────────────────────────────────
    print("\n========== Server Response ==========")
    print(f"decision_id      : {resp['decision_id']}")
    print(f"decision_str     : {resp['decision_str']!r}")
    print(f"inference_time_s : {resp['inference_time_s']:.4f}")

    print(f"\nTrajectory ({len(resp['trajectory'])} waypoints):")
    for i, wp in enumerate(resp["trajectory"]):
        yaw = np.degrees(np.arctan2(wp["sin_yaw"], wp["cos_yaw"]))
        print(f"  [{i}] x={wp['x']:+.3f}  y={wp['y']:+.3f}  yaw={yaw:+.2f} deg")

    print(f"\nLanguage progress ({len(resp['progress'])} tokens):")
    for tp in resp["progress"]:
        print(f"  {tp['token']:<18} -> {tp['phase']} ({tp['phase_str']})")

    print("\nOK: client <-> server communication verified.")


if __name__ == "__main__":
    main()
