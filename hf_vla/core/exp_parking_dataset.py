import os
import json
import random
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence

def to_local_coords(target_x, target_y, target_yaw, curr_x, curr_y, curr_yaw):
    """全局坐标转换到自车局部坐标系"""
    curr_yaw = np.deg2rad(curr_yaw)
    target_yaw = np.deg2rad(target_yaw)
    dx = target_x - curr_x
    dy = target_y - curr_y
    local_x = dx * np.cos(curr_yaw) + dy * np.sin(curr_yaw)
    local_y = -dx * np.sin(curr_yaw) + dy * np.cos(curr_yaw)
    local_yaw = target_yaw - curr_yaw
    return np.array([local_x, local_y, np.cos(local_yaw), np.sin(local_yaw)])

def find_traj_dirs(data_root, max_depth=2):
    """从 data_root 向下查找所有轨迹目录（含 decision.txt），深度上限 max_depth。

    兼容：
      data_root/decision.txt                  (0 层)
      data_root/<traj>/decision.txt           (1 层)
      data_root/<group>/<traj>/decision.txt   (2 层)
    一旦某目录被判定为轨迹目录，就不再向其内部递归。
    """
    traj_dirs = []

    def _walk(cur_dir, depth):
        if os.path.exists(os.path.join(cur_dir, "decision.txt")):
            traj_dirs.append(cur_dir)
            return
        if depth >= max_depth:
            return
        for d in sorted(os.listdir(cur_dir)):
            sub = os.path.join(cur_dir, d)
            if os.path.isdir(sub):
                _walk(sub, depth + 1)

    _walk(data_root, 0)
    return sorted(traj_dirs)

class ExpParkingDataset(Dataset):
    def __init__(self, data_root, tokenizer, image_transform, max_history=8, future_steps=8,
                 history_mode='fixed_count', distance_interval=0.5, turn_yaw_thresh=5.0, turn_dense_interval=0.2):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_history = max_history
        self.future_steps = future_steps
        self.history_mode = history_mode
        self.distance_interval = distance_interval
        self.turn_yaw_thresh = turn_yaw_thresh
        self.turn_dense_interval = turn_dense_interval

        self.samples = []
        self._build_dataset()

    def _build_dataset(self):
        traj_dirs = find_traj_dirs(self.data_root)

        for traj_dir in traj_dirs:
            with open(os.path.join(traj_dir, "decision.txt"), 'r') as f:
                lines = f.readlines()
                instruction = lines[0].strip()
                true_decision_id = int(lines[1].strip())

            odom_df = pd.read_csv(os.path.join(traj_dir, "odom.csv"))
            with open(os.path.join(traj_dir, "parking_slots.txt"), 'r') as f:
                slots_data = [json.loads(line) for line in f.readlines()]

            num_frames = len(odom_df)
            for t in range(num_frames):
                curr_slots = slots_data[t]
                visible_ids = [slot['id'] for slot in curr_slots]
                decision_id = true_decision_id if true_decision_id in visible_ids else 0

                self.samples.append({
                    'traj_dir': traj_dir,
                    't': t,
                    'instruction': instruction,
                    'decision_id': decision_id,
                    'slots': curr_slots,
                    'odom': odom_df
                })

        print(f"Dataset built: {len(self.samples)} samples from {len(traj_dirs)} trajectories.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        t = sample['t']
        odom = sample['odom']
        traj_dir = sample['traj_dir']
        
        # 1. 四视角图片提取
        images = {}
        for view in ['front', 'rear', 'left', 'right']:
            # 加1是因为图片文件名是从 000001.png 开始的，而 t 是从 0 开始的
            img_path = os.path.join(traj_dir, "images", view, f"{t+1:06d}.png")
            img = Image.open(img_path).convert("RGB")
            images[view] = self.image_transform(img)
            
        # 2. 历史轨迹 (Memory)
        curr_state = odom.iloc[t]
        hist_df = odom.iloc[0:t]
        history_traj = []
        
        if t > 0:
            if self.history_mode == 'fixed_count':
                if t <= self.max_history:
                    history_traj = [to_local_coords(row['x'], row['y'], row['yaw'], curr_state['x'], curr_state['y'], curr_state['yaw']) for _, row in hist_df.iterrows()]
                else:
                    indices = np.linspace(0, t - 1, self.max_history, dtype=int)
                    history_traj = [to_local_coords(row['x'], row['y'], row['yaw'], curr_state['x'], curr_state['y'], curr_state['yaw']) for _, row in odom.iloc[indices].iterrows()]
            
            elif self.history_mode == 'fixed_distance':
                last_idx = -1
                for i in range(t):
                    if i == 0 or i == t - 1:
                        history_traj.append(to_local_coords(hist_df.iloc[i]['x'], hist_df.iloc[i]['y'], hist_df.iloc[i]['yaw'], curr_state['x'], curr_state['y'], curr_state['yaw']))
                        last_idx = i
                    else:
                        dist = np.hypot(hist_df.iloc[i]['x'] - hist_df.iloc[last_idx]['x'], hist_df.iloc[i]['y'] - hist_df.iloc[last_idx]['y'])
                        if dist >= self.distance_interval:
                            history_traj.append(to_local_coords(hist_df.iloc[i]['x'], hist_df.iloc[i]['y'], hist_df.iloc[i]['yaw'], curr_state['x'], curr_state['y'], curr_state['yaw']))
                            last_idx = i

            elif self.history_mode == 'full':
                # 全量历史：0..t-1 每一帧都转局部坐标，不做任何抽稀
                history_traj = [to_local_coords(row['x'], row['y'], row['yaw'], curr_state['x'], curr_state['y'], curr_state['yaw']) for _, row in hist_df.iterrows()]

            elif self.history_mode == 'smart':
                states = []
                for i in range(t):
                    start = max(0, i - 2) # 以当前帧为中心，向前后各取2帧计算转向状态
                    end = min(t - 1, i + 2)
                    yaw_diff = abs(hist_df.iloc[end]['yaw'] - hist_df.iloc[start]['yaw']) % 360
                    if yaw_diff > 180:
                        yaw_diff = 360 - yaw_diff
                    states.append('Turn' if yaw_diff > self.turn_yaw_thresh else 'Straight')

                last_idx = -1
                for i in range(t):
                    row = hist_df.iloc[i]
                    pt = to_local_coords(row['x'], row['y'], row['yaw'], curr_state['x'], curr_state['y'], curr_state['yaw'])
                    if i == 0 or i == t - 1:
                        history_traj.append(pt)
                        last_idx = i
                    else:
                        if states[i] == 'Straight':
                            # 保留直行的首尾边界点
                            if states[i] != states[i-1] or (i + 1 < t and states[i] != states[i+1]):
                                history_traj.append(pt)
                                last_idx = i
                        else:
                            dist = np.hypot(row['x'] - hist_df.iloc[last_idx]['x'], row['y'] - hist_df.iloc[last_idx]['y'])
                            if dist >= self.turn_dense_interval:
                                history_traj.append(pt)
                                last_idx = i

        if len(history_traj) == 0:
            history_tensor = torch.zeros((1, 4), dtype=torch.float32)
        else:
            history_tensor = torch.tensor(np.array(history_traj), dtype=torch.float32)
        
        # 3. 未来轨迹预测 (Action)
        future_df = odom.iloc[t+1 : t+1+self.future_steps]
        future_traj = [to_local_coords(row['x'], row['y'], row['yaw'], curr_state['x'], curr_state['y'], curr_state['yaw']) for _, row in future_df.iterrows()]
        
        while len(future_traj) < self.future_steps:
            if len(future_traj) > 0:
                future_traj.append(future_traj[-1])
            else:
                future_traj.append(np.zeros(4))
                
        action_tensor = torch.tensor(np.array(future_traj), dtype=torch.float32)
        
        # 4. 文本提示生成
        slots_str = "[]"
        if len(sample['slots']) > 0:
            slots_str = "Detected parking slots:[" + ",".join([f"(id={s['id']},x={s['x']:.2f},y={s['y']:.2f})" for s in sample['slots']]) + "]"
            
        decision_token_id = self.tokenizer.encode(str(sample['decision_id']), add_special_tokens=False)[-1]
        
        def tokenize(text):
            return torch.tensor(self.tokenizer.encode(text, add_special_tokens=False), dtype=torch.long)
            
        return {
            'pixel_values_front': images['front'], 'pixel_values_rear': images['rear'],
            'pixel_values_left': images['left'], 'pixel_values_right': images['right'],
            'history_traj': history_tensor, 'action_gt': action_tensor,
            'decision_gt': torch.tensor(decision_token_id, dtype=torch.long),
            'instruction': sample['instruction'],
            
            'sys_prompt_ids': tokenize("Please predict the future trajectory and select the parking slot id based on the human instruction, the current four observation images, the history trajectory, and the detected parking slots."),
            'instruct_ids': tokenize(f"Instruction: {sample['instruction']}"),
            'p_front_ids': tokenize("Front view:"),
            'p_rear_ids': tokenize("Rear view:"),
            'p_left_ids': tokenize("Left view:"),
            'p_right_ids': tokenize("Right view:"),
            'p_hist_ids': tokenize("History trajectory:"),
            'p_slots_ids': tokenize(f"Detected parking slot information: {slots_str}"),
            # 注：已移除决策占位提示和动作占位提示，将直接拼接全0 Embedding
        }

class BalancedBatchSampler(Sampler):
    def __init__(self, pos_indices, neg_indices, num_neg_per_batch):
        self.pos_indices = list(pos_indices)
        self.neg_indices = list(neg_indices)
        self.num_neg_per_batch = num_neg_per_batch
        
        # 一个epoch的定义：所有负样本被遍历一次
        self.num_batches = (len(self.neg_indices) + self.num_neg_per_batch - 1) // self.num_neg_per_batch

    def __iter__(self):
        # 每个epoch开始前打乱负样本
        neg_indices = self.neg_indices.copy()
        random.shuffle(neg_indices)
        
        # 打乱正样本
        pos_indices = self.pos_indices.copy()
        random.shuffle(pos_indices)
        pos_ptr = 0
        
        for i in range(self.num_batches):
            # 取出当前批次的负样本
            start_idx = i * self.num_neg_per_batch
            end_idx = min(start_idx + self.num_neg_per_batch, len(neg_indices))
            batch_neg = neg_indices[start_idx:end_idx]
            
            # 取出1个正样本
            batch_pos = [pos_indices[pos_ptr]]
            pos_ptr += 1
            # 正样本数量较少，如果用完则重新打乱并循环使用
            if pos_ptr >= len(pos_indices):
                random.shuffle(pos_indices)
                pos_ptr = 0
            
            # 组合并打乱当前batch内的数据 (负样本 + 1个正样本)
            batch = batch_pos + batch_neg
            random.shuffle(batch)
            yield batch

    def __len__(self): 
        return self.num_batches


def custom_collate_fn(batch, pad_token_id=0):
    """支持变长文本序列的合并"""
    collated = {}
    for key in batch[0].keys():
        if key == 'instruction':
            collated[key] = [item[key] for item in batch]
        elif key.endswith('_ids'):
            # pad_sequence 默认模式即为右填充 (Right Padding)
            collated[key] = pad_sequence([item[key] for item in batch], batch_first=True, padding_value=pad_token_id)
        elif key == 'history_traj':
            # 返回原始List，将Padding与Mask的计算完全移交给 Model 的前向传播过程
            collated[key] = [item[key] for item in batch]
        else:
            collated[key] = torch.stack([item[key] for item in batch], dim=0)
            
    # 将 pad_token_id 存入 collated，保障模型能精准计算 text 的 attention_mask
    collated['pad_token_id'] = pad_token_id
    return collated


# ============================================================
# ClassicVLA: 文本自回归版数据集
# ============================================================

def traj_to_text(traj: np.ndarray) -> str:
    """将 [N,4] 轨迹数组转成 '[-0.00,0.00,1.00,0.00],...' 格式文本"""
    return ",".join(
        f"[{x:.2f},{y:.2f},{c:.2f},{s:.2f}]" for x, y, c, s in traj
    )


def build_output_json(decision_id: int, future_traj: np.ndarray) -> str:
    """构造输出 JSON 字符串，保留两位小数"""
    traj_str = traj_to_text(future_traj)
    return '{' + f'"decision":{decision_id},"trajectory":[{traj_str}]' + '}'


class ClassicVLADataset(Dataset):
    """ClassicVLA 训练集：历史轨迹文本化，输出为自回归 JSON"""
    def __init__(self, data_root, tokenizer, image_transform, max_history=8, future_steps=8,
                 history_mode='fixed_count', distance_interval=0.5, turn_yaw_thresh=5.0, turn_dense_interval=0.2):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_history = max_history
        self.future_steps = future_steps
        self.history_mode = history_mode
        self.distance_interval = distance_interval
        self.turn_yaw_thresh = turn_yaw_thresh
        self.turn_dense_interval = turn_dense_interval

        self.samples = []
        self._build_dataset()

    def _build_dataset(self):
        traj_dirs = find_traj_dirs(self.data_root)
        for traj_dir in traj_dirs:
            with open(os.path.join(traj_dir, "decision.txt"), 'r') as f:
                lines = f.readlines()
                instruction = lines[0].strip()
                true_decision_id = int(lines[1].strip())

            odom_df = pd.read_csv(os.path.join(traj_dir, "odom.csv"))
            with open(os.path.join(traj_dir, "parking_slots.txt"), 'r') as f:
                slots_data = [json.loads(line) for line in f.readlines()]

            num_frames = len(odom_df)
            for t in range(num_frames):
                curr_slots = slots_data[t]
                visible_ids = [slot['id'] for slot in curr_slots]
                # 与 LangPark 一致：当前帧车位列表里出现最终决策 id 即打 gt 标签，否则为 0
                decision_id = true_decision_id if true_decision_id in visible_ids else 0

                sample = {
                    'traj_dir': traj_dir,
                    't': t,
                    'instruction': instruction,
                    'decision_id': decision_id,
                    'slots': curr_slots,
                    'odom': odom_df
                }
                self.samples.append(sample)
        print(f"ClassicVLADataset built: {len(self.samples)} samples from {len(traj_dirs)} trajectories.")

    def __len__(self):
        return len(self.samples)

    def _get_history_traj(self, odom, t, curr_state):
        hist_df = odom.iloc[0:t]
        history_traj = []
        if t > 0:
            if self.history_mode == 'fixed_count':
                if t <= self.max_history:
                    history_traj = [to_local_coords(row['x'], row['y'], row['yaw'],
                                                    curr_state['x'], curr_state['y'], curr_state['yaw'])
                                    for _, row in hist_df.iterrows()]
                else:
                    indices = np.linspace(0, t - 1, self.max_history, dtype=int)
                    history_traj = [to_local_coords(row['x'], row['y'], row['yaw'],
                                                    curr_state['x'], curr_state['y'], curr_state['yaw'])
                                    for _, row in odom.iloc[indices].iterrows()]
            elif self.history_mode == 'fixed_distance':
                last_idx = -1
                for i in range(t):
                    if i == 0 or i == t - 1:
                        history_traj.append(to_local_coords(hist_df.iloc[i]['x'], hist_df.iloc[i]['y'],
                                                            hist_df.iloc[i]['yaw'], curr_state['x'],
                                                            curr_state['y'], curr_state['yaw']))
                        last_idx = i
                    else:
                        dist = np.hypot(hist_df.iloc[i]['x'] - hist_df.iloc[last_idx]['x'],
                                        hist_df.iloc[i]['y'] - hist_df.iloc[last_idx]['y'])
                        if dist >= self.distance_interval:
                            history_traj.append(to_local_coords(hist_df.iloc[i]['x'], hist_df.iloc[i]['y'],
                                                                hist_df.iloc[i]['yaw'], curr_state['x'],
                                                                curr_state['y'], curr_state['yaw']))
                            last_idx = i
            elif self.history_mode == 'smart':
                states = []
                for i in range(t):
                    start = max(0, i - 2)
                    end = min(t - 1, i + 2)
                    yaw_diff = abs(hist_df.iloc[end]['yaw'] - hist_df.iloc[start]['yaw']) % 360
                    if yaw_diff > 180:
                        yaw_diff = 360 - yaw_diff
                    states.append('Turn' if yaw_diff > self.turn_yaw_thresh else 'Straight')
                last_idx = -1
                for i in range(t):
                    row = hist_df.iloc[i]
                    pt = to_local_coords(row['x'], row['y'], row['yaw'],
                                        curr_state['x'], curr_state['y'], curr_state['yaw'])
                    if i == 0 or i == t - 1:
                        history_traj.append(pt)
                        last_idx = i
                    else:
                        if states[i] == 'Straight':
                            if states[i] != states[i - 1] or (i + 1 < t and states[i] != states[i + 1]):
                                history_traj.append(pt)
                                last_idx = i
                        else:
                            dist = np.hypot(row['x'] - hist_df.iloc[last_idx]['x'],
                                            row['y'] - hist_df.iloc[last_idx]['y'])
                            if dist >= self.turn_dense_interval:
                                history_traj.append(pt)
                                last_idx = i
        if len(history_traj) == 0:
            history_traj = [np.zeros(4)]
        return np.array(history_traj)  # [N, 4]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        t = sample['t']
        odom = sample['odom']
        traj_dir = sample['traj_dir']

        images = {}
        for view in ['front', 'rear', 'left', 'right']:
            img_path = os.path.join(traj_dir, "images", view, f"{t+1:06d}.png")
            img = Image.open(img_path).convert("RGB")
            images[view] = self.image_transform(img)

        curr_state = odom.iloc[t]
        history_arr = self._get_history_traj(odom, t, curr_state)  # [N, 4]
        # print("history_arr shape:", history_arr.shape)  # Debug 输出，验证历史轨迹数组形状

        # 未来轨迹 (GT) —— 以文本 token 形式表示，不使用浮点张量
        future_df = odom.iloc[t + 1: t + 1 + self.future_steps]
        future_traj = [to_local_coords(row['x'], row['y'], row['yaw'],
                                       curr_state['x'], curr_state['y'], curr_state['yaw'])
                       for _, row in future_df.iterrows()]
        while len(future_traj) < self.future_steps:
            future_traj.append(future_traj[-1] if future_traj else np.zeros(4))
        future_arr = np.array(future_traj)  # [future_steps, 4]，仅用于构建 output_json

        # 停车槽文本
        slots_str = "[]"
        if len(sample['slots']) > 0:
            slots_str = "[" + ",".join(
                [f"(id={s['id']},x={s['x']:.2f},y={s['y']:.2f})" for s in sample['slots']]) + "]"

        # 历史轨迹文本化
        hist_text = traj_to_text(history_arr)

        def tokenize(text):
            return torch.tensor(self.tokenizer.encode(text, add_special_tokens=False), dtype=torch.long)

        # 输出 JSON 目标
        output_json = build_output_json(sample['decision_id'], future_arr)
        # print(f"Sample {idx} Output JSON: {output_json}")  # Debug 输出，验证 JSON 格式
        output_ids = tokenize(output_json)
        # EOS token
        eos_id = torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)

        # prompt 各分段 ids（不含 BOS，BOS 在 model forward 中添加）
        sys_prompt_ids = tokenize("Please predict the future trajectory and select the parking slot id based on the human instruction, the current four observation images, the history trajectory, and the detected parking slots.")
        instruct_ids = tokenize(f"Instruction: {sample['instruction']}")
        p_front_ids = tokenize("Front view:")
        p_rear_ids = tokenize("Rear view:")
        p_left_ids = tokenize("Left view:")
        p_right_ids = tokenize("Right view:")
        p_hist_ids = tokenize(f"History trajectory:{hist_text}")
        p_slots_ids = tokenize(f"Detected parking slot information: {slots_str}")
        p_answer_ids = tokenize("Answer: ")

        # labels：prompt 部分用 IGNORE_INDEX 掩盖，只对输出 JSON + EOS 计算 loss
        from .constants import IGNORE_INDEX
        prompt_len = (1  # BOS（model forward 里加）
                      + sys_prompt_ids.shape[0]
                      + instruct_ids.shape[0]
                      + p_front_ids.shape[0]
                      # 图像 patch token 数量在 model forward 里才知道，这里用 -1 标记，collate 时不用处理
                      # labels 中图像位置会在 model.forward 里用 IGNORE_INDEX 填充
                      )
        # 注：labels 的完整拼接在 model.forward 中完成，dataset 只需返回各段 ids 和 output_ids

        return {
            'pixel_values_front': images['front'],
            'pixel_values_rear': images['rear'],
            'pixel_values_left': images['left'],
            'pixel_values_right': images['right'],
            'instruction': sample['instruction'],
            'sys_prompt_ids': sys_prompt_ids,
            'instruct_ids': instruct_ids,
            'p_front_ids': p_front_ids,
            'p_rear_ids': p_rear_ids,
            'p_left_ids': p_left_ids,
            'p_right_ids': p_right_ids,
            'p_hist_ids': p_hist_ids,
            'p_slots_ids': p_slots_ids,
            'p_answer_ids': p_answer_ids,
            'output_ids': output_ids,
            'eos_ids': eos_id,
        }


class SingleTrajClassicTestDataset(Dataset):
    """ClassicVLA 推理用：顺序读取单条轨迹"""
    def __init__(self, traj_dir, tokenizer, image_transform, max_history=8, future_steps=8,
                 history_mode='fixed_count', distance_interval=0.5, turn_yaw_thresh=5.0, turn_dense_interval=0.1):
        self.traj_dir = traj_dir
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_history = max_history
        self.future_steps = future_steps
        self.history_mode = history_mode
        self.distance_interval = distance_interval
        self.turn_yaw_thresh = turn_yaw_thresh
        self.turn_dense_interval = turn_dense_interval
        self.samples = []
        self._build_dataset()

    def _build_dataset(self):
        with open(os.path.join(self.traj_dir, "decision.txt"), 'r') as f:
            lines = f.readlines()
            instruction = lines[0].strip()
            true_decision_id = int(lines[1].strip())

        odom_df = pd.read_csv(os.path.join(self.traj_dir, "odom.csv"))
        with open(os.path.join(self.traj_dir, "parking_slots.txt"), 'r') as f:
            slots_data = [json.loads(line) for line in f.readlines()]

        num_frames = len(odom_df)
        for t in range(num_frames):
            curr_slots = slots_data[t]
            visible_ids = [slot['id'] for slot in curr_slots]
            # 与 LangPark / 训练集一致：可见即打 gt 标签，否则为 0
            decision_id = true_decision_id if true_decision_id in visible_ids else 0
            self.samples.append({
                'traj_dir': self.traj_dir,
                't': t,
                'instruction': instruction,
                'decision_id': decision_id,
                'slots': curr_slots,
                'odom': odom_df
            })

    def __len__(self):
        return len(self.samples)

    def _get_history_traj(self, odom, t, curr_state):
        """与 ClassicVLADataset 相同的历史轨迹提取逻辑"""
        hist_df = odom.iloc[0:t]
        history_traj = []
        if t > 0:
            if self.history_mode == 'fixed_count':
                if t <= self.max_history:
                    history_traj = [to_local_coords(row['x'], row['y'], row['yaw'],
                                                    curr_state['x'], curr_state['y'], curr_state['yaw'])
                                    for _, row in hist_df.iterrows()]
                else:
                    indices = np.linspace(0, t - 1, self.max_history, dtype=int)
                    history_traj = [to_local_coords(row['x'], row['y'], row['yaw'],
                                                    curr_state['x'], curr_state['y'], curr_state['yaw'])
                                    for _, row in odom.iloc[indices].iterrows()]
            elif self.history_mode == 'fixed_distance':
                last_idx = -1
                for i in range(t):
                    if i == 0 or i == t - 1:
                        history_traj.append(to_local_coords(hist_df.iloc[i]['x'], hist_df.iloc[i]['y'],
                                                            hist_df.iloc[i]['yaw'], curr_state['x'],
                                                            curr_state['y'], curr_state['yaw']))
                        last_idx = i
                    else:
                        dist = np.hypot(hist_df.iloc[i]['x'] - hist_df.iloc[last_idx]['x'],
                                        hist_df.iloc[i]['y'] - hist_df.iloc[last_idx]['y'])
                        if dist >= self.distance_interval:
                            history_traj.append(to_local_coords(hist_df.iloc[i]['x'], hist_df.iloc[i]['y'],
                                                                hist_df.iloc[i]['yaw'], curr_state['x'],
                                                                curr_state['y'], curr_state['yaw']))
                            last_idx = i
            elif self.history_mode == 'smart':
                states = []
                for i in range(t):
                    start = max(0, i - 2)
                    end = min(t - 1, i + 2)
                    yaw_diff = abs(hist_df.iloc[end]['yaw'] - hist_df.iloc[start]['yaw']) % 360
                    if yaw_diff > 180:
                        yaw_diff = 360 - yaw_diff
                    states.append('Turn' if yaw_diff > self.turn_yaw_thresh else 'Straight')
                last_idx = -1
                for i in range(t):
                    row = hist_df.iloc[i]
                    pt = to_local_coords(row['x'], row['y'], row['yaw'],
                                        curr_state['x'], curr_state['y'], curr_state['yaw'])
                    if i == 0 or i == t - 1:
                        history_traj.append(pt)
                        last_idx = i
                    else:
                        if states[i] == 'Straight':
                            if states[i] != states[i - 1] or (i + 1 < t and states[i] != states[i + 1]):
                                history_traj.append(pt)
                                last_idx = i
                        else:
                            dist = np.hypot(row['x'] - hist_df.iloc[last_idx]['x'],
                                            row['y'] - hist_df.iloc[last_idx]['y'])
                            if dist >= self.turn_dense_interval:
                                history_traj.append(pt)
                                last_idx = i
        if len(history_traj) == 0:
            history_traj = [np.zeros(4)]
        return np.array(history_traj)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        t = sample['t']
        odom = sample['odom']
        traj_dir = sample['traj_dir']

        images = {}
        for view in ['front', 'rear', 'left', 'right']:
            img_path = os.path.join(traj_dir, "images", view, f"{t+1:06d}.png")
            img = Image.open(img_path).convert("RGB")
            images[view] = self.image_transform(img)

        curr_state = odom.iloc[t]
        history_arr = self._get_history_traj(odom, t, curr_state)

        future_df = odom.iloc[t + 1: t + 1 + self.future_steps]
        future_traj = [to_local_coords(row['x'], row['y'], row['yaw'],
                                       curr_state['x'], curr_state['y'], curr_state['yaw'])
                       for _, row in future_df.iterrows()]
        while len(future_traj) < self.future_steps:
            future_traj.append(future_traj[-1] if future_traj else np.zeros(4))
        action_tensor = torch.tensor(np.array(future_traj), dtype=torch.float32)

        slots_str = "[]"
        if len(sample['slots']) > 0:
            slots_str = "[" + ",".join(
                [f"(id={s['id']},x={s['x']:.2f},y={s['y']:.2f})" for s in sample['slots']]) + "]"

        hist_text = traj_to_text(history_arr)

        def tokenize(text):
            return torch.tensor(self.tokenizer.encode(text, add_special_tokens=False), dtype=torch.long)

        return {
            'pixel_values_front': images['front'],
            'pixel_values_rear': images['rear'],
            'pixel_values_left': images['left'],
            'pixel_values_right': images['right'],
            'action_gt': action_tensor,
            'decision_gt': torch.tensor(sample['decision_id'], dtype=torch.long),
            'instruction': sample['instruction'],
            'sys_prompt_ids': tokenize("Please predict the future trajectory and select the parking slot id based on the human instruction, the current four observation images, the history trajectory, and the detected parking slots."),
            'instruct_ids': tokenize(f"Instruction: {sample['instruction']}"),
            'p_front_ids': tokenize("Front view:"),
            'p_rear_ids': tokenize("Rear view:"),
            'p_left_ids': tokenize("Left view:"),
            'p_right_ids': tokenize("Right view:"),
            'p_hist_ids': tokenize(f"History trajectory:{hist_text}"),
            'p_slots_ids': tokenize(f"Detected parking slot information: {slots_str}"),
            'p_answer_ids': tokenize("Answer: "),
        }


class MultiTrajClassicTestDataset(Dataset):
    """ClassicVLA 推理用：跨所有轨迹展开为统一样本集，支持批量 / 多卡并行测试。

    与 ClassicVLADataset 的历史/未来/slots 处理逻辑保持一致；额外返回 action_gt、
    decision_gt(int) 以及元数据 traj_name / step_idx / is_last（是否轨迹末帧），
    供测试时按轨迹聚合指标、计算 Parking Success。
    """
    def __init__(self, data_root, tokenizer, image_transform, max_history=8, future_steps=8,
                 history_mode='fixed_count', distance_interval=0.5, turn_yaw_thresh=5.0, turn_dense_interval=0.1):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_history = max_history
        self.future_steps = future_steps
        self.history_mode = history_mode
        self.distance_interval = distance_interval
        self.turn_yaw_thresh = turn_yaw_thresh
        self.turn_dense_interval = turn_dense_interval
        self.samples = []
        self._build_dataset()

    def _build_dataset(self):
        traj_dirs = find_traj_dirs(self.data_root)
        for traj_dir in traj_dirs:
            with open(os.path.join(traj_dir, "decision.txt"), 'r') as f:
                lines = f.readlines()
                instruction = lines[0].strip()
                true_decision_id = int(lines[1].strip())

            odom_df = pd.read_csv(os.path.join(traj_dir, "odom.csv"))
            with open(os.path.join(traj_dir, "parking_slots.txt"), 'r') as f:
                slots_data = [json.loads(line) for line in f.readlines()]

            traj_name = os.path.basename(traj_dir)
            num_frames = len(odom_df)
            for t in range(num_frames):
                curr_slots = slots_data[t]
                visible_ids = [slot['id'] for slot in curr_slots]
                decision_id = true_decision_id if true_decision_id in visible_ids else 0
                self.samples.append({
                    'traj_dir': traj_dir,
                    'traj_name': traj_name,
                    't': t,
                    'is_last': (t == num_frames - 1),
                    'instruction': instruction,
                    'decision_id': decision_id,
                    'slots': curr_slots,
                    'odom': odom_df
                })
        print(f"MultiTrajClassicTestDataset built: {len(self.samples)} samples from {len(traj_dirs)} trajectories.")

    def __len__(self):
        return len(self.samples)

    # 历史轨迹提取逻辑与 ClassicVLADataset 完全一致
    _get_history_traj = ClassicVLADataset._get_history_traj

    def __getitem__(self, idx):
        sample = self.samples[idx]
        t = sample['t']
        odom = sample['odom']
        traj_dir = sample['traj_dir']

        images = {}
        for view in ['front', 'rear', 'left', 'right']:
            img_path = os.path.join(traj_dir, "images", view, f"{t+1:06d}.png")
            img = Image.open(img_path).convert("RGB")
            images[view] = self.image_transform(img)

        curr_state = odom.iloc[t]
        history_arr = self._get_history_traj(odom, t, curr_state)

        future_df = odom.iloc[t + 1: t + 1 + self.future_steps]
        future_traj = [to_local_coords(row['x'], row['y'], row['yaw'],
                                       curr_state['x'], curr_state['y'], curr_state['yaw'])
                       for _, row in future_df.iterrows()]
        while len(future_traj) < self.future_steps:
            future_traj.append(future_traj[-1] if future_traj else np.zeros(4))
        action_tensor = torch.tensor(np.array(future_traj), dtype=torch.float32)

        slots_str = "[]"
        if len(sample['slots']) > 0:
            slots_str = "[" + ",".join(
                [f"(id={s['id']},x={s['x']:.2f},y={s['y']:.2f})" for s in sample['slots']]) + "]"

        hist_text = traj_to_text(history_arr)

        def tokenize(text):
            return torch.tensor(self.tokenizer.encode(text, add_special_tokens=False), dtype=torch.long)

        return {
            'pixel_values_front': images['front'],
            'pixel_values_rear': images['rear'],
            'pixel_values_left': images['left'],
            'pixel_values_right': images['right'],
            'action_gt': action_tensor,
            'decision_gt': torch.tensor(sample['decision_id'], dtype=torch.long),
            'instruction': sample['instruction'],
            'traj_name': sample['traj_name'],
            'step_idx': torch.tensor(sample['t'], dtype=torch.long),
            'is_last': torch.tensor(1 if sample['is_last'] else 0, dtype=torch.long),
            'sys_prompt_ids': tokenize("Please predict the future trajectory and select the parking slot id based on the human instruction, the current four observation images, the history trajectory, and the detected parking slots."),
            'instruct_ids': tokenize(f"Instruction: {sample['instruction']}"),
            'p_front_ids': tokenize("Front view:"),
            'p_rear_ids': tokenize("Rear view:"),
            'p_left_ids': tokenize("Left view:"),
            'p_right_ids': tokenize("Right view:"),
            'p_hist_ids': tokenize(f"History trajectory:{hist_text}"),
            'p_slots_ids': tokenize(f"Detected parking slot information: {slots_str}"),
            'p_answer_ids': tokenize("Answer: "),
        }


def classic_collate_fn(batch, pad_token_id=0):
    """ClassicVLA 专用 collate：output_ids/eos_ids 也做右 padding"""
    from .constants import IGNORE_INDEX
    collated = {}
    for key in batch[0].keys():
        if key in ('instruction', 'traj_name'):
            collated[key] = [item[key] for item in batch]
        elif key.endswith('_ids'):
            collated[key] = pad_sequence(
                [item[key] for item in batch], batch_first=True, padding_value=pad_token_id
            )
        else:
            collated[key] = torch.stack([item[key] for item in batch], dim=0)
    # output_ids padding 对应的 label 应为 IGNORE_INDEX
    collated['pad_token_id'] = pad_token_id
    return collated
