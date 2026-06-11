import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .exp_parking_dataset import to_local_coords


class LangParkDataset(Dataset):
    """
    Dataset for LangPark VLA training.

    Extends the original parking dataset with:
      - full_history_traj: all historical odometry frames in current-frame coords (for MEM module)
      - align_label: per-instruction-token phase labels from lang_label_frames.csv (0=done/1=active/2=upcoming)
      - Updated decision GT: any frame where the target slot is visible is a positive sample
      - No pos/neg split; use standard DataLoader(shuffle=True)
    """

    def __init__(
        self,
        data_root: str,
        tokenizer,
        image_transform,
        max_history: int = 8,
        future_steps: int = 8,
        history_mode: str = 'smart',
        distance_interval: float = 0.5,
        turn_yaw_thresh: float = 5.0,
        turn_dense_interval: float = 0.1,
    ):
        self.data_root         = data_root
        self.tokenizer         = tokenizer
        self.image_transform   = image_transform
        self.max_history       = max_history
        self.future_steps      = future_steps
        self.history_mode      = history_mode
        self.distance_interval = distance_interval
        self.turn_yaw_thresh   = turn_yaw_thresh
        self.turn_dense_interval = turn_dense_interval

        # {traj_dir: {frame_id(int): label_str}}
        self.lang_labels_by_traj: dict = {}

        self.samples: list = []
        self._build_dataset()

    # ------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------

    def _build_dataset(self):
        # Support two-level layout: data_root/<scenario>/<timestamp>/decision.txt
        # A valid trajectory directory is identified by containing decision.txt.
        # Walk up to two levels deep so both flat (data_root/<traj>/) and
        # nested (data_root/<scenario>/<traj>/) layouts are handled automatically.
        traj_dirs = []
        for entry in sorted(os.listdir(self.data_root)):
            level1 = os.path.join(self.data_root, entry)
            if not os.path.isdir(level1):
                continue
            if os.path.exists(os.path.join(level1, "decision.txt")):
                # Flat layout: data_root/<traj>/
                traj_dirs.append(level1)
            else:
                # Nested layout: data_root/<scenario>/<traj>/
                for sub in sorted(os.listdir(level1)):
                    level2 = os.path.join(level1, sub)
                    if os.path.isdir(level2) and os.path.exists(os.path.join(level2, "decision.txt")):
                        traj_dirs.append(level2)

        for traj_dir in traj_dirs:
            with open(os.path.join(traj_dir, "decision.txt"), 'r') as f:
                lines = f.readlines()
                instruction     = lines[0].strip()
                true_decision_id = int(lines[1].strip())

            odom_df = pd.read_csv(os.path.join(traj_dir, "odom.csv"))
            with open(os.path.join(traj_dir, "parking_slots.txt"), 'r') as f:
                slots_data = [json.loads(line) for line in f.readlines()]

            # Load lang_label_frames.csv once per trajectory
            lang_label_path = os.path.join(traj_dir, "lang_label_frames.csv")
            if os.path.exists(lang_label_path):
                ldf = pd.read_csv(lang_label_path, dtype={'token_mask_012': str})
                self.lang_labels_by_traj[traj_dir] = {
                    int(row['frame_id']): row['token_mask_012']
                    for _, row in ldf.iterrows()
                }
            else:
                self.lang_labels_by_traj[traj_dir] = {}

            num_frames = len(odom_df)
            for t in range(num_frames):
                curr_slots  = slots_data[t]
                visible_ids = [slot['id'] for slot in curr_slots]

                # New GT: target slot is visible → predict the correct slot id
                decision_id = true_decision_id if true_decision_id in visible_ids else 0

                self.samples.append({
                    'traj_dir':        traj_dir,
                    't':               t,
                    'instruction':     instruction,
                    'true_decision_id': true_decision_id,
                    'decision_id':     decision_id,
                    'slots':           curr_slots,
                    'odom':            odom_df,   # shared reference (same object for all frames)
                })

        print(
            f"LangParkDataset built: {len(self.samples)} samples "
            f"from {len(traj_dirs)} trajectories."
        )

    # ------------------------------------------------------------------
    # Smart history sampling (mirrors exp_parking_dataset logic)
    # ------------------------------------------------------------------

    def _get_smart_history(self, odom: pd.DataFrame, t: int, curr_state) -> list:
        hist_df       = odom.iloc[0:t]
        history_traj  = []

        if t == 0:
            return history_traj

        if self.history_mode == 'fixed_count':
            if t <= self.max_history:
                history_traj = [
                    to_local_coords(row['x'], row['y'], row['yaw'],
                                    curr_state['x'], curr_state['y'], curr_state['yaw'])
                    for _, row in hist_df.iterrows()
                ]
            else:
                indices = np.linspace(0, t - 1, self.max_history, dtype=int)
                history_traj = [
                    to_local_coords(row['x'], row['y'], row['yaw'],
                                    curr_state['x'], curr_state['y'], curr_state['yaw'])
                    for _, row in odom.iloc[indices].iterrows()
                ]

        elif self.history_mode == 'fixed_distance':
            last_idx = -1
            for i in range(t):
                if i == 0 or i == t - 1:
                    history_traj.append(
                        to_local_coords(hist_df.iloc[i]['x'], hist_df.iloc[i]['y'],
                                        hist_df.iloc[i]['yaw'],
                                        curr_state['x'], curr_state['y'], curr_state['yaw'])
                    )
                    last_idx = i
                else:
                    dist = np.hypot(
                        hist_df.iloc[i]['x'] - hist_df.iloc[last_idx]['x'],
                        hist_df.iloc[i]['y'] - hist_df.iloc[last_idx]['y'],
                    )
                    if dist >= self.distance_interval:
                        history_traj.append(
                            to_local_coords(hist_df.iloc[i]['x'], hist_df.iloc[i]['y'],
                                            hist_df.iloc[i]['yaw'],
                                            curr_state['x'], curr_state['y'], curr_state['yaw'])
                        )
                        last_idx = i

        elif self.history_mode == 'smart':
            states = []
            for i in range(t):
                start    = max(0, i - 2)
                end      = min(t - 1, i + 2)
                yaw_diff = abs(hist_df.iloc[end]['yaw'] - hist_df.iloc[start]['yaw']) % 360
                if yaw_diff > 180:
                    yaw_diff = 360 - yaw_diff
                states.append('Turn' if yaw_diff > self.turn_yaw_thresh else 'Straight')

            last_idx = -1
            for i in range(t):
                row = hist_df.iloc[i]
                pt  = to_local_coords(row['x'], row['y'], row['yaw'],
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
                        dist = np.hypot(
                            row['x'] - hist_df.iloc[last_idx]['x'],
                            row['y'] - hist_df.iloc[last_idx]['y'],
                        )
                        if dist >= self.turn_dense_interval:
                            history_traj.append(pt)
                            last_idx = i

        return history_traj

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample   = self.samples[idx]
        t        = sample['t']
        odom     = sample['odom']
        traj_dir = sample['traj_dir']

        # ── 1. Images ──────────────────────────────────────────────────
        images = {}
        for view in ['front', 'rear', 'left', 'right']:
            img_path = os.path.join(traj_dir, "images", view, f"{t + 1:06d}.png")
            images[view] = self.image_transform(Image.open(img_path).convert("RGB"))

        curr_state = odom.iloc[t]

        # ── 2. Smart-sampled history (for LLM context) ─────────────────
        history_traj = self._get_smart_history(odom, t, curr_state)
        if len(history_traj) == 0:
            history_tensor = torch.zeros((1, 4), dtype=torch.float32)
        else:
            history_tensor = torch.tensor(np.array(history_traj), dtype=torch.float32)

        # ── 3. Full history (all frames, for MEM module) ───────────────
        full_hist = [
            to_local_coords(odom.iloc[i]['x'], odom.iloc[i]['y'], odom.iloc[i]['yaw'],
                            curr_state['x'], curr_state['y'], curr_state['yaw'])
            for i in range(t)
        ]
        if len(full_hist) == 0:
            full_history_tensor = torch.zeros((1, 4), dtype=torch.float32)
        else:
            full_history_tensor = torch.tensor(np.array(full_hist), dtype=torch.float32)

        # ── 4. Future trajectory (GT) ──────────────────────────────────
        future_df   = odom.iloc[t + 1: t + 1 + self.future_steps]
        future_traj = [
            to_local_coords(row['x'], row['y'], row['yaw'],
                            curr_state['x'], curr_state['y'], curr_state['yaw'])
            for _, row in future_df.iterrows()
        ]
        while len(future_traj) < self.future_steps:
            future_traj.append(future_traj[-1] if future_traj else np.zeros(4))
        action_tensor = torch.tensor(np.array(future_traj), dtype=torch.float32)

        # ── 5. Parking slots text ──────────────────────────────────────
        slots_str = "[]"
        if len(sample['slots']) > 0:
            slots_str = "[" + ",".join(
                [f"(id={s['id']},x={s['x']:.2f},y={s['y']:.2f})" for s in sample['slots']]
            ) + "]"

        # ── 6. Tokenisation ────────────────────────────────────────────
        def tokenize(text):
            return torch.tensor(
                self.tokenizer.encode(text, add_special_tokens=False), dtype=torch.long
            )

        # instruct_ids uses add_special_tokens=False → no BOS prepended.
        # NOTE: the "Instruction:" prefix is intentionally NOT included here. It
        # lives at the end of sys_prompt_ids instead, so that instruct_ids contains
        # ONLY the raw instruction tokens. This makes instruct_ids align 1:1 with the
        # per-token phase labels in lang_label_frames.csv (token_mask_012), which were
        # generated from tokenize(instruction) without any prefix. Since sys_prompt and
        # instruct are concatenated adjacently in the model forward, the text the LLM
        # sees is unchanged.
        instruct_ids = tokenize(sample['instruction'])

        decision_token_id = self.tokenizer.encode(
            str(sample['decision_id']), add_special_tokens=False
        )[-1]

        # ── 7. Instruction alignment labels ───────────────────────────
        lang_labels = self.lang_labels_by_traj.get(traj_dir, {})
        if t in lang_labels:
            label_str = lang_labels[t]
            inst_len  = instruct_ids.shape[0]
            if len(label_str) != inst_len:
                # Graceful mismatch handling: truncate or pad with -100
                label_list = [int(c) for c in label_str]
                if len(label_list) < inst_len:
                    label_list = label_list + [-100] * (inst_len - len(label_list))
                else:
                    label_list = label_list[:inst_len]
                align_label = torch.tensor(label_list, dtype=torch.long)
            else:
                align_label = torch.tensor([int(c) for c in label_str], dtype=torch.long)
        else:
            # No label file or missing frame → all positions ignored
            align_label = torch.full((instruct_ids.shape[0],), -100, dtype=torch.long)

        return {
            'pixel_values_front': images['front'],
            'pixel_values_rear':  images['rear'],
            'pixel_values_left':  images['left'],
            'pixel_values_right': images['right'],
            'history_traj':       history_tensor,        # [N_hist, 4], variable length
            'full_history_traj':  full_history_tensor,   # [t, 4],      variable length
            'action_gt':          action_tensor,          # [future_steps, 4]
            'decision_gt':        torch.tensor(decision_token_id, dtype=torch.long),
            'align_label':        align_label,            # [L_inst]
            'instruction':        sample['instruction'],
            'sys_prompt_ids': tokenize(
                "Please predict the future trajectory and select the parking slot id based on "
                "the human instruction, the current four observation images, the history "
                "trajectory, and the detected parking slots. Instruction:"
            ),
            'instruct_ids':   instruct_ids,
            'p_front_ids':    tokenize("Front view:"),
            'p_rear_ids':     tokenize("Rear view:"),
            'p_left_ids':     tokenize("Left view:"),
            'p_right_ids':    tokenize("Right view:"),
            'p_hist_ids':     tokenize("History trajectory:"),
            'p_slots_ids':    tokenize(f"Detected parking slot information: {slots_str}"),
        }


# ──────────────────────────────────────────────────────────────────────
# Collate function
# ──────────────────────────────────────────────────────────────────────

def langpark_collate_fn(batch: list, pad_token_id: int = 0) -> dict:
    """
    Collate for LangParkDataset.

    - Variable-length trajectory lists (history_traj, full_history_traj) are returned
      as Python lists; padding is handled inside the model forward.
    - align_label is right-padded with -100 (IGNORE_INDEX).
    - All *_ids tensors are right-padded with pad_token_id.
    - Everything else is stacked.
    """
    collated = {}
    for key in batch[0].keys():
        if key == 'instruction':
            collated[key] = [item[key] for item in batch]

        elif key in ('history_traj', 'full_history_traj'):
            # Variable-length; leave as list, model forward handles padding
            collated[key] = [item[key] for item in batch]

        elif key == 'align_label':
            collated[key] = pad_sequence(
                [item[key] for item in batch],
                batch_first=True,
                padding_value=-100,
            )

        elif key.endswith('_ids'):
            collated[key] = pad_sequence(
                [item[key] for item in batch],
                batch_first=True,
                padding_value=pad_token_id,
            )

        else:
            collated[key] = torch.stack([item[key] for item in batch], dim=0)

    collated['pad_token_id'] = pad_token_id
    return collated
