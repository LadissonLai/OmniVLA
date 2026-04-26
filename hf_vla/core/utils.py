import torch
import os
import requests
import matplotlib.pyplot as plt
import numpy as np

from .constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX

def get_current_action_mask(token_ids):
    newline_positions = token_ids != IGNORE_INDEX
    cumsum = torch.cumsum(newline_positions, dim=1)
    mask = (1 <= cumsum) & (cumsum <= ACTION_DIM)
    action_tokens_only_mask = token_ids > ACTION_TOKEN_BEGIN_IDX
    mask = action_tokens_only_mask * mask
    return mask

def get_next_actions_mask(token_ids):
    newline_positions = token_ids != IGNORE_INDEX
    cumsum = torch.cumsum(newline_positions, dim=1)
    mask = cumsum > ACTION_DIM
    action_tokens_only_mask = token_ids > ACTION_TOKEN_BEGIN_IDX
    mask = action_tokens_only_mask * mask
    return mask

def compute_token_accuracy(predicted_token_ids, ground_truth_token_ids, mask):
    correct_preds = (predicted_token_ids == ground_truth_token_ids) & mask
    accuracy = correct_preds.sum().float() / mask.sum().float()
    return accuracy

def model_is_on_hf_hub(model_id):
    """Check if model exists on HF Hub vs local path"""
    if os.path.exists(model_id):
        return False
    try:
        response = requests.head(f"https://huggingface.co/{model_id}")
        return response.status_code == 200
    except:
        return False

def check_model_logic_mismatch(*args, **kwargs):
    pass

def update_auto_map(*args, **kwargs):
    pass

def visualize_train_expvla(
    project_folder: str,
    pred_actions: torch.Tensor,
    gt_actions: torch.Tensor,
    pred_decisions: list,
    gt_decisions: list,
    instructions: list,
    images_front: torch.Tensor,
    images_rear: torch.Tensor,
    images_left: torch.Tensor,
    images_right: torch.Tensor,
    epoch: int,
    step: int,
    num_images_log: int = 1,
):
    visualize_path = os.path.join(
        project_folder,
        f"epoch_{epoch}",
    )
    os.makedirs(visualize_path, exist_ok=True)
    
    # helper func to convert normalized tensor back to viewable image
    def to_img(tensor_img):
        # tensor_img shape is likely [6, 224, 224] for PrismaticProcessor (where 3 channels for clip and 3 for dino)
        # or maybe [3, H, W], we only need first 3 channels for visualizing RGB
        if tensor_img.shape[0] > 3:
            tensor_img = tensor_img[:3, :, :]
            
        # assume tensor_img is [3, H, W] normalized with some mean/std, or [0, 1]
        # if not normalized, we can simply detach and permute
        img = tensor_img.detach().cpu().numpy().transpose(1, 2, 0)
        # simplistic un-normalization (if it was normalized with standard mean/std, you'd apply inverse here)
        # Just clamp to [0, 1] for visualization stability if values are out of bounds
        img = np.clip((img - img.min()) / (img.max() - img.min() + 1e-8), 0, 1)
        return img
    
    bs = min(num_images_log, pred_actions.shape[0])
    for i in range(bs):
        fig = plt.figure(figsize=(20, 10), dpi=80)
        gs = fig.add_gridspec(2, 4)
        
        # 轨迹图 - 右半部分
        ax_traj = fig.add_subplot(gs[0:2, 2:4])
        
        pred_x = pred_actions[i, :, 0].detach().cpu().to(torch.float32).numpy()
        pred_y = pred_actions[i, :, 1].detach().cpu().to(torch.float32).numpy()
        gt_x = gt_actions[i, :, 0].detach().cpu().to(torch.float32).numpy()
        gt_y = gt_actions[i, :, 1].detach().cpu().to(torch.float32).numpy()
        
        ax_traj.plot(-pred_y, pred_x, marker='o', linewidth=3, markersize=8, color='blue', label='Predicted')
        ax_traj.plot(-gt_y, gt_x, marker='*', linewidth=3, markersize=8, color='red', label='Ground Truth')
        
        ax_traj.set_title(f"{instructions[i]}\nPred Decision: {pred_decisions[i]} | GT Decision: {gt_decisions[i]}")
        ax_traj.legend(loc='best')
        ax_traj.grid(True)
        ax_traj.set_xlabel("-Y")
        ax_traj.set_ylabel("X")
        
        # 将原点(0,0)放置在图表的中下部
        max_abs_y = max(np.max(np.abs(pred_y)), np.max(np.abs(gt_y)))
        max_abs_y = max(max_abs_y, 0.5)  # 避免全0情况
        ax_traj.set_xlim(-max_abs_y * 1.5, max_abs_y * 1.5)
        
        max_x = max(np.max(pred_x), np.max(gt_x))
        min_x = min(np.min(pred_x), np.min(gt_x), 0.0)
        ax_traj.set_ylim(min_x - 0.2, max(max_x * 1.2, 1.0))
        
        # 图像部分 - 左半部分
        ax_front = fig.add_subplot(gs[0, 0])
        ax_front.imshow(to_img(images_front[i]))
        ax_front.set_title("Front")
        ax_front.axis("off")
        
        ax_rear = fig.add_subplot(gs[0, 1])
        ax_rear.imshow(to_img(images_rear[i]))
        ax_rear.set_title("Rear")
        ax_rear.axis("off")
        
        ax_left = fig.add_subplot(gs[1, 0])
        ax_left.imshow(to_img(images_left[i]))
        ax_left.set_title("Left")
        ax_left.axis("off")
        
        ax_right = fig.add_subplot(gs[1, 1])
        ax_right.imshow(to_img(images_right[i]))
        ax_right.set_title("Right")
        ax_right.axis("off")
        
        save_path = os.path.join(visualize_path, f"step_{step}_sample_{i}.png")
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

