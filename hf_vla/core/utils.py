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

def visualize_test_expvla(
    save_path: str,
    pred_actions: torch.Tensor,
    gt_actions: torch.Tensor,
    past_traj: torch.Tensor,
    pred_decision,
    gt_decision,
    instruction: str,
    image_front: torch.Tensor,
    image_rear: torch.Tensor,
    image_left: torch.Tensor,
    image_right: torch.Tensor,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # helper func to convert normalized tensor back to viewable image
    def to_img(tensor_img):
        if tensor_img.shape[0] > 3:
            tensor_img = tensor_img[:3, :, :]
        img = tensor_img.detach().cpu().numpy().transpose(1, 2, 0)
        img = np.clip((img - img.min()) / (img.max() - img.min() + 1e-8), 0, 1)
        return img
    
    fig = plt.figure(figsize=(20, 10), dpi=80)
    gs = fig.add_gridspec(2, 4)
    
    # 轨迹图 - 右半部分
    ax_traj = fig.add_subplot(gs[0:2, 2:4])
    
    pred_x = pred_actions[:, 0].detach().cpu().to(torch.float32).numpy()
    pred_y = pred_actions[:, 1].detach().cpu().to(torch.float32).numpy()
    gt_x = gt_actions[:, 0].detach().cpu().to(torch.float32).numpy()
    gt_y = gt_actions[:, 1].detach().cpu().to(torch.float32).numpy()
    
    if past_traj is not None and past_traj.shape[0] > 0:
        hist_x = past_traj[:, 0].detach().cpu().to(torch.float32).numpy()
        hist_y = past_traj[:, 1].detach().cpu().to(torch.float32).numpy()
        ax_traj.plot(-hist_y, hist_x, marker='s', linewidth=2, markersize=6, color='green', label='History')

    ax_traj.plot(-pred_y, pred_x, marker='o', linewidth=3, markersize=8, color='blue', label='Predicted')
    ax_traj.plot(-gt_y, gt_x, marker='*', linewidth=3, markersize=8, color='red', label='Ground Truth')
    
    # Ego vehicle at origin
    ax_traj.plot(0, 0, marker='^', markersize=10, color='orange', label='Ego (0,0)')
    
    ax_traj.set_title(f"{instruction}\nPred Decision: {pred_decision} | GT Decision: {gt_decision}")
    # 采用半透明背景和自适应的最佳位置，防止图例遮挡关键轨迹
    ax_traj.legend(loc='best', framealpha=0.8)
    ax_traj.grid(True)
    ax_traj.set_xlabel("-Y")
    ax_traj.set_ylabel("X")
    
    # 动态调整坐标轴比例尺
    # 获取所有的绘制坐标(图表的X轴对应汽车-Y，Y轴对应汽车X，包含自车原点)
    all_plot_x = np.concatenate([-pred_y, -gt_y, [0.0]])
    all_plot_y = np.concatenate([pred_x, gt_x, [0.0]])
    
    if past_traj is not None and past_traj.shape[0] > 0:
        all_plot_x = np.concatenate([all_plot_x, -hist_y])
        all_plot_y = np.concatenate([all_plot_y, hist_x])
        
    # 要求自车(0,0)在图片的中央，因此x和y必须各自以0为中心绝对对称
    max_abs_x = max(np.max(np.abs(all_plot_x)), 0.5)
    max_abs_y = max(np.max(np.abs(all_plot_y)), 0.5)
    
    # 统一x和y的最大数值作为两者共同的量程，保证缩放完全一致且自车居中
    max_dist = max(max_abs_x, max_abs_y)
    
    # 刻度范围放大20%作为边距，确保所有轨迹清晰显示在画布内
    ax_traj.set_xlim(-max_dist * 1.2, max_dist * 1.2)
    ax_traj.set_ylim(-max_dist * 1.2, max_dist * 1.2)
    
    # 锁定物理世界真正的 1:1 比例，由于我们统一了量程范围，这回不再会被无理压缩了
    ax_traj.set_aspect('equal')
    
    # 图像部分 - 左半部分
    ax_front = fig.add_subplot(gs[0, 0])
    ax_front.imshow(to_img(image_front))
    ax_front.set_title("Front")
    ax_front.axis("off")
    
    ax_rear = fig.add_subplot(gs[0, 1])
    ax_rear.imshow(to_img(image_rear))
    ax_rear.set_title("Rear")
    ax_rear.axis("off")
    
    ax_left = fig.add_subplot(gs[1, 0])
    ax_left.imshow(to_img(image_left))
    ax_left.set_title("Left")
    ax_left.axis("off")
    
    ax_right = fig.add_subplot(gs[1, 1])
    ax_right.imshow(to_img(image_right))
    ax_right.set_title("Right")
    ax_right.axis("off")
    
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def visualize_langpark(
    save_path: str,
    pred_actions: torch.Tensor,   # [future_steps, 4]
    gt_actions: torch.Tensor,     # [future_steps, 4]
    past_traj,                     # tensor [N, 4] or None
    pred_decision: str,
    gt_decision: str,
    instruction: str,
    image_front: torch.Tensor,
    image_rear: torch.Tensor,
    image_left: torch.Tensor,
    image_right: torch.Tensor,
    token_texts: list,             # decoded instruction tokens
    gt_labels: list,               # per-token GT labels (0=done/1=active/2=upcoming/-100=ignore)
    pred_labels: list,             # per-token predicted labels (0/1/2)
):
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    def to_img(t):
        if t.shape[0] > 3:
            t = t[:3]
        img = t.detach().cpu().numpy().transpose(1, 2, 0)
        return np.clip((img - img.min()) / (img.max() - img.min() + 1e-8), 0, 1)

    fig = plt.figure(figsize=(20, 14), dpi=80)
    gs  = fig.add_gridspec(3, 4, height_ratios=[2, 2, 1.5], hspace=0.45, wspace=0.3)

    # Camera images (top-left 2x2 block)
    for spec, title, img in [
        (gs[0, 0], "Front", image_front),
        (gs[0, 1], "Rear",  image_rear),
        (gs[1, 0], "Left",  image_left),
        (gs[1, 1], "Right", image_right),
    ]:
        ax = fig.add_subplot(spec)
        ax.imshow(to_img(img))
        ax.set_title(title)
        ax.axis("off")

    # Trajectory plot (rows 0-1, cols 2-3)
    ax_traj = fig.add_subplot(gs[0:2, 2:4])
    px = pred_actions[:, 0].detach().cpu().float().numpy()
    py = pred_actions[:, 1].detach().cpu().float().numpy()
    gx = gt_actions[:, 0].detach().cpu().float().numpy()
    gy = gt_actions[:, 1].detach().cpu().float().numpy()

    all_plot_x = np.concatenate([-py, -gy, [0.]])
    all_plot_y = np.concatenate([px,   gx, [0.]])

    if past_traj is not None and past_traj.shape[0] > 1:
        hx = past_traj[:, 0].detach().cpu().float().numpy()
        hy = past_traj[:, 1].detach().cpu().float().numpy()
        ax_traj.plot(-hy, hx, marker='s', lw=2, ms=6, color='green', label='History')
        all_plot_x = np.concatenate([all_plot_x, -hy])
        all_plot_y = np.concatenate([all_plot_y,  hx])

    ax_traj.plot(-py, px, marker='o', lw=3, ms=8, color='blue', label='Predicted')
    ax_traj.plot(-gy, gx, marker='*', lw=3, ms=8, color='red',  label='GT')
    ax_traj.plot(0, 0, marker='^', ms=10, color='orange', label='Ego (0,0)')
    ax_traj.set_title(f"{instruction}\nPred: {pred_decision} | GT: {gt_decision}")
    ax_traj.legend(loc='best', framealpha=0.8)
    ax_traj.grid(True)
    ax_traj.set_xlabel("-Y")
    ax_traj.set_ylabel("X")
    d = max(np.max(np.abs(all_plot_x)), np.max(np.abs(all_plot_y)), 0.5)
    ax_traj.set_xlim(-d * 1.2, d * 1.2)
    ax_traj.set_ylim(-d * 1.2, d * 1.2)
    ax_traj.set_aspect('equal')

    # Language progress panels (row 2: GT left / Pred right)
    ax_gt   = fig.add_subplot(gs[2, 0:2])
    ax_pred = fig.add_subplot(gs[2, 2:4])

    TOKEN_FONTSIZE = 14
    TITLE_FONTSIZE = 11

    # Ensure a renderer exists so token widths can be measured exactly.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def render_tokens(ax, texts, colors, title):
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=TITLE_FONTSIZE, loc='left', pad=2)
        inv    = ax.transAxes.inverted()
        x, y   = 0.0, 0.90
        line_h = 0.30
        for tok, color in zip(texts, colors):
            disp = tok.replace('▁', ' ').replace('Ġ', ' ')  # strip ▁ / Ġ prefix
            if not disp:
                continue
            t = ax.text(x, y, disp, color=color, fontsize=TOKEN_FONTSIZE,
                        transform=ax.transAxes, va='top', ha='left',
                        fontfamily='monospace')
            # Measure the rendered glyph width and convert px -> axes fraction.
            bbox = t.get_window_extent(renderer=renderer)
            w    = inv.transform((bbox.x1, 0))[0] - inv.transform((bbox.x0, 0))[0]
            # Wrap to a new line if this token overflows the panel width.
            if x > 0.0 and x + w > 0.98:
                t.remove()
                x  = 0.0
                y -= line_h
                if y < 0.0:
                    break
                t = ax.text(x, y, disp, color=color, fontsize=TOKEN_FONTSIZE,
                            transform=ax.transAxes, va='top', ha='left',
                            fontfamily='monospace')
                bbox = t.get_window_extent(renderer=renderer)
                w    = inv.transform((bbox.x1, 0))[0] - inv.transform((bbox.x0, 0))[0]
            x += w

    # GT panel: skip -100 positions; active(1) → red, others → gray
    gt_t, gt_c = [], []
    for tok, lbl in zip(token_texts, gt_labels):
        if lbl == -100:
            continue
        gt_t.append(tok)
        gt_c.append('red' if lbl == 1 else 'gray')
    render_tokens(ax_gt, gt_t, gt_c,
                  "GT Alignment  [red=active | gray=other]")

    # Pred panel: all tokens, active(1) → blue, others → gray
    pred_c = ['blue' if lbl == 1 else 'gray' for lbl in pred_labels]
    render_tokens(ax_pred, token_texts, pred_c,
                  "Pred Alignment  [blue=active | gray=other]")

    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

