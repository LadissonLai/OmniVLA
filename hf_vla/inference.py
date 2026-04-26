import torch
import os
import argparse
from PIL import Image
import numpy as np

# Use the core representations
from core.modeling_prismatic import ExpVLAForActionPrediction
from core.configuration_prismatic import OpenVLAConfig
from core.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from transformers import AutoProcessor, AutoModelForVision2Seq
from safetensors.torch import load_file
from peft import PeftModel

def load_vla_model(base_model_path, checkpoint_path=None, use_lora=True, device="cuda:0"):
    print(f"Loading processor from {base_model_path}")
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    
    print(f"Loading base model from {base_model_path}")
    vla = ExpVLAForActionPrediction.from_pretrained(
        base_model_path, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True
    ).to(device)

    # Load custom heads
    llm_dim = vla.config.text_config.hidden_size
    vocab_size = vla.config.text_config.vocab_size
    
    import torch.nn as nn
    past_traj_projector = nn.Sequential(
        nn.Linear(4, llm_dim // 2), nn.GELU(), nn.Linear(llm_dim // 2, llm_dim)
    ).to(device).to(torch.bfloat16)
    
    action_head = nn.Sequential(
        nn.Linear(llm_dim, llm_dim), nn.GELU(), nn.Linear(llm_dim, 1)
    ).to(device).to(torch.bfloat16)
    
    decision_head = nn.Linear(llm_dim, vocab_size).to(device).to(torch.bfloat16)

    # Load trained weights if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        if use_lora:
            lora_adapter_path = os.path.join(checkpoint_path, "lora_adapter")
            if os.path.exists(lora_adapter_path):
                vla = PeftModel.from_pretrained(vla, lora_adapter_path)
            
        proj_path = os.path.join(checkpoint_path, "past_traj_projector.pt")
        act_path = os.path.join(checkpoint_path, "action_head.pt")
        dec_path = os.path.join(checkpoint_path, "decision_head.pt")

        if os.path.exists(proj_path):
            past_traj_projector.load_state_dict(torch.load(proj_path, map_location=device))
        if os.path.exists(act_path):
            action_head.load_state_dict(torch.load(act_path, map_location=device))
        if os.path.exists(dec_path):
            decision_head.load_state_dict(torch.load(dec_path, map_location=device))

    vla.eval()
    past_traj_projector.eval()
    action_head.eval()
    decision_head.eval()
    
    return processor, vla, past_traj_projector, action_head, decision_head

def run_inference(sample_data_dir, base_model, checkpoint_dir=None):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # 1. Load model and weights
    processor, vla, past_traj_projector, action_head, decision_head = load_vla_model(
        base_model, checkpoint_dir, device=device
    )
    
    # 2. Prepare inputs (mock random data for demonstration or pull from dir)
    # You would typically use processor to preprocess your actual multi-view images
    IMG_SIZE = 224
    dummy_imgs = [Image.fromarray(np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)) for _ in range(4)]
    
    # Preprocess text and images
    instruction = "Park the vehicle in slot 2."
    prompt = f"Please predict the future trajectory and select the parking slot id... Instruction: {instruction}"
    inputs = processor(text=prompt, images=dummy_imgs[0], return_tensors="pt").to(device, torch.bfloat16)
    
    # We build the dictionary required by forward
    batch = {
        "pixel_values_front": processor.image_processor(dummy_imgs[0], return_tensors="pt")['pixel_values'].to(device, torch.bfloat16),
        "pixel_values_rear": processor.image_processor(dummy_imgs[1], return_tensors="pt")['pixel_values'].to(device, torch.bfloat16),
        "pixel_values_left": processor.image_processor(dummy_imgs[2], return_tensors="pt")['pixel_values'].to(device, torch.bfloat16),
        "pixel_values_right": processor.image_processor(dummy_imgs[3], return_tensors="pt")['pixel_values'].to(device, torch.bfloat16),
        "history_traj": torch.zeros((1, 8, 4), device=device, dtype=torch.bfloat16),
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
    }
    
    # 3. Model Forward
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = vla(batch, past_traj_projector=past_traj_projector)
            last_hidden_states = outputs.hidden_states[-1] 
            
            decision_hidden = last_hidden_states[:, -35, :] 
            actions_hidden = last_hidden_states[:, -34:-2, :] 
            
            pred_decision_logits = decision_head(decision_hidden)
            pred_actions_flat = action_head(actions_hidden).squeeze(-1)
            pred_actions = pred_actions_flat.view(-1, 8, 4)
            
            predicted_slot_token = torch.argmax(pred_decision_logits, dim=-1)
            predicted_slot = processor.tokenizer.decode(predicted_slot_token)
            
            print(f"Predicted Slot ID: {predicted_slot}")
            print(f"Predicted Action Trajectory: \n{pred_actions.cpu().numpy()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="openvla/openvla-7b")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    args = parser.parse_args()
    
    run_inference(None, args.base_model, args.checkpoint_dir)
