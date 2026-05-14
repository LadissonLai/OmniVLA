import os
import torch
from dataclasses import dataclass
from transformers import AutoProcessor, AutoConfig, AutoImageProcessor, AutoModelForVision2Seq
from peft import PeftModel
import draccus

from core.configuration_prismatic import OpenVLAConfig
from core.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from core.modeling_prismatic import ClassicVLAForActionPrediction

@dataclass
class MergeConfig:
    # 基础模型路径
    base_model_path: str = "/public/home/lqq_202430131053/codes/OmniVLA/openvla-7b"
    # 训练好的 LoRA 权重路径（注意指向包含 adapter_config.json 的目录）
    lora_path: str = "/public/home/lqq_202430131053/codes/OmniVLA/runs_classicvla_smart_history/2026-05-04_14-42/step_56010_loss_0.1578_ckpt/lora_adapter"
    # 合并后模型保存的新目录
    output_dir: str = "/public/home/lqq_202430131053/codes/OmniVLA/merged/openvla-7b-classic-merged"

@draccus.wrap()
def main(cfg: MergeConfig):
    print(f"Base model path: {cfg.base_model_path}")
    print(f"LoRA adapter path: {cfg.lora_path}")
    print(f"Output directory: {cfg.output_dir}")

    # 1. 注册所需的 Custom classes
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, ClassicVLAForActionPrediction)

    # 2. 加载 Processor
    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained(cfg.base_model_path, trust_remote_code=True)

    # 3. 加载基础 VLA 模型
    print("Loading base model...")
    base_model = ClassicVLAForActionPrediction.from_pretrained(
        cfg.base_model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    # 4. 加载 LoRA 权重至基础模型
    print("Loading LoRA adapter into base model...")
    model = PeftModel.from_pretrained(base_model, cfg.lora_path)

    # 5. 合并并卸载 LoRA
    print("Merging LoRA weights (merge_and_unload)...")
    merged_model = model.merge_and_unload()

    # 6. 保存为全新的全量参数模型
    print(f"Saving merged model to {cfg.output_dir}...")
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    merged_model.save_pretrained(cfg.output_dir)
    processor.save_pretrained(cfg.output_dir)

    print("✅ Merge complete! The merged model can now be loaded directly via from_pretrained.")

if __name__ == "__main__":
    main()
