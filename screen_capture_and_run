import os
import time
from mss import mss
import cv2
import numpy as np
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel, LoraConfig, TaskType
import torch

# 自定义路径设置
model_dir_path = "./models/Qwen2-VL-2B-Instruct"  # 模型下载路径
checkpoint_path = "./outputs/Qwen2-VL-2B/checkpoint-62"  # 测试时加载的检查点路径

def load_model_and_processor(model_dir_path, checkpoint_path):
    """加载模型和处理器"""
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=True,
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
    )
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_dir_path, torch_dtype="auto", device_map="auto")
    model = PeftModel.from_pretrained(model, model_id=checkpoint_path, config=config)
    processor = AutoProcessor.from_pretrained(model_dir_path)
    
    return model, processor

def capture_screen_and_infer(model, processor):
    """实时监控屏幕并进行推理"""
    with mss() as sct:
        monitor = sct.monitors[1]
        
        while True:
            screenshot = sct.grab(monitor)
            img_np = np.array(Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX"))  # 转换为numpy数组
            
            image = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "COCO Yes:"},
                    ],
                }
            ]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")

            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            print(f"Screen Capture -> Output: {output_text}")
            
            time.sleep(1)  # 控制帧率，避免过快

model, processor = load_model_and_processor(model_dir_path, checkpoint_path)

capture_screen_and_infer(model, processor)
