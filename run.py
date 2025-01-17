import os
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel, LoraConfig, TaskType
import torch

# 自定义路径设置
model_dir_path = "./models/Qwen2-VL-2B-Instruct"  # 模型下载路径
checkpoint_path = "./outputs/Qwen2-VL-2B/checkpoint-62"  # 测试时加载的检查点路径
image_dir_path = "测试图像目录"  # 测试图像所在的目录

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

def infer_images_in_directory(image_dir_path, model, processor):
    """对指定目录下的所有图像进行推理"""
    for image_file in os.listdir(image_dir_path):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):  # 支持更多格式请自行添加
            image_path = os.path.join(image_dir_path, image_file)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
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

            print(f"Image: {image_file} -> Output: {output_text}")

model, processor = load_model_and_processor(model_dir_path, checkpoint_path)

infer_images_in_directory(image_dir_path, model, processor)
