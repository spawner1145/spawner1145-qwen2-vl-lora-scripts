import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.transformers import SwanLabCallback
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)
import swanlab
import json

# 用户可自定义的配置

# 模型相关路径（训练和测试模式共用）
model_repo = "Qwen/Qwen2-VL-2B-Instruct"  # 模型在ModelScope上的路径
cache_dir = "./models"  # 缓存模型的目录（训练和测试模式共用）

# 数据集路径（训练模式专用）
train_json_path = "data_vl.json"  # 训练数据集JSON文件路径

# 输出路径（训练模式专用）
output_dir = "./output/Qwen2-VL-2B"  # 训练时保存检查点和日志的输出目录

# 测试模型检查点路径（测试模式专用）
checkpoint_path = "./output/Qwen2-VL-2B/checkpoint-62"  # 测试用的检查点路径

# 模式选择: 'train' 或 'test'
mode = 'train'  # 更改为 'test' 来运行测试模式

if mode not in ['train', 'test']:
    raise ValueError("模式必须是'train'或'test'.")

def process_func(example):
    """将数据集进行预处理"""
    MAX_LENGTH = 8192
    conversation = example["conversations"]
    input_content = conversation[0]["value"]
    output_content = conversation[1]["value"]
    file_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[0]  # 获取图像路径
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"{file_path}", "resized_height": 280, "resized_width": 280},
                {"type": "text", "text": "COCO Yes:"}
            ]
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = {key: value.tolist() for key, value in inputs.items()}
    instruction = inputs

    response = tokenizer(f"{output_content}", add_special_tokens=False)

    input_ids = instruction["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"][0] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"][0]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)
    inputs['pixel_values'] = torch.tensor(inputs['pixel_values'])
    inputs['image_grid_thw'] = torch.tensor(inputs['image_grid_thw']).squeeze(0)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
            "pixel_values": inputs['pixel_values'], "image_grid_thw": inputs['image_grid_thw']}

def predict(messages, model):
    """准备推理"""
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

# 在modelscope上下载Qwen2-VL模型到本地目录下（训练和测试模式均需要）
model_dir = snapshot_download(model_repo, cache_dir=cache_dir, revision="master")

# 使用Transformers加载模型权重（训练和测试模式均需要）
tokenizer = AutoTokenizer.from_pretrained(f"{cache_dir}/{model_repo.split('/')[-1]}", use_fast=False, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(f"{cache_dir}/{model_repo.split('/')[-1]}")

model = Qwen2VLForConditionalGeneration.from_pretrained(f"{cache_dir}/{model_repo.split('/')[-1]}", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True,)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

if mode == 'train':
    with open(train_json_path, 'r') as f:
        data = json.load(f)
        train_data = data[:-4]
        test_data = data[-4:]

    with open("data_vl_train.json", "w") as f:
        json.dump(train_data, f)

    with open("data_vl_test.json", "w") as f:
        json.dump(test_data, f)

    train_ds = Dataset.from_json("data_vl_train.json")
    train_dataset = train_ds.map(process_func)

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
    )

    peft_model = get_peft_model(model, config)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=10,
        logging_first_step=5,
        num_train_epochs=2,
        save_steps=100,
        learning_rate=1e-4,  # 初始学习率
        lr_scheduler_type="linear",  # 使用线性学习率调度器，可根据需要更改为其他类型，例如："linear", "cosine", "constant"等。
        warmup_steps=500,  # 预热步骤数量，根据你的数据集大小和训练计划进行调整
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none",
    )
        
    swanlab_callback = SwanLabCallback(
        project="Qwen2-VL-finetune",
        experiment_name="qwen2-vl-coco2014",
        config={
            "model": "https://modelscope.cn/models/Qwen/Qwen2-VL-2B-Instruct",
            "dataset": "https://modelscope.cn/datasets/modelscope/coco_2014_caption/quickstart",
            "github": "https://github.com/datawhalechina/self-llm",
            "prompt": "COCO Yes: ",
            "train_data_number": len(train_data),
            "lora_rank": 64,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
        },
    )

    trainer = Trainer(
        model=peft_model,
        args=args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=[swanlab_callback],
    )

    trainer.train()

elif mode == 'test':
    # 获取测试模型（测试模式专用）
    val_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=True,
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
    )

    val_peft_model = PeftModel.from_pretrained(model, model_id=checkpoint_path, config=val_config)

    with open("data_vl_test.json", "r") as f:
        test_dataset = json.load(f)

    test_image_list = []
    for item in test_dataset:
        input_image_prompt = item["conversations"][0]["value"]
        origin_image_path = input_image_prompt.split("<|vision_start|>")[1].split("<|vision_end|>")[0]

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": origin_image_path},
                {"type": "text", "text": "COCO Yes:"}
            ]}]

        response = predict(messages, val_peft_model)
        messages.append({"role": "assistant", "content": f"{response}"})
        print(messages[-1])

        test_image_list.append(swanlab.Image(origin_image_path, caption=response))

    swanlab.log({"Prediction": test_image_list})

    swanlab.finish()
