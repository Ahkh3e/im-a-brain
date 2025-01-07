from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from datasets import Dataset, load_dataset
import torch
import os
import json
from PIL import Image

# === Setup Constants ===
CACHE_DIR = "cache"
IMAGE_CACHE_DIR = os.path.join(CACHE_DIR, "images")
JSON_FILE_PATH = os.path.join(CACHE_DIR, "processed_results.json")
JSON_LINES_FILE = os.path.join(CACHE_DIR, "processed_results.jsonl")
MODEL_NAME = "unsloth/Llama-3.2-11B-Vision-Instruct-4bit"
OUTPUT_DIR = "outputs"
memory_log_count = 0
# === Load Image Helper ===
def load_image(image_id, image_dir=IMAGE_CACHE_DIR):
    image_path = os.path.join(image_dir, f"{image_id}.jpg")
    if os.path.exists(image_path):
        return Image.open(image_path).convert("RGB")
    raise FileNotFoundError(f"Image {image_id} not found in {image_dir}.")

# === Convert Data to Conversation Format ===
def convert_to_conversation(sample, image_dir=IMAGE_CACHE_DIR):
    if 'image_id' not in sample or not sample['image_id']:
        raise ValueError("Sample missing 'image_id'.")
    image_path = os.path.join(image_dir, f"{sample['image_id']}.jpg")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image {sample['image_id']} not found in {image_dir}.")
    image = Image.open(image_path).convert("RGB")
    
    conversation_instance= [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": sample.get("user_instruction", "")},
                {"type": "image", "image": image}
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample.get("assistant_response", "")}]
        }
    ]
    return { "messages" : conversation_instance}

import psutil
import os

def log_memory_usage(step):
    global memory_log_count
    memory_log_count += 1
    process = psutil.Process(os.getpid())
    memory = process.memory_info().rss / (1024 * 1024)  # Memory in MB
    print(f"Step {step}: Memory usage is {memory:.2f} MB - {memory_log_count}")
# === Process Dataset ===
def process_dataset(json_file_path,max_images, image_dir=IMAGE_CACHE_DIR ):
    with open(json_file_path, "r") as f:
        raw_data = json.load(f)
    limited_data = list(raw_data.items())[:max_images]
    processed_conversations = []
    conversation = [convert_to_conversation(sample, image_dir=image_dir)for example_id, sample in limited_data]
    return conversation

# === Save Dataset as JSON Lines ===
def save_as_json_lines(data, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for record in data:
            f.write(json.dumps(record) + "\n")
    print(f"Saved dataset to {output_file} as JSON Lines.")

# === Main Script ===
print("Processing dataset...")
max_images=100
processed_conversations = process_dataset(JSON_FILE_PATH,max_images, IMAGE_CACHE_DIR)

#print("Saving dataset as JSON Lines...")
#save_as_json_lines(processed_conversations, JSON_LINES_FILE)



def check_for_none(data):
    if data is None:
        return True
    if isinstance(data, dict):
        return any(check_for_none(v) for v in data.values())
    if isinstance(data, list):
        return any(check_for_none(v) for v in data)
    return False

# Check the data for any NoneType values
# print("Converting to Hugging Face Dataset...")
# hf_dataset = Dataset.from_list(processed_conversations)
# contains_none = check_for_none(hf_dataset)
# print(f"Does the data contain NoneType values? {contains_none}")
# print("Created dataset")
# Split into training and validation sets
# Split into training and validation sets

# === Load and Prepare Model ===
print("Loading model...")
model, tokenizer = FastVisionModel.from_pretrained(
    model_name=MODEL_NAME,
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth"
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config = None
)

# === Configure Trainer ===
print("Configuring trainer...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=processed_conversations,
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=5,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_DIR,
        report_to = "none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_num_proc=4,
        max_seq_length=2048,
    ),
)

# === Train Model ===
print("Starting training...")
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
torch.cuda.empty_cache()
torch.cuda.reset_max_memory_allocated()
import gc

gc.collect()

torch.cuda.empty_cache()
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")
print(torch.cuda.memory_summary(device=None, abbreviated=False))
trainer_stats = trainer.train()
print(f"Training runtime: {trainer_stats.metrics['train_runtime']} seconds.")
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
# === Save Model ===
print("Saving model...")
model.save_pretrained("/content/fine_tuned_model")
tokenizer.save_pretrained("/content/fine_tuned_model")

model.save_pretrained_gguf("/content/fine_tuned_gguf", tokenizer, quantization_method="f16")
print("Training completed and model saved!")
