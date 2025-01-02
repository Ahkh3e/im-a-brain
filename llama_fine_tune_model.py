import os
import json
from PIL import Image

# Load class labels from a JSON file
def load_classes(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

class_labels = load_classes("class_labels.json")

instruction = "You are an expert in object detection. Identify all objects in this image with bounding boxes and descriptions."

# Convert YOLO format to conversation format
def convert_yolo_to_conversation(image_dir, label_dir):
    conversation_data = []
    for image_file in os.listdir(image_dir):
        # Handle both PNG and JPG files
        if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        label_file = os.path.join(label_dir, image_file.replace(".png", ".txt").replace(".jpg", ".txt"))
        image_path = os.path.join(image_dir, image_file)

        annotations = []
        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                for line in f.readlines():
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    class_name = class_labels.get(str(int(class_id)), "unknown")
                    annotations.append(f"{class_name} at x_center={x_center}, y_center={y_center}, width={width}, height={height}.")

        messages = [
            {"role": "user", "content": [{"type": "text", "text": instruction}, {"type": "image", "image": Image.open(image_path)}]},
            {"role": "assistant", "content": [{"type": "text", "text": " ".join(annotations)}]},
        ]
        conversation_data.append({"messages": messages})
    return conversation_data

# Prepare train and validation datasets
train_data = convert_yolo_to_conversation("Custom_Data/images/train", "Custom_Data/labels/train")
val_data = convert_yolo_to_conversation("Custom_Data/images/val", "Custom_Data/labels/val")






from unsloth import FastVisionModel
import torch
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit", # Llama 3.2 vision support
    "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
    "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit", # Can fit in a 80GB card!
    "unsloth/Llama-3.2-90B-Vision-bnb-4bit",

    "unsloth/Pixtral-12B-2409-bnb-4bit",              # Pixtral fits in 16GB!
    "unsloth/Pixtral-12B-Base-2409-bnb-4bit",         # Pixtral base model

    "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",          # Qwen2 VL support
    "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
    "unsloth/Qwen2-VL-72B-Instruct-bnb-4bit",

    "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit",      # Any Llava variant works!
    "unsloth/llava-1.5-7b-hf-bnb-4bit",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastVisionModel.from_pretrained(
    model_name= "unsloth/Llama-3.2-11B-Vision-Instruct",
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 16,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)


FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=train_data,
    eval_dataset=val_data,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=30,  # Set `num_train_epochs` for full training
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        remove_unused_columns=False,
        max_seq_length=2048,
    ),
)

#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# Train the model
trainer_stats = trainer.train()


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

# Save the fine-tuned model in GGUF format
model.save_pretrained_gguf(
    save_directory="model",
    tokenizer=tokenizer,
    quantization_method="f16"  # Quantize weights to float16 for efficiency
)

# Save the fine-tuned model
model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")

print("Model and tokenizer saved to 'fine_tuned_model' directory.")