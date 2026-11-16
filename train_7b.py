from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import os

model_name = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
max_seq_length = 512 

print(f"Loading {model_name} in 4-bit QLoRA mode...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

def format_prompt(examples):
    texts = []
    for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
        user_prompt = f"{instruction}\nContext: {input_text}"
        messages = [
            {"role": "system", "content": "You are an expert archaeological AI assistant."},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": output}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return {"text": texts}

# Load the full generated dataset
print("Loading synthetic archaeological dataset...")
dataset = load_dataset("json", data_files="data/archaeology_qa_full.jsonl", split="train")
dataset = dataset.map(format_prompt, batched=True)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 2, # Train for 2 full passes over the dataset
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

print("Starting full QLoRA training loop...")
trainer.train()

print("Saving production LoRA adapters...")
model.save_pretrained("strata-qwen-7b-archaeology-lora-production")
tokenizer.save_pretrained("strata-qwen-7b-archaeology-lora-production")
print("Success! The production model is saved.")
