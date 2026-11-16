from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import json
import os

# Create a tiny dummy dataset to test the training loop
dummy_data = [
    {
        "instruction": "What is the significance of the sterile ash layer at Balathal?",
        "input": "The stone enclosure was overlain by an undisturbed layer of sterile, white ashy soil 20–30 cm in thickness.",
        "output": "The sterile ash layer represents a 1000-year chronological separation between the Chalcolithic and Early Historic occupations, coinciding with regional aridification."
    },
    {
        "instruction": "What pottery was found in Trench 4?",
        "input": "Trench 4 yielded significant quantities of Samian ware pottery dating to the late 1st century AD.",
        "output": "Trench 4 contained Samian ware pottery from the late 1st century AD."
    }
]

os.makedirs("data", exist_ok=True)
with open("data/archaeology_qa.jsonl", "w") as f:
    for item in dummy_data:
        f.write(json.dumps(item) + "\n")

# 1. Load the 7B model (using Unsloth's pre-quantized 4-bit version to load faster)
model_name = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
max_seq_length = 512 # KEEP THIS LOW (512 or 1024 max) so you don't exceed 8GB VRAM

print(f"Loading {model_name} in 4-bit QLoRA mode...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

# 2. Attach LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rank of the adapter
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",
    use_gradient_checkpointing = "unsloth", # CRITICAL for 8GB VRAM
    random_state = 3407,
)

# 3. Format the dataset into the Qwen Chat Template
def format_prompt(examples):
    texts = []
    for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
        # Combining instruction and context just like our RAG generator does
        user_prompt = f"{instruction}\nContext: {input_text}"
        
        messages = [
            {"role": "system", "content": "You are an expert archaeological AI assistant."},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": output}
        ]
        
        # Apply the chat template so the model learns the exact format it will use in production
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return {"text": texts}

dataset = load_dataset("json", data_files="data/archaeology_qa.jsonl", split="train")
dataset = dataset.map(format_prompt, batched=True)

# 4. Train the model
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = TrainingArguments(
        per_device_train_batch_size = 1,      # MUST BE 1 for 8GB VRAM
        gradient_accumulation_steps = 4,      # Simulates a batch size of 4
        warmup_steps = 5,
        max_steps = 20,                       # Small number just to test if the loop runs without crashing
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",                 # 8-bit optimizer saves VRAM
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

print("Starting 7B training loop. Watch your GPU memory closely...")
trainer.train()

# 5. Save the fine-tuned adapters locally
print("Saving LoRA adapters...")
model.save_pretrained("strata-qwen-7b-archaeology-lora")
tokenizer.save_pretrained("strata-qwen-7b-archaeology-lora")
print("Success! The domain-adapted 7B model is saved.")
