from unsloth import FastLanguageModel

# Load your fine-tuned production model
print("Loading production model for GGUF conversion...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="strata-qwen-7b-archaeology-lora-production",
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
)

# Export to 4-bit GGUF (Q4_K_M is the best balance of speed and quality)
print("Exporting to GGUF format...")
model.save_pretrained_gguf(
    "strata_qwen_7b_archaeology", 
    tokenizer, 
    quantization_method="q4_k_m"
)
print("Success! GGUF export complete.")
