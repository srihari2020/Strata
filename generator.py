import torch
from unsloth import FastLanguageModel
from retriever import StrataRetriever

class StrataGenerator:
    def __init__(self, model_path="strata-qwen-7b-archaeology-lora-production"):
        print(f"Loading Fine-Tuned 7B LLM from {model_path}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the 4-bit model with your custom LoRA weights applied
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_path,
            max_seq_length = 8192,
            dtype = None,
            load_in_4bit = True,
        )
        
        # Enable Unsloth's highly optimized inference mode
        FastLanguageModel.for_inference(self.model)
        
        # Initialize the BAAI dense embedding retriever
        self.retriever = StrataRetriever()

    def generate_answer(self, question, context):
        prompt = (
            f"You are Strata, a highly intelligent archaeological AI assistant.\n"
            f"First, attempt to answer the user's question using the Context provided below.\n"
            f"If the Context does not contain the answer, use your broad general knowledge to provide a helpful, informative response, but politely note that the information comes from your general knowledge rather than the Strata database.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
        )
        
        messages = [
            {"role": "system", "content": "You are a helpful and expert AI assistant."},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=400, temperature=0.4)
            
        return self.tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
