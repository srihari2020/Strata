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
            max_seq_length = 4096,
            dtype = None,
            load_in_4bit = True,
        )
        
        # Enable Unsloth's highly optimized inference mode
        FastLanguageModel.for_inference(self.model)
        print(f"Total Parameters: {self.model.num_parameters():,}")
        memory_gb = self.model.get_memory_footprint() / (1024**3)
        print(f"Model Size in VRAM: {memory_gb:.2f} GB")
        
        self.retriever = StrataRetriever()

    def ask(self, query):
        print(f"\nSearching index for: '{query}'...")
        results = self.retriever.search(query, top_k=2)
        context_text = "\n".join([f"- {r['document']}" for r in results])
        print(f"Retrieved Context:\n{context_text}\n")
        
        user_prompt = f"{query}\nContext: {context_text}"
        
        messages = [
            {"role": "system", "content": "You are an expert archaeological AI assistant."},
            {"role": "user", "content": user_prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        print("Synthesizing answer with custom 7B model...")
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=150,
                temperature=0.3,
                do_sample=True
            )
            
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

if __name__ == "__main__":
    rag = StrataGenerator()
    question = "What is the earliest widely accepted evidence for leprosy according to the text?"
    
    print("\n==================================")
    print(f"QUESTION: {question}")
    print("==================================")
    
    answer = rag.ask(question)
    
    print("==================================")
    print(f"STRATA AI ANSWER:\n{answer}")
    print("==================================")
