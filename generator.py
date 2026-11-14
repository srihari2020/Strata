import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from retriever import StrataRetriever

class StrataGenerator:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        print(f"Loading LLM: {model_name} onto RTX 5060...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Using torch.bfloat16 saves VRAM while maintaining accuracy
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        
        # Initialize our hybrid retriever
        self.retriever = StrataRetriever()

    def ask(self, query):
        print(f"\nSearching index for: '{query}'...")
        
        # 1. Retrieve the best context using our hybrid search
        results = self.retriever.search(query, top_k=2)
        
        # Combine the retrieved documents into a single context string
        context_text = "\n".join([f"- {r['document']}" for r in results])
        print(f"Retrieved Context:\n{context_text}\n")
        
        # 2. Construct the prompt using the model's chat template
            system_prompt = (
                "You are an expert archaeological AI assistant. Answer the user's question based strictly on the provided field notes. "
                "If the context provided does not contain the answer, say 'I cannot answer this based on the provided field notes.' "
                "Do not include information from the context that is irrelevant to the question."
            )
	user_prompt = f"Context (Field Notes):\n{context_text}\n\nQuestion: {query}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        # 3. Generate the answer (PyTorch)
        print("Synthesizing answer...")
        with torch.no_grad(): # Disable gradient calculation for faster inference
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=150,
                temperature=0.3, # Low temperature for factual consistency
                do_sample=True
            )
            
        # Slice the output to only return the newly generated tokens, not the prompt
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

if __name__ == "__main__":
    rag = StrataGenerator()
    
    question = "Based on the field notes, what types of pottery were found and in which contexts/trenches?"
    print("\n==================================")
    print(f"QUESTION: {question}")
    print("==================================")
    
    answer = rag.ask(question)
    
    print("==================================")
    print(f"STRATA AI ANSWER:\n{answer}")
    print("==================================")
