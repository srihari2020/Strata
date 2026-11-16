import pickle
import json
import torch
import re
from unsloth import FastLanguageModel

def generate_synthetic_data():
    print("Loading Strata Index...")
    with open('index/strata_index.pkl', 'rb') as f:
        index = pickle.load(f)
    documents = index['documents']

    print("Loading 7B LLM for Data Generation...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    qa_dataset = []

    print(f"\nGenerating Q&A pairs from {len(documents)} document chunks...")
    for i, doc in enumerate(documents):
        # Prompt the model to act as a dataset generator
        prompt = (
            "Read the following archaeological text and generate exactly 2 Question and Answer pairs based strictly on the facts presented.\n"
            "You MUST format your output exactly like this:\n"
            "Q1: [Your First Question]\n"
            "A1: [Your First Answer]\n"
            "Q2: [Your Second Question]\n"
            "A2: [Your Second Answer]\n\n"
            f"Text: {doc}\n"
        )
        
        messages = [
            {"role": "system", "content": "You are a dataset generator that strictly follows formatting rules."},
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.3)
        
        response = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        
        # Parse the model's output using Regex
        q1_match = re.search(r'Q1:\s*(.*?)\nA1:\s*(.*?)(?=\nQ2:|$)', response, re.DOTALL)
        q2_match = re.search(r'Q2:\s*(.*?)\nA2:\s*(.*?)(?=$)', response, re.DOTALL)
        
        if q1_match:
            qa_dataset.append({"instruction": q1_match.group(1).strip(), "input": doc, "output": q1_match.group(2).strip()})
        if q2_match:
            qa_dataset.append({"instruction": q2_match.group(1).strip(), "input": doc, "output": q2_match.group(2).strip()})
            
        print(f"Processed chunk {i+1}/{len(documents)}")

    # Save the generated dataset
    with open('data/archaeology_qa_full.jsonl', 'w') as f:
        for item in qa_dataset:
            f.write(json.dumps(item) + '\n')

    print(f"\nSuccessfully generated {len(qa_dataset)} Q&A pairs and saved to data/archaeology_qa_full.jsonl")

if __name__ == "__main__":
    generate_synthetic_data()
