import fitz  # PyMuPDF
import pickle
import os

def extract_and_chunk_pdf(pdf_path, chunk_size=350, overlap=50):
    print(f"Opening PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    
    all_text = ""
    for page in doc:
        all_text += page.get_text("text") + " "
        
    words = all_text.split()
    chunks = []
    
    # Sliding window chunking
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.split()) > 50:  # Ignore tiny trailing chunks
            chunks.append(chunk)
            
    print(f"Extracted {len(chunks)} overlapping text chunks strictly under the limit.")
    return chunks

if __name__ == "__main__":
    os.makedirs("index", exist_ok=True)
    
    # 1. Extract token-aware chunks
    documents = extract_and_chunk_pdf("data/report.pdf")
    
    # 2. Save raw chunks temporarily for the indexer to pick up
    with open('index/raw_documents.pkl', 'wb') as f:
        pickle.dump(documents, f)
    
    print("Chunks saved. Run indexer.py next to rebuild the embeddings.")