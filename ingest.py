import pymupdf  # Replaced fitz to clear the deprecation warning
import pickle
import os
import glob     # Added the missing import to scan the data/ folder

def extract_and_chunk_pdf(pdf_path, chunk_size=350, overlap=50):
    print(f"Opening PDF: {pdf_path}")
    doc = pymupdf.open(pdf_path)
    
    all_text = ""
    for page in doc:
        all_text += page.get_text("text") + " "
        
    words = all_text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.split()) > 50:
            chunks.append(chunk)
            
    print(f"Extracted {len(chunks)} chunks from {os.path.basename(pdf_path)}.")
    return chunks

if __name__ == "__main__":
    os.makedirs("index", exist_ok=True)
    all_documents = []
    
    # 1. Process all PDFs
    pdf_files = glob.glob("data/*.pdf")
    print(f"Found {len(pdf_files)} PDF(s).")
    for pdf_path in pdf_files:
        all_documents.extend(extract_and_chunk_pdf(pdf_path))
        
    # 2. Process all scraped Text files
    txt_files = glob.glob("data/*.txt")
    print(f"Found {len(txt_files)} scraped text file(s).")
    for txt_path in txt_files:
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
            words = text.split()
            chunks = []
            # Sliding window for text files
            for i in range(0, len(words), 350 - 50):
                chunk = " ".join(words[i:i + 350])
                if len(chunk.split()) > 50:
                    chunks.append(chunk)
            all_documents.extend(chunks)
            print(f"Extracted {len(chunks)} chunks from {os.path.basename(txt_path)}.")
        
    with open('index/raw_documents.pkl', 'wb') as f:
        pickle.dump(all_documents, f)
        
    print(f"Total {len(all_documents)} chunks saved. Run indexer.py next to update ChromaDB.")
