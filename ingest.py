import pymupdf
import os
from indexer import StrataIndexer

def extract_and_chunk_pdf(pdf_path, min_chunk_length=100):
    print(f"Opening PDF: {pdf_path}")
    doc = pymupdf.open(pdf_path)
    chunks = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        
        # Split page into paragraphs
        paragraphs = text.split('\n\n')
        
        for p in paragraphs:
            # Clean up whitespace and line breaks within paragraphs
            clean_p = " ".join(p.split())
            
            # Filter out tiny snippets (like page numbers or tiny headers)
            if len(clean_p) > min_chunk_length:
                chunks.append(clean_p)
                
    print(f"Extracted {len(chunks)} viable text chunks from {len(doc)} pages.")
    return chunks

if __name__ == "__main__":
    # Ensure the data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Specify the PDF file you want to read
    pdf_filename = "data/report.pdf" 
    
    if not os.path.exists(pdf_filename):
        print(f"Error: Please place a PDF named 'report.pdf' inside the 'data/' folder.")
    else:
        documents = extract_and_chunk_pdf(pdf_filename)
        
        # Build the new index with the real data
        print("Initializing StrataIndexer...")
        indexer = StrataIndexer()
        indexer.build_index(documents)
