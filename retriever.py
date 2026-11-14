import torch
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class StrataRetriever:
    def __init__(self, index_path='index/strata_index.pkl', embed_model_name='all-MiniLM-L6-v2'):
        print("Loading Strata Index...")
        with open(index_path, 'rb') as f:
            self.index = pickle.load(f)
            
        self.documents = self.index['documents']
        
        # Convert numpy embeddings back to PyTorch tensors and move to RTX 5060 for fast similarity math
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.doc_embeddings = torch.tensor(self.index['embeddings']).to(self.device)
        
        self.tfidf_vectorizer = self.index['tfidf_vectorizer']
        self.tfidf_matrix = self.index['tfidf_matrix']
        
        print(f"Loading embedding model: {embed_model_name}...")
        self.embedder = SentenceTransformer(embed_model_name)
        
    def search(self, query, top_k=2):
        # 1. Dense Semantic Search (PyTorch)
        # Understands concepts like "stone tool making" = "lithic assemblage"
        query_emb = self.embedder.encode(query, convert_to_tensor=True).to(self.device)
        
        # Compute cosine similarity using PyTorch tensor operations
        dense_scores = torch.nn.functional.cosine_similarity(query_emb.unsqueeze(0), self.doc_embeddings)
        dense_scores_np = dense_scores.cpu().numpy()
        
        # 2. Sparse Keyword Search (Sklearn TF-IDF)
        # Finds exact matches for codes like "1045" or "OXA-1234"
        query_tfidf = self.tfidf_vectorizer.transform([query])
        sparse_scores = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]
        
        # 3. Hybrid Scoring (Combining PyTorch understanding with Sklearn precision)
        # We give a 1.5x multiplier to sparse scores to heavily reward exact ID/typology matches
        hybrid_scores = dense_scores_np + (sparse_scores * 1.5)
        
        # Get top K results
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'hybrid_score': hybrid_scores[idx],
                'dense_score': dense_scores_np[idx],
                'sparse_score': sparse_scores[idx],
                'cluster': self.index['cluster_labels'][idx]
            })
            
        return results

if __name__ == "__main__":
    retriever = StrataRetriever()
    
    # Test 1: Conceptual search (PyTorch should do the heavy lifting here)
    print("\n--- Test 1: Conceptual Query ---")
    query1 = "Evidence of stone tool making"
    print(f"Query: '{query1}'")
    results1 = retriever.search(query1, top_k=1)
    for r in results1:
        print(f"Match: {r['document']}")
        print(f"(Scores -> Hybrid: {r['hybrid_score']:.3f} | Dense: {r['dense_score']:.3f} | Sparse: {r['sparse_score']:.3f})")
        
    # Test 2: Exact identifier search (Sklearn should do the heavy lifting here)
    print("\n--- Test 2: Exact Typology/ID Query ---")
    query2 = "What was found in context 1045?"
    print(f"Query: '{query2}'")
    results2 = retriever.search(query2, top_k=1)
    for r in results2:
        print(f"Match: {r['document']}")
        print(f"(Scores -> Hybrid: {r['hybrid_score']:.3f} | Dense: {r['dense_score']:.3f} | Sparse: {r['sparse_score']:.3f})")
