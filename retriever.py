import torch
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class StrataRetriever:
    def __init__(self, index_path='index/strata_index.pkl', embed_model_name='BAAI/bge-large-en-v1.5'):
        print("Loading Strata Index...")
        with open(index_path, 'rb') as f:
            self.index = pickle.load(f)
            
        self.documents = self.index['documents']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.doc_embeddings = torch.tensor(self.index['embeddings']).to(self.device)
        
        self.tfidf_vectorizer = self.index['tfidf_vectorizer']
        self.tfidf_matrix = self.index['tfidf_matrix']
        
        print(f"Loading Qwen embedding model: {embed_model_name} in fp16...")
        self.embedder = SentenceTransformer(
            embed_model_name,
            model_kwargs={"torch_dtype": torch.float16},
            trust_remote_code=False
        )
        
    def search(self, query, top_k=2):
        query_emb = self.embedder.encode(query, convert_to_tensor=True).to(self.device)
        dense_scores = torch.nn.functional.cosine_similarity(query_emb.unsqueeze(0), self.doc_embeddings)
        dense_scores_np = dense_scores.cpu().numpy()
        
        query_tfidf = self.tfidf_vectorizer.transform([query])
        sparse_scores = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]
        
        hybrid_scores = dense_scores_np + (sparse_scores * 1.5)
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
