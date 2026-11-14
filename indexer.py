import torch
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import os

class StrataIndexer:
    def __init__(self, embed_model_name='all-MiniLM-L6-v2', num_clusters=3):
        print(f"Loading embedding model: {embed_model_name}...")
        # Automatically uses GPU if available
        self.embedder = SentenceTransformer(embed_model_name)
        self.num_clusters = num_clusters
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        
    def build_index(self, documents):
        print(f"Processing {len(documents)} documents...")
        
        # 1. PyTorch Semantic Embeddings
        print("Generating dense embeddings on GPU...")
        embeddings = self.embedder.encode(documents, convert_to_tensor=True)
        embeddings_np = embeddings.cpu().numpy()
        
        # 2. Sklearn Semantic Clustering
        print("Clustering documents by archaeological theme...")
        cluster_labels = self.kmeans.fit_predict(embeddings_np)
        
        # 3. Sklearn TF-IDF (Sparse matrix for exact keyword/typology matches)
        print("Building TF-IDF sparse matrix...")
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        
        # 4. Save everything to disk for the Retriever to use later
        index_data = {
            'documents': documents,
            'embeddings': embeddings_np,
            'cluster_labels': cluster_labels,
            'kmeans_model': self.kmeans,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'tfidf_matrix': tfidf_matrix
        }
        
        os.makedirs('index', exist_ok=True)
        with open('index/strata_index.pkl', 'wb') as f:
            pickle.dump(index_data, f)
            
        print("Index successfully built and saved to 'index/strata_index.pkl'")
        
        # Preview the clusters
        for i in range(self.num_clusters):
            print(f"\n--- Cluster {i} ---")
            cluster_docs = [documents[j] for j in range(len(documents)) if cluster_labels[j] == i]
            for doc in cluster_docs[:2]: # Show first 2 of each cluster
                print(f"- {doc}")

if __name__ == "__main__":
    # Synthetic archaeological dataset
    sample_data = [
        "Context [1045] revealed a dark grey silty clay containing fragmented Terra Sigillata (Dragendorff 37).",
        "Trench 4 yielded significant quantities of Samian ware pottery dating to the late 1st century AD.",
        "The lithic assemblage consists primarily of Mesolithic microblades struck from local flint nodules.",
        "Flint cores and debitage found in context [2001] suggest localized tool manufacturing.",
        "Radiocarbon dating of the human mandible (Sample OXA-1234) suggests a late Neolithic burial practice.",
        "Osteological analysis of the remains in Grave 5 indicates severe joint disease, likely osteoarthritis."
    ]
    
    indexer = StrataIndexer()
    indexer.build_index(sample_data)
