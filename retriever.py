import chromadb
from sentence_transformers import SentenceTransformer

class StrataRetriever:
    def __init__(self):
        print("Connecting to local ChromaDB cluster...")
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_collection(name="strata_global")
        
        self.model = SentenceTransformer('BAAI/bge-large-en-v1.5')

    def search(self, query, top_k=7):
        query_embedding = self.model.encode(query, normalize_embeddings=True).tolist()
        
        # Lightning-fast vector similarity search on disk
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Safely return just the flat list of text chunks
        return results['documents'][0]
