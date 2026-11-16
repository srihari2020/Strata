import pickle
import chromadb
from sentence_transformers import SentenceTransformer
import os

print("Loading raw chunked documents...")
with open('index/raw_documents.pkl', 'rb') as f:
    documents = pickle.load(f)

print("Initializing ChromaDB Vector Storage...")
# This creates a folder on your drive to store the global database permanently
client = chromadb.PersistentClient(path="./chroma_db")

# Clear the old collection if it exists so we have a clean slate
try:
    client.delete_collection(name="strata_global")
except:
    pass

collection = client.create_collection(name="strata_global")

print("Loading BAAI embedding model...")
model = SentenceTransformer('BAAI/bge-large-en-v1.5')

print(f"Generating embeddings for {len(documents)} chunks...")
embeddings = model.encode(documents, normalize_embeddings=True)

print("Writing to disk-backed vector database...")
# ChromaDB requires unique IDs for every chunk
ids = [f"archaeology_doc_{i}" for i in range(len(documents))]

collection.add(
    embeddings=embeddings.tolist(),
    documents=documents,
    ids=ids
)

print("Migration complete! Global infrastructure is now live in ./chroma_db")
