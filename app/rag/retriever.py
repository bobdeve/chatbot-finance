import faiss
import numpy as np
import pickle

from sentence_transformers import SentenceTransformer  # Or your Hugging Face API function

# Load FAISS index and metadata
def load_faiss_index(index_path, metadata_path):
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

# Embed user query
def embed_query(query, model=None):
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode([query])[0]

# Perform similarity search
def retrieve_top_k(query, index, metadata, k=5, model=None):
    query_vec = embed_query(query, model).astype("float32").reshape(1, -1)
    D, I = index.search(query_vec, k)
    return [metadata[i] for i in I[0]]
