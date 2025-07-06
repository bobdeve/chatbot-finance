import faiss
import numpy as np
import pickle
import os

def build_faiss_index(embedded_chunks):
    vectors = np.array([chunk["embedding"] for chunk in embedded_chunks]).astype("float32")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    metadata = [
        {
            "doc_id": chunk["doc_id"],
            "product": chunk.get("product", ""),
            "chunk_index": chunk["chunk_index"],
            "chunk_text": chunk["chunk_text"]
        }
        for chunk in embedded_chunks
    ]

    return index, metadata



def save_faiss_index(index, metadata, base_path="faiss_index"):
    faiss.write_index(index, base_path + ".bin")
    with open(base_path + "_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
