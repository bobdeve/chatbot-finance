from sentence_transformers import SentenceTransformer

# Load the model once
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text_chunks(text_chunks):
    texts = [chunk["chunk_text"] for chunk in text_chunks]
    vectors = model.encode(texts, show_progress_bar=True, batch_size=32)

    embedded = []
    for chunk, vector in zip(text_chunks, vectors):
        embedded.append({
            "doc_id": chunk["doc_id"],
            "product": chunk.get("product", ""),
            "chunk_index": chunk["chunk_index"],
            "chunk_text": chunk["chunk_text"],
            "embedding": vector.tolist()
        })
    return embedded
