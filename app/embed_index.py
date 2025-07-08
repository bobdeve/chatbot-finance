import os
import numpy as np
import pandas as pd
import pickle
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Load your cleaned dataframe (limit to 50,000)
df_main = pd.read_csv("../data/filtered_complaints.csv").dropna(subset=["Consumer complaint narrative"])
df_main = df_main.head(50000)

# Extract necessary columns
texts = df_main["Consumer complaint narrative"].tolist()
complaint_ids = df_main["Complaint ID"].astype(str).tolist()
products = df_main["Product"].tolist()
dates = df_main["Date received"].tolist()

# --- Step 1: Chunking ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "]
)

all_chunks = []
chunk_metadata = []
chunk_ids = []

for i, (text, cid, product, date) in enumerate(zip(texts, complaint_ids, products, dates)):
    chunks = text_splitter.split_text(text)
    for j, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        chunk_metadata.append({
            "product": product,
            "date_received": date,
            "source_id": cid,
            "chunk_id": f"{cid}_{j}"
        })
        chunk_ids.append(f"{cid}_{j}")

print(f"✅ Total chunks created: {len(all_chunks)}")

# --- Step 2: Embedding ---
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(all_chunks, batch_size=64, show_progress_bar=True)

# --- Step 3: Initialize ChromaDB ---
client = chromadb.Client()
try:
    client.delete_collection("complaints_chunks")
except:
    pass

collection = client.create_collection(
    name="complaints_chunks",
    metadata={"description": "Chunked and embedded consumer complaint narratives"}
)

# --- Step 4: Add to ChromaDB ---
collection.add(
    documents=all_chunks,
    embeddings=embeddings,
    metadatas=chunk_metadata,
    ids=chunk_ids
)

print(f"✅ Added {collection.count()} chunks to ChromaDB")

# --- Step 5: Save to disk ---
# os.makedirs("vector_store", exist_ok=True)

np.save("../data/new/embeddings.npy", embeddings)

# with open("vector_store/chunk_texts.pkl", "wb") as f:
#     pickle.dump(all_chunks, f)

# with open("vector_store/chunk_metadata.pkl", "wb") as f:
#     pickle.dump(chunk_metadata, f)

# with open("vector_store/chunk_ids.pkl", "wb") as f:
#     pickle.dump(chunk_ids, f)

print("✅ Vector store and chunk data saved to 'vector_store/' directory")
