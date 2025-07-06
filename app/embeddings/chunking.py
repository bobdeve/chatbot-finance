from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_texts(texts_dict, chunk_size=800, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []
    for doc_id, text in texts_dict.items():
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "doc_id": doc_id,
                "chunk_index": i,
                "chunk_text": chunk
            })
    return all_chunks
