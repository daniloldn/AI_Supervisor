import faiss
import numpy as np
from pathlib import Path
import pickle


def build_faiss_index(embedded_chunks: list[dict], index_path="vectorstore/faiss.index"):
    """
    Build and save FAISS index from embedded chunks.
    """
    Path("vectorstore").mkdir(exist_ok=True)

    embeddings = np.array([chunk["embedding"] for chunk in embedded_chunks]).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, index_path)

    print(f"FAISS index saved to {index_path}")


def save_metadata(embedded_chunks, metadata_path="vectorstore/metadata.pkl"):
    """
    Save chunk metadata without embeddings.
    """
    clean_chunks = []

    for chunk in embedded_chunks:
        clean_chunks.append({
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "text": chunk["text"]
        })

    with open(metadata_path, "wb") as f:
        pickle.dump(clean_chunks, f)

    print(f"Metadata saved to {metadata_path}")