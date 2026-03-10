from src.loaders import load_pdf
from src.chunking import chunk_docs
from src.embedding import embed_chunks
from src.utils import save_chunks_metadata
import pickle
from pathlib import Path

def main():

    if Path("vectorstore/chunks_metadata.pkl").exists():
        print("Embeddings already exist. Skipping embedding step.")
        return None
    
    #embedding
    documents = load_pdf("data/raw")
    chunks = chunk_docs(documents, chunk_size=800, overlap=150)

    print(f"Created {len(chunks)} chunks.")
    print("Embedding first 3 chunks...\n")

    embedded = embed_chunks(chunks[:3])

    print("Saving chunk metadata...")
    save_chunks_metadata(embedded)
    print("Saved.")

    for c in embedded:
        print("=" * 50)
        print(f"Chunk ID: {c['chunk_id']}")
        print(f"Embedding length: {len(c['embedding'])}")
        print(f"First 5 dims: {c['embedding'][:5]}")


if __name__ == "__main__":
    main()