from src.retrieval import load_faiss_index, load_metadata

def main():
    index = load_faiss_index()
    metadata = load_metadata()

    print("FAISS total vectors:", index.ntotal)
    print("Metadata rows:", len(metadata))

    for row in metadata[:10]:
        print(row["chunk_id"], row["source"], len(row["text"]))

if __name__ == "__main__":
    main()