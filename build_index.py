from src.loaders import load_pdf


def main():
    documents = load_pdf("data/raw")

    print(f"Loaded {len(documents)} documents.\n")

    for doc in documents:
        print(f"Source: {doc['source']}")
        print(f"Character count: {len(doc['text'])}")
        print("Preview:")
        print("\n")


if __name__ == "__main__":
    main()
