



def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:

    """
    split text into overlapping chunks.

    Args:
        text: The full document text
        chunk_size: Maximum size of each chunk
        overlap: Number of characters shared between consecutive chunks

    Returns:
        List of text chunks
       """
    
    if not text:
        return []
    
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)



    return chunks

def chunk_docs(documents: list[dict], chunk_size: int = 800, overlap: int = 150) -> list[dict]:

    """
    Chunk a list of loaded documents and preserve metadata.

    Returns:
        A list of chunk dictionaries like:
        [
            {"chunk_id": 0, "source": "...", "text": "..."},
            ...
        ]
    """

    all_chunks = []
    chunk_id = 0

    for doc in documents:
        source = doc["source"]
        text = doc["text"]

        chunks = chunk_text(text, chunk_size, overlap)

        for chunk in chunks:
            all_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source": source,
                    "text": chunk
                }
            )
            chunk_id += 1

    return all_chunks
