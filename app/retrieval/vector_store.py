from langchain_community.vectorstores import Chroma


def create_vectorstore(
    chunks,
    embeddings,
    persist_directory
):

    texts = [c["text"] for c in chunks]
    metadatas = [
        {
            "chunk_id": c["chunk_id"],
            "document": c["document"],
            "chunk_index": c["chunk_index"]
        }
        for c in chunks
    ]
    ids = [c["chunk_id"] for c in chunks]

    return Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        ids=ids,
        persist_directory=persist_directory
    )


def load_vectorstore(
    embeddings,
    persist_directory
):

    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )