from langchain_community.vectorstores import Chroma


def create_vectorstore(
    chunks,
    embeddings,
    persist_directory
):

    return Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )