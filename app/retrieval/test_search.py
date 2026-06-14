from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def main():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory="data/vectorstores/test_store",
        embedding_function=embeddings
    )

    query = "What are the responsibilities of the contractor?"

    results = vectorstore.similarity_search(
        query,
        k=3
    )

    print("\nQUERY:")
    print(query)

    print("\nTOP 3 RESULTS\n")

    for i, doc in enumerate(results, start=1):

        print("=" * 60)
        print(f"Result {i}")
        print(doc.page_content[:700])
        print()


if __name__ == "__main__":
    main()