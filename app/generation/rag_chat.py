from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


load_dotenv()


def main():

    query = "What are the responsibilities of the contractor?"

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory="data/vectorstores/test_store",
        embedding_function=embeddings
    )

    docs = vectorstore.similarity_search(
        query,
        k=3
    )

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile"
    )

    prompt = f"""
You are a contract analysis assistant.

Answer ONLY using the provided context.

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    print("\nQUESTION:")
    print(query)

    print("\nANSWER:")
    print(response.content)


if __name__ == "__main__":
    main()