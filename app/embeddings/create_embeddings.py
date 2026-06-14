from pathlib import Path

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


PDF_PATH = Path(
    "data/raw_documents/Contract document.pdf"
)


def load_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def create_chunks(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_text(text)


def main():

    print("Loading document...")

    text = load_pdf_text(PDF_PATH)

    chunks = create_chunks(text)

    print(f"Chunks created: {len(chunks)}")

    print("\nLoading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Creating vector database...")

    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="data/vectorstores/test_store"
    )

    print("\nVector database created successfully")

    print(f"Stored chunks: {len(chunks)}")


if __name__ == "__main__":
    main()