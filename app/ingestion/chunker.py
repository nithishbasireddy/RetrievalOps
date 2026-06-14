from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


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
        chunk_overlap=200,
        length_function=len
    )

    chunks = splitter.split_text(text)

    return chunks


def main():

    text = load_pdf_text(PDF_PATH)

    chunks = create_chunks(text)

    print("\n" + "=" * 60)
    print(f"Document: {PDF_PATH.name}")
    print(f"Total Characters: {len(text)}")
    print(f"Number of Chunks: {len(chunks)}")

    print("\nFIRST CHUNK")
    print("-" * 60)
    print(chunks[0][:500])

    print("\nLAST CHUNK")
    print("-" * 60)
    print(chunks[-1][:500])


if __name__ == "__main__":
    main()