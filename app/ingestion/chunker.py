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


def create_chunks(
    text,
    chunk_size=1000,
    chunk_overlap=200,
    filename="unknown_document"
):
    import re

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    raw_chunks = splitter.split_text(text)

    clean_filename = re.sub(r"[^a-zA-Z0-9_]", "_", filename)

    chunks = []
    for idx, chunk_text in enumerate(raw_chunks):
        chunk_id = f"{clean_filename}_chunk_{idx:04d}"
        chunks.append({
            "chunk_id": chunk_id,
            "document": filename,
            "chunk_index": idx,
            "text": chunk_text
        })

    return chunks


def main():

    text = load_pdf_text(PDF_PATH)

    chunks = create_chunks(text, filename=PDF_PATH.name)

    print("\n" + "=" * 60)
    print(f"Document: {PDF_PATH.name}")
    print(f"Total Characters: {len(text)}")
    print(f"Number of Chunks: {len(chunks)}")

    print("\nFIRST CHUNK")
    print("-" * 60)
    print(chunks[0]["text"][:500])

    print("\nLAST CHUNK")
    print("-" * 60)
    print(chunks[-1]["text"][:500])


if __name__ == "__main__":
    main()