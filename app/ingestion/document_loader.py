from pathlib import Path

from pypdf import PdfReader
from docx import Document

DOCUMENTS_PATH = Path("data/raw_documents")


def read_pdf(file_path):
    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n"

    return text


def read_docx(file_path):
    doc = Document(file_path)

    text = "\n".join(
        paragraph.text
        for paragraph in doc.paragraphs
    )

    return text


def load_documents():
    files = list(DOCUMENTS_PATH.glob("*"))

    print(f"\nFound {len(files)} documents\n")

    for file in files:

        if file.suffix.lower() == ".pdf":
            text = read_pdf(file)

        elif file.suffix.lower() == ".docx":
            text = read_docx(file)

        else:
            continue

        print("=" * 50)
        print(file.name)
        print(f"Characters: {len(text)}")
        print(text[:300])
        print("\n")


if __name__ == "__main__":
    load_documents()