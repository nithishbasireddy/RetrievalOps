from app.ingestion.document_loader import load_documents
from app.ingestion.chunker import create_chunks


documents = load_documents()

first_doc = documents[0]

chunks = create_chunks(
    first_doc["text"],
    chunk_size=1000,
    chunk_overlap=200
)

print(f"Document: {first_doc['filename']}")
print(f"Chunks Created: {len(chunks)}")