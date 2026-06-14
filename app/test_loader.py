from app.ingestion.document_loader import load_documents

documents = load_documents()

print(f"Documents Loaded: {len(documents)}")

print("\nFirst Document:")

print(documents[0]["filename"])
print(len(documents[0]["text"]))