from app.pipeline.builder import build_pipeline
from app.embeddings.embedding_manager import (
    get_embedding_model
)


config = build_pipeline(
    "dense_e5"
)

embedding_model = get_embedding_model(
    config["embedding_model"]
)

print("\nEmbedding model loaded successfully")