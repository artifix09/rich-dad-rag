from sentence_transformers import SentenceTransformer
from backend.core.config import settings
from backend.core.logger import logger


_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:

    global _model

    if _model is None:
        logger.info(
            f"Loading embedding model: {settings.EMBEDDING_MODEL}"
        )
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info("Embedding model loaded and cached")

    return _model


def embed_chunks(chunks: list[dict]) -> list[dict]:

    model = get_embedding_model()

    texts = [chunk["text"] for chunk in chunks]

    logger.info(
        f"Embedding {len(texts)} chunks | "
        f"model: {settings.EMBEDDING_MODEL}"
    )

    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    logger.info(
        f"Embedding complete | "
        f"vector shape: {vectors.shape} | "
        f"dimension: {vectors.shape[1]}"
    )

    if vectors.shape[1] != settings.EMBEDDING_DIMENSION:
        raise ValueError(
            f"Embedding dimension mismatch. "
            f"Expected {settings.EMBEDDING_DIMENSION}, "
            f"got {vectors.shape[1]}"
        )

    embedded_chunks = []

    for i, chunk in enumerate(chunks):
        embedded_chunk = {
            **chunk,
            "embedding": vectors[i].tolist(),
        }
        embedded_chunks.append(embedded_chunk)

    logger.info(f"Attached embeddings to {len(embedded_chunks)} chunks")

    return embedded_chunks


def embed_query(query: str) -> list[float]:

    model = get_embedding_model()

    logger.debug(f"Embedding query: {query[:80]}...")

    vector = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return vector.tolist()