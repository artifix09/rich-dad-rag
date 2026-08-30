from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PayloadSchemaType,
)
from backend.core.config import settings
from backend.core.logger import logger


_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:

    global _client

    if _client is None:
        logger.info(f"Connecting to Qdrant at: {settings.QDRANT_URL}")

        _client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None,
        )

        logger.info("Qdrant client connected")

    return _client


def create_collection(recreate: bool = False) -> None:

    client = get_qdrant_client()
    collection = settings.QDRANT_COLLECTION

    exists = client.collection_exists(collection)

    if exists and not recreate:
        logger.info(f"Collection '{collection}' already exists — skipping creation")
        return

    if exists and recreate:
        logger.warning(f"Deleting existing collection '{collection}'")
        client.delete_collection(collection)

    logger.info(f"Creating collection '{collection}'")

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(
            size=settings.EMBEDDING_DIMENSION,
            distance=Distance.COSINE,
        ),
        hnsw_config=HnswConfigDiff(
            m=16,
            ef_construct=100,
            full_scan_threshold=10000,
        ),
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=20000,
        ),
    )

    client.create_payload_index(
        collection_name=collection,
        field_name="source",
        field_schema=PayloadSchemaType.KEYWORD,
    )

    client.create_payload_index(
        collection_name=collection,
        field_name="page_numbers",
        field_schema=PayloadSchemaType.INTEGER,
    )

    logger.info(f"Collection '{collection}' created with HNSW + payload indexes")


def upsert_chunks(embedded_chunks: list[dict]) -> None:

    client = get_qdrant_client()
    collection = settings.QDRANT_COLLECTION

    logger.info(
        f"Preparing {len(embedded_chunks)} points for upsert "
        f"into '{collection}'"
    )

    points = []

    for i, chunk in enumerate(embedded_chunks):
        point = PointStruct(
            id=i,
            vector=chunk["embedding"],
            payload={
                "chunk_id":     chunk["chunk_id"],
                "text":         chunk["text"],
                "source":       chunk["source"],
                "page_numbers": chunk["page_numbers"],
                "char_count":   chunk["char_count"],
                "token_estimate": chunk["token_estimate"],
            },
        )
        points.append(point)

    batch_size = 50
    total_batches = (len(points) + batch_size - 1) // batch_size

    for batch_num in range(total_batches):
        start = batch_num * batch_size
        end = start + batch_size
        batch = points[start:end]

        client.upsert(
            collection_name=collection,
            points=batch,
        )

        logger.debug(
            f"Upserted batch {batch_num + 1}/{total_batches} "
            f"({len(batch)} points)"
        )

    logger.info(
        f"Upsert complete | "
        f"total points in Qdrant: {len(embedded_chunks)}"
    )


def get_collection_info() -> dict:

    client = get_qdrant_client()

    info = client.get_collection(settings.QDRANT_COLLECTION)

    return {
        "name":         settings.QDRANT_COLLECTION,
        "total_points": info.points_count,
        "status":       str(info.status),
        "dimension":    info.config.params.vectors.size,
    }