import re
from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from backend.core.config import settings
from backend.core.logger import logger


def build_full_text(pages: list[dict]) -> str:

    full_text = ""

    for page in pages:
        marker = f"[PAGE_{page['page_number']}]\n"
        full_text += marker + page["text"] + "\n\n"

    return full_text


def extract_page_numbers(chunk_text: str) -> list[int]:

    matches = re.findall(r"\[PAGE_(\d+)\]", chunk_text)
    return [int(m) for m in matches] if matches else []


def clean_chunk_text(text: str) -> str:

    text = re.sub(r"\[PAGE_\d+\]\n?", "", text)
    return text.strip()


def apply_overlap(chunks: list[dict]) -> list[dict]:

    if not chunks:
        return chunks

    for i in range(1, len(chunks)):
        prev_text = chunks[i - 1]["text"]
        overlap_chars = int(len(prev_text) * settings.CHUNK_OVERLAP_PERCENT)

        if overlap_chars > 0:
            overlap_snippet = prev_text[-overlap_chars:]
            chunks[i]["text"] = overlap_snippet + " " + chunks[i]["text"]
            chunks[i]["char_count"] = len(chunks[i]["text"])
            chunks[i]["token_estimate"] = chunks[i]["char_count"] // 4

    return chunks


def chunk_pages(pages: list[dict]) -> list[dict]:

    logger.info("Loading embedding model for semantic chunking...")

    embed_model = HuggingFaceEmbedding(
        model_name=settings.EMBEDDING_MODEL
    )

    splitter = SemanticSplitterNodeParser(
        embed_model=embed_model,
        breakpoint_percentile_threshold=95,
        buffer_size=1,
    )

    source_file = pages[0]["source"]
    full_text = build_full_text(pages)

    logger.info(
        f"Built full document | "
        f"chars: {len(full_text)} | "
        f"pages: {len(pages)}"
    )

    document = Document(
        text=full_text,
        metadata={"source": source_file}
    )

    nodes = splitter.get_nodes_from_documents([document])

    logger.info(f"Semantic splitter produced {len(nodes)} raw nodes")

    chunks = []
    last_known_page = 1

    for i, node in enumerate(nodes):

        raw_text = node.get_content()
        page_nums = extract_page_numbers(raw_text)
        clean_text = clean_chunk_text(raw_text)

        if not clean_text:
            logger.debug(f"Skipping empty node {i}")
            continue

        if page_nums:
            last_known_page = page_nums[-1]
        else:
            page_nums = [last_known_page]

        char_count = len(clean_text)
        token_estimate = char_count // 4

        chunk = {
            "chunk_id": f"chunk_{i:04d}",
            "text": clean_text,
            "source": source_file,
            "page_numbers": page_nums,
            "char_count": char_count,
            "token_estimate": token_estimate,
        }

        chunks.append(chunk)

        logger.debug(
            f"Chunk {i:04d} | "
            f"pages: {page_nums} | "
            f"chars: {char_count} | "
            f"tokens~: {token_estimate}"
        )

    chunks = apply_overlap(chunks)

    total_chunks = len(chunks)
    avg_chars = sum(c["char_count"] for c in chunks) // total_chunks

    logger.info(
        f"Chunking complete | "
        f"total chunks: {total_chunks} | "
        f"avg chars/chunk: {avg_chars} | "
        f"avg tokens~/chunk: {avg_chars // 4}"
    )

    return chunks