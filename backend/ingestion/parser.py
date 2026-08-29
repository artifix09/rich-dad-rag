import re 
import pymupdf as fitz
from pathlib import Path
from backend.core.config import settings
from backend.core.logger import logger


def clean_text(text : str) -> str:

    text = text.replace("\f" , " ")

    text = re.sub(r" +" , " " , text)

    text = re.sub(r"\n{3,}", "\n\n", text)

    text = re.sub(r"[^\x09\x0A\x20-\x7E]", "", text)


    return text.strip()


def extract_text_from_pdf(filename : str) -> list[dict]:

    pdf_path: Path = settings.RAW_DIR / filename

    if not pdf_path.exists():
        logger.error(f"pdf not found at path: {pdf_path}")
        raise FileNotFoundError(f"No PDF found at {pdf_path}")

    logger.info(f"opening pdf: {filename}")

    doc = fitz.open(str(pdf_path))
    pages = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        raw_text = page.get_text("text")
        cleaned = clean_text(raw_text)

        if not cleaned:
            logger.debug(f"skip empty pages : {page_index + 1}")
            continue


        pages.append({
            "page_number" : page_index + 1 ,
            "text" : cleaned,
            "source" : filename,
            "char_count" : len(cleaned)  
        })

        logger.debug(
            f"page {page_index + 1} ectracted |"
            f"chars: {len(cleaned)}"
        )

    doc.close()    

    logger.info(
        f"exraction complete :"
        f"total pages : {len(pages)}"
        f"total characters : {sum(p['char_count'] for p in pages)}"
    )

    return pages

