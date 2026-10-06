"""
Text Splitter module for 'Ask My Notes'.
Splits extracted document pages into small, context-preserving chunks
(around 500 words each by default), maintaining exact document and page metadata.
"""

from typing import List, Dict, Any
import re


def split_text_into_chunks(
    text: str,
    target_words: int = 500,
    overlap_words: int = 60,
) -> List[str]:
    """
    Splits text into chunks of roughly `target_words` words with `overlap_words` overlap.
    Attempts to split along paragraph and sentence boundaries before raw word breaks.
    """
    if not text or not text.strip():
        return []

    # First, split into paragraph blocks
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    # Break paragraphs into sentence or sub-paragraph units
    units: List[str] = []
    for p in paragraphs:
        # If paragraph is long, split by sentence boundaries (.?! followed by space)
        sentences = re.split(r"(?<=[.!?])\s+", p)
        for s in sentences:
            if s.strip():
                units.append(s.strip())

    chunks: List[str] = []
    current_chunk_words: List[str] = []
    current_word_count = 0

    for unit in units:
        unit_words = unit.split()
        unit_len = len(unit_words)

        # If a single sentence exceeds target_words, break it into word segments
        if unit_len > target_words:
            start = 0
            while start < unit_len:
                end = min(start + target_words, unit_len)
                sub_slice = unit_words[start:end]
                chunks.append(" ".join(sub_slice))
                start += max(1, target_words - overlap_words)
            continue

        if current_word_count + unit_len <= target_words:
            current_chunk_words.append(unit)
            current_word_count += unit_len
        else:
            # Finish current chunk
            if current_chunk_words:
                chunks.append(" ".join(current_chunk_words))

            # Start new chunk with overlap from the previous words
            if overlap_words > 0 and current_chunk_words:
                all_prev_words = (" ".join(current_chunk_words)).split()
                overlap_slice = all_prev_words[-overlap_words:] if len(all_prev_words) > overlap_words else all_prev_words
                current_chunk_words = [" ".join(overlap_slice), unit]
                current_word_count = len(overlap_slice) + unit_len
            else:
                current_chunk_words = [unit]
                current_word_count = unit_len

    if current_chunk_words:
        chunks.append(" ".join(current_chunk_words))

    return chunks


def chunk_document_pages(
    pages: List[Dict[str, Any]],
    target_words: int = 500,
    overlap_words: int = 60,
) -> List[Dict[str, Any]]:
    """
    Splits a list of page dicts into chunks, retaining document name and page number.

    Args:
        pages: List of page dictionaries from pdf_loader.extract_pages_from_pdf
        target_words: Target word count per chunk (~500 words)
        overlap_words: Word overlap between consecutive chunks

    Returns:
        List of chunk dicts:
        [
            {
                "chunk_id": "doc.pdf_p1_c0",
                "text": "...",
                "document_name": "doc.pdf",
                "page_number": 1,
                "total_pages": 4,
                "word_count": 482,
                "char_count": 3120,
            },
            ...
        ]
    """
    all_chunks: List[Dict[str, Any]] = []

    for page in pages:
        page_text = page.get("text", "")
        doc_name = page.get("document_name", "unknown")
        page_num = page.get("page_number", 1)
        total_pages = page.get("total_pages", 1)

        if not page_text or len(page_text.split()) < 10:
            # Ignore empty or near-empty pages (blank spacers)
            continue

        raw_chunks = split_text_into_chunks(
            page_text,
            target_words=target_words,
            overlap_words=overlap_words,
        )

        for chunk_idx, chunk_str in enumerate(raw_chunks):
            words = chunk_str.split()
            chunk_id = f"{doc_name}_p{page_num}_c{chunk_idx}"

            all_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk_str,
                    "document_name": doc_name,
                    "page_number": page_num,
                    "total_pages": total_pages,
                    "word_count": len(words),
                    "char_count": len(chunk_str),
                }
            )

    return all_chunks
