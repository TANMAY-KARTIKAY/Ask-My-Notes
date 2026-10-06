"""
PDF Loader module for 'Ask My Notes'.
Extracts text page-by-page from PDF files using pypdf,
retaining page numbers and document metadata.
"""

from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any, Union
import re
from pypdf import PdfReader


def clean_text(text: str) -> str:
    """
    Cleans raw extracted PDF text:
    - Normalizes multiple spaces and tabs to single spaces
    - Normalizes consecutive newlines
    - Strips leading/trailing whitespace
    """
    if not text:
        return ""
    # Normalize carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace non-breaking spaces
    text = text.replace("\xa0", " ")
    # Replace sequences of 3+ newlines with double newline (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Replace tabs and multiple horizontal spaces with single space
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_pages_from_pdf(
    file_source: Union[str, Path, BytesIO, bytes],
    filename: str = "document.pdf",
) -> List[Dict[str, Any]]:
    """
    Extracts text from each page of a PDF document.

    Args:
        file_source: File path, BytesIO stream, or raw bytes of the PDF.
        filename: Original name of the PDF file for citation attribution.

    Returns:
        List of dicts, each representing a page:
        [
            {
                "document_name": filename,
                "page_number": 1,        # 1-indexed
                "total_pages": total,
                "text": "Cleaned page text...",
                "char_count": 1234,
                "word_count": 210
            },
            ...
        ]
    """
    if isinstance(file_source, bytes):
        stream = BytesIO(file_source)
    elif isinstance(file_source, (str, Path)):
        stream = open(file_source, "rb")
    else:
        stream = file_source

    try:
        reader = PdfReader(stream)
        total_pages = len(reader.pages)
        pages_data = []

        for page_idx, page in enumerate(reader.pages):
            page_num = page_idx + 1  # 1-indexed for human readability
            try:
                raw_text = page.extract_text() or ""
            except Exception as e:
                raw_text = f"[Warning: Could not extract text from page {page_num}: {e}]"

            cleaned = clean_text(raw_text)

            # Keep page even if sparse, but mark word count
            words = cleaned.split() if cleaned else []
            pages_data.append(
                {
                    "document_name": filename,
                    "page_number": page_num,
                    "total_pages": total_pages,
                    "text": cleaned,
                    "char_count": len(cleaned),
                    "word_count": len(words),
                }
            )

        return pages_data
    finally:
        # If we opened a file from string/Path, close it
        if isinstance(file_source, (str, Path)) and hasattr(stream, "close"):
            stream.close()
