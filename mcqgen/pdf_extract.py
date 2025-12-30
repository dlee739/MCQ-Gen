# mcqgen/pdf_extract.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF


def extract_pdf_pages_to_jsonl(
    context_pdf: Path,
    out_pages_jsonl: Path,
    logger: Optional[object] = None
) -> int:
    """
    Extract per-page text from a PDF and write JSONL:
      {"page_num": 1, "text": "..."}
    Returns total number of pages.
    """
    if not context_pdf.exists():
        raise FileNotFoundError(f"Context PDF not found: {context_pdf}")

    out_pages_jsonl.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(context_pdf))
    total_pages = doc.page_count
    if logger:
        logger.info("PDF opened: %s (pages=%d)", context_pdf.name, total_pages)

    with out_pages_jsonl.open("w", encoding="utf-8") as f:
        for i in range(total_pages):
            page = doc.load_page(i)
            # "text" is usually best for a v1; you can upgrade later if needed
            text = page.get_text("text") or ""
            row = {"page_num": i + 1, "text": text.strip()}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if logger and (i + 1) % 10 == 0:
                logger.info("Extracted %d/%d pages...", i + 1, total_pages)

    doc.close()
    if logger:
        logger.info("Wrote pages.jsonl: %s", str(out_pages_jsonl))

    return total_pages
