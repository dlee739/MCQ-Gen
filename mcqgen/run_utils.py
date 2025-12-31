from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def now_iso_local() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def slugify(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        elif ch in (" ", "."):
            keep.append("_")
    return "".join(keep).strip("_")


def make_run_id(context_pdf: Path) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = slugify(context_pdf.stem)
    return f"{ts}__{base}"


def make_chunks(total_pages: int, pages_per_partition: int, overlap_pages: int) -> List[Dict[str, Any]]:
    stride = pages_per_partition - overlap_pages
    chunks: List[Dict[str, Any]] = []

    start = 1
    idx = 1
    while start <= total_pages:
        end = min(start + pages_per_partition - 1, total_pages)
        chunks.append({
            "chunk_id": f"chunk_{idx:03d}",
            "page_start": start,
            "page_end": end
        })
        if end == total_pages:
            break
        start += stride
        idx += 1

    return chunks
