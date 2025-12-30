from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def load_pages_map(pages_jsonl: Path) -> Dict[int, str]:
    """
    Reads runs/<run>/pages.jsonl and returns {page_num: text}.
    """
    pages: Dict[int, str] = {}
    with pages_jsonl.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            page_num = int(obj["page_num"])
            text = str(obj.get("text", ""))
            pages[page_num] = text
    return pages


def build_chunk_text(pages_map: Dict[int, str], page_start: int, page_end: int) -> str:
    """
    Builds a single context string for a chunk, annotated with page markers.
    """
    parts: List[str] = []
    for p in range(page_start, page_end + 1):
        txt = pages_map.get(p, "")
        # Keep it simple: marker + text (text may be empty)
        parts.append(f"[Page {p}]\n{txt}".strip())
    return "\n\n".join(parts).strip()
