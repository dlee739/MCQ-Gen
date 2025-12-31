from __future__ import annotations


def escape_braces(text: str) -> str:
    # Python str.format treats { } specially; escape them in user/PDF text.
    return text.replace("{", "{{").replace("}", "}}")
