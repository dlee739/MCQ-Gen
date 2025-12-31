from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from openai import OpenAI


def _read_api_key_from_file(path: Path) -> Optional[str]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in raw.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    return None


def make_client() -> OpenAI:
    # Uses OPENAI_API_KEY from env automatically, or .openai_key in repo root.
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        key_path = Path(__file__).resolve().parents[1] / ".openai_key"
        if key_path.exists():
            api_key = _read_api_key_from_file(key_path)
    if api_key:
        return OpenAI(api_key=api_key)
    return OpenAI()


def call_responses_structured(
    *,
    client: OpenAI,
    model: str,
    prompt: str,
    schema_name: str,
    schema: Dict[str, Any],
    reasoning_effort: str = "none",          # "none" | "low" | "medium" | "high" | "xhigh"
    temperature: Optional[float] = 0.2,      # only allowed when reasoning_effort == "none" for GPT-5.2
    max_output_tokens: int = 4000,
    logger: Optional[object] = None
) -> Dict[str, Any]:
    """
    Calls Responses API and returns the parsed JSON that conforms to the given JSON schema.
    """
    req: Dict[str, Any] = {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_output_tokens,
        "reasoning": {"effort": reasoning_effort},
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": schema,
                "strict": True
            }
        }
    }

    # GPT-5.2 compatibility: temperature only with reasoning_effort == "none"
    if temperature is not None:
        if reasoning_effort == "none":
            req["temperature"] = float(temperature)
        else:
            if logger:
                logger.info(
                    "Omitting temperature because reasoning_effort=%s (GPT-5.2 constraint).",
                    reasoning_effort
                )

    if logger:
        logger.debug("Responses request model=%s reasoning_effort=%s", model, reasoning_effort)

    resp = client.responses.create(**req)

    raw = resp.output_text
    raw_text = raw() if callable(raw) else raw

    try:
        return json.loads(raw_text)
    except Exception:
        if logger:
            logger.exception("Failed to parse structured JSON from model output.")
            logger.debug("Raw output_text: %s", raw_text)
        raise
