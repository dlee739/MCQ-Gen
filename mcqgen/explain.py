from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from mcqgen.text_utils import escape_braces


def render_explain_prompt(template_path: Path, variables: Dict[str, Any]) -> str:
    template = template_path.read_text(encoding="utf-8")
    return template.format(**variables)


def format_choices_for_prompt(choices: List[Dict[str, Any]]) -> str:
    # Keep it simple + readable for the model.
    # Example:
    # c1: ...
    # c2: ...
    lines = []
    for c in choices:
        cid = c.get("id", "")
        text = c.get("text", "")
        lines.append(f"{cid}: {text}")
    return "\n".join(lines)

def format_choices_with_letters(choices: List[Dict[str, Any]]) -> str:
    # Label choices in their current order as A, B, C, ...
    lines = []
    for idx, c in enumerate(choices):
        letter = chr(ord("A") + idx)
        text = c.get("text", "")
        lines.append(f"{letter}. {text}")
    return "\n".join(lines)


def format_correct_letters(choices: List[Dict[str, Any]], correct_ids: List[str]) -> str:
    id_to_letter: Dict[str, str] = {}
    for idx, c in enumerate(choices):
        cid = c.get("id", "")
        if cid:
            id_to_letter[cid] = chr(ord("A") + idx)
    letters = [id_to_letter.get(cid, "?") for cid in correct_ids]
    return ", ".join(letters)

def call_responses_text(
    *,
    client: OpenAI,
    model: str,
    prompt: str,
    reasoning_effort: str = "none",
    temperature: Optional[float] = 0.2,
    max_output_tokens: int = 500,
    logger: Optional[object] = None
) -> str:
    """
    Plain-text Responses API call (no JSON schema).
    Uses `output_text` helper to extract combined text output.
    """
    req: Dict[str, Any] = {
        "model": model,
        "input": prompt,
        "max_output_tokens": int(max_output_tokens),
        "reasoning": {"effort": reasoning_effort},
    }

    # GPT-5.2 constraint: temperature only allowed when reasoning.effort == "none".
    if temperature is not None:
        if reasoning_effort == "none":
            req["temperature"] = float(temperature)
        else:
            if logger:
                logger.info("Omitting temperature (reasoning_effort=%s)", reasoning_effort)

    resp = client.responses.create(**req)

    raw = resp.output_text
    return raw() if callable(raw) else raw


def add_explanations_for_wrong_questions(
    *,
    output_json_path: Path,
    wrong_ids: List[str],
    explain_prompt_file: Path,
    client: OpenAI,
    model: str,
    reasoning_effort: str = "none",
    temperature: Optional[float] = 0.2,
    max_output_tokens: int = 500,
    retries: int = 3,
    logger: Optional[object] = None
) -> Dict[str, Any]:
    """
    Mutates output.json in-memory: fills `explanation` field for wrong questions.
    Returns updated output object.
    """
    import json

    output = json.loads(output_json_path.read_text(encoding="utf-8"))
    questions: List[Dict[str, Any]] = output.get("questions", [])
    by_id = {q.get("id"): q for q in questions}

    updated = 0
    skipped_missing = 0
    skipped_failed = 0

    for qid in wrong_ids:
        q = by_id.get(qid)
        if not q:
            skipped_missing += 1
            if logger:
                logger.warning("Skipping unknown question id: %s", qid)
            continue

        stem = str(q.get("stem", ""))
        choices = q.get("choices", [])
        correct_ids = q.get("correct_choice_ids", [])

        prompt = render_explain_prompt(
            explain_prompt_file,
            {
                "stem": escape_braces(stem),
                "choices": escape_braces(format_choices_with_letters(choices)),
                "correct_choice_ids": escape_braces(format_correct_letters(choices, correct_ids))
            }
        )

        if logger:
            logger.info("Explaining %s ...", qid)

        explanation = None
        for attempt in range(1, retries + 1):
            try:
                explanation = call_responses_text(
                    client=client,
                    model=model,
                    prompt=prompt,
                    reasoning_effort=reasoning_effort,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    logger=logger
                ).strip()
                break
            except Exception as exc:
                if logger:
                    if attempt < retries:
                        logger.warning(
                            "Explain failed for %s (attempt %d/%d): %s",
                            qid,
                            attempt,
                            retries,
                            exc
                        )
                    else:
                        logger.exception(
                            "Explain failed for %s after %d attempts; skipping.",
                            qid,
                            retries
                        )
        if explanation is None:
            skipped_failed += 1
            continue

        q["explanation"] = explanation
        updated += 1

    if logger:
        logger.info(
            "Explanations added: %d | missing ids skipped: %d | failed: %d",
            updated,
            skipped_missing,
            skipped_failed
        )

    return output
