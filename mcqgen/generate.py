from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import random
from openai import OpenAI

from mcqgen.llm_client import call_responses_structured
from mcqgen.llm_schemas import generation_schema
from mcqgen.text_utils import escape_braces


def render_generator_prompt(user_prompt_path: Path, fixed_prompt_path: Path, variables: Dict[str, Any]) -> str:
    user_text = user_prompt_path.read_text(encoding="utf-8").strip()
    fixed_text = fixed_prompt_path.read_text(encoding="utf-8").strip()
    combined = f"{user_text}\n\n{fixed_text}".strip()
    return combined.format(**variables)

def llm_generate_questions_for_chunk(
    *,
    client: OpenAI,
    chunk: Dict[str, Any],
    cfg: Dict[str, Any],
    chunk_text: str,
    retries: int = 3,
    logger: Optional[object] = None
) -> List[Dict[str, Any]]:
    """
    Real LLM generation using Responses API + Structured Outputs.
    Returns list of questions in the internal format expected by postprocess_questions.
    """
    qtype = cfg["generation"]["question_type"]
    choices_per_q = int(cfg["generation"]["choices_per_question"])
    questions_per_partition = int(cfg["generation"]["questions_per_partition"])

    schema_obj = generation_schema(choices_per_q)
    schema_name = schema_obj["name"]
    schema = schema_obj["schema"]

    user_prompt_file = Path(cfg["prompts"]["user_prompt_file"])
    fixed_prompt_file = Path(
        cfg["prompts"]["mcq_prompt_file"] if qtype == "MCQ" else cfg["prompts"]["sata_prompt_file"]
    )

    correct_counts_note = ""
    if qtype == "SATA":
        counts = [random.randint(1, choices_per_q) for _ in range(questions_per_partition)]
        counts_str = ", ".join(str(x) for x in counts)
        correct_counts_note = (
            "SATA only: Use these exact correct-choice counts per question, in order: "
            f"{counts_str}."
        )

    prompt = render_generator_prompt(
        user_prompt_file,
        fixed_prompt_file,
        {
            "question_type": qtype,
            "choices_per_question": choices_per_q,
            "questions_per_partition": questions_per_partition,
            "context": escape_braces(chunk_text),
            "correct_counts_note": correct_counts_note
        }
    )

    reasoning_effort = cfg["llm"].get("reasoning_effort", "none")
    temperature = cfg["llm"].get("temperature", 0.2)
    max_output_tokens = int(cfg["llm"].get("max_output_tokens", 4000))

    data = None
    for attempt in range(1, retries + 1):
        try:
            data = call_responses_structured(
                client=client,
                model=cfg["llm"]["model"],          # e.g., "gpt-5.2"
                prompt=prompt,
                schema_name=schema_name,
                schema=schema,
                reasoning_effort=reasoning_effort,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                logger=logger
            )
            break
        except Exception as exc:
            if logger:
                if attempt < retries:
                    logger.warning(
                        "LLM generation failed for %s (attempt %d/%d): %s",
                        chunk["chunk_id"],
                        attempt,
                        retries,
                        exc
                    )
                else:
                    logger.exception(
                        "LLM generation failed for %s after %d attempts; skipping chunk.",
                        chunk["chunk_id"],
                        retries
                    )
            if attempt < retries:
                continue
            return []

    out: List[Dict[str, Any]] = []
    for q in data["questions"]:
        out.append({
            "question_type": qtype,
            "stem": q["stem"],
            "choices": q["choices"],
            "correct_choice_ids": q["correct_choice_ids"],
            "chunk_id": chunk["chunk_id"],
            "explanation": ""
        })

    if logger:
        logger.info("LLM returned %d questions for %s", len(out), chunk["chunk_id"])

    return out
