from __future__ import annotations

import random
from typing import Any, Dict, List


def postprocess_questions(
    questions: List[Dict[str, Any]],
    question_type: str,
    choices_per_question: int,
    randomize_questions: bool,
    randomize_options: bool
) -> List[Dict[str, Any]]:
    # A) validate minimal structure + rules
    seen_ids = set()
    for q in questions:
        # required keys
        for k in ("stem", "choices", "correct_choice_ids", "chunk_id"):
            if k not in q:
                raise ValueError(f"Question missing '{k}': {q}")

        if q.get("question_type") != question_type:
            raise ValueError("Question type mismatch with config")

        choices = q["choices"]
        if not isinstance(choices, list) or len(choices) != choices_per_question:
            raise ValueError("Choices count mismatch")

        # ensure stable choice ids are present
        choice_ids = [c.get("id") for c in choices]
        if any((not cid) for cid in choice_ids) or len(set(choice_ids)) != len(choice_ids):
            raise ValueError("Choice ids must be unique and non-empty")

        correct = q["correct_choice_ids"]
        if not isinstance(correct, list) or len(correct) < 1:
            raise ValueError("correct_choice_ids must be a non-empty list")

        if question_type == "MCQ" and len(correct) != 1:
            raise ValueError("MCQ must have exactly 1 correct choice")
        # SATA: allow >=1 (or change to >=2 if you decide)

        if any(cid not in set(choice_ids) for cid in correct):
            raise ValueError("correct_choice_ids must refer to existing choices")

        # Ensure explanation field exists (empty string ok)
        q.setdefault("explanation", "")

    # B) shuffle questions (no seed)
    if randomize_questions:
        random.shuffle(questions)

    # C) shuffle options (safe because choice ids are stable)
    if randomize_options:
        for q in questions:
            random.shuffle(q["choices"])

    # D) assign sequential question IDs at the end (stable, clean)
    for i, q in enumerate(questions, start=1):
        q["id"] = f"q_{i:04d}"

    return questions
