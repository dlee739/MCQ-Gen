from __future__ import annotations

from typing import Any, Dict, List, Optional


def mock_generate_questions_for_chunk(
    chunk: Dict[str, Any],
    cfg: Dict[str, Any],
    logger: Optional[object] = None
) -> List[Dict[str, Any]]:
    """
    Return a list of fake questions for a single chunk.
    Output format must match what postprocess_questions expects:
      - stem
      - question_type
      - choices (with stable ids c1..cN)
      - correct_choice_ids (subset of choices)
      - chunk_id
      - explanation (optional; postprocess will default it)
    """
    qtype = cfg["generation"]["question_type"]
    n_choices = int(cfg["generation"]["choices_per_question"])
    n_q = int(cfg["generation"]["questions_per_partition"])

    page_start = chunk["page_start"]
    page_end = chunk["page_end"]
    chunk_id = chunk["chunk_id"]

    questions: List[Dict[str, Any]] = []
    for i in range(1, n_q + 1):
        # choices with stable IDs so shuffling doesn't require remapping
        choices = [
            {"id": f"c{k}", "text": f"Mock option {k} (chunk {chunk_id})"}
            for k in range(1, n_choices + 1)
        ]

        if qtype == "MCQ":
            correct = ["c2"] if n_choices >= 2 else ["c1"]
        else:  # SATA
            # pick two correct if possible; otherwise one
            correct = (
                ["c2", "c4"] if n_choices >= 4 else (["c1", "c2"] if n_choices >= 2 else ["c1"])
            )

        stem = (
            f"[MOCK] Q{i} for {chunk_id} (pages {page_start}-{page_end}): "
            f"Which option is correct?"
        )

        questions.append({
            "question_type": qtype,
            "stem": stem,
            "choices": choices,
            "correct_choice_ids": correct,
            "chunk_id": chunk_id,
            "explanation": ""
        })

    if logger:
        logger.debug("Mock generated %d questions for %s", len(questions), chunk_id)

    return questions
