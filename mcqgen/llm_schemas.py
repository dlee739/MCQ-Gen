from __future__ import annotations

from typing import Any, Dict


def generation_schema(choices_per_question: int) -> Dict[str, Any]:
    """
    JSON Schema for one chunk's generation output.

    Model should return:
      { "questions": [ { "stem": "...", "choices": [...], "correct_choice_ids": [...] }, ... ] }

    Notes:
    - choice ids should be stable like c1..cN (recommended)
    - we enforce exact choice count via minItems/maxItems
    """
    if choices_per_question < 2:
        raise ValueError("choices_per_question must be >= 2")

    return {
        "name": "mcqgen_chunk_questions_v1",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["questions"],
            "properties": {
                "questions": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["stem", "choices", "correct_choice_ids"],
                        "properties": {
                            "stem": {"type": "string", "minLength": 1},
                            "choices": {
                                "type": "array",
                                "minItems": choices_per_question,
                                "maxItems": choices_per_question,
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": ["id", "text"],
                                    "properties": {
                                        "id": {"type": "string", "minLength": 1},
                                        "text": {"type": "string", "minLength": 1}
                                    }
                                }
                            },
                            "correct_choice_ids": {
                                "type": "array",
                                "minItems": 1,
                                "items": {"type": "string", "minLength": 1}
                            }
                        }
                    }
                }
            }
        }
    }
