from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from mcqgen.pdf_extract import extract_pdf_pages_to_jsonl
from mcqgen.pages import load_pages_map, build_chunk_text
from mcqgen.logging_utils import setup_run_logger
from mcqgen.io_utils import write_json, write_jsonl
from mcqgen.run_utils import make_run_id, make_chunks, now_iso_local
from mcqgen.postprocess import postprocess_questions
from mcqgen.llm_client import make_client
from mcqgen.generate import llm_generate_questions_for_chunk
from mcqgen.mock_llm import mock_generate_questions_for_chunk


def run_generate_pipeline(
    *,
    context_pdf: Path,
    cfg: Dict[str, Any],
    runs_dir: Path,
    logger: Optional[object] = None,
    use_mock_llm: bool = False,
    run_dir: Optional[Path] = None,
    config_path: Optional[Path] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Path:
    if run_dir is None:
        run_id = make_run_id(context_pdf)
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        run_id = run_dir.name

    if logger is None:
        logger = setup_run_logger(run_dir, verbose=False)

    logger.info("Run: %s", run_id)
    logger.info("Context: %s", context_pdf.name)
    if config_path:
        logger.info("Config: %s", str(config_path))

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": now_iso_local(),
        "context_file": context_pdf.name,
        "config_file": str(config_path) if config_path else None,
        "prompt_files": {
            "user": cfg["prompts"]["user_prompt_file"],
            "mcq_fixed": cfg["prompts"]["mcq_prompt_file"],
            "sata_fixed": cfg["prompts"]["sata_prompt_file"],
            "explanation": cfg["prompts"]["explanation_prompt_file"],
        },
        "llm": {
            "provider": cfg["llm"]["provider"],
            "model": cfg["llm"]["model"],
            "temperature": cfg["llm"].get("temperature", 0.2),
            "max_output_tokens": cfg["llm"].get("max_output_tokens", 4000),
            "reasoning_effort": cfg["llm"].get("reasoning_effort", "none"),
        }
    }
    write_json(run_dir / "manifest.json", manifest)
    logger.info("Wrote manifest.json")

    pages_jsonl = run_dir / "pages.jsonl"
    logger.info("Extracting PDF -> %s", pages_jsonl)
    total_pages = extract_pdf_pages_to_jsonl(context_pdf, pages_jsonl, logger=logger)
    logger.info("Extracted %d pages", total_pages)

    chunks = make_chunks(
        total_pages=total_pages,
        pages_per_partition=cfg["partitioning"]["pages_per_partition"],
        overlap_pages=cfg["partitioning"]["overlap_pages"]
    )
    write_jsonl(run_dir / "chunks.jsonl", chunks)
    logger.info("Wrote chunks.jsonl (%d chunks)", len(chunks))

    pages_map = load_pages_map(pages_jsonl)
    logger.info("Loaded %d pages from pages.jsonl", len(pages_map))

    client = None
    if not use_mock_llm:
        client = make_client()

    all_questions: List[Dict[str, Any]] = []
    total_chunks = len(chunks)
    for idx, ch in enumerate(chunks, start=1):
        logger.info("Generating for %s (p%d-%d)", ch["chunk_id"], ch["page_start"], ch["page_end"])
        chunk_text = build_chunk_text(pages_map, ch["page_start"], ch["page_end"])
        logger.debug("Chunk %s context length: %d chars", ch["chunk_id"], len(chunk_text))

        if use_mock_llm:
            qs = mock_generate_questions_for_chunk(ch, cfg, logger=logger)
        else:
            qs = llm_generate_questions_for_chunk(
                client=client,
                chunk=ch,
                cfg=cfg,
                chunk_text=chunk_text,
                logger=logger
            )
        logger.info("  -> got %d questions", len(qs))
        all_questions.extend(qs)

        if progress_callback:
            progress_callback(idx, total_chunks)

    processed = postprocess_questions(
        questions=all_questions,
        question_type=cfg["generation"]["question_type"],
        choices_per_question=cfg["generation"]["choices_per_question"],
        randomize_questions=cfg["randomization"]["randomize_questions"],
        randomize_options=cfg["randomization"]["randomize_options"],
    )
    logger.info("Postprocess complete. Total questions: %d", len(processed))

    output = {
        "schema_version": 1,
        "run_id": run_id,
        "context_file": context_pdf.name,
        "settings": {
            "question_type": cfg["generation"]["question_type"],
            "choices_per_question": cfg["generation"]["choices_per_question"],
            "randomize_questions": cfg["randomization"]["randomize_questions"],
            "randomize_options": cfg["randomization"]["randomize_options"],
        },
        "questions": processed
    }
    write_json(run_dir / "output.json", output)
    logger.info("Wrote output.json")

    return run_dir
