from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

from mcqgen.explain import add_explanations_for_wrong_questions
from mcqgen.generate import llm_generate_questions_for_chunk
from mcqgen.io_utils import read_json, read_yaml, write_json, write_jsonl
from mcqgen.llm_client import make_client
from mcqgen.logging_utils import setup_run_logger
from mcqgen.mock_llm import mock_generate_questions_for_chunk
from mcqgen.pages import load_pages_map, build_chunk_text
from mcqgen.pdf_extract import extract_pdf_pages_to_jsonl
from mcqgen.postprocess import postprocess_questions
from mcqgen.run_utils import make_chunks, make_run_id, now_iso_local

app = typer.Typer(no_args_is_help=True)

# -------------------------
# Utilities
# -------------------------

def list_pdfs(contexts_dir: Path) -> List[Path]:
    if not contexts_dir.exists():
        return []
    return sorted([p for p in contexts_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"])

def choose_context_interactively(contexts_dir: Path) -> Path:
    pdfs = list_pdfs(contexts_dir)
    if not pdfs:
        raise typer.BadParameter(f"No PDFs found in {contexts_dir}")

    typer.echo("Select a context PDF:")
    for i, p in enumerate(pdfs, start=1):
        typer.echo(f"  [{i}] {p.name}")

    idx = typer.prompt("Enter number", type=int)
    if idx < 1 or idx > len(pdfs):
        raise typer.BadParameter("Invalid selection.")
    return pdfs[idx - 1]

def validate_config(cfg: Dict[str, Any]) -> None:
    # Minimal validation for lean v1
    try:
        gen = cfg["generation"]
        part = cfg["partitioning"]
        rand = cfg["randomization"]
        ans = cfg["answers"]
        prompts = cfg["prompts"]
        llm = cfg["llm"]
    except KeyError as e:
        raise typer.BadParameter(f"Missing config section: {e}")

    qt = gen.get("question_type")
    if qt not in ("MCQ", "SATA"):
        raise typer.BadParameter("generation.question_type must be MCQ or SATA")

    c = int(gen.get("choices_per_question", 0))
    if c < 2:
        raise typer.BadParameter("generation.choices_per_question must be >= 2")

    qpp = int(gen.get("questions_per_partition", 0))
    if qpp < 1:
        raise typer.BadParameter("generation.questions_per_partition must be >= 1")

    w = int(part.get("pages_per_partition", 0))
    o = int(part.get("overlap_pages", 0))
    if w < 1:
        raise typer.BadParameter("partitioning.pages_per_partition must be >= 1")
    if o < 0 or o >= w:
        raise typer.BadParameter("partitioning.overlap_pages must satisfy 0 <= overlap < pages_per_partition")

    if not isinstance(rand.get("randomize_questions"), bool):
        raise typer.BadParameter("randomization.randomize_questions must be boolean")
    if not isinstance(rand.get("randomize_options"), bool):
        raise typer.BadParameter("randomization.randomize_options must be boolean")

    if ans.get("explanation_mode") not in ("none", "gpt"):
        raise typer.BadParameter("answers.explanation_mode must be none or gpt")
    if not isinstance(ans.get("explain_only_wrong"), bool):
        raise typer.BadParameter("answers.explain_only_wrong must be boolean")

    if not Path(prompts.get("user_prompt_file", "")).exists():
        raise typer.BadParameter("prompts.user_prompt_file does not exist")
    if not Path(prompts.get("mcq_prompt_file", "")).exists():
        raise typer.BadParameter("prompts.mcq_prompt_file does not exist")
    if not Path(prompts.get("sata_prompt_file", "")).exists():
        raise typer.BadParameter("prompts.sata_prompt_file does not exist")
    if ans.get("explanation_mode") == "gpt" and not Path(prompts.get("explanation_prompt_file", "")).exists():
        raise typer.BadParameter("prompts.explanation_prompt_file does not exist")

    if llm.get("provider") != "openai":
        raise typer.BadParameter("llm.provider must be openai for v1")
    if not llm.get("model"):
        raise typer.BadParameter("llm.model must be set")




# -------------------------
# Commands
# -------------------------

@app.command()
def generate(
    config: Path = typer.Option(..., "--config", exists=True),
    contexts_dir: Path = typer.Option(Path("./contexts"), "--contexts-dir"),
    runs_dir: Path = typer.Option(Path("./runs"), "--runs-dir"),
    context: Optional[str] = typer.Option(None, "--context"),
    non_interactive: bool = typer.Option(False, "--non-interactive"),
    verbose: bool = typer.Option(False, "--verbose"),
    mock_llm: bool = typer.Option(False, "--mock-llm", help="Use mock LLM output (no API calls)."),
):
    """
    Generate MCQs/SATA from a selected context PDF and write one clean output.json.
    """
    cfg = read_yaml(config)
    validate_config(cfg)

    # Choose context
    contexts_dir = contexts_dir.resolve()
    runs_dir = runs_dir.resolve()
    if context is not None:
        context_pdf = (contexts_dir / context).resolve()
        if not context_pdf.exists() or context_pdf.suffix.lower() != ".pdf":
            raise typer.BadParameter("--context must be a PDF filename in contexts/")
    else:
        if non_interactive:
            raise typer.BadParameter("Context must be provided via --context in --non-interactive mode.")
        context_pdf = choose_context_interactively(contexts_dir)

    run_id = make_run_id(context_pdf)
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    logger = setup_run_logger(run_dir, verbose=verbose)
    logger.info("LLM mode: %s", "mock" if mock_llm else "real")
    logger.info("Run: %s", run_id)
    logger.info("Context: %s", context_pdf.name)
    logger.info("Config: %s", str(config))
    client = None
    if not mock_llm:
        client = make_client()

    try:
        # Manifest (slim)
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "created_at": now_iso_local(),
            "context_file": context_pdf.name,
            "config_file": str(config),
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
            }
        }
        write_json(run_dir / "manifest.json", manifest)
        logger.info("Wrote manifest.json")

        # Extract
        pages_jsonl = run_dir / "pages.jsonl"
        logger.info("Extracting PDF -> %s", pages_jsonl)
        total_pages = extract_pdf_pages_to_jsonl(context_pdf, pages_jsonl, logger=logger)
        logger.info("Extracted %d pages", total_pages)
        pages_map = load_pages_map(pages_jsonl)
        logger.info("Loaded %d pages from pages.jsonl", len(pages_map))

        # Chunk
        chunks = make_chunks(
            total_pages=total_pages,
            pages_per_partition=cfg["partitioning"]["pages_per_partition"],
            overlap_pages=cfg["partitioning"]["overlap_pages"]
        )
        write_jsonl(run_dir / "chunks.jsonl", chunks)
        logger.info("Wrote chunks.jsonl (%d chunks)", len(chunks))

        # Generate
        all_questions: List[Dict[str, Any]] = []
        for ch in chunks:
            logger.info("Generating for %s (p%d-%d)", ch["chunk_id"], ch["page_start"], ch["page_end"])
            chunk_text = build_chunk_text(pages_map, ch["page_start"], ch["page_end"])
            logger.debug("Chunk %s context length: %d chars", ch["chunk_id"], len(chunk_text))
            if mock_llm:
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

        # Postprocess + output
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

        typer.echo(f"Run created: {run_dir}")
        typer.echo("Wrote: manifest.json, pages.jsonl, chunks.jsonl, output.json")

    except Exception:
        # This logs a full traceback to runs/<run_id>/run.log
        logger.exception("Run failed with an exception.")
        raise typer.Exit(code=1)


@app.command()
def explain(
    run: Path = typer.Option(..., "--run", exists=True, file_okay=False, dir_okay=True),
    wrong: Path = typer.Option(..., "--wrong", exists=True, file_okay=True, dir_okay=False),
    verbose: bool = typer.Option(False, "--verbose"),
):
    """
    Add GPT explanations for wrong questions listed in wrong_ids.json.
    (Lean mode: no test UI; user provides wrong IDs.)
    """
    run = run.resolve()
    output_path = run / "output.json"
    manifest_path = run / "manifest.json"
    if not output_path.exists() or not manifest_path.exists():
        raise typer.BadParameter("Run folder must contain output.json and manifest.json")

    logger = setup_run_logger(run, verbose=verbose)
    logger.info("Explain run: %s", run.name)

    wrong_obj = read_json(wrong)
    wrong_ids = wrong_obj.get("wrong_question_ids", [])
    if not isinstance(wrong_ids, list) or not all(isinstance(x, str) for x in wrong_ids):
        raise typer.BadParameter("wrong_ids.json must contain {\"wrong_question_ids\": [\"q_0001\", ...]}")

    manifest = read_json(manifest_path)
    cfg_prompt_path = Path(manifest["prompt_files"]["explanation"])

    client = make_client()
    model = manifest["llm"]["model"]
    temperature = manifest["llm"].get("temperature", 0.2)
    max_output_tokens = 500
    reasoning_effort = "none"

    def safe_echo(message: str) -> None:
        try:
            typer.echo(message)
        except OSError:
            # Avoid crashing if stdout is unavailable (e.g., piped/closed).
            pass

    try:
        updated_output = add_explanations_for_wrong_questions(
            output_json_path=output_path,
            wrong_ids=wrong_ids,
            explain_prompt_file=cfg_prompt_path,
            client=client,
            model=model,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            logger=logger
        )
        write_json(output_path, updated_output)
        safe_echo(f"Updated explanations in: {output_path}")
    except Exception:
        logger.exception("Explain failed with an exception.")
        safe_echo(f"Error. See log: {run / 'run.log'}")
        raise typer.Exit(code=1)


@app.command("list-contexts")
def list_contexts(
    contexts_dir: Path = typer.Option(Path("./contexts"), "--contexts-dir"),
):
    pdfs = list_pdfs(contexts_dir.resolve())
    for p in pdfs:
        typer.echo(p.name)


@app.command("list-runs")
def list_runs(
    runs_dir: Path = typer.Option(Path("./runs"), "--runs-dir"),
    limit: int = typer.Option(50, "--limit"),
):
    runs_dir = runs_dir.resolve()
    if not runs_dir.exists():
        return
    dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    dirs = sorted(dirs, reverse=True)[:limit]
    for d in dirs:
        typer.echo(d.name)


if __name__ == "__main__":
    app()
