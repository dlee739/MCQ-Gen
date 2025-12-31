from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_run_logger(run_dir: Path, verbose: bool = False) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"

    logger = logging.getLogger(f"mcqgen.{run_dir.name}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # avoid duplicate logs if root logger exists

    # Clear handlers if reusing in tests/dev
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    # Console handler for user feedback
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    logger.debug("Logger initialized. Log file: %s", log_path)
    return logger
