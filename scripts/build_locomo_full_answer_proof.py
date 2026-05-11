#!/usr/bin/env python3
"""Build the full comparator-ready LoCoMo answer proof report.

Runs the existing LoCoMo answer-mode harness against the official 1540-row
public release and writes the checked-in full report fixture. This keeps the
slow full-proof lane explicit and reproducible without dragging it into the
default fast test loop.

Usage:
    python3 scripts/build_locomo_full_answer_proof.py
    python3 scripts/build_locomo_full_answer_proof.py --jobs 12 --fresh-corpus
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import eval_locomo

REPO_ROOT = eval_locomo.REPO_ROOT
SCRIPT_PATH = REPO_ROOT / "scripts" / "eval_locomo.py"
DEFAULT_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "locomo10.json"
DEFAULT_OUTPUT = REPO_ROOT / "tests" / "fixtures" / "locomo_answer_full_report.json"
DEFAULT_JOBS = max(1, min(os.cpu_count() or 1, 16))
DEFAULT_TIMEOUT_SECS = eval_locomo.ENV_TIMEOUT_SECS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--jobs",
        type=int,
        default=DEFAULT_JOBS,
        help="How many LoCoMo query subprocesses to run in parallel",
    )
    parser.add_argument(
        "--timeout-secs",
        type=int,
        default=DEFAULT_TIMEOUT_SECS,
        help="Per-cortyx command timeout passed through to scripts/eval_locomo.py",
    )
    parser.add_argument(
        "--min-answer-confidence",
        type=float,
        default=None,
        help="Optional cortyx --min-answer-confidence value for answer-mode runs",
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Enable the optional LLM judge in the underlying evaluation run",
    )
    parser.add_argument(
        "--fresh-corpus",
        action="store_true",
        help="Force the underlying evaluation to rebuild its staged corpus cache",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.jobs <= 0:
        print("ERROR: --jobs must be positive")
        return 1

    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "--fixture",
        args.fixture,
        "--answer-mode",
        "--profile",
        "full",
        "--jobs",
        str(args.jobs),
        "--timeout-secs",
        str(args.timeout_secs),
        "--output",
        args.output,
    ]
    if args.min_answer_confidence is not None:
        command.extend(["--min-answer-confidence", str(args.min_answer_confidence)])
    if args.llm_judge:
        command.append("--llm-judge")
    if args.fresh_corpus:
        command.append("--fresh-corpus")

    return subprocess.run(command, cwd=str(REPO_ROOT)).returncode


if __name__ == "__main__":
    raise SystemExit(main())
