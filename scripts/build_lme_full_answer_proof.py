#!/usr/bin/env python3
"""Build a full comparator-ready LongMemEval answer proof report.

Runs the existing LongMemEval answer-mode harness once per category, then merges
those category reports back into one full 500-row report plus the official
question_id/hypothesis JSONL surface. This keeps the heavy full-proof lane out
of the default fast loop while reusing the same report/public-surface format.

Usage:
    python3 scripts/build_lme_full_answer_proof.py
    python3 scripts/build_lme_full_answer_proof.py --jobs 4 --timeout-secs 1500
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import eval_lme

REPO_ROOT = eval_lme.REPO_ROOT
SCRIPT_PATH = REPO_ROOT / "scripts" / "eval_lme.py"
DEFAULT_OUTPUT = REPO_ROOT / "tests" / "fixtures" / "longmemeval_answer_full_report.json"
DEFAULT_WORKDIR = REPO_ROOT / ".cortyx_eval_work" / "lme_full_answer_proof"
DEFAULT_TIMEOUT_SECS = 1500
DEFAULT_JOBS = max(1, min(len(eval_lme.CATEGORIES), os.cpu_count() or 1, 4))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--fixture", default=eval_lme.FIXTURE_DEFAULT)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Where to write the merged full-report JSON fixture",
    )
    parser.add_argument(
        "--workdir",
        default=str(DEFAULT_WORKDIR),
        help="Directory for temporary per-category reports",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=DEFAULT_JOBS,
        help="How many category evaluations to run in parallel",
    )
    parser.add_argument(
        "--timeout-secs",
        type=int,
        default=DEFAULT_TIMEOUT_SECS,
        help="Per-cortyx command timeout passed through to scripts/eval_lme.py",
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
        help="Use the optional LLM judge in each per-category eval",
    )
    parser.add_argument(
        "--fresh-corpus",
        action="store_true",
        help="Force each per-category eval to rebuild its staged corpus cache",
    )
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="Keep per-category reports instead of cleaning them after merge",
    )
    return parser.parse_args()


def load_fixture_entries(fixture_path: Path) -> list[dict]:
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def build_question_order(entries: list[dict]) -> dict[str, int]:
    order: dict[str, int] = {}
    for index, entry in enumerate(entries):
        question_id = eval_lme.entry_question_id(entry)
        if question_id:
            order[question_id] = index
    return order


def relative_path(path: Path) -> str:
    return eval_lme.display_path(path)


def segment_output_path(workdir: Path, category: str) -> Path:
    return workdir / f"{category}.json"


def run_segment(
    *,
    category: str,
    fixture_path: Path,
    workdir: Path,
    timeout_secs: int,
    min_answer_confidence: float | None,
    llm_judge: bool,
    fresh_corpus: bool,
) -> tuple[str, Path, float]:
    output_path = segment_output_path(workdir, category)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "--fixture",
        str(fixture_path),
        "--answer-mode",
        "--categories",
        category,
        "--timeout-secs",
        str(timeout_secs),
        "--output",
        str(output_path),
    ]
    if min_answer_confidence is not None:
        command.extend(["--min-answer-confidence", str(min_answer_confidence)])
    if llm_judge:
        command.append("--llm-judge")
    if fresh_corpus:
        command.append("--fresh-corpus")

    start = time.perf_counter()
    print(f"[start] {category}")
    result = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        detail = stderr or stdout or f"command exited with code {result.returncode}"
        raise RuntimeError(f"{category} segment failed: {detail}")

    print(f"[done]  {category} ({elapsed:.1f}s)")
    return category, output_path, elapsed


def load_segment_reports(paths: dict[str, Path]) -> dict[str, dict]:
    reports: dict[str, dict] = {}
    for category, path in paths.items():
        reports[category] = json.loads(path.read_text(encoding="utf-8"))
    return reports


def aggregate_corpus(segment_reports: dict[str, dict]) -> dict:
    segments: list[dict] = []
    reused_segments = 0
    conversation_inputs = 0
    project_inputs = 0
    files_staged = 0
    binary = None

    for category in eval_lme.CATEGORIES:
        report = segment_reports.get(category)
        if not report:
            continue
        corpus = report.get("run", {}).get("corpus_cache") or {}
        if corpus.get("reused"):
            reused_segments += 1
        conversation_inputs += int(corpus.get("conversation_inputs", 0) or 0)
        project_inputs += int(corpus.get("project_inputs", 0) or 0)
        files_staged += int(corpus.get("files_staged", 0) or 0)
        if binary is None and corpus.get("binary"):
            binary = corpus.get("binary")
        segments.append(
            {
                "category": category,
                "reused": bool(corpus.get("reused")),
                "cache_key": corpus.get("cache_key"),
                "path": corpus.get("path"),
                "conversation_inputs": corpus.get("conversation_inputs", 0),
                "project_inputs": corpus.get("project_inputs", 0),
                "files_staged": corpus.get("files_staged", 0),
                "stage_secs": corpus.get("stage_secs", 0.0),
                "mine_secs": corpus.get("mine_secs", 0.0),
            }
        )

    return {
        "strategy": "segmented_by_category",
        "segment_count": len(segments),
        "reused_segments": reused_segments,
        "conversation_inputs": conversation_inputs,
        "project_inputs": project_inputs,
        "files_staged": files_staged,
        "binary": binary or eval_lme.binary_fingerprint(),
        "segments": segments,
    }


def aggregate_timings(
    *,
    segment_reports: dict[str, dict],
    wall_clock_secs: float,
    query_count: int,
    jobs: int,
) -> dict:
    corpus_stage_secs = 0.0
    corpus_mine_secs = 0.0
    query_secs_total = 0.0
    segment_total_secs = 0.0
    query_secs_by_category: dict[str, float] = {}

    for category in eval_lme.CATEGORIES:
        report = segment_reports.get(category)
        if not report:
            continue
        timings = report.get("run", {}).get("timings") or {}
        corpus_stage_secs += float(timings.get("corpus_stage_secs", 0.0) or 0.0)
        corpus_mine_secs += float(timings.get("corpus_mine_secs", 0.0) or 0.0)
        query_secs_total += float(timings.get("query_secs_total", 0.0) or 0.0)
        segment_total_secs += float(timings.get("total_secs", 0.0) or 0.0)
        category_times = timings.get("query_secs_by_category") or {}
        category_query_secs = float(category_times.get(category, timings.get("query_secs_total", 0.0)) or 0.0)
        query_secs_by_category[category] = round(category_query_secs, 4)

    return {
        "corpus_stage_secs": round(corpus_stage_secs, 4),
        "corpus_mine_secs": round(corpus_mine_secs, 4),
        "query_secs_total": round(query_secs_total, 4),
        "avg_query_secs": round(query_secs_total / query_count, 4) if query_count else 0.0,
        "query_count": query_count,
        "query_secs_by_category": query_secs_by_category,
        "total_secs": round(wall_clock_secs, 4),
        "segment_total_secs": round(segment_total_secs, 4),
        "parallel_jobs": jobs,
    }


def merge_case_rows(
    *,
    entries: list[dict],
    segment_reports: dict[str, dict],
    failures: list[dict],
) -> list[dict]:
    order = build_question_order(entries)
    case_rows: list[dict] = []
    seen_question_ids: set[str] = set()

    for category in eval_lme.CATEGORIES:
        report = segment_reports.get(category)
        if not report:
            continue
        for row in report.get("cases", {}).get("rows", []):
            question_id = str(row.get("question_id") or "").strip()
            if question_id:
                if question_id in seen_question_ids:
                    failures.append(
                        {
                            "stage": "merge",
                            "group": category,
                            "command": "merge-case-rows",
                            "cwd": str(REPO_ROOT),
                            "timed_out": False,
                            "returncode": 1,
                            "stderr_excerpt": f"duplicate question_id detected during merge: {question_id}",
                        }
                    )
                    continue
                seen_question_ids.add(question_id)
            case_rows.append(row)

    case_rows.sort(
        key=lambda row: (
            order.get(str(row.get("question_id") or "").strip(), len(order)),
            row.get("category", ""),
            row.get("question", ""),
        )
    )
    return case_rows


def merge_reports(
    *,
    fixture_path: Path,
    entries: list[dict],
    output_path: Path,
    segment_reports: dict[str, dict],
    timeout_secs: int,
    min_answer_confidence: float | None,
    llm_judge: bool,
    wall_clock_secs: float,
    jobs: int,
) -> dict:
    failures: list[dict] = []
    results: dict[str, dict] = {}

    for category in eval_lme.CATEGORIES:
        report = segment_reports.get(category)
        if not report:
            failures.append(
                {
                    "stage": "merge",
                    "group": category,
                    "command": "load-segment-report",
                    "cwd": str(REPO_ROOT),
                    "timed_out": False,
                    "returncode": 1,
                    "stderr_excerpt": "missing per-category report",
                }
            )
            continue

        failures.extend(report.get("diagnostics", {}).get("infra_failures", []))
        results.update(report.get("results", {}))

    case_rows = merge_case_rows(entries=entries, segment_reports=segment_reports, failures=failures)
    selected_categories = {eval_lme.entry_category(entry) for entry in entries}

    evaluation = {
        "results": results,
        "overall": eval_lme.build_overall(results, answer_mode=True),
        "case_rows": case_rows,
        "diagnostics": {
            "infra_failures": failures,
            "hard_cases": eval_lme.hardest_cases(case_rows, "answer_recall"),
        },
        "corpus": aggregate_corpus(segment_reports),
        "timings": aggregate_timings(
            segment_reports=segment_reports,
            wall_clock_secs=wall_clock_secs,
            query_count=len(case_rows),
            jobs=jobs,
        ),
    }
    public_surface = eval_lme.build_public_surface_artifact(
        case_rows,
        answer_mode=True,
        output_path=output_path,
    )
    report = eval_lme.build_report(
        fixture_path,
        entries,
        entries,
        profile="full",
        answer_mode=True,
        use_llm=llm_judge,
        timeout_secs=timeout_secs,
        min_answer_confidence=min_answer_confidence,
        selected_categories=selected_categories,
        evaluation=evaluation,
        public_surface=public_surface,
    )
    report["run"]["proof_strategy"] = {
        "mode": "segmented_by_category",
        "jobs": jobs,
        "categories": list(eval_lme.CATEGORIES),
    }
    return report


def main() -> int:
    args = parse_args()
    fixture_path = Path(args.fixture)
    if not fixture_path.exists():
        print(f"ERROR: fixture not found: {fixture_path}")
        return 1
    if args.jobs <= 0:
        print("ERROR: --jobs must be positive")
        return 1

    output_path = Path(args.output)
    workdir = Path(args.workdir)
    entries = load_fixture_entries(fixture_path)

    if workdir.exists():
        shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    segment_paths: dict[str, Path] = {}

    try:
        with ThreadPoolExecutor(max_workers=min(args.jobs, len(eval_lme.CATEGORIES))) as executor:
            futures = {
                executor.submit(
                    run_segment,
                    category=category,
                    fixture_path=fixture_path,
                    workdir=workdir,
                    timeout_secs=args.timeout_secs,
                    min_answer_confidence=args.min_answer_confidence,
                    llm_judge=args.llm_judge,
                    fresh_corpus=args.fresh_corpus,
                ): category
                for category in eval_lme.CATEGORIES
            }
            for future in as_completed(futures):
                category, path, _elapsed = future.result()
                segment_paths[category] = path

        segment_reports = load_segment_reports(segment_paths)
        wall_clock_secs = time.perf_counter() - start
        report = merge_reports(
            fixture_path=fixture_path,
            entries=entries,
            output_path=output_path,
            segment_reports=segment_reports,
            timeout_secs=args.timeout_secs,
            min_answer_confidence=args.min_answer_confidence,
            llm_judge=args.llm_judge,
            wall_clock_secs=wall_clock_secs,
            jobs=min(args.jobs, len(eval_lme.CATEGORIES)),
        )
        eval_lme.print_summary(report)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(eval_lme.json_ready(report), indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\n✓ Results saved to {relative_path(output_path)}")
        return 0 if report["proof"]["reproducible"] else 2
    finally:
        if not args.keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
