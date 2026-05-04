#!/usr/bin/env python3
"""Proper evaluation harness for LoCoMo benchmark.

Computes token-level F1, Exact Match, and Recall per question type against the
real LoCoMo gold answers. Replaces the single-keyword hit check used in the
basic bench_locomo test.

Usage:
    python3 scripts/eval_locomo.py
    python3 scripts/eval_locomo.py --fixture tests/fixtures/locomo_sample.json
    python3 scripts/eval_locomo.py --profile quick
    python3 scripts/eval_locomo.py --fixture tests/fixtures/locomo10.json --answer-mode
    python3 scripts/eval_locomo.py --answer-mode
    python3 scripts/eval_locomo.py --fresh-corpus
    python3 scripts/eval_locomo.py --llm-judge
    python3 scripts/eval_locomo.py --llm-answer   # Cortyx retrieves → Ollama synthesises → F1

LoCoMo metrics (per arXiv:2402.17753):
    Default mode:
        F1     — token overlap between retrieved context and gold answer
        EM     — exact token-set match
        Recall — fraction of gold tokens present in retrieved context
    Answer mode:
        F1 / EM / Recall operate on the predicted answer string instead.

Proof-grade report additions:
    - fixture hash + git revision metadata
    - partial-run / comparator blockers
    - full per-case answer surface
    - hardest misses + infra failures

Baseline reference:
    Hindsight   89.6% LoCoMo
    Zep          ~85%
    Letta/MemGPT ~83.2%
    Mem0          58–67%
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from gen_locomo import convert_locomo

REPO_ROOT = Path(__file__).resolve().parent.parent
CORTYX_BIN = os.environ.get("CORTYX_BIN", "")
ENV_TIMEOUT_SECS = int(os.environ.get("CORTYX_TIMEOUT_SECS", "180"))
ENV_ANSWER_MODE = os.environ.get("CORTYX_ANSWER_MODE", "").lower() not in ("", "0", "false", "no")
FIXTURE_DEFAULT = str(REPO_ROOT / "tests" / "fixtures" / "locomo_sample.json")
DEFAULT_OUTPUT = str(REPO_ROOT / "locomo_eval_results.json")

QUESTION_TYPES = ["single_hop", "multi_hop", "temporal", "open_qa"]
OFFICIAL_PUBLIC_RELEASE_ROWS = 1540

PROFILE_DEFAULTS = {
    "full": {"max_entries": 0, "max_per_question_type": 0},
    "quick": {"max_entries": 0, "max_per_question_type": 2},
    "smoke": {"max_entries": 0, "max_per_question_type": 1},
}


@dataclass
class RunResult:
    args: list[str]
    cwd: str
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


# ── Scoring helpers ────────────────────────────────────────────────────────────


def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def f1_score(prediction: str, gold: str) -> float:
    pred = Counter(_tokenise(prediction))
    ref = Counter(_tokenise(gold))
    common = sum((pred & ref).values())
    if common == 0:
        return 0.0
    precision = common / sum(pred.values()) if pred else 0.0
    recall = common / sum(ref.values()) if ref else 0.0
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, gold: str) -> bool:
    return set(_tokenise(prediction)) == set(_tokenise(gold))


def recall_score(prediction: str, gold: str) -> float:
    pred_tokens = set(_tokenise(prediction))
    gold_tokens = _tokenise(gold)
    if not gold_tokens:
        return 1.0
    hits = sum(1 for token in gold_tokens if token in pred_tokens)
    return hits / len(gold_tokens)


# ── LLM judge ─────────────────────────────────────────────────────────────────


def llm_judge(question: str, retrieved: str, gold: str, *, answer_mode: bool) -> float:
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return -1.0

    if answer_mode:
        prompt = (
            f"Question: {question}\n\n"
            f"Predicted answer:\n{retrieved[:1500]}\n\n"
            f"Gold answer: {gold}\n\n"
            "Is the predicted answer correct? Reply with exactly: YES, PARTIAL, or NO"
        )
    else:
        prompt = (
            f"Question: {question}\n\n"
            f"Retrieved context (first 1500 chars):\n{retrieved[:1500]}\n\n"
            f"Gold answer: {gold}\n\n"
            "Does the retrieved context contain sufficient information to correctly "
            "answer the question? Reply with exactly: YES, PARTIAL, or NO"
        )

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import urllib.request
            import json as _json

            body = _json.dumps(
                {
                    "model": "claude-haiku-4-5",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": prompt}],
                }
            ).encode()
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=body,
                headers={
                    "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = _json.loads(resp.read())["content"][0]["text"].strip().upper()
            return {"YES": 1.0, "PARTIAL": 0.5, "NO": 0.0}.get(result, 0.0)
        except Exception:
            return -1.0

    return -1.0


def ollama_answer(question: str, context: str) -> str:
    """Synthesise an answer from retrieved context using a local Ollama model.

    Uses CORTYX_OLLAMA_URL (default http://localhost:11434) and
    CORTYX_ANSWER_MODEL (default qwen3:8b).  Returns empty string on failure.
    """
    import json as _json
    import urllib.request

    ollama_url = os.environ.get("CORTYX_OLLAMA_URL", "http://localhost:11434")
    model = os.environ.get("CORTYX_ANSWER_MODEL", "qwen3:8b")

    prompt = (
        "You are answering questions based on retrieved memory context. "
        "Use ONLY the provided context to answer. "
        "If the answer is not in the context, say 'I don't know'.\n\n"
        f"Context:\n{context[:3000]}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely in 1-3 sentences:"
    )
    body = _json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {"num_predict": 512, "temperature": 0.1},
        }
    ).encode()
    try:
        req = urllib.request.Request(
            f"{ollama_url}/api/generate",
            data=body,
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = _json.loads(resp.read())
            return (data.get("response") or data.get("thinking") or "").strip()
    except Exception:
        return ""


# ── Runner helpers ─────────────────────────────────────────────────────────────


def _decode_output(payload) -> str:
    if payload is None:
        return ""
    if isinstance(payload, bytes):
        return payload.decode("utf-8", errors="replace")
    return str(payload)


def resolve_binary() -> str:
    candidates: list[Path] = []
    if CORTYX_BIN:
        candidates.append(Path(CORTYX_BIN))
    candidates.extend(
        [
            REPO_ROOT / "target" / "release" / "cortyx",
            REPO_ROOT / "target" / "debug" / "cortyx",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    print("ERROR: cortyx binary not found. Run: cargo build --release", file=sys.stderr)
    sys.exit(1)


def run_cortyx(args_list: list[str], cwd: Path, timeout_secs: int) -> RunResult:
    binary = resolve_binary()
    full_args = [binary] + args_list
    try:
        result = subprocess.run(
            full_args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_secs,
        )
        return RunResult(
            args=full_args,
            cwd=str(cwd),
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
        )
    except subprocess.TimeoutExpired as exc:
        return RunResult(
            args=full_args,
            cwd=str(cwd),
            stdout=_decode_output(exc.stdout),
            stderr=_decode_output(exc.stderr),
            returncode=124,
            timed_out=True,
        )


def run_git(args: list[str]) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_metadata() -> dict:
    status = run_git(["status", "--porcelain"])
    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "dirty": bool(status),
        "status_entries": len(status.splitlines()) if status else 0,
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalise_question_type(value: str) -> str:
    return str(value or "single_hop").replace("-", "_")


def parse_csv_set(raw: str) -> set[str]:
    return {normalise_question_type(part.strip()) for part in raw.split(",") if part.strip()}


def is_converted_fixture(payload) -> bool:
    if not isinstance(payload, list):
        return False
    if not payload:
        return True
    first = payload[0]
    return isinstance(first, dict) and {"session", "query", "expected_answer"} <= set(first)


def is_raw_locomo_fixture(payload) -> bool:
    def looks_like_conversation(item) -> bool:
        return isinstance(item, dict) and any(
            key in item for key in ("qa", "qa_pairs", "qas", "questions")
        )

    if isinstance(payload, list):
        return not payload or looks_like_conversation(payload[0])
    if isinstance(payload, dict):
        if "entries" in payload:
            return False
        values = list(payload.values())
        return not values or looks_like_conversation(values[0])
    return False


def load_fixture(fixture_path: Path) -> tuple[list[dict], dict]:
    payload = json.loads(fixture_path.read_text())
    fixture_meta: dict = {}
    payload_body = payload

    if isinstance(payload, dict) and "entries" in payload:
        fixture_meta = payload.get("fixture") or payload.get("meta") or {}
        payload_body = payload["entries"]

    if is_converted_fixture(payload_body):
        entries = payload_body
        source_format = "converted_entries"
        conversation_count = fixture_meta.get("conversation_count")
    elif is_raw_locomo_fixture(payload_body):
        entries = convert_locomo(payload_body)
        source_format = "raw_public_release"
        conversation_count = len(payload_body)
    else:
        raise ValueError(f"unsupported LoCoMo fixture format: {fixture_path}")

    sample_fixture = bool(fixture_meta.get("sample_fixture")) or "sample" in fixture_path.stem.lower()
    official_public_release = (
        bool(fixture_meta.get("official_public_release"))
        or source_format == "raw_public_release"
        or (len(entries) == OFFICIAL_PUBLIC_RELEASE_ROWS and not sample_fixture)
    )
    return entries, {
        "source_format": source_format,
        "sample_fixture": sample_fixture,
        "conversation_count": conversation_count,
        "official_public_release": official_public_release,
        "official_qa_rows_expected": int(
            fixture_meta.get("official_qa_rows_expected", OFFICIAL_PUBLIC_RELEASE_ROWS)
        ),
    }


def question_type_counts(entries: list[dict]) -> dict[str, int]:
    counts = Counter(normalise_question_type(entry.get("question_type", "single_hop")) for entry in entries)
    ordered = [question_type for question_type in QUESTION_TYPES if counts.get(question_type)]
    ordered.extend(sorted(key for key in counts if key not in QUESTION_TYPES))
    return {key: counts[key] for key in ordered}


def select_entries(
    entries: list[dict],
    *,
    max_entries: int,
    max_per_question_type: int,
    allowed_question_types: set[str],
) -> list[dict]:
    selected: list[dict] = []
    counts: dict[str, int] = defaultdict(int)
    for entry in entries:
        question_type = normalise_question_type(entry.get("question_type", "single_hop"))
        if allowed_question_types and question_type not in allowed_question_types:
            continue
        if max_per_question_type and counts[question_type] >= max_per_question_type:
            continue
        selected.append(entry)
        counts[question_type] += 1
        if max_entries and len(selected) >= max_entries:
            break
    return selected


def apply_profile_defaults(
    profile: str,
    *,
    max_entries: int | None,
    max_per_question_type: int | None,
) -> tuple[int, int]:
    defaults = PROFILE_DEFAULTS[profile]
    resolved_max_entries = defaults["max_entries"] if max_entries is None else max_entries
    resolved_max_per_question_type = (
        defaults["max_per_question_type"]
        if max_per_question_type is None
        else max_per_question_type
    )
    if resolved_max_entries < 0 or resolved_max_per_question_type < 0:
        raise ValueError("selection limits must be non-negative")
    return resolved_max_entries, resolved_max_per_question_type


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, RunResult):
        return asdict(value)
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def binary_fingerprint() -> dict:
    binary_path = Path(resolve_binary())
    stat = binary_path.stat()
    return {
        "path": resolve_binary(),
        "sha256": sha256_file(binary_path),
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    }


def build_locomo_manifest(entries: list[dict]) -> tuple[dict[str, str], list[dict], dict]:
    staged_files: dict[str, str] = {}
    manifest: list[dict] = []
    unique_conversations: dict[str, str] = {}

    for index, entry in enumerate(entries):
        conv_id = entry.get("conv_id", str(index))
        unique_conversations.setdefault(conv_id, entry["session"])

    for index, (conv_id, session) in enumerate(unique_conversations.items()):
        safe_id = re.sub(r"[^a-zA-Z0-9]", "_", conv_id)[:30]
        rel_path = Path("conversations") / f"session_{safe_id}_{index}.txt"
        rel_key = rel_path.as_posix()
        staged_files[rel_key] = session
        manifest.append({"path": rel_key, "sha256": sha256_text(session)})

    manifest.sort(key=lambda item: item["path"])
    return staged_files, manifest, {
        "conversation_inputs": len(unique_conversations),
        "files_staged": len(manifest),
    }


def prepare_locomo_corpus(
    entries: list[dict],
    *,
    timeout_secs: int,
    fresh_corpus: bool,
) -> dict:
    staged_files, manifest, counts = build_locomo_manifest(entries)
    binary = binary_fingerprint()
    # Cache key is content-only (manifest hash). Binary version is stored in
    # corpus.json for reference but does NOT invalidate the cache — the mined
    # index is a pure function of the conversation content, not the binary.
    cache_key = hashlib.sha256(
        json.dumps(
            {
                "benchmark": "locomo",
                "manifest": manifest,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:20]

    cache_dir = REPO_ROOT / ".cortyx_eval_work" / "cache" / "locomo" / cache_key
    project_dir = cache_dir / "project"
    conversations_dir = cache_dir / "conversations"
    metadata_path = cache_dir / "corpus.json"
    index_path = project_dir / ".cortyx" / "index.json"

    if fresh_corpus and cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)

    reused = metadata_path.exists() and index_path.exists()
    stage_secs = 0.0
    mine_secs = 0.0
    mine_run = RunResult(
        args=[resolve_binary(), "mine", str(conversations_dir)],
        cwd=str(project_dir),
        stdout="",
        stderr="",
        returncode=0,
    )

    if not reused:
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
        project_dir.mkdir(parents=True, exist_ok=True)
        conversations_dir.mkdir(parents=True, exist_ok=True)

        stage_start = time.perf_counter()
        for rel_key, content in staged_files.items():
            destination = cache_dir / rel_key
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")
        stage_secs = time.perf_counter() - stage_start

        mine_start = time.perf_counter()
        mine_timeout = max(timeout_secs, 600)
        mine_run = run_cortyx(["mine", str(conversations_dir)], project_dir, mine_timeout)
        mine_secs = time.perf_counter() - mine_start
        if not mine_run.ok:
            shutil.rmtree(cache_dir, ignore_errors=True)
            return {
                "reused": False,
                "cache_key": cache_key,
                "path": str(cache_dir),
                "project_dir": project_dir,
                "conversations_dir": conversations_dir,
                "binary": binary,
                "stage_secs": round(stage_secs, 4),
                "mine_secs": round(mine_secs, 4),
                "metadata": {},
                "mine_run": mine_run,
                **counts,
            }

        metadata = {
            "generated_at_utc": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "cache_key": cache_key,
            "binary": binary,
            **counts,
            "stage_secs": round(stage_secs, 4),
            "mine_secs": round(mine_secs, 4),
            "project_dir": str(project_dir),
            "conversations_dir": str(conversations_dir),
        }
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    else:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}

    return {
        "reused": reused,
        "cache_key": cache_key,
        "path": str(cache_dir),
        "project_dir": project_dir,
        "conversations_dir": conversations_dir,
        "binary": binary,
        "stage_secs": round(stage_secs, 4),
        "mine_secs": round(mine_secs, 4),
        "metadata": metadata,
        "mine_run": mine_run,
        **counts,
    }


def trim_text(text: str, limit: int = 240) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def record_failure(failures: list[dict], stage: str, group: str, run: RunResult) -> None:
    failures.append(failure_from_run(stage, group, run))


def failure_from_run(stage: str, group: str, run: RunResult) -> dict:
    return {
        "stage": stage,
        "group": group,
        "command": " ".join(run.args),
        "cwd": run.cwd,
        "timed_out": run.timed_out,
        "returncode": run.returncode,
        "stderr_excerpt": trim_text(run.stderr or run.stdout, limit=320),
    }


def evaluate_entry(
    *,
    index: int,
    entry: dict,
    question_type: str,
    project_dir: Path,
    answer_mode: bool,
    use_llm: bool,
    llm_answer: bool,
    timeout_secs: int,
    min_answer_confidence: float | None,
) -> dict:
    query = entry["query"]
    gold = entry.get("expected_answer", "")
    args = [
        "get-contexts",
        "--task",
        query,
        "--max-tokens",
        "4000",
        "--kind",
        "conversation",
    ]
    if answer_mode:
        args.append("--answer-mode")
    if min_answer_confidence is not None:
        args.extend(["--min-answer-confidence", str(min_answer_confidence)])

    query_start = time.perf_counter()
    query_run = run_cortyx(args, project_dir, timeout_secs)
    query_elapsed = time.perf_counter() - query_start

    retrieved = query_run.stdout if query_run.stdout else query_run.stderr
    prediction = retrieved.strip()

    if llm_answer and retrieved.strip():
        synth = ollama_answer(query, retrieved)
        if synth:
            prediction = synth

    f1 = f1_score(prediction, gold)
    em = exact_match(prediction, gold)
    recall = recall_score(prediction, gold)

    judge_score = None
    if use_llm and gold:
        llm_score = llm_judge(query, retrieved, gold, answer_mode=answer_mode)
        if llm_score >= 0:
            judge_score = llm_score

    return {
        "index": index,
        "f1": f1,
        "em": em,
        "recall": recall,
        "judge_score": judge_score,
        "query_secs": query_elapsed,
        "failure": None
        if query_run.ok
        else failure_from_run("query", question_type, query_run),
        "row": {
            "conv_id": str(entry.get("conv_id", index)),
            "question_type": question_type,
            "query": query,
            "gold_answer": gold,
            "prediction": prediction,
            "prediction_excerpt": trim_text(prediction, 240),
            "f1": round(f1, 4),
            "em": round(float(em), 4),
            "recall": round(recall, 4),
            "timed_out": query_run.timed_out,
            "returncode": query_run.returncode,
            "query_secs": round(query_elapsed, 4),
        },
    }


# ── Evaluation ─────────────────────────────────────────────────────────────────


def hardest_cases(case_rows: list[dict]) -> list[dict]:
    return sorted(
        case_rows,
        key=lambda item: (item["f1"], item["em"], item["recall"], item["question_type"], item["query"]),
    )[:5]


def build_overall(results: dict) -> dict:
    if not results:
        return {}

    total_n = sum(value["n"] for value in results.values())
    return {
        "groups_scored": len(results),
        "entries_scored": total_n,
        "macro_f1": round(sum(value["f1"] for value in results.values()) / len(results), 4),
        "macro_em": round(sum(value["em"] for value in results.values()) / len(results), 4),
        "macro_recall": round(sum(value["recall"] for value in results.values()) / len(results), 4),
        "micro_f1": round(
            sum(value["f1"] * value["n"] for value in results.values()) / total_n,
            4,
        ),
        "micro_em": round(
            sum(value["em"] * value["n"] for value in results.values()) / total_n,
            4,
        ),
        "micro_recall": round(
            sum(value["recall"] * value["n"] for value in results.values()) / total_n,
            4,
        ),
    }


def build_proof(
    fixture_path: Path,
    *,
    fixture_info: dict,
    entries_total: int,
    entries_evaluated: int,
    answer_mode: bool,
    llm_answer: bool,
    failures: list[dict],
) -> dict:
    blockers: list[str] = []
    if entries_evaluated < entries_total:
        blockers.append(f"partial run evaluated {entries_evaluated}/{entries_total} LoCoMo rows")
    if fixture_info["sample_fixture"]:
        blockers.append(
            f"fixture {fixture_path.name} is a sample slice, not the official full LoCoMo split"
        )
    elif entries_total != fixture_info["official_qa_rows_expected"]:
        blockers.append(
            "fixture "
            f"{fixture_path.name} has {entries_total} rows; comparator-ready LoCoMo proof expects "
            f"{fixture_info['official_qa_rows_expected']} answer-graded QA rows"
        )
    if not answer_mode and not llm_answer:
        blockers.append(
            "published LoCoMo references are answer-grade QA F1, not retrieval-context overlap"
        )
    if failures:
        blockers.append(f"{len(failures)} infrastructure failure(s) occurred during evaluation")

    return {
        "reproducible": not failures,
        "comparator_ready": not blockers and not failures,
        "blocking_reasons": blockers,
    }


def evaluate(
    entries: list[dict],
    *,
    answer_mode: bool,
    use_llm: bool,
    llm_answer: bool,
    timeout_secs: int,
    min_answer_confidence: float | None,
    fresh_corpus: bool,
    jobs: int,
) -> dict:
    run_start = time.perf_counter()
    print(f"Evaluating {len(entries)} entries")
    print("=" * 60)

    by_type: dict[str, list] = defaultdict(list)
    for entry in entries:
        by_type[normalise_question_type(entry.get("question_type", "single_hop"))].append(entry)

    results: dict[str, dict] = {}
    failures: list[dict] = []
    case_rows: list[dict] = []

    if not entries:
        return {
            "results": results,
            "overall": build_overall(results),
            "case_rows": case_rows,
            "diagnostics": {
                "infra_failures": failures,
                "hard_cases": hardest_cases(case_rows),
            },
            "corpus": {},
            "timings": {
                "corpus_stage_secs": 0.0,
                "corpus_mine_secs": 0.0,
                "query_secs_total": 0.0,
                "avg_query_secs": 0.0,
                "query_count": 0,
                "query_secs_by_question_type": {},
                "parallel_jobs": jobs,
                "total_secs": round(time.perf_counter() - run_start, 4),
            },
        }

    corpus = prepare_locomo_corpus(
        entries,
        timeout_secs=timeout_secs,
        fresh_corpus=fresh_corpus,
    )
    corpus_state = "reused" if corpus["reused"] else "built"
    print(
        "[corpus] "
        f"{corpus_state} {corpus['path']} "
        f"({corpus['conversation_inputs']} conversations)"
    )
    if not corpus["mine_run"].ok:
        record_failure(failures, "mine", "corpus", corpus["mine_run"])
        print(
            "  WARNING: corpus mine failed: "
            f"{trim_text(corpus['mine_run'].stderr or corpus['mine_run'].stdout)}"
        )
        return {
            "results": results,
            "overall": build_overall(results),
            "case_rows": case_rows,
            "diagnostics": {
                "infra_failures": failures,
                "hard_cases": hardest_cases(case_rows),
            },
            "corpus": corpus,
            "timings": {
                "corpus_stage_secs": corpus["stage_secs"],
                "corpus_mine_secs": corpus["mine_secs"],
                "query_secs_total": 0.0,
                "avg_query_secs": 0.0,
                "query_count": 0,
                "query_secs_by_question_type": {},
                "parallel_jobs": jobs,
                "total_secs": round(time.perf_counter() - run_start, 4),
            },
        }

    project_dir = corpus["project_dir"]
    query_secs_total = 0.0
    query_secs_by_question_type: dict[str, float] = {}

    for question_type in QUESTION_TYPES:
        type_entries = by_type.get(question_type, [])
        if not type_entries:
            continue

        print(f"\n[{question_type}] {len(type_entries)} entries")

        f1_sum = 0.0
        em_count = 0
        recall_sum = 0.0
        llm_sum = 0.0
        llm_n = 0
        type_query_secs = 0.0
        evaluated_entries: list[dict] = []

        def accumulate(evaluated: dict, completed: int) -> None:
            nonlocal f1_sum, em_count, recall_sum, llm_sum, llm_n, type_query_secs, query_secs_total

            f1_sum += evaluated["f1"]
            em_count += int(evaluated["em"])
            recall_sum += evaluated["recall"]
            type_query_secs += evaluated["query_secs"]
            query_secs_total += evaluated["query_secs"]
            if evaluated["judge_score"] is not None:
                llm_sum += evaluated["judge_score"]
                llm_n += 1
            if evaluated["failure"] is not None:
                failures.append(evaluated["failure"])

            if completed % 20 == 0:
                print(
                    f"  … {completed}/{len(type_entries)} "
                    f"(F1={f1_sum / completed:.3f}, EM={em_count / completed:.3f}, "
                    f"Recall={recall_sum / completed:.3f})",
                    flush=True,
                )

        if jobs == 1 or len(type_entries) == 1:
            for i, entry in enumerate(type_entries):
                evaluated = evaluate_entry(
                    index=i,
                    entry=entry,
                    question_type=question_type,
                    project_dir=project_dir,
                    answer_mode=answer_mode,
                    use_llm=use_llm,
                    llm_answer=llm_answer,
                    timeout_secs=timeout_secs,
                    min_answer_confidence=min_answer_confidence,
                )
                evaluated_entries.append(evaluated)
                accumulate(evaluated, i + 1)
        else:
            with ThreadPoolExecutor(max_workers=min(jobs, len(type_entries))) as executor:
                futures = {
                    executor.submit(
                        evaluate_entry,
                        index=i,
                        entry=entry,
                        question_type=question_type,
                        project_dir=project_dir,
                        answer_mode=answer_mode,
                        use_llm=use_llm,
                        llm_answer=llm_answer,
                        timeout_secs=timeout_secs,
                        min_answer_confidence=min_answer_confidence,
                    ): i
                    for i, entry in enumerate(type_entries)
                }
                for completed, future in enumerate(as_completed(futures), start=1):
                    evaluated = future.result()
                    evaluated_entries.append(evaluated)
                    accumulate(evaluated, completed)

        evaluated_entries.sort(key=lambda item: item["index"])
        case_rows.extend(item["row"] for item in evaluated_entries)

        n = len(type_entries)
        type_result = {
            "n": n,
            "f1": round(f1_sum / n, 4),
            "em": round(em_count / n, 4),
            "recall": round(recall_sum / n, 4),
        }
        if llm_n > 0:
            type_result["llm_judge"] = round(llm_sum / llm_n, 4)

        results[question_type] = type_result
        query_secs_by_question_type[question_type] = round(type_query_secs, 4)
        print(
            f"  → F1={type_result['f1']:.3f}  EM={type_result['em']:.3f}  "
            f"Recall={type_result['recall']:.3f}"
        )

    return {
        "results": results,
        "overall": build_overall(results),
        "case_rows": case_rows,
        "diagnostics": {
            "infra_failures": failures,
            "hard_cases": hardest_cases(case_rows),
        },
        "corpus": corpus,
        "timings": {
            "corpus_stage_secs": corpus["stage_secs"],
            "corpus_mine_secs": corpus["mine_secs"],
            "query_secs_total": round(query_secs_total, 4),
            "avg_query_secs": round(query_secs_total / len(entries), 4) if entries else 0.0,
            "query_count": len(entries),
            "query_secs_by_question_type": query_secs_by_question_type,
            "parallel_jobs": jobs,
            "total_secs": round(time.perf_counter() - run_start, 4),
        },
    }


# ── Reporting ──────────────────────────────────────────────────────────────────


def build_report(
    fixture_path: Path,
    all_entries: list[dict],
    selected_entries: list[dict],
    *,
    fixture_info: dict,
    profile: str,
    answer_mode: bool,
    use_llm: bool,
    llm_answer: bool,
    timeout_secs: int,
    min_answer_confidence: float | None,
    selected_question_types: set[str],
    evaluation: dict,
    jobs: int,
) -> dict:
    failures = evaluation["diagnostics"]["infra_failures"]
    if llm_answer:
        mode = "llm-answer"
    elif answer_mode:
        mode = "answer"
    else:
        mode = "retrieval"
    return {
        "benchmark": "locomo",
        "mode": mode,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "fixture": {
            "path": str(fixture_path),
            "sha256": sha256_file(fixture_path),
            "entries_total": len(all_entries),
            "entries_evaluated": len(selected_entries),
            "sample_fixture": fixture_info["sample_fixture"],
            "source_format": fixture_info["source_format"],
            "conversation_count": fixture_info["conversation_count"],
            "official_public_release": fixture_info["official_public_release"],
            "official_qa_rows_expected": fixture_info["official_qa_rows_expected"],
        },
        "selection": {
            "profile": profile,
            "question_types": sorted(selected_question_types),
            "question_type_counts_total": question_type_counts(all_entries),
            "question_type_counts_evaluated": question_type_counts(selected_entries),
            "full_run": len(selected_entries) == len(all_entries),
        },
        "run": {
            "cortyx_bin": evaluation.get("corpus", {}).get("binary", {}).get(
                "path", resolve_binary()
            ),
            "timeout_secs": timeout_secs,
            "llm_judge": use_llm,
            "llm_answer": llm_answer,
            "llm_answer_model": os.environ.get("CORTYX_ANSWER_MODEL", "qwen3:8b") if llm_answer else None,
            "min_answer_confidence": min_answer_confidence,
            "git": git_metadata(),
            "corpus_cache": evaluation.get("corpus", {}),
            "timings": evaluation.get("timings", {}),
            "proof_strategy": {
                "mode": "parallel_queries" if jobs > 1 else "serial_queries",
                "jobs": jobs,
            },
        },
        "overall": evaluation["overall"],
        "results": evaluation["results"],
        "cases": {
            "count": len(evaluation["case_rows"]),
            "rows": evaluation["case_rows"],
        },
        "diagnostics": evaluation["diagnostics"],
        "proof": build_proof(
            fixture_path,
            fixture_info=fixture_info,
            entries_total=len(all_entries),
            entries_evaluated=len(selected_entries),
            answer_mode=answer_mode,
            llm_answer=llm_answer,
            failures=failures,
        ),
    }


def print_summary(report: dict):
    print("\n" + "=" * 60)
    print("LoCoMo Evaluation Summary")
    print("=" * 60)

    results = report["results"]
    if not results:
        print("No scored LoCoMo question types matched this fixture selection.")
        return

    print(f"{'Question Type':<20} {'N':>5} {'F1':>7} {'EM':>7} {'Recall':>8}")
    print("-" * 52)
    for question_type, values in results.items():
        print(
            f"{question_type:<20} {values['n']:>5} {values['f1']:>7.3f} "
            f"{values['em']:>7.3f} {values['recall']:>8.3f}"
        )

    overall = report["overall"]
    print("-" * 52)
    print(
        f"{'OVERALL (macro avg)':<20} {'':>5} "
        f"{overall['macro_f1']:>7.3f} {overall['macro_em']:>7.3f} {overall['macro_recall']:>8.3f}"
    )

    selection = report.get("selection", {})
    run = report.get("run", {})
    corpus = run.get("corpus_cache") or {}
    timings = run.get("timings") or {}
    print("\nFeedback loop:")
    print(f"  profile: {selection.get('profile', 'full')}")
    if corpus:
        corpus_state = "reused" if corpus.get("reused") else "built"
        print(
            f"  corpus: {corpus_state} {corpus.get('path')} "
            f"({corpus.get('conversation_inputs', 0)} conversations)"
        )
    if timings:
        print(
            "  timings: "
            f"total={timings.get('total_secs', 0.0):.2f}s | "
            f"stage={timings.get('corpus_stage_secs', 0.0):.2f}s | "
            f"mine={timings.get('corpus_mine_secs', 0.0):.2f}s | "
            f"query={timings.get('query_secs_total', 0.0):.2f}s "
            f"({timings.get('avg_query_secs', 0.0):.2f}s/query)"
        )
        if timings.get("parallel_jobs", 1) > 1:
            print(f"  query jobs: {timings['parallel_jobs']}")

    print("\nBaseline reference (LoCoMo QA F1):")
    print("  Hindsight (open-source):  ~89.6%")
    print("  Zep:                       ~85.0%")
    print("  Letta / MemGPT:            ~83.2%")
    print("  Mem0:                      58–67%")
    if report["mode"] == "llm-answer":
        model = report.get("run", {}).get("llm_answer_model", "qwen3:8b")
        print(f"  Note: llm-answer mode — Cortyx retrieves context, {model} synthesises the answer.")
        print("        Directly comparable to Hindsight/Zep/Letta/Mem0 published F1 scores.")
    elif report["mode"] == "answer":
        print("  Note: answer mode scores predicted answers, which matches public QA-style baselines.")
    else:
        print("  Note: retrieval mode scores supporting context, not final QA answers.")

    proof = report["proof"]
    print("\nProof status:")
    print(f"  reproducible: {'yes' if proof['reproducible'] else 'no'}")
    print(f"  comparator ready: {'yes' if proof['comparator_ready'] else 'no'}")
    for blocker in proof["blocking_reasons"]:
        print(f"  blocker: {blocker}")

    hard_cases = report["diagnostics"].get("hard_cases", [])
    if hard_cases:
        print("\nHard cases:")
        for case in hard_cases[:3]:
            print(
                f"  - [{case['question_type']}] {case['query']} "
                f"(F1={case['f1']:.3f}, EM={case['em']:.3f}, Recall={case['recall']:.3f})"
            )


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--fixture", default=FIXTURE_DEFAULT)
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Where to write the structured JSON report",
    )
    parser.add_argument(
        "--answer-mode",
        action="store_true",
        help="Score predicted answers instead of retrieved context blocks",
    )
    parser.add_argument(
        "--min-answer-confidence",
        type=float,
        default=None,
        help="Optional cortyx --min-answer-confidence value for answer-mode runs",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_DEFAULTS),
        default="full",
        help="Sampling preset: full=all rows, quick=2 rows/question type, smoke=1 row/question type",
    )
    parser.add_argument("--llm-judge", action="store_true")
    parser.add_argument(
        "--llm-answer",
        action="store_true",
        help=(
            "Cortyx retrieves context, then Ollama synthesises the answer (CORTYX_ANSWER_MODEL, "
            "default qwen3:8b). Produces answer-grade F1 directly comparable to "
            "Hindsight/Zep/Letta/Mem0 published baselines. Requires ollama serve."
        ),
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Limit to first N selected rows after applying --profile defaults (0 = all)",
    )
    parser.add_argument(
        "--max-per-question-type",
        type=int,
        default=None,
        help="Cap selected rows per question type after applying --profile defaults",
    )
    parser.add_argument(
        "--question-types",
        default="",
        help="Comma-separated question type filter (default: all question types)",
    )
    parser.add_argument(
        "--timeout-secs",
        type=int,
        default=ENV_TIMEOUT_SECS,
        help="Per-cortyx command timeout in seconds",
    )
    parser.add_argument(
        "--fresh-corpus",
        action="store_true",
        help="Rebuild the staged corpus/index instead of reusing the cached corpus for the same binary+selection",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Run up to N query subprocesses in parallel within each question type",
    )
    args = parser.parse_args()

    fixture_path = Path(args.fixture)
    if not fixture_path.exists():
        print(f"ERROR: fixture not found: {fixture_path}")
        print("Generate it first: python3 scripts/gen_locomo.py")
        return 1

    answer_mode = args.answer_mode or ENV_ANSWER_MODE
    if args.jobs <= 0:
        print("ERROR: --jobs must be positive")
        return 1
    if args.min_answer_confidence is not None and not answer_mode:
        print("ERROR: --min-answer-confidence requires --answer-mode")
        return 1

    try:
        all_entries, fixture_info = load_fixture(fixture_path)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1
    allowed_question_types = parse_csv_set(args.question_types)
    try:
        max_entries, max_per_question_type = apply_profile_defaults(
            args.profile,
            max_entries=args.max_entries,
            max_per_question_type=args.max_per_question_type,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1
    selected_entries = select_entries(
        all_entries,
        max_entries=max_entries,
        max_per_question_type=max_per_question_type,
        allowed_question_types=allowed_question_types,
    )
    selected_question_types = {
        normalise_question_type(entry.get("question_type", "single_hop"))
        for entry in selected_entries
    }

    print(
        f"Evaluating {len(selected_entries)} entries from {fixture_path} "
        f"(profile={args.profile})"
    )
    try:
        evaluation = evaluate(
            selected_entries,
            answer_mode=answer_mode,
            use_llm=args.llm_judge,
            llm_answer=args.llm_answer,
            timeout_secs=args.timeout_secs,
            min_answer_confidence=args.min_answer_confidence,
            fresh_corpus=args.fresh_corpus,
            jobs=args.jobs,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1
    report = build_report(
        fixture_path,
        all_entries,
        selected_entries,
        fixture_info=fixture_info,
        profile=args.profile,
        answer_mode=answer_mode,
        use_llm=args.llm_judge,
        llm_answer=args.llm_answer,
        timeout_secs=args.timeout_secs,
        min_answer_confidence=args.min_answer_confidence,
        selected_question_types=selected_question_types,
        evaluation=evaluation,
        jobs=args.jobs,
    )
    print_summary(report)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(json_ready(report), indent=2) + "\n")
    print(f"\n✓ Results saved to {output_path}")

    return 0 if report["proof"]["reproducible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
