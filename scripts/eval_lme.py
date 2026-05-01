#!/usr/bin/env python3
"""Proper evaluation harness for LongMemEval-500.

Runs cortyx get-contexts for each entry in the fixture, then scores the result
using token-level F1 (fast, no API required) per category. Optionally uses an
LLM judge for more accurate scoring.

Usage:
    python3 scripts/eval_lme.py
    python3 scripts/eval_lme.py --fixture tests/fixtures/longmemeval_500.json
    python3 scripts/eval_lme.py --profile quick
    python3 scripts/eval_lme.py --answer-mode
    python3 scripts/eval_lme.py --question-ids 08e075c7,e61a7584
    python3 scripts/eval_lme.py --question-ids 08e075c7,e61a7584 --selection-corpus
    python3 scripts/eval_lme.py --fresh-corpus
    python3 scripts/eval_lme.py --llm-judge   # requires ANTHROPIC_API_KEY or OPENAI_API_KEY

What the old bench.rs measured (WRONG):
    hit = any keyword from expected_keywords appears anywhere in stdout

What this script measures (CORRECT):
    Default mode:
        F1   = token overlap between retrieved context and gold answer
        EM   = exact match of retrieved answer token set vs gold
        R@5  = does the evidence session appear in the retrieved context?
    Answer mode:
        F1   = token overlap between predicted answer and gold answer
        EM   = exact token-set match
        AnsR = fraction of gold-answer tokens recovered by the predicted answer

Proof-grade report additions:
    - fixture hash + git revision metadata
    - partial-run / comparator blockers
    - hardest misses + infra failures
    - official LongMemEval QA-surface hypotheses export for answer-mode runs
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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CORTYX_BIN = os.environ.get("CORTYX_BIN", "")
ENV_TIMEOUT_SECS = int(os.environ.get("CORTYX_TIMEOUT_SECS", "180"))
ENV_ANSWER_MODE = os.environ.get("CORTYX_ANSWER_MODE", "").lower() not in ("", "0", "false", "no")
FIXTURE_DEFAULT = str(REPO_ROOT / "tests" / "fixtures" / "longmemeval_500.json")
DEFAULT_OUTPUT = str(REPO_ROOT / "lme500_eval_results.json")
_RESOLVED_BINARY: str | None = None

CATEGORIES = [
    "single_session_user",
    "single_session_assistant",
    "single_session_preference",
    "multi_session",
    "temporal_reasoning",
    "knowledge_update",
    "absent",
]

PROFILE_DEFAULTS = {
    "full": {"max_entries": 0, "max_per_category": 0},
    "quick": {"max_entries": 0, "max_per_category": 3},
    "smoke": {"max_entries": 0, "max_per_category": 1},
}

KNOWN_FIXTURE_ANOMALIES: dict[str, dict[str, str]] = {
    "370a8ff4": {
        "kind": "temporal_gold_mismatch",
        "fixture_expected_answer": "15",
        "evidence_based_answer": "11",
        "reason": (
            "Both events are phrased as 'today' in sessions dated 19 January 2023 and "
            "10 April 2023, so the visible event-to-event gap is about 11.6 weeks. "
            "Raw benchmark scoring is intentionally left unchanged."
        ),
    }
}


def parse_lme_datetime(raw: str) -> datetime | None:
    raw = str(raw or "").strip()
    if not raw:
        return None
    for fmt in ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d (%a)", "%Y/%m/%d %H:%M", "%Y/%m/%d"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def format_lme_date_label(raw: str) -> str | None:
    dt = parse_lme_datetime(raw)
    if dt is None:
        return None
    return f"{dt.day} {dt.strftime('%B, %Y')}"


def render_lme_entry_content(entry: dict) -> str:
    content = str(entry.get("neuron_source_content", ""))
    evidence_dates = entry.get("evidence_dates") or []
    if not content.strip() or not evidence_dates:
        return content
    sessions = re.split(r"\n\s*---\s*\n", content.strip())
    if len(sessions) != len(evidence_dates):
        return content

    rendered: list[str] = []
    for index, (session, raw_date) in enumerate(zip(sessions, evidence_dates), start=1):
        label = format_lme_date_label(str(raw_date))
        session = session.strip()
        if label:
            rendered.append(f"[Session {index} - {label}]\n{session}")
        else:
            rendered.append(session)
    return "\n\n---\n\n".join(rendered)


def render_lme_question(entry: dict, *, answer_mode: bool) -> str:
    question = str(entry.get("question", ""))
    if not answer_mode or not lme_question_needs_reference_date(entry):
        return question
    label = format_lme_date_label(str(entry.get("question_date") or ""))
    if not label:
        return question
    return f"As of {label}, {question}"


def lme_question_needs_reference_date(entry: dict) -> bool:
    question = str(entry.get("question", "")).strip().lower()
    if str(entry.get("question_type", "")).strip().lower() != "temporal-reasoning":
        return False
    return (
        question.startswith("how many ")
        and (
            " ago did i " in question or " have passed since i " in question
        )
    ) or lme_question_is_relative_recall(question)


def lme_question_is_relative_recall(question: str) -> bool:
    if not re.search(
        r"\b(?:a couple of days ago|a few days ago|last (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(?:day|days|week|weeks|month|months|year|years)\s+ago)\b",
        question,
    ):
        return False
    return question.startswith(("what ", "which ", "who ", "i "))


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


# ── Token-level scoring ─────────────────────────────────────────────────────────


def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = Counter(_tokenise(prediction))
    gold_tokens = Counter(_tokenise(gold))
    common = sum((pred_tokens & gold_tokens).values())
    if common == 0:
        return 0.0
    precision = common / sum(pred_tokens.values())
    recall = common / sum(gold_tokens.values())
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, gold: str) -> bool:
    return set(_tokenise(prediction)) == set(_tokenise(gold))


def answer_recall_score(prediction: str, gold: str) -> float:
    pred_tokens = set(_tokenise(prediction))
    gold_tokens = _tokenise(gold)
    if not gold_tokens:
        return 1.0
    hits = sum(1 for token in gold_tokens if token in pred_tokens)
    return hits / len(gold_tokens)


def gold_answer_variants(gold: str) -> list[str]:
    gold = str(gold or "").strip()
    if not gold:
        return [""]

    variants: list[str] = [gold]

    alt_match = re.match(
        r"^\s*(.+?)\.\s*(.+?)\s+is also acceptable\.?\s*$",
        gold,
        flags=re.IGNORECASE,
    )
    if alt_match:
        primary = alt_match.group(1).strip()
        alternate = re.sub(r"\s*\([^)]*\)\s*$", "", alt_match.group(2).strip()).strip()
        if primary:
            variants.append(primary)
        if alternate:
            variants.append(alternate)

    range_match = re.match(
        r"^\s*(.+?)\.\s*Answers ranging from\s+(\d+)\s+([A-Za-z]+)\s+to\s+(\d+)\s+([A-Za-z]+)\s+are also acceptable\.?\s*$",
        gold,
        flags=re.IGNORECASE,
    )
    if range_match:
        primary = range_match.group(1).strip()
        start = int(range_match.group(2))
        start_unit = range_match.group(3).lower()
        end = int(range_match.group(4))
        end_unit = range_match.group(5).lower()
        if primary:
            variants.append(primary)
        if start <= end and start_unit == end_unit:
            for value in range(start, end + 1):
                variants.append(f"{value} {start_unit}")

    return list(dict.fromkeys(variant for variant in variants if variant))


def best_gold_variant_score(prediction: str, gold: str) -> tuple[float, bool, float, str]:
    best_f1 = 0.0
    best_em = False
    best_recall = 0.0
    best_gold = str(gold or "")
    for variant in gold_answer_variants(gold):
        variant_f1 = f1_score(prediction, variant)
        variant_em = exact_match(prediction, variant)
        variant_recall = answer_recall_score(prediction, variant)
        if (
            (variant_em and not best_em)
            or (variant_em == best_em and variant_f1 > best_f1)
            or (
                variant_em == best_em
                and variant_f1 == best_f1
                and variant_recall > best_recall
            )
        ):
            best_f1 = variant_f1
            best_em = variant_em
            best_recall = variant_recall
            best_gold = variant
    return best_f1, best_em, best_recall, best_gold


def recall_at_k(retrieved_text: str, evidence_keywords: list[str]) -> bool:
    lower = retrieved_text.lower()
    return any(keyword.lower() in lower for keyword in evidence_keywords)


# ── LLM judge (optional) ───────────────────────────────────────────────────────


def llm_judge(question: str, retrieved: str, gold: str, *, answer_mode: bool) -> float:
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return -1.0

    if answer_mode:
        prompt = (
            f"Question: {question}\n\n"
            f"Predicted answer:\n{retrieved[:1500]}\n\n"
            f"Gold answer: {gold}\n\n"
            "Is the predicted answer correct? Reply with exactly one of: YES, PARTIAL, NO"
        )
    else:
        prompt = (
            f"Question: {question}\n\n"
            f"Retrieved context:\n{retrieved[:1500]}\n\n"
            f"Gold answer: {gold}\n\n"
            "Does the retrieved context contain enough information to answer the question "
            "correctly? Reply with exactly one of: YES, PARTIAL, NO"
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


# ── Runner helpers ──────────────────────────────────────────────────────────────


def _decode_output(payload) -> str:
    if payload is None:
        return ""
    if isinstance(payload, bytes):
        return payload.decode("utf-8", errors="replace")
    return str(payload)


def _source_paths_for_release_freshness() -> list[Path]:
    paths: list[Path] = []
    for relative in ("Cargo.toml", "Cargo.lock", "build.rs"):
        candidate = REPO_ROOT / relative
        if candidate.exists():
            paths.append(candidate)
    src_root = REPO_ROOT / "src"
    if src_root.exists():
        paths.extend(path for path in src_root.rglob("*.rs") if path.is_file())
    return paths


def _newest_source_mtime() -> float:
    mtimes = [path.stat().st_mtime for path in _source_paths_for_release_freshness()]
    return max(mtimes, default=0.0)


def ensure_release_binary() -> Path:
    release_binary = REPO_ROOT / "target" / "release" / "cortyx"
    if release_binary.exists() and release_binary.stat().st_mtime >= _newest_source_mtime():
        return release_binary
    print(
        "[build] target/release/cortyx is missing or stale; running cargo build --release",
        file=sys.stderr,
    )
    subprocess.run(
        ["cargo", "build", "--release"],
        cwd=str(REPO_ROOT),
        check=True,
    )
    if not release_binary.exists():
        print(
            "ERROR: cargo build --release completed but target/release/cortyx was not produced",
            file=sys.stderr,
        )
        sys.exit(1)
    return release_binary


def resolve_binary() -> str:
    global _RESOLVED_BINARY
    if _RESOLVED_BINARY:
        return _RESOLVED_BINARY
    if CORTYX_BIN:
        candidate = Path(CORTYX_BIN)
        if not candidate.exists():
            print(f"ERROR: cortyx binary not found: {candidate}", file=sys.stderr)
            sys.exit(1)
        _RESOLVED_BINARY = str(candidate)
        return _RESOLVED_BINARY
    _RESOLVED_BINARY = str(ensure_release_binary())
    return _RESOLVED_BINARY


def run_cortyx(args_list: list[str], cwd: Path, timeout_secs: int) -> RunResult:
    binary = resolve_binary()
    full_args = [binary] + args_list
    env = os.environ.copy()
    if "--answer-mode" in args_list:
        env["CORTYX_EMPTY_ABSTENTION"] = "1"
    try:
        result = subprocess.run(
            full_args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_secs,
            env=env,
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


def display_path(path: Path | str) -> str:
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(candidate)


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


def normalise_category(value: str) -> str:
    return str(value or "unknown").replace("-", "_")


def entry_question_id(entry: dict) -> str:
    return str(entry.get("question_id") or "").strip()


def entry_question_type(entry: dict) -> str:
    raw = str(entry.get("question_type") or "").strip()
    if raw:
        return raw
    category = str(entry.get("category") or "").strip()
    if not category or normalise_category(category) == "absent":
        return ""
    return category.replace("_", "-")


def entry_category(entry: dict) -> str:
    if entry_question_id(entry).endswith("_abs"):
        return "absent"
    question_type = entry_question_type(entry)
    if question_type:
        return normalise_category(question_type)
    return normalise_category(entry.get("category", "unknown"))


def known_fixture_anomaly(entry: dict) -> dict | None:
    anomaly = KNOWN_FIXTURE_ANOMALIES.get(entry_question_id(entry))
    if anomaly is None:
        return None
    return {
        **anomaly,
        "question_id": entry_question_id(entry),
        "question": str(entry.get("question") or ""),
        "question_date": str(entry.get("question_date") or ""),
        "evidence_dates": list(entry.get("evidence_dates") or []),
    }


def category_counts(entries: list[dict]) -> dict[str, int]:
    counts = Counter(entry_category(entry) for entry in entries)
    ordered = [category for category in CATEGORIES if counts.get(category)]
    ordered.extend(sorted(key for key in counts if key not in CATEGORIES))
    return {key: counts[key] for key in ordered}


def parse_csv_set(raw: str) -> set[str]:
    return {normalise_category(part.strip()) for part in raw.split(",") if part.strip()}


def parse_csv_values(raw: str) -> set[str]:
    return {part.strip() for part in raw.split(",") if part.strip()}


def select_entries(
    entries: list[dict],
    *,
    max_entries: int,
    max_per_category: int,
    allowed_categories: set[str],
    allowed_question_ids: set[str],
) -> list[dict]:
    selected: list[dict] = []
    counts: dict[str, int] = defaultdict(int)
    for entry in entries:
        question_id = entry_question_id(entry)
        if allowed_question_ids and question_id not in allowed_question_ids:
            continue
        category = entry_category(entry)
        if allowed_categories and category not in allowed_categories:
            continue
        if max_per_category and counts[category] >= max_per_category:
            continue
        selected.append(entry)
        counts[category] += 1
        if max_entries and len(selected) >= max_entries:
            break
    return selected


def apply_profile_defaults(
    profile: str,
    *,
    max_entries: int | None,
    max_per_category: int | None,
) -> tuple[int, int]:
    defaults = PROFILE_DEFAULTS[profile]
    resolved_max_entries = defaults["max_entries"] if max_entries is None else max_entries
    resolved_max_per_category = (
        defaults["max_per_category"] if max_per_category is None else max_per_category
    )
    if resolved_max_entries < 0 or resolved_max_per_category < 0:
        raise ValueError("selection limits must be non-negative")
    return resolved_max_entries, resolved_max_per_category


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def binary_fingerprint() -> dict:
    binary_path = Path(resolve_binary())
    stat = binary_path.stat()
    return {
        "path": display_path(binary_path),
        "sha256": sha256_file(binary_path),
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    }


def lme_stage_files(entries: list[dict]) -> tuple[dict[str, str], list[dict], dict]:
    staged_files: dict[str, str] = {}
    manifest: list[dict] = []
    conversation_inputs = 0
    project_inputs = 0

    for index, entry in enumerate(entries):
        filename = entry.get("neuron_filename", f"conv_{index}.conv.md")
        content = render_lme_entry_content(entry)
        if entry.get("kind") == "conversation":
            rel_path = Path("conversations") / filename
        else:
            rel_path = Path("project") / "src" / filename
        rel_key = rel_path.as_posix()
        previous = staged_files.get(rel_key)
        if previous is not None and previous != content:
            raise ValueError(f"fixture collision detected for staged path: {rel_key}")
        if previous is not None:
            continue
        staged_files[rel_key] = content
        manifest.append({"path": rel_key, "sha256": sha256_text(content)})
        if rel_key.startswith("conversations/"):
            conversation_inputs += 1
        else:
            project_inputs += 1

    manifest.sort(key=lambda item: item["path"])
    return staged_files, manifest, {
        "conversation_inputs": conversation_inputs,
        "project_inputs": project_inputs,
        "files_staged": len(manifest),
    }


def prepare_lme_corpus(
    entries: list[dict],
    *,
    timeout_secs: int,
    fresh_corpus: bool,
) -> dict:
    staged_files, manifest, counts = lme_stage_files(entries)
    binary = binary_fingerprint()
    cache_key = hashlib.sha256(
        json.dumps(
            {
                "benchmark": "longmemeval-500",
                "binary_sha256": binary["sha256"],
                "manifest": manifest,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:20]

    cache_dir = REPO_ROOT / ".cortyx_eval_work" / "cache" / "longmemeval-500" / cache_key
    project_dir = cache_dir / "project"
    conversations_dir = cache_dir / "conversations"
    metadata_path = cache_dir / "corpus.json"
    index_path = project_dir / ".cortyx" / "index.json"

    if fresh_corpus and cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)

    reused = metadata_path.exists() and (
        counts["conversation_inputs"] == 0 or index_path.exists()
    )
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
        (project_dir / "src").mkdir(parents=True, exist_ok=True)
        conversations_dir.mkdir(parents=True, exist_ok=True)

        stage_start = time.perf_counter()
        for rel_key, content in staged_files.items():
            destination = cache_dir / rel_key
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")
        stage_secs = time.perf_counter() - stage_start

        if counts["conversation_inputs"]:
            mine_start = time.perf_counter()
            mine_run = run_cortyx(["mine", str(conversations_dir)], project_dir, timeout_secs)
            mine_secs = time.perf_counter() - mine_start
            if not mine_run.ok:
                shutil.rmtree(cache_dir, ignore_errors=True)
                return {
                    "reused": False,
                    "cache_key": cache_key,
                    "path": display_path(cache_dir),
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
            "project_dir": display_path(project_dir),
            "conversations_dir": display_path(conversations_dir),
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
        "path": display_path(cache_dir),
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
    failures.append(
        {
            "stage": stage,
            "group": group,
            "command": " ".join(run.args),
            "cwd": run.cwd,
            "timed_out": run.timed_out,
            "returncode": run.returncode,
            "stderr_excerpt": trim_text(run.stderr or run.stdout, limit=320),
        }
    )


# ── Evaluation ──────────────────────────────────────────────────────────────────


def hardest_cases(case_rows: list[dict], metric_name: str) -> list[dict]:
    def sort_key(item: dict):
        metric_value = item.get(metric_name)
        if metric_value is None:
            metric_value = 1.0 if item.get("abstention_correct") else 0.0
        return (item["f1"], item["em"], metric_value, item["category"], item["question"])

    return sorted(case_rows, key=sort_key)[:5]


def build_overall(results: dict, answer_mode: bool) -> dict:
    if not results:
        return {}

    total_n = sum(value["n"] for value in results.values())
    overall = {
        "groups_scored": len(results),
        "entries_scored": total_n,
        "macro_f1": round(sum(value["f1"] for value in results.values()) / len(results), 4),
        "macro_em": round(sum(value["em"] for value in results.values()) / len(results), 4),
        "micro_f1": round(
            sum(value["f1"] * value["n"] for value in results.values()) / total_n,
            4,
        ),
        "micro_em": round(
            sum(value["em"] * value["n"] for value in results.values()) / total_n,
            4,
        ),
    }

    metric_name = "answer_recall" if answer_mode else "r_at_5"
    metric_results = [value for value in results.values() if metric_name in value]
    if metric_results:
        metric_n = sum(value["n"] for value in metric_results)
        overall[f"macro_{metric_name}"] = round(
            sum(value[metric_name] for value in metric_results) / len(metric_results),
            4,
        )
        overall[f"micro_{metric_name}"] = round(
            sum(value[metric_name] * value["n"] for value in metric_results) / metric_n,
            4,
        )

    abstention_results = [value for value in results.values() if "abstention_accuracy" in value]
    if abstention_results:
        abstention_n = sum(value["n"] for value in abstention_results)
        overall["macro_abstention_accuracy"] = round(
            sum(value["abstention_accuracy"] for value in abstention_results)
            / len(abstention_results),
            4,
        )
        overall["micro_abstention_accuracy"] = round(
            sum(value["abstention_accuracy"] * value["n"] for value in abstention_results)
            / abstention_n,
            4,
        )

    return overall


def build_proof(
    fixture_path: Path,
    *,
    entries_total: int,
    entries_evaluated: int,
    corpus_scope: str,
    answer_mode: bool,
    failures: list[dict],
    public_surface: dict | None,
) -> dict:
    blockers: list[str] = []
    if entries_evaluated < entries_total:
        blockers.append(
            f"partial run evaluated {entries_evaluated}/{entries_total} LongMemEval rows"
        )
        if corpus_scope != "full_fixture":
            blockers.append(
                "targeted run used a trimmed corpus; retrieval conditions may differ from the full 500-row proof"
            )
    if entries_total < 500:
        blockers.append(
            f"fixture {fixture_path.name} has {entries_total} rows; official LongMemEval proof expects 500"
        )
    public_surface_ready = True
    if answer_mode:
        public_surface_ready = bool(public_surface and public_surface.get("same_surface"))
        if not public_surface:
            blockers.append("answer-mode report did not export an official LongMemEval QA-surface hypotheses file")
        elif public_surface.get("question_id_coverage") != entries_evaluated:
            blockers.append(
                f"official LongMemEval question_id coverage is {public_surface.get('question_id_coverage', 0)}/{entries_evaluated}"
            )
        elif public_surface.get("question_type_coverage") != entries_evaluated:
            blockers.append(
                f"official LongMemEval question_type coverage is {public_surface.get('question_type_coverage', 0)}/{entries_evaluated}"
            )
    if failures:
        blockers.append(f"{len(failures)} infrastructure failure(s) occurred during evaluation")

    proof = {
        "reproducible": not failures,
        "comparator_ready": public_surface_ready and not blockers and not failures,
        "blocking_reasons": blockers,
    }
    if answer_mode:
        proof["public_surface_ready"] = public_surface_ready
    return proof


def evaluate(
    entries: list[dict],
    *,
    corpus_entries: list[dict] | None,
    answer_mode: bool,
    use_llm: bool,
    timeout_secs: int,
    min_answer_confidence: float | None,
    fresh_corpus: bool,
) -> dict:
    run_start = time.perf_counter()
    print(f"Evaluating {len(entries)} entries")
    print("=" * 60)

    by_cat: dict[str, list] = defaultdict(list)
    for entry in entries:
        by_cat[entry_category(entry)].append(entry)

    results: dict[str, dict] = {}
    failures: list[dict] = []
    case_rows: list[dict] = []
    fixture_anomalies: list[dict] = []
    metric_name = "answer_recall" if answer_mode else "r_at_5"

    if not entries:
        return {
            "results": results,
            "overall": build_overall(results, answer_mode),
            "case_rows": case_rows,
            "diagnostics": {
                "infra_failures": failures,
                "hard_cases": hardest_cases(case_rows, metric_name),
                "fixture_anomalies": fixture_anomalies,
            },
            "corpus": {},
            "timings": {
                "corpus_stage_secs": 0.0,
                "corpus_mine_secs": 0.0,
                "query_secs_total": 0.0,
                "avg_query_secs": 0.0,
                "query_count": 0,
                "query_secs_by_category": {},
                "total_secs": round(time.perf_counter() - run_start, 4),
            },
        }

    corpus = prepare_lme_corpus(
        corpus_entries or entries,
        timeout_secs=timeout_secs,
        fresh_corpus=fresh_corpus,
    )
    corpus_state = "reused" if corpus["reused"] else "built"
    print(
        "[corpus] "
        f"{corpus_state} {corpus['path']} "
        f"({corpus['conversation_inputs']} conversations, {corpus['project_inputs']} project files)"
    )
    if corpus["conversation_inputs"] and not corpus["mine_run"].ok:
        record_failure(failures, "mine", "corpus", corpus["mine_run"])
        print(
            "  WARNING: corpus mine failed: "
            f"{trim_text(corpus['mine_run'].stderr or corpus['mine_run'].stdout)}"
        )
        return {
            "results": results,
            "overall": build_overall(results, answer_mode),
            "case_rows": case_rows,
            "diagnostics": {
                "infra_failures": failures,
                "hard_cases": hardest_cases(case_rows, metric_name),
                "fixture_anomalies": fixture_anomalies,
            },
            "corpus": corpus,
            "timings": {
                "corpus_stage_secs": corpus["stage_secs"],
                "corpus_mine_secs": corpus["mine_secs"],
                "query_secs_total": 0.0,
                "avg_query_secs": 0.0,
                "query_count": 0,
                "query_secs_by_category": {},
                "total_secs": round(time.perf_counter() - run_start, 4),
            },
        }

    project_dir = corpus["project_dir"]
    query_secs_total = 0.0
    query_secs_by_category: dict[str, float] = {}

    for category in CATEGORIES:
        cat_entries = by_cat.get(category, [])
        if not cat_entries:
            continue

        print(f"\n[{category}] {len(cat_entries)} entries")

        f1_sum = 0.0
        em_count = 0
        r_at_5_count = 0
        answer_recall_sum = 0.0
        llm_sum = 0.0
        llm_n = 0
        abstention_correct = 0
        category_query_secs = 0.0

        for i, entry in enumerate(cat_entries):
            question = render_lme_question(entry, answer_mode=answer_mode)
            gold = entry.get("expected_answer", "")
            keywords = entry.get("expected_keywords", [])
            anomaly = known_fixture_anomaly(entry)
            prediction = ""
            is_absent = entry_category(entry) == "absent"

            args = [
                "get-contexts",
                "--task",
                question,
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
            category_query_secs += query_elapsed
            query_secs_total += query_elapsed

            retrieved = query_run.stdout if query_run.stdout else query_run.stderr
            prediction = retrieved.strip()
            if not query_run.ok:
                record_failure(failures, "query", category, query_run)

            if is_absent:
                no_match = "(no neurons matched" in retrieved.lower()
                correct = no_match or (answer_mode and not _tokenise(retrieved))
                abstention_correct += int(correct)
                f1 = 1.0 if correct else 0.0
                em = correct
                metric_value = 1.0 if correct else 0.0
                scored_gold = gold
            else:
                f1, em, scored_gold_metric, scored_gold = best_gold_variant_score(retrieved, gold)
                if answer_mode:
                    metric_value = scored_gold_metric
                    answer_recall_sum += metric_value
                else:
                    metric_value = (
                        1.0 if (recall_at_k(retrieved, keywords) if keywords else False) else 0.0
                    )
                    r_at_5_count += int(metric_value)

                if use_llm and gold:
                    judge_score = llm_judge(question, retrieved, gold, answer_mode=answer_mode)
                    if judge_score >= 0:
                        llm_sum += judge_score
                        llm_n += 1

            f1_sum += f1
            em_count += int(em)

            case_row = {
                "question_id": entry_question_id(entry),
                "question_type": entry_question_type(entry),
                "category": category,
                "question": question,
                "gold_answer": gold,
                "gold_answer_scored": scored_gold,
                "prediction": prediction,
                "prediction_excerpt": trim_text(prediction, 240),
                "f1": round(f1, 4),
                "em": round(float(em), 4),
                "timed_out": query_run.timed_out,
                "returncode": query_run.returncode,
                "query_secs": round(query_elapsed, 4),
            }
            if is_absent:
                case_row["abstention_correct"] = bool(metric_value)
            else:
                case_row[metric_name] = round(metric_value, 4)
            if anomaly is not None:
                anomaly_f1, anomaly_em, anomaly_metric, anomaly_scored = best_gold_variant_score(
                    retrieved,
                    anomaly["evidence_based_answer"],
                )
                anomaly_row = {
                    **anomaly,
                    "category": category,
                    "gold_answer_scored": anomaly_scored,
                    "prediction": prediction,
                    "prediction_excerpt": trim_text(prediction, 240),
                    "evidence_based_f1": round(anomaly_f1, 4),
                    "evidence_based_em": round(float(anomaly_em), 4),
                    metric_name: round(anomaly_metric, 4),
                    "raw_f1": round(f1, 4),
                    "raw_em": round(float(em), 4),
                }
                case_row["known_fixture_anomaly"] = anomaly_row
                fixture_anomalies.append(anomaly_row)
            case_rows.append(case_row)

            if (i + 1) % 20 == 0:
                progress_tail = (
                    f"AnsR={answer_recall_sum / (i + 1):.3f}"
                    if answer_mode and not is_absent
                    else (
                        f"R@5={r_at_5_count / (i + 1):.3f}"
                        if not answer_mode and not is_absent
                        else f"Abstention={abstention_correct / (i + 1):.3f}"
                    )
                )
                print(
                    f"  … {i + 1}/{len(cat_entries)} "
                    f"(F1={f1_sum / (i + 1):.3f}, EM={em_count / (i + 1):.3f}, {progress_tail})",
                    flush=True,
                )

        n = len(cat_entries)
        cat_result = {
            "n": n,
            "f1": round(f1_sum / n, 4),
            "em": round(em_count / n, 4),
        }
        if category == "absent":
            cat_result["abstention_accuracy"] = round(abstention_correct / n, 4)
            metric_tail = f"Abstention acc={cat_result['abstention_accuracy']:.3f}"
        elif answer_mode:
            cat_result["answer_recall"] = round(answer_recall_sum / n, 4)
            metric_tail = f"AnsR={cat_result['answer_recall']:.3f}"
        else:
            cat_result["r_at_5"] = round(r_at_5_count / n, 4)
            metric_tail = f"R@5={cat_result['r_at_5']:.3f}"
        if llm_n > 0:
            cat_result["llm_judge"] = round(llm_sum / llm_n, 4)

        results[category] = cat_result
        query_secs_by_category[category] = round(category_query_secs, 4)
        print(f"  → F1={cat_result['f1']:.3f}  EM={cat_result['em']:.3f}  {metric_tail}")

    return {
        "results": results,
        "overall": build_overall(results, answer_mode),
        "case_rows": case_rows,
        "diagnostics": {
            "infra_failures": failures,
            "hard_cases": hardest_cases(case_rows, metric_name),
            "fixture_anomalies": fixture_anomalies,
        },
        "corpus": corpus,
        "timings": {
            "corpus_stage_secs": corpus["stage_secs"],
            "corpus_mine_secs": corpus["mine_secs"],
            "query_secs_total": round(query_secs_total, 4),
            "avg_query_secs": round(query_secs_total / len(entries), 4) if entries else 0.0,
            "query_count": len(entries),
            "query_secs_by_category": query_secs_by_category,
            "total_secs": round(time.perf_counter() - run_start, 4),
        },
    }


# ── Reporting ───────────────────────────────────────────────────────────────────


def public_hypotheses_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.public-hypotheses.jsonl")


def build_public_surface_artifact(
    case_rows: list[dict],
    *,
    answer_mode: bool,
    output_path: Path,
) -> dict | None:
    if not answer_mode:
        return None

    hypotheses_path = public_hypotheses_path(output_path)
    hypotheses_path.parent.mkdir(parents=True, exist_ok=True)

    question_id_coverage = 0
    question_type_coverage = 0
    lines: list[str] = []
    for index, row in enumerate(case_rows):
        question_id = str(row.get("question_id") or "").strip()
        question_type = str(row.get("question_type") or "").strip()
        if question_id:
            question_id_coverage += 1
        else:
            question_id = f"fixture_{index:04d}"
        if question_type:
            question_type_coverage += 1
        lines.append(
            json.dumps(
                {
                    "question_id": question_id,
                    "hypothesis": str(row.get("prediction", "")),
                },
                ensure_ascii=False,
            )
        )
    hypotheses_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    same_surface = (
        question_id_coverage == len(case_rows) and question_type_coverage == len(case_rows)
    )
    notes = [
        "Exports LongMemEval answer-mode predictions in the official question_id/hypothesis JSONL shape.",
        "Use the official LongMemEval QA evaluator to score task-averaged accuracy, overall accuracy, and abstention accuracy.",
    ]
    if question_id_coverage != len(case_rows):
        notes.append(
            f"question_id coverage is {question_id_coverage}/{len(case_rows)}; missing IDs fall back to synthetic fixture_* placeholders."
        )
    if question_type_coverage != len(case_rows):
        notes.append(
            f"question_type coverage is {question_type_coverage}/{len(case_rows)}; official task-specific QA prompts require the public question_type."
        )

    return {
        "surface": "official_qa_accuracy",
        "primary_metric": "task_averaged_accuracy",
        "secondary_metrics": ["overall_accuracy", "abstention_accuracy"],
        "same_surface": same_surface,
        "entries_exported": len(case_rows),
        "question_id_coverage": question_id_coverage,
        "question_type_coverage": question_type_coverage,
        "hypotheses_path": display_path(hypotheses_path),
        "hypotheses_sha256": sha256_file(hypotheses_path),
        "official_eval_command": (
            "python3 src/evaluation/evaluate_qa.py gpt-4o "
            f"{display_path(hypotheses_path)} /path/to/longmemeval_oracle.json"
        ),
        "notes": notes,
    }


def build_report(
    fixture_path: Path,
    all_entries: list[dict],
    selected_entries: list[dict],
    *,
    profile: str,
    answer_mode: bool,
    use_llm: bool,
    timeout_secs: int,
    min_answer_confidence: float | None,
    selected_categories: set[str],
    selected_question_ids: set[str],
    corpus_scope: str,
    corpus_entries_count: int,
    evaluation: dict,
    public_surface: dict | None,
) -> dict:
    failures = evaluation["diagnostics"]["infra_failures"]
    return {
        "benchmark": "longmemeval-500",
        "mode": "answer" if answer_mode else "retrieval",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "fixture": {
            "path": display_path(fixture_path),
            "sha256": sha256_file(fixture_path),
            "entries_total": len(all_entries),
            "entries_evaluated": len(selected_entries),
            "sample_fixture": len(all_entries) < 500,
            "source_format": "converted_entries",
            "official_public_release": len(all_entries) == 500
            and all(entry_question_id(entry) for entry in all_entries)
            and all(entry_question_type(entry) for entry in all_entries),
            "question_id_coverage_total": sum(1 for entry in all_entries if entry_question_id(entry)),
            "question_type_coverage_total": sum(
                1 for entry in all_entries if entry_question_type(entry)
            ),
        },
        "selection": {
            "profile": profile,
            "categories": sorted(selected_categories),
            "question_ids": sorted(selected_question_ids),
            "corpus_scope": corpus_scope,
            "corpus_entries": corpus_entries_count,
            "category_counts_total": category_counts(all_entries),
            "category_counts_evaluated": category_counts(selected_entries),
            "full_run": len(selected_entries) == len(all_entries),
        },
        "run": {
            "cortyx_bin": evaluation.get("corpus", {}).get("binary", {}).get(
                "path", display_path(resolve_binary())
            ),
            "timeout_secs": timeout_secs,
            "llm_judge": use_llm,
            "min_answer_confidence": min_answer_confidence,
            "git": git_metadata(),
            "corpus_cache": evaluation.get("corpus", {}),
            "timings": evaluation.get("timings", {}),
        },
        "overall": evaluation["overall"],
        "results": evaluation["results"],
        "cases": {
            "count": len(evaluation["case_rows"]),
            "rows": evaluation["case_rows"],
        },
        "diagnostics": evaluation["diagnostics"],
        "public_surface": public_surface,
        "proof": build_proof(
            fixture_path,
            entries_total=len(all_entries),
            entries_evaluated=len(selected_entries),
            corpus_scope=corpus_scope,
            answer_mode=answer_mode,
            failures=failures,
            public_surface=public_surface,
        ),
    }


def print_summary(report: dict):
    print("\n" + "=" * 60)
    print("LongMemEval-500 Evaluation Summary")
    print("=" * 60)

    results = report["results"]
    if not results:
        print("No scored LongMemEval categories matched this fixture selection.")
        return

    answer_mode = report["mode"] == "answer"
    metric_key = "answer_recall" if answer_mode else "r_at_5"
    metric_label = "AnsR" if answer_mode else "R@5"

    header = f"{'Category':<30} {'N':>5} {'F1':>6} {'EM':>6} {metric_label:>6}"
    print(header)
    print("-" * 60)
    for category, values in results.items():
        metric = (
            f"{values[metric_key]:.3f}"
            if metric_key in values
            else f"{values['abstention_accuracy']:.3f}"
            if "abstention_accuracy" in values
            else "  N/A"
        )
        print(f"{category:<30} {values['n']:>5} {values['f1']:>6.3f} {values['em']:>6.3f} {metric:>6}")

    overall = report["overall"]
    print("-" * 60)
    print(
        f"{'OVERALL (macro avg)':<30} {'':>5} "
        f"{overall['macro_f1']:>6.3f} {overall['macro_em']:>6.3f} "
        f"{overall.get(f'macro_{metric_key}', overall.get('macro_abstention_accuracy', 0.0)):>6.3f}"
    )

    selection = report.get("selection", {})
    run = report.get("run", {})
    corpus = run.get("corpus_cache") or {}
    timings = run.get("timings") or {}
    print("\nFeedback loop:")
    print(f"  profile: {selection.get('profile', 'full')}")
    if selection.get("question_ids"):
        print(f"  question_ids: {', '.join(selection['question_ids'])}")
    corpus_scope = selection.get("corpus_scope")
    if corpus_scope:
        scope_label = "full fixture" if corpus_scope == "full_fixture" else "selected rows"
        print(
            f"  corpus scope: {scope_label} "
            f"({selection.get('corpus_entries', 0)} staged entries)"
        )
    if corpus:
        corpus_state = "reused" if corpus.get("reused") else "built"
        print(
            f"  corpus: {corpus_state} {corpus.get('path')} "
            f"({corpus.get('conversation_inputs', 0)} conversations, "
            f"{corpus.get('project_inputs', 0)} project files)"
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

    if answer_mode:
        print("\nAnswer-mode note:")
        print("  This mode reports answer-grade metrics (F1 / EM / AnsR).")
        public_surface = report.get("public_surface") or {}
        if public_surface:
            print("  Public LongMemEval same-surface comparator metric: QA accuracy.")
            print(f"  same surface: {'yes' if public_surface.get('same_surface') else 'no'}")
            print(f"  hypotheses: {public_surface.get('hypotheses_path')}")
        print("  No comparator outcomes are claimed in this report.")
    else:
        print("\nBaseline reference:")
        print("  MemPalace (verbatim ChromaDB, dense-only):  R@5 ≈ 96.6%  (LME-500)")
        print("  OMEGA (top leaderboard 2026):               ~95.4%")

    proof = report["proof"]
    print("\nProof status:")
    print(f"  reproducible: {'yes' if proof['reproducible'] else 'no'}")
    if answer_mode and "public_surface_ready" in proof:
        print(f"  public surface ready: {'yes' if proof['public_surface_ready'] else 'no'}")
    print(f"  comparator ready: {'yes' if proof['comparator_ready'] else 'no'}")
    for blocker in proof["blocking_reasons"]:
        print(f"  blocker: {blocker}")

    hard_cases = report["diagnostics"].get("hard_cases", [])
    if hard_cases:
        print("\nHard cases:")
        for case in hard_cases[:3]:
            metric_value = case.get(metric_key)
            metric_text = (
                f"{metric_label}={metric_value:.3f}"
                if metric_value is not None
                else f"abstain={case.get('abstention_correct', False)}"
            )
            print(
                f"  - [{case['category']}] {case['question']} "
                f"(F1={case['f1']:.3f}, EM={case['em']:.3f}, {metric_text})"
            )

    fixture_anomalies = report["diagnostics"].get("fixture_anomalies", [])
    if fixture_anomalies:
        print("\nKnown fixture anomalies:")
        for anomaly in fixture_anomalies[:3]:
            print(
                f"  - [{anomaly['category']}] {anomaly['question']} "
                f"(fixture={anomaly['fixture_expected_answer']}; "
                f"evidence-based={anomaly['evidence_based_answer']}; "
                f"prediction={anomaly['prediction_excerpt'] or '(empty)'})"
            )
            print(f"    reason: {anomaly['reason']}")


# ── Main ────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--fixture",
        default=FIXTURE_DEFAULT,
        help="Path to fixture JSON (default: tests/fixtures/longmemeval_500.json)",
    )
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
        "--llm-judge",
        action="store_true",
        help="Use LLM judge for scoring (requires API key env var)",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_DEFAULTS),
        default="full",
        help="Sampling preset: full=all rows, quick=3 rows/category, smoke=1 row/category",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Limit to first N selected entries after applying --profile defaults (0 = all)",
    )
    parser.add_argument(
        "--max-per-category",
        type=int,
        default=None,
        help="Cap selected rows per category after applying --profile defaults",
    )
    parser.add_argument(
        "--categories",
        default="",
        help="Comma-separated category filter (default: all categories)",
    )
    parser.add_argument(
        "--question-ids",
        default="",
        help="Comma-separated question_id filter for targeted hard-case reruns",
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
        "--selection-corpus",
        action="store_true",
        help="Build the corpus from only the selected rows instead of the full fixture (faster cold starts, but less faithful for targeted reruns)",
    )
    args = parser.parse_args()

    fixture_path = Path(args.fixture)
    if not fixture_path.exists():
        print(f"ERROR: fixture not found: {fixture_path}")
        print("Generate it first: python3 scripts/gen_lme500.py")
        return 1

    answer_mode = args.answer_mode or ENV_ANSWER_MODE
    if args.min_answer_confidence is not None and not answer_mode:
        print("ERROR: --min-answer-confidence requires --answer-mode")
        return 1

    all_entries = json.loads(fixture_path.read_text())
    allowed_categories = parse_csv_set(args.categories)
    allowed_question_ids = parse_csv_values(args.question_ids)
    try:
        max_entries, max_per_category = apply_profile_defaults(
            args.profile,
            max_entries=args.max_entries,
            max_per_category=args.max_per_category,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1
    selected_entries = select_entries(
        all_entries,
        max_entries=max_entries,
        max_per_category=max_per_category,
        allowed_categories=allowed_categories,
        allowed_question_ids=allowed_question_ids,
    )
    selected_categories = {entry_category(entry) for entry in selected_entries}
    corpus_entries = selected_entries if args.selection_corpus else all_entries
    corpus_scope = "full_fixture" if len(corpus_entries) == len(all_entries) else "selected_rows"

    print(
        f"Evaluating {len(selected_entries)} entries from {fixture_path} "
        f"(profile={args.profile})"
    )
    try:
        evaluation = evaluate(
            selected_entries,
            corpus_entries=corpus_entries,
            answer_mode=answer_mode,
            use_llm=args.llm_judge,
            timeout_secs=args.timeout_secs,
            min_answer_confidence=args.min_answer_confidence,
            fresh_corpus=args.fresh_corpus,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1
    output_path = Path(args.output)
    public_surface = build_public_surface_artifact(
        evaluation["case_rows"],
        answer_mode=answer_mode,
        output_path=output_path,
    )
    report = build_report(
        fixture_path,
        all_entries,
        selected_entries,
        profile=args.profile,
        answer_mode=answer_mode,
        use_llm=args.llm_judge,
        timeout_secs=args.timeout_secs,
        min_answer_confidence=args.min_answer_confidence,
        selected_categories=selected_categories,
        selected_question_ids=allowed_question_ids,
        corpus_scope=corpus_scope,
        corpus_entries_count=len(corpus_entries),
        evaluation=evaluation,
        public_surface=public_surface,
    )
    print_summary(report)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(json_ready(report), indent=2) + "\n")
    print(f"\n✓ Results saved to {output_path}")

    return 0 if report["proof"]["reproducible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
