#!/usr/bin/env python3
"""Generate tests/fixtures/longmemeval_500.json from the real LongMemEval dataset.

LongMemEval: "Benchmarking Chat Assistants on Long-Term Interactive Memory"
arXiv:2410.10813 — https://github.com/xiaowu0162/LongMemEval

Requirements:
    pip install requests

Usage:
    python3 scripts/gen_lme500.py
    python3 scripts/gen_lme500.py --output tests/fixtures/longmemeval_500.json
    python3 scripts/gen_lme500.py --local path/to/longmemeval_oracle.json

The script produces a JSON array of entries, each:
{
  "question":               str,   # the evaluation question
  "expected_answer":        str,   # gold answer (for LLM-judge / EM eval)
  "expected_keywords":      [str], # key terms from expected_answer (for fast bench.rs R@5 check)
  "neuron_source_content":  str,   # the evidence session(s) as plain text
  "neuron_filename":        str,   # filename for cortyx compile (used as neuron ID)
  "kind":                   str,   # always "conversation"
  "category":               str,   # question type (see CATEGORIES below)
  "session_id":             str    # original session ID for debugging
}

CATEGORIES (maps to LongMemEval question_type):
  single_session_user      — fact stated by the user in one session
  single_session_assistant — fact stated by the assistant in one session
  multi_session            — requires evidence from multiple sessions
  temporal_reasoning       — requires reasoning about time / order
  knowledge_update         — a fact changed; what is current?
  absent                   — answer is NOT in the history (abstention)
"""

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

# ── Dataset location ───────────────────────────────────────────────────────────

REPO_RAW = "https://raw.githubusercontent.com/xiaowu0162/LongMemEval/main"
ORACLE_URL = f"{REPO_RAW}/data/longmemeval_oracle.json"
HAYSTACK_URL = f"{REPO_RAW}/data/longmemeval_haystack.json"

# Category → human tag
CATEGORY_MAP = {
    "single_session_user":      "single_session_user",
    "single_session_assistant": "single_session_assistant",
    "multi_session":            "multi_session",
    "temporal_reasoning":       "temporal_reasoning",
    "knowledge_update":         "knowledge_update",
    "absent":                   "absent",
    # Alternate spellings found in some dataset versions
    "single-session-user":      "single_session_user",
    "multi-session":            "multi_session",
    "temporal":                 "temporal_reasoning",
    "knowledge-update":         "knowledge_update",
    "abstention":               "absent",
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _fetch_json(url: str) -> object:
    print(f"  Downloading {url} …", flush=True)
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.loads(resp.read().decode())


def _load_json(path: str) -> object:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _keywords_from_answer(answer: str, n: int = 5) -> list[str]:
    """Extract up to n meaningful keywords from a gold answer string.

    Uses two extraction passes:
    1. Pure numerics (\b\d+\b) — captures single-digit counting answers like "2","3"
       that the old {2,} alpha-only regex would drop.
    2. Dollar amounts (\$\d+(?:\.\d+)?) — captures "$5", "$10" style answers.
    3. Ratio/fraction tokens (\d+:\d+) — captures "3:1" style answers.
    4. Alpha words ([a-zA-Z][a-zA-Z0-9'_-]+, ≥2 alpha chars) — skips stopwords.
    """
    stopwords = {
        "the", "a", "an", "is", "was", "are", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "i", "you", "he", "she",
        "we", "they", "it", "my", "your", "his", "her", "our", "their",
        "in", "on", "at", "to", "for", "of", "with", "by", "from", "about",
        "that", "this", "these", "those", "and", "or", "but", "not",
        "very", "just", "also", "so", "if", "when", "then", "there",
    }
    answer = str(answer)  # handles int/float answers
    ans_lower = answer.lower()

    seen: list[str] = []

    # Pass 1: ratio tokens (e.g. "3:1") — before splitting on digits
    for m in re.finditer(r"\b\d+:\d+\b", ans_lower):
        w = m.group()
        if w not in seen:
            seen.append(w)
        if len(seen) >= n:
            return seen

    # Pass 2: dollar amounts (e.g. "$5", "$10.50")
    for m in re.finditer(r"\$\d+(?:\.\d+)?", ans_lower):
        w = m.group()
        if w not in seen:
            seen.append(w)
        if len(seen) >= n:
            return seen

    # Pass 3: pure numerics including single digits (e.g. "2", "17")
    for m in re.finditer(r"\b\d+\b", ans_lower):
        w = m.group()
        if w not in seen:
            seen.append(w)
        if len(seen) >= n:
            return seen

    # Pass 4: alpha words (≥2 chars, not stopwords)
    for w in re.findall(r"[a-zA-Z][a-zA-Z0-9'_-]+", ans_lower):
        if w not in stopwords and w not in seen:
            seen.append(w)
        if len(seen) >= n:
            return seen

    return seen


def _session_to_text(session: list[dict]) -> str:
    """Convert a list of {speaker, text} turns into a plain-text block."""
    lines = []
    for turn in session:
        speaker = turn.get("speaker") or turn.get("role") or "user"
        text = turn.get("text") or turn.get("content") or ""
        lines.append(f"{speaker.capitalize()}: {text.strip()}")
    return "\n".join(lines)


def _safe_filename(session_id: str, idx: int) -> str:
    """Sanitise a session ID into a safe filename."""
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", str(session_id))[:40]
    return f"lme_{idx:04d}_{slug}.conv.md"


# ── Conversion ─────────────────────────────────────────────────────────────────

def convert_oracle(oracle: list[dict]) -> list[dict]:
    """Convert LongMemEval oracle entries to Cortyx fixture entries."""
    out = []
    for idx, entry in enumerate(oracle):
        question    = entry.get("question") or entry.get("query", "")
        answer      = str(entry.get("answer") or entry.get("gold_answer") or "")
        qtype       = entry.get("question_type") or entry.get("type", "")
        session_id  = entry.get("session_id") or entry.get("id", str(idx))

        # Normalise category
        category = CATEGORY_MAP.get(qtype.lower(), qtype or "single_session_user")

        # --- Build neuron content -----------------------------------------
        # For the oracle dataset, use the answer_session_ids to pick the
        # evidence sessions from haystack_sessions (those with has_answer turns).
        haystack = entry.get("haystack_sessions") or []
        answer_ids = set(entry.get("answer_session_ids") or [])

        if answer_ids and haystack:
            # Filter to sessions that contain the answer
            evidence = [s for s in haystack if any(
                t.get("has_answer") for t in (s if isinstance(s, list) else [])
            )]
            if not evidence:
                evidence = haystack[:1]  # fallback to first session
        elif haystack:
            evidence = haystack
        else:
            evidence = entry.get("evidence_session") or []

        # evidence is a list of sessions (list-of-lists) or a single session (list-of-dicts)
        if evidence and isinstance(evidence[0], list):
            session_texts = [_session_to_text(s) for s in evidence]
            content = "\n\n---\n\n".join(session_texts)
        elif evidence and isinstance(evidence[0], dict):
            content = _session_to_text(evidence)
        else:
            content = str(evidence)

        if not content.strip():
            content = f"(no session content available for entry {idx})"

        out.append({
            "question":              question,
            "expected_answer":       answer,
            "expected_keywords":     _keywords_from_answer(answer),
            "neuron_source_content": content,
            "neuron_filename":       _safe_filename(session_id, idx),
            "kind":                  "conversation",
            "category":              category,
            "session_id":            str(session_id),
        })

    return out


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="tests/fixtures/longmemeval_500.json",
                        help="Output fixture path (default: tests/fixtures/longmemeval_500.json)")
    parser.add_argument("--local", default=None,
                        help="Path to a local longmemeval_oracle.json (skip download)")
    args = parser.parse_args()

    print("=== gen_lme500.py — LongMemEval-500 fixture generator ===")

    # Load oracle
    if args.local:
        print(f"Loading local dataset: {args.local}")
        oracle = _load_json(args.local)
    else:
        try:
            oracle = _fetch_json(ORACLE_URL)
        except Exception as exc:
            print(f"\nERROR: Could not download dataset: {exc}")
            print("\nAlternatives:")
            print("  1. Clone the repo:  git clone https://github.com/xiaowu0162/LongMemEval")
            print("     Then:            python3 scripts/gen_lme500.py --local LongMemEval/data/longmemeval_oracle.json")
            print("  2. Download manually from https://github.com/xiaowu0162/LongMemEval/tree/main/data")
            sys.exit(1)

    if isinstance(oracle, dict):
        # Some versions wrap the list in a dict
        oracle = oracle.get("data") or oracle.get("entries") or list(oracle.values())

    print(f"  Loaded {len(oracle)} entries.")

    # Convert
    print("  Converting to Cortyx fixture format …")
    entries = convert_oracle(oracle)

    # Category stats
    from collections import Counter
    cats = Counter(e["category"] for e in entries)
    print(f"  Category breakdown: {dict(cats)}")

    # Write
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2, ensure_ascii=False)

    print(f"\n✓ Wrote {len(entries)} entries → {out_path}")
    print("\nNext steps:")
    print("  cargo test --test bench bench_retrieval_accuracy_500q -- --ignored --nocapture")
    print("  python3 scripts/eval_lme.py")


if __name__ == "__main__":
    main()
