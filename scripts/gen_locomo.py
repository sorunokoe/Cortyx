#!/usr/bin/env python3
"""Generate tests/fixtures/locomo_sample.json from the real LoCoMo dataset.

LoCoMo: "Evaluating Very Long-Term Conversational Memory of LLM Agents"
arXiv:2402.17753 — https://github.com/snap-research/LoCoMo

Requirements:
    pip install requests

Usage:
    python3 scripts/gen_locomo.py
    python3 scripts/gen_locomo.py --output tests/fixtures/locomo_sample.json
    python3 scripts/gen_locomo.py --local path/to/locomo10.json
    python3 scripts/gen_locomo.py --per-type 50

The script produces a JSON array of entries for the bench_locomo test, each:
{
  "session":           str,   # full conversation history as plain text
  "query":             str,   # the QA question
  "expected_answer":   str,   # gold answer (for F1/EM eval)
  "expected_keyword":  str,   # primary anchor used in basic bench_locomo hit check
  "expected_keywords": [str], # alternate anchors used in basic bench_locomo hit check
  "question_type":     str,   # single_hop | multi_hop | temporal | open_qa
  "conv_id":           str,   # conversation ID for debugging
  "gold_tokens":       [str]  # tokenised gold answer for F1 calculation
}

By default it writes a **deterministic stratified sample** (50 entries per
question type) rather than the full released QA set. This keeps `bench_locomo`
and `eval_locomo.py` in the "benchmark" regime instead of turning them into an
hours-long exhaustive replay of all LoCoMo QA items.

QUESTION TYPES:
  single_hop — answer comes from one turn, no reasoning required
  multi_hop  — answer requires connecting facts from multiple turns
  temporal   — requires temporal reasoning or ordering
  open_qa    — open-ended, no single correct answer (evaluated by F1 / human)
"""

import argparse
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

# ── Dataset location ───────────────────────────────────────────────────────────

REPO_RAW = "https://raw.githubusercontent.com/snap-research/LoCoMo/main"
LOCOMO_URLS = [
    f"{REPO_RAW}/data/locomo10.json",
    f"{REPO_RAW}/locomo10.json",
    f"{REPO_RAW}/data/locomo.json",
]

QTYPE_MAP = {
    "single_hop":  "single_hop",
    "multi_hop":   "multi_hop",
    "multihop":    "multi_hop",
    "temporal":    "temporal",
    "open_qa":     "open_qa",
    "open":        "open_qa",
    # Alternate spellings
    "single":      "single_hop",
    "multi":       "multi_hop",
    # Current public LoCoMo release uses numeric categories.
    "1":           "multi_hop",
    "2":           "temporal",
    "3":           "open_qa",
    "4":           "single_hop",
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _fetch_json(url: str) -> object:
    print(f"  Trying {url} …", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        print(f"    urllib failed ({exc}); retrying with curl …", flush=True)
        result = subprocess.run(
            ["curl", "-fsSL", url],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout:
            return json.loads(result.stdout)
        raise


def _load_json(path: str) -> object:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _tokenise(text: str) -> list[str]:
    """Simple whitespace + punctuation tokeniser for F1 computation."""
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


ANCHOR_STOPWORDS = {
    "the", "a", "an", "is", "was", "are", "were", "i", "he", "she", "it",
    "they", "we", "you", "my", "his", "her", "their", "its", "and", "or",
    "but", "for", "with", "from", "that", "this", "have", "just", "into",
    "what", "when", "where", "after", "before", "over", "under", "through",
    "about", "your", "been", "than", "then", "because", "while", "would",
    "could", "should", "did", "does", "doing", "done", "made", "make", "made",
}

WEAK_ANCHORS = {
    "own", "really", "very", "more", "some", "many", "such", "good", "great",
    "nice", "first", "last", "next", "later", "early", "approximately",
    "nearly", "almost", "attending", "visiting", "working", "started",
    "joining", "joined", "went", "going", "taking", "having", "getting",
    "received", "support", "life", "thing", "things", "place", "places",
}


def _answer_anchors(answer: str, session_text: str, *, limit: int = 4) -> list[str]:
    """Return deterministic answer anchors for the basic LoCoMo recall check."""
    session_lower = session_text.lower()
    raw_tokens = re.findall(r"[a-zA-Z0-9']+", answer.lower())

    seen = set()
    ranked: list[tuple[int, str]] = []
    for idx, token in enumerate(raw_tokens):
        if token in seen:
            continue
        seen.add(token)

        if token in ANCHOR_STOPWORDS:
            continue
        if len(token) < 3 and not token.isdigit():
            continue

        in_session = token in session_lower
        strong = token not in WEAK_ANCHORS
        numeric = token.isdigit()

        score = 0
        if in_session:
            score += 100
        if strong:
            score += 20
        if numeric:
            score += 10
        score += min(len(token), 12)
        score -= idx  # prefer earlier anchors when otherwise similar
        ranked.append((score, token))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    anchors = [token for _, token in ranked[:limit]]

    if not anchors:
        fallback = [t for t in raw_tokens if len(t) >= 3]
        if fallback:
            anchors = [fallback[0]]
        elif answer.split():
            anchors = [answer.split()[0].lower()]
        else:
            anchors = ["answer"]

    return anchors


def _extract_answer(qa: dict) -> str:
    """Return the canonical gold answer text from a LoCoMo QA record."""
    answer = qa.get("answer")
    if isinstance(answer, dict):
        for key in ("answer", "text", "gold_answer"):
            value = answer.get(key)
            if value:
                return str(value)
    elif answer:
        return str(answer)

    gold_answer = qa.get("gold_answer")
    if gold_answer:
        return str(gold_answer)

    answers = qa.get("answers")
    if isinstance(answers, list) and answers:
        return str(answers[0])
    if answers:
        return str(answers)

    return ""


def _turns_to_text(turns: list[dict]) -> str:
    """Convert a list of conversation turns to plain text."""
    lines = []
    for turn in turns:
        # Handle various field naming conventions
        speaker = (turn.get("speaker") or turn.get("role") or
                   turn.get("name") or "user")
        text = (turn.get("text") or turn.get("content") or
                turn.get("message") or "")
        if text:
            lines.append(f"{speaker.capitalize()}: {text.strip()}")
    return "\n".join(lines)


def _sessions_to_text(sessions) -> str:
    """Convert sessions (list of session-dicts or list of turn-lists) to text."""
    if not sessions:
        return ""

    if isinstance(sessions, dict):
        parts = []
        speaker_a = sessions.get("speaker_a")
        speaker_b = sessions.get("speaker_b")
        if speaker_a or speaker_b:
            speaker_lines = []
            if speaker_a:
                speaker_lines.append(f"Speaker A: {speaker_a}")
            if speaker_b:
                speaker_lines.append(f"Speaker B: {speaker_b}")
            if speaker_lines:
                parts.append("\n".join(speaker_lines))

        session_keys = sorted(
            [
                key for key in sessions
                if re.fullmatch(r"session_\d+", key)
            ],
            key=lambda key: int(key.split("_")[1]),
        )
        for key in session_keys:
            turns = sessions.get(key) or []
            date = sessions.get(f"{key}_date_time")
            block = _turns_to_text(turns if isinstance(turns, list) else [turns])
            if not block.strip():
                continue
            header = key.replace("_", " ").title()
            if date:
                header = f"{header} — {date}"
            parts.append(f"[{header}]\n{block}")

        return "\n\n".join(parts)

    # sessions may be: list[list[turn]], list[dict with 'turns'], or list[turn]
    parts = []
    for i, sess in enumerate(sessions):
        if isinstance(sess, list):
            # list of turns
            block = _turns_to_text(sess)
        elif isinstance(sess, dict):
            # dict with 'turns', 'messages', or direct speaker/text
            turns = (sess.get("turns") or sess.get("messages") or
                     sess.get("conversation") or [])
            if turns:
                block = _turns_to_text(turns)
            elif "speaker" in sess or "role" in sess:
                block = _turns_to_text([sess])
            else:
                block = str(sess)
        else:
            block = str(sess)

        if block.strip():
            parts.append(f"[Session {i + 1}]\n{block}")

    return "\n\n".join(parts)


# ── Conversion ─────────────────────────────────────────────────────────────────

def convert_locomo(data) -> list[dict]:
    """Convert LoCoMo data to Cortyx bench fixture entries."""
    out = []

    # The dataset may be a list of conversations, or a dict keyed by conv_id
    if isinstance(data, dict):
        conversations = list(data.values())
    elif isinstance(data, list):
        conversations = data
    else:
        print(f"WARNING: unexpected data type {type(data)}, wrapping as list")
        conversations = [data]

    for conv in conversations:
        conv_id = str(conv.get("conv_id") or conv.get("sample_id") or conv.get("id") or len(out))

        # Extract sessions / full conversation
        sessions = (conv.get("sessions") or conv.get("history") or
                     conv.get("conversation") or conv.get("turns") or [])
        session_text = _sessions_to_text(sessions) if sessions else ""

        # QA pairs
        qa_pairs = (conv.get("qa_pairs") or conv.get("questions") or
                    conv.get("qas") or conv.get("qa") or [])

        for qa in qa_pairs:
            # The current public LoCoMo release includes category 5 adversarial
            # questions whose correct behaviour is abstention, not answer recall.
            # Skip them in this fixture: the basic bench and eval harness both
            # target single-hop / multi-hop / temporal / open QA.
            if str(qa.get("category")) == "5":
                continue
            question = (qa.get("question") or qa.get("query") or "")
            answer = _extract_answer(qa)
            qtype_raw = (qa.get("question_type") or qa.get("type") or
                         qa.get("category") or "single_hop")
            qtype = QTYPE_MAP.get(str(qtype_raw).lower(), "single_hop")

            # Evidence for this QA (may be a subset of sessions)
            evidence = qa.get("evidence_session") or qa.get("evidence") or []
            if evidence and isinstance(evidence, list):
                if isinstance(evidence[0], dict):
                    neuron_text = _turns_to_text(evidence)
                elif isinstance(evidence[0], list):
                    neuron_text = _sessions_to_text(evidence)
                else:
                    neuron_text = session_text  # fall back to full history
            else:
                neuron_text = session_text

            if not neuron_text.strip():
                neuron_text = session_text

            anchors = _answer_anchors(str(answer), session_text or neuron_text)
            out.append({
                "session":          session_text or neuron_text or "(empty session)",
                "query":            question,
                "expected_answer":  str(answer),
                "expected_keyword": anchors[0],
                "expected_keywords": anchors,
                "question_type":    qtype,
                "conv_id":          conv_id,
                "gold_tokens":      _tokenise(str(answer)),
            })

    return out


def _evenly_sample(entries: list[dict], limit: int) -> list[dict]:
    """Deterministically sample across the full list span instead of taking a prefix."""
    if limit <= 0 or len(entries) <= limit:
        return entries
    if limit == 1:
        return [entries[len(entries) // 2]]

    last = len(entries) - 1
    idxs = {round(i * last / (limit - 1)) for i in range(limit)}
    return [entries[i] for i in sorted(idxs)]


def stratified_sample(entries: list[dict], per_type: int) -> list[dict]:
    """Return up to `per_type` entries per question type, preserving type balance."""
    if per_type <= 0:
        return entries

    grouped = {}
    for entry in entries:
        grouped.setdefault(entry["question_type"], []).append(entry)

    sampled = []
    for qtype in ["single_hop", "multi_hop", "temporal", "open_qa"]:
        sampled.extend(_evenly_sample(grouped.get(qtype, []), per_type))
    return sampled


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="tests/fixtures/locomo_sample.json",
                        help="Output fixture path (default: tests/fixtures/locomo_sample.json)")
    parser.add_argument("--local", default=None,
                        help="Path to a local LoCoMo JSON file (skip download)")
    parser.add_argument("--per-type", type=int, default=50,
                        help="Deterministic sample size per question type (default: 50, use 0 for full dataset)")
    args = parser.parse_args()

    print("=== gen_locomo.py — LoCoMo fixture generator ===")

    if args.local:
        print(f"Loading local dataset: {args.local}")
        data = _load_json(args.local)
    else:
        data = None
        last_err = None
        for url in LOCOMO_URLS:
            try:
                data = _fetch_json(url)
                break
            except Exception as exc:
                last_err = exc
                continue

        if data is None:
            print(f"\nERROR: Could not download LoCoMo dataset: {last_err}")
            print("\nAlternatives:")
            print("  1. Clone the repo:  git clone https://github.com/snap-research/LoCoMo")
            print("     Then:            python3 scripts/gen_locomo.py --local LoCoMo/data/locomo10.json")
            print("  2. Download manually from https://github.com/snap-research/LoCoMo")
            sys.exit(1)

    print(f"  Raw data type: {type(data).__name__}, "
          f"len={len(data) if isinstance(data, (list, dict)) else 'N/A'}")

    entries = convert_locomo(data)
    print(f"  Converted {len(entries)} QA entries.")

    sampled_entries = stratified_sample(entries, args.per_type)
    if args.per_type > 0:
        print(f"  Sampled {len(sampled_entries)} QA entries ({args.per_type} per type max).")
    entries = sampled_entries

    # Category stats
    from collections import Counter
    qtypes = Counter(e["question_type"] for e in entries)
    print(f"  Question type breakdown: {dict(qtypes)}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2, ensure_ascii=False)

    print(f"\n✓ Wrote {len(entries)} entries → {out_path}")
    print("\nNext steps:")
    print("  cargo test --test bench bench_locomo -- --ignored --nocapture")
    print("  python3 scripts/eval_locomo.py")


if __name__ == "__main__":
    main()
