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

The script produces a JSON array of entries for the bench_locomo test, each:
{
  "session":           str,   # full conversation history as plain text
  "query":             str,   # the QA question
  "expected_answer":   str,   # gold answer (for F1/EM eval)
  "expected_keyword":  str,   # single keyword used in basic bench_locomo hit check
  "question_type":     str,   # single_hop | multi_hop | temporal | open_qa
  "conv_id":           str,   # conversation ID for debugging
  "gold_tokens":       [str]  # tokenised gold answer for F1 calculation
}

QUESTION TYPES:
  single_hop — answer comes from one turn, no reasoning required
  multi_hop  — answer requires connecting facts from multiple turns
  temporal   — requires temporal reasoning or ordering
  open_qa    — open-ended, no single correct answer (evaluated by F1 / human)
"""

import argparse
import json
import re
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
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _fetch_json(url: str) -> object:
    print(f"  Trying {url} …", flush=True)
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.loads(resp.read().decode())


def _load_json(path: str) -> object:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _tokenise(text: str) -> list[str]:
    """Simple whitespace + punctuation tokeniser for F1 computation."""
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def _first_keyword(answer: str) -> str:
    """Return the first meaningful word from an answer string."""
    stopwords = {"the", "a", "an", "is", "was", "are", "were", "i", "he", "she",
                 "it", "they", "we", "you", "my", "his", "her", "their", "its"}
    for w in re.findall(r"[a-zA-Z0-9]{3,}", answer.lower()):
        if w not in stopwords:
            return w
    return answer.split()[0].lower() if answer.split() else "answer"


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
        conv_id = str(conv.get("conv_id") or conv.get("id") or len(out))

        # Extract sessions / full conversation
        sessions = (conv.get("sessions") or conv.get("history") or
                    conv.get("conversation") or conv.get("turns") or [])
        session_text = _sessions_to_text(sessions) if sessions else ""

        # QA pairs
        qa_pairs = (conv.get("qa_pairs") or conv.get("questions") or
                    conv.get("qas") or [])

        for qa in qa_pairs:
            question = (qa.get("question") or qa.get("query") or "")
            answer   = (qa.get("answer") or qa.get("gold_answer") or
                        qa.get("answers", [""])[0] if isinstance(qa.get("answers"), list)
                        else qa.get("answers", ""))
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

            out.append({
                "session":          neuron_text or "(empty session)",
                "query":            question,
                "expected_answer":  str(answer),
                "expected_keyword": _first_keyword(str(answer)),
                "question_type":    qtype,
                "conv_id":          conv_id,
                "gold_tokens":      _tokenise(str(answer)),
            })

    return out


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="tests/fixtures/locomo_sample.json",
                        help="Output fixture path (default: tests/fixtures/locomo_sample.json)")
    parser.add_argument("--local", default=None,
                        help="Path to a local LoCoMo JSON file (skip download)")
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
