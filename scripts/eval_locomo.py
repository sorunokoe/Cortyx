#!/usr/bin/env python3
"""Proper evaluation harness for LoCoMo benchmark.

Computes token-level F1, Exact Match, and Recall per question type against
the real LoCoMo gold answers.  Replaces the single-keyword hit check used in
the basic bench_locomo test.

Usage:
    python3 scripts/eval_locomo.py
    python3 scripts/eval_locomo.py --fixture tests/fixtures/locomo_sample.json
    python3 scripts/eval_locomo.py --llm-judge

LoCoMo metrics (per arXiv:2402.17753):
    F1     — token overlap between retrieved context and gold answer
    EM     — exact token-set match
    Recall — fraction of gold tokens present in retrieved context

Baseline reference:
    Hindsight   89.6% LoCoMo
    Zep          ~85%
    Letta/MemGPT ~83.2%
    Mem0          58–67%
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

CORTYX_BIN = str(Path(__file__).parent.parent / "target" / "debug" / "cortyx")
FIXTURE_DEFAULT = str(Path(__file__).parent.parent / "tests" / "fixtures" / "locomo_sample.json")

QUESTION_TYPES = ["single_hop", "multi_hop", "temporal", "open_qa"]

# ── Scoring helpers ────────────────────────────────────────────────────────────

def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def f1_score(prediction: str, gold: str) -> float:
    pred = Counter(_tokenise(prediction))
    ref  = Counter(_tokenise(gold))
    common = sum((pred & ref).values())
    if common == 0:
        return 0.0
    p = common / sum(pred.values()) if pred else 0.0
    r = common / sum(ref.values())  if ref  else 0.0
    return 2 * p * r / (p + r)


def exact_match(prediction: str, gold: str) -> bool:
    return set(_tokenise(prediction)) == set(_tokenise(gold))


def recall_score(prediction: str, gold: str) -> float:
    """Fraction of gold tokens present in prediction."""
    pred_tokens = set(_tokenise(prediction))
    gold_tokens = _tokenise(gold)
    if not gold_tokens:
        return 1.0
    hits = sum(1 for t in gold_tokens if t in pred_tokens)
    return hits / len(gold_tokens)


# ── LLM judge ─────────────────────────────────────────────────────────────────

def llm_judge(question: str, retrieved: str, gold: str) -> float:
    """Optional LLM-judge score: 1.0 YES / 0.5 PARTIAL / 0.0 NO."""
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return -1.0
    prompt = (
        f"Question: {question}\n\n"
        f"Retrieved context (first 1500 chars):\n{retrieved[:1500]}\n\n"
        f"Gold answer: {gold}\n\n"
        "Does the retrieved context contain sufficient information to correctly "
        "answer the question? Reply with exactly: YES, PARTIAL, or NO"
    )
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import urllib.request, json as _json
            body = _json.dumps({
                "model": "claude-haiku-4-5",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": prompt}],
            }).encode()
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


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_cortyx(args_list: list[str], cwd: str) -> tuple[str, bool]:
    binary = CORTYX_BIN
    if not Path(binary).exists():
        binary = str(Path(__file__).parent.parent / "target" / "release" / "cortyx")
    if not Path(binary).exists():
        print("ERROR: cortyx binary not found. Run: cargo build", file=sys.stderr)
        sys.exit(1)
    result = subprocess.run(
        [binary] + args_list,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout, result.returncode == 0


def evaluate(fixture_path: str, use_llm: bool, max_entries: int) -> dict:
    entries = json.loads(Path(fixture_path).read_text())
    if max_entries:
        entries = entries[:max_entries]

    print(f"Evaluating {len(entries)} entries from {fixture_path}")
    print("=" * 60)

    by_type: dict[str, list] = defaultdict(list)
    for e in entries:
        by_type[e.get("question_type", "single_hop")].append(e)

    results = {}

    for qtype in QUESTION_TYPES:
        type_entries = by_type.get(qtype, [])
        if not type_entries:
            continue

        print(f"\n[{qtype}] {len(type_entries)} entries")

        f1_sum = 0.0
        em_count = 0
        recall_sum = 0.0
        llm_sum = 0.0
        llm_n = 0

        tmpdir = tempfile.mkdtemp(prefix="locomo_eval_")
        try:
            for i, entry in enumerate(type_entries):
                session  = entry["session"]
                query    = entry["query"]
                gold     = entry.get("expected_answer", "")
                conv_id  = entry.get("conv_id", str(i))

                # Write session as a plain text file for cortyx mine
                fname = f"session_{re.sub(r'[^a-zA-Z0-9]', '_', conv_id)[:30]}_{i}.txt"
                session_path = os.path.join(tmpdir, fname)
                Path(session_path).write_text(session, encoding="utf-8")

            # Mine all sessions into the index
            _, _ = run_cortyx(["compile"], tmpdir)
            _, _ = run_cortyx(["mine", tmpdir], tmpdir)

            for i, entry in enumerate(type_entries):
                query = entry["query"]
                gold  = entry.get("expected_answer", "")

                retrieved, _ = run_cortyx(
                    ["get-contexts", "--task", query, "--max-tokens", "4000"],
                    tmpdir,
                )

                f1  = f1_score(retrieved, gold)
                em  = exact_match(retrieved, gold)
                rec = recall_score(retrieved, gold)

                f1_sum     += f1
                em_count   += int(em)
                recall_sum += rec

                if use_llm and gold:
                    lscore = llm_judge(query, retrieved, gold)
                    if lscore >= 0:
                        llm_sum += lscore
                        llm_n += 1

                if (i + 1) % 20 == 0:
                    print(f"  … {i + 1}/{len(type_entries)} "
                          f"(F1={f1_sum/(i+1):.3f}, EM={em_count/(i+1):.3f}, "
                          f"Recall={recall_sum/(i+1):.3f})",
                          flush=True)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        n = len(type_entries)
        cat_result = {
            "n":      n,
            "f1":     round(f1_sum / n, 4),
            "em":     round(em_count / n, 4),
            "recall": round(recall_sum / n, 4),
        }
        if llm_n > 0:
            cat_result["llm_judge"] = round(llm_sum / llm_n, 4)

        results[qtype] = cat_result
        print(f"  → F1={cat_result['f1']:.3f}  EM={cat_result['em']:.3f}  "
              f"Recall={cat_result['recall']:.3f}")

    return results


def print_summary(results: dict):
    print("\n" + "=" * 60)
    print("LoCoMo Evaluation Summary")
    print("=" * 60)
    header = f"{'Question Type':<20} {'N':>5} {'F1':>7} {'EM':>7} {'Recall':>8}"
    print(header)
    print("-" * 52)

    all_f1 = [v["f1"] for v in results.values()]
    all_em = [v["em"] for v in results.values()]
    all_rec = [v["recall"] for v in results.values()]

    for qtype, v in results.items():
        print(f"{qtype:<20} {v['n']:>5} {v['f1']:>7.3f} {v['em']:>7.3f} {v['recall']:>8.3f}")

    print("-" * 52)
    print(f"{'OVERALL (macro avg)':<20} {'':>5} "
          f"{sum(all_f1)/len(all_f1):>7.3f} "
          f"{sum(all_em)/len(all_em):>7.3f} "
          f"{sum(all_rec)/len(all_rec):>8.3f}")

    print("\nBaseline reference (LoCoMo QA F1):")
    print("  Hindsight (open-source):  ~89.6%")
    print("  Zep:                       ~85.0%")
    print("  Letta / MemGPT:            ~83.2%")
    print("  Mem0:                      58–67%")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fixture", default=FIXTURE_DEFAULT)
    p.add_argument("--llm-judge", action="store_true")
    p.add_argument("--max-entries", type=int, default=0)
    args = p.parse_args()

    if not Path(args.fixture).exists():
        print(f"ERROR: fixture not found: {args.fixture}")
        print("Generate it first: python3 scripts/gen_locomo.py")
        sys.exit(1)

    results = evaluate(args.fixture, args.llm_judge, args.max_entries)
    print_summary(results)

    report_path = Path("locomo_eval_results.json")
    report_path.write_text(json.dumps(results, indent=2))
    print(f"\n✓ Results saved to {report_path}")


if __name__ == "__main__":
    main()
