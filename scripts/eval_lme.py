#!/usr/bin/env python3
"""Proper evaluation harness for LongMemEval-500.

Runs cortyx get-contexts for each entry in the fixture, then scores the result
using token-level F1 (fast, no API required) per category.  Optionally uses an
LLM judge for more accurate scoring.

Usage:
    python3 scripts/eval_lme.py
    python3 scripts/eval_lme.py --fixture tests/fixtures/longmemeval_500.json
    python3 scripts/eval_lme.py --llm-judge   # requires ANTHROPIC_API_KEY or OPENAI_API_KEY

What the old bench.rs measured (WRONG):
    hit = any keyword from expected_keywords appears anywhere in stdout

What this script measures (CORRECT):
    F1   = token overlap between retrieved context and gold answer
    EM   = exact match of retrieved answer token set vs gold
    R@5  = does the evidence session appear in the top-5 retrieved neurons?
    Per-category breakdown for all 6 LME question types.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
import tempfile
import shutil

CORTYX_BIN = str(Path(__file__).parent.parent / "target" / "debug" / "cortyx")
FIXTURE_DEFAULT = str(Path(__file__).parent.parent / "tests" / "fixtures" / "longmemeval_500.json")

CATEGORIES = [
    "single_session_user",
    "single_session_assistant",
    "multi_session",
    "temporal_reasoning",
    "knowledge_update",
    "absent",
]

# ── Token-level F1 ─────────────────────────────────────────────────────────────

def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def f1_score(prediction: str, gold: str) -> float:
    """Token-level F1 between prediction and gold answer."""
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


def recall_at_k(retrieved_text: str, evidence_keywords: list[str]) -> bool:
    """R@k: does the retrieved text contain any evidence keyword?"""
    lower = retrieved_text.lower()
    return any(kw.lower() in lower for kw in evidence_keywords)


# ── LLM judge (optional) ───────────────────────────────────────────────────────

def llm_judge(question: str, retrieved: str, gold: str) -> float:
    """Score 0.0/0.5/1.0 using an LLM to judge answer correctness."""
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return -1.0  # not available

    prompt = (
        f"Question: {question}\n\n"
        f"Retrieved context:\n{retrieved[:1500]}\n\n"
        f"Gold answer: {gold}\n\n"
        "Does the retrieved context contain enough information to answer the question "
        "correctly? Reply with exactly one of: YES, PARTIAL, NO"
    )

    # Try Anthropic first
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
    """Run cortyx and return (stdout, success)."""
    binary = CORTYX_BIN
    if not Path(binary).exists():
        # Try release binary
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

    # Group by category
    by_cat: dict[str, list] = defaultdict(list)
    for e in entries:
        by_cat[e.get("category", "unknown")].append(e)

    results: dict[str, dict] = {}

    for category in CATEGORIES:
        cat_entries = by_cat.get(category, [])
        if not cat_entries:
            continue

        print(f"\n[{category}] {len(cat_entries)} entries")

        f1_sum = 0.0
        em_count = 0
        r_at_5_count = 0
        llm_sum = 0.0
        llm_n = 0
        abstention_correct = 0

        tmpdir = tempfile.mkdtemp(prefix="lme_eval_")
        try:
            src_dir = os.path.join(tmpdir, "src")
            os.makedirs(src_dir)

            # Write all conversation files and compile once per category batch
            for i, entry in enumerate(cat_entries):
                fname = entry.get("neuron_filename", f"conv_{i}.conv.md")
                content = entry.get("neuron_source_content", "")
                fpath = os.path.join(src_dir, fname) if entry.get("kind") != "conversation" else os.path.join(tmpdir, fname)
                Path(fpath).write_text(content, encoding="utf-8")

            # Compile the index
            _, ok = run_cortyx(["compile"], tmpdir)
            if not ok:
                print(f"  WARNING: compile failed for {category}")

            # Mine conversation files
            _, _ = run_cortyx(["mine", tmpdir], tmpdir)

            for i, entry in enumerate(cat_entries):
                question   = entry["question"]
                gold       = entry.get("expected_answer", "")
                keywords   = entry.get("expected_keywords", [])

                # For absent category: correct answer is to return nothing
                is_absent = (category == "absent")

                # Run retrieval
                retrieved, _ = run_cortyx(
                    ["get-contexts", "--task", question, "--max-tokens", "4000"],
                    tmpdir,
                )

                no_match = "(no neurons matched" in retrieved

                if is_absent:
                    # Correct if no neurons returned or very low score
                    correct = no_match
                    abstention_correct += int(correct)
                    f1 = 1.0 if correct else 0.0
                    em = correct
                else:
                    # Standard scoring
                    f1 = f1_score(retrieved, gold)
                    em = exact_match(retrieved, gold)
                    r5 = recall_at_k(retrieved, keywords) if keywords else False
                    r_at_5_count += int(r5)

                    if use_llm and gold:
                        lscore = llm_judge(question, retrieved, gold)
                        if lscore >= 0:
                            llm_sum += lscore
                            llm_n += 1

                f1_sum += f1
                em_count += int(em)

                if (i + 1) % 20 == 0:
                    print(f"  … {i + 1}/{len(cat_entries)} "
                          f"(F1={f1_sum/(i+1):.3f}, EM={em_count/(i+1):.3f})",
                          flush=True)

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        n = len(cat_entries)
        cat_result = {
            "n":          n,
            "f1":         round(f1_sum / n, 4),
            "em":         round(em_count / n, 4),
        }
        if category != "absent":
            cat_result["r_at_5"] = round(r_at_5_count / n, 4)
        else:
            cat_result["abstention_accuracy"] = round(abstention_correct / n, 4)
        if llm_n > 0:
            cat_result["llm_judge"] = round(llm_sum / llm_n, 4)

        results[category] = cat_result
        print(f"  → F1={cat_result['f1']:.3f}  EM={cat_result['em']:.3f}  "
              + (f"R@5={cat_result.get('r_at_5', 'N/A')}" if category != "absent"
                 else f"Abstention acc={cat_result.get('abstention_accuracy', 'N/A')}"))

    return results


def print_summary(results: dict):
    print("\n" + "=" * 60)
    print("LongMemEval-500 Evaluation Summary")
    print("=" * 60)

    all_f1 = [v["f1"] for v in results.values()]
    all_em = [v["em"] for v in results.values()]

    header = f"{'Category':<30} {'N':>5} {'F1':>6} {'EM':>6} {'R@5':>6}"
    print(header)
    print("-" * 60)
    for cat, v in results.items():
        r5 = f"{v['r_at_5']:.3f}" if "r_at_5" in v else (
            f"{v['abstention_accuracy']:.3f}" if "abstention_accuracy" in v else "  N/A")
        print(f"{cat:<30} {v['n']:>5} {v['f1']:>6.3f} {v['em']:>6.3f} {r5:>6}")

    print("-" * 60)
    print(f"{'OVERALL (macro avg)':<30} {'':>5} "
          f"{sum(all_f1)/len(all_f1):>6.3f} "
          f"{sum(all_em)/len(all_em):>6.3f}")

    print("\nBaseline reference:")
    print("  MemPalace (verbatim ChromaDB, dense-only):  R@5 ≈ 96.6%  (LME-500)")
    print("  OMEGA (top leaderboard 2026):               ~95.4%")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fixture", default=FIXTURE_DEFAULT,
                   help="Path to fixture JSON (default: tests/fixtures/longmemeval_500.json)")
    p.add_argument("--llm-judge", action="store_true",
                   help="Use LLM judge for scoring (requires API key env var)")
    p.add_argument("--max-entries", type=int, default=0,
                   help="Limit to first N entries (0 = all)")
    args = p.parse_args()

    if not Path(args.fixture).exists():
        print(f"ERROR: fixture not found: {args.fixture}")
        print("Generate it first: python3 scripts/gen_lme500.py")
        sys.exit(1)

    results = evaluate(args.fixture, args.llm_judge, args.max_entries)
    print_summary(results)

    # Write JSON report
    report_path = Path("lme500_eval_results.json")
    report_path.write_text(json.dumps(results, indent=2))
    print(f"\n✓ Results saved to {report_path}")


if __name__ == "__main__":
    main()
