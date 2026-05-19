#!/usr/bin/env python3
"""Generate tests/fixtures/lmev2_fixture.jsonl from the LME-V2 dataset.

LME-V2: "Long-term Memory Evaluation for Coding Agents V2"
arXiv:2605.12493 (May 2026) — HuggingFace dataset required (HF_TOKEN)

Dataset source:
    https://huggingface.co/datasets/lme-benchmark/lme-v2

Requirements:
    pip install datasets huggingface_hub tqdm

Usage:
    export HF_TOKEN=<your_token>
    python3 scripts/gen_lmev2.py
    python3 scripts/gen_lmev2.py --output /tmp/lmev2_fixture.jsonl
    python3 scripts/gen_lmev2.py --sample-size 200  # 0 = full dataset

Output format (JSONL, one entry per line):
    {
      "question_id":   str,  # unique row ID from the LME-V2 dataset
      "question":      str,  # retrieval query / task description
      "expected":      str,  # gold context content (for R@5 computation)
      "category":      str,  # sub-benchmark category
      "trajectory":    str,  # web-agent trajectory (haystack)
    }

The bench_retrieval_accuracy_lmev2 test reads this file and evaluates Cortyx
R@5 against the gold context. SOTA on LME-V2 is 74.9% (from the paper).
"""

import argparse
import json
import os
import sys
from pathlib import Path


DATASET_NAME = "lme-benchmark/lme-v2"
DEFAULT_OUTPUT = Path(__file__).parent.parent / "tests" / "fixtures" / "lmev2_fixture.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSONL path")
    parser.add_argument("--sample-size", type=int, default=0, help="Number of rows (0 = full dataset)")
    parser.add_argument("--split", default="test", help="Dataset split to use (default: test)")
    return parser.parse_args()


def require_hf_token():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN environment variable is required to download LME-V2.", file=sys.stderr)
        print("Set it via: export HF_TOKEN=hf_...", file=sys.stderr)
        sys.exit(1)
    return token


def main():
    args = parse_args()
    token = require_hf_token()

    try:
        from datasets import load_dataset
        from tqdm import tqdm
    except ImportError:
        print("ERROR: Required packages not installed. Run: pip install datasets tqdm", file=sys.stderr)
        sys.exit(1)

    print(f"Downloading LME-V2 dataset ({DATASET_NAME}, split={args.split})...")
    dataset = load_dataset(DATASET_NAME, split=args.split, token=token)

    rows = list(dataset)
    if args.sample_size > 0:
        rows = rows[:args.sample_size]
        print(f"Using {len(rows)} rows (sample-size={args.sample_size})")
    else:
        print(f"Using full dataset: {len(rows)} rows")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in tqdm(rows, desc="Writing JSONL"):
            entry = {
                "question_id": row.get("id", ""),
                "question": row.get("query", row.get("question", "")),
                "expected": row.get("gold_context", row.get("answer", "")),
                "category": row.get("category", "unknown"),
                "trajectory": row.get("trajectory", ""),
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} rows to {output_path}")


if __name__ == "__main__":
    main()
