#!/usr/bin/env python3
"""Cortyx ablation baseline runner.

Runs eval_lme.py with different Cortyx build configurations to quantify each
retrieval feature's contribution. Results are saved to benchmarks/ablation_results.json.

Currently-supported ablations (no code changes needed):
  bm25-only   — cargo build --release (no --features), pure BM25 + synapse + Hebbian
  embed       — cargo build --release --features embed, BM25 + dense cosine RRF

TODO — deeper ablations require binary flags that are not yet implemented:
  no-hebbian  — BM25 + synapse graph, Hebbian co-return disabled
  no-synapse  — BM25 + Hebbian, synapse graph traversal disabled

Usage:
    python3 scripts/run_ablations.py
    python3 scripts/run_ablations.py --configs bm25-only,embed
    python3 scripts/run_ablations.py --profile quick
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CONFIGS: dict[str, dict] = {
    "bm25-only": {
        "label": "BM25 + synapse + Hebbian (no dense embeddings)",
        "build_features": None,
        "bin_name": "cortyx",
        "notes": "Default release build. Pure BM25 with synapse graph and Hebbian co-return feedback.",
    },
    "embed": {
        "label": "BM25 + dense cosine RRF (hybrid retrieval)",
        "build_features": "embed",
        "bin_name": "cortyx-embed",
        "notes": "Hybrid build: BM25 + all-MiniLM-L6-v2 (384-dim) via fastembed, merged with RRF. "
                 "Downloads ~80MB model on first run.",
    },
}


def build(config_id: str, config: dict) -> Path:
    features = config.get("build_features")
    bin_name = config["bin_name"]
    bin_path = REPO_ROOT / "target" / "release" / bin_name

    cmd = ["cargo", "build", "--release"]
    if features:
        cmd += ["--features", features]

    print(f"\n▶ Building {config_id} ({config['label']}) ...")
    t0 = time.monotonic()
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        print(f"  ✗ Build failed ({elapsed:.0f}s):\n{result.stderr[-2000:]}", file=sys.stderr)
        return Path("")

    # Rename to avoid collision when running multiple configs.
    default_bin = REPO_ROOT / "target" / "release" / "cortyx"
    if bin_name != "cortyx" and default_bin.exists():
        default_bin.rename(bin_path)
    elif not bin_path.exists() and default_bin.exists():
        bin_path = default_bin

    print(f"  ✓ Built in {elapsed:.0f}s → {bin_path}")
    return bin_path


def run_eval(bin_path: Path, config_id: str, profile: str) -> dict:
    env_overrides = {"CORTYX_BIN": str(bin_path)}

    cmd = [sys.executable, "scripts/eval_lme.py", "--fresh-corpus"]
    if profile != "full":
        cmd += ["--profile", profile]

    print(f"\n▶ Running eval_lme.py for {config_id} (profile={profile}) ...")
    t0 = time.monotonic()
    result = subprocess.run(cmd, cwd=REPO_ROOT, env={**__import__("os").environ, **env_overrides},
                            capture_output=True, text=True)
    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        print(f"  ✗ Eval failed ({elapsed:.0f}s):\n{result.stderr[-2000:]}", file=sys.stderr)
        return {"error": "eval failed", "elapsed_secs": elapsed}

    # Parse the output JSON.
    output_path = REPO_ROOT / "lme500_eval_results.json"
    if output_path.exists():
        data = json.loads(output_path.read_text())
        overall = data.get("overall", {})
        print(f"  ✓ {elapsed:.0f}s  macro_f1={overall.get('macro_f1', '?')}  "
              f"answer_recall={overall.get('macro_answer_recall', '?')}")
        return {"overall": overall, "elapsed_secs": elapsed}

    return {"error": "output file not found", "elapsed_secs": elapsed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", default=",".join(CONFIGS),
                        help="Comma-separated list of config IDs to run (default: all)")
    parser.add_argument("--profile", default="quick",
                        choices=["smoke", "quick", "full"],
                        help="Eval profile (default: quick)")
    parser.add_argument("--output", default="benchmarks/ablation_results.json",
                        help="Output file for results")
    args = parser.parse_args()

    requested = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in requested if c not in CONFIGS]
    if unknown:
        parser.error(f"Unknown config(s): {', '.join(unknown)}. Available: {', '.join(CONFIGS)}")

    results: dict = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "ablations": {},
        "notes": (
            "These are internal Cortyx ablations, not competitor comparisons. "
            "Deeper ablations (no-hebbian, no-synapse) require feature flags that "
            "are not yet implemented in the binary."
        ),
    }

    for config_id in requested:
        config = CONFIGS[config_id]
        print(f"\n{'='*60}")
        print(f"Ablation: {config_id} — {config['label']}")
        print(f"{'='*60}")

        bin_path = build(config_id, config)
        if not bin_path or not bin_path.exists():
            results["ablations"][config_id] = {
                "config": config,
                "error": "build failed",
            }
            continue

        eval_result = run_eval(bin_path, config_id, args.profile)
        results["ablations"][config_id] = {
            "config": config,
            "result": eval_result,
        }

    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\n✓ Ablation results saved to {output_path}")

    print("\n## Summary")
    print(f"{'Config':<20} {'macro_f1':<12} {'answer_recall':<15}")
    print("-" * 47)
    for config_id, entry in results["ablations"].items():
        overall = entry.get("result", {}).get("overall", {})
        f1 = overall.get("macro_f1", "error")
        recall = overall.get("macro_answer_recall", "error")
        print(f"{config_id:<20} {str(f1):<12} {str(recall):<15}")


if __name__ == "__main__":
    main()
