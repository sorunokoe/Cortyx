#!/usr/bin/env python3
"""Graph reasoning convergence benchmark for Cortyx.

Runs the Rust integration test `bench_graph_reasoning` and reports
the TraversalStats proof evidence used to move the `graph-reasoning`
proof_matrix dimension from "smoke" to "proven".

Usage:
    python3 scripts/benchmark_graph_reasoning.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "benchmarks" / "registry.json"


def run_bench() -> tuple[bool, dict[str, object]]:
    """Run the Rust benchmark test and parse its printed stats."""
    result = subprocess.run(
        [
            "cargo",
            "test",
            "--test",
            "bench",
            "bench_graph_reasoning",
            "--",
            "--ignored",
            "--nocapture",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    output = result.stdout + result.stderr
    passed = "bench_graph_reasoning ... ok" in output

    stats: dict[str, object] = {}
    for line in output.splitlines():
        line = line.strip()
        m = re.match(r"nodes_by_depth:\s+(\[.*?\])", line)
        if m:
            stats["nodes_by_depth"] = json.loads(m.group(1))
        m = re.match(r"max_depth_reached:\s+(\d+)", line)
        if m:
            stats["max_depth_reached"] = int(m.group(1))
        m = re.match(r"total_expansions:\s+(\d+)", line)
        if m:
            stats["total_expansions"] = int(m.group(1))
        m = re.match(r"converged:\s+(true|false)", line)
        if m:
            stats["converged"] = m.group(1) == "true"
        m = re.match(r"total_nodes:\s+(\d+)", line)
        if m:
            stats["total_nodes"] = int(m.group(1))
        m = re.match(r"depth_coverage:\s+([0-9.]+)", line)
        if m:
            stats["depth_coverage"] = float(m.group(1))
        m = re.match(r"nodes found:\s+(\d+)", line)
        if m:
            stats["report_nodes"] = int(m.group(1))

    return passed, stats


def main() -> None:
    print("[benchmark_graph_reasoning] Building and running graph reasoning test…")
    passed, stats = run_bench()

    print()
    print("=== Graph Reasoning Traversal Stats ===")
    for key, val in stats.items():
        print(f"  {key:<22} {val}")
    print()

    if not passed:
        print("[FAIL] bench_graph_reasoning test did not pass.")
        print("       Run manually: cargo test --test bench bench_graph_reasoning -- --ignored --nocapture")
        sys.exit(1)

    # Minimum quality bar for "proven" status:
    #   - traversal reaches ≥ 2 hops
    #   - small graphs (≤ 64 nodes) converge naturally
    #   - depth coverage ≥ 0.5 for a 3-hop graph
    checks = [
        ("max_depth_reached ≥ 2", lambda s: s.get("max_depth_reached", 0) >= 2),
        ("converged = true",       lambda s: s.get("converged") is True),
        ("depth_coverage ≥ 0.50", lambda s: s.get("depth_coverage", 0.0) >= 0.50),
        ("total_nodes ≥ 3",       lambda s: s.get("total_nodes", 0) >= 3),
    ]

    all_pass = True
    for label, check in checks:
        ok = check(stats)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}")
        if not ok:
            all_pass = False

    print()
    if all_pass:
        print("[benchmark_graph_reasoning] PASS — graph-reasoning proof criteria met.")
        print("  proof_status: proven")
        print("  Claim: multi-hop graph traversal converges with per-depth coverage tracking.")
    else:
        print("[benchmark_graph_reasoning] FAIL — one or more criteria not met.")
        sys.exit(1)


if __name__ == "__main__":
    main()
