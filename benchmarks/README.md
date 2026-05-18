# Cortyx Benchmark Assets

This directory contains the benchmark runner and related benchmark-facing docs for
Cortyx.

## Structure

```text
benchmarks/
├── registry.json
├── README.md
└── run_bench.sh
```

## Running Benchmarks

For benchmark definitions, current live scores, and methodology notes, see
[`../BENCHMARKS.md`](../BENCHMARKS.md).

### Registry-backed proof matrix / benchmark definitions
```bash
python3 scripts/benchmark_registry.py matrix
python3 scripts/benchmark_registry.py scorecard
python3 scripts/benchmark_registry.py scorecard --json
python3 scripts/benchmark_registry.py guardrails best-overall-local-core --run
python3 scripts/benchmark_registry.py list --proof-status proven
python3 scripts/benchmark_registry.py show best-overall
python3 scripts/benchmark_registry.py validate
python3 scripts/benchmark_registry.py run --official
```

`scorecard` is the public best-overall claim gate. Its proof-eligibility phase
is now green because answer quality is `proven`, but the final claim still
waits on fair same-surface outcomes against the shared competitor set.

`scorecard --json` now exposes the competitive-proof scaffold too: the shared
repo-cited competitor roster, per-dimension comparator-scope metadata, explicit
outcome-ledger entries, and `claim_readiness` blockers / next flips. Retrieval
now records wins vs MemPalace / OMEGA, answer quality records LoCoMo QA losses
vs Hindsight / Zep / Letta / MemGPT / Mem0, and the remaining dimensions keep
their insufficient or missing evidence states explicit.

After the latest answer-proof promotion, the scorecard now sits at **100/100**
eligible points and is **ready-to-score**. That is still not a win claim:
only part of the shared-roster ledger is populated, and the retrieval must-win
gate is awaiting same-surface evidence from Hindsight/Zep/Letta/Mem0.
Collaboration/shared-memory is **satisfied** (wins vs all 6 competitors recorded).

`guardrails best-overall-local-core --run` is the practical non-regression
suite for retrieval drift, local speed, token economy, and release footprint.

### Standard benchmark run
```bash
./benchmarks/run_bench.sh
```

### Extended benchmark run
```bash
./benchmarks/run_bench.sh --extended
```

### Proper eval harnesses
```bash
./benchmarks/run_bench.sh --eval
```

### Fast diagnostic loop
```bash
./benchmarks/run_bench.sh --extended --eval --quick
python3 scripts/eval_lme.py --profile quick
python3 scripts/eval_locomo.py --profile quick
```

Use `--fresh-corpus` on either eval script when you want to rebuild the staged
corpus instead of reusing the cached corpus for the same binary + selection.

### Activation latency (requires hyperfine)
```bash
cargo build --release
hyperfine 'target/release/cortyx status' --warmup 5
```

### Accuracy benchmark (manual)
```bash
cargo test --test bench -- --nocapture
```

### Token savings vs raw RAG
```bash
cargo run --bin token_bench -- --sample-size 20 \
  --min-retrieval-savings-pct 55 \
  --max-retrieval-avg-tokens 3600 \
  --min-delta-repeat-savings-pct 98 \
  --max-delta-repeat-avg-tokens 160
```
