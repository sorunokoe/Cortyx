#!/usr/bin/env bash
set -euo pipefail

# Slow-but-honest proof/perf lanes that are intentionally kept out of the
# default developer `cargo test` loop.
# --no-default-features: embed/rerank require downloaded model files (not
# available in CI); core BM25 retrieval accuracy is fully exercised without them.

cargo test --test answer_proof --no-default-features -- --ignored --nocapture

for test_name in \
  bench_longmemeval_100_r_at_5 \
  bench_compile_100_files \
  bench_compile_500_files \
  bench_status_cold_start \
  bench_token_savings_estimate \
  bench_retrieval_accuracy_50q \
  bench_lme_regression_guard \
  bench_locomo_regression_guard
do
  cargo test --test bench --no-default-features "$test_name" -- --ignored --nocapture
done

python3 scripts/benchmark_registry.py validate
python3 scripts/benchmark_registry.py guardrails best-overall-local-core --run
