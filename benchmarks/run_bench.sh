#!/usr/bin/env bash
# benchmarks/run_bench.sh — Run the full Cortyx benchmark suite
#
# Standard benchmarks (always fast, no fixtures needed):
#   ./benchmarks/run_bench.sh
#
# Extended benchmarks (require fixture generation first):
#   python3 scripts/gen_lme500.py && python3 scripts/gen_locomo.py
#   ./benchmarks/run_bench.sh --extended
#
# Full benchmarks with dense embeddings (best accuracy, needs model download):
#   ./benchmarks/run_bench.sh --embed
#   ./benchmarks/run_bench.sh --embed --extended
#
# Proper eval harnesses (F1/EM scoring, replaces keyword-match in bench.rs):
#   ./benchmarks/run_bench.sh --eval

set -euo pipefail

EMBED=0
EXTENDED=0
EVAL=0
RELEASE=0

for arg in "$@"; do
  case $arg in
    --embed)    EMBED=1 ;;
    --extended) EXTENDED=1 ;;
    --eval)     EVAL=1 ;;
    --release)  RELEASE=1 ;;
    --help|-h)
      sed -n '2,20p' "$0"
      exit 0
      ;;
  esac
done

# ── Feature flags ──────────────────────────────────────────────────────────────

FEATURES=""
if [[ $EMBED -eq 1 ]]; then
  FEATURES="embed"
  echo "▶ Building with dense embeddings (--features embed)"
  echo "  Note: first run downloads ~80MB model weights from HuggingFace"
fi

# ── Build the cortyx binary ────────────────────────────────────────────────────

BUILD_FLAGS=""
if [[ $RELEASE -eq 1 ]]; then
  BUILD_FLAGS="--release"
fi

if [[ -n $FEATURES ]]; then
  cargo build $BUILD_FLAGS --features "$FEATURES" 2>&1
else
  cargo build $BUILD_FLAGS 2>&1
fi

echo ""
echo "══════════════════════════════════════════════════════"
echo "  Cortyx Benchmark Suite"
echo "══════════════════════════════════════════════════════"
echo ""

# ── Standard benchmarks (always run) ──────────────────────────────────────────

echo "▶ Running standard benchmarks …"
if [[ -n $FEATURES ]]; then
  cargo test --test bench --features "$FEATURES" -- --nocapture 2>&1
else
  cargo test --test bench -- --nocapture 2>&1
fi

# ── Extended benchmarks (require fixtures) ─────────────────────────────────────

if [[ $EXTENDED -eq 1 ]]; then
  echo ""
  echo "▶ Running extended benchmarks (LME-500, LoCoMo) …"

  FIXTURE_LME="tests/fixtures/longmemeval_500.json"
  FIXTURE_LOCOMO="tests/fixtures/locomo_sample.json"

  if [[ ! -f "$FIXTURE_LME" ]]; then
    echo "  Generating LME-500 fixture …"
    python3 scripts/gen_lme500.py
  fi

  if [[ ! -f "$FIXTURE_LOCOMO" ]]; then
    echo "  Generating LoCoMo fixture …"
    python3 scripts/gen_locomo.py
  fi

  if [[ -n $FEATURES ]]; then
    cargo test --test bench --features "$FEATURES" bench_retrieval_accuracy_500q bench_locomo -- --ignored --nocapture 2>&1
  else
    cargo test --test bench bench_retrieval_accuracy_500q bench_locomo -- --ignored --nocapture 2>&1
  fi
fi

# ── Proper eval harnesses (F1/EM scoring) ─────────────────────────────────────

if [[ $EVAL -eq 1 ]]; then
  echo ""
  echo "▶ Running proper evaluation harnesses …"

  if [[ -f "tests/fixtures/longmemeval_500.json" ]]; then
    echo "  eval_lme.py (LME-500 F1/EM) …"
    python3 scripts/eval_lme.py
  else
    echo "  Skipping eval_lme.py — fixture missing (run gen_lme500.py first)"
  fi

  if [[ -f "tests/fixtures/locomo_sample.json" ]]; then
    echo "  eval_locomo.py (LoCoMo F1/EM) …"
    python3 scripts/eval_locomo.py
  else
    echo "  Skipping eval_locomo.py — fixture missing (run gen_locomo.py first)"
  fi
fi

echo ""
echo "══════════════════════════════════════════════════════"
echo "  Done."
echo "══════════════════════════════════════════════════════"
