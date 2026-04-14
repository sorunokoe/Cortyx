# Cortyx Benchmark Results

Cortyx is benchmarked across three dimensions: **retrieval accuracy** (R@5),
**activation latency** (p95 ms), and **token efficiency** (% savings vs
full-context injection). All benchmarks run against the debug build unless
noted; release adds ~25% speed improvement.

> **Note on comparisons:** The competitive table at the bottom of this file
> compares Cortyx against other memory systems on the same independent datasets
> (LongMemEval-500, LoCoMo). Earlier versions of this file compared Cortyx's
> score on a synthetic self-made fixture against MemPalace's score on the real
> LongMemEval-500 — that comparison was not apples-to-apples and has been removed.

---

## Quick Start

```bash
# Standard (always fast, no fixtures needed):
./benchmarks/run_bench.sh

# With dense embeddings (best accuracy, downloads ~80MB model once):
./benchmarks/run_bench.sh --embed

# Extended — requires fixture generation first (see below):
python3 scripts/gen_lme500.py && python3 scripts/gen_locomo.py
./benchmarks/run_bench.sh --extended

# Proper F1/EM evaluation harness:
./benchmarks/run_bench.sh --eval
```

---

## LongMemEval-100 (LME-100) — Internal Accuracy Smoke-Test

**What it measures:** Retrieval Recall@5 on a 100-entry synthetic fixture of
code and conversation neurons built by the Cortyx team.

> ⚠️ This is an **internal smoke-test**, not the official LongMemEval benchmark.
> The fixture uses Cortyx's own code-retrieval workload and is not comparable
> to MemPalace's score on the real LongMemEval-500.

**Fixture:** `tests/fixtures/longmemeval_100.json` — 100 synthetic QA pairs
across Core, Concept, and Verbatim neurons.

**Run:**
```bash
cargo test --test bench bench_retrieval_accuracy_50q -- --nocapture
```

| Metric | Value |
|--------|-------|
| R@5    | **99%** (99/100, live run) |
| Total latency | ~6s (100 compile+query cycles) |
| Per-query p50 | ~46ms |
| Per-query p95 | ~120ms |

---

## LongMemEval-500 (LME-500) — Official Accuracy Benchmark

**What it measures:** The real LongMemEval benchmark (arXiv:2410.10813, ICLR 2025,
UC Santa Barbara). 500 questions across 5 types embedded in multi-session conversation
histories up to 1.5M tokens:

| Category | What it tests |
|---|---|
| `single_session_user` | Direct fact recall from one session |
| `single_session_assistant` | Assistant-stated fact recall |
| `multi_session` | Synthesize evidence across multiple sessions |
| `temporal_reasoning` | When / before / after / most recent |
| `knowledge_update` | Fact changed — what is current? |
| `absent` | Answer NOT in history — system must abstain |

**Fixture:** `tests/fixtures/longmemeval_500.json` — generated from the real dataset.

**Generate the fixture:**
```bash
python3 scripts/gen_lme500.py
# Or from a local clone: python3 scripts/gen_lme500.py --local LongMemEval/data/longmemeval_oracle.json
```

**Run (quick, ~6s on 50 entries):**
```bash
QUICK=1 cargo test --test bench bench_retrieval_accuracy_500q -- --ignored --nocapture
```

**Run (full 500, ~10 min with 5000-char session truncation):**
```bash
cargo test --test bench bench_retrieval_accuracy_500q -- --ignored --nocapture
```

**Run (proper F1/EM eval harness):**
```bash
python3 scripts/eval_lme.py
# With LLM judge: python3 scripts/eval_lme.py --llm-judge
```

**Abstention support:** Use `--min-confidence 0.5` with `get-contexts` so that
absent questions return `(no neurons matched — confidence below threshold)` instead
of a false-positive low-relevance result:
```bash
cortyx get-contexts --task "..." --min-confidence 0.5
```

**Live results (BM25-only, no dense embeddings, debug build):**

| Category | n | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Δ R4→R5 |
|---|---|---|---|---|---|---|---|
| single-session-preference | 30 | 100.0% | 100.0% | 100.0% | 100.0% | **100.0%** | — |
| single-session-assistant | 56 | 92.9% | 92.9% | 92.9% | 92.9% | **92.9%** | — |
| single-session-user | 70 | 78.6% | 77.1% | 71.4% | 75.7% | **74.3%** | -1.4% |
| temporal-reasoning | 133 | 78.2% | 80.5% | 78.9% | **82.7%** | 78.2% | ⚠️ -4.5% |
| knowledge-update | 78 | 55.1% | **57.7%** | 55.1% | 55.1% | 53.8% | -1.3% |
| multi-session | 133 | 45.9% | 48.9% | 47.4% | 48.1% | **48.9%** | +0.8% |
| **Overall** | **500** | 69.0% | **70.6%** | 68.6% | 70.4% | **69.0%** | **⚠️ -1.4%** |

**Timing:**

| Run | Mine | Queries | Total |
|---|---|---|---|
| Run 1 | ~216s | ~324s | ~540s |
| Run 2 | ~707s | ~814s | ~1521s |
| Run 3 | ~844s | ~1104s | ~1948s |
| Run 4 | ~569s | ~676s | ~1245s |
| Run 5 | **~568s** | **~674s** | **~1242s** ✅ stable |

**Target with dense embeddings (`--features embed`):** R@5 ≥ 97% (beats MemPalace 96.6%)  
**Proper eval target:** F1 ≥ 85% overall across all 5 categories

---

## LoCoMo — Conversation Memory Benchmark

**What it measures:** Long-term conversation memory recall across multi-session
agent diaries, per [arXiv:2402.17753](https://arxiv.org/abs/2402.17753).

**Metrics (per the paper):** F1, Exact Match (EM), Recall per question type
(single_hop, multi_hop, temporal, open_qa).

**Fixture:** `tests/fixtures/locomo_sample.json` — generated from the real dataset.

**Generate the fixture:**
```bash
python3 scripts/gen_locomo.py
# Or from a local clone: python3 scripts/gen_locomo.py --local LoCoMo/data/locomo10.json
```

**Run (basic keyword check):**
```bash
cargo test --test bench bench_locomo -- --ignored --nocapture
```

**Run (proper F1/EM eval harness):**
```bash
python3 scripts/eval_locomo.py
```

**Target:** F1 ≥ 87% (beats Zep ~85%, approaches Hindsight 89.6%)

---

## Activation Latency — p95 < 50ms

**What it measures:** Time from `get_contexts` call to result delivery on a
100-neuron index (typical project size).

**Run:**
```bash
cargo test --test bench bench_latency_p95_100_neurons -- --nocapture
```

| Percentile | Target | Result |
|-----------|--------|--------|
| p50 | < 15ms | ~8ms |
| p95 | < 50ms | ~22ms |
| p99 | < 100ms | ~38ms |

---

## Token Efficiency

**What it measures:** Tokens delivered vs tokens in full-context injection
(naive approach = send all neurons every time).

| Scenario | Full-context | Cortyx | Savings |
|----------|-------------|--------|---------|
| Simple query (1 neuron) | ~8,000 tok | ~400 tok | **95%** |
| Complex refactoring (5 neurons) | ~8,000 tok | ~2,000 tok | **75%** |
| Wake-up priming (S5) | n/a | ~170 tok | — |

---

## Binary Size

Release binary target: ≤ 8MB (zero runtime dependencies, pure Rust).

| Build | Size |
|-------|------|
| Debug | ~25MB |
| Release | ~7MB |

```bash
cargo test --test bench bench_binary_size -- --nocapture
```

---

## Competitive Comparison

### Accuracy on Independent Benchmarks

> Scores marked **[target]** are not yet achieved — they are the goal after
> completing the remaining retrieval improvements (dense embed, multi-hop, temporal
> query routing). Scores marked **[live]** are from real benchmark runs.

| System | LME-500 R@5 | LoCoMo QA F1 | Notes |
|---|---|---|---|
| **Cortyx (BM25 only, live)** | **69.0%** | — | Pure Rust, debug build (best run: 70.6% R2) |
| **Cortyx (BM25 + embed target)** | **[target] ≥97%** | **[target] ≥87%** | `--features embed`, release build |
| MemPalace | 96.6% | not entered | Verbatim ChromaDB, Python |
| OMEGA | 95.4% | — | Cloud |
| Zep | ~81.6% | ~85% | Graph-based, self-host |
| Letta / MemGPT | ~79% | ~83.2% | Agentic, open-source |
| Mem0 | — | 58–67% | Cloud, production-ready |

> **Note on domains:** Cortyx is primarily a *code context retrieval* tool (MCP
> for IDEs). MemPalace is a *conversational memory* tool. The comparison above
> is meaningful because both run on the same LME-500 dataset; the workloads are
> different but the dataset is independent.

### Feature Comparison

| Feature | Cortyx | MemPalace | mem0 |
|---|---|---|---|
| Activation latency p95 | **~22ms** | ~200ms | ~500ms+ |
| Token cost (simple query) | **~400 tok** | ~2,000 tok | ~3,000 tok |
| Binary size | **7MB** | n/a (Python) | n/a (Python) |
| Zero dependencies at runtime | **Yes** | No | No |
| MCP tools | **25** | 19 | ~10 |
| Temporal KG | **Yes** | No | Limited |
| Contradiction detection | **Yes** | No | No |
| Knowledge-update supersession | **Yes** | No | No |
| Abstention signal (`--min-confidence`) | **Yes** | No | No |
| Git-tracked neurons | **Yes** | No | No |
| Dense embedding (hybrid BM25+dense) | **Yes** (`--features embed`) | Yes (only) | Yes |
| Auto-install | **Yes** | No | No |

---

## Running All Benchmarks

```bash
# Recommended: use the benchmark runner script
./benchmarks/run_bench.sh                  # standard only
./benchmarks/run_bench.sh --embed          # with dense embeddings (best accuracy)
./benchmarks/run_bench.sh --extended       # include LME-500 + LoCoMo
./benchmarks/run_bench.sh --eval           # proper F1/EM harness

# Manual: standard (always run, ~7s):
cargo test --test bench -- --nocapture

# Manual: extended (require fixtures):
python3 scripts/gen_lme500.py
python3 scripts/gen_locomo.py
cargo test --test bench -- --ignored --nocapture

# Manual: proper eval harnesses:
python3 scripts/eval_lme.py
python3 scripts/eval_locomo.py

# Release binary size check:
cargo build --release
cargo test --test bench bench_binary_size -- --nocapture
```

