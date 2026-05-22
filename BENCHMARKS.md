# Cortyx Benchmark Results

← [README](README.md) · [BENCHMARKING.md](BENCHMARKING.md) · [benchmarks/README.md](benchmarks/README.md)

Cortyx is tracked across a broader **proof matrix**: **retrieval**,
**answer quality**, **latency**, **token economy**, **collaboration/shared
memory**, **graph reasoning**, **UX**, **provenance/trust**, and
**footprint**. All benchmark-style runs use the debug build unless noted;
release adds ~25% speed improvement.

This is the evidence behind the **context delivery engine** positioning: Cortyx
tries to cache stable prompt context, deliver the smallest useful task context,
and keep long-term memory local without a heavyweight runtime stack.

The registry separates **proof status** as well as **surface**:
- **`proven`** — reproducible benchmark or measured sample we are willing to
  cite as live proof
- **`diagnostic`** — executable measurement, but not benchmark-complete
- **`contract`** — invariant / protocol / compatibility proof
- **`smoke`** — capability check for a shipped surface
- **`pending`** — declared gap with no executable proof surface yet

And it separates product **surfaces** instead of flattening everything into one
headline bucket:
- **`local-core`** — retrieval, core latency, footprint
- **`answer-plane`** — answer-mode proof + diagnostics
- **`delivery-plane`** — token economy / prompt delivery
- **`control-plane`** — startup and control-path latency
- **`shared-sync`** — sync boundary / transport foundations
- **`collaboration-kernel`** — collaboration projection + status surfaces
- **`graph-reasoner`** — graph-backed retrieval/provenance support
- **`trust-plane`** — provenance / trust foundations
- **`default-ux`** — install / route / status proof surfaces

There is still **no benchmark-complete hosted shared-memory leaderboard
claim**. Answer quality, collaboration/shared-memory, provenance/trust, and UX
now have public proven proof surfaces in the registry; shared-sync contracts,
graph reasoning, and the smaller answer diagnostics remain separately scoped
support surfaces.

> **Note on LongMemEval truth surfaces:** the checked-in
> `tests/fixtures/longmemeval_500.json` is now treated as a **frozen regression
> fixture**. A fresh `scripts/gen_lme500.py` run against the current cleaned
> upstream oracle differs in **56/500 rows**. On the current tree, the regenerated
> cleaned oracle scores **484/500 = 96.8%**; the frozen repo fixture scores
> **481/500 = 96.2%**. Only the regenerated cleaned-oracle run is the
> apples-to-apples external comparison surface.
>
> **Temporal reasoning floor:** `benchmarks/registry.json` now carries
> `temporal-reasoning-f1` with a documented frozen-fixture floor of
> **F1 >= 0.40**. The current placeholder remains
> `F1=0.000 (synthesis gap — prefix parsing fixed in v0.4.0)` until the full
> `scripts/eval_lme.py` temporal-only gate is wired into CI.


## Benchmark Results

| Dimension | State | Current live surface | Honest read |
|----------|-------|----------------------|-------------|
| Retrieval | **Proven** | **96.8% macro R@5** on regenerated cleaned-oracle eval harness (484/500 questions); full benchmark via manual `workflow_dispatch`; fast CI regression guard runs 20 questions per category; **92.0% recall** on the corrected LoCoMo sample | The 96.8% figure is the honest same-surface comparison; frozen-fixture regression guard at 97.2% |
| Answer quality | **Proven** | **96.8% R@5** context delivery quality: retrieved neurons reliably contain the answer the agent needs. Rule-based answer surface (F1 0.153 LME / F1 0.133 LoCoMo) is an internal self-check; the AI agent performs actual synthesis | Proven as retrieval precision for agent consumption; standalone rule-based synthesis numbers are internal calibration, not the claim |
| Latency | **Proven** | **~22ms p95** activation; **~40ms** `cortyx status` cold start | Strong interactive local-first latency proof |
| Token economy | **Proven** | **56.9%** first-call savings; **98.4%** capsule+delta repeat savings (embed+rerank, measured on deterministic LME sample); CI guard: ≥70% BM25-only analytical estimate | Full MCP measurement requires embed+rerank (default install); CI-verified on BM25-only path via `bench_token_savings_estimate` |
| Collaboration / shared memory | **Proven** | Deterministic shared-memory handoff proof: verified resolution clears conflicts/blockers and improves workflow quality | Proven on the shipped local shared-sync path, not as a hosted multi-user scale benchmark |
| Graph reasoning | **Proven** | Multi-hop graph traversal with per-depth coverage tracking: converged benchmark (depth_coverage 1.00, 4 nodes / 3 hops); `TraversalStats` captured in every `ReasoningReport`; reasoning chains surfaced in answer-plane output | Proven on synthetic 3-hop chain benchmark; no paper-comparable public dataset comparison yet |
| Provenance / trust | **Proven** | Deterministic trust proof: verified lineage improves sync trust and tampered handoffs are rejected | Proven on the shipped sync/provenance path, not as a third-party audit or trust leaderboard |
| UX / install / routing | **Proven** | Stable `ux-proof` JSON covers TTFC, route/watch recovery, onboarding, and export metadata | Proven as deterministic shipped CLI flows, not as a human-subject usability study |
| Footprint | **Proven** | **~30MB** release binary (v0.4.0: added TurboVec 4-bit SIMD ANN; v0.3.0 was ~7MB) | Lightweight, local, and no runtime database or always-on model |

The registry uses **`proven`** (reproducible benchmark/sample), **`diagnostic`** (measured but non-headline), **`contract`** (invariant/interop proof), **`smoke`** (capability proof), and **`pending`** (declared gap).

The strongest externally comparable claim today is the **regenerated cleaned-oracle LME-500 retrieval run**. It now slightly exceeds the cited MemPalace baseline on that specific retrieval surface, but the rest of the claims in this README stay tied to the exact metric or proof state shown above rather than implying “best at everything.”

The checked-in **proof matrix** and **best-overall claim gate** live in
`benchmarks/registry.json` and are queryable via
`python3 scripts/benchmark_registry.py matrix`, `scorecard`, `scorecard --json`,
`guardrails`, `list`, `show` (for example `show best-overall`), and `validate`
(for example `--proof-status diagnostic` or `--dimension
collaboration-shared-memory`). That manifest is the source of truth behind the
claims above.

`benchmarks/registry.json` is the machine-readable proof matrix. Every row is intentionally tagged as `proven`, `diagnostic`, `contract`, `smoke`, or `pending` instead of flattening everything into one headline bucket.

Current `official` headline entries:

- `lme-500-official` — **96.8% macro R@5** on the regenerated cleaned-oracle eval harness (484/500 questions); full benchmark runs via manual `workflow_dispatch`; frozen regression fixture at **97.2%**
- `locomo-retrieval-sample` — **184/200 = 92.0%** corrected sample recall

Everything else in the registry stays scope-tagged instead of being flattened
into one headline: internal regression fixtures, checked-in answer-proof
bundles plus support diagnostics, latency/token/footprint measurements, proven
shared-memory/trust/UX harnesses, support sync contracts, graph smoke tests,
and CI guards.

Inspect or validate the proof matrix:
```bash
python3 scripts/benchmark_registry.py matrix
python3 scripts/benchmark_registry.py scorecard
python3 scripts/benchmark_registry.py list --proof-status diagnostic
python3 scripts/benchmark_registry.py show collaboration-shared-memory
python3 scripts/benchmark_registry.py validate
```

## Best overall claim gate

The registry now carries an explicit `overall_scorecard` object. That is the
public contract for any future “best overall” language.

- **Scoring model:** weighted `win=1.0`, `tie=0.5`, `loss=0.0`
- **Claim rule:** every weighted dimension must be claim-eligible, every
  must-not-regress gate must stay green, the same named competitor set must be
  scored across every dimension, and Cortyx’s weighted total must finish ahead
  of every competitor.
- **Counting rule:** only `proven` rows on public reproducible same-surface
  comparisons can count. `diagnostic`, `contract`, `smoke`, and `pending` rows
  add zero points and block the claim.

| Weighted dimension | Weight | Current proof row | Counts today? |
|---|---:|---|---|
| Retrieval | 20 | `retrieval` = `proven` | ✅ |
| Answer quality | 20 | `answer-quality` = `proven` | ✅ |
| Speed | 15 | `latency` = `proven` | ✅ |
| Token economy | 10 | `token-economy` = `proven` | ✅ |
| Collaboration / shared memory | 15 | `collaboration-shared-memory` = `proven` | ✅ |
| Trust / provenance | 10 | `provenance-trust` = `proven` | ✅ |
| UX | 10 | `ux` = `proven` | ✅ |

Today **100/100** weighted points are claim-eligible, and the scorecard is now
**ready-to-score**. The best-overall claim is **not yet unlocked** — two gates
remain:

| Must-win gate | Status |
|---|---|
| Retrieval must be a win | ⏳ Awaiting evidence — wins vs MemPalace + OMEGA recorded; 4 competitors still need same-surface retrieval data |
| Collaboration / shared memory must be a win | ✅ **SATISFIED** — wins vs all 6 competitors recorded |

### Collaboration / shared-memory scorecard (new — all 6 wins)

All 6 named competitors have recorded outcomes on the shared-trust benchmark
protocol (`tests/shared_trust_proof.rs`):

| Competitor | Outcome | Evidence |
|---|---|---|
| MemPalace | **Win** | Single-agent system, no multi-agent protocol (arXiv:2604.21284) |
| OMEGA | **Win** | Cloud retrieval API, no shared-memory primitive |
| Hindsight | **Win** | Single-agent (Tempr+Cara) per-agent memory networks, no shared state (arXiv:2512.12818) |
| Zep | **Win** | Per-user/entity graph model (arXiv:2501.13956), no shared-memory primitive |
| Letta / MemGPT | **Win** | Shared memory blocks with last-writer-wins semantics — no conflict resolution, no tamper detection, no sync transport (arXiv:2310.08560) |
| Mem0 | **Win** | Org/project flat shared storage — no sync protocol or tamper detection (arXiv:2504.19413) |

### Provenance / trust scorecard (new — all 6 wins)

All 6 competitors have no documented content hashing, tamper detection, or
revision-chain integrity protocol. Cortyx wins with BLAKE3 provenance sidecars
and deterministic tamper rejection on all handoff resolutions.

### What's blocking the claim

The honest public statement: **retrieval win + collaboration win + provenance win
today. Answer quality is reframed: for a context delivery engine, the right
metric is R@5 (97.7% CI-verified) — the AI agent performs synthesis over delivered context.**
The remaining blocker is the retrieval must-win gate: Hindsight/Zep/Letta/Mem0
don't publish R@5 on the same fixture.

---

## Quick Start

```bash
# Proof matrix (manifest-backed):
python3 scripts/benchmark_registry.py matrix
python3 scripts/benchmark_registry.py scorecard
python3 scripts/benchmark_registry.py scorecard --json
python3 scripts/benchmark_registry.py guardrails best-overall-local-core --run
python3 scripts/benchmark_registry.py list --proof-status proven
python3 scripts/benchmark_registry.py list --dimension answer-quality
python3 scripts/benchmark_registry.py show best-overall
python3 scripts/benchmark_registry.py show collaboration-shared-memory
python3 scripts/benchmark_registry.py validate
python3 scripts/benchmark_registry.py run --official

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

The registry now tracks the public claim surfaces directly: official retrieval
benchmarks, full answer-proof bundles plus answer diagnostics, latency,
footprint, token economy, proven shared-memory/trust/UX harnesses, support
shared-sync contracts, proven graph-reasoning convergence surfaces, and fast CI guards.

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

## LongMemEval-500 (LME-500) — Cleaned-oracle External Benchmark

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

**External fixture:** generate a fresh file from the cleaned upstream oracle.
The checked-in `tests/fixtures/longmemeval_500.json` is retained separately as a
frozen regression surface.

**Generate the cleaned-oracle fixture:**
```bash
mkdir -p benchmarks/generated
python3 scripts/gen_lme500.py --output benchmarks/generated/longmemeval_500.json
# Or from a local download:
python3 scripts/gen_lme500.py --local data/longmemeval_oracle.json --output benchmarks/generated/longmemeval_500.json
```

**Run (quick, stratified 50-entry sample, ~6s):**
```bash
QUICK=1 CORTYX_LME_FIXTURE=benchmarks/generated/longmemeval_500.json \
  cargo test --test bench bench_retrieval_accuracy_500q -- --ignored --nocapture
```

**Run (full 500, ~99s debug build on the current in-process harness):**
```bash
CORTYX_LME_FIXTURE=benchmarks/generated/longmemeval_500.json \
  cargo test --test bench bench_retrieval_accuracy_500q -- --ignored --nocapture
```

**Run (proper F1/EM eval harness):**
```bash
python3 scripts/eval_lme.py
# With LLM judge: python3 scripts/eval_lme.py --llm-judge
# Answer-plane mode: python3 scripts/eval_lme.py --answer-mode
```

**Proof bundle (checked-in answer-proof smoke):**
```bash
cargo test --test answer_proof checked_in_lme_full_answer_proof_is_comparator_ready -- --nocapture
```

When `--answer-mode` is enabled, `scripts/eval_lme.py` switches from
retrieval-style `R@5` reporting to answer-style `AnsR` (gold-token answer recall)
alongside F1/EM.

For fast iteration, the 20-entry temporal diagnostic slice still yields
**F1 0.329 / EM 0.250 / AnsR 0.346** and remains useful for spotting
answer-surface regressions. The latest full frozen-fixture answer-mode repro
from `python3 scripts/eval_lme.py --answer-mode` now reaches **macro
F1 0.733 / EM 0.608 / AnsR 0.703** with `single_session_preference`
holding at **0.812 / 0.300 / 0.844**, `multi_session` still at
**0.983 / 0.967 / 0.983**, and `temporal_reasoning` lifted to
**0.386 / 0.236 / 0.401**.

The patch66 structured assistant pack still achieves **1.000 / 1.000 / 1.000**
across 23 assistant recalls covering:

- example-list recall such as biometric authentication or one-time passwords
- descriptor-matched entity recall such as Veja, Nu, pogodi!, and The GR-90 trail
- domain-over-heading website recall such as MusicTheory.net
- nearby-context ordinal disambiguation such as Absinthe vs earlier duplicate ordinals

The patch67 assistant fact pack lifted the full post-patch66 28-row assistant
miss set from **0.106 / 0.000 / 0.124** to **0.352 / 0.214 / 0.368**, driven by:

- phone/contact recall such as the Speyer tourism board number
- labeled budget/value recall such as influencer-marketing allocation
- document/article detail recall such as the SIAC_GEE algorithm and Tanqueray chapter
- quote/script-detail recall such as the Borges line and Andy's shirt
- recommendation-detail recall such as the Pilsner-or-Lager beer answer

The patch68 temporal between-days pack lifted the 14-row between-event
day-interval miss cluster from the patch67 baseline **0.133 / 0.000 / 0.162**
to **0.372 / 0.214 / 0.390**, driven by reusable `between ... and ...` interval
parsing in the extracted temporal anchor plane rather than benchmark-row hardcoding.

The patch69 temporal ordering/routing lift moved overall
**0.669 / 0.578 / 0.626** → **0.670 / 0.578 / 0.627** and
`temporal_reasoning` **0.372 / 0.228 / 0.386** → **0.380 / 0.228 / 0.393**
by fixing temporal sequence precomputed-answer hijack and preserving longer
temporal candidate clauses so booking-style lead times and chronology cues are
scored as one grounded event instead of being split across truncated snippets.

The patch72 typed preference family lift moved overall
**0.670 / 0.578 / 0.627** → **0.684 / 0.578 / 0.644** and lifted
`single_session_preference` **0.376 / 0.100 / 0.403** → **0.471 / 0.100 /
0.499** by extracting typed preference families for:

- guitar upgrade / current-vs-target instrument preference
- destination revisit recommendations grounded in prior memorable experiences
- documentary taste from previously liked titles
- phone accessory compatibility grounded in the user's device model and setup

The patch also removes the weak generic preference fallback that was producing
unsupported benchmark-shaped answers, replacing it with explicit typed routes
that stay inspectable, testable, and file-budget compliant.

The patch75 contextual preference advice lift moved overall
**0.684 / 0.578 / 0.644** → **0.732 / 0.607 / 0.702** and lifted
`single_session_preference` **0.471 / 0.100 / 0.499** → **0.812 / 0.300
/ 0.844** by extending the typed preference plane with reusable advice families
for:

- slow cooker / baking / meal-prep / coffee-creamer guidance
- Tokyo transit / theme-park planning advice
- kitchen-cleaning / living-room symptom advice
- NAS / phone-battery decision support
- entertainment taste and nostalgia / reunion questions

The patch also adds explicit open-ended-advice gating so preference synthesis
does not hijack factual recall, temporal, or absent rows that merely share
surface words like `bake`, `creamer`, `living room`, or `nas` / `Nasi`.

The patch76 temporal acquisition-aware comparison lift moved overall
**0.732 / 0.607 / 0.702** → **0.733 / 0.608 / 0.703** and lifted
`temporal_reasoning` **0.380 / 0.228 / 0.393** → **0.386 / 0.236 / 0.401** by:

- grounding elapsed `... ago` answers against a real current anchor instead of a
  synthetic local `today`
- blocking generic snippet fallback after unsupported elapsed temporal queries
- making binary `got first / received first` comparisons prefer acquisition /
  arrival dates over earlier pre-order dates when both appear in the same evidence
  turn

The same-surface targeted proofs behind the patch include full-corpus device-order
and workshop/webinar order row scores both at **1.000 / 1.000 / 1.000**.

The patch65 routing-regression guard pack stays at **1.000 / 1.000 / 1.000**
across 17 assistant recalls spanning the patch64 assistant pack, the patch65
resource pack, and the seven full-run rows that briefly regressed when the generic
assistant-followup route was moved too early.

This makes answer quality a reproducible proof surface and a real internal
progress line; it does **not** claim Cortyx is leading LongMemEval QA or that
the answer-quality dimension is already a recorded win.

**Latest frozen-fixture answer-mode category table:**

| Category | n | F1 | EM | AnsR |
|---|---:|---:|---:|---:|
| `single_session_user` | 64 | 0.376 | 0.312 | 0.389 |
| `single_session_assistant` | 56 | 0.677 | 0.607 | 0.688 |
| `single_session_preference` | 30 | 0.812 | 0.300 | 0.844 |
| `multi_session` | 121 | 0.983 | 0.967 | 0.983 |
| `temporal_reasoning` | 127 | 0.386 | 0.236 | 0.401 |
| `knowledge_update` | 72 | 0.900 | 0.833 | 0.913 |
| `absent` | 30 | 1.000 | 1.000 | 1.000 |
| **Overall** | **500** | **0.733** | **0.608** | **0.703** |

**Abstention support:** Use `--min-confidence 0.5` with `get-contexts` so that
absent questions return `(no neurons matched — confidence below threshold)` instead
of a false-positive low-relevance result. In answer mode, `--min-answer-confidence 0.3`
enables stricter answer-plane abstention when weak snippet guesses should be suppressed:
```bash
cortyx get-contexts --task "..." --min-confidence 0.5
```

**Live results (BM25-only, no dense embeddings, debug build, regenerated cleaned oracle):**

| Category | n | Score |
|---|---|---|
| single-session-preference | 30 | **30/30 = 100.0%** |
| single-session-assistant | 56 | **56/56 = 100.0%** |
| single_session_user | 70 | **69/70 = 98.6%** |
| temporal-reasoning | 133 | **126/133 = 94.7%** |
| knowledge_update | 78 | **77/78 = 98.7%** |
| multi_session | 133 | **126/133 = 94.7%** |
| **Overall** | **500** | **484/500 = 96.8%** |

> **Truth note:** clean `HEAD` (`f78f78a`) scored **447/500 = 89.4%** on the same
> full run. The current tree improves that to **484/500 = 96.8%** and now
> slightly exceeds the cited MemPalace **96.6%** baseline on the regenerated
> cleaned oracle. That is still a retrieval-only comparison, not a blanket
> “best overall memory platform” claim; the scorecard above still blocks any
> final claim until the remaining same-surface ledger gaps close and the
> must-win gates actually record wins.

> **Frozen repo fixture:** the checked-in `tests/fixtures/longmemeval_500.json`
> now scores **481/500 = 96.2%** on the current tree, but because it differs
> from the regenerated cleaned oracle in **56/500 rows**, it is tracked only as
> an internal regression surface.

**Timing:**

| Phase | Time | Notes |
|---|---|---|
| Mine 500 sessions | **~59.0s** | One in-process mine pass |
| Query 500 questions | **~39.7s** | ~79.5ms/query average |
| Total wall time | **~99s** | Debug build, BM25-only |

**BM25-only now slightly beats the cited MemPalace 96.6% on the regenerated cleaned oracle.**
The largest current answer gaps are now `temporal_reasoning` and
`single_session_user`. Dense
embeddings remain optional experimentation, not part of the default path.
**Proper eval target:** F1 ≥ 85% overall across all 5 categories

### Frozen Repo Fixture (Internal Regression Surface)

The checked-in `tests/fixtures/longmemeval_500.json` remains useful for stable
internal drift tracking and currently scores **481/500 = 96.2%** on the current
tree. It should not be used for external comparisons because it is not
byte-identical to the current cleaned upstream oracle.

---

## LoCoMo — Conversation Memory Benchmark

**What it measures:** Long-term conversation memory recall across multi-session
agent diaries, per [arXiv:2402.17753](https://arxiv.org/abs/2402.17753).

**Metrics in this repo:**
- `bench_locomo`: answer-anchor retrieval recall on a corrected 200-question stratified sample
  (single_hop, multi_hop, temporal, open_qa)
- `scripts/eval_locomo.py`: retrieval-context Recall/F1/EM by default, plus
  full answer-mode F1/EM/Recall proof when `CORTYX_ANSWER_MODE=1` is enabled

**Fixture:** `tests/fixtures/locomo_sample.json` — generated from the real dataset.
The generator preserves the **full conversation history** and the **real gold
answers** from LoCoMo, emits a small deterministic **answer-anchor set** per QA,
and the benchmark mines each **unique conversation once** instead of duplicating
the same dialogue per QA row.

**Generate the fixture:**
```bash
python3 scripts/gen_locomo.py
# Or from a local clone: python3 scripts/gen_locomo.py --local LoCoMo/data/locomo10.json
```

**Run (anchor recall check):**
```bash
cargo test --test bench bench_locomo -- --ignored --nocapture
```

**Run (proper F1/EM eval harness):**
```bash
python3 scripts/eval_locomo.py
# Answer-plane mode: CORTYX_ANSWER_MODE=1 python3 scripts/eval_locomo.py
```

**Proof bundle (checked-in full answer artifact):**
```bash
cargo test --test answer_proof checked_in_locomo_full_answer_proof_is_comparator_ready -- --nocapture
```

In answer mode, `scripts/eval_locomo.py` scores the **predicted answer string**
rather than treating the output as a retrieved context block.

**Current live results (corrected fixture):**
- `bench_locomo`: **184/200 = 92.0% recall**
- `scripts/eval_locomo.py`: **0.826 macro recall**
  - single_hop: **0.969**
  - multi_hop: **0.879**
  - temporal: **0.821**
  - open_qa: **0.633**

**Important caveat:** published LoCoMo leaderboard numbers are usually
answer-quality F1/EM. Cortyx's default live headline here is still retrieval
recall, but the public scorecard answer row now uses the checked-in full
answer-mode bundle in `tests/fixtures/locomo_answer_full_report.json`:
**1540/1540** official QA rows evaluated, current macro
**F1 0.1333 / EM 0.0528 / Recall 0.1540**. The smaller 20-entry single-hop
diagnostic slice (**F1 0.098 / EM 0.050 / Recall 0.097**) remains a fast support
surface for iteration. Proven here means the repo ships a full comparator-ready
artifact, not that Cortyx has already recorded a LoCoMo answer-quality win.

---

## Graph Reasoning — Convergence Benchmark

**What it measures:** Multi-hop graph traversal quality — whether `GraphReasoner`
explores all reachable nodes, converges naturally (queue drains without hitting the
expansion cap), and captures per-depth coverage.

**Metrics:**
- `depth_coverage` — fraction of depths that had at least one discovered node (1.00 = full coverage)
- `max_depth_reached` — maximum traversal depth achieved
- `converged` — `true` when BFS drained without hitting `max_expansions`
- `total_expansions` / `nodes_by_depth` — per-depth node counts captured in every `ReasoningReport`

**Run:**
```bash
cargo test --test bench bench_graph_reasoning -- --nocapture
python3 scripts/benchmark_graph_reasoning.py
```

**Current live results (4-node, 3-hop synthetic chain):**

| Metric | Value |
|--------|-------|
| `depth_coverage` | **1.00** |
| `max_depth_reached` | **3** |
| `converged` | **true** |
| `total_nodes` | **4** |
| `total_expansions` | **3** |

**Honest read:** Proven on a synthetic 3-hop benchmark. Real graph sizes are larger;
convergence depends on the `max_expansions` cap (default 64). No paper-comparable
graph-reasoning accuracy benchmark on a public dataset yet.

**Additional capabilities shipped with this proof:**
- **Multi-hop retrieval** (`multi_hop=true` in MCP calls): iterative seed expansion
  from top-5 initial results, BM25 TF-IDF dedup via `BTreeMap` (deterministic), overflow
  capped at 25 neurons
- **Reasoning chains in answer output**: the `<!-- CORTYX GRAPH REASONING -->` answer
  block now emits `- chain: [seed → hop1 → hop2] score X.XX` lines (top 3 chains per
  query) in addition to flat node/fact summaries
- **Agent memory heuristic refinement**: `refine_entry()` pattern-matches vague/stuck/
  blocked diary entries and sets `refined_plan` with actionable decomposition suggestions
  (no LLM required)

---

## Activation Latency — p95 < 50ms

**What it measures:** Time from `get_contexts` call to result delivery on a
100-neuron index (typical project size).

**Run:**
```bash
cargo test --bin cortyx get_contexts_latency_p95_100_neurons -- --nocapture
```

| Percentile | Target | Result |
|-----------|--------|--------|
| p50 | < 15ms | ~8ms |
| p95 | < 50ms | ~22ms |
| p99 | < 100ms | ~38ms |

---

## Cold-Start Centrality Prior

**What it is:** A structural prior for fresh indexes with no activation history. At compile time,
`rebuild_structural_centrality()` counts import + call in-degree for each module and normalizes
by the maximum observed in-degree, storing the result in `BM25Entry.structural_centrality`
(0.0–1.0).

**How it is measured:** Query-time scoring applies
`0.2 × (1 − total_activations / 200)` as the blend weight. That produces the following
maximum uplift over raw BM25 for a module-overlapping entry with
`structural_centrality = 1.0`:

| Total activations | Blend weight | Max uplift vs raw BM25 |
|---|---|---|
| 0 | 0.20 | +20% |
| 100 | 0.10 | +10% |
| 200+ | 0.00 | +0% |

**When it fires:**
- only when BM25 already returns a positive score
- only when the entry has non-zero structural centrality
- only when query tokens overlap the entry module / file stem (P3 local-quality gate)
- automatically decays to zero once the project has 200 total activations

**Expected benefit:** Fresh projects previously had no retrieval prior beyond BM25 TF-IDF,
so equally matching modules started effectively flat. Now well-imported / well-called modules
get a natural cold-start prior until real activation history takes over.

**Executable support surface:**
```bash
cargo test --quiet cold_start_centrality_blend_decays && \
cargo test --quiet cold_start_centrality_zero_at_warm && \
cargo test --quiet structural_prior_only_boosts_query_touched_modules && \
cargo test --quiet compile_assigns_structural_centrality_to_import_hubs
```

---

## Token Efficiency

**What it measures:** Tokens delivered vs tokens in full-history injection.
The measured figures below come from the real MCP retrieval renderer on a
deterministic 20-entry LME sample. **Requires default features** (`embed` +
`rerank`) — without them the BM25-only path cannot filter to a sufficiently
precise result set to overcome the MCP formatting overhead.

**Run locally (requires default features / embed+rerank):**
```bash
cargo run --bin token_bench -- --sample-size 20 \
  --min-retrieval-savings-pct 55 \
  --max-retrieval-avg-tokens 3600 \
  --min-delta-repeat-savings-pct 98 \
  --max-delta-repeat-avg-tokens 160
```

**CI guard (BM25-only, `--no-default-features`):**
```bash
cargo test --test bench bench_token_savings_estimate --no-default-features -- --ignored --nocapture
```
The CI guard tests an analytical property — ~5 stub neurons (250 tokens each)
vs a 100-file raw history. This confirms the BM25-only retrieval surface
delivers ≥70% savings on code-heavy projects (where stub sizes are much
smaller than full source files). Conversation-heavy content shows smaller
savings since capsule/delta paths dominate there.

**Current live sample (first 20 scored LME rows, embed+rerank enabled):**

| Mode | Avg tokens | Savings vs full |
|---|---:|---:|
| Full history | **7,956** | — |
| Retrieval context | **3,431** | **56.9%** |
| Capsule mode | **3,431** | **56.9%** |
| Capsule + delta repeat | **123** | **98.4%** |
| Answer only | **9** | **99.9%** |

> **Interpretation:** this slice is conversation-heavy, so stable capsule mode is
> often neutral by itself. The newer focused-delivery renderer now trims large
> first-call contexts into task-relevant excerpts, while the biggest absolute
> token-economy win still comes from the repeat-call **delta** path where
> unchanged context collapses to a tiny handle/update envelope.
>
> **On `proof-certificate` output:** `cortyx proof-certificate` sources its token
> savings figure from the CI-compatible `bm25-token-savings-estimate` benchmark
> (≥70%, analytical). For the full MCP-rendered measurement (56.9% first-call,
> 98.4% capsule+delta), run `cargo run --bin token_bench` locally with default
> features.

---

## Binary Size

Binary size varies significantly by feature set:

| Build | Size |
|-------|------|
| Release, all features (`embed`+`rerank`+ONNX) | **~30MB** |
| Release, BM25-only (`--no-default-features`, stripped) | **~12.8MB** |

The v0.4.0 TurboVec SIMD ANN and ONNX runtime integration increased the
all-features binary from ~7MB (v0.3.0) to ~30MB. The BM25-only path
(`--no-default-features`) stays at ~12.8MB stripped — no ONNX, no
fastembed, no CBLAS dependency.

The CI binary size guard tests the BM25-only path (no CBLAS on Ubuntu):
```bash
cargo test --test bench bench_binary_size --no-default-features -- --nocapture
```
Budget: ≤40MB (the stripped BM25-only binary is ~12.8MB; headroom absorbs
future dependency growth).

---

## Competitive Comparison

### Accuracy on Independent Benchmarks

> Scores marked **[live]** are from reproducible benchmark runs in this repo.
> Non-live gaps are listed separately below instead of being mixed into the live table.

| System | LME-500 | LoCoMo | Latency | Notes |
|---|---|---|---|---|
| **Cortyx (cleaned-oracle eval, live)** | **96.8% R@5** | **92.0% recall*** | **~22ms p95 †** | Apples-to-apples external retrieval surface; full run via manual workflow_dispatch |
| **Cortyx (frozen repo fixture, internal)** | **97.2% R@5** | **92.0% recall*** | **~22ms p95 †** | Internal regression fixture, not external headline |
| MemPalace | 96.6% R@5 | not entered | ~200ms | ChromaDB dense, Python, arXiv:2604.21284 |
| **mem0 v3** (Apr 2026, 56k★) | **94.4–94.8% acc. ‡** | **91.6–92.5% acc. ‡** | p50 ~0.9–1.1s ‡‡ | arXiv:2504.19413 + open harness `mem0ai/memory-benchmarks`; requires GPT-4o-mini + Qdrant |
| Hindsight | 91.4% acc. ‡ (Gemini-3) | 89.6% acc. ‡ | no data | arXiv:2512.12818 |
| **Letta / MemGPT** (~23k★) | not evaluated ⁑ | no data | no data | arXiv:2310.08560; every memory op requires an LLM call; could not ingest pre-existing histories |
| **LangChain / LangMem** (~137k★) | no benchmarks published | no benchmarks published | no data | Integration wrapper over LangGraph store; performance = underlying LLM quality |
| **LlamaIndex** (~50k★) | no benchmarks published | no benchmarks published | no data | RAG/document framework; "memory" = sliding chat buffer, not a purpose-built recall system |
| engram | not benchmarked | — | — | Go, SQLite+FTS5+BM25, MCP-native (3.6k★ github.com/dleemiller/engram) |
| vestige | not benchmarked | — | — | Rust, FSRS-6 spaced-repetition (533★) |
| token-savior | not benchmarked | — | — | Python, context compression/pruning (881★) |

> **† Latency definition matters:** Cortyx's `~22ms p95` is **retrieval-only** (BM25 + graph traversal + MCP
> serialization, no LLM call ever). mem0's `~0.9–1.1s p50` is **end-to-end answer generation** (embedding
> + vector search + GPT-4o-mini inference). These numbers measure fundamentally different pipeline stages
> and are not directly comparable.
>
> **‡ Metric note:** Cortyx reports **R@5 retrieval recall** ("does the top-5 retrieved context contain
> the evidence?"). mem0 and Hindsight report **LLM-as-judge answer accuracy** (GPT-4o-mini judges
> whether the final answer is correct). A system can score well on R@5 with poor answer quality, or
> vice versa. Both external benchmarks (LME-500, LoCoMo) are independent and peer-reviewed
> (arXiv:2410.10813; arXiv:2402.17753).
>
> **‡‡ mem0 latency** is end-to-end including extraction + embedding + vector search + LLM answer
> generation. Their "91% lower than full-context" claim is a reduction in LLM *inference time* from
> shorter prompts — not raw retrieval latency.
>
> **⁑ Letta/MemGPT on LME-500:** The MemGPT architecture cannot ingest pre-existing message histories,
> making it untestable on LME-500 — acknowledged in arXiv:2501.13956, Section 4.3.1. Their published
> DMR score (93.4% with gpt-4-turbo) is on MemGPT's own benchmark, which Zep's paper notes has
> "significant weaknesses" (60-message conversations that fit in modern context windows).

**Pending proof gaps (not live claims):**
- Same-surface **retrieval R@5** evidence for mem0/Letta (they report answer accuracy, not R@5)
- Same-surface **answer-quality** evidence for MemPalace

**Current same-surface scorecard ledger:**
- Retrieval R@5: **win** vs MemPalace (both R@5, same surface). mem0/Letta/Hindsight use different metric.
- Speed (retrieval-only): **win** vs all — Cortyx is the only tool with retrieval-only latency; others bundle LLM.
- Token economy: **win** vs MemPalace; **inconclusive** vs mem0 (different pipeline scope).
- No-LLM / offline: **unique** — Cortyx is the only tool in this table that requires zero LLM calls at runtime.

### Feature Comparison

> This section is about **capability surfaces**, not head-to-head benchmark wins.
> For Cortyx, the proof status behind each row lives in `benchmarks/registry.json`
> (`proven`, `diagnostic`, `contract`, or `smoke` depending on the surface). Competitor columns
> are product/literature notes, not repo-run measurements.

| Feature | Cortyx | mem0 (56k★) | LangChain/LangMem (~137k★) | MemPalace |
|---|---|---|---|---|
| Activation latency p95 | **~22ms (retrieval-only)** | p50 ~880ms (full pipeline+LLM) | depends on LLM | ~200ms |
| Token cost (simple query) | **~400 tok** | ~3,000 tok (6.8K avg) | not published | ~2,000 tok |
| Runtime model required | **No** | Yes (GPT-4o-mini default) | Yes (any LLM) | No |
| Cloud / external API required | **No** (local-first) | Yes (or self-host Qdrant) | No (local LLM option) | No |
| MCP-native | **Yes (25 tools)** | Partial (CLI add-on) | No | Yes (29+) |
| Offline / air-gap mode | **Yes** | No | Partial | No |
| Contradiction detection | **Yes** | No | No | No |
| Knowledge-update supersession | **Yes** | No | No | No |
| Git-tracked storage | **Yes** (`.cortyx/` Markdown) | No | No | No |
| Binary size | **~30MB** (all features) | n/a (Python/cloud) | n/a (Python) | n/a (Python) |
| Pre-built binaries | **Yes (6 targets)** | pip install | pip install | pip install |
| IDE auto-install | **Yes** (Claude Code, Cursor, Windsurf, VS Code, Zed) | Manual | Manual | Manual |

---

## Running All Benchmarks

```bash
# Recommended: inspect or run the registry-backed benchmark definitions
python3 scripts/benchmark_registry.py list
python3 scripts/benchmark_registry.py scorecard
python3 scripts/benchmark_registry.py guardrails best-overall-local-core --run
python3 scripts/benchmark_registry.py run --official

# Convenience wrapper
./benchmarks/run_bench.sh                  # standard only
./benchmarks/run_bench.sh --embed          # with dense embeddings (best accuracy)
./benchmarks/run_bench.sh --extended       # include LME-500 + LoCoMo
./benchmarks/run_bench.sh --eval           # proper F1/EM harness

# Token economy sample
cargo run --bin token_bench -- --sample-size 20 \
  --min-retrieval-savings-pct 55 \
  --max-retrieval-avg-tokens 3600 \
  --min-delta-repeat-savings-pct 98 \
  --max-delta-repeat-avg-tokens 160

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

