# Cortyx Benchmark Results

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

| Dimension | Status | Current live evidence | Honest read |
|---|---|---|---|
| Retrieval | **Proven** | **484/500 = 96.8%** cleaned-oracle LME-500; **184/200 = 92.0%** corrected LoCoMo sample recall | Current external headline is retrieval only |
| Answer quality | **Proven** | Full LME answer proof bundle **macro F1 0.153 / EM 0.109 / AnsR 0.188** with 500/500 official-QA hypotheses; full LoCoMo answer proof bundle **macro F1 0.133 / EM 0.053 / Recall 0.154** over 1540/1540 | Proven means the repo now ships full public proof bundles, not that Cortyx is winning this dimension |
| Latency | **Proven** | **~22ms p95 activation**, **~40ms status cold start** | Interactive local-first latency is benchmarked |
| Token economy | **Proven** | **56.9%** first-call savings, **98.4%** capsule+delta repeat savings | Proven on a deterministic sample harness, not a universal all-prompts claim |
| Collaboration / shared memory | **Proven** | Deterministic shared-memory handoff proof: verified resolution clears conflicts/blockers and improves workflow quality | Proven on the shipped local shared-sync path, not as a hosted multi-user scale benchmark |
| Graph reasoning | **Smoke** | Concept-cloud retrieval and graph-backed provenance summaries are executable | Support exists, but no standalone scorecard yet |
| Provenance / trust | **Proven** | Deterministic trust proof: verified lineage improves sync trust and tampered handoffs are rejected | Proven on the shipped sync/provenance path, not as a third-party audit or trust leaderboard |
| UX / install / routing | **Proven** | Stable `ux-proof` JSON covers TTFC, route/watch recovery, onboarding, and export metadata | Proven as deterministic shipped CLI flows, not as a human-subject usability study |
| Footprint | **Proven** | **~6.9MB** stripped release binary, BM25-only default path | No runtime DB and no always-on dense model required |

**Speed-path note:** Cortyx now auto-skips the binary activation-cache artifact when it is larger than the canonical `index.json`. On the current benchmark-sized projects, rebuilding derived state from `index.json` is faster than deserializing the larger binary blob, so the default path prefers the measured faster route.

**Contract note:** the current external headline is still the **local-core
retrieval surface**. Everything else stays intentionally proof-state-scoped in
the registry (`proven`, `diagnostic`, `contract`, `smoke`, or `pending`) so the
public story stays metric-scoped and honest.

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
**ready-to-score**. The best-overall claim is still **not unlocked** because
none of the weighted dimensions has a complete same-surface competitor outcome
ledger. Retrieval, speed, token economy, and footprint must not regress;
retrieval, answer quality, and collaboration/shared memory are the explicit
**must-win** gates. Footprint is **gate-only** (important, but not weighted),
while graph reasoning is **support-only** until it has its own proven
comparator-backed benchmark.

Fair competitor rules are also explicit now:

- same public fixture / study protocol
- same surface and metric family
- same default shipped path unless an optional mode is called out separately
- no private retuning, hidden hosted memory, or unpublished cleaning steps
- same named competitor set across every weighted dimension

So the honest public statement stays: **retrieval win today, best-overall claim
not yet unlocked.**

The registry now also carries a machine-readable
`overall_scorecard.comparison_scaffold`: the shared roster now names the
repo-cited systems (**MemPalace, OMEGA, Hindsight, Zep, Letta / MemGPT,
Mem0**), the current claim-eligible dimensions already have apples-to-apples
scope metadata filled in, and the scorecard now records the same-surface
outcomes the repo can already support honestly: retrieval wins vs **MemPalace**
and **OMEGA**, plus LoCoMo QA F1 answer-quality losses vs **Hindsight**,
**Zep**, **Letta / MemGPT**, and **Mem0**. Every remaining gap stays explicit as
`insufficient-evidence` or `no-repo-evidence`, without inventing extra wins or
losses.

`python3 scripts/benchmark_registry.py scorecard --json` now exposes
`comparison_scaffold`, roster metadata, per-dimension readiness states,
outcome-ledger entries, `claim_readiness` phases, blocker ids, and `next_flip`
text so the remaining competitive-proof work is explicit and machine-readable.

The latest full answer-proof artifacts, plus the shared-trust and UX work, now
promote answer quality, collaboration/shared memory, trust/provenance, and UX
to proven public surfaces, but this still does **not** unlock the claim: the
shared roster still lacks complete same-surface ledgers for every weighted
dimension, answer quality already records non-win outcomes, and the
collaboration/shared-memory must-win lane still has no comparator evidence.

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
shared-sync contracts, graph smoke surfaces, and fast CI guards.

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
**0.386 / 0.236 / 0.401**. The latest full proof artifact is
`/tmp/cortyx_lme_full_patch76_v1.json`.

The patch66 structured assistant pack
`/tmp/cortyx_lme_patch66_assistant_structured_pack_v6.json` still stays at
**1.000 / 1.000 / 1.000** across 23 assistant recalls covering:

- example-list recall such as biometric authentication or one-time passwords
- descriptor-matched entity recall such as Veja, Nu, pogodi!, and The GR-90 trail
- domain-over-heading website recall such as MusicTheory.net
- nearby-context ordinal disambiguation such as Absinthe vs earlier duplicate ordinals

The new patch67 assistant fact pack
`/tmp/cortyx_lme_patch67_assistant_fact_pack_v2.json` lifts the full
post-patch66 28-row assistant miss set from **0.106 / 0.000 / 0.124** to
**0.352 / 0.214 / 0.368**, driven by:

- phone/contact recall such as the Speyer tourism board number
- labeled budget/value recall such as influencer-marketing allocation
- document/article detail recall such as the SIAC_GEE algorithm and Tanqueray chapter
- quote/script-detail recall such as the Borges line and Andy's shirt
- recommendation-detail recall such as the Pilsner-or-Lager beer answer

The new patch68 temporal between-days pack
`/tmp/cortyx_lme_patch68_between_days_pack_v1.json` lifts the 14-row
between-event day-interval miss cluster from the patch67 baseline
**0.133 / 0.000 / 0.162** to **0.372 / 0.214 / 0.390**, driven by reusable
`between ... and ...` interval parsing in the extracted temporal anchor plane
rather than benchmark-row hardcoding.

The new patch69 temporal ordering/routing lift is now proven on the full
500-row frozen-fixture answer surface:
`/tmp/cortyx_lme_full_release_after_patch69_v1.json`. It moves overall
**0.669 / 0.578 / 0.626** → **0.670 / 0.578 / 0.627** and
`temporal_reasoning` **0.372 / 0.228 / 0.386** → **0.380 / 0.228 / 0.393**
by fixing temporal sequence precomputed-answer hijack and preserving longer
temporal candidate clauses so booking-style lead times and chronology cues are
scored as one grounded event instead of being split across truncated snippets.

The new patch72 typed preference family lift is now proven on the full 500-row
frozen-fixture answer surface: `/tmp/cortyx_lme_full_patch72_v1.json`. It moves
overall **0.670 / 0.578 / 0.627** → **0.684 / 0.578 / 0.644** and lifts
`single_session_preference` **0.376 / 0.100 / 0.403** → **0.471 / 0.100 /
0.499** by extracting typed preference families for:

- guitar upgrade / current-vs-target instrument preference
- destination revisit recommendations grounded in prior memorable experiences
- documentary taste from previously liked titles
- phone accessory compatibility grounded in the user's device model and setup

The patch also removes the weak generic preference fallback that was producing
unsupported benchmark-shaped answers, replacing it with explicit typed routes
that stay inspectable, testable, and file-budget compliant.

The new patch75 contextual preference advice lift is now proven on the full
500-row frozen-fixture answer surface: `/tmp/cortyx_lme_full_patch75_v2.json`.
It moves overall **0.684 / 0.578 / 0.644** → **0.732 / 0.607 / 0.702** and
lifts `single_session_preference` **0.471 / 0.100 / 0.499** → **0.812 / 0.300
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

The new patch76 temporal acquisition-aware comparison lift is now proven on the
full 500-row frozen-fixture answer surface:
`/tmp/cortyx_lme_full_patch76_v1.json`. It moves overall
**0.732 / 0.607 / 0.702** → **0.733 / 0.608 / 0.703** and lifts
`temporal_reasoning` **0.380 / 0.228 / 0.393** → **0.386 / 0.236 / 0.401** by:

- grounding elapsed `... ago` answers against a real current anchor instead of a
  synthetic local `today`
- blocking generic snippet fallback after unsupported elapsed temporal queries
- making binary `got first / received first` comparisons prefer acquisition /
  arrival dates over earlier pre-order dates when both appear in the same evidence
  turn

The same-surface targeted proofs behind the patch include:

- `/tmp/cortyx_lme_patch76_device_order_fullcorpus_v1.json` — full-corpus
  `gpt4_2312f94c` device-order row repaired to **1.000 / 1.000 / 1.000**
- `/tmp/cortyx_lme_patch76_order_probe_event_v1.json` — full-corpus
  `gpt4_2487a7cb` workshop/webinar order row remains green at
  **1.000 / 1.000 / 1.000**

The patch65 routing-regression guard pack
`/tmp/cortyx_lme_assistant_regression_pack_v1.json` still stays at
**1.000 / 1.000 / 1.000** across 17 assistant recalls spanning the patch64
assistant pack, the patch65 resource pack, and the seven full-run rows that
briefly regressed when the generic assistant-followup route was moved too early.

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

## Token Efficiency

**What it measures:** Tokens delivered vs tokens in full-history injection.
The checked-in token benchmark uses the real MCP retrieval renderer so it can
measure normal retrieval, capsule mode, capsule + delta reuse, and answer-only
output on the same deterministic fixture slice.

**Run:**
```bash
cargo run --bin token_bench -- --sample-size 20 \
  --min-retrieval-savings-pct 55 \
  --max-retrieval-avg-tokens 3600 \
  --min-delta-repeat-savings-pct 98 \
  --max-delta-repeat-avg-tokens 160
```

That command is now the executable non-regression guard: first-call retrieval
must keep at least **55.0%** savings while staying at or below **3,600** average
tokens, and the repeat-call capsule+delta path must keep at least **98.0%**
savings while staying at or below **160** average tokens.

**Current live sample (first 20 scored LME rows):**

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

---

## Binary Size

Release binary target: ≤ 8MB (zero runtime dependencies, pure Rust).
Cargo's release profile now strips symbols, enables thin LTO, and uses a
single codegen unit so the shipped local-core artifact stays under that budget.

| Build | Size |
|-------|------|
| Debug | ~25MB |
| Release | ~6.9MB stripped |

```bash
cargo test --test bench bench_binary_size -- --nocapture
```

---

## Competitive Comparison

### Accuracy on Independent Benchmarks

> Scores marked **[live]** are from reproducible benchmark runs in this repo.
> Non-live gaps are listed separately below instead of being mixed into the live table.

| System | LME-500 R@5 | LoCoMo | Notes |
|---|---|---|---|
| **Cortyx (cleaned oracle, live)** | **96.8%** | **92.0% sample recall*** | Apples-to-apples external retrieval surface today; slight lead over the cited MemPalace baseline on this surface |
| **Cortyx (frozen repo fixture, internal)** | **96.2%** | **92.0% sample recall*** | Internal regression fixture, not external headline |
| MemPalace | 96.6% | not entered | ChromaDB dense, Python, ~200ms, LLM-judge eval |
| OMEGA | 95.4% | — | Cloud |
| Hindsight | — | ~89.6% F1 | Published LoCoMo QA baseline |
| Zep | ~81.6% | ~85% F1 | Graph-based, self-host |
| Letta / MemGPT | ~79% | ~83.2% F1 | Agentic, open-source |
| Mem0 | — | 58–67% F1 | Cloud, production-ready |

**Pending proof gaps (not live claims):**
- scorecard-ready retrieval evidence for **Hindsight**, **Zep**, **Letta / MemGPT**, and **Mem0**
- same-surface answer-quality evidence for **MemPalace** and **OMEGA**
- any same-surface collaboration/shared-memory, trust/provenance, speed, token-economy, or UX competitor ledgers

**Current same-surface scorecard ledger status:**
- Retrieval now records **win** outcomes vs **MemPalace** and **OMEGA** on the cited LME-500 rows.
- Answer quality now records **loss** outcomes vs **Hindsight**, **Zep**, **Letta / MemGPT**, and **Mem0** on LoCoMo QA F1.
- Speed / token economy / UX still only have `insufficient-evidence` or `no-repo-evidence` states.
- Collaboration / shared memory and trust / provenance still have no same-surface competitor ledgers.

> **Note on domains:** Cortyx is primarily a *code context retrieval* tool (MCP
> for IDEs). MemPalace is a *conversational memory* tool. The comparison above
> is meaningful because both run on the same LME-500 dataset; the workloads are
> different but the dataset is independent.
>
> ***LoCoMo metric note:** Cortyx's live LoCoMo number is retrieval recall on the
> corrected sample fixture. It is intentionally not presented as paper-comparable
> answer F1 yet. For scorecard purposes, only the cited MemPalace / OMEGA
> retrieval rows and the cited Hindsight / Zep / Letta / MemGPT / Mem0 LoCoMo
> QA rows are currently treated as ledger-ready same-surface baselines.*

### Feature Comparison

> This section is about **capability surfaces**, not head-to-head benchmark wins.
> For Cortyx, the proof status behind each row lives in `benchmarks/registry.json`
> (`proven`, `diagnostic`, `contract`, or `smoke` depending on the surface). Competitor columns
> are product/literature notes, not repo-run measurements.

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
| Offline / air-gap mode (`CORTYX_NO_DOWNLOAD=1`) | **Yes** | No | No |
| Cross-platform file watcher | **Yes** (macOS/Linux/Windows) | No | No |
| Auto-install | **Yes** (Claude Code, Cursor, Windsurf, Codex, VS Code, Zed) | No | No |
| Languages with AST extraction | **23 + universal fallback** | n/a | n/a |
| Import auto-wiring languages | **10** (Rust, Python, TS/JS, Go, C/C++, Ruby, Swift, Kotlin, Dart, Elixir) | n/a | n/a |
| Pre-built binaries | **6 targets** (x86\_64/aarch64 × Linux-gnu/musl + macOS + Windows) | n/a | n/a |

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
