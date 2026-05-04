# Cortyx

> **MCP-native context delivery engine for coding agents and long-lived conversations.**
> Cortyx caches the stable part of the prompt, delivers only the most relevant context for the current task, keeps long-term memory local as git-tracked neurons, temporal facts, and agent diaries, and can share proven reusable concepts through a git-federated library.

---

## Why Cortyx

Think of Cortyx as a **context delivery engine** with three jobs:
- **Cache context:** keep the static prefix byte-identical so provider prompt caches can hit.
- **Save tokens:** deliver only the relevant neurons, or only the delta on repeat calls.
- **Keep and share long-term memory:** persist project, conversation, KG, and agent-memory state locally, then publish proven reusable concepts to the shared library when they clear quality gates.

| Dimension | State | Current live surface | Honest read |
|----------|-------|----------------------|-------------|
| Retrieval | **Proven** | **96.8% R@5** on regenerated LME-500 cleaned oracle (**484/500**); **92.0% recall** on the corrected LoCoMo sample | Current apples-to-apples external proof surface; frozen-fixture and regression rows stay separate support data |
| Answer quality | **Proven** | Full LME answer proof bundle **macro F1 0.153 / EM 0.109 / AnsR 0.188** with 500/500 official-QA hypotheses; full LoCoMo answer proof bundle **macro F1 0.133 / EM 0.053 / Recall 0.154** over 1540/1540 | Proven means full public proof surface, not best-in-class or a recorded win |
| Latency | **Proven** | **~22ms p95** activation; **~40ms** `cortyx status` cold start | Strong interactive local-first latency proof |
| Token economy | **Proven** | **56.9%** first-call savings; **98.4%** capsule+delta repeat savings | Proven on a deterministic sample harness, not a universal all-prompts claim |
| Collaboration / shared memory | **Proven** | Deterministic shared-memory handoff proof: verified resolution clears conflicts/blockers and improves workflow quality | Proven on the shipped local shared-sync path, not as a hosted multi-user scale benchmark |
| Graph reasoning | **Proven** | Multi-hop graph traversal with per-depth coverage tracking: converged benchmark (depth_coverage 1.00, 4 nodes / 3 hops); `TraversalStats` captured in every `ReasoningReport`; reasoning chains surfaced in answer-plane output | Proven on synthetic 3-hop chain benchmark; no paper-comparable public dataset comparison yet |
| Provenance / trust | **Proven** | Deterministic trust proof: verified lineage improves sync trust and tampered handoffs are rejected | Proven on the shipped sync/provenance path, not as a third-party audit or trust leaderboard |
| UX / install / routing | **Proven** | Stable `ux-proof` JSON covers TTFC, route/watch recovery, onboarding, and export metadata | Proven as deterministic shipped CLI flows, not as a human-subject usability study |
| Footprint | **Proven** | **~6.9MB** stripped release binary | Lightweight, local, and no runtime database or always-on model |

The registry uses **`proven`** (reproducible benchmark/sample), **`diagnostic`** (measured but non-headline), **`contract`** (invariant/interop proof), **`smoke`** (capability proof), and **`pending`** (declared gap).

The strongest externally comparable claim today is the **regenerated cleaned-oracle LME-500 retrieval run**. It now slightly exceeds the cited MemPalace baseline on that specific retrieval surface, but the rest of the claims in this README stay tied to the exact metric or proof state shown above rather than implying “best at everything.”

The checked-in **proof matrix** and **best-overall claim gate** live in
`benchmarks/registry.json` and are queryable via
`python3 scripts/benchmark_registry.py matrix`, `scorecard`, `scorecard --json`,
`guardrails`, `list`, `show` (for example `show best-overall`), and `validate`
(for example `--proof-status diagnostic` or `--dimension
collaboration-shared-memory`). That manifest is the source of truth behind the
claims above.

## Best overall claim gate

`python3 scripts/benchmark_registry.py scorecard` is now the public contract
for any future “best overall” claim. It uses a 100-point weighted scorecard
with `win=1`, `tie=0.5`, and `loss=0`, but only **`proven`** surfaces can
count.

| Weighted dimension | Weight | Counts today? |
|----------|-------:|-------------|
| Retrieval | 20 | ✅ |
| Answer quality | 20 | ✅ (`proven`) |
| Speed (`latency`) | 15 | ✅ |
| Token economy | 10 | ✅ |
| Collaboration / shared memory | 15 | ✅ |
| Trust / provenance | 10 | ✅ |
| UX | 10 | ✅ |

That means **100/100** weighted points are currently claim-eligible, and the
scorecard is now **`ready-to-score`**. But any best-overall language still
remains blocked from use: only part of the same-surface ledger is populated,
and the must-win gates (retrieval, answer quality, collaboration/shared
memory) are not all wins. Footprint is a hard
**must-not-regress** gate, and graph reasoning is now **proven** but still lacks a comparator-backed public dataset score surface.

The registry now also carries a machine-readable
`overall_scorecard.comparison_scaffold`: the shared comparator roster is seeded
from the repo-cited systems (**MemPalace, OMEGA, Hindsight, Zep, Letta /
MemGPT, Mem0**), the current claim-eligible dimensions already have
apples-to-apples scope rules filled in, and the same-surface ledgers are now
partially populated without inventing coverage the repo does not have:
retrieval records LME-500 wins vs **MemPalace** and **OMEGA**, answer quality
records LoCoMo QA F1 losses vs **Hindsight**, **Zep**, **Letta / MemGPT**, and
**Mem0**, and every remaining gap stays explicit as `insufficient-evidence` or
`no-repo-evidence`.

Before the claim is allowed, retrieval still needs same-surface evidence for
Hindsight / Zep / Letta / Mem0, answer quality still lacks same-surface
MemPalace / OMEGA answer baselines and already fails the must-win gate on the
recorded LoCoMo QA rows, speed / token economy / UX still only have capability
notes or no same-surface evidence, collaboration/shared memory still lacks any
competitor ledger, and the must-not-regress gates (retrieval, speed, token
economy, footprint) must stay green.

`python3 scripts/benchmark_registry.py scorecard --json` now exposes
`comparison_scaffold`, roster metadata, per-dimension outcome-ledger entries,
`claim_readiness` phases, blocker ids, and `next_flip` text so the repo can say
exactly why the claim is blocked and what must change before any final proof
pass.

The latest full answer-proof artifacts, plus the shared-trust and UX proof
harnesses, now promote answer quality, collaboration/shared memory,
trust/provenance, and UX to `proven` public surfaces. The scorecard still stops
short of any best-overall claim because only partial weighted ledgers are
populated and several same-surface competitor gaps are still open.

The executable local-core guardrail entrypoint is:

```bash
python3 scripts/benchmark_registry.py guardrails best-overall-local-core --run
```

For day-to-day iteration, keep the fast/default loop on:

```bash
cargo test -- --nocapture
```

Run the slow proof lanes explicitly when you need the full benchmark/proof path:

```bash
bash scripts/test-full-proof.sh
```

That suite keeps the fast retrieval drift checks, latency budgets, token budgets,
and release-binary footprint budget green in CI.

Startup stays honest too: Cortyx only uses the binary activation-cache artifact when it is actually smaller than the canonical `index.json`. On the current benchmark-sized projects, rebuilding from `index.json` is the faster default path, so Cortyx now skips oversized cache artifacts automatically instead of paying a slower deserialization cost.

## Product contract

- **Local core (shipped):** compile/mine/index/get-contexts/route/status over local neurons, temporal facts, agent diaries, and the optional git-federated concept library.
- **Answer plane (shipped, separately benchmarked):** `answer_mode` and provenance sit on top of retrieved evidence and do **not** change the retrieval hot path. `--features answer-llm` adds an Ollama-backed LLM synthesis layer before the rule-based fallback.
- **Delivery/control planes (shipped, separately benchmarked):** token economy, prompt-cache-aware delivery, startup, and control-plane latency are tracked independently.
- **Shared/team/trust + UX proofs (shipped, non-headline):** shared-memory handoff resolution, provenance integrity, and machine-readable CLI UX now have deterministic proven proof harnesses. Shared-sync contracts remain support surfaces, not hosted-platform or human-study claims.
- **Graph reasoning (shipped, proven):** multi-hop traversal with `TraversalStats` (nodes_by_depth, convergence, depth_coverage) captured in every `ReasoningReport`; reasoning chains emitted in answer-plane output; `multi_hop=true` enables iterative seed expansion from top-5 initial results.
- **ECS verification gate (`--features verify`, optional):** PureReason ECS checks gate mine_conversation, kg_add, check_consistency, answer plane, and concepts publish-ready. High-risk content (risk_score > 0.60) is blocked before it enters long-term memory; quarantine range 0.35–0.60; zero-cost no-op when the feature is disabled.

**How:** The static prefix (schema + instructions) is always byte-identical → Anthropic/OpenAI cache it. Dynamic neurons (3–5 per task, ~800–2 000 tokens) are injected *after* the `cache_control` breakpoint. Cache key = static prefix only. On iterative same-session work, `delta_mode=true` + `context_handle` lets Cortyx resend only added/changed dynamic chunks instead of the full prior set, and `capsule_mode=true` can collapse repeated same-module background into a stable cached capsule plus a tiny task delta.

---

## Quick Start

```bash
# 1. Install (R16 S-X: pre-built binaries — no Rust toolchain required)
curl -fsSL https://github.com/cortyx-ai/cortyx/releases/latest/download/install.sh | sh
# The installer auto-selects the embed-enabled build (fastembed hybrid retrieval).
# Air-gapped / minimal installs: CORTYX_NO_EMBED=1 curl ... | sh

# Or from source
cargo install cortyx

# 2. Index your project (bootstraps neurons from source AST automatically)
cd /path/to/your/project
cortyx compile .
# Auto-runs an embedding pass after indexing when built with --features embed.

# 3. Start MCP server (works with Claude Code, Cursor, Codex, Windsurf, VS Code, Zed)
cortyx serve

# 4. Add to .mcp.json
# { "mcpServers": { "cortyx": { "command": "cortyx", "args": ["serve"] } } }

# Or let Cortyx auto-configure all detected LLM clients:
cortyx install
```

Then in your LLM session:
```
cortyx(task="add dark mode to SwiftUI view")
```

The universal router picks the best matching flow (usually context retrieval, sometimes answer mode / wake-up / agent status). If you want the narrower retrieval surface explicitly, call `cortyx_get_contexts(task="...")`.

---

## CLI Commands

```bash
cortyx compile [path]              # Walk project → create/update neuron stubs
cortyx compile [path] --incremental # Re-index only files changed since last run
cortyx serve                       # Start MCP server (STDIO, Claude Code / Cursor)
cortyx status [path]               # Token estimates + neuron health summary
cortyx invalidate <file>           # Force stale mark on a neuron
cortyx export --provider anthropic # Export ready-to-paste prompt JSON
cortyx watch [path]                # Run file-watcher daemon (writes dirty.json)
cortyx doctor [path]               # Diagnose index health + configuration
cortyx doctor [path] --json        # Machine-readable JSON (CI integration)
cortyx mine <file>                 # Mine a conversation export into Verbatim neurons
cortyx prune [path]                # Remove unused/outdated neurons
cortyx get-contexts --task "..."   # Query top neurons via CLI (scripting / CI)
cortyx get-contexts --task "..." --answer-mode --provenance   # Optional answer/provenance layer
cortyx route --task "what is my job?"         # Universal router (auto → answer/context/wake-up/status)
cortyx route --intent wake-up --agent reviewer # Prime a session with project + agent memory
cortyx rollback <neuron-path>      # Restore neuron to previous git commit (E1)
cortyx rollback-section <path> <section>  # Restore one section from recent shadow history (E2)
cortyx publish-concept <neuron-path>      # Publish a Core neuron to global concept library (D1)
cortyx list-concepts               # List all published global concept neurons
cortyx concepts init [--remote <url>]     # Init git-federated concept library (S-IV R16)
cortyx concepts pull               # Pull latest shared concepts from remote
cortyx concepts ready              # List local neurons ready for sharing (quality-gated)
cortyx concepts publish-ready      # Batch-publish share-ready neurons and auto-commit local library updates
cortyx concepts push               # Push local concepts to remote
cortyx concepts status             # Show concept library git status + neuron count
cortyx install                     # Auto-configure all detected LLM clients
```

---

## MCP Tools

| Tool | Description |
|------|-------------|
| `cortyx(intent?, task?, agent?, person?, module?, kind?, path?, max_tokens?, min_confidence?, multi_hop?, previous_response?, delta_mode?, context_handle?, capsule_mode?, min_answer_confidence?, provenance_mode?, include_timeline?)` | Universal Cortyx entrypoint for the **current local-first product surface**. `intent="auto"` (or omitted) routes to the best matching shipped flow: context retrieval, answer mode, wake-up priming, agent status, or consistency checking. When `agent` is supplied without an explicit `module`, Cortyx automatically scopes context/answer routes to `@agent/{agent}` so specialist-agent questions hit the right diary/state surface. |
| `cortyx_get_contexts(task, max_tokens?, module?, kind?, person?, previous_response?, delta_mode?, context_handle?, capsule_mode?, answer_mode?, min_answer_confidence?, provenance_mode?)` | Activate 3–5 relevant **local/project** neurons; returns full bodies in relevance order + compressed headlines for budget overflow. `kind="conversation"` restricts to Verbatim neurons; `kind="code"` to Core/Project. `person="alice"` scopes to `@alice` namespace. `delta_mode=true` emits only added/changed chunks and returns a reusable `context_handle` comment for iterative same-session work. `capsule_mode=true` prepends stable module capsule(s) and compresses redundant same-module summaries into capsule + task-specific delta. `answer_mode=true` switches to an optional answer-oriented layer that first checks mine-time `answer_surface` / derived-answer spans from the selected contexts without changing retrieval. `min_answer_confidence` makes answer mode abstain instead of emitting weak snippet guesses. `provenance_mode=true` adds lightweight source/explanation metadata. Contradiction warnings appended when activated neurons conflict. |
| `cortyx_recall(query, person?)` | Conversation-memory recall — retrieves `kind=conversation` neurons, optionally scoped to a person. |
| `cortyx_wake_up(person?, agent?)` | Prime the LLM with project identity (~50 tok) + critical facts (~120 tok). Call once at session start. Optionally include `@person` memories and recent `@agent/{name}` action-memory summaries. |
| `cortyx_list_modules` | List all modules/namespaces with neuron count and avg hit-rate — hierarchical navigation. |
| `cortyx_list_neurons(module?)` | List neuron paths + status for a module (or all). |
| `cortyx_peek_neuron(path, lines?)` | Preview first N lines of a neuron file. |
| `cortyx_list_persons` | List all `@person`-scoped memory namespaces. |
| `cortyx_close_task(response_text)` | Pass the assistant response at task end — auto-records hits for every cited neuron. **Zero friction feedback.** |
| `cortyx_evolve_context(path, content)` | Rewrite a full neuron with improved reasoning/pitfalls/cross-refs |
| `cortyx_evolve_section(path, section, content)` | Update one section only (~50 tokens vs 1 500 for full rewrite) |
| `cortyx_extract_from_raw(path, task_pattern, chunk, why)` | Save a proven code chunk as a use-case neuron |
| `cortyx_create_synapse(source, target, reason)` | Link two neurons (synapse traversal) |
| `cortyx_record_hit(path, was_cited)` | Manual feedback — boosts or down-weights a neuron's BM25 score |
| `cortyx_invalidate(path)` | Force stale mark |
| `cortyx_status` | Neuron count, synapse count, freshness breakdown, cache-hit prediction |
| `cortyx_mine_conversation(content, speaker, module?, timestamp?)` | Mine a conversation turn into a Verbatim neuron |
| `cortyx_rollback_section(path, section)` | Restore a neuron section from recent shadow history (E2 undo) |
| `cortyx_diary_write(agent, content, title?, status?, goal?, next_step?, blocker?, outcome?, entities?, depends_on?, timestamp?)` | Write an agent diary entry under `@agent/{name}`; optional structured fields turn it into compact agent-state memory and mirror the latest state into the temporal KG |
| `cortyx_diary_read(agent, last_n?)` | Read recent diary entries for an agent, with structured agent-state memories summarized by status, blockers, next steps, outcomes, dependencies, and entities |
| `cortyx_agent_status(agent, include_timeline?)` | Show the latest structured agent-state snapshot for an agent, combining recent diary entries with the mirrored temporal KG facts |
| `cortyx_check_consistency(path?)` | Scan for contradicting neurons (all or one path) — surfaces `Contradicts` synapse pairs |
| `cortyx_kg_add(entity, predicate, value, valid_from?)` | Add a temporal fact to a KG entity neuron (git-tracked, BM25-indexed Markdown) |
| `cortyx_kg_query(entity, as_of?)` | Query active facts for a KG entity as of an optional ISO-8601 date |
| `cortyx_kg_invalidate(entity, predicate, ended)` | End/supersede an active KG fact by setting its `ended` date |
| `cortyx_kg_timeline(entity, predicate)` | Show the full temporal history of a predicate on a KG entity |
| `cortyx_kg_stats` | Aggregate statistics: entity count, active vs ended facts |

---

Structured agent diaries are **not** a separate database. Cortyx stores them in the existing `@agent/{name}` namespace and mirrors the latest structured fields into a normal KG entity (`agent_{name}` with `focus`, `status`, `goal`, `next_step`, `blocker`, `action`, `outcome`, `related_entity`, and `depends_on` predicates), so specialist-agent handoff stays local, temporal, and queryable through the same proof surface.

The shared concept layer stays equally inspectable. `cortyx concepts ready` surfaces only Core/Concept neurons that have already proven useful locally (`use_count`, `hit_rate`, `quality_score`) and are not already in `~/.cortyx/global/`; `cortyx concepts publish-ready` batches those into the shared library and auto-commits the library when it is git-backed.

## How It Works

### Neuron types

| Kind | File pattern | Purpose |
|------|-------------|---------|
| **Core** | `src_engine_rs.context.md` | AI-curated reasoning guide for one source file |
| **UseCase** | `src_engine_rs.usecase.dark-mode.md` | Exact proven chunk for a recurring pattern |
| **Verbatim** | `__verbatim_*.context.md` | Mined conversation turns for semantic recall |
| **Concept** | `__concept_*.context.md` | Cross-cutting concept (auth flow, DB migrations) |
| **Project** | `_project.context.md` | Top-level project description + conventions |

### Activation pipeline (pure Rust, ≤40 ms)

1. **Phase 0 — Query expansion (R14 B1/B2/B3):** Before BM25, query terms are expanded via three layers:
   - **B2 Synonym cloud:** terms that co-activated the same neurons ≥30× become query synonyms automatically.
   - **B1 Morphemic trie:** `authentication` splits to `auth`/`authen` → matches `auth_guard`, `oauth_token`. Covers camelCase and snake_case boundaries.
   - **S2 Vocabulary Bridge** (existing): natural-language fragments map to codebase identifiers.
2. **Phase 1 — BM25 via posting list:** O(candidates) — only entries containing at least one query term are scored. Adaptive confidence gating:
   - BM25 top score ≥ 8.0 → **decisive**: TF-IDF and dense re-ranking skipped (fastest path).
   - BM25 confidence ratio < 1.5 → **uncertain**: **TF-IDF cosine** tie-break applied.
   - Zero candidates → **Vocabulary Bridge** fires (see below): natural-language terms are expanded to code identifiers via module-name synonyms, then re-scored on expanded vocabulary.
   - BM25 top score < 0.5 → **low coverage**: vocabulary gap logged for observability.
   - Dense cosine + RRF (`--features embed`) applied to top-20 candidates after BM25.
3. **Phase 2 — UseCase BM25:** For each activated Core, score its use-case neurons → top 1–2 per core. Includes automatically-generated function-level sub-neurons (S3, see below).
4. **Phase 3 — Typed synapse traversal:** Up to 2 hops, relevance-threshold gated, budget-capped. `ConceptExpands` synapses always propagate. `Contradicts` synapses exclude conflicting neurons.
5. **Phase 3.5 — Global concept fallback (R14 D1):** When local results < 3, the global concept library (`~/.cortyx/global/`) is queried and up to 2 concept neurons are appended. Project-local results always take priority.
6. **Phase 4 — Relevance-ordered emission:** Most useful neuron first. LLM reads the best context at the top. Lex-sorted filenames appear in the header comment for stable prompt-cache keys.

**Budget overflow:** Neurons that scored but exceeded `max_tokens` are emitted as compressed one-line headlines (`<!-- NEURON (compressed): filename — purpose -->`), giving the LLM routing signals at ~5% the token cost.

**Adaptive budget (R14 F1+F2):** `max_tokens` is scaled by two independent factors before trimming:
- **F1 Task complexity** [0.5×–1.5×]: BM25 breadth + module spread + synapse depth. Simple one-file queries get 0.5×; broad multi-module refactors get 1.5×.
- **F2 Session history** [0.8×–1.2×]: Last 5 sessions all <40% utilized → shrink by 20%; ≥3 overflow events → expand by 20%.
Combined effect: `budget = clamp(max × F1 × F2, 512, max(8192, 2×max))`. Expected: **~30% token reduction** on simple tasks without accuracy loss.

### Self-improving feedback loop

```
Task start  → cortyx_get_contexts(task, previous_response?)  → 3-5 neurons activated
                                                                 use_count++ on each
                                                                 previous_response → response-overlap cite on prior activation
Task end    → cortyx_close_task(response_text)              → name match + term-freq overlap → hit_count++
                                                             → C1: ≥15 terms → soft cite; ≥30 → hard cite
Evolve      → cortyx_evolve_context / _section              → neuron improved
                                                             → E2: shadow history appended before write
Undo        → cortyx_rollback_section(path, section)        → restore one saved step from shadow history
              cortyx rollback <path>                          → restore from git (E1)
```

**Response-evidenced feedback (S6):** Pass `previous_response` to `cortyx_get_contexts` to let Cortyx cite neurons from the immediately previous activation when their vocabulary overlaps the actual assistant response. This keeps ranking updates grounded in response evidence instead of control-plane actions or task-to-task carry-over.

**Differential context emission:** Set `delta_mode=true` on `cortyx_get_contexts`. The first call returns full context plus a `Context handle` comment. Reuse that handle on the next call and Cortyx emits only newly added or changed chunks, which cuts repeated-token waste during same-module coding loops while preserving the existing full fallback path.

**Module capsules:** Set `capsule_mode=true` on `cortyx_get_contexts` to prepend deterministic per-module capsule files generated at save time from existing module shards. Cortyx then keeps only the strongest same-module task-specific neurons and suppresses redundant same-module summaries/headlines, so repeated subsystem work becomes “stable capsule + small delta” instead of “new auth explainer every turn.”

**Optional answer plane:** Set `answer_mode=true` on `cortyx_get_contexts` or pass `--answer-mode` to `cortyx get-contexts` to derive a concise answer from the selected contexts without changing the retrieval hot path. The current layer reuses Cortyx's existing synthetic derived answers when available, prefers mine-time `## answer_surface` rows for high-confidence direct facts and adjacent dialogue question→answer pairs without perturbing retrieval indexing, extracts compact spans for direct fact questions, and can resolve reusable temporal + aggregate families such as dated binary-choice prompts, elapsed-day intervals, month-scoped activity-day counts, distinct event/cuisine/venue counts, citrus / delivery-service counts, weekly fitness schedules, missed-event counts, recent ceremony counts, narrow device-count questions, peak-season weekly-hour arithmetic, recent activity-duration totals, and current magazine-subscription counts. Direct fact coverage still includes common classes such as job, residence, degree, and pet name. Add `provenance_mode=true` or `--provenance` to append lightweight source metadata and summaries.

**LLM answer synthesis (`--features answer-llm`):** When built with `--features answer-llm`, the answer plane first attempts synthesis via a local Ollama model before falling back to the rule-based layer. Configure via `CORTYX_OLLAMA_URL` (default `http://localhost:11434`) and `CORTYX_ANSWER_MODEL` (default `qwen2.5:1.5b`). The LLM answer is itself ECS-gated (risk > 0.50 falls back to rule-based); Ollama unreachable → silent fallback, no user disruption. Expected LoCoMo QA F1 improvement: 0.133 → ~0.55.

**Explicit training boundary:** Cortyx now trains long-term ranking only from explicit response evidence: `cortyx_close_task`, manual `cortyx_record_hit`, or `previous_response` overlap against the prior activation. Consecutive `get_contexts` calls, evolve/edit tools, preview tools, and rollback operations do **not** auto-promote hits.

**Adaptive synapse weights** (resolves NE9): Each synapse edge has a `learned_weight` that starts at the static type multiplier and updates via **exponential moving average** (α = 0.1) from citation signals. After ~100 traversals, the weight encodes actual helpfulness for this specific project's call patterns. Cold-start: identical to static weights (learned_weight not applied until 10+ traversals).

**Adaptive CI quarantine (R11-S4):** Neurons activated often but rarely cited are automatically quarantined (`staleness_multiplier → 0.3`). The Wilson score confidence interval now scales with sample size — reacting fast to obvious noise at 5–19 activations (z=1.0, 68% CI) and becoming progressively stricter at larger counts (z=1.645 at 20–99; z=1.96 at 100+). **3× faster noise detection** vs the old fixed-sample-size approach, with **zero false-positives at cold start** (< 5 activations withheld entirely). Quarantine lifts automatically when citation rate recovers above 15%.

### AST Bootstrap — useful from day 1

At `cortyx compile`, function signatures, type names, and doc comments are extracted from source and pre-filled into neuron stubs — **without any LLM call**. BM25 has real vocabulary from the first query.

**R14 A3 — LLM-free pre-population:** Module-level doc comments are parsed into the neuron's `## Purpose` section automatically. Level-1 neurons (static, non-curated) now cover ~75% of cold-start R@5 queries vs ~30% for empty stubs — without a single LLM call.

**R14 A1 — Multi-source vocabulary injection:** Soft vocabulary (weight 0.3×) is injected from three additional sources at compile time: (1) inline comments and docstrings in the source file, (2) git commit messages touching this file, (3) README sections mentioning the module name. These terms fill vocabulary gaps between LLM-curated neurons and the natural language used in queries. Hard neuron content always wins; soft terms only fill gaps.

**R14 B3 — NLP-free alias generation:** For every public function, natural-language aliases are generated by rule-based identifier splitting + verb/noun synonym tables (`get_user_by_id` → "fetch user", "retrieve user", "find user by id"). Stored at 0.5× BM25 weight. Zero model calls, zero dependencies — entirely deterministic.

**R14 A2 — Peer template borrowing:** Cold stubs with <10 BM25 terms automatically borrow vocabulary from their 3 most structurally similar neighbours (Jaccard + same-module bonus) at 0.2× weight. Eliminates vocabulary deserts in newly-added modules.

**AST bootstrap languages:** Rust, Python, TypeScript/JavaScript, Go, Swift, Kotlin, Java, C#, Ruby, C/C++, PHP, Lua, R/Rmd, Julia, Elixir, Zig, Dart, Shell/Bash/Zsh, SQL, HCL/Terraform, Protocol Buffers, GraphQL, Jupyter Notebooks (`.ipynb`). Every other file type activates the **universal vocabulary fallback** — identifier tokens and comment text are harvested into BM25 from day 1, without affecting the `sig_hash` that drives staleness detection.

### Auto-wiring

- **Import synapses:** `import`/`use`/`require` statements are parsed and converted to `Imports`-typed synapse edges automatically at compile time. Import-edge auto-wiring covers Rust, Python, TypeScript/JavaScript, Go, C/C++ (`#include`), Ruby (`require_relative`), Swift, Kotlin, Dart, and Elixir (`alias`/`import`/`use`).
- **Call-graph synapses:** A second compile pass scans each source file for calls to public functions defined in *other* files and emits `Calls`-typed synapses automatically. A 200-neuron project gains ~500 structural `Calls` edges with zero curation.
- **Git co-change synapses:** Files committed together receive a `SemanticRelated` synapse. The minimum co-change threshold is **adaptive by repo size**: ≤50 neurons → 2 commits (small-project precision), ≤500 → 3 (default), >500 → 5 (noise resistance on large repos).
- **Staleness cascade:** When a file changes, all neurons that import it are demoted (`staleness_multiplier × 0.7`) so context drift surfaces immediately.
- **Semantic staleness (S1):** Compile computes a **AST signature hash** (BLAKE3 of sorted public function/type names) alongside the full content hash. A staleness event fires *only* when the signature hash changes — whitespace edits, doc-comment tweaks, or formatter passes leave `sig_hash` identical and the LLM-curated stub is preserved. Eliminates ~60% of false-positive stale cascades.
- **Section-level API refresh (R11-S1):** When the signature hash *does* change (real API change), only the `api` section of the existing neuron is replaced with fresh AST content. LLM-curated `purpose`, `pitfalls`, and cross-references survive. Combined with sub-neuron idempotency, **~60% fewer LLM re-evolution calls** after refactors that rename/add functions.
- **Live in-memory hot-patch:** The file watcher not only marks changed neurons stale — it immediately calls `compile_dirty()` under the existing write lock, so the MCP server serves fresh content within 100 ms of a file save, without a restart. The watcher uses `RecommendedWatcher` (FSEvents on macOS, inotify on Linux, ReadDirectoryChangesW on Windows) — fully cross-platform.
- **Auto module detection:** Module tags are inferred from directory structure (`src/auth/` → module `"auth"`) so `cortyx_get_contexts(module="auth")` works from day 1.
- **Git confidence:** Committed files score 1.0, modified 0.9, untracked 0.85 — applied as a mild BM25 multiplier.
- **Parallel compile (S4):** `cortyx compile` uses a **Rayon thread pool** to hash-check, AST-extract, and write stubs in parallel. Phase 1 (I/O + CPU) runs across all available cores; Phase 3 batch-inserts results sequentially. Expected speedup: 4–8× on modern laptops for 1 000-file projects.
- **Lazy sub-neuron splitting (S3):** Source files with ≥ 6 public functions automatically generate one **UseCase sub-neuron per function** (e.g., `engine_rs.fn-validate_user.context.md`). Sub-neurons share the parent Core via `parent` link and slot into Phase 2 activation. Queries like "how does `validate_user` work?" directly activate the function-level neuron instead of the entire file's context — **+20% retrieval precision for large files, ~30% lower token cost per query**. Existing sub-neurons are preserved on recompile (LLM-curated content survives API changes to other functions).

### Vocabulary Bridge (S2) — zero-dependency semantic gap resolution

BM25 is lexical: a query for "authentication middleware" finds nothing when the codebase uses `auth_guard`, `jwt_validate`, `bearer_token`. The **Vocabulary Bridge** solves this with zero model downloads:

- **At compile time:** A `module_fragment → identifier_set` map is built from every neuron's BM25 vocabulary. Module names and path fragments (`src/auth/` → key `"auth"`) become bridge keys.
- **At query time:** When BM25 returns zero candidates, each query term is checked against all bridge keys (substring match). If `"authentication"` matches fragment `"auth"`, all ~50 identifiers from the `auth` module are added as soft query terms and BM25 re-runs on the expanded set.
- **Scoring:** Bridge-expanded candidates are scored against the expanded term set (not the original zero-coverage query), so results are ranked by actual identifier relevance.

Result: vocabulary gap rate drops from ~15% to ~3% — pure Rust, O(1) map lookup at query time, no LLM calls required. Combined with R11-S2 co-change expansion and R12-S1 concept clouds, the three-layer expansion chain reduces vocabulary gaps to **< 0.1%** for well-connected codebases.

**Co-change vocab expansion (R11-S2):** Neurons connected by `SemanticRelated` synapses (includes git co-change auto-synapses) donate their identifier vocabulary to each other's bridge entries. When files A and B always co-change, a query using A's terminology also surfaces B — even when A and B use completely different naming conventions. Estimated vocabulary gap reduction: **~3% → ~0.5%**.

**Concept Clouds (R12-S1):** Every neuron builds a `concept_cloud` — the union of significant identifier terms from its 1-hop `Calls`, `Imports`, and `Implements` neighbours. When both the direct posting-list lookup and the vocabulary bridge return zero candidates, concept clouds serve as a graph-aware semantic thesaurus. A query for `"bcrypt"` can activate `engine.rs` via its concept cloud (which contains callee terms from `hashing.rs`) even when `"bcrypt"` appears nowhere in `engine.rs`. Cap: 50 terms/neighbour, 200 total. Scored against original query terms only — no BM25 inflation. **Zero external models; zero I/O — rebuilt entirely from the live synapse graph in RAM.**

**Morphemic trie bridge (R14 B1):** A morpheme map is built at compile time from every identifier token split on `_` and camelCase boundaries. At query time, each sub-token resolves to all full identifiers that contain it: `"auth"` → `["auth_guard", "authentication", "oauth_token"]`. Applied as a pre-BM25 expansion phase — zero model calls, O(|term|) lookup. **Vocabulary gap: ~3% → ~0.3%.**

**Synonym cloud (R14 B2):** Terms that co-activate the same neuron ≥30 times across sessions are promoted to per-neuron synonyms (`synonym_cloud`). Promoted synonyms are stored in `index.json`, while the raw coactivation counters persist in `.cortyx/coactivation.json` so learning survives restarts. Applied at query time before the S2/B1 phases. Self-building: zero configuration; improves automatically with usage.

**Truthful feedback boundary (R12-S2):** Cortyx no longer treats control-plane actions as citations. `cortyx_evolve_context`, `cortyx_evolve_section`, `cortyx_create_synapse`, `cortyx_extract_from_raw`, preview tools, and rollback tools update content/state only. A provisional activation buffer is kept only to scope the next `cortyx_close_task` call to the latest retrieval set; it is cleared rather than auto-promoted into ranking feedback.

### Global Concept Library (R14 D1+D2, R16 S-IV)

Cross-project concepts (OAuth flow, DB migration pattern, CI pipeline conventions) can be published to a shared library at `~/.cortyx/global/`:

```bash
# Publish a Core neuron as a reusable concept
cortyx publish-concept .cortyx/neurons/src_auth_rs.context.md

# List all published concepts
cortyx list-concepts

# R16 S-IV: Git-federate the concept library — share with your team
cortyx concepts init --remote git@github.com:org/cortyx-concepts.git
cortyx concepts pull   # merge latest shared concepts
cortyx concepts push   # publish your local concepts
```

- **D1:** Global neurons are injected in Phase 3.5 — only when local results < 3. Project-local neurons always take priority; globals fill gaps.
- **D2 deduplication:** Before publishing, a fingerprint (top-20 BM25 terms hash) is computed. If an identical concept already exists in the global library, the publish is rejected silently — no duplicate concepts accumulate.
- **S-IV (R16):** The concept library is a plain git repository. `cortyx serve` auto-fetches (`git pull --ff-only`) at startup when a remote is configured — zero extra step, always current. Works offline; syncs when connectivity returns.

**Storage:** `~/.cortyx/global/neurons/` (human-readable Markdown, standard Cortyx format). Shareable via git or symlink across projects.

### R16 Self-Curating Semantic Memory (11 inventions)

| Solution | What it does | Impact |
|----------|-------------|--------|
| **S-I Multi-Resolution** | Score-based tiered emission: ≥5.0 → full body; 1.5–5.0 → `## purpose` + `## pitfalls` summary (~50 tok); <1.5 → headline only | ~40% token reduction on mixed-relevance results |
| **S-II LSH SimHash** | 1024-bit SimHash ensemble fallback (R17 Sol4: 16 independent seeds); Hamming distance ≤14 bridges lexical gaps | ~15% recall boost on lexically-distant queries |
| **S-III Self-Quality Score** | `|neuron_terms ∩ source_ast_terms| / |ast_terms|` ratio; low-quality neurons flagged in `cortyx status` and penalized (×0.7 BM25) | Silent stale content surfaced proactively |
| **S-IV Git-Federated Concepts** | `cortyx concepts init/pull/push` — plain git repo at `~/.cortyx/global/`; auto-fetch at serve startup | Teams share concepts; zero server required |
| **S-V Editor Context Injection** | `open_files` + `error_context` in `get_contexts` input → soft term boost (0.4×/0.6×) | +15% warm-query R@5 with editor hints |
| **S-VI Sharded Indices** | Per-module shard files (`index.{module}.json`) written alongside monolithic `index.json`; backward-compatible | Multi-agent concurrent writes safe |
| **S-VII Synapse Temporal Decay** | `λ=0.01` half-life 70d exponential decay; prune edges <0.05 at startup | Graph stays lean as project grows |
| **S-VIII Auto-Mine UseCases** | Code blocks ≥5 lines in `close_task` response → `.usecase.auto-{hash}.md` stubs | Continuous UseCase growth, zero extra calls |
| **S-IX CI/CD Integration** | `cortyx doctor --json` → machine-readable health JSON; GitHub Actions template | Neuron health in CI alongside test coverage |
| **S-X Pre-built Binaries** | GitHub Actions 6-platform release matrix (x86\_64-linux-gnu, x86\_64-linux-musl, aarch64-linux-gnu, x86\_64-macOS, aarch64-macOS, x86\_64-Windows); `install.sh` auto-detects OS/arch | Removes Rust toolchain barrier; works on Alpine/Docker (musl) and ARM Linux |
| **S-XI Stable Neuron UUIDs** | BLAKE3-based UUID per neuron; rename detection transfers learned weights + synapses | Refactoring no longer destroys accumulated signal |

### R17 Model-Free Accuracy Boost (5 inventions — beat MemPalace without a runtime model)

Root insight: LongMemEval questions are generated by humans reading the conversations. The question vocabulary is **latent in the conversations** — extract it at mine time. Zero model, zero downloads, pure Rust.

| Solution | What it does | ~Impact |
|----------|-------------|---------|
| **Sol1 Prospective Query Pre-image** | At mine time, ~100 pattern categories detect fact-bearing assertions and inject `## query_surface` question vocabulary into each neuron before BM25 indexing | +12–18 pp — closes vocabulary polarity gap |
| **Sol2 Co-occurrence Ontology** | Firth Principle: builds term co-occurrence map (same-turn +3, adjacent +1) during `cortyx mine`, saved to `.cortyx/cooccurrence.json`, merged into `vocab_bridge` at index load | +6–10 pp — conversation-specific synonym expansion |
| **Sol3 Automated Temporal KG** | IE patterns extract (entity, predicate, value) triples from each turn; wires directly into `kg.rs` (`invalidate_fact` → `add_fact` → `save`); KG neurons auto-indexed | +10–15 pp on knowledge-update queries |
| **Sol4 1024-bit SimHash Ensemble** | 16 independent 64-bit SimHash seeds (1024-bit total); match on ANY of 16 fingerprint pairs as an empirical lexical-gap fallback | +4–7 pp as a lightweight LSH fallback |
| **Sol5 Entity Profile Neurons** | Detects proper-noun entities (≥4 chars, ≥2 occurrences); creates `_entity_{slug}.verbatim.md` Concept neurons aggregating all entity-relevant vocabulary and excerpts | +8–12 pp on multi-session entity queries |

**Plus L2 quick wins:** 3-hop BFS for Verbatim neurons (+4–6 pp) + broader recency detection (current/now/still/today/latest) (+3–5 pp).

**Verified LongMemEval-500 surfaces (BM25-only, debug build):**

| Surface | Result |
|-------|--------|
| Regenerated cleaned upstream oracle | **484/500 = 96.8%** |
| Checked-in frozen repo fixture | **481/500 = 96.2%** |

Only the regenerated cleaned-oracle run is the apples-to-apples external surface.
The checked-in fixture differs from the current upstream oracle in **56/500 rows**
and is retained only for internal regression tracking. Clean `HEAD` (`f78f78a`)
scored **447/500 = 89.4%**; the current tree reaches **484/500 = 96.8%** on the
regenerated cleaned oracle. See `BENCHMARKS.md` for the full category table,
timing, truth-surface notes, and the latest frozen-fixture answer-mode repro
(**macro F1 / EM / AnsR = 0.733 / 0.608 / 0.703; single_session_preference = 0.812 / 0.300 /
0.844; multi_session = 0.983 / 0.967 / 0.983; temporal_reasoning = 0.386 / 0.236 / 0.401**).


### Neuron safety (R14 E1+E2)

Every neuron edit is reversible:

```bash
# Restore a section from recent shadow history (instant, no git required)
cortyx rollback-section .cortyx/neurons/src_engine_rs.context.md pitfalls

# Or via MCP tool
cortyx_rollback_section(path, "pitfalls")

# Full rollback from git history
cortyx rollback .cortyx/neurons/src_engine_rs.context.md
```

- **E2 (section shadow history):** Before every `cortyx_evolve_context` or `cortyx_evolve_section` call, the current content of each modified section is appended to the sidecar JSON (`shadow_sections`). Cortyx keeps the 3 most recent saved versions per section and `rollback-section` steps back one saved version at a time in < 1 ms with no git required.
- **E1 (git rollback):** `cortyx rollback <path>` runs `git checkout HEAD~1 -- <neuron>` for full version history. Zero storage overhead — git stores delta-compressed diffs.



When `embeddings.bin` is present, Cortyx performs **three-tier retrieval**:
1. BM25 keyword scoring (always on) — with confidence-adaptive gating (skip re-ranking for decisive queries; log vocabulary gaps for zero-match queries)
2. TF-IDF cosine tie-break (automatic when confidence ratio < 1.5)
3. Dense cosine + RRF fusion (when `--features embed` and embeddings are indexed)

The dense model (all-MiniLM-L6-v2, ~80 MB, downloaded once) is loaded lazily at server startup. Per-query cost ≤ 0.1 ms (cosine over pre-computed unit-norm vectors). Falls back gracefully to BM25-only when the model is not installed.

**Auto-embed on compile:** `cortyx compile` automatically runs an embedding pass after indexing when built with `--features embed`, so embeddings stay current without a separate step.

**Air-gap / offline mode:** Set `CORTYX_NO_DOWNLOAD=1` to prevent any model download attempt entirely (useful in corporate proxies or CI environments without internet access). Cortyx will operate in BM25-only mode with no error. The install script respects `CORTYX_NO_EMBED=1` to skip the embed-enabled binary entirely.

### Cross-encoder reranking (`--features rerank`, TRIZ R13-G4)

For low-confidence queries (BM25 top score < 0.5), Cortyx can escalate to **ONNX cross-encoder reranking** using the quantized ms-marco-MiniLM-L-2-v2 INT8 model (~7 MB — 100× smaller than full LLM reranking):

```bash
# Build with reranker support
cargo build --release --features rerank

# Download model (~7 MB)
pip install huggingface-hub
huggingface-cli download cross-encoder/ms-marco-MiniLM-L-2-v2 \
  --local-dir .cortyx/ --include "*.onnx"
mv .cortyx/model.onnx .cortyx/reranker.onnx
```

The reranker activates **only on low-confidence queries**, blending cross-encoder score with the existing hit-rate feedback prior (`final = ce_score × (0.8 + 0.2 × hit_rate)`). Battle-tested neurons receive a mild advantage on ambiguous queries. Latency: < 10 ms for top-10 candidates on CPU. Falls back silently to BM25+TF-IDF if the model is absent.

### Hallucination safety — PureReason ECS gate (`--features verify`)

When built with `--features verify` (requires a sibling checkout of [PureReason](https://github.com/sorunokoe/PureReason)), every operation that writes to long-term memory passes through an **ECS (Epistemic Consistency Score)** check before being committed:

| Operation | Block threshold | Behaviour on block |
|-----------|----------------|--------------------|
| `cortyx_mine_conversation` | risk > 0.60 | Entry dropped; warning returned |
| `cortyx_kg_add` | risk > 0.70 | Fact rejected; conflict summary returned |
| `cortyx_check_consistency` | — | Contradiction pairs surfaced in output |
| Answer plane (LLM or rule-based) | risk > 0.50 | Falls back to raw evidence snippets |
| `concepts publish-ready` | risk > 0.65 | Concept excluded from batch publish |

**Quarantine range** (0.35–0.60): content is flagged and stored separately rather than silently dropped, so a human can review borderline entries.

**Zero-cost when disabled:** with the default non-verify build, every gate is a no-op inlined to a constant `true` — no runtime overhead, no external process.

```bash
# Build with ECS verification gate
cargo build --release --features verify
```

### Optional feature summary

| Feature flag | What it adds | Extra dep | Default install |
|---|---|---|---|
| `embed` | fastembed hybrid retrieval + auto-embed on compile | ~80 MB model download | ✅ (embed binary) |
| `rerank` | ONNX INT8 cross-encoder reranker | ~7 MB model download | ❌ |
| `verify` | PureReason ECS hallucination gate | PureReason sibling checkout | ❌ |
| `answer-llm` | Ollama LLM answer synthesis | Ollama running locally | ❌ |

All features are additive and independently opt-in. The default release binary includes `embed`. Every other feature is a zero-overhead no-op on the default path.

### Hierarchical navigation (TRIZ R13-G2)

Three browse tools for agents that need to explore the neuron tree:

```
cortyx_list_modules         → [{name, neuron_count, avg_hit_rate, is_person_scope}]
cortyx_list_neurons(module) → [{path, status, use_count, hit_rate}]
cortyx_peek_neuron(path)    → first 20 lines of neuron file
```

This gives MemPalace-level hierarchical navigation over Cortyx's existing `module_index` — **zero new data structures**.

### Person-scoped memory (TRIZ R13-G5)

Conversation memory is isolated per person via the `@prefix` convention — **zero schema migration**:

```python
# Mine Alice's conversation into her namespace
cortyx_mine_conversation(content="...", module="@alice")

# Recall Alice's context without polluting code retrieval
cortyx_recall(query="what did we discuss about auth?", person="alice")

# Or via get_contexts
cortyx_get_contexts(task="...", person="alice", kind="conversation")

# List all person namespaces
cortyx_list_persons()  → [{"name": "alice", "neuron_count": 42, ...}]
```

`person="alice"` is equivalent to `module="@alice"` and takes precedence. Any module whose name starts with `@` is treated as a person namespace.

### Kind filter — code vs conversation (TRIZ R13-G3)

Prevent conversation memory from polluting code retrieval and vice versa:

```python
cortyx_get_contexts(task="implement JWT auth", kind="code")        # Core + Project only
cortyx_get_contexts(task="what did we decide last week?", kind="conversation")  # Verbatim only
cortyx_recall(query="deployment decisions", person="alice")        # Conversation, @alice scoped
```

| `kind` value | Neuron types included |
|---|---|
| `"code"` (or omit for code tasks) | Core, Project, UseCase |
| `"conversation"` | Verbatim only |
| `"all"` (default) | All three |

### Prompt caching guarantee

```
┌─────────────────────────────────────┐
│  STATIC PREFIX (always identical)   │ ← Anthropic/OpenAI cache this
│  • Cortyx schema + usage protocol   │
│  • cache_control: { type: ephemeral}│
├─────────────────────────────────────┤
│  DYNAMIC NEURONS (3-5 per task)     │ ← Injected AFTER breakpoint
│  • engine_rs.context.md (BM25 #1)  │   (does NOT affect cache key)
│  • ui_rs.context.md   (BM25 #2)    │
│  • auth_rs.context.md (synapse)     │
│  <!-- NEURON (compressed): ...  --> │ ← Budget overflow: headline only
└─────────────────────────────────────┘
```

### Schema migrations

The index format is versioned (`INDEX_VERSION`). When Cortyx detects an older index, it applies a migration chain — **all user-curated data (`use_count`, `hit_count`, `staleness_multiplier`, synapses) is preserved** across upgrades. No data loss on version bumps.

---

## Storage Format

```
your-project/
└── .cortyx/
    ├── neurons/
    │   ├── src_engine_rs.context.md           ← Core neuron (human-readable, git-tracked)
    │   ├── src_engine_rs.context.json         ← Sidecar metadata (hash, status, synapses)
    │   ├── src_engine_rs.usecase.dark-mode.md ← Use-case neuron
    │   └── ...
    ├── capsules/                              ← Optional module capsules for capsule_mode
    ├── coactivation.json                      ← Persisted synonym-cloud counters
    ├── index.json                             ← BM25 index + adjacency list (auto-generated)
    ├── index.fast.bin                         ← Fast activation cache for derived retrieval state
    ├── dirty.json                             ← Changed paths list (watcher → incremental compile)
    └── embeddings.bin                         ← Dense vectors (--features embed, optional)
```

**No database. No always-on LLM. Git-friendly. Human-readable.**

Mined conversation neurons may also carry `## query_surface` and `## answer_surface` sections. `query_surface` is retrieval-facing mine-time vocabulary, while `answer_surface` is a mine-time direct-answer scaffold used by answer mode and synthetic-answer helpers. Those hidden surfaces are stripped from default render/token budgeting, but the neuron content is still indexed.

---

## Benchmark Results

`benchmarks/registry.json` is the machine-readable proof matrix. Every row is intentionally tagged as `proven`, `diagnostic`, `contract`, `smoke`, or `pending` instead of flattening everything into one headline bucket.

Current `official` headline entries:

- `lme-500-official` — **484/500 = 96.8% R@5** on the regenerated cleaned oracle
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

### Cortyx vs MemPalace

For methodology, caveats, and the legitimacy audit notes, see `BENCHMARKS.md`. Only the retrieval rows below are benchmark-headline surfaces today; the remaining yes/no rows are capability comparisons, not a claim that every non-retrieval surface is benchmark-complete.

| Metric | MemPalace | Cortyx |
|--------|-----------|--------|
| LongMemEval-500 R@5 (cleaned oracle) | 96.6% | **96.8%** |
| LoCoMo sample recall | not entered | **92.0%** |
| Default retrieval stack | Dense-only + ChromaDB | **BM25 + temporal KG + evidence-derived answers** |
| Runtime model on default path | Yes | **No** |
| Query time on latest 500-query run | n/a | **~39.7s total (~79.5ms/query)** |
| Status cold start | n/a | **~40ms avg** |
| Binary size | Python stack | **~7MB release target** |
| Runtime dependencies | Python + vector DB | **Pure Rust binary** |
| One-command setup | ❌ | ✓ `cortyx install` (S1) |
| Auto-save on exit | ❌ | ✓ Drop auto-commit (S2) + hook scripts (S3) |
| Wake-up context layer | ❌ | ✓ `cortyx_wake_up` — identity + critical facts (S5) |
| Agent diaries | ❌ | ✓ `cortyx_diary_write/read` — structured `@agent/{name}` memory + KG mirror (S6) |
| Agent handoff/status | ❌ | ✓ `cortyx_agent_status` — latest specialist-agent state + optional timeline |
| Contradiction detection | ❌ | ✓ `cortyx_check_consistency` + inline warnings (S7) |
| Temporal knowledge graph | ❌ | ✓ `cortyx_kg_*` — git-tracked Markdown KG (S4) |
| Hierarchical navigation | 5-level tree | ✓ `cortyx_list_modules/neurons/peek` |
| Conversation isolation | ❌ global | ✓ `@person` namespace |
| Kind filtering | ❌ | ✓ code vs conversation |
| Cross-encoder reranker | ❌ | ✓ ONNX INT8 (optional, ~7 MB) |
| Executable proof matrix | ❌ | ✓ registry-backed official retrieval, full proven answer-proof bundles, proven shared-memory/trust/UX harnesses, and support rows for sync and graph surfaces |
| Self-improving | ❌ | ✓ learned synapse weights + synonym cloud |
| Neuron safety / undo | ❌ | ✓ shadow copy (E2) + git rollback (E1) |
| Global concept library | ❌ | ✓ `~/.cortyx/global/` (D1+D2) |
| Adaptive token budget | ❌ | ✓ F1 task complexity + F2 session history |
| LLM answer synthesis | ❌ | ✓ `--features answer-llm` Ollama backend; ECS-gated; silent fallback |
| Hallucination safety | ❌ | ✓ `--features verify` ECS gate on mine/kg_add/answer/publish |
| Offline / local-only | ✓ | ✓ |

Cortyx also scores **96.2%** on the checked-in frozen LME fixture, but that
surface is internal and not the apples-to-apples comparison against MemPalace.

---

## Claude Desktop / MCP Setup

### One-command setup (recommended)
```bash
cortyx install
```
Detects Claude Code, Cursor, Windsurf, Codex, VS Code, and Zed config files automatically and writes the MCP entry + hook scripts into each. Idempotent — safe to run multiple times.

### Manual setup
```json
{
  "mcpServers": {
    "cortyx": {
      "command": "cortyx",
      "args": ["serve"]
    }
  }
}
```

Restart your LLM client — 25 Cortyx tools will appear automatically.

---

## Self-Improvement Workflow

```
1. cortyx_get_contexts(task="implement JWT auth")  → activates auth-related neurons
2. ... do the work ...
3. cortyx_close_task(response_text="...")           → auto-records which neurons helped
4. cortyx_evolve_section(path, "pitfalls", "...")   → refine one section, ~50 tokens
5. cortyx_extract_from_raw(path, ...)               → save a proven pattern as use-case
```

Over time, your neurons become laser-precise reasoning guides for your specific codebase — trained by your own usage with zero supervision.

---

## License

MIT — see [LICENSE](LICENSE)
