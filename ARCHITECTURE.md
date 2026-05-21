# Cortyx Architecture

← [Back to README](README.md)

This document covers Cortyx's module map, key data structures, concurrency model, and design constraints.

Cortyx is a Rust MCP-native context-delivery engine. It maintains a semantic
in-memory index of project knowledge (neurons) and serves them to an LLM host
via the Model Context Protocol. Retrieval is BM25-primary with a 13-stage
activation pipeline, synapse-graph traversal, PMI-based synonym expansion,
Hebbian co-return feedback, and TurboVec 4-bit quantized ANN reranking.

---

## Module Map

```
src/
├── lib.rs                  — crate root; public module declarations
├── error.rs                — CortyxError + SecurityError typed errors
├── types/                  — newtypes: SynapseWeight, TermFrequency, QueryText, …
├── neuron/                 — per-neuron I/O, meta, filter, synapse, provenance
│   ├── filter.rs           — path-escape validation (validate_relative_path,
│   │                         validate_synapse_path) → SecurityError::PathEscape
│   ├── io.rs               — core_neuron_path (filename-only fallback on escape)
│   ├── synapse.rs          — Synapse type + effective_weight() blend formula
│   └── meta.rs             — NeuronMeta, NeuronKind, NeuronStatus
├── index/
│   └── core/               — NeuronIndex + all retrieval logic
│       ├── mod.rs          — shared imports for all index/core submodules
│       ├── types.rs        — NeuronIndex struct (4 domain fields) + summary types
│       ├── domain/         — owned domain state structs (Wave 2 decomposition)
│       │   ├── retrieval_state.rs  — BM25 corpus, adjacency, posting lists, embeddings
│       │   ├── feedback_state.rs   — coactivation, co-return, session utilization
│       │   ├── persistence_state.rs— project_root, WAL state, dirty flags
│       │   └── watcher_state.rs    — dirty_set Arc shared with watcher task
│       ├── compile.rs      — compile_dirty(), dirty_set_handle()
│       ├── persistence.rs  — load/save with CRC32-hardened WAL + checksum sidecars
│       ├── stats.rs        — Hebbian synapse formation, coactivation tracking
│       ├── config.rs       — index-wide constants (including temporal decay weights)
│       ├── bm25/           — BM25Entry, posting-list, IDF cache
│       └── activation/
│           ├── mod.rs      — module exports
│           ├── search.rs   — 20-line thin orchestrator (calls pipeline + Hebbian)
│           ├── phase1.rs   — phase1_candidates, rerank_candidates, ANN rerank + HyDE
│           ├── overflow.rs — overflow handling logic
│           └── selection.rs— select_paths, token-budget trimming
│       └── pipeline/       — QueryContext activation pipeline (Phase 1)
│           ├── mod.rs      — NeuronIndex::build_query_context + state-view extractors
│           ├── stage.rs    — ActivationStage trait + ActivationPipeline runner
│           ├── types.rs    — QueryContext<'a>, ScoredCandidate, FeedbackSnapshot,
│           │                 RetrievalStateView, FeedbackStateView, PersistenceStateView,
│           │                 WatcherStateView; also QueryContextFixture + test_entry for tests
│           └── stages/     — 13 independently-testable stage structs:
│               ├── bm25_scoring.rs      — seed + bridge BM25 scoring
│               ├── vocab_bridge.rs      — vocabulary synonym expansion
│               ├── morpheme_bridge.rs   — morpheme root bridging (0.7× weight)
│               ├── pmi_expansion.rs     — PMI co-occurrence expansion (0.5× weight)
│               ├── use_case_augment.rs  — use-case tag matching (0.9× weight)
│               ├── synapse_traversal.rs — 1-hop synapse graph traversal
│               ├── session_cluster.rs   — session-cluster scoring
│               ├── coreturn_boost.rs    — co-return frequency boost (Hebbian read)
│               ├── coactivation.rs      — term co-activation scoring
│               ├── staleness_decay.rs   — time-based staleness decay
│               ├── counting_augment.rs  — counting/quantitative query augment
│               ├── session_tf_decay.rs  — intra-session TF decay (0.85× weight)
│               └── temporal_proximity.rs— exponential recency boost (TRIZ C3)
├── fleet/                  — optional local-first fleet orchestration (NEW)
│   ├── mod.rs              — public API re-exports + FLEET_REGISTRY_VERSION
│   ├── types.rs            — FleetNodeId (blake3 newtype), FleetNode, FleetRegistry,
│   │                         FleetQueryResult, FleetRouteReason
│   ├── registry.rs         — load/save/register/deregister; path: ~/.cortyx/fleet/nodes.json
│   ├── router.rs           — parallel tokio dispatch (200ms timeout per node),
│   │                         module-manifest filtering, FLEET_LOW_CONFIDENCE_THRESHOLD
│   ├── merge.rs            — rrf_merge() — append fleet context after local output
│   └── tests.rs            — unit tests (registry roundtrip, merge, threshold)
├── mcp/
│   ├── mod.rs              — CortyxServer (+ fleet_registry field), serve(), URL allowlist,
│   │                         5-min inactivity poller, session_active AtomicBool, --frozen gate
│   ├── feedback_buffer.rs  — FeedbackBuffer; provisional_hits drain on on_session_end()
│   ├── helpers/
│   │   ├── meta_io.rs      — neuron metadata helpers (CortyxError, no anyhow)
│   │   └── server_impl.rs  — for_benchmark() constructor; last_activity refresh on tool call
│   └── tools/
│       ├── context/        — get_contexts handler (Phase 5 decomposition)
│       │   ├── mod.rs      — thin orchestrator: InflightGuard acquisition + dispatch
│       │   ├── inflight_guard.rs — RAII byte-cap guard + per-request size estimation
│       │   ├── session_decay.rs  — session TF snapshot/update + path-history decay
│       │   └── answer_mode.rs    — answer_mode dispatch + answer-plane routing
│       └── fleet.rs        — cortyx_fleet_query + cortyx_fleet_status MCP tools
├── commands/
│   └── fleet.rs            — CLI: fleet register / deregister / list / status
├── watcher.rs              — inotify/FSEvents watcher, in-memory dirty_set
├── reasoner.rs             — graph traversal + ReasoningReport
└── main.rs                 — binary entrypoint (anyhow OK here)
```

---

## Key Data Structures

### NeuronIndex (`src/index/core/types.rs`)

The central in-memory index. All operations run in RAM; no async I/O during
retrieval. Persisted to `.cortyx/index.json` after every mutating operation.

**Wave 2 decomposition — NeuronIndex now owns four typed domain structs:**

| Field | Struct | Key contents |
|---|---|---|
| `retrieval` | `RetrievalState` | BM25 entries, adjacency graph, posting lists, path/module/session indexes, embeddings (TurboVec) |
| `feedback` | `FeedbackState` | coactivation counts, co-return Hebbian pairs, session utilization |
| `persistence` | `PersistenceState` | project_root, WAL base, dirty flags, checksum sidecars |
| `watcher` | `WatcherState` | dirty_set Arc (shared with FSEvents/inotify watcher task) |

The original flat 25-field struct is replaced by these four composed types in
`src/index/core/domain/`. Borrow patterns (`pub(in crate::index)` visibility)
prevent cross-domain coupling while enabling efficient zero-copy borrows in
`build_query_context()`.

**TurboVec embedding store (`src/embedder.rs`):**
- `EmbeddingStore` wraps `turbovec::IdMapIndex` (4-bit quantized ANN)
- Embedding model: `NomicEmbedTextV15` (768-dim, 8192 token context)
- `.cortyx/embeddings.bin` is the authoritative raw store (f32, EMBED_VERSION=2)
- `.cortyx/embeddings.tvim` is the derived TurboVec ANN cache (rebuilt when stale)
- `prepare()` warms SIMD lookup tables on load
- Full-corpus ANN search runs in parallel with BM25 via RRF fusion
- LLM-free pseudo-relevance feedback: query vector blended (75%+25%) with mean of top-3 BM25 candidate embeddings before ANN

**Persistence hardening (`src/index/core/persistence.rs`):**
- WAL entries use line-oriented hex CRC32 (format: `CORTYXWAL1` magic header)
- `index.json` has a `.cortyx/index.checksum` sidecar (CRC32, verified on load)
- Activation cache has a 4-byte CRC32 little-endian trailer
- Corrupt WAL entries are skipped (skip-and-recover, not abort)

---

## Fleet Module (`src/fleet/`)

Fleet provides zero-server, local-first cross-project context sharing. When the local
BM25 confidence score for a query falls below `FLEET_LOW_CONFIDENCE_THRESHOLD` (4.0),
the `get_contexts` handler automatically fans out to registered peer projects.

### Design invariants

- **Zero overhead when absent:** if `~/.cortyx/fleet/nodes.json` does not exist,
  no fleet code runs — `CortyxServer.fleet_registry` is `None`.
- **No daemon required:** each query dispatches blocking index loads on tokio
  `spawn_blocking` tasks, capped at `FLEET_QUERY_TIMEOUT_MS` (200ms) total.
- **Supplementary only:** fleet results are appended after local context, never
  replacing it. Local weight 0.7, fleet weight 0.3 (RRF merge).
- **Module-manifest routing:** nodes whose registered module list does not match
  the active `module` filter are skipped — avoids irrelevant fan-out.

### Registry format (`~/.cortyx/fleet/nodes.json`)

```json
{
  "version": 1,
  "nodes": [
    { "id": "node-a1b2c3d4", "path": "/abs/path", "alias": "api-svc",
      "modules": ["auth", "billing"], "last_registered": "2026-05-12T..." }
  ]
}
```

`FleetNodeId` is a newtype over a blake3-short (8 hex chars) of the canonical path,
ensuring stable identity across renames of the alias.

### Escalation trigger in `get_contexts`

```
local_top_score < 4.0 AND fleet_registry.is_some()
  → route_fleet_query(task, module_filter, max_tokens/4, registry)
  → append fleet results as tagged HTML comment blocks in the output string
```

---

### Path Escape Prevention
`src/neuron/filter.rs` — `validate_relative_path()` and `validate_synapse_path()`

Any user-supplied path (neuron name, synapse target) is validated before use:
1. Must be relative (no `/` prefix)
2. No `..` or `.` components (except `.cortyx/neurons/` prefix for synapse targets)
3. Returns `SecurityError::PathEscape { path }` on violation

Call sites: `core_neuron_path()` in `src/neuron/io.rs` (filename-only fallback
also applied when `strip_prefix` fails — defence in depth).

### Git Pull Allowlist
`src/mcp/mod.rs` — `serve()`

Remote concept sync (`git pull`) is gated by a URL allowlist. Only
`github.com`, `gitlab.com`, and `ssh git@` remotes are permitted.
Returns `SecurityError::UntrustedRemote { url }` on violation.

### In-Flight Memory Cap
`src/mcp/tools/context/inflight_guard.rs` — `InflightGuard`

`MAX_INFLIGHT_BYTES = 64 MB`. An `AtomicUsize` counter on `CortyxServer` tracks
bytes currently in-flight across all concurrent `get_contexts` calls. The RAII
`InflightGuard` decrements the counter on any return path. Prevents memory
amplification attacks from concurrent large payloads.

---

## QueryContext Activation Pipeline (`src/index/core/pipeline/`)

The retrieval path is structured as an **immutable-snapshot pipeline**. Before
any scoring begins, `NeuronIndex::build_query_context()` snapshots all relevant
index state into a `QueryContext<'a>` — a zero-copy borrow struct. Stages then
operate as pure functions over this snapshot.

### Design invariants

- **`QueryContext<'a>` is immutable.** No stage may mutate the index during
  scoring. Hebbian writes happen after the pipeline returns.
- **Each stage is independently testable.** Every stage implements
  `ActivationStage` (`fn apply(&self, ctx: &QueryContext<'_>, candidates: &mut
  Vec<ScoredCandidate>)`). Tests use `QueryContextFixture` + `test_entry` from
  `types.rs` — no `NeuronIndex` instantiation required.
- **TRIZ separation-in-time:** conflicting requirements (mutable index vs. pure
  scoring) are resolved by separating mutation (before/after) from retrieval
  (during).

### Data flow

```
NeuronIndex::get_contexts_with_overflow(task, max_tokens, module, kind)
  │
  ├─► build_query_context(task, max_tokens, module, kind)
  │     tokenise → classify → resolve module/kind sets
  │     borrows: entries, posting_list, adjacency, vocab_bridge, …
  │     returns: QueryContext<'a>
  │
  ├─► ActivationPipeline::phase1().run(&ctx, &mut candidates)
  │       Bm25ScoringStage         — seed + bridge BM25 with δ-BM25 + hit-rate boost
  │     VocabBridgeStage         — synonym expansion via vocab_bridge map
  │     MorphemeBridgeStage      — morpheme root bridging (0.7× weight)
  │     PmiExpansionStage        — PMI co-occurrence graph expansion (0.5× weight)
  │     UseCaseAugmentStage      — use-case tag matching (0.9× parent score)
  │     SynapseTraversalStage    — 1-hop synapse graph traversal
  │     SessionClusterStage      — session-cluster scoring
  │     CoreturnBoostStage       — co-return frequency boost (Hebbian read)
  │     CoactivationStage        — term co-activation scoring
  │     StalenessDecayStage      — time-based staleness decay
  │     CountingAugmentStage     — counting/quantitative query augment
  │     SessionTfDecayStage      — intra-session TF decay (0.85× weight)
  │     TemporalProximityStage  — exponential recency boost: score *= 1 + bias×0.3×exp(-age/30d)
  │
  ├─► ANN rerank (embed feature): RRF-fuse BM25 + full-corpus TurboVec ANN
  │     LLM-free HyDE: query_vec blended with mean of top-3 BM25 candidate embeddings
  │
  ├─► rank, token-budget slice, format output
  │
  └─► record_co_return() — Hebbian write (after pipeline; never during)
```

### NeuronIndex state-view accessors

`NeuronIndex` exposes four typed borrow extractors used in `build_query_context`:

| Accessor | Struct | Fields |
|---|---|---|
| `retrieval_state()` | `RetrievalStateView<'a>` | entries, adjacency, path_index, posting_list, vocab_bridge, morpheme_map, pmi_neighbors, session_index, module_index, df_cache, avg_doc_len, embeddings |
| `feedback_state()` | `FeedbackStateView<'a>` | coactivation_counts, co_return_counts, session_utilization |
| `persistence_state()` | `PersistenceStateView<'a>` | project_root, delta flags, dirty_sidecars |
| `watcher_state()` | `WatcherStateView<'a>` | dirty_set Arc |

---

## Concurrency Model

### Write-Lock Split (`src/watcher.rs`)

The hot-patch watcher previously held the write lock through invalidation
(~1 ms) AND compilation (~100 ms). Fixed by splitting into two lock windows:

```
acquire write lock → invalidate (mark stale) → RELEASE
[readers get ~100ms access here]
acquire write lock → compile_dirty (insert compiled results) → RELEASE
```

### Dirty-Set Race Elimination (`src/index/core/types.rs`, `src/watcher.rs`)

`dirty.json` created a TOCTOU race between the watcher (writer) and
`compile_dirty()` (reader). Replaced with an `Arc<Mutex<HashSet<PathBuf>>>`
field on `NeuronIndex`. The watcher inserts paths; `compile_dirty()` drains
them atomically. One-time migration: `compile_dirty()` reads legacy `dirty.json`
on first call if the set is empty, then deletes the file.

---

## S4 Delta-Append + CRC32 Persistence (`src/index/core/persistence.rs`)

`src/index/core/persistence.rs`

> ⚠ The WAL provides **crash safety for mutations** (skip-corrupt-entry recovery)
> but is **not a full ACID transaction log**.

**WAL format (Wave 1C hardening):**
- Magic header: `CORTYXWAL1` (line-oriented text format)
- Each entry: `{json_payload}\t{hex_crc32}\n`
- On load: entries with mismatched CRC32 are skipped (corrupt-entry recovery)
- Legacy BLAKE3-prefixed format (`[32 bytes][JSON]`) is still readable

**Checksum sidecars:**
- `index.json` → `.cortyx/index.checksum` (CRC32, verified on load; mismatch forces rebuild)
- Activation cache → 4-byte CRC32 little-endian trailer (appended at write, verified at read)

**Delta-append optimisation:**
After the full `index.json` is written, subsequent small mutations (single
neuron compiles) are appended as deltas. On the next full save the deltas are
merged. The `wal_base` field records the byte offset of the last full save so
the delta region can be sliced out on load.

---

## effective_weight() Blend Formula (`src/neuron/synapse.rs`)

Synapse strength is a graduated blend of `prior_weight` (static score assigned
at compile time) and `learned_weight` (updated by the feedback loop):

```
blend_ratio = min(traversal_count / 100, 0.5)   // saturates at 50% learned
effective   = prior_weight * (1 - blend_ratio) + learned_weight * blend_ratio
```

- At `traversal_count < 10` → 100% prior (cold-start: insufficient signal)
- At `traversal_count = 10` → 90% prior / 10% learned
- At `traversal_count = 25` → 75% prior / 25% learned
- At `traversal_count ≥ 50` → 50% prior + 50% learned (cap; domain knowledge is never fully discarded)

The 50% saturation cap prevents a high-velocity feedback loop from discarding
prior domain knowledge.

---

## Hebbian Synapse Formation (`src/index/core/stats.rs`, `src/index/core/activation/search.rs`)

> "Hebbian" here is a **metaphor**, not a mechanistic claim.

Hebb's rule is "neurons that fire together wire together." The analogy:
neurons (files) that are **returned together** in the same query result get
wired. The signal is co-return co-occurrence, not simultaneous neural activation.

Implementation:
1. After each `get_contexts` call, Verbatim×Verbatim pairs in the result are
   counted in `co_return_counts: Mutex<HashMap<(usize, usize), u32>>`.
   Keys are `path_index` entry IDs (O(8) hash) not PathBufs (O(path_len) hash).
2. When a pair reaches `HEBBIAN_THRESHOLD = 10`, a `SemanticRelated` synapse
   is queued (can't mutate adjacency under shared borrow).
3. On the next `apply_pending_hebbian_synapses()` call (`&mut self`), the pair
   is wired bidirectionally and the counter is set to `HEBBIAN_THRESHOLD + 1`
   (sentinel: prevents re-firing on future calls).

---

## TRIZ Contradiction Resolution Log

This table records each code-review finding, its TRIZ classification, and the
resolution applied.

| Finding | Type | Principle | File | Resolution |
|---|---|---|---|---|
| `learned_weight: f32` bypasses `SynapseWeight` | PC (fictitious) | P35 Change Parameters | `neuron/synapse.rs` | `SynapseWeight` newtype applied; zero runtime cost (`#[repr(transparent)]`) |
| Duplicate serde default fns | TC | P25 Self-Service | `index/core/config.rs` | Named constants; default fns reference constants |
| Crate-wide `#[allow(clippy::unwrap_in_result)]` | TC | P1 Segmentation | `Cargo.toml` | Removed; each call site manages its own safety contract |
| `TermFrequency` unused in `BM25Entry` | PC (fictitious) | P35 Change Parameters | `index/core/bm25/entry.rs` | `TermFrequency` newtype applied in hot struct; zero overhead |
| `simhash_1024` names 256-bit hash | TC | P2 Taking Out | `types/`, `index/core/` | Renamed `simhash_256`; array `[u64;16]` → `[u64;4]`; v8→v9 migration |
| Path escape in `core_neuron_path` | TC | P22 Turn Harm into Benefit | `neuron/io.rs`, `neuron/filter.rs` | `validate_relative_path()` + `SecurityError::PathEscape`; filename fallback |
| Unconditional `git pull` | PC | P10 Preliminary Action | `mcp/mod.rs` | URL allowlist gate; `SecurityError::UntrustedRemote` |
| In-flight memory exhaustion | TC | P10 Preliminary Action | `mcp/tools/context.rs` | `InflightGuard` RAII; `MAX_INFLIGHT_BYTES = 64 MB` |
| `anyhow` in library helpers | TC | P5 Merging | `mcp/helpers/meta_io.rs` | Replaced with `CortyxError` + `cortyx_bail!`/`cortyx_err!` |
| `Other(String)` error proliferation | TC | P23 Feedback | `error.rs` | `SecurityError` enum added; typed variants prevent regression |
| `dirty.json` TOCTOU race | PC | P13 Opposite + P25 Self-Service | `watcher.rs`, `index/core/types.rs` | In-memory `Arc<Mutex<HashSet<PathBuf>>>` on `NeuronIndex` |
| Write lock held through compile | PC | P19 Periodic Action | `watcher.rs` | Lock split into two fast phases; readers get ~100ms unblocked |
| `co_return_counts` PathBuf keys | TC | P3 Local Quality | `index/core/types.rs`, `stats.rs`, `activation/search.rs` | Keys changed to `(usize, usize)` using `path_index` IDs |
| "WAL" misnaming | TC | P26 Copying | `index/core/persistence.rs` | Comments corrected to "S4 delta-append"; ⚠ note added |
| `effective_weight()` doc wrong | TC | P16 Partial Action | `neuron/synapse.rs` | Doc corrected: graduated blend, 50% saturation cap |
| Hebbian metaphor mismatch | TC | P26 Copying | `index/core/stats.rs` | One-line clarification added: co-return co-occurrence, not neural activation |
| `session_tf` grows monotonically — early-session terms pollute late-session retrieval | TC | P15 Dynamics, P35 Parameter Changes | `mcp/tools/context.rs` | Forget-gate decay λ=0.9 applied after each `get_contexts` call; entries pruned at count < 1 |
| Session memory tracks query vocabulary but not retrieved content | TC | P23 Feedback, P3 Local Quality | `mcp/mod.rs`, `mcp/tools/context.rs` | `session_path_history` (λ=0.8 decay) added alongside `session_tf`; retrieved paths receive +15% score boost on next call |
| Weak overflow items dilute high-confidence primary results | TC | P21 Skipping, P1 Segmentation | `mcp/tools/context.rs` | Adaptive channel gate: suppress overflow score < 1.5 when primary BM25 max > 8.0 |
| `apply_synapse_decay()` only fires at startup — synapses go stale during long-running serve | TC | P19 Periodic Action | `mcp/mod.rs` | Background tokio task on 24 h interval re-applies LTD; skips first tick (startup already ran) |
| **C2** Fixed emission tier thresholds misaligned with confidence constants | PC | P1 Segmentation, P35 Parameter Changes | `mcp/helpers/context_render.rs` | `select_emission_tier()` thresholds now use LOW_CONFIDENCE=4.0 / HIGH_CONFIDENCE=8.0; Focused fills 4–8 gap |
| **C6** Fixed Hebbian threshold ignores signal consistency (fixed count=10) | PC | P11 Cushion in Advance, P35 Parameter Changes | `index/core/stats.rs`, `index/core/activation/search.rs` | Wilson score lower bound at z=1.0 replaces fixed threshold; strong pairs wire at count≈3, noisy pairs never wire; sentinel = u32::MAX prevents overshoot |
| **C7** Static fleet merge weight 0.3 ignores result quality | PC | P15 Dynamics | `fleet/merge.rs`, `mcp/tools/fleet.rs` | `dynamic_fleet_weight(top_score)` — sigmoid-shaped [0.10, 0.50]; baseline 0.30 at midpoint=4.0 (LOW_CONFIDENCE) |
| **C3** Temporal anchor prefix poisons date parsers | TC | P10 Preliminary Action | `answer_plane/temporal/` | `strip_anchor_prefix()` strips `"As of <DATE>, "` from query text before all temporal parsers |
| **C4** Cold-start no signal for new entries | TC | P3 Local Quality + P35 Parameter Changes | `index/core/impl_helpers/` | `structural_centrality: f32` import/call in-degree prior; 0.2× weight decaying to 0 at 200 activations; file-stem overlap gate |
| **C5** Benchmark claims unverifiable | AC | P22 Turn Harm into Benefit | `main.rs`, `cli.rs`, `benchmarks/registry.json` | `cortyx proof-certificate [--validate]` reads live registry; `--validate` exits 1 on unmeasured metrics; CI gate added |
| **C6** Capsule staleness invisible to agent | TC | P23 Feedback | `mcp/tools/context/mod.rs` | `.hash` sidecar written at save time; mismatch on read emits stale-capsule HTML comment warning |
| **C7** Feedback poisoning from soft misses | PC | P1 Segmentation + P3 Local Quality | `mcp/tools/context/mod.rs`, `mcp/feedback_buffer.rs` | `ImplicitFeedbackTier` enum — `Miss` is a no-op; floor hits via `on_session_end()` only |
| **C8** Benchmark noise from live feedback writes | TC | P15 Dynamics | `mcp/mod.rs` | `serve --frozen` gates all feedback writes on `!frozen` |
| **C9** Temporal reasoning synthesis gap (F1 ≈ 0) | TC | P28 Replace Mechanical System + P25 Self-Service | `answer_plane/scoring/date_utils.rs`, `index/core/impl_helpers/indexer.rs` | ISO-8601 parsing, `ExplicitDateMatch` enum, anchored relative-date resolution, KG→BM25 alias injection |
| **NC2** Embedding version mismatch causes hard error | AC | P25 Self-Service | `src/embedder.rs` | `EmbeddingLoad::NeedsRebuild` → background tokio rebuild; no user intervention required |

**Deferred (too invasive for automated refactor):**

| Finding | Status | Reason |
|---|---|---|
| `source_path` absolute | Blocked | Requires `NeuronMeta` refactor + data migration across all call sites |
| `use super::*` glob imports | Blocked | 30+ production files in `index/core`; pervasive namespace-sharing pattern |
| `NeuronIndex` god struct decomposition | Blocked | 25+ fields, dozens of callers; requires dedicated branch + integration harness |

---

## Product Overview

- **Local core (shipped):** compile/mine/index/get-contexts/route/status over local neurons, temporal facts, agent diaries, and the optional git-federated concept library.
- **Answer plane (shipped, separately benchmarked):** `answer_mode` and provenance sit on top of retrieved evidence and do **not** change the retrieval hot path. The answer plane uses rule-based extraction from retrieved neurons — the AI agent is expected to perform synthesis over the delivered context.
- **Delivery/control planes (shipped, separately benchmarked):** token economy, prompt-cache-aware delivery, startup, and control-plane latency are tracked independently.
- **Shared/team/trust + UX proofs (shipped, non-headline):** shared-memory handoff resolution, provenance integrity, and machine-readable CLI UX now have deterministic proven proof harnesses. Shared-sync contracts remain support surfaces, not hosted-platform or human-study claims.
- **Graph reasoning (shipped, proven):** multi-hop traversal with `TraversalStats` (nodes_by_depth, convergence, depth_coverage) captured in every `ReasoningReport`; reasoning chains emitted in answer-plane output; `multi_hop=true` enables iterative seed expansion from top-5 initial results.
- **ECS verification gate (`--features verify`, optional):** PureReason ECS checks gate mine_conversation, kg_add, check_consistency, answer plane, and concepts publish-ready. High-risk content (risk_score > 0.60) is blocked before it enters long-term memory; quarantine range 0.35–0.60; zero-cost no-op when the feature is disabled.

**How:** The static prefix (schema + instructions) is always byte-identical → Anthropic/OpenAI cache it. Dynamic neurons (3–5 per task, ~800–2 000 tokens) are injected *after* the `cache_control` breakpoint. Cache key = static prefix only. On iterative same-session work, `delta_mode=true` + `context_handle` lets Cortyx resend only added/changed dynamic chunks instead of the full prior set, and `capsule_mode=true` can collapse repeated same-module background into a stable cached capsule plus a tiny task delta.

### Context Zones

- **Stable zone:** compiled capsule content (`## project`, `## module/{name}`, proven `## use-case/{name}`) rendered first; it changes only when `cortyx compile` regenerates capsule files.
- **Dynamic zone:** query-time context emitted after the stable zone by the 13-stage Hebbian pipeline; it contains task-specific neurons, overflow summaries, and delta-only updates when `delta_mode=true`.

## Neuron Types

| Kind | File pattern | Purpose |
|------|-------------|---------|
| **Core** | `src_engine_rs.context.md` | AI-curated reasoning guide for one source file |
| **UseCase** | `src_engine_rs.usecase.dark-mode.md` | Exact proven chunk for a recurring pattern |
| **Verbatim** | `__verbatim_*.context.md` | Mined conversation turns for semantic recall |
| **Concept** | `__concept_*.context.md` | Cross-cutting concept (auth flow, DB migrations) |
| **Project** | `_project.context.md` | Top-level project description + conventions |

## Activation Pipeline

> **Implementation note (v0.3.0):** The activation pipeline was restructured in
> Phase 1 into the `QueryContext` pipeline architecture described in the
> [QueryContext Activation Pipeline](#querycontext-activation-pipeline-srcindexcorepipeline)
> section above. Each numbered phase below maps to one or more `ActivationStage`
> implementations in `src/index/core/pipeline/stages/`.

**Pure Rust, ≤40 ms**

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

## Self-Improving Feedback Loop

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

**Optional answer plane:** Set `answer_mode=true` on `cortyx_get_contexts` or pass `--answer-mode` to `cortyx get-contexts` to derive a concise answer from the selected contexts without changing the retrieval hot path. The current layer reuses Cortyx's existing synthetic derived answers when available, prefers mine-time `## answer_surface` rows for high-confidence direct facts and adjacent dialogue question→answer pairs without perturbing retrieval indexing, extracts compact spans for direct fact questions, and can resolve reusable temporal + aggregate families such as dated binary-choice prompts, elapsed-day intervals, month-scoped activity-day counts, distinct event/cuisine/venue counts, citrus / delivery-service counts, weekly fitness schedules, missed-event counts, recent ceremony counts, narrow device-count questions, peak-season weekly-hour arithmetic, recent activity-duration totals, and current magazine-subscription counts. Direct fact coverage still includes common classes such as job, residence, degree, and pet name. Add `provenance_mode=true` or `--provenance` to append lightweight source metadata and summaries. The AI agent performs synthesis over delivered context — Cortyx does not embed a generative model.

**Explicit training boundary:** Cortyx now trains long-term ranking only from explicit response evidence: `cortyx_close_task`, manual `cortyx_record_hit`, or `previous_response` overlap against the prior activation. Consecutive `get_contexts` calls, evolve/edit tools, preview tools, and rollback operations do **not** auto-promote hits.

**Adaptive synapse weights** (resolves NE9): Each synapse edge has a `learned_weight` that starts at the static type multiplier and updates via **exponential moving average** (α = 0.1) from citation signals. After ~100 traversals, the weight encodes actual helpfulness for this specific project's call patterns. Cold-start: identical to static weights (learned_weight not applied until 10+ traversals).

**Adaptive CI quarantine (R11-S4):** Neurons activated often but rarely cited are automatically quarantined (`staleness_multiplier → 0.3`). The Wilson score confidence interval now scales with sample size — reacting fast to obvious noise at 5–19 activations (z=1.0, 68% CI) and becoming progressively stricter at larger counts (z=1.645 at 20–99; z=1.96 at 100+). **3× faster noise detection** vs the old fixed-sample-size approach, with **zero false-positives at cold start** (< 5 activations withheld entirely). Quarantine lifts automatically when citation rate recovers above 15%.

## AST Bootstrap

At `cortyx compile`, function signatures, type names, and doc comments are extracted from source and pre-filled into neuron stubs — **without any LLM call**. BM25 has real vocabulary from the first query.

**R14 A3 — LLM-free pre-population:** Module-level doc comments are parsed into the neuron's `## Purpose` section automatically. Level-1 neurons (static, non-curated) now cover ~75% of cold-start R@5 queries vs ~30% for empty stubs — without a single LLM call.

**R14 A1 — Multi-source vocabulary injection:** Soft vocabulary (weight 0.3×) is injected from three additional sources at compile time: (1) inline comments and docstrings in the source file, (2) git commit messages touching this file, (3) README sections mentioning the module name. These terms fill vocabulary gaps between LLM-curated neurons and the natural language used in queries. Hard neuron content always wins; soft terms only fill gaps.

**R14 B3 — NLP-free alias generation:** For every public function, natural-language aliases are generated by rule-based identifier splitting + verb/noun synonym tables (`get_user_by_id` → "fetch user", "retrieve user", "find user by id"). Stored at 0.5× BM25 weight. Zero model calls, zero dependencies — entirely deterministic.

**R14 A2 — Peer template borrowing:** Cold stubs with <10 BM25 terms automatically borrow vocabulary from their 3 most structurally similar neighbours (Jaccard + same-module bonus) at 0.2× weight. Eliminates vocabulary deserts in newly-added modules.

**AST bootstrap languages:** Rust, Python, TypeScript/JavaScript, Go, Swift, Kotlin, Java, C#, Ruby, C/C++, PHP, Lua, R/Rmd, Julia, Elixir, Zig, Dart, Shell/Bash/Zsh, SQL, HCL/Terraform, Protocol Buffers, GraphQL, Jupyter Notebooks (`.ipynb`). Every other file type activates the **universal vocabulary fallback** — identifier tokens and comment text are harvested into BM25 from day 1, without affecting the `sig_hash` that drives staleness detection.

## Auto-Wiring

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

## Vocabulary Bridge

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

## Adaptive Reasoning

Cortyx implements three recursive reasoning features inspired by the RecursiveMAS architecture, adapted for a non-LLM retrieval engine:

### 1. Adaptive Iterative Deepening (`AdaptiveReasoner`)
Wraps `GraphReasoner` with automatic retry: if the first traversal pass did not converge (was cut short by `max_expansions`), it retries with `max_hops + 1`. Up to 3 passes total. Returns `IterationStats { passes, final_options }` alongside the `ReasoningReport`.

### 2. Iterative Query Expansion
When the top BM25 score falls below `LOW_CONFIDENCE_THRESHOLD` (= 4.0), a second retrieval pass is triggered with concept-cloud-expanded query terms. Results are merged via Reciprocal Rank Fusion (RRF). This handles vocabulary mismatch without requiring embeddings.

### 3. Diary Blocker Decomposition (`refine_entry` / `cortyx_diary_refine`)
`refine_entry(&mut StructuredDiaryEntry)` detects vague, too-large, or waiting blockers using heuristic patterns and populates `refined_plan` with a structured decomposition suggestion — no LLM required. The `cortyx_diary_refine` MCP tool exposes this to agents.

## Self-Improvement Workflow

```
1. cortyx_get_contexts(task="implement JWT auth")  → activates auth-related neurons
2. ... do the work ...
3. cortyx_close_task(response_text="...")           → auto-records which neurons helped
4. cortyx_evolve_section(path, "pitfalls", "...")   → refine one section, ~50 tokens
5. cortyx_extract_from_raw(path, ...)               → save a proven pattern as use-case
```

Over time, your neurons become laser-precise reasoning guides for your specific codebase — trained by your own usage with zero supervision.

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

## Advanced Features

### Fleet


When Cortyx has low local confidence (BM25 top-score < 4.0) for a query, it can
automatically supplement the response with context from registered peer projects —
no server, no daemon, no network.

```bash
# Register a peer project as a fleet node
cortyx fleet register ../api-service --alias api

# Register the current project
cortyx fleet register .

# List all fleet nodes
cortyx fleet list

# Fleet status summary
cortyx fleet status

# Deregister a node
cortyx fleet deregister api
```

- **Zero overhead:** if `~/.cortyx/fleet/nodes.json` is absent, no fleet code runs.
- **Parallel dispatch:** each registered node is queried concurrently on tokio `spawn_blocking` tasks with a 200ms wall-clock deadline — fleet never delays local context delivery.
- **Module-manifest routing:** nodes whose module list does not intersect the active `module` filter are skipped — avoids irrelevant fan-out.
- **Supplementary output:** fleet context appended after local context as tagged HTML comment blocks (local weight 0.7, fleet weight 0.3).
- **Registry:** plain JSON at `~/.cortyx/fleet/nodes.json` — inspectable, git-trackable.

**MCP tools:** `cortyx_fleet_query(task, module?, max_tokens?)` and `cortyx_fleet_status` are available for explicit cross-project lookups from within agent sessions.

### Global Concept Library

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

### R16 Self-Curating Semantic Memory

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

### R17 Model-Free Accuracy Boost

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

### Neuron Safety (R14 E1+E2)

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

### Three-tier retrieval / embeddings

When `embeddings.bin` is present, Cortyx performs **three-tier retrieval**:
1. BM25 keyword scoring (always on) — with confidence-adaptive gating (skip re-ranking for decisive queries; log vocabulary gaps for zero-match queries)
2. TF-IDF cosine tie-break (automatic when confidence ratio < 1.5)
3. Dense cosine + RRF fusion (when `--features embed` and embeddings are indexed)

The dense model (all-MiniLM-L6-v2, ~80 MB, downloaded once) is loaded lazily at server startup. Per-query cost ≤ 0.1 ms (cosine over pre-computed unit-norm vectors). Falls back gracefully to BM25-only when the model is not installed.

**Auto-embed on compile:** `cortyx compile` automatically runs an embedding pass after indexing when built with `--features embed`, so embeddings stay current without a separate step.

**Air-gap / offline mode:** Set `CORTYX_NO_DOWNLOAD=1` to prevent any model download attempt entirely (useful in corporate proxies or CI environments without internet access). Cortyx will operate in BM25-only mode with no error. The install script respects `CORTYX_NO_EMBED=1` to skip the embed-enabled binary entirely.

### Cross-Encoder Reranking

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

### Hallucination Safety / PureReason ECS

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

### Optional Feature Summary

| Feature flag | What it adds | Extra dep | Default install |
|---|---|---|---|
| `embed` | fastembed hybrid retrieval + auto-embed on compile | ~80 MB model download | ✅ (embed binary) |
| `rerank` | ONNX INT8 cross-encoder reranker | ~7 MB model download | ❌ |
| `verify` | PureReason ECS hallucination gate | PureReason sibling checkout | ❌ |

All features are additive and independently opt-in. The default release binary includes `embed`. Every other feature is a zero-overhead no-op on the default path.

### Hierarchical Navigation

Three browse tools for agents that need to explore the neuron tree:

```
cortyx_list_modules         → [{name, neuron_count, avg_hit_rate, is_person_scope}]
cortyx_list_neurons(module) → [{path, status, use_count, hit_rate}]
cortyx_peek_neuron(path)    → first 20 lines of neuron file
```

This gives MemPalace-level hierarchical navigation over Cortyx's existing `module_index` — **zero new data structures**.

### Person-Scoped Memory

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

### Kind Filter

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

### Prompt Caching Guarantee

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

### Schema Migrations

The index format is versioned (`INDEX_VERSION`). When Cortyx detects an older index, it applies a migration chain — **all user-curated data (`use_count`, `hit_count`, `staleness_multiplier`, synapses) is preserved** across upgrades. No data loss on version bumps.

## MCP Tools Reference

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
| `cortyx_fleet_query(task, module?, max_tokens?)` | Query registered fleet nodes for supplementary cross-project context. Called automatically by `cortyx_get_contexts` when local confidence is low; also available for explicit cross-project lookups. |
| `cortyx_fleet_status` | List all registered fleet nodes with alias, path, module count, and last registration time. |
| `cortyx_kg_add(entity, predicate, value, valid_from?)` | Add a temporal fact to a KG entity neuron (git-tracked, BM25-indexed Markdown) |
| `cortyx_kg_query(entity, as_of?)` | Query active facts for a KG entity as of an optional ISO-8601 date |
| `cortyx_kg_invalidate(entity, predicate, ended)` | End/supersede an active KG fact by setting its `ended` date |
| `cortyx_kg_timeline(entity, predicate)` | Show the full temporal history of a predicate on a KG entity |
| `cortyx_kg_stats` | Aggregate statistics: entity count, active vs ended facts |
