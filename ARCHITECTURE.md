# Cortyx Architecture

Cortyx is a Rust MCP-native context-delivery engine. It maintains a semantic
in-memory index of project knowledge (neurons) and serves them to an LLM host
via the Model Context Protocol. Retrieval is BM25-primary with synapse-graph
traversal, PMI-based synonym expansion, and Hebbian co-return feedback.

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
│       ├── types.rs        — NeuronIndex struct (25+ fields)
│       ├── compile.rs      — compile_dirty(), dirty_set_handle()
│       ├── persistence.rs  — load/save with S4 delta-append optimisation
│       ├── stats.rs        — Hebbian synapse formation, coactivation tracking
│       ├── config.rs       — index-wide constants
│       ├── bm25/           — BM25Entry, posting-list, IDF cache
│       └── activation/
│           └── search.rs   — get_contexts_with_overflow, Hebbian tracking
├── mcp/
│   ├── mod.rs              — CortyxServer, serve(), URL allowlist, inflight cap
│   ├── helpers/
│   │   ├── meta_io.rs      — neuron metadata helpers (CortyxError, no anyhow)
│   │   └── server_impl.rs  — for_benchmark() constructor (used in tests)
│   └── tools/
│       └── context.rs      — get_contexts handler + InflightGuard RAII
├── watcher.rs              — inotify/FSEvents watcher, in-memory dirty_set
├── reasoner.rs             — graph traversal + ReasoningReport
└── main.rs                 — binary entrypoint (anyhow OK here)
```

---

## Key Data Structures

### NeuronIndex (`src/index/core/types.rs`)

The central in-memory index. All operations run in RAM; no async I/O during
retrieval. Persisted to `.cortyx/index.json` after every mutating operation.

**Retrieval pipeline fields:**
- `entries: Vec<BM25Entry>` — BM25 corpus; index position = stable path ID
- `path_index: HashMap<PathBuf, usize>` — maps path → entries index (path interner)
- `posting_list: HashMap<String, Vec<usize>>` — term → entry indices
- `df_cache: HashMap<String, u32>` — document frequency cache
- `adjacency: HashMap<PathBuf, Vec<Synapse>>` — synapse graph

**Feedback fields:**
- `coactivation_counts: HashMap<PathBuf, HashMap<String, u32>>` — term promotion
  (persisted in `.cortyx/coactivation.json`)
- `co_return_counts: Mutex<HashMap<(usize, usize), u32>>` — Hebbian pair counts
  (in-memory only; keys are `path_index` IDs — O(8) hash vs O(path_len) PathBuf)
- `dirty_set: Arc<Mutex<HashSet<PathBuf>>>` — hot-reload dirty registry
  (shared with the watcher task; replaces dirty.json to eliminate TOCTOU race)

**Write-optimization field:**
- `wal_base: Option<u64>` — S4 delta-append size marker (⚠ NOT a WAL; no crash
  safety, no checksums — see "S4 Delta-Append" section below)

---

## Security Model

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
`src/mcp/tools/context.rs` — `InflightGuard`

`MAX_INFLIGHT_BYTES = 64 MB`. An `AtomicUsize` counter on `CortyxServer` tracks
bytes currently in-flight across all concurrent `get_contexts` calls. The RAII
`InflightGuard` decrements the counter on any return path. Prevents memory
amplification attacks from concurrent large payloads.

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

## S4 Delta-Append (Persistence Optimisation)

`src/index/core/persistence.rs`

> ⚠ This is **not** a Write-Ahead Log. It does not provide crash safety.

After the full `index.json` is written, subsequent small mutations (single
neuron compiles) are appended as deltas. On the next full save the deltas are
merged. The `wal_base` field records the byte offset of the last full save so
the delta region can be sliced out on load.

There are **no checksums** and **no recovery guarantees**. If the process is
killed mid-append, the delta is silently dropped and the last full save is used.
Field names (`wal_base`, `needs_full_save`) are kept for serialisation
compatibility; the comments now describe the actual mechanism.

---

## effective_weight() Blend Formula (`src/neuron/synapse.rs`)

Synapse strength is a graduated blend of `prior_weight` (static score assigned
at compile time) and `learned_weight` (updated by the feedback loop):

```
blend_ratio = min(use_count / 20, 0.5)   // saturates at 50% learned
effective   = prior_weight * (1 - blend_ratio) + learned_weight * blend_ratio
```

- At `use_count = 0` → 100% prior (cold start safety)
- At `use_count = 10` → 50/50 blend
- At `use_count ≥ 20` → 50% prior + 50% learned (never fully discards prior)

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

**Deferred (too invasive for automated refactor):**

| Finding | Status | Reason |
|---|---|---|
| `source_path` absolute | Blocked | Requires `NeuronMeta` refactor + data migration across all call sites |
| `use super::*` glob imports | Blocked | 30+ production files in `index/core`; pervasive namespace-sharing pattern |
| `NeuronIndex` god struct decomposition | Blocked | 25+ fields, dozens of callers; requires dedicated branch + integration harness |
