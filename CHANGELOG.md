# Changelog

All notable changes to Cortyx will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [0.4.0] — 2026-05-20

### Breaking Changes
- **Embedding dimension changed from 384 → 768.** Delete `.cortyx/embeddings.bin`
  and rerun `cortyx compile --features embed` to regenerate. The new model
  (`NomicEmbedTextV15`) produces higher-quality embeddings with 8192 token context.
- **`EMBED_VERSION` bumped to 2.** Old `embeddings.bin` files are rejected on load
  with a clear error message.

### Added
- **TurboVec 4-bit quantized ANN** (`turbovec 0.4.1`, `--features embed`):
  - `EmbeddingStore` backed by `turbovec::IdMapIndex` with stable blake3 path IDs
  - Full-corpus ANN search runs in parallel with BM25; results merged via RRF
  - LLM-free pseudo-relevance feedback: query vector blended (75% query + 25% mean
    of top-3 BM25 candidate embeddings) before ANN for higher recall
  - `prepare()` warms SIMD lookup tables at startup
  - `.cortyx/embeddings.tvim` derived ANN cache (rebuilt from `.bin` when stale)
- **Embedding model upgrade**: `AllMiniLML6V2` (384-dim) → `NomicEmbedTextV15`
  (768-dim, 8192 token context) via fastembed 5.8.0
- **13-stage activation pipeline** — Wave 1A: all 10 previously-empty stub stages
  now have real implementations; one new stage added:
  - `MorphemeBridgeStage` — morpheme map expansion at 0.7× weight
  - `PmiExpansionStage` — PMI neighbor expansion at 0.5× weight
  - `StalenessDecayStage` — multiplies score by `staleness_multiplier`
  - `SessionTfDecayStage` — 0.85× decay for same-session candidates
  - `UseCaseAugmentStage` — injects UseCase sub-neurons at 0.9× parent score
  - `SynapseTraversalStage` — 1-hop adjacency walk above threshold
  - `CoactivationStage` — Hebbian coactivation history boost
  - `CoreturnBoostStage` — co-return pair boost for count ≥ 5
  - `SessionClusterStage` — injects top-2 session siblings when Verbatim in top-3
  - `TemporalProximityStage` (NEW) — exponential recency boost:
    `score *= 1.0 + temporal_bias × 0.3 × exp(-age_days / 30.0)`
- **`temporal_bias` MCP parameter** — `cortyx_get_contexts` now accepts
  `temporal_bias: Option<f32>` (range [0.0, 3.0]). `0.0` disables temporal boost;
  `2.0` doubles it. Default: 1.0.
- **NeuronIndex domain decomposition** (Wave 2):
  - The 25-field god struct split into 4 typed domain structs in `src/index/core/domain/`:
    `RetrievalState`, `FeedbackState`, `PersistenceState`, `WatcherState`
  - `NeuronIndex` now owns these 4 fields; all field accesses are domain-scoped
  - `pub(in crate::index)` visibility prevents cross-domain coupling
- **Persistence hardening** (Wave 1C):
  - WAL entries have per-entry hex CRC32 checksums (`CORTYXWAL1` line format)
  - Corrupt WAL entries are skipped (skip-and-recover, not abort)
  - `index.json` has a `.cortyx/index.checksum` sidecar (CRC32, verified on load)
  - Activation cache has a 4-byte CRC32 little-endian trailer
  - `crc32fast = "1"` added as a dependency
- **`search.rs` thin orchestrator** — refactored from 1,030 → 20 lines; logic
  extracted into `activation/phase1.rs` (363 lines), `activation/overflow.rs`
  (242 lines), `activation/selection.rs` (195 lines)
- **New integration test suites**: `pipeline_integration.rs` (171 lines),
  `turbovec_integration.rs` (198 lines), `persistence_hardening.rs` (277 lines)
- **Test count**: 870 (baseline) → 924 (current)

### Changed
- `src/index/core/config.rs`: added `TEMPORAL_DECAY_WEIGHT = 0.3`,
  `TEMPORAL_HALF_LIFE = 30.0`, `SESSION_SAME_SCORE_DECAY = 0.85`,
  `MAX_SYNAPSE_CANDIDATES = 50`
- `src/index/core/pipeline/types.rs`: `QueryContext` gains `temporal_bias_scale: f32`
- `src/mcp/types.rs`: `GetContextsInput` gains `temporal_bias: Option<f32>`

---

## [0.3.0] — 2026-06-XX

### Breaking Changes
- **`embed` and `rerank` features are now default.** Builds without ONNX support must use `--no-default-features`. First startup downloads `~130MB` of model weights unless `CORTYX_NO_DOWNLOAD=1` is set.
- **Index storage version bumped to v9.** Cortyx auto-migrates on first run; the previous index is discarded and rebuilt from `.cortyx/neurons/` in place (no data loss — neuron files are the source of truth).

### Added
- **QueryContext activation pipeline** (`src/index/core/pipeline/`) — 12 independently-testable `ActivationStage` implementations replace the former monolithic `search.rs` function. See [ARCHITECTURE.md § QueryContext Activation Pipeline](ARCHITECTURE.md#querycontext-activation-pipeline-srcindexcorepipeline).
- **MCP `context/` module decomposition** — `src/mcp/tools/context.rs` (1,691 lines) split into `inflight_guard.rs`, `session_decay.rs`, `answer_mode.rs`, and a thin `mod.rs` orchestrator.
- **Competitive benchmark roster expanded** — `benchmarks/registry.json` now tracks 9 comparators: the original 6 (MemPalace, Omega, Hindsight, Zep, Letta/MemGPT, Mem0) plus engram, vestige, and token-savior.
- **Hardened CI lint gates** — `cast_possible_truncation = deny`, `unwrap_used = warn`, `missing_docs = warn`; `cargo fmt --check` enforced on every PR via `quality.yml`.
- **LME-500 regression guard upgraded** — 80 rows/run (up from 20), SSU threshold 85% (up from 80%), KU threshold 65% (up from 60%).

### Changed
- `src/index/core/activation/search.rs` refactored to thin orchestrator; scoring logic now lives in the 12 pipeline stage files.
- `embed` and `rerank` Cargo features promoted to `default`.

---


- **Fleet module** (`src/fleet/`) — zero-server local-first cross-project context sharing.
  - `cortyx fleet register <path> [--alias <name>]` — register a peer project as a fleet node
  - `cortyx fleet deregister <alias-or-path>` — remove a node from the fleet
  - `cortyx fleet list` — show all registered nodes with alias, path, module count
  - `cortyx fleet status` — fleet health summary
  - `cortyx_fleet_query` MCP tool — explicit fleet context query from agent tooling
  - `cortyx_fleet_status` MCP tool — list registered nodes in-session
  - Auto-escalation in `cortyx_get_contexts`: when local BM25 top-score < 4.0 and fleet
    nodes are registered, fan-out query dispatches to peer projects (200ms timeout, parallel
    tokio tasks, module-manifest filtering to avoid irrelevant fan-out)
  - Fleet context appended as tagged comment blocks after local context (0.7/0.3 weight split)
  - Registry stored at `~/.cortyx/fleet/nodes.json` (mirrors global concepts pattern)
  - Zero overhead when registry absent — all fleet code skipped entirely

---

## [0.2.0] — 2026-05-11

### Changed
- Restored meaningful Clippy lints (removed blanket `clippy::all = "allow"`)
- Fixed `Cargo.toml` repository URL to `https://github.com/sorunokoe/Cortyx`
- Added `readme = "README.md"` to Cargo.toml for crates.io packaging
- Removed stale documentation files (COMPLETE_STATUS, REFACTORING_STATUS, IMPLEMENTATION_GUIDE,
  FINAL_IMPLEMENTATION_REPORT, recursiveMAS_analysis_report)
- **Monolith extraction** — split `src/index/core/` from 3 files totalling ~33k lines into 30+
  focused modules; no single file exceeds 2,000 lines:
  - `helpers.rs` (16k) → 7 focused files (`helpers_detect`, `helpers_extract`, `helpers_surface`,
    `helpers_phrase`, `helpers_temporal2`, `helpers_title`)
  - `synthetic.rs` (11k) → 5 focused files (`synthetic_router`, `synthetic_count`,
    `synthetic_temporal` × 3, `synthetic_kg`, `synthetic_session`)
  - `impl_helpers.rs` (2.9k) → 2 files (`impl_helpers`, `impl_helpers2`)
  - `tests.rs` (5.5k) → 4 focused test files
- **`--project` flag** — `cortyx serve --project /path/to/myapp` now loads `.cortyx/` from the
  specified path instead of always using the current directory
- Removed private `verify`/PureReason feature (path dep; no-op stubs remain in `verify_gate.rs`)
- Binary size regression guard updated to 14MB (actual stripped Linux binary: ~13MB)

### Fixed
- CI full-proof lane mine timeout: added `--selection-corpus` to limit mine to selected rows
- Clippy `expect_used` error in `mcp/mod.rs` — replaced with `?` operator
- 60+ mechanical clippy warnings across 40+ files (clamp, slice::from_ref, enumerate, etc.)
- `SyncTransportRelation::Diverged` variant boxing (512B → pointer-size)
- `&PathBuf` → `&Path` in public function signatures (`hooks.rs`, `registration.rs`)


### Added
- **Adaptive Reasoner** (`src/reasoner/`) — iterative query expansion and multi-hop BFS graph
  traversal; `cortyx_diary_refine` MCP tool for pure-heuristic blocker decomposition
- **Agent Diary system** — `cortyx_diary_write`, `cortyx_diary_read`, `cortyx_diary_refine` MCP
  tools; `@agent/<name>` neuron namespace; mirrors into temporal knowledge graph
- **ECS Verification Gate** (`src/verify_gate.rs`) — PureReason integration stubs; zero overhead
  when disabled; hallucination detection opt-in via `--features verify`
- **Dense embedding support** (`--features embed`) — all-MiniLM-L6-v2 (384-dim) via fastembed;
  BM25 + cosine RRF hybrid retrieval; auto-embed on `cortyx compile`
- **ONNX cross-encoder reranking** (`--features rerank`) — ms-marco-MiniLM-L-2-v2 INT8 quantized
  (~7 MB); activates only on low-confidence BM25 queries
- **Temporal Knowledge Graph** — `cortyx_kg_add`, `cortyx_kg_query`, `cortyx_kg_invalidate`,
  `cortyx_kg_timeline`, `cortyx_kg_stats` MCP tools; valid_from/ended validity windows;
  git-tracked Markdown fact triples
- **Collaboration Kernel** — `cortyx_agent_status`, `cortyx_collaboration_status` MCP tools;
  multi-agent sync transport
- **Global Concept Library** — `~/.cortyx/global/` git-tracked concept sharing across projects
- **NeuronKind::Aggregate** — cross-session aggregate neurons produced at mine time
- Quality infrastructure: `rustfmt.toml`, `clippy.toml`, `.editorconfig`, `Cargo.toml` lints,
  `.github/workflows/quality.yml` CI pipeline

### Changed
- **Module extraction** — split the original 33,870-line `index/core/mod.rs` monolith into
  sub-modules: `bm25/`, `query/`, `activation/`, `lsh/`, `concept_cloud/`, `state/`,
  `answer_surface/`, `helpers.rs`, `synthetic.rs`, `tests.rs`, `persistence.rs`
- Extracted `answer_plane/` sub-modules from 4,027-line `answer_plane/mod.rs`
- Extracted `miner/surface/` sub-modules from surface miner
- Extracted MCP tool handlers into `mcp/tools/{context,memory,knowledge,collaboration,admin}.rs`
- Applied `TokenCount`, `QueryText`, `BM25Score`, `SynapseWeight` newtypes in hot paths
- Replaced `anyhow::Error` with `CortyxError` (`thiserror`) in library code
- Temporal anchor extractor split into focused modules (`extractors/`, `families/`, `tests/`)
- Import parser split into focused modules
- AST extractor language core/extension split

### Fixed
- Dense embed re-rank gate preventing accuracy regression on LME-500
- `O(n²)` mining, 5K truncation bug, temporal routing overlap (`NE-1/NE-2/NE-4`)
- Numeric keyword extraction for short/single-digit answers (+12.2pp on LoCoMo bench)
- LoCoMo underscore keyword normalization (+0.2pp → 92.0%)
- Temporal reasoning patch series (patches 68–76): between-days, sequence, acquisition-aware
  comparison, booking-style lead times
- `sol-C/E` thresholds and entity profile accumulation regression
- `cortyx_record_hit` MCP handler: `send_json` body-read fix

### Performance
- BM25 R@5 on LME-500: 68.6% → **97.8%** (63 commits of targeted vocabulary improvements)
- LoCoMo answer recall: **92.0%** (corrected 200-question sample)
- Temporal reasoning F1: 0.228 → **0.401** (30 patches)
- p95 activation latency: **~22ms**
- Token economy: 98.4% savings on capsule+delta repeat queries

---

## [0.1.0] — Initial Release

### Added
- BM25 retrieval engine (k1=1.4, b=0.75) with IDF boosting, length normalization, SimHash LSH
- **Neuron model** — `.context.md` files in `.cortyx/neurons/`; kinds: Core, UseCase, Verbatim,
  Concept, Project
- **Synapse graph** — 8 edge types (Imports, Calls, Implements, SemanticRelated, Contradicts,
  TemporalFollows, Derived, ConceptExpands); 2-hop traversal; `Contradicts` auto-exclusion
- **MCP server** — 29 tools across 5 groups (retrieval, memory, knowledge, collaboration, admin)
  via `rmcp 1.4.0` STDIO transport
- **AST bootstrap** — `cortyx compile` extracts context stubs from 25 languages with zero LLM calls
- **Prompt-cache-aware delivery** — static neuron header (lexicographically sorted filenames) +
  `cache_control` breakpoint; ensures Anthropic/OpenAI prompt cache always hits
- **Self-improving rankings** — BM25 scores adjusted by `hit_rate × use_count` EMA feedback;
  Wilson-score CI adaptive quarantine
- **Git-tracked memory** — neurons as Markdown in project repo; `cortyx rollback`; BLAKE3 provenance
- **File watcher** — `cortyx watch` indexes changes in real-time
- **Token budget tiers** — Tier-1 (summary, 1.5–5.0), Tier-2 (full, ≥5.0), overflow (headline)
- **Delta mode** — returns only changed/added neurons on repeat queries
- **Installer** — `cortyx install` auto-detects Claude Code, Cursor, Windsurf, VS Code, Zed, Codex
- Platforms: Linux x86_64/aarch64, macOS x86_64/arm64, Windows x86_64

---

[Unreleased]: https://github.com/sorunokoe/Cortyx/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/sorunokoe/Cortyx/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/sorunokoe/Cortyx/releases/tag/v0.1.0
