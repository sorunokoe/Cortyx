# Changelog

All notable changes to Cortyx will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Changed
- Restored meaningful Clippy lints (removed blanket `clippy::all = "allow"`)
- Fixed `Cargo.toml` repository URL to `https://github.com/sorunokoe/Cortyx`
- Removed stale documentation files (COMPLETE_STATUS, REFACTORING_STATUS, IMPLEMENTATION_GUIDE,
  FINAL_IMPLEMENTATION_REPORT, recursiveMAS_analysis_report)

---

## [0.2.0] — Unreleased

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
