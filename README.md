# Cortyx

<p align="center">
  <img src=".github/logo.svg" alt="two neurons connected by a synapse" width="280">
</p>

[![CI](https://github.com/sorunokoe/Cortyx/actions/workflows/ci.yml/badge.svg)](https://github.com/sorunokoe/Cortyx/actions/workflows/ci.yml)  [![Quality](https://github.com/sorunokoe/Cortyx/actions/workflows/quality.yml/badge.svg)](https://github.com/sorunokoe/Cortyx/actions/workflows/quality.yml)  [![Benchmarks](https://github.com/sorunokoe/Cortyx/actions/workflows/benchmarks.yml/badge.svg)](https://github.com/sorunokoe/Cortyx/actions/workflows/benchmarks.yml)

> MCP-native context delivery engine for coding agents and long-lived conversations.<br>
> 96.8% R@5 on LME-500 · local-first · pure Rust · no runtime model

## Why Cortyx

- **Context delivery, not synthesis** — Cortyx puts the right neurons into the LLM window; your agent does the synthesis.
- **Local-first** — neurons are human-readable Markdown in `.cortyx/`, git-tracked, and never require a cloud backend.
- **Proven retrieval** — **96.8% macro R@5** on LME-500 (regenerated cleaned-oracle run; full 500-question benchmark via manual trigger), beating MemPalace **96.6%** on the same surface.

## Benchmark Results

Cortyx leads with retrieval-first proof, then backs it up with latency, token, and footprint measurements on the shipped path.

| Metric | Cortyx | MemPalace | engram | vestige | token-savior |
|--------|--------|-----------|--------|---------|--------------|
| LME-500 R@5 | **96.8%** ¹ | 96.6% | not benchmarked | not benchmarked | not benchmarked |
| LoCoMo recall | **92.0%** | — | — | — | — |
| Activation latency p95 | **~22ms** | ~200ms | — | — | — |
| Token savings (first call) | **56.9%** | — | — | — | — |
| Token savings (capsule+delta repeat) | **98.4%** | — | — | — | — |
| Binary size | **~30MB** (v0.4.0: TurboVec SIMD) | Python stack | ~12MB (Go) | ~8MB (Rust) | Python stack |
| Runtime model on default path | **No** | Yes | No | No | No |

> ¹ 96.8% R@5 is from the regenerated cleaned-oracle eval harness run (484/500 questions).
> The full 500-question benchmark runs via manual `workflow_dispatch`; the fast CI regression
> guard runs 20 questions per category. See [BENCHMARKS.md](BENCHMARKS.md) for full methodology.
> **Verify claims:** Run `cortyx proof-certificate` for a live reproducible summary; claims are sourced from `benchmarks/registry.json`.

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

## MCP Setup

### One-Command Setup (Recommended)
```bash
cortyx install
```
Detects Claude Code, Cursor, Windsurf, Codex, VS Code, and Zed config files automatically and writes the MCP entry + hook scripts into each. Registers four Claude Code hooks: `SessionStart` (auto-primes context index), `Stop`, `PreCompact`, and `PostToolUse` (quality-gated auto-capture of tool observations). Idempotent — safe to run multiple times.

### Manual Setup
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

Restart your LLM client — all Cortyx tools will appear automatically.

## MCP Tools

| Tool | Description |
|------|-------------|
| `cortyx` | Universal router — auto-selects best flow (retrieval, answer, wake-up, status) |
| `cortyx_get_contexts` | Retrieve 3–5 relevant neurons for the current task |
| `cortyx_get_evidence` | Structured evidence facts extracted from top-k neurons for a task |
| `cortyx_recall` | Conversation-memory recall, optionally scoped to a person |
| `cortyx_wake_up` | Prime LLM with project identity + critical facts at session start |
| `cortyx_list_modules` | List all modules with neuron counts and hit-rates |
| `cortyx_list_neurons` | List neuron paths + status for a module |
| `cortyx_peek_neuron` | Preview first N lines of a neuron |
| `cortyx_read_section` | Read a single named section (e.g. `purpose`, `api`) from a neuron |
| `cortyx_explore_tree` | Navigate the project's neuron hierarchy like a table of contents |
| `cortyx_search_literal` | Exact string search across all neuron bodies with surrounding context |
| `cortyx_search_regex` | Regex search across all neuron bodies with neuron-path results |
| `cortyx_search_raw` | Search raw source files directly (not neuron bodies) |
| `cortyx_list_persons` | List all `@person`-scoped memory namespaces |
| `cortyx_close_task` | Record which neurons helped at task end (zero-friction feedback) |
| `cortyx_evolve_context` | Rewrite a full neuron with improved reasoning |
| `cortyx_evolve_section` | Update one section of a neuron (~50 tokens) |
| `cortyx_extract_from_raw` | Save a proven code chunk as a use-case neuron |
| `cortyx_create_synapse` | Link two neurons |
| `cortyx_record_hit` | Manual feedback — boost or down-weight a neuron |
| `cortyx_invalidate` | Force stale mark on a neuron |
| `cortyx_status` | Neuron count, synapse count, freshness breakdown |
| `cortyx_mine_conversation` | Mine a conversation turn into a Verbatim neuron |
| `cortyx_rollback_section` | Restore a neuron section from shadow history |
| `cortyx_diary_write` | Write a structured agent diary entry |
| `cortyx_diary_read` | Read recent diary entries for an agent |
| `cortyx_diary_refine` | Analyse a recent diary entry and populate `refined_plan` with a heuristic decomposition suggestion |
| `cortyx_diary_consolidate` | Promote frequently-used diary entries to permanent Verbatim neurons |
| `cortyx_session_timeline` | Chronological timeline of diary entries, activated neurons, and KG facts |
| `cortyx_agent_status` | Show latest agent-state snapshot |
| `cortyx_collaboration_status` | Summarize collaboration-kernel state across agents, shared modules, and sync activity |
| `cortyx_check_consistency` | Scan for contradicting neurons |
| `cortyx_fleet_query` | Query registered fleet nodes for cross-project context |
| `cortyx_fleet_status` | List all registered fleet nodes |
| `cortyx_fleet_register` | Register a local project or git-backed corpus as a fleet node |
| `cortyx_kg_add` | Add a temporal fact to the knowledge graph |
| `cortyx_kg_query` | Query active KG facts for an entity |
| `cortyx_kg_invalidate` | End/supersede an active KG fact |
| `cortyx_kg_timeline` | Show temporal history of a KG predicate |
| `cortyx_kg_stats` | Aggregate KG statistics |

→ Full parameter signatures and examples: [ARCHITECTURE.md](ARCHITECTURE.md#mcp-tools-reference)

Structured agent diaries are **not** a separate database. Cortyx stores them in the existing `@agent/{name}` namespace and mirrors the latest structured fields into a normal KG entity, so specialist-agent handoff stays local, temporal, and queryable through the same proof surface.

The shared concept layer stays equally inspectable. `cortyx concepts ready` surfaces only Core/Concept neurons that have already proven useful locally, and `cortyx concepts publish-ready` batches those into the shared library when it is git-backed.

## CLI Reference

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
cortyx mine-observation            # Mine a single tool observation (reads stdin; used by PostToolUse hook)
cortyx precompact-snapshot         # Snapshot activated session context before compaction (used by PreCompact hook)
cortyx timeline [--since 2d] [--agent <name>] [--limit N]  # Session timeline: diary + activated neurons + KG facts
cortyx consolidate [--min-refs N] [--dry-run]  # Promote diary entries used >= N times to permanent neurons
cortyx insights [--since 2w] [--top N]         # Neuron health: top activated, stalest, per-module stats
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
cortyx fleet register <path> [--alias <name>]  # Register a peer project as a fleet node
cortyx fleet deregister <alias-or-path>        # Remove a node from the fleet
cortyx fleet list                              # Show all registered fleet nodes
cortyx fleet status                            # Fleet health summary (node count, modules)
cortyx proof-certificate            # Print live benchmark proof sourced from benchmarks/registry.json
cortyx proof-certificate --validate # Exit 1 if any metric is unmeasured (CI gate)
cortyx serve --frozen               # Start MCP server in frozen mode (all feedback writes disabled)
cortyx install                     # Auto-configure all detected LLM clients
```

## How It Works

Cortyx runs a ≤40ms activation pipeline: hybrid BM25 + dense retrieval (embed and rerank are enabled by default; compile with `--no-default-features` for a lighter binary), synapse graph traversal (up to 3 hops), and a 12-stage query-context pipeline. On each query, 3–5 neurons are selected, ordered by relevance, and injected after the prompt-cache breakpoint — the static prefix stays byte-identical so provider caches always hit, typically saving 56–98% of input tokens on repeat calls. `cortyx_close_task` records which neurons helped, feeding a self-improving ranking loop.

Neurons are plain Markdown files in `.cortyx/neurons/` — human-readable, git-tracked, editable.

→ Deep dive — activation pipeline, storage format, vocabulary bridge, advanced features: [ARCHITECTURE.md](ARCHITECTURE.md)

## Neuron Types

| Kind | File pattern | Purpose |
|------|-------------|---------|
| **Core** | `src_engine_rs.context.md` | AI-curated reasoning guide for one source file |
| **UseCase** | `src_engine_rs.usecase.dark-mode.md` | Exact proven chunk for a recurring pattern |
| **Verbatim** | `__verbatim_*.context.md` | Mined conversation turns for semantic recall |
| **Concept** | `__concept_*.context.md` | Cross-cutting concept (auth flow, DB migrations) |
| **Project** | `_project.context.md` | Top-level project description + conventions |

## Documentation

| Document | Description |
|---|---|
| [BENCHMARKS.md](BENCHMARKS.md) | Full benchmark results, methodology, comparison tables |
| [BENCHMARKING.md](BENCHMARKING.md) | How to run benchmarks and submit results |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Module map and design decisions |
| [MIGRATION.md](MIGRATION.md) | v0.2.0 → v0.3.0 breaking changes and migration steps |
| [CHANGELOG.md](CHANGELOG.md) | Release history |

## License

MIT — see [LICENSE](LICENSE)
