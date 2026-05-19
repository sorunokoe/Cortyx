# Migration Guide

← [Back to README](README.md)

This document covers breaking changes and migration steps when upgrading Cortyx.

---

## v0.2.0 → v0.3.0

### 1. `embed` and `rerank` are now default features

**What changed:** The `embed` (dense ONNX embeddings) and `rerank` (cross-encoder
re-ranking) features are now compiled in by default.

**Impact on users:** On first startup after upgrade, Cortyx downloads ~130 MB of
model weights if they are not already cached.

**To opt out (air-gap / CI environments):**

```bash
# Build without ONNX models
cargo build --no-default-features

# Serve without embedding models
CORTYX_NO_DOWNLOAD=1 cortyx serve
```

**CI / Clippy:** Always use `--no-default-features` in CI to avoid the model
download:

```yaml
- run: cargo clippy --no-default-features -- -D warnings
- run: cargo test --no-default-features
```

---

### 2. Index storage version bump (v8 → v9)

**What changed:** The in-memory index format was bumped to version 9 as part of
the QueryContext pipeline refactor.

**Impact:** On first run, Cortyx detects the version mismatch and rebuilds the
index from your `.cortyx/neurons/` source files. This takes 1–10 seconds
depending on corpus size.

**No manual action required.** Your neuron `.context.md` files are the source of
truth; nothing is lost during the migration.

If you see:

```
[WARN] Migrated index from v8 to v9 (structural change; rebuilt from neurons)
```

this is expected and normal.

---

### 3. `src/mcp/tools/context.rs` is now a directory

**Impact on contributors only.** If you have a local branch that touched
`src/mcp/tools/context.rs`, you will get a merge conflict. The file has been
split into:

```
src/mcp/tools/context/
├── mod.rs             — thin orchestrator
├── inflight_guard.rs  — RAII byte-cap guard
├── session_decay.rs   — session TF + path-history decay
└── answer_mode.rs     — answer_mode dispatch
```

Resolve by rebasing onto `master` and re-applying your changes to the appropriate
sub-module.

---

### 4. Lint level changes

**What changed:** Two Clippy lints were promoted:

| Lint | Before | After |
|---|---|---|
| `cast_possible_truncation` | warn | **deny** |
| `unwrap_used` | allow | warn |
| `missing_docs` | allow | warn |

**Impact on contributors:** Code that previously compiled with warnings may now
fail to compile (`cast_possible_truncation`) or emit new warnings
(`unwrap_used`, `missing_docs`).

**Fix `cast_possible_truncation`:**

```rust
// Before (deny in v0.3.0):
let n = some_usize as u32;

// After:
let n = u32::try_from(some_usize).unwrap_or(u32::MAX);
// or, if you've verified the range:
let n = some_usize as u32; // safe: value fits in u32 because …
#[allow(clippy::cast_possible_truncation)]
```

**Fix `unwrap_used`:**

Prefer `expect("reason")` over `.unwrap()` and propagate errors with `?` where
possible.

---

### 5. LME-500 regression guard thresholds raised

**Impact on contributors running benchmarks:**

| Threshold | v0.2.0 | v0.3.0 |
|---|---|---|
| Rows per category | 5 (20 total) | 20 (80 total) |
| SSU accuracy | 80% | 85% |
| KU accuracy | 60% | 65% |
| Quick-mode rows/cat | 2 | 5 |
| Quick-mode SSU/KU | 50% | 60% |

If you have local changes that reduce retrieval accuracy, the regression guard
will now fail at these higher thresholds. Run:

```bash
cargo test --no-default-features --test bench bench_lme_regression_guard -- --nocapture
```

---

## v0.1.0 → v0.2.0

See [CHANGELOG.md § 0.2.0](CHANGELOG.md#020--2026-05-11) for the full list of
changes. The main breaking change was the monolith extraction: if you have a
local fork that patched `src/index/core/helpers.rs` or `src/index/core/synthetic.rs`,
those files were split into multiple focused modules — rebase required.
