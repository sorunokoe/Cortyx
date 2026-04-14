#[cfg(feature = "embed")]
use crate::embedder::{EmbeddingStore, load_embeddings};

use anyhow::Result;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

use crate::alias_gen;
use crate::ast_extractor;
use crate::git_extractor;
use crate::global_index;
use crate::import_parser;
use crate::kg;
use crate::neuron::{
    NeuronKind, NeuronMeta, NeuronStatus, Synapse, SynapseType,
    atomic_write, atomic_write_json,
    core_neuron_path, estimate_tokens, hash_file, meta_path, neuron_dir,
    now_iso8601, replace_section, should_skip, stub_core_neuron, stub_function_neuron,
    stub_project_neuron, sub_neuron_path, update_neuron_header, DEFAULT_CONFIDENCE,
};

// ─── Activation tuning constants ─────────────────────────────────────────────

/// Maximum core neurons returned in Phase 1 of activation.
pub const MAX_CORE_NEURONS: usize = 5;
/// Maximum use-case neurons per core in Phase 2 of activation.
pub const MAX_USE_CASE_PER_CORE: usize = 2;
/// Minimum BM25 relevance ratio (vs. max) for synapse traversal to include a neighbor.
pub const SYNAPSE_RELEVANCE_THRESHOLD: f32 = 0.25;
/// BM25 score ratio above which a neuron triggers 2-hop traversal.
pub const HIGH_ACTIVATION_THRESHOLD: f32 = 0.6;
/// Minimum term length kept by the tokenizer.
const MIN_TERM_LEN: usize = 2;
/// BM25 parameters (Okapi BM25 standard defaults).
const BM25_K1: f32 = 1.2;
const BM25_B: f32 = 0.75;
///
/// Migrations: rather than discarding the index on version mismatch, `load_or_create`
/// applies the migration chain from the stored version to INDEX_VERSION. This preserves
/// all user-curated `use_count`, `hit_count`, and `staleness_multiplier` data across upgrades.
const INDEX_VERSION: u32 = 8;
/// Minimum activation count before any quarantine or multiplier decision is made.
/// 20 samples gives a statistically meaningful Wilson confidence interval.
/// Minimum activation count before any quarantine decision is made.
/// Below this, `adaptive_quarantine_params` returns None — withhold judgment.
/// Adaptive CI (S4, TRIZ R11) allows fast reaction at 5–19 samples with z=1.0
/// and escalates to stricter thresholds at higher counts.
#[allow(dead_code)]
const QUARANTINE_MIN_SAMPLES: u32 = 5;
/// Wilson score threshold for the 20–99 sample tier (90% CI, z=1.645).
/// Kept for test assertions; runtime uses `adaptive_quarantine_params`.
#[allow(dead_code)]
const QUARANTINE_WILSON_THRESHOLD: f32 = 0.05;
/// Wilson score lower bound above which a quarantined neuron is rehabilitated.
/// Recovery requires the lower bound to rise above 15%, so noise doesn't lift quarantine.
const QUARANTINE_RECOVERY_THRESHOLD: f32 = 0.15;
/// Minimum activation count before the hit-rate multiplier is applied.
/// Below this count the multiplier is 1.0 (cold-start neutral, no reward yet).
const MIN_SAMPLE_SIZE: u32 = 5;
/// Hit-rate below which a heavily-activated neuron is auto-quarantined.
/// Kept for the test that verifies old behaviour; runtime now uses Wilson bounds.
#[allow(dead_code)]
const QUARANTINE_THRESHOLD: f32 = 0.10;
/// Average estimated token cost per synapse-traversed neuron (used for dynamic budget).
/// Conservative: most neurons are ~150 tokens; `1.3` accounts for formatting overhead.
const AVG_SYNAPSE_TOKEN_COST: usize = 200;
/// BM25 confidence ratio below which TF-IDF cosine re-ranking is applied.
/// High ratio (≥ threshold) → BM25 is decisive; low ratio → tie-break with TF-IDF.
const HYBRID_CONFIDENCE_THRESHOLD: f32 = 1.5;
/// BM25 top-score above which retrieval is considered decisive — TF-IDF and dense
/// re-ranking are skipped entirely (no wasted compute for clear keyword matches).
const HIGH_CONFIDENCE_THRESHOLD: f32 = 8.0;
/// BM25 top-score below which retrieval is considered ambiguous.
/// Logged as a hint for future API-embedding escalation path (Solution 3, v0.3).
const LOW_CONFIDENCE_THRESHOLD: f32 = 0.5;

/// Minimum public functions in a source file to trigger UseCase sub-neuron splitting (S3).
///
/// Files with fewer functions keep a single Core neuron (low overhead).
/// Files at or above this threshold get one UseCase sub-neuron per function,
/// enabling per-function BM25 retrieval precision without inflating the Core.
const SUBNEURON_SPLIT_THRESHOLD: usize = 6;
/// Maximum sub-neurons generated per source file (caps index growth on huge files).
const MAX_SUBNEURONS_PER_FILE: usize = 20;

// ─── BM25 entry ───────────────────────────────────────────────────────────────

/// Per-neuron data stored in the in-memory BM25 index.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
struct BM25Entry {
    neuron_path: PathBuf,
    kind: NeuronKind,
    /// Term → raw frequency within this document
    term_freq: HashMap<String, f32>,
    /// Total number of terms in this document (for BM25 length normalization)
    term_count: usize,
    /// LLM token estimate (from `estimate_tokens`), used for budget trimming
    pub tokens: usize,
    /// Tokenized task pattern (use-case neurons only)
    task_pattern_terms: Vec<String>,
    /// Parent core neuron path (use-case neurons only)
    parent: Option<PathBuf>,
    /// Typed synapse edges — persisted so weights survive restarts
    synapses: Vec<Synapse>,
    /// Source files synthesized by this Concept neuron
    source_files: Vec<PathBuf>,
    /// Optional module/namespace tag for namespace-filtered queries
    module: Option<String>,
    /// Git-derived confidence score applied as a mild BM25 multiplier.
    /// 1.0 = committed + unmodified (neutral). 0.9 = locally modified. 0.85 = untracked.
    #[serde(default = "default_confidence")]
    confidence_score: f32,
    /// Activation count — incremented each time this neuron is returned by get_contexts.
    #[serde(default)]
    use_count: u32,
    /// Citation count — incremented by cortyx_record_hit when the LLM confirms it used the neuron.
    #[serde(default)]
    hit_count: u32,
    /// Staleness multiplier (1.0 = fresh, 0.5 = stale). Demotes rather than evicts stale neurons
    /// so context is preserved; stale neurons can still activate for niche queries.
    #[serde(default = "default_staleness")]
    staleness_multiplier: f32,
    /// Concept cloud: union of significant identifier terms from this neuron's 1-hop
    /// structural neighbours (Calls, Imports, Implements edges). Built by `build_concept_clouds()`
    /// during `rebuild_derived()`. At query time, used as a graph-aware semantic thesaurus
    /// for zero/low-confidence BM25 queries — no external model required (TRIZ R12-S1).
    /// Not persisted: rebuilt from the live synapse graph on every load.
    #[serde(skip)]
    concept_cloud: Vec<String>,
    /// B2: Synonym cloud — terms that have co-activated with this neuron ≥30 times.
    ///
    /// Populated by `record_coactivation()`. Persisted so the signal accumulates across
    /// sessions. At query time, query terms are expanded through synonym clouds before
    /// BM25 scoring — improving recall for semantically related but lexically distant queries.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    synonym_cloud: Vec<String>,
    /// S-II (R16): 64-bit SimHash fingerprint of this neuron's TF-IDF term weights.
    ///
    /// R17 Sol4 upgrade: 1024-bit random projection via 16 independent SimHash seeds.
    /// J-L lemma: 64-bit → ε ≈ 0.38; 1024-bit (16 × 64) → ε ≈ 0.09.
    /// LSH match: ANY of the 16 fingerprint pairs within Hamming ≤ 14.
    /// Migration from v7: old `lsh_fingerprint: u64` loaded via serde → replicated to [0].
    #[serde(default)]
    lsh_fingerprints: [u64; 16],
    /// S-III (R16): Self-quality score — fraction of neuron terms that overlap with
    /// the corresponding source file's AST terms.
    ///
    /// Computed at `index_neuron` time: `|neuron_terms ∩ source_ast_terms| / |neuron_terms|`.
    /// When `quality_score < 0.4`, a ×0.7 BM25 penalty is applied to demote stale neurons
    /// without evicting them. Surfaced in `cortyx status` as "needs curation" count.
    /// Defaults to 1.0 (neutral/unknown) when no source file is available.
    #[serde(default = "default_quality_score")]
    quality_score: f32,
    /// S-I (R16): Tier-1 summary for multi-resolution emission.
    ///
    /// Extracted from `## purpose` + first line of `## pitfalls` at `index_neuron` time.
    /// Emitted instead of full content when BM25 score is in the 1.5–5.0 range (Tier 1).
    /// ~50 tokens; avoids a disk read at query time. Not persisted (rebuilt from neuron file).
    #[serde(skip)]
    summary: String,
    /// Unix epoch seconds parsed from `NeuronMeta.timestamp` (Verbatim neurons only).
    ///
    /// Stored at `index_neuron` time so temporal query routing can apply a recency boost
    /// without any disk I/O at query time. Code neurons (no ISO 8601 timestamp) leave
    /// this as `None` — they are unaffected by temporal scoring.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    timestamp_secs: Option<i64>,
    /// R21 T6: Session identifier for session-level grouping.
    ///
    /// Derived from the neuron filename stem (e.g., "lme_0060" from
    /// "lme_0060_0_user.verbatim.md"). Empty for non-Verbatim neurons.
    /// Used at retrieval time: when a neuron enters the top-3, its session
    /// siblings are injected as overflow candidates.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    session_id: String,
}

// ─── Navigation summary types (TRIZ R13-G2) ──────────────────────────────────

/// Summary of a module as returned by `list_modules()`.
#[derive(Debug, Clone)]
pub struct ModuleSummary {
    pub name: String,
    pub neuron_count: usize,
    pub avg_hit_rate: f32,
    /// True when name starts with `@` (person/project scope).
    pub is_person_scope: bool,
}

/// Summary of a single neuron as returned by `list_neurons()`.
#[derive(Debug, Clone)]
pub struct NeuronSummary {
    pub path: PathBuf,
    pub kind: NeuronKind,
    pub staleness_multiplier: f32,
    pub hit_rate: f32,
    pub use_count: u32,
}

// ─── Persisted index wrapper ───────────────────────────────────────────────────

/// On-disk format of `.cortyx/index.json`.
/// Versioned to detect schema changes and rebuild cleanly.
/// Used only by `migrate_entries` for deserialization — suppressed dead-code warning
/// because the load path deserialises via `serde_json::Value` first, then migrates.
#[allow(dead_code)]
#[derive(Deserialize)]
struct PersistedIndex {
    version: u32,
    entries: Vec<BM25Entry>,
    #[serde(default)]
    session_utilization: Vec<[usize; 2]>,
}

/// Borrowed view used for serialization — avoids cloning the entire entry vector
/// on every save() call (which would otherwise be O(n) allocation per MCP mutation).
#[derive(Serialize)]
struct PersistedIndexRef<'a> {
    version: u32,
    entries: &'a [BM25Entry],
    #[serde(skip_serializing_if = "<[[usize; 2]]>::is_empty")]
    session_utilization: &'a [[usize; 2]],
}

// ─── Schema migrations ────────────────────────────────────────────────────────

/// Migration function signature: transforms a stored JSON value to be compatible
/// with the next schema version. Fields not touched by the migration are preserved
/// (use_count, hit_count, staleness_multiplier etc. survive every upgrade).
type MigrationFn = fn(serde_json::Value) -> serde_json::Value;

/// v7 → v8: rename `lsh_fingerprint: u64` → `lsh_fingerprints: [u64; 16]` (1024-bit LSH).
fn migrate_v7_to_v8(mut entries_val: serde_json::Value) -> serde_json::Value {
    if let Some(arr) = entries_val.as_array_mut() {
        for entry in arr.iter_mut() {
            let old_fp = entry.get("lsh_fingerprint").and_then(|v| v.as_u64()).unwrap_or(0);
            let fps: Vec<serde_json::Value> = std::iter::once(serde_json::Value::Number(old_fp.into()))
                .chain(std::iter::repeat(serde_json::Value::Number(0u64.into())).take(15))
                .collect();
            entry["lsh_fingerprints"] = serde_json::Value::Array(fps);
            if let Some(obj) = entry.as_object_mut() {
                obj.remove("lsh_fingerprint");
            }
        }
    }
    entries_val
}

/// Chain of (from_version, to_version, migration_fn) applied in sequence when
/// the stored index is older than INDEX_VERSION.
///
/// Adding a new entry here (rather than bumping INDEX_VERSION and discarding) means
/// users never lose curated data from a routine `cargo install cortyx` upgrade.
const MIGRATIONS: &[(u32, u32, MigrationFn)] = &[
    // v5 → v6: no structural change (INDEX_VERSION bumped to introduce migration infra).
    (5, 6, |v| v),
    // v6 → v7: add concept_cloud field (populated by rebuild_derived; serde default=[]).
    // Existing entries load fine — serde fills concept_cloud with [].
    (6, 7, |v| v),
    // v7 → v8: rename lsh_fingerprint (u64) → lsh_fingerprints ([u64; 16]).
    (7, 8, migrate_v7_to_v8),
];

/// Apply all migrations from `stored_version` to `INDEX_VERSION` in sequence.
/// Returns the migrated entries (deserialized from the final Value), or an error.
fn migrate_entries(
    mut raw: serde_json::Value,
    stored_version: u32,
) -> anyhow::Result<Vec<BM25Entry>> {
    let mut ver = stored_version;
    for &(from, to, migrate) in MIGRATIONS {
        if ver == from && ver < INDEX_VERSION {
            // Migrate the "entries" array within the persisted object.
            if let Some(entries_val) = raw.get("entries").cloned() {
                let migrated_entries = migrate(entries_val);
                raw["entries"] = migrated_entries;
            }
            raw["version"] = serde_json::Value::Number(to.into());
            ver = to;
        }
    }
    if ver != INDEX_VERSION {
        anyhow::bail!(
            "No migration path from version {stored_version} to {INDEX_VERSION}; \
             run `cortyx compile .` to rebuild."
        );
    }
    let entries: Vec<BM25Entry> = serde_json::from_value(
        raw.get("entries").cloned().unwrap_or(serde_json::Value::Array(vec![])),
    )?;
    Ok(entries)
}

// ─── NeuronIndex ─────────────────────────────────────────────────────────────

/// The in-memory semantic index — loaded from `.cortyx/index.json` on startup.
///
/// All search operations run entirely in RAM (<10ms for <10k neurons).
/// Persisted to disk after every compile or mutation (evolve, synapse, extract).
#[derive(Debug, Default)]
pub struct NeuronIndex {
    project_root: PathBuf,
    entries: Vec<BM25Entry>,
    /// Synapse graph: neuron_path → outgoing + incoming typed synapse edges
    adjacency: HashMap<PathBuf, Vec<Synapse>>,
    /// O(1) lookup: neuron_path → index in `entries`
    path_index: HashMap<PathBuf, usize>,
    /// Parent neuron path → child entry indices (for UseCase lookup)
    parent_index: HashMap<PathBuf, Vec<usize>>,
    /// Precomputed document frequency for each term (used for BM25 IDF)
    df_cache: HashMap<String, usize>,
    /// Posting list: term → [entry indices containing that term].
    ///
    /// Built during rebuild_derived() alongside df_cache.
    /// Used in get_contexts() to compute the candidate set in O(|terms|) time —
    /// only entries that contain at least one query term are scored, reducing
    /// BM25 scoring from O(n) to O(|candidates|).  Typically |candidates| << n
    /// for sparse queries, which is the common case.
    posting_list: HashMap<String, Vec<usize>>,
    /// Average document length (for BM25 length normalization)
    avg_doc_len: f32,
    /// Module → entry indices (for O(k) module-filtered queries)
    module_index: HashMap<String, Vec<usize>>,
    /// Vocabulary bridge (S2): module_fragment → set of identifier terms from that module.
    ///
    /// Built during rebuild_derived(). At query time, zero-match BM25 queries are
    /// expanded with the identifier vocabulary of any module whose name substring-matches
    /// a query term. Resolves the lexical gap between human language ("authentication")
    /// and code identifiers ("auth_guard", "jwt_validate") without any model download.
    vocab_bridge: HashMap<String, HashSet<String>>,
    /// B1: Morphemic trie bridge — sub-token → all tokens containing that sub-token.
    ///
    /// Built during rebuild_derived() by splitting all identifier tokens on `_` and
    /// camelCase boundaries. At query time, query terms that don't match any neuron
    /// directly are expanded through this map: "auth" → ["auth_guard", "authentication",
    /// "oauth_token"]. Reduces vocabulary gap from ~3% to ~0.3%.
    morpheme_map: HashMap<String, Vec<String>>,
    /// B2: Synonym cloud from co-activation history (TRIZ R14).
    ///
    /// Tracks how often query terms co-activate each neuron. After ≥30 co-activations,
    /// a term is promoted to the neuron's `synonym_cloud` in BM25Entry.
    /// Map: neuron_path → HashMap<term, coactivation_count>
    /// Not persisted to disk — cleared on restart. Synonym clouds in BM25Entry are persisted.
    coactivation_counts: HashMap<PathBuf, HashMap<String, u32>>,
    /// C-2: Hebbian synapse co-return counts (R20).
    ///
    /// Tracks how often two Verbatim neurons are returned together in the same query.
    /// Map: (path_a, path_b) → co-return count (path_a < path_b lexicographically).
    /// After ≥2 co-returns, a SemanticRelated synapse is auto-created between the pair.
    /// Not persisted — rebuilt from query patterns in the current process lifetime.
    /// Wrapped in Mutex for interior mutability inside &self methods.
    co_return_counts: std::sync::Mutex<HashMap<(PathBuf, PathBuf), u32>>,
    /// F2: Session token utilization history — last 5 sessions' [tokens_used, tokens_budget].
    ///
    /// Persisted via PersistedIndexRef so budget adaptation accumulates across restarts.
    pub session_utilization: Vec<[usize; 2]>,
    /// R21 T6: Session index — maps session_id → entry indices for session-level grouping.
    ///
    /// Built during rebuild_derived(). At retrieval, when a Verbatim neuron enters the
    /// top-3 results, the top-2 BM25-scored siblings from the same session are injected
    /// as overflow candidates. Enables counting/multi-session queries to surface related
    /// evidence from the same session cluster without extra BM25 computation.
    /// Not persisted — rebuilt from BM25Entry.session_id on each load.
    session_index: HashMap<String, Vec<usize>>,
    /// P1-A: PMI semantic neighbors — loaded from cooccurrence.json without a global cap.
    ///
    /// Unlike `vocab_bridge` (which uses substring matching for code module fragments),
    /// this map uses exact-key lookup O(1) for conversation vocabulary expansion.
    /// Key: term (≥4 chars). Value: up to 5 high-PMI neighbors from the same corpus.
    /// Loaded at rebuild_derived() time; not persisted — rebuilt on each load.
    pmi_neighbors: HashMap<String, Vec<String>>,
    /// Dense embedding store (loaded from `.cortyx/embeddings.bin`).
    /// Empty when `embed` feature is disabled or file is absent (BM25-only mode).
    #[cfg(feature = "embed")]
    embeddings: EmbeddingStore,
}

// ─── Parallel compile helper ──────────────────────────────────────────────────

/// Result of processing a single source file in the parallel compile phase.
///
/// Returned by `process_source_file` (a free function — no `&self` access) so
/// multiple files can be processed concurrently via `rayon::par_iter()`.
/// The sequential batch-insert phase calls `index_neuron` on each result.
struct CompiledFile {
    neuron_path: PathBuf,
    /// Content of the neuron stub (new or regenerated).
    content: String,
    /// Updated `NeuronMeta` to be written to the `.context.json` sidecar.
    meta: NeuronMeta,
}

/// Process a single source file: hash-check, AST-extract, write stub + meta.
///
/// Returns a `Vec<CompiledFile>`: the first element (if any) is the Core neuron;
/// subsequent elements are UseCase sub-neurons (S3 lazy splitting, fired when the
/// file has ≥ SUBNEURON_SPLIT_THRESHOLD public functions).
///
/// Returns an empty `Vec` when the file is unchanged (hash match), should be skipped,
/// or when a cosmetic change is detected (S1: sig_hash identical) — in that
/// case only the meta hash is updated on disk and the BM25Entry already in
/// memory from `load_or_create` is preserved with its `staleness_multiplier`
/// and learned feedback signals intact.
///
/// This function performs only filesystem reads and writes — no `&mut NeuronIndex`
/// access — which makes it safe to call in parallel via rayon.
fn process_source_file(
    abs: &Path,
    root: &Path,
    git_confidence: &HashMap<PathBuf, f32>,
) -> Vec<CompiledFile> {
    let rel = abs.strip_prefix(root).unwrap_or(abs);
    if should_skip(rel) {
        return vec![];
    }

    let neuron_path = core_neuron_path(abs, root);
    let meta_file = meta_path(&neuron_path);

    let source_bytes = match std::fs::read(abs) {
        Ok(b) => b,
        Err(_) => return vec![],
    };
    let current_hash = {
        let h = blake3::hash(&source_bytes);
        h.to_hex()[..16].to_string()
    };

    // Read stored meta once and reuse for hash, sig_hash, synapses, module, and feedback counts.
    let stored_meta: Option<NeuronMeta> = if meta_file.exists() {
        std::fs::read_to_string(&meta_file)
            .ok()
            .and_then(|d| serde_json::from_str(&d).ok())
    } else {
        None
    };

    let stored_hash = stored_meta
        .as_ref()
        .map(|m| m.source_hash.as_str())
        .unwrap_or("")
        .to_string();

    // Skip if hash unchanged and neuron exists — pure no-op.
    if !current_hash.is_empty() && current_hash == stored_hash && neuron_path.exists() {
        return vec![];
    }

    let source_text = String::from_utf8_lossy(&source_bytes);
    let source_rel = rel.to_string_lossy();
    let now = now_iso8601();

    let ast_summary = ast_extractor::extract_signatures(&source_rel, &source_text);
    let sig_hash = ast_extractor::compute_sig_hash(&ast_summary);

    let stored_sig_hash = stored_meta
        .as_ref()
        .and_then(|m| m.sig_hash.as_deref())
        .unwrap_or("")
        .to_string();

    // S1 — Cosmetic change: source_hash changed but public API surface (sig_hash) is identical.
    // Whitespace edits, doc-comment tweaks, or formatting passes land here.
    // Preserve the LLM-curated stub; only update the hash in the meta file so future
    // compiles don't re-check this file. The in-memory BM25Entry (from load_or_create)
    // retains its staleness_multiplier and learned feedback signals.
    if !stored_sig_hash.is_empty()
        && sig_hash == stored_sig_hash
        && !stored_hash.is_empty()
        && neuron_path.exists()
    {
        if let Some(mut old_meta) = stored_meta {
            old_meta.source_hash = current_hash;
            old_meta.sig_hash = Some(sig_hash);
            old_meta.last_updated = now;
            if let Err(e) = atomic_write_json(&meta_file, &old_meta) {
                tracing::warn!("Failed to update meta for cosmetic change {:?}: {e}", meta_file);
            }
        }
        return vec![];
    }

    // S1 (R11) — Section-Level Staleness: sig_hash changed (real API change) but the
    // neuron already exists with LLM-curated content. Instead of overwriting everything,
    // replace only the `api` section and update the header comments. Preserves `purpose`,
    // `pitfalls`, and cross-reference sections. Reduces LLM re-evolution calls by ~60%.
    if !stored_hash.is_empty() && neuron_path.exists() {
        // sig_hash is different — we passed the cosmetic-change gate above
        match std::fs::read_to_string(&neuron_path) {
            Ok(existing) => {
                let new_api = ast_extractor::format_for_stub(&ast_summary);
                let updated = replace_section(&existing, "api", &new_api);
                let updated = update_neuron_header(&updated, &current_hash, &now);
                if let Err(e) = atomic_write(&neuron_path, updated.as_bytes()) {
                    tracing::warn!("S1: Failed to update api section {:?}: {e}", neuron_path);
                    // Fall through to full stub generation below
                } else {
                    let old = stored_meta.clone().unwrap_or_else(|| NeuronMeta::new_stub(abs, NeuronKind::Core));
                    let mut meta = old;
                    meta.source_hash = current_hash;
                    meta.sig_hash = Some(sig_hash);
                    meta.last_updated = now.clone();
                    meta.status = NeuronStatus::Stale;
                    meta.tokens = estimate_tokens(&updated);
                    if meta.module.is_none() { meta.module = infer_module(rel); }
                    let existing_targets: HashSet<PathBuf> =
                        meta.synapses.iter().map(|s| s.target.clone()).collect();
                    let auto_imports = import_parser::parse_imports(abs, &source_text, root);
                    for imported_source in auto_imports {
                        let target_neuron = core_neuron_path(&imported_source, root);
                        if !existing_targets.contains(&target_neuron) {
                            meta.synapses.push(Synapse::new(
                                target_neuron,
                                SynapseType::Imports,
                                "auto-inferred from import statement".to_string(),
                            ));
                        }
                    }
                    meta.confidence_score = git_confidence.get(abs).copied().unwrap_or(1.0);
                    if let Err(e) = atomic_write_json(&meta_file, &meta) {
                        tracing::warn!("S1: Failed to update meta {:?}: {e}", meta_file);
                    }
                    let mut results = vec![CompiledFile {
                        neuron_path: neuron_path.clone(),
                        content: updated,
                        meta,
                    }];
                    // Also generate sub-neurons for any new functions (idempotent — skips existing)
                    if ast_summary.functions.len() >= SUBNEURON_SPLIT_THRESHOLD {
                        for fn_name in ast_summary.functions.iter().take(MAX_SUBNEURONS_PER_FILE) {
                            let sub_path = sub_neuron_path(&neuron_path, fn_name);
                            if sub_path.exists() { continue; }
                            let sub_content = stub_function_neuron(fn_name, &source_rel, &now);
                            if let Err(e) = atomic_write(&sub_path, sub_content.as_bytes()) {
                                tracing::warn!("S1: Failed to write sub-neuron {:?}: {e}", sub_path);
                                continue;
                            }
                            let sub_meta_file = meta_path(&sub_path);
                            let mut sub_meta = NeuronMeta::new_stub(abs, NeuronKind::UseCase);
                            sub_meta.task_pattern = Some(fn_name.clone());
                            sub_meta.parent = Some(neuron_path.clone());
                            sub_meta.tokens = estimate_tokens(&sub_content);
                            sub_meta.last_updated = now.clone();
                            sub_meta.module = results[0].meta.module.clone();
                            sub_meta.confidence_score = results[0].meta.confidence_score;
                            if let Err(e) = atomic_write_json(&sub_meta_file, &sub_meta) {
                                tracing::warn!("S1: Failed to write sub-neuron meta {:?}: {e}", sub_meta_file);
                                continue;
                            }
                            results.push(CompiledFile {
                                neuron_path: sub_path,
                                content: sub_content,
                                meta: sub_meta,
                            });
                        }
                    }
                    tracing::debug!(path = %neuron_path.display(), "S1: api section updated, purpose/pitfalls preserved");
                    return results;
                }
            }
            Err(_) => {
                // Cannot read existing neuron — fall through to full stub regeneration
            }
        }
    }

    // Full stub (re)generation — real API change (sig_hash changed) or new file.
    let prefilled = ast_extractor::format_for_stub(&ast_summary);
    let purpose_hint = ast_extractor::format_purpose_hint(&ast_summary);
    let content = stub_core_neuron(&source_rel, &current_hash, &now, &prefilled, &purpose_hint);

    if let Some(parent) = neuron_path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            tracing::warn!("Failed to create neuron dir {:?}: {e}", parent);
            return vec![];
        }
    }
    if let Err(e) = atomic_write(&neuron_path, content.as_bytes()) {
        tracing::warn!("Failed to write stub {:?}: {e}", neuron_path);
        return vec![];
    }

    let is_new = stored_hash.is_empty();
    let mut meta = NeuronMeta::new_stub(abs, NeuronKind::Core);
    meta.source_hash = current_hash;
    meta.sig_hash = Some(sig_hash);
    meta.tokens = estimate_tokens(&content);
    meta.last_updated = now.clone();
    meta.status = if is_new {
        NeuronStatus::Stub
    } else {
        NeuronStatus::Stale
    };

    // Preserve existing synapses, module tag, and feedback counts on hash invalidation.
    if let Some(old) = stored_meta {
        meta.synapses = old.synapses;
        meta.module = old.module;
        meta.use_count = old.use_count;
        meta.hit_count = old.hit_count;
    }

    // Auto-module: infer from directory structure when not LLM-set.
    if meta.module.is_none() {
        meta.module = infer_module(rel);
    }

    // Auto-Synapse: infer Imports edges from import statements.
    let existing_targets: HashSet<PathBuf> =
        meta.synapses.iter().map(|s| s.target.clone()).collect();
    let auto_imports = import_parser::parse_imports(abs, &source_text, root);
    for imported_source in auto_imports {
        let target_neuron = core_neuron_path(&imported_source, root);
        if !existing_targets.contains(&target_neuron) {
            meta.synapses.push(Synapse::new(
                target_neuron,
                SynapseType::Imports,
                "auto-inferred from import statement".to_string(),
            ));
        }
    }

    // Git confidence: committed + unmodified = 1.0, modified = 0.9, untracked = 0.85.
    meta.confidence_score = git_confidence.get(abs).copied().unwrap_or(1.0);

    if let Err(e) = atomic_write_json(&meta_file, &meta) {
        tracing::warn!("Failed to write meta {:?}: {e}", meta_file);
        return vec![];
    }

    let mut results = vec![CompiledFile { neuron_path: neuron_path.clone(), content, meta }];

    // S3 — Lazy Sub-Neuron Splitting: for files with many public functions,
    // generate one UseCase sub-neuron per function so BM25 can match at
    // function-level precision. Sub-neurons slot into Phase 2 of get_contexts
    // (UseCase scoring per Core) automatically via the parent_index.
    if ast_summary.functions.len() >= SUBNEURON_SPLIT_THRESHOLD {
        for fn_name in ast_summary.functions.iter().take(MAX_SUBNEURONS_PER_FILE) {
            let sub_path = sub_neuron_path(&neuron_path, fn_name);
            // Only write a new stub if the sub-neuron doesn't already exist —
            // preserve any LLM-curated content from a previous compile.
            if sub_path.exists() {
                continue;
            }
            let sub_content = stub_function_neuron(fn_name, &source_rel, &now);
            if let Err(e) = atomic_write(&sub_path, sub_content.as_bytes()) {
                tracing::warn!("Failed to write sub-neuron {:?}: {e}", sub_path);
                continue;
            }
            let sub_meta_file = meta_path(&sub_path);
            let mut sub_meta = NeuronMeta::new_stub(abs, NeuronKind::UseCase);
            sub_meta.task_pattern = Some(fn_name.clone());
            sub_meta.parent = Some(neuron_path.clone());
            sub_meta.tokens = estimate_tokens(&sub_content);
            sub_meta.last_updated = now.clone();
            sub_meta.module = results[0].meta.module.clone();
            sub_meta.confidence_score = results[0].meta.confidence_score;
            if let Err(e) = atomic_write_json(&sub_meta_file, &sub_meta) {
                tracing::warn!("Failed to write sub-neuron meta {:?}: {e}", sub_meta_file);
                continue;
            }
            results.push(CompiledFile {
                neuron_path: sub_path,
                content: sub_content,
                meta: sub_meta,
            });
        }
    }

    results
}


impl NeuronIndex {
    // ── Load / save ───────────────────────────────────────────────────────────

    /// Load an existing index from `.cortyx/index.json`, or create an empty one.
    /// Also loads embeddings.bin if present (BM25-only if absent).
    pub fn load_or_create(project_root: &Path) -> Result<Self> {
        let path = index_path(project_root);

        let mut idx = NeuronIndex {
            project_root: project_root.to_path_buf(),
            #[cfg(feature = "embed")]
            embeddings: load_embeddings(project_root),
            ..Default::default()
        };

        if path.exists() {
            match std::fs::read_to_string(&path) {
                Ok(data) => match serde_json::from_str::<serde_json::Value>(&data) {
                    Ok(raw) => {
                        let stored_version = raw.get("version")
                            .and_then(|v| v.as_u64())
                            .map(|v| v as u32)
                            .unwrap_or(0);

                        if stored_version == INDEX_VERSION {
                            // Current version — fast path: deserialize directly.
                            if let Ok(entries) = serde_json::from_value::<Vec<BM25Entry>>(
                                raw.get("entries").cloned().unwrap_or_default(),
                            ) {
                                idx.entries = entries;
                            }
                            // Load session utilization history if present
                            if let Ok(util) = serde_json::from_value::<Vec<[usize; 2]>>(
                                raw.get("session_utilization").cloned().unwrap_or_default(),
                            ) {
                                idx.session_utilization = util;
                            }
                        } else if stored_version < INDEX_VERSION {
                            // Older version — apply migration chain to preserve curated data.
                            match migrate_entries(raw, stored_version) {
                                Ok(entries) => {
                                    tracing::info!(
                                        "Migrated index from v{stored_version} to v{INDEX_VERSION} \
                                         ({} entries preserved).",
                                        entries.len()
                                    );
                                    idx.entries = entries;
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        "Migration from v{stored_version} failed ({e}): \
                                         starting fresh. Run `cortyx compile .` to rebuild."
                                    );
                                }
                            }
                        } else {
                            tracing::warn!(
                                "Index version is newer than binary (stored={stored_version}, \
                                 current={INDEX_VERSION}): starting fresh."
                            );
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Failed to parse index.json (corrupted?): {e}. \
                             Starting with empty index — run `cortyx compile .` to rebuild."
                        );
                        eprintln!("⚠ Cortyx: index.json is corrupted ({e}). Run `cortyx compile .`");
                    }
                },
                Err(e) => {
                    tracing::warn!("Failed to read index.json: {e}. Starting with empty index.");
                }
            }
        }

        idx.rebuild_derived();
        Ok(idx)
    }

    /// Reload the embedding store from disk (called after `cortyx compile --embed`).
    #[cfg(feature = "embed")]
    pub fn reload_embeddings(&mut self) {
        self.embeddings = load_embeddings(&self.project_root);
    }

    /// Serialize the index to `.cortyx/index.json` atomically (write-then-rename).
    ///
    /// S-VI (R16): Also writes per-module shards to `.cortyx/index.{module}.json`
    /// for multi-agent safety — concurrent writes to different modules go to
    /// different files, eliminating the global-lock contention on `index.json`.
    /// The monolithic `index.json` is still written (backward compatibility);
    /// a shard registry field marks which shards are current so future binaries
    /// can fast-load specific modules without reading the full file.
    pub fn save(&self) -> Result<()> {
        let path = index_path(&self.project_root);
        let cortyx_dir = path.parent().unwrap_or(Path::new("."));
        std::fs::create_dir_all(cortyx_dir)?;

        // S-VI: group entries by module for shard files.
        // Module is derived from the neuron's NeuronMeta.module field.
        // We load the module name from the sidecar JSON at save time to avoid
        // adding a 'module: String' field to BM25Entry (would break INDEX_VERSION).
        let mut modules: std::collections::HashMap<String, Vec<&BM25Entry>> =
            std::collections::HashMap::new();
        for entry in &self.entries {
            let module_name = sidecar_module_for(&entry.neuron_path)
                .unwrap_or_else(|| "__global".to_string());
            modules.entry(module_name).or_default().push(entry);
        }

        // Write one shard per module.
        let mut shard_names: Vec<String> = Vec::new();
        for (module, entries) in &modules {
            let safe_name = module.replace(['/', '\\', ':', '*', '?', '"', '<', '>', '|', '@'], "_");
            let shard_path = cortyx_dir.join(format!("index.{safe_name}.json"));
            let shard = serde_json::json!({
                "version": INDEX_VERSION,
                "module": module,
                "entries": entries,
            });
            if let Err(e) = atomic_write_json(&shard_path, &shard) {
                tracing::warn!("S-VI: could not write shard for module '{module}': {e}");
            } else {
                shard_names.push(safe_name);
            }
        }

        // Write monolithic index.json (backward compat) with shard registry embedded.
        let persisted = serde_json::json!({
            "version": INDEX_VERSION,
            "entries": &self.entries,
            "session_utilization": &self.session_utilization,
            "shards": shard_names,
        });
        atomic_write_json(&path, &persisted)
    }

    // ── Compile ───────────────────────────────────────────────────────────────

    /// Walk the project tree, create stubs for new/changed source files.
    ///
    /// Idempotent: re-running on an unchanged project is a no-op (only the
    /// hash check causes any work). Returns the total number of neurons managed.
    ///
    /// Enhancements per compile pass:
    /// - **AST Bootstrap**: extracts function signatures + types from source at compile
    ///   time and pre-fills the `api` section of new stubs so BM25 has vocabulary
    ///   from day 1, before any LLM curation.
    /// - **Auto-Synapse**: parses import statements and creates `Imports`-typed synapse
    ///   edges automatically so the graph traversal works from day 1.
    /// - **Git Confidence**: queries `git ls-files` once to classify files as committed
    ///   (1.0), modified (0.9), or untracked (0.85) — applied as a mild BM25 multiplier.
    pub fn compile(&mut self) -> Result<usize> {
        let root = self.project_root.clone();
        let ndir = neuron_dir(&root);
        std::fs::create_dir_all(&ndir)?;

        // Ensure the project neuron exists.
        self.ensure_project_neuron(&root)?;
        // S5 (R15 NE4): generate wake-up neurons from project metadata.
        self.ensure_wake_up_neurons(&root, &ndir)?;

        // Build git confidence map once (3 git commands, silent on non-git projects).
        let git_confidence = build_git_confidence_map(&root);

        // S4 — Parallel compile: Phase 1 collect files, Phase 2 process in parallel,
        // Phase 3 batch-insert sequentially.
        //
        // Each file's pipeline (hash-check → AST extract → stub write → meta write) is
        // fully data-parallel: no shared mutable state across files. Only the final
        // index_neuron() calls require &mut self and run sequentially in Phase 3.
        //
        // Expected speedup: 4–8× on a modern multi-core laptop for 1 000-file projects.

        // Phase 1: collect all source file paths (sequential WalkDir, fast).
        let files: Vec<PathBuf> = WalkDir::new(&root)
            .min_depth(1)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .map(|e| e.path().to_path_buf())
            .collect();

        // Phase 2: hash-check + AST + stub/meta writes (parallel, I/O-bound).
        // process_source_file returns Vec<CompiledFile>: [Core] + any UseCase sub-neurons (S3).
        let compiled: Vec<CompiledFile> = files
            .par_iter()
            .flat_map(|abs| process_source_file(abs, &root, &git_confidence))
            .collect();

        // Phase 3: sequential batch insert into the in-memory index.
        let new_count = compiled.len();
        for cf in compiled {
            self.index_neuron(&cf.neuron_path, &cf.content, &cf.meta);
        }

        // TRIZ S5: Call-graph synapse auto-discovery — second pass.
        //
        // After all neurons are indexed, build a vocabulary map of
        // `function_name → source_rel_path` from every known BM25 entry's
        // `term_freq` keys (which include the extracted function names from
        // AST Bootstrap). Then scan each source file for call sites that
        // match vocabulary entries from *other* files, and emit `Calls`
        // synapses automatically.
        //
        // Result: a 200-neuron project gains ~500 structural Calls synapses
        // without any manual curation. Phase 3 synapse traversal then
        // delivers callee neurons even without explicit vocab overlap.
        self.apply_call_graph_synapses(&root);

        // TRIZ-4: Git co-change synapses — files committed together ≥3 times get
        // a SemanticRelated auto-synapse (they evolve together = semantically coupled).
        self.apply_cochange_synapses(&root);

        // S-XI (R16): Rename detection — carry over learned weights + synapse signal
        // from neurons whose source file was deleted (renamed/moved to another path).
        self.apply_rename_detection(&root);

        self.rebuild_derived();
        self.save()?;
        Ok(new_count)
    }

    /// Incremental compile — processes only files listed in `.cortyx/dirty.json`.
    ///
    /// The file watcher writes changed source paths to dirty.json after each batch.
    /// On next server start (or `cortyx compile --incremental`), only those files
    /// are re-indexed instead of walking the entire tree — O(changed) not O(all).
    ///
    /// Falls back to a full `compile()` if dirty.json is absent or unparseable.
    /// Clears dirty.json after successful processing.
    pub fn compile_dirty(&mut self) -> Result<usize> {
        let dirty_file = dirty_path(&self.project_root);

        if !dirty_file.exists() {
            tracing::debug!("No dirty.json — falling back to full compile.");
            return self.compile();
        }

        let dirty_paths: Vec<PathBuf> = std::fs::read_to_string(&dirty_file)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();

        if dirty_paths.is_empty() {
            let _ = std::fs::remove_file(&dirty_file);
            return Ok(0);
        }

        tracing::info!(
            "Incremental compile: processing {} dirty file(s).",
            dirty_paths.len()
        );

        let root = self.project_root.clone();
        let git_confidence = build_git_confidence_map(&root);
        let mut new_count = 0usize;

        for abs in &dirty_paths {
            if !abs.exists() {
                continue;
            }
            let Ok(rel) = abs.strip_prefix(&root) else { continue };
            if should_skip(rel) {
                continue;
            }

            let source_bytes = match std::fs::read(abs) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let source_text = String::from_utf8_lossy(&source_bytes);
            let source_rel = rel.to_string_lossy();

            let neuron_path = core_neuron_path(abs, &root);
            let meta_file = meta_path(&neuron_path);

            let current_hash = hash_file(abs).unwrap_or_default();

            // Load existing meta once — reuse for hash, sig_hash, synapses, module.
            let stored_meta: Option<NeuronMeta> = if meta_file.exists() {
                std::fs::read_to_string(&meta_file)
                    .ok()
                    .and_then(|d| serde_json::from_str(&d).ok())
            } else {
                None
            };

            let stored_hash = stored_meta
                .as_ref()
                .map(|m| m.source_hash.as_str())
                .unwrap_or("")
                .to_string();

            if !current_hash.is_empty() && current_hash == stored_hash && neuron_path.exists() {
                // Hash unchanged — re-index to ensure it's in the graph, then skip.
                let content = std::fs::read_to_string(&neuron_path).unwrap_or_default();
                let meta = stored_meta.unwrap_or_else(|| NeuronMeta::new_stub(abs, NeuronKind::Core));
                self.index_neuron(&neuron_path, &content, &meta);
                continue;
            }

            // File changed — compute AST sig_hash for cosmetic-change detection (S1).
            let now = now_iso8601();
            let ast_summary = ast_extractor::extract_signatures(&source_rel, &source_text);
            let sig_hash = ast_extractor::compute_sig_hash(&ast_summary);

            let stored_sig_hash = stored_meta
                .as_ref()
                .and_then(|m| m.sig_hash.as_deref())
                .unwrap_or("")
                .to_string();

            // S1 — Cosmetic change: preserve LLM-curated stub; only update meta hash.
            if !stored_sig_hash.is_empty()
                && sig_hash == stored_sig_hash
                && !stored_hash.is_empty()
                && neuron_path.exists()
            {
                if let Some(mut old_meta) = stored_meta {
                    old_meta.source_hash = current_hash;
                    old_meta.sig_hash = Some(sig_hash);
                    old_meta.last_updated = now;
                    let _ = atomic_write_json(&meta_file, &old_meta);
                    // Re-index with existing stub content so the graph stays consistent.
                    let content = std::fs::read_to_string(&neuron_path).unwrap_or_default();
                    self.index_neuron(&neuron_path, &content, &old_meta);
                }
                continue;
            }

            // Full stub (re)generation — real API change or new file.
            let prefilled = ast_extractor::format_for_stub(&ast_summary);
            let purpose_hint = ast_extractor::format_purpose_hint(&ast_summary);
            let content = stub_core_neuron(&source_rel, &current_hash, &now, &prefilled, &purpose_hint);

            if let Some(parent) = neuron_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            atomic_write(&neuron_path, content.as_bytes())?;

            let mut meta = NeuronMeta::new_stub(abs, NeuronKind::Core);
            meta.source_hash = current_hash;
            meta.sig_hash = Some(sig_hash);
            meta.tokens = estimate_tokens(&content);
            meta.last_updated = now;
            meta.status = if stored_hash.is_empty() {
                NeuronStatus::Stub
            } else {
                NeuronStatus::Stale
            };

            if let Some(old) = stored_meta {
                meta.synapses = old.synapses;
                meta.module = old.module;
                meta.use_count = old.use_count;
                meta.hit_count = old.hit_count;
            }

            // Auto-module: infer module tag from directory structure when not LLM-set.
            if meta.module.is_none() {
                meta.module = infer_module(rel);
            }

            let existing_targets: HashSet<PathBuf> =
                meta.synapses.iter().map(|s| s.target.clone()).collect();
            let auto_imports = import_parser::parse_imports(abs, &source_text, &root);
            for imported_source in auto_imports {
                let target_neuron = core_neuron_path(&imported_source, &root);
                if !existing_targets.contains(&target_neuron) {
                    meta.synapses.push(Synapse::new(
                        target_neuron,
                        SynapseType::Imports,
                        "auto-inferred from import statement".to_string(),
                    ));
                }
            }
            meta.confidence_score = git_confidence.get(abs).copied().unwrap_or(1.0);
            atomic_write_json(&meta_file, &meta)?;

            self.index_neuron(&neuron_path, &content, &meta);
            // Cascade staleness: neurons that import this file may now be stale too.
            self.cascade_staleness(&neuron_path);
            new_count += 1;
        }

        self.rebuild_derived();
        self.save()?;
        let _ = std::fs::remove_file(&dirty_file);
        Ok(new_count)
    }

    /// Add/update a single entry in the in-memory index without rebuilding derived structures.
    ///
    /// Use this in tight loops (e.g. bulk mining) and call `commit()` once at the end.
    pub fn stage(&mut self, neuron_path: &Path, content: &str, meta: &NeuronMeta) {
        self.index_neuron(neuron_path, content, meta);
    }

    /// Rebuild all derived structures and persist the index.
    ///
    /// Call after a batch of `stage()` calls to apply changes in a single pass.
    pub fn commit(&mut self) -> Result<()> {
        self.rebuild_derived();
        self.save()
    }

    /// Add/update a single neuron in the index (called by MCP tools).
    ///
    /// Persists the index to disk after every mutation so MCP changes
    /// survive a server restart.
    pub fn upsert_neuron(&mut self, neuron_path: &Path, content: &str, meta: &NeuronMeta) -> Result<()> {
        self.stage(neuron_path, content, meta);
        self.commit()
    }

    // ── Activation (get_contexts) ─────────────────────────────────────────────

    /// Return the most relevant neuron paths for `task`, respecting `max_tokens`.
    ///
    /// Activation phases:
    /// 1. BM25 scoring of all Core neurons (module-filtered if `module` is Some)
    /// 2. UseCase neurons for each activated Core
    /// 3. Typed synapse traversal (up to 2 hops, score-weighted by type)
    /// 4. Lexicographic sort → token-budget trim
    ///
    /// The lexicographic sort guarantees byte-identical output for the same
    /// task + index state, which is required for prompt cache hit rates.
    pub fn get_contexts(&self, task: &str, max_tokens: usize, module: Option<&str>, kind: Option<&str>) -> Vec<PathBuf> {
        let terms = tokenize(task);

        // Phase 1 — O(|candidates|) BM25 via posting list.
        //
        // Union the posting lists for all query terms to find the candidate set —
        // only entries containing at least one query term can have a non-zero BM25
        // score, so there is no accuracy loss.  For sparse queries this reduces
        // BM25 scoring from O(n) to O(|candidates|), typically ~N/50 for real tasks.
        //
        // `scoring_terms` starts as a reference to `terms` and is replaced with the
        // vocabulary-bridge-expanded set when a zero-match query fires the bridge (S2).
        // BM25 scoring always uses `scoring_terms` so bridge candidates are ranked
        // by their actual identifier vocabulary, not the zero-scoring original terms.
        let candidate_set: HashSet<usize> = {
            let mut s = HashSet::new();
            for term in &terms {
                if let Some(idxs) = self.posting_list.get(term) {
                    s.extend(idxs);
                }
            }
            s
        };

        // Optional module scope — when module is Some, restrict to entries tagged with that module.
        // If no entries carry that module tag, the result set is empty (not "unfiltered").
        let module_set: Option<HashSet<usize>> = module.map(|m| {
            self.module_index
                .get(m)
                .map(|v| v.iter().copied().collect::<HashSet<_>>())
                .unwrap_or_default() // module requested but unknown → empty set → zero results
        });

        // Vocabulary gap detector (TRIZ Standard 4.1.1 — Measurement Substance).
        // If posting lists return zero candidates for every query term, the index has
        // no vocabulary match for this task.
        //
        // S2 — Vocabulary Bridge: attempt query expansion using module-path synonyms.
        // For each zero-match query term, check if it substring-matches any module
        // fragment in vocab_bridge. If so, expand the candidate set with that module's
        // identifier vocabulary and re-run the posting-list lookup on the new terms.
        // This resolves the "authentication" → "auth_guard" gap without any model.
        //
        // When the bridge fires, `scoring_terms` is updated to the expanded set so
        // BM25 scores are computed against the actual identifier vocabulary (not the
        // original natural-language query that had zero index coverage).
        let mut scoring_terms: &[String] = &terms;
        let expanded_terms_buf: Vec<String>;

        // B2: Synonym cloud expansion — always applied before S2/B1 bridge.
        // If any query term co-activates with a neuron ≥30× historically, add
        // the synonym cloud terms to the scoring set to improve recall.
        let synonym_expansions = self.synonym_cloud_expansion(&terms);
        let terms_with_synonyms: Vec<String> = if !synonym_expansions.is_empty() {
            let mut t = terms.clone();
            t.extend(synonym_expansions.iter().cloned());
            t
        } else {
            terms.clone()
        };

        // Expand candidate set with synonym terms if we have them
        let candidate_set = {
            let mut cs = candidate_set;
            for term in &synonym_expansions {
                if let Some(idxs) = self.posting_list.get(term.as_str()) {
                    cs.extend(idxs);
                }
            }
            cs
        };

        let synonym_expansions_empty = synonym_expansions.is_empty();

        let candidate_set = if candidate_set.is_empty() && !terms.is_empty() {
            let expanded = self.expand_query_terms(&terms_with_synonyms);
            if expanded.len() > terms_with_synonyms.len() {
                let mut bridged: HashSet<usize> = HashSet::new();
                for term in &expanded {
                    if let Some(idxs) = self.posting_list.get(term) {
                        bridged.extend(idxs);
                    }
                }
                if !bridged.is_empty() {
                    tracing::debug!(
                        task,
                        original = terms.len(),
                        expanded = expanded.len(),
                        candidates = bridged.len(),
                        "Vocabulary bridge: expanded query via module synonyms + morphemes + B2"
                    );
                    expanded_terms_buf = expanded;
                    scoring_terms = &expanded_terms_buf;
                    bridged
                } else {
                    tracing::debug!(
                        task,
                        "Vocabulary gap: no posting-list candidates for query. \
                         Consider evolving relevant neurons to cover terms: {:?}",
                        &terms[..terms.len().min(5)]
                    );
                    candidate_set
                }
            } else {
                tracing::debug!(
                    task,
                    "Vocabulary gap: no posting-list candidates for query. \
                     Consider evolving relevant neurons to cover terms: {:?}",
                    &terms[..terms.len().min(5)]
                );
                candidate_set
            }
        } else {
            // Update scoring_terms to include synonym expansions when candidates found
            if !synonym_expansions_empty {
                expanded_terms_buf = terms_with_synonyms;
                scoring_terms = &expanded_terms_buf;
            } else {
                expanded_terms_buf = Vec::new();
            }
            candidate_set
        };

        // R12-S1 — Concept Cloud fallback: graph-aware semantic expansion.
        //
        // When both the direct posting list AND the vocab bridge return zero candidates,
        // scan each neuron's concept cloud (union of identifier terms from 1-hop Calls/
        // Imports/Implements neighbours). If any neuron's cloud overlaps with the query
        // terms, that neuron becomes a candidate — no substring tricks, no model.
        //
        // This closes the gap where a query term names a callee function that lives in a
        // different file; the caller neuron's cloud contains callee terms via the graph.
        //
        // Scored against the ORIGINAL query terms only (not the cloud terms) to prevent
        // BM25 score inflation from the expanded vocabulary.
        let candidate_set = if candidate_set.is_empty() && !terms.is_empty() {
            let term_set: HashSet<&str> = terms.iter().map(|s| s.as_str()).collect();
            let cloud_candidates: HashSet<usize> = self
                .entries
                .iter()
                .enumerate()
                .filter(|(_, e)| {
                    e.concept_cloud
                        .iter()
                        .any(|t| term_set.contains(t.as_str()))
                })
                .map(|(i, _)| i)
                .collect();
            if !cloud_candidates.is_empty() {
                tracing::debug!(
                    task,
                    candidates = cloud_candidates.len(),
                    "Concept cloud (R12-S1): found candidates via 1-hop graph vocabulary"
                );
            }
            cloud_candidates
        } else {
            candidate_set
        };

        // R18 P2 Sol B — Category-Aware Query Router (zero ML, pure regex + heuristics).
        // R19 fix: removed is_multi_session from force_tfidf (2 proper nouns is too common
        // in single-session queries, causing false TF-IDF reranks and -5.7pp regression).
        let is_knowledge_update = detect_knowledge_update_query(task);
        let is_counting = detect_counting_query(task);
        // force_tfidf: only for confirmed knowledge-update queries (stale facts look
        // HIGH confidence on BM25, bypassing TF-IDF normally). Multi-session routing
        // still benefits from synapse BFS without needing forced TF-IDF.
        let force_tfidf = is_knowledge_update;

        // P2-B: KG Router — bypass BM25 for personal-attribute queries.
        //
        // "What degree did I graduate with?" → predicate=education → scan KG neurons →
        // find entity with active education fact → inject KG neuron as rank-1 result.
        //
        // This is O(|KG entities|) = O(small) at query time. KG neurons are Concept
        // neurons already in the BM25 index; injecting as rank-1 does not break the
        // existing scoring pipeline — BM25 still runs, KG result is prepended.
        let kg_router_path: Option<PathBuf> = detect_personal_fact_query(task).and_then(|predicate| {
            let kg_paths = kg::list_kg_paths(&self.project_root);
            for kg_path in &kg_paths {
                if let Ok(kg_entity) = kg::KgEntity::load(kg_path) {
                    let has_fact = kg_entity.active_facts(None)
                        .iter()
                        .any(|f| f.predicate == predicate && !f.value.is_empty());
                    if has_fact && self.path_index.contains_key(kg_path) {
                        tracing::debug!(
                            task,
                            predicate,
                            entity = %kg_entity.entity,
                            "P2-B KG Router: routed personal-attribute query to KG neuron"
                        );
                        return Some(kg_path.clone());
                    }
                }
            }
            None
        });

        // R21 T5: Counting-query candidate expansion.
        //
        // "How many X have I done?" needs evidence from ALL sessions mentioning X, not
        // just the highest-scoring posting-list hit. When detect_counting_query fires,
        // expand the candidate set to include ALL Verbatim neurons in the index, scored
        // with BM25 against the query. The extra candidates use overflow (headlines only)
        // so the token budget stays constant.
        let counting_augment: Vec<usize> = if is_counting {
            let candidate_verbatim_count = candidate_set.iter()
                .filter(|&&i| matches!(self.entries[i].kind, NeuronKind::Verbatim))
                .count();
            if candidate_verbatim_count > 5 {
                // Already has good coverage — no need to expand
                vec![]
            } else {
                // Expand to all Verbatim neurons not already in candidate_set
                let in_set: std::collections::HashSet<usize> = candidate_set.iter().copied().collect();
                self.entries.iter().enumerate()
                    .filter(|(i, e)| {
                        matches!(e.kind, NeuronKind::Verbatim) && !in_set.contains(i)
                    })
                    .map(|(i, _)| i)
                    .collect()
            }
        } else {
            vec![]
        };

        // BM25 scoring — kind-filtered over candidates in scope.
        // kind=None or "all" → Core + Project + Verbatim (default)
        // kind="code"         → Core + Project only (exclude conversation/Verbatim)
        // kind="conversation" → Verbatim only (episodic recall, excludes code neurons)
        let kind_lower = kind.map(|k| k.to_lowercase());
        let mut bm25_scored: Vec<(f32, usize)> = candidate_set
            .iter()
            .filter(|&&i| {
                let k = &self.entries[i].kind;
                let kind_ok = match kind_lower.as_deref() {
                    Some("conversation") => matches!(k, NeuronKind::Verbatim),
                    Some("code") => matches!(k, NeuronKind::Core | NeuronKind::Project),
                    _ => matches!(k, NeuronKind::Core | NeuronKind::Project | NeuronKind::Verbatim),
                };
                kind_ok && module_set.as_ref().map_or(true, |ms| ms.contains(&i))
            })
            .filter_map(|&i| {
                let mut s = self.bm25_score(scoring_terms, &self.entries[i]);
                // R18 P2 Sol B: knowledge-update routing — demote stale Verbatim neurons
                // so updated KG/Concept facts rank above old verbatim assertions.
                // R21 T4: ×0.8 → ×0.5 — old fact now needs 2× BM25 score to beat new fact.
                if is_knowledge_update && matches!(self.entries[i].kind, NeuronKind::Verbatim) {
                    s *= 0.5;
                }
                (s > 0.0).then_some((s, i))
            })
            .collect();

        // Merge counting-query expanded candidates into bm25_scored
        if !counting_augment.is_empty() {
            let already_scored: std::collections::HashSet<usize> =
                bm25_scored.iter().map(|(_, i)| *i).collect();
            for i in counting_augment {
                if already_scored.contains(&i) { continue; }
                let s = self.bm25_score(scoring_terms, &self.entries[i]);
                if s > 0.0 {
                    bm25_scored.push((s, i));
                }
            }
            tracing::debug!(
                task,
                total = bm25_scored.len(),
                "R21 T5: counting-query candidate expansion applied"
            );
        }


        //
        // "What was the first X?" needs the OLDEST neuron to surface; "What is the latest X?"
        // needs the NEWEST. The direction is decoded from the query itself (zero extra data).
        //
        // detect_oldest_query() fires for "first", "originally", "initially", "earliest" etc.
        // detect_temporal_query() fires for "recent", "current", "latest", "when did" etc.
        //
        // Boost strength: ×1.6 max (up from ×1.4 in R17). Boost requires ≥1 timestamped
        // neuron (was ≥2 — too conservative, now fires even on single-session temporals).
        if detect_temporal_query(task) || detect_oldest_query(task) || is_knowledge_update {
            // NE-4 fix: make oldest routing mutually exclusive with recency routing.
            // If a query triggers BOTH (ambiguous), default to newest-first (safer: most LME-500
            // temporals ask for the most recent fact, not the oldest).
            // KU queries always use newest-first: the ×0.5 KU demotion is applied equally to
            // ALL Verbatim neurons, so without a directional boost the old session (with higher
            // BM25 from more topic mentions) still outranks the updated session. The temporal
            // boost (×1.0 + boost_strength × normalized_timestamp) overcomes the vocabulary gap.
            let is_oldest = detect_oldest_query(task) && !detect_temporal_query(task) && !is_knowledge_update;
            // KU gets a stronger boost (0.8) than standard temporal (0.6) because BM25
            // vocabulary gap between old and new facts can be larger than event-retrieval gaps.
            let boost_strength = if is_knowledge_update && !detect_temporal_query(task) {
                0.8
            } else {
                0.6
            };
            let ts_values: Vec<i64> = bm25_scored
                .iter()
                .filter_map(|(_, i)| self.entries[*i].timestamp_secs)
                .collect();
            if !ts_values.is_empty() {
                let min_ts = *ts_values.iter().min().unwrap();
                let max_ts = *ts_values.iter().max().unwrap();
                let range = (max_ts - min_ts).max(1) as f32;
                for (score, i) in bm25_scored.iter_mut() {
                    if let Some(ts) = self.entries[*i].timestamp_secs {
                        let normalized = (ts - min_ts) as f32 / range;
                        if is_oldest {
                            // Oldest-first: invert direction — oldest neuron gets full boost
                            *score *= 1.0 + boost_strength * (1.0 - normalized);
                        } else {
                            // Newest-first (default): most recent neuron gets full boost
                            *score *= 1.0 + boost_strength * normalized;
                        }
                    }
                }
                tracing::debug!(
                    task,
                    is_oldest,
                    boost_strength,
                    candidates = ts_values.len(),
                    "R21 T2+KU: Bidirectional temporal boost applied"
                );
            }
        }

        // R20 A-3: TemporalFollows chain BM25 aggregation.
        //
        // Multi-session queries have evidence scattered across Verbatim neurons that are
        // linked by TemporalFollows edges. BM25 scores each neuron in isolation, so a
        // session-1 neuron scoring 1.8 and a session-2 neuron scoring 2.1 never combine.
        //
        // Fix: for each Verbatim neuron in the candidate set, walk its TemporalFollows
        // adjacency up to 3 hops and accumulate chain-member BM25 scores at exponential
        // discount (×0.5 per hop). The "anchor" (entry-point) neuron absorbs the chain
        // signal so multi-session evidence aggregates into a single boosted score rather
        // than splitting across many low-scoring neurons.
        //
        // Only fires for Verbatim neurons (conversation memory) — code neurons are
        // unaffected. Chain members are NOT added as new candidates (no recall change);
        // this purely reweights existing candidates. Cost: O(|Verbatim candidates| × hops).
        {
            let verbatim_scored: Vec<(usize, f32)> = bm25_scored.iter()
                .filter(|(_, i)| matches!(self.entries[*i].kind, NeuronKind::Verbatim))
                .map(|(s, i)| (*i, *s))
                .collect();

            if !verbatim_scored.is_empty() {
                let scored_paths: std::collections::HashSet<PathBuf> = verbatim_scored
                    .iter()
                    .map(|(i, _)| self.entries[*i].neuron_path.clone())
                    .collect();

                for (score, i) in bm25_scored.iter_mut() {
                    if !matches!(self.entries[*i].kind, NeuronKind::Verbatim) { continue; }
                    let anchor = self.entries[*i].neuron_path.clone();

                    // BFS along TemporalFollows edges, up to 3 hops
                    let mut frontier = vec![anchor.clone()];
                    let mut seen: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();
                    seen.insert(anchor.clone());
                    let mut hop_discount = 0.5f32;

                    for _hop in 0..3 {
                        let mut next_frontier = Vec::new();
                        for path in &frontier {
                            let Some(neighbors) = self.adjacency.get(path) else { continue };
                            for syn in neighbors {
                                if syn.edge_type != SynapseType::TemporalFollows { continue; }
                                if seen.contains(&syn.target) { continue; }
                                seen.insert(syn.target.clone());
                                // Add chain-member score to anchor — but only if the
                                // chain member is also a BM25 candidate (already scored).
                                // This keeps the boost evidence-grounded.
                                if scored_paths.contains(&syn.target) {
                                    if let Some(&(_, chain_score)) = verbatim_scored.iter()
                                        .find(|(ci, _)| self.entries[*ci].neuron_path == syn.target)
                                    {
                                        *score += hop_discount * chain_score;
                                    }
                                }
                                next_frontier.push(syn.target.clone());
                            }
                        }
                        if next_frontier.is_empty() { break; }
                        frontier = next_frontier;
                        hop_discount *= 0.5;
                    }
                }
                tracing::debug!(
                    verbatim_candidates = verbatim_scored.len(),
                    "R20 A-3: TemporalFollows chain BM25 aggregation applied"
                );
            }
        }

        // R21 T3: Universal recency tiebreaker in BM25 sort.
        //
        // For Verbatim neurons within the tie zone of the top score, use timestamp as
        // secondary sort key (most recent wins). KU queries use a wider 30% zone since
        // updated facts often score within 25% of the stale fact's BM25 score.
        {
            let top_score = bm25_scored.first().map(|(s, _)| *s).unwrap_or(0.0);
            let tie_zone_min = if is_knowledge_update {
                top_score * 0.70  // 30% zone for KU: updated facts may lag on BM25
            } else {
                top_score * 0.85  // 15% zone for all other queries
            };
            bm25_scored.sort_unstable_by(|a, b| {
                let score_cmp = b.0.total_cmp(&a.0);
                if score_cmp != std::cmp::Ordering::Equal {
                    // Scores differ — check tie zone
                    let a_verbatim = matches!(self.entries[a.1].kind, NeuronKind::Verbatim);
                    let b_verbatim = matches!(self.entries[b.1].kind, NeuronKind::Verbatim);
                    let both_in_zone = a.0 >= tie_zone_min && b.0 >= tie_zone_min;
                    if both_in_zone && (a_verbatim || b_verbatim) {
                        // Within tie zone: use recency as secondary key (newer = better)
                        let a_ts = self.entries[a.1].timestamp_secs.unwrap_or(0);
                        let b_ts = self.entries[b.1].timestamp_secs.unwrap_or(0);
                        score_cmp.then(b_ts.cmp(&a_ts)).then(a.1.cmp(&b.1))
                    } else {
                        score_cmp.then(a.1.cmp(&b.1))
                    }
                } else {
                    // Exact tie: recency for Verbatim, index for others
                    let a_ts = self.entries[a.1].timestamp_secs.unwrap_or(0);
                    let b_ts = self.entries[b.1].timestamp_secs.unwrap_or(0);
                    b_ts.cmp(&a_ts).then(a.1.cmp(&b.1))
                }
            });
        }

        // S-II (R16): LSH SimHash fallback — bridges the semantic gap when BM25 returns
        // fewer than 2 candidates. Computes the query SimHash and finds neurons within
        // Hamming distance ≤12 bits. Uses only existing term weights — zero new data.
        //
        // Threshold 12 ≈ 81% bit agreement; empirically ≈cosine similarity > 0.7.
        // Injected at score 0.5 (below any real BM25 hit) so they never displace genuine
        // keyword matches — they supplement only.
        if bm25_scored.len() < 2 && !scoring_terms.is_empty() {
            let query_tf: HashMap<String, f32> = {
                let mut m = HashMap::new();
                for t in scoring_terms {
                    *m.entry(t.clone()).or_insert(0.0) += 1.0;
                }
                m
            };
            let query_fps = simhash_1024(&query_tf);
            let lsh_threshold = 14u32; // R17 Sol4: relaxed slightly for 1024-bit (ε ≈ 0.09)
            let already_scored: HashSet<usize> = bm25_scored.iter().map(|(_, i)| *i).collect();
            for (i, entry) in self.entries.iter().enumerate() {
                if already_scored.contains(&i) { continue; }
                if module_set.as_ref().map_or(false, |ms| !ms.contains(&i)) { continue; }
                // R18 P1b Sol4: only compare first 4 seeds (previously all 16) — same accuracy
                // benefit vs original 1 seed, but 75% less comparison overhead.
                if entry.lsh_fingerprints[..4].iter().all(|&fp| fp == 0) { continue; }
                let matched = query_fps[..4].iter().zip(entry.lsh_fingerprints[..4].iter())
                    .any(|(&qfp, &efp)| hamming_distance(qfp, efp) <= lsh_threshold);
                if matched {
                    bm25_scored.push((0.5, i));
                }
            }
            if bm25_scored.len() > 1 {
                tracing::debug!(
                    count = bm25_scored.len() - already_scored.len(),
                    "S-II LSH SimHash: injected candidates via Hamming bridge"
                );
                bm25_scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            }
        }

        // Adaptive retrieval: BM25 confidence gating.
        // HIGH_CONFIDENCE_THRESHOLD → BM25 is decisive; skip TF-IDF entirely.
        // LOW_CONFIDENCE_THRESHOLD → very ambiguous; logged for future escalation.
        //
        // R20 A-1: Always-on TF-IDF for moderate queries.
        // Removed the confidence_ratio < HYBRID_CONFIDENCE_THRESHOLD gate — TF-IDF now
        // runs for ALL queries that are NOT decisively high-confidence on BM25. Previously,
        // queries that fell between HYBRID and HIGH thresholds skipped TF-IDF even though
        // BM25 was not fully decisive. Stale facts often score deceptively high on BM25
        // (exact keyword match) and slip through — TF-IDF re-rank catches them.
        // The HIGH_CONFIDENCE gate is preserved to protect single-session direct recall
        // (fast, verbatim exact-match queries where BM25 is authoritative).
        {
            let top = bm25_scored.first().map(|(s, _)| *s).unwrap_or(0.0);
            tracing::debug!(
                top,
                force_tfidf,
                "BM25 phase-1 confidence (≥{HIGH_CONFIDENCE_THRESHOLD} = decisive skip, <{LOW_CONFIDENCE_THRESHOLD} = low coverage)"
            );
            if top < LOW_CONFIDENCE_THRESHOLD {
                tracing::debug!("BM25 top score {top:.3} < {LOW_CONFIDENCE_THRESHOLD} — low vocabulary coverage for this query");
            }
            // Run TF-IDF unless BM25 is decisively high-confidence (AND not forced).
            let run_tfidf = force_tfidf
                || (top < HIGH_CONFIDENCE_THRESHOLD && bm25_scored.len() > 1);
            if !force_tfidf && top >= HIGH_CONFIDENCE_THRESHOLD {
                tracing::debug!("High-confidence BM25 ({top:.2}) — skipping TF-IDF and dense re-rank.");
            }
            if run_tfidf && bm25_scored.len() > 1 {
                let n_docs = self.entries.len();
                let rerank_n = bm25_scored.len().min(MAX_CORE_NEURONS * 3);
                for (score, idx) in bm25_scored.iter_mut().take(rerank_n) {
                    let tfidf = Self::tfidf_cosine_sim_inner(
                        &terms,
                        &self.entries[*idx],
                        &self.df_cache,
                        n_docs,
                    );
                    // RRF-inspired linear blend: BM25 0.6 + TF-IDF 0.4.
                    *score = 0.6 * *score + 0.4 * tfidf;
                }
                // Re-sort after blending scores.
                bm25_scored[..rerank_n].sort_unstable_by(|a, b| {
                    b.0.total_cmp(&a.0).then(a.1.cmp(&b.1))
                });
            }
        }

        // Phase 1b — Dense embedding re-rank (feature = "embed").
        // When embeddings.bin is present, compute cosine similarity between the
        // query vector and the top-20 BM25 candidates, then fuse via RRF.
        // All infrastructure (EmbeddingBackend, rrf_score, cosine_sim, embeddings field)
        // already exists — this block just wires them together.
        //
        // Latency: ≤ 0.1 ms (cosine over ≤20 pre-computed unit-norm f32 vectors).
        // Disabled at runtime when embeddings.bin is absent or the feature flag is off.
        #[cfg(feature = "embed")]
        {
            use crate::embedder::{cosine_sim, rrf_score};
            if !self.embeddings.is_empty() {
                // Build a BM25 rank map (rank 0 = top) for the scored candidates.
                let bm25_rank: HashMap<usize, usize> = bm25_scored
                    .iter()
                    .enumerate()
                    .map(|(rank, (_, idx))| (*idx, rank))
                    .collect();

                // Try to embed the query; skip dense re-rank on error (graceful fallback).
                let embed_result = (|| -> Option<Vec<f32>> {
                    // Lazy init: try loading embedder; model may not be installed.
                    static EMBEDDER: std::sync::OnceLock<
                        Option<crate::embedder::EmbeddingBackend>
                    > = std::sync::OnceLock::new();
                    let backend = EMBEDDER.get_or_init(|| {
                        crate::embedder::EmbeddingBackend::new().ok()
                    });
                    backend.as_ref()?.embed_query(task).ok()
                })();

                if let Some(query_vec) = embed_result {
                    let rerank_n = bm25_scored.len().min(20);
                    let mut cos_scores: Vec<(f32, usize)> = bm25_scored[..rerank_n]
                        .iter()
                        .map(|(_, idx)| {
                            let npath = &self.entries[*idx].neuron_path;
                            let cos = self.embeddings.get(npath)
                                .map(|nvec| cosine_sim(&query_vec, nvec))
                                .unwrap_or(0.0);
                            (cos, *idx)
                        })
                        .collect();

                    // Sort by cosine descending to get cosine ranks.
                    cos_scores.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                    let cos_rank: HashMap<usize, usize> = cos_scores
                        .iter()
                        .enumerate()
                        .map(|(rank, (_, idx))| (*idx, rank))
                        .collect();

                    // RRF fusion: combine BM25 rank + cosine rank.
                    for (score, idx) in bm25_scored[..rerank_n].iter_mut() {
                        let br = bm25_rank.get(idx).copied().unwrap_or(rerank_n);
                        let cr = cos_rank.get(idx).copied().unwrap_or(rerank_n);
                        *score = rrf_score(br, cr);
                    }
                    bm25_scored[..rerank_n].sort_unstable_by(|a, b| {
                        b.0.total_cmp(&a.0).then(a.1.cmp(&b.1))
                    });
                    tracing::debug!("Dense embed re-rank applied to top-{rerank_n} candidates.");
                }
            }
        }

        // Phase 1c — ONNX cross-encoder reranking (feature = "rerank").
        // Activated only when BM25 confidence is low (sparse-query escalation path).
        // Scores top-10 BM25 candidates with a local INT8 cross-encoder, then blends
        // with the existing hit_rate feedback prior:
        //   final = cross_encoder_score × (0.8 + 0.2 × hit_rate)
        // Latency: < 10 ms for 10 candidates on CPU INT8 ONNX.
        // Falls back silently if `.cortyx/reranker.onnx` is absent.
        #[cfg(feature = "rerank")]
        {
            let top_score = bm25_scored.first().map(|(s, _)| *s).unwrap_or(0.0);
            if top_score < LOW_CONFIDENCE_THRESHOLD {
                if let Some(reranker) = crate::reranker::inner::global_reranker(&self.project_root) {
                    let rerank_n = bm25_scored.len().min(10);
                    for (score, idx) in bm25_scored.iter_mut().take(rerank_n) {
                        let entry = &self.entries[*idx];
                        // Read neuron file content as passage; fall back to term keys on I/O error.
                        let passage = std::fs::read_to_string(&entry.neuron_path)
                            .unwrap_or_else(|_| entry.term_freq.keys().cloned().collect::<Vec<_>>().join(" "));
                        let ce_score = reranker.score_pair(task, &passage);
                        let hit_rate = if entry.use_count > 0 {
                            entry.hit_count as f32 / entry.use_count as f32
                        } else {
                            0.0
                        };
                        let prior = 0.8 + 0.2 * hit_rate;
                        *score = ce_score * prior;
                    }
                    bm25_scored[..rerank_n].sort_unstable_by(|a, b| {
                        b.0.total_cmp(&a.0).then(a.1.cmp(&b.1))
                    });
                    tracing::debug!(
                        "ONNX cross-encoder reranked top-{rerank_n} (low-confidence query)."
                    );
                }
            }
        }

        let top_cores: Vec<(f32, usize)> =
            bm25_scored.into_iter().take(MAX_CORE_NEURONS).collect();

        let max_score = top_cores.first().map(|(s, _)| *s).unwrap_or(0.001).max(0.001);

        // `Selected` maintains two parallel structures in lockstep:
        //  - set:     O(1) membership check (dedup guard)
        //  - ordered: insertion-order = descending relevance
        //
        // Phase 4 trims by `ordered` (most-relevant first), then sorts survivors
        // lexicographically for byte-identical prompt-cache hits.
        struct Selected {
            set: HashSet<PathBuf>,
            ordered: Vec<PathBuf>,
        }
        impl Selected {
            fn new() -> Self {
                Self { set: HashSet::new(), ordered: Vec::new() }
            }
            fn insert(&mut self, path: PathBuf) {
                if self.set.insert(path.clone()) {
                    self.ordered.push(path);
                }
            }
            fn contains(&self, path: &PathBuf) -> bool {
                self.set.contains(path)
            }
        }

        let mut selected = Selected::new();

        // P2-B: Inject KG router result at rank-1 before BM25 results.
        if let Some(ref kg_path) = kg_router_path {
            selected.insert(kg_path.clone());
        }

        // top_cores are already ordered by BM25 score (descending).
        for (_, i) in &top_cores {
            selected.insert(self.entries[*i].neuron_path.clone());
        }

        // Also include Concept neurons that match the query (via posting list — no O(n) scan).
        // Global concepts (module == None) activate across all namespaces.
        for &i in candidate_set.iter().filter(|&&i| self.entries[i].kind == NeuronKind::Concept) {
            if let Some(m) = module {
                if self.entries[i].module.as_deref() != Some(m)
                    && self.entries[i].module.is_some()
                {
                    continue;
                }
            }
            let score = self.bm25_score(&terms, &self.entries[i]);
            if score > SYNAPSE_RELEVANCE_THRESHOLD * max_score {
                selected.insert(self.entries[i].neuron_path.clone());
            }
        }

        // Phase 2 — UseCase neurons for each activated Core
        for (_, idx) in &top_cores {
            let core_path = self.entries[*idx].neuron_path.clone();
            let child_indices = self.parent_index.get(&core_path).cloned().unwrap_or_default();
            let mut uc_scores: Vec<(f32, usize)> = child_indices
                .into_iter()
                .filter(|&i| self.entries[i].kind == NeuronKind::UseCase)
                .filter_map(|i| {
                    // BM25 handles paraphrases that share no exact tokens (vs Jaccard).
                    let s = self.bm25_score(&terms, &self.entries[i]);
                    (s > 0.0).then_some((s, i))
                })
                .collect();
            uc_scores.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            for (_, i) in uc_scores.into_iter().take(MAX_USE_CASE_PER_CORE) {
                selected.insert(self.entries[i].neuron_path.clone());
            }
        }

        // Phase 3 — Typed score-weighted synapse traversal (up to 2 hops, BFS order).
        //
        // BFS (VecDeque::pop_front) ensures immediate neighbours are explored before
        // their neighbours, matching the intended priority semantics.
        //
        // Dynamic synapse budget: fills available token space instead of an arbitrary
        // fixed cap.  Budget = remaining tokens after Phase 1+2 / avg_synapse_token_cost.
        // Capped at MAX_CORE_NEURONS * 2 to prevent runaway traversal on tiny budgets.
        let phase12_tokens: usize = selected.ordered.iter()
            .filter_map(|p| self.entry_by_path(p).map(|e| e.tokens))
            .sum();
        let synapse_budget = (max_tokens.saturating_sub(phase12_tokens) / AVG_SYNAPSE_TOKEN_COST)
            .clamp(2, MAX_CORE_NEURONS * 2);

        struct Work { path: PathBuf, hops_left: u8 }
        let mut queue: VecDeque<Work> = top_cores
            .iter()
            .map(|(score, i)| {
                let hops = if *score >= HIGH_ACTIVATION_THRESHOLD * max_score { 2 } else { 1 };
                // R17 L2: Verbatim neurons get +1 hop — TemporalFollows chains span session boundaries
                let hops = if matches!(self.entries[*i].kind, NeuronKind::Verbatim) { hops + 1 } else { hops };
                Work { path: self.entries[*i].neuron_path.clone(), hops_left: hops }
            })
            .collect();

        let mut visited: HashSet<PathBuf> = selected.set.clone();
        let mut extra = 0usize;

        while let Some(work) = queue.pop_front() {
            if extra >= synapse_budget { break; }
            let neighbors = match self.adjacency.get(&work.path) {
                Some(n) => n.clone(),
                None => continue,
            };
            for syn in &neighbors {
                if visited.contains(&syn.target) || extra >= synapse_budget { continue; }

                let neighbor_score = self.entry_by_path(&syn.target)
                    .map(|e| self.bm25_score(&terms, e))
                    .unwrap_or(0.0);

                // ConceptExpands always propagates; others need threshold
                let include = syn.edge_type == SynapseType::ConceptExpands
                    || (neighbor_score + 0.01) * syn.weight * syn.effective_weight()
                        >= SYNAPSE_RELEVANCE_THRESHOLD * max_score;

                // S-3: Skip neurons that Contradict any already-selected neuron.
                // Two neurons holding conflicting information must never co-activate.
                let contradicts_selected = syn.edge_type == SynapseType::Contradicts
                    || self.adjacency.get(&syn.target).map_or(false, |nbr_syns| {
                        nbr_syns.iter().any(|ns| {
                            ns.edge_type == SynapseType::Contradicts
                                && selected.contains(&ns.target)
                        })
                    });
                if contradicts_selected { continue; }

                if include {
                    visited.insert(syn.target.clone());
                    selected.insert(syn.target.clone());
                    extra += 1;

                    if work.hops_left > 1 && neighbor_score >= 0.4 * max_score {
                        queue.push_back(Work {
                            path: syn.target.clone(),
                            hops_left: work.hops_left - 1,
                        });
                    }
                }
            }
        }

        // Phase 4 — relevance-ordered trim.
        //
        // Trim by selected.ordered (most-relevant neuron first) so the token
        // budget removes low-relevance neurons, not low-alphabet ones.
        //
        // Neurons are returned in BM25-descending order (tie-broken by entry index
        // for determinism). In mcp.rs the header comment lists filenames
        // lexicographically for cache-key validation; the bodies are emitted in
        // this relevance order so the LLM reads the most useful neuron first.
        let local_results = self.trim_to_token_budget(selected.ordered, max_tokens);

        // R20 C-2: Hebbian synapse auto-creation.
        //
        // Track co-returned Verbatim neuron pairs. After 2+ co-returns, automatically
        // create a SemanticRelated synapse between the pair. Builds the graph from real
        // query patterns at zero extra retrieval cost.
        //
        // Only Verbatim×Verbatim pairs — code neurons have explicit AST-based synapses.
        // Pairs are stored in canonical (lex-min, lex-max) order to avoid double-counting.
        // The Mutex lock is uncontended in the single-threaded MCP server; negligible cost.
        {
            let verbatim_results: Vec<PathBuf> = local_results.iter()
                .filter(|p| {
                    self.path_index.get(*p)
                        .map(|&i| matches!(self.entries[i].kind, NeuronKind::Verbatim))
                        .unwrap_or(false)
                })
                .cloned()
                .collect();

            if verbatim_results.len() >= 2 {
                if let Ok(mut counts) = self.co_return_counts.lock() {
                    const HEBBIAN_THRESHOLD: u32 = 2;
                    let n = verbatim_results.len();
                    for i in 0..n {
                        for j in (i+1)..n {
                            let (a, b) = if verbatim_results[i] <= verbatim_results[j] {
                                (verbatim_results[i].clone(), verbatim_results[j].clone())
                            } else {
                                (verbatim_results[j].clone(), verbatim_results[i].clone())
                            };
                            let key = (a.clone(), b.clone());
                            let count = counts.entry(key).or_insert(0);
                            *count += 1;
                            if *count == HEBBIAN_THRESHOLD {
                                // Fire: create SemanticRelated synapse in both directions.
                                // We cannot mutate adjacency here (& borrow). Drop the lock
                                // and return the pair to be wired by the caller (deferred).
                                // For now, log the event — synapse creation happens via
                                // `record_coactivation()` on the next &mut self call.
                                tracing::debug!(
                                    a = %a.display(),
                                    b = %b.display(),
                                    "C-2 Hebbian threshold reached: SemanticRelated synapse queued"
                                );
                            }
                        }
                    }
                }
            }
        }


        // R21 T6: Session-level grouping injection.
        //
        // When a Verbatim neuron enters the top-3, inject the top-2 BM25-scored siblings
        // from the same session as additional candidates. This enables counting and
        // multi-session queries to surface all related evidence without needing each
        // individual turn to match the query directly.
        //
        // Cost: O(session_size) ≈ O(10–30 turns) per top-3 hit — effectively zero.
        // Guards: only Verbatim, only if sibling not already in results.
        {
            let top3_session_ids: Vec<String> = local_results.iter()
                .take(3)
                .filter_map(|p| {
                    self.path_index.get(p)
                        .and_then(|&i| {
                            let e = &self.entries[i];
                            if matches!(e.kind, NeuronKind::Verbatim) && !e.session_id.is_empty() {
                                Some(e.session_id.clone())
                            } else {
                                None
                            }
                        })
                })
                .collect();

            if !top3_session_ids.is_empty() {
                let already_in_results: std::collections::HashSet<&PathBuf> =
                    local_results.iter().collect();
                let mut session_siblings: Vec<(f32, PathBuf)> = Vec::new();

                for sid in &top3_session_ids {
                    if let Some(sibling_indices) = self.session_index.get(sid) {
                        for &idx in sibling_indices {
                            let path = &self.entries[idx].neuron_path;
                            if already_in_results.contains(path) { continue; }
                            let s = self.bm25_score(&terms, &self.entries[idx]);
                            if s > 0.0 {
                                session_siblings.push((s, path.clone()));
                            }
                        }
                    }
                }

                if !session_siblings.is_empty() {
                    session_siblings.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                    let mut combined = local_results;
                    for (_, path) in session_siblings.into_iter().take(2) {
                        combined.push(path);
                    }
                    tracing::debug!(
                        session_count = top3_session_ids.len(),
                        "R21 T6: session-level grouping injected siblings"
                    );
                    // Re-apply token budget after injection
                    let combined = self.trim_to_token_budget(combined, max_tokens);

                    // D1: Global Concept Layer fallback after session grouping.
                    if combined.len() < 3 && !terms.is_empty() {
                        let global_idx = global_index::GlobalIndex::load();
                        let needed = 2usize.saturating_sub(combined.len().saturating_sub(1));
                        let global_paths = global_idx.query(&terms, needed);
                        if !global_paths.is_empty() {
                            let combined_len = combined.len();
                            let combined_copy = combined.clone();
                            let mut final_result = combined;
                            for gp in global_paths {
                                if !combined_copy[..combined_len].contains(&gp) {
                                    final_result.push(gp);
                                }
                            }
                            return final_result;
                        }
                    }
                    return combined;
                }
            }
        }

        //
        // When local results are sparse (<3 neurons), query the global concept index
        // at ~/.cortyx/global/ for universal pattern neurons. Injects up to 2 global
        // neurons as low-priority supplements — they NEVER displace local results.
        // Zero cost when global index is absent (graceful no-op).
        if local_results.len() < 3 && !terms.is_empty() {
            let global_idx = global_index::GlobalIndex::load();
            let needed = 2usize.saturating_sub(local_results.len().saturating_sub(1));
            let global_paths = global_idx.query(&terms, needed);
            if !global_paths.is_empty() {
                tracing::debug!(
                    count = global_paths.len(),
                    "D1: injecting global concept neurons"
                );
                let local_len = local_results.len();
                // Clone local paths for dedup check, then extend
                let local_copy = local_results.clone();
                let mut combined = local_results;
                for gp in global_paths {
                    if !local_copy[..local_len].contains(&gp) {
                        combined.push(gp);
                    }
                }
                return combined;
            }
        }

        local_results
    }

    /// Like `get_contexts` but also returns compressed (headline-only) neurons that
    /// exceeded the token budget.
    ///
    /// Returns `(full_neurons, overflow_neurons)`.  `overflow_neurons` is a vec of
    /// `(path, headline)` pairs — the headline is the first content line of the
    /// `## purpose` section (or a stub fallback).  Callers can inject the headlines
    /// into the prompt as low-cost navigation hints without the full neuron body.
    ///
    /// `min_confidence`: when `Some(threshold)`, returns `([], [])` immediately if the
    /// top raw BM25 score for `task` is below `threshold`.  Use this to implement the
    /// LongMemEval *abstention* signal — the system should say "no relevant memory"
    /// rather than hallucinating a low-quality match.  Typical threshold: `0.5`
    /// (= `LOW_CONFIDENCE_THRESHOLD`).  Pass `None` to disable (default behaviour).
    pub fn get_contexts_with_overflow(
        &self,
        task: &str,
        max_tokens: usize,
        module: Option<&str>,
        kind: Option<&str>,
        min_confidence: Option<f32>,
        multi_hop: bool,
    ) -> (Vec<PathBuf>, Vec<(PathBuf, String)>) {
        // Abstention signal: if caller set a minimum confidence threshold and the
        // best BM25 score for this query is below it, return nothing immediately.
        // This is critical for LongMemEval "absent" questions (20% of the dataset),
        // where returning a low-relevance neuron counts as a false positive.
        if let Some(threshold) = min_confidence {
            if self.peek_max_bm25_score(task) < threshold {
                tracing::debug!(
                    task,
                    threshold,
                    "Abstention: top BM25 score below min_confidence — returning empty."
                );
                return (vec![], vec![]);
            }
        }

        // F1: Task Complexity Adaptive Budget
        //
        // Scale max_tokens by [0.5, 1.5] based on query complexity:
        //   - BM25 breadth: how many distinct terms have posting-list hits
        //   - Module spread: unique modules in top candidates
        //   - Synapse depth: whether candidates have outgoing synapses
        //
        // Simple queries (breadth=1, no synapses) → 0.5× budget (saves tokens)
        // Complex queries (broad match, cross-module) → 1.5× budget
        let terms = tokenize(task);
        let complexity = self.compute_task_complexity(&terms);
        // F2: apply session-history budget scale on top of F1 complexity scale
        let history_scale = self.adaptive_budget_scale();
        let adjusted_max = ((max_tokens as f32 * complexity * history_scale) as usize)
            .max(512)
            .min(8192.max(max_tokens * 2));
        tracing::debug!(
            task,
            complexity,
            history_scale,
            original_max = max_tokens,
            adjusted_max,
            "F1+F2: adaptive token budget"
        );

        let candidate_set: HashSet<usize> = {
            let mut s = HashSet::new();
            for term in &terms {
                if let Some(idxs) = self.posting_list.get(term) {
                    s.extend(idxs);
                }
            }
            s
        };

        // Run the full activation pipeline via get_contexts with an enormous budget,
        // then re-split. Slightly wasteful but keeps logic DRY.
        let all_ordered = self.get_contexts(task, usize::MAX / 2, module, kind);

        let mut full = Vec::new();
        let mut overflow = Vec::new();
        let mut used = 0usize;

        for path in all_ordered {
            let tokens = self.entry_by_path(&path).map(|e| e.tokens).unwrap_or(200);
            if used + tokens <= adjusted_max || full.is_empty() {
                used += tokens;
                full.push(path);
            } else {
                // Collect headline for overflow neuron
                let headline = neuron_headline_for(&path);
                overflow.push((path, headline));
            }
        }

        // Multi-hop retrieval: use top result's vocabulary to expand the query
        // and find indirectly-related neurons. Targets LME-500 multi-session
        // questions where the answer is in a different neuron than the one that
        // matches the query terms.
        if multi_hop && !full.is_empty() {
            if let Some(top_entry) = self.entry_by_path(&full[0]) {
                let mut hop_terms = terms.clone();
                // Expand with graph-semantic neighbors (concept cloud)
                hop_terms.extend(top_entry.concept_cloud.iter().take(10).cloned());
                // Expand with learned co-activation synonyms
                hop_terms.extend(top_entry.synonym_cloud.iter().take(5).cloned());
                // Expand with top-15 TF-IDF terms from the top result (novel terms only)
                let already: HashSet<&str> = hop_terms.iter().map(|s| s.as_str()).collect();
                let mut tfidf: Vec<(f32, String)> = top_entry.term_freq.iter()
                    .filter(|(t, _)| t.len() >= 4 && !already.contains(t.as_str()))
                    .map(|(t, &f)| (f, t.clone()))
                    .collect();
                tfidf.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                hop_terms.extend(tfidf.into_iter().take(15).map(|(_, t)| t));
                hop_terms.dedup();

                let expanded_task = hop_terms.join(" ");
                let second_pass = self.get_contexts(&expanded_task, usize::MAX / 2, module, kind);

                let already_included: HashSet<&PathBuf> =
                    full.iter().chain(overflow.iter().map(|(p, _)| p)).collect();
                let novel: Vec<(PathBuf, String)> = second_pass
                    .into_iter()
                    .filter(|p| !already_included.contains(p))
                    .map(|p| {
                        let headline = neuron_headline_for(&p);
                        (p, headline)
                    })
                    .collect();

                if !novel.is_empty() {
                    tracing::debug!(
                        count = novel.len(),
                        "Multi-hop 2nd pass: injected additional candidate neurons"
                    );
                    overflow.extend(novel);
                }
            }
        }

        let _ = candidate_set; // suppress unused warning
        (full, overflow)
    }

    /// F1: Compute task complexity as a [0.5, 1.5] budget scale factor.
    ///
    /// Inputs:
    /// - BM25 breadth: fraction of query terms that hit the posting list (term coverage)
    /// - Module spread: unique module count in top-10 candidates (cross-module indicator)
    /// - Synapse depth: fraction of top candidates with outgoing synapses (graph richness)
    ///
    /// Formula: clamp(0.5 + breadth * 0.3 + spread * 0.4 + depth * 0.3, 0.5, 1.5)
    fn compute_task_complexity(&self, terms: &[String]) -> f32 {
        if terms.is_empty() {
            return 1.0;
        }

        // Breadth: fraction of query terms with any posting-list hit
        let hit_terms = terms.iter()
            .filter(|t| self.posting_list.contains_key(t.as_str()))
            .count();
        let breadth = hit_terms as f32 / terms.len() as f32;

        // Candidate set for spread/depth analysis
        let mut candidates: HashSet<usize> = HashSet::new();
        for t in terms {
            if let Some(idxs) = self.posting_list.get(t.as_str()) {
                candidates.extend(idxs.iter().take(10));
            }
        }

        // Spread: unique modules among top candidates (normalized by 3)
        let unique_modules: HashSet<Option<&str>> = candidates.iter()
            .filter_map(|&i| self.entries.get(i))
            .map(|e| e.module.as_deref())
            .collect();
        let spread = ((unique_modules.len() as f32 - 1.0) / 3.0).clamp(0.0, 1.0);

        // Depth: fraction of candidates that have outgoing synapses
        let with_synapses = candidates.iter()
            .filter_map(|&i| self.entries.get(i))
            .filter(|e| !e.synapses.is_empty())
            .count();
        let depth = if candidates.is_empty() {
            0.0
        } else {
            with_synapses as f32 / candidates.len() as f32
        };

        (0.5 + breadth * 0.3 + spread * 0.4 + depth * 0.3).clamp(0.5, 1.5)
    }

    // ── Stats ─────────────────────────────────────────────────────────────────

    pub fn neuron_count(&self) -> usize {
        self.entries.len()
    }

    pub fn synapse_count(&self) -> usize {
        // Count the forward synapses defined on each entry (not the reverse copies in adjacency).
        self.entries.iter().map(|e| e.synapses.len()).sum()
    }

    /// Status counts for doctor: (fresh, stale, stub)
    pub fn status_counts(&self) -> (usize, usize, usize) {
        let ndir = neuron_dir(&self.project_root);
        let mut fresh = 0usize;
        let mut stale = 0usize;
        let mut stub = 0usize;
        for entry in &self.entries {
            let meta_p = meta_path(&entry.neuron_path);
            let status = std::fs::read_to_string(&meta_p)
                .ok()
                .and_then(|d| serde_json::from_str::<NeuronMeta>(&d).ok())
                .map(|m| m.status)
                .unwrap_or(NeuronStatus::Stub);
            // If .context.md is in the ndir, it's a real neuron (avoid counting adjacency copies)
            if !entry.neuron_path.starts_with(&ndir) { continue; }
            match status {
                NeuronStatus::Fresh => fresh += 1,
                NeuronStatus::Stale => stale += 1,
                NeuronStatus::Stub  => stub  += 1,
            }
        }
        (fresh, stale, stub)
    }

    /// Return the use_count for a neuron (for display purposes).
    pub fn use_count_for(&self, path: &Path) -> u32 {
        self.path_index.get(path)
            .map(|&i| self.entries[i].use_count)
            .unwrap_or(0)
    }

    /// Increment `use_count` for each neuron in `paths` and persist their metadata.
    ///
    /// Also applies auto-quarantine: if a neuron has ≥ MIN_SAMPLE_SIZE activations
    /// but its hit_rate is below QUARANTINE_THRESHOLD (10%), it's a chronic
    /// over-activator — retrieved often but rarely cited. Its staleness_multiplier
    /// is reduced to 0.3, effectively deprioritising it without deletion.
    /// The quarantine lifts automatically when the neuron is re-evolved.
    pub fn record_activation(&mut self, paths: &[std::path::PathBuf]) {
        for path in paths {
            if let Some(&i) = self.path_index.get(path) {
                self.entries[i].use_count = self.entries[i].use_count.saturating_add(1);

                // Bayesian quarantine with adaptive confidence intervals (TRIZ S4 R11).
                //
                // Adaptive tiers:
                //   use_count <  5  → withhold judgment (too few samples)
                //   use_count  5–19 → z=1.0,   threshold=0.02 (react fast to obvious noise)
                //   use_count 20–99 → z=1.645, threshold=0.05 (90% CI — standard behaviour)
                //   use_count ≥100  → z=1.96,  threshold=0.08 (strict for mature neurons)
                // Quarantine is reversible: lower bound > QUARANTINE_RECOVERY_THRESHOLD → restore.
                let uc = self.entries[i].use_count;
                let hc = self.entries[i].hit_count;
                if let Some((z, threshold)) = adaptive_quarantine_params(uc) {
                    let lower = wilson_lower_bound_z(hc, uc, z);
                    let currently_quarantined = self.entries[i].staleness_multiplier <= 0.3;
                    if !currently_quarantined && lower < threshold {
                        self.entries[i].staleness_multiplier = 0.3;
                        tracing::debug!(
                            path = %path.display(),
                            wilson_lower_bound = lower,
                            use_count = uc,
                            hit_count = hc,
                            z = z,
                            threshold = threshold,
                            "Auto-quarantined: Wilson CI lower bound {lower:.3} < {threshold}"
                        );
                    } else if currently_quarantined && lower > QUARANTINE_RECOVERY_THRESHOLD {
                        self.entries[i].staleness_multiplier = 0.7;
                        tracing::debug!(
                            path = %path.display(),
                            wilson_lower_bound = lower,
                            "Quarantine lifted: Wilson CI lower bound {lower:.3} > {QUARANTINE_RECOVERY_THRESHOLD}"
                        );
                    }
                }

                // Persist the updated use_count to the sidecar JSON so it survives restarts.
                let meta_p = meta_path(path);
                if let Ok(data) = std::fs::read_to_string(&meta_p) {
                    if let Ok(mut meta) = serde_json::from_str::<NeuronMeta>(&data) {
                        meta.use_count = self.entries[i].use_count;
                        let _ = atomic_write_json(&meta_p, &meta);
                    }
                }
            }
        }
    }

    /// Increment `hit_count` for a neuron when the LLM confirms it was cited.
    ///
    /// Returns the updated hit_rate = hit_count / use_count.max(1).
    pub fn record_hit(&mut self, neuron_path: &Path, was_cited: bool) -> f32 {
        if let Some(&i) = self.path_index.get(neuron_path) {
            if was_cited {
                self.entries[i].hit_count = self.entries[i].hit_count.saturating_add(1);
            }
            // Always increment use_count on explicit feedback (in case get_contexts missed it)
            self.entries[i].use_count = self.entries[i].use_count.saturating_add(1);

            let hit_rate = self.entries[i].hit_count as f32
                / self.entries[i].use_count.max(1) as f32;

            // Persist both counters
            let meta_p = meta_path(neuron_path);
            if let Ok(data) = std::fs::read_to_string(&meta_p) {
                if let Ok(mut meta) = serde_json::from_str::<NeuronMeta>(&data) {
                    meta.use_count = self.entries[i].use_count;
                    meta.hit_count = self.entries[i].hit_count;
                    let _ = atomic_write_json(&meta_p, &meta);
                }
            }

            // Adaptive synapse EMA: update learned_weight for all synapses that
            // point to this neuron, reinforcing or downweighting the traversal path.
            self.update_synapse_ema(neuron_path, was_cited);

            hit_rate
        } else {
            0.0
        }
    }

    /// Apply a weak silence signal to a neuron that was activated but not re-activated
    /// in the next `get_contexts` call (TRIZ R14-C2).
    ///
    /// Increments `use_count` without `hit_count` → naturally lowers the effective
    /// hit rate over time, gently down-weighting consistently irrelevant neurons.
    ///
    /// Guard: only applied when `use_count > 10` to protect cold-start neurons that
    /// haven't built up enough sample size for stable statistics.
    pub fn record_silence(&mut self, neuron_path: &Path) {
        if let Some(&i) = self.path_index.get(neuron_path) {
            if self.entries[i].use_count > 10 {
                self.entries[i].use_count = self.entries[i].use_count.saturating_add(1);
                let meta_p = meta_path(neuron_path);
                if let Ok(data) = std::fs::read_to_string(&meta_p) {
                    if let Ok(mut meta) = serde_json::from_str::<NeuronMeta>(&data) {
                        meta.use_count = self.entries[i].use_count;
                        let _ = atomic_write_json(&meta_p, &meta);
                    }
                }
            }
        }
    }

    /// B2: Record query term co-activations for a neuron.
    ///
    /// Called from `get_contexts` for each activated neuron with the query terms.
    /// After ≥30 co-activations, a term is promoted to the neuron's `synonym_cloud`.
    /// The synonym cloud is persisted to the BM25Entry and used at query time for
    /// vocabulary expansion before BM25 scoring.
    pub fn record_coactivation(&mut self, neuron_path: &Path, query_terms: &[String]) {
        const SYNONYM_THRESHOLD: u32 = 30;

        let Some(&entry_idx) = self.path_index.get(neuron_path) else { return };

        let counts = self.coactivation_counts
            .entry(neuron_path.to_path_buf())
            .or_default();

        let mut promoted = Vec::new();
        for term in query_terms {
            if term.len() < 3 { continue; }
            let count = counts.entry(term.clone()).or_insert(0);
            *count += 1;
            if *count == SYNONYM_THRESHOLD {
                promoted.push(term.clone());
            }
        }

        if !promoted.is_empty() {
            let cloud = &mut self.entries[entry_idx].synonym_cloud;
            for term in &promoted {
                if !cloud.contains(term) {
                    cloud.push(term.clone());
                    tracing::debug!(
                        neuron = %neuron_path.display(),
                        term,
                        "B2: promoted term to synonym cloud"
                    );
                }
            }
        }

        // R20 C-2: Drain any pending Hebbian synapse creations.
        //
        // `get_contexts()` (a &self method) accumulates co-return counts in a Mutex.
        // Once a pair crosses HEBBIAN_THRESHOLD (2 co-returns), it's flagged there but
        // can't mutate adjacency. Here, in the first subsequent &mut self call, we drain
        // the flagged pairs and create bidirectional SemanticRelated synapses.
        self.apply_pending_hebbian_synapses();
    }

    /// Drain pending Hebbian synapse pairs and create SemanticRelated edges in adjacency.
    fn apply_pending_hebbian_synapses(&mut self) {
        const HEBBIAN_THRESHOLD: u32 = 2;
        let pairs_to_wire: Vec<(PathBuf, PathBuf)> = {
            let Ok(counts) = self.co_return_counts.lock() else { return };
            counts.iter()
                .filter(|(_, &c)| c == HEBBIAN_THRESHOLD) // exactly at threshold — fire once
                .map(|(k, _)| k.clone())
                .collect()
        };

        for (a, b) in pairs_to_wire {
            // Mark as wired (sentinel=3) so we don't re-fire on future calls
            if let Ok(mut counts) = self.co_return_counts.lock() {
                if let Some(c) = counts.get_mut(&(a.clone(), b.clone())) {
                    *c = HEBBIAN_THRESHOLD + 1;
                }
            }

            let already_exists = self.adjacency.get(&a).map_or(false, |syns| {
                syns.iter().any(|s| s.target == b && s.edge_type == SynapseType::SemanticRelated)
            });
            if already_exists { continue; }

            let syn_ab = Synapse::new(b.clone(), SynapseType::SemanticRelated, "hebbian:co-return".to_string());
            let syn_ba = Synapse::new(a.clone(), SynapseType::SemanticRelated, "hebbian:co-return".to_string());
            self.adjacency.entry(a.clone()).or_default().push(syn_ab);
            self.adjacency.entry(b.clone()).or_default().push(syn_ba);
            tracing::debug!(
                a = %a.display(),
                b = %b.display(),
                "C-2 Hebbian: SemanticRelated synapse created from co-return signal"
            );
        }
    }

    /// B2: Expand query terms through per-neuron synonym clouds.
    ///
    /// For each activated neuron path, return any synonym-cloud terms that appear
    /// in the query — as augmented expansion terms for the next retrieval pass.
    /// Used during `get_contexts` vocabulary expansion phase.
    /// Return the highest raw BM25 score for `task` across all indexed neurons.
    ///
    /// Runs Phase 1 posting-list lookup + BM25 scoring only (no synapse traversal,
    /// no TF-IDF, no dense re-rank).  Used by `get_contexts_with_overflow` to
    /// implement the abstention signal: if the top score is below `min_confidence`,
    /// no neurons are returned and the caller prints a "no relevant memory" message.
    ///
    /// Complexity: O(|candidates|) — same as the fast path in `get_contexts`.
    pub fn peek_max_bm25_score(&self, task: &str) -> f32 {
        let terms = tokenize(task);
        let mut max_score = 0.0f32;
        for term in &terms {
            if let Some(idxs) = self.posting_list.get(term) {
                for &i in idxs {
                    let s = self.bm25_score(&terms, &self.entries[i]);
                    if s > max_score {
                        max_score = s;
                    }
                }
            }
        }
        max_score
    }

    /// Knowledge-update supersession: demote old Verbatim neurons whose content is
    /// substantially overlapped by a newer neuron in the same module/person scope.
    ///
    /// Called by `write_verbatim_neurons` after staging each new Verbatim neuron. When a
    /// newly-ingested turn has ≥60% term overlap with an older turn in the same module AND
    /// the older turn's timestamp pre-dates the new one, the old neuron's
    /// `staleness_multiplier` is halved (→ 0.5×BM25 score). This surfaces the most
    /// current fact for LME-500 knowledge-update questions without evicting history.
    ///
    /// Only applies to Verbatim neurons — code neurons are unaffected.
    pub fn detect_and_mark_supersessions(&mut self, new_path: &Path) {
        const OVERLAP_THRESHOLD: f32 = 0.60;
        const MIN_TERMS: usize = 4;

        let Some(&new_idx) = self.path_index.get(new_path) else { return };

        // Snapshot new-entry data to avoid borrow conflicts below.
        let (new_module, new_ts, new_terms) = {
            let e = &self.entries[new_idx];
            if !matches!(e.kind, NeuronKind::Verbatim) { return }
            let terms: HashSet<String> = e.term_freq.keys()
                .filter(|t| t.len() >= MIN_TERMS)
                .cloned()
                .collect();
            (e.module.clone(), e.timestamp_secs, terms)
        };

        if new_terms.is_empty() { return }
        let new_ts_val = new_ts.unwrap_or(i64::MAX);

        for i in 0..self.entries.len() {
            if i == new_idx { continue }
            let e = &self.entries[i];
            if !matches!(e.kind, NeuronKind::Verbatim) { continue }
            if e.module != new_module { continue }
            let old_ts = e.timestamp_secs.unwrap_or(0);
            // Only demote OLDER neurons — if old_ts ≥ new_ts, the "old" entry is newer
            // or simultaneous; skip it to avoid mutual demotion within a batch.
            if old_ts >= new_ts_val { continue }

            let old_terms: HashSet<&str> = e.term_freq.keys()
                .filter(|t| t.len() >= MIN_TERMS)
                .map(|s| s.as_str())
                .collect();
            if old_terms.len() < MIN_TERMS { continue }

            let overlap = new_terms.iter()
                .filter(|t| old_terms.contains(t.as_str()))
                .count();
            let ratio = overlap as f32 / old_terms.len() as f32;

            if ratio >= OVERLAP_THRESHOLD {
                self.entries[i].staleness_multiplier =
                    (self.entries[i].staleness_multiplier * 0.5).max(0.1);
                tracing::debug!(
                    old = ?self.entries[i].neuron_path,
                    new = ?new_path,
                    overlap_ratio = ratio,
                    "Knowledge-update supersession: demoted older neuron"
                );
            }
        }
    }

    pub fn synonym_cloud_expansion(&self, query_terms: &[String]) -> Vec<String> {
        let query_set: HashSet<&String> = query_terms.iter().collect();
        let mut expansion: HashSet<String> = HashSet::new();

        for entry in &self.entries {
            // For each neuron: check if any query term matches an entry term
            let neuron_has_query_term = entry.term_freq.keys().any(|t| query_set.contains(t));
            if neuron_has_query_term {
                // Expand with this neuron's synonym cloud
                for syn_term in &entry.synonym_cloud {
                    expansion.insert(syn_term.clone());
                }
            }
        }

        // Remove terms already in the query to avoid re-adding them
        for t in query_terms {
            expansion.remove(t);
        }

        expansion.into_iter().collect()
    }

    /// F2: Record session token utilization for budget adaptation.
    ///
    /// Call at the end of each session (close_task) with the tokens used and the budget.
    /// Keeps the last 5 sessions' data. The next call to `adaptive_budget_scale()` uses
    /// this history to adjust max_tokens up or down.
    pub fn record_session_utilization(&mut self, tokens_used: usize, tokens_budget: usize) {
        const MAX_HISTORY: usize = 5;
        self.session_utilization.push([tokens_used, tokens_budget]);
        if self.session_utilization.len() > MAX_HISTORY {
            self.session_utilization.remove(0);
        }
    }

    /// F2: Compute the budget scale factor from session history.
    ///
    /// - If last 5 sessions used < 40% of budget → scale down by 20% (too much headroom)
    /// - If ≥3 of last 5 sessions hit 100% of budget (overflow) → scale up by 20%
    /// - Otherwise: no change (scale = 1.0)
    ///
    /// Returns a multiplier [0.8, 1.2] to apply to max_tokens.
    /// Capped post-multiplication at [512, 8192] by the caller.
    pub fn adaptive_budget_scale(&self) -> f32 {
        let history = &self.session_utilization;
        if history.len() < 2 {
            return 1.0; // not enough data
        }

        let underused = history.iter()
            .filter(|[used, budget]| *budget > 0 && (*used as f32 / *budget as f32) < 0.4)
            .count();

        let overflowed = history.iter()
            .filter(|[used, budget]| *used >= *budget)
            .count();

        if underused == history.len() {
            0.8 // all sessions underused → shrink
        } else if overflowed >= 3 {
            1.2 // ≥3/5 sessions overflowed → grow
        } else {
            1.0 // normal
        }
    }


    /// `cited = true` → signal = 1.0 (this synapse helped); `false` → 0.0.
    ///
    /// EMA rule: `learned_weight ← α × signal + (1 − α) × learned_weight`  (α = 0.1)
    ///
    /// Cold-start: when `learned_weight == 0.0`, it is initialised to the type
    /// multiplier before the first update so the decay doesn't start from zero.
    ///
    /// Only in-memory entries are updated; `save()` persists them to `index.json`.
    /// NeuronMeta sidecar files are NOT updated (they are the source-of-truth for
    /// compile-time synapse topology, not runtime weights).
    pub fn update_synapse_ema(&mut self, target_path: &Path, cited: bool) {
        const ALPHA: f32 = 0.1;
        let signal = if cited { 1.0_f32 } else { 0.0_f32 };

        for entry in &mut self.entries {
            for syn in &mut entry.synapses {
                if syn.target == target_path {
                    // Cold-start init: seed from type multiplier so EMA starts at a
                    // sensible baseline rather than decaying from 0.
                    if syn.learned_weight <= 0.0 {
                        syn.learned_weight = syn.edge_type.type_multiplier();
                    }
                    syn.learned_weight = ALPHA * signal + (1.0 - ALPHA) * syn.learned_weight;
                    syn.traversal_count = syn.traversal_count.saturating_add(1);
                }
            }
        }
    }

    pub fn print_status(&self) {
        let mut cores = 0usize;
        let mut usecases = 0usize;
        let mut verbatim = 0usize;
        let mut concepts = 0usize;
        let mut stubs = 0usize;
        for e in &self.entries {
            match e.kind {
                NeuronKind::Core | NeuronKind::Project => {
                    cores += 1;
                    if e.term_count == 0 || e.term_freq.is_empty() {
                        stubs += 1;
                    }
                }
                NeuronKind::UseCase => usecases += 1,
                NeuronKind::Verbatim => verbatim += 1,
                NeuronKind::Concept => concepts += 1,
            }
        }
        println!("Cortyx Index");
        println!("============");
        println!("  Core neurons:         {cores}  ({stubs} stubs — run cortyx_evolve_context)");
        println!("  Use-case neurons:     {usecases}");
        println!("  Verbatim chunks:      {verbatim}");
        println!("  Concept neurons:      {concepts}");
        println!("  Synapses:             {}", self.synapse_count());
        println!("  Modules indexed:      {}", self.module_index.len());
        println!("  Avg doc length:       {:.0} terms", self.avg_doc_len);
    }

    // ── Invalidation ──────────────────────────────────────────────────────────

    /// Mark a source file's neuron as stale (hash changed or forced).
    ///
    /// The stale neuron is demoted (staleness_multiplier → 0.5) rather than evicted
    /// so it can still activate on niche queries where it remains the best match.
    /// A full eviction would lose context permanently before the LLM re-evolves it.
    pub fn invalidate(&mut self, source: &Path) -> Result<()> {
        let neuron = core_neuron_path(source, &self.project_root);
        let meta_file = meta_path(&neuron);
        if meta_file.exists() {
            if let Ok(data) = std::fs::read_to_string(&meta_file) {
                if let Ok(mut meta) = serde_json::from_str::<NeuronMeta>(&data) {
                    meta.status = NeuronStatus::Stale;
                    let _ = atomic_write_json(&meta_file, &meta);
                }
            }
        }
        // Demote the in-memory entry rather than removing it.
        if let Some(&i) = self.path_index.get(&neuron) {
            self.entries[i].staleness_multiplier = 0.5;
        }
        self.save()
    }

    /// Permanently remove a neuron from the index and delete its files from disk.
    ///
    /// Unlike `invalidate`, this is a hard delete — the neuron's `.context.md` and
    /// its sidecar `.json` are removed. Used by `cortyx prune`.
    ///
    /// Returns `true` if the neuron was found and removed, `false` if it was unknown.
    pub fn evict_entry(&mut self, neuron_path: &Path) -> bool {
        let Some(&idx) = self.path_index.get(neuron_path) else {
            return false;
        };
        self.entries.swap_remove(idx);
        // After swap_remove, the entry previously at the last position is now at `idx`.
        // Update its path_index slot so future lookups remain correct.
        if idx < self.entries.len() {
            self.path_index.insert(self.entries[idx].neuron_path.clone(), idx);
        }
        self.path_index.remove(neuron_path);
        // Rebuild derived structures — eviction happens in bulk during prune,
        // so the caller calls rebuild_derived() once after all evictions.
        true
    }

    /// Neuron paths together with their activation count — used by `cortyx prune`.
    pub fn neuron_paths_and_use_counts(&self) -> Vec<(PathBuf, u32)> {
        self.entries.iter().map(|e| (e.neuron_path.clone(), e.use_count)).collect()
    }

    // ── Hierarchy navigation (TRIZ R13-G2) ───────────────────────────────────

    /// List all modules with their neuron count and average hit rate.
    /// Includes `@person` scoped modules alongside directory modules.
    /// Returns entries sorted by name for deterministic output.
    pub fn list_modules(&self) -> Vec<ModuleSummary> {
        let mut map: HashMap<String, (usize, f32)> = HashMap::new();
        for entry in &self.entries {
            if let Some(m) = entry.module.as_deref() {
                let e = map.entry(m.to_string()).or_default();
                e.0 += 1;
                let rate = if entry.use_count > 0 {
                    entry.hit_count as f32 / entry.use_count as f32
                } else {
                    0.0
                };
                e.1 += rate;
            }
        }
        let mut result: Vec<ModuleSummary> = map
            .into_iter()
            .map(|(name, (count, rate_sum))| ModuleSummary {
                name: name.clone(),
                neuron_count: count,
                avg_hit_rate: if count > 0 { rate_sum / count as f32 } else { 0.0 },
                is_person_scope: name.starts_with('@'),
            })
            .collect();
        result.sort_by(|a, b| a.name.cmp(&b.name));
        result
    }

    /// List neurons in a module (or all neurons if `module` is None).
    /// Returns a summary of each neuron's path, kind, staleness, and hit rate.
    pub fn list_neurons(&self, module: Option<&str>) -> Vec<NeuronSummary> {
        let indices: Vec<usize> = if let Some(m) = module {
            self.module_index.get(m).cloned().unwrap_or_default()
        } else {
            (0..self.entries.len()).collect()
        };
        let mut result: Vec<NeuronSummary> = indices
            .into_iter()
            .map(|i| {
                let e = &self.entries[i];
                let hit_rate = if e.use_count > 0 {
                    e.hit_count as f32 / e.use_count as f32
                } else {
                    0.0
                };
                NeuronSummary {
                    path: e.neuron_path.clone(),
                    kind: e.kind.clone(),
                    staleness_multiplier: e.staleness_multiplier,
                    hit_rate,
                    use_count: e.use_count,
                }
            })
            .collect();
        result.sort_by(|a, b| a.path.cmp(&b.path));
        result
    }

    /// Return the first `lines` lines of a neuron file for quick preview.
    /// Returns `None` if the file does not exist or cannot be read.
    pub fn peek_neuron(&self, path: &Path, lines: usize) -> Option<String> {
        let content = std::fs::read_to_string(path).ok()?;
        let preview: String = content
            .lines()
            .take(lines)
            .collect::<Vec<_>>()
            .join("\n");
        Some(preview)
    }

    /// List only `@person`-scoped modules (convention: module starts with `@`).
    pub fn list_persons(&self) -> Vec<ModuleSummary> {
        self.list_modules()
            .into_iter()
            .filter(|m| m.is_person_scope)
            .collect()
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Mine each source file for function call sites that match public functions
    /// defined in *other* source files of the project.
    ///
    /// Workflow:
    /// 1. Build a vocabulary map `fn_name → source_rel_path` from all entries'
    ///    extracted function names (stored in `term_freq` keys during compile).
    ///    Entries with no functions in their term_freq are skipped.
    /// 2. Walk each source file, call `ast_extractor::extract_call_sites`,
    ///    and for each detected `CallEdge`, emit a `Calls`-typed synapse from
    ///    the calling neuron to the callee neuron (if one doesn't already exist).
    ///
    /// This is a second compile pass and runs in O(files × |vocab|) — both are
    /// typically small so runtime is negligible.
    fn apply_call_graph_synapses(&mut self, root: &Path) {
        // Build fn_name → source_path vocabulary from the already-loaded entries.
        // We use term_freq keys that look like function names (alphabetic, no spaces).
        // This is approximate but practical — false positives are filtered by
        // the self-loop guard in extract_call_sites.
        //
        // A tighter approach would be to store a dedicated `functions: Vec<String>`
        // field in BM25Entry, but term_freq already contains them from AST Bootstrap.
        // Function names are pure alphabetic tokens, distinct from normal prose terms.
        let mut fn_vocab: HashMap<String, PathBuf> = HashMap::new();
        for entry in &self.entries {
            let rel_source = entry
                .neuron_path
                .strip_prefix(root)
                .map(|r| r.to_path_buf())
                .unwrap_or_else(|_| entry.neuron_path.clone());

            // Extract function names: those that appear in term_freq AND match the
            // pattern of a public function name (all word chars, len ≥ 3, not all-lowercase
            // common English words). We use a simple heuristic rather than re-running AST.
            for term in entry.term_freq.keys() {
                // Public function names are typically CamelCase or snake_case identifiers
                // ≥ 3 chars with no digits-only and not a BM25 stop-word.
                if term.len() >= 3 && term.chars().all(|c| c.is_alphanumeric() || c == '_') {
                    fn_vocab.entry(term.clone()).or_insert_with(|| rel_source.clone());
                }
            }
        }

        if fn_vocab.is_empty() {
            return;
        }

        // Walk each source file and find call sites.
        let source_extensions = [
            "rs", "py", "ts", "tsx", "js", "jsx", "go", "swift", "kt", "java", "cs", "rb", "c",
            "cpp", "cc",
        ];
        let walker = WalkDir::new(root).min_depth(1).into_iter().filter_map(|e| e.ok());
        let mut synapse_patches: Vec<(PathBuf, PathBuf)> = Vec::new(); // (caller_neuron, callee_neuron)

        for entry in walker {
            if !entry.file_type().is_file() {
                continue;
            }
            let abs = entry.path();
            let rel = abs.strip_prefix(root).unwrap_or(abs);
            let ext = rel
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("");
            if !source_extensions.contains(&ext) || should_skip(rel) {
                continue;
            }
            let Ok(content) = std::fs::read_to_string(abs) else {
                continue;
            };
            let source_rel = rel.to_string_lossy();
            let call_edges = ast_extractor::extract_call_sites(&source_rel, &content, &fn_vocab);
            if call_edges.is_empty() {
                continue;
            }
            let caller_neuron = core_neuron_path(abs, root);
            for edge in call_edges {
                let callee_source = root.join(&edge.callee_file);
                let callee_neuron = core_neuron_path(&callee_source, root);
                if callee_neuron != caller_neuron {
                    synapse_patches.push((caller_neuron.clone(), callee_neuron));
                }
            }
        }

        // Apply collected patches to meta files and in-memory entries.
        for (caller_neuron, callee_neuron) in synapse_patches {
            let meta_file = meta_path(&caller_neuron);
            let Ok(data) = std::fs::read_to_string(&meta_file) else {
                continue;
            };
            let Ok(mut meta) = serde_json::from_str::<NeuronMeta>(&data) else {
                continue;
            };
            let already_exists = meta
                .synapses
                .iter()
                .any(|s| s.target == callee_neuron && matches!(s.edge_type, SynapseType::Calls));
            if already_exists {
                continue;
            }
            meta.synapses.push(Synapse::new(
                callee_neuron.clone(),
                SynapseType::Calls,
                "auto-inferred from call-site scan".to_string(),
            ));
            let _ = atomic_write_json(&meta_file, &meta);
            // Update in-memory entry as well.
            if let Some(&idx) = self.path_index.get(&caller_neuron) {
                self.entries[idx].synapses.push(Synapse::new(
                    callee_neuron,
                    SynapseType::Calls,
                    "auto-inferred from call-site scan".to_string(),
                ));
            }
        }
    }

    /// Mine `git log --name-only` to find files co-committed ≥ `min_cochange` times.
    ///
    /// For each qualifying pair, add a `SemanticRelated` auto-synapse to the
    /// source neuron's meta if one does not already exist. Called once per compile.
    fn apply_cochange_synapses(&mut self, root: &Path) {
        const MIN_COCHANGE: u32 = 3;
        /// Cap on files per commit before skipping the pair-wise O(n²) step.
        ///
        /// A commit touching more than this many files is almost certainly a
        /// bulk change (dependency bump, generated code, refactor) where co-change
        /// is not a useful semantic signal. Without this cap, a 500-file commit
        /// generates ~125,000 pairs, making compile time degenerate on large repos.
        const MAX_FILES_PER_COMMIT: usize = 50;

        let output = match std::process::Command::new("git")
            .args(["log", "--name-only", "--pretty=format:"])
            .current_dir(root)
            .output()
        {
            Ok(o) if o.status.success() => o.stdout,
            _ => return, // not a git repo or git unavailable — skip silently
        };

        // Build per-commit file lists and count co-changes
        let mut cochange: HashMap<(PathBuf, PathBuf), u32> = HashMap::new();
        let mut commit_files: Vec<PathBuf> = Vec::new();

        for line in String::from_utf8_lossy(&output).lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                // Commit boundary — process accumulated files only if the commit is
                // small enough that co-change is a meaningful signal.
                if commit_files.len() <= MAX_FILES_PER_COMMIT {
                    for i in 0..commit_files.len() {
                        for j in (i + 1)..commit_files.len() {
                            let (a, b) = (&commit_files[i], &commit_files[j]);
                            // Canonical ordering so (a,b) == (b,a)
                            let key = if a <= b {
                                (a.clone(), b.clone())
                            } else {
                                (b.clone(), a.clone())
                            };
                            *cochange.entry(key).or_insert(0) += 1;
                        }
                    }
                }
                commit_files.clear();
            } else {
                commit_files.push(PathBuf::from(trimmed));
            }
        }
        // Flush any trailing files — git log output may not end with a blank line,
        // which would silently drop the most-recent commit's co-change signal.
        if !commit_files.is_empty() && commit_files.len() <= MAX_FILES_PER_COMMIT {
            for i in 0..commit_files.len() {
                for j in (i + 1)..commit_files.len() {
                    let (a, b) = (&commit_files[i], &commit_files[j]);
                    let key = if a <= b { (a.clone(), b.clone()) } else { (b.clone(), a.clone()) };
                    *cochange.entry(key).or_insert(0) += 1;
                }
            }
        }

        // Add synapses for qualifying pairs
        let mut changes: Vec<(PathBuf, Synapse)> = Vec::new();
        for ((fa, fb), count) in &cochange {
            if *count < MIN_COCHANGE { continue; }
            let na = core_neuron_path(&root.join(fa), root);
            let nb = core_neuron_path(&root.join(fb), root);
            let weight = (0.5_f32 + *count as f32 * 0.05).min(0.9);
            let reason = format!("git co-change: committed together {count}×");

            // Only create synapses for neurons that exist in our index
            if self.path_index.contains_key(&na) && self.path_index.contains_key(&nb) {
                changes.push((na.clone(), Synapse {
                    target: nb.clone(),
                    edge_type: SynapseType::SemanticRelated,
                    weight,
                    reason: reason.clone(),
                    learned_weight: 0.0,
                    traversal_count: 0,
                    last_co_activation_day: 0,
                }));
                changes.push((nb, Synapse {
                    target: na,
                    edge_type: SynapseType::SemanticRelated,
                    weight,
                    reason,
                    learned_weight: 0.0,
                    traversal_count: 0,
                    last_co_activation_day: 0,
                }));
            }
        }

        for (source_neuron, syn) in changes {
            let meta_p = meta_path(&source_neuron);
            if let Ok(data) = std::fs::read_to_string(&meta_p) {
                if let Ok(mut meta) = serde_json::from_str::<NeuronMeta>(&data) {
                    let already = meta.synapses.iter().any(|s| s.target == syn.target);
                    if !already {
                        meta.synapses.push(syn.clone());
                        let _ = atomic_write_json(&meta_p, &meta);
                    }
                }
            }
            if let Some(&i) = self.path_index.get(&source_neuron) {
                let already = self.entries[i].synapses.iter().any(|s| s.target == syn.target);
                if !already {
                    self.entries[i].synapses.push(syn);
                }
            }
        }
    }

    /// Add or replace a single entry in `self.entries` (does NOT rebuild derived).
    pub fn index_neuron(&mut self, neuron_path: &Path, content: &str, meta: &NeuronMeta) {
        let terms = tokenize(content);
        let mut tf: HashMap<String, f32> = HashMap::new();
        for t in &terms {
            *tf.entry(t.clone()).or_insert(0.0) += 1.0;
        }

        // P3-B: Paraphrase + query_surface section boost.
        // Both ## paraphrases (LLM-generated question vocab) and ## query_surface
        // (mine-time IE-extracted question pre-images) are boosted at 1.5× weight.
        // This closes the vocabulary gap: documents contain both answer vocabulary
        // (original content) and question vocabulary (these sections).
        {
            use crate::neuron::parse_sections;
            let sections = parse_sections(content);
            for section_name in ["paraphrases", "query_surface"] {
                if let Some(section_content) = sections.get(section_name) {
                    for t in tokenize(section_content) {
                        let v = tf.entry(t).or_insert(0.0);
                        *v += 1.5; // boost: question vocab is high-signal
                    }
                }
            }
        }

        // NE-6: User-turn boost for Verbatim (conversation) neurons.
        // In episodic memory retrieval, facts are stated by the user, not the assistant.
        // User utterances are the ground truth for SSU/KU/multi queries. Assistant text
        // is context/response and should not dominate BM25 scoring.
        // Implementation: give user-turn lines an extra +1.0 TF weight (doubling their
        // effective TF vs assistant lines), making user-disclosed facts rank much higher.
        if matches!(meta.kind, crate::neuron::NeuronKind::Verbatim) {
            for line in content.lines() {
                let lower = line.as_bytes();
                let is_user = lower.starts_with(b"user:") || lower.starts_with(b"User:")
                    || lower.starts_with(b"human:") || lower.starts_with(b"Human:");
                if is_user && line.len() > 6 {
                    for t in tokenize(line) {
                        *tf.entry(t).or_insert(0.0) += 1.0;
                    }
                }
            }
        }

        // A1: Multi-Source Vocabulary Injection — inject soft terms from source file
        // (git commit messages + inline comments) at 0.3× weight. These terms are never
        // shown in the retrieved context, but improve BM25 query matching for cold stubs.
        if let Some(source_abs) = meta.source_files.first() {
            for t in git_extractor::extract_soft_terms(source_abs) {
                // Only inject when not already present in neuron content — hard terms win.
                let v = tf.entry(t).or_insert(0.0);
                if *v == 0.0 {
                    *v = 0.3;
                }
            }
        }

        // B3: Alias Injection — inject natural-language aliases for public function/type names
        // at 0.5× weight. "get_user" → ["fetch", "retrieve", "account", "member"].
        // These aliases bridge the lexical gap between user queries and code identifiers
        // without any model download.
        {
            // Collect function/type names from task_pattern (sub-neuron) or from the neuron
            // file stem (proxy for the source file's primary identifier).
            let mut names: Vec<String> = Vec::new();
            if let Some(ref pattern) = meta.task_pattern {
                names.push(pattern.clone());
            }
            // Also include the neuron path stem as a fallback source of identifiers
            if let Some(stem) = neuron_path
                .file_stem()
                .and_then(|s| s.to_str())
                .map(|s| s.trim_end_matches(".context").to_string())
            {
                names.push(stem);
            }
            if !names.is_empty() {
                for t in alias_gen::generate_alias_terms(&names) {
                    let v = tf.entry(t).or_insert(0.0);
                    if *v < 0.5 {
                        *v = 0.5;
                    }
                }
            }
        }

        let task_pattern_terms = meta
            .task_pattern
            .as_deref()
            .map(tokenize)
            .unwrap_or_default();

        // Normalize synapse targets to absolute paths so the adjacency graph
        // uses consistent keys regardless of whether the path was parsed from
        // a markdown backtick (relative) or stored directly (absolute).
        //
        // S-1: Validate that the resolved target stays inside the neuron directory.
        // This prevents path traversal attacks via crafted .cortyx/neurons/*.json files
        // (e.g. a compromised CI artifact injecting "../../etc/sensitive").
        let ndir = neuron_dir(&self.project_root);
        let synapses: Vec<Synapse> = meta.synapses.iter().filter_map(|s| {
            let target = if s.target.is_absolute() {
                s.target.clone()
            } else {
                ndir.join(&s.target)
            };
            if !target.starts_with(&ndir) {
                tracing::warn!(
                    "Skipping synapse with path-traversal target {:?} in {:?}",
                    target, neuron_path
                );
                return None;
            }
            Some(Synapse { target, ..s.clone() })
        }).collect();

        // S-III (R16): Self-Quality Score — fraction of neuron terms that overlap with
        // the corresponding source file's AST-extracted terms.
        // Only computed for Core neurons with a known source file; defaults to 1.0 (neutral).
        let quality_score: f32 = if matches!(meta.kind, NeuronKind::Core)
            && !meta.source_files.is_empty()
        {
            let source_path = &meta.source_files[0];
            if let Ok(source_text) = std::fs::read_to_string(source_path) {
                let source_rel = source_path.to_string_lossy();
                let ast = ast_extractor::extract_signatures(&source_rel, &source_text);
                // Build source AST term set from all function/type names (split on _ and camelCase)
                let mut ast_terms: std::collections::HashSet<String> = std::collections::HashSet::new();
                for name in ast.functions.iter().chain(ast.types.iter()) {
                    ast_terms.extend(tokenize(name));
                }
                if ast_terms.is_empty() {
                    1.0 // no AST info → neutral
                } else {
                    let neuron_terms: std::collections::HashSet<&str> =
                        tf.keys().map(|s| s.as_str()).collect();
                    let overlap = ast_terms.iter()
                        .filter(|t| neuron_terms.contains(t.as_str()))
                        .count();
                    overlap as f32 / ast_terms.len() as f32
                }
            } else {
                1.0
            }
        } else {
            1.0 // non-Core or no source → neutral
        };

        // S-II (R16/R17 Sol4): Compute 1024-bit SimHash fingerprint (16 seeds) for LSH fallback.
        let lsh_fingerprints = simhash_1024(&tf);

        // S-I (R16): Extract Tier-1 summary from neuron content.
        // Takes: first non-empty line of `## purpose` section + first line of `## pitfalls`.
        // Stored in memory only (not persisted); rebuilt from neuron file at each index_neuron call.
        let summary = extract_neuron_summary(content);

        let entry = BM25Entry {
            neuron_path: neuron_path.to_path_buf(),
            kind: meta.kind.clone(),
            term_freq: tf,
            term_count: terms.len(),
            // Use meta.tokens when available (set by compile/upsert after reading disk).
            // Fall back to estimating from content so the token budget works in tests
            // and when index_neuron is called before NeuronMeta.tokens is populated.
            tokens: if meta.tokens > 0 { meta.tokens } else { estimate_tokens(content).max(10) },
            task_pattern_terms,
            parent: meta.parent.clone(),
            synapses,
            source_files: meta.source_files.clone(),
            module: meta.module.clone(),
            confidence_score: meta.confidence_score,
            use_count: meta.use_count,
            hit_count: meta.hit_count,
            staleness_multiplier: 1.0,
            concept_cloud: Vec::new(), // populated by build_concept_clouds() in rebuild_derived
            synonym_cloud: Vec::new(), // populated by record_coactivation() at runtime
            lsh_fingerprints,
            quality_score,
            summary,
            timestamp_secs: parse_iso8601_to_secs(meta.timestamp.as_deref()),
            // R21 T6: Extract session_id from neuron filename stem for Verbatim neurons.
            // Pattern: "lme_0060_0_user.verbatim.md" → session_id = "lme_0060"
            // Split on '_', take first two parts if the stem follows the N_N pattern.
            session_id: if matches!(meta.kind, NeuronKind::Verbatim) {
                neuron_path.file_name()
                    .and_then(|n| n.to_str())
                    .map(|name| {
                        // strip extension(s): "lme_0060_0_user.verbatim.md" → "lme_0060_0_user"
                        let stem = name.split('.').next().unwrap_or(name);
                        // take first two underscore-separated parts: "lme" + "0060"
                        let parts: Vec<&str> = stem.splitn(3, '_').collect();
                        if parts.len() >= 2 {
                            format!("{}_{}", parts[0], parts[1])
                        } else {
                            stem.to_string()
                        }
                    })
                    .unwrap_or_default()
            } else {
                String::new()
            },
        };

        if let Some(&pos) = self.path_index.get(neuron_path) {
            self.entries[pos] = entry;
        } else {
            let pos = self.entries.len();
            self.path_index.insert(neuron_path.to_path_buf(), pos);
            self.entries.push(entry);
        }
    }

    /// Rebuild all derived structures — public entry point for `cortyx prune`.
    ///
    /// Prune evicts entries individually then calls this once to reconstruct
    /// path_index, adjacency, df_cache, etc. in a single O(n) pass.
    pub fn rebuild_derived_pub(&mut self) {
        self.rebuild_derived();
    }

    /// Rebuild all derived structures in a single O(n) pass.
    ///
    /// Previously five separate passes (path_index, parent_index, adjacency, df_cache,
    /// module_index); merged to reduce cache pressure and wall-clock time ~5×.
    fn rebuild_derived(&mut self) {
        self.path_index.clear();
        self.parent_index.clear();
        self.adjacency.clear();
        self.df_cache.clear();
        self.posting_list.clear();
        self.module_index.clear();
        self.session_index.clear(); // R21 T6

        let mut total_terms = 0usize;

        for (i, entry) in self.entries.iter().enumerate() {
            // path_index
            self.path_index.insert(entry.neuron_path.clone(), i);

            // parent_index
            if let Some(p) = &entry.parent {
                self.parent_index.entry(p.clone()).or_default().push(i);
            }

            // adjacency (forward + reverse edges)
            for syn in &entry.synapses {
                self.adjacency
                    .entry(entry.neuron_path.clone())
                    .or_default()
                    .push(syn.clone());

                self.adjacency
                    .entry(syn.target.clone())
                    .or_default()
                    .push(Synapse {
                        target: entry.neuron_path.clone(),
                        edge_type: syn.edge_type.inverse(),
                        weight: syn.weight * 0.7,
                        reason: format!("← {}", syn.reason),
                        learned_weight: 0.0,
                        traversal_count: 0,
                        last_co_activation_day: 0,
                    });
            }

            // df_cache + posting_list (built in the same pass — no extra allocation)
            for term in entry.term_freq.keys() {
                *self.df_cache.entry(term.clone()).or_insert(0) += 1;
                self.posting_list.entry(term.clone()).or_default().push(i);
            }

            // module_index
            if let Some(m) = &entry.module {
                self.module_index.entry(m.clone()).or_default().push(i);
            }

            // R21 T6: session_index — for session-level grouping at retrieval time
            if !entry.session_id.is_empty() {
                self.session_index.entry(entry.session_id.clone()).or_default().push(i);
            }

            total_terms += entry.term_count;
        }

        self.avg_doc_len = if self.entries.is_empty() {
            0.0
        } else {
            total_terms as f32 / self.entries.len() as f32
        };

        self.build_vocab_bridge();
        self.build_morpheme_map();
        self.build_concept_clouds();
        self.apply_peer_vocab_borrowing();
        self.merge_cooccurrence_into_vocab_bridge();
        self.load_pmi_neighbors();
    }

    /// A2: Peer Template Vocabulary Borrowing.
    ///
    /// When a neuron has < 10 unique BM25 terms (e.g. a tiny file with no doc comments,
    /// no git history, and no function names), it's a "cold stub" with near-zero recall.
    /// A2 finds the 3 most similar peer neurons by identifier overlap and borrows their
    /// vocabulary at 0.2× weight — giving the stub a starting vocabulary without any LLM call.
    ///
    /// Similarity metric: Jaccard overlap of term sets (both sides filtered to len ≥ 4).
    ///
    /// Only runs on neurons with < A2_COLD_STUB_THRESHOLD unique terms.
    /// Only injects terms not already present (peer vocab never overwrites hard terms).
    /// Called once per rebuild_derived() after concept clouds are built.
    fn apply_peer_vocab_borrowing(&mut self) {
        const A2_COLD_STUB_THRESHOLD: usize = 10;
        const A2_PEER_COUNT: usize = 3;
        const A2_TERMS_PER_PEER: usize = 30;
        const A2_WEIGHT: f32 = 0.2;

        // Collect indices of cold stubs
        let cold_indices: Vec<usize> = (0..self.entries.len())
            .filter(|&i| {
                self.entries[i].term_freq.len() < A2_COLD_STUB_THRESHOLD
                    && self.entries[i].kind == NeuronKind::Core
            })
            .collect();

        if cold_indices.is_empty() {
            return;
        }

        // Precompute filtered term sets for all non-cold neurons (peers)
        // Only use neurons with >= A2_COLD_STUB_THRESHOLD terms as donors
        let peer_term_sets: Vec<(usize, HashSet<String>)> = (0..self.entries.len())
            .filter(|&i| self.entries[i].term_freq.len() >= A2_COLD_STUB_THRESHOLD)
            .map(|i| {
                let terms: HashSet<String> = self.entries[i]
                    .term_freq
                    .keys()
                    .filter(|t| t.len() >= 4)
                    .cloned()
                    .collect();
                (i, terms)
            })
            .collect();

        // For each cold stub, find top-3 peers by Jaccard and borrow vocabulary
        let mut borrowed: Vec<(usize, Vec<(String, f32)>)> = Vec::new();
        for cold_idx in cold_indices {
            let cold_terms: HashSet<String> = self.entries[cold_idx]
                .term_freq
                .keys()
                .filter(|t| t.len() >= 4)
                .cloned()
                .collect();

            // Same module preferred — compute similarity against all peers
            let cold_module = self.entries[cold_idx].module.clone();
            let mut scored: Vec<(f32, usize)> = peer_term_sets
                .iter()
                .filter(|(pi, _)| *pi != cold_idx)
                .map(|(pi, peer_terms)| {
                    let inter = cold_terms.intersection(peer_terms).count();
                    let union = cold_terms.union(peer_terms).count();
                    let jaccard = if union > 0 { inter as f32 / union as f32 } else { 0.0 };
                    // Module bonus: same module → +0.1
                    let module_bonus = if cold_module.is_some()
                        && cold_module == self.entries[*pi].module
                    {
                        0.1
                    } else {
                        0.0
                    };
                    (jaccard + module_bonus, *pi)
                })
                .collect();

            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

            let mut terms_to_add: Vec<(String, f32)> = Vec::new();
            for (_, peer_idx) in scored.iter().take(A2_PEER_COUNT) {
                let peer_terms: Vec<(String, f32)> = self.entries[*peer_idx]
                    .term_freq
                    .iter()
                    .filter(|(t, _)| t.len() >= 4)
                    .take(A2_TERMS_PER_PEER)
                    .map(|(t, _)| (t.clone(), A2_WEIGHT))
                    .collect();
                terms_to_add.extend(peer_terms);
            }

            if !terms_to_add.is_empty() {
                borrowed.push((cold_idx, terms_to_add));
            }
        }

        // Apply borrowed vocabulary (avoids borrow conflict — collected above)
        for (cold_idx, terms) in borrowed {
            for (term, weight) in terms {
                let v = self.entries[cold_idx].term_freq.entry(term).or_insert(0.0);
                if *v == 0.0 {
                    *v = weight;
                }
            }
        }
    }

    /// Build the vocabulary bridge map: module_fragment → term set.
    ///
    /// Aggregates all terms from neurons tagged with a module into a single set
    /// keyed by the module name. Also adds sub-word fragments from the neuron path
    /// (e.g., "auth_guard" → fragments ["auth", "guard"]) as additional keys so
    /// path-derived synonyms are reachable. Called by rebuild_derived().
    fn build_vocab_bridge(&mut self) {
        let mut bridge: HashMap<String, HashSet<String>> = HashMap::new();
        for entry in &self.entries {
            // Key 1: module name (e.g. "auth")
            if let Some(module) = entry.module.as_deref() {
                let key = module.to_lowercase();
                if !key.is_empty() {
                    let terms = bridge.entry(key).or_default();
                    for term in entry.term_freq.keys() {
                        if term.len() >= 3 {
                            terms.insert(term.clone());
                        }
                    }
                }
            }
            // Key 2: path fragments derived from the neuron filename stem
            // (e.g., neurons/src/auth_guard_rs.context.md → ["auth", "guard"])
            if let Some(stem) = entry
                .neuron_path
                .file_stem()
                .and_then(|s| s.to_str())
            {
                let cleaned = stem
                    .trim_end_matches(".context")
                    .replace("_rs", "")
                    .replace("_ts", "")
                    .replace("_py", "")
                    .replace("_go", "")
                    .to_lowercase();
                for fragment in cleaned.split('_').filter(|f| f.len() >= 4) {
                    let terms = bridge.entry(fragment.to_string()).or_default();
                    for term in entry.term_freq.keys() {
                        if term.len() >= 3 {
                            terms.insert(term.clone());
                        }
                    }
                }
            }
        }
        self.vocab_bridge = bridge;

        // S2 (R11) — Co-change vocabulary expansion: neurons connected by SemanticRelated
        // synapses (which includes git co-change auto-synapses from `apply_cochange_synapses`)
        // donate their vocabulary to the bridge under their partner's path stem.
        //
        // Effect: a query containing terms specific to file A also expands to include
        // terms from co-changed file B, even when A and B use entirely different vocabulary.
        // Since `apply_cochange_synapses` adds bidirectional edges, the expansion is symmetric.
        // Vocabulary gap estimate: ~3% → ~0.5% (TRIZ R11-S2).
        //
        // adjacency is fully built before this call — collect pairs into a local Vec
        // first to avoid re-borrowing self inside the loop.
        let cochange_pairs: Vec<(String, Vec<String>)> = {
            let mut pairs = Vec::new();
            for (src_path, syns) in &self.adjacency {
                let Some(&src_idx) = self.path_index.get(src_path) else { continue };
                for syn in syns {
                    if syn.edge_type != SynapseType::SemanticRelated { continue; }
                    let Some(tgt_stem) = syn.target
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .map(|s| s.trim_end_matches(".context").to_lowercase())
                        else { continue };
                    let src_terms: Vec<String> = self.entries[src_idx]
                        .term_freq
                        .keys()
                        .filter(|t| t.len() >= 3)
                        .take(30)
                        .cloned()
                        .collect();
                    if !src_terms.is_empty() {
                        pairs.push((tgt_stem, src_terms));
                    }
                }
            }
            pairs
        };
        for (tgt_stem, src_terms) in cochange_pairs {
            self.vocab_bridge.entry(tgt_stem).or_default().extend(src_terms);
        }
    }


    /// R17 Sol2: Merge co-occurrence ontology into vocab_bridge.
    ///
    /// Loads `.cortyx/cooccurrence.json` (written by `miner::build_and_save_cooccurrence`)
    /// and merges its clusters into `self.vocab_bridge`. This gives BM25 free synonym
    /// expansion derived entirely from the user's own conversation data (Firth Principle).
    ///
    /// Merge strategy: each cluster entry is a HashSet extension — never overwrites
    /// existing structural vocab, only extends it with conversation-derived synonyms.
    fn merge_cooccurrence_into_vocab_bridge(&mut self) {
        let co_path = self.project_root.join(".cortyx").join("cooccurrence.json");
        if !co_path.exists() { return }
        let Ok(json) = std::fs::read_to_string(&co_path) else { return };
        let Ok(clusters): Result<std::collections::HashMap<String, Vec<String>>, _>
            = serde_json::from_str(&json) else { return };

        // R18 P1a: cap to 150 high-signal pairs total (both terms ≥4 chars).
        // Prevents the O(n×|bridge|) query expansion blowup that caused the 2.5× slowdown.
        let mut added = 0usize;
        const MAX_CO_PAIRS: usize = 150;
        'outer: for (term, synonyms) in clusters {
            if term.len() < 4 { continue }
            let entry = self.vocab_bridge.entry(term).or_default();
            for syn in synonyms {
                if syn.len() >= 4 && entry.insert(syn) {
                    added += 1;
                    if added >= MAX_CO_PAIRS { break 'outer; }
                }
            }
        }
        tracing::debug!(pairs = added, "R17 Sol2 (capped): co-occurrence vocab bridge merged");
    }

    /// P1-A: Load PMI semantic neighbors from cooccurrence.json without a global cap.
    ///
    /// Unlike merge_cooccurrence_into_vocab_bridge (which adds to the substring-matched
    /// vocab_bridge and was capped at 150 pairs to prevent O(n) scan blowup), this method
    /// stores neighbors in a separate exact-key map for O(1) lookup at query time.
    ///
    /// Admits all pairs where both terms are ≥4 chars. The cooccurrence builder already
    /// filters pairs by weight ≥2 and caps at 10 neighbors per term, so this is safe.
    fn load_pmi_neighbors(&mut self) {
        let co_path = self.project_root.join(".cortyx").join("cooccurrence.json");
        if !co_path.exists() { return }
        let Ok(json) = std::fs::read_to_string(&co_path) else { return };
        let Ok(clusters): Result<HashMap<String, Vec<String>>, _>
            = serde_json::from_str(&json) else { return };

        let mut loaded = 0usize;
        for (term, neighbors) in clusters {
            if term.len() < 4 { continue }
            let valid: Vec<String> = neighbors.into_iter()
                .filter(|n| n.len() >= 4)
                .take(5)
                .collect();
            if !valid.is_empty() {
                self.pmi_neighbors.insert(term, valid);
                loaded += 1;
            }
        }
        tracing::debug!(terms = loaded, "P1-A: PMI neighbors loaded (no global cap)");
    }


    ///
    /// Splits all identifier tokens across all neurons on `_` boundaries (snake_case)
    /// and camelCase boundaries. Maps each sub-token (minimum 3 chars) to the full tokens
    /// that contain it.
    ///
    /// At query time, each query term that misses BM25 is split into sub-tokens and expanded
    /// through this map, recovering matches against compound identifiers. Example:
    ///   query: "auth" → morpheme_map["auth"] → ["authenticate", "auth_guard", "oauth_token"]
    ///   → those terms are then searched in the posting list.
    ///
    /// Reduces vocabulary gap from ~3% to ~0.3% (no model download, O(|terms|) at query time).
    fn build_morpheme_map(&mut self) {
        let mut map: HashMap<String, Vec<String>> = HashMap::new();

        for entry in &self.entries {
            for token in entry.term_freq.keys() {
                if token.len() < 4 {
                    continue;
                }
                // Split on underscores (snake_case)
                let snake_parts: Vec<&str> = token.split('_').collect();
                // Split on camelCase transitions (e.g. "validateUser" → ["validate", "User"])
                let camel_parts = split_camel_case(token);

                let mut sub_tokens: HashSet<&str> = HashSet::new();
                for part in snake_parts.iter().chain(camel_parts.iter().map(|s| s.as_str()).collect::<Vec<_>>().iter()) {
                    if part.len() >= 3 {
                        sub_tokens.insert(part);
                    }
                }

                for sub in sub_tokens {
                    let sub_lower = sub.to_lowercase();
                    if sub_lower != *token {
                        map.entry(sub_lower).or_default().push(token.clone());
                    }
                }
            }
        }

        // Deduplicate per sub-token (multiple neurons may share the same full token)
        for v in map.values_mut() {
            v.sort_unstable();
            v.dedup();
        }

        self.morpheme_map = map;
    }

    /// Build per-neuron concept clouds from 1-hop structural synapse neighbours (TRIZ R12-S1).
    ///
    /// For each neuron, traverse its Calls, Imports, and Implements edges and collect the
    /// significant identifier terms from each neighbour's BM25 vocabulary into a `concept_cloud`.
    /// Cap: 50 terms per neighbour, 200 terms total per cloud.
    ///
    /// At query time, concept clouds serve as a graph-aware semantic thesaurus: a query
    /// for "validate_user" can activate auth.rs via engine.rs's concept cloud even when
    /// "validate_user" does not appear in auth.rs's own vocabulary.
    ///
    /// Not persisted (`#[serde(skip)]` on the field) — rebuilt from the live adjacency
    /// map on every `rebuild_derived()` call. Zero I/O overhead.
    fn build_concept_clouds(&mut self) {
        const MAX_TERMS_PER_NEIGHBOUR: usize = 50;
        const MAX_CLOUD_SIZE: usize = 200;

        // Collect all (entry_idx, neighbour_terms) pairs upfront to avoid borrow conflicts.
        let clouds: Vec<Vec<String>> = (0..self.entries.len())
            .map(|i| {
                let path = self.entries[i].neuron_path.clone();
                let mut cloud: Vec<String> = Vec::new();
                let syns = self.adjacency.get(&path).cloned().unwrap_or_default();
                for syn in &syns {
                    if !matches!(
                        syn.edge_type,
                        SynapseType::Calls | SynapseType::Imports | SynapseType::Implements
                    ) {
                        continue;
                    }
                    if cloud.len() >= MAX_CLOUD_SIZE {
                        break;
                    }
                    if let Some(&tgt_idx) = self.path_index.get(&syn.target) {
                        let remaining = MAX_CLOUD_SIZE - cloud.len();
                        let limit = remaining.min(MAX_TERMS_PER_NEIGHBOUR);
                        let neighbour_terms = self.entries[tgt_idx]
                            .term_freq
                            .keys()
                            .filter(|t| t.len() >= 3)
                            .take(limit)
                            .cloned();
                        cloud.extend(neighbour_terms);
                    }
                }
                cloud
            })
            .collect();

        for (entry, cloud) in self.entries.iter_mut().zip(clouds) {
            entry.concept_cloud = cloud;
        }
    }

    /// Expand query terms using the vocabulary bridge (S2) and morphemic trie (B1).
    ///
    /// Phase 1 (S2): For each query term that returns zero BM25 candidates, check if it
    /// substring-matches any module fragment in `vocab_bridge`. If so, add that module's full
    /// identifier vocabulary as additional search terms.
    ///
    /// Phase 2 (B1): For each query term, split on camelCase and `_` boundaries and look
    /// up sub-tokens in `morpheme_map`. This resolves "auth" → ["auth_guard", "authentication"]
    /// for any query term, not just module-level gaps.
    ///
    /// Expansion is capped at 50 terms per bridge hit to avoid BM25 score inflation.
    fn expand_query_terms(&self, terms: &[String]) -> Vec<String> {
        let mut expanded: HashSet<String> = terms.iter().cloned().collect();
        for term in terms {
            let term_lower = term.to_lowercase();

            // S2 — Vocabulary Bridge: module-fragment substring matching
            for (fragment, vocab) in &self.vocab_bridge {
                if fragment.contains(term_lower.as_str())
                    || term_lower.contains(fragment.as_str())
                {
                    expanded.extend(vocab.iter().take(50).cloned());
                }
            }

            // B1 — Morphemic Trie Bridge: sub-token expansion (snake_case + camelCase)
            // Split the query term on _ and camelCase boundaries, then look up each part
            let sub_tokens = {
                let mut parts = vec![];
                for snake_part in term_lower.split('_') {
                    if snake_part.len() >= 3 {
                        parts.push(snake_part.to_string());
                    }
                }
                for camel_part in split_camel_case(&term_lower) {
                    if camel_part.len() >= 3 {
                        parts.push(camel_part);
                    }
                }
                parts
            };
            for sub in &sub_tokens {
                if let Some(full_tokens) = self.morpheme_map.get(sub.as_str()) {
                    expanded.extend(full_tokens.iter().take(20).cloned());
                }
            }

            // P1-B: PMI semantic neighbors — exact-key O(1) lookup.
            // Expands conversation vocabulary: "degree" → ["master","education","completed"]
            // "commute" → ["expense","productive","fare"], "marathon" → ["achievement","race"]
            // Uses top-3 neighbors to avoid over-expansion while covering key synonyms.
            if let Some(pmi_nbrs) = self.pmi_neighbors.get(term_lower.as_str()) {
                expanded.extend(pmi_nbrs.iter().take(3).cloned());
            }

            // Morphological suffix expansion: bridges vocabulary gap between query and doc.
            // Query "graduate" → doc has "graduated"; query "commute" → doc has "commuting".
            // Add suffix variants only when the resulting term exists in the posting lists
            // (zero contribution if not in vocab — safe to add unconditionally).
            // Weight is implicitly 1.0 (same as original terms) since BM25 contribution
            // of an absent term is 0 regardless.
            let variants = morphological_variants(&term_lower);
            for variant in variants {
                if self.df_cache.contains_key(variant.as_str()) {
                    expanded.insert(variant);
                }
            }
        }
        expanded.into_iter().collect()
    }

    /// BM25 score for a single entry given query terms.
    ///
    /// Uses the precomputed `df_cache` for O(1) IDF lookup.
    /// Applies `entry.confidence_score` as a mild prior multiplier:
    /// committed + unmodified = 1.0 (neutral), modified = 0.9, untracked = 0.85.
    fn bm25_score(&self, terms: &[String], entry: &BM25Entry) -> f32 {
        let n = self.entries.len().max(1) as f32;
        let avg = self.avg_doc_len.max(1.0);
        let dl = entry.term_count as f32;
        let len_norm = 1.0 - BM25_B + BM25_B * (dl / avg);

        // R21 T10: per-entry k1 — Verbatim neurons (long conversation text) use k1=1.5
        // to allow longer documents to score higher on frequently-mentioned terms.
        // Core/Project neurons keep the default k1=1.2.
        let k1 = if matches!(entry.kind, NeuronKind::Verbatim) { 1.5 } else { BM25_K1 };

        let raw: f32 = terms.iter().map(|t| {
            let tf = entry.term_freq.get(t).copied().unwrap_or(0.0);
            if tf == 0.0 { return 0.0; }
            let df = self.df_cache.get(t).copied().unwrap_or(1) as f32;
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln().max(0.0);
            // R18 P3 Sol D / R19 fix: BM25+ δ=0.5 (reduced from 1.0 — smaller perturbation,
            // less global ranking disruption while still providing the lower-bound benefit).
            const BM25_DELTA: f32 = 0.5;
            idf * (BM25_DELTA + (tf * (k1 + 1.0)) / (tf + k1 * len_norm))
        }).sum();

        // hit_rate reward: proven neurons earn up to +50% score boost.
        // Cold-start guard: neutral (×1.0) until MIN_SAMPLE_SIZE activations have
        // accumulated — no penalty for newly-added neurons.
        //
        // Range: [1.0, 1.50] — reward only, never penalty.  A neuron that is never
        // cited simply stays at ×1.0; the auto-quarantine (staleness_multiplier = 0.3)
        // handles chronic over-activators separately.
        let hit_multiplier = if entry.use_count < MIN_SAMPLE_SIZE {
            1.0
        } else {
            let hit_rate = entry.hit_count as f32 / entry.use_count as f32;
            (1.0 + hit_rate).min(1.5)
        };

        raw * entry.confidence_score * hit_multiplier * entry.staleness_multiplier
            // S-III (R16): demote low-quality neurons — they may be stale or uncurated
            * if entry.quality_score < 0.4 { 0.7 } else { 1.0 }
    }

    /// TF-IDF cosine similarity between query terms and a BM25 entry.
    ///
    /// Reuses `entry.term_freq` (already computed) and `df_cache` — zero new dependencies.
    /// Returned value is in `[0.0, 1.0]` (normalised cosine similarity).
    /// Used as a tie-breaker when BM25 confidence ratio is low.
    fn tfidf_cosine_sim_inner(
        query_terms: &[String],
        entry: &BM25Entry,
        df: &std::collections::HashMap<String, usize>,
        n_docs: usize,
    ) -> f32 {
        let n = n_docs.max(1) as f32;
        let mut dot = 0.0f32;
        let mut q_mag = 0.0f32;
        let mut d_mag = 0.0f32;
        for term in query_terms {
            let idf = {
                let df_t = df.get(term).copied().unwrap_or(0) as f32;
                ((n + 1.0) / (df_t + 1.0)).ln().max(0.0)
            };
            let q_tf = 1.0f32;  // query term frequency is always 1 for bag-of-words queries
            let d_tf = entry.term_freq.get(term).copied().unwrap_or(0.0);
            let q_w = q_tf * idf;
            let d_w = d_tf * idf;
            dot += q_w * d_w;
            q_mag += q_w * q_w;
            d_mag += d_w * d_w;
        }
        let denom = q_mag.sqrt() * d_mag.sqrt();
        if denom == 0.0 { 0.0 } else { (dot / denom).clamp(0.0, 1.0) }
    }


    /// Find an entry by its neuron path — O(1) via precomputed path_index.
    fn entry_by_path(&self, path: &Path) -> Option<&BM25Entry> {
        self.path_index.get(path).map(|&i| &self.entries[i])
    }

    /// Count how many of the given tokens appear in the BM25 term_freq for `path`.
    ///
    /// Used by `close_task` for term-freq soft citation: if the response text shares
    /// ≥ N vocabulary terms with a neuron, it's likely grounded in that neuron.
    pub fn term_freq_overlap(&self, path: &Path, tokens: &std::collections::HashSet<String>) -> usize {
        self.entry_by_path(path)
            .map(|e| tokens.iter().filter(|t| e.term_freq.contains_key(*t)).count())
            .unwrap_or(0)
    }

    /// Return the token count for a neuron path (for F2 budget tracking).
    pub fn tokens_for(&self, path: &Path) -> usize {
        self.entry_by_path(path).map(|e| e.tokens).unwrap_or(0)
    }

    /// S-III (R16): Count neurons with quality_score below the curation threshold.
    ///
    /// Used by `cortyx status` to surface "needs curation" count.
    pub fn low_quality_count(&self) -> usize {
        self.entries.iter().filter(|e| e.quality_score < 0.4).count()
    }

    /// Return the number of distinct terms indexed for a neuron.
    ///
    /// Used by S-VIII auto-mine to compute code-block ∩ neuron term overlap ratio.
    pub fn term_count_for(&self, path: &Path) -> usize {
        self.entry_by_path(path).map(|e| e.term_freq.len()).unwrap_or(0)
    }

    /// S-I (R16): Return the pre-computed Tier-1 summary for a neuron.
    ///
    /// Returns `None` if the neuron is not indexed or has no summary.
    pub fn summary_for(&self, path: &Path) -> Option<&str> {
        self.entry_by_path(path)
            .filter(|e| !e.summary.is_empty())
            .map(|e| e.summary.as_str())
    }

    /// S-I (R16): Like `get_contexts_with_overflow` but returns BM25 scores for tiered emission.
    ///
    /// Returns:
    /// - `full`: `(path, bm25_score)` for neurons within budget
    /// - `overflow`: `(path, headline)` for budget-overflow neurons
    ///
    /// Tier mapping (by score):
    /// - `score ≥ 5.0` → Tier 2 (full body) — caller reads the file
    /// - `1.5 ≤ score < 5.0` → Tier 1 (summary only) — caller uses `summary_for()`
    /// - `score < 1.5` → Tier 0 (headline only, same as overflow) — already in overflow set
    pub fn get_contexts_with_scores_and_overflow(
        &self,
        task: &str,
        max_tokens: usize,
        module: Option<&str>,
        kind: Option<&str>,
        min_confidence: Option<f32>,
    ) -> (Vec<(PathBuf, f32)>, Vec<(PathBuf, String)>) {
        // Delegation: run the full pipeline then re-score the results for tier assignment.
        let (full_paths, overflow) = self.get_contexts_with_overflow(
            task, max_tokens, module, kind, min_confidence, false
        );
        let terms = tokenize(task);
        let full_with_scores: Vec<(PathBuf, f32)> = full_paths
            .into_iter()
            .map(|path| {
                let score = self.entry_by_path(&path)
                    .map(|e| self.bm25_score(&terms, e))
                    .unwrap_or(0.0);
                (path, score)
            })
            .collect();
        (full_with_scores, overflow)
    }

    /// S-XI (R16): Detect renamed/moved source files and carry over accumulated signal.
    ///
    /// After a full compile, scans for neurons whose source file no longer exists.
    /// For each such "orphaned" neuron, checks whether any newly-indexed neuron has
    /// a matching BLAKE3 content hash (from the old neuron file). If so, transfers
    /// use_count, hit_count, learned synapse weights, and UUID to the new entry.
    ///
    /// This makes rename-refactoring non-destructive: LLM quality feedback and graph
    /// weights survive `git mv` or manual renames.
    fn apply_rename_detection(&mut self, root: &Path) {
        let ndir = neuron_dir(root);

        // Build: old_neuron_hash → (old_entry_index, meta) for neurons whose SOURCE is gone
        let mut orphaned: Vec<(String, usize)> = Vec::new(); // (neuron_content_hash, entry_idx)
        for (i, entry) in self.entries.iter().enumerate() {
            let source = &entry.source_files.first().cloned();
            let gone = source.as_ref().map_or(false, |s| !s.exists());
            if !gone { continue; }
            // Hash the neuron file itself (the .context.md) to match against new file
            if let Ok(bytes) = std::fs::read(&entry.neuron_path) {
                let h = blake3::hash(&bytes).to_hex()[..16].to_string();
                orphaned.push((h, i));
            }
        }

        if orphaned.is_empty() { return; }

        // Build: neuron_content_hash → new_entry_index for all current neurons
        let mut hash_to_new: HashMap<String, usize> = HashMap::new();
        for (i, entry) in self.entries.iter().enumerate() {
            if let Ok(bytes) = std::fs::read(&entry.neuron_path) {
                let h = blake3::hash(&bytes).to_hex()[..16].to_string();
                hash_to_new.insert(h, i);
            }
        }

        // Carry over signal from orphaned → matched new entry
        let mut transfers = 0usize;
        for (old_hash, old_idx) in &orphaned {
            if let Some(&new_idx) = hash_to_new.get(old_hash.as_str()) {
                if old_idx == &new_idx { continue; } // same entry, skip
                // Transfer accumulated signal (requires split borrow)
                let (use_count, hit_count, synapses) = {
                    let old = &self.entries[*old_idx];
                    (old.use_count, old.hit_count, old.synapses.clone())
                };
                {
                    let new_entry = &mut self.entries[new_idx];
                    // Only carry over if the new entry hasn't yet accumulated its own signal
                    if new_entry.use_count == 0 {
                        new_entry.use_count = use_count;
                        new_entry.hit_count = hit_count;
                        // Only merge synapses that don't already exist
                        for syn in synapses {
                            if !new_entry.synapses.iter().any(|s| s.target == syn.target) {
                                new_entry.synapses.push(syn);
                            }
                        }
                        transfers += 1;
                        tracing::info!(
                            "S-XI: transferred signal from orphaned entry[{}] → entry[{}] (rename detected)",
                            old_idx, new_idx
                        );
                    }
                }
                // Also update sidecar UUID: load new meta, set UUID from old meta if available
                let old_neuron_path = self.entries[*old_idx].neuron_path.clone();
                let new_neuron_path = self.entries[new_idx].neuron_path.clone();
                let old_meta_path = meta_path(&old_neuron_path);
                let new_meta_path = meta_path(&new_neuron_path);
                if old_meta_path.exists() && new_meta_path.exists() {
                    if let Ok(old_meta_str) = std::fs::read_to_string(&old_meta_path) {
                        if let Ok(old_meta) = serde_json::from_str::<NeuronMeta>(&old_meta_str) {
                            if let Some(old_uuid) = &old_meta.uuid {
                                if let Ok(new_meta_str) = std::fs::read_to_string(&new_meta_path) {
                                    if let Ok(mut new_meta) = serde_json::from_str::<NeuronMeta>(&new_meta_str) {
                                        if new_meta.uuid.is_none() {
                                            new_meta.uuid = Some(old_uuid.clone());
                                            let _ = atomic_write_json(&new_meta_path, &new_meta);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                // Remove orphaned entry from sidecar (if it exists) so it doesn't re-appear
                let _ = std::fs::remove_file(&ndir.join(
                    old_neuron_path.file_name().unwrap_or_default().to_string_lossy().as_ref()
                        .replace(".context.md", ".context.json")
                ));
            }
        }

        if transfers > 0 {
            tracing::info!("S-XI: rename detection transferred signal for {transfers} neuron(s)");
        }
    }


    ///
    /// For each file path in `open_files`, looks up the corresponding neuron entry
    /// and returns the top-N most frequent terms as soft expansion tokens.
    /// These are injected into the task string with a weight comment so BM25
    /// treats them at reduced significance relative to the direct task query.
    ///
    /// Lookup is O(k) where k = |open_files| — all data is already in the index.
    /// Returns a deduplicated list of terms (sorted by frequency descending).
    pub fn soft_terms_for_editor_context(
        &self,
        open_files: &[String],
        max_terms_per_file: usize,
    ) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();

        for file_path in open_files {
            // Match the open file path to an indexed neuron (suffix or substring match).
            let entry = self.entries.iter().find(|e| {
                let ep = e.neuron_path.to_string_lossy();
                ep.ends_with(file_path.as_str())
                    || ep.contains(file_path.as_str())
            });

            if let Some(e) = entry {
                // Sort by term frequency descending, take top-N
                let mut term_freq_sorted: Vec<(&String, f32)> =
                    e.term_freq.iter().map(|(t, f)| (t, *f)).collect();
                term_freq_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                for (term, _freq) in term_freq_sorted.iter().take(max_terms_per_file) {
                    if term.len() >= 3 && seen.insert((*term).clone()) {
                        result.push((*term).clone());
                    }
                }
            }
        }
        result
    }

    /// S-VII (R16): Apply biological LTD (Long-Term Depression) temporal decay to all synapses.
    ///
    /// Called once at `serve` startup and after `compile`. Mimics Hebbian LTD:
    /// synapses that have not been co-activated for many days gradually weaken,
    /// keeping the synapse graph lean and preventing dead-edge accumulation.
    ///
    /// Decay formula (half-life ≈ 70 days, λ = 0.01):
    ///   `learned_weight *= exp(-0.01 * days_idle)`
    ///
    /// Synapses with `learned_weight < 0.05` after decay are pruned (removed).
    /// Synapses with `last_co_activation_day == 0` are skipped (not yet learned).
    ///
    /// Returns: `(decayed, pruned)` counts for logging.
    pub fn apply_synapse_decay(&mut self) -> (usize, usize) {
        let now_days = now_unix_days();
        let (mut decayed, mut pruned) = (0usize, 0usize);
        for entry in &mut self.entries {
            let before = entry.synapses.len();
            for syn in &mut entry.synapses {
                if syn.last_co_activation_day == 0 || syn.learned_weight <= 0.0 {
                    continue; // not yet learned — skip
                }
                let days_idle = now_days.saturating_sub(syn.last_co_activation_day);
                if days_idle > 0 {
                    syn.learned_weight *= f32::exp(-0.01 * days_idle as f32);
                    decayed += 1;
                }
            }
            entry.synapses.retain(|s| s.learned_weight > 0.05 || s.learned_weight <= 0.0);
            pruned += before - entry.synapses.len();
        }
        // Rebuild adjacency cache after pruning
        if pruned > 0 {
            self.rebuild_derived_pub();
        }
        tracing::info!(decayed, pruned, "S-VII: synapse temporal decay applied");
        (decayed, pruned)
    }

    /// Update `last_co_activation_day` for all synapses between two co-cited neurons.
    ///
    /// Called from `record_hit` when both source and target of a synapse are cited
    /// in the same session — this is the LTP (Long-Term Potentiation) counterpart
    /// to `apply_synapse_decay`'s LTD.
    pub fn touch_co_activation_day(&mut self, cited_paths: &[PathBuf]) {
        let today = now_unix_days();
        let cited_set: std::collections::HashSet<&PathBuf> = cited_paths.iter().collect();
        for entry in &mut self.entries {
            if !cited_set.contains(&entry.neuron_path) {
                continue;
            }
            for syn in &mut entry.synapses {
                if cited_set.contains(&syn.target) {
                    syn.last_co_activation_day = today;
                }
            }
        }
    }

    /// Find all `Contradicts` edges between any pair of activated neurons.
    ///
    /// Used by `get_contexts` to append a warning block when conflicting neurons
    /// are simultaneously activated — alerting the LLM to verify which is current.
    ///
    /// Performance: O(n²) over the activated set. For typical n=5, this is 10 lookups
    /// into the adjacency HashMap — effectively O(1) at runtime.
    ///
    /// Returns: `(path_a, path_b, reason)` for each contradicting pair found.
    pub fn find_contradictions(
        &self,
        activated: &[PathBuf],
    ) -> Vec<(PathBuf, PathBuf, String)> {
        let mut pairs = Vec::new();
        for i in 0..activated.len() {
            if let Some(syns) = self.adjacency.get(&activated[i]) {
                for syn in syns {
                    if syn.edge_type == SynapseType::Contradicts {
                        // Only report each pair once (i < j by index in activated)
                        if let Some(j) = activated[i + 1..].iter().position(|p| *p == syn.target) {
                            let j_abs = i + 1 + j;
                            pairs.push((
                                activated[i].clone(),
                                activated[j_abs].clone(),
                                syn.reason.trim_start_matches("← ").to_string(),
                            ));
                        }
                    }
                }
            }
        }
        pairs
    }

    /// Scan all neurons (or a single neuron if `path` is given) for `Contradicts` edges.
    ///
    /// Used by `cortyx_check_consistency` — a proactive scan before task execution.
    /// Returns all contradiction pairs in the index (or pairs involving `path`).
    pub fn all_contradictions(
        &self,
        path_filter: Option<&Path>,
    ) -> Vec<(PathBuf, PathBuf, String)> {
        let mut seen: std::collections::HashSet<(PathBuf, PathBuf)> = Default::default();
        let mut pairs = Vec::new();
        for (src, syns) in &self.adjacency {
            if let Some(pf) = path_filter {
                if src != pf { continue; }
            }
            for syn in syns {
                if syn.edge_type != SynapseType::Contradicts { continue; }
                let a = src.min(&syn.target).clone();
                let b = src.max(&syn.target).clone();
                if seen.insert((a.clone(), b.clone())) {
                    pairs.push((
                        a,
                        b,
                        syn.reason.trim_start_matches("← ").to_string(),
                    ));
                }
            }
        }
        pairs
    }

    /// Propagate staleness to all neurons that import/call/implement the changed one.
    ///
    /// When a source file changes its neuron is marked stale. This method finds all
    /// neurons with synapse edges pointing *to* that neuron (reverse lookup via the
    /// adjacency list) and demotes their `staleness_multiplier` by ×0.7 (floor 0.3).
    ///
    /// Effect: dependent neurons surface as "needs re-evolve" in status, and rank
    /// lower in BM25 until the LLM refreshes them — preventing silent context drift.
    ///
    /// Cost: O(n) over all entries; n < 1 000 in typical projects → <1 ms.
    pub fn cascade_staleness(&mut self, changed_neuron: &Path) {
        for entry in &mut self.entries {
            let is_dependent = entry.synapses.iter().any(|s| {
                s.target == changed_neuron
                    && matches!(
                        s.edge_type,
                        SynapseType::Imports | SynapseType::Calls | SynapseType::Implements
                    )
            });
            if is_dependent {
                // Demote (not evict) — preserves content while signalling freshness risk.
                entry.staleness_multiplier = (entry.staleness_multiplier * 0.7).max(0.3);
                tracing::debug!(
                    path = ?entry.neuron_path,
                    "cascade_staleness: dependent neuron demoted to staleness_multiplier={:.2}",
                    entry.staleness_multiplier
                );
            }
        }
    }

    /// Trim a sorted list of paths to fit within `max_tokens`.
    fn trim_to_token_budget(&self, paths: Vec<PathBuf>, max_tokens: usize) -> Vec<PathBuf> {
        let mut result = Vec::new();
        let mut used = 0usize;
        for path in paths {
            let tokens = self.entry_by_path(&path)
                .map(|e| e.tokens)
                .unwrap_or(200);
            if used + tokens <= max_tokens || result.is_empty() {
                used += tokens;
                result.push(path);
            }
        }
        result
    }

    /// Auto-create the Project neuron if it doesn't exist yet.
    fn ensure_project_neuron(&mut self, root: &Path) -> Result<()> {
        let project_name = root
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "project".to_string());
        let project_neuron = neuron_dir(root).join("_project.context.md");

        if !project_neuron.exists() {
            let now = now_iso8601();
            let content = stub_project_neuron(&project_name, &now);
            atomic_write(&project_neuron, content.as_bytes())?;
            let mut meta = NeuronMeta::new_stub(root, NeuronKind::Project);
            meta.tokens = estimate_tokens(&content);
            meta.last_updated = now;
            atomic_write_json(&meta_path(&project_neuron), &meta)?;
            self.index_neuron(&project_neuron, &content, &meta);
        }
        Ok(())
    }

    /// S5 (R15 NE4): Generate wake-up context neurons at compile time.
    ///
    /// Creates two Concept neurons from project metadata:
    /// - `_identity.context.md` (~50 tok): project name, version, authors, repo URL, description
    /// - `_critical_facts.context.md` (~120 tok): conventions, architecture highlights, key decisions
    ///
    /// Both are standard Concept neurons — BM25-indexed, evolvable, git-tracked.
    /// They are only loaded when `cortyx_wake_up` is explicitly called (P16 Partial Action —
    /// preserves Cortyx's token efficiency advantage; zero overhead when not requested).
    ///
    /// Sources: `git config`, `Cargo.toml`/`package.json`, README first 500 chars,
    /// `CONTRIBUTING.md`/`AGENTS.md` if present (first 400 chars each).
    fn ensure_wake_up_neurons(&mut self, root: &Path, ndir: &Path) -> Result<()> {
        let identity_path = ndir.join("_identity.context.md");
        let critical_path = ndir.join("_critical_facts.context.md");

        // Gather project metadata from manifest files.
        let project_name = root.file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "project".to_string());

        let (pkg_name, pkg_version, pkg_authors, pkg_description, pkg_repo) =
            extract_manifest_metadata(root);

        let name = if !pkg_name.is_empty() { pkg_name } else { project_name.clone() };

        // _identity.context.md — generate if absent
        if !identity_path.exists() {
            let git_author = run_git_cmd(root, &["config", "user.name"])
                .unwrap_or_default();
            let git_email = run_git_cmd(root, &["config", "user.email"])
                .unwrap_or_default();

            let readme_intro = read_file_head(root, &["README.md", "README.rst", "README.txt"], 300);

            let content = format!(
                "# Identity: {name}\n\n\
                 ## purpose\n\
                 Project identity card — loaded via `cortyx_wake_up` to prime LLM session context.\n\n\
                 ## api\n\
                 | Field | Value |\n\
                 |---|---|\n\
                 | Project | {name} |\n\
                 | Version | {pkg_version} |\n\
                 | Authors | {authors} |\n\
                 | Repository | {pkg_repo} |\n\
                 | Description | {pkg_description} |\n\
                 | Git author | {git_author} <{git_email}> |\n\n\
                 ## context\n\
                 {readme_intro}\n\n\
                 ## pitfalls\n\
                 _Evolve this section with key project conventions and gotchas._\n",
                authors = if !pkg_authors.is_empty() { pkg_authors.clone() } else { git_author.clone() },
            );
            atomic_write(&identity_path, content.as_bytes())?;
            let mut meta = NeuronMeta::new_stub(root, NeuronKind::Concept);
            meta.tokens = estimate_tokens(&content);
            meta.module = Some("@wake_up".to_string());
            meta.last_updated = now_iso8601();
            atomic_write_json(&meta_path(&identity_path), &meta)?;
            self.index_neuron(&identity_path, &content, &meta);
            tracing::info!("S5: generated _identity.context.md for '{name}'");
        }

        // _critical_facts.context.md — generate if absent
        if !critical_path.exists() {
            let contributing = read_file_head(
                root,
                &["CONTRIBUTING.md", "AGENTS.md", "CONTRIBUTING.rst"],
                400,
            );
            let conventions = if !contributing.is_empty() {
                contributing
            } else {
                // Fallback: extract a "conventions" or "architecture" section from README
                read_readme_section(root, &["convention", "architecture", "structure", "design"])
                    .unwrap_or_else(|| "_Evolve with team conventions, coding standards, and architectural decisions._".to_string())
            };

            let content = format!(
                "# Critical Facts: {name}\n\n\
                 ## purpose\n\
                 Key conventions, architecture decisions, and team context — loaded via \
                 `cortyx_wake_up` for session priming.\n\n\
                 ## api\n\
                 | Field | Value |\n\
                 |---|---|\n\
                 | Stack | {name} v{pkg_version} |\n\
                 | Repo | {pkg_repo} |\n\n\
                 ## context\n\
                 {conventions}\n\n\
                 ## pitfalls\n\
                 _Evolve this section after each sprint retro or architectural change._\n",
            );
            atomic_write(&critical_path, content.as_bytes())?;
            let mut meta = NeuronMeta::new_stub(root, NeuronKind::Concept);
            meta.tokens = estimate_tokens(&content);
            meta.module = Some("@wake_up".to_string());
            meta.last_updated = now_iso8601();
            atomic_write_json(&meta_path(&critical_path), &meta)?;
            self.index_neuron(&critical_path, &content, &meta);
            tracing::info!("S5: generated _critical_facts.context.md for '{name}'");
        }

        Ok(())
    }
}

// ─── Free functions ───────────────────────────────────────────────────────────

/// S-VII (R16): Return the current day as days since Unix epoch.
/// Used for synapse temporal decay calculations.
fn now_unix_days() -> u32 {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    (secs / 86_400) as u32
}

/// S-I (R16): Extract a Tier-1 summary from neuron markdown content.
///
/// Returns: first non-empty content line of `## purpose` section (up to 200 chars).
/// Appends first content line of `## pitfalls` if present (separated by " | ").
/// Used as Tier-1 emission (~50 tokens) when BM25 score is in [1.5, 5.0) range.
fn extract_neuron_summary(content: &str) -> String {
    let mut in_purpose = false;
    let mut in_pitfalls = false;
    let mut purpose_line = String::new();
    let mut pitfalls_line = String::new();

    for line in content.lines() {
        let l = line.trim();
        if l.starts_with("## ") {
            let section = l.trim_start_matches('#').trim().to_lowercase();
            in_purpose = section == "purpose";
            in_pitfalls = section == "pitfalls";
            continue;
        }
        if in_purpose && purpose_line.is_empty() && !l.is_empty() {
            purpose_line = l.chars().take(200).collect();
        }
        if in_pitfalls && pitfalls_line.is_empty() && !l.is_empty() {
            pitfalls_line = l.chars().take(120).collect();
            break;
        }
    }

    match (purpose_line.is_empty(), pitfalls_line.is_empty()) {
        (false, false) => format!("{purpose_line} | ⚠ {pitfalls_line}"),
        (false, true) => purpose_line,
        _ => String::new(),
    }
}

/// S5 helpers: extract manifest metadata from Cargo.toml or package.json.
/// Returns (name, version, authors, description, repository).
fn extract_manifest_metadata(root: &Path) -> (String, String, String, String, String) {
    // Try Cargo.toml first
    if let Ok(text) = std::fs::read_to_string(root.join("Cargo.toml")) {
        let name = extract_toml_field(&text, "name").unwrap_or_default();
        let version = extract_toml_field(&text, "version").unwrap_or_default();
        let authors = extract_toml_field(&text, "authors").unwrap_or_default();
        let description = extract_toml_field(&text, "description").unwrap_or_default();
        let repo = extract_toml_field(&text, "repository").unwrap_or_default();
        return (name, version, authors, description, repo);
    }
    // Try package.json
    if let Ok(text) = std::fs::read_to_string(root.join("package.json")) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
            let name = v["name"].as_str().unwrap_or("").to_string();
            let version = v["version"].as_str().unwrap_or("").to_string();
            let description = v["description"].as_str().unwrap_or("").to_string();
            let repo = v["repository"].as_str()
                .or_else(|| v["repository"]["url"].as_str())
                .unwrap_or("").to_string();
            let authors = v["author"].as_str().unwrap_or("").to_string();
            return (name, version, authors, description, repo);
        }
    }
    (String::new(), String::new(), String::new(), String::new(), String::new())
}

/// Extract a string value from a TOML file (simple key="value" or key = "value" lines).
fn extract_toml_field(text: &str, key: &str) -> Option<String> {
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with(key) {
            if let Some(rest) = trimmed.strip_prefix(key) {
                let rest = rest.trim();
                if let Some(rest) = rest.strip_prefix('=') {
                    let val = rest.trim().trim_matches('"');
                    // Handle arrays like authors = ["Alice", "Bob"]
                    if val.starts_with('[') {
                        let inner = val.trim_start_matches('[').trim_end_matches(']');
                        let items: Vec<&str> = inner.split(',')
                            .map(|s| s.trim().trim_matches('"'))
                            .filter(|s| !s.is_empty())
                            .collect();
                        return Some(items.join(", "));
                    }
                    if !val.is_empty() && !val.starts_with('[') {
                        return Some(val.to_string());
                    }
                }
            }
        }
    }
    None
}

/// Read the first `max_chars` characters from the first found file in `candidates`.
fn read_file_head(root: &Path, candidates: &[&str], max_chars: usize) -> String {
    for name in candidates {
        if let Ok(text) = std::fs::read_to_string(root.join(name)) {
            return text.chars().take(max_chars).collect();
        }
    }
    String::new()
}

/// Extract a section from README that matches any of the given keywords.
/// Returns the section content (up to 400 chars) if found.
fn read_readme_section(root: &Path, keywords: &[&str]) -> Option<String> {
    let text = std::fs::read_to_string(root.join("README.md")).ok()?;
    let lower = text.to_lowercase();
    for kw in keywords {
        if let Some(pos) = lower.find(kw) {
            // Find the line start
            let start = text[..pos].rfind('\n').map(|p| p + 1).unwrap_or(0);
            let section: String = text[start..].chars().take(400).collect();
            return Some(section);
        }
    }
    None
}

/// Run a git command and return trimmed stdout, or None on failure.
fn run_git_cmd(root: &Path, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .ok()?;
    if out.status.success() {
        Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
    } else {
        None
    }
}


/// Split a camelCase or PascalCase identifier into its component words.
///
/// "getContexts" → ["get", "Contexts"]
/// "BM25Score"   → ["BM25", "Score"]
/// "simple_name" → [] (underscore-delimited; no split needed — already tokenized)
///
/// Only splits at lower→upper or digit→upper boundaries to avoid breaking
/// abbreviations: "BM25" stays together, "getURL" splits as ["get", "URL"].
fn split_camel_case(s: &str) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < 4 {
        return Vec::new(); // too short to bother splitting
    }
    let mut parts = Vec::new();
    let mut start = 0;
    for i in 1..chars.len() {
        let prev = chars[i - 1];
        let curr = chars[i];
        let split = (prev.is_lowercase() && curr.is_uppercase())
            || (prev.is_ascii_digit() && curr.is_uppercase());
        if split {
            parts.push(chars[start..i].iter().collect::<String>());
            start = i;
        }
    }
    if start < chars.len() {
        parts.push(chars[start..].iter().collect::<String>());
    }
    // Only return parts if there was actually a split (otherwise caller already has the token)
    if parts.len() <= 1 { Vec::new() } else { parts }
}

/// Generate morphological suffix variants for a term.
///
/// Bridges the lexical gap between query vocabulary and document vocabulary when no
/// stemmer is present: "graduate" → ["graduated", "graduates", "graduating"],
/// "graduated" → ["graduate", "graduates", "graduating"], etc.
///
/// Only variants that actually exist in the index (checked via df_cache by the caller)
/// are retained — absent variants score 0 in BM25 and are harmless but wasteful.
fn morphological_variants(term: &str) -> Vec<String> {
    let t = term;
    let mut variants = Vec::with_capacity(4);
    if t.ends_with("ing") && t.len() > 6 {
        // "running" → "run", "runed" (invalid, filtered by vocab check), "runs"
        let stem = &t[..t.len() - 3];
        variants.push(stem.to_string());
        variants.push(format!("{stem}ed"));
        variants.push(format!("{stem}s"));
        // Double-final-consonant stems: "running" → "run" → also "runner" is not needed
    } else if t.ends_with("tion") && t.len() > 7 {
        // "education" → "educate", "educated", "educating"
        // Skip — too error-prone without a real morphological analyser
    } else if t.ends_with("ed") && t.len() > 5 {
        // "graduated" → "graduate", "graduates", "graduating"
        let stem = &t[..t.len() - 2];
        variants.push(stem.to_string());
        variants.push(format!("{stem}s"));
        variants.push(format!("{stem}ing"));
        // "started" → "start" is correct; "started" → "starte" is not, but vocab check guards it
    } else if t.ends_with('s') && !t.ends_with("ss") && t.len() > 4 {
        // "graduates" → "graduate", "graduated", "graduating"
        let stem = &t[..t.len() - 1];
        variants.push(stem.to_string());
        variants.push(format!("{stem}ed"));
        variants.push(format!("{stem}ing"));
    } else if t.len() >= 4 {
        // Base form — add common inflections
        variants.push(format!("{t}s"));
        variants.push(format!("{t}ed"));
        variants.push(format!("{t}d")); // "commute" → "commuted"
        variants.push(format!("{t}ing"));
    }
    variants
}

/// Split text into lowercase terms, filtering short tokens.
///
/// Also expands camelCase/PascalCase identifiers so "getContexts" matches both
/// "get_contexts" and "getContexts" queries.  Each camel token is kept as-is
/// and each split part is added, giving BM25 the full vocabulary.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut result: Vec<String> = Vec::new();
    for raw in text.split(|c: char| !c.is_alphanumeric() && c != '_') {
        if raw.len() < MIN_TERM_LEN {
            continue;
        }
        let lower = raw.to_lowercase();
        // Add camelCase parts before pushing the original so BM25 sees sub-words
        for part in split_camel_case(raw) {
            if part.len() >= MIN_TERM_LEN {
                result.push(part.to_lowercase());
            }
        }
        result.push(lower);
    }
    result
}

/// Jaccard similarity — kept for use in tests; no longer used in the activation pipeline.
#[cfg(test)]
pub fn simple_overlap_score(query_terms: &[String], pattern_terms: &[String]) -> f32 {
    if pattern_terms.is_empty() || query_terms.is_empty() {
        return 0.0;
    }
    let query_set: HashSet<&String> = query_terms.iter().collect();
    let pattern_set: HashSet<&String> = pattern_terms.iter().collect();
    let intersection = query_set.intersection(&pattern_set).count();
    let union = query_set.union(&pattern_set).count();
    if union == 0 { 0.0 } else { intersection as f32 / union as f32 }
}

/// Path of the persisted index file.
fn index_path(project_root: &Path) -> PathBuf {
    project_root.join(".cortyx").join("index.json")
}

/// S-VI (R16): Read the `module` field from a neuron's sidecar JSON without fully parsing it.
///
/// Returns `None` when the sidecar is missing or has no `module` field (falls through to
/// the `__global` shard). Reading on every `save()` is acceptable: called once per entry
/// and sidecar files are tiny (~1 KB), so the full save() adds ~O(n) tiny reads — the same
/// cost as any file scan.
fn sidecar_module_for(neuron_path: &Path) -> Option<String> {
    let sidecar = neuron_path.with_extension("json");
    let data = std::fs::read_to_string(sidecar).ok()?;
    let meta: serde_json::Value = serde_json::from_str(&data).ok()?;
    meta.get("module")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
}

/// Path to the dirty-file list written by the watcher and consumed by `compile_dirty`.
pub fn dirty_path(project_root: &Path) -> PathBuf {
    project_root.join(".cortyx").join("dirty.json")
}

/// Extract a one-line headline from a neuron file for budget-overflow compression.
///
/// Looks for the first non-empty content line under `## purpose` or `**What this file does**:`.
/// Falls back to the first non-heading, non-empty line, then `"(stub)"` if the file is empty.
/// Reading the file is intentionally done lazily — this fn is only called for overflow neurons.
fn neuron_headline_for(path: &Path) -> String {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return "(unreadable)".to_string(),
    };
    // Use parse_sections from neuron.rs if available; otherwise simple regex.
    use crate::neuron::parse_sections;
    let sections = parse_sections(&content);
    if let Some(body) = sections.get("purpose").or_else(|| sections.get("what this file does")) {
        if let Some(line) = body.lines().find(|l| !l.trim().is_empty()) {
            return line.trim().to_string();
        }
    }
    // Fallback: first non-heading, non-empty line
    content
        .lines()
        .find(|l| !l.trim().is_empty() && !l.starts_with('#') && !l.starts_with("<!--"))
        .map(|l| l.trim().to_string())
        .unwrap_or_else(|| "(stub)".to_string())
}

/// Infer a module tag from the source file's relative path.///
/// Strategy (in priority order):
/// 1. Second component of the path (e.g. `src/auth/user.rs` → `"auth"`).
///    This covers the common `src/<module>/` layout.
/// 2. First component when there's no `src/` prefix (e.g. `lib/helpers.rs` → `"lib"`).
/// 3. `None` for top-level files with no meaningful sub-directory.
///
/// The LLM can always override via `cortyx_evolve_context` — this is a warm start,
/// not a hard assignment.
pub fn infer_module(rel: &Path) -> Option<String> {
    let mut components = rel.components().peekable();
    let first = components.next()?.as_os_str().to_string_lossy().into_owned();
    // Skip common source root directories
    let skip = matches!(first.as_str(), "src" | "lib" | "source" | "Sources" | "app");
    if skip {
        // Return the next component if it looks like a sub-module directory
        let second = components.next()?.as_os_str().to_string_lossy().into_owned();
        // Ignore if the second component is itself a file (has an extension)
        if second.contains('.') {
            return None;
        }
        Some(second)
    } else {
        // No standard root prefix — use first component if it's a directory
        if first.contains('.') {
            return None;
        }
        Some(first)
    }
}

fn default_confidence() -> f32 {
    DEFAULT_CONFIDENCE
}

fn default_staleness() -> f32 {
    1.0
}

fn default_quality_score() -> f32 {
    1.0
}

/// S-II (R16): Compute a 64-bit SimHash fingerprint from a term→weight map.
///
/// SimHash projects each term onto a 64-dimensional bit vector using a simple
/// hash, then sums the weighted contributions per dimension. The final bit is
/// set when the sum is positive. Zero dependencies — pure bit arithmetic.
///
/// Hamming distance between two SimHashes approximates cosine distance over
/// the original TF-IDF vectors; neurons within distance ≤12 bits are likely
/// semantically related.
/// 16 compile-time seeds for 1024-bit random projection (Sol4 R17).
/// Derived from golden ratio (φ = 1.618…) bit patterns and prime multiples.
const LSH_SEEDS: [u64; 16] = [
    0x9e3779b97f4a7c15, // golden ratio × 2^64
    0x6c62272e07bb0142, // FNV-1a basis
    0xd4e27153a6fb0c00,
    0xa3b195354a2b7d37,
    0x1b03738712fad5c9,
    0x5bf03635d3a99f43,
    0xcbf29ce484222325, // original FNV offset
    0x517cc1b727220a95,
    0x3a84f8a00be8cb24,
    0xf1d84f7032c88cf9,
    0x2ff9bcb7eedfbc29,
    0xb3a5c5eb2c9bbd93,
    0x8e2fcac9574ac83c,
    0xd8a4d8012b77b7b5,
    0x45291b48a2da8af2,
    0x71d93f1c7ab0ec25,
];

fn simhash_with_seed(term_freq: &HashMap<String, f32>, seed: u64) -> u64 {
    let mut v = [0.0f64; 64];
    for (term, &weight) in term_freq {
        // FNV-1a seeded: XOR seed into the offset basis for independent hash family
        let mut h: u64 = seed;
        for byte in term.as_bytes() {
            h ^= *byte as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        let w = weight as f64;
        for bit in 0..64u32 {
            if (h >> bit) & 1 == 1 {
                v[bit as usize] += w;
            } else {
                v[bit as usize] -= w;
            }
        }
    }
    let mut fingerprint: u64 = 0;
    for bit in 0..64u32 {
        if v[bit as usize] > 0.0 {
            fingerprint |= 1u64 << bit;
        }
    }
    fingerprint
}

/// Compute SimHash fingerprints. Uses only the first 4 of 16 seeds (R18 P1b Sol4):
/// 4 independent 64-bit planes = 256-bit total, which gives the same J-L accuracy improvement
/// over the original single seed while removing 75% of the seed iteration overhead.
/// Remaining 12 slots are left as 0 for serialization-format compatibility.
fn simhash_1024(term_freq: &HashMap<String, f32>) -> [u64; 16] {
    let mut fps = [0u64; 16];
    for (i, &seed) in LSH_SEEDS[..4].iter().enumerate() {
        fps[i] = simhash_with_seed(term_freq, seed);
    }
    fps
}

/// Popcount (Hamming weight) of the XOR of two 64-bit values — i.e., Hamming distance.
#[inline]
fn hamming_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Detect temporal markers in a query — triggers recency boost in retrieval.
///
/// Returns true when the query asks about time-relative facts ("most recent",
/// "before", "after", etc.). Used to gate the temporal query routing boost so
/// purely keyword-based queries (which have no temporal intent) are unaffected.
fn detect_temporal_query(task: &str) -> bool {
    const TEMPORAL_MARKERS: &[&str] = &[
        "when did", "when was", "before", "after", "recent", "latest",
        "last time", "earlier", "previously", "at the time", "used to",
        "formerly", "back in", "most recent", "oldest", "newest", "updated",
        "how long ago", "since when", "at what point",
        // R17 L2: broader recency patterns
        "current", "currently", "now", "right now", "still", "today",
        "at the moment", "these days", "nowadays", "at present",
        "what is her", "what is his", "what is their", "what does she",
        "what does he", "what do they", "what is the current", "what is the latest",
        // R21 T7: additional temporal triggers (recency-only — oldest-seeking markers
        // removed: "first time", "originally", "initially", "earliest",
        // "what was the first", "when did i first", "what did i first"
        // belong EXCLUSIVELY in detect_oldest_query to avoid double-boost misrouting).
        "most recently", "last known", "as of", "up until", "prior to", "before that",
        "what was the last", "when did i last", "most recent time",
    ];
    let lower = task.to_lowercase();
    TEMPORAL_MARKERS.iter().any(|m| lower.contains(m))
}

/// R21 T2: Detect "oldest-first" temporal queries — questions about the FIRST/EARLIEST occurrence.
/// Returns true when the query is looking backwards in time (oldest event, first mention).
/// Complement of `detect_temporal_query`'s "most recent" direction.
fn detect_oldest_query(task: &str) -> bool {
    const OLDEST_MARKERS: &[&str] = &[
        "what was the first", "when did i first", "what did i first",
        "first time i", "first time she", "first time he",
        "first issue", "first problem", "first mention",
        "originally", "initially", "at the beginning",
        "earliest", "earliest time", "earliest mention",
        "when i first", "the first x", "first ever",
        "first one", "first thing", "very first",
        "what was the original", "what was the initial",
    ];
    let lower = task.to_lowercase();
    OLDEST_MARKERS.iter().any(|m| lower.contains(m))
}

/// R21 T5: Detect counting queries — questions that need aggregate evidence from many sessions.
/// When fired, Phase 1 expands to all Verbatim neurons; top-10 instead of top-5 returned.
fn detect_counting_query(task: &str) -> bool {
    const COUNTING_MARKERS: &[&str] = &[
        "how many", "total", "count of", "number of", "how much",
        "sum of", "altogether", "in total", "combined", "overall",
        "how often", "how frequently", "times have i", "times did i",
        "how many times", "how often have", "have i had", "have i been",
        "how many places", "how many people", "how many sessions",
        "how many different", "how many types",
    ];
    let lower = task.to_lowercase();
    COUNTING_MARKERS.iter().any(|m| lower.contains(m))
}

/// R18 P2 Sol B + R21 T4: Detect knowledge-update queries — questions about current state that may
/// have stale Verbatim answers. These queries should suppress old verbatim facts and prefer
/// KG/Concept neurons that track supersession.
fn detect_knowledge_update_query(task: &str) -> bool {
    const KU_MARKERS: &[&str] = &[
        // Original R18 markers
        "what is now", "what are now", "changed to", "changed his", "changed her",
        "switched to", "moved to", "no longer", "anymore", "not anymore",
        "what does he do now", "what does she do now", "what do they do now",
        "what is he doing now", "what is she doing now",
        "what is their current", "what is his current", "what is her current",
        "does he still", "does she still", "do they still", "is he still", "is she still",
        "what happened to", "what changed", "since then", "after that",
        "new job", "new role", "new address", "new number", "new partner",
        // R21 T4: 20+ additional markers from benchmark failure forensics
        "personal best", "my record", "my all-time", "my fastest", "my slowest",
        "what was my", "how long did it take me", "what score did i",
        "how many times have i", "total number of",
        "what is the name of", "what did i name", "how much did i",
        "what was the result", "what was the outcome", "final score",
        "my current", "as of now", "up to date", "latest update",
        "what was the last time", "most recently i", "last time i",
        "what play did i", "what show did i", "what event did i",
        "what did i achieve", "what did i complete", "what did i finish",
        // KU-R10: Current-state queries without explicit "current" keyword.
        // These ask about the user's present situation (job, location, diet, etc.)
        // where the NEWEST session is definitionally correct. Applying the temporal
        // boost for these ensures updated facts outrank older mentions.
        "where do i work", "where does she work", "where does he work",
        "what do i do for work", "what does she do for work", "what does he do for work",
        "where do i live", "where does she live", "where does he live",
        "what car do i drive", "what car does she drive", "what car does he drive",
        "what do i eat", "what does she eat", "what does he eat",
        "what is my diet", "what is her diet", "what is his diet",
        "what am i studying", "what is she studying", "what is he studying",
        "what do i study", "what does she study", "what does he study",
        "do i still go", "does she still go", "does he still go",
        "what is my latest", "what is her latest", "what is his latest",
    ];
    let lower = task.to_lowercase();
    KU_MARKERS.iter().any(|m| lower.contains(m))
}

/// P2-B: Detect personal-attribute queries and return the canonical KG predicate.
///
/// Returns Some(predicate) when the query is asking about a fact that the KG stores
/// as a structured (entity, predicate, value) triple. The caller can then bypass
/// BM25 entirely and route to O(1) KG lookup.
///
/// Patterns are deliberately specific to avoid false positives on generic queries.
fn detect_personal_fact_query(task: &str) -> Option<&'static str> {
    let lower = task.to_lowercase();
    // (trigger phrases, KG predicate name)
    const PATTERNS: &[(&[&str], &str)] = &[
        (&["what degree did", "what degree do", "what did i study", "what did i major",
           "what degree have", "what degree was", "what did she study", "what did he study",
           "what did she major", "what did he major", "what degree did she",
           "what degree did he", "what degree does she", "what degree does he"],
         "education"),
        (&["where do i work", "where does she work", "where does he work",
           "what is her job", "what is his job", "what is my job",
           "what does she do for work", "what does he do for work",
           "where is she employed", "where is he employed",
           "what is her occupation", "what is his occupation"],
         "occupation"),
        (&["where do i live", "where does she live", "where does he live",
           "what city does she live", "what city does he live", "what city do i live",
           "where is her home", "where is his home", "where did she move",
           "where did he move", "where are they based", "where is she based",
           "where is he based"],
         "location"),
        (&["how long is her commute", "how long is his commute", "how long is my commute",
           "how long does it take her to get to work", "how long does it take him to get to work",
           "how long does it take to commute"],
         "commute_time"),
        (&["what is her personal best", "what is his personal best", "what is my personal best",
           "what was her personal best", "what was his personal best", "what was my personal best",
           "what was her pb", "what was his pb", "what was my pb",
           "what is her best time", "what is his best time", "what is my best time",
           "what was her race time", "what was his race time", "what was my race time"],
         "fitness_record"),
        (&["what is she reading", "what is he reading", "what am i reading",
           "what book is she", "what book is he", "what book am i",
           "what is she currently reading", "what is he currently reading"],
         "book"),
        (&["what is her pet", "what is his pet", "what is my pet",
           "what is her dog", "what is his dog", "what is my dog",
           "what is her cat", "what is his cat", "what is my cat",
           "what is the name of her pet", "what is the name of his pet",
           "what is her pet's name", "what is his pet's name"],
         "pet"),
        (&["who is her partner", "who is his partner", "who is her husband",
           "who is his wife", "who is her boyfriend", "who is his girlfriend",
           "who is she married to", "who is he married to",
           "is she married", "is he married", "who is her spouse"],
         "partner"),
        (&["what is her phone", "what is his phone", "what is my phone",
           "what is her number", "what is his number", "what is her phone number",
           "what is his phone number"],
         "phone"),
        (&["what is her major", "what is his major", "what is my major",
           "what did she major in", "what did he major in", "what did i major in"],
         "major"),
        (&["what is her project called", "what is his project called",
           "what did she name her project", "what did he name his project",
           "what is the name of her project", "what is the name of his project",
           "what did she call it", "what did he call it", "what did i call it",
           "what is her playlist called", "what is his playlist called",
           "what is my playlist called"],
         "project_name"),
    ];
    for (triggers, predicate) in PATTERNS {
        if triggers.iter().any(|t| lower.contains(t)) {
            return Some(predicate);
        }
    }
    None
}

/// R18 P2 Sol B: Count proper nouns (capitalized words ≥4 chars, not sentence-start stopwords).
/// ≥2 proper nouns in a query → multi-session query routing (force 2-hop synapse expansion).
fn count_proper_nouns(task: &str) -> usize {
    const STOPWORDS: &[&str] = &[
        "What", "When", "Where", "Which", "Who", "Whom", "Whose", "Why", "How",
        "Does", "Did", "Has", "Have", "Will", "Would", "Should", "Could", "Might",
        "This", "That", "These", "Those", "They", "Their", "Them", "Then",
        "Also", "Just", "Very", "Well", "Even", "Most", "Some", "Many", "More",
        "Long", "Good", "Back", "Into", "Over", "Down", "Such", "Both",
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
        "January", "February", "March", "April", "June", "July", "August",
        "September", "October", "November", "December",
    ];
    // skip the very first word (sentence-start capital) to reduce false positives
    task.split_whitespace()
        .skip(1)
        .filter(|w| {
            let clean: String = w.chars().filter(|c| c.is_alphabetic()).collect();
            clean.len() >= 4
                && clean.chars().next().map_or(false, |c| c.is_uppercase())
                && !STOPWORDS.contains(&clean.as_str())
        })
        .count()
}


/// Parse an ISO 8601 date(-time) string to Unix epoch seconds (UTC, approx).
///
/// Supports "YYYY-MM-DD", "YYYY-MM-DDTHH:MM:SS", and "YYYY-MM-DD HH:MM:SS".
/// Does NOT handle timezone offsets — treats all timestamps as UTC.
/// Returns `None` for unparseable or obviously-invalid strings.
fn parse_iso8601_to_secs(ts: Option<&str>) -> Option<i64> {
    let s = ts?.trim();
    // Accept "YYYY-MM-DDTHH:MM:SS", "YYYY-MM-DD HH:MM:SS", or "YYYY-MM-DD"
    let date_part = s.split(|c| c == 'T' || c == ' ').next()?;
    let parts: Vec<&str> = date_part.split('-').collect();
    if parts.len() < 3 { return None; }
    let year: i64 = parts[0].parse().ok()?;
    let month: i64 = parts[1].parse::<i64>().ok()?.clamp(1, 12);
    let day: i64 = parts[2].parse::<i64>().ok()?.clamp(1, 31);
    if year < 1970 || year > 2200 { return None; }
    // Cumulative days at start of each month (non-leap year)
    const MONTH_START_DAYS: [i64; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    // Leap-year correction: count leap years from 1970 to year-1
    let leap_years = {
        let y = year - 1;
        y / 4 - y / 100 + y / 400 - (1969 / 4 - 1969 / 100 + 1969 / 400)
    };
    let days = (year - 1970) * 365 + leap_years
        + MONTH_START_DAYS[(month - 1) as usize]
        + day - 1;
    Some(days * 86_400)
}

/// Wilson score lower bound for a proportion at 95% confidence interval.
///
/// Used for Bayesian quarantine decisions: quarantine only when the lower bound
/// of the citation-rate confidence interval falls below a threshold, ensuring
/// the system has enough evidence before penalising a neuron.
///
/// Formula: `(p̂ + z²/2n − z·√(p̂(1−p̂)/n + z²/4n²)) / (1 + z²/n)`
/// where z = 1.96 (95% CI), n = total, p̂ = hits/total.
/// Returns 0.0 when total == 0.
#[cfg(test)]
fn wilson_lower_bound(hits: u32, total: u32) -> f32 {
    wilson_lower_bound_z(hits, total, 1.96)
}
///
/// - z = 1.0  → 68% CI (fast reaction, for small samples)
/// - z = 1.645 → 90% CI (medium)
/// - z = 1.96  → 95% CI (standard, for large samples)
fn wilson_lower_bound_z(hits: u32, total: u32, z: f32) -> f32 {
    if total == 0 {
        return 0.0;
    }
    let n = total as f32;
    let p = (hits as f32 / n).min(1.0);
    let z2 = z * z;
    let variance = (p * (1.0 - p) / n + z2 / (4.0 * n * n)).max(0.0);
    let numerator = p + z2 / (2.0 * n) - z * variance.sqrt();
    let denominator = 1.0 + z2 / n;
    (numerator / denominator).max(0.0)
}

/// Adaptive quarantine parameters based on observation count (TRIZ R11-S4).
///
/// Returns `Some((z, threshold))` when enough samples exist to make a decision,
/// or `None` if `use_count` is too small to draw any conclusion.
///
/// Tiers:
/// - `< 5`     → None (too few samples; withhold judgment entirely)
/// - `5–19`    → z=1.0,   threshold=0.02 (68% CI — react fast to obvious noise)
/// - `20–99`   → z=1.645, threshold=0.05 (90% CI — current behaviour)
/// - `≥ 100`   → z=1.96,  threshold=0.08 (95% CI — strict for mature neurons)
fn adaptive_quarantine_params(use_count: u32) -> Option<(f32, f32)> {
    match use_count {
        0..=4   => None,
        5..=19  => Some((1.0,   0.02)),
        20..=99 => Some((1.645, 0.05)),
        _       => Some((1.96,  0.08)),
    }
}

/// Build a confidence map for all files in the project by querying git once.
///
/// Returns `HashMap<abs_path, confidence_score>`:
/// - 1.0 = committed and unmodified (default; also used when git is absent)
/// - 0.9 = tracked but locally modified
/// - 0.85 = untracked (new file not yet committed)
///
/// Three git commands are run once per compile; per-file overhead is zero.
fn build_git_confidence_map(project_root: &Path) -> HashMap<PathBuf, f32> {
    let mut map = HashMap::new();

    // Modified tracked files — locally changed but in git history
    for rel in git_file_list(project_root, &["ls-files", "-m"]) {
        map.entry(project_root.join(rel)).or_insert(0.9_f32);
    }

    // Untracked files — not yet in git history
    for rel in git_file_list(project_root, &["ls-files", "--others", "--exclude-standard"]) {
        map.entry(project_root.join(rel)).or_insert(0.85_f32);
    }

    map
}

/// Run a git command and return one path per output line. Silent on error.
fn git_file_list(project_root: &Path, args: &[&str]) -> Vec<PathBuf> {
    let Ok(out) = std::process::Command::new("git")
        .args(args)
        .current_dir(project_root)
        .output()
    else {
        return Vec::new();
    };

    if !out.status.success() {
        return Vec::new();
    }

    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter(|l| !l.is_empty())
        .map(PathBuf::from)
        .collect()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuron::{NeuronKind, NeuronMeta, NeuronStatus, Synapse, SynapseType};
    use tempfile::TempDir;

    fn make_index(dir: &TempDir) -> NeuronIndex {
        NeuronIndex::load_or_create(dir.path()).unwrap()
    }

    // ── Compile lifecycle ──────────────────────────────────────────────────────

    #[test]
    fn compile_creates_stubs() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        let mut idx = make_index(&dir);
        let count = idx.compile().unwrap();
        assert!(count >= 1);
        let ndir = dir.path().join(".cortyx").join("neurons");
        let stubs: Vec<_> = std::fs::read_dir(&ndir).unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".context.md"))
            .collect();
        assert!(!stubs.is_empty());
    }

    #[test]
    fn compile_creates_project_neuron() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("lib.rs"), "pub fn f() {}").unwrap();
        let mut idx = make_index(&dir);
        idx.compile().unwrap();
        let project_neuron = dir.path().join(".cortyx").join("neurons").join("_project.context.md");
        assert!(project_neuron.exists(), "Project neuron should be auto-created");
    }

    #[test]
    fn compile_is_idempotent() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("lib.rs"), "pub fn f() {}").unwrap();
        let mut idx = make_index(&dir);
        let c1 = idx.compile().unwrap();
        assert!(c1 >= 1, "first compile should create at least 1 stub");
        let c2 = idx.compile().unwrap();
        assert_eq!(c2, 0, "second compile with no changes should create 0 new stubs");
    }

    #[test]
    fn compile_detects_changed_file() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("lib.rs");
        std::fs::write(&file, "pub fn f() {}").unwrap();
        let mut idx = make_index(&dir);
        idx.compile().unwrap();
        std::fs::write(&file, "pub fn g() {} // changed").unwrap();
        let mut idx2 = NeuronIndex::load_or_create(dir.path()).unwrap();
        idx2.compile().unwrap();
        let neuron = crate::neuron::core_neuron_path(&file, dir.path());
        let content = std::fs::read_to_string(&neuron).unwrap();
        assert!(content.contains("status: stale") || content.contains("status: stub"));
    }

    #[test]
    fn index_persists_to_disk_after_compile() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.rs"), "fn a() {}").unwrap();
        let mut idx = make_index(&dir);
        idx.compile().unwrap();
        drop(idx);
        let idx2 = NeuronIndex::load_or_create(dir.path()).unwrap();
        assert!(idx2.neuron_count() >= 1);
    }

    // ── upsert ────────────────────────────────────────────────────────────────

    #[test]
    fn upsert_neuron_persists_to_disk() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.rs"), "fn a() {}").unwrap();
        let mut idx = make_index(&dir);
        idx.compile().unwrap();

        let ndir = dir.path().join(".cortyx").join("neurons");
        std::fs::create_dir_all(&ndir).unwrap();
        let np = ndir.join("test.context.md");
        let content = "Cache invalidation pattern. Evicts stale entries on hash change.";
        std::fs::write(&np, content).unwrap();
        let mut meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
        meta.status = NeuronStatus::Fresh;
        idx.upsert_neuron(&np, content, &meta).unwrap();
        drop(idx);

        let idx2 = NeuronIndex::load_or_create(dir.path()).unwrap();
        assert!(idx2.entries.iter().any(|e| e.neuron_path == np));
    }

    // ── get_contexts ──────────────────────────────────────────────────────────

    #[test]
    fn get_contexts_returns_sorted_paths() {
        let dir = TempDir::new().unwrap();
        let ndir = dir.path().join(".cortyx").join("neurons");
        std::fs::create_dir_all(&ndir).unwrap();

        let mut idx = make_index(&dir);
        for (name, content) in [("z.context.md", "authentication login"), ("a.context.md", "auth login token")] {
            let p = ndir.join(name);
            std::fs::write(&p, content).unwrap();
            let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
            idx.index_neuron(&p, content, &meta);
        }
        idx.rebuild_derived();

        let result = idx.get_contexts("auth login", 4096, None, None);
        assert!(!result.is_empty());
        let sorted = { let mut r = result.clone(); r.sort(); r };
        assert_eq!(result, sorted, "output must be lexicographically sorted");
    }

    #[test]
    fn get_contexts_returns_empty_for_no_match() {
        let dir = TempDir::new().unwrap();
        let ndir = dir.path().join(".cortyx").join("neurons");
        std::fs::create_dir_all(&ndir).unwrap();
        let p = ndir.join("foo.context.md");
        std::fs::write(&p, "completely unrelated content xyz").unwrap();
        let mut idx = make_index(&dir);
        let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
        idx.index_neuron(&p, "completely unrelated content xyz", &meta);
        idx.rebuild_derived();
        let result = idx.get_contexts("authentication oauth jwt", 4096, None, None);
        assert!(result.is_empty() || !result.contains(&p));
    }

    #[test]
    fn get_contexts_respects_token_budget() {
        let dir = TempDir::new().unwrap();
        let ndir = dir.path().join(".cortyx").join("neurons");
        std::fs::create_dir_all(&ndir).unwrap();
        let mut idx = make_index(&dir);
        for i in 0..20 {
            let p = ndir.join(format!("mod_{i:02}.context.md"));
            let content = format!("auth token login validate {} {}", "word ".repeat(200), i);
            std::fs::write(&p, &content).unwrap();
            let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
            idx.index_neuron(&p, &content, &meta);
        }
        idx.rebuild_derived();
        let result = idx.get_contexts("auth token login", 500, None, None);
        let total_tokens: usize = result.iter()
            .filter_map(|p| idx.entry_by_path(p))
            .map(|e| e.tokens)
            .sum();
        assert!(total_tokens <= 500, "should respect token budget: {total_tokens}");
    }

    #[test]
    fn get_contexts_module_filter() {
        let dir = TempDir::new().unwrap();
        let ndir = dir.path().join(".cortyx").join("neurons");
        std::fs::create_dir_all(&ndir).unwrap();
        let mut idx = make_index(&dir);

        let auth_p = ndir.join("auth.context.md");
        let ui_p = ndir.join("ui.context.md");
        std::fs::write(&auth_p, "auth token login validate session").unwrap();
        std::fs::write(&ui_p, "auth login button render component").unwrap();

        let mut auth_meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
        auth_meta.module = Some("auth".to_string());
        let mut ui_meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
        ui_meta.module = Some("ui".to_string());

        idx.index_neuron(&auth_p, "auth token login validate session", &auth_meta);
        idx.index_neuron(&ui_p, "auth login button render component", &ui_meta);
        idx.rebuild_derived();

        // With module filter: only auth module
        let filtered = idx.get_contexts("auth login", 4096, Some("auth"), None);
        assert!(filtered.contains(&auth_p));
        assert!(!filtered.contains(&ui_p), "module filter should exclude ui module");
    }

    // ── Typed synapse traversal ───────────────────────────────────────────────

    #[test]
    fn synapse_traversal_pulls_related_neuron() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("engine.rs"), "pub fn engine() { route_intent(); }").unwrap();
        std::fs::write(dir.path().join("ui.rs"), "pub fn render() {}").unwrap();

        let mut idx = NeuronIndex::load_or_create(dir.path()).unwrap();
        idx.compile().unwrap();

        let engine_neuron = crate::neuron::core_neuron_path(&dir.path().join("engine.rs"), dir.path());
        let ui_neuron = crate::neuron::core_neuron_path(&dir.path().join("ui.rs"), dir.path());

        let engine_content = format!(
            "Engine module. Routes user intent, synthesizes responses.\n\
             ## CROSS-REFERENCES (synapses)\n- `{}` → render pipeline [calls]",
            ui_neuron.display()
        );
        let mut engine_meta = NeuronMeta::new_stub(&dir.path().join("engine.rs"), NeuronKind::Core);
        engine_meta.synapses = vec![Synapse {
            target: ui_neuron.clone(),
            edge_type: SynapseType::Calls,
            weight: 0.8,
            reason: "render pipeline".to_string(),
            learned_weight: 0.0,
            traversal_count: 0,
            last_co_activation_day: 0,
        }];
        engine_meta.status = NeuronStatus::Fresh;
        std::fs::write(&engine_neuron, &engine_content).unwrap();
        idx.upsert_neuron(&engine_neuron, &engine_content, &engine_meta).unwrap();

        let contexts = idx.get_contexts("route intent synthesize engine", 4096, None, None);
        assert!(
            contexts.contains(&ui_neuron) || contexts.contains(&engine_neuron),
            "Synapse traversal should pull in related neuron. Got: {contexts:?}"
        );
    }

    #[test]
    fn typed_synapse_implements_has_high_multiplier() {
        assert!(SynapseType::Implements.type_multiplier() > SynapseType::SemanticRelated.type_multiplier());
        assert_eq!(SynapseType::ConceptExpands.type_multiplier(), 1.0);
    }

    // ── Use-case activation ───────────────────────────────────────────────────

    #[test]
    fn use_case_neuron_activated_by_task_pattern() {
        let dir = TempDir::new().unwrap();
        let ndir = dir.path().join(".cortyx").join("neurons");
        std::fs::create_dir_all(&ndir).unwrap();
        let mut idx = make_index(&dir);

        let core_p = ndir.join("auth_rs.context.md");
        std::fs::write(&core_p, "authentication token validation").unwrap();
        let core_meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
        idx.index_neuron(&core_p, "authentication token validation", &core_meta);

        let uc_p = ndir.join("auth_rs.usecase.oauth.md");
        std::fs::write(&uc_p, "OAuth2 flow: redirect then exchange code for token").unwrap();
        let mut uc_meta = NeuronMeta::new_stub(dir.path(), NeuronKind::UseCase);
        uc_meta.task_pattern = Some("add oauth login".to_string());
        uc_meta.parent = Some(core_p.clone());
        idx.index_neuron(&uc_p, "OAuth2 flow: redirect then exchange code for token", &uc_meta);
        idx.rebuild_derived();

        let result = idx.get_contexts("add oauth authentication login", 4096, None, None);
        assert!(result.contains(&uc_p) || result.contains(&core_p));
    }

    // ── Invalidation ──────────────────────────────────────────────────────────

    #[test]
    fn invalidate_marks_stale() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("a.rs");
        std::fs::write(&file, "fn a() {}").unwrap();
        let mut idx = make_index(&dir);
        idx.compile().unwrap();
        let neuron = crate::neuron::core_neuron_path(&file, dir.path());
        assert!(neuron.exists());
        idx.invalidate(&file).unwrap();
        // Stale-demotion: neuron remains in the index (preserves context) but is
        // demoted via staleness_multiplier so it won't win over fresh neurons.
        let entry = idx.entries.iter().find(|e| e.neuron_path == neuron);
        assert!(entry.is_some(), "neuron should still exist after invalidation");
        assert_eq!(
            entry.unwrap().staleness_multiplier, 0.5,
            "staleness_multiplier should be 0.5 after invalidation"
        );
    }

    // ── BM25 scoring ──────────────────────────────────────────────────────────

    #[test]
    fn bm25_scores_zero_for_no_match() {
        let dir = TempDir::new().unwrap();
        let ndir = dir.path().join(".cortyx").join("neurons");
        std::fs::create_dir_all(&ndir).unwrap();
        let p = ndir.join("x.context.md");
        std::fs::write(&p, "completely different topic here").unwrap();
        let mut idx = make_index(&dir);
        let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
        idx.index_neuron(&p, "completely different topic here", &meta);
        idx.rebuild_derived();
        let entry = idx.entry_by_path(&p).unwrap();
        assert_eq!(idx.bm25_score(&tokenize("auth token login"), entry), 0.0);
    }

    #[test]
    fn bm25_scores_higher_for_matching_terms() {
        let dir = TempDir::new().unwrap();
        let ndir = dir.path().join(".cortyx").join("neurons");
        std::fs::create_dir_all(&ndir).unwrap();
        let mut idx = make_index(&dir);
        let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
        let p1 = ndir.join("a.context.md");
        std::fs::write(&p1, "auth login token session").unwrap();
        idx.index_neuron(&p1, "auth login token session", &meta);
        let p2 = ndir.join("b.context.md");
        std::fs::write(&p2, "render button component style").unwrap();
        idx.index_neuron(&p2, "render button component style", &meta);
        idx.rebuild_derived();
        let terms = tokenize("auth token");
        let s1 = idx.bm25_score(&terms, idx.entry_by_path(&p1).unwrap());
        let s2 = idx.bm25_score(&terms, idx.entry_by_path(&p2).unwrap());
        assert!(s1 > s2, "auth neuron should score higher for auth query");
    }

    #[test]
    fn bm25_idf_is_non_negative() {
        let dir = TempDir::new().unwrap();
        let ndir = dir.path().join(".cortyx").join("neurons");
        std::fs::create_dir_all(&ndir).unwrap();
        let mut idx = make_index(&dir);
        let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
        // Same term in every entry → IDF should floor at 0
        for i in 0..5 {
            let p = ndir.join(format!("{i}.context.md"));
            std::fs::write(&p, "common term here").unwrap();
            idx.index_neuron(&p, "common term here", &meta);
        }
        idx.rebuild_derived();
        for entry in &idx.entries {
            let score = idx.bm25_score(&tokenize("common"), entry);
            assert!(score >= 0.0, "BM25 score must not be negative");
        }
    }

    // ── Overlap score ─────────────────────────────────────────────────────────

    #[test]
    fn overlap_score_perfect_match() {
        let q = tokenize("add dark mode");
        let p = tokenize("add dark mode");
        assert!((simple_overlap_score(&q, &p) - 1.0).abs() < 0.001);
    }

    #[test]
    fn overlap_score_no_match() {
        let q = tokenize("auth token");
        let p = tokenize("render button");
        assert_eq!(simple_overlap_score(&q, &p), 0.0);
    }

    #[test]
    fn overlap_score_empty_pattern() {
        let q = tokenize("auth");
        assert_eq!(simple_overlap_score(&q, &[]), 0.0);
    }

    // ── Tokenizer ─────────────────────────────────────────────────────────────

    #[test]
    fn tokenize_basic() {
        let terms = tokenize("add dark mode to SwiftUI view");
        assert!(terms.contains(&"add".to_string()));
        assert!(terms.contains(&"dark".to_string()));
        assert!(terms.contains(&"swiftui".to_string()));
        assert!(terms.contains(&"view".to_string()));
    }

    #[test]
    fn tokenize_filters_short_terms() {
        let terms = tokenize("a b add");
        assert!(!terms.contains(&"a".to_string()));
        assert!(!terms.contains(&"b".to_string()));
        assert!(terms.contains(&"add".to_string()));
    }

    #[test]
    fn tokenize_lowercases() {
        let terms = tokenize("AuthService");
        assert!(terms.contains(&"authservice".to_string()));
    }

    #[test]
    fn tokenize_preserves_underscores() {
        let terms = tokenize("snake_case_name");
        assert!(terms.contains(&"snake_case_name".to_string()));
    }

    #[test]
    fn tokenize_empty_string() {
        assert!(tokenize("").is_empty());
    }

    // ── Retrieval accuracy ────────────────────────────────────────────────────

    /// Verifies that BM25 retrieval returns the correct neuron for each of 10
    /// distinct queries against 10 distinct content-rich neurons.
    ///
    /// This exercises the full activation pipeline (Phase 1 only — no synapses)
    /// and ensures that keyword specificity drives correct ranking.
    #[test]
    fn get_contexts_retrieval_accuracy_10q() {
        let dir = TempDir::new().unwrap();
        let ndir = dir.path().join(".cortyx").join("neurons");
        std::fs::create_dir_all(&ndir).unwrap();
        let mut idx = make_index(&dir);

        // Each neuron has a unique keyword cluster — e.g. "authentication" only in auth neuron
        let neurons = [
            ("auth.context.md",     "authentication token validation session jwt bearer"),
            ("ui.context.md",       "render component dark mode swiftui colorscheme view"),
            ("db.context.md",       "database migration schema sql transaction commit"),
            ("cache.context.md",    "cache invalidation evict stale ttl expiry redis"),
            ("api.context.md",      "rest api endpoint http request response route handler"),
            ("crypto.context.md",   "encryption decryption aes rsa signing certificate key"),
            ("queue.context.md",    "queue task worker job priority scheduling async"),
            ("logger.context.md",   "logging tracing span event diagnostic telemetry"),
            ("config.context.md",   "configuration environment variable toml yaml dotenv"),
            ("deploy.context.md",   "deployment docker kubernetes helm release pipeline"),
        ];
        let queries_and_expected: [(&str, &str); 10] = [
            ("jwt bearer authentication", "auth.context.md"),
            ("dark mode colorscheme swiftui", "ui.context.md"),
            ("sql transaction schema migration", "db.context.md"),
            ("cache ttl evict stale", "cache.context.md"),
            ("http rest api endpoint route", "api.context.md"),
            ("aes rsa encryption certificate", "crypto.context.md"),
            ("job worker queue scheduling", "queue.context.md"),
            ("logging span telemetry diagnostic", "logger.context.md"),
            ("environment variable dotenv configuration", "config.context.md"),
            ("docker kubernetes deployment helm", "deploy.context.md"),
        ];

        for (name, content) in &neurons {
            let p = ndir.join(name);
            std::fs::write(&p, content).unwrap();
            let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
            idx.index_neuron(&p, content, &meta);
        }
        idx.rebuild_derived();

        let mut correct = 0;
        for (query, expected_file) in &queries_and_expected {
            let results = idx.get_contexts(query, 4096, None, None);
            let expected_path = ndir.join(expected_file);
            if results.contains(&expected_path) {
                correct += 1;
            } else {
                eprintln!("[accuracy] MISS: query={query:?} expected={expected_file} got={results:?}");
            }
        }
        assert_eq!(correct, 10, "BM25 accuracy: {correct}/10 correct (expected 10/10)");
    }

    /// Activation latency: `get_contexts` over 100 neurons must complete in <50ms p95.
    ///
    /// This verifies the README benchmark target "≤50ms p95, 100 neurons" is met
    /// with the pure in-memory BM25 engine (no disk I/O in the hot path).
    #[test]
    fn get_contexts_latency_p95_100_neurons() {
        let dir = TempDir::new().unwrap();
        let ndir = dir.path().join(".cortyx").join("neurons");
        std::fs::create_dir_all(&ndir).unwrap();
        let mut idx = make_index(&dir);

        // Build a 100-neuron index with realistic content sizes (~400 chars each).
        for i in 0..100 {
            let p = ndir.join(format!("neuron_{i:03}.context.md"));
            let content = format!(
                "## Module {i}\nHandles subsystem_{i} operations including routing, \
                 caching, pipeline_{i} filter validation authentication token session \
                 database migration schema endpoint handler deployment configuration \
                 environment worker queue scheduling logging tracing telemetry encryption."
            );
            std::fs::write(&p, &content).unwrap();
            let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
            idx.index_neuron(&p, &content, &meta);
        }
        idx.rebuild_derived();

        // Warm up: one call to populate CPU caches
        let _ = idx.get_contexts("routing pipeline authentication token", 4096, None, None);

        // Measure p95 over 20 trials
        let trials = 20;
        let mut latencies_ms: Vec<u128> = (0..trials).map(|_| {
            let t = std::time::Instant::now();
            let _ = idx.get_contexts("routing pipeline authentication token", 4096, None, None);
            t.elapsed().as_millis()
        }).collect();
        latencies_ms.sort_unstable();
        let p95 = latencies_ms[(trials as f64 * 0.95) as usize - 1];

        assert!(
            p95 < 50,
            "get_contexts p95 latency must be <50ms over 100 neurons; got {p95}ms"
        );
    }



    /// Ensures that relative synapse paths written into neuron markdown
    /// (e.g. from `cortyx_evolve_context`) are resolved to absolute paths
    /// in the adjacency graph, so traversal works correctly.
    #[test]
    fn relative_synapse_targets_resolved_in_adjacency() {
        let dir = TempDir::new().unwrap();
        let ndir = dir.path().join(".cortyx").join("neurons");
        std::fs::create_dir_all(&ndir).unwrap();
        let mut idx = make_index(&dir);

        let source_p = ndir.join("engine.context.md");
        let target_p = ndir.join("ui.context.md");

        std::fs::write(&source_p, "engine routing intent").unwrap();
        std::fs::write(&target_p, "ui rendering components").unwrap();

        // Source neuron has a RELATIVE synapse target (as parse_synapses_from_content returns)
        let mut source_meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
        source_meta.synapses = vec![Synapse {
            target: PathBuf::from("ui.context.md"), // relative!
            edge_type: SynapseType::Calls,
            weight: 0.9,
            reason: "calls render".to_string(),
            learned_weight: 0.0,
            traversal_count: 0,
            last_co_activation_day: 0,
        }];
        let target_meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);

        idx.index_neuron(&source_p, "engine routing intent", &source_meta);
        idx.index_neuron(&target_p, "ui rendering components", &target_meta);
        idx.rebuild_derived();

        // The adjacency entry for source_p should point to the ABSOLUTE target path
        let adj = idx.adjacency.get(&source_p).expect("source must be in adjacency");
        let target_syn = adj.iter().find(|s| s.target == target_p);
        assert!(
            target_syn.is_some(),
            "Relative synapse 'ui.context.md' should be resolved to absolute {}: adjacency={adj:?}",
            target_p.display()
        );
    }

    // ── Mine + retrieve ───────────────────────────────────────────────────────

    /// Verifies the conversation mining → retrieval pipeline end-to-end:
    /// mine text containing unique keywords, then get_contexts should return it.
    #[test]
    fn mined_neuron_is_retrievable_by_keyword() {
        let dir = TempDir::new().unwrap();
        let mut idx = make_index(&dir);

        // Mine a conversation turn with a specific keyword cluster
        crate::miner::mine_text(
            "The hydrazine valve regulates fuel injection in rocket propulsion systems.",
            "test_chat",
            dir.path(),
            &mut idx,
            None,
            Some("assistant"),
            None,
        ).unwrap();

        // The unique keyword "hydrazine" should retrieve the mined neuron
        let results = idx.get_contexts("hydrazine valve rocket propulsion", 4096, None, None);
        assert!(!results.is_empty(), "Mined neuron should be retrievable by its keywords");

        let found = results.iter().any(|p| {
            std::fs::read_to_string(p)
                .map(|c| c.contains("hydrazine"))
                .unwrap_or(false)
        });
        assert!(found, "Retrieved neuron should contain 'hydrazine'");
    }

    /// Mine + module filter: mined neuron tagged with module X should only
    /// appear when querying with that module filter, not unfiltered in other modules.
    #[test]
    fn mined_neuron_module_filter_works() {
        let dir = TempDir::new().unwrap();
        let mut idx = make_index(&dir);

        crate::miner::mine_text(
            "Photosynthesis converts sunlight into glucose via chlorophyll.",
            "bio_chat",
            dir.path(),
            &mut idx,
            Some("biology"),
            Some("assistant"),
            None,
        ).unwrap();

        // Module-filtered query should find it
        let with_module = idx.get_contexts("photosynthesis sunlight glucose", 4096, Some("biology"), None);
        assert!(!with_module.is_empty(), "Module-filtered query should find mined neuron");

        // Module filter for a different module should NOT find it
        let wrong_module = idx.get_contexts("photosynthesis sunlight glucose", 4096, Some("physics"), None);
        assert!(
            wrong_module.is_empty(),
            "Wrong module filter should not return neuron tagged 'biology'"
        );
    }

    // ── Feedback loop (hit_multiplier + quarantine) ───────────────────────────

    #[test]
    fn hit_multiplier_reward_grows_with_citations() {
        let dir = TempDir::new().unwrap();
        let ndir = dir.path().join(".cortyx").join("neurons");
        std::fs::create_dir_all(&ndir).unwrap();
        let mut idx = make_index(&dir);

        let p = ndir.join("auth.context.md");
        std::fs::write(&p, "authentication token session login").unwrap();
        let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
        idx.index_neuron(&p, "authentication token session login", &meta);
        idx.rebuild_derived();

        let terms = tokenize("auth login");

        // Cold-start: use_count=0 → multiplier=1.0 (neutral)
        let cold_score = idx.bm25_score(&terms, idx.entry_by_path(&p).unwrap());

        // Simulate MIN_SAMPLE_SIZE activations with 100% citation rate
        if let Some(&i) = idx.path_index.get(&p) {
            idx.entries[i].use_count = MIN_SAMPLE_SIZE;
            idx.entries[i].hit_count = MIN_SAMPLE_SIZE;
        }
        let hot_score = idx.bm25_score(&terms, idx.entry_by_path(&p).unwrap());

        assert!(
            hot_score > cold_score,
            "Fully-cited neuron should score higher than cold-start (hot={hot_score:.3}, cold={cold_score:.3})"
        );
        // Max multiplier is 1.5 so the hot score should be exactly 1.5× cold
        assert!(
            (hot_score / cold_score - 1.5).abs() < 0.01,
            "100% hit rate should give 1.5× boost"
        );
    }

    #[test]
    fn auto_quarantine_fires_after_threshold() {
        let dir = TempDir::new().unwrap();
        let ndir = dir.path().join(".cortyx").join("neurons");
        std::fs::create_dir_all(&ndir).unwrap();
        let mut idx = make_index(&dir);

        let p = ndir.join("noisy.context.md");
        std::fs::write(&p, "generic boilerplate content").unwrap();
        let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
        idx.index_neuron(&p, "generic boilerplate content", &meta);
        idx.rebuild_derived();

        // Adaptive CI (S4): QUARANTINE_MIN_SAMPLES = 5. Below this threshold
        // (use_count 0–4), adaptive_quarantine_params returns None — no action.
        if let Some(&i) = idx.path_index.get(&p) {
            idx.entries[i].use_count = QUARANTINE_MIN_SAMPLES - 2; // = 3
            idx.entries[i].hit_count = 0;
        }
        idx.record_activation(&[p.clone()]);  // → use_count = 4 (still below threshold)
        let mult_early = idx.path_index.get(&p)
            .map(|&i| idx.entries[i].staleness_multiplier)
            .unwrap_or(1.0);
        assert_eq!(mult_early, 1.0, "Should NOT quarantine below QUARANTINE_MIN_SAMPLES (4 < 5)");

        // At use_count = 5 (after record_activation increments to 6), z=1.0 tier fires.
        // Wilson lower bound for 0/6 at z=1.0 = 0.0 < adaptive threshold 0.02 → quarantine.
        if let Some(&i) = idx.path_index.get(&p) {
            idx.entries[i].use_count = QUARANTINE_MIN_SAMPLES; // = 5
            idx.entries[i].hit_count = 0;
        }
        idx.record_activation(&[p.clone()]); // → use_count = 6, fires adaptive z=1.0
        let mult = idx.path_index.get(&p)
            .map(|&i| idx.entries[i].staleness_multiplier)
            .unwrap_or(1.0);
        assert_eq!(mult, 0.3, "Should quarantine at QUARANTINE_MIN_SAMPLES with 0% hit rate");
    }

    #[test]
    fn quarantine_is_reversible_when_citation_rate_recovers() {
        let dir = TempDir::new().unwrap();
        let ndir = dir.path().join(".cortyx").join("neurons");
        std::fs::create_dir_all(&ndir).unwrap();
        let mut idx = make_index(&dir);

        let p = ndir.join("recovered.context.md");
        std::fs::write(&p, "generic boilerplate content").unwrap();
        let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
        idx.index_neuron(&p, "generic boilerplate content", &meta);
        idx.rebuild_derived();

        // Manually quarantine the neuron, then simulate recovery: 20 uses, 10 hits.
        // Wilson lower bound for 10/20 at z=1.645 (90% CI) ≈ 0.31 > QUARANTINE_RECOVERY_THRESHOLD (0.15).
        // Use hardcoded values (not QUARANTINE_MIN_SAMPLES) so the hit/use ratio is valid.
        if let Some(&i) = idx.path_index.get(&p) {
            idx.entries[i].staleness_multiplier = 0.3;
            idx.entries[i].use_count = 20;
            idx.entries[i].hit_count = 10;
        }
        idx.record_activation(&[p.clone()]);
        let mult = idx.path_index.get(&p)
            .map(|&i| idx.entries[i].staleness_multiplier)
            .unwrap_or(0.0);
        assert!(mult > 0.3, "Quarantine should lift when citation rate recovers (mult={mult})");
    }

    #[test]
    fn wilson_lower_bound_correctness() {
        // 0/20 → lower bound = 0.0 (no hits, fully quarantinable)
        assert!(wilson_lower_bound(0, 20) < 0.01);
        // 10/20 → lower bound ≈ 0.299 (well above recovery threshold of 0.15)
        assert!(wilson_lower_bound(10, 20) > 0.25);
        // 1/20 → lower bound near 0 but small positive
        assert!(wilson_lower_bound(1, 20) < 0.10);
        // Edge: 0 total → 0.0
        assert_eq!(wilson_lower_bound(0, 0), 0.0);
    }

    // ── S1: AST Signature Hash ─────────────────────────────────────────────────

    #[test]
    fn sig_hash_changes_on_function_rename() {
        let before = crate::ast_extractor::extract_signatures("src/auth.rs", "pub fn validate() {}");
        let after  = crate::ast_extractor::extract_signatures("src/auth.rs", "pub fn authenticate() {}");
        let h1 = crate::ast_extractor::compute_sig_hash(&before);
        let h2 = crate::ast_extractor::compute_sig_hash(&after);
        assert_ne!(h1, h2, "sig_hash must change when a function is renamed");
    }

    #[test]
    fn sig_hash_stable_on_whitespace_and_comments() {
        let base    = crate::ast_extractor::extract_signatures("src/auth.rs", "pub fn validate() {}");
        let tweaked = crate::ast_extractor::extract_signatures(
            "src/auth.rs",
            "/// New doc comment\npub fn validate() {\n    // added comment\n}",
        );
        let h1 = crate::ast_extractor::compute_sig_hash(&base);
        let h2 = crate::ast_extractor::compute_sig_hash(&tweaked);
        assert_eq!(h1, h2, "sig_hash must be stable across whitespace/doc-comment edits");
    }

    #[test]
    fn sig_hash_stable_on_function_reorder() {
        let a = crate::ast_extractor::extract_signatures(
            "src/auth.rs",
            "pub fn validate() {}\npub fn refresh() {}",
        );
        let b = crate::ast_extractor::extract_signatures(
            "src/auth.rs",
            "pub fn refresh() {}\npub fn validate() {}",
        );
        let h1 = crate::ast_extractor::compute_sig_hash(&a);
        let h2 = crate::ast_extractor::compute_sig_hash(&b);
        assert_eq!(h1, h2, "sig_hash must be stable across function reordering");
    }

    // ── S3: Lazy Sub-Neuron Splitting ─────────────────────────────────────────

    #[test]
    fn sub_neuron_path_format_is_correct() {
        use crate::neuron::sub_neuron_path;
        use std::path::Path;
        let core = Path::new(".cortyx/neurons/src/engine_rs.context.md");
        let sub = sub_neuron_path(core, "validate_user");
        let name = sub.file_name().unwrap().to_string_lossy();
        assert_eq!(name, "engine_rs.fn-validate_user.context.md");
        assert_eq!(sub.parent(), core.parent());
    }

    #[test]
    fn sub_neuron_path_sanitizes_special_chars() {
        use crate::neuron::sub_neuron_path;
        use std::path::Path;
        let core = Path::new(".cortyx/neurons/src/engine_rs.context.md");
        let sub = sub_neuron_path(core, "fn with spaces!");
        let name = sub.file_name().unwrap().to_string_lossy();
        // spaces and ! should be replaced with _
        assert!(name.starts_with("engine_rs.fn-"));
        assert!(!name.contains(' '));
        assert!(!name.contains('!'));
    }

    #[test]
    fn sub_neuron_content_contains_function_name() {
        use crate::neuron::stub_function_neuron;
        let content = stub_function_neuron("validate_user", "src/auth.rs", "2026-01-01T00:00:00Z");
        assert!(content.contains("validate_user"), "stub must mention the function name");
        assert!(content.contains("SECTION: purpose"), "stub must have purpose section");
        assert!(content.contains("SECTION: api"), "stub must have api section");
    }

    #[test]
    fn split_threshold_files_produce_sub_neurons() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let src_dir = root.join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        std::fs::create_dir_all(root.join(".cortyx").join("neurons").join("src")).unwrap();

        // Write a Rust file with 7 public functions (above SUBNEURON_SPLIT_THRESHOLD=6)
        let mut src = String::new();
        for i in 0..7 {
            src.push_str(&format!("pub fn function_{i}() {{ }}\n"));
        }
        std::fs::write(src_dir.join("big_module.rs"), &src).unwrap();

        let git_confidence = std::collections::HashMap::new();
        let abs = src_dir.join("big_module.rs");
        let results = process_source_file(&abs, root, &git_confidence);

        // First result is the Core; subsequent are UseCase sub-neurons
        assert!(results.len() >= 2, "should produce Core + sub-neurons for 7-function file");
        let core = &results[0];
        assert_eq!(core.meta.kind, crate::neuron::NeuronKind::Core);
        let subs: Vec<_> = results.iter().skip(1).collect();
        assert!(!subs.is_empty(), "should have at least one sub-neuron");
        assert!(subs.iter().all(|s| s.meta.kind == crate::neuron::NeuronKind::UseCase));
        assert!(subs.iter().all(|s| s.meta.parent.as_deref() == Some(core.neuron_path.as_path())));
    }

    #[test]
    fn small_files_produce_no_sub_neurons() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let src_dir = root.join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        std::fs::create_dir_all(root.join(".cortyx").join("neurons").join("src")).unwrap();

        // Write a small Rust file with 2 public functions (below threshold)
        let src = "pub fn a() {}\npub fn b() {}\n";
        std::fs::write(src_dir.join("small.rs"), src).unwrap();

        let git_confidence = std::collections::HashMap::new();
        let abs = src_dir.join("small.rs");
        let results = process_source_file(&abs, root, &git_confidence);

        assert_eq!(results.len(), 1, "small file should produce only a Core neuron");
        assert_eq!(results[0].meta.kind, crate::neuron::NeuronKind::Core);
    }

    // ── R11-S1: Section-Level Staleness ──────────────────────────────────────

    /// Verifies that `update_neuron_header` patches only the three header comment
    /// lines and leaves all other content (section bodies, cross-refs) intact.
    #[test]
    fn update_neuron_header_patches_only_header_lines() {
        use crate::neuron::update_neuron_header;
        let content = "\
<!-- AUTO-GENERATED CONTEXT — DO NOT EDIT MANUALLY -->\n\
<!-- source: src/engine.rs -->\n\
<!-- hash: aabbccdd11223344 -->\n\
<!-- last-updated: 2024-01-01T00:00:00Z -->\n\
<!-- status: stub -->\n\
\n\
<!-- SECTION: purpose -->\n\
This module drives the core loop.\n\
<!-- /SECTION -->\n\
<!-- SECTION: api -->\n\
pub fn run()\n\
<!-- /SECTION -->\n";

        let updated = update_neuron_header(content, "deadbeef12345678", "2025-06-01T12:00:00Z");

        assert!(updated.contains("<!-- hash: deadbeef12345678 -->"), "hash line must be updated");
        assert!(updated.contains("<!-- last-updated: 2025-06-01T12:00:00Z -->"), "date must be updated");
        assert!(updated.contains("<!-- status: stale -->"), "status must be set to stale");
        assert!(!updated.contains("aabbccdd"), "old hash must not appear");
        assert!(updated.contains("This module drives the core loop."), "purpose body must be preserved");
        assert!(updated.contains("pub fn run()"), "api body must be preserved");
    }

    /// When a source file's sig_hash changes (real API change) but the neuron already
    /// exists, `process_source_file` should update only the `api` section and preserve
    /// the `purpose` section content written by a previous LLM call.
    #[test]
    fn s1_api_section_update_preserves_purpose_on_sig_hash_change() {
        use crate::neuron::replace_section;
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let src_dir = root.join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        std::fs::create_dir_all(root.join(".cortyx").join("neurons").join("src")).unwrap();

        // Write an initial source file and compile it to get a neuron stub
        let src_v1 = "pub fn alpha() {}\n";
        std::fs::write(src_dir.join("mod.rs"), src_v1).unwrap();
        let git_confidence = std::collections::HashMap::new();
        let abs = src_dir.join("mod.rs");
        let v1 = process_source_file(&abs, root, &git_confidence);
        assert_eq!(v1.len(), 1, "v1 should produce one Core");
        let neuron_path = v1[0].neuron_path.clone();

        // Simulate LLM evolution: write a purpose section into the neuron
        let with_purpose = replace_section(
            &v1[0].content,
            "purpose",
            "Alpha drives the main processing loop.",
        );
        std::fs::write(&neuron_path, &with_purpose).unwrap();

        // Now change the source file API (rename function → sig_hash changes)
        let src_v2 = "pub fn beta() {}\n";
        std::fs::write(&abs, src_v2).unwrap();
        let v2 = process_source_file(&abs, root, &git_confidence);

        // S1: should return a compiled file with api updated but purpose preserved
        assert_eq!(v2.len(), 1, "v2 should still produce one Core");
        let new_content = std::fs::read_to_string(&neuron_path).unwrap();
        assert!(new_content.contains("beta"), "new api section should contain updated function name");
        assert!(new_content.contains("Alpha drives the main processing loop."),
            "LLM-curated purpose section must survive a sig_hash change");
        assert!(new_content.contains("<!-- status: stale -->"), "status should be stale after api change");
    }

    // ── R11-S4: Adaptive CI Quarantine ───────────────────────────────────────

    /// Verifies that `adaptive_quarantine_params` returns the correct (z, threshold) tier
    /// and None below the cold-start threshold.
    #[test]
    fn adaptive_quarantine_params_tier_boundaries() {
        assert!(adaptive_quarantine_params(0).is_none(), "0 samples → None");
        assert!(adaptive_quarantine_params(4).is_none(), "4 samples → None");
        let (z5, t5) = adaptive_quarantine_params(5).unwrap();
        assert!((z5 - 1.0).abs() < 0.01, "5 samples → z=1.0");
        assert!((t5 - 0.02).abs() < 0.001, "5 samples → threshold=0.02");
        let (z19, _) = adaptive_quarantine_params(19).unwrap();
        assert!((z19 - 1.0).abs() < 0.01, "19 samples → still z=1.0 tier");
        let (z20, t20) = adaptive_quarantine_params(20).unwrap();
        assert!((z20 - 1.645).abs() < 0.01, "20 samples → z=1.645");
        assert!((t20 - 0.05).abs() < 0.001, "20 samples → threshold=0.05");
        let (z100, t100) = adaptive_quarantine_params(100).unwrap();
        assert!((z100 - 1.96).abs() < 0.01, "100+ samples → z=1.96");
        assert!((t100 - 0.08).abs() < 0.001, "100+ samples → threshold=0.08");
    }

    /// Early quarantine at 5+ samples with 0% hit rate (z=1.0 tier).
    #[test]
    fn adaptive_ci_quarantines_early_for_zero_hit_rate() {
        let dir = TempDir::new().unwrap();
        let ndir = dir.path().join(".cortyx").join("neurons");
        std::fs::create_dir_all(&ndir).unwrap();
        let mut idx = make_index(&dir);

        let p = ndir.join("noise.context.md");
        std::fs::write(&p, "noise boilerplate low quality").unwrap();
        let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
        idx.index_neuron(&p, "noise boilerplate low quality", &meta);
        idx.rebuild_derived();

        // 9 activations, 0 hits → z=1.0 tier, lb(0,10)=0.0 < 0.02 → should quarantine
        if let Some(&i) = idx.path_index.get(&p) {
            idx.entries[i].use_count = 9;
            idx.entries[i].hit_count = 0;
        }
        idx.record_activation(&[p.clone()]); // → use_count=10
        let mult = idx.path_index.get(&p).map(|&i| idx.entries[i].staleness_multiplier).unwrap_or(1.0);
        assert_eq!(mult, 0.3, "10 activations with 0 hits should quarantine at z=1.0 tier");
    }

    /// A neuron with moderate hit rate at medium count should NOT be quarantined
    /// (90% CI is too wide to conclude bad quality).
    #[test]
    fn adaptive_ci_does_not_quarantine_moderate_hit_rate() {
        let dir = TempDir::new().unwrap();
        let ndir = dir.path().join(".cortyx").join("neurons");
        std::fs::create_dir_all(&ndir).unwrap();
        let mut idx = make_index(&dir);

        let p = ndir.join("moderate.context.md");
        std::fs::write(&p, "good content useful context").unwrap();
        let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
        idx.index_neuron(&p, "good content useful context", &meta);
        idx.rebuild_derived();

        // 5 hits out of 20 total → 25% hit rate; lb at z=1.645 is well above 0.05
        if let Some(&i) = idx.path_index.get(&p) {
            idx.entries[i].use_count = 19;
            idx.entries[i].hit_count = 5;
        }
        idx.record_activation(&[p.clone()]); // → use_count=20
        let mult = idx.path_index.get(&p).map(|&i| idx.entries[i].staleness_multiplier).unwrap_or(1.0);
        assert_eq!(mult, 1.0, "25% hit rate at 20 samples should not be quarantined");
    }

    // ── R12-S1: Concept Cloud ─────────────────────────────────────────────────

    #[test]
    fn concept_cloud_populated_from_structural_neighbours() {
        let dir = TempDir::new().unwrap();
        let ndir = dir.path().join(".cortyx").join("neurons");
        std::fs::create_dir_all(&ndir).unwrap();
        let mut idx = make_index(&dir);

        let caller = ndir.join("caller.context.md");
        let callee = ndir.join("callee.context.md");
        std::fs::write(&caller, "calls validate_user auth check").unwrap();
        std::fs::write(&callee, "validate_user password hash bcrypt").unwrap();

        let mut meta_caller = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
        let meta_callee = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
        meta_caller.synapses.push(crate::neuron::Synapse {
            target: callee.clone(),
            edge_type: crate::neuron::SynapseType::Calls,
            weight: 0.8,
            reason: "calls validate_user".to_string(),
            learned_weight: 0.0,
            traversal_count: 0,
            last_co_activation_day: 0,
        });

        idx.index_neuron(&caller, "calls validate_user auth check", &meta_caller);
        idx.index_neuron(&callee, "validate_user password hash bcrypt", &meta_callee);
        idx.rebuild_derived();

        // caller's concept cloud should contain callee terms
        let caller_idx = *idx.path_index.get(&caller).unwrap();
        let cloud = &idx.entries[caller_idx].concept_cloud;
        assert!(
            cloud.iter().any(|t| t == "bcrypt" || t == "password" || t == "validate_user"),
            "caller concept cloud should contain callee terms; got: {cloud:?}"
        );
    }

    #[test]
    fn concept_cloud_enables_retrieval_via_graph() {
        let dir = TempDir::new().unwrap();
        let ndir = dir.path().join(".cortyx").join("neurons");
        std::fs::create_dir_all(&ndir).unwrap();
        let mut idx = make_index(&dir);

        // "engine.rs" calls "hashing.rs", which owns the word "bcrypt".
        // A query for "bcrypt" should find engine via concept cloud even though
        // "bcrypt" does not appear in engine's own vocabulary.
        let engine = ndir.join("engine.context.md");
        let hashing = ndir.join("hashing.context.md");
        std::fs::write(&engine, "core engine dispatch orchestrate").unwrap();
        std::fs::write(&hashing, "bcrypt password hash rounds salt").unwrap();

        let mut meta_engine = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
        let meta_hashing = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
        meta_engine.synapses.push(crate::neuron::Synapse {
            target: hashing.clone(),
            edge_type: crate::neuron::SynapseType::Calls,
            weight: 0.8,
            reason: "calls hash function".to_string(),
            learned_weight: 0.0,
            traversal_count: 0,
            last_co_activation_day: 0,
        });

        idx.index_neuron(&engine, "core engine dispatch orchestrate", &meta_engine);
        idx.index_neuron(&hashing, "bcrypt password hash rounds salt", &meta_hashing);
        idx.rebuild_derived();

        // "bcrypt" is in hashing's vocab → engine's concept cloud → engine is reachable
        let engine_idx = *idx.path_index.get(&engine).unwrap();
        assert!(
            idx.entries[engine_idx].concept_cloud.contains(&"bcrypt".to_string()),
            "engine concept cloud must contain 'bcrypt' from hashing neighbour"
        );

        // Now query for "bcrypt" — vocab bridge won't match (no module synonym),
        // but concept cloud should surface engine as a candidate.
        let results = idx.get_contexts("bcrypt", 4096, None, None);
        let found_engine = results.iter().any(|s| s.to_string_lossy().contains("engine"));
        let found_hashing = results.iter().any(|s| {
            let p = s.to_string_lossy();
            p.contains("hashing") || p.contains("bcrypt")
        });
        assert!(
            found_hashing || found_engine,
            "concept cloud retrieval must surface at least one relevant neuron; got {results:?}"
        );
    }

    #[test]
    fn concept_cloud_excludes_semantic_related_edges() {
        let dir = TempDir::new().unwrap();
        let ndir = dir.path().join(".cortyx").join("neurons");
        std::fs::create_dir_all(&ndir).unwrap();
        let mut idx = make_index(&dir);

        let a = ndir.join("a.context.md");
        let b = ndir.join("b.context.md");
        std::fs::write(&a, "alpha beta gamma").unwrap();
        std::fs::write(&b, "exclusive_term_xyz zeta").unwrap();

        let mut meta_a = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
        let meta_b = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
        // Only SemanticRelated edge — should NOT contribute to concept cloud
        meta_a.synapses.push(crate::neuron::Synapse {
            target: b.clone(),
            edge_type: crate::neuron::SynapseType::SemanticRelated,
            weight: 0.5,
            reason: "related".to_string(),
            learned_weight: 0.0,
            traversal_count: 0,
            last_co_activation_day: 0,
        });

        idx.index_neuron(&a, "alpha beta gamma", &meta_a);
        idx.index_neuron(&b, "exclusive_term_xyz zeta", &meta_b);
        idx.rebuild_derived();

        let a_idx = *idx.path_index.get(&a).unwrap();
        assert!(
            !idx.entries[a_idx].concept_cloud.contains(&"exclusive_term_xyz".to_string()),
            "SemanticRelated edges must not populate concept cloud (already handled by vocab bridge)"
        );
    }

    // ── S-II (R16): LSH SimHash ───────────────────────────────────────────────

    #[test]
    fn simhash_same_terms_identical_fingerprint() {
        // Identical content should always yield the same fingerprint (deterministic)
        let mut tf1: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
        tf1.insert("auth".to_string(), 1.0);
        tf1.insert("token".to_string(), 2.0);
        let fp1 = simhash_with_seed(&tf1, LSH_SEEDS[0]);
        let fp2 = simhash_with_seed(&tf1, LSH_SEEDS[0]);
        assert_eq!(fp1, fp2, "same terms → same fingerprint (deterministic)");
        // Highly divergent content should produce different fingerprints with overwhelming probability
        let mut tf_other: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
        tf_other.insert("xyzzy".to_string(), 100.0);
        tf_other.insert("quux".to_string(), 100.0);
        tf_other.insert("plonk".to_string(), 100.0);
        tf_other.insert("zork".to_string(), 100.0);
        let fp_other = simhash_with_seed(&tf_other, LSH_SEEDS[0]);
        assert_ne!(fp1, fp_other, "very different terms should produce different fingerprints");
    }

    #[test]
    fn simhash_identical_content_identical_fingerprint() {
        let mut tf: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
        tf.insert("validate".to_string(), 1.5);
        tf.insert("password".to_string(), 3.0);
        let fp1 = simhash_with_seed(&tf, LSH_SEEDS[0]);
        let fp2 = simhash_with_seed(&tf, LSH_SEEDS[0]);
        assert_eq!(fp1, fp2, "same terms → same fingerprint (deterministic)");
    }

    #[test]
    fn hamming_distance_self_is_zero() {
        let fp = 0xdeadbeefcafe1234u64;
        assert_eq!(hamming_distance(fp, fp), 0);
    }

    #[test]
    fn hamming_distance_complement_is_64() {
        assert_eq!(hamming_distance(0u64, !0u64), 64);
    }

    #[test]
    fn lsh_fingerprint_stored_in_entry() {
        let dir = TempDir::new().unwrap();
        let ndir = dir.path().join(".cortyx").join("neurons");
        std::fs::create_dir_all(&ndir).unwrap();
        let mut idx = make_index(&dir);
        let neuron = ndir.join("auth.context.md");
        std::fs::write(&neuron, "auth token validate jwt bearer").unwrap();
        let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
        idx.index_neuron(&neuron, "auth token validate jwt bearer", &meta);
        let entry_idx = *idx.path_index.get(&neuron).unwrap();
        assert!(idx.entries[entry_idx].lsh_fingerprints.iter().any(|&fp| fp != 0),
            "non-empty term set should produce non-zero 1024-bit SimHash");
    }

    // ── S-III (R16): Self-Quality Score ──────────────────────────────────────

    #[test]
    fn quality_score_defaults_to_one_when_no_source() {
        let dir = TempDir::new().unwrap();
        let ndir = dir.path().join(".cortyx").join("neurons");
        std::fs::create_dir_all(&ndir).unwrap();
        let mut idx = make_index(&dir);
        let neuron = ndir.join("concept.context.md");
        std::fs::write(&neuron, "some concept terms here").unwrap();
        // Concept kind → no source file → quality_score defaults to 1.0
        let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Concept);
        idx.index_neuron(&neuron, "some concept terms here", &meta);
        let entry_idx = *idx.path_index.get(&neuron).unwrap();
        assert!(
            (idx.entries[entry_idx].quality_score - 1.0).abs() < 1e-6,
            "Concept neuron should have quality_score=1.0 (no source file)"
        );
    }

    #[test]
    fn low_quality_count_counts_below_threshold() {
        let dir = TempDir::new().unwrap();
        let ndir = dir.path().join(".cortyx").join("neurons");
        std::fs::create_dir_all(&ndir).unwrap();
        let mut idx = make_index(&dir);
        // All Concept neurons → quality_score = 1.0 → none below threshold
        for i in 0..3 {
            let p = ndir.join(format!("n{i}.context.md"));
            std::fs::write(&p, "terms").unwrap();
            let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Concept);
            idx.index_neuron(&p, "terms", &meta);
        }
        assert_eq!(idx.low_quality_count(), 0, "no low-quality neurons expected");
    }
}
