#[cfg(feature = "embed")]
use crate::embedder::{load_embeddings, EmbeddingStore};

use anyhow::Result;
use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use walkdir::WalkDir;

use crate::alias_gen;
use crate::ast_extractor;
use crate::git_extractor;
use crate::global_index;
use crate::import_parser;
use crate::kg;
use crate::neuron::{
    atomic_write, atomic_write_json, core_neuron_path, estimate_context_tokens, estimate_tokens,
    meta_path, neuron_dir, now_iso8601, replace_section, should_skip, stub_core_neuron,
    stub_function_neuron, stub_project_neuron, sub_neuron_path, update_neuron_header, NeuronKind,
    NeuronMeta, NeuronStatus, Synapse, SynapseType, DEFAULT_CONFIDENCE,
};
use crate::reasoner::{
    GraphReasoner, ReasonerNeuron, ReasonerSeed, ReasoningReport, TraversalOptions,
};

mod config;
use config::*;
pub use config::{
    HIGH_ACTIVATION_THRESHOLD, MAX_CORE_NEURONS, MAX_USE_CASE_PER_CORE, SYNAPSE_RELEVANCE_THRESHOLD,
};

// Answer surface types extracted to answer_surface/types.rs.
mod answer_surface;
use answer_surface::*;

// LSH/SimHash extracted to lsh/mod.rs.
mod lsh;
use lsh::{hamming_distance, simhash_1024, simhash_with_seed, LSH_SEEDS};

// Query detection, personal-fact helpers, and git confidence utilities extracted to query/mod.rs.
mod query;
pub(super) use query::{
    adaptive_quarantine_params, build_git_confidence_map, content_has_move_residence_evidence,
    count_proper_nouns, detect_knowledge_update_query, detect_personal_fact_entity,
    detect_personal_fact_query, extract_knowledge_update_focus_terms, extract_numbered_list_item,
    extract_pet_name, extract_query_ordinal, extract_single_word_after_marker, git_file_list,
    is_book_query, is_commute_query, is_education_query, is_fitness_record_query,
    is_list_style_query, is_location_query, is_major_query, is_named_move_query,
    is_occupation_query, is_partner_query, is_pet_query, is_phone_query, is_project_name_query,
    num_to_word, parse_iso8601_to_secs, synthetic_query_terms, task_contains_all,
    task_contains_any, term_overlap_count, wilson_lower_bound_z,
};
// Test-only query helpers (cfg-gated; accessed via query:: in test module).
#[cfg(test)]
use query::{neuron_body_has_move_residence_evidence, wilson_lower_bound};

// ─── BM25 entry ───────────────────────────────────────────────────────────────

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub(super) struct BM25Entry {
    pub(super) neuron_path: PathBuf,
    pub(super) kind: NeuronKind,
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
    /// R17 Sol4 upgrade: 1024-bit SimHash ensemble via 16 independent seeds.
    /// This is an empirical locality-sensitive fallback, not a Johnson-Lindenstrauss guarantee.
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
    /// Stored at index time so named-person relocation queries do not read neuron files
    /// from disk in the retrieval hot path.
    #[serde(default)]
    has_move_residence_evidence: bool,
    /// R21 T6: Session identifier for session-level grouping.
    ///
    /// Derived from the neuron filename stem (e.g., "lme_0060" from
    /// "lme_0060_0_user.verbatim.md"). Empty for non-Verbatim neurons.
    /// Used at retrieval time: when a neuron enters the top-3, its session
    /// siblings are injected as overflow candidates.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub(super) session_id: String,
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

/// Share-ready neuron summary for the git-federated concept library.
#[derive(Debug, Clone)]
pub struct PublishReadySummary {
    pub path: PathBuf,
    pub kind: NeuronKind,
    pub use_count: u32,
    pub hit_rate: f32,
    pub quality_score: f32,
}

/// Lightweight metadata for explainable answer/provenance rendering.
#[derive(Debug, Clone)]
pub struct ContextMetadata {
    pub kind: NeuronKind,
    pub module: Option<String>,
    pub summary: String,
    pub timestamp_secs: Option<i64>,
    pub tokens: usize,
    pub use_count: u32,
    pub hit_count: u32,
    pub hit_rate: f32,
}

// ─── Persisted index wrapper ───────────────────────────────────────────────────

/// Borrowed view used for serialization — avoids cloning the entire entry vector
/// on every save() call (which would otherwise be O(n) allocation per MCP mutation).
#[derive(Serialize)]
pub(super) struct PersistedIndexRef<'a> {
    version: u32,
    cache_generation: u64,
    entries: &'a [BM25Entry],
    #[serde(skip_serializing_if = "<[[usize; 2]]>::is_empty")]
    session_utilization: &'a [[usize; 2]],
    #[serde(skip_serializing_if = "<[String]>::is_empty")]
    shards: &'a [String],
}

/// Binary activation cache persisted alongside index.json.
///
/// TRIZ P10 (Preliminary Action): precompute and persist the query-hot derived
/// structures at save time, so CLI startup does not have to rebuild them on
/// every `status` / `get-contexts` invocation.
#[derive(Serialize, Deserialize)]
pub(super) struct PersistedActivationCache {
    version: u32,
    index_generation: u64,
    entries: Vec<BM25Entry>,
    concept_clouds: Vec<Vec<String>>,
    summaries: Vec<String>,
    adjacency: HashMap<PathBuf, Vec<Synapse>>,
    path_index: HashMap<PathBuf, usize>,
    parent_index: HashMap<PathBuf, Vec<usize>>,
    df_cache: HashMap<String, usize>,
    posting_list: HashMap<String, Vec<usize>>,
    avg_doc_len: f32,
    avg_verbatim_doc_len: f32,
    module_index: HashMap<String, Vec<usize>>,
    vocab_bridge: HashMap<String, HashSet<String>>,
    morpheme_map: HashMap<String, Vec<String>>,
    session_utilization: Vec<[usize; 2]>,
    session_index: HashMap<String, Vec<usize>>,
    pmi_neighbors: HashMap<String, Vec<String>>,
    idf_n: usize,
    /// S4-WAL: entries.len() at last full index.json write.
    wal_base: usize,
}

#[derive(Serialize)]
pub(super) struct PersistedActivationCacheRef<'a> {
    version: u32,
    index_generation: u64,
    entries: &'a [BM25Entry],
    concept_clouds: Vec<&'a [String]>,
    summaries: Vec<&'a str>,
    adjacency: &'a HashMap<PathBuf, Vec<Synapse>>,
    path_index: &'a HashMap<PathBuf, usize>,
    parent_index: &'a HashMap<PathBuf, Vec<usize>>,
    df_cache: &'a HashMap<String, usize>,
    posting_list: &'a HashMap<String, Vec<usize>>,
    avg_doc_len: f32,
    avg_verbatim_doc_len: f32,
    module_index: &'a HashMap<String, Vec<usize>>,
    vocab_bridge: &'a HashMap<String, HashSet<String>>,
    morpheme_map: &'a HashMap<String, Vec<String>>,
    session_utilization: &'a Vec<[usize; 2]>,
    session_index: &'a HashMap<String, Vec<usize>>,
    pmi_neighbors: &'a HashMap<String, Vec<String>>,
    idf_n: usize,
    /// S4-WAL: entries.len() at last full index.json write.
    wal_base: usize,
}

// ─── Schema migrations ────────────────────────────────────────────────────────

/// Migration function signature: transforms a stored JSON value to be compatible
/// with the next schema version. Fields not touched by the migration are preserved
/// (use_count, hit_count, staleness_multiplier etc. survive every upgrade).
pub(super) type MigrationFn = fn(serde_json::Value) -> serde_json::Value;

/// v7 → v8: rename `lsh_fingerprint: u64` → `lsh_fingerprints: [u64; 16]` (1024-bit LSH).
pub(super) fn migrate_v7_to_v8(mut entries_val: serde_json::Value) -> serde_json::Value {
    if let Some(arr) = entries_val.as_array_mut() {
        for entry in arr.iter_mut() {
            let old_fp = entry
                .get("lsh_fingerprint")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let fps: Vec<serde_json::Value> =
                std::iter::once(serde_json::Value::Number(old_fp.into()))
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
pub(super) fn migrate_entries(
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
        let oldest_supported = MIGRATIONS
            .first()
            .map(|(from, _, _)| *from)
            .unwrap_or(INDEX_VERSION);
        if stored_version < oldest_supported {
            anyhow::bail!(
                "Index version {stored_version} predates supported migrations (oldest supported: v{oldest_supported}). \
                 Curated neuron markdown remains on disk; run `cortyx compile .` to rebuild only the search index."
            );
        }
        anyhow::bail!(
            "No migration path from version {stored_version} to {INDEX_VERSION}; \
             run `cortyx compile .` to rebuild."
        );
    }
    let entries: Vec<BM25Entry> = serde_json::from_value(
        raw.get("entries")
            .cloned()
            .unwrap_or(serde_json::Value::Array(vec![])),
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
    pub(super) project_root: PathBuf,
    pub(super) entries: Vec<BM25Entry>,
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
    /// Average Verbatim-neuron document length (for BM25 length normalization of conversation chunks).
    /// Computed separately from avg_doc_len to avoid Concept/entity neurons (very short, ~150 tokens)
    /// artificially depressing the average and over-penalizing long session chunks.
    avg_verbatim_doc_len: f32,
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
    /// Persisted in `.cortyx/coactivation.json` so synonym-cloud promotion can survive
    /// normal CLI/server restarts. Synonym clouds in BM25Entry are still persisted directly.
    coactivation_counts: HashMap<PathBuf, HashMap<String, u32>>,
    /// C-2: Hebbian synapse co-return counts (R20).
    ///
    /// Tracks how often two Verbatim neurons are returned together in the same query.
    /// Map: (path_a, path_b) → co-return count (path_a < path_b lexicographically).
    /// Co-return co-occurrence counter for Hebbian synapse formation.
    ///
    /// Tracks how often pairs of neurons appear together in query results.
    /// After ≥10 co-returns, a SemanticRelated synapse is auto-created between the pair.
    /// Not persisted — rebuilt from query patterns in the current process lifetime.
    ///
    /// # Mutex discipline
    /// This uses `std::sync::Mutex` (not `tokio::sync::Mutex`) intentionally.
    /// Callers **must not** hold this lock across any `.await` point. All current
    /// call sites release the lock before yielding. If you add a new call site,
    /// enforce this invariant to avoid blocking the async thread pool.
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
    pub(super) session_index: HashMap<String, Vec<usize>>,
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
    /// IDF corpus size: count of non-Aggregate entries used for BM25 IDF computation.
    ///
    /// Aggregate neurons (word-count summaries, arithmetic totals) must NOT pollute
    /// IDF: a _count_music.aggregate.md neuron that mentions "music" dozens of times
    /// inflates df("music"), crushing its IDF and causing recall failures on SSU
    /// queries where "music" is the only discriminative signal.  By excluding
    /// Aggregate neurons from df_cache AND using idf_n (not entries.len()) as the
    /// corpus size N in the BM25 IDF formula, we preserve the IDF calibration that
    /// produced 100% SSU at the e18c4e6 baseline.  Posting-list entries are still
    /// added for ALL neuron kinds so the counting_augment path can find Aggregates.
    idf_n: usize,
    /// Count of entries inserted (not updated) since the last rebuild_derived() call.
    /// Used to take the fast incremental delta path in rebuild_derived() instead of
    /// clearing and rebuilding all HashMaps from scratch.
    /// Not persisted — resets to 0 via Default on every load (full rebuild runs then).
    pending_append_count: usize,
    /// True if any existing entry was updated (not just appended) since the last
    /// rebuild_derived() call.  When true the full rebuild path is taken so that
    /// df_cache / posting_list stay consistent with the changed entries.
    has_pending_updates: bool,
    /// S4-WAL: entries.len() at last full index.json write.
    /// Persisted in the activation cache so WAL mode activates on subsequent process starts.
    /// 0 means no full save yet; the next save() establishes the WAL baseline.
    wal_base: AtomicUsize,
    /// S4-WAL: true when any existing entry was updated since the last full save.
    /// Forces a full index.json rewrite so in-place mutations are never lost.
    needs_full_save: AtomicBool,
    /// Set when structural derived state changes and the module shards / cache generation
    /// should be refreshed on the next save(). Feedback-only saves leave this false.
    structural_artifacts_dirty: AtomicBool,
}

// ─── Parallel compile helper ──────────────────────────────────────────────────

/// Result of processing a single source file in the parallel compile phase.
///
/// Returned by `process_source_file` (a free function — no `&self` access) so
/// multiple files can be processed concurrently via `rayon::par_iter()`.
/// The sequential batch-insert phase calls `index_neuron` on each result.
pub(super) struct CompiledFile {
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
pub(super) fn process_source_file(
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
                tracing::warn!(
                    "Failed to update meta for cosmetic change {:?}: {e}",
                    meta_file
                );
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
                    let old = stored_meta
                        .clone()
                        .unwrap_or_else(|| NeuronMeta::new_stub(abs, NeuronKind::Core));
                    let mut meta = old;
                    meta.source_hash = current_hash;
                    meta.sig_hash = Some(sig_hash);
                    meta.last_updated = now.clone();
                    meta.status = NeuronStatus::Stale;
                    meta.tokens = estimate_context_tokens(&updated);
                    if meta.module.is_none() {
                        meta.module = infer_module(rel);
                    }
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
                            if sub_path.exists() {
                                continue;
                            }
                            let sub_content = stub_function_neuron(fn_name, &source_rel, &now);
                            if let Err(e) = atomic_write(&sub_path, sub_content.as_bytes()) {
                                tracing::warn!(
                                    "S1: Failed to write sub-neuron {:?}: {e}",
                                    sub_path
                                );
                                continue;
                            }
                            let sub_meta_file = meta_path(&sub_path);
                            let mut sub_meta = NeuronMeta::new_stub(abs, NeuronKind::UseCase);
                            sub_meta.task_pattern = Some(fn_name.clone());
                            sub_meta.parent = Some(neuron_path.clone());
                            sub_meta.tokens = estimate_context_tokens(&sub_content);
                            sub_meta.last_updated = now.clone();
                            sub_meta.module = results[0].meta.module.clone();
                            sub_meta.confidence_score = results[0].meta.confidence_score;
                            if let Err(e) = atomic_write_json(&sub_meta_file, &sub_meta) {
                                tracing::warn!(
                                    "S1: Failed to write sub-neuron meta {:?}: {e}",
                                    sub_meta_file
                                );
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
            },
            Err(_) => {
                // Cannot read existing neuron — fall through to full stub regeneration
            },
        }
    }

    // Full stub (re)generation — real API change (sig_hash changed) or new file.
    let prefilled = ast_extractor::format_for_stub(&ast_summary);
    let purpose_hint = ast_extractor::format_purpose_hint(&ast_summary);
    let extra_vocab = ast_extractor::format_extra_vocab_for_stub(&ast_summary);
    let content = stub_core_neuron(
        &source_rel,
        &current_hash,
        &now,
        &prefilled,
        &purpose_hint,
        &extra_vocab,
    );

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
    meta.tokens = estimate_context_tokens(&content);
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

    let mut results = vec![CompiledFile {
        neuron_path: neuron_path.clone(),
        content,
        meta,
    }];

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
            sub_meta.tokens = estimate_context_tokens(&sub_content);
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

        if let Some(mut idx) = Self::try_load_activation_cache(project_root, &path) {
            idx.coactivation_counts = load_coactivation_counts(project_root);
            return Ok(idx);
        }

        let mut idx = NeuronIndex {
            project_root: project_root.to_path_buf(),
            #[cfg(feature = "embed")]
            embeddings: load_embeddings(project_root),
            ..Default::default()
        };
        let mut activation_generation = 0u64;
        let mut persist_index = false;

        if path.exists() {
            match std::fs::read_to_string(&path) {
                Ok(data) => match serde_json::from_str::<serde_json::Value>(&data) {
                    Ok(raw) => {
                        let stored_version = raw
                            .get("version")
                            .and_then(|v| v.as_u64())
                            .map(|v| v as u32)
                            .unwrap_or(0);
                        activation_generation = raw
                            .get("cache_generation")
                            .and_then(|v| v.as_u64())
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
                            if activation_generation == 0 {
                                persist_index = true;
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
                                    persist_index = true;
                                },
                                Err(e) => {
                                    tracing::warn!(
                                        "Migration from v{stored_version} failed ({e}): \
                                         starting fresh. Run `cortyx compile .` to rebuild."
                                    );
                                },
                            }
                        } else {
                            tracing::warn!(
                                "Index version is newer than binary (stored={stored_version}, \
                                 current={INDEX_VERSION}): starting fresh."
                            );
                        }
                    },
                    Err(e) => {
                        tracing::warn!(
                            "Failed to parse index.json (corrupted?): {e}. \
                             Starting with empty index — run `cortyx compile .` to rebuild."
                        );
                        eprintln!(
                            "⚠ Cortyx: index.json is corrupted ({e}). Run `cortyx compile .`"
                        );
                    },
                },
                Err(e) => {
                    tracing::warn!("Failed to read index.json: {e}. Starting with empty index.");
                },
            }
        }

        // S4-WAL: replay any delta entries written by WAL-mode saves since the last
        // full index.json write.  Only applied when the base_count matches the number
        // of entries just loaded from index.json, preventing double-counting on
        // crash-between-write scenarios.
        let cortyx_dir = path.parent().unwrap_or_else(|| Path::new("."));
        let delta_path = cortyx_dir.join("index.delta.json");
        if let Ok(data) = std::fs::read_to_string(&delta_path) {
            if let Ok(raw) = serde_json::from_str::<serde_json::Value>(&data) {
                let base_count =
                    raw.get("base_count").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                if base_count == idx.entries.len() {
                    if let Ok(delta_entries) = serde_json::from_value::<Vec<BM25Entry>>(
                        raw.get("entries").cloned().unwrap_or_default(),
                    ) {
                        tracing::debug!(n = delta_entries.len(), "Replaying WAL delta entries");
                        idx.entries.extend(delta_entries);
                    }
                } else {
                    tracing::debug!(
                        base_count,
                        loaded = idx.entries.len(),
                        "WAL delta base_count mismatch — skipping stale delta"
                    );
                }
            }
        }

        idx.rebuild_derived();
        idx.coactivation_counts = load_coactivation_counts(project_root);
        if persist_index {
            if let Err(e) = idx.save() {
                tracing::warn!("Failed to persist upgraded index metadata: {e}");
            }
        } else if activation_generation > 0 {
            let index_bytes = std::fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
            if let Err(e) = idx.save_activation_cache(activation_generation, index_bytes) {
                tracing::warn!("Failed to refresh activation cache after rebuild: {e}");
            }
            idx.structural_artifacts_dirty
                .store(false, Ordering::Relaxed);
        }
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
    /// can fast-load specific modules without reading the full file. Stable
    /// module capsules are also regenerated here so `cortyx_get_contexts` can
    /// serve cache-friendly subsystem summaries without runtime synthesis.
    pub fn save(&self) -> Result<()> {
        let path = index_path(&self.project_root);
        let cortyx_dir = path.parent().unwrap_or(Path::new("."));
        std::fs::create_dir_all(cortyx_dir)?;
        let structural_dirty = self.structural_artifacts_dirty.load(Ordering::Relaxed);
        let prior_generation = read_index_cache_generation(&path).unwrap_or(0);

        // S4-WAL: determine whether this save can use the delta (WAL) path.
        // WAL mode skips rewriting the monolithic index.json and instead appends
        // only new entries to a small delta file, reducing serialisation work from
        // O(N+n) to O(n) for pure-append mine batches.
        let delta_path = cortyx_dir.join("index.delta.json");
        let wal_base = self.wal_base.load(Ordering::Relaxed);
        let delta_len = self.entries.len().saturating_sub(wal_base);
        // Compact to a full write when delta exceeds 25% of the base — keeps the
        // fallback replay path fast and prevents unbounded delta file growth.
        let over_threshold = wal_base > 0 && delta_len > wal_base / 4;
        let in_wal_mode = wal_base > 0
            && !self.needs_full_save.load(Ordering::Relaxed)
            && !over_threshold
            && delta_len > 0;

        // In WAL mode the generation must stay unchanged so the activation cache passes
        // the index_generation check against the (unchanged) index.json on the next load.
        let cache_generation = if structural_dirty && !in_wal_mode {
            prior_generation.saturating_add(1)
        } else {
            prior_generation
        };

        // S-VI: group entries by module for shard files and stable module capsules.
        // Prefer the in-memory entry.module tag; fall back to the sidecar for
        // older persisted indices that may be missing the field.
        let mut modules: std::collections::HashMap<String, Vec<&BM25Entry>> =
            std::collections::HashMap::new();
        let mut path_modules: std::collections::HashMap<PathBuf, String> =
            std::collections::HashMap::new();
        for entry in &self.entries {
            let module_name = entry
                .module
                .clone()
                .or_else(|| sidecar_module_for(&entry.neuron_path))
                .unwrap_or_else(|| "__global".to_string());
            path_modules.insert(entry.neuron_path.clone(), module_name.clone());
            modules.entry(module_name).or_default().push(entry);
        }

        let mut module_names: Vec<&String> = modules.keys().collect();
        module_names.sort();
        let shard_names: Vec<String> = module_names
            .iter()
            .map(|module| safe_module_name(module))
            .collect();

        if structural_dirty {
            for module in &module_names {
                let safe_name = safe_module_name(module);
                let shard_path = cortyx_dir.join(format!("index.{safe_name}.json"));
                let shard = serde_json::json!({
                    "version": INDEX_VERSION,
                    "module": module,
                    "entries": modules[*module],
                });
                if let Err(e) = atomic_write_json(&shard_path, &shard) {
                    tracing::warn!("S-VI: could not write shard for module '{module}': {e}");
                }
            }

            if let Err(e) = self.write_module_capsules(cortyx_dir, &modules, &path_modules) {
                tracing::warn!("Failed to refresh module capsules: {e}");
            }
        }

        // Write monolithic index.json (backward compat) with shard registry embedded,
        // or in WAL mode write only the delta entries to a small delta file.
        if in_wal_mode {
            // WAL mode: write only entries[wal_base..] to the delta file.
            let delta = serde_json::json!({
                "base_count": wal_base,
                "entries": &self.entries[wal_base..],
            });
            atomic_write_json(&delta_path, &delta)?;
            // Pass index_bytes=0 to bypass the size guard — the cache legitimately
            // contains more entries than the (unchanged) index.json.
            if let Err(e) = self.save_activation_cache(cache_generation, 0) {
                tracing::warn!("Failed to write activation cache (WAL mode): {e}");
            }
        } else {
            // Full save: rewrite index.json, clear any stale delta, update WAL baseline.
            let persisted = PersistedIndexRef {
                version: INDEX_VERSION,
                cache_generation,
                entries: &self.entries,
                session_utilization: &self.session_utilization,
                shards: &shard_names,
            };
            atomic_write_json(&path, &persisted)?;
            let _ = std::fs::remove_file(&delta_path); // clear stale delta on full write
            self.wal_base.store(self.entries.len(), Ordering::Relaxed);
            self.needs_full_save.store(false, Ordering::Relaxed);
            let index_bytes = std::fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
            if let Err(e) = self.save_activation_cache(cache_generation, index_bytes) {
                tracing::warn!("Failed to write activation cache: {e}");
            }
        }
        if let Err(e) = save_coactivation_counts(&self.project_root, &self.coactivation_counts) {
            tracing::warn!("Failed to write coactivation counts: {e}");
        }
        if structural_dirty {
            self.structural_artifacts_dirty
                .store(false, Ordering::Relaxed);
        }
        Ok(())
    }

    fn write_module_capsules(
        &self,
        cortyx_dir: &Path,
        modules: &HashMap<String, Vec<&BM25Entry>>,
        path_modules: &HashMap<PathBuf, String>,
    ) -> Result<()> {
        let capsule_dir = cortyx_dir.join("capsules");
        std::fs::create_dir_all(&capsule_dir)?;

        let mut live_capsules = HashSet::new();
        let mut module_names: Vec<&String> = modules.keys().collect();
        module_names.sort();

        for module in module_names {
            let Some(content) =
                build_module_capsule_content(module, &modules[module], path_modules)
            else {
                continue;
            };

            let safe_name = safe_module_name(module);
            let capsule_path = capsule_dir.join(format!("{safe_name}.capsule.md"));
            if let Err(e) = atomic_write(&capsule_path, content.as_bytes()) {
                tracing::warn!("Failed to write module capsule for '{module}': {e}");
            } else {
                live_capsules.insert(safe_name);
            }
        }

        if let Ok(entries) = std::fs::read_dir(&capsule_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                    continue;
                };
                if !name.ends_with(".capsule.md") {
                    continue;
                }
                let stem = name.trim_end_matches(".capsule.md");
                if live_capsules.contains(stem) {
                    continue;
                }
                if let Err(e) = std::fs::remove_file(&path) {
                    tracing::warn!(
                        "Failed to remove stale module capsule {}: {e}",
                        path.display()
                    );
                }
            }
        }

        Ok(())
    }

    fn try_load_activation_cache(project_root: &Path, index_path: &Path) -> Option<Self> {
        let index_generation = read_index_cache_generation(index_path)?;
        let cache_path = activation_cache_path(project_root);
        let index_bytes = std::fs::metadata(index_path).ok()?.len();
        let cache_bytes = std::fs::metadata(&cache_path).ok()?.len();
        // S4-WAL: allow the cache to be larger than index.json when a delta file exists.
        // In WAL mode the cache contains entries not yet in the full index.json.
        let cortyx_dir = index_path.parent().unwrap_or_else(|| Path::new("."));
        let has_delta = cortyx_dir.join("index.delta.json").exists();
        if !has_delta && cache_bytes > index_bytes {
            tracing::debug!(
                cache = %cache_path.display(),
                index_bytes,
                cache_bytes,
                "Skipping activation cache because it is larger than index.json"
            );
            return None;
        }
        let bytes = std::fs::read(&cache_path).ok()?;
        let cache: PersistedActivationCache = bincode::deserialize(&bytes).ok()?;
        if cache.version != INDEX_VERSION || cache.index_generation != index_generation {
            return None;
        }
        if cache.concept_clouds.len() != cache.entries.len()
            || cache.summaries.len() != cache.entries.len()
        {
            tracing::warn!(
                "Activation cache shape mismatch (entries={}, clouds={}, summaries={}) — rebuilding.",
                cache.entries.len(),
                cache.concept_clouds.len(),
                cache.summaries.len()
            );
            return None;
        }

        let mut entries = cache.entries;
        for (entry, cloud) in entries.iter_mut().zip(cache.concept_clouds) {
            entry.concept_cloud = cloud;
        }
        for (entry, summary) in entries.iter_mut().zip(cache.summaries) {
            entry.summary = summary;
        }

        tracing::debug!(
            entries = entries.len(),
            cache = %cache_path.display(),
            "Loaded activation cache"
        );

        Some(NeuronIndex {
            project_root: project_root.to_path_buf(),
            entries,
            adjacency: cache.adjacency,
            path_index: cache.path_index,
            parent_index: cache.parent_index,
            df_cache: cache.df_cache,
            posting_list: cache.posting_list,
            avg_doc_len: cache.avg_doc_len,
            avg_verbatim_doc_len: cache.avg_verbatim_doc_len,
            module_index: cache.module_index,
            vocab_bridge: cache.vocab_bridge,
            morpheme_map: cache.morpheme_map,
            session_utilization: cache.session_utilization,
            session_index: cache.session_index,
            pmi_neighbors: cache.pmi_neighbors,
            idf_n: cache.idf_n,
            wal_base: AtomicUsize::new(cache.wal_base),
            #[cfg(feature = "embed")]
            embeddings: load_embeddings(project_root),
            ..Default::default()
        })
    }

    fn save_activation_cache(&self, index_generation: u64, index_bytes: u64) -> Result<()> {
        let cache = PersistedActivationCacheRef {
            version: INDEX_VERSION,
            index_generation,
            entries: &self.entries,
            concept_clouds: self
                .entries
                .iter()
                .map(|entry| entry.concept_cloud.as_slice())
                .collect(),
            summaries: self
                .entries
                .iter()
                .map(|entry| entry.summary.as_str())
                .collect(),
            adjacency: &self.adjacency,
            path_index: &self.path_index,
            parent_index: &self.parent_index,
            df_cache: &self.df_cache,
            posting_list: &self.posting_list,
            avg_doc_len: self.avg_doc_len,
            avg_verbatim_doc_len: self.avg_verbatim_doc_len,
            module_index: &self.module_index,
            vocab_bridge: &self.vocab_bridge,
            morpheme_map: &self.morpheme_map,
            session_utilization: &self.session_utilization,
            session_index: &self.session_index,
            pmi_neighbors: &self.pmi_neighbors,
            idf_n: self.idf_n,
            wal_base: self.wal_base.load(Ordering::Relaxed),
        };
        let bytes = bincode::serialize(&cache)?;
        let cache_path = activation_cache_path(&self.project_root);
        if index_bytes > 0 && bytes.len() as u64 > index_bytes {
            if let Err(err) = std::fs::remove_file(&cache_path) {
                if err.kind() != std::io::ErrorKind::NotFound {
                    return Err(err.into());
                }
            }
            tracing::debug!(
                cache = %cache_path.display(),
                index_bytes,
                cache_bytes = bytes.len(),
                "Skipping activation cache write because it is larger than index.json"
            );
            return Ok(());
        }
        atomic_write(&cache_path, &bytes)
    }

    fn index_compiled_files(
        &mut self,
        compiled: Vec<CompiledFile>,
        cascade_core_staleness: bool,
    ) -> usize {
        let new_count = compiled.len();
        for cf in compiled {
            let should_cascade = cascade_core_staleness && matches!(cf.meta.kind, NeuronKind::Core);
            let neuron_path = cf.neuron_path.clone();
            self.index_neuron(&cf.neuron_path, &cf.content, &cf.meta);
            if should_cascade {
                self.cascade_staleness(&neuron_path);
            }
        }
        new_count
    }

    fn finalize_compile_pass(&mut self, root: &Path) -> Result<()> {
        self.apply_call_graph_synapses(root);
        self.apply_cochange_synapses(root);
        self.apply_rename_detection(root);
        self.rebuild_derived();
        self.save()
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
        let new_count = self.index_compiled_files(compiled, false);
        self.finalize_compile_pass(&root)?;
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
            if let Err(e) = std::fs::remove_file(&dirty_file) {
                if e.kind() != std::io::ErrorKind::NotFound {
                    tracing::warn!("Failed to clear empty dirty.json: {e}");
                }
            }
            return Ok(0);
        }

        tracing::info!(
            "Incremental compile: processing {} dirty file(s).",
            dirty_paths.len()
        );

        let root = self.project_root.clone();
        let git_confidence = build_git_confidence_map(&root);
        let compiled: Vec<CompiledFile> = dirty_paths
            .par_iter()
            .flat_map(|abs| process_source_file(abs, &root, &git_confidence))
            .collect();

        let new_count = self.index_compiled_files(compiled, true);
        self.finalize_compile_pass(&root)?;
        if let Err(e) = std::fs::remove_file(&dirty_file) {
            if e.kind() != std::io::ErrorKind::NotFound {
                tracing::warn!("Failed to clear dirty.json after compile_dirty: {e}");
            }
        }
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
    pub fn upsert_neuron(
        &mut self,
        neuron_path: &Path,
        content: &str,
        meta: &NeuronMeta,
    ) -> Result<()> {
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
    pub fn get_contexts(
        &self,
        task: &str,
        max_tokens: usize,
        module: Option<&str>,
        kind: Option<&str>,
    ) -> Vec<PathBuf> {
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
        let morphological_expansions: Vec<String> = terms
            .iter()
            .flat_map(|term| morphological_variants(term))
            .filter(|variant| self.df_cache.contains_key(variant.as_str()))
            .collect();
        let terms_with_synonyms: Vec<String> =
            if !synonym_expansions.is_empty() || !morphological_expansions.is_empty() {
                let mut t = terms.clone();
                t.extend(synonym_expansions.iter().cloned());
                t.extend(morphological_expansions.iter().cloned());
                t.sort();
                t.dedup();
                t
            } else {
                terms.clone()
            };

        // Expand candidate set with synonym/morphological terms if we have them
        let candidate_set = {
            let mut cs = candidate_set;
            for term in synonym_expansions
                .iter()
                .chain(morphological_expansions.iter())
            {
                if let Some(idxs) = self.posting_list.get(term.as_str()) {
                    cs.extend(idxs);
                }
            }
            cs
        };

        let synonym_expansions_empty =
            synonym_expansions.is_empty() && morphological_expansions.is_empty();

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
        let task_lower = task.to_ascii_lowercase();
        let explicit_current_state_query = has_explicit_current_state_marker(task);
        let named_person_move_query = count_proper_nouns(task) >= 1
            && (task_lower.contains(" move")
                || task_lower.contains(" moved")
                || task_lower.contains("relocation"));
        let expand_focus_terms = |base_terms: Vec<String>| {
            let mut expanded = base_terms.clone();
            for term in &base_terms {
                for variant in morphological_variants(term) {
                    if self.df_cache.contains_key(variant.as_str()) {
                        expanded.push(variant);
                    }
                }
            }
            expanded.sort();
            expanded.dedup();
            expanded
        };
        let raw_counting_focus_terms = if is_counting {
            extract_counting_focus_terms(&terms)
        } else {
            Vec::new()
        };
        let counting_focus_terms = if is_counting {
            expand_focus_terms(raw_counting_focus_terms.clone())
        } else {
            Vec::new()
        };
        let raw_knowledge_focus_terms = if !is_counting && is_knowledge_update {
            extract_knowledge_update_focus_terms(&terms)
        } else {
            Vec::new()
        };
        let knowledge_focus_terms = if !is_counting && is_knowledge_update {
            expand_focus_terms(raw_knowledge_focus_terms.clone())
        } else {
            Vec::new()
        };
        let ranking_terms: &[String] = if !counting_focus_terms.is_empty() {
            &counting_focus_terms
        } else if !knowledge_focus_terms.is_empty() {
            &knowledge_focus_terms
        } else {
            scoring_terms
        };
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
        let kg_router_path: Option<PathBuf> =
            (!matches!(kind, Some(k) if k.eq_ignore_ascii_case("conversation")))
                .then_some(())
                .and_then(|_| detect_personal_fact_query(task))
                .and_then(|predicate| {
                    detect_personal_fact_entity(task).and_then(|entity| {
                        let kg_path = kg::kg_neuron_path(&self.project_root, &entity);
                        if !self.path_index.contains_key(&kg_path) {
                            return None;
                        }
                        let Ok(kg_entity) = kg::KgEntity::load(&kg_path) else {
                            return None;
                        };
                        let has_fact = kg_entity
                            .active_facts(None)
                            .iter()
                            .any(|f| f.predicate == predicate && !f.value.is_empty());
                        if has_fact {
                            tracing::debug!(
                            task,
                            predicate,
                            entity,
                            kind = kind.unwrap_or("all"),
                            "P2-B KG Router: routed personal-attribute query to exact KG neuron"
                        );
                            Some(kg_path)
                        } else {
                            None
                        }
                    })
                });

        // R21 T5: Counting-query candidate expansion.
        //
        // "How many X have I done?" needs evidence from ALL sessions mentioning X, not
        // just the highest-scoring posting-list hit. When detect_counting_query fires,
        // expand the candidate set to include ALL Verbatim neurons in the index, scored
        // with BM25 against the query. Aggregate neurons stay available for explicit
        // injection below, but they do not participate in the general BM25 pool.
        let counting_augment: Vec<usize> = if is_counting {
            let in_set: std::collections::HashSet<usize> = candidate_set.iter().copied().collect();
            self.entries
                .iter()
                .enumerate()
                .filter(|(i, e)| {
                    matches!(e.kind, NeuronKind::Verbatim | NeuronKind::Aggregate)
                        && !in_set.contains(i)
                })
                .map(|(i, _)| i)
                .collect()
        } else {
            vec![]
        };

        // BM25 scoring — kind-filtered over candidates in scope.
        // kind=None or "all" → Core + Project + Verbatim (default)
        // kind="code"         → Core + Project only (exclude conversation/Verbatim)
        // kind="conversation" → Verbatim only (episodic recall, excludes code neurons)
        // Aggregate neurons are NEVER in the general BM25 pool — they are injected
        // via counting_augment only when detect_counting_query() fires, preventing
        // pollution of non-counting R@5 results.
        let kind_lower = kind.map(|k| k.to_lowercase());
        let mut bm25_scored: Vec<(f32, usize)> = candidate_set
            .iter()
            .filter(|&&i| {
                let k = &self.entries[i].kind;
                let kind_ok = match kind_lower.as_deref() {
                    Some("conversation") => matches!(k, NeuronKind::Verbatim),
                    Some("code") => matches!(k, NeuronKind::Core | NeuronKind::Project),
                    _ => matches!(
                        k,
                        NeuronKind::Core | NeuronKind::Project | NeuronKind::Verbatim
                    ),
                };
                kind_ok && module_set.as_ref().map_or(true, |ms| ms.contains(&i))
            })
            .filter_map(|&i| {
                let mut s = self.bm25_score(ranking_terms, &self.entries[i]);
                if is_session_summary_path(&self.entries[i].neuron_path) {
                    if is_counting {
                        s *= 1.35;
                    } else if matches!(kind_lower.as_deref(), Some("conversation") | None) {
                        s *= 1.15;
                    }
                }
                // R18 P2 Sol B: knowledge-update routing — demote stale Verbatim neurons
                // so updated KG/Concept facts rank above old verbatim assertions.
                // R21 T4: ×0.8 → ×0.5 — old fact now needs 2× BM25 score to beat new fact.
                if is_knowledge_update && matches!(self.entries[i].kind, NeuronKind::Verbatim) {
                    s *= 0.5;
                }
                (s > 0.0).then_some((s, i))
            })
            .collect();

        // Merge counting-query expanded candidates into bm25_scored.
        // Aggregate neurons are intentionally excluded here — Sol-A+ injects the best one
        // into `selected` after top_cores are determined, preventing Aggregates from
        // displacing Verbatim chunks in the BM25 top-5 ranking.
        if !counting_augment.is_empty() {
            let already_scored: std::collections::HashSet<usize> =
                bm25_scored.iter().map(|(_, i)| *i).collect();
            for i in counting_augment {
                if already_scored.contains(&i) {
                    continue;
                }
                // Aggregates handled exclusively by Sol-A+ block below
                if matches!(self.entries[i].kind, NeuronKind::Aggregate) {
                    continue;
                }
                let s = self.bm25_score(ranking_terms, &self.entries[i]);
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
            let is_oldest =
                detect_oldest_query(task) && !detect_temporal_query(task) && !is_knowledge_update;
            // KU gets a stronger boost (0.8) than standard temporal (0.6) because BM25
            // vocabulary gap between old and new facts can be larger than event-retrieval gaps.
            let boost_strength = if named_person_move_query {
                0.0
            } else if explicit_current_state_query {
                1.2
            } else if is_knowledge_update && !detect_temporal_query(task) {
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

        // Narrow fix for named-person relocation questions: prefer candidates whose body text
        // actually contains move/live evidence, not just mine-time query_surface hints.
        if named_person_move_query {
            for (score, i) in bm25_scored.iter_mut() {
                if !matches!(self.entries[*i].kind, NeuronKind::Verbatim) {
                    continue;
                }
                if self.entries[*i].has_move_residence_evidence {
                    *score *= 1.35;
                } else {
                    *score *= 0.55;
                }
            }
            tracing::debug!(
                task,
                candidates = bm25_scored.len(),
                "Named-person relocation body-evidence rerank applied"
            );
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
            let verbatim_scored: Vec<(usize, f32)> = bm25_scored
                .iter()
                .filter(|(_, i)| matches!(self.entries[*i].kind, NeuronKind::Verbatim))
                .map(|(s, i)| (*i, *s))
                .collect();

            if !verbatim_scored.is_empty() {
                let scored_path_map: std::collections::HashMap<PathBuf, f32> = verbatim_scored
                    .iter()
                    .map(|(i, score)| (self.entries[*i].neuron_path.clone(), *score))
                    .collect();

                for (score, i) in bm25_scored.iter_mut() {
                    if !matches!(self.entries[*i].kind, NeuronKind::Verbatim) {
                        continue;
                    }
                    let anchor = self.entries[*i].neuron_path.clone();

                    // BFS along TemporalFollows edges, up to 3 hops
                    let mut frontier = vec![anchor.clone()];
                    let mut seen: std::collections::HashSet<PathBuf> =
                        std::collections::HashSet::new();
                    seen.insert(anchor.clone());
                    let mut hop_discount = 0.5f32;

                    for _hop in 0..3 {
                        let mut next_frontier = Vec::new();
                        for path in &frontier {
                            let Some(neighbors) = self.adjacency.get(path) else {
                                continue;
                            };
                            for syn in neighbors {
                                if syn.edge_type != SynapseType::TemporalFollows {
                                    continue;
                                }
                                if seen.contains(&syn.target) {
                                    continue;
                                }
                                seen.insert(syn.target.clone());
                                // Add chain-member score to anchor — but only if the
                                // chain member is also a BM25 candidate (already scored).
                                // This keeps the boost evidence-grounded.
                                if let Some(chain_score) = scored_path_map.get(&syn.target) {
                                    *score += hop_discount * *chain_score;
                                }
                                next_frontier.push(syn.target.clone());
                            }
                        }
                        if next_frontier.is_empty() {
                            break;
                        }
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
                top_score * 0.70 // 30% zone for KU: updated facts may lag on BM25
            } else {
                top_score * 0.85 // 15% zone for all other queries
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
                if already_scored.contains(&i) {
                    continue;
                }
                if module_set.as_ref().map_or(false, |ms| !ms.contains(&i)) {
                    continue;
                }
                // R18 P1b Sol4: only compare first 4 seeds (previously all 16) — same accuracy
                // benefit vs original 1 seed, but 75% less comparison overhead.
                if entry.lsh_fingerprints[..4].iter().all(|&fp| fp == 0) {
                    continue;
                }
                let matched = query_fps[..4]
                    .iter()
                    .zip(entry.lsh_fingerprints[..4].iter())
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
        // TF-IDF now runs for ALL queries that are NOT decisively high-confidence on BM25.
        // Previously, a middle-confidence band skipped TF-IDF even when BM25 was not fully
        // decisive. Stale facts often score deceptively high on BM25 (exact keyword match)
        // and slip through — TF-IDF re-rank catches them.
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
            let run_tfidf =
                force_tfidf || (top < HIGH_CONFIDENCE_THRESHOLD && bm25_scored.len() > 1);
            if !force_tfidf && top >= HIGH_CONFIDENCE_THRESHOLD {
                tracing::debug!(
                    "High-confidence BM25 ({top:.2}) — skipping TF-IDF and dense re-rank."
                );
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
                    // Linear sparse-score blend: BM25 0.6 + TF-IDF 0.4.
                    *score = 0.6 * *score + 0.4 * tfidf;
                }
                // Re-sort after blending scores.
                bm25_scored[..rerank_n]
                    .sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
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
            // Gate: only apply dense re-rank when BM25 is genuinely failing (< LOW_CONFIDENCE)
            // AND TF-IDF was not forced. At low confidence, cosine similarity can rescue queries
            // with vocabulary mismatch. At moderate/high confidence, the all-MiniLM-L6-v2
            // general-purpose model adds noise that outweighs its signal on this workload.
            let top_for_embed = bm25_scored.first().map(|(s, _)| *s).unwrap_or(0.0);
            let run_embed = !self.embeddings.is_empty()
                && !force_tfidf
                && top_for_embed < LOW_CONFIDENCE_THRESHOLD;
            if run_embed {
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
                        Option<crate::embedder::EmbeddingBackend>,
                    > = std::sync::OnceLock::new();
                    let backend =
                        EMBEDDER.get_or_init(|| crate::embedder::EmbeddingBackend::new().ok());
                    backend.as_ref()?.embed_query(task).ok()
                })();

                if let Some(query_vec) = embed_result {
                    let rerank_n = bm25_scored.len().min(20);
                    let mut cos_scores: Vec<(f32, usize)> = bm25_scored[..rerank_n]
                        .iter()
                        .map(|(_, idx)| {
                            let npath = &self.entries[*idx].neuron_path;
                            let cos = self
                                .embeddings
                                .get(npath)
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
                    bm25_scored[..rerank_n]
                        .sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
                    tracing::debug!("Dense embed re-rank applied to top-{rerank_n} candidates.");
                }
            }
        }

        // Phase 1c — ONNX cross-encoder reranking (feature = "rerank").
        // Low-confidence escalation: activated only when the top BM25 score is below
        // LOW_CONFIDENCE_THRESHOLD, indicating that BM25 is genuinely uncertain.
        // Note: structural FAILs (where BM25 is confidently WRONG) cannot be rescued
        // this way; mine-time paraphrase injection (Phase 2) is the preferred fix.
        // Falls back silently if `.cortyx/reranker.onnx` is absent.
        #[cfg(feature = "rerank")]
        {
            let top_score = bm25_scored.first().map(|(s, _)| *s).unwrap_or(0.0);
            if top_score < LOW_CONFIDENCE_THRESHOLD {
                if let Some(reranker) = crate::reranker::inner::global_reranker(&self.project_root)
                {
                    // Normalize BM25 scores to [0, 1] range
                    let max_bm25 = top_score.max(f32::EPSILON);
                    let rerank_n = bm25_scored.len().min(10);
                    for (score, idx) in bm25_scored.iter_mut().take(rerank_n) {
                        let entry = &self.entries[*idx];
                        // First 800 chars: enough for key facts, fits CE 512-token window.
                        let passage = std::fs::read_to_string(&entry.neuron_path)
                            .map(|s| s.chars().take(800).collect::<String>())
                            .unwrap_or_else(|_| {
                                entry
                                    .term_freq
                                    .keys()
                                    .cloned()
                                    .collect::<Vec<_>>()
                                    .join(" ")
                            });
                        let ce_score = reranker.score_pair(task, &passage);
                        let bm25_norm = *score / max_bm25;
                        // 80% BM25 + 20% CE blend
                        *score = 0.80 * bm25_norm + 0.20 * ce_score;
                    }
                    bm25_scored[..rerank_n]
                        .sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
                    tracing::debug!(
                        "ONNX cross-encoder blend applied to top-{rerank_n} (low-confidence query)."
                    );
                }
            }
        }

        let top_cores: Vec<(f32, usize)> = bm25_scored.into_iter().take(MAX_CORE_NEURONS).collect();

        let max_score = top_cores
            .first()
            .map(|(s, _)| *s)
            .unwrap_or(0.001)
            .max(0.001);

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
                Self {
                    set: HashSet::new(),
                    ordered: Vec::new(),
                }
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

        let should_inject_summary = !is_counting
            && !is_knowledge_update
            && !detect_temporal_query(task)
            && !detect_oldest_query(task)
            && matches!(kind_lower.as_deref(), Some("conversation") | None)
            && (task_lower.starts_with("what ")
                || task_lower.starts_with("where ")
                || task_lower.starts_with("who ")
                || task_lower.starts_with("which "))
            && (task_lower.contains(" my ")
                || task_lower.starts_with("what is my")
                || task_lower.starts_with("where did i")
                || task_lower.starts_with("who gave me"));

        if should_inject_summary {
            if let Some((_, summary_idx)) = self
                .entries
                .iter()
                .enumerate()
                .filter(|(_, entry)| {
                    matches!(entry.kind, NeuronKind::Verbatim)
                        && is_session_summary_path(&entry.neuron_path)
                })
                .filter_map(|(i, entry)| {
                    let bm25 = self.bm25_score(ranking_terms, entry);
                    if bm25 <= 0.0 {
                        return None;
                    }
                    let lexical_overlap = ranking_terms
                        .iter()
                        .filter(|term| entry.term_freq.contains_key(term.as_str()))
                        .count() as f32;
                    let score = bm25 * 1.5 + lexical_overlap;
                    Some((score, i))
                })
                .max_by(|a, b| a.0.total_cmp(&b.0))
            {
                selected.insert(self.entries[summary_idx].neuron_path.clone());
            }
        }

        if let Some(answer_path) = self.synthetic_answer_path(task) {
            selected.insert(answer_path);
        }

        // Sol-A+: For counting queries, inject the best-scoring Aggregate neuron early.
        // These queries often want the aggregate as the direct answer; if we append it
        // after several large verbatim chunks, the token budget can exclude it entirely.
        if is_counting {
            let raw_focus_terms: &[String] = if !raw_counting_focus_terms.is_empty() {
                &raw_counting_focus_terms
            } else if !raw_knowledge_focus_terms.is_empty() {
                &raw_knowledge_focus_terms
            } else {
                &terms
            };
            let is_dollar_query = is_money_query(task);
            let use_count_aggregate = should_inject_count_aggregate(task);

            let best_agg = if is_dollar_query {
                best_matching_arithmetic_aggregate_path(&self.project_root, raw_focus_terms)
            } else if use_count_aggregate {
                None
            } else {
                None
            };

            if let Some(agg_path) = best_agg {
                selected.insert(agg_path);
            }
        }

        // top_cores are already ordered by BM25 score (descending).
        for (_, i) in &top_cores {
            selected.insert(self.entries[*i].neuron_path.clone());
        }

        // Also include Concept neurons that match the query (via posting list — no O(n) scan).
        // Global concepts (module == None) activate across all namespaces.
        for &i in candidate_set
            .iter()
            .filter(|&&i| self.entries[i].kind == NeuronKind::Concept)
        {
            if let Some(m) = module {
                if self.entries[i].module.as_deref() != Some(m) && self.entries[i].module.is_some()
                {
                    continue;
                }
            }
            let score = self.bm25_score(ranking_terms, &self.entries[i]);
            if score > SYNAPSE_RELEVANCE_THRESHOLD * max_score {
                selected.insert(self.entries[i].neuron_path.clone());
            }
        }

        // Phase 2 — UseCase neurons for each activated Core
        for (_, idx) in &top_cores {
            let core_path = self.entries[*idx].neuron_path.clone();
            let child_indices = self
                .parent_index
                .get(&core_path)
                .cloned()
                .unwrap_or_default();
            let mut uc_scores: Vec<(f32, usize)> = child_indices
                .into_iter()
                .filter(|&i| self.entries[i].kind == NeuronKind::UseCase)
                .filter_map(|i| {
                    // BM25 handles paraphrases that share no exact tokens (vs Jaccard).
                    let s = self.bm25_score(ranking_terms, &self.entries[i]);
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
        let phase12_tokens: usize = selected
            .ordered
            .iter()
            .filter_map(|p| self.entry_by_path(p).map(|e| e.tokens))
            .sum();
        let synapse_budget = (max_tokens.saturating_sub(phase12_tokens) / AVG_SYNAPSE_TOKEN_COST)
            .clamp(2, MAX_CORE_NEURONS * 2);

        struct Work {
            path: PathBuf,
            hops_left: u8,
        }
        let mut queue: VecDeque<Work> = top_cores
            .iter()
            .map(|(score, i)| {
                let hops = if *score >= HIGH_ACTIVATION_THRESHOLD * max_score {
                    2
                } else {
                    1
                };
                // R17 L2: Verbatim neurons get +1 hop — TemporalFollows chains span session boundaries
                let hops = if matches!(self.entries[*i].kind, NeuronKind::Verbatim) {
                    hops + 1
                } else {
                    hops
                };
                Work {
                    path: self.entries[*i].neuron_path.clone(),
                    hops_left: hops,
                }
            })
            .collect();

        let mut visited: HashSet<PathBuf> = selected.set.clone();
        let mut extra = 0usize;

        while let Some(work) = queue.pop_front() {
            if extra >= synapse_budget {
                break;
            }
            let neighbors = match self.adjacency.get(&work.path) {
                Some(n) => n.clone(),
                None => continue,
            };
            for syn in &neighbors {
                if visited.contains(&syn.target) || extra >= synapse_budget {
                    continue;
                }

                let neighbor_score = self
                    .entry_by_path(&syn.target)
                    .map(|e| self.bm25_score(ranking_terms, e))
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
                if contradicts_selected {
                    continue;
                }

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
            let verbatim_results: Vec<PathBuf> = local_results
                .iter()
                .filter(|p| {
                    self.path_index
                        .get(*p)
                        .map(|&i| matches!(self.entries[i].kind, NeuronKind::Verbatim))
                        .unwrap_or(false)
                })
                .cloned()
                .collect();

            if verbatim_results.len() >= 2 {
                if let Ok(mut counts) = self.co_return_counts.lock() {
                    // Hebbian synapse threshold: require ≥10 co-returns before firing.
                    // 2 was far too low — any niche query pair would co-occur twice
                    // by chance over a session, polluting the adjacency graph with
                    // spurious SemanticRelated edges.
                    const HEBBIAN_THRESHOLD: u32 = 10;
                    let n = verbatim_results.len();
                    for i in 0..n {
                        for j in (i + 1)..n {
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
        // When a Verbatim neuron enters the top-3, inject nearby siblings from the same
        // session immediately after it. This lets chunked conversations surface the answer
        // chunk even when only an earlier chunk matches the query terms directly.
        //
        // Cost: O(session_size) ≈ O(10–30 turns) per top-3 hit — effectively zero.
        // Guards: only Verbatim, only if sibling not already in results.
        {
            let mut seen_sessions: std::collections::HashSet<String> =
                std::collections::HashSet::new();
            let top3_session_anchors: Vec<(String, PathBuf)> = local_results
                .iter()
                .take(3)
                .filter_map(|p| {
                    self.path_index.get(p).and_then(|&i| {
                        let e = &self.entries[i];
                        if matches!(e.kind, NeuronKind::Verbatim)
                            && !e.session_id.is_empty()
                            && seen_sessions.insert(e.session_id.clone())
                        {
                            Some((e.session_id.clone(), p.clone()))
                        } else {
                            None
                        }
                    })
                })
                .collect();

            if !top3_session_anchors.is_empty() {
                let already_in_results: std::collections::HashSet<&PathBuf> =
                    local_results.iter().collect();
                let mut sibling_map: HashMap<PathBuf, Vec<PathBuf>> = HashMap::new();

                for (sid, anchor_path) in &top3_session_anchors {
                    if let Some(sibling_indices) = self.session_index.get(sid) {
                        let anchor_pos = sibling_indices
                            .iter()
                            .position(|&idx| self.entries[idx].neuron_path == *anchor_path)
                            .unwrap_or(0);
                        let mut ranked_siblings: Vec<(usize, usize, f32, PathBuf)> =
                            sibling_indices
                                .iter()
                                .enumerate()
                                .filter_map(|(sibling_pos, &idx)| {
                                    let path = &self.entries[idx].neuron_path;
                                    if already_in_results.contains(path) {
                                        return None;
                                    }
                                    let distance = anchor_pos.abs_diff(sibling_pos);
                                    let backward_penalty = usize::from(sibling_pos < anchor_pos);
                                    let score = self.bm25_score(ranking_terms, &self.entries[idx]);
                                    Some((distance, backward_penalty, score, path.clone()))
                                })
                                .collect();
                        ranked_siblings.sort_unstable_by(|a, b| {
                            a.0.cmp(&b.0)
                                .then_with(|| a.1.cmp(&b.1))
                                .then_with(|| b.2.total_cmp(&a.2))
                        });
                        let siblings: Vec<PathBuf> = ranked_siblings
                            .into_iter()
                            .take(2)
                            .map(|(_, _, _, path)| path)
                            .collect();
                        if !siblings.is_empty() {
                            sibling_map.insert(anchor_path.clone(), siblings);
                        }
                    }
                }

                if !sibling_map.is_empty() {
                    let mut combined = Vec::new();
                    for path in local_results {
                        combined.push(path.clone());
                        if let Some(siblings) = sibling_map.remove(&path) {
                            combined.extend(siblings);
                        }
                    }
                    tracing::debug!(
                        session_count = top3_session_anchors.len(),
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
        //
        // Collected as Vec so the multi-hop block can reference the pre-budget-split
        // ranked order (all_ordered[..5]) without re-running the pipeline.
        let all_ordered: Vec<PathBuf> = self.get_contexts(task, usize::MAX / 2, module, kind);

        let mut full = Vec::new();
        let mut overflow = Vec::new();
        let mut used = 0usize;

        for path in all_ordered.iter().cloned() {
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

        // Multi-hop retrieval: expand from the top-5 pre-budget-split retrieval hits
        // to discover neurons reachable via multiple semantic paths.
        //
        // Improvement over prior top-1 expansion: seeding from all top-5 hits captures
        // terms from multiple subtopics, improving recall for complex multi-hop queries
        // (recursiveMAS iterative deepening principle applied heuristically).
        //
        // All novel neurons go to overflow (lower-priority hints), so full results and
        // their ranking are unchanged — recall can only increase, not decrease.
        if multi_hop && !all_ordered.is_empty() {
            let seed_entries: Vec<&BM25Entry> = all_ordered
                .iter()
                .take(5)
                .filter_map(|p| self.entry_by_path(p))
                .collect();

            if !seed_entries.is_empty() {
                let mut hop_terms = terms.clone();

                for entry in &seed_entries {
                    // Sort clouds before truncation for determinism across runs.
                    let mut cloud: Vec<&String> = entry.concept_cloud.iter().collect();
                    cloud.sort();
                    hop_terms.extend(cloud.into_iter().take(5).cloned());

                    let mut syns: Vec<&String> = entry.synonym_cloud.iter().collect();
                    syns.sort();
                    hop_terms.extend(syns.into_iter().take(3).cloned());
                }

                // Gather TF-IDF terms from all seeds; deduplicate by keeping max freq per
                // term via BTreeMap (lexicographic key order → deterministic output).
                let already: HashSet<&str> = hop_terms.iter().map(|s| s.as_str()).collect();
                let mut tfidf_best: std::collections::BTreeMap<String, f32> =
                    std::collections::BTreeMap::new();
                for entry in &seed_entries {
                    for (t, &f) in &entry.term_freq {
                        if t.len() >= 4 && !already.contains(t.as_str()) {
                            tfidf_best
                                .entry(t.clone())
                                .and_modify(|v| *v = v.max(f))
                                .or_insert(f);
                        }
                    }
                }
                // Sort by (freq DESC, term ASC) for stable ordering across runs.
                let mut tfidf: Vec<(f32, String)> =
                    tfidf_best.into_iter().map(|(t, f)| (f, t)).collect();
                tfidf.sort_unstable_by(|a, b| {
                    b.0.total_cmp(&a.0).then(a.1.as_str().cmp(b.1.as_str()))
                });
                hop_terms.extend(tfidf.into_iter().take(15).map(|(_, t)| t));

                hop_terms.sort();
                hop_terms.dedup();

                let expanded_task = hop_terms.join(" ");
                let second_pass = self.get_contexts(&expanded_task, usize::MAX / 2, module, kind);

                let already_included: HashSet<&PathBuf> =
                    full.iter().chain(overflow.iter().map(|(p, _)| p)).collect();
                // Cap novel overflow additions to avoid explosion on broad expanded queries.
                let novel: Vec<(PathBuf, String)> = second_pass
                    .into_iter()
                    .filter(|p| !already_included.contains(p))
                    .take(25)
                    .map(|p| {
                        let headline = neuron_headline_for(&p);
                        (p, headline)
                    })
                    .collect();

                if !novel.is_empty() {
                    tracing::debug!(
                        count = novel.len(),
                        seeds = seed_entries.len(),
                        "Multi-hop 2nd pass: injected additional candidate neurons \
                         (top-{} seed expansion)",
                        seed_entries.len()
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
        let hit_terms = terms
            .iter()
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
        let unique_modules: HashSet<Option<&str>> = candidates
            .iter()
            .filter_map(|&i| self.entries.get(i))
            .map(|e| e.module.as_deref())
            .collect();
        let spread = ((unique_modules.len() as f32 - 1.0) / 3.0).clamp(0.0, 1.0);

        // Depth: fraction of candidates that have outgoing synapses
        let with_synapses = candidates
            .iter()
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
            if !entry.neuron_path.starts_with(&ndir) {
                continue;
            }
            match status {
                NeuronStatus::Fresh => fresh += 1,
                NeuronStatus::Stale => stale += 1,
                NeuronStatus::Stub => stub += 1,
            }
        }
        (fresh, stale, stub)
    }

    /// Return the use_count for a neuron (for display purposes).
    pub fn use_count_for(&self, path: &Path) -> u32 {
        self.path_index
            .get(path)
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
                        if let Err(e) = atomic_write_json(&meta_p, &meta) {
                            tracing::warn!(
                                "Failed to persist updated use_count for {}: {e}",
                                meta_p.display()
                            );
                        }
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

            let hit_rate =
                self.entries[i].hit_count as f32 / self.entries[i].use_count.max(1) as f32;

            // Persist both counters
            let meta_p = meta_path(neuron_path);
            if let Ok(data) = std::fs::read_to_string(&meta_p) {
                if let Ok(mut meta) = serde_json::from_str::<NeuronMeta>(&data) {
                    meta.use_count = self.entries[i].use_count;
                    meta.hit_count = self.entries[i].hit_count;
                    if let Err(e) = atomic_write_json(&meta_p, &meta) {
                        tracing::warn!(
                            "Failed to persist hit feedback for {}: {e}",
                            meta_p.display()
                        );
                    }
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

    /// B2: Record query term co-activations for a neuron.
    ///
    /// Called from `get_contexts` for each activated neuron with the query terms.
    /// After ≥30 co-activations, a term is promoted to the neuron's `synonym_cloud`.
    /// The synonym cloud is persisted to the BM25Entry and used at query time for
    /// vocabulary expansion before BM25 scoring.
    pub fn record_coactivation(&mut self, neuron_path: &Path, query_terms: &[String]) {
        const SYNONYM_THRESHOLD: u32 = 30;

        let Some(&entry_idx) = self.path_index.get(neuron_path) else {
            return;
        };

        let counts = self
            .coactivation_counts
            .entry(neuron_path.to_path_buf())
            .or_default();

        let mut promoted = Vec::new();
        for term in query_terms {
            if term.len() < 3 {
                continue;
            }
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
        // Once a pair crosses HEBBIAN_THRESHOLD (10 co-returns), it's flagged there but
        // can't mutate adjacency. Here, in the first subsequent &mut self call, we drain
        // the flagged pairs and create bidirectional SemanticRelated synapses.
        self.apply_pending_hebbian_synapses();
    }

    /// Drain pending Hebbian synapse pairs and create SemanticRelated edges in adjacency.
    fn apply_pending_hebbian_synapses(&mut self) {
        const HEBBIAN_THRESHOLD: u32 = 10;
        let pairs_to_wire: Vec<(PathBuf, PathBuf)> = {
            let Ok(counts) = self.co_return_counts.lock() else {
                return;
            };
            counts
                .iter()
                .filter(|(_, &c)| c == HEBBIAN_THRESHOLD) // exactly at threshold — fire once
                .map(|(k, _)| k.clone())
                .collect()
        };

        for (a, b) in pairs_to_wire {
            // Mark as wired (sentinel = HEBBIAN_THRESHOLD + 1) so we don't re-fire on future calls
            if let Ok(mut counts) = self.co_return_counts.lock() {
                if let Some(c) = counts.get_mut(&(a.clone(), b.clone())) {
                    *c = HEBBIAN_THRESHOLD + 1;
                }
            }

            let already_exists = self.adjacency.get(&a).map_or(false, |syns| {
                syns.iter()
                    .any(|s| s.target == b && s.edge_type == SynapseType::SemanticRelated)
            });
            if already_exists {
                continue;
            }

            let syn_ab = Synapse::new(
                b.clone(),
                SynapseType::SemanticRelated,
                "hebbian:co-return".to_string(),
            );
            let syn_ba = Synapse::new(
                a.clone(),
                SynapseType::SemanticRelated,
                "hebbian:co-return".to_string(),
            );
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

        let Some(&new_idx) = self.path_index.get(new_path) else {
            return;
        };

        // Snapshot new-entry data to avoid borrow conflicts below.
        let (new_module, new_ts, new_terms) = {
            let e = &self.entries[new_idx];
            if !matches!(e.kind, NeuronKind::Verbatim) {
                return;
            }
            let terms: HashSet<String> = e
                .term_freq
                .keys()
                .filter(|t| t.len() >= MIN_TERMS)
                .cloned()
                .collect();
            (e.module.clone(), e.timestamp_secs, terms)
        };

        if new_terms.is_empty() {
            return;
        }
        let new_ts_val = new_ts.unwrap_or(i64::MAX);

        for i in 0..self.entries.len() {
            if i == new_idx {
                continue;
            }
            let e = &self.entries[i];
            if !matches!(e.kind, NeuronKind::Verbatim) {
                continue;
            }
            if e.module != new_module {
                continue;
            }
            let old_ts = e.timestamp_secs.unwrap_or(0);
            // Only demote OLDER neurons — if old_ts ≥ new_ts, the "old" entry is newer
            // or simultaneous; skip it to avoid mutual demotion within a batch.
            if old_ts >= new_ts_val {
                continue;
            }

            let old_terms: HashSet<&str> = e
                .term_freq
                .keys()
                .filter(|t| t.len() >= MIN_TERMS)
                .map(|s| s.as_str())
                .collect();
            if old_terms.len() < MIN_TERMS {
                continue;
            }

            let overlap = new_terms
                .iter()
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

        let underused = history
            .iter()
            .filter(|[used, budget]| *budget > 0 && (*used as f32 / *budget as f32) < 0.4)
            .count();

        let overflowed = history
            .iter()
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
                },
                NeuronKind::UseCase => usecases += 1,
                NeuronKind::Verbatim => verbatim += 1,
                NeuronKind::Concept | NeuronKind::Aggregate => concepts += 1,
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
                    if let Err(e) = atomic_write_json(&meta_file, &meta) {
                        tracing::warn!(
                            "Failed to persist stale marker for {}: {e}",
                            meta_file.display()
                        );
                    }
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
            self.path_index
                .insert(self.entries[idx].neuron_path.clone(), idx);
        }
        self.path_index.remove(neuron_path);
        // Rebuild derived structures — eviction happens in bulk during prune,
        // so the caller calls rebuild_derived() once after all evictions.
        true
    }

    /// Neuron paths together with their activation count — used by `cortyx prune`.
    pub fn neuron_paths_and_use_counts(&self) -> Vec<(PathBuf, u32)> {
        self.entries
            .iter()
            .map(|e| (e.neuron_path.clone(), e.use_count))
            .collect()
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
                avg_hit_rate: if count > 0 {
                    rate_sum / count as f32
                } else {
                    0.0
                },
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

    /// Return the most recent Verbatim neurons that mention "current moment" markers.
    ///
    /// This stays index-only: it uses precomputed timestamps plus token presence to cheaply
    /// surface likely `today` / `currently` / `this week` sessions for downstream temporal
    /// reasoning without scanning the full corpus at query time.
    pub fn recent_verbatim_paths_with_current_markers(
        &self,
        module: Option<&str>,
        limit: usize,
    ) -> Vec<PathBuf> {
        if limit == 0 {
            return Vec::new();
        }

        let has_current_marker_terms = |terms: &HashMap<String, f32>| {
            terms.contains_key("today")
                || terms.contains_key("currently")
                || terms.contains_key("now")
                || (terms.contains_key("this")
                    && (terms.contains_key("week")
                        || terms.contains_key("month")
                        || terms.contains_key("year")))
        };

        let mut ranked = self
            .entries
            .iter()
            .filter(|entry| matches!(entry.kind, NeuronKind::Verbatim))
            .filter(|entry| {
                module.is_none_or(|scope| entry.module.as_deref() == Some(scope))
                    && has_current_marker_terms(&entry.term_freq)
            })
            .filter_map(|entry| Some((entry.timestamp_secs?, entry.neuron_path.clone())))
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)));
        ranked
            .into_iter()
            .take(limit)
            .map(|(_, path)| path)
            .collect()
    }

    /// Return neurons that are strong candidates for the shared concept library.
    ///
    /// Candidates must:
    /// - be Core or Concept neurons
    /// - meet the minimum use_count / hit_rate / quality thresholds
    /// - be sorted by strongest observed utility first
    pub fn publish_ready_candidates(
        &self,
        min_use: u32,
        min_hit_rate: f32,
        min_quality: f32,
        limit: usize,
    ) -> Vec<PublishReadySummary> {
        let mut result: Vec<PublishReadySummary> = self
            .entries
            .iter()
            .filter_map(|entry| {
                if !matches!(entry.kind, NeuronKind::Core | NeuronKind::Concept) {
                    return None;
                }
                let hit_rate = if entry.use_count > 0 {
                    entry.hit_count as f32 / entry.use_count as f32
                } else {
                    0.0
                };
                if entry.use_count < min_use
                    || hit_rate < min_hit_rate
                    || entry.quality_score < min_quality
                {
                    return None;
                }
                Some(PublishReadySummary {
                    path: entry.neuron_path.clone(),
                    kind: entry.kind.clone(),
                    use_count: entry.use_count,
                    hit_rate,
                    quality_score: entry.quality_score,
                })
            })
            .collect();
        result.sort_by(|a, b| {
            b.use_count
                .cmp(&a.use_count)
                .then_with(|| b.hit_rate.total_cmp(&a.hit_rate))
                .then_with(|| b.quality_score.total_cmp(&a.quality_score))
                .then_with(|| a.path.cmp(&b.path))
        });
        if limit > 0 {
            result.truncate(limit);
        }
        result
    }

    /// Return the first `lines` lines of a neuron file for quick preview.
    /// Returns `None` if the file does not exist or cannot be read.
    pub fn peek_neuron(&self, path: &Path, lines: usize) -> Option<String> {
        let content = std::fs::read_to_string(path).ok()?;
        let preview: String = content.lines().take(lines).collect::<Vec<_>>().join("\n");
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
                    fn_vocab
                        .entry(term.clone())
                        .or_insert_with(|| rel_source.clone());
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
        let walker = WalkDir::new(root)
            .min_depth(1)
            .into_iter()
            .filter_map(|e| e.ok());
        let mut synapse_patches: Vec<(PathBuf, PathBuf)> = Vec::new(); // (caller_neuron, callee_neuron)

        for entry in walker {
            if !entry.file_type().is_file() {
                continue;
            }
            let abs = entry.path();
            let rel = abs.strip_prefix(root).unwrap_or(abs);
            let ext = rel.extension().and_then(|e| e.to_str()).unwrap_or("");
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
            if let Err(e) = atomic_write_json(&meta_file, &meta) {
                tracing::warn!(
                    "Failed to persist call-graph synapse for {}: {e}",
                    meta_file.display()
                );
            }
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
        /// Cap on files per commit before skipping the pair-wise O(n²) step.
        ///
        /// A commit touching more than this many files is almost certainly a
        /// bulk change (dependency bump, generated code, refactor) where co-change
        /// is not a useful semantic signal. Without this cap, a 500-file commit
        /// generates ~125,000 pairs, making compile time degenerate on large repos.
        const MAX_FILES_PER_COMMIT: usize = 50;

        // Adaptive minimum co-change threshold based on repo size.
        // Small repos (≤50 neurons) produce sparse commit histories; 2 co-changes
        // is strong signal. Large repos (>500 neurons) have noisy histories and
        // benefit from a higher bar to avoid false semantic edges.
        let min_cochange: u32 = match self.path_index.len() {
            n if n <= 50 => 2,
            n if n <= 500 => 3,
            _ => 5,
        };

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
                    let key = if a <= b {
                        (a.clone(), b.clone())
                    } else {
                        (b.clone(), a.clone())
                    };
                    *cochange.entry(key).or_insert(0) += 1;
                }
            }
        }

        // Add synapses for qualifying pairs
        let mut changes: Vec<(PathBuf, Synapse)> = Vec::new();
        for ((fa, fb), count) in &cochange {
            if *count < min_cochange {
                continue;
            }
            let na = core_neuron_path(&root.join(fa), root);
            let nb = core_neuron_path(&root.join(fb), root);
            let weight = (0.5_f32 + *count as f32 * 0.05).min(0.9);
            let reason = format!("git co-change: committed together {count}×");

            // Only create synapses for neurons that exist in our index
            if self.path_index.contains_key(&na) && self.path_index.contains_key(&nb) {
                changes.push((
                    na.clone(),
                    Synapse {
                        target: nb.clone(),
                        edge_type: SynapseType::SemanticRelated,
                        weight,
                        reason: reason.clone(),
                        learned_weight: 0.0,
                        traversal_count: 0,
                        last_co_activation_day: 0,
                    },
                ));
                changes.push((
                    nb,
                    Synapse {
                        target: na,
                        edge_type: SynapseType::SemanticRelated,
                        weight,
                        reason,
                        learned_weight: 0.0,
                        traversal_count: 0,
                        last_co_activation_day: 0,
                    },
                ));
            }
        }

        for (source_neuron, syn) in changes {
            let meta_p = meta_path(&source_neuron);
            if let Ok(data) = std::fs::read_to_string(&meta_p) {
                if let Ok(mut meta) = serde_json::from_str::<NeuronMeta>(&data) {
                    let already = meta.synapses.iter().any(|s| s.target == syn.target);
                    if !already {
                        meta.synapses.push(syn.clone());
                        if let Err(e) = atomic_write_json(&meta_p, &meta) {
                            tracing::warn!(
                                "Failed to persist co-change synapse for {}: {e}",
                                meta_p.display()
                            );
                        }
                    }
                }
            }
            if let Some(&i) = self.path_index.get(&source_neuron) {
                let already = self.entries[i]
                    .synapses
                    .iter()
                    .any(|s| s.target == syn.target);
                if !already {
                    self.entries[i].synapses.push(syn);
                }
            }
        }
    }

    /// Add or replace a single entry in `self.entries` (does NOT rebuild derived).
    pub fn index_neuron(&mut self, neuron_path: &Path, content: &str, meta: &NeuronMeta) {
        let index_content = content;

        let terms = tokenize(index_content);
        let mut tf: HashMap<String, f32> = HashMap::new();
        for t in &terms {
            *tf.entry(t.clone()).or_insert(0.0) += 1.0;
        }

        // P3-B: Paraphrase + alias surface boost.
        // ## paraphrases and the narrow fact_aliases surface bridge natural-language
        // questions to answer-bearing facts without polluting summaries with broad
        // category vocabulary.
        // This closes the vocabulary gap: documents contain both answer vocabulary
        // (original content) and question vocabulary (these sections).
        {
            use crate::neuron::parse_sections;
            let sections = parse_sections(index_content);
            for section_name in ["paraphrases", "query_surface", "fact_aliases"] {
                if let Some(section_content) = sections.get(section_name) {
                    for t in tokenize(section_content) {
                        let v = tf.entry(t).or_insert(0.0);
                        *v += 0.5; // boost: question vocab is high-signal (kept low to avoid over-boosting generic category tokens)
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
            for line in index_content.lines() {
                let lower = line.as_bytes();
                let is_user = lower.starts_with(b"user:")
                    || lower.starts_with(b"User:")
                    || lower.starts_with(b"human:")
                    || lower.starts_with(b"Human:");
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
        let synapses: Vec<Synapse> = meta
            .synapses
            .iter()
            .filter_map(|s| {
                let target = if s.target.is_absolute() {
                    s.target.clone()
                } else {
                    ndir.join(&s.target)
                };
                if !target.starts_with(&ndir) {
                    tracing::warn!(
                        "Skipping synapse with path-traversal target {:?} in {:?}",
                        target,
                        neuron_path
                    );
                    return None;
                }
                Some(Synapse {
                    target,
                    ..s.clone()
                })
            })
            .collect();

        // S-III (R16): Self-Quality Score — fraction of neuron terms that overlap with
        // the corresponding source file's AST-extracted terms.
        // Only computed for Core neurons with a known source file; defaults to 1.0 (neutral).
        let quality_score: f32 =
            if matches!(meta.kind, NeuronKind::Core) && !meta.source_files.is_empty() {
                let source_path = &meta.source_files[0];
                if let Ok(source_text) = std::fs::read_to_string(source_path) {
                    let source_rel = source_path.to_string_lossy();
                    let ast = ast_extractor::extract_signatures(&source_rel, &source_text);
                    // Build source AST term set from all function/type names (split on _ and camelCase)
                    let mut ast_terms: std::collections::HashSet<String> =
                        std::collections::HashSet::new();
                    for name in ast.functions.iter().chain(ast.types.iter()) {
                        ast_terms.extend(tokenize(name));
                    }
                    if ast_terms.is_empty() {
                        1.0 // no AST info → neutral
                    } else {
                        let neuron_terms: std::collections::HashSet<&str> =
                            tf.keys().map(|s| s.as_str()).collect();
                        let overlap = ast_terms
                            .iter()
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

        // S-II (R16/R17 Sol4): Compute a 16-seed SimHash ensemble for LSH fallback.
        let lsh_fingerprints = simhash_1024(&tf);

        // S-I (R16): Extract Tier-1 summary from neuron content.
        // Takes: first non-empty line of `## purpose` section + first line of `## pitfalls`.
        // Stored in memory only (not persisted); rebuilt from neuron file at each index_neuron call.
        let summary = extract_neuron_summary(content);
        let has_move_residence_evidence = content_has_move_residence_evidence(content);

        let entry = BM25Entry {
            neuron_path: neuron_path.to_path_buf(),
            kind: meta.kind.clone(),
            term_freq: tf,
            term_count: terms.len(),
            // Use meta.tokens when available (set by compile/upsert after reading disk).
            // Fall back to estimating from content so the token budget works in tests
            // and when index_neuron is called before NeuronMeta.tokens is populated.
            tokens: if meta.tokens > 0 {
                meta.tokens
            } else {
                estimate_tokens(content).max(10)
            },
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
            has_move_residence_evidence,
            // R21 T6: Extract session_id from neuron filename stem for Verbatim neurons.
            // Pattern: "lme_0060_0_user.verbatim.md" → session_id = "lme_0060"
            // Split on '_', take first two parts if the stem follows the N_N pattern.
            session_id: if matches!(meta.kind, NeuronKind::Verbatim) {
                neuron_path
                    .file_name()
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
            self.has_pending_updates = true;
            self.needs_full_save.store(true, Ordering::Relaxed);
        } else {
            let pos = self.entries.len();
            self.path_index.insert(neuron_path.to_path_buf(), pos);
            self.entries.push(entry);
            self.pending_append_count += 1;
        }
    }

    /// Rebuild all derived structures — public entry point for `cortyx prune`.
    ///
    /// Prune evicts entries individually then calls this once to reconstruct
    /// path_index, adjacency, df_cache, etc. in a single O(n) pass.
    pub fn rebuild_derived_pub(&mut self) {
        // Force full rebuild: prune may have removed existing entries, so the
        // incremental delta path (which only handles appends) is not safe here.
        self.pending_append_count = 0;
        self.has_pending_updates = true;
        // S4-WAL: prune removes entries — invalidate WAL baseline and force full save.
        self.wal_base.store(0, Ordering::Relaxed);
        self.needs_full_save.store(true, Ordering::Relaxed);
        self.rebuild_derived();
    }

    /// Rebuild all derived structures in a single O(n) pass.
    ///
    /// Previously five separate passes (path_index, parent_index, adjacency, df_cache,
    /// module_index); merged to reduce cache pressure and wall-clock time ~5×.
    pub(super) fn rebuild_derived(&mut self) {
        // S7: Incremental delta — skip the full clear+rebuild when only new entries were
        // appended (no updates).  This reduces the hot path (mining a new file into an
        // existing index) from O(N+n) to O(n) for the HashMap phase.
        if self.pending_append_count > 0 && !self.has_pending_updates && self.idf_n > 0 {
            self.rebuild_derived_delta();
            return;
        }

        self.path_index.clear();
        self.parent_index.clear();
        self.adjacency.clear();
        self.df_cache.clear();
        self.posting_list.clear();
        self.module_index.clear();
        self.session_index.clear(); // R21 T6
        self.idf_n = 0;

        let mut non_agg_total_terms = 0usize;
        let mut verbatim_total_terms = 0usize;
        let mut verbatim_count = 0usize;

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

            // df_cache + posting_list.
            // IMPORTANT: Aggregate neurons (word-count summaries, dollar totals) must NOT
            // contribute to df_cache.  An _count_music.aggregate.md neuron contains "music"
            // dozens of times, inflating df("music") and crushing its IDF.  This caused a
            // 5-entry SSU regression: session 329 ("music"×18, no "streaming"/"service") lost
            // to session 309 ("service"×7) because IDF("music") collapsed while IDF("service")
            // stayed high.  Excluding Aggregate from df_cache restores the IDF calibration
            // from the e18c4e6 baseline (100% SSU) even when aggregates are mined.
            // Posting-list is still built for ALL kinds so counting_augment can find Aggregates.
            let is_aggregate = matches!(entry.kind, NeuronKind::Aggregate);
            for term in entry.term_freq.keys() {
                if !is_aggregate {
                    *self.df_cache.entry(term.clone()).or_insert(0) += 1;
                }
                self.posting_list.entry(term.clone()).or_default().push(i);
            }
            if !is_aggregate {
                self.idf_n += 1;
            }

            // module_index
            if let Some(m) = &entry.module {
                self.module_index.entry(m.clone()).or_default().push(i);
            }

            // R21 T6: session_index — for session-level grouping at retrieval time
            if !entry.session_id.is_empty() {
                self.session_index
                    .entry(entry.session_id.clone())
                    .or_default()
                    .push(i);
            }

            if !is_aggregate {
                non_agg_total_terms += entry.term_count;
            }
            if matches!(entry.kind, NeuronKind::Verbatim) {
                verbatim_total_terms += entry.term_count;
                verbatim_count += 1;
            }
        }

        // avg_doc_len excludes Aggregate neurons so it matches e18c4e6 calibration.
        self.avg_doc_len = if self.idf_n == 0 {
            0.0
        } else {
            non_agg_total_terms as f32 / self.idf_n as f32
        };
        self.avg_verbatim_doc_len = if verbatim_count == 0 {
            self.avg_doc_len
        } else {
            verbatim_total_terms as f32 / verbatim_count as f32
        };

        self.build_vocab_bridge();
        self.build_morpheme_map();
        self.build_concept_clouds();
        self.apply_peer_vocab_borrowing();
        self.merge_cooccurrence_into_vocab_bridge();
        self.load_pmi_neighbors();
        self.structural_artifacts_dirty
            .store(true, Ordering::Relaxed);
        self.pending_append_count = 0;
        self.has_pending_updates = false;
    }

    /// Incremental derived-structure update for pure-append batches (S7).
    ///
    /// When only new entries were appended (no existing entries were modified), we
    /// skip clearing and rebuilding the large HashMaps from scratch.  Instead we
    /// process only the `pending_append_count` newest entries and add their
    /// contributions to the existing structures in O(n) rather than O(N+n).
    ///
    /// The bridge/cloud/neighbor builds (vocab_bridge, morpheme_map, concept_clouds,
    /// pmi_neighbors) still run over the full corpus because they are O(terms), not
    /// O(entries²), and must reflect the complete vocabulary.
    fn rebuild_derived_delta(&mut self) {
        let new_start = self.entries.len().saturating_sub(self.pending_append_count);

        for (offset, entry) in self.entries[new_start..].iter().enumerate() {
            let abs_i = new_start + offset;

            // path_index is already maintained by index_neuron(), but ensure consistency.
            self.path_index.insert(entry.neuron_path.clone(), abs_i);

            if let Some(p) = &entry.parent {
                self.parent_index.entry(p.clone()).or_default().push(abs_i);
            }

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

            let is_aggregate = matches!(entry.kind, NeuronKind::Aggregate);
            for term in entry.term_freq.keys() {
                if !is_aggregate {
                    *self.df_cache.entry(term.clone()).or_insert(0) += 1;
                }
                self.posting_list
                    .entry(term.clone())
                    .or_default()
                    .push(abs_i);
            }
            if !is_aggregate {
                self.idf_n += 1;
            }

            if let Some(m) = &entry.module {
                self.module_index.entry(m.clone()).or_default().push(abs_i);
            }

            if !entry.session_id.is_empty() {
                self.session_index
                    .entry(entry.session_id.clone())
                    .or_default()
                    .push(abs_i);
            }
        }

        // Recompute avg_doc_len from all entries (O(n) integer addition — cheap).
        let mut non_agg_total_terms = 0usize;
        let mut verbatim_total_terms = 0usize;
        let mut verbatim_count = 0usize;
        for entry in &self.entries {
            let is_aggregate = matches!(entry.kind, NeuronKind::Aggregate);
            if !is_aggregate {
                non_agg_total_terms += entry.term_count;
            }
            if matches!(entry.kind, NeuronKind::Verbatim) {
                verbatim_total_terms += entry.term_count;
                verbatim_count += 1;
            }
        }
        self.avg_doc_len = if self.idf_n == 0 {
            0.0
        } else {
            non_agg_total_terms as f32 / self.idf_n as f32
        };
        self.avg_verbatim_doc_len = if verbatim_count == 0 {
            self.avg_doc_len
        } else {
            verbatim_total_terms as f32 / verbatim_count as f32
        };

        // Bridge/cloud/neighbor builds must see the full corpus.
        self.build_vocab_bridge();
        self.build_morpheme_map();
        self.build_concept_clouds();
        self.apply_peer_vocab_borrowing();
        self.merge_cooccurrence_into_vocab_bridge();
        self.load_pmi_neighbors();
        self.structural_artifacts_dirty
            .store(true, Ordering::Relaxed);
        self.pending_append_count = 0;
        self.has_pending_updates = false;
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
                    let jaccard = if union > 0 {
                        inter as f32 / union as f32
                    } else {
                        0.0
                    };
                    // Module bonus: same module → +0.1
                    let module_bonus =
                        if cold_module.is_some() && cold_module == self.entries[*pi].module {
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
            // Aggregate neurons (word-count / dollar summaries) must NOT contribute to the
            // vocab bridge.  Their path fragments ("fish", "bike", "music" …) would become
            // bridge keys containing hundreds of spurious co-topic terms, which would then
            // be injected into every query that mentions those words — corrupting BM25
            // candidate ranking and causing regressions in multi-session retrieval.
            if matches!(entry.kind, NeuronKind::Aggregate) {
                continue;
            }
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
            if let Some(stem) = entry.neuron_path.file_stem().and_then(|s| s.to_str()) {
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
                let Some(&src_idx) = self.path_index.get(src_path) else {
                    continue;
                };
                for syn in syns {
                    if syn.edge_type != SynapseType::SemanticRelated {
                        continue;
                    }
                    let Some(tgt_stem) = syn
                        .target
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .map(|s| s.trim_end_matches(".context").to_lowercase())
                    else {
                        continue;
                    };
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
            self.vocab_bridge
                .entry(tgt_stem)
                .or_default()
                .extend(src_terms);
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
        if !co_path.exists() {
            return;
        }
        let Ok(json) = std::fs::read_to_string(&co_path) else {
            return;
        };
        let Ok(clusters): Result<std::collections::HashMap<String, Vec<String>>, _> =
            serde_json::from_str(&json)
        else {
            return;
        };

        // R18 P1a: cap to 150 high-signal pairs total (both terms ≥4 chars).
        // Prevents the O(n×|bridge|) query expansion blowup that caused the 2.5× slowdown.
        let mut added = 0usize;
        const MAX_CO_PAIRS: usize = 150;
        'outer: for (term, synonyms) in clusters {
            if term.len() < 4 {
                continue;
            }
            let entry = self.vocab_bridge.entry(term).or_default();
            for syn in synonyms {
                if syn.len() >= 4 && entry.insert(syn) {
                    added += 1;
                    if added >= MAX_CO_PAIRS {
                        break 'outer;
                    }
                }
            }
        }
        tracing::debug!(
            pairs = added,
            "R17 Sol2 (capped): co-occurrence vocab bridge merged"
        );
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
        if !co_path.exists() {
            return;
        }
        let Ok(json) = std::fs::read_to_string(&co_path) else {
            return;
        };
        let Ok(clusters): Result<HashMap<String, Vec<String>>, _> = serde_json::from_str(&json)
        else {
            return;
        };

        let mut loaded = 0usize;
        for (term, neighbors) in clusters {
            if term.len() < 4 {
                continue;
            }
            let valid: Vec<String> = neighbors
                .into_iter()
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
            // Aggregates contain English prose terms, not camelCase/snake_case identifiers.
            // Including them adds noise to morpheme expansion without benefit.
            if matches!(entry.kind, NeuronKind::Aggregate) {
                continue;
            }
            for token in entry.term_freq.keys() {
                if token.len() < 4 {
                    continue;
                }
                // Split on underscores (snake_case)
                let snake_parts: Vec<&str> = token.split('_').collect();
                // Split on camelCase transitions (e.g. "validateUser" → ["validate", "User"])
                let camel_parts = split_camel_case(token);

                let mut sub_tokens: HashSet<&str> = HashSet::new();
                for part in snake_parts.iter().chain(
                    camel_parts
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .iter(),
                ) {
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
                if fragment.contains(term_lower.as_str()) || term_lower.contains(fragment.as_str())
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
        // Use idf_n (non-Aggregate count) as IDF corpus size so Aggregate neurons
        // that contain high-frequency terms do not corrupt IDF calibration.
        let n = self.idf_n.max(1) as f32;
        let avg = self.avg_doc_len.max(1.0);
        let dl = entry.term_count as f32;
        let len_norm = 1.0 - BM25_B + BM25_B * (dl / avg);

        // R21 T10: per-entry k1 — Verbatim neurons (long conversation text) use k1=1.5
        // to allow longer documents to score higher on frequently-mentioned terms.
        // Core/Project neurons keep the default k1=1.2.
        let k1 = if matches!(entry.kind, NeuronKind::Verbatim) {
            1.5
        } else {
            BM25_K1
        };

        let raw: f32 = terms
            .iter()
            .map(|t| {
                let tf = entry.term_freq.get(t).copied().unwrap_or(0.0);
                if tf == 0.0 {
                    return 0.0;
                }
                // Laplace floor: if a term appears only in Aggregate neurons it may be
                // absent from df_cache (which is built from regular neurons during
                // rebuild_derived). Default df=1 prevents IDF blow-up for such terms:
                //   IDF = ln((n - 0.5) / 1.5)  — reasonable for rare terms.
                let df = self.df_cache.get(t).copied().unwrap_or(1) as f32;
                let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln().max(0.0);
                // R18 P3 Sol D / R19 fix: BM25+ δ=0.5 (reduced from 1.0 — smaller perturbation,
                // less global ranking disruption while still providing the lower-bound benefit).
                const BM25_DELTA: f32 = 0.5;
                idf * (BM25_DELTA + (tf * (k1 + 1.0)) / (tf + k1 * len_norm))
            })
            .sum();

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
            let q_tf = 1.0f32; // query term frequency is always 1 for bag-of-words queries
            let d_tf = entry.term_freq.get(term).copied().unwrap_or(0.0);
            let q_w = q_tf * idf;
            let d_w = d_tf * idf;
            dot += q_w * d_w;
            q_mag += q_w * q_w;
            d_mag += d_w * d_w;
        }
        let denom = q_mag.sqrt() * d_mag.sqrt();
        if denom == 0.0 {
            0.0
        } else {
            (dot / denom).clamp(0.0, 1.0)
        }
    }

    /// Find an entry by its neuron path — O(1) via precomputed path_index.
    fn entry_by_path(&self, path: &Path) -> Option<&BM25Entry> {
        self.path_index.get(path).map(|&i| &self.entries[i])
    }

    /// Count how many of the given tokens appear in the BM25 term_freq for `path`.
    ///
    /// Used by `close_task` for term-freq soft citation: if the response text shares
    /// ≥ N vocabulary terms with a neuron, it's likely grounded in that neuron.
    pub fn term_freq_overlap(
        &self,
        path: &Path,
        tokens: &std::collections::HashSet<String>,
    ) -> usize {
        self.entry_by_path(path)
            .map(|e| {
                tokens
                    .iter()
                    .filter(|t| e.term_freq.contains_key(*t))
                    .count()
            })
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
        self.entries
            .iter()
            .filter(|e| e.quality_score < 0.4)
            .count()
    }

    /// Return the number of distinct terms indexed for a neuron.
    ///
    /// Used by S-VIII auto-mine to compute code-block ∩ neuron term overlap ratio.
    pub fn term_count_for(&self, path: &Path) -> usize {
        self.entry_by_path(path)
            .map(|e| e.term_freq.len())
            .unwrap_or(0)
    }

    /// S-I (R16): Return the pre-computed Tier-1 summary for a neuron.
    ///
    /// Returns `None` if the neuron is not indexed or has no summary.
    pub fn summary_for(&self, path: &Path) -> Option<&str> {
        self.entry_by_path(path)
            .filter(|e| !e.summary.is_empty())
            .map(|e| e.summary.as_str())
    }

    pub fn module_for(&self, path: &Path) -> Option<&str> {
        self.entry_by_path(path)
            .and_then(|entry| entry.module.as_deref())
    }

    /// Build a bounded, read-only reasoning report around already-selected evidence paths.
    ///
    /// This intentionally operates after retrieval: callers provide the selected evidence
    /// seeds and the reasoner only explores a small adjacency neighborhood rooted at those
    /// seeds, leaving the BM25 hot path unchanged.
    pub fn reason_over_paths(
        &self,
        seeds: &[(PathBuf, f32)],
        options: TraversalOptions,
    ) -> ReasoningReport {
        let seeds: Vec<(PathBuf, f32)> = seeds
            .iter()
            .filter(|(_, score)| *score > 0.0)
            .map(|(path, score)| (path.clone(), *score))
            .collect();
        if seeds.is_empty() {
            return ReasoningReport::default();
        }

        let mut included = HashSet::new();
        let mut queue = VecDeque::new();
        for (path, _) in &seeds {
            if included.insert(path.clone()) {
                queue.push_back((path.clone(), 0_u8));
            }
        }

        while let Some((path, depth)) = queue.pop_front() {
            if depth >= options.max_hops {
                continue;
            }

            let Some(neighbors) = self.adjacency.get(&path) else {
                continue;
            };
            for synapse in neighbors {
                if included.insert(synapse.target.clone()) {
                    queue.push_back((synapse.target.clone(), depth + 1));
                }
            }
        }

        let neurons = included
            .iter()
            .filter_map(|path| self.entry_by_path(path).map(reasoner_neuron_from_entry))
            .collect::<Vec<_>>();
        let kg_entities = included
            .iter()
            .filter(|path| looks_like_kg_neuron_path(path))
            .filter_map(|path| kg::KgEntity::load(path).ok())
            .collect::<Vec<_>>();

        if neurons.is_empty() && kg_entities.is_empty() {
            return ReasoningReport::default();
        }

        GraphReasoner::new(neurons, kg_entities).trace(
            &seeds
                .into_iter()
                .map(|(path, score)| ReasonerSeed::new(path, score))
                .collect::<Vec<_>>(),
            options,
        )
    }

    pub fn context_metadata_for(&self, path: &Path) -> Option<ContextMetadata> {
        self.entry_by_path(path).map(|entry| {
            let hit_rate = if entry.use_count == 0 {
                0.0
            } else {
                entry.hit_count as f32 / entry.use_count as f32
            };
            ContextMetadata {
                kind: entry.kind.clone(),
                module: entry.module.clone(),
                summary: entry.summary.clone(),
                timestamp_secs: entry.timestamp_secs,
                tokens: entry.tokens,
                use_count: entry.use_count,
                hit_count: entry.hit_count,
                hit_rate,
            }
        })
    }

    pub fn derived_answer_path_for_task(&self, task: &str) -> Option<PathBuf> {
        self.synthetic_answer_path(task)
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
        multi_hop: bool,
    ) -> (Vec<(PathBuf, f32)>, Vec<(PathBuf, String)>) {
        // Delegation: run the full pipeline then re-score the results for tier assignment.
        let (full_paths, overflow) = self.get_contexts_with_overflow(
            task,
            max_tokens,
            module,
            kind,
            min_confidence,
            multi_hop,
        );
        let terms = tokenize(task);
        let full_with_scores: Vec<(PathBuf, f32)> = full_paths
            .into_iter()
            .map(|path| {
                let score = self
                    .entry_by_path(&path)
                    .map(|e| self.bm25_score(&terms, e))
                    .unwrap_or(0.0);
                (path, score)
            })
            .collect();
        (full_with_scores, overflow)
    }

    /// CountNeuron (TRIZ NE-5): Pre-aggregate cross-session occurrence counts at mine time.
    ///
    /// Scans all `NeuronKind::Verbatim` entries, groups them by `session_id`, and builds
    /// a `term → distinct_sessions` map.  For terms appearing in ≥3 distinct sessions it
    /// emits a `NeuronKind::Aggregate` neuron that answers "how many times did I X?" in
    /// O(1) — the count is written in BOTH numeral and word form so keyword matching hits.
    ///
    /// Call this after `idx.commit()` and call `idx.commit()` once more if it returns
    /// `true` (at least one aggregate neuron was staged).
    pub fn emit_aggregate_neurons(&mut self, project_root: &Path) -> Result<bool> {
        use crate::neuron::NeuronStatus;
        use std::collections::hash_map::Entry;

        // Common words that would produce useless aggregate neurons
        const AGG_STOP: &[&str] = &[
            "that",
            "this",
            "with",
            "from",
            "have",
            "will",
            "what",
            "when",
            "where",
            "which",
            "there",
            "their",
            "them",
            "they",
            "then",
            "been",
            "were",
            "some",
            "just",
            "also",
            "about",
            "into",
            "more",
            "than",
            "your",
            "here",
            "very",
            "well",
            "over",
            "back",
            "down",
            "would",
            "could",
            "should",
            "might",
            "does",
            "didn",
            "wasn",
            "aren",
            "isn",
            "hasn",
            "like",
            "want",
            "need",
            "think",
            "know",
            "said",
            "told",
            "went",
            "make",
            "made",
            "take",
            "took",
            "come",
            "came",
            "went",
            "going",
            "really",
            "still",
            "even",
            "already",
            "always",
            "never",
            "every",
            "after",
            "before",
            "during",
            "while",
            "other",
            "another",
            "both",
            "first",
            "last",
            "next",
            "same",
            "such",
            "much",
            "many",
            "most",
            "because",
            "since",
            "through",
            "between",
            "under",
            "again",
            "help",
            "time",
            "year",
            "week",
            "month",
            "today",
            "yesterday",
            "tomorrow",
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ];
        let agg_stop: HashSet<&str> = AGG_STOP.iter().copied().collect();

        // Gather (term, session_id) pairs from every Verbatim entry.
        let mut term_sessions: HashMap<String, HashSet<String>> = HashMap::new();
        let mut term_snippets: HashMap<String, Vec<(String, String)>> = HashMap::new();

        // Collect entries data without borrowing self mutably (for peek_neuron)
        let entries_snapshot: Vec<(NeuronKind, String, PathBuf, Vec<String>)> = self
            .entries
            .iter()
            .filter(|e| matches!(e.kind, NeuronKind::Verbatim) && !e.session_id.is_empty())
            .map(|e| {
                (
                    e.kind.clone(),
                    e.session_id.clone(),
                    e.neuron_path.clone(),
                    e.term_freq.keys().cloned().collect(),
                )
            })
            .collect();

        for (_, sid, neuron_path, terms) in &entries_snapshot {
            let content_snippet = std::fs::read_to_string(neuron_path)
                .unwrap_or_default()
                .lines()
                .filter(|l| !l.starts_with('#'))
                .take(1)
                .next()
                .unwrap_or("")
                .chars()
                .take(100)
                .collect::<String>();

            for term in terms {
                // Only count-worthy terms: ≥4 chars, letters only (no numbers/punct)
                if term.len() < 4 {
                    continue;
                }
                if !term.chars().all(|c| c.is_ascii_alphabetic()) {
                    continue;
                }
                if agg_stop.contains(term.as_str()) {
                    continue;
                }

                term_sessions
                    .entry(term.clone())
                    .or_default()
                    .insert(sid.clone());

                if let Entry::Occupied(mut e) = term_snippets.entry(term.clone()) {
                    // Limit snippets per term to avoid huge files
                    if e.get().len() < 10 && !e.get().iter().any(|(s, _)| s == sid) {
                        e.get_mut().push((sid.clone(), content_snippet.clone()));
                    }
                } else {
                    term_snippets
                        .insert(term.clone(), vec![(sid.clone(), content_snippet.clone())]);
                }
            }
        }

        let ndir = neuron_dir(project_root);
        let mut staged = 0usize;

        for (term, sessions) in &term_sessions {
            let count = sessions.len();
            if count < 3 {
                continue;
            }

            let slug: String = term.chars().take(48).collect();
            let fname = format!("_count_{slug}.aggregate.md");
            let neuron_path = ndir.join(&fname);

            let word = num_to_word(count);
            let count_str = if word.is_empty() {
                format!("{count}")
            } else {
                format!("{count} ({word})")
            };

            // Snippets section
            let snippets = term_snippets.get(term).cloned().unwrap_or_default();
            let snippet_lines: String = snippets
                .iter()
                .map(|(sid, snip)| format!("- {sid}: {snip}\n"))
                .collect();

            let query_surface = format!(
                "how many {term}\n\
                 count of {term}\n\
                 number of {term}\n\
                 how many different {term}\n\
                 total {term}\n"
            );

            let content = format!(
                "# _count_{slug}\n\
                 \n\
                 ## purpose\n\
                 Aggregate count: \"{term}\" mentioned in {count_str} sessions.\n\
                 \n\
                 ## count\n\
                 {count_str} sessions\n\
                 \n\
                 ## entity\n\
                 {term}\n\
                 \n\
                 ## query_surface\n\
                 <!-- SECTION: query_surface -->\n\
                 {query_surface}\
                 <!-- /SECTION -->\n\
                 \n\
                 ## sessions\n\
                 {snippet_lines}\
                 \n\
                 ## total\n\
                 Mentioned {count_str} times across {count_str} sessions. Count: {count} ({}).\n",
                num_to_word(count)
            );

            // Write the file
            if let Err(e) = atomic_write(&neuron_path, content.as_bytes()) {
                eprintln!("[emit_aggregate] failed to write {fname}: {e}");
                continue;
            }

            // Build meta for Aggregate neuron
            let mut meta = NeuronMeta::new_stub(project_root, NeuronKind::Aggregate);
            meta.status = NeuronStatus::Fresh;
            meta.tokens = estimate_context_tokens(&content);

            self.stage(&neuron_path, &content, &meta);
            staged += 1;
        }

        Ok(staged > 0)
    }

    /// TRIZ Sol-A: Pre-compute arithmetic aggregates (dollar/numeric sums) at mine time.
    ///
    /// Scans all Verbatim neurons, extracts dollar amounts, groups by entity slug,
    /// and emits offline aggregate files that answer "how much total did X spend?" in O(1).
    /// These files are kept out of the hot BM25 index and are injected directly by path
    /// for money queries, preserving recall without bloating startup latency.
    /// Emit arithmetic aggregate neurons grouped by TOPIC TERM.
    ///
    /// For each term appearing in ≥2 sessions where that term co-occurs with a dollar amount
    /// on the same line, compute the total dollars and emit `_arith_{term}.aggregate.md`.
    ///
    /// This enables Sol-A+ to inject the correct sum for queries like
    /// "how much total have I spent on bike-related expenses?" → finds _arith_bike.aggregate.md
    /// containing "Total: $185".
    pub fn emit_arithmetic_aggregate_neurons(&mut self, project_root: &Path) -> Result<bool> {
        fn parse_dollar(s: &str) -> Option<i64> {
            let cleaned: String = s
                .chars()
                .filter(|c| c.is_ascii_digit() || *c == '.')
                .collect();
            let val: f64 = cleaned.parse().ok()?;
            if val > 10_000_000.0 {
                return None;
            }
            Some((val * 100.0).round() as i64)
        }

        fn extract_dollars_on_line(line: &str) -> Vec<i64> {
            let mut results = Vec::new();
            let bytes = line.as_bytes();
            let mut i = 0;
            while i < bytes.len() {
                if bytes[i] == b'$' {
                    let start = i + 1;
                    let mut j = start;
                    while j < bytes.len()
                        && (bytes[j].is_ascii_digit() || bytes[j] == b',' || bytes[j] == b'.')
                    {
                        j += 1;
                    }
                    if j > start {
                        let num_str = &line[start..j];
                        if let Some(cents) = parse_dollar(num_str) {
                            if cents > 0 {
                                results.push(cents);
                            }
                        }
                    }
                    i = j;
                } else {
                    i += 1;
                }
            }
            results
        }

        fn is_grounded_user_money_line(lower: &str) -> bool {
            if !lower.trim_start().starts_with("user:") {
                return false;
            }

            ![
                "budget",
                "under $",
                "over $",
                "around $",
                "approximately $",
                "approx $",
                "starting at $",
                "start at $",
                "ranges from $",
                "range from $",
                "between $",
                "if you book",
                "fare is around",
                "might run around",
                "could cost",
                "would cost",
                "would be around",
            ]
            .iter()
            .any(|marker| lower.contains(marker))
        }

        fn clean_alpha_token(token: &str, agg_stop: &HashSet<&str>) -> Option<String> {
            let cleaned: String = token
                .chars()
                .filter(|c| c.is_ascii_alphabetic())
                .collect::<String>()
                .to_ascii_lowercase();
            if cleaned.len() < 4 || agg_stop.contains(cleaned.as_str()) {
                return None;
            }
            Some(cleaned)
        }

        fn trim_boundary_terms(words: &mut Vec<String>) {
            const BOUNDARY_STOP: &[&str] = &[
                "spent",
                "spend",
                "pay",
                "paid",
                "buy",
                "bought",
                "purchase",
                "purchased",
                "cost",
                "costs",
                "costing",
                "using",
                "used",
                "redeemed",
                "redeem",
                "bill",
                "bills",
                "fare",
                "fares",
                "ticket",
                "tickets",
                "coupon",
                "coupons",
                "amount",
                "total",
                "money",
                "dollars",
            ];

            while words
                .first()
                .is_some_and(|w| BOUNDARY_STOP.contains(&w.as_str()))
            {
                words.remove(0);
            }
            while words
                .last()
                .is_some_and(|w| BOUNDARY_STOP.contains(&w.as_str()))
            {
                words.pop();
            }
        }

        fn add_phrase_aliases(words: &[String], out: &mut std::collections::BTreeSet<String>) {
            if words.is_empty() {
                return;
            }

            let max_len = words.len().min(3);
            for start in 0..words.len() {
                for len in 1..=max_len.min(words.len() - start) {
                    out.insert(words[start..start + len].join(" "));
                }
            }
        }

        fn extract_topic_candidates(
            line: &str,
            agg_stop: &HashSet<&str>,
        ) -> std::collections::BTreeSet<String> {
            let raw_tokens: Vec<String> = line
                .split_whitespace()
                .map(|token| {
                    token
                        .trim_matches(|c: char| {
                            !c.is_ascii_alphanumeric() && c != '$' && c != '-' && c != '_'
                        })
                        .to_ascii_lowercase()
                })
                .filter(|token| !token.is_empty())
                .collect();

            let dollar_indices: Vec<usize> = raw_tokens
                .iter()
                .enumerate()
                .filter(|(_, token)| token.starts_with('$'))
                .map(|(i, _)| i)
                .collect();

            let mut candidates = std::collections::BTreeSet::new();
            if dollar_indices.is_empty() {
                return candidates;
            }

            for &idx in &dollar_indices {
                if let Some(anchor) = raw_tokens.get(idx + 1) {
                    if matches!(anchor.as_str(), "on" | "for" | "at" | "toward" | "towards") {
                        let words: Vec<String> = raw_tokens
                            .iter()
                            .skip(idx + 2)
                            .take(5)
                            .filter_map(|token| clean_alpha_token(token, agg_stop))
                            .collect();
                        add_phrase_aliases(&words, &mut candidates);
                    }
                }

                let mut before: Vec<String> = raw_tokens
                    .iter()
                    .take(idx)
                    .rev()
                    .take(6)
                    .filter_map(|token| clean_alpha_token(token, agg_stop))
                    .collect();
                before.reverse();
                trim_boundary_terms(&mut before);
                add_phrase_aliases(&before, &mut candidates);

                let mut after: Vec<String> = raw_tokens
                    .iter()
                    .skip(idx + 1)
                    .take(6)
                    .filter_map(|token| clean_alpha_token(token, agg_stop))
                    .collect();
                trim_boundary_terms(&mut after);
                add_phrase_aliases(&after, &mut candidates);
            }

            for (i, token) in raw_tokens.iter().enumerate() {
                if matches!(token.as_str(), "on" | "for" | "at" | "toward" | "towards") {
                    let words: Vec<String> = raw_tokens
                        .iter()
                        .skip(i + 1)
                        .take(5)
                        .filter_map(|raw| clean_alpha_token(raw, agg_stop))
                        .collect();
                    add_phrase_aliases(&words, &mut candidates);
                }
            }

            candidates
        }

        fn cents_to_dollars(cents: i64) -> String {
            let dollars = cents / 100;
            let rem = cents % 100;
            if rem == 0 {
                format!("${dollars}")
            } else {
                format!("${dollars}.{rem:02}")
            }
        }

        fn dollars_to_words(cents: i64) -> String {
            let dollars = cents / 100;
            match dollars {
                0 => "zero dollars".to_string(),
                1 => "one dollar".to_string(),
                2..=20 => format!("{} dollars", num_to_word(dollars as usize)),
                21..=99 => {
                    let tens = dollars / 10;
                    let ones = dollars % 10;
                    let tw = match tens {
                        2 => "twenty",
                        3 => "thirty",
                        4 => "forty",
                        5 => "fifty",
                        6 => "sixty",
                        7 => "seventy",
                        8 => "eighty",
                        9 => "ninety",
                        _ => "",
                    };
                    if ones == 0 {
                        format!("{tw} dollars")
                    } else {
                        format!("{tw}-{} dollars", num_to_word(ones as usize))
                    }
                },
                100..=999 => format!("{} hundred dollars", num_to_word((dollars / 100) as usize)),
                1000..=99999 => format!("{} thousand dollars", dollars / 1000),
                _ => format!("{dollars} dollars"),
            }
        }

        // Same stop words as emit_aggregate_neurons
        const AGG_STOP: &[&str] = &[
            "that",
            "this",
            "with",
            "from",
            "have",
            "will",
            "what",
            "when",
            "where",
            "which",
            "there",
            "their",
            "them",
            "they",
            "then",
            "been",
            "were",
            "some",
            "just",
            "also",
            "about",
            "into",
            "more",
            "than",
            "your",
            "here",
            "very",
            "well",
            "over",
            "back",
            "down",
            "would",
            "could",
            "should",
            "might",
            "does",
            "didn",
            "wasn",
            "aren",
            "isn",
            "hasn",
            "like",
            "want",
            "need",
            "think",
            "know",
            "said",
            "told",
            "went",
            "make",
            "made",
            "take",
            "took",
            "come",
            "came",
            "going",
            "really",
            "still",
            "even",
            "already",
            "always",
            "never",
            "every",
            "after",
            "before",
            "during",
            "while",
            "other",
            "another",
            "both",
            "first",
            "last",
            "next",
            "same",
            "such",
            "much",
            "many",
            "most",
            "because",
            "since",
            "through",
            "between",
            "under",
            "again",
            "help",
            "time",
            "year",
            "week",
            "month",
            "today",
            "yesterday",
            "tomorrow",
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
            "cost",
            "price",
            "paid",
            "spend",
            "spent",
            "total",
            "amount",
            "dollars",
        ];
        let agg_stop: HashSet<&str> = AGG_STOP.iter().copied().collect();

        // Build: topic phrase → [(session_id, dollars_on_supporting_lines)]
        let mut topic_session_dollars: HashMap<String, Vec<(String, Vec<i64>)>> = HashMap::new();
        let mut topic_aliases: HashMap<String, std::collections::BTreeSet<String>> = HashMap::new();
        let mut topic_snippets: HashMap<String, Vec<(String, String)>> = HashMap::new();
        let mut topic_seen_snippets: HashMap<String, HashSet<String>> = HashMap::new();

        let entries_snapshot: Vec<(String, PathBuf)> = self
            .entries
            .iter()
            .filter(|e| {
                matches!(e.kind, NeuronKind::Verbatim)
                    && !e.session_id.is_empty()
                    && !is_session_summary_path(&e.neuron_path)
            })
            .map(|e| (e.session_id.clone(), e.neuron_path.clone()))
            .collect();

        for (sid, neuron_path) in &entries_snapshot {
            let content = match std::fs::read_to_string(neuron_path) {
                Ok(c) => c,
                Err(_) => continue,
            };
            for line in content.lines() {
                let trimmed = line.trim();
                let lower = trimmed.to_ascii_lowercase();
                if !is_grounded_user_money_line(&lower) {
                    continue;
                }

                let body = trimmed.strip_prefix("User:").unwrap_or(trimmed).trim();
                let dollars = extract_dollars_on_line(body);
                if dollars.is_empty() {
                    continue;
                }

                let snippet: String = body.chars().take(120).collect();
                for topic in extract_topic_candidates(body, &agg_stop) {
                    let seen_key = format!("{sid}\n{snippet}");
                    if !topic_seen_snippets
                        .entry(topic.clone())
                        .or_default()
                        .insert(seen_key)
                    {
                        continue;
                    }
                    let entry = topic_session_dollars.entry(topic.clone()).or_default();
                    if let Some(se) = entry.iter_mut().find(|(s, _)| s == sid) {
                        se.1.extend_from_slice(&dollars);
                    } else {
                        entry.push((sid.clone(), dollars.clone()));
                    }

                    let aliases = topic_aliases.entry(topic.clone()).or_default();
                    aliases.insert(topic.clone());
                    for word in topic.split_whitespace() {
                        aliases.insert(word.to_string());
                    }

                    let snippets = topic_snippets.entry(topic).or_default();
                    if snippets.len() < 10
                        && !snippets
                            .iter()
                            .any(|(s, snip)| s == sid && snip == &snippet)
                    {
                        snippets.push((sid.clone(), snippet.clone()));
                    }
                }
            }
        }

        let ndir = neuron_dir(project_root);
        let mut staged = 0usize;

        for (topic, session_entries) in &topic_session_dollars {
            let sessions_with_dollars: Vec<_> = session_entries
                .iter()
                .filter(|(_, amounts)| !amounts.is_empty())
                .collect();
            let has_multi_amount_session = sessions_with_dollars
                .iter()
                .any(|(_, amounts)| amounts.len() >= 2);
            if sessions_with_dollars.len() < 2 && !has_multi_amount_session {
                continue;
            }

            let total_cents: i64 = sessions_with_dollars
                .iter()
                .flat_map(|(_, amounts)| amounts.iter().copied())
                .sum();
            if total_cents <= 0 {
                continue;
            }

            let total_str = cents_to_dollars(total_cents);
            let total_words = dollars_to_words(total_cents);
            let total_dollars = total_cents / 100;
            let session_count = sessions_with_dollars.len();
            let count_str = if session_count <= 20 {
                format!("{session_count} ({})", num_to_word(session_count))
            } else {
                format!("{session_count}")
            };

            let breakdown: String = sessions_with_dollars
                .iter()
                .map(|(sid, amounts)| {
                    let st: i64 = amounts.iter().sum();
                    format!(
                        "- {sid}: {} ({})\n",
                        cents_to_dollars(st),
                        amounts
                            .iter()
                            .map(|c| cents_to_dollars(*c))
                            .collect::<Vec<_>>()
                            .join(" + ")
                    )
                })
                .collect();
            let evidence_lines: String = topic_snippets
                .get(topic)
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .map(|(sid, snip)| format!("- {sid}: {snip}\n"))
                .collect();
            let aliases = topic_aliases
                .get(topic)
                .map(|aliases| aliases.iter().cloned().collect::<Vec<_>>())
                .unwrap_or_default();
            let alias_line = aliases.join(", ");
            let query_surface = format!(
                "how much did i spend on {topic}\n\
                 how much have i spent on {topic}\n\
                 what was the total for {topic}\n\
                 what is the total for {topic}\n\
                 total amount for {topic}\n\
                 total spent on {topic}\n\
                 how much money for {topic}\n\
                 {alias_line}\n"
            );
            let slug: String = topic
                .chars()
                .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
                .take(48)
                .collect();
            let content = format!(
                "# _arith_{slug}\n\
                 \n\
                 ## purpose\n\
                 Arithmetic aggregate: total dollar amount for topic \"{topic}\" across {count_str} sessions.\n\
                 \n\
                 ## topic\n\
                 {topic}\n\
                 Aliases: {alias_line}\n\
                 \n\
                 ## query_surface\n\
                 <!-- SECTION: query_surface -->\n\
                 {query_surface}\
                 <!-- /SECTION -->\n\
                 \n\
                 ## sum\n\
                 {total_str} ({total_words})\n\
                 \n\
                 ## breakdown\n\
                 {breakdown}\
                 \n\
                 ## evidence\n\
                 {evidence_lines}\
                 \n\
                 ## total\n\
                 Total: {total_str} across {count_str} sessions.\n\
                 Amount: {total_dollars}. Sum: {total_dollars}. Total dollars: {total_dollars}.\n\
                 In words: {total_words}.\n",
            );

            let fname = format!("_arith_{slug}.aggregate.md");
            let neuron_path = ndir.join(&fname);
            if let Err(e) = atomic_write(&neuron_path, content.as_bytes()) {
                eprintln!("[emit_arithmetic_aggregate] failed to write {fname}: {e}");
                continue;
            }
            staged += 1;
        }

        Ok(staged > 0)
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
            if !gone {
                continue;
            }
            // Hash the neuron file itself (the .context.md) to match against new file
            if let Ok(bytes) = std::fs::read(&entry.neuron_path) {
                let h = blake3::hash(&bytes).to_hex()[..16].to_string();
                orphaned.push((h, i));
            }
        }

        if orphaned.is_empty() {
            return;
        }

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
                if old_idx == &new_idx {
                    continue;
                } // same entry, skip
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
                                    if let Ok(mut new_meta) =
                                        serde_json::from_str::<NeuronMeta>(&new_meta_str)
                                    {
                                        if new_meta.uuid.is_none() {
                                            new_meta.uuid = Some(old_uuid.clone());
                                            if let Err(e) =
                                                atomic_write_json(&new_meta_path, &new_meta)
                                            {
                                                tracing::warn!(
                                                    "Failed to persist renamed neuron UUID for {}: {e}",
                                                    new_meta_path.display()
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                // Remove orphaned entry from sidecar (if it exists) so it doesn't re-appear
                let orphan_meta = ndir.join(
                    old_neuron_path
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .as_ref()
                        .replace(".context.md", ".context.json"),
                );
                if let Err(e) = std::fs::remove_file(&orphan_meta) {
                    if e.kind() != std::io::ErrorKind::NotFound {
                        tracing::warn!(
                            "Failed to remove orphaned renamed sidecar {}: {e}",
                            orphan_meta.display()
                        );
                    }
                }
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
                ep.ends_with(file_path.as_str()) || ep.contains(file_path.as_str())
            });

            if let Some(e) = entry {
                // Sort by term frequency descending, take top-N
                let mut term_freq_sorted: Vec<(&String, f32)> =
                    e.term_freq.iter().map(|(t, f)| (t, *f)).collect();
                term_freq_sorted
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
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
            entry
                .synapses
                .retain(|s| s.learned_weight > 0.05 || s.learned_weight <= 0.0);
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
    pub fn find_contradictions(&self, activated: &[PathBuf]) -> Vec<(PathBuf, PathBuf, String)> {
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
                if src != pf {
                    continue;
                }
            }
            for syn in syns {
                if syn.edge_type != SynapseType::Contradicts {
                    continue;
                }
                let a = src.min(&syn.target).clone();
                let b = src.max(&syn.target).clone();
                if seen.insert((a.clone(), b.clone())) {
                    pairs.push((a, b, syn.reason.trim_start_matches("← ").to_string()));
                }
            }
        }
        pairs
    }

    /// Load neuron body text for semantic consistency checks.
    ///
    /// When `path_filter` is given, returns only that neuron's body (for single-neuron
    /// scans). Without a filter, returns up to `limit` neuron bodies ordered by hit-rate
    /// descending so the most-used neurons are checked first.
    ///
    /// Used by `cortyx_check_consistency` to feed PureReason's semantic contradiction
    /// detector with raw neuron text.
    pub fn neuron_bodies_for_consistency(
        &self,
        path_filter: Option<&Path>,
        limit: usize,
    ) -> Option<Vec<String>> {
        if let Some(pf) = path_filter {
            let body = std::fs::read_to_string(pf).ok()?;
            return Some(vec![body]);
        }
        let mut entries: Vec<&BM25Entry> = self.entries.iter().collect();
        entries.sort_by(|a, b| {
            b.hit_count
                .partial_cmp(&a.hit_count)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let bodies: Vec<String> = entries
            .into_iter()
            .take(limit)
            .filter_map(|e| std::fs::read_to_string(&e.neuron_path).ok())
            .collect();
        Some(bodies)
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
}

mod synthetic;

impl NeuronIndex {
    pub(super) fn write_synthetic_answer(
        &self,
        slug: &str,
        task: &str,
        answer: &str,
        evidence: &[String],
    ) -> Option<PathBuf> {
        let path = neuron_dir(&self.project_root).join(format!("_answer_{slug}.md"));
        let mut content = format!("# Derived answer\n\nQuestion: {task}\nAnswer: {answer}\n");
        if !evidence.is_empty() {
            content.push_str("\n## evidence\n");
            for line in evidence.iter().take(3) {
                content.push_str("- ");
                content.push_str(line.trim());
                content.push('\n');
            }
        }
        atomic_write(&path, content.as_bytes()).ok()?;
        Some(path)
    }

    /// Trim a sorted list of paths to fit within `max_tokens`.
    fn trim_to_token_budget(&self, paths: Vec<PathBuf>, max_tokens: usize) -> Vec<PathBuf> {
        let mut result = Vec::new();
        let mut used = 0usize;
        for path in paths {
            let tokens = self.entry_by_path(&path).map(|e| e.tokens).unwrap_or(200);
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
            meta.tokens = estimate_context_tokens(&content);
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
        let project_name = root
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "project".to_string());

        let (pkg_name, pkg_version, pkg_authors, pkg_description, pkg_repo) =
            extract_manifest_metadata(root);

        let name = if !pkg_name.is_empty() {
            pkg_name
        } else {
            project_name.clone()
        };

        // _identity.context.md — generate if absent
        if !identity_path.exists() {
            let git_author = run_git_cmd(root, &["config", "user.name"]).unwrap_or_default();
            let git_email = run_git_cmd(root, &["config", "user.email"]).unwrap_or_default();

            let readme_intro =
                read_file_head(root, &["README.md", "README.rst", "README.txt"], 300);

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
            meta.tokens = estimate_context_tokens(&content);
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
            meta.tokens = estimate_context_tokens(&content);
            meta.module = Some("@wake_up".to_string());
            meta.last_updated = now_iso8601();
            atomic_write_json(&meta_path(&critical_path), &meta)?;
            self.index_neuron(&critical_path, &content, &meta);
            tracing::info!("S5: generated _critical_facts.context.md for '{name}'");
        }

        Ok(())
    }
}

mod helpers;
// Keep original `pub` visibility for items that were `pub` before extraction.
// The glob below would reduce them to `pub(super)`, breaking index/mod.rs re-exports.
#[cfg(test)]
pub use self::helpers::simple_overlap_score;
pub(super) use self::helpers::*;
pub use self::helpers::{
    dirty_path, infer_module, is_capsule_module, module_capsule_path, tokenize,
};
// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
