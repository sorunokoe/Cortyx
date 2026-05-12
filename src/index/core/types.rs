//! Core index types - summary and metadata structures.

use super::*;

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

// ─── NeuronIndex ─────────────────────────────────────────────────────────────

/// The in-memory semantic index — loaded from `.cortyx/index.json` on startup.
///
/// All search operations run entirely in RAM (<10ms for <10k neurons).
/// Persisted to disk after every compile or mutation (evolve, synapse, extract).
#[derive(Debug, Default)]
pub struct NeuronIndex {
    pub(in crate::index) project_root: PathBuf,
    pub(in crate::index) entries: Vec<BM25Entry>,
    /// Synapse graph: neuron_path → outgoing + incoming typed synapse edges
    pub(in crate::index) adjacency: HashMap<PathBuf, Vec<Synapse>>,
    /// O(1) lookup: neuron_path → index in `entries`
    pub(in crate::index) path_index: HashMap<PathBuf, usize>,
    /// Parent neuron path → child entry indices (for UseCase lookup)
    pub(in crate::index) parent_index: HashMap<PathBuf, Vec<usize>>,
    /// Precomputed document frequency for each term (used for BM25 IDF)
    pub(in crate::index) df_cache: HashMap<String, usize>,
    /// Posting list: term → [entry indices containing that term].
    ///
    /// Built during rebuild_derived() alongside df_cache.
    /// Used in get_contexts() to compute the candidate set in O(|terms|) time —
    /// only entries that contain at least one query term are scored, reducing
    /// BM25 scoring from O(n) to O(|candidates|).  Typically |candidates| << n
    /// for sparse queries, which is the common case.
    pub(in crate::index) posting_list: HashMap<String, Vec<usize>>,
    /// Average document length (for BM25 length normalization)
    pub(in crate::index) avg_doc_len: f32,
    /// Average Verbatim-neuron document length (for BM25 length normalization of conversation chunks).
    /// Computed separately from avg_doc_len to avoid Concept/entity neurons (very short, ~150 tokens)
    /// artificially depressing the average and over-penalizing long session chunks.
    pub(in crate::index) avg_verbatim_doc_len: f32,
    /// Module → entry indices (for O(k) module-filtered queries)
    pub(in crate::index) module_index: HashMap<String, Vec<usize>>,
    /// Vocabulary bridge (S2): module_fragment → set of identifier terms from that module.
    ///
    /// Built during rebuild_derived(). At query time, zero-match BM25 queries are
    /// expanded with the identifier vocabulary of any module whose name substring-matches
    /// a query term. Resolves the lexical gap between human language ("authentication")
    /// and code identifiers ("auth_guard", "jwt_validate") without any model download.
    pub(in crate::index) vocab_bridge: HashMap<String, HashSet<String>>,
    /// B1: Morphemic trie bridge — sub-token → all tokens containing that sub-token.
    ///
    /// Built during rebuild_derived() by splitting all identifier tokens on `_` and
    /// camelCase boundaries. At query time, query terms that don't match any neuron
    /// directly are expanded through this map: "auth" → ["auth_guard", "authentication",
    /// "oauth_token"]. Reduces vocabulary gap from ~3% to ~0.3%.
    pub(in crate::index) morpheme_map: HashMap<String, Vec<String>>,
    /// B2: Synonym cloud from co-activation history (TRIZ R14).
    ///
    /// Tracks how often query terms co-activate each neuron. After ≥30 co-activations,
    /// a term is promoted to the neuron's `synonym_cloud` in BM25Entry.
    /// Map: neuron_path → HashMap<term, coactivation_count>
    /// Persisted in `.cortyx/coactivation.json` so synonym-cloud promotion can survive
    /// normal CLI/server restarts. Synonym clouds in BM25Entry are still persisted directly.
    pub(in crate::index) coactivation_counts: HashMap<PathBuf, HashMap<String, u32>>,
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
    /// Keys are (path_index_a, path_index_b) with a ≤ b — using the usize IDs from
    /// `path_index` eliminates per-lookup PathBuf hashing (O(path_len) → O(8 bytes)).
    pub(in crate::index) co_return_counts: std::sync::Mutex<HashMap<(usize, usize), u32>>,
    /// F2: Session token utilization history — last 5 sessions' [tokens_used, tokens_budget].
    ///
    /// Persisted via PersistedIndexRef so budget adaptation accumulates across restarts.
    pub(in crate::index) session_utilization: Vec<[usize; 2]>,
    /// R21 T6: Session index — maps session_id → entry indices for session-level grouping.
    ///
    /// Built during rebuild_derived(). At retrieval, when a Verbatim neuron enters the
    /// top-3 results, the top-2 BM25-scored siblings from the same session are injected
    /// as overflow candidates. Enables counting/multi-session queries to surface related
    /// evidence from the same session cluster without extra BM25 computation.
    /// Not persisted — rebuilt from BM25Entry.session_id on each load.
    pub(in crate::index) session_index: HashMap<String, Vec<usize>>,
    /// P1-A: PMI semantic neighbors — loaded from cooccurrence.json without a global cap.
    ///
    /// Unlike `vocab_bridge` (which uses substring matching for code module fragments),
    /// this map uses exact-key lookup O(1) for conversation vocabulary expansion.
    /// Key: term (≥4 chars). Value: up to 5 high-PMI neighbors from the same corpus.
    /// Loaded at rebuild_derived() time; not persisted — rebuilt on each load.
    pub(in crate::index) pmi_neighbors: HashMap<String, Vec<String>>,
    /// Dense embedding store (loaded from `.cortyx/embeddings.bin`).
    /// Empty when `embed` feature is disabled or file is absent (BM25-only mode).
    #[cfg(feature = "embed")]
    pub(in crate::index) embeddings: EmbeddingStore,
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
    pub(in crate::index) idf_n: usize,
    /// Count of entries inserted (not updated) since the last rebuild_derived() call.
    /// Used to take the fast incremental delta path in rebuild_derived() instead of
    /// clearing and rebuilding all HashMaps from scratch.
    /// Not persisted — resets to 0 via Default on every load (full rebuild runs then).
    pub(in crate::index) pending_append_count: usize,
    /// True if any existing entry was updated (not just appended) since the last
    /// rebuild_derived() call.  When true the full rebuild path is taken so that
    /// df_cache / posting_list stay consistent with the changed entries.
    pub(in crate::index) has_pending_updates: bool,
    /// S4 delta-append: entries.len() at last full index.json write.
    /// Persisted in the activation cache so delta-append mode activates on subsequent process starts.
    /// 0 means no full save yet; the next save() establishes the delta baseline.
    pub(in crate::index) wal_base: AtomicUsize,
    /// S4 delta-append: true when any existing entry was updated since the last full save.
    /// Forces a full index.json rewrite so in-place mutations are never lost.
    pub(in crate::index) needs_full_save: AtomicBool,
    /// Set when structural derived state changes and the module shards / cache generation
    /// should be refreshed on the next save(). Feedback-only saves leave this false.
    pub(in crate::index) structural_artifacts_dirty: AtomicBool,
    /// In-memory dirty set: source paths changed by the file watcher and not yet
    /// compiled into the index.
    ///
    /// The watcher inserts into this set; `compile_dirty()` drains it atomically.
    /// Eliminates the `dirty.json` TOCTOU race: file-system reads/writes are replaced
    /// by a single mutex-protected in-memory swap.
    pub(in crate::index) dirty_set: std::sync::Arc<std::sync::Mutex<HashSet<PathBuf>>>,
}

// ─── Parallel compile helper ──────────────────────────────────────────────────

/// Result of processing a single source file in the parallel compile phase.
///
/// Returned by `process_source_file` (a free function — no `&self` access) so
/// multiple files can be processed concurrently via `rayon::par_iter()`.
/// The sequential batch-insert phase calls `index_neuron` on each result.
pub(in crate::index) struct CompiledFile {
    pub(in crate::index) neuron_path: PathBuf,
    /// Content of the neuron stub (new or regenerated).
    pub(in crate::index) content: String,
    /// Updated `NeuronMeta` to be written to the `.context.json` sidecar.
    pub(in crate::index) meta: NeuronMeta,
}
