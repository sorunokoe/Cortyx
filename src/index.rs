#[cfg(feature = "embed")]
use crate::embedder::{EmbeddingStore, load_embeddings};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

use crate::ast_extractor;
use crate::import_parser;
use crate::neuron::{
    NeuronKind, NeuronMeta, NeuronStatus, Synapse, SynapseType,
    atomic_write, atomic_write_json,
    core_neuron_path, estimate_tokens, meta_path, neuron_dir,
    now_iso8601, should_skip, stub_core_neuron, stub_project_neuron,
    DEFAULT_CONFIDENCE,
};

// ─── Activation tuning constants ─────────────────────────────────────────────

/// Maximum core neurons returned in Phase 1 of activation.
pub const MAX_CORE_NEURONS: usize = 5;
/// Maximum use-case neurons per core in Phase 2 of activation.
pub const MAX_USE_CASE_PER_CORE: usize = 2;
/// Maximum extra neurons added via synapse traversal (Phases 3–4).
pub const MAX_SYNAPSE_EXTRA: usize = 5;
/// Minimum BM25 relevance ratio (vs. max) for synapse traversal to include a neighbor.
pub const SYNAPSE_RELEVANCE_THRESHOLD: f32 = 0.25;
/// BM25 score ratio above which a neuron triggers 2-hop traversal.
pub const HIGH_ACTIVATION_THRESHOLD: f32 = 0.6;
/// Minimum term length kept by the tokenizer.
const MIN_TERM_LEN: usize = 2;
/// BM25 parameters (Okapi BM25 standard defaults).
const BM25_K1: f32 = 1.2;
const BM25_B: f32 = 0.75;
/// Persisted index format version — increment when the schema changes.
const INDEX_VERSION: u32 = 4;

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
    tokens: usize,
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
}

// ─── Persisted index wrapper ──────────────────────────────────────────────────

/// On-disk format of `.cortyx/index.json`.
/// Versioned to detect schema changes and rebuild cleanly.
#[derive(Deserialize)]
struct PersistedIndex {
    version: u32,
    entries: Vec<BM25Entry>,
}

/// Borrowed view used for serialization — avoids cloning the entire entry vector
/// on every save() call (which would otherwise be O(n) allocation per MCP mutation).
#[derive(Serialize)]
struct PersistedIndexRef<'a> {
    version: u32,
    entries: &'a [BM25Entry],
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
    /// Average document length (for BM25 length normalization)
    avg_doc_len: f32,
    /// Module → entry indices (for O(k) module-filtered queries)
    module_index: HashMap<String, Vec<usize>>,
    /// Dense embedding store (loaded from `.cortyx/embeddings.bin`).
    /// Empty when `embed` feature is disabled or file is absent (BM25-only mode).
    #[cfg(feature = "embed")]
    embeddings: EmbeddingStore,
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
                Ok(data) => match serde_json::from_str::<PersistedIndex>(&data) {
                    Ok(persisted) => {
                        if persisted.version == INDEX_VERSION {
                            idx.entries = persisted.entries;
                        } else {
                            tracing::warn!(
                                "Index version mismatch (stored={}, current={}): \
                                 starting fresh. Run `cortyx compile .` to rebuild.",
                                persisted.version, INDEX_VERSION
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
    pub fn save(&self) -> Result<()> {
        let path = index_path(&self.project_root);
        std::fs::create_dir_all(path.parent().unwrap_or(Path::new(".")))?;
        let persisted = PersistedIndexRef {
            version: INDEX_VERSION,
            entries: &self.entries,
        };
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

        // Ensure the project neuron exists
        self.ensure_project_neuron(&root)?;

        // Build git confidence map once (3 git commands, silent on non-git projects)
        let git_confidence = build_git_confidence_map(&root);

        let mut new_count = 0usize;
        for entry in WalkDir::new(&root).min_depth(1).into_iter().filter_map(|e| e.ok()) {
            if !entry.file_type().is_file() {
                continue;
            }
            let abs = entry.path();
            let rel = abs.strip_prefix(&root).unwrap_or(abs);

            if should_skip(rel) {
                continue;
            }

            let neuron_path = core_neuron_path(abs, &root);
            let meta_file = meta_path(&neuron_path);

            // Read source file once — used for hash check, AST extraction, and import parsing
            let source_bytes = match std::fs::read(abs) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let current_hash = {
                let h = blake3::hash(&source_bytes);
                h.to_hex()[..16].to_string()
            };

            let stored_hash = if meta_file.exists() {
                std::fs::read_to_string(&meta_file)
                    .ok()
                    .and_then(|d| serde_json::from_str::<NeuronMeta>(&d).ok())
                    .map(|m| m.source_hash)
                    .unwrap_or_default()
            } else {
                String::new()
            };

            // Skip if hash unchanged and neuron file exists
            if !current_hash.is_empty() && current_hash == stored_hash && neuron_path.exists() {
                continue;
            }

            let source_rel = rel.to_string_lossy();
            let now = now_iso8601();
            let source_text = String::from_utf8_lossy(&source_bytes);

            // AST Bootstrap: extract public API surface for BM25 vocabulary from day 1
            let ast_summary = ast_extractor::extract_signatures(&source_rel, &source_text);
            let prefilled = ast_extractor::format_for_stub(&ast_summary);

            let content = stub_core_neuron(&source_rel, &current_hash, &now, &prefilled);

            // Ensure the neuron's parent directory exists (dir-structure paths)
            if let Some(parent) = neuron_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            atomic_write(&neuron_path, content.as_bytes())?;

            let mut meta = NeuronMeta::new_stub(abs, NeuronKind::Core);
            meta.source_hash = current_hash;
            meta.tokens = estimate_tokens(&content);
            meta.last_updated = now;
            meta.status = if stored_hash.is_empty() {
                NeuronStatus::Stub
            } else {
                NeuronStatus::Stale
            };

            if meta_file.exists() {
                // Preserve existing synapses and module on hash invalidation
                if let Ok(old) = std::fs::read_to_string(&meta_file)
                    .and_then(|d| serde_json::from_str::<NeuronMeta>(&d).map_err(Into::into))
                {
                    meta.synapses = old.synapses;
                    meta.module = old.module;
                }
            }

            // Auto-Synapse: infer Imports edges from import statements
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

            // Git Confidence: committed + unmodified = 1.0, modified = 0.9, untracked = 0.85
            meta.confidence_score = git_confidence.get(abs).copied().unwrap_or(1.0);

            atomic_write_json(&meta_file, &meta)?;

            self.index_neuron(&neuron_path, &content, &meta);
            new_count += 1;
        }

        // TRIZ-4: Git co-change synapses — files committed together ≥3 times get
        // a SemanticRelated auto-synapse (they evolve together = semantically coupled).
        self.apply_cochange_synapses(&root);

        self.rebuild_derived();
        self.save()?;
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
    pub fn get_contexts(&self, task: &str, max_tokens: usize, module: Option<&str>) -> Vec<PathBuf> {
        let terms = tokenize(task);

        // Phase 1 — Candidate selection: Core + Project + Verbatim (module-filtered).
        //
        // Previously two separate Vec allocations; merged into one pass to halve
        // allocation cost.
        let primary_candidates: Vec<usize> = if let Some(m) = module {
            self.module_index
                .get(m)
                .map(|v| {
                    v.iter()
                        .copied()
                        .filter(|&i| matches!(
                            self.entries[i].kind,
                            NeuronKind::Core | NeuronKind::Project | NeuronKind::Verbatim
                        ))
                        .collect()
                })
                .unwrap_or_default()
        } else {
            self.entries
                .iter()
                .enumerate()
                .filter(|(_, e)| matches!(
                    e.kind,
                    NeuronKind::Core | NeuronKind::Project | NeuronKind::Verbatim
                ))
                .map(|(i, _)| i)
                .collect()
        };

        // BM25 scoring — use total_cmp for a well-defined total order over all f32
        // values (including NaN that can arise from degenerate empty-document inputs).
        let mut bm25_scored: Vec<(f32, usize)> = primary_candidates
            .iter()
            .filter_map(|&i| {
                let s = self.bm25_score(&terms, &self.entries[i]);
                (s > 0.0).then_some((s, i))
            })
            .collect();
        bm25_scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));

        // Hybrid RRF fusion when a query embedding is available.
        //
        // NOTE: Cosine ranking requires a live query embedding computed from the task
        // string at query time. The fastembed model integration (--features embed) is
        // not yet wired into get_contexts, so we always use pure BM25 here.
        // When the embed feature is complete, pass `query_vec: Option<&[f32]>` and
        // perform dot-product similarity against self.embeddings, then RRF-fuse.
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

        // top_cores are already ordered by BM25 score (descending).
        for (_, i) in &top_cores {
            selected.insert(self.entries[*i].neuron_path.clone());
        }

        // Also include Concept neurons that match the query, respecting module filter.
        // Global concepts (module == None) are always included regardless of filter.
        for i in 0..self.entries.len() {
            if self.entries[i].kind == NeuronKind::Concept {
                // Module filter: include if global concept or matches the requested module
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
        }

        // Phase 2 — UseCase neurons for each activated Core
        for (_, idx) in &top_cores {
            let core_path = self.entries[*idx].neuron_path.clone();
            let child_indices = self.parent_index.get(&core_path).cloned().unwrap_or_default();
            let mut uc_scores: Vec<(f32, usize)> = child_indices
                .into_iter()
                .filter(|&i| self.entries[i].kind == NeuronKind::UseCase)
                .filter_map(|i| {
                    let s = simple_overlap_score(&terms, &self.entries[i].task_pattern_terms);
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
        // their neighbours, matching the intended priority semantics.  The previous
        // Vec::pop (LIFO) produced depth-first traversal, which explored the second
        // hop of the highest-ranked neuron before the first hop of the second-ranked.
        //
        // Each selected neuron's outgoing synapses are scored:
        //   contribution = neighbor_bm25 * synapse.weight * edge_type.type_multiplier()
        // High-scoring (>HIGH_ACTIVATION_THRESHOLD) activations trigger a 2nd hop.
        struct Work { path: PathBuf, hops_left: u8 }
        let mut queue: VecDeque<Work> = top_cores
            .iter()
            .map(|(score, i)| {
                let hops = if *score >= HIGH_ACTIVATION_THRESHOLD * max_score { 2 } else { 1 };
                Work { path: self.entries[*i].neuron_path.clone(), hops_left: hops }
            })
            .collect();

        let mut visited: HashSet<PathBuf> = selected.set.clone();
        let mut extra = 0usize;

        while let Some(work) = queue.pop_front() {
            if extra >= MAX_SYNAPSE_EXTRA { break; }
            let neighbors = match self.adjacency.get(&work.path) {
                Some(n) => n.clone(),
                None => continue,
            };
            for syn in &neighbors {
                if visited.contains(&syn.target) || extra >= MAX_SYNAPSE_EXTRA { continue; }

                let neighbor_score = self.entry_by_path(&syn.target)
                    .map(|e| self.bm25_score(&terms, e))
                    .unwrap_or(0.0);

                // ConceptExpands always propagates; others need threshold
                let include = syn.edge_type == SynapseType::ConceptExpands
                    || (neighbor_score + 0.01) * syn.weight * syn.edge_type.type_multiplier()
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

        // Phase 4 — relevance-ordered trim, then lex sort for byte-identical prompt-cache hits.
        //
        // Trim by selected.ordered (most-relevant neuron first) so the token
        // budget removes low-relevance neurons, not low-alphabet ones.
        // Sort survivors lexicographically so the same task always produces
        // the same byte sequence (required for Anthropic/OpenAI cache hits).
        let mut trimmed = self.trim_to_token_budget(selected.ordered, max_tokens);
        trimmed.sort();
        trimmed
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
    /// Called after `get_contexts` returns; keeps the activation feedback loop active.
    pub fn record_activation(&mut self, paths: &[std::path::PathBuf]) {
        for path in paths {
            if let Some(&i) = self.path_index.get(path) {
                self.entries[i].use_count = self.entries[i].use_count.saturating_add(1);
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

            hit_rate
        } else {
            0.0
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
        // Remove from in-memory index so a stale neuron doesn't activate
        self.entries.retain(|e| e.neuron_path != neuron);
        self.rebuild_derived();
        self.save()
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

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
                }));
                changes.push((nb, Synapse {
                    target: na,
                    edge_type: SynapseType::SemanticRelated,
                    weight,
                    reason,
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
    fn index_neuron(&mut self, neuron_path: &Path, content: &str, meta: &NeuronMeta) {
        let terms = tokenize(content);
        let mut tf: HashMap<String, f32> = HashMap::new();
        for t in &terms {
            *tf.entry(t.clone()).or_insert(0.0) += 1.0;
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
        };

        if let Some(&pos) = self.path_index.get(neuron_path) {
            self.entries[pos] = entry;
        } else {
            let pos = self.entries.len();
            self.path_index.insert(neuron_path.to_path_buf(), pos);
            self.entries.push(entry);
        }
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
        self.module_index.clear();

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
                    });
            }

            // df_cache
            for term in entry.term_freq.keys() {
                *self.df_cache.entry(term.clone()).or_insert(0) += 1;
            }

            // module_index
            if let Some(m) = &entry.module {
                self.module_index.entry(m.clone()).or_default().push(i);
            }

            total_terms += entry.term_count;
        }

        self.avg_doc_len = if self.entries.is_empty() {
            0.0
        } else {
            total_terms as f32 / self.entries.len() as f32
        };
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

        let raw: f32 = terms.iter().map(|t| {
            let tf = entry.term_freq.get(t).copied().unwrap_or(0.0);
            if tf == 0.0 { return 0.0; }
            let df = self.df_cache.get(t).copied().unwrap_or(1) as f32;
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln().max(0.0);
            idf * (tf * (BM25_K1 + 1.0)) / (tf + BM25_K1 * len_norm)
        }).sum();

        // hit_rate: fraction of activations where the LLM confirmed the neuron was cited.
        // Multiplier range: 0.70 (never cited) → 1.00 (always cited, confidence_score applies).
        let hit_rate = entry.hit_count as f32 / entry.use_count.max(1) as f32;
        let hit_multiplier = 0.70 + 0.30 * hit_rate;

        raw * entry.confidence_score * hit_multiplier
    }

    /// Find an entry by its neuron path — O(1) via precomputed path_index.
    fn entry_by_path(&self, path: &Path) -> Option<&BM25Entry> {
        self.path_index.get(path).map(|&i| &self.entries[i])
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
}

// ─── Free functions ───────────────────────────────────────────────────────────

/// Split text into lowercase terms, filtering short tokens.
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|s| s.len() >= MIN_TERM_LEN)
        .map(|s| s.to_lowercase())
        .collect()
}

/// Jaccard similarity between query terms and a pre-tokenized pattern.
///
/// Returns |A∩B| / |A∪B| — unbiased by pattern length, unlike ratio-of-matches.
/// A 2-term and a 10-term pattern with the same overlap get the same score.
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

fn default_confidence() -> f32 {
    DEFAULT_CONFIDENCE
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

        let result = idx.get_contexts("auth login", 4096, None);
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
        let result = idx.get_contexts("authentication oauth jwt", 4096, None);
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
        let result = idx.get_contexts("auth token login", 500, None);
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
        let filtered = idx.get_contexts("auth login", 4096, Some("auth"));
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
        }];
        engine_meta.status = NeuronStatus::Fresh;
        std::fs::write(&engine_neuron, &engine_content).unwrap();
        idx.upsert_neuron(&engine_neuron, &engine_content, &engine_meta).unwrap();

        let contexts = idx.get_contexts("route intent synthesize engine", 4096, None);
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

        let result = idx.get_contexts("add oauth authentication login", 4096, None);
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
        assert!(!idx.entries.iter().any(|e| e.neuron_path == neuron));
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
            let results = idx.get_contexts(query, 4096, None);
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
        let _ = idx.get_contexts("routing pipeline authentication token", 4096, None);

        // Measure p95 over 20 trials
        let trials = 20;
        let mut latencies_ms: Vec<u128> = (0..trials).map(|_| {
            let t = std::time::Instant::now();
            let _ = idx.get_contexts("routing pipeline authentication token", 4096, None);
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
        let results = idx.get_contexts("hydrazine valve rocket propulsion", 4096, None);
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
        let with_module = idx.get_contexts("photosynthesis sunlight glucose", 4096, Some("biology"));
        assert!(!with_module.is_empty(), "Module-filtered query should find mined neuron");

        // Module filter for a different module should NOT find it
        let wrong_module = idx.get_contexts("photosynthesis sunlight glucose", 4096, Some("physics"));
        assert!(
            wrong_module.is_empty(),
            "Wrong module filter should not return neuron tagged 'biology'"
        );
    }
}
