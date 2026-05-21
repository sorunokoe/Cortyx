//! Core index types - summary and metadata structures.

use super::family_prelude::*;

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
    pub(in crate::index) retrieval: RetrievalState,
    pub(in crate::index) feedback: FeedbackState,
    pub(in crate::index) persistence: PersistenceState,
    pub(in crate::index) watcher: WatcherState,
}

impl NeuronIndex {
    #[cfg(debug_assertions)]
    pub fn verify_invariants(&self) {
        let entry_paths: HashSet<_> = self
            .retrieval
            .entries
            .iter()
            .map(|entry| entry.neuron_path.as_path())
            .collect();

        for entry in &self.retrieval.entries {
            debug_assert!(
                self.retrieval.adjacency.contains_key(&entry.neuron_path),
                "adjacency missing entry for {:?}",
                entry.neuron_path
            );
        }
        for path in self.retrieval.adjacency.keys() {
            debug_assert!(
                entry_paths.contains(path.as_path()),
                "adjacency contains unknown entry for {:?}",
                path
            );
        }

        if self.persistence.has_pending_updates.load(Ordering::Acquire) {
            let has_dirty_sidecar = self
                .persistence
                .dirty_sidecars
                .lock()
                .map(|dirty| !dirty.is_empty())
                .unwrap_or(true);
            debug_assert!(
                self.persistence.delta_dirty.load(Ordering::Acquire) || has_dirty_sidecar,
                "pending updates require a full save or at least one dirty sidecar"
            );
        }
        // Add more invariants here as the struct evolves.
    }
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
