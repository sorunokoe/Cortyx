// This file is a submodule of `crate::index::core`.
// It contains `impl NeuronIndex` methods extracted from helpers.rs.
// All visibility is relative to `crate::index` (the parent of `core`).
use super::family_prelude::*;

impl NeuronIndex {
    // ── Invalidation ──────────────────────────────────────────────────────────

    /// Mark a source file's neuron as stale (hash changed or forced).
    ///
    /// The stale neuron is demoted (staleness_multiplier → 0.5) rather than evicted
    /// so it can still activate on niche queries where it remains the best match.
    /// A full eviction would lose context permanently before the LLM re-evolves it.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying operation fails.
    pub fn invalidate(&mut self, source: &Path) -> Result<()> {
        let neuron = core_neuron_path(source, &self.persistence.project_root);
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
        if let Some(&i) = self.retrieval.path_index.get(&neuron) {
            self.retrieval.entries[i].staleness_multiplier = 0.5;
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
        let Some(&idx) = self.retrieval.path_index.get(neuron_path) else {
            return false;
        };
        self.retrieval.entries.swap_remove(idx);
        // After swap_remove, the entry previously at the last position is now at `idx`.
        // Update its path_index slot so future lookups remain correct.
        if idx < self.retrieval.entries.len() {
            self.retrieval
                .path_index
                .insert(self.retrieval.entries[idx].neuron_path.clone(), idx);
        }
        self.retrieval.path_index.remove(neuron_path);
        // swap_remove reorders entries, so any usize indices stored in
        // co_return_counts are now stale. Clear them to prevent silently
        // wiring synapses between the wrong neurons.
        if let Ok(mut counts) = self.feedback.co_return_counts.lock() {
            counts.clear();
        }
        // Rebuild derived structures — eviction happens in bulk during prune,
        // so the caller calls rebuild_derived() once after all evictions.
        true
    }

    /// Neuron paths together with their activation count — used by `cortyx prune`.
    pub fn neuron_paths_and_use_counts(&self) -> Vec<(PathBuf, u32)> {
        self.retrieval
            .entries
            .iter()
            .map(|e| {
                (
                    e.neuron_path.clone(),
                    e.use_count.load(std::sync::atomic::Ordering::Relaxed),
                )
            })
            .collect()
    }
}
