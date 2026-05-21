//! Domain state for persistence, filesystem layout, and dirty tracking.

use super::super::pipeline::PersistenceStateView;
use super::super::*;
use super::RetrievalState;
use std::path::Path;

/// Owned persistence state for `NeuronIndex`.
#[derive(Debug, Default)]
pub(crate) struct PersistenceState {
    pub(in crate::index) project_root: PathBuf,
    pub(in crate::index) pending_append_count: usize,
    pub(in crate::index) has_pending_updates: AtomicBool,
    pub(in crate::index) delta_base: AtomicUsize,
    pub(in crate::index) delta_dirty: AtomicBool,
    pub(in crate::index) structural_artifacts_dirty: AtomicBool,
    #[cfg(feature = "embed")]
    pub(in crate::index) embedding_rebuild_needed: AtomicBool,
    pub(in crate::index) dirty_sidecars: std::sync::Mutex<HashSet<PathBuf>>,
}

impl PersistenceState {
    #[allow(dead_code)]
    pub(in crate::index::core) fn new() -> Self {
        Self::default()
    }

    #[allow(dead_code)]
    pub(in crate::index::core) fn view(&self) -> PersistenceStateView<'_> {
        PersistenceStateView {
            project_root: &self.project_root,
            pending_append_count: self.pending_append_count,
            has_pending_updates: &self.has_pending_updates,
            delta_base: &self.delta_base,
            delta_dirty: &self.delta_dirty,
            structural_artifacts_dirty: &self.structural_artifacts_dirty,
            dirty_sidecars: &self.dirty_sidecars,
        }
    }

    pub(in crate::index::core) fn mark_sidecar_dirty(&self, path: &Path) {
        match self.dirty_sidecars.lock() {
            Ok(mut dirty_sidecars) => {
                dirty_sidecars.insert(path.to_path_buf());
            },
            Err(err) => tracing::warn!("Failed to lock dirty sidecar set: {err}"),
        }
    }

    pub(in crate::index::core) fn persist_feedback_sidecar(
        &self,
        neuron_path: &Path,
        retrieval: &RetrievalState,
    ) -> bool {
        let Some(&index) = retrieval.path_index.get(neuron_path) else {
            return true;
        };

        let meta_path = meta_path(neuron_path);
        let Ok(data) = std::fs::read_to_string(&meta_path) else {
            return true;
        };
        let Ok(mut meta) = serde_json::from_str::<NeuronMeta>(&data) else {
            return true;
        };

        meta.use_count = retrieval.entries[index]
            .use_count
            .load(std::sync::atomic::Ordering::Relaxed);
        meta.hit_count = retrieval.entries[index].hit_count;
        if let Err(err) = atomic_write_json(&meta_path, &meta) {
            tracing::warn!(
                "Failed to persist feedback sidecar for {}: {err}",
                meta_path.display()
            );
            return false;
        }

        true
    }

    pub(in crate::index::core) fn flush_dirty_sidecars(&self, retrieval: &RetrievalState) {
        let dirty_paths: Vec<PathBuf> = match self.dirty_sidecars.lock() {
            Ok(mut dirty_sidecars) => dirty_sidecars.drain().collect(),
            Err(err) => {
                tracing::warn!("Failed to lock dirty sidecar set for flush: {err}");
                return;
            },
        };

        if dirty_paths.is_empty() {
            return;
        }

        let failed_paths: Vec<PathBuf> = dirty_paths
            .into_iter()
            .filter(|path| !self.persist_feedback_sidecar(path, retrieval))
            .collect();

        if failed_paths.is_empty() {
            return;
        }

        match self.dirty_sidecars.lock() {
            Ok(mut dirty_sidecars) => {
                dirty_sidecars.extend(failed_paths);
            },
            Err(err) => tracing::warn!("Failed to restore dirty sidecars after flush: {err}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn default_is_empty() {
        let state = PersistenceState::default();
        assert!(state.project_root.as_os_str().is_empty());
        assert_eq!(state.pending_append_count, 0);
        assert!(state.dirty_sidecars.lock().expect("lock").is_empty());
    }

    #[test]
    fn atomic_flags_default_false_or_zero() {
        let state = PersistenceState::new();
        assert!(!state.has_pending_updates.load(Ordering::Relaxed));
        assert_eq!(state.delta_base.load(Ordering::Relaxed), 0);
        assert!(!state.delta_dirty.load(Ordering::Relaxed));
        assert!(!state.structural_artifacts_dirty.load(Ordering::Relaxed));
    }
}
