//! Domain state for watcher-owned dirty path tracking.

use super::super::pipeline::WatcherStateView;
use super::super::*;

/// Owned watcher state for `NeuronIndex`.
#[derive(Debug, Default)]
pub(crate) struct WatcherState {
    pub(in crate::index) dirty_set: std::sync::Arc<std::sync::Mutex<HashSet<PathBuf>>>,
}

impl WatcherState {
    #[allow(dead_code)]
    pub(in crate::index::core) fn new() -> Self {
        Self::default()
    }

    #[allow(dead_code)]
    pub(in crate::index::core) fn view(&self) -> WatcherStateView<'_> {
        WatcherStateView {
            dirty_set: &self.dirty_set,
        }
    }

    pub(in crate::index::core) fn handle(
        &self,
    ) -> std::sync::Arc<std::sync::Mutex<HashSet<PathBuf>>> {
        std::sync::Arc::clone(&self.dirty_set)
    }

    pub(in crate::index::core) fn extend(&self, paths: Vec<PathBuf>) {
        let mut dirty = self
            .dirty_set
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        dirty.extend(paths);
    }

    pub(in crate::index::core) fn drain_paths(&self) -> Vec<PathBuf> {
        let mut dirty = self
            .dirty_set
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let drained: HashSet<PathBuf> = std::mem::take(&mut *dirty);
        drained.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_wraps_empty_arc_mutex() {
        let state = WatcherState::default();
        assert!(state.dirty_set.lock().expect("lock").is_empty());
    }

    #[test]
    fn handle_shares_same_dirty_set() {
        let state = WatcherState::new();
        let handle = state.handle();
        handle
            .lock()
            .expect("lock")
            .insert(PathBuf::from("src/lib.rs"));
        assert!(state
            .dirty_set
            .lock()
            .expect("relock")
            .contains(&PathBuf::from("src/lib.rs")));
    }
}
