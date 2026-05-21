use std::{collections::HashMap, path::PathBuf, sync::atomic::AtomicU64};
use tokio::sync::Mutex;

/// Per-session volatile state: context handles, vocabulary adaptation, path continuity.
///
/// Each field has independent synchronization so callers only lock what they touch.
/// `CortyxServer` holds this behind `Arc<SessionState>` for cheap clone across tool handlers.
#[derive(Default)]
pub struct SessionState {
    /// Server-side snapshots for delta-mode context emission.
    pub context_sessions: Mutex<HashMap<String, ContextSnapshot>>,
    /// Monotonic counter for context-handle IDs.
    pub next_context_handle: AtomicU64,
    /// Session-scoped term frequency for vocabulary adaptation (TRIZ Innovation A).
    pub session_tf: Mutex<HashMap<String, f32>>,
    /// Session-scoped path history with exponential decay (λ=0.8) for continuity boosts.
    pub session_path_history: Mutex<HashMap<PathBuf, f32>>,
}

#[derive(Clone, Default)]
pub(super) struct ContextSnapshot {
    pub order: u64,
    pub chunks: HashMap<PathBuf, String>,
    pub overflow: HashMap<PathBuf, String>,
}
