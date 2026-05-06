//! Type-state patterns for compile-time state validation.
//!
//! These state machines use the typestate pattern to ensure invalid state transitions
//! are caught at compile time rather than runtime.

use std::marker::PhantomData;

// ─── Index State Machine ──────────────────────────────────────────────────────

/// Index in uninitialized state.
pub struct Uninitialized;

/// Index is currently loading from disk.
pub struct Loading;

/// Index is ready for queries.
pub struct Ready;

/// Index data is stale and needs rebuilding.
pub struct Stale;

/// Represents the state of the neuron index.
///
/// Transitions:
/// - `Uninitialized → Loading` (via `begin_load()`)
/// - `Loading → Ready` (via `mark_ready()`)
/// - `Ready → Stale` (via `mark_stale()`)
/// - `Stale → Loading` (via `rebuild()`)
pub struct IndexState<S> {
    _state: PhantomData<S>,
}

impl IndexState<Uninitialized> {
    /// Create a new uninitialized index state.
    pub fn new() -> Self {
        Self {
            _state: PhantomData,
        }
    }

    /// Begin loading the index.
    pub fn begin_load(self) -> IndexState<Loading> {
        IndexState {
            _state: PhantomData,
        }
    }
}

impl Default for IndexState<Uninitialized> {
    fn default() -> Self {
        Self::new()
    }
}

impl IndexState<Loading> {
    /// Mark the index as ready after successful load.
    pub fn mark_ready(self) -> IndexState<Ready> {
        IndexState {
            _state: PhantomData,
        }
    }

    /// Mark as failed (returns to uninitialized).
    pub fn mark_failed(self) -> IndexState<Uninitialized> {
        IndexState {
            _state: PhantomData,
        }
    }
}

impl IndexState<Ready> {
    /// Mark the index as stale.
    pub fn mark_stale(self) -> IndexState<Stale> {
        IndexState {
            _state: PhantomData,
        }
    }
}

impl IndexState<Stale> {
    /// Begin rebuilding the index.
    pub fn rebuild(self) -> IndexState<Loading> {
        IndexState {
            _state: PhantomData,
        }
    }
}

// ─── Neuron Lifecycle State Machine ───────────────────────────────────────────

/// Neuron exists as a stub (no content yet).
pub struct Stub;

/// Neuron has been indexed and is searchable.
pub struct Indexed;

/// Neuron has been validated (content matches source).
pub struct Validated;

/// Neuron has been archived (no longer active).
pub struct Archived;

/// Represents the lifecycle state of a neuron.
///
/// Transitions:
/// - `Stub → Indexed` (via `index()`)
/// - `Indexed → Validated` (via `validate()`)
/// - `Validated → Archived` (via `archive()`)
/// - `Indexed → Stub` (via `revert()`)
pub struct NeuronLifecycle<S> {
    _state: PhantomData<S>,
}

impl NeuronLifecycle<Stub> {
    /// Create a new stub neuron.
    pub fn new() -> Self {
        Self {
            _state: PhantomData,
        }
    }

    /// Index the neuron.
    pub fn index(self) -> NeuronLifecycle<Indexed> {
        NeuronLifecycle {
            _state: PhantomData,
        }
    }
}

impl Default for NeuronLifecycle<Stub> {
    fn default() -> Self {
        Self::new()
    }
}

impl NeuronLifecycle<Indexed> {
    /// Validate the neuron.
    pub fn validate(self) -> NeuronLifecycle<Validated> {
        NeuronLifecycle {
            _state: PhantomData,
        }
    }

    /// Revert to stub state.
    pub fn revert(self) -> NeuronLifecycle<Stub> {
        NeuronLifecycle {
            _state: PhantomData,
        }
    }
}

impl NeuronLifecycle<Validated> {
    /// Archive the neuron.
    pub fn archive(self) -> NeuronLifecycle<Archived> {
        NeuronLifecycle {
            _state: PhantomData,
        }
    }
}

impl NeuronLifecycle<Archived> {
    // Terminal state - no transitions out
}

// ─── Sync State Machine ───────────────────────────────────────────────────────

/// Sync is idle (no active operations).
pub struct Idle;

/// Sync is actively syncing.
pub struct Syncing;

/// Sync encountered a conflict.
pub struct Conflicted;

/// Conflict has been resolved.
pub struct Resolved;

/// Represents the state of a sync operation.
///
/// Transitions:
/// - `Idle → Syncing` (via `begin_sync()`)
/// - `Syncing → Idle` (via `complete()`)
/// - `Syncing → Conflicted` (via `conflict()`)
/// - `Conflicted → Resolved` (via `resolve()`)
/// - `Resolved → Syncing` (via `retry()`)
pub struct SyncState<S> {
    _state: PhantomData<S>,
}

impl SyncState<Idle> {
    /// Create a new idle sync state.
    pub fn new() -> Self {
        Self {
            _state: PhantomData,
        }
    }

    /// Begin a sync operation.
    pub fn begin_sync(self) -> SyncState<Syncing> {
        SyncState {
            _state: PhantomData,
        }
    }
}

impl Default for SyncState<Idle> {
    fn default() -> Self {
        Self::new()
    }
}

impl SyncState<Syncing> {
    /// Complete the sync successfully.
    pub fn complete(self) -> SyncState<Idle> {
        SyncState {
            _state: PhantomData,
        }
    }

    /// Encounter a conflict during sync.
    pub fn conflict(self) -> SyncState<Conflicted> {
        SyncState {
            _state: PhantomData,
        }
    }
}

impl SyncState<Conflicted> {
    /// Resolve the conflict.
    pub fn resolve(self) -> SyncState<Resolved> {
        SyncState {
            _state: PhantomData,
        }
    }

    /// Abort and return to idle.
    pub fn abort(self) -> SyncState<Idle> {
        SyncState {
            _state: PhantomData,
        }
    }
}

impl SyncState<Resolved> {
    /// Retry the sync after resolution.
    pub fn retry(self) -> SyncState<Syncing> {
        SyncState {
            _state: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_state_transitions() {
        let state = IndexState::<Uninitialized>::new();
        let state = state.begin_load();
        let state = state.mark_ready();
        let state = state.mark_stale();
        let _state = state.rebuild();
    }

    #[test]
    fn neuron_lifecycle_transitions() {
        let lifecycle = NeuronLifecycle::<Stub>::new();
        let lifecycle = lifecycle.index();
        let lifecycle = lifecycle.validate();
        let _lifecycle = lifecycle.archive();
    }

    #[test]
    fn sync_state_transitions() {
        let state = SyncState::<Idle>::new();
        let state = state.begin_sync();
        let state = state.conflict();
        let state = state.resolve();
        let state = state.retry();
        let _state = state.complete();
    }

    #[test]
    fn index_state_failed_transition() {
        let state = IndexState::<Uninitialized>::new();
        let state = state.begin_load();
        let _state = state.mark_failed(); // Back to uninitialized
    }

    #[test]
    fn sync_state_abort() {
        let state = SyncState::<Idle>::new();
        let state = state.begin_sync();
        let state = state.conflict();
        let _state = state.abort(); // Back to idle
    }
}
