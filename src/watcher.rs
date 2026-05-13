//! File system watcher for hot-reloading neurons.
//!
//! Monitors project files for changes and automatically updates the index.

use crate::error::Result;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use tokio::sync::{mpsc, RwLock};

use crate::index::NeuronIndex;
use crate::neuron::should_skip;

/// Debounce window — events within this interval are coalesced into a single batch.
///
/// A 50ms window absorbs editor "atomic save" sequences (write tmp → rename) that
/// arrive as two events in ~1ms, preventing duplicate invalidations per save.
const DEBOUNCE_MS: u64 = 50;
pub const HOT_PATCH_WATCH_SUMMARY: &str = "in-memory dirty-set hot patching is active";

/// Starts a background file watcher that marks neurons stale when source files change.
///
/// Events are batched with a 50ms debounce window to coalesce burst saves into a
/// single invalidation pass. An overflow buffer ensures no event is ever silently
/// dropped when the primary channel fills during burst saves (e.g. `cargo fmt`,
/// `git checkout`). Returns the watcher handle — keep alive for server lifetime.
///
/// Changed paths are inserted into `dirty_handle` (obtained via
/// `NeuronIndex::dirty_set_handle()`).  `compile_dirty()` drains that set atomically,
/// which eliminates the TOCTOU race that existed when the watcher wrote to `dirty.json`
/// and `compile_dirty()` simultaneously read, wrote, and deleted the same file.
pub fn start_watcher(
    project_root: PathBuf,
    index: Arc<RwLock<NeuronIndex>>,
    dirty_handle: Arc<Mutex<HashSet<PathBuf>>>,
) -> Result<RecommendedWatcher> {
    let (tx, mut rx) = mpsc::channel::<PathBuf>(256);
    // Overflow buffer: when the primary channel is full, events are queued here
    // and drained into the next batch cycle instead of being dropped.
    let overflow: Arc<Mutex<Vec<PathBuf>>> = Arc::new(Mutex::new(Vec::new()));
    let overflow_for_task = Arc::clone(&overflow);
    let root_for_task = project_root.clone();

    tokio::spawn(async move {
        let mut batch: Vec<PathBuf> = Vec::new();
        let debounce = Duration::from_millis(DEBOUNCE_MS);

        loop {
            // Drain overflow buffer first — these are events that couldn't fit in the
            // primary channel during a burst save (guaranteed delivery on next cycle).
            {
                let mut ov = match overflow_for_task.lock() {
                    Ok(guard) => guard,
                    Err(poisoned) => {
                        tracing::warn!(
                            "Watcher overflow mutex poisoned — recovering inner state: {poisoned}"
                        );
                        poisoned.into_inner()
                    },
                };
                batch.append(&mut *ov);
            }

            // Collect events for up to DEBOUNCE_MS or until channel closes.
            match tokio::time::timeout(debounce, rx.recv()).await {
                Ok(Some(path)) => {
                    batch.push(path);
                    // Drain any immediately available events into the same batch.
                    while let Ok(p) = rx.try_recv() {
                        batch.push(p);
                    }
                },
                Ok(None) => break, // channel closed
                Err(_timeout) => {
                    // Debounce window expired — flush batch if non-empty.
                    if batch.is_empty() {
                        continue;
                    }
                },
            }

            if batch.is_empty() {
                continue;
            }

            // Deduplicate paths within the batch before any work.
            batch.sort_unstable();
            batch.dedup();

            // Insert changed paths into the in-memory dirty set.
            // The set is protected by a Mutex so insertions from this task and drains
            // from compile_dirty() never race — no file I/O, no TOCTOU window.
            {
                match dirty_handle.lock() {
                    Ok(mut set) => {
                        set.extend(batch.iter().cloned());
                    },
                    Err(poisoned) => {
                        tracing::warn!("Watcher dirty_set mutex poisoned — recovering inner state");
                        let mut set = poisoned.into_inner();
                        set.extend(batch.iter().cloned());
                    },
                }
            }

            // Invalidate (mark stale) — fast, holds write lock for μs per path.
            {
                let mut idx = index.write().await;
                for path in &batch {
                    if let Err(e) = idx.invalidate(path) {
                        tracing::warn!("Failed to invalidate {}: {e}", path.display());
                    } else {
                        tracing::debug!("Marked stale: {}", path.display());
                    }
                }
            }
            batch.clear();

            // Hot-patch: re-index the changed files in-memory so the MCP server
            // immediately returns fresh content without a restart.
            //
            // compile_dirty() drains the in-memory dirty set (atomically via Mutex swap)
            // and re-processes only changed files — O(changed) not O(all).
            // The write lock is re-acquired only for the fast insertion phase.
            {
                let mut idx = index.write().await;
                match idx.compile_dirty() {
                    Ok(n) if n > 0 => tracing::info!("Hot-patched {n} neuron(s) in-memory."),
                    Ok(_) => {},
                    Err(e) => tracing::warn!("Hot-patch compile_dirty failed: {e}"),
                }
            }
        }
    });

    let mut watcher = notify::recommended_watcher(move |res: notify::Result<Event>| {
        let Ok(event) = res else { return };
        if !matches!(
            event.kind,
            EventKind::Modify(_) | EventKind::Create(_) | EventKind::Remove(_)
        ) {
            return;
        }
        for path in event.paths {
            let Ok(rel) = path.strip_prefix(&root_for_task) else {
                continue;
            };
            if !should_skip(rel) {
                // Fast path: primary channel. On overflow, push to the overflow
                // buffer so no invalidation is ever silently dropped.
                if tx.try_send(path.clone()).is_err() {
                    match overflow.lock() {
                        Ok(mut guard) => guard.push(path),
                        Err(poisoned) => {
                            tracing::warn!(
                                "Watcher overflow mutex poisoned — recovering inner state: {poisoned}"
                            );
                            poisoned.into_inner().push(path);
                        },
                    };
                }
            }
        }
    })?;

    watcher.watch(&project_root, RecursiveMode::Recursive)?;
    tracing::info!(
        "Watching {} for changes ({HOT_PATCH_WATCH_SUMMARY}, debounce={}ms)",
        project_root.display(),
        DEBOUNCE_MS
    );
    Ok(watcher)
}
