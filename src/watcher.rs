use anyhow::Result;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use tokio::sync::{RwLock, mpsc};

use crate::index::{NeuronIndex, dirty_path};
use crate::neuron::should_skip;

/// Debounce window — events within this interval are coalesced into a single batch.
///
/// A 50ms window absorbs editor "atomic save" sequences (write tmp → rename) that
/// arrive as two events in ~1ms, preventing duplicate invalidations per save.
const DEBOUNCE_MS: u64 = 50;

/// Starts a background file watcher that marks neurons stale when source files change.
///
/// Events are batched with a 50ms debounce window to coalesce burst saves into a
/// single invalidation pass. An overflow buffer ensures no event is ever silently
/// dropped when the primary channel fills during burst saves (e.g. `cargo fmt`,
/// `git checkout`). Returns the watcher handle — keep alive for server lifetime.
pub fn start_watcher(
    project_root: PathBuf,
    index: Arc<RwLock<NeuronIndex>>,
) -> Result<RecommendedWatcher> {
    let (tx, mut rx) = mpsc::channel::<PathBuf>(256);
    // Overflow buffer: when the primary channel is full, events are queued here
    // and drained into the next batch cycle instead of being dropped.
    let overflow: Arc<Mutex<Vec<PathBuf>>> = Arc::new(Mutex::new(Vec::new()));
    let overflow_for_task = Arc::clone(&overflow);
    let root_for_task = project_root.clone();
    let root_for_dirty = project_root.clone();

    tokio::spawn(async move {
        let mut batch: Vec<PathBuf> = Vec::new();
        let debounce = Duration::from_millis(DEBOUNCE_MS);

        loop {
            // Drain overflow buffer first — these are events that couldn't fit in the
            // primary channel during a burst save (guaranteed delivery on next cycle).
            {
                let mut ov = overflow_for_task.lock().unwrap();
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
                }
                Ok(None) => break, // channel closed
                Err(_timeout) => {
                    // Debounce window expired — flush batch if non-empty.
                    if batch.is_empty() {
                        continue;
                    }
                }
            }

            if batch.is_empty() {
                continue;
            }

            // Deduplicate paths within the batch before acquiring the write lock.
            batch.sort_unstable();
            batch.dedup();

            // Write changed source paths to dirty.json for incremental compile.
            // Append to any existing dirty set — multiple watcher cycles accumulate.
            let dirty_file = dirty_path(&root_for_dirty);
            let existing_dirty: Vec<PathBuf> = std::fs::read_to_string(&dirty_file)
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or_default();
            let mut merged: Vec<PathBuf> = existing_dirty;
            merged.extend(batch.iter().cloned());
            merged.sort_unstable();
            merged.dedup();
            if let Ok(json) = serde_json::to_string(&merged) {
                let _ = std::fs::create_dir_all(dirty_file.parent().unwrap_or(std::path::Path::new(".")));
                let _ = std::fs::write(&dirty_file, json);
            }

            let mut idx = index.write().await;
            for path in batch.drain(..) {
                if let Err(e) = idx.invalidate(&path) {
                    tracing::warn!("Failed to invalidate {}: {e}", path.display());
                } else {
                    tracing::debug!("Marked stale: {}", path.display());
                }
            }

            // Hot-patch: re-index the changed files in-memory so the MCP server
            // immediately returns fresh content without a restart.
            //
            // compile_dirty() reads dirty.json (written above), re-processes only
            // changed files (O(changed) not O(all)), then clears dirty.json.
            // The write lock is held for ≤100 ms; queued reads resume normally.
            match idx.compile_dirty() {
                Ok(n) if n > 0 => tracing::info!("Hot-patched {n} neuron(s) in-memory."),
                Ok(_) => {}
                Err(e) => tracing::warn!("Hot-patch compile_dirty failed: {e}"),
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
                    overflow.lock().unwrap().push(path);
                }
            }
        }
    })?;

    watcher.watch(&project_root, RecursiveMode::Recursive)?;
    tracing::info!("Watching {} for changes", project_root.display());
    Ok(watcher)
}

