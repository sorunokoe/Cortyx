use anyhow::Result;
use std::path::PathBuf;
use std::sync::Arc;

use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use tokio::sync::{RwLock, mpsc};

use crate::index::NeuronIndex;
use crate::neuron::should_skip;

/// Starts a background file watcher that marks neurons stale when source files change.
///
/// Returns the watcher handle — it must be kept alive for the duration of the server.
/// Events are processed on a Tokio task; invalidation is serialized through the mpsc channel.
pub fn start_watcher(
    project_root: PathBuf,
    index: Arc<RwLock<NeuronIndex>>,
) -> Result<RecommendedWatcher> {
    let (tx, mut rx) = mpsc::channel::<PathBuf>(256);
    let root_for_task = project_root.clone();

    tokio::spawn(async move {
        while let Some(path) = rx.recv().await {
            let mut idx = index.write().await;
            if let Err(e) = idx.invalidate(&path) {
                tracing::warn!("Failed to invalidate {}: {e}", path.display());
            } else {
                tracing::debug!("Marked stale: {}", path.display());
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
            // Only process events within the project root.
            // strip_prefix can fail if the OS delivers an absolute path outside the root.
            let Ok(rel) = path.strip_prefix(&root_for_task) else {
                continue;
            };
            if !should_skip(rel) {
                // try_send avoids blocking the watcher callback; warn on overflow.
                if tx.try_send(path.clone()).is_err() {
                    tracing::warn!(
                        "Watcher channel full — invalidation dropped for: {}",
                        path.display()
                    );
                }
            }
        }
    })?;

    watcher.watch(&project_root, RecursiveMode::Recursive)?;
    tracing::info!("Watching {} for changes", project_root.display());
    Ok(watcher)
}
