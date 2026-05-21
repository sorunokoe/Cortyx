//! CortyxServer implementation: Drop, benchmarks, and server utilities.

use super::super::*;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

pub async fn flush_provisional_hits_async(
    _index: &Arc<RwLock<NeuronIndex>>,
    provisional_hits: &Mutex<Vec<PathBuf>>,
) -> Result<usize> {
    let pending = {
        let mut prov = provisional_hits.lock().await;
        if prov.is_empty() {
            return Ok(0);
        }
        std::mem::take(&mut *prov)
    };
    Ok(pending.len())
}

pub fn flush_provisional_hits_blocking(
    _index: &Arc<RwLock<NeuronIndex>>,
    provisional_hits: &Mutex<Vec<PathBuf>>,
) -> Result<usize> {
    let pending = {
        let mut prov = provisional_hits.blocking_lock();
        if prov.is_empty() {
            return Ok(0);
        }
        std::mem::take(&mut *prov)
    };
    Ok(pending.len())
}

/// Clear provisional carry-over on unexpected process exit (STDIO EOF).
///
/// Explicit citation feedback must come from `close_task`, `record_hit`, or previous-response
/// evidence. `Drop` remains a last-resort cleanup for abnormal shutdown so this buffer does not
/// leak across clones.
impl Drop for CortyxServer {
    fn drop(&mut self) {
        // Only flush when this is the last CortyxServer instance.
        // CortyxServer derives Clone; rmcp may hold short-lived clones per request.
        if Arc::strong_count(&self.feedback) > 1 {
            return;
        }
        if tokio::runtime::Handle::try_current().is_ok() {
            tracing::debug!(
                "S2: skipping blocking provisional buffer clear from async Drop context"
            );
            return;
        }
        match flush_provisional_hits_blocking(&self.index, &self.feedback.provisional_hits) {
            Ok(0) => {},
            Ok(n) => tracing::info!("S2: Drop cleared {n} provisional paths on exit"),
            Err(e) => tracing::warn!("S2: failed to clear provisional buffer during Drop: {e}"),
        }
    }
}

impl CortyxServer {
    #[allow(dead_code)]
    pub fn for_benchmark(project_root: PathBuf, idx: NeuronIndex) -> Self {
        let index = Arc::new(RwLock::new(idx));

        Self {
            project_root,
            index,
            session: Arc::new(SessionState::default()),
            feedback: Arc::new(FeedbackBuffer::default()),
            inflight_bytes: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            fleet_registry: None,
            tool_router: CortyxServer::tool_router(),
        }
    }

    #[allow(dead_code)]
    pub async fn benchmark_get_contexts(&self, input: GetContextsInput) -> String {
        self.get_contexts(Parameters(input)).await
    }

    #[allow(dead_code)]
    pub async fn benchmark_cortyx(&self, input: CortyxInput) -> String {
        self.cortyx(Parameters(input)).await
    }

    #[allow(dead_code)]
    pub async fn benchmark_collaboration_status(&self, input: CollaborationStatusInput) -> String {
        self.collaboration_status(Parameters(input)).await
    }

    /// Return a project-relative display string for `path`.
    ///
    /// Strips the project root prefix so absolute internal filesystem paths
    /// (including usernames) are never exposed to MCP clients.
    pub(in crate::mcp) fn rel_display<'a>(&self, path: &'a Path) -> std::borrow::Cow<'a, str> {
        path.strip_prefix(&self.project_root)
            .unwrap_or(path)
            .to_string_lossy()
    }

    pub(in crate::mcp) async fn render_cortyx_capability_summary(&self) -> String {
        let (neuron_count, synapse_count, projection) = {
            let idx = self.index.read().await;
            (
                idx.neuron_count(),
                idx.synapse_count(),
                build_collaboration_projection(&idx, &self.project_root),
            )
        };
        let sync_statuses = load_collaboration_sync_statuses(&self.project_root);
        let pending_sync = sync_statuses
            .iter()
            .filter(|status| status.pending_incoming() || status.pending_outgoing())
            .count();
        let conflict_count: usize = sync_statuses
            .iter()
            .map(SyncTransportStatus::conflict_count)
            .sum();
        let collaboration_line =
            if projection.collaborators.is_empty() && projection.modules.is_empty() {
                " - collaboration: no tracked agent or shared-module state yet\n".to_string()
            } else {
                format!(
                    " - collaboration: {} collaborator(s), {} shared module(s)\n",
                    projection.collaborators.len(),
                    projection.modules.len()
                )
            };

        format!(
            "Cortyx capability summary\n\
             Default entrypoint: cortyx(task=\"...\") for auto routing, or cortyx(intent=\"context\", task=\"...\") when you need cache-safe raw context.\n\
             Call cortyx() with no args any time to see this summary again.\n\n\
             Available routes:\n\
             - context — retrieve the highest-signal local/project neurons for a task\n\
             - answer — reuse the retrieval path but return a concise answer layer\n\
             - wake_up — prime a person/agent session from diary + collaboration state\n\
             - agent_status — summarize one agent's focus, goal, blocker, and next step\n\
             - consistency — check a path's temporal KG surface for contradictions\n\n\
             Project snapshot:\n\
             - neurons: {neuron_count}, synapses: {synapse_count}\n\
             {collaboration_line}\
             - shared sync: {pending_sync} pending item(s), {conflict_count} conflict(s)\n\n\
             Examples:\n\
             - cortyx(task=\"trace the auth flow\")\n\
             - cortyx(intent=\"answer\", task=\"What is the reviewer's blocker?\", agent=\"reviewer\")\n\
             - cortyx(intent=\"wake_up\", agent=\"reviewer\")\n\
             - cortyx(intent=\"consistency\", path=\"src/auth.rs\")"
        )
    }

    pub(in crate::mcp) async fn ensure_context_handle(&self, requested: Option<&str>) -> String {
        requested
            .filter(|handle| !handle.trim().is_empty())
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| {
                let next = self
                    .session
                    .next_context_handle
                    .fetch_add(1, Ordering::Relaxed)
                    + 1;
                format!("ctx-{next}")
            })
    }

    pub(in crate::mcp) async fn load_context_snapshot(
        &self,
        handle: &str,
    ) -> Option<ContextSnapshot> {
        self.session
            .context_sessions
            .lock()
            .await
            .get(handle)
            .cloned()
    }

    pub(in crate::mcp) async fn store_context_snapshot(
        &self,
        handle: String,
        chunks: &[RenderedContextItem],
        overflow: &[RenderedContextItem],
    ) {
        let order = self
            .session
            .next_context_handle
            .fetch_add(1, Ordering::Relaxed)
            + 1;
        let snapshot = ContextSnapshot {
            order,
            chunks: chunks
                .iter()
                .map(|item| (item.path.clone(), item.fingerprint.clone()))
                .collect(),
            overflow: overflow
                .iter()
                .map(|item| (item.path.clone(), item.fingerprint.clone()))
                .collect(),
        };

        let mut sessions = self.session.context_sessions.lock().await;
        if sessions.len() >= 128 {
            if let Some(oldest) = sessions
                .iter()
                .min_by_key(|(_, snapshot)| snapshot.order)
                .map(|(handle, _)| handle.clone())
            {
                sessions.remove(&oldest);
            }
        }
        sessions.insert(handle, snapshot);
    }
}
