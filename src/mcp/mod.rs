//! Model Context Protocol (MCP) server implementation.
//!
//! Exposes Cortyx functionality via the MCP protocol for LLM integration.

use crate::error::Result;
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{Implementation, ServerCapabilities, ServerInfo},
    tool_handler, ServerHandler, ServiceExt,
};
#[cfg(test)]
use std::collections::HashMap;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::agent_memory::{
    has_structured_diary_fields, parse_structured_diary_entry, render_structured_diary_entry,
    summarize_structured_diary_entry, StructuredDiaryEntry,
};
use crate::answer_plane;
use crate::collaboration_kernel::{
    agent_entity_name, project_collaboration_state, CollaborationAttention,
    CollaborationDiaryRecord, CollaborationEvidenceSummary, CollaborationStateProjection,
    CollaborationTimelineEvent, CollaborationTimelineKind, CollaboratorSummary,
    ModuleCollaborationState, AGENT_ACTION_PREDICATE, AGENT_BLOCKER_PREDICATE,
    AGENT_DEPENDS_ON_PREDICATE, AGENT_FOCUS_PREDICATE, AGENT_GOAL_PREDICATE,
    AGENT_NEXT_STEP_PREDICATE, AGENT_OUTCOME_PREDICATE, AGENT_RELATED_ENTITY_PREDICATE,
    AGENT_STATUS_PREDICATE,
};
use crate::index::{is_capsule_module, module_capsule_path, tokenize, NeuronIndex};
use crate::kg;
use crate::miner;
use crate::neuron::provenance::{
    record_content_provenance_edit, ProvenanceEdit, ProvenanceOperation, ProvenanceSource,
};
use crate::neuron::{
    atomic_write, atomic_write_json, core_neuron_path, estimate_context_tokens, hash_file,
    latest_shadow, meta_path, neuron_dir, now_iso8601, parse_sections, parse_synapses_from_content,
    pop_shadow, push_shadow, replace_section, unix_secs_to_datetime, validate_relative_path,
    NeuronKind, NeuronMeta, NeuronStatus, Synapse, SynapseType,
};
use crate::reasoner::{ReasoningReport, TraversalOptions};
use crate::sync_transport::{sync_transport_dir, SyncTransportRepository, SyncTransportStatus};
use crate::verify_gate;
use crate::watcher;

mod feedback_buffer;
mod helpers;
mod session_state;
mod tools;
mod types;

use self::feedback_buffer::FeedbackBuffer;
use self::helpers::*;
use self::session_state::{ContextSnapshot, SessionState};
use self::tools::tool_router as build_tool_router;
pub use types::*;

// ─── MCP Server ───────────────────────────────────────────────────────────────

/// Maximum byte size for content fields in MCP tool inputs.
///
/// Prevents OOM from a runaway or malicious LLM agent submitting unbounded payloads.
const MAX_CONTENT_BYTES: usize = 1_048_576; // 1 MB

/// Maximum byte length for task/query strings.
const MAX_TASK_BYTES: usize = 4_096;

/// Maximum total in-flight bytes across all concurrent tool-handler executions.
///
/// Bounds aggregate memory when multiple LLM agents share a single Cortyx process.
/// The check is advisory — it guards response-building work, not input deserialization.
const MAX_INFLIGHT_BYTES: usize = 64 * 1_048_576; // 64 MB

/// RAII guard that releases reserved in-flight bytes when the handler finishes.
pub(crate) struct InflightGuard<'a>(&'a std::sync::atomic::AtomicUsize, usize);

impl Drop for InflightGuard<'_> {
    fn drop(&mut self) {
        self.0
            .fetch_sub(self.1, std::sync::atomic::Ordering::AcqRel);
    }
}

#[derive(Clone)]
pub struct CortyxServer {
    project_root: PathBuf,
    index: Arc<RwLock<NeuronIndex>>,
    session: Arc<SessionState>,
    feedback: Arc<FeedbackBuffer>,
    frozen: bool,
    /// Running sum of bytes currently being processed across all concurrent handlers.
    /// Handlers that build large responses increment this before work and decrement after.
    inflight_bytes: Arc<std::sync::atomic::AtomicUsize>,
    fleet_registry: Option<Arc<crate::fleet::FleetRegistry>>,
    // Kept for the rmcp macro-generated dispatch table; not called directly.
    #[allow(dead_code)]
    tool_router: ToolRouter<Self>,
}

#[derive(Clone)]
struct RenderedContextItem {
    path: PathBuf,
    rendered: String,
    fingerprint: String,
}

#[derive(Default)]
struct DeltaSelection {
    emitted: Vec<RenderedContextItem>,
    unchanged: usize,
    removed: usize,
}

impl CortyxServer {
    fn tool_router() -> ToolRouter<Self> {
        build_tool_router()
    }
}

#[tool_handler]
impl ServerHandler for CortyxServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("cortyx", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "Cortyx: MCP-native context delivery engine for coding agents. \
                Prefer cortyx(task=..., intent?='auto') as the default entrypoint at task start, \
                or call cortyx() with no args for a capability summary; \
                use cortyx_get_contexts when you want the narrow retrieval surface explicitly. \
                At task end call cortyx_close_task(response_text) to record response-grounded hits, \
                or cortyx_record_hit(path, was_cited) for fine control. \
                Use cortyx_diary_write(agent, ..., title?, status?, goal?, next_step?, blocker?, \
                outcome?, entities?, depends_on?) to persist agent state, \
                cortyx_agent_status(agent) or cortyx_wake_up(person?, agent?) for handoff/priming, \
                cortyx_evolve_section to improve one section, cortyx_extract_from_raw to save a \
                proven pattern, and cortyx_create_synapse to link related neurons.",
            )
    }
}

// ─── Server entrypoint ────────────────────────────────────────────────────────

/// Derive a deterministic startup offset (0..3600s) from the project root path.
/// Spreads fleet processes without randomness — same root always gets the same offset.
fn decay_jitter_secs(project_root: &Path) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    project_root.hash(&mut hasher);
    hasher.finish() % 3600
}

/// Start the MCP server on STDIO (compatible with Claude Code, Cursor, Codex, Windsurf).
///
/// # Errors
///
/// Returns an error if the underlying operation fails.
pub async fn serve(project: Option<PathBuf>, frozen: bool) -> Result<()> {
    let project_root = match project {
        Some(p) => p.canonicalize().unwrap_or(p),
        None => std::env::current_dir()?,
    };
    tracing::info!("Starting Cortyx MCP server for: {}", project_root.display());
    if frozen {
        tracing::info!("serve --frozen: feedback writes disabled");
    }

    let mut idx = NeuronIndex::load_or_create(&project_root)?;

    // Auto-compile on first run — turns `cortyx serve` into a one-step setup.
    if idx.neuron_count() == 0 {
        tracing::info!("No neurons found — running initial compile...");
        let count = idx.compile()?;
        tracing::info!("Auto-compiled {count} neurons (AST Bootstrap + Auto-Synapse active)");
        eprintln!("✓ Cortyx: auto-compiled {count} neurons on first run.");
    }

    // S-VII (R16): Apply synapse temporal decay at startup — self-cleaning graph.
    let (decayed, pruned) = idx.apply_synapse_decay();
    if decayed > 0 || pruned > 0 {
        tracing::info!(
            decayed,
            pruned,
            "S-VII: synapse temporal decay applied at startup"
        );
    }

    let index = Arc::new(RwLock::new(idx));

    // P19 Periodic Action: re-apply synapse LTD every 24 h during active serve.
    // Ensures synapses go stale in long-running servers without a restart.
    {
        let index = Arc::clone(&index);
        let project_root_for_decay = project_root.clone();
        tokio::spawn(async move {
            let jitter = decay_jitter_secs(&project_root_for_decay);
            tokio::time::sleep(std::time::Duration::from_secs(jitter)).await;
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(86_400));
            loop {
                interval.tick().await;
                let mut idx = index.write().await;
                idx.apply_synapse_decay();
            }
        });
    }

    // S-IV (R16): Auto-fetch global concepts — fire-and-forget so a slow/hanging
    // network call never blocks the MCP server from serving its first request.
    // Skipped entirely when FETCH_HEAD is younger than 1 hour (staleness gate).
    {
        let global_dir = crate::global_index::global_dir();
        if global_dir.join(".git").exists() {
            tokio::spawn(async move {
                // Staleness gate: skip fetch if we pulled within the last hour.
                let fetch_head = global_dir.join(".git/FETCH_HEAD");
                let stale = fetch_head
                    .metadata()
                    .and_then(|m| m.modified())
                    .map(|t| {
                        t.elapsed().unwrap_or(std::time::Duration::MAX)
                            > std::time::Duration::from_secs(3600)
                    })
                    .unwrap_or(true);
                if !stale {
                    tracing::debug!("S-IV: global concepts cache fresh — skipping fetch");
                    return;
                }

                let remote_url = tokio::process::Command::new(crate::git_util::git_binary())
                    .args(["remote", "get-url", "origin"])
                    .current_dir(&global_dir)
                    .output()
                    .await
                    .ok()
                    .filter(|o| o.status.success())
                    .and_then(|o| String::from_utf8(o.stdout).ok());

                let Some(url) = remote_url else { return };
                let url = url.trim();

                // Allowlist: only pull from trusted HTTPS hosts.
                // SSH git@ URLs from these same hosts are also accepted.
                // This prevents a compromised or user-misconfigured global-concepts
                // directory from silently pulling from an arbitrary server.
                let trusted = [
                    "https://github.com/",
                    "https://gitlab.com/",
                    "git@github.com:",
                    "git@gitlab.com:",
                ];
                if !trusted.iter().any(|prefix| url.starts_with(prefix)) {
                    tracing::warn!(
                        remote_url = url,
                        "S-IV: global concepts auto-fetch skipped — remote URL not in allowlist"
                    );
                    return;
                }

                // 5-second hard timeout — network hang cannot stall the server.
                let pull_fut = async {
                    tokio::process::Command::new(crate::git_util::git_binary())
                        .args(["pull", "--ff-only", "origin", "main"])
                        .current_dir(&global_dir)
                        .output()
                        .await
                        .or(tokio::process::Command::new(crate::git_util::git_binary())
                            .args(["pull", "--ff-only", "origin", "master"])
                            .current_dir(&global_dir)
                            .output()
                            .await)
                };
                match tokio::time::timeout(std::time::Duration::from_secs(5), pull_fut).await {
                    Ok(Ok(o)) if o.status.success() => {
                        tracing::debug!("S-IV: global concepts auto-fetch OK")
                    },
                    Ok(Ok(_)) => tracing::warn!(
                        "S-IV: global concepts auto-fetch skipped (not fast-forward)"
                    ),
                    Ok(Err(e)) => tracing::debug!("S-IV: global concepts auto-fetch skipped: {e}"),
                    Err(_) => tracing::warn!(
                        "S-IV: global concepts auto-fetch timed out after 5s — using cached"
                    ),
                }
            });
        }
    }

    // Fleet git-backed nodes: fire-and-forget sync at serve startup.
    // Same pattern as global concepts auto-fetch — non-blocking, offline-safe.
    tokio::spawn(async move {
        if let Err(e) = crate::fleet::sync_git_nodes() {
            tracing::warn!("Fleet: git-backed node sync error: {e}");
        }
    });

    // Embed feature active — hybrid BM25 + dense retrieval is wired into get_contexts.
    // Embeddings load from .cortyx/embeddings.tvim with .cortyx/embeddings.bin as the
    // raw authoritative companion; falls back gracefully to BM25-only when absent.
    #[cfg(feature = "embed")]
    tracing::info!("--features embed: hybrid BM25 + dense cosine retrieval active.");

    let feedback = Arc::new(FeedbackBuffer::default());
    let session = Arc::new(SessionState::default());
    let fleet_registry = match crate::fleet::load_registry() {
        Ok(registry) if !registry.nodes.is_empty() => {
            tracing::info!("Fleet: {} registered node(s)", registry.nodes.len());
            Some(Arc::new(registry))
        },
        Ok(_) => None,
        Err(err) => {
            tracing::warn!("Fleet: failed to load registry: {err}");
            None
        },
    };

    let dirty_handle = index.read().await.dirty_set_handle();
    let _watcher = watcher::start_watcher(project_root.clone(), Arc::clone(&index), dirty_handle)?;

    let server = CortyxServer {
        project_root,
        index: Arc::clone(&index),
        session: Arc::clone(&session),
        feedback: Arc::clone(&feedback),
        frozen,
        inflight_bytes: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        fleet_registry: fleet_registry.clone(),
        tool_router: CortyxServer::tool_router(),
    };

    let service = server
        .serve(rmcp::transport::stdio())
        .await
        .map_err(|e| crate::error::CortyxError::Other(e.to_string()))?;
    service
        .waiting()
        .await
        .map_err(|e| crate::error::CortyxError::Other(e.to_string()))?;
    let flushed = flush_provisional_hits_async(&index, &feedback.provisional_hits).await?;
    if flushed > 0 {
        tracing::info!("S2: explicitly cleared {flushed} provisional paths before shutdown");
    }
    Ok(())
}

#[cfg(test)]
mod tests;
