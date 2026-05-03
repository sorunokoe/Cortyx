//! Model Context Protocol (MCP) server implementation.
//!
//! Exposes Cortyx functionality via the MCP protocol for LLM integration.

use anyhow::Result;
use rmcp::schemars;
use rmcp::schemars::JsonSchema;
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{Implementation, ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router, ServerHandler, ServiceExt,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

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
use crate::watcher;

// ─── Tool input types ─────────────────────────────────────────────────────────

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct GetContextsInput {
    /// Natural language description of the task (drives neuron activation)
    pub task: String,
    /// Maximum tokens to return (default: 4096)
    pub max_tokens: Option<usize>,
    /// Optional module filter — restricts activation to a tagged namespace
    pub module: Option<String>,
    /// Optional person scope shorthand — equivalent to module="@{person}".
    /// Example: person="alice" → activates only neurons tagged module="@alice".
    pub person: Option<String>,
    /// Optional kind filter: "code" | "conversation" | "all" (default: "all").
    /// "code" → Core + Project neurons only. "conversation" → Verbatim only.
    pub kind: Option<String>,
    /// Optional minimum BM25 confidence required to return any neurons.
    /// Useful for abstention on absent or ambiguous memory questions.
    pub min_confidence: Option<f64>,
    /// Optional second-pass graph/vocabulary expansion for indirect matches.
    /// Targets multi-session and diffuse conversation-memory queries.
    pub multi_hop: Option<bool>,
    /// Optional: pass your previous assistant response here to close the feedback
    /// loop without a separate cortyx_close_task call. Cortyx soft-cites neurons
    /// from the last activation whose vocabulary overlaps the response text.
    pub previous_response: Option<String>,
    /// Optional: list of file paths currently open in the editor (e.g. ["src/auth.rs"]).
    /// Their BM25 term sets are injected as soft query terms (0.4× weight) before
    /// scoring — zero extra disk I/O, all terms are already in the posting lists.
    /// Improves recall for tasks like "fix this" when the relevant file is open.
    pub open_files: Option<Vec<String>>,
    /// Optional: recent error message or compiler output.
    /// Terms extracted and added to query expansion at 0.6× weight.
    /// Example: pass the last `cargo build` error to activate the relevant neuron.
    pub error_context: Option<String>,
    /// Optional: enable differential context emission.
    /// When true, Cortyx returns only added/changed context chunks relative to `context_handle`
    /// and emits/refreshes a handle comment in the response for the client to reuse.
    pub delta_mode: Option<bool>,
    /// Optional handle returned by a prior delta-mode `cortyx_get_contexts` call.
    /// Reuse the same handle across iterative same-session work to avoid re-sending unchanged
    /// context bodies and summaries.
    pub context_handle: Option<String>,
    /// Optional: enable stable module capsules.
    /// When true, Cortyx prepends deterministic per-module capsules for the explicit module
    /// filter or a dominant retrieved module cluster, then compresses redundant same-module
    /// summaries/headlines into capsule + task-specific delta neurons.
    pub capsule_mode: Option<bool>,
    /// Optional: return a concise answer-oriented output derived from the selected
    /// contexts instead of the full context bodies. Keeps ranking unchanged.
    pub answer_mode: Option<bool>,
    /// Optional minimum answer confidence required when answer_mode is enabled.
    /// Low-support heuristic snippet guesses abstain instead of returning weak answers.
    pub min_answer_confidence: Option<f64>,
    /// Optional: include lightweight provenance/explanation metadata for the
    /// selected contexts or derived answer output.
    pub provenance_mode: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct CortyxInput {
    /// Optional high-level intent. Supported values: `auto`, `context`, `answer`,
    /// `wake_up`, `agent_status`, `consistency`, `capabilities`.
    pub intent: Option<String>,
    /// Optional task or question to route. Required for `context` and `answer`.
    pub task: Option<String>,
    /// Optional agent identifier for agent-status or wake-up flows.
    pub agent: Option<String>,
    /// Optional person scope for wake-up or retrieval flows.
    pub person: Option<String>,
    /// Optional module filter for routed retrieval flows.
    pub module: Option<String>,
    /// Optional kind filter for routed retrieval flows: "code" | "conversation" | "all".
    pub kind: Option<String>,
    /// Optional path filter for consistency checks.
    pub path: Option<String>,
    /// Optional maximum tokens for routed retrieval flows.
    pub max_tokens: Option<usize>,
    /// Optional minimum BM25 confidence for routed retrieval flows.
    pub min_confidence: Option<f64>,
    /// Optional 2-hop retrieval for routed retrieval flows.
    pub multi_hop: Option<bool>,
    /// Optional previous assistant response for routed retrieval flows.
    pub previous_response: Option<String>,
    /// Optional delta-mode context emission for routed retrieval flows.
    pub delta_mode: Option<bool>,
    /// Optional reusable delta-mode context handle.
    pub context_handle: Option<String>,
    /// Optional stable capsule emission for routed retrieval flows.
    pub capsule_mode: Option<bool>,
    /// Optional minimum answer confidence for routed answer-mode flows.
    pub min_answer_confidence: Option<f64>,
    /// Optional provenance metadata in routed outputs.
    pub provenance_mode: Option<bool>,
    /// Optional agent-status timeline expansion.
    pub include_timeline: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct EvolveContextInput {
    /// Source file path relative to project root (e.g. "src/engine.rs")
    pub path: String,
    /// Full new markdown content for the `.context.md` neuron
    pub content: String,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct ExtractFromRawInput {
    /// Source file path relative to project root
    pub path: String,
    /// Short task pattern phrase (e.g. "add dark mode to SwiftUI view")
    pub task_pattern: String,
    /// The exact relevant chunk extracted from the raw source
    pub chunk: String,
    /// Why this chunk was useful for the task
    pub why: String,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct MineConversationInput {
    /// Raw conversation content (Claude MD, ChatGPT JSON, plain text, etc.)
    pub content: String,
    /// Optional speaker label for single-turn mining (e.g. "user" or "assistant")
    pub speaker: Option<String>,
    /// Optional module tag for filtered queries
    pub module: Option<String>,
    /// Optional person scope — equivalent to module="@{person}". Takes precedence over module.
    /// Example: person="alice" → all mined neurons tagged module="@alice".
    pub person: Option<String>,
    /// Optional ISO 8601 timestamp for the turn
    pub timestamp: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct CreateSynapseInput {
    /// Source neuron filename (relative to `.cortyx/neurons/`)
    pub source: String,
    /// Target neuron filename (relative to `.cortyx/neurons/`)
    pub target: String,
    /// Human-readable reason for the connection
    pub reason: String,
    /// Semantic edge type — defaults to `semantic_related` if omitted.
    /// Allowed values: `semantic_related`, `imports`, `calls`, `implements`,
    /// `contradicts`, `temporal_follows`, `derived`, `concept_expands`.
    pub edge_type: Option<SynapseType>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct InvalidateInput {
    /// Source file path relative to project root
    pub path: String,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct ListNeuronsInput {
    /// Optional module name to filter by (e.g. "auth" or "@alice"). Omit for all neurons.
    pub module: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct PeekNeuronInput {
    /// Full path to the neuron file (as returned by cortyx_list_neurons)
    pub path: String,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct RecallInput {
    /// Natural language query for episodic recall (e.g. "what did I decide about auth?")
    pub query: String,
    /// Optional person scope — restrict to memories tagged @person
    pub person: Option<String>,
    /// Maximum tokens to return (default: 4096)
    pub max_tokens: Option<usize>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct EvolveSectionInput {
    /// Source file path relative to project root (e.g. "src/engine.rs")
    pub path: String,
    /// Section name to update (e.g. "purpose", "api", "pitfalls")
    pub section: String,
    /// New markdown content for this section (replaces existing body)
    pub content: String,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct RecordHitInput {
    /// Source file path relative to project root (same as used in get_contexts)
    pub path: String,
    /// true if the neuron was actually cited in your response; false if it was irrelevant
    pub was_cited: bool,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct CloseTaskInput {
    /// The full assistant response text for the completed task.
    /// Cortyx scans it for neuron content to auto-record hits.
    pub response_text: String,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct RollbackSectionInput {
    /// Full path to the neuron file (as returned by cortyx_list_neurons)
    pub neuron_path: String,
    /// Section to restore: "purpose", "api", "pitfalls", etc., or "_full" for the whole neuron.
    /// Only sections shadowed before the most recent evolve call can be restored.
    pub section: String,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct DiaryWriteInput {
    /// Agent identifier, e.g. "reviewer", "architect", "ops". Stored under @agent/{agent}.
    pub agent: String,
    /// Freeform action/notes body for this memory entry.
    pub content: String,
    /// Optional short task or decision title.
    pub title: Option<String>,
    /// Optional action status (e.g. "planned", "in_progress", "done", "blocked").
    pub status: Option<String>,
    /// Optional broader objective this action is serving.
    pub goal: Option<String>,
    /// Optional next concrete step for the agent.
    pub next_step: Option<String>,
    /// Optional current blocker or dependency gap.
    pub blocker: Option<String>,
    /// Optional concise result/conclusion for this action.
    pub outcome: Option<String>,
    /// Optional related modules/files/entities. Useful for later retrieval and summaries.
    pub entities: Option<Vec<String>>,
    /// Optional people/systems/tasks this work currently depends on.
    pub depends_on: Option<Vec<String>>,
    /// Optional ISO 8601 timestamp. Defaults to current time.
    pub timestamp: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct DiaryReadInput {
    /// Agent identifier matching the one used with diary_write.
    pub agent: String,
    /// Approximate number of recent entries to return (default: 10).
    pub last_n: Option<usize>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct CheckConsistencyInput {
    /// Optional neuron path to scope the check. If omitted, scans all neurons.
    pub path: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct WakeUpInput {
    /// Optional person to include their most recent conversation memories (~60 tokens).
    pub person: Option<String>,
    /// Optional agent to include recent structured agent memories (~3 recent summaries).
    pub agent: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct AgentStatusInput {
    /// Agent identifier matching the one used with diary_write.
    pub agent: String,
    /// When true, append recent focus/status/goal/next-step/blocker/outcome timelines from the temporal KG mirror.
    pub include_timeline: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct CollaborationStatusInput {
    /// Optional agent identifier to scope the report to one collaborator.
    pub agent: Option<String>,
    /// Optional shared module/entity label to scope the report to one collaboration surface.
    pub module: Option<String>,
    /// When true, append recent collaboration timeline events.
    pub include_timeline: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CortyxRouteKind {
    Context,
    Answer,
    WakeUp,
    AgentStatus,
    Consistency,
    Capabilities,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CortyxRoutePlan {
    kind: CortyxRouteKind,
    task: Option<String>,
    agent: Option<String>,
}

fn normalize_cortyx_route_intent(intent: &str) -> Option<CortyxRouteKind> {
    match intent.trim().to_ascii_lowercase().as_str() {
        "" | "auto" => None,
        "context" | "retrieve" | "retrieval" | "search" | "get_contexts" => {
            Some(CortyxRouteKind::Context)
        },
        "answer" | "qa" | "question" => Some(CortyxRouteKind::Answer),
        "wake" | "wake_up" | "wake-up" | "prime" | "priming" => Some(CortyxRouteKind::WakeUp),
        "agent" | "agent_status" | "agent-status" | "status" => Some(CortyxRouteKind::AgentStatus),
        "consistency" | "contradiction" | "conflict" | "check" => {
            Some(CortyxRouteKind::Consistency)
        },
        "capability" | "capabilities" | "describe" | "help" => Some(CortyxRouteKind::Capabilities),
        _ => None,
    }
}

fn looks_like_wake_up_request(task_lower: &str) -> bool {
    task_lower.contains("wake up")
        || task_lower.contains("wake-up")
        || task_lower.contains("prime session")
        || task_lower.contains("prime the session")
        || task_lower.contains("prime context")
}

fn looks_like_consistency_request(task_lower: &str) -> bool {
    task_lower.contains("consistency")
        || task_lower.contains("contradiction")
        || task_lower.contains("conflict")
}

fn looks_like_question(task: &str) -> bool {
    let trimmed = task.trim();
    if trimmed.ends_with('?') {
        return true;
    }
    matches!(
        trimmed
            .split_whitespace()
            .next()
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "what"
            | "who"
            | "when"
            | "where"
            | "why"
            | "how"
            | "which"
            | "is"
            | "are"
            | "did"
            | "does"
            | "do"
            | "can"
            | "could"
            | "should"
            | "will"
    )
}

fn derive_cortyx_route(input: &CortyxInput) -> std::result::Result<CortyxRoutePlan, String> {
    if let Some(intent) = input.intent.as_deref() {
        if !intent.trim().is_empty() && !intent.eq_ignore_ascii_case("auto") {
            let Some(kind) = normalize_cortyx_route_intent(intent) else {
                return Err(format!(
                    "unsupported intent '{}'; use auto, context, answer, wake_up, agent_status, consistency, or capabilities",
                    intent
                ));
            };
            let task = input.task.clone().filter(|task| !task.trim().is_empty());
            let agent = input.agent.clone().filter(|agent| !agent.trim().is_empty());
            match kind {
                CortyxRouteKind::Context | CortyxRouteKind::Answer => {
                    if task.is_none() {
                        return Err("task is required for context or answer intent".to_string());
                    }
                },
                CortyxRouteKind::AgentStatus => {
                    if agent.is_none() {
                        return Err("agent is required for agent_status intent".to_string());
                    }
                },
                CortyxRouteKind::WakeUp
                | CortyxRouteKind::Consistency
                | CortyxRouteKind::Capabilities => {},
            }
            return Ok(CortyxRoutePlan { kind, task, agent });
        }
    }

    let task = input.task.clone().filter(|task| !task.trim().is_empty());
    let agent = input.agent.clone().filter(|agent| !agent.trim().is_empty());

    if task.is_none() {
        if agent.is_some() {
            return Ok(CortyxRoutePlan {
                kind: CortyxRouteKind::AgentStatus,
                task: None,
                agent,
            });
        }
        if input
            .person
            .as_ref()
            .is_some_and(|person| !person.trim().is_empty())
        {
            return Ok(CortyxRoutePlan {
                kind: CortyxRouteKind::WakeUp,
                task: None,
                agent: None,
            });
        }
        if input
            .path
            .as_ref()
            .is_some_and(|path| !path.trim().is_empty())
        {
            return Ok(CortyxRoutePlan {
                kind: CortyxRouteKind::Consistency,
                task: None,
                agent: None,
            });
        }
        return Ok(CortyxRoutePlan {
            kind: CortyxRouteKind::Capabilities,
            task: None,
            agent: None,
        });
    }

    let task_value = task.clone().unwrap();
    let task_lower = task_value.to_ascii_lowercase();
    let kind = if looks_like_wake_up_request(&task_lower) {
        CortyxRouteKind::WakeUp
    } else if looks_like_consistency_request(&task_lower) {
        CortyxRouteKind::Consistency
    } else if agent.is_some()
        && (task_lower.contains("status")
            || task_lower.contains("working on")
            || task_lower.contains("blocked")
            || task_lower.contains("next step"))
    {
        CortyxRouteKind::AgentStatus
    } else if looks_like_question(&task_value) {
        CortyxRouteKind::Answer
    } else {
        CortyxRouteKind::Context
    };

    Ok(CortyxRoutePlan {
        kind,
        task: Some(task_value),
        agent,
    })
}

// ─── S4 KG input structs ──────────────────────────────────────────────────────

#[derive(Debug, Deserialize, JsonSchema)]
pub struct KgAddInput {
    /// Entity name/slug (e.g. "project_meta", "team", "dependencies").
    pub entity: String,
    /// Predicate / relationship type (e.g. "language", "lead", "version").
    pub predicate: String,
    /// Fact value.
    pub value: String,
    /// Optional ISO-8601 start date for this fact (e.g. "2024-01-01").
    pub valid_from: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct KgQueryInput {
    /// Entity name to query.
    pub entity: String,
    /// Optional ISO-8601 date to filter active facts (defaults to "now = all open-ended").
    pub as_of: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct KgInvalidateInput {
    /// Entity name.
    pub entity: String,
    /// Predicate of the fact to end.
    pub predicate: String,
    /// ISO-8601 date when this fact was superseded.
    pub ended: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct KgTimelineInput {
    /// Entity name.
    pub entity: String,
    /// Predicate to show the full history for.
    pub predicate: String,
}

// ─── MCP Server ───────────────────────────────────────────────────────────────

/// Maximum byte size for content fields in MCP tool inputs.
///
/// Prevents OOM from a runaway or malicious LLM agent submitting unbounded payloads.
const MAX_CONTENT_BYTES: usize = 1_048_576; // 1 MB

/// Maximum byte length for task/query strings.
const MAX_TASK_BYTES: usize = 4_096;

#[derive(Clone)]
pub struct CortyxServer {
    project_root: PathBuf,
    index: Arc<RwLock<NeuronIndex>>,
    /// Paths returned by the most recent cortyx_get_contexts call.
    /// Used by cortyx_close_task to auto-record hits without an explicit list.
    last_activated: Arc<Mutex<Vec<PathBuf>>>,
    /// Ephemeral carry-over of the last returned paths.
    /// This is cleared on the next get_contexts or close_task so external shutdown hooks
    /// do not silently convert control-plane activity into training signals.
    provisional_hits: Arc<Mutex<Vec<PathBuf>>>,
    /// Server-side snapshots for delta-mode context emission.
    context_sessions: Arc<Mutex<HashMap<String, ContextSnapshot>>>,
    next_context_handle: Arc<AtomicU64>,
    // Kept for the rmcp macro-generated dispatch table; not called directly.
    #[allow(dead_code)]
    tool_router: ToolRouter<Self>,
}

#[derive(Clone, Default)]
struct ContextSnapshot {
    order: u64,
    chunks: HashMap<PathBuf, String>,
    overflow: HashMap<PathBuf, String>,
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

async fn flush_provisional_hits_async(
    _index: &Arc<RwLock<NeuronIndex>>,
    provisional_hits: &Arc<Mutex<Vec<PathBuf>>>,
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

fn flush_provisional_hits_blocking(
    _index: &Arc<RwLock<NeuronIndex>>,
    provisional_hits: &Arc<Mutex<Vec<PathBuf>>>,
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
        if Arc::strong_count(&self.provisional_hits) > 1 {
            return;
        }
        if tokio::runtime::Handle::try_current().is_ok() {
            tracing::debug!(
                "S2: skipping blocking provisional buffer clear from async Drop context"
            );
            return;
        }
        match flush_provisional_hits_blocking(&self.index, &self.provisional_hits) {
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
        let provisional_hits = Arc::new(Mutex::new(Vec::new()));
        let context_sessions = Arc::new(Mutex::new(HashMap::new()));
        let next_context_handle = Arc::new(AtomicU64::new(0));

        Self {
            project_root,
            index,
            last_activated: Arc::new(Mutex::new(Vec::new())),
            provisional_hits,
            context_sessions,
            next_context_handle,
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
    fn rel_display<'a>(&self, path: &'a Path) -> std::borrow::Cow<'a, str> {
        path.strip_prefix(&self.project_root)
            .unwrap_or(path)
            .to_string_lossy()
    }

    async fn render_cortyx_capability_summary(&self) -> String {
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

    async fn ensure_context_handle(&self, requested: Option<&str>) -> String {
        requested
            .filter(|handle| !handle.trim().is_empty())
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| {
                let next = self.next_context_handle.fetch_add(1, Ordering::Relaxed) + 1;
                format!("ctx-{next}")
            })
    }

    async fn load_context_snapshot(&self, handle: &str) -> Option<ContextSnapshot> {
        self.context_sessions.lock().await.get(handle).cloned()
    }

    async fn store_context_snapshot(
        &self,
        handle: String,
        chunks: &[RenderedContextItem],
        overflow: &[RenderedContextItem],
    ) {
        let order = self.next_context_handle.fetch_add(1, Ordering::Relaxed) + 1;
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

        let mut sessions = self.context_sessions.lock().await;
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

#[tool_router]
impl CortyxServer {
    #[tool(
        name = "cortyx",
        description = "Universal Cortyx entrypoint for the current local-first Cortyx surface. Routes a high-level intent to context retrieval, answer mode, wake-up priming, agent status, consistency checks, or a capability summary. Use intent='auto' or omit it to infer the best route from the supplied task/agent/person/path; call it with no task/agent/path inputs to get the current capability summary."
    )]
    async fn cortyx(&self, Parameters(input): Parameters<CortyxInput>) -> String {
        let route = match derive_cortyx_route(&input) {
            Ok(route) => route,
            Err(err) => return format!("ERROR: {err}"),
        };

        match route.kind {
            CortyxRouteKind::Context | CortyxRouteKind::Answer => {
                let task = route.task.unwrap_or_default();
                let module = input.module.clone().or_else(|| {
                    route
                        .agent
                        .as_ref()
                        .map(|agent| format!("@agent/{}", agent.trim()))
                });
                self.get_contexts(Parameters(GetContextsInput {
                    task,
                    max_tokens: input.max_tokens,
                    module,
                    person: input.person,
                    kind: input.kind,
                    min_confidence: input.min_confidence,
                    multi_hop: input.multi_hop,
                    previous_response: input.previous_response,
                    open_files: None,
                    error_context: None,
                    delta_mode: input.delta_mode,
                    context_handle: input.context_handle,
                    capsule_mode: input.capsule_mode,
                    answer_mode: Some(route.kind == CortyxRouteKind::Answer),
                    min_answer_confidence: input.min_answer_confidence,
                    provenance_mode: input.provenance_mode,
                }))
                .await
            },
            CortyxRouteKind::WakeUp => {
                self.wake_up(Parameters(WakeUpInput {
                    person: input.person,
                    agent: route.agent,
                }))
                .await
            },
            CortyxRouteKind::AgentStatus => {
                self.agent_status(Parameters(AgentStatusInput {
                    agent: route.agent.unwrap_or_default(),
                    include_timeline: input.include_timeline,
                }))
                .await
            },
            CortyxRouteKind::Consistency => {
                self.check_consistency(Parameters(CheckConsistencyInput { path: input.path }))
                    .await
            },
            CortyxRouteKind::Capabilities => self.render_cortyx_capability_summary().await,
        }
    }

    /// Activate the most relevant neurons for a task.
    ///
    /// Returns context files sorted lexicographically — place them AFTER the
    /// `cache_control: {type: "ephemeral"}` breakpoint in your prompt to keep
    /// the static prefix byte-identical across calls (enabling prompt cache hits
    /// on the static block).
    #[tool(
        name = "cortyx_get_contexts",
        description = "Get the most relevant local/project context neurons for a task. Returns 3-5 .context.md files, sorted deterministically. Inject after your cache_control breakpoint to keep the static prefix byte-identical for prompt caching. Pass your previous assistant response in `previous_response` to close the feedback loop automatically — no separate cortyx_close_task call needed. Set `delta_mode=true` and reuse `context_handle` to receive only added/changed context on iterative same-session work. Set `capsule_mode=true` to prepend stable module capsules and compress redundant same-module summaries into capsule + task delta. Set `answer_mode=true` to return an optional answer-layer derived from the selected contexts without changing the retrieval hot path. Set `min_answer_confidence` to require stronger answer support before answer-mode emits a result. Set `provenance_mode=true` to include lightweight source/explanation metadata."
    )]
    async fn get_contexts(&self, Parameters(input): Parameters<GetContextsInput>) -> String {
        if input.task.len() > MAX_TASK_BYTES {
            return format!("ERROR: task exceeds {MAX_TASK_BYTES} byte limit");
        }
        if let Some(prev_resp) = &input.previous_response {
            if prev_resp.len() > MAX_CONTENT_BYTES {
                return format!("ERROR: previous_response exceeds {MAX_CONTENT_BYTES} byte limit");
            }
        }

        // S6 — Implicit feedback: if the caller supplied their previous response,
        // apply soft-citation against last_activated before running the new query.
        // This eliminates the need for a separate cortyx_close_task call.
        if let Some(prev_resp) = &input.previous_response {
            let activated = self.last_activated.lock().await.clone();
            if !activated.is_empty() && !prev_resp.is_empty() {
                let response_lower = prev_resp.to_lowercase();
                let response_tokens: std::collections::HashSet<String> =
                    tokenize(prev_resp).into_iter().collect();
                let citation_decisions: Vec<(PathBuf, bool)> = {
                    let idx = self.index.read().await;
                    activated
                        .iter()
                        .map(|path| {
                            let stem = path
                                .file_stem()
                                .unwrap_or_default()
                                .to_string_lossy()
                                .to_lowercase();
                            let stem = stem.trim_end_matches(".context");
                            let explicit_cited = !stem.is_empty() && response_lower.contains(stem);
                            let soft_cited = if !explicit_cited && !response_tokens.is_empty() {
                                idx.term_freq_overlap(path, &response_tokens) >= 20
                            } else {
                                false
                            };
                            (path.clone(), explicit_cited || soft_cited)
                        })
                        .collect()
                };
                let mut idx = self.index.write().await;
                let mut implicit_hits = 0usize;
                for (path, cited) in &citation_decisions {
                    idx.record_hit(path, *cited);
                    if *cited {
                        implicit_hits += 1;
                    }
                }
                tracing::debug!(
                    hits = implicit_hits,
                    total = activated.len(),
                    "S6 implicit feedback applied from previous_response"
                );
            }
        }
        let max_tokens = input.max_tokens.unwrap_or(4096);
        let min_confidence = input.min_confidence.map(|value| value as f32);
        let multi_hop = input.multi_hop.unwrap_or(false);
        let capsule_mode = input.capsule_mode.unwrap_or(false);
        let answer_mode = input.answer_mode.unwrap_or(false);
        let min_answer_confidence = input.min_answer_confidence.map(|value| value as f32);
        let provenance_mode = input.provenance_mode.unwrap_or(false);
        let effective_module: Option<String> = input
            .person
            .as_ref()
            .map(|p| format!("@{}", p))
            .or_else(|| input.module.clone());

        // Clear the previous provisional buffer. Only explicit citation evidence should
        // train long-term ranking; carry-over paths are kept solely for in-session close_task.
        let old_provisional = std::mem::take(&mut *self.provisional_hits.lock().await);

        let augmented_task = {
            let idx = self.index.read().await;
            build_augmented_task(&idx, &input)
        };

        let (mut paths_with_scores, mut overflow) = {
            let idx = self.index.read().await;
            // S-I (R16): Multi-resolution emission — use scored variant for tiered output
            idx.get_contexts_with_scores_and_overflow(
                &augmented_task,
                max_tokens,
                effective_module.as_deref(),
                input.kind.as_deref(),
                min_confidence,
                multi_hop,
            )
        };

        let mut capsule_items = Vec::new();
        if capsule_mode {
            let idx = self.index.read().await;
            let path_modules = build_path_module_map(&paths_with_scores, &overflow, &idx);
            drop(idx);

            let candidate_modules = select_capsule_modules(
                &paths_with_scores,
                effective_module.as_deref(),
                &path_modules,
            );
            let available_capsules: Vec<(String, RenderedContextItem)> = candidate_modules
                .into_iter()
                .filter_map(|module| {
                    render_module_capsule(&self.project_root, &module).map(|item| (module, item))
                })
                .collect();

            if !available_capsules.is_empty() {
                let active_capsule_modules: HashSet<String> = available_capsules
                    .iter()
                    .map(|(module, _)| module.clone())
                    .collect();
                let keep_paths = select_capsule_anchor_paths(
                    &paths_with_scores,
                    &active_capsule_modules,
                    &path_modules,
                );
                paths_with_scores.retain(|(path, _)| match path_modules.get(path) {
                    Some(module) if active_capsule_modules.contains(module) => {
                        keep_paths.contains(path)
                    },
                    _ => true,
                });
                overflow.retain(|(path, _)| match path_modules.get(path) {
                    Some(module) => !active_capsule_modules.contains(module),
                    None => true,
                });
                capsule_items = available_capsules
                    .into_iter()
                    .map(|(_, item)| item)
                    .collect();
            }
        }

        // Flatten paths for backward-compatible downstream use
        let paths: Vec<PathBuf> = paths_with_scores.iter().map(|(p, _)| p.clone()).collect();
        if !old_provisional.is_empty() {
            tracing::debug!(
                cleared = old_provisional.len(),
                "Dropped provisional carry-over without applying implicit ranking feedback"
            );
        }

        // Increment use_count for all returned neurons — activates the feedback loop.
        // Also capture any Contradicts pairs for the warning block (S7).
        let contradictions = if !paths.is_empty() {
            let mut idx = self.index.write().await;
            idx.record_activation(&paths);
            // B2: Record co-activation of query terms with each activated neuron.
            // After ≥30 co-activations, terms are promoted to synonym clouds for
            // query expansion — improving recall for semantically related queries.
            // Use the effective augmented retrieval task, not only the raw user text.
            let terms = crate::index::tokenize(&augmented_task);
            for path in &paths {
                idx.record_coactivation(path, &terms);
            }
            // S7: Check for contradicting pairs among activated neurons.
            idx.find_contradictions(&paths)
        } else {
            Vec::new()
        };

        // Store for cortyx_close_task — replaces previous task's activation list.
        *self.last_activated.lock().await = paths.clone();
        // Set provisional carry-over for in-session close_task tracking only.
        *self.provisional_hits.lock().await = paths.clone();

        if paths.is_empty() && capsule_items.is_empty() {
            if input.min_confidence.is_some() {
                return "(no neurons matched — confidence below threshold)".to_string();
            }
            return "No relevant neurons found. Run `cortyx compile .` first, then call \
                cortyx_evolve_context to fill stubs."
                .to_string();
        }

        if answer_mode {
            let idx_read = self.index.read().await;
            return match answer_plane::render_answer_output_decision(
                &idx_read,
                &input.task,
                &paths_with_scores,
                provenance_mode,
                min_answer_confidence,
            ) {
                Ok(answer) => answer,
                Err(answer_plane::AnswerAbstentionReason::LowFormConfidence)
                    if min_answer_confidence.is_some() =>
                {
                    "(no confident answer — answer confidence below threshold)".to_string()
                },
                Err(answer_plane::AnswerAbstentionReason::LowFormConfidence)
                | Err(answer_plane::AnswerAbstentionReason::Unsupported) => String::new(),
            };
        }

        // Filenames sorted lexicographically in the header — stable, byte-identical for the same
        // neuron set regardless of retrieval order. Used for cache-key validation by the client.
        // Bodies below are emitted in BM25-relevance order (most useful neuron first).
        let mut lex_names: Vec<String> = paths
            .iter()
            .map(|p| {
                p.file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        lex_names.sort();

        let mut out = format!(
            "<!-- CORTYX CONTEXT — injected after cache_control breakpoint -->\n\
             <!-- Task: {} -->\n\
             <!-- Neurons (lex): {} -->\n\n",
            sanitize_comment(&input.task),
            lex_names.join(", "),
        );
        if !capsule_items.is_empty() {
            let mut capsule_names: Vec<String> = capsule_items
                .iter()
                .map(|item| {
                    item.path
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .into_owned()
                })
                .collect();
            capsule_names.sort();
            out.push_str(&format!(
                "<!-- Module capsules: {} -->\n<!-- Capsule mode: stable module capsule + task delta -->\n\n",
                capsule_names.join(", "),
            ));
        }
        if provenance_mode {
            let idx_read = self.index.read().await;
            if let Some(block) =
                answer_plane::render_provenance_output(&idx_read, &paths_with_scores)
            {
                out.push_str(&block);
            }
        }

        // S-I (R16): Tiered emission — full body for Tier 2 (score ≥ 5.0),
        // summary for Tier 1 (1.5 ≤ score < 5.0), already handled as overflow for Tier 0.
        let render_terms = tokenize(&input.task);
        let idx_read = self.index.read().await;
        let mut rendered_chunks = capsule_items;
        rendered_chunks.extend(
            paths_with_scores
                .iter()
                .map(|(path, score)| render_context_item(path, *score, &render_terms, &idx_read)),
        );
        drop(idx_read);
        let rendered_overflow: Vec<RenderedContextItem> = overflow
            .iter()
            .map(|(path, headline)| render_overflow_item(path, headline))
            .collect();

        let delta_mode = input.delta_mode.unwrap_or(false);
        let mut chunks_to_emit = rendered_chunks.clone();
        let mut overflow_to_emit = rendered_overflow.clone();
        if delta_mode {
            let context_handle = self
                .ensure_context_handle(input.context_handle.as_deref())
                .await;
            let previous_snapshot = self.load_context_snapshot(&context_handle).await;
            let chunk_delta = select_delta_items(
                &rendered_chunks,
                previous_snapshot.as_ref().map(|s| &s.chunks),
            );
            let overflow_delta = select_delta_items(
                &rendered_overflow,
                previous_snapshot.as_ref().map(|s| &s.overflow),
            );

            chunks_to_emit = chunk_delta.emitted;
            overflow_to_emit = overflow_delta.emitted;

            let emitted_total = chunks_to_emit.len() + overflow_to_emit.len();
            let unchanged_total = chunk_delta.unchanged + overflow_delta.unchanged;
            let removed_total = chunk_delta.removed + overflow_delta.removed;
            let mode_label = if previous_snapshot.is_some() {
                "delta"
            } else {
                "full"
            };

            out.push_str(&format!(
                "<!-- Context handle: {} -->\n<!-- Context mode: {mode_label}; emitted={emitted_total}; unchanged={unchanged_total}; removed={removed_total} -->\n",
                sanitize_comment(&context_handle),
            ));
            if emitted_total == 0 {
                out.push_str(
                    "<!-- Context delta: no new or changed chunks; reuse previously injected context. -->\n",
                );
            }
            out.push('\n');

            self.store_context_snapshot(context_handle, &rendered_chunks, &rendered_overflow)
                .await;
        }

        for chunk in &chunks_to_emit {
            out.push_str(&chunk.rendered);
        }

        // Compressed overflow neurons: emit one-line headlines for neurons that
        // were relevant but exceeded the token budget. Gives the LLM routing
        // signals at ~5% of the token cost of the full neuron.
        if !overflow_to_emit.is_empty() {
            out.push_str("<!-- === COMPRESSED CONTEXT (budget overflow) === -->\n");
            for item in &overflow_to_emit {
                out.push_str(&item.rendered);
            }
            out.push_str("<!-- === END COMPRESSED === -->\n");
        }

        // S7: Append contradiction warning block if any activated neurons conflict.
        if !contradictions.is_empty() {
            out.push_str(
                "\n## ⚠ Contradictions Detected\n\
                The following neuron pairs hold conflicting information. \
                Verify which is current before proceeding.\n\n",
            );
            for (a, b, reason) in &contradictions {
                let a_name = a.file_name().unwrap_or_default().to_string_lossy();
                let b_name = b.file_name().unwrap_or_default().to_string_lossy();
                out.push_str(&format!(
                    "- **{}** ↔ **{}**\n  Reason: {}\n\n",
                    a_name, b_name, reason
                ));
            }
        }

        out.push_str("<!-- === END CORTYX CONTEXT === -->\n");
        out
    }

    /// Rewrite a neuron with improved content (self-improvement during normal usage).
    #[tool(
        name = "cortyx_evolve_context",
        description = "Evolve (rewrite) a neuron with AI-curated content. Call after a task reveals better reasoning instructions, pitfalls, or cross-references. Atomically updates the .context.md file and refreshes the index. IMPORTANT: When writing neuron content for conversation/memory neurons, append a '## paraphrases' section containing 8-10 natural-language questions that this neuron directly answers. Example: '## paraphrases\\nWhat degree did she graduate with?\\nWhere did she go to school?\\nWhat did she study?' This pre-generates question vocabulary that BM25 uses at query time, dramatically improving recall without any model at query time."
    )]
    async fn evolve_context(&self, Parameters(input): Parameters<EvolveContextInput>) -> String {
        let rel = match validate_relative_path(&input.path) {
            Ok(p) => p,
            Err(e) => return format!("ERROR: Invalid path: {e}"),
        };
        if input.content.is_empty() {
            return "ERROR: Content must not be empty".to_string();
        }
        if input.content.len() > MAX_CONTENT_BYTES {
            return format!("ERROR: content exceeds {MAX_CONTENT_BYTES} byte limit");
        }

        let source = self.project_root.join(&rel);
        let neuron_path = core_neuron_path(&source, &self.project_root);
        let existed_before = neuron_path.exists();

        if let Err(e) = std::fs::create_dir_all(
            neuron_path
                .parent()
                .ok_or_else(|| anyhow::anyhow!("bad path"))
                .unwrap_or(Path::new(".")),
        ) {
            return format!("ERROR: Failed to create neuron dir: {e}");
        }

        // E2: Save full content shadow before overwriting — enables instant undo.
        if let Ok(prev_content) = std::fs::read_to_string(&neuron_path) {
            let meta_file_shadow = meta_path(&neuron_path);
            let mut meta_shadow = load_or_new_meta(&meta_file_shadow, &source, NeuronKind::Core);
            push_shadow(&mut meta_shadow.shadow_sections, "_full", prev_content);
            if let Err(e) = save_meta(&meta_file_shadow, &meta_shadow) {
                return format!("ERROR: Failed to save rollback shadow: {e}");
            }
        }

        if let Err(e) = atomic_write(&neuron_path, input.content.as_bytes()) {
            return format!("ERROR: Failed to write neuron: {e}");
        }

        let source_hash = hash_file(&source).unwrap_or_default();
        let now = now_iso8601();
        let meta_file = meta_path(&neuron_path);
        let mut meta = load_or_new_meta(&meta_file, &source, NeuronKind::Core);
        refresh_meta_after_content_write(&mut meta, &input.content);
        meta.source_hash = source_hash;
        meta.last_updated = now;

        if let Err(e) = save_meta(&meta_file, &meta) {
            return format!(
                "ERROR: Failed to save meta for {}: {e}",
                self.rel_display(&neuron_path)
            );
        }
        let provenance_result = record_mutation_provenance(
            &neuron_path,
            &meta,
            &input.content,
            if existed_before {
                ProvenanceOperation::Update
            } else {
                ProvenanceOperation::Create
            },
            ProvenanceSource::Local,
            None,
            Some(if existed_before {
                format!("rewrote neuron from {}", rel.display())
            } else {
                format!("created neuron from {}", rel.display())
            }),
        );

        let mut idx = self.index.write().await;
        if let Err(e) = idx.upsert_neuron(&neuron_path, &input.content, &meta) {
            return format!("ERROR: Failed to update index: {e}");
        }
        finalize_mutation_message(
            format!(
                "Neuron evolved: {} ({} tokens, {} synapses)",
                self.rel_display(&neuron_path),
                meta.tokens,
                meta.synapses.len()
            ),
            provenance_result,
        )
    }

    /// Update a single named section within a neuron — surgical and token-efficient.
    #[tool(
        name = "cortyx_evolve_section",
        description = "Update one named section (e.g. 'purpose', 'api', 'pitfalls') within a neuron. ~50 tokens instead of a full 1500-token rewrite. Use when only one section needs improving."
    )]
    async fn evolve_section(&self, Parameters(input): Parameters<EvolveSectionInput>) -> String {
        let rel = match validate_relative_path(&input.path) {
            Ok(p) => p,
            Err(e) => return format!("ERROR: Invalid path: {e}"),
        };
        if input.section.is_empty() {
            return "ERROR: section must not be empty".to_string();
        }
        if input.content.is_empty() {
            return "ERROR: content must not be empty".to_string();
        }

        let source = self.project_root.join(&rel);
        let neuron_path = core_neuron_path(&source, &self.project_root);
        let section_key = input.section.to_lowercase();

        let existing = match std::fs::read_to_string(&neuron_path) {
            Ok(c) => c,
            Err(e) => {
                return format!("ERROR: Cannot read neuron (run `cortyx compile` first): {e}");
            },
        };

        // E2: Save previous section content to shadow before overwriting.
        {
            let meta_file_shadow = meta_path(&neuron_path);
            let mut meta_shadow = load_or_new_meta(&meta_file_shadow, &source, NeuronKind::Core);
            // Extract current section body from existing content and save as shadow
            let current_sections = parse_sections(&existing);
            if let Some(prev_body) = current_sections.get(&section_key) {
                push_shadow(
                    &mut meta_shadow.shadow_sections,
                    &section_key,
                    prev_body.clone(),
                );
            } else {
                // Save the full content as a fallback shadow
                push_shadow(&mut meta_shadow.shadow_sections, "_full", existing.clone());
            }
            if let Err(e) = save_meta(&meta_file_shadow, &meta_shadow) {
                return format!("ERROR: Failed to save rollback shadow: {e}");
            }
        }

        let new_content = replace_section(&existing, &input.section, &input.content);

        if let Err(e) = atomic_write(&neuron_path, new_content.as_bytes()) {
            return format!("ERROR: Failed to write neuron: {e}");
        }

        let now = now_iso8601();
        let meta_file = meta_path(&neuron_path);
        let mut meta = load_or_new_meta(&meta_file, &source, NeuronKind::Core);
        refresh_meta_after_content_write(&mut meta, &new_content);
        meta.last_updated = now;

        if let Err(e) = save_meta(&meta_file, &meta) {
            return format!("ERROR: Failed to save meta: {e}");
        }
        let provenance_result = record_mutation_provenance(
            &neuron_path,
            &meta,
            &new_content,
            ProvenanceOperation::SectionUpdate,
            ProvenanceSource::Local,
            Some(section_key.clone()),
            Some(format!(
                "updated {section_key} section for {}",
                rel.display()
            )),
        );

        let mut idx = self.index.write().await;
        if let Err(e) = idx.upsert_neuron(&neuron_path, &new_content, &meta) {
            return format!("ERROR: Failed to update index: {e}");
        }
        let sections = parse_sections(&new_content);
        finalize_mutation_message(
            format!(
                "Section '{}' updated in {} ({} tokens, {} sections)",
                input.section,
                self.rel_display(&neuron_path),
                meta.tokens,
                sections.len()
            ),
            provenance_result,
        )
    }

    /// Create a use-case neuron — a proven concrete chunk for a specific task pattern.
    #[tool(
        name = "cortyx_extract_from_raw",
        description = "Save a proven relevant chunk as a use-case neuron. Activated automatically for similar future tasks without re-reading the raw source."
    )]
    async fn extract_from_raw(&self, Parameters(input): Parameters<ExtractFromRawInput>) -> String {
        let rel = match validate_relative_path(&input.path) {
            Ok(p) => p,
            Err(e) => return format!("ERROR: Invalid path: {e}"),
        };
        if input.chunk.len() > MAX_CONTENT_BYTES {
            return format!("ERROR: chunk exceeds {MAX_CONTENT_BYTES} byte limit");
        }

        let source = self.project_root.join(&rel);
        // Truncate kebab to avoid exceeding OS filename limits (max 255 chars total)
        let task_kebab = truncate_str(&to_kebab(&input.task_pattern), 64);
        let source_rel = rel.to_string_lossy().replace(['/', '\\'], "_");

        let neuron_filename = format!("{source_rel}.usecase.{task_kebab}.md");
        let neuron_path = neuron_dir(&self.project_root).join(&neuron_filename);
        let existed_before = neuron_path.exists();
        let now = now_iso8601();

        let content = format!(
            "<!-- Task pattern: {} -->\n\
             <!-- parent: {source_rel}.context.md -->\n\
             <!-- created: {now} | uses: 0 -->\n\n\
             **Exact relevant chunk (proven):**\n\n{}\n\n\
             **Why it was used:**\n{}\n",
            input.task_pattern, input.chunk, input.why
        );

        if let Err(e) = std::fs::create_dir_all(neuron_path.parent().unwrap_or(Path::new("."))) {
            return format!("ERROR: Failed to create dir: {e}");
        }
        if let Err(e) = atomic_write(&neuron_path, content.as_bytes()) {
            return format!("ERROR: Failed to write use-case neuron: {e}");
        }

        let parent_neuron = core_neuron_path(&source, &self.project_root);
        let mut meta = NeuronMeta::new_stub(&source, NeuronKind::UseCase);
        meta.task_pattern = Some(input.task_pattern.clone());
        meta.parent = Some(parent_neuron);
        meta.tokens = estimate_context_tokens(&content);
        meta.last_updated = now;
        meta.source_hash = hash_file(&source).unwrap_or_default();
        meta.status = NeuronStatus::Fresh;

        let meta_file = meta_path(&neuron_path);
        if let Err(e) = save_meta(&meta_file, &meta) {
            tracing::warn!("Failed to save meta for {}: {e}", neuron_path.display());
        }
        let provenance_result = record_mutation_provenance(
            &neuron_path,
            &meta,
            &content,
            if existed_before {
                ProvenanceOperation::Update
            } else {
                ProvenanceOperation::Create
            },
            ProvenanceSource::Import,
            None,
            Some(format!(
                "extracted raw chunk for pattern \"{}\"",
                input.task_pattern
            )),
        );

        let mut idx = self.index.write().await;
        if let Err(e) = idx.upsert_neuron(&neuron_path, &content, &meta) {
            return format!("ERROR: Failed to update index: {e}");
        }
        finalize_mutation_message(
            format!(
                "Use-case neuron created: {} for pattern \"{}\"",
                self.rel_display(&neuron_path),
                input.task_pattern
            ),
            provenance_result,
        )
    }

    /// Add a synapse (cross-reference edge) between two neurons.
    #[tool(
        name = "cortyx_create_synapse",
        description = "Create a synapse between two neurons. The activation engine traverses 1-hop synapses to pull in related context for tasks spanning multiple files."
    )]
    async fn create_synapse(&self, Parameters(input): Parameters<CreateSynapseInput>) -> String {
        // Validate both source and target are safe paths
        let source_rel = match validate_relative_path(&input.source) {
            Ok(p) => p,
            Err(e) => return format!("ERROR: Invalid source: {e}"),
        };
        let target_rel = match validate_relative_path(&input.target) {
            Ok(p) => p,
            Err(e) => return format!("ERROR: Invalid target: {e}"),
        };

        let ndir = neuron_dir(&self.project_root);
        let source_path = ndir.join(&source_rel);
        let target_path = ndir.join(&target_rel);

        for path in [&source_path, &target_path] {
            if !path.exists() {
                return format!(
                    "ERROR: Neuron not found: {}. Create it first with cortyx_evolve_context.",
                    path.display()
                );
            }
        }

        let mut content = match std::fs::read_to_string(&source_path) {
            Ok(c) => c,
            Err(e) => return format!("ERROR: Cannot read source neuron: {e}"),
        };

        if !content.contains("## CROSS-REFERENCES") {
            content.push_str("\n## CROSS-REFERENCES (synapses)\n");
        }
        // Use the relative path so neurons remain portable across machines.
        content.push_str(&format!(
            "\n- `{}` → {}",
            target_rel.display(),
            input.reason
        ));

        if let Err(e) = atomic_write(&source_path, content.as_bytes()) {
            return format!("ERROR: Failed to write synapse: {e}");
        }

        let meta_file = meta_path(&source_path);
        let mut meta = load_or_new_meta(&meta_file, &source_path, NeuronKind::Core);
        if let Some(source_hash) = hash_file(&meta.source_path) {
            meta.source_hash = source_hash;
        }
        meta.tokens = estimate_context_tokens(&content);
        meta.last_updated = now_iso8601();
        meta.status = NeuronStatus::Fresh;
        let edge_type = input.edge_type.unwrap_or(SynapseType::SemanticRelated);
        if !meta.synapses.iter().any(|s| s.target == target_path) {
            meta.synapses.push(Synapse::new(
                target_path.clone(),
                edge_type,
                input.reason.clone(),
            ));
        }
        if let Err(e) = save_meta(&meta_file, &meta) {
            tracing::warn!("Failed to save synapse meta: {e}");
        }
        let provenance_result = record_mutation_provenance(
            &source_path,
            &meta,
            &content,
            ProvenanceOperation::SectionUpdate,
            ProvenanceSource::Local,
            Some("cross-references".to_string()),
            Some(format!("added synapse to {}", target_rel.display())),
        );

        let mut idx = self.index.write().await;
        if let Err(e) = idx.upsert_neuron(&source_path, &content, &meta) {
            return format!("ERROR: Failed to update index: {e}");
        }
        finalize_mutation_message(
            format!(
                "Synapse created: {} → {} ({})",
                input.source, input.target, input.reason
            ),
            provenance_result,
        )
    }

    /// Force a neuron to be marked stale.
    #[tool(
        name = "cortyx_invalidate",
        description = "Mark a neuron stale, forcing re-evaluation on the next cortyx_get_contexts call."
    )]
    async fn invalidate(&self, Parameters(input): Parameters<InvalidateInput>) -> String {
        let rel = match validate_relative_path(&input.path) {
            Ok(p) => p,
            Err(e) => return format!("ERROR: Invalid path: {e}"),
        };
        let source = self.project_root.join(&rel);
        let mut idx = self.index.write().await;
        match idx.invalidate(&source) {
            Ok(()) => format!("Marked stale: {}", input.path),
            Err(e) => format!("ERROR: {e}"),
        }
    }

    // ── Hierarchy navigation tools (TRIZ R13-G2) ─────────────────────────────

    /// List all modules (directories and @person scopes) with their neuron count
    /// and average hit rate. Equivalent to MemPalace list_wings.
    #[tool(
        name = "cortyx_list_modules",
        description = "List all modules (code namespaces and @person scopes) with neuron count and avg hit rate. \
                       Equivalent to MemPalace list_wings. Returns JSON array."
    )]
    async fn list_modules(&self) -> String {
        let idx = self.index.read().await;
        let modules = idx.list_modules();
        if modules.is_empty() {
            return "No modules found. Run cortyx_compile first.".to_string();
        }
        let rows: Vec<serde_json::Value> = modules
            .iter()
            .map(|m| {
                serde_json::json!({
                    "name": m.name,
                    "neuron_count": m.neuron_count,
                    "avg_hit_rate": format!("{:.2}", m.avg_hit_rate),
                    "person_scope": m.is_person_scope,
                })
            })
            .collect();
        serde_json::to_string_pretty(&rows)
            .unwrap_or_else(|_| "ERROR: serialization failed".to_string())
    }

    /// List neurons in a module (or all neurons if module is omitted).
    /// Returns neuron paths, kind, staleness, and hit rate.
    #[tool(
        name = "cortyx_list_neurons",
        description = "List neurons in a module (or all neurons if module is omitted). \
                       Returns path, kind, staleness, and hit_rate for each neuron."
    )]
    async fn list_neurons(&self, Parameters(input): Parameters<ListNeuronsInput>) -> String {
        let idx = self.index.read().await;
        let neurons = idx.list_neurons(input.module.as_deref());
        if neurons.is_empty() {
            return format!(
                "No neurons found{}.",
                input
                    .module
                    .as_ref()
                    .map(|m| format!(" in module '{m}'"))
                    .unwrap_or_default()
            );
        }
        let rows: Vec<serde_json::Value> = neurons
            .iter()
            .map(|n| {
                serde_json::json!({
                    "path": self.rel_display(&n.path).as_ref().to_string(),
                    "kind": format!("{:?}", n.kind),
                    "staleness": format!("{:.1}", n.staleness_multiplier),
                    "hit_rate": format!("{:.2}", n.hit_rate),
                    "use_count": n.use_count,
                })
            })
            .collect();
        serde_json::to_string_pretty(&rows)
            .unwrap_or_else(|_| "ERROR: serialization failed".to_string())
    }

    /// Return the first N lines of a neuron file for quick preview.
    #[tool(
        name = "cortyx_peek_neuron",
        description = "Return the first 20 lines of a neuron file for quick preview without full activation. \
                       Path is the full neuron path as returned by cortyx_list_neurons."
    )]
    async fn peek_neuron(&self, Parameters(input): Parameters<PeekNeuronInput>) -> String {
        let path = match resolve_neuron_store_path(&input.path, &self.project_root) {
            Ok(path) => path,
            Err(err) => return format!("ERROR: Invalid neuron path: {err}"),
        };
        let preview = {
            let idx = self.index.read().await;
            idx.peek_neuron(&path, 20)
        };
        match preview {
            Some(p) => p,
            None => format!("ERROR: Neuron not found or unreadable: {}", input.path),
        }
    }

    // ── Person scope tools (TRIZ R13-G5) ─────────────────────────────────────

    /// Restore a single section of a neuron to its shadow copy (E2: section shadow, TRIZ R14).
    ///
    /// Before each evolve_context or evolve_section call, Cortyx automatically saves
    /// the previous content. Use this tool to step backward through recent evolutions.
    #[tool(
        name = "cortyx_rollback_section",
        description = "Restore a neuron section to its previous version (saved before recent evolve calls). \
                       Use section=\"_full\" to restore the entire neuron. \
                       Useful when an LLM evolution produces worse content than the original."
    )]
    async fn rollback_section(
        &self,
        Parameters(input): Parameters<RollbackSectionInput>,
    ) -> String {
        use crate::neuron::NeuronMeta;

        let neuron_path = match resolve_neuron_store_path(&input.neuron_path, &self.project_root) {
            Ok(path) => path,
            Err(err) => return format!("ERROR: Invalid neuron path: {err}"),
        };
        let meta_file = meta_path(&neuron_path);

        let meta_data = match std::fs::read_to_string(&meta_file) {
            Ok(d) => d,
            Err(e) => return format!("ERROR: Cannot read sidecar: {e}"),
        };
        let mut meta: NeuronMeta = match serde_json::from_str(&meta_data) {
            Ok(m) => m,
            Err(e) => return format!("ERROR: Cannot parse sidecar: {e}"),
        };

        let shadow = match latest_shadow(&meta.shadow_sections, &input.section) {
            Some(s) => s.to_string(),
            None => {
                return format!(
                    "ERROR: No shadow for section '{}'. Shadows are saved before each evolve call.",
                    input.section
                )
            },
        };

        if input.section == "_full" {
            if let Err(e) = atomic_write(&neuron_path, shadow.as_bytes()) {
                return format!("ERROR: Failed to write neuron: {e}");
            }
            pop_shadow(&mut meta.shadow_sections, "_full");
            refresh_meta_after_content_write(&mut meta, &shadow);
            if let Err(e) = save_meta(&meta_file, &meta) {
                return format!("ERROR: Failed to save meta: {e}");
            }
            let provenance_result = record_mutation_provenance(
                &neuron_path,
                &meta,
                &shadow,
                ProvenanceOperation::Rollback,
                ProvenanceSource::Local,
                None,
                Some("restored full neuron from rollback shadow".to_string()),
            );
            let mut idx = self.index.write().await;
            if let Err(e) = idx.upsert_neuron(&neuron_path, &shadow, &meta) {
                return format!("ERROR: Failed to update index: {e}");
            }
            finalize_mutation_message(
                format!(
                    "✓ Restored full neuron {} from shadow.",
                    self.rel_display(&neuron_path)
                ),
                provenance_result,
            )
        } else {
            let existing = match std::fs::read_to_string(&neuron_path) {
                Ok(c) => c,
                Err(e) => return format!("ERROR: Cannot read neuron file: {e}"),
            };
            let restored = replace_section(&existing, &input.section, &shadow);
            if let Err(e) = atomic_write(&neuron_path, restored.as_bytes()) {
                return format!("ERROR: Failed to write neuron: {e}");
            }
            pop_shadow(&mut meta.shadow_sections, &input.section);
            refresh_meta_after_content_write(&mut meta, &restored);
            if let Err(e) = save_meta(&meta_file, &meta) {
                return format!("ERROR: Failed to save meta: {e}");
            }
            let section_key = input.section.to_lowercase();
            let provenance_result = record_mutation_provenance(
                &neuron_path,
                &meta,
                &restored,
                ProvenanceOperation::Rollback,
                ProvenanceSource::Local,
                Some(section_key.clone()),
                Some(format!("restored {section_key} from rollback shadow")),
            );
            let mut idx = self.index.write().await;
            if let Err(e) = idx.upsert_neuron(&neuron_path, &restored, &meta) {
                return format!("ERROR: Failed to update index: {e}");
            }
            finalize_mutation_message(
                format!(
                    "✓ Restored section '{}' in {} from shadow.",
                    input.section,
                    self.rel_display(&neuron_path)
                ),
                provenance_result,
            )
        }
    }

    /// List all @person-scoped memory namespaces.
    #[tool(
        name = "cortyx_list_persons",
        description = "List all @person-scoped memory namespaces (created via mine_conversation with person=...). \
                       Returns person name, neuron count, and avg hit rate."
    )]
    async fn list_persons(&self) -> String {
        let idx = self.index.read().await;
        let persons = idx.list_persons();
        if persons.is_empty() {
            return "No person-scoped memories found. Use mine_conversation with person=\"alice\" to create some.".to_string();
        }
        let rows: Vec<serde_json::Value> = persons
            .iter()
            .map(|p| {
                serde_json::json!({
                    "person": p.name.trim_start_matches('@'),
                    "module": p.name,
                    "neuron_count": p.neuron_count,
                    "avg_hit_rate": format!("{:.2}", p.avg_hit_rate),
                })
            })
            .collect();
        serde_json::to_string_pretty(&rows)
            .unwrap_or_else(|_| "ERROR: serialization failed".to_string())
    }

    // ── Conversation recall tool (TRIZ R13-G3) ────────────────────────────────

    /// Retrieve conversation memories (Verbatim neurons) for a query.
    /// Isolates episodic recall from code retrieval — equivalent to MemPalace's
    /// dedicated episodic store but with zero storage overhead (query-time predicate).
    #[tool(
        name = "cortyx_recall",
        description = "Retrieve conversation memories (Verbatim neurons) matching a query. \
                       Optionally scope to a person's memories with person=\"alice\". \
                       Use for 'what did I decide last month?' style queries."
    )]
    async fn recall(&self, Parameters(input): Parameters<RecallInput>) -> String {
        let idx = self.index.read().await;
        let effective_module: Option<String> = input.person.as_ref().map(|p| format!("@{}", p));
        let paths = idx.get_contexts(
            &input.query,
            input.max_tokens.unwrap_or(4096),
            effective_module.as_deref(),
            Some("conversation"),
        );
        if paths.is_empty() {
            return "No conversation memories found for this query. \
                    Use cortyx_mine_conversation to index conversations first."
                .to_string();
        }
        let mut out = format!("<!-- cortyx:recall {} memories -->\n", paths.len());
        for path in &paths {
            if let Ok(content) = std::fs::read_to_string(path) {
                out.push_str(&content);
                out.push('\n');
            }
        }
        out
    }

    /// Show neuron stats and cache-hit prediction.
    #[tool(
        name = "cortyx_status",
        description = "Show neuron count, synapse count, freshness, and cache-hit prediction."
    )]
    async fn status(&self) -> String {
        let idx = self.index.read().await;
        let low_quality = idx.low_quality_count();
        let quality_note = if low_quality > 0 {
            format!("\nNeeds curation (quality<40%): {low_quality}")
        } else {
            String::new()
        };
        format!(
            "Cortyx Status\n\
             =============\n\
             Neurons (total):       {}\n\
             Synapses:              {}{}\n\
             \n\
             Prompt caching:        ✓ Static prefix byte-identical on every call\n\
             Activation latency:    ~BM25 in-memory (<10ms for <10k neurons)\n\
             Instructions: Call cortyx_get_contexts(task) at the start of each task.",
            idx.neuron_count(),
            idx.synapse_count(),
            quality_note
        )
    }

    /// Mine a raw conversation turn into the live index as a Verbatim neuron.
    ///
    /// Accepts any format: Claude MD, ChatGPT JSON, LongMemEval JSON, or plain text.
    /// Consecutive calls automatically create TemporalFollows synapse chains.
    /// Use `module` to tag memories for namespace-filtered retrieval.
    #[tool(
        name = "cortyx_mine_conversation",
        description = "Mine a conversation turn (or whole file export) into Verbatim neurons for semantic recall. \
                       Returns the number of neurons created and the first neuron path."
    )]
    async fn mine_conversation(
        &self,
        Parameters(input): Parameters<MineConversationInput>,
    ) -> String {
        if input.content.len() > MAX_CONTENT_BYTES {
            return format!("ERROR: content exceeds {MAX_CONTENT_BYTES} byte limit");
        }
        let mut idx = self.index.write().await;
        let effective_module: Option<String> = input
            .person
            .as_ref()
            .map(|p| format!("@{}", p))
            .or_else(|| input.module.clone());
        match miner::mine_text(
            &input.content,
            "mcp-inline",
            &self.project_root,
            &mut idx,
            effective_module.as_deref(),
            input.speaker.as_deref(),
            input.timestamp.as_deref(),
        ) {
            Ok(count) => format!(
                "Mined {count} Verbatim neuron(s). Total neurons: {}.",
                idx.neuron_count()
            ),
            Err(e) => format!("ERROR: {e}"),
        }
    }

    /// Implicit hit-rate feedback — scan the task response for cited neurons.
    ///
    /// Pass the assistant's full response text; Cortyx scans for neuron file stems
    /// and auto-increments hit_count for each match. No per-neuron tool calls needed.
    /// Call once at task end instead of cortyx_record_hit for each neuron.
    #[tool(
        name = "cortyx_close_task",
        description = "Pass the assistant response text. Cortyx auto-records hits for neurons whose filenames appear in the response. Zero friction — one call closes the feedback loop for the whole task."
    )]
    async fn close_task(&self, Parameters(input): Parameters<CloseTaskInput>) -> String {
        if input.response_text.len() > MAX_CONTENT_BYTES {
            return format!("ERROR: response_text exceeds {MAX_CONTENT_BYTES} byte limit");
        }
        // R12-S2-B: Clear provisional hits — close_task provides actual citation evidence,
        // so the optimistic provisional buffer is no longer needed.
        self.provisional_hits.lock().await.clear();
        let activated = self.last_activated.lock().await.clone();
        if activated.is_empty() {
            return "No neurons from last cortyx_get_contexts call to evaluate.".to_string();
        }

        let response_lower = input.response_text.to_lowercase();
        // C1: Graded response-diff citation (TRIZ R14).
        // Tokenize response text and compute overlap with each activated neuron's vocabulary.
        // ≥15 terms → soft cite (record_hit once)
        // ≥30 terms → hard cite (record_hit twice — stronger feedback signal)
        // This is a tighter, graded version of the prior flat ≥20-term threshold.
        let response_tokens: std::collections::HashSet<String> =
            tokenize(&input.response_text).into_iter().collect();
        let mut hits = 0usize;

        // Phase 1 (immutable): decide citation for each activated neuron.
        // Returns (path, explicit_cited, soft_weight) where soft_weight: 0=miss, 1=soft, 2=hard
        let citation_decisions: Vec<(PathBuf, bool, u8)> = {
            let idx = self.index.read().await;
            activated
                .iter()
                .map(|path| {
                    let stem = path
                        .file_stem()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_lowercase();
                    let stem = stem.trim_end_matches(".context");
                    let explicit_cited = !stem.is_empty() && response_lower.contains(stem);
                    if explicit_cited {
                        return (path.clone(), true, 1u8);
                    }
                    // C1 graded: measure term overlap
                    let overlap = if !response_tokens.is_empty() {
                        idx.term_freq_overlap(path, &response_tokens)
                    } else {
                        0
                    };
                    let weight = if overlap >= 30 {
                        2 // hard cite
                    } else if overlap >= 15 {
                        1 // soft cite
                    } else {
                        0 // miss
                    };
                    (path.clone(), weight >= 1, weight)
                })
                .collect()
        };

        // Phase 2 (mutable): apply citation signals.
        let mut idx = self.index.write().await;
        for (path, cited, weight) in &citation_decisions {
            idx.record_hit(path, *cited);
            if *cited {
                hits += 1;
                // Hard cite: record a second hit to double the feedback signal
                if *weight >= 2 {
                    idx.record_hit(path, true);
                    tracing::debug!(path = %path.display(), "Hard citation (≥30 term overlap)");
                } else if *weight == 1 {
                    tracing::debug!(path = %path.display(), "Soft citation (≥15 term overlap)");
                }
            }
        }

        // S-VII (R16): Update last_co_activation_day for all cited neuron pairs (LTP).
        let cited_paths: Vec<PathBuf> = citation_decisions
            .iter()
            .filter(|(_, cited, _)| *cited)
            .map(|(p, _, _)| p.clone())
            .collect();
        if cited_paths.len() >= 2 {
            idx.touch_co_activation_day(&cited_paths);
        }

        // S-VIII (R16): Auto-mine UseCase stubs from code blocks in the response.
        // Code blocks ≥5 lines with ≥60% overlap with a cited neuron → write stub.
        let mined =
            auto_mine_code_blocks(&input.response_text, &cited_paths, &self.project_root, &idx);
        let mined_note = if mined > 0 {
            format!(" Auto-mined {mined} UseCase stub(s).")
        } else {
            String::new()
        };

        // F2: Record session token utilization for budget adaptation.
        // Count total tokens used in this session from the activated neurons.
        let tokens_used: usize = activated.iter().map(|p| idx.tokens_for(p)).sum();
        let tokens_budget = input.response_text.len() / 4; // rough estimate of budget from response size; actual budget not stored here
        if tokens_used > 0 {
            idx.record_session_utilization(tokens_used, tokens_budget.max(1));
            if let Err(e) = idx.save() {
                tracing::warn!("Failed to persist close_task feedback: {e}");
                return format!(
                    "Closed task: {hits}/{} neurons cited (auto-detected from response).{} Warning: feedback was applied in-memory but could not be saved: {e}",
                    activated.len(),
                    mined_note
                );
            }
        }

        format!(
            "Closed task: {hits}/{} neurons cited (auto-detected from response).{mined_note}",
            activated.len()
        )
    }

    /// Record whether a neuron was actually cited — closes the self-improvement feedback loop.
    ///
    /// Call after each task for each neuron returned by cortyx_get_contexts.
    /// Cited neurons get a higher hit_rate → higher BM25 score → activated more readily in future.
    /// Irrelevant neurons are down-weighted over time without any manual curation.
    #[tool(
        name = "cortyx_record_hit",
        description = "Tell Cortyx whether a neuron was actually useful. was_cited=true boosts it; false down-weights it. Closes the self-improvement loop — neurons get smarter with every task."
    )]
    async fn record_hit(&self, Parameters(input): Parameters<RecordHitInput>) -> String {
        let rel = match validate_relative_path(&input.path) {
            Ok(p) => p,
            Err(e) => return format!("ERROR: Invalid path: {e}"),
        };
        let source = self.project_root.join(&rel);
        let neuron_path = core_neuron_path(&source, &self.project_root);

        if !neuron_path.exists() {
            return format!(
                "ERROR: Neuron not found: {}",
                self.rel_display(&neuron_path)
            );
        }

        let mut idx = self.index.write().await;
        let hit_rate = idx.record_hit(&neuron_path, input.was_cited);
        let use_count = idx.use_count_for(&neuron_path);

        format!(
            "Recorded {} for {} — hit_rate now {:.0}% ({} uses)",
            if input.was_cited { "hit" } else { "miss" },
            self.rel_display(&neuron_path),
            hit_rate * 100.0,
            use_count
        )
    }

    /// Write an agent diary entry (S6 — NE5).
    ///
    /// Agent diaries are Verbatim neurons stored under the `@agent/{agent}` module namespace.
    /// They use the existing @prefix isolation mechanism — zero new storage, zero new retrieval
    /// logic. Each diary entry is BM25-indexed and searchable via cortyx_get_contexts.
    #[tool(
        name = "cortyx_diary_write",
        description = "Write an observation or decision to an agent's diary. Stored as a Verbatim neuron under @agent/{agent} — BM25-indexed, searchable via get_contexts. Optional title/status/goal/next_step/blocker/outcome/entities/depends_on fields turn it into structured agent-state memory without adding a new storage layer."
    )]
    async fn diary_write(&self, Parameters(input): Parameters<DiaryWriteInput>) -> String {
        if input.agent.is_empty() {
            return "ERROR: agent name must not be empty".to_string();
        }
        let entities = input.entities.clone().unwrap_or_default();
        let depends_on = input.depends_on.clone().unwrap_or_default();
        let structured = has_structured_diary_fields(
            input.title.as_deref(),
            input.status.as_deref(),
            input.goal.as_deref(),
            input.next_step.as_deref(),
            input.blocker.as_deref(),
            input.outcome.as_deref(),
            &entities,
            &depends_on,
        );
        let body = if structured {
            render_structured_diary_entry(
                input.agent.trim(),
                &input.content,
                input.title.as_deref(),
                input.status.as_deref(),
                input.goal.as_deref(),
                input.next_step.as_deref(),
                input.blocker.as_deref(),
                input.outcome.as_deref(),
                &entities,
                &depends_on,
            )
        } else {
            input.content.trim().to_string()
        };
        if body.is_empty() {
            return "ERROR: content must not be empty unless structured diary fields are supplied"
                .to_string();
        }
        if body.len() > MAX_CONTENT_BYTES {
            return format!("ERROR: content exceeds {MAX_CONTENT_BYTES} byte limit");
        }
        let structured_entry = structured
            .then(|| parse_structured_diary_entry(&body))
            .flatten();
        let effective_timestamp = input.timestamp.clone().unwrap_or_else(now_iso8601);
        let module = format!("@agent/{}", input.agent.trim());
        let mut idx = self.index.write().await;
        match miner::mine_text(
            &body,
            "diary",
            &self.project_root,
            &mut idx,
            Some(&module),
            Some(input.agent.trim()),
            Some(effective_timestamp.as_str()),
        ) {
            Ok(count) => {
                if let Some(entry) = structured_entry.as_ref() {
                    if let Err(err) = sync_structured_diary_to_kg(
                        &self.project_root,
                        &mut idx,
                        input.agent.trim(),
                        entry,
                        &effective_timestamp,
                    ) {
                        return format!("ERROR syncing agent memory to temporal KG: {err}");
                    }
                    format!(
                        "Diary entry written for agent '{}' ({count} neuron(s) created, temporal KG synced).",
                        input.agent
                    )
                } else {
                    format!(
                        "Diary entry written for agent '{}' ({count} neuron(s) created).",
                        input.agent
                    )
                }
            },
            Err(e) => format!("ERROR: {e}"),
        }
    }

    /// Read recent diary entries for an agent (S6 — NE5).
    #[tool(
        name = "cortyx_diary_read",
        description = "Read recent diary entries for an agent. Returns last_n entries (default 10) from @agent/{agent} namespace, most recent first. Structured action memories are summarized with status/outcome/entity fields."
    )]
    async fn diary_read(&self, Parameters(input): Parameters<DiaryReadInput>) -> String {
        if input.agent.is_empty() {
            return "ERROR: agent name must not be empty".to_string();
        }
        let last_n = input.last_n.unwrap_or(10);
        let module = format!("@agent/{}", input.agent.trim());
        let idx = self.index.read().await;
        let results = recent_module_paths(&idx, &module, last_n, Some(NeuronKind::Verbatim));
        if results.is_empty() {
            return format!("No diary entries found for agent '{}'.", input.agent);
        }
        let mut out = format!("## Agent Diary: {} (last {})\n\n", input.agent, last_n);
        for path in results {
            let timestamp_secs = idx
                .context_metadata_for(&path)
                .and_then(|metadata| metadata.timestamp_secs);
            match std::fs::read_to_string(&path) {
                Ok(content) => {
                    if let Some(entry) = parse_structured_diary_entry(&content) {
                        out.push_str(&render_structured_diary_history_entry(
                            &entry,
                            timestamp_secs,
                        ));
                    } else {
                        out.push_str(&format!("---\n{}\n", content));
                    }
                },
                Err(err) => {
                    out.push_str(&format!(
                        "- {} — read error: {}\n",
                        path.display(),
                        sanitize_comment(&err.to_string())
                    ));
                },
            }
        }
        out
    }

    #[tool(
        name = "cortyx_agent_status",
        description = "Show the latest structured collaboration snapshot for an agent by combining recent @agent diary entries, mirrored temporal KG facts, and shared-sync status. Useful for specialist-agent handoff, wake-up, and coordination."
    )]
    async fn agent_status(&self, Parameters(input): Parameters<AgentStatusInput>) -> String {
        if input.agent.trim().is_empty() {
            return "ERROR: agent name must not be empty".to_string();
        }
        let idx = self.index.read().await;
        render_agent_status_report(
            &idx,
            &self.project_root,
            input.agent.trim(),
            input.include_timeline.unwrap_or(false),
        )
        .unwrap_or_else(|| {
            format!(
                "No structured agent memory found for agent '{}'.",
                input.agent
            )
        })
    }

    #[tool(
        name = "cortyx_collaboration_status",
        description = "Summarize collaboration-kernel state across agents, shared modules, and sync activity. Optionally scope to one agent or one module, and append recent collaboration timeline events."
    )]
    async fn collaboration_status(
        &self,
        Parameters(input): Parameters<CollaborationStatusInput>,
    ) -> String {
        if input
            .agent
            .as_deref()
            .is_some_and(|agent| agent.trim().is_empty())
        {
            return "ERROR: agent name must not be empty".to_string();
        }
        if input
            .module
            .as_deref()
            .is_some_and(|module| module.trim().is_empty())
        {
            return "ERROR: module name must not be empty".to_string();
        }
        if input.agent.is_some() && input.module.is_some() {
            return "ERROR: agent and module filters are mutually exclusive".to_string();
        }

        let idx = self.index.read().await;
        let projection = build_collaboration_projection(&idx, &self.project_root);
        render_collaboration_status_report(
            &projection,
            input.agent.as_deref(),
            input.module.as_deref(),
            input.include_timeline.unwrap_or(false),
        )
        .unwrap_or_else(|| {
            if let Some(agent) = input.agent.as_deref() {
                format!("No collaboration state found for agent '{}'.", agent.trim())
            } else if let Some(module) = input.module.as_deref() {
                format!(
                    "No collaboration state found for module '{}'.",
                    module.trim()
                )
            } else {
                "No collaboration state found.".to_string()
            }
        })
    }

    /// Check for contradicting neuron pairs (S7 — NE6).
    ///
    /// Proactively scans all neurons (or a single neuron) for `Contradicts` synapse edges.
    /// Use before starting a task to surface known conflicts. Contradictions are also
    /// automatically surfaced by `cortyx_get_contexts` at query time.
    #[tool(
        name = "cortyx_check_consistency",
        description = "Check for contradictions in the neuron graph. Scans all Contradicts synapse edges and returns conflicting pairs with reasons. Scope to a single neuron with the optional path argument. Contradictions are also surfaced automatically during cortyx_get_contexts."
    )]
    async fn check_consistency(
        &self,
        Parameters(input): Parameters<CheckConsistencyInput>,
    ) -> String {
        let path_filter: Option<PathBuf> = if let Some(ref p) = input.path {
            match validate_relative_path(p) {
                Ok(rel) => {
                    let src = self.project_root.join(&rel);
                    Some(core_neuron_path(&src, &self.project_root))
                },
                Err(e) => return format!("ERROR: Invalid path: {e}"),
            }
        } else {
            None
        };

        let idx = self.index.read().await;
        let pairs = idx.all_contradictions(path_filter.as_deref());

        if pairs.is_empty() {
            return "No contradictions detected.".to_string();
        }

        let mut out = format!("## Contradictions Found ({})\n\n", pairs.len());
        for (a, b, reason) in &pairs {
            let a_name = a.file_name().unwrap_or_default().to_string_lossy();
            let b_name = b.file_name().unwrap_or_default().to_string_lossy();
            out.push_str(&format!(
                "- **{}** ↔ **{}**\n  Reason: {}\n  Action: use `cortyx_create_synapse` to update or \
                 `cortyx_invalidate` to retire the outdated neuron.\n\n",
                a_name, b_name, reason
            ));
        }
        out
    }

    /// Session priming — load identity and critical-facts wake-up neurons (S5 — NE4).
    ///
    /// Returns both L0 (_identity) and L1 (_critical_facts) neurons (~170 tokens total)
    /// plus optional @person memories. Call at the start of a new session to prime the
    /// LLM with project identity — equivalent to MemPalace L0+L1 but lossless (plain
    /// Markdown, not AAAK-encoded).
    ///
    /// Zero tokens unless explicitly called — preserves Cortyx's token efficiency advantage.
    #[tool(
        name = "cortyx_wake_up",
        description = "Prime the LLM with project identity and critical facts. Returns _identity.context.md (~50 tokens) + _critical_facts.context.md (~120 tokens). Optionally include person memories and recent structured agent memories. Call once at session start — lossless Markdown (vs MemPalace AAAK-encoding)."
    )]
    async fn wake_up(&self, Parameters(input): Parameters<WakeUpInput>) -> String {
        let ndir = neuron_dir(&self.project_root);
        let mut out = String::new();

        // Load identity neuron (L0)
        let identity_path = ndir.join("_identity.context.md");
        if identity_path.exists() {
            if let Ok(content) = std::fs::read_to_string(&identity_path) {
                out.push_str("<!-- CORTYX WAKE-UP: L0 identity -->\n");
                out.push_str(&content);
                out.push('\n');
            }
        } else {
            out.push_str(
                "<!-- _identity.context.md not found — run `cortyx compile .` first -->\n",
            );
        }

        // Load critical-facts neuron (L1)
        let critical_path = ndir.join("_critical_facts.context.md");
        if critical_path.exists() {
            if let Ok(content) = std::fs::read_to_string(&critical_path) {
                out.push_str("<!-- CORTYX WAKE-UP: L1 critical facts -->\n");
                out.push_str(&content);
                out.push('\n');
            }
        } else {
            out.push_str(
                "<!-- _critical_facts.context.md not found — run `cortyx compile .` first -->\n",
            );
        }

        if input.person.is_some() || input.agent.is_some() {
            let idx = self.index.read().await;

            // Optional @person memories (recent conversation neurons for the person)
            if let Some(ref person) = input.person {
                let module = format!("@{}", person.trim());
                let paths = idx.get_contexts("", 600, Some(&module), Some("conversation"));
                if !paths.is_empty() {
                    out.push_str(&format!("\n<!-- CORTYX WAKE-UP: @{person} memories -->\n"));
                    for path in paths.iter().take(3) {
                        if let Ok(content) = std::fs::read_to_string(path) {
                            out.push_str(&content);
                            out.push('\n');
                        }
                    }
                }
            }

            if let Some(ref agent) = input.agent {
                if let Some(block) = render_recent_agent_memory_block(&idx, agent, 3) {
                    out.push('\n');
                    out.push_str(&block);
                }
            }
        }

        if out.is_empty() {
            out.push_str("No wake-up neurons found. Run `cortyx compile .` to generate them.");
        }
        out
    }

    // ─── S4: Temporal Knowledge Graph (NE3) ──────────────────────────────────

    /// Add a fact to a KG entity neuron (creating the entity if needed).
    #[tool(
        name = "cortyx_kg_add",
        description = "Add a fact triple to a KG entity (creates entity if absent). \
                       KG neurons are git-tracked, BM25-indexed Markdown files. \
                       Example: entity='project_meta', predicate='language', value='Rust', valid_from='2024-01-01'."
    )]
    async fn kg_add(&self, Parameters(input): Parameters<KgAddInput>) -> String {
        let path = kg::kg_neuron_path(&self.project_root, &input.entity);
        let mut entity = match kg::KgEntity::load(&path) {
            Ok(e) => e,
            Err(e) => return format!("ERROR loading KG entity: {e}"),
        };
        entity.add_fact(&input.predicate, &input.value, input.valid_from.as_deref());
        if let Err(e) = entity.save() {
            return format!("ERROR saving KG entity: {e}");
        }
        let mut idx = self.index.write().await;
        if let Err(err) = index_kg_entity_path(&mut idx, &path) {
            return format!(
                "ERROR reloading KG entity {} after save: {err}",
                path.display()
            );
        }
        format!(
            "KG fact added: {entity} / {pred} = {val} (from: {from})",
            entity = input.entity,
            pred = input.predicate,
            val = input.value,
            from = input.valid_from.as_deref().unwrap_or(""),
        )
    }

    /// Query active facts for a KG entity as of an optional date.
    #[tool(
        name = "cortyx_kg_query",
        description = "Query active facts for a KG entity. Pass as_of (ISO-8601) to filter by date. \
                       Returns a Markdown table of active fact triples."
    )]
    async fn kg_query(&self, Parameters(input): Parameters<KgQueryInput>) -> String {
        let path = kg::kg_neuron_path(&self.project_root, &input.entity);
        let entity = match kg::KgEntity::load(&path) {
            Ok(e) => e,
            Err(e) => return format!("ERROR loading KG entity: {e}"),
        };
        let facts = entity.active_facts(input.as_of.as_deref());
        if facts.is_empty() {
            return format!(
                "No active facts for entity '{}' (as_of: {:?})",
                input.entity, input.as_of
            );
        }
        let mut out = format!("## KG: {} (active facts)\n\n| predicate | value | valid_from | ended |\n|---|---|---|---|\n", input.entity);
        for f in facts {
            out.push_str(&format!(
                "| {} | {} | {} | {} |\n",
                f.predicate, f.value, f.valid_from, f.ended
            ));
        }
        out
    }

    /// Invalidate (end) an active KG fact by setting its `ended` date.
    #[tool(
        name = "cortyx_kg_invalidate",
        description = "Invalidate (end) the currently active fact for a predicate on a KG entity. \
                       Sets the `ended` date; does NOT delete the historical record."
    )]
    async fn kg_invalidate(&self, Parameters(input): Parameters<KgInvalidateInput>) -> String {
        let path = kg::kg_neuron_path(&self.project_root, &input.entity);
        let mut entity = match kg::KgEntity::load(&path) {
            Ok(e) => e,
            Err(e) => return format!("ERROR loading KG entity: {e}"),
        };
        if let Err(e) = entity.invalidate_fact(&input.predicate, &input.ended) {
            return format!("ERROR: {e}");
        }
        if let Err(e) = entity.save() {
            return format!("ERROR saving KG entity: {e}");
        }
        let mut idx = self.index.write().await;
        if let Err(err) = index_kg_entity_path(&mut idx, &path) {
            return format!(
                "ERROR reloading KG entity {} after save: {err}",
                path.display()
            );
        }
        format!(
            "KG fact invalidated: {}/{} ended on {}",
            input.entity, input.predicate, input.ended
        )
    }

    /// Show the full temporal timeline for a predicate on a KG entity.
    #[tool(
        name = "cortyx_kg_timeline",
        description = "Show the full temporal history of a predicate on a KG entity — all past, \
                       present, and future values with their validity windows."
    )]
    async fn kg_timeline(&self, Parameters(input): Parameters<KgTimelineInput>) -> String {
        let path = kg::kg_neuron_path(&self.project_root, &input.entity);
        let entity = match kg::KgEntity::load(&path) {
            Ok(e) => e,
            Err(e) => return format!("ERROR loading KG entity: {e}"),
        };
        let timeline = entity.timeline_for(&input.predicate);
        if timeline.is_empty() {
            return format!("No facts found for {}/{}", input.entity, input.predicate);
        }
        let mut out = format!(
            "## Timeline: {}/{}\n\n| # | value | valid_from | ended |\n|---|---|---|---|\n",
            input.entity, input.predicate
        );
        for (i, f) in timeline.iter().enumerate() {
            let ended = if f.ended.is_empty() {
                "active"
            } else {
                &f.ended
            };
            out.push_str(&format!(
                "| {} | {} | {} | {} |\n",
                i + 1,
                f.value,
                f.valid_from,
                ended
            ));
        }
        out
    }

    /// Return aggregate statistics for all KG entities in this project.
    #[tool(
        name = "cortyx_kg_stats",
        description = "Return aggregate statistics for all KG entities: entity count, total facts, \
                       active facts, ended/invalidated facts."
    )]
    async fn kg_stats(&self, _params: Parameters<serde_json::Value>) -> String {
        let stats = kg::compute_stats(&self.project_root);
        format!(
            "KG stats: {} entities, {} total facts ({} active, {} ended)",
            stats.entity_count, stats.total_facts, stats.active_facts, stats.ended_facts
        )
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

/// Start the MCP server on STDIO (compatible with Claude Code, Cursor, Codex, Windsurf).
pub async fn serve(project_name: Option<String>) -> Result<()> {
    if let Some(ref name) = project_name {
        tracing::warn!(
            "--project '{}' is accepted but not yet implemented — \
             planned for v0.2 multi-folder support. Server will use the current directory.",
            name
        );
    }
    let project_root = std::env::current_dir()?;
    tracing::info!("Starting Cortyx MCP server for: {}", project_root.display());

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

                let remote_ok = tokio::process::Command::new("git")
                    .args(["remote", "get-url", "origin"])
                    .current_dir(&global_dir)
                    .output()
                    .await
                    .map(|o| o.status.success())
                    .unwrap_or(false);
                if !remote_ok {
                    return;
                }

                // 5-second hard timeout — network hang cannot stall the server.
                let pull_fut = async {
                    tokio::process::Command::new("git")
                        .args(["pull", "--ff-only", "origin", "main"])
                        .current_dir(&global_dir)
                        .output()
                        .await
                        .or(tokio::process::Command::new("git")
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

    // Embed feature active — hybrid BM25 + dense retrieval is wired into get_contexts.
    // Embeddings will be loaded from .cortyx/embeddings.bin if present; falls back
    // gracefully to BM25-only when embeddings.bin is absent or model not installed.
    #[cfg(feature = "embed")]
    tracing::info!("--features embed: hybrid BM25 + dense cosine retrieval active.");

    let index = Arc::new(RwLock::new(idx));
    let provisional_hits = Arc::new(Mutex::new(Vec::new()));
    let context_sessions = Arc::new(Mutex::new(HashMap::new()));
    let next_context_handle = Arc::new(AtomicU64::new(0));

    let _watcher = watcher::start_watcher(project_root.clone(), Arc::clone(&index))?;

    let server = CortyxServer {
        project_root,
        index: Arc::clone(&index),
        last_activated: Arc::new(Mutex::new(Vec::new())),
        provisional_hits: Arc::clone(&provisional_hits),
        context_sessions,
        next_context_handle,
        tool_router: CortyxServer::tool_router(),
    };

    let service = server.serve(rmcp::transport::stdio()).await?;
    service.waiting().await?;
    let flushed = flush_provisional_hits_async(&index, &provisional_hits).await?;
    if flushed > 0 {
        tracing::info!("S2: explicitly cleared {flushed} provisional paths before shutdown");
    }
    Ok(())
}

// ─── Helpers ─────────────────────────────────────────────────────────────────/// Convert a task pattern string to a URL-safe kebab-case identifier.
fn to_kebab(s: &str) -> String {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s: &&str| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}

/// Truncate a string to at most `max_chars` characters (byte boundary safe for ASCII).
fn truncate_str(s: &str, max_chars: usize) -> String {
    s.chars().take(max_chars).collect()
}

fn recent_module_paths(
    index: &NeuronIndex,
    module: &str,
    limit: usize,
    kind_filter: Option<NeuronKind>,
) -> Vec<PathBuf> {
    let mut items: Vec<(i64, PathBuf)> = index
        .list_neurons(Some(module))
        .into_iter()
        .filter(|summary| {
            kind_filter
                .as_ref()
                .map(|kind| summary.kind == *kind)
                .unwrap_or(true)
        })
        .map(|summary| {
            let timestamp = index
                .context_metadata_for(&summary.path)
                .and_then(|metadata| metadata.timestamp_secs)
                .unwrap_or(i64::MIN);
            (timestamp, summary.path)
        })
        .collect();
    items.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    items
        .into_iter()
        .take(limit)
        .map(|(_, path)| path)
        .collect()
}

const DEFAULT_COLLAB_SUMMARY_LIMIT: usize = 5;
const DEFAULT_COLLAB_TIMELINE_LIMIT: usize = 8;

fn build_collaboration_projection(
    index: &NeuronIndex,
    project_root: &Path,
) -> CollaborationStateProjection {
    let diaries = collect_collaboration_diaries(index);
    let sync_statuses = load_collaboration_sync_statuses(project_root);
    let kg_entities = load_collaboration_kg_entities(project_root);
    let reasoning = build_collaboration_reasoning(index, project_root, &diaries, &sync_statuses);
    project_collaboration_state(&diaries, &sync_statuses, &kg_entities, reasoning.as_ref())
}

fn collect_collaboration_diaries(index: &NeuronIndex) -> Vec<CollaborationDiaryRecord> {
    let mut diaries = Vec::new();
    for summary in index.list_neurons(None) {
        if summary.kind != NeuronKind::Verbatim {
            continue;
        }
        let Some(module) = index.module_for(&summary.path) else {
            continue;
        };
        let Some(collaborator) = module.strip_prefix("@agent/") else {
            continue;
        };
        let collaborator = collaborator.trim();
        if collaborator.is_empty() {
            continue;
        }
        let Ok(content) = std::fs::read_to_string(&summary.path) else {
            continue;
        };
        let Some(entry) = parse_structured_diary_entry(&content) else {
            continue;
        };
        let when = index
            .context_metadata_for(&summary.path)
            .and_then(|metadata| metadata.timestamp_secs)
            .map(format_timestamp_secs);
        diaries.push(CollaborationDiaryRecord {
            collaborator: collaborator.to_string(),
            when,
            path: Some(summary.path.clone()),
            entry,
        });
    }
    diaries.sort_by(|left, right| {
        right
            .when
            .cmp(&left.when)
            .then_with(|| left.collaborator.cmp(&right.collaborator))
            .then_with(|| left.path.cmp(&right.path))
    });
    diaries
}

fn load_collaboration_sync_statuses(project_root: &Path) -> Vec<SyncTransportStatus> {
    let sync_root = sync_transport_dir(project_root);
    if !sync_root.exists() {
        return Vec::new();
    }
    SyncTransportRepository::for_project(project_root)
        .list_statuses()
        .unwrap_or_default()
}

fn load_collaboration_kg_entities(project_root: &Path) -> Vec<kg::KgEntity> {
    kg::list_kg_paths(project_root)
        .into_iter()
        .filter_map(|path| kg::KgEntity::load(&path).ok())
        .collect()
}

fn build_collaboration_reasoning(
    index: &NeuronIndex,
    project_root: &Path,
    diaries: &[CollaborationDiaryRecord],
    sync_statuses: &[SyncTransportStatus],
) -> Option<ReasoningReport> {
    let mut seeds = HashMap::new();
    for diary in diaries {
        if let Some(path) = &diary.path {
            merge_collaboration_seed(&mut seeds, path.clone(), 0.6);
        }
        for entity in diary
            .entry
            .entities
            .iter()
            .chain(diary.entry.depends_on.iter())
        {
            let kg_path = kg::kg_neuron_path(project_root, entity);
            if kg_path.exists() {
                merge_collaboration_seed(&mut seeds, kg_path, 0.8);
            }
        }
    }
    for status in sync_statuses {
        if let Some(source_path) = status.source_path() {
            let source = if source_path.is_absolute() {
                source_path.to_path_buf()
            } else {
                project_root.join(source_path)
            };
            if source.starts_with(project_root) {
                let neuron_path = core_neuron_path(&source, project_root);
                if neuron_path.exists() {
                    merge_collaboration_seed(&mut seeds, neuron_path, 1.0);
                }
            }
        }
        if let Some(module) = status.module() {
            let kg_path = kg::kg_neuron_path(project_root, module);
            if kg_path.exists() {
                merge_collaboration_seed(&mut seeds, kg_path, 0.7);
            }
        }
    }
    if seeds.is_empty() {
        return None;
    }

    let report = index.reason_over_paths(
        &seeds.into_iter().collect::<Vec<_>>(),
        TraversalOptions {
            max_hops: 1,
            max_expansions: 32,
            ..TraversalOptions::default()
        },
    );
    (!report.nodes.is_empty() || !report.facts.is_empty() || !report.conflicts.is_empty())
        .then_some(report)
}

fn merge_collaboration_seed(seeds: &mut HashMap<PathBuf, f32>, path: PathBuf, score: f32) {
    seeds
        .entry(path)
        .and_modify(|existing| *existing = existing.max(score))
        .or_insert(score);
}

/// Load existing metadata or create a fresh stub.
fn load_or_new_meta(meta_file: &Path, source: &Path, kind: NeuronKind) -> NeuronMeta {
    if let Ok(data) = std::fs::read_to_string(meta_file) {
        if let Ok(meta) = serde_json::from_str::<NeuronMeta>(&data) {
            return meta;
        }
    }
    NeuronMeta::new_stub(source, kind)
}

/// Serialize and write metadata to disk atomically.
fn save_meta(meta_file: &Path, meta: &NeuronMeta) -> Result<()> {
    atomic_write_json(meta_file, meta)
}

fn refresh_meta_after_content_write(meta: &mut NeuronMeta, content: &str) {
    if let Some(source_hash) = hash_file(&meta.source_path) {
        meta.source_hash = source_hash;
    }
    meta.tokens = estimate_context_tokens(content);
    meta.last_updated = now_iso8601();
    meta.status = NeuronStatus::Fresh;
    meta.synapses = parse_synapses_from_content(content);
}

fn record_mutation_provenance(
    neuron_path: &Path,
    meta: &NeuronMeta,
    content: &str,
    operation: ProvenanceOperation,
    source: ProvenanceSource,
    section: Option<String>,
    summary: Option<String>,
) -> Result<()> {
    record_content_provenance_edit(
        neuron_path,
        meta,
        content,
        ProvenanceEdit {
            operation,
            source,
            section,
            summary,
            ..Default::default()
        },
    )
    .map(|_| ())
}

fn finalize_mutation_message(message: String, provenance_result: Result<()>) -> String {
    match provenance_result {
        Ok(()) => message,
        Err(err) => format!("{message}\nWARNING: Failed to record provenance: {err}"),
    }
}

fn resolve_neuron_store_path(raw_path: &str, project_root: &Path) -> Result<PathBuf> {
    let neuron_root = neuron_dir(project_root)
        .canonicalize()
        .map_err(|err| anyhow::anyhow!("cannot access neuron directory: {err}"))?;
    let candidate = Path::new(raw_path);
    let resolved = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        neuron_root.join(candidate)
    };
    let canonical = resolved
        .canonicalize()
        .map_err(|err| anyhow::anyhow!("cannot access neuron path: {err}"))?;
    if !canonical.starts_with(&neuron_root) {
        anyhow::bail!(
            "path {} is outside neuron directory {}",
            canonical.display(),
            neuron_root.display()
        );
    }
    if canonical.is_dir() {
        anyhow::bail!(
            "path {} is a directory, not a neuron file",
            canonical.display()
        );
    }
    Ok(canonical)
}

fn build_augmented_task(index: &NeuronIndex, input: &GetContextsInput) -> String {
    let mut extra = String::new();

    if let Some(ref open_files) = input.open_files {
        if !open_files.is_empty() {
            let soft = index.soft_terms_for_editor_context(open_files, 8);
            if !soft.is_empty() {
                extra.push(' ');
                extra.push_str(&soft.join(" "));
                tracing::debug!(
                    files = open_files.len(),
                    soft_terms = soft.len(),
                    "S-V: editor context injected"
                );
            }
        }
    }

    if let Some(ref err_ctx) = input.error_context {
        if !err_ctx.is_empty() {
            let err_terms = tokenize(err_ctx);
            if !err_terms.is_empty() {
                extra.push(' ');
                extra.push_str(&err_terms.join(" "));
                tracing::debug!(err_terms = err_terms.len(), "S-V: error_context injected");
            }
        }
    }

    if extra.is_empty() {
        input.task.clone()
    } else {
        format!("{}{}", input.task, extra)
    }
}

fn index_kg_entity_path(index: &mut NeuronIndex, path: &Path) -> Result<()> {
    let content = std::fs::read_to_string(path)
        .map_err(|err| anyhow::anyhow!("reload KG entity {}: {err}", path.display()))?;
    let mut meta = NeuronMeta::new_stub(path, NeuronKind::Concept);
    meta.module = Some("@kg".to_string());
    meta.tokens = estimate_context_tokens(&content);
    index.index_neuron(path, &content, &meta);
    Ok(())
}

fn sync_structured_diary_to_kg(
    project_root: &Path,
    index: &mut NeuronIndex,
    agent: &str,
    entry: &StructuredDiaryEntry,
    effective_from: &str,
) -> Result<()> {
    let path = kg::kg_neuron_path(project_root, &agent_entity_name(agent));
    let mut entity = kg::KgEntity::load(&path)?;
    entity.replace_active_fact(
        AGENT_FOCUS_PREDICATE,
        entry.title.as_deref().unwrap_or(""),
        effective_from,
    );
    entity.replace_active_fact(
        AGENT_STATUS_PREDICATE,
        entry.status.as_deref().unwrap_or(""),
        effective_from,
    );
    entity.replace_active_fact(
        AGENT_GOAL_PREDICATE,
        entry.goal.as_deref().unwrap_or(""),
        effective_from,
    );
    entity.replace_active_fact(
        AGENT_NEXT_STEP_PREDICATE,
        entry.next_step.as_deref().unwrap_or(""),
        effective_from,
    );
    entity.replace_active_fact(
        AGENT_BLOCKER_PREDICATE,
        entry.blocker.as_deref().unwrap_or(""),
        effective_from,
    );
    entity.replace_active_fact(
        AGENT_OUTCOME_PREDICATE,
        entry.outcome.as_deref().unwrap_or(""),
        effective_from,
    );
    entity.replace_active_fact(
        AGENT_ACTION_PREDICATE,
        entry.action.as_deref().unwrap_or(""),
        effective_from,
    );
    entity.sync_active_values(
        AGENT_RELATED_ENTITY_PREDICATE,
        &entry.entities,
        effective_from,
    );
    entity.sync_active_values(
        AGENT_DEPENDS_ON_PREDICATE,
        &entry.depends_on,
        effective_from,
    );
    entity.save()?;
    index_kg_entity_path(index, &path)
}

/// Strip HTML comment delimiters and control characters from user-supplied strings
/// before embedding them in comment blocks, preventing comment breakout and prompt injection.
fn sanitize_comment(s: &str) -> String {
    let clean: String = s
        .chars()
        .map(|c| {
            if c.is_ascii_control() && c != '\t' {
                ' '
            } else {
                c
            }
        })
        .collect();
    let clean = clean.replace("-->", "—>").replace("<!--", "<—");
    // Truncate to 500 chars to prevent unbounded comment sections
    clean.chars().take(500).collect()
}

fn render_recent_agent_memory_block(
    index: &NeuronIndex,
    agent: &str,
    limit: usize,
) -> Option<String> {
    let module = format!("@agent/{}", agent.trim());
    let paths = recent_module_paths(index, &module, limit, Some(NeuronKind::Verbatim));
    if paths.is_empty() {
        return None;
    }

    let mut out = format!(
        "<!-- CORTYX WAKE-UP: @agent/{} memories -->\n",
        agent.trim()
    );
    for path in paths {
        let timestamp_secs = index
            .context_metadata_for(&path)
            .and_then(|metadata| metadata.timestamp_secs);
        match std::fs::read_to_string(&path) {
            Ok(content) => {
                out.push_str(&render_agent_memory_summary(&content, timestamp_secs));
                out.push('\n');
            },
            Err(err) => {
                out.push_str(&format!(
                    "- {} — read error: {}\n",
                    path.display(),
                    sanitize_comment(&err.to_string())
                ));
            },
        }
    }
    Some(out)
}

fn render_agent_memory_summary(content: &str, timestamp_secs: Option<i64>) -> String {
    let timestamp = timestamp_secs
        .map(format_timestamp_secs)
        .unwrap_or_else(|| "unknown-time".to_string());
    if let Some(entry) = parse_structured_diary_entry(content) {
        format!(
            "- {timestamp} — {}",
            summarize_structured_diary_entry(&entry)
        )
    } else {
        format!(
            "- {timestamp} — {}",
            truncate_str(&summarize_plain_diary_content(content), 180)
        )
    }
}

fn render_structured_diary_history_entry(
    entry: &crate::agent_memory::StructuredDiaryEntry,
    timestamp_secs: Option<i64>,
) -> String {
    let timestamp = timestamp_secs
        .map(format_timestamp_secs)
        .unwrap_or_else(|| "unknown-time".to_string());
    let mut out = format!(
        "- {timestamp} — {}",
        summarize_structured_diary_entry(entry)
    );
    if let Some(action) = &entry.action {
        out.push_str(&format!(
            "\n  action: {}",
            truncate_str(&summarize_plain_diary_content(action), 200)
        ));
    }
    if let Some(goal) = &entry.goal {
        out.push_str(&format!("\n  goal: {}", truncate_str(goal, 200)));
    }
    if let Some(next_step) = &entry.next_step {
        out.push_str(&format!("\n  next step: {}", truncate_str(next_step, 200)));
    }
    if let Some(blocker) = &entry.blocker {
        out.push_str(&format!("\n  blocker: {}", truncate_str(blocker, 200)));
    }
    if let Some(outcome) = &entry.outcome {
        out.push_str(&format!(
            "\n  outcome: {}",
            truncate_str(&summarize_plain_diary_content(outcome), 200)
        ));
    }
    if !entry.depends_on.is_empty() {
        out.push_str(&format!("\n  depends on: {}", entry.depends_on.join(", ")));
    }
    out.push('\n');
    out
}

fn render_agent_status_report(
    index: &NeuronIndex,
    project_root: &Path,
    agent: &str,
    include_timeline: bool,
) -> Option<String> {
    let projection = build_collaboration_projection(index, project_root);
    let summary = projection
        .collaborators
        .iter()
        .find(|summary| matches_collaboration_filter(&summary.collaborator, agent))?;
    Some(render_collaborator_status_report(
        summary,
        &projection,
        include_timeline,
    ))
}

fn render_collaboration_status_report(
    projection: &CollaborationStateProjection,
    agent: Option<&str>,
    module: Option<&str>,
    include_timeline: bool,
) -> Option<String> {
    if let Some(agent) = agent.filter(|agent| !agent.trim().is_empty()) {
        let summary = projection
            .collaborators
            .iter()
            .find(|summary| matches_collaboration_filter(&summary.collaborator, agent))?;
        return Some(render_collaborator_status_report(
            summary,
            projection,
            include_timeline,
        ));
    }
    if let Some(module) = module.filter(|module| !module.trim().is_empty()) {
        let state = projection
            .modules
            .iter()
            .find(|state| matches_collaboration_filter(&state.module, module))?;
        return Some(render_module_status_report(
            state,
            projection,
            include_timeline,
        ));
    }
    if projection.collaborators.is_empty()
        && projection.modules.is_empty()
        && projection.timeline.is_empty()
    {
        return None;
    }
    Some(render_project_collaboration_status(
        projection,
        include_timeline,
    ))
}

fn render_project_collaboration_status(
    projection: &CollaborationStateProjection,
    include_timeline: bool,
) -> String {
    let latest_activity = projection
        .timeline
        .iter()
        .find_map(|event| event.at.clone())
        .or_else(|| {
            projection
                .collaborators
                .iter()
                .filter_map(|summary| summary.last_updated.clone())
                .max()
        })
        .or_else(|| {
            projection
                .modules
                .iter()
                .filter_map(|state| state.last_updated.clone())
                .max()
        });
    let pending_sync_count = projection
        .timeline
        .iter()
        .filter(|event| matches!(event.kind, CollaborationTimelineKind::Sync))
        .count();
    let conflict_count = projection
        .timeline
        .iter()
        .filter(|event| matches!(event.kind, CollaborationTimelineKind::Conflict))
        .count();
    let trust_scores = projection
        .collaborators
        .iter()
        .filter_map(|summary| summary.trust_score)
        .chain(
            projection
                .modules
                .iter()
                .filter_map(|state| state.trust_score),
        )
        .collect::<Vec<_>>();
    let average_trust_score = (!trust_scores.is_empty())
        .then_some(trust_scores.iter().sum::<f32>() / trust_scores.len() as f32);
    let handoff_risk_count = projection
        .collaborators
        .iter()
        .map(|summary| summary.evidence.untrusted_handoff_count)
        .sum::<usize>();
    let integrity_issue_count = projection
        .collaborators
        .iter()
        .map(|summary| summary.evidence.integrity_issue_count)
        .sum::<usize>();
    let top_attention = match (projection.collaborators.first(), projection.modules.first()) {
        (Some(collaborator), Some(module))
            if module.attention_score > collaborator.attention_score =>
        {
            module.attention
        },
        (Some(collaborator), _) => collaborator.attention,
        (None, Some(module)) => module.attention,
        (None, None) => CollaborationAttention::Nominal,
    };

    let mut out = "## Collaboration Status\n\n".to_string();
    out.push_str(&format!(
        "- collaborators: {}\n",
        projection.collaborators.len()
    ));
    out.push_str(&format!("- modules: {}\n", projection.modules.len()));
    if let Some(latest_activity) = latest_activity {
        out.push_str(&format!("- latest activity: {latest_activity}\n"));
    }
    if top_attention != CollaborationAttention::Nominal {
        out.push_str(&format!(
            "- attention: {}\n",
            format_collaboration_attention(top_attention)
        ));
    }
    out.push_str(&format!("- pending sync items: {pending_sync_count}\n"));
    out.push_str(&format!("- sync conflicts: {conflict_count}\n"));
    if let Some(trust_score) = average_trust_score {
        out.push_str(&format!(
            "- average trust score: {}\n",
            format_trust_score(trust_score)
        ));
    }
    if handoff_risk_count > 0 {
        out.push_str(&format!("- handoff risks: {handoff_risk_count}\n"));
    }
    if integrity_issue_count > 0 {
        out.push_str(&format!("- integrity issues: {integrity_issue_count}\n"));
    }

    if !projection.collaborators.is_empty() {
        out.push_str("\n## Top collaborators\n");
        for summary in projection
            .collaborators
            .iter()
            .take(DEFAULT_COLLAB_SUMMARY_LIMIT)
        {
            out.push_str(&format!("- {}\n", render_collaborator_brief(summary)));
        }
    }
    if !projection.modules.is_empty() {
        out.push_str("\n## Shared modules\n");
        for state in projection.modules.iter().take(DEFAULT_COLLAB_SUMMARY_LIMIT) {
            out.push_str(&format!("- {}\n", render_module_brief(state)));
        }
    }
    if include_timeline {
        append_collaboration_timeline(&mut out, &projection.timeline, None, None);
    }
    out
}

fn render_collaborator_status_report(
    summary: &CollaboratorSummary,
    projection: &CollaborationStateProjection,
    include_timeline: bool,
) -> String {
    let mut out = format!("## Agent Status: {}\n\n", summary.collaborator);
    if let Some(updated) = &summary.last_updated {
        out.push_str(&format!("- updated: {updated}\n"));
    }
    out.push_str(&format!(
        "- attention: {}\n",
        format_collaboration_attention(summary.attention)
    ));
    if let Some(focus) = &summary.focus {
        out.push_str(&format!("- focus: {focus}\n"));
    }
    if let Some(status) = &summary.status {
        out.push_str(&format!("- status: {status}\n"));
    }
    if let Some(goal) = &summary.goal {
        out.push_str(&format!("- goal: {goal}\n"));
    }
    if let Some(next_step) = &summary.next_step {
        out.push_str(&format!("- next step: {next_step}\n"));
    }
    if let Some(blocker) = &summary.blocker {
        out.push_str(&format!("- blocker: {blocker}\n"));
    }
    if let Some(action) = &summary.action {
        out.push_str(&format!(
            "- action: {}\n",
            truncate_str(&summarize_plain_diary_content(action), 220)
        ));
    }
    if let Some(outcome) = &summary.outcome {
        out.push_str(&format!(
            "- outcome: {}\n",
            truncate_str(&summarize_plain_diary_content(outcome), 220)
        ));
    }
    if !summary.related_entities.is_empty() {
        out.push_str(&format!(
            "- related entities: {}\n",
            summary.related_entities.join(", ")
        ));
    }
    if !summary.depends_on.is_empty() {
        out.push_str(&format!(
            "- depends on: {}\n",
            summary.depends_on.join(", ")
        ));
    }
    if !summary.touched_modules.is_empty() {
        out.push_str(&format!(
            "- touched modules: {}\n",
            summary.touched_modules.join(", ")
        ));
    }
    out.push_str(&format!(
        "- pending sync: {}\n",
        if summary.pending_sync { "yes" } else { "no" }
    ));
    if let Some(trust_score) = summary.trust_score {
        out.push_str(&format!(
            "- trust score: {}\n",
            format_trust_score(trust_score)
        ));
    }
    let evidence = format_collaboration_evidence(&summary.evidence);
    if !evidence.is_empty() {
        out.push_str(&format!("- evidence: {evidence}\n"));
    }
    out.push_str(&format!(
        "- sources: @agent/{} diary, @kg/{}\n",
        summary.collaborator, summary.entity
    ));

    append_supporting_facts(&mut out, &summary.supporting_facts);

    let related_modules: Vec<&ModuleCollaborationState> = projection
        .modules
        .iter()
        .filter(|state| {
            summary
                .touched_modules
                .iter()
                .any(|module| matches_collaboration_filter(&state.module, module))
        })
        .take(3)
        .collect();
    if !related_modules.is_empty() {
        out.push_str("\n## Shared modules\n");
        for state in related_modules {
            out.push_str(&format!("- {}\n", render_module_brief(state)));
        }
    }

    if include_timeline {
        append_collaboration_timeline(
            &mut out,
            &projection.timeline,
            Some(&summary.collaborator),
            None,
        );
    }
    out
}

fn render_module_status_report(
    state: &ModuleCollaborationState,
    projection: &CollaborationStateProjection,
    include_timeline: bool,
) -> String {
    let mut out = format!("## Collaboration Module: {}\n\n", state.module);
    if let Some(updated) = &state.last_updated {
        out.push_str(&format!("- updated: {updated}\n"));
    }
    out.push_str(&format!(
        "- attention: {}\n",
        format_collaboration_attention(state.attention)
    ));
    if !state.collaborators.is_empty() {
        out.push_str(&format!(
            "- collaborators: {}\n",
            state.collaborators.join(", ")
        ));
    }
    if !state.focuses.is_empty() {
        out.push_str(&format!("- focuses: {}\n", state.focuses.join(", ")));
    }
    if !state.blockers.is_empty() {
        out.push_str(&format!("- blockers: {}\n", state.blockers.join(", ")));
    }
    if !state.related_entities.is_empty() {
        out.push_str(&format!(
            "- related entities: {}\n",
            state.related_entities.join(", ")
        ));
    }
    out.push_str(&format!(
        "- pending sync: {}\n",
        if state.pending_sync { "yes" } else { "no" }
    ));
    if let Some(trust_score) = state.trust_score {
        out.push_str(&format!(
            "- trust score: {}\n",
            format_trust_score(trust_score)
        ));
    }
    if let Some(edit_id) = &state.latest_sync_edit_id {
        out.push_str(&format!("- latest sync edit: {edit_id}\n"));
    }
    let evidence = format_collaboration_evidence(&state.evidence);
    if !evidence.is_empty() {
        out.push_str(&format!("- evidence: {evidence}\n"));
    }

    append_supporting_facts(&mut out, &state.supporting_facts);

    let matching_collaborators: Vec<&CollaboratorSummary> = projection
        .collaborators
        .iter()
        .filter(|summary| {
            summary
                .touched_modules
                .iter()
                .any(|module| matches_collaboration_filter(module, &state.module))
        })
        .take(3)
        .collect();
    if !matching_collaborators.is_empty() {
        out.push_str("\n## Collaborators\n");
        for summary in matching_collaborators {
            out.push_str(&format!("- {}\n", render_collaborator_brief(summary)));
        }
    }

    if include_timeline {
        append_collaboration_timeline(&mut out, &projection.timeline, None, Some(&state.module));
    }
    out
}

fn summarize_plain_diary_content(content: &str) -> String {
    content
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.starts_with("<!--") && !line.starts_with('#'))
        .unwrap_or("(empty diary entry)")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn format_timestamp_secs(timestamp_secs: i64) -> String {
    if timestamp_secs < 0 {
        return timestamp_secs.to_string();
    }
    let (y, mo, d, h, mi, s) = unix_secs_to_datetime(timestamp_secs as u64);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

#[allow(dead_code)]
fn latest_active_kg_value(entity: &kg::KgEntity, predicate: &str) -> Option<String> {
    entity
        .active_values_for_predicate(predicate, None)
        .last()
        .map(|fact| fact.value.clone())
}

#[allow(dead_code)]
fn active_kg_values(entity: &kg::KgEntity, predicate: &str) -> Vec<String> {
    entity
        .active_values_for_predicate(predicate, None)
        .into_iter()
        .map(|fact| fact.value.clone())
        .collect()
}

fn render_collaborator_brief(summary: &CollaboratorSummary) -> String {
    let mut details = vec![format_collaboration_attention(summary.attention).to_string()];
    if let Some(focus) = &summary.focus {
        details.push(format!("focus: {}", truncate_str(focus, 80)));
    }
    if !summary.touched_modules.is_empty() {
        details.push(format!("modules: {}", summary.touched_modules.join(", ")));
    }
    if summary.pending_sync {
        details.push("pending sync".to_string());
    }
    if summary.evidence.conflict_count > 0 {
        details.push(format!("conflicts: {}", summary.evidence.conflict_count));
    }
    if summary.evidence.integrity_issue_count > 0 {
        details.push(format!(
            "integrity issues: {}",
            summary.evidence.integrity_issue_count
        ));
    }
    if let Some(trust_score) = summary.trust_score {
        details.push(format!("trust: {}", format_trust_score(trust_score)));
    }
    format!("{} — {}", summary.collaborator, details.join("; "))
}

fn render_module_brief(state: &ModuleCollaborationState) -> String {
    let mut details = vec![format_collaboration_attention(state.attention).to_string()];
    if !state.collaborators.is_empty() {
        details.push(format!("collaborators: {}", state.collaborators.join(", ")));
    }
    if state.pending_sync {
        details.push("pending sync".to_string());
    }
    if state.evidence.conflict_count > 0 {
        details.push(format!("conflicts: {}", state.evidence.conflict_count));
    }
    if state.evidence.integrity_issue_count > 0 {
        details.push(format!(
            "integrity issues: {}",
            state.evidence.integrity_issue_count
        ));
    }
    if let Some(trust_score) = state.trust_score {
        details.push(format!("trust: {}", format_trust_score(trust_score)));
    }
    format!("{} — {}", state.module, details.join("; "))
}

fn append_supporting_facts(out: &mut String, facts: &[String]) {
    if facts.is_empty() {
        return;
    }
    out.push_str("\n## Supporting facts\n");
    for fact in facts.iter().take(6) {
        out.push_str(&format!("- {fact}\n"));
    }
}

fn append_collaboration_timeline(
    out: &mut String,
    timeline: &[CollaborationTimelineEvent],
    collaborator: Option<&str>,
    module: Option<&str>,
) {
    let mut filtered = timeline
        .iter()
        .filter(|event| {
            collaborator
                .map(|target| {
                    event
                        .collaborator
                        .as_deref()
                        .map(|value| matches_collaboration_filter(value, target))
                        .unwrap_or(false)
                })
                .unwrap_or(true)
                && module
                    .map(|target| {
                        event
                            .module
                            .as_deref()
                            .map(|value| matches_collaboration_filter(value, target))
                            .unwrap_or(false)
                    })
                    .unwrap_or(true)
        })
        .take(DEFAULT_COLLAB_TIMELINE_LIMIT)
        .peekable();
    if filtered.peek().is_none() {
        return;
    }
    out.push_str("\n## Collaboration timeline\n");
    for event in filtered {
        out.push_str(&format!(
            "- {}\n",
            render_collaboration_timeline_event(event)
        ));
    }
}

fn render_collaboration_timeline_event(event: &CollaborationTimelineEvent) -> String {
    let at = event.at.as_deref().unwrap_or("unknown-time");
    let mut scope = Vec::new();
    if let Some(collaborator) = &event.collaborator {
        scope.push(collaborator.as_str());
    }
    if let Some(module) = &event.module {
        scope.push(module.as_str());
    }
    let prefix = if scope.is_empty() {
        String::new()
    } else {
        format!("{} — ", scope.join(" / "))
    };
    let mut rendered = format!("{at} — {prefix}{}", event.summary);
    if event.attention != CollaborationAttention::Nominal {
        rendered.push_str(&format!(
            " ({})",
            format_collaboration_attention(event.attention)
        ));
    }
    rendered
}

fn format_collaboration_attention(attention: CollaborationAttention) -> &'static str {
    match attention {
        CollaborationAttention::Nominal => "nominal",
        CollaborationAttention::NeedsFollowUp => "needs_follow_up",
        CollaborationAttention::Blocked => "blocked",
        CollaborationAttention::SyncConflict => "sync_conflict",
    }
}

fn format_collaboration_evidence(evidence: &CollaborationEvidenceSummary) -> String {
    let mut parts = Vec::new();
    if evidence.diary_entries > 0 {
        parts.push(format!("diary={}", evidence.diary_entries));
    }
    if evidence.kg_fact_count > 0 {
        parts.push(format!("kg={}", evidence.kg_fact_count));
    }
    if evidence.sync_neuron_count > 0 {
        parts.push(format!("sync={}", evidence.sync_neuron_count));
    }
    if evidence.verified_sync_count > 0 {
        parts.push(format!("verified_sync={}", evidence.verified_sync_count));
    }
    if evidence.pending_sync_count > 0 {
        parts.push(format!("pending_sync={}", evidence.pending_sync_count));
    }
    if evidence.conflict_count > 0 {
        parts.push(format!("conflicts={}", evidence.conflict_count));
    }
    if evidence.integrity_issue_count > 0 {
        parts.push(format!(
            "integrity_issues={}",
            evidence.integrity_issue_count
        ));
    }
    if evidence.untrusted_handoff_count > 0 {
        parts.push(format!(
            "handoff_risks={}",
            evidence.untrusted_handoff_count
        ));
    }
    if evidence.reasoning_node_count > 0 {
        parts.push(format!("reasoning_nodes={}", evidence.reasoning_node_count));
    }
    if evidence.reasoning_fact_count > 0 {
        parts.push(format!("reasoning_facts={}", evidence.reasoning_fact_count));
    }
    parts.join(", ")
}

fn format_trust_score(score: f32) -> String {
    format!("{score:.1}")
}

fn matches_collaboration_filter(value: &str, filter: &str) -> bool {
    kg::slugify(value) == kg::slugify(filter)
}

fn fingerprint_rendered_context(rendered: &str) -> String {
    blake3::hash(rendered.as_bytes()).to_hex()[..16].to_string()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EmissionTier {
    Full,
    Focused,
    Summary,
}

fn render_context_item(
    path: &Path,
    score: f32,
    task_terms: &[String],
    index: &NeuronIndex,
) -> RenderedContextItem {
    let rendered = match std::fs::read_to_string(path) {
        Ok(content) => {
            let content = strip_render_only_sections(&content);
            match select_emission_tier(score, &content) {
                EmissionTier::Full => format!(
                    "<!-- === NEURON: {} === -->\n{}\n\n",
                    path.display(),
                    content
                ),
                EmissionTier::Focused => {
                    let focused = build_focused_context(&content, task_terms);
                    format!(
                        "<!-- === NEURON (focused, score={:.1}): {} === -->\n{}\n\n",
                        score,
                        path.file_name().unwrap_or_default().to_string_lossy(),
                        focused
                    )
                },
                EmissionTier::Summary => {
                    let summary = index
                        .summary_for(path)
                        .filter(|s| !s.is_empty())
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| {
                            fallback_excerpt(&content, 3)
                                .lines()
                                .take(3)
                                .collect::<Vec<_>>()
                                .join("\n")
                        });
                    format!(
                        "<!-- === NEURON (summary, score={:.1}): {} === -->\n{}\n\n",
                        score,
                        path.file_name().unwrap_or_default().to_string_lossy(),
                        sanitize_comment(&summary),
                    )
                },
            }
        },
        Err(err) => {
            if score >= 5.0 {
                format!("<!-- NEURON {} — read error: {err} -->\n\n", path.display())
            } else {
                tracing::warn!(
                    "Failed to read {} while building summary context: {}",
                    path.display(),
                    err
                );
                format!(
                    "<!-- === NEURON (summary, score={:.1}): {} === -->\n{}\n\n",
                    score,
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    sanitize_comment(&format!("(read error: {err})")),
                )
            }
        },
    };

    RenderedContextItem {
        path: path.to_path_buf(),
        fingerprint: fingerprint_rendered_context(&rendered),
        rendered,
    }
}

fn select_emission_tier(score: f32, content: &str) -> EmissionTier {
    let tokens = estimate_context_tokens(content);
    if score < 5.0 {
        EmissionTier::Summary
    } else if score >= 9.0 || tokens <= 160 {
        EmissionTier::Full
    } else {
        EmissionTier::Focused
    }
}

fn build_focused_context(content: &str, task_terms: &[String]) -> String {
    if let Some(sectioned) = render_focused_sections(content, task_terms) {
        return sectioned;
    }
    render_focused_excerpt(content, task_terms)
}

fn render_focused_sections(content: &str, task_terms: &[String]) -> Option<String> {
    let sections = parse_markdown_sections(content);
    if sections.len() < 2 {
        return None;
    }

    let focus_terms = significant_task_terms(task_terms);
    let debug_task = focus_terms.iter().any(|term| {
        matches!(
            term.as_str(),
            "fix" | "bug" | "error" | "errors" | "failing" | "failure" | "debug" | "issue"
        )
    });
    let guidance_task = focus_terms.iter().any(|term| {
        matches!(
            term.as_str(),
            "implement" | "implementation" | "how" | "why" | "use" | "usage" | "example"
        )
    });

    let mut selected = std::collections::BTreeSet::new();
    if let Some(idx) = sections
        .iter()
        .position(|(name, _)| name.eq_ignore_ascii_case("purpose"))
    {
        selected.insert(idx);
    }

    let mut scored: Vec<(i32, usize)> = sections
        .iter()
        .enumerate()
        .map(|(idx, (name, body))| {
            let lower_name = name.to_ascii_lowercase();
            let section_terms: std::collections::HashSet<String> =
                tokenize(body).into_iter().collect();
            let overlap = section_terms
                .iter()
                .filter(|term| focus_terms.contains(*term))
                .count() as i32;
            let mut score = overlap * 10;
            match lower_name.as_str() {
                "purpose" => score += 15,
                "api" => score += 12,
                "pitfalls" if debug_task => score += 14,
                "patterns" | "examples" if guidance_task => score += 12,
                "auto_evolved" => score += 6,
                "notes" => score += 2,
                _ => {},
            }
            (score, idx)
        })
        .collect();
    scored.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    for (_, idx) in scored {
        selected.insert(idx);
        if selected.len() >= 3 {
            break;
        }
    }

    if selected.is_empty() {
        return None;
    }

    let title = content
        .lines()
        .find(|line| line.trim_start().starts_with("# "))
        .map(str::trim)
        .filter(|line| !line.is_empty());
    let mut out = String::new();
    if let Some(title) = title {
        out.push_str(title);
        out.push_str("\n\n");
    }
    for idx in selected {
        let (name, body) = &sections[idx];
        let body = trim_body_lines(body, 8);
        if body.is_empty() {
            continue;
        }
        out.push_str("## ");
        out.push_str(name);
        out.push('\n');
        out.push_str(&body);
        out.push_str("\n\n");
    }

    let trimmed = out.trim().to_string();
    (!trimmed.is_empty()).then_some(trimmed)
}

fn render_focused_excerpt(content: &str, task_terms: &[String]) -> String {
    let focus_terms = significant_task_terms(task_terms);
    let lines: Vec<&str> = content.lines().collect();
    let mut scored: Vec<(usize, usize)> = lines
        .iter()
        .enumerate()
        .filter_map(|(idx, line)| {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                return None;
            }
            let overlap = tokenize(trimmed)
                .into_iter()
                .filter(|term| focus_terms.contains(term))
                .count();
            if overlap == 0 {
                return None;
            }
            let speaker_bonus =
                usize::from(trimmed.starts_with("User:") || trimmed.starts_with("Assistant:"));
            Some((overlap + speaker_bonus, idx))
        })
        .collect();

    if scored.is_empty() {
        return fallback_excerpt(content, 6);
    }

    scored.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    let mut chosen = std::collections::BTreeSet::new();
    for (_, idx) in scored.into_iter().take(3) {
        let start = idx.saturating_sub(1);
        let end = (idx + 1).min(lines.len().saturating_sub(1));
        for line_idx in start..=end {
            if !lines[line_idx].trim().is_empty() {
                chosen.insert(line_idx);
            }
        }
    }

    let excerpt_lines: Vec<&str> = chosen
        .into_iter()
        .filter_map(|idx| lines.get(idx).copied())
        .filter(|line| !line.trim().is_empty())
        .take(10)
        .collect();
    if excerpt_lines.is_empty() {
        fallback_excerpt(content, 6)
    } else {
        excerpt_lines.join("\n")
    }
}

fn parse_markdown_sections(content: &str) -> Vec<(String, String)> {
    let mut sections = Vec::new();
    let mut current_name: Option<String> = None;
    let mut body_lines: Vec<&str> = Vec::new();

    for line in content.lines() {
        if let Some(name) = line.trim_start().strip_prefix("## ") {
            if let Some(prev) = current_name.take() {
                sections.push((prev, body_lines.join("\n").trim().to_string()));
                body_lines.clear();
            }
            current_name = Some(name.trim().to_string());
        } else if current_name.is_some() {
            body_lines.push(line);
        }
    }

    if let Some(name) = current_name {
        sections.push((name, body_lines.join("\n").trim().to_string()));
    }

    sections.retain(|(_, body)| !body.is_empty());
    sections
}

fn trim_body_lines(body: &str, max_nonempty_lines: usize) -> String {
    body.lines()
        .filter(|line| !line.trim().is_empty())
        .take(max_nonempty_lines)
        .collect::<Vec<_>>()
        .join("\n")
}

fn fallback_excerpt(content: &str, max_nonempty_lines: usize) -> String {
    content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .take(max_nonempty_lines)
        .collect::<Vec<_>>()
        .join("\n")
}

fn significant_task_terms(task_terms: &[String]) -> std::collections::HashSet<String> {
    const STOPWORDS: &[&str] = &[
        "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "for", "from", "have", "how",
        "i", "in", "into", "is", "it", "me", "my", "of", "on", "or", "our", "that", "the", "their",
        "them", "they", "this", "to", "was", "what", "when", "where", "which", "who", "why",
        "with", "you", "your",
    ];
    task_terms
        .iter()
        .map(|term| term.to_ascii_lowercase())
        .filter(|term| term.len() >= 3 && !STOPWORDS.contains(&term.as_str()))
        .collect()
}

fn strip_render_only_sections(content: &str) -> String {
    let without_query = strip_named_render_section(content, "query_surface");
    strip_named_render_section(&without_query, "answer_surface")
}

fn strip_named_render_section(content: &str, section_name: &str) -> String {
    let header = format!("## {section_name}");
    let marker = format!("<!-- SECTION: {section_name} -->");
    let end_marker = "<!-- /SECTION -->";
    let Some(header_start) = content.find(&header) else {
        return content.to_string();
    };
    let Some(section_start_rel) = content[header_start..].find(&marker) else {
        return content.to_string();
    };
    let section_start = header_start + section_start_rel;
    let Some(section_end_rel) = content[section_start..].find(end_marker) else {
        return content.to_string();
    };
    let section_end = section_start + section_end_rel + end_marker.len();

    let mut stripped = String::with_capacity(content.len());
    stripped.push_str(content[..header_start].trim_end());
    if !stripped.ends_with('\n') {
        stripped.push('\n');
    }
    let tail = content[section_end..].trim_start_matches('\n');
    if !tail.is_empty() {
        stripped.push('\n');
        stripped.push_str(tail);
    }
    stripped
}

fn render_overflow_item(path: &Path, headline: &str) -> RenderedContextItem {
    let rendered = format!(
        "<!-- NEURON (compressed): {} — {} -->\n",
        path.file_name().unwrap_or_default().to_string_lossy(),
        sanitize_comment(headline),
    );
    RenderedContextItem {
        path: path.to_path_buf(),
        fingerprint: fingerprint_rendered_context(&rendered),
        rendered,
    }
}

fn render_module_capsule(project_root: &Path, module: &str) -> Option<RenderedContextItem> {
    let path = module_capsule_path(project_root, module);
    let content = match std::fs::read_to_string(&path) {
        Ok(content) => content,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return None,
        Err(err) => {
            tracing::warn!(
                "Failed to read module capsule {} for {}: {}",
                path.display(),
                module,
                err
            );
            return Some(RenderedContextItem {
                path,
                fingerprint: fingerprint_rendered_context(&format!(
                    "<!-- MODULE CAPSULE {} — read error: {} -->\n\n",
                    sanitize_comment(module),
                    sanitize_comment(&err.to_string())
                )),
                rendered: format!(
                    "<!-- MODULE CAPSULE {} — read error: {} -->\n\n",
                    sanitize_comment(module),
                    sanitize_comment(&err.to_string())
                ),
            });
        },
    };
    let rendered = format!(
        "<!-- === MODULE CAPSULE: {} === -->\n{}\n\n",
        sanitize_comment(module),
        content
    );
    Some(RenderedContextItem {
        path,
        fingerprint: fingerprint_rendered_context(&rendered),
        rendered,
    })
}

fn build_path_module_map(
    paths_with_scores: &[(PathBuf, f32)],
    overflow: &[(PathBuf, String)],
    index: &NeuronIndex,
) -> HashMap<PathBuf, String> {
    let mut path_modules = HashMap::new();
    for (path, _) in paths_with_scores {
        if let Some(module) = index.module_for(path) {
            path_modules.insert(path.clone(), module.to_string());
        }
    }
    for (path, _) in overflow {
        if let Some(module) = index.module_for(path) {
            path_modules.insert(path.clone(), module.to_string());
        }
    }
    path_modules
}

fn select_capsule_modules(
    paths_with_scores: &[(PathBuf, f32)],
    explicit_module: Option<&str>,
    path_modules: &HashMap<PathBuf, String>,
) -> Vec<String> {
    if let Some(module) = explicit_module {
        return if is_capsule_module(module) {
            vec![module.to_string()]
        } else {
            Vec::new()
        };
    }

    let module_tagged_total = paths_with_scores
        .iter()
        .filter(|(path, _)| {
            path_modules
                .get(path)
                .is_some_and(|module| is_capsule_module(module))
        })
        .count();
    if module_tagged_total < 2 {
        return Vec::new();
    }

    let mut counts: HashMap<&str, usize> = HashMap::new();
    for (path, _) in paths_with_scores {
        let Some(module) = path_modules.get(path) else {
            continue;
        };
        if !is_capsule_module(module) {
            continue;
        }
        *counts.entry(module.as_str()).or_insert(0) += 1;
    }

    let mut ranked: Vec<(&str, usize)> = counts.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    ranked
        .into_iter()
        .find(|(_, count)| *count >= 2 && *count * 2 >= module_tagged_total)
        .map(|(module, _)| vec![module.to_string()])
        .unwrap_or_default()
}

fn select_capsule_anchor_paths(
    paths_with_scores: &[(PathBuf, f32)],
    capsule_modules: &HashSet<String>,
    path_modules: &HashMap<PathBuf, String>,
) -> HashSet<PathBuf> {
    const CAPSULE_DYNAMIC_NEURONS_PER_MODULE: usize = 2;
    const CAPSULE_FULL_BODY_SCORE_THRESHOLD: f32 = 5.0;

    let mut grouped: HashMap<&str, Vec<(&PathBuf, f32)>> = HashMap::new();
    for (path, score) in paths_with_scores {
        let Some(module) = path_modules.get(path) else {
            continue;
        };
        if capsule_modules.contains(module) {
            grouped
                .entry(module.as_str())
                .or_default()
                .push((path, *score));
        }
    }

    let mut keep = HashSet::new();
    for items in grouped.values_mut() {
        items.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        let mut kept = 0usize;
        for (path, score) in items.iter() {
            if *score >= CAPSULE_FULL_BODY_SCORE_THRESHOLD
                && kept < CAPSULE_DYNAMIC_NEURONS_PER_MODULE
            {
                keep.insert((*path).clone());
                kept += 1;
            }
        }
        if kept == 0 {
            if let Some((path, _)) = items.first() {
                keep.insert((*path).clone());
            }
        }
    }

    keep
}

fn select_delta_items(
    items: &[RenderedContextItem],
    previous: Option<&HashMap<PathBuf, String>>,
) -> DeltaSelection {
    let current_paths: std::collections::HashSet<&PathBuf> =
        items.iter().map(|item| &item.path).collect();

    let mut emitted = Vec::new();
    let mut unchanged = 0usize;
    for item in items {
        if previous
            .and_then(|snapshot| snapshot.get(&item.path))
            .is_some_and(|fingerprint| fingerprint == &item.fingerprint)
        {
            unchanged += 1;
        } else {
            emitted.push(item.clone());
        }
    }

    let removed = previous
        .map(|snapshot| {
            snapshot
                .keys()
                .filter(|path| !current_paths.contains(*path))
                .count()
        })
        .unwrap_or(0);

    DeltaSelection {
        emitted,
        unchanged,
        removed,
    }
}

/// S-VIII (R16): Auto-mine UseCase stubs from code blocks in an LLM response.
///
/// Scans `response_text` for fenced code blocks (``` ... ```) with ≥5 lines.
/// For each block, finds the cited neuron with the highest term overlap.
/// If overlap ≥ 60% of the neuron's own terms, writes a UseCase stub to
/// `.cortyx/neurons/{neuron}.usecase.auto-{hash}.md` with `status: Stub`.
///
/// Returns the count of stubs written.
fn auto_mine_code_blocks(
    response_text: &str,
    cited_paths: &[PathBuf],
    project_root: &Path,
    index: &NeuronIndex,
) -> usize {
    if cited_paths.is_empty() {
        return 0;
    }

    // Extract fenced code blocks: ```[lang]\n<body>\n```
    let mut blocks: Vec<String> = Vec::new();
    let mut in_block = false;
    let mut current_block = Vec::new();
    for line in response_text.lines() {
        let trimmed = line.trim();
        if !in_block && trimmed.starts_with("```") {
            in_block = true;
            current_block.clear();
        } else if in_block && trimmed.starts_with("```") {
            if current_block.len() >= 5 {
                blocks.push(current_block.join("\n"));
            }
            in_block = false;
            current_block.clear();
        } else if in_block {
            current_block.push(line.to_string());
        }
    }

    if blocks.is_empty() {
        return 0;
    }

    let ndir = neuron_dir(project_root);
    let mut written = 0usize;

    for block in &blocks {
        let block_terms: std::collections::HashSet<String> = tokenize(block).into_iter().collect();
        if block_terms.is_empty() {
            continue;
        }

        // Find the cited neuron with highest term overlap
        let best = cited_paths
            .iter()
            .filter_map(|path| {
                let overlap = index.term_freq_overlap(path, &block_terms);
                let total_neuron_terms = index.term_count_for(path);
                if total_neuron_terms == 0 {
                    return None;
                }
                let ratio = overlap as f32 / total_neuron_terms as f32;
                Some((ratio, path))
            })
            .max_by(|a, b| a.0.total_cmp(&b.0));

        let Some((ratio, best_path)) = best else {
            continue;
        };
        if ratio < 0.6 {
            continue;
        }

        // Derive the output filename from the parent neuron stem + a short hash
        let stem = best_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .trim_end_matches(".context")
            .to_string();

        let hash_bytes = blake3::hash(block.as_bytes());
        let short_hash = &hash_bytes.to_hex()[..8];
        let usecase_filename = format!("{stem}.usecase.auto-{short_hash}.md");
        let usecase_path = ndir.join(&usecase_filename);

        if usecase_path.exists() {
            continue; // already mined
        }

        let content = format!(
            "# {stem} — auto-mined UseCase\n\
             status: Stub\n\
             source: auto-mined from close_task\n\n\
             ## task\n\
             (edit: describe the task pattern this code solves)\n\n\
             ## example\n\
             ```\n{block}\n```\n"
        );
        if let Err(e) = std::fs::write(&usecase_path, &content) {
            tracing::warn!(
                "S-VIII: failed to write UseCase stub {:?}: {e}",
                usecase_path
            );
        } else {
            tracing::debug!(
                "S-VIII: wrote UseCase stub {:?} (ratio={ratio:.2})",
                usecase_path
            );
            written += 1;
        }
    }

    written
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::NeuronIndex;
    use crate::kg::KgFact;
    use crate::neuron::provenance::{
        load_provenance, provenance_content_hash, ProvenanceOperation, ProvenanceSource,
    };
    use crate::reasoner::{ReasonedFact, ReasonedNode, ReasoningReport};
    use crate::sync_transport::{
        SyncHandoffIssue, SyncHandoffState, SyncHandoffSummary, SyncRevisionState,
    };
    use std::fs;

    fn test_item(path: &str, rendered: &str) -> RenderedContextItem {
        RenderedContextItem {
            path: PathBuf::from(path),
            rendered: rendered.to_string(),
            fingerprint: fingerprint_rendered_context(rendered),
        }
    }

    fn sample_collaboration_projection() -> CollaborationStateProjection {
        let mut diary = CollaborationDiaryRecord::new(
            "reviewer",
            StructuredDiaryEntry {
                agent: Some("reviewer".to_string()),
                title: Some("Audit auth middleware".to_string()),
                status: Some("blocked".to_string()),
                goal: Some("Close the auth bypass.".to_string()),
                next_step: Some("Wait for api-owner approval.".to_string()),
                blocker: Some("Waiting on api-owner.".to_string()),
                outcome: None,
                entities: vec!["auth".to_string(), "engine".to_string()],
                depends_on: vec!["api-owner".to_string()],
                action: None,
                refined_plan: None,
            },
        );
        diary.when = Some("2026-04-17T10:04:00Z".to_string());

        let trusted_integrity =
            |fingerprint: &str| crate::neuron::provenance::ProvenanceIntegritySummary {
                trusted: true,
                score: 100,
                fingerprint: Some(fingerprint.to_string()),
                revision_count: 1,
                author_count: 1,
                authorship_present: true,
                latest_author_present: true,
                identity_verified: true,
                content_verified: true,
                chain_verified: true,
                timestamps_monotonic: true,
                issues: Vec::new(),
            };

        let sync = SyncTransportStatus {
            neuron_uuid: "uuid-1234".to_string(),
            snapshot: Some(SyncRevisionState {
                source_path: PathBuf::from("src/engine.rs"),
                module: Some("engine".to_string()),
                content_hash: "hash-snapshot".to_string(),
                edit_id: Some("edit-2".to_string()),
                parent_edit_id: Some("edit-1".to_string()),
                edited_at: Some("2026-04-17T10:05:00Z".to_string()),
                author_id: Some("agent:reviewer".to_string()),
                author_display: Some("Reviewer".to_string()),
                created_by_id: Some("agent:reviewer".to_string()),
                created_by_display: Some("Reviewer".to_string()),
                provenance_fingerprint: Some("prov-snapshot".to_string()),
                revision_count: 1,
                author_count: 1,
                summary: Some("local auth hardening".to_string()),
                integrity: trusted_integrity("prov-snapshot"),
            }),
            outgoing: Some(SyncRevisionState {
                source_path: PathBuf::from("src/engine.rs"),
                module: Some("engine".to_string()),
                content_hash: "hash-outgoing".to_string(),
                edit_id: Some("edit-2".to_string()),
                parent_edit_id: Some("edit-1".to_string()),
                edited_at: Some("2026-04-17T10:05:00Z".to_string()),
                author_id: Some("agent:reviewer".to_string()),
                author_display: Some("Reviewer".to_string()),
                created_by_id: Some("agent:reviewer".to_string()),
                created_by_display: Some("Reviewer".to_string()),
                provenance_fingerprint: Some("prov-snapshot".to_string()),
                revision_count: 1,
                author_count: 1,
                summary: Some("local auth hardening".to_string()),
                integrity: trusted_integrity("prov-snapshot"),
            }),
            incoming: Some(SyncRevisionState {
                source_path: PathBuf::from("src/engine.rs"),
                module: Some("engine".to_string()),
                content_hash: "hash-incoming".to_string(),
                edit_id: Some("edit-3".to_string()),
                parent_edit_id: Some("edit-1".to_string()),
                edited_at: Some("2026-04-17T10:06:00Z".to_string()),
                author_id: Some("agent:reviewer".to_string()),
                author_display: Some("Reviewer".to_string()),
                created_by_id: Some("agent:reviewer".to_string()),
                created_by_display: Some("Reviewer".to_string()),
                provenance_fingerprint: Some("prov-incoming".to_string()),
                revision_count: 1,
                author_count: 1,
                summary: Some("remote auth edit".to_string()),
                integrity: trusted_integrity("prov-incoming"),
            }),
            conflict_paths: vec![PathBuf::from(
                ".cortyx/sync/conflicts/uu/uuid-1234--edit-2--edit-3.json",
            )],
            outgoing_pending: true,
            incoming_pending: true,
            handoff: SyncHandoffSummary {
                state: SyncHandoffState::Conflict,
                shared_edit_id: Some("edit-1".to_string()),
                local_edit_id: Some("edit-2".to_string()),
                remote_edit_id: Some("edit-3".to_string()),
                continuity_verified: false,
                integrity_verified: true,
                score: 60,
                issues: vec![SyncHandoffIssue::ConflictRecorded],
            },
        };

        let kg_entities = vec![
            kg::KgEntity {
                entity: agent_entity_name("reviewer"),
                facts: vec![
                    KgFact {
                        predicate: AGENT_ACTION_PREDICATE.to_string(),
                        value: "Investigating auth middleware.".to_string(),
                        valid_from: "2026-04-17T10:01:00Z".to_string(),
                        ended: String::new(),
                    },
                    KgFact {
                        predicate: AGENT_OUTCOME_PREDICATE.to_string(),
                        value: "Found a legacy bypass.".to_string(),
                        valid_from: "2026-04-17T10:02:00Z".to_string(),
                        ended: String::new(),
                    },
                ],
                path: PathBuf::from(".cortyx/neurons/_kg_agent_reviewer.context.md"),
            },
            kg::KgEntity {
                entity: "auth".to_string(),
                facts: vec![KgFact {
                    predicate: "owner".to_string(),
                    value: "platform-team".to_string(),
                    valid_from: "2026-04-17T10:03:00Z".to_string(),
                    ended: String::new(),
                }],
                path: PathBuf::from(".cortyx/neurons/_kg_auth.context.md"),
            },
        ];
        let reasoning = ReasoningReport {
            nodes: vec![ReasonedNode {
                path: PathBuf::from(".cortyx/neurons/src/engine.context.md"),
                score: 0.82,
                depth: 1,
                kind: Some(NeuronKind::Core),
                module: Some("engine".to_string()),
                summary: Some("Engine auth entrypoint".to_string()),
                supporting: vec![PathBuf::from(".cortyx/neurons/src/auth.context.md")],
                strongest_step: None,
                is_seed: false,
                is_kg_entity: false,
            }],
            facts: vec![ReasonedFact::new(
                PathBuf::from(".cortyx/neurons/_kg_auth.context.md"),
                "auth".to_string(),
                "owner".to_string(),
                "platform-team".to_string(),
                0.91,
                vec![PathBuf::from(".cortyx/neurons/src/engine.context.md")],
                true,
                "2026-04-17T10:03:00Z".to_string(),
                String::new(),
            )],
            conflicts: Vec::new(),
            ..Default::default()
        };

        project_collaboration_state(&[diary], &[sync], &kg_entities, Some(&reasoning))
    }

    #[test]
    fn select_delta_items_emits_only_new_and_changed_chunks() {
        let old_a = test_item("a.context.md", "A1");
        let old_b = test_item("b.context.md", "B1");
        let old_d = test_item("d.context.md", "D1");
        let previous = HashMap::from([
            (old_a.path.clone(), old_a.fingerprint.clone()),
            (old_b.path.clone(), old_b.fingerprint.clone()),
            (old_d.path.clone(), old_d.fingerprint.clone()),
        ]);

        let items = vec![
            test_item("a.context.md", "A1"),
            test_item("b.context.md", "B2"),
            test_item("c.context.md", "C1"),
        ];
        let delta = select_delta_items(&items, Some(&previous));

        assert_eq!(delta.unchanged, 1);
        assert_eq!(delta.removed, 1);
        assert_eq!(delta.emitted.len(), 2);
        assert_eq!(delta.emitted[0].path, PathBuf::from("b.context.md"));
        assert_eq!(delta.emitted[1].path, PathBuf::from("c.context.md"));
    }

    #[test]
    fn select_delta_items_emits_full_set_without_snapshot() {
        let items = vec![
            test_item("a.context.md", "A1"),
            test_item("b.context.md", "B1"),
        ];
        let delta = select_delta_items(&items, None);

        assert_eq!(delta.unchanged, 0);
        assert_eq!(delta.removed, 0);
        assert_eq!(delta.emitted.len(), 2);
    }

    #[test]
    fn select_capsule_modules_prefers_explicit_or_dominant_module() {
        let items = vec![
            (PathBuf::from("auth.context.md"), 8.0),
            (PathBuf::from("guard.context.md"), 6.5),
            (PathBuf::from("ui.context.md"), 4.0),
        ];
        let modules = HashMap::from([
            (PathBuf::from("auth.context.md"), "auth".to_string()),
            (PathBuf::from("guard.context.md"), "auth".to_string()),
            (PathBuf::from("ui.context.md"), "ui".to_string()),
        ]);

        assert_eq!(select_capsule_modules(&items, None, &modules), vec!["auth"]);
        assert_eq!(
            select_capsule_modules(&items, Some("ui"), &modules),
            vec!["ui"]
        );
        assert!(select_capsule_modules(&items, Some("@alice"), &modules).is_empty());
    }

    #[test]
    fn select_capsule_anchor_paths_keeps_top_dynamic_neurons() {
        let items = vec![
            (PathBuf::from("auth.context.md"), 9.0),
            (PathBuf::from("guard.context.md"), 6.0),
            (PathBuf::from("session.context.md"), 4.0),
            (PathBuf::from("ui.context.md"), 7.0),
        ];
        let modules = HashMap::from([
            (PathBuf::from("auth.context.md"), "auth".to_string()),
            (PathBuf::from("guard.context.md"), "auth".to_string()),
            (PathBuf::from("session.context.md"), "auth".to_string()),
            (PathBuf::from("ui.context.md"), "ui".to_string()),
        ]);
        let active = HashSet::from(["auth".to_string()]);

        let keep = select_capsule_anchor_paths(&items, &active, &modules);
        assert!(keep.contains(&PathBuf::from("auth.context.md")));
        assert!(keep.contains(&PathBuf::from("guard.context.md")));
        assert!(!keep.contains(&PathBuf::from("session.context.md")));
        assert!(!keep.contains(&PathBuf::from("ui.context.md")));
    }

    #[test]
    fn render_context_item_reports_summary_read_error() {
        let path = PathBuf::from("missing.context.md");
        let rendered = render_context_item(&path, 4.0, &[], &NeuronIndex::default());
        assert!(rendered.rendered.contains("read error"));
    }

    #[test]
    fn render_context_item_strips_answer_and_query_surface_sections() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("example.context.md");
        fs::write(
            &path,
            "# Example\n\nUseful body.\n\n## answer_surface\n<!-- SECTION: answer_surface -->\n| question_pattern | answer_span | confidence |\n| --- | --- | --- |\n| role | reviewer | 0.9 |\n<!-- /SECTION -->\n\n## query_surface\n<!-- SECTION: query_surface -->\n- audit auth route\n<!-- /SECTION -->\n",
        )
        .unwrap();

        let rendered = render_context_item(&path, 9.0, &[], &NeuronIndex::default());
        assert!(rendered.rendered.contains("Useful body."));
        assert!(!rendered.rendered.contains("answer_surface"));
        assert!(!rendered.rendered.contains("query_surface"));
        assert!(!rendered.rendered.contains("reviewer"));
        assert!(!rendered.rendered.contains("audit auth route"));
    }

    #[test]
    fn render_context_item_uses_focused_excerpt_for_large_verbatim_contexts() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("conversation.verbatim.md");
        let filler =
            "Assistant: We also talked about side topics that are not relevant right now.\n"
                .repeat(24);
        let content = format!(
            "User: I asked about travel plans.\nAssistant: We discussed itineraries.\n{filler}User: The venue I picked was Revolution Hall in Portland.\nAssistant: Revolution Hall is a great choice for indie music.\nUser: I also booked dinner nearby.\nAssistant: Enjoy the show.\n"
        );
        fs::write(&path, content).unwrap();

        let rendered = render_context_item(
            &path,
            6.0,
            &[
                "portland".to_string(),
                "venue".to_string(),
                "indie".to_string(),
            ],
            &NeuronIndex::default(),
        );
        assert!(rendered.rendered.contains("focused"));
        assert!(rendered.rendered.contains("Revolution Hall"));
        assert!(!rendered.rendered.contains("travel plans"));
    }

    #[test]
    fn render_context_item_prefers_key_markdown_sections_in_focused_mode() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("auth.context.md");
        let notes =
            "Additional historical notes about auth migrations and rollout details.\n".repeat(24);
        let content = format!(
            "# Auth\n\n## purpose\nKeep tokens low for auth tasks.\n\n## api\nUse `require_auth()` before accessing the session.\n\n## pitfalls\nDo not trust unsigned cookies.\n\n## notes\n{notes}\n"
        );
        fs::write(&path, content).unwrap();

        let rendered = render_context_item(
            &path,
            6.2,
            &["auth".to_string(), "fix".to_string(), "session".to_string()],
            &NeuronIndex::default(),
        );
        assert!(rendered.rendered.contains("## purpose"));
        assert!(rendered.rendered.contains("## api"));
        assert!(rendered.rendered.contains("## pitfalls"));
        assert!(!rendered.rendered.contains("## notes"));
    }

    #[test]
    fn render_module_capsule_reports_read_error() {
        let dir = tempfile::tempdir().unwrap();
        let capsule_path = module_capsule_path(dir.path(), "auth");
        fs::create_dir_all(&capsule_path).unwrap();

        let rendered = render_module_capsule(dir.path(), "auth").unwrap();
        assert!(rendered.rendered.contains("read error"));
    }

    #[test]
    fn render_agent_memory_summary_uses_structured_diary_fields() {
        let content = render_structured_diary_entry(
            "reviewer",
            "Investigated auth middleware coverage.",
            Some("Audit auth middleware"),
            Some("done"),
            Some("Close the auth bypass."),
            Some("Patch the legacy REST route."),
            Some("Waiting on route ownership clarification."),
            Some("Found a bypass in the legacy route."),
            &["auth".to_string(), "middleware".to_string()],
            &["router-owner".to_string()],
        );
        let summary = render_agent_memory_summary(&content, Some(1_710_000_000));
        assert!(summary.contains("Audit auth middleware"));
        assert!(summary.contains("status: done"));
        assert!(summary.contains("goal: Close the auth bypass."));
        assert!(summary.contains("blocker: Waiting on route ownership clarification."));
        assert!(summary.contains("Found a bypass in the legacy route."));
    }

    #[test]
    fn flush_provisional_hits_blocking_clears_without_training_feedback() {
        let dir = tempfile::tempdir().unwrap();
        let neuron_path = dir.path().join("example.context.md");
        fs::write(&neuron_path, "example").unwrap();
        let meta = NeuronMeta::new_stub(&neuron_path, NeuronKind::Core);
        fs::write(
            meta_path(&neuron_path),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();

        let mut idx = NeuronIndex::default();
        idx.index_neuron(&neuron_path, "example context body", &meta);
        let before = idx.use_count_for(&neuron_path);
        let index = std::sync::Arc::new(tokio::sync::RwLock::new(idx));
        let provisional = std::sync::Arc::new(tokio::sync::Mutex::new(vec![neuron_path.clone()]));

        let cleared = flush_provisional_hits_blocking(&index, &provisional).unwrap();

        assert_eq!(cleared, 1);
        assert!(provisional.blocking_lock().is_empty());
        assert_eq!(index.blocking_read().use_count_for(&neuron_path), before);
    }

    #[test]
    fn resolve_neuron_store_path_accepts_neuron_and_rejects_escape() {
        let dir = tempfile::tempdir().unwrap();
        let neuron_root = neuron_dir(dir.path());
        fs::create_dir_all(&neuron_root).unwrap();
        let neuron_path = neuron_root.join("example.context.md");
        fs::write(&neuron_path, "hello").unwrap();
        let outside = dir.path().join("outside.context.md");
        fs::write(&outside, "nope").unwrap();

        let resolved =
            resolve_neuron_store_path(&neuron_path.display().to_string(), dir.path()).unwrap();
        assert_eq!(resolved, neuron_path.canonicalize().unwrap());
        assert!(resolve_neuron_store_path(&outside.display().to_string(), dir.path()).is_err());
    }

    #[test]
    fn build_augmented_task_includes_editor_and_error_terms() {
        let dir = tempfile::tempdir().unwrap();
        let mut index = NeuronIndex::default();
        let source = dir.path().join("src").join("auth.rs");
        fs::create_dir_all(source.parent().unwrap()).unwrap();
        fs::write(&source, "fn auth() {}").unwrap();
        let neuron_path = core_neuron_path(&source, dir.path());
        fs::create_dir_all(neuron_path.parent().unwrap()).unwrap();
        let mut meta = NeuronMeta::new_stub(&source, NeuronKind::Core);
        meta.tokens = crate::neuron::estimate_context_tokens(
            "token validation middleware refresh auth session",
        );
        index.index_neuron(
            &neuron_path,
            "token validation middleware refresh auth session",
            &meta,
        );

        let input = GetContextsInput {
            task: "fix auth".to_string(),
            max_tokens: None,
            module: None,
            person: None,
            kind: None,
            min_confidence: None,
            multi_hop: None,
            previous_response: None,
            open_files: Some(vec!["src/auth.rs".to_string()]),
            error_context: Some("middleware validation failed".to_string()),
            delta_mode: None,
            context_handle: None,
            capsule_mode: None,
            answer_mode: None,
            min_answer_confidence: None,
            provenance_mode: None,
        };

        let augmented = build_augmented_task(&index, &input);
        assert!(augmented.contains("fix auth"));
        assert!(augmented.contains("middleware"));
        assert!(augmented.contains("validation"));
    }

    #[test]
    fn cortyx_route_auto_uses_answer_for_questions() {
        let route = derive_cortyx_route(&CortyxInput {
            intent: None,
            task: Some("What is my job?".to_string()),
            agent: None,
            person: None,
            module: None,
            kind: None,
            path: None,
            max_tokens: None,
            min_confidence: None,
            multi_hop: None,
            previous_response: None,
            delta_mode: None,
            context_handle: None,
            capsule_mode: None,
            min_answer_confidence: None,
            provenance_mode: None,
            include_timeline: None,
        })
        .unwrap();
        assert_eq!(route.kind, CortyxRouteKind::Answer);
    }

    #[test]
    fn cortyx_route_auto_uses_agent_status_for_agent_only() {
        let route = derive_cortyx_route(&CortyxInput {
            intent: None,
            task: None,
            agent: Some("reviewer".to_string()),
            person: None,
            module: None,
            kind: None,
            path: None,
            max_tokens: None,
            min_confidence: None,
            multi_hop: None,
            previous_response: None,
            delta_mode: None,
            context_handle: None,
            capsule_mode: None,
            min_answer_confidence: None,
            provenance_mode: None,
            include_timeline: None,
        })
        .unwrap();
        assert_eq!(route.kind, CortyxRouteKind::AgentStatus);
        assert_eq!(route.agent.as_deref(), Some("reviewer"));
    }

    #[test]
    fn cortyx_route_auto_without_inputs_uses_capability_summary() {
        let route = derive_cortyx_route(&CortyxInput {
            intent: None,
            task: None,
            agent: None,
            person: None,
            module: None,
            kind: None,
            path: None,
            max_tokens: None,
            min_confidence: None,
            multi_hop: None,
            previous_response: None,
            delta_mode: None,
            context_handle: None,
            capsule_mode: None,
            min_answer_confidence: None,
            provenance_mode: None,
            include_timeline: None,
        })
        .unwrap();
        assert_eq!(route.kind, CortyxRouteKind::Capabilities);
        assert!(route.task.is_none());
        assert!(route.agent.is_none());
    }

    #[test]
    fn cortyx_route_auto_uses_wake_up_for_priming_request() {
        let route = derive_cortyx_route(&CortyxInput {
            intent: None,
            task: Some("Wake up the session with reviewer memory".to_string()),
            agent: Some("reviewer".to_string()),
            person: None,
            module: None,
            kind: None,
            path: None,
            max_tokens: None,
            min_confidence: None,
            multi_hop: None,
            previous_response: None,
            delta_mode: None,
            context_handle: None,
            capsule_mode: None,
            min_answer_confidence: None,
            provenance_mode: None,
            include_timeline: None,
        })
        .unwrap();
        assert_eq!(route.kind, CortyxRouteKind::WakeUp);
    }

    #[tokio::test]
    async fn benchmark_cortyx_routes_answer_intent_to_answer_mode() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = NeuronIndex::load_or_create(dir.path()).unwrap();
        miner::mine_text(
            "I work as a pediatric nurse at the city hospital.",
            "diary",
            dir.path(),
            &mut idx,
            None,
            Some("user"),
            Some("2026-04-17T10:00:00Z"),
        )
        .unwrap();
        let server = CortyxServer::for_benchmark(dir.path().to_path_buf(), idx);
        let output = server
            .benchmark_cortyx(CortyxInput {
                intent: Some("answer".to_string()),
                task: Some("What is my job?".to_string()),
                agent: None,
                person: None,
                module: None,
                kind: None,
                path: None,
                max_tokens: Some(4000),
                min_confidence: None,
                multi_hop: None,
                previous_response: None,
                delta_mode: None,
                context_handle: None,
                capsule_mode: None,
                min_answer_confidence: None,
                provenance_mode: Some(false),
                include_timeline: None,
            })
            .await;
        assert!(output.to_ascii_lowercase().contains("pediatric nurse"));
    }

    #[tokio::test]
    async fn benchmark_cortyx_without_inputs_returns_capability_summary() {
        let dir = tempfile::tempdir().unwrap();
        let idx = NeuronIndex::load_or_create(dir.path()).unwrap();
        let server = CortyxServer::for_benchmark(dir.path().to_path_buf(), idx);

        let output = server
            .benchmark_cortyx(CortyxInput {
                intent: None,
                task: None,
                agent: None,
                person: None,
                module: None,
                kind: None,
                path: None,
                max_tokens: None,
                min_confidence: None,
                multi_hop: None,
                previous_response: None,
                delta_mode: None,
                context_handle: None,
                capsule_mode: None,
                min_answer_confidence: None,
                provenance_mode: None,
                include_timeline: None,
            })
            .await;

        assert!(output.contains("Cortyx capability summary"));
        assert!(output.contains("Default entrypoint: cortyx(task=\"...\")"));
        assert!(output.contains("shared sync: 0 pending item(s), 0 conflict(s)"));
    }

    #[tokio::test]
    async fn benchmark_cortyx_scopes_agent_questions_to_agent_memory() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = NeuronIndex::load_or_create(dir.path()).unwrap();
        let diary = render_structured_diary_entry(
            "reviewer",
            "Audited the legacy auth route.",
            Some("Close auth bypass"),
            Some("in_progress"),
            Some("Close the auth bypass without regressing login."),
            Some("Patch the legacy REST route after ownership is confirmed."),
            Some("Waiting on route ownership clarification."),
            Some("Confirmed the bypass only exists on the legacy REST path."),
            &["auth".to_string(), "routing".to_string()],
            &["router-owner".to_string()],
        );
        miner::mine_text(
            &diary,
            "diary",
            dir.path(),
            &mut idx,
            Some("@agent/reviewer"),
            None,
            Some("2026-04-17T10:00:00Z"),
        )
        .unwrap();
        let server = CortyxServer::for_benchmark(dir.path().to_path_buf(), idx);
        let output = server
            .benchmark_cortyx(CortyxInput {
                intent: None,
                task: Some("What is the reviewer's goal?".to_string()),
                agent: Some("reviewer".to_string()),
                person: None,
                module: None,
                kind: None,
                path: None,
                max_tokens: Some(4000),
                min_confidence: None,
                multi_hop: None,
                previous_response: None,
                delta_mode: None,
                context_handle: None,
                capsule_mode: None,
                min_answer_confidence: None,
                provenance_mode: Some(false),
                include_timeline: None,
            })
            .await;
        assert!(output
            .to_ascii_lowercase()
            .contains("close the auth bypass without regressing login"));
    }

    #[tokio::test]
    async fn benchmark_cortyx_answer_mode_can_abstain_with_min_answer_confidence() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = NeuronIndex::load_or_create(dir.path()).unwrap();
        miner::mine_text(
            "Work has been stressful lately, and I keep thinking about my future career path.",
            "diary",
            dir.path(),
            &mut idx,
            None,
            Some("user"),
            Some("2026-04-17T10:00:00Z"),
        )
        .unwrap();
        let server = CortyxServer::for_benchmark(dir.path().to_path_buf(), idx);
        let output = server
            .benchmark_cortyx(CortyxInput {
                intent: Some("answer".to_string()),
                task: Some("What is my job?".to_string()),
                agent: None,
                person: None,
                module: None,
                kind: None,
                path: None,
                max_tokens: Some(4000),
                min_confidence: None,
                multi_hop: None,
                previous_response: None,
                delta_mode: None,
                context_handle: None,
                capsule_mode: None,
                min_answer_confidence: Some(0.6),
                provenance_mode: Some(false),
                include_timeline: None,
            })
            .await;
        assert!(output.trim().is_empty());
    }

    #[tokio::test]
    async fn mutation_tools_record_provenance_history() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("src").join("auth.rs");
        let target_source = dir.path().join("src").join("guard.rs");
        fs::create_dir_all(source.parent().unwrap()).unwrap();
        fs::write(&source, "fn auth() -> bool { true }\n").unwrap();
        fs::write(&target_source, "fn guard() {}\n").unwrap();

        let idx = NeuronIndex::load_or_create(dir.path()).unwrap();
        let server = CortyxServer::for_benchmark(dir.path().to_path_buf(), idx);

        let initial_content =
            "# Auth\n\n## purpose\n<!-- SECTION: purpose -->\nInitial purpose.\n<!-- /SECTION -->\n";
        let evolve = server
            .evolve_context(Parameters(EvolveContextInput {
                path: "src/auth.rs".to_string(),
                content: initial_content.to_string(),
            }))
            .await;
        assert!(evolve.contains("Neuron evolved"));

        let neuron_path = core_neuron_path(&source, dir.path());
        let create_provenance = load_provenance(&neuron_path).unwrap().unwrap();
        assert_eq!(create_provenance.edit_history.len(), 1);
        let create_edit = &create_provenance.edit_history[0];
        let initial_hash = provenance_content_hash(initial_content);
        assert_eq!(create_edit.operation, ProvenanceOperation::Create);
        assert_eq!(create_edit.source, ProvenanceSource::Local);
        assert_eq!(
            create_edit.summary.as_deref(),
            Some("created neuron from src/auth.rs")
        );
        assert_eq!(
            create_edit.content_hash.as_deref(),
            Some(initial_hash.as_str())
        );
        let create_edit_id = create_edit.edit_id.clone();

        let update = server
            .evolve_section(Parameters(EvolveSectionInput {
                path: "src/auth.rs".to_string(),
                section: "purpose".to_string(),
                content: "Refined purpose.".to_string(),
            }))
            .await;
        assert!(update.contains("Section 'purpose' updated"));

        let updated_provenance = load_provenance(&neuron_path).unwrap().unwrap();
        assert_eq!(updated_provenance.edit_history.len(), 2);
        let section_edit = updated_provenance.edit_history.last().unwrap();
        assert_eq!(section_edit.operation, ProvenanceOperation::SectionUpdate);
        assert_eq!(section_edit.source, ProvenanceSource::Local);
        assert_eq!(section_edit.section.as_deref(), Some("purpose"));
        assert_eq!(
            section_edit.summary.as_deref(),
            Some("updated purpose section for src/auth.rs")
        );
        assert_eq!(
            section_edit.parent_edit_id.as_deref(),
            Some(create_edit_id.as_str())
        );

        let rollback = server
            .rollback_section(Parameters(RollbackSectionInput {
                neuron_path: neuron_path.display().to_string(),
                section: "purpose".to_string(),
            }))
            .await;
        assert!(rollback.contains("Restored section 'purpose'"));

        let rolled_back_provenance = load_provenance(&neuron_path).unwrap().unwrap();
        assert_eq!(rolled_back_provenance.edit_history.len(), 3);
        let rollback_edit = rolled_back_provenance.edit_history.last().unwrap();
        assert_eq!(rollback_edit.operation, ProvenanceOperation::Rollback);
        assert_eq!(rollback_edit.source, ProvenanceSource::Local);
        assert_eq!(rollback_edit.section.as_deref(), Some("purpose"));
        assert_eq!(
            rollback_edit.summary.as_deref(),
            Some("restored purpose from rollback shadow")
        );
        let rollback_edit_id = rollback_edit.edit_id.clone();

        let target_neuron = core_neuron_path(&target_source, dir.path());
        fs::create_dir_all(target_neuron.parent().unwrap()).unwrap();
        fs::write(&target_neuron, "# Guard\n").unwrap();

        let neuron_root = neuron_dir(dir.path());
        let source_rel = neuron_path
            .strip_prefix(&neuron_root)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        let target_rel = target_neuron
            .strip_prefix(&neuron_root)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");

        let synapse = server
            .create_synapse(Parameters(CreateSynapseInput {
                source: source_rel.clone(),
                target: target_rel.clone(),
                reason: "imports guard helpers".to_string(),
                edge_type: Some(SynapseType::Imports),
            }))
            .await;
        assert!(synapse.contains("Synapse created"));

        let synapse_provenance = load_provenance(&neuron_path).unwrap().unwrap();
        assert_eq!(synapse_provenance.edit_history.len(), 4);
        let synapse_edit = synapse_provenance.edit_history.last().unwrap();
        let synapse_summary = format!("added synapse to {target_rel}");
        let current_hash = provenance_content_hash(&fs::read_to_string(&neuron_path).unwrap());
        assert_eq!(synapse_edit.operation, ProvenanceOperation::SectionUpdate);
        assert_eq!(synapse_edit.source, ProvenanceSource::Local);
        assert_eq!(synapse_edit.section.as_deref(), Some("cross-references"));
        assert_eq!(
            synapse_edit.summary.as_deref(),
            Some(synapse_summary.as_str())
        );
        assert_eq!(
            synapse_edit.parent_edit_id.as_deref(),
            Some(rollback_edit_id.as_str())
        );
        assert_eq!(
            synapse_edit.content_hash.as_deref(),
            Some(current_hash.as_str())
        );
    }

    #[tokio::test]
    async fn extract_from_raw_records_import_provenance() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("src").join("router.rs");
        fs::create_dir_all(source.parent().unwrap()).unwrap();
        fs::write(&source, "fn route() {}\n").unwrap();

        let idx = NeuronIndex::load_or_create(dir.path()).unwrap();
        let server = CortyxServer::for_benchmark(dir.path().to_path_buf(), idx);

        let task_pattern = "audit auth routes";
        let extracted = server
            .extract_from_raw(Parameters(ExtractFromRawInput {
                path: "src/router.rs".to_string(),
                task_pattern: task_pattern.to_string(),
                chunk: "fn route() {}".to_string(),
                why: "It shows the auth guard order.".to_string(),
            }))
            .await;
        assert!(extracted.contains("Use-case neuron created"));

        let source_rel = "src/router.rs".replace(['/', '\\'], "_");
        let task_kebab = truncate_str(&to_kebab(task_pattern), 64);
        let neuron_path =
            neuron_dir(dir.path()).join(format!("{source_rel}.usecase.{task_kebab}.md"));
        let content = fs::read_to_string(&neuron_path).unwrap();
        let provenance = load_provenance(&neuron_path).unwrap().unwrap();
        assert_eq!(provenance.source_path.as_deref(), Some(source.as_path()));
        assert_eq!(provenance.edit_history.len(), 1);
        let edit = &provenance.edit_history[0];
        let expected_hash = provenance_content_hash(&content);
        assert_eq!(edit.operation, ProvenanceOperation::Create);
        assert_eq!(edit.source, ProvenanceSource::Import);
        assert_eq!(
            edit.summary.as_deref(),
            Some("extracted raw chunk for pattern \"audit auth routes\"")
        );
        assert_eq!(edit.content_hash.as_deref(), Some(expected_hash.as_str()));

        let meta: NeuronMeta =
            serde_json::from_str(&fs::read_to_string(meta_path(&neuron_path)).unwrap()).unwrap();
        assert!(!meta.source_hash.is_empty());
    }

    #[test]
    fn sync_structured_diary_to_kg_replaces_active_agent_state() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = NeuronIndex::default();
        let first = parse_structured_diary_entry(&render_structured_diary_entry(
            "reviewer",
            "Investigated auth middleware coverage.",
            Some("Audit auth middleware"),
            Some("in_progress"),
            Some("Close the auth bypass."),
            Some("Patch the legacy REST route."),
            Some("Waiting on route ownership clarification."),
            Some("Tracing the auth bypass."),
            &["auth".to_string()],
            &["router-owner".to_string()],
        ))
        .unwrap();
        sync_structured_diary_to_kg(
            dir.path(),
            &mut idx,
            "reviewer",
            &first,
            "2026-04-17T10:00:00Z",
        )
        .unwrap();

        let second = parse_structured_diary_entry(&render_structured_diary_entry(
            "reviewer",
            "Patched the legacy REST route.",
            Some("Close auth bypass"),
            Some("done"),
            Some("Close the auth bypass."),
            Some("Ship the regression tests."),
            None,
            Some("Removed the legacy auth bypass."),
            &["auth".to_string(), "routing".to_string()],
            &["qa".to_string()],
        ))
        .unwrap();
        sync_structured_diary_to_kg(
            dir.path(),
            &mut idx,
            "reviewer",
            &second,
            "2026-04-17T10:05:00Z",
        )
        .unwrap();

        let entity = kg::KgEntity::load(&kg::kg_neuron_path(
            dir.path(),
            &agent_entity_name("reviewer"),
        ))
        .unwrap();
        assert_eq!(
            latest_active_kg_value(&entity, AGENT_STATUS_PREDICATE).as_deref(),
            Some("done")
        );
        assert_eq!(
            latest_active_kg_value(&entity, AGENT_FOCUS_PREDICATE).as_deref(),
            Some("Close auth bypass")
        );
        assert_eq!(
            latest_active_kg_value(&entity, AGENT_GOAL_PREDICATE).as_deref(),
            Some("Close the auth bypass.")
        );
        assert_eq!(
            latest_active_kg_value(&entity, AGENT_NEXT_STEP_PREDICATE).as_deref(),
            Some("Ship the regression tests.")
        );
        assert_eq!(
            latest_active_kg_value(&entity, AGENT_BLOCKER_PREDICATE),
            None
        );
        let related = active_kg_values(&entity, AGENT_RELATED_ENTITY_PREDICATE);
        assert_eq!(related, vec!["auth".to_string(), "routing".to_string()]);
        let depends_on = active_kg_values(&entity, AGENT_DEPENDS_ON_PREDICATE);
        assert_eq!(depends_on, vec!["qa".to_string()]);
        let status_timeline = entity.timeline_for(AGENT_STATUS_PREDICATE);
        assert_eq!(status_timeline.len(), 2);
        assert_eq!(status_timeline[0].ended, "2026-04-17T10:05:00Z");
        let blocker_timeline = entity.timeline_for(AGENT_BLOCKER_PREDICATE);
        assert_eq!(blocker_timeline.len(), 1);
        assert_eq!(blocker_timeline[0].ended, "2026-04-17T10:05:00Z");
    }

    #[test]
    fn render_collaboration_status_report_summarizes_team_sync_and_modules() {
        let projection = sample_collaboration_projection();

        let report =
            render_collaboration_status_report(&projection, None, None, true).expect("report");

        assert!(report.contains("## Collaboration Status"));
        assert!(report.contains("collaborators: 1"));
        assert!(report.contains("modules: 1"));
        assert!(report.contains("sync conflicts: 1"));
        assert!(report.contains("average trust score:"));
        assert!(report.contains("## Top collaborators"));
        assert!(report.contains("reviewer — sync_conflict"));
        assert!(report.contains("## Shared modules"));
        assert!(report.contains("engine — sync_conflict"));
        assert!(report.contains("## Collaboration timeline"));
    }

    #[test]
    fn render_collaboration_status_report_filters_to_module() {
        let projection = sample_collaboration_projection();

        let report = render_collaboration_status_report(&projection, None, Some("engine"), true)
            .expect("module report");

        assert!(report.contains("## Collaboration Module: engine"));
        assert!(report.contains("attention: sync_conflict"));
        assert!(report.contains("collaborators: reviewer"));
        assert!(report.contains("pending sync: yes"));
        assert!(report.contains("trust score:"));
        assert!(report.contains("## Collaboration timeline"));
    }

    #[test]
    fn render_agent_status_report_uses_temporal_kg_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = NeuronIndex::default();
        let entry = parse_structured_diary_entry(&render_structured_diary_entry(
            "reviewer",
            "Patched the legacy REST route.",
            Some("Close auth bypass"),
            Some("done"),
            Some("Close the auth bypass."),
            Some("Ship the regression tests."),
            Some("Waiting on QA sign-off."),
            Some("Removed the legacy auth bypass."),
            &["auth".to_string(), "routing".to_string()],
            &["qa".to_string()],
        ))
        .unwrap();
        sync_structured_diary_to_kg(
            dir.path(),
            &mut idx,
            "reviewer",
            &entry,
            "2026-04-17T10:05:00Z",
        )
        .unwrap();

        let report = render_agent_status_report(&idx, dir.path(), "reviewer", true).unwrap();
        assert!(report.contains("## Agent Status: reviewer"));
        assert!(report.contains("attention: blocked"));
        assert!(report.contains("focus: Close auth bypass"));
        assert!(report.contains("status: done"));
        assert!(report.contains("goal: Close the auth bypass."));
        assert!(report.contains("next step: Ship the regression tests."));
        assert!(report.contains("blocker: Waiting on QA sign-off."));
        assert!(report.contains("depends on: qa"));
        assert!(report.contains("Removed the legacy auth bypass."));
        assert!(report.contains("pending sync: no"));
        assert!(report.contains("## Supporting facts"));
        assert!(report.contains("## Collaboration timeline"));
    }
}
