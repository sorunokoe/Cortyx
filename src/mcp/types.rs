//! MCP tool input and KG input type definitions.

use crate::neuron::SynapseType;
use rmcp::schemars::JsonSchema;
use serde::{Deserialize, Serialize};

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
    /// ECS risk threshold above which the write is rejected (0.0–1.0, default 0.60).
    /// Requires `--features verify`. Has no effect on default builds.
    pub min_ecs_threshold: Option<f64>,
    /// When `true`, skip ECS verification entirely for this call (e.g. trusted curator content).
    /// Requires `--features verify`. Has no effect on default builds.
    pub skip_verify: Option<bool>,
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
pub struct DiaryRefineInput {
    /// Agent identifier matching the one used with diary_write.
    pub agent: String,
    /// Optional explicit diary entry path to refine instead of the latest entry.
    pub entry_path: Option<String>,
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
pub(super) enum CortyxRouteKind {
    Context,
    Answer,
    WakeUp,
    AgentStatus,
    Consistency,
    Capabilities,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CortyxRoutePlan {
    pub(super) kind: CortyxRouteKind,
    pub(super) task: Option<String>,
    pub(super) agent: Option<String>,
}

pub(super) fn normalize_cortyx_route_intent(intent: &str) -> Option<CortyxRouteKind> {
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

pub(super) fn derive_cortyx_route(
    input: &CortyxInput,
) -> std::result::Result<CortyxRoutePlan, String> {
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

    let Some(task_value) = task else {
        unreachable!("task.is_none() returns above")
    };
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
