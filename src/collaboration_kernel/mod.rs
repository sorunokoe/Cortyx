//! Collaboration-kernel state projection helpers.
//!
//! This layer composes structured agent diaries, mirrored KG facts, shared-sync
//! snapshots, and optional graph-reasoner evidence into reusable state summaries
//! that later CLI/MCP/status surfaces can render without reimplementing the merge logic.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::agent_memory::StructuredDiaryEntry;
use crate::kg;

mod attention;
mod projection;
#[cfg(test)]
mod tests;
mod timeline;
mod util;

pub use attention::{
    collaboration_attention_score, collaboration_evidence_score, CollaborationAttention,
    CollaborationEvidenceSummary, CollaborationWorkflowMetrics, SharedTrustOutcomeReport,
};
pub use projection::{
    compare_shared_trust_outcomes, project_collaboration_state, summarize_collaboration_workflow,
    CollaborationStateProjection, CollaboratorSummary, ModuleCollaborationState,
};
pub use timeline::{
    merge_collaboration_timeline, CollaborationTimelineEvent, CollaborationTimelineKind,
};

pub const AGENT_FOCUS_PREDICATE: &str = "focus";
pub const AGENT_STATUS_PREDICATE: &str = "status";
pub const AGENT_GOAL_PREDICATE: &str = "goal";
pub const AGENT_NEXT_STEP_PREDICATE: &str = "next_step";
pub const AGENT_BLOCKER_PREDICATE: &str = "blocker";
pub const AGENT_OUTCOME_PREDICATE: &str = "outcome";
pub const AGENT_ACTION_PREDICATE: &str = "action";
pub const AGENT_RELATED_ENTITY_PREDICATE: &str = "related_entity";
pub const AGENT_DEPENDS_ON_PREDICATE: &str = "depends_on";

const DIRECT_AGENT_FACT_PREDICATES: &[&str] = &[
    AGENT_FOCUS_PREDICATE,
    AGENT_STATUS_PREDICATE,
    AGENT_GOAL_PREDICATE,
    AGENT_NEXT_STEP_PREDICATE,
    AGENT_BLOCKER_PREDICATE,
    AGENT_OUTCOME_PREDICATE,
    AGENT_ACTION_PREDICATE,
];

const IDENTITY_STOPWORDS: &[&str] = &[
    "agent", "copilot", "device", "github", "local", "noreply", "user", "users",
];

pub fn agent_entity_name(agent: &str) -> String {
    format!("agent_{}", kg::slugify(agent))
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CollaborationDiaryRecord {
    pub collaborator: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub when: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<PathBuf>,
    pub entry: StructuredDiaryEntry,
}

impl CollaborationDiaryRecord {
    pub fn new(collaborator: impl Into<String>, entry: StructuredDiaryEntry) -> Self {
        Self {
            collaborator: collaborator.into(),
            when: None,
            path: None,
            entry,
        }
    }
}
