use serde::{Deserialize, Serialize};

use crate::sync_transport::SyncTransportStatus;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CollaborationAttention {
    #[default]
    Nominal,
    NeedsFollowUp,
    Blocked,
    SyncConflict,
}

impl CollaborationAttention {
    pub(super) fn severity(self) -> u8 {
        match self {
            Self::Nominal => 0,
            Self::NeedsFollowUp => 1,
            Self::Blocked => 2,
            Self::SyncConflict => 3,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct CollaborationEvidenceSummary {
    pub diary_entries: usize,
    pub kg_fact_count: usize,
    pub sync_neuron_count: usize,
    pub verified_sync_count: usize,
    pub pending_sync_count: usize,
    pub conflict_count: usize,
    pub integrity_issue_count: usize,
    pub untrusted_handoff_count: usize,
    pub reasoning_node_count: usize,
    pub reasoning_fact_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct CollaborationWorkflowMetrics {
    pub collaborator_count: usize,
    pub module_count: usize,
    pub timeline_event_count: usize,
    pub nominal_collaborator_count: usize,
    pub follow_up_collaborator_count: usize,
    pub blocked_collaborator_count: usize,
    pub sync_conflict_collaborator_count: usize,
    pub nominal_module_count: usize,
    pub follow_up_module_count: usize,
    pub blocked_module_count: usize,
    pub sync_conflict_module_count: usize,
    pub active_blocker_count: usize,
    pub pending_collaborator_count: usize,
    pub pending_module_count: usize,
    pub collaborator_attention_score_total: f32,
    pub module_attention_score_total: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub average_collaborator_trust_score: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub average_module_trust_score: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SharedTrustOutcomeReport {
    pub baseline_sync: crate::sync_transport::SyncTrustMetrics,
    pub candidate_sync: crate::sync_transport::SyncTrustMetrics,
    pub baseline_workflow: CollaborationWorkflowMetrics,
    pub candidate_workflow: CollaborationWorkflowMetrics,
    pub conflict_delta: isize,
    pub pending_sync_delta: isize,
    pub integrity_issue_delta: isize,
    pub trust_attention_delta: isize,
    pub fully_verified_delta: isize,
    pub active_blocker_delta: isize,
    pub collaborator_attention_score_delta: f32,
    pub module_attention_score_delta: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub average_sync_trust_score_delta: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub average_collaborator_trust_score_delta: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub average_module_trust_score_delta: Option<f32>,
    pub workflow_improved: bool,
    pub trust_improved: bool,
}

#[must_use]
pub fn collaboration_evidence_score(evidence: &CollaborationEvidenceSummary) -> f32 {
    evidence.diary_entries as f32
        + evidence.kg_fact_count as f32 * 0.4
        + evidence.sync_neuron_count as f32 * 0.8
        + evidence.verified_sync_count as f32 * 0.9
        + evidence.pending_sync_count as f32 * 0.5
        + evidence.conflict_count as f32 * 1.5
        + evidence.integrity_issue_count as f32 * 0.35
        + evidence.untrusted_handoff_count as f32 * 0.8
        + evidence.reasoning_node_count as f32 * 0.3
        + evidence.reasoning_fact_count as f32 * 0.45
}

pub fn collaboration_attention_score(
    status: Option<&str>,
    blocker: Option<&str>,
    has_next_step: bool,
    pending_sync_count: usize,
    conflict_count: usize,
    dependency_count: usize,
    integrity_issue_count: usize,
    untrusted_handoff_count: usize,
) -> (CollaborationAttention, f32) {
    let status_kind = classify_status(status);
    let blocked = blocker
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .is_some()
        || matches!(status_kind, Some(StatusKind::Blocked));
    let mut attention = CollaborationAttention::Nominal;
    let mut score = 0.0;

    if conflict_count > 0 {
        attention = CollaborationAttention::SyncConflict;
        score += 10.0 + conflict_count as f32;
    }
    if blocked {
        if attention.severity() < CollaborationAttention::Blocked.severity() {
            attention = CollaborationAttention::Blocked;
        }
        score += 7.0;
    }
    if pending_sync_count > 0 || dependency_count > 0 || has_next_step {
        if attention.severity() < CollaborationAttention::NeedsFollowUp.severity() {
            attention = CollaborationAttention::NeedsFollowUp;
        }
        score += pending_sync_count as f32 * 0.75;
        score += dependency_count as f32 * 0.5;
        if has_next_step {
            score += 1.0;
        }
    }
    if integrity_issue_count > 0 || untrusted_handoff_count > 0 {
        if attention.severity() < CollaborationAttention::NeedsFollowUp.severity() {
            attention = CollaborationAttention::NeedsFollowUp;
        }
        score += integrity_issue_count as f32 * 0.75;
        score += untrusted_handoff_count as f32 * 1.25;
    }

    if score == 0.0
        && matches!(
            status_kind,
            Some(StatusKind::InProgress | StatusKind::Planned)
        )
    {
        attention = CollaborationAttention::NeedsFollowUp;
        score = 1.0;
    }

    (attention, score)
}

pub(super) fn aggregate_trust_score<'a, I>(statuses: I) -> Option<f32>
where
    I: IntoIterator<Item = &'a SyncTransportStatus>,
{
    let scores = statuses
        .into_iter()
        .map(SyncTransportStatus::trust_score)
        .collect::<Vec<_>>();
    average_scores(&scores)
}

pub(super) fn score_not_worse(after: Option<f32>, before: Option<f32>) -> bool {
    match (after, before) {
        (Some(after), Some(before)) => after + f32::EPSILON >= before,
        (Some(_), None) => true,
        (None, Some(_)) => false,
        (None, None) => true,
    }
}

pub(super) fn average_scores(values: &[f32]) -> Option<f32> {
    (!values.is_empty()).then_some(values.iter().sum::<f32>() / values.len() as f32)
}

pub(super) fn delta_usize(after: usize, before: usize) -> isize {
    after as isize - before as isize
}

pub(super) fn delta_score(after: Option<f32>, before: Option<f32>) -> Option<f32> {
    match (after, before) {
        (Some(after), Some(before)) => Some(after - before),
        (Some(after), None) => Some(after),
        (None, Some(before)) => Some(-before),
        (None, None) => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum StatusKind {
    Planned,
    InProgress,
    Blocked,
    Done,
    Unknown,
}

pub(super) fn classify_status(status: Option<&str>) -> Option<StatusKind> {
    let status = status?.trim().to_ascii_lowercase();
    if status.is_empty() {
        return None;
    }
    Some(
        if status.contains("block") || status.contains("stuck") || status.contains("wait") {
            StatusKind::Blocked
        } else if status.contains("done")
            || status.contains("complete")
            || status.contains("resolved")
            || status.contains("fixed")
        {
            StatusKind::Done
        } else if status.contains("progress")
            || status.contains("active")
            || status.contains("working")
            || status.contains("doing")
        {
            StatusKind::InProgress
        } else if status.contains("todo")
            || status.contains("plan")
            || status.contains("queued")
            || status.contains("pending")
        {
            StatusKind::Planned
        } else {
            StatusKind::Unknown
        },
    )
}
