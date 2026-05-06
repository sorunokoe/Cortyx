use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::agent_memory::summarize_structured_diary_entry;
use crate::kg::KgEntity;
use crate::sync_transport::{SyncHandoffState, SyncTransportStatus};

use super::attention::{collaboration_attention_score, CollaborationAttention};
use super::util::{is_collaboration_module, modules_for_diary_entry, normalize_summary_key};
use super::{
    CollaborationDiaryRecord, AGENT_BLOCKER_PREDICATE, AGENT_NEXT_STEP_PREDICATE,
    AGENT_STATUS_PREDICATE, DIRECT_AGENT_FACT_PREDICATES,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CollaborationTimelineKind {
    Diary,
    Fact,
    Sync,
    Conflict,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CollaborationTimelineEvent {
    pub kind: CollaborationTimelineKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collaborator: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub module: Option<String>,
    pub summary: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub related_entities: Vec<String>,
    pub attention: CollaborationAttention,
    pub attention_score: f32,
}

pub fn merge_collaboration_timeline<I>(events: I) -> Vec<CollaborationTimelineEvent>
where
    I: IntoIterator<Item = CollaborationTimelineEvent>,
{
    let mut deduped = BTreeMap::new();
    for event in events {
        let key = (
            event.kind,
            event.at.clone().unwrap_or_default(),
            event.collaborator.clone().unwrap_or_default(),
            event.module.clone().unwrap_or_default(),
            normalize_summary_key(&event.summary),
        );
        deduped.entry(key).or_insert(event);
    }

    let mut merged: Vec<CollaborationTimelineEvent> = deduped.into_values().collect();
    merged.sort_by(|left, right| {
        right
            .at
            .cmp(&left.at)
            .then_with(|| right.attention_score.total_cmp(&left.attention_score))
            .then_with(|| left.summary.cmp(&right.summary))
    });
    merged
}

pub(super) fn diary_timeline_events(
    collaborator: &str,
    diary_records: &[&CollaborationDiaryRecord],
    module_lookup: &BTreeMap<String, String>,
) -> Vec<CollaborationTimelineEvent> {
    use super::util::normalize_values;
    diary_records
        .iter()
        .map(|record| {
            let related_entities = normalize_values(
                &record
                    .entry
                    .entities
                    .iter()
                    .chain(record.entry.depends_on.iter())
                    .cloned()
                    .collect::<Vec<_>>(),
            );
            let (attention, attention_score) = collaboration_attention_score(
                record.entry.status.as_deref(),
                record.entry.blocker.as_deref(),
                record.entry.next_step.is_some(),
                0,
                0,
                record.entry.depends_on.len(),
                0,
                0,
            );
            CollaborationTimelineEvent {
                kind: CollaborationTimelineKind::Diary,
                at: record.when.clone(),
                collaborator: Some(collaborator.to_string()),
                module: modules_for_diary_entry(&record.entry, module_lookup)
                    .into_iter()
                    .next(),
                summary: summarize_structured_diary_entry(&record.entry),
                related_entities,
                attention,
                attention_score,
            }
        })
        .collect()
}

pub(super) fn sync_timeline_events(
    collaborator: &str,
    sync_statuses: &[&SyncTransportStatus],
) -> Vec<CollaborationTimelineEvent> {
    sync_statuses
        .iter()
        .filter_map(|status| {
            let module = status
                .module()
                .filter(|module| is_collaboration_module(module))
                .map(str::to_string);
            let target = module.clone().or_else(|| {
                status
                    .source_path()
                    .map(|path| path.display().to_string())
                    .filter(|value| !value.is_empty())
            });

            if status.conflict_count() > 0 {
                let (attention, attention_score) = collaboration_attention_score(
                    None,
                    None,
                    false,
                    0,
                    status.conflict_count(),
                    0,
                    status.integrity_issue_count(),
                    status.requires_trust_attention() as usize,
                );
                Some(CollaborationTimelineEvent {
                    kind: CollaborationTimelineKind::Conflict,
                    at: status.latest_activity_at().map(str::to_string),
                    collaborator: Some(collaborator.to_string()),
                    module,
                    summary: format!(
                        "sync conflict on {}{}{}",
                        target.unwrap_or_else(|| status.neuron_uuid.clone()),
                        status
                            .handoff_shared_edit_id()
                            .map(|edit_id| format!(" since {edit_id}"))
                            .unwrap_or_default(),
                        status
                            .latest_summary()
                            .map(|summary| format!(" — {summary}"))
                            .unwrap_or_default()
                    ),
                    related_entities: Vec::new(),
                    attention,
                    attention_score,
                })
            } else if status.pending_outgoing() || status.pending_incoming() {
                let (attention, attention_score) = collaboration_attention_score(
                    None,
                    None,
                    false,
                    status.pending_outgoing() as usize + status.pending_incoming() as usize,
                    0,
                    0,
                    status.integrity_issue_count(),
                    status.requires_trust_attention() as usize,
                );
                Some(CollaborationTimelineEvent {
                    kind: CollaborationTimelineKind::Sync,
                    at: status.latest_activity_at().map(str::to_string),
                    collaborator: Some(collaborator.to_string()),
                    module,
                    summary: format!(
                        "{} for {}{}{}",
                        sync_timeline_label(status.handoff.state),
                        target.unwrap_or_else(|| status.neuron_uuid.clone()),
                        status
                            .handoff_shared_edit_id()
                            .map(|edit_id| format!(" since {edit_id}"))
                            .unwrap_or_default(),
                        status
                            .latest_summary()
                            .map(|summary| format!(" — {summary}"))
                            .unwrap_or_default()
                    ),
                    related_entities: Vec::new(),
                    attention,
                    attention_score,
                })
            } else {
                None
            }
        })
        .collect()
}

pub(super) fn agent_fact_timeline_events(
    collaborator: &str,
    agent_kg: &KgEntity,
) -> Vec<CollaborationTimelineEvent> {
    let mut events = Vec::new();
    for predicate in DIRECT_AGENT_FACT_PREDICATES {
        let timeline = agent_kg.timeline_for(predicate);
        let Some(fact) = timeline.last() else {
            continue;
        };
        let (attention, attention_score) = collaboration_attention_score(
            (*predicate == AGENT_STATUS_PREDICATE).then_some(fact.value.as_str()),
            (*predicate == AGENT_BLOCKER_PREDICATE).then_some(fact.value.as_str()),
            *predicate == AGENT_NEXT_STEP_PREDICATE,
            0,
            0,
            0,
            0,
            0,
        );
        events.push(CollaborationTimelineEvent {
            kind: CollaborationTimelineKind::Fact,
            at: (!fact.valid_from.is_empty()).then_some(fact.valid_from.clone()),
            collaborator: Some(collaborator.to_string()),
            module: None,
            summary: format!("{predicate}: {}", fact.value),
            related_entities: Vec::new(),
            attention,
            attention_score,
        });
    }
    merge_collaboration_timeline(events)
}

pub(super) fn sync_author_label(status: &SyncTransportStatus) -> Option<String> {
    use super::util::identity_tokens;
    status
        .latest_author_display()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| {
            status
                .latest_author_id()
                .and_then(|value| identity_tokens(value).into_iter().next())
        })
        .or_else(|| {
            status
                .module()
                .and_then(|module| module.strip_prefix("@agent/"))
                .map(str::to_string)
        })
}

pub(super) fn collaborator_matches_sync(
    key: &str,
    display: &str,
    status: &SyncTransportStatus,
) -> bool {
    use super::util::{collaborator_key, identity_tokens};
    use std::collections::BTreeSet;

    if status
        .module()
        .and_then(|module| module.strip_prefix("@agent/"))
        .map(collaborator_key)
        .as_deref()
        == Some(key)
    {
        return true;
    }

    let collaborator_tokens: BTreeSet<String> = identity_tokens(display).into_iter().collect();
    if collaborator_tokens.is_empty() {
        return false;
    }

    [status.latest_author_display(), status.latest_author_id()]
        .into_iter()
        .flatten()
        .map(identity_tokens)
        .any(|tokens| {
            tokens
                .iter()
                .any(|token| collaborator_tokens.contains(token))
        })
}

fn sync_timeline_label(state: SyncHandoffState) -> &'static str {
    match state {
        SyncHandoffState::Idle => "sync",
        SyncHandoffState::PendingOutgoing => "outgoing handoff pending",
        SyncHandoffState::PendingIncoming => "incoming handoff pending",
        SyncHandoffState::Applied => "applied handoff",
        SyncHandoffState::Conflict => "conflicted handoff",
    }
}
