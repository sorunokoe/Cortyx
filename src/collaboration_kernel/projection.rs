use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde::{Deserialize, Serialize};

use crate::kg::KgEntity;
use crate::reasoner::ReasoningReport;
use crate::sync_transport::{summarize_sync_trust, SyncTransportStatus};

use super::attention::{
    aggregate_trust_score, average_scores, collaboration_attention_score,
    collaboration_evidence_score, delta_score, delta_usize, score_not_worse,
    CollaborationAttention, CollaborationEvidenceSummary, CollaborationWorkflowMetrics,
    SharedTrustOutcomeReport,
};
use super::timeline::{
    agent_fact_timeline_events, collaborator_matches_sync, diary_timeline_events,
    merge_collaboration_timeline, sync_author_label, sync_timeline_events,
    CollaborationTimelineEvent,
};
use super::util::{
    collaborator_from_agent_entity, collaborator_key, collect_supporting_facts,
    is_collaboration_module, latest_diary_record, max_optional_pair, max_optional_strings,
    merge_unique, modules_for_diary_entry, normalize_values, normalized_label,
};
use super::{
    agent_entity_name, CollaborationDiaryRecord, AGENT_ACTION_PREDICATE, AGENT_BLOCKER_PREDICATE,
    AGENT_DEPENDS_ON_PREDICATE, AGENT_FOCUS_PREDICATE, AGENT_GOAL_PREDICATE,
    AGENT_NEXT_STEP_PREDICATE, AGENT_OUTCOME_PREDICATE, AGENT_RELATED_ENTITY_PREDICATE,
    AGENT_STATUS_PREDICATE,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CollaboratorSummary {
    pub collaborator: String,
    pub entity: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub focus: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub goal: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_step: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocker: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub related_entities: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub depends_on: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub touched_modules: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub supporting_facts: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_updated: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latest_sync_edit_id: Option<String>,
    pub pending_sync: bool,
    pub attention: CollaborationAttention,
    pub attention_score: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trust_score: Option<f32>,
    pub evidence_score: f32,
    pub evidence: CollaborationEvidenceSummary,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModuleCollaborationState {
    pub module: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub collaborators: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub focuses: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blockers: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub related_entities: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub supporting_facts: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_updated: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latest_sync_edit_id: Option<String>,
    pub pending_sync: bool,
    pub attention: CollaborationAttention,
    pub attention_score: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trust_score: Option<f32>,
    pub evidence_score: f32,
    pub evidence: CollaborationEvidenceSummary,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct CollaborationStateProjection {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub collaborators: Vec<CollaboratorSummary>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub modules: Vec<ModuleCollaborationState>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub timeline: Vec<CollaborationTimelineEvent>,
}

pub fn summarize_collaboration_workflow(
    projection: &CollaborationStateProjection,
) -> CollaborationWorkflowMetrics {
    let mut metrics = CollaborationWorkflowMetrics {
        collaborator_count: projection.collaborators.len(),
        module_count: projection.modules.len(),
        timeline_event_count: projection.timeline.len(),
        ..Default::default()
    };
    let mut blockers = BTreeSet::new();
    let mut collaborator_trust_scores = Vec::new();
    let mut module_trust_scores = Vec::new();

    for summary in &projection.collaborators {
        match summary.attention {
            CollaborationAttention::Nominal => metrics.nominal_collaborator_count += 1,
            CollaborationAttention::NeedsFollowUp => metrics.follow_up_collaborator_count += 1,
            CollaborationAttention::Blocked => metrics.blocked_collaborator_count += 1,
            CollaborationAttention::SyncConflict => metrics.sync_conflict_collaborator_count += 1,
        }
        metrics.pending_collaborator_count += summary.pending_sync as usize;
        metrics.collaborator_attention_score_total += summary.attention_score;
        if let Some(blocker) = summary
            .blocker
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            blockers.insert(blocker.to_string());
        }
        if let Some(trust_score) = summary.trust_score {
            collaborator_trust_scores.push(trust_score);
        }
    }

    for state in &projection.modules {
        match state.attention {
            CollaborationAttention::Nominal => metrics.nominal_module_count += 1,
            CollaborationAttention::NeedsFollowUp => metrics.follow_up_module_count += 1,
            CollaborationAttention::Blocked => metrics.blocked_module_count += 1,
            CollaborationAttention::SyncConflict => metrics.sync_conflict_module_count += 1,
        }
        metrics.pending_module_count += state.pending_sync as usize;
        metrics.module_attention_score_total += state.attention_score;
        for blocker in &state.blockers {
            let blocker = blocker.trim();
            if !blocker.is_empty() {
                blockers.insert(blocker.to_string());
            }
        }
        if let Some(trust_score) = state.trust_score {
            module_trust_scores.push(trust_score);
        }
    }

    metrics.active_blocker_count = blockers.len();
    metrics.average_collaborator_trust_score = average_scores(&collaborator_trust_scores);
    metrics.average_module_trust_score = average_scores(&module_trust_scores);
    metrics
}

pub fn compare_shared_trust_outcomes(
    baseline_sync_statuses: &[SyncTransportStatus],
    baseline_projection: &CollaborationStateProjection,
    candidate_sync_statuses: &[SyncTransportStatus],
    candidate_projection: &CollaborationStateProjection,
) -> SharedTrustOutcomeReport {
    let baseline_sync = summarize_sync_trust(baseline_sync_statuses);
    let candidate_sync = summarize_sync_trust(candidate_sync_statuses);
    let baseline_workflow = summarize_collaboration_workflow(baseline_projection);
    let candidate_workflow = summarize_collaboration_workflow(candidate_projection);

    let conflict_delta = delta_usize(candidate_sync.conflict_count, baseline_sync.conflict_count);
    let pending_sync_delta = delta_usize(
        candidate_sync.pending_outgoing_count + candidate_sync.pending_incoming_count,
        baseline_sync.pending_outgoing_count + baseline_sync.pending_incoming_count,
    );
    let integrity_issue_delta = delta_usize(
        candidate_sync.integrity_issue_count,
        baseline_sync.integrity_issue_count,
    );
    let trust_attention_delta = delta_usize(
        candidate_sync.trust_attention_count,
        baseline_sync.trust_attention_count,
    );
    let fully_verified_delta = delta_usize(
        candidate_sync.fully_verified_neuron_count,
        baseline_sync.fully_verified_neuron_count,
    );
    let active_blocker_delta = delta_usize(
        candidate_workflow.active_blocker_count,
        baseline_workflow.active_blocker_count,
    );
    let collaborator_attention_score_delta = candidate_workflow.collaborator_attention_score_total
        - baseline_workflow.collaborator_attention_score_total;
    let module_attention_score_delta = candidate_workflow.module_attention_score_total
        - baseline_workflow.module_attention_score_total;
    let average_sync_trust_score_delta = delta_score(
        candidate_sync.average_trust_score,
        baseline_sync.average_trust_score,
    );
    let average_collaborator_trust_score_delta = delta_score(
        candidate_workflow.average_collaborator_trust_score,
        baseline_workflow.average_collaborator_trust_score,
    );
    let average_module_trust_score_delta = delta_score(
        candidate_workflow.average_module_trust_score,
        baseline_workflow.average_module_trust_score,
    );

    let workflow_strict_improvement = active_blocker_delta < 0
        || candidate_workflow.pending_collaborator_count
            < baseline_workflow.pending_collaborator_count
        || candidate_workflow.pending_module_count < baseline_workflow.pending_module_count
        || collaborator_attention_score_delta < -f32::EPSILON
        || module_attention_score_delta < -f32::EPSILON
        || average_collaborator_trust_score_delta.unwrap_or_default() > f32::EPSILON
        || average_module_trust_score_delta.unwrap_or_default() > f32::EPSILON;
    let workflow_improved = workflow_strict_improvement
        && candidate_workflow.active_blocker_count <= baseline_workflow.active_blocker_count
        && candidate_workflow.pending_collaborator_count
            <= baseline_workflow.pending_collaborator_count
        && candidate_workflow.pending_module_count <= baseline_workflow.pending_module_count
        && candidate_workflow.collaborator_attention_score_total
            <= baseline_workflow.collaborator_attention_score_total + f32::EPSILON
        && candidate_workflow.module_attention_score_total
            <= baseline_workflow.module_attention_score_total + f32::EPSILON
        && score_not_worse(
            candidate_workflow.average_collaborator_trust_score,
            baseline_workflow.average_collaborator_trust_score,
        )
        && score_not_worse(
            candidate_workflow.average_module_trust_score,
            baseline_workflow.average_module_trust_score,
        );

    let trust_strict_improvement = conflict_delta < 0
        || pending_sync_delta < 0
        || integrity_issue_delta < 0
        || trust_attention_delta < 0
        || fully_verified_delta > 0
        || average_sync_trust_score_delta.unwrap_or_default() > f32::EPSILON;
    let trust_improved = trust_strict_improvement
        && candidate_sync.conflict_count <= baseline_sync.conflict_count
        && candidate_sync.integrity_issue_count <= baseline_sync.integrity_issue_count
        && candidate_sync.trust_attention_count <= baseline_sync.trust_attention_count
        && candidate_sync.pending_outgoing_count + candidate_sync.pending_incoming_count
            <= baseline_sync.pending_outgoing_count + baseline_sync.pending_incoming_count
        && candidate_sync.fully_verified_neuron_count >= baseline_sync.fully_verified_neuron_count
        && score_not_worse(
            candidate_sync.average_trust_score,
            baseline_sync.average_trust_score,
        )
        && score_not_worse(
            candidate_sync.average_handoff_score,
            baseline_sync.average_handoff_score,
        );

    SharedTrustOutcomeReport {
        baseline_sync,
        candidate_sync,
        baseline_workflow,
        candidate_workflow,
        conflict_delta,
        pending_sync_delta,
        integrity_issue_delta,
        trust_attention_delta,
        fully_verified_delta,
        active_blocker_delta,
        collaborator_attention_score_delta,
        module_attention_score_delta,
        average_sync_trust_score_delta,
        average_collaborator_trust_score_delta,
        average_module_trust_score_delta,
        workflow_improved,
        trust_improved,
    }
}

pub fn project_collaboration_state(
    diaries: &[CollaborationDiaryRecord],
    sync_statuses: &[SyncTransportStatus],
    kg_entities: &[KgEntity],
    reasoning: Option<&ReasoningReport>,
) -> CollaborationStateProjection {
    let kg_by_entity: HashMap<String, &KgEntity> = kg_entities
        .iter()
        .map(|entity| (entity.entity.clone(), entity))
        .collect();
    let module_lookup = build_module_lookup(sync_statuses, reasoning);

    let mut display_by_key = BTreeMap::new();
    let mut diaries_by_key: BTreeMap<String, Vec<&CollaborationDiaryRecord>> = BTreeMap::new();
    for diary in diaries {
        let display = diary.collaborator.trim();
        if display.is_empty() {
            continue;
        }
        let key = collaborator_key(display);
        display_by_key
            .entry(key.clone())
            .or_insert_with(|| display.to_string());
        diaries_by_key.entry(key).or_default().push(diary);
    }
    for status in sync_statuses {
        if let Some(display) = sync_author_label(status) {
            display_by_key
                .entry(collaborator_key(&display))
                .or_insert(display);
        }
    }
    for entity in kg_entities {
        if let Some(display) = collaborator_from_agent_entity(&entity.entity) {
            display_by_key
                .entry(collaborator_key(&display))
                .or_insert(display);
        }
    }

    let mut collaborators = Vec::new();
    let mut timeline = Vec::new();

    for (key, display) in display_by_key {
        let diary_records = diaries_by_key.remove(&key).unwrap_or_default();
        let matched_statuses: Vec<&SyncTransportStatus> = sync_statuses
            .iter()
            .filter(|status| collaborator_matches_sync(&key, &display, status))
            .collect();
        let entity_name = agent_entity_name(&display);
        let agent_kg = kg_by_entity.get(&entity_name).copied();

        if let Some(summary) = build_collaborator_summary(
            &display,
            &entity_name,
            &diary_records,
            &matched_statuses,
            agent_kg,
            &kg_by_entity,
            &module_lookup,
            reasoning,
        ) {
            timeline.extend(diary_timeline_events(
                &display,
                &diary_records,
                &module_lookup,
            ));
            timeline.extend(sync_timeline_events(&display, &matched_statuses));
            if diary_records.is_empty() {
                if let Some(agent_kg) = agent_kg {
                    timeline.extend(agent_fact_timeline_events(&display, agent_kg));
                }
            }
            collaborators.push(summary);
        }
    }

    collaborators.sort_by(|left, right| {
        right
            .attention_score
            .total_cmp(&left.attention_score)
            .then_with(|| right.last_updated.cmp(&left.last_updated))
            .then_with(|| left.collaborator.cmp(&right.collaborator))
    });

    let modules = build_module_states(
        &collaborators,
        diaries,
        sync_statuses,
        &kg_by_entity,
        reasoning,
        &module_lookup,
    );

    CollaborationStateProjection {
        collaborators,
        modules,
        timeline: merge_collaboration_timeline(timeline),
    }
}

#[derive(Default)]
struct ModuleAccumulator {
    collaborators: BTreeSet<String>,
    focuses: BTreeSet<String>,
    blockers: BTreeSet<String>,
    related_entities: BTreeSet<String>,
    last_updated: Option<String>,
    latest_sync_edit_id: Option<String>,
    evidence: CollaborationEvidenceSummary,
    has_next_step: bool,
    trust_score_total: f32,
    trust_score_count: usize,
}

pub(super) fn build_collaborator_summary(
    display: &str,
    entity_name: &str,
    diary_records: &[&CollaborationDiaryRecord],
    sync_statuses: &[&SyncTransportStatus],
    agent_kg: Option<&KgEntity>,
    kg_by_entity: &HashMap<String, &KgEntity>,
    module_lookup: &BTreeMap<String, String>,
    reasoning: Option<&ReasoningReport>,
) -> Option<CollaboratorSummary> {
    let latest_diary = latest_diary_record(diary_records);
    let focus = latest_diary
        .and_then(|record| record.entry.title.clone())
        .or_else(|| agent_kg.and_then(|entity| entity.latest_active_value(AGENT_FOCUS_PREDICATE)));
    let status = latest_diary
        .and_then(|record| record.entry.status.clone())
        .or_else(|| agent_kg.and_then(|entity| entity.latest_active_value(AGENT_STATUS_PREDICATE)));
    let goal = latest_diary
        .and_then(|record| record.entry.goal.clone())
        .or_else(|| agent_kg.and_then(|entity| entity.latest_active_value(AGENT_GOAL_PREDICATE)));
    let next_step = latest_diary
        .and_then(|record| record.entry.next_step.clone())
        .or_else(|| {
            agent_kg.and_then(|entity| entity.latest_active_value(AGENT_NEXT_STEP_PREDICATE))
        });
    let blocker = latest_diary
        .and_then(|record| record.entry.blocker.clone())
        .or_else(|| {
            agent_kg.and_then(|entity| entity.latest_active_value(AGENT_BLOCKER_PREDICATE))
        });
    let outcome = latest_diary
        .and_then(|record| record.entry.outcome.clone())
        .or_else(|| {
            agent_kg.and_then(|entity| entity.latest_active_value(AGENT_OUTCOME_PREDICATE))
        });
    let action = latest_diary
        .and_then(|record| record.entry.action.clone())
        .or_else(|| agent_kg.and_then(|entity| entity.latest_active_value(AGENT_ACTION_PREDICATE)));

    let mut related_entities = latest_diary
        .map(|record| normalize_values(&record.entry.entities))
        .unwrap_or_default();
    if let Some(entity) = agent_kg {
        merge_unique(
            &mut related_entities,
            entity.active_value_strings(AGENT_RELATED_ENTITY_PREDICATE),
        );
    }

    let mut depends_on = latest_diary
        .map(|record| normalize_values(&record.entry.depends_on))
        .unwrap_or_default();
    if let Some(entity) = agent_kg {
        merge_unique(
            &mut depends_on,
            entity.active_value_strings(AGENT_DEPENDS_ON_PREDICATE),
        );
    }

    let mut touched_modules = BTreeSet::new();
    for status in sync_statuses {
        if let Some(module) = status
            .module()
            .filter(|module| is_collaboration_module(module))
        {
            touched_modules.insert(module.to_string());
        }
    }
    for value in related_entities.iter().chain(depends_on.iter()) {
        if let Some(module) = module_lookup.get(&normalized_label(value)) {
            touched_modules.insert(module.clone());
        }
    }
    let touched_modules: Vec<String> = touched_modules.into_iter().collect();

    let reasoning_node_count = reasoning_nodes_for_modules(reasoning, &touched_modules);
    let latest_sync = sync_statuses.iter().copied().max_by(|left, right| {
        left.latest_activity_at()
            .cmp(&right.latest_activity_at())
            .then_with(|| left.latest_edit_id().cmp(&right.latest_edit_id()))
    });
    let mut probe_entities = related_entities.clone();
    merge_unique(&mut probe_entities, depends_on.clone());
    merge_unique(&mut probe_entities, touched_modules.clone());
    let (supporting_facts, reasoning_fact_count, kg_fact_count) =
        collect_supporting_facts(agent_kg, &probe_entities, kg_by_entity, reasoning);
    let trust_score = aggregate_trust_score(sync_statuses.iter().copied());

    let evidence = CollaborationEvidenceSummary {
        diary_entries: diary_records.len(),
        kg_fact_count,
        sync_neuron_count: sync_statuses.len(),
        verified_sync_count: sync_statuses
            .iter()
            .filter(|status| status.fully_verified())
            .count(),
        pending_sync_count: sync_statuses
            .iter()
            .map(|status| status.pending_outgoing() as usize + status.pending_incoming() as usize)
            .sum(),
        conflict_count: sync_statuses
            .iter()
            .map(|status| status.conflict_count())
            .sum(),
        integrity_issue_count: sync_statuses
            .iter()
            .map(|status| status.integrity_issue_count())
            .sum(),
        untrusted_handoff_count: sync_statuses
            .iter()
            .filter(|status| status.requires_trust_attention())
            .count(),
        reasoning_node_count,
        reasoning_fact_count,
    };
    let (attention, attention_score) = collaboration_attention_score(
        status.as_deref(),
        blocker.as_deref(),
        next_step.is_some(),
        evidence.pending_sync_count,
        evidence.conflict_count,
        depends_on.len(),
        evidence.integrity_issue_count,
        evidence.untrusted_handoff_count,
    );
    let last_updated = max_optional_strings(
        [
            latest_diary.and_then(|record| record.when.clone()),
            agent_kg.and_then(|entity| entity.latest_active_timestamp()),
            latest_sync.and_then(|status| status.latest_activity_at().map(str::to_string)),
        ],
    );

    if focus.is_none()
        && status.is_none()
        && goal.is_none()
        && next_step.is_none()
        && blocker.is_none()
        && outcome.is_none()
        && action.is_none()
        && related_entities.is_empty()
        && depends_on.is_empty()
        && touched_modules.is_empty()
        && evidence.diary_entries == 0
        && evidence.kg_fact_count == 0
        && evidence.sync_neuron_count == 0
    {
        return None;
    }

    Some(CollaboratorSummary {
        collaborator: display.to_string(),
        entity: entity_name.to_string(),
        focus,
        status,
        goal,
        next_step,
        blocker,
        outcome,
        action,
        related_entities,
        depends_on,
        touched_modules,
        supporting_facts,
        last_updated,
        latest_sync_edit_id: latest_sync
            .and_then(|status| status.latest_edit_id().map(str::to_string)),
        pending_sync: evidence.pending_sync_count > 0,
        attention,
        attention_score,
        trust_score,
        evidence_score: collaboration_evidence_score(&evidence),
        evidence,
    })
}

pub(super) fn build_module_states(
    collaborators: &[CollaboratorSummary],
    diaries: &[CollaborationDiaryRecord],
    sync_statuses: &[SyncTransportStatus],
    kg_by_entity: &HashMap<String, &KgEntity>,
    reasoning: Option<&ReasoningReport>,
    module_lookup: &BTreeMap<String, String>,
) -> Vec<ModuleCollaborationState> {
    let mut modules: BTreeMap<String, ModuleAccumulator> = BTreeMap::new();

    for collaborator in collaborators {
        for module in &collaborator.touched_modules {
            let entry = modules.entry(module.clone()).or_default();
            entry
                .collaborators
                .insert(collaborator.collaborator.clone());
            if let Some(focus) = &collaborator.focus {
                entry.focuses.insert(focus.clone());
            }
            if let Some(blocker) = &collaborator.blocker {
                entry.blockers.insert(blocker.clone());
            }
            entry
                .related_entities
                .extend(collaborator.related_entities.iter().cloned());
            entry.has_next_step |= collaborator.next_step.is_some();
            entry.last_updated =
                max_optional_pair(entry.last_updated.take(), collaborator.last_updated.clone());
        }
    }

    for diary in diaries {
        for module in modules_for_diary_entry(&diary.entry, module_lookup) {
            let entry = modules.entry(module).or_default();
            entry.collaborators.insert(diary.collaborator.clone());
            entry.evidence.diary_entries += 1;
            if let Some(title) = &diary.entry.title {
                entry.focuses.insert(title.clone());
            }
            if let Some(blocker) = &diary.entry.blocker {
                entry.blockers.insert(blocker.clone());
            }
            entry
                .related_entities
                .extend(normalize_values(&diary.entry.entities));
            entry.has_next_step |= diary.entry.next_step.is_some();
            entry.last_updated = max_optional_pair(entry.last_updated.take(), diary.when.clone());
        }
    }

    for status in sync_statuses {
        let Some(module) = status
            .module()
            .filter(|module| is_collaboration_module(module))
        else {
            continue;
        };
        let entry = modules.entry(module.to_string()).or_default();
        entry.evidence.sync_neuron_count += 1;
        entry.evidence.verified_sync_count += status.fully_verified() as usize;
        entry.evidence.pending_sync_count +=
            status.pending_outgoing() as usize + status.pending_incoming() as usize;
        entry.evidence.conflict_count += status.conflict_count();
        entry.evidence.integrity_issue_count += status.integrity_issue_count();
        entry.evidence.untrusted_handoff_count += status.requires_trust_attention() as usize;
        entry.trust_score_total += status.trust_score();
        entry.trust_score_count += 1;
        entry.last_updated = max_optional_pair(
            entry.last_updated.take(),
            status.latest_activity_at().map(str::to_string),
        );
        if let Some(edit_id) = status.latest_edit_id() {
            entry.latest_sync_edit_id = Some(edit_id.to_string());
        }
    }

    let mut projected = Vec::new();
    for (module, mut entry) in modules {
        let mut entity_names = vec![module.clone()];
        merge_unique(&mut entity_names, entry.related_entities.iter().cloned());
        let (supporting_facts, reasoning_fact_count, kg_fact_count) =
            collect_supporting_facts(None, &entity_names, kg_by_entity, reasoning);
        entry.evidence.kg_fact_count += kg_fact_count;
        entry.evidence.reasoning_fact_count += reasoning_fact_count;
        entry.evidence.reasoning_node_count +=
            reasoning_nodes_for_modules(reasoning, std::slice::from_ref(&module));

        let blockers: Vec<String> = entry.blockers.into_iter().collect();
        let (attention, attention_score) = collaboration_attention_score(
            None,
            blockers.first().map(String::as_str),
            entry.has_next_step,
            entry.evidence.pending_sync_count,
            entry.evidence.conflict_count,
            0,
            entry.evidence.integrity_issue_count,
            entry.evidence.untrusted_handoff_count,
        );
        let evidence_score = collaboration_evidence_score(&entry.evidence);
        let trust_score = (entry.trust_score_count > 0)
            .then_some(entry.trust_score_total / entry.trust_score_count as f32);

        projected.push(ModuleCollaborationState {
            module,
            collaborators: entry.collaborators.into_iter().collect(),
            focuses: entry.focuses.into_iter().collect(),
            blockers,
            related_entities: entry.related_entities.into_iter().collect(),
            supporting_facts,
            last_updated: entry.last_updated,
            latest_sync_edit_id: entry.latest_sync_edit_id,
            pending_sync: entry.evidence.pending_sync_count > 0,
            attention,
            attention_score,
            trust_score,
            evidence_score,
            evidence: entry.evidence,
        });
    }

    projected.sort_by(|left, right| {
        right
            .attention_score
            .total_cmp(&left.attention_score)
            .then_with(|| right.last_updated.cmp(&left.last_updated))
            .then_with(|| left.module.cmp(&right.module))
    });
    projected
}

pub(super) fn build_module_lookup(
    sync_statuses: &[SyncTransportStatus],
    reasoning: Option<&ReasoningReport>,
) -> BTreeMap<String, String> {
    let mut lookup = BTreeMap::new();
    for status in sync_statuses {
        if let Some(module) = status
            .module()
            .filter(|module| is_collaboration_module(module))
        {
            lookup
                .entry(normalized_label(module))
                .or_insert_with(|| module.to_string());
        }
    }
    if let Some(reasoning) = reasoning {
        for node in &reasoning.nodes {
            if let Some(module) = node
                .module
                .as_deref()
                .filter(|module| is_collaboration_module(module))
            {
                lookup
                    .entry(normalized_label(module))
                    .or_insert_with(|| module.to_string());
            }
        }
    }
    lookup
}

fn reasoning_nodes_for_modules(reasoning: Option<&ReasoningReport>, modules: &[String]) -> usize {
    let module_keys: BTreeSet<String> = modules
        .iter()
        .map(|module| normalized_label(module))
        .collect();
    reasoning
        .map(|report| {
            report
                .nodes
                .iter()
                .filter(|node| {
                    node.module
                        .as_deref()
                        .map(normalized_label)
                        .map(|module| module_keys.contains(&module))
                        .unwrap_or(false)
                })
                .count()
        })
        .unwrap_or(0)
}
