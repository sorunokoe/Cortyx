use std::path::PathBuf;

use super::*;
use crate::kg::KgEntity;
use crate::kg::KgFact;
use crate::neuron::provenance::ProvenanceIntegritySummary;
use crate::neuron::NeuronKind;
use crate::reasoner::{ReasonedFact, ReasonedNode, ReasoningReport};
use crate::sync_transport::{
    SyncHandoffIssue, SyncHandoffState, SyncHandoffSummary, SyncRevisionState, SyncTransportStatus,
};

fn diary_entry() -> StructuredDiaryEntry {
    StructuredDiaryEntry {
        agent: Some("reviewer".to_string()),
        title: Some("Audit auth middleware".to_string()),
        status: Some("blocked".to_string()),
        goal: Some("Close the legacy auth bypass.".to_string()),
        next_step: Some("Wait for api-owner approval.".to_string()),
        blocker: Some("Waiting on api-owner.".to_string()),
        outcome: None,
        entities: vec!["auth".to_string(), "engine".to_string()],
        depends_on: vec!["api-owner".to_string()],
        action: None,
        refined_plan: None,
    }
}

fn agent_kg(entity: &str) -> KgEntity {
    KgEntity {
        entity: entity.to_string(),
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
        path: PathBuf::from(format!(".cortyx/neurons/_kg_{entity}.context.md")),
    }
}

fn knowledge_kg(entity: &str, predicate: &str, value: &str) -> KgEntity {
    KgEntity {
        entity: entity.to_string(),
        facts: vec![KgFact {
            predicate: predicate.to_string(),
            value: value.to_string(),
            valid_from: "2026-04-17T10:03:00Z".to_string(),
            ended: String::new(),
        }],
        path: PathBuf::from(format!(".cortyx/neurons/_kg_{entity}.context.md")),
    }
}

fn sync_status() -> SyncTransportStatus {
    let trusted_integrity = |fingerprint: &str| ProvenanceIntegritySummary {
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
    SyncTransportStatus {
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
    }
}

fn reasoning_report() -> ReasoningReport {
    ReasoningReport {
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
    }
}

#[test]
fn collaboration_projection_merges_diary_sync_and_reasoning_signals() {
    let mut diary = CollaborationDiaryRecord::new("reviewer", diary_entry());
    diary.when = Some("2026-04-17T10:04:00Z".to_string());
    let projection = project_collaboration_state(
        &[diary],
        &[sync_status()],
        &[
            agent_kg(&agent_entity_name("reviewer")),
            knowledge_kg("auth", "owner", "platform-team"),
        ],
        Some(&reasoning_report()),
    );

    assert_eq!(projection.collaborators.len(), 1);
    let reviewer = &projection.collaborators[0];
    assert_eq!(reviewer.collaborator, "reviewer");
    assert_eq!(reviewer.focus.as_deref(), Some("Audit auth middleware"));
    assert_eq!(
        reviewer.action.as_deref(),
        Some("Investigating auth middleware.")
    );
    assert_eq!(reviewer.outcome.as_deref(), Some("Found a legacy bypass."));
    assert_eq!(reviewer.touched_modules, vec!["engine".to_string()]);
    assert!(reviewer.pending_sync);
    assert_eq!(reviewer.attention, CollaborationAttention::SyncConflict);
    assert_eq!(reviewer.evidence.diary_entries, 1);
    assert_eq!(reviewer.evidence.sync_neuron_count, 1);
    assert_eq!(reviewer.evidence.conflict_count, 1);
    assert_eq!(reviewer.evidence.integrity_issue_count, 0);
    assert_eq!(reviewer.evidence.untrusted_handoff_count, 1);
    assert_eq!(reviewer.evidence.reasoning_fact_count, 1);
    assert_eq!(reviewer.evidence.reasoning_node_count, 1);
    assert!(reviewer.trust_score.unwrap() > 80.0);
    assert!(reviewer
        .supporting_facts
        .iter()
        .any(|fact| fact.contains("auth.owner = platform-team")));

    let module = projection
        .modules
        .iter()
        .find(|module| module.module == "engine")
        .expect("engine module state");
    assert_eq!(module.collaborators, vec!["reviewer".to_string()]);
    assert!(module.pending_sync);
    assert_eq!(module.attention, CollaborationAttention::SyncConflict);
    assert!(module.trust_score.unwrap() > 80.0);
    assert!(projection
        .timeline
        .iter()
        .any(|event| event.kind == CollaborationTimelineKind::Diary));
    assert!(projection.timeline.iter().any(|event| {
        event.kind == CollaborationTimelineKind::Conflict && event.summary.contains("since edit-1")
    }));
}

#[test]
fn collaboration_projection_falls_back_to_agent_kg_without_diary_entries() {
    let agent = KgEntity {
        entity: agent_entity_name("planner"),
        facts: vec![
            KgFact {
                predicate: AGENT_FOCUS_PREDICATE.to_string(),
                value: "Design sync rollout".to_string(),
                valid_from: "2026-04-17T09:00:00Z".to_string(),
                ended: String::new(),
            },
            KgFact {
                predicate: AGENT_STATUS_PREDICATE.to_string(),
                value: "in_progress".to_string(),
                valid_from: "2026-04-17T09:05:00Z".to_string(),
                ended: String::new(),
            },
            KgFact {
                predicate: AGENT_RELATED_ENTITY_PREDICATE.to_string(),
                value: "sync".to_string(),
                valid_from: "2026-04-17T09:10:00Z".to_string(),
                ended: String::new(),
            },
        ],
        path: PathBuf::from(".cortyx/neurons/_kg_agent_planner.context.md"),
    };

    let projection = project_collaboration_state(&[], &[], &[agent], None);

    assert_eq!(projection.collaborators.len(), 1);
    let planner = &projection.collaborators[0];
    assert_eq!(planner.collaborator, "planner");
    assert_eq!(planner.focus.as_deref(), Some("Design sync rollout"));
    assert_eq!(planner.status.as_deref(), Some("in_progress"));
    assert_eq!(planner.related_entities, vec!["sync".to_string()]);
    assert!(planner.trust_score.is_none());
    assert!(projection
        .timeline
        .iter()
        .any(|event| event.kind == CollaborationTimelineKind::Fact));
}

#[test]
fn merge_collaboration_timeline_deduplicates_and_orders() {
    let later = CollaborationTimelineEvent {
        kind: CollaborationTimelineKind::Diary,
        at: Some("2026-04-17T10:00:00Z".to_string()),
        collaborator: Some("reviewer".to_string()),
        module: Some("engine".to_string()),
        summary: "latest update".to_string(),
        related_entities: vec!["engine".to_string()],
        attention: CollaborationAttention::NeedsFollowUp,
        attention_score: 2.0,
    };
    let earlier = CollaborationTimelineEvent {
        kind: CollaborationTimelineKind::Fact,
        at: Some("2026-04-17T09:00:00Z".to_string()),
        collaborator: Some("reviewer".to_string()),
        module: None,
        summary: "status: in_progress".to_string(),
        related_entities: Vec::new(),
        attention: CollaborationAttention::Nominal,
        attention_score: 0.5,
    };

    let merged = merge_collaboration_timeline(vec![earlier.clone(), later.clone(), later]);

    assert_eq!(merged.len(), 2);
    assert_eq!(merged[0].summary, "latest update");
    assert_eq!(merged[1], earlier);
}
