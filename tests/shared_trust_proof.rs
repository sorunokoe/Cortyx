use cortyx::agent_memory::StructuredDiaryEntry;
use cortyx::collaboration_kernel::{
    compare_shared_trust_outcomes, project_collaboration_state, CollaborationDiaryRecord,
};
use cortyx::neuron::provenance::{
    provenance_content_hash, AuthorshipRecord, NeuronProvenance, ProvenanceAuthor,
    ProvenanceEditRecord, ProvenanceOperation, ProvenanceSource,
};
use cortyx::neuron::{NeuronKind, NeuronMeta, NeuronStatus};
use cortyx::sync_transport::{
    summarize_sync_trust, SyncPullStatus, SyncResolutionStrategy, SyncTransportEnvelope,
    SyncTransportRepository,
};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static TEST_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

struct ProofDir {
    path: PathBuf,
}

impl ProofDir {
    fn new(name: &str) -> Self {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("test-artifacts")
            .join(format!(
                "{name}-{}-{}",
                std::process::id(),
                TEST_DIR_COUNTER.fetch_add(1, Ordering::Relaxed)
            ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(path.join("src")).expect("create proof dir");
        fs::write(
            path.join("src/engine.rs"),
            "pub fn engine() -> &'static str { \"engine\" }\n",
        )
        .expect("write proof source");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ProofDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

struct EditSpec<'a> {
    id: &'a str,
    parent: Option<&'a str>,
    operation: ProvenanceOperation,
    source: ProvenanceSource,
    edited_at: &'a str,
    summary: &'a str,
    body: &'a str,
    tampered_content_hash: Option<&'a str>,
}

fn reviewer() -> ProvenanceAuthor {
    ProvenanceAuthor {
        author_id: "agent:reviewer".to_string(),
        display_name: Some("Reviewer".to_string()),
        device_id: Some("proof-rig".to_string()),
    }
}

fn sync_meta(project_root: &Path) -> NeuronMeta {
    let mut meta = NeuronMeta::new_stub(&project_root.join("src/engine.rs"), NeuronKind::Core);
    meta.status = NeuronStatus::Fresh;
    meta.source_hash = "src-hash-engine".to_string();
    meta.sig_hash = Some("sig-hash-engine".to_string());
    meta.last_updated = "2026-04-17T10:00:00Z".to_string();
    meta.module = Some("engine".to_string());
    meta.uuid = Some("uuid-1234".to_string());
    meta
}

fn envelope_with_history(
    meta: &NeuronMeta,
    current_body: &str,
    specs: &[EditSpec<'_>],
) -> SyncTransportEnvelope {
    let author = reviewer();
    let provenance = NeuronProvenance {
        version: 1,
        neuron_uuid: meta.uuid.clone(),
        source_path: Some(meta.source_path.clone()),
        authorship: Some(AuthorshipRecord {
            created_by: author.clone(),
            created_at: specs
                .first()
                .map(|spec| spec.edited_at.to_string())
                .unwrap_or_default(),
        }),
        edit_history: specs
            .iter()
            .map(|spec| ProvenanceEditRecord {
                edit_id: spec.id.to_string(),
                parent_edit_id: spec.parent.map(str::to_string),
                operation: spec.operation.clone(),
                source: spec.source.clone(),
                edited_at: spec.edited_at.to_string(),
                author: Some(author.clone()),
                section: Some("body".to_string()),
                summary: Some(spec.summary.to_string()),
                content_hash: Some(
                    spec.tampered_content_hash
                        .map(str::to_string)
                        .unwrap_or_else(|| provenance_content_hash(spec.body)),
                ),
            })
            .collect(),
    };
    SyncTransportEnvelope::from_parts(meta, current_body, Some(&provenance))
        .expect("proof envelope should be syncable")
}

fn blocked_diary() -> CollaborationDiaryRecord {
    let mut record = CollaborationDiaryRecord::new(
        "reviewer",
        StructuredDiaryEntry {
            agent: Some("reviewer".to_string()),
            title: Some("Resolve engine shared-trust handoff".to_string()),
            status: Some("blocked".to_string()),
            goal: Some("Keep engine memory verified across sync boundaries.".to_string()),
            next_step: Some("Resolve the conflicting sync handoff.".to_string()),
            blocker: Some("Conflicting local and remote shared-memory edits.".to_string()),
            outcome: None,
            entities: vec!["engine".to_string()],
            depends_on: Vec::new(),
            action: Some("Inspect shared lineage and decide a resolution.".to_string()),
            refined_plan: None,
        },
    );
    record.when = Some("2026-04-17T10:03:00Z".to_string());
    record
}

fn resolved_diary() -> CollaborationDiaryRecord {
    let mut record = CollaborationDiaryRecord::new(
        "reviewer",
        StructuredDiaryEntry {
            agent: Some("reviewer".to_string()),
            title: Some("Resolve engine shared-trust handoff".to_string()),
            status: Some("done".to_string()),
            goal: Some("Keep engine memory verified across sync boundaries.".to_string()),
            next_step: None,
            blocker: None,
            outcome: Some("Resolved the handoff and preserved trusted lineage.".to_string()),
            entities: vec!["engine".to_string()],
            depends_on: Vec::new(),
            action: Some(
                "Promote the verified resolution and wait for acknowledgement.".to_string(),
            ),
            refined_plan: None,
        },
    );
    record.when = Some("2026-04-17T10:07:00Z".to_string());
    record
}

fn seed_conflict(
    repo: &SyncTransportRepository,
    meta: &NeuronMeta,
) -> (
    SyncTransportEnvelope,
    SyncTransportEnvelope,
    SyncTransportEnvelope,
) {
    let base_body = "shared memory v1";
    let local_body = "local reviewer refinement";
    let remote_body = "remote reviewer refinement";
    let base = envelope_with_history(
        meta,
        base_body,
        &[EditSpec {
            id: "edit-1",
            parent: None,
            operation: ProvenanceOperation::Create,
            source: ProvenanceSource::Local,
            edited_at: "2026-04-17T10:00:00Z",
            summary: "bootstrap shared memory",
            body: base_body,
            tampered_content_hash: None,
        }],
    );
    let local = envelope_with_history(
        meta,
        local_body,
        &[
            EditSpec {
                id: "edit-1",
                parent: None,
                operation: ProvenanceOperation::Create,
                source: ProvenanceSource::Local,
                edited_at: "2026-04-17T10:00:00Z",
                summary: "bootstrap shared memory",
                body: base_body,
                tampered_content_hash: None,
            },
            EditSpec {
                id: "edit-2",
                parent: Some("edit-1"),
                operation: ProvenanceOperation::Update,
                source: ProvenanceSource::Local,
                edited_at: "2026-04-17T10:01:00Z",
                summary: "local reviewer refinement",
                body: local_body,
                tampered_content_hash: None,
            },
        ],
    );
    let remote = envelope_with_history(
        meta,
        remote_body,
        &[
            EditSpec {
                id: "edit-1",
                parent: None,
                operation: ProvenanceOperation::Create,
                source: ProvenanceSource::Local,
                edited_at: "2026-04-17T10:00:00Z",
                summary: "bootstrap shared memory",
                body: base_body,
                tampered_content_hash: None,
            },
            EditSpec {
                id: "edit-3",
                parent: Some("edit-1"),
                operation: ProvenanceOperation::Update,
                source: ProvenanceSource::Sync,
                edited_at: "2026-04-17T10:02:00Z",
                summary: "remote reviewer refinement",
                body: remote_body,
                tampered_content_hash: None,
            },
        ],
    );

    repo.apply_remote(&base).expect("apply base");
    repo.stage_local(&local).expect("stage local");
    repo.apply_remote(&remote).expect("apply remote conflict");
    (base, local, remote)
}

fn resolved_envelope(meta: &NeuronMeta) -> SyncTransportEnvelope {
    let base_body = "shared memory v1";
    let local_body = "local reviewer refinement";
    let resolved_body = "resolved shared memory with verified lineage";
    envelope_with_history(
        meta,
        resolved_body,
        &[
            EditSpec {
                id: "edit-1",
                parent: None,
                operation: ProvenanceOperation::Create,
                source: ProvenanceSource::Local,
                edited_at: "2026-04-17T10:00:00Z",
                summary: "bootstrap shared memory",
                body: base_body,
                tampered_content_hash: None,
            },
            EditSpec {
                id: "edit-2",
                parent: Some("edit-1"),
                operation: ProvenanceOperation::Update,
                source: ProvenanceSource::Local,
                edited_at: "2026-04-17T10:01:00Z",
                summary: "local reviewer refinement",
                body: local_body,
                tampered_content_hash: None,
            },
            EditSpec {
                id: "edit-4",
                parent: Some("edit-2"),
                operation: ProvenanceOperation::Merge,
                source: ProvenanceSource::Local,
                edited_at: "2026-04-17T10:04:00Z",
                summary: "resolve shared handoff",
                body: resolved_body,
                tampered_content_hash: None,
            },
        ],
    )
}

fn tampered_resolution_envelope(meta: &NeuronMeta) -> SyncTransportEnvelope {
    let base_body = "shared memory v1";
    let local_body = "local reviewer refinement";
    let resolved_body = "resolved shared memory with verified lineage";
    envelope_with_history(
        meta,
        resolved_body,
        &[
            EditSpec {
                id: "edit-1",
                parent: None,
                operation: ProvenanceOperation::Create,
                source: ProvenanceSource::Local,
                edited_at: "2026-04-17T10:00:00Z",
                summary: "bootstrap shared memory",
                body: base_body,
                tampered_content_hash: None,
            },
            EditSpec {
                id: "edit-2",
                parent: Some("edit-1"),
                operation: ProvenanceOperation::Update,
                source: ProvenanceSource::Local,
                edited_at: "2026-04-17T10:01:00Z",
                summary: "local reviewer refinement",
                body: local_body,
                tampered_content_hash: None,
            },
            EditSpec {
                id: "edit-4",
                parent: Some("edit-2"),
                operation: ProvenanceOperation::Merge,
                source: ProvenanceSource::Local,
                edited_at: "2026-04-17T10:04:00Z",
                summary: "resolve shared handoff",
                body: resolved_body,
                tampered_content_hash: Some("tampered-content-hash"),
            },
        ],
    )
}

#[test]
fn shared_trust_proof_harness_proves_resolution_improves_workflow_and_integrity() {
    let dir = ProofDir::new("shared-trust-proof-success");
    let repo = SyncTransportRepository::for_project(dir.path());
    let meta = sync_meta(dir.path());
    seed_conflict(&repo, &meta);

    let baseline_statuses = repo.list_statuses().expect("baseline statuses");
    let baseline_metrics = summarize_sync_trust(&baseline_statuses);
    assert_eq!(baseline_metrics.conflict_count, 1);
    assert_eq!(baseline_metrics.fully_verified_neuron_count, 0);

    let baseline_projection =
        project_collaboration_state(&[blocked_diary()], &baseline_statuses, &[], None);

    let resolved = resolved_envelope(&meta);
    let resolution = repo
        .resolve_handoff(&resolved, SyncResolutionStrategy::StageLocal)
        .expect("resolve handoff");
    assert_eq!(resolution.cleared_conflict_paths.len(), 1);
    assert!(repo
        .load_incoming("uuid-1234")
        .expect("load incoming")
        .is_none());

    let staged_status = repo
        .status_for("uuid-1234")
        .expect("staged status")
        .unwrap();
    assert_eq!(staged_status.conflict_count(), 0);
    assert!(staged_status.pending_outgoing());

    let ack = repo.apply_remote(&resolved).expect("ack resolved handoff");
    assert_eq!(ack.status, SyncPullStatus::AlreadyCurrent);

    let candidate_statuses = repo.list_statuses().expect("candidate statuses");
    let candidate_metrics = summarize_sync_trust(&candidate_statuses);
    assert_eq!(candidate_metrics.conflict_count, 0);
    assert_eq!(candidate_metrics.fully_verified_neuron_count, 1);
    assert_eq!(candidate_metrics.trust_attention_count, 0);

    let candidate_projection =
        project_collaboration_state(&[resolved_diary()], &candidate_statuses, &[], None);
    let report = compare_shared_trust_outcomes(
        &baseline_statuses,
        &baseline_projection,
        &candidate_statuses,
        &candidate_projection,
    );

    assert!(report.workflow_improved, "{report:#?}");
    assert!(report.trust_improved, "{report:#?}");
    assert_eq!(report.conflict_delta, -1);
    assert_eq!(report.fully_verified_delta, 1);
    assert_eq!(report.active_blocker_delta, -1);
    assert!(
        report.average_sync_trust_score_delta.unwrap_or_default() > 0.0,
        "{report:#?}"
    );
}

#[test]
fn shared_trust_proof_harness_rejects_tampered_resolution_integrity() {
    let dir = ProofDir::new("shared-trust-proof-tampered");
    let repo = SyncTransportRepository::for_project(dir.path());
    let meta = sync_meta(dir.path());
    seed_conflict(&repo, &meta);

    let baseline_statuses = repo.list_statuses().expect("baseline statuses");
    let baseline_projection =
        project_collaboration_state(&[blocked_diary()], &baseline_statuses, &[], None);

    let tampered = tampered_resolution_envelope(&meta);
    repo.resolve_handoff(&tampered, SyncResolutionStrategy::StageLocal)
        .expect("stage tampered resolution");
    repo.apply_remote(&tampered)
        .expect("ack tampered resolution");

    let candidate_statuses = repo.list_statuses().expect("candidate statuses");
    let candidate_projection =
        project_collaboration_state(&[resolved_diary()], &candidate_statuses, &[], None);
    let report = compare_shared_trust_outcomes(
        &baseline_statuses,
        &baseline_projection,
        &candidate_statuses,
        &candidate_projection,
    );

    assert!(!report.trust_improved, "{report:#?}");
    assert!(
        report.candidate_sync.integrity_issue_count > report.baseline_sync.integrity_issue_count
    );
    assert!(
        report.average_sync_trust_score_delta.unwrap_or_default() < 0.0,
        "{report:#?}"
    );
}
