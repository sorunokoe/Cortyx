use super::*;
use crate::neuron::{
    provenance::{
        AuthorshipRecord, NeuronProvenance, ProvenanceAuthor, ProvenanceEditRecord,
        ProvenanceOperation, ProvenanceSource,
    },
    sync::{hash_sync_body, SyncConflictKind},
    NeuronKind, NeuronMeta, NeuronStatus, Synapse, SynapseType,
};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static TEST_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

struct TestDir {
    path: PathBuf,
}

impl TestDir {
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
        fs::create_dir_all(&path).expect("create test dir");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn sync_meta(project_root: &Path) -> NeuronMeta {
    let mut meta = NeuronMeta::new_stub(&project_root.join("src/engine.rs"), NeuronKind::Core);
    meta.status = NeuronStatus::Fresh;
    meta.source_hash = "src-hash-1".to_string();
    meta.sig_hash = Some("sig-hash-1".to_string());
    meta.last_updated = "2026-01-02T03:04:05Z".to_string();
    meta.module = Some("engine".to_string());
    meta.uuid = Some("uuid-1234".to_string());
    meta.source_files = vec![project_root.join("src/a.rs"), project_root.join("src/b.rs")];
    meta.synapses = vec![
        test_synapse(
            ".cortyx/neurons/src/b.context.md",
            SynapseType::Imports,
            0.8,
            "imports cache",
        ),
        test_synapse(
            ".cortyx/neurons/src/a.context.md",
            SynapseType::Calls,
            0.6,
            "calls parser",
        ),
    ];
    meta
}

fn test_synapse(target: &str, edge_type: SynapseType, weight: f32, reason: &str) -> Synapse {
    Synapse {
        target: PathBuf::from(target),
        edge_type,
        weight,
        reason: reason.to_string(),
        learned_weight: 0.9,
        traversal_count: 12,
        last_co_activation_day: 77,
    }
}

fn test_provenance(
    latest_edit_id: &str,
    parent_edit_id: Option<&str>,
    body: &str,
) -> NeuronProvenance {
    let author = ProvenanceAuthor {
        author_id: "agent:reviewer".to_string(),
        display_name: Some("Reviewer".to_string()),
        device_id: Some("macbook".to_string()),
    };
    NeuronProvenance {
        version: 1,
        neuron_uuid: Some("uuid-1234".to_string()),
        source_path: Some(PathBuf::from("src/engine.rs")),
        authorship: Some(AuthorshipRecord {
            created_by: author.clone(),
            created_at: "2026-01-02T03:04:05Z".to_string(),
        }),
        edit_history: vec![ProvenanceEditRecord {
            edit_id: latest_edit_id.to_string(),
            parent_edit_id: parent_edit_id.map(str::to_string),
            operation: ProvenanceOperation::Update,
            source: ProvenanceSource::Sync,
            edited_at: "2026-01-02T03:05:05Z".to_string(),
            author: Some(author),
            section: Some("purpose".to_string()),
            summary: Some("sync update".to_string()),
            content_hash: Some(hash_sync_body(body)),
        }],
    }
}

fn envelope(
    project_root: &Path,
    body: &str,
    latest_edit_id: &str,
    parent_edit_id: Option<&str>,
) -> SyncTransportEnvelope {
    let meta = sync_meta(project_root);
    let provenance = test_provenance(latest_edit_id, parent_edit_id, body);
    SyncTransportEnvelope::from_parts(&meta, body, Some(&provenance))
        .expect("syncable transport envelope")
}

#[test]
fn sync_transport_layout_buckets_records_under_cortyx_sync() {
    let dir = TestDir::new("sync-transport-layout");
    let repo = SyncTransportRepository::for_project(dir.path());

    repo.ensure_layout().unwrap();

    assert_eq!(
        sync_transport_dir(dir.path()),
        dir.path().join(".cortyx").join("sync")
    );
    assert_eq!(
        repo.layout.snapshot_path("uuid-1234"),
        dir.path()
            .join(".cortyx")
            .join("sync")
            .join("snapshots")
            .join("uu")
            .join("uuid-1234.json")
    );
    assert!(repo.layout.outgoing_dir.exists());
    assert!(repo.layout.incoming_dir.exists());
    assert!(repo.layout.conflicts_dir.exists());
}

#[test]
fn sync_transport_envelope_round_trips_json_with_provenance() {
    let dir = TestDir::new("sync-transport-roundtrip");
    let envelope = envelope(
        dir.path(),
        "line one\r\nline two\r\n",
        "edit-2",
        Some("edit-1"),
    );

    let json = envelope.to_json_pretty().unwrap();
    let decoded = SyncTransportEnvelope::from_json_str(&json).unwrap();

    assert_eq!(decoded, envelope);
    assert_eq!(decoded.syncable.body, "line one\nline two");
    assert_eq!(
        decoded
            .provenance
            .as_ref()
            .and_then(|provenance| provenance.edit_history.last())
            .map(|edit| edit.edit_id.as_str()),
        Some("edit-2")
    );
}

#[test]
fn sync_transport_stages_local_fast_forward_and_clears_acknowledged_outbox() {
    let dir = TestDir::new("sync-transport-stage-local");
    let repo = SyncTransportRepository::for_project(dir.path());

    let v1 = envelope(dir.path(), "body v1", "edit-1", None);
    let queued = repo.stage_local(&v1).unwrap();
    assert_eq!(queued.status, SyncStageStatus::QueuedNew);
    assert_eq!(repo.load_snapshot("uuid-1234").unwrap().unwrap(), v1);
    assert_eq!(repo.load_outgoing("uuid-1234").unwrap().unwrap(), v1);

    let v2 = envelope(dir.path(), "body v2", "edit-2", Some("edit-1"));
    let fast_forwarded = repo.stage_local(&v2).unwrap();
    assert_eq!(fast_forwarded.status, SyncStageStatus::QueuedFastForward);
    assert_eq!(repo.load_snapshot("uuid-1234").unwrap().unwrap(), v2);
    assert_eq!(repo.load_outgoing("uuid-1234").unwrap().unwrap(), v2);

    let acknowledged = repo.apply_remote(&v2).unwrap();
    assert_eq!(acknowledged.status, SyncPullStatus::AlreadyCurrent);
    assert_eq!(repo.load_incoming("uuid-1234").unwrap().unwrap(), v2);
    assert!(repo.load_outgoing("uuid-1234").unwrap().is_none());
}

#[test]
fn sync_transport_applies_remote_fast_forward() {
    let dir = TestDir::new("sync-transport-remote-fast-forward");
    let repo = SyncTransportRepository::for_project(dir.path());

    let base = envelope(dir.path(), "body v1", "edit-1", None);
    let applied_new = repo.apply_remote(&base).unwrap();
    assert_eq!(applied_new.status, SyncPullStatus::AppliedNew);
    assert_eq!(repo.load_snapshot("uuid-1234").unwrap().unwrap(), base);

    let remote_ahead = envelope(dir.path(), "body v2", "edit-2", Some("edit-1"));
    let applied_fast_forward = repo.apply_remote(&remote_ahead).unwrap();

    assert_eq!(
        applied_fast_forward.status,
        SyncPullStatus::AppliedFastForward
    );
    assert_eq!(
        repo.load_snapshot("uuid-1234").unwrap().unwrap(),
        remote_ahead
    );
    assert_eq!(
        repo.load_incoming("uuid-1234").unwrap().unwrap(),
        remote_ahead
    );

    let status = repo.status_for("uuid-1234").unwrap().unwrap();
    assert!(!status.pending_incoming());
    assert_eq!(status.handoff.state, SyncHandoffState::Applied);
    assert!(status.handoff.integrity_verified);
    assert!(status.trust_score() >= 90.0);
}

#[test]
fn sync_transport_flags_identical_content_with_diverged_provenance_handoff() {
    let dir = TestDir::new("sync-transport-provenance-divergence");
    let repo = SyncTransportRepository::for_project(dir.path());

    let local = envelope(dir.path(), "same body", "edit-1", None);
    repo.apply_remote(&local).unwrap();

    let remote = envelope(dir.path(), "same body", "edit-2", Some("other-base"));
    let result = repo.apply_remote(&remote).unwrap();
    assert_eq!(result.status, SyncPullStatus::AlreadyCurrent);

    let status = repo.status_for("uuid-1234").unwrap().unwrap();
    assert!(!status.pending_incoming());
    assert_eq!(status.handoff.state, SyncHandoffState::Applied);
    assert!(!status.handoff.integrity_verified);
    assert!(status
        .handoff
        .issues
        .contains(&SyncHandoffIssue::ProvenanceDiverged));
    assert!(status.requires_trust_attention());
}

#[test]
fn sync_transport_records_divergent_remote_conflicts_without_overwriting_snapshot() {
    let dir = TestDir::new("sync-transport-conflict");
    let repo = SyncTransportRepository::for_project(dir.path());

    let base = envelope(dir.path(), "body v1", "edit-1", None);
    repo.apply_remote(&base).unwrap();

    let local = envelope(dir.path(), "local body", "edit-2", Some("edit-1"));
    repo.stage_local(&local).unwrap();

    let remote = envelope(dir.path(), "remote body", "edit-3", Some("edit-1"));
    let result = repo.apply_remote(&remote).unwrap();

    assert_eq!(result.status, SyncPullStatus::ConflictRecorded);
    assert_eq!(repo.load_snapshot("uuid-1234").unwrap().unwrap(), local);
    assert_eq!(repo.load_incoming("uuid-1234").unwrap().unwrap(), remote);

    let expected_conflict = local.detect_conflict(&remote).unwrap();
    let conflict_path = result.conflict_path.clone().expect("conflict path");
    assert_eq!(repo.layout.conflict_path(&expected_conflict), conflict_path);

    let artifact = SyncTransportConflictArtifact::load_from_path(&conflict_path)
        .unwrap()
        .unwrap();
    assert_eq!(artifact.conflict.kind, SyncConflictKind::ContentDiverged);
    assert_eq!(artifact.conflict, expected_conflict);
    assert_eq!(artifact.local, local);
    assert_eq!(artifact.remote, remote);
}

#[test]
fn sync_transport_status_surfaces_snapshot_queue_and_conflict_state() {
    let dir = TestDir::new("sync-transport-status");
    let repo = SyncTransportRepository::for_project(dir.path());

    let base = envelope(dir.path(), "body v1", "edit-1", None);
    repo.apply_remote(&base).unwrap();

    let local = envelope(dir.path(), "local body", "edit-2", Some("edit-1"));
    repo.stage_local(&local).unwrap();

    let remote = envelope(dir.path(), "remote body", "edit-3", Some("edit-1"));
    repo.apply_remote(&remote).unwrap();

    let status = repo.status_for("uuid-1234").unwrap().unwrap();
    assert_eq!(
        status
            .snapshot
            .as_ref()
            .and_then(|revision| revision.edit_id.as_deref()),
        Some("edit-2")
    );
    assert_eq!(
        status
            .outgoing
            .as_ref()
            .and_then(|revision| revision.edit_id.as_deref()),
        Some("edit-2")
    );
    assert_eq!(
        status
            .incoming
            .as_ref()
            .and_then(|revision| revision.edit_id.as_deref()),
        Some("edit-3")
    );
    assert!(status.pending_outgoing());
    assert!(status.pending_incoming());
    assert_eq!(status.conflict_count(), 1);
    assert_eq!(status.handoff.state, SyncHandoffState::Conflict);
    assert_eq!(status.handoff.shared_edit_id.as_deref(), Some("edit-1"));
    assert!(status.handoff.score < 70);
    assert_eq!(status.integrity_issue_count(), 0);
    assert_eq!(status.verified_revision_count(), 2);
    assert_eq!(status.module(), Some("engine"));
    let expected_source_path = dir.path().join("src/engine.rs");
    assert_eq!(status.source_path(), Some(expected_source_path.as_path()));

    let statuses = repo.list_statuses().unwrap();
    assert_eq!(statuses.len(), 1);
    assert_eq!(statuses[0], status);
}
