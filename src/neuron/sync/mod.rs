//! Additive sync-boundary model for shareable neurons.
//!
//! Keeps sync concerns out of the retrieval hot path while defining a
//! canonical, local-first payload for future transport layers.

mod conflict;
mod hash;
mod neuron;
mod types;

// Re-export public API
pub use conflict::detect_sync_conflict;
pub use hash::{hash_sync_body, hash_sync_payload};
pub use neuron::{is_syncable, sync_boundary_reason, SyncableNeuron};
pub use types::{
    SyncBoundaryReason, SyncConflict, SyncConflictKind, SyncConflictVersion, SyncHeaders,
    SyncSynapse, SYNC_HEADERS_VERSION,
};

#[cfg(test)]
mod tests {
    use super::super::meta::NeuronMeta;
    use super::super::provenance::{
        AuthorshipRecord, NeuronProvenance, ProvenanceAuthor, ProvenanceEditRecord,
        ProvenanceOperation, ProvenanceSource,
    };
    use super::super::synapse::{Synapse, SynapseType};
    use super::*;
    use crate::neuron::kind::{NeuronKind, NeuronStatus};
    use std::path::{Path, PathBuf};
    use types::{SyncBoundaryReason, SyncConflictKind};

    fn sync_meta(kind: NeuronKind, status: NeuronStatus) -> NeuronMeta {
        let mut meta = NeuronMeta::new_stub(Path::new("src/engine.rs"), kind);
        meta.status = status;
        meta.source_hash = "src-hash-1".to_string();
        meta.sig_hash = Some("sig-hash-1".to_string());
        meta.last_updated = "2026-01-02T03:04:05Z".to_string();
        meta.module = Some("engine".to_string());
        meta.uuid = Some("uuid-1234".to_string());
        meta
    }

    fn test_synapse(target: &str, edge_type: SynapseType, weight: f32, reason: &str) -> Synapse {
        Synapse {
            target: PathBuf::from(target),
            edge_type,
            weight: crate::types::SynapseWeight::new(weight),
            reason: reason.to_string(),
            learned_weight: crate::types::SynapseWeight::new(0.9),
            traversal_count: 12,
            last_co_activation_day: 77,
        }
    }

    fn test_provenance(latest_edit_id: &str, parent_edit_id: Option<&str>) -> NeuronProvenance {
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
                content_hash: Some("ctx-hash".to_string()),
            }],
        }
    }

    #[test]
    fn sync_boundary_rejects_local_only_or_incomplete_neurons() {
        let verbatim = sync_meta(NeuronKind::Verbatim, NeuronStatus::Fresh);
        assert_eq!(
            sync_boundary_reason(&verbatim, "conversation chunk"),
            Some(SyncBoundaryReason::LocalOnlyKind)
        );

        let stub = sync_meta(NeuronKind::Core, NeuronStatus::Stub);
        assert_eq!(
            sync_boundary_reason(&stub, "ready body"),
            Some(SyncBoundaryReason::StubStatus)
        );

        let mut missing_uuid = sync_meta(NeuronKind::Core, NeuronStatus::Fresh);
        missing_uuid.uuid = None;
        assert_eq!(
            sync_boundary_reason(&missing_uuid, "ready body"),
            Some(SyncBoundaryReason::MissingUuid)
        );

        let fresh = sync_meta(NeuronKind::Core, NeuronStatus::Fresh);
        assert_eq!(
            sync_boundary_reason(&fresh, " \r\n\r\n"),
            Some(SyncBoundaryReason::EmptyBody)
        );
        assert!(is_syncable(&fresh, "ready body"));
    }

    #[test]
    fn sync_boundary_hashes_ignore_local_learning_state() {
        let mut meta_a = sync_meta(NeuronKind::Concept, NeuronStatus::Fresh);
        meta_a.source_files = vec![PathBuf::from("src/b.rs"), PathBuf::from("src/a.rs")];
        meta_a.synapses = vec![
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

        let mut meta_b = meta_a.clone();
        meta_b.source_files.reverse();
        meta_b.synapses.reverse();
        meta_b.synapses[0].learned_weight = crate::types::SynapseWeight::ZERO;
        meta_b.synapses[0].traversal_count = 999;
        meta_b.synapses[0].last_co_activation_day = 1;
        meta_b.synapses[1].learned_weight = crate::types::SynapseWeight::ONE;
        meta_b.synapses[1].traversal_count = 0;
        meta_b.synapses[1].last_co_activation_day = 2048;

        let sync_a = SyncableNeuron::from_parts(&meta_a, "line one\r\nline two\r\n", None).unwrap();
        let sync_b = SyncableNeuron::from_parts(&meta_b, "line one\nline two\n", None).unwrap();

        assert_eq!(sync_a.body, "line one\nline two");
        assert_eq!(sync_a.headers.body_hash, sync_b.headers.body_hash);
        assert_eq!(sync_a.headers.content_hash, sync_b.headers.content_hash);
        assert_eq!(sync_a.synapses, sync_b.synapses);
        assert_eq!(sync_a.source_files, sync_b.source_files);
    }

    #[test]
    fn sync_boundary_captures_latest_provenance_revision() {
        let meta = sync_meta(NeuronKind::Core, NeuronStatus::Fresh);
        let provenance = test_provenance("edit-2", Some("edit-1"));

        let headers = SyncHeaders::from_meta(&meta, "body", Some(&provenance)).unwrap();

        assert_eq!(headers.latest_edit_id.as_deref(), Some("edit-2"));
        assert_eq!(headers.parent_edit_id.as_deref(), Some("edit-1"));
        assert!(headers.provenance_fingerprint.is_some());
        assert_eq!(headers.revision_count, 1);
        assert_eq!(headers.author_count, 1);
    }

    #[test]
    fn sync_boundary_detects_fast_forward_and_divergence() {
        let meta = sync_meta(NeuronKind::Core, NeuronStatus::Fresh);
        let local =
            SyncableNeuron::from_parts(&meta, "body v1", Some(&test_provenance("edit-1", None)))
                .unwrap();
        let remote_ahead = SyncableNeuron::from_parts(
            &meta,
            "body v2",
            Some(&test_provenance("edit-2", Some("edit-1"))),
        )
        .unwrap();
        assert!(local.detect_conflict(&remote_ahead).is_none());

        let remote_diverged = SyncableNeuron::from_parts(
            &meta,
            "body v3",
            Some(&test_provenance("edit-3", Some("other-base"))),
        )
        .unwrap();
        let conflict = local.detect_conflict(&remote_diverged).unwrap();
        assert_eq!(conflict.kind, SyncConflictKind::ContentDiverged);
        assert_eq!(conflict.neuron_uuid, "uuid-1234");
        assert_eq!(conflict.local.latest_edit_id.as_deref(), Some("edit-1"));
        assert_eq!(conflict.remote.latest_edit_id.as_deref(), Some("edit-3"));
    }

    #[test]
    fn sync_boundary_detects_metadata_only_divergence() {
        let local_meta = sync_meta(NeuronKind::Core, NeuronStatus::Fresh);
        let mut remote_meta = local_meta.clone();
        remote_meta.source_path = PathBuf::from("src/engine_v2.rs");
        remote_meta.source_hash = "src-hash-2".to_string();

        let local = SyncableNeuron::from_parts(
            &local_meta,
            "same body",
            Some(&test_provenance("edit-1", Some("base"))),
        )
        .unwrap();
        let remote = SyncableNeuron::from_parts(
            &remote_meta,
            "same body",
            Some(&test_provenance("edit-2", Some("other-base"))),
        )
        .unwrap();

        let conflict = detect_sync_conflict(&local.headers, &remote.headers).unwrap();
        assert_eq!(conflict.kind, SyncConflictKind::MetadataDiverged);
        assert_eq!(conflict.local.body_hash, conflict.remote.body_hash);
    }
}
