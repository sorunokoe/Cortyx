//! Syncable neuron construction and boundary checking.

use super::hash::{canonicalize_paths, hash_sync_body, hash_sync_payload, provenance_sync_summary};
use super::types::{SyncBoundaryReason, SyncConflict, SyncHeaders, SyncSynapse, SYNC_HEADERS_VERSION};
use crate::neuron::kind::{NeuronKind, NeuronStatus};
use crate::neuron::meta::NeuronMeta;
use crate::neuron::provenance::NeuronProvenance;
use crate::neuron::synapse::Synapse;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Canonical sync payload for a neuron.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SyncableNeuron {
    pub headers: SyncHeaders,
    pub body: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub synapses: Vec<SyncSynapse>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub module: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_pattern: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub source_files: Vec<PathBuf>,
}

impl SyncableNeuron {
    pub fn from_parts(
        meta: &NeuronMeta,
        body: &str,
        provenance: Option<&NeuronProvenance>,
    ) -> Option<Self> {
        let body = super::hash::normalize_sync_body(body);
        if sync_boundary_reason(meta, &body).is_some() {
            return None;
        }

        let neuron_uuid = meta.uuid.as_deref()?.trim().to_string();
        if neuron_uuid.is_empty() {
            return None;
        }

        let synapses = canonical_sync_synapses(&meta.synapses);
        let source_files = canonicalize_paths(&meta.source_files);
        let (latest_edit_id, parent_edit_id, provenance_fingerprint, revision_count, author_count) =
            provenance_sync_summary(provenance);

        let mut syncable = Self {
            headers: SyncHeaders {
                version: SYNC_HEADERS_VERSION,
                neuron_uuid,
                kind: meta.kind.clone(),
                status: meta.status.clone(),
                source_path: meta.source_path.clone(),
                source_hash: meta.source_hash.clone(),
                sig_hash: meta.sig_hash.clone(),
                last_updated: meta.last_updated.clone(),
                body_hash: hash_sync_body(&body),
                content_hash: String::new(),
                latest_edit_id,
                parent_edit_id,
                provenance_fingerprint,
                revision_count,
                author_count,
            },
            body,
            synapses,
            module: meta.module.clone(),
            task_pattern: meta.task_pattern.clone(),
            parent: meta.parent.clone(),
            source_files,
        };
        syncable.headers.content_hash = hash_sync_payload(&syncable);
        Some(syncable)
    }

    pub fn detect_conflict(&self, remote: &SyncableNeuron) -> Option<SyncConflict> {
        super::conflict::detect_sync_conflict(&self.headers, &remote.headers)
    }
}

impl SyncHeaders {
    pub fn from_meta(
        meta: &NeuronMeta,
        body: &str,
        provenance: Option<&NeuronProvenance>,
    ) -> Option<Self> {
        SyncableNeuron::from_parts(meta, body, provenance).map(|syncable| syncable.headers)
    }
}

/// Decide whether a neuron belongs to the shared-sync boundary.
pub fn is_syncable(meta: &NeuronMeta, body: &str) -> bool {
    sync_boundary_reason(meta, body).is_none()
}

pub fn sync_boundary_reason(meta: &NeuronMeta, body: &str) -> Option<SyncBoundaryReason> {
    if meta
        .uuid
        .as_deref()
        .map(str::trim)
        .filter(|uuid| !uuid.is_empty())
        .is_none()
    {
        return Some(SyncBoundaryReason::MissingUuid);
    }
    if matches!(meta.kind, NeuronKind::Verbatim | NeuronKind::Aggregate) {
        return Some(SyncBoundaryReason::LocalOnlyKind);
    }
    if matches!(meta.status, NeuronStatus::Stub) {
        return Some(SyncBoundaryReason::StubStatus);
    }
    if super::hash::normalize_sync_body(body).trim().is_empty() {
        return Some(SyncBoundaryReason::EmptyBody);
    }
    None
}

fn canonical_sync_synapses(synapses: &[Synapse]) -> Vec<SyncSynapse> {
    let mut canonical = synapses.iter().map(SyncSynapse::from).collect::<Vec<_>>();
    canonical.sort_by_key(super::hash::sync_synapse_sort_key);
    canonical
}
