//! Hash computation and normalization for sync payloads.

use super::neuron::SyncableNeuron;
use super::types::{SyncHashSynapse, SyncSynapse, SYNC_HASH_LEN};
use crate::neuron::kind::{NeuronKind, NeuronStatus};
use crate::neuron::provenance::NeuronProvenance;
use serde::Serialize;
use std::path::{Path, PathBuf};

/// BLAKE3 over the shareable neuron body after newline normalization.
pub fn hash_sync_body(body: &str) -> String {
    blake3_hex(normalize_sync_body(body).as_bytes())
}

/// BLAKE3 over the canonical sync payload.
///
/// Excludes `last_updated` and revision IDs so semantically equivalent payloads
/// hash the same across devices while lineage still lives in `SyncHeaders`.
pub fn hash_sync_payload(syncable: &SyncableNeuron) -> String {
    let body_hash = hash_sync_body(&syncable.body);
    let source_files = canonicalize_paths(&syncable.source_files);
    let synapses = canonicalize_sync_synapse_vec(&syncable.synapses);
    let input = SyncHashInput {
        version: syncable.headers.version,
        neuron_uuid: &syncable.headers.neuron_uuid,
        kind: &syncable.headers.kind,
        status: &syncable.headers.status,
        source_path: path_to_string(&syncable.headers.source_path),
        source_hash: &syncable.headers.source_hash,
        sig_hash: syncable.headers.sig_hash.as_deref(),
        module: syncable.module.as_deref(),
        task_pattern: syncable.task_pattern.as_deref(),
        parent: syncable.parent.as_deref().map(path_to_string),
        source_files: source_files
            .iter()
            .map(|path| path_to_string(path))
            .collect(),
        synapses: synapses.iter().map(SyncHashSynapse::from).collect(),
        body_hash: &body_hash,
    };
    blake3_hash_json(&input)
}

pub(super) fn normalize_sync_body(body: &str) -> String {
    let normalized = body.replace("\r\n", "\n").replace('\r', "\n");
    normalized.trim_end_matches('\n').to_string()
}

pub(super) fn canonicalize_paths(paths: &[PathBuf]) -> Vec<PathBuf> {
    let mut canonical = paths.to_vec();
    canonical.sort_by_key(|path| path_to_string(path));
    canonical
}

pub(super) fn canonicalize_sync_synapse_vec(synapses: &[SyncSynapse]) -> Vec<SyncSynapse> {
    let mut canonical = synapses.to_vec();
    canonical.sort_by_key(sync_synapse_sort_key);
    canonical
}

pub(super) fn sync_synapse_sort_key(synapse: &SyncSynapse) -> (String, String, u32, String) {
    (
        path_to_string(&synapse.target),
        serde_json::to_string(&synapse.edge_type).expect("sync synapse type is serializable"),
        synapse.weight.to_bits(),
        synapse.reason.clone(),
    )
}

pub(super) fn provenance_sync_summary(
    provenance: Option<&NeuronProvenance>,
) -> (Option<String>, Option<String>, Option<String>, usize, usize) {
    let (latest_edit_id, parent_edit_id) = provenance_revision_ids(provenance);
    (
        latest_edit_id,
        parent_edit_id,
        provenance.map(NeuronProvenance::fingerprint),
        provenance
            .map(|provenance| provenance.edit_history.len())
            .unwrap_or(0),
        provenance
            .map(NeuronProvenance::author_count)
            .unwrap_or_default(),
    )
}

fn provenance_revision_ids(
    provenance: Option<&NeuronProvenance>,
) -> (Option<String>, Option<String>) {
    let latest = provenance.and_then(|provenance| provenance.edit_history.last());
    (
        latest.map(|edit| edit.edit_id.clone()),
        latest.and_then(|edit| edit.parent_edit_id.clone()),
    )
}

pub(super) fn path_to_string(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn blake3_hash_json<T: Serialize>(value: &T) -> String {
    let bytes = serde_json::to_vec(value).expect("sync payload is serializable");
    blake3_hex(&bytes)
}

fn blake3_hex(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex()[..SYNC_HASH_LEN].to_string()
}

impl From<&SyncSynapse> for SyncHashSynapse {
    fn from(synapse: &SyncSynapse) -> Self {
        Self {
            target: path_to_string(&synapse.target),
            edge_type: synapse.edge_type.clone(),
            weight_bits: synapse.weight.to_bits(),
            reason: synapse.reason.clone(),
        }
    }
}

#[derive(Serialize)]
struct SyncHashInput<'a> {
    version: u32,
    neuron_uuid: &'a str,
    kind: &'a NeuronKind,
    status: &'a NeuronStatus,
    source_path: String,
    source_hash: &'a str,
    sig_hash: Option<&'a str>,
    module: Option<&'a str>,
    task_pattern: Option<&'a str>,
    parent: Option<String>,
    source_files: Vec<String>,
    synapses: Vec<SyncHashSynapse>,
    body_hash: &'a str,
}
