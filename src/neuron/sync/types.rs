//! Core sync types for shareable neurons.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use super::super::kind::{NeuronKind, NeuronStatus};
use super::super::synapse::{Synapse, SynapseType};

pub const SYNC_HEADERS_VERSION: u32 = 1;
pub(super) const SYNC_HASH_LEN: usize = 32;

pub(super) fn default_sync_headers_version() -> u32 {
    SYNC_HEADERS_VERSION
}

pub(super) fn is_zero(value: &usize) -> bool {
    *value == 0
}

/// Why a neuron is excluded from shared sync.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyncBoundaryReason {
    LocalOnlyKind,
    StubStatus,
    MissingUuid,
    EmptyBody,
}

/// Shareable header block used for sync identity, hashing, and conflict checks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyncHeaders {
    #[serde(default = "default_sync_headers_version")]
    pub version: u32,
    pub neuron_uuid: String,
    pub kind: NeuronKind,
    pub status: NeuronStatus,
    pub source_path: PathBuf,
    pub source_hash: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sig_hash: Option<String>,
    pub last_updated: String,
    pub body_hash: String,
    pub content_hash: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latest_edit_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_edit_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub revision_count: usize,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub author_count: usize,
}

impl SyncHeaders {
    /// True when `self` is a direct descendant of `base`, so the update can be
    /// fast-forwarded instead of treated as a divergence.
    #[must_use]
    pub fn fast_forwards(&self, base: &SyncHeaders) -> bool {
        match (
            self.parent_edit_id.as_deref(),
            base.latest_edit_id.as_deref(),
        ) {
            (Some(parent), Some(base_latest)) => parent == base_latest,
            _ => false,
        }
    }
}

/// Sync-safe synapse snapshot that excludes local traversal telemetry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SyncSynapse {
    pub target: PathBuf,
    pub edge_type: SynapseType,
    pub weight: f32,
    pub reason: String,
}

impl From<&Synapse> for SyncSynapse {
    fn from(synapse: &Synapse) -> Self {
        Self {
            target: synapse.target.clone(),
            edge_type: synapse.edge_type.clone(),
            weight: synapse.weight.get(),
            reason: synapse.reason.clone(),
        }
    }
}

/// Conflict category for two sync payloads that share a neuron UUID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyncConflictKind {
    ContentDiverged,
    MetadataDiverged,
}

/// Snapshot of one side of a sync conflict.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyncConflictVersion {
    pub kind: NeuronKind,
    pub status: NeuronStatus,
    pub content_hash: String,
    pub body_hash: String,
    pub source_path: PathBuf,
    pub source_hash: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sig_hash: Option<String>,
    pub last_updated: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latest_edit_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_edit_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub revision_count: usize,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub author_count: usize,
}

impl From<&SyncHeaders> for SyncConflictVersion {
    fn from(headers: &SyncHeaders) -> Self {
        Self {
            kind: headers.kind.clone(),
            status: headers.status.clone(),
            content_hash: headers.content_hash.clone(),
            body_hash: headers.body_hash.clone(),
            source_path: headers.source_path.clone(),
            source_hash: headers.source_hash.clone(),
            sig_hash: headers.sig_hash.clone(),
            last_updated: headers.last_updated.clone(),
            latest_edit_id: headers.latest_edit_id.clone(),
            parent_edit_id: headers.parent_edit_id.clone(),
            provenance_fingerprint: headers.provenance_fingerprint.clone(),
            revision_count: headers.revision_count,
            author_count: headers.author_count,
        }
    }
}

/// Divergent sync state for the same neuron UUID.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyncConflict {
    pub kind: SyncConflictKind,
    pub neuron_uuid: String,
    pub local: SyncConflictVersion,
    pub remote: SyncConflictVersion,
}

/// Hash synapse representation used for internal hashing (excludes reason text).
#[derive(Serialize)]
pub(super) struct SyncHashSynapse {
    pub target: String,
    pub edge_type: SynapseType,
    pub weight_bits: u32,
    pub reason: String,
}
