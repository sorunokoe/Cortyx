use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::neuron::{
    provenance::NeuronProvenance,
    sync::{SyncConflict, SyncConflictKind, SyncConflictVersion, SyncHeaders, SyncableNeuron},
    NeuronMeta,
};

pub(super) fn default_sync_transport_version() -> u32 {
    super::SYNC_TRANSPORT_VERSION
}

/// Persisted transport envelope stored in snapshot/inbox/outbox files.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SyncTransportEnvelope {
    #[serde(default = "default_sync_transport_version")]
    pub version: u32,
    pub syncable: SyncableNeuron,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<NeuronProvenance>,
}

impl SyncTransportEnvelope {
    #[must_use]
    pub fn from_syncable(syncable: SyncableNeuron, provenance: Option<NeuronProvenance>) -> Self {
        Self {
            version: super::SYNC_TRANSPORT_VERSION,
            syncable,
            provenance,
        }
    }

    #[must_use]
    pub fn from_parts(
        meta: &NeuronMeta,
        body: &str,
        provenance: Option<&NeuronProvenance>,
    ) -> Option<Self> {
        let syncable = SyncableNeuron::from_parts(meta, body, provenance)?;
        let provenance = provenance.map(|provenance| {
            let mut provenance = provenance.clone();
            provenance.sync_from_meta(meta);
            provenance
        });
        Some(Self::from_syncable(syncable, provenance))
    }

    #[must_use]
    pub fn headers(&self) -> &SyncHeaders {
        &self.syncable.headers
    }

    #[must_use]
    pub fn neuron_uuid(&self) -> &str {
        &self.headers().neuron_uuid
    }

    #[must_use]
    pub fn latest_edit_id(&self) -> Option<&str> {
        self.headers().latest_edit_id.as_deref()
    }

    #[must_use]
    pub fn content_hash(&self) -> &str {
        &self.headers().content_hash
    }

    #[must_use]
    pub fn fast_forwards(&self, base: &Self) -> bool {
        self.headers().fast_forwards(base.headers())
    }

    #[must_use]
    pub fn detect_conflict(&self, other: &Self) -> Option<SyncConflict> {
        self.syncable.detect_conflict(&other.syncable)
    }

    #[must_use]
    pub fn relation_to(&self, other: &Self) -> SyncTransportRelation {
        compare_sync_transport(self, other)
    }

    /// # Errors
    ///
    /// Returns an error if the underlying operation fails.
    pub fn to_json_pretty(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// # Errors
    ///
    /// Returns an error if the underlying operation fails.
    pub fn from_json_str(data: &str) -> Result<Self> {
        let mut envelope: Self = serde_json::from_str(data)?;
        envelope.normalize_version();
        Ok(envelope)
    }

    /// # Errors
    ///
    /// Returns an error if the underlying operation fails.
    pub fn load_from_path(path: &Path) -> Result<Option<Self>> {
        let Some(mut envelope) = super::read_json::<Self>(path)? else {
            return Ok(None);
        };
        envelope.normalize_version();
        Ok(Some(envelope))
    }

    /// # Errors
    ///
    /// Returns an error if the underlying operation fails.
    pub fn save_to_path(&self, path: &Path) -> Result<()> {
        super::ensure_parent_dir(path)?;
        crate::neuron::atomic_write_json(path, self)
    }

    pub(super) fn normalize_version(&mut self) {
        if self.version == 0 {
            self.version = super::SYNC_TRANSPORT_VERSION;
        }
    }
}

/// High-level relationship between two transport envelopes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SyncTransportRelation {
    DifferentNeuron,
    Identical,
    LocalAhead,
    RemoteAhead,
    Diverged(Box<SyncConflict>),
}

#[must_use]
pub fn compare_sync_transport(
    local: &SyncTransportEnvelope,
    remote: &SyncTransportEnvelope,
) -> SyncTransportRelation {
    if local.neuron_uuid() != remote.neuron_uuid() {
        return SyncTransportRelation::DifferentNeuron;
    }
    if local.content_hash() == remote.content_hash() {
        return SyncTransportRelation::Identical;
    }
    if local.fast_forwards(remote) {
        return SyncTransportRelation::LocalAhead;
    }
    if remote.fast_forwards(local) {
        return SyncTransportRelation::RemoteAhead;
    }
    let conflict = local
        .detect_conflict(remote)
        .unwrap_or_else(|| SyncConflict {
            kind: if local.headers().body_hash == remote.headers().body_hash {
                SyncConflictKind::MetadataDiverged
            } else {
                SyncConflictKind::ContentDiverged
            },
            neuron_uuid: local.neuron_uuid().to_string(),
            local: SyncConflictVersion::from(local.headers()),
            remote: SyncConflictVersion::from(remote.headers()),
        });
    SyncTransportRelation::Diverged(Box::new(conflict))
}
