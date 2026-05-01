//! Sync conflict detection logic.

use super::types::{SyncConflict, SyncConflictKind, SyncConflictVersion, SyncHeaders};

/// Detect a true divergence between two revisions of the same syncable neuron.
pub fn detect_sync_conflict(local: &SyncHeaders, remote: &SyncHeaders) -> Option<SyncConflict> {
    if local.neuron_uuid != remote.neuron_uuid {
        return None;
    }
    if local.content_hash == remote.content_hash {
        return None;
    }
    if local.fast_forwards(remote) || remote.fast_forwards(local) {
        return None;
    }

    Some(SyncConflict {
        kind: if local.body_hash == remote.body_hash {
            SyncConflictKind::MetadataDiverged
        } else {
            SyncConflictKind::ContentDiverged
        },
        neuron_uuid: local.neuron_uuid.clone(),
        local: SyncConflictVersion::from(local),
        remote: SyncConflictVersion::from(remote),
    })
}
