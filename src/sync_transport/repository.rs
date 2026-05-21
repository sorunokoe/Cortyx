use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use crate::neuron::sync::SyncConflict;

use super::*;

fn default_sync_transport_version() -> u32 {
    SYNC_TRANSPORT_VERSION
}

/// Stored conflict artifact kept beside local/incoming snapshots.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SyncTransportConflictArtifact {
    #[serde(default = "default_sync_transport_version")]
    pub version: u32,
    pub conflict: SyncConflict,
    pub local: SyncTransportEnvelope,
    pub remote: SyncTransportEnvelope,
}

impl SyncTransportConflictArtifact {
    #[must_use]
    pub fn from_envelopes(
        conflict: SyncConflict,
        local: &SyncTransportEnvelope,
        remote: &SyncTransportEnvelope,
    ) -> Self {
        Self {
            version: SYNC_TRANSPORT_VERSION,
            conflict,
            local: local.clone(),
            remote: remote.clone(),
        }
    }

    /// # Errors
    ///
    /// Returns an error if the underlying operation fails.
    pub fn load_from_path(path: &Path) -> Result<Option<Self>> {
        let Some(mut artifact) = super::read_json::<Self>(path)? else {
            return Ok(None);
        };
        artifact.normalize_versions();
        Ok(Some(artifact))
    }

    /// # Errors
    ///
    /// Returns an error if the underlying operation fails.
    pub fn save_to_path(&self, path: &Path) -> Result<()> {
        super::ensure_parent_dir(path)?;
        crate::neuron::atomic_write_json(path, self)
    }

    fn normalize_versions(&mut self) {
        if self.version == 0 {
            self.version = SYNC_TRANSPORT_VERSION;
        }
        self.local.normalize_version();
        self.remote.normalize_version();
    }
}

/// Deterministic filesystem layout for local snapshots, queues, and conflicts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncRepositoryLayout {
    pub root: PathBuf,
    pub snapshots_dir: PathBuf,
    pub outgoing_dir: PathBuf,
    pub incoming_dir: PathBuf,
    pub conflicts_dir: PathBuf,
}

impl SyncRepositoryLayout {
    #[must_use]
    pub fn for_project(project_root: &Path) -> Self {
        Self::new(sync_transport_dir(project_root))
    }

    pub fn new(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        Self {
            snapshots_dir: root.join("snapshots"),
            outgoing_dir: root.join("outgoing"),
            incoming_dir: root.join("incoming"),
            conflicts_dir: root.join("conflicts"),
            root,
        }
    }

    /// # Errors
    ///
    /// Returns an error if the underlying operation fails.
    pub fn ensure_exists(&self) -> Result<()> {
        for dir in [
            &self.root,
            &self.snapshots_dir,
            &self.outgoing_dir,
            &self.incoming_dir,
            &self.conflicts_dir,
        ] {
            std::fs::create_dir_all(dir)?;
        }
        Ok(())
    }

    #[must_use]
    pub fn snapshot_path(&self, neuron_uuid: &str) -> PathBuf {
        bucketed_transport_path(&self.snapshots_dir, neuron_uuid)
    }

    #[must_use]
    pub fn outgoing_path(&self, neuron_uuid: &str) -> PathBuf {
        bucketed_transport_path(&self.outgoing_dir, neuron_uuid)
    }

    #[must_use]
    pub fn incoming_path(&self, neuron_uuid: &str) -> PathBuf {
        bucketed_transport_path(&self.incoming_dir, neuron_uuid)
    }

    #[must_use]
    pub fn conflict_path(&self, conflict: &SyncConflict) -> PathBuf {
        let neuron_uuid = safe_component(&conflict.neuron_uuid);
        let local_revision = revision_component(
            conflict.local.latest_edit_id.as_deref(),
            &conflict.local.content_hash,
        );
        let remote_revision = revision_component(
            conflict.remote.latest_edit_id.as_deref(),
            &conflict.remote.content_hash,
        );
        let file_name = format!("{neuron_uuid}--{local_revision}--{remote_revision}");
        self.conflicts_dir
            .join(bucket_component(&neuron_uuid))
            .join(format!("{file_name}.json"))
    }
}

/// File-backed local-first repository used by transport tests and future sync plumbing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncTransportRepository {
    pub layout: SyncRepositoryLayout,
}

impl SyncTransportRepository {
    #[must_use]
    pub fn for_project(project_root: &Path) -> Self {
        Self {
            layout: SyncRepositoryLayout::for_project(project_root),
        }
    }

    pub fn from_sync_root(sync_root: impl Into<PathBuf>) -> Self {
        Self {
            layout: SyncRepositoryLayout::new(sync_root),
        }
    }

    /// # Errors
    ///
    /// Returns an error if the underlying operation fails.
    pub fn ensure_layout(&self) -> Result<()> {
        self.layout.ensure_exists()
    }

    /// # Errors
    ///
    /// Returns an error if the underlying operation fails.
    pub fn load_snapshot(&self, neuron_uuid: &str) -> Result<Option<SyncTransportEnvelope>> {
        SyncTransportEnvelope::load_from_path(&self.layout.snapshot_path(neuron_uuid))
    }

    /// # Errors
    ///
    /// Returns an error if the underlying operation fails.
    pub fn load_outgoing(&self, neuron_uuid: &str) -> Result<Option<SyncTransportEnvelope>> {
        SyncTransportEnvelope::load_from_path(&self.layout.outgoing_path(neuron_uuid))
    }

    /// # Errors
    ///
    /// Returns an error if the underlying operation fails.
    pub fn load_incoming(&self, neuron_uuid: &str) -> Result<Option<SyncTransportEnvelope>> {
        SyncTransportEnvelope::load_from_path(&self.layout.incoming_path(neuron_uuid))
    }

    /// # Errors
    ///
    /// Returns an error if the underlying operation fails.
    pub fn load_conflict(
        &self,
        conflict: &SyncConflict,
    ) -> Result<Option<SyncTransportConflictArtifact>> {
        SyncTransportConflictArtifact::load_from_path(&self.layout.conflict_path(conflict))
    }

    /// # Errors
    ///
    /// Returns an error if the underlying operation fails.
    pub fn status_for(&self, neuron_uuid: &str) -> Result<Option<SyncTransportStatus>> {
        self.ensure_layout()?;
        let conflict_paths = self.conflict_paths_for(neuron_uuid)?;
        build_sync_transport_status(
            neuron_uuid,
            self.load_snapshot(neuron_uuid)?,
            self.load_outgoing(neuron_uuid)?,
            self.load_incoming(neuron_uuid)?,
            conflict_paths,
        )
    }

    /// # Errors
    ///
    /// Returns an error if the underlying operation fails.
    pub fn list_statuses(&self) -> Result<Vec<SyncTransportStatus>> {
        self.ensure_layout()?;

        let mut uuids = BTreeSet::new();
        for base in [
            &self.layout.snapshots_dir,
            &self.layout.outgoing_dir,
            &self.layout.incoming_dir,
        ] {
            for path in collect_transport_json_paths(base)? {
                if let Some(uuid) = transport_uuid_from_path(&path) {
                    uuids.insert(uuid);
                }
            }
        }

        let mut conflict_paths_by_uuid: HashMap<String, Vec<PathBuf>> = HashMap::new();
        for path in collect_transport_json_paths(&self.layout.conflicts_dir)? {
            let Some(artifact) = SyncTransportConflictArtifact::load_from_path(&path)? else {
                continue;
            };
            let uuid = artifact.conflict.neuron_uuid;
            uuids.insert(uuid.clone());
            conflict_paths_by_uuid.entry(uuid).or_default().push(path);
        }

        let mut statuses = Vec::new();
        for uuid in uuids {
            let mut conflict_paths = conflict_paths_by_uuid.remove(&uuid).unwrap_or_default();
            conflict_paths.sort();
            if let Some(status) = build_sync_transport_status(
                &uuid,
                self.load_snapshot(&uuid)?,
                self.load_outgoing(&uuid)?,
                self.load_incoming(&uuid)?,
                conflict_paths,
            )? {
                statuses.push(status);
            }
        }

        statuses.sort_by(|left, right| {
            right
                .latest_activity_at()
                .cmp(&left.latest_activity_at())
                .then_with(|| left.neuron_uuid.cmp(&right.neuron_uuid))
        });
        Ok(statuses)
    }

    /// # Errors
    ///
    /// Returns an error if the underlying operation fails.
    pub fn stage_local(&self, envelope: &SyncTransportEnvelope) -> Result<SyncStageResult> {
        self.ensure_layout()?;

        let neuron_uuid = envelope.neuron_uuid();
        let snapshot_path = self.layout.snapshot_path(neuron_uuid);
        let outgoing_path = self.layout.outgoing_path(neuron_uuid);

        let Some(snapshot) = self.load_snapshot(neuron_uuid)? else {
            envelope.save_to_path(&snapshot_path)?;
            envelope.save_to_path(&outgoing_path)?;
            return Ok(SyncStageResult {
                status: SyncStageStatus::QueuedNew,
                snapshot_path,
                outgoing_path: Some(outgoing_path),
                conflict_path: None,
            });
        };

        match compare_sync_transport(envelope, &snapshot) {
            SyncTransportRelation::DifferentNeuron => unreachable!("snapshot path is UUID-scoped"),
            SyncTransportRelation::Identical => Ok(SyncStageResult {
                status: SyncStageStatus::AlreadyCurrent,
                snapshot_path,
                outgoing_path: self
                    .load_outgoing(neuron_uuid)?
                    .map(|_| self.layout.outgoing_path(neuron_uuid)),
                conflict_path: None,
            }),
            SyncTransportRelation::LocalAhead => {
                envelope.save_to_path(&snapshot_path)?;
                envelope.save_to_path(&outgoing_path)?;
                Ok(SyncStageResult {
                    status: SyncStageStatus::QueuedFastForward,
                    snapshot_path,
                    outgoing_path: Some(outgoing_path),
                    conflict_path: None,
                })
            },
            SyncTransportRelation::RemoteAhead => Ok(SyncStageResult {
                status: SyncStageStatus::IgnoredStale,
                snapshot_path,
                outgoing_path: self
                    .load_outgoing(neuron_uuid)?
                    .map(|_| self.layout.outgoing_path(neuron_uuid)),
                conflict_path: None,
            }),
            SyncTransportRelation::Diverged(conflict) => {
                let conflict_path = self.record_conflict(&snapshot, envelope, *conflict)?;
                Ok(SyncStageResult {
                    status: SyncStageStatus::ConflictRecorded,
                    snapshot_path,
                    outgoing_path: self
                        .load_outgoing(neuron_uuid)?
                        .map(|_| self.layout.outgoing_path(neuron_uuid)),
                    conflict_path: Some(conflict_path),
                })
            },
        }
    }

    /// # Errors
    ///
    /// Returns an error if the underlying operation fails.
    pub fn apply_remote(&self, envelope: &SyncTransportEnvelope) -> Result<SyncPullResult> {
        self.ensure_layout()?;

        let neuron_uuid = envelope.neuron_uuid();
        let incoming_path = self.layout.incoming_path(neuron_uuid);
        let snapshot_path = self.layout.snapshot_path(neuron_uuid);

        envelope.save_to_path(&incoming_path)?;

        let Some(local) = self.load_snapshot(neuron_uuid)? else {
            envelope.save_to_path(&snapshot_path)?;
            self.clear_outgoing(neuron_uuid)?;
            return Ok(SyncPullResult {
                status: SyncPullStatus::AppliedNew,
                snapshot_path,
                incoming_path,
                conflict_path: None,
            });
        };

        match compare_sync_transport(&local, envelope) {
            SyncTransportRelation::DifferentNeuron => unreachable!("snapshot path is UUID-scoped"),
            SyncTransportRelation::Identical => {
                self.clear_outgoing(neuron_uuid)?;
                Ok(SyncPullResult {
                    status: SyncPullStatus::AlreadyCurrent,
                    snapshot_path,
                    incoming_path,
                    conflict_path: None,
                })
            },
            SyncTransportRelation::LocalAhead => Ok(SyncPullResult {
                status: SyncPullStatus::IgnoredStale,
                snapshot_path,
                incoming_path,
                conflict_path: None,
            }),
            SyncTransportRelation::RemoteAhead => {
                envelope.save_to_path(&snapshot_path)?;
                self.clear_outgoing(neuron_uuid)?;
                Ok(SyncPullResult {
                    status: SyncPullStatus::AppliedFastForward,
                    snapshot_path,
                    incoming_path,
                    conflict_path: None,
                })
            },
            SyncTransportRelation::Diverged(conflict) => {
                let conflict_path = self.record_conflict(&local, envelope, *conflict)?;
                Ok(SyncPullResult {
                    status: SyncPullStatus::ConflictRecorded,
                    snapshot_path,
                    incoming_path,
                    conflict_path: Some(conflict_path),
                })
            },
        }
    }

    /// # Errors
    ///
    /// Returns an error if the underlying operation fails.
    pub fn resolve_handoff(
        &self,
        envelope: &SyncTransportEnvelope,
        strategy: SyncResolutionStrategy,
    ) -> Result<SyncResolutionResult> {
        self.ensure_layout()?;

        let neuron_uuid = envelope.neuron_uuid();
        let snapshot_path = self.layout.snapshot_path(neuron_uuid);
        let outgoing_path = self.layout.outgoing_path(neuron_uuid);
        let incoming_path = self.layout.incoming_path(neuron_uuid);
        let cleared_conflict_paths = self.clear_conflicts(neuron_uuid)?;

        let result = match strategy {
            SyncResolutionStrategy::AdoptRemote => {
                envelope.save_to_path(&incoming_path)?;
                envelope.save_to_path(&snapshot_path)?;
                self.clear_outgoing(neuron_uuid)?;
                SyncResolutionResult {
                    status: SyncResolutionStatus::AdoptedRemote,
                    snapshot_path,
                    outgoing_path: None,
                    incoming_path: Some(incoming_path),
                    cleared_conflict_paths,
                }
            },
            SyncResolutionStrategy::StageLocal => {
                envelope.save_to_path(&snapshot_path)?;
                envelope.save_to_path(&outgoing_path)?;
                self.clear_incoming(neuron_uuid)?;
                SyncResolutionResult {
                    status: SyncResolutionStatus::StagedLocal,
                    snapshot_path,
                    outgoing_path: Some(outgoing_path),
                    incoming_path: None,
                    cleared_conflict_paths,
                }
            },
        };

        Ok(result)
    }

    fn record_conflict(
        &self,
        local: &SyncTransportEnvelope,
        remote: &SyncTransportEnvelope,
        conflict: SyncConflict,
    ) -> Result<PathBuf> {
        let path = self.layout.conflict_path(&conflict);
        let artifact = SyncTransportConflictArtifact::from_envelopes(conflict, local, remote);
        artifact.save_to_path(&path)?;
        Ok(path)
    }

    fn clear_outgoing(&self, neuron_uuid: &str) -> Result<()> {
        super::clear_optional_file(&self.layout.outgoing_path(neuron_uuid))
    }

    fn clear_incoming(&self, neuron_uuid: &str) -> Result<()> {
        super::clear_optional_file(&self.layout.incoming_path(neuron_uuid))
    }

    fn clear_conflicts(&self, neuron_uuid: &str) -> Result<Vec<PathBuf>> {
        let paths = self.conflict_paths_for(neuron_uuid)?;
        for path in &paths {
            super::clear_optional_file(path)?;
        }
        Ok(paths)
    }

    fn conflict_paths_for(&self, neuron_uuid: &str) -> Result<Vec<PathBuf>> {
        let mut paths = Vec::new();
        for path in collect_transport_json_paths(&self.layout.conflicts_dir)? {
            let Some(artifact) = SyncTransportConflictArtifact::load_from_path(&path)? else {
                continue;
            };
            if artifact.conflict.neuron_uuid == neuron_uuid {
                paths.push(path);
            }
        }
        paths.sort();
        Ok(paths)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncStageStatus {
    QueuedNew,
    QueuedFastForward,
    AlreadyCurrent,
    IgnoredStale,
    ConflictRecorded,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncStageResult {
    pub status: SyncStageStatus,
    pub snapshot_path: PathBuf,
    pub outgoing_path: Option<PathBuf>,
    pub conflict_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncPullStatus {
    AppliedNew,
    AppliedFastForward,
    AlreadyCurrent,
    IgnoredStale,
    ConflictRecorded,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncPullResult {
    pub status: SyncPullStatus,
    pub snapshot_path: PathBuf,
    pub incoming_path: PathBuf,
    pub conflict_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncResolutionStrategy {
    AdoptRemote,
    StageLocal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncResolutionStatus {
    AdoptedRemote,
    StagedLocal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncResolutionResult {
    pub status: SyncResolutionStatus,
    pub snapshot_path: PathBuf,
    pub outgoing_path: Option<PathBuf>,
    pub incoming_path: Option<PathBuf>,
    pub cleared_conflict_paths: Vec<PathBuf>,
}

// ─── Private helper functions ─────────────────────────────────────────────────

fn incoming_requires_review(
    snapshot: Option<&SyncTransportEnvelope>,
    incoming: Option<&SyncTransportEnvelope>,
) -> bool {
    let Some(incoming) = incoming else {
        return false;
    };
    let Some(snapshot) = snapshot else {
        return true;
    };
    matches!(
        compare_sync_transport(snapshot, incoming),
        SyncTransportRelation::DifferentNeuron
            | SyncTransportRelation::RemoteAhead
            | SyncTransportRelation::Diverged(_)
    )
}

fn build_sync_handoff_summary(
    snapshot: Option<&SyncTransportEnvelope>,
    outgoing: Option<&SyncTransportEnvelope>,
    incoming: Option<&SyncTransportEnvelope>,
    snapshot_state: Option<&SyncRevisionState>,
    outgoing_state: Option<&SyncRevisionState>,
    incoming_state: Option<&SyncRevisionState>,
    outgoing_pending: bool,
    incoming_pending: bool,
    has_conflict: bool,
) -> SyncHandoffSummary {
    let local_envelope = outgoing.or(snapshot);
    let local_state = outgoing_state.or(snapshot_state);
    let relation = match (local_envelope, incoming) {
        (Some(local), Some(remote)) => Some(compare_sync_transport(local, remote)),
        _ => None,
    };
    let shared_edit_id =
        shared_edit_id_between(local_envelope, incoming, local_state, incoming_state);
    let local_integrity_verified = local_state.map(|s| s.integrity.trusted).unwrap_or(true);
    let remote_integrity_verified = incoming_state.map(|s| s.integrity.trusted).unwrap_or(true);
    let provenance_diverged = local_state
        .and_then(|s| s.provenance_fingerprint.as_deref())
        .zip(incoming_state.and_then(|s| s.provenance_fingerprint.as_deref()))
        .map(|(local, remote)| {
            local != remote
                && local_state.map(|s| s.content_hash.as_str())
                    == incoming_state.map(|s| s.content_hash.as_str())
        })
        .unwrap_or(false);
    let integrity_verified =
        local_integrity_verified && remote_integrity_verified && !provenance_diverged;
    let continuity_verified = if matches!(
        relation.as_ref(),
        Some(
            SyncTransportRelation::Identical
                | SyncTransportRelation::LocalAhead
                | SyncTransportRelation::RemoteAhead
        )
    ) {
        true
    } else if let (Some(snapshot), Some(outgoing)) = (snapshot, outgoing) {
        matches!(
            compare_sync_transport(outgoing, snapshot),
            SyncTransportRelation::Identical | SyncTransportRelation::LocalAhead
        )
    } else {
        local_state
            .or(incoming_state)
            .map(|s| s.integrity.trusted)
            .unwrap_or(false)
    };
    let state =
        if has_conflict || matches!(relation.as_ref(), Some(SyncTransportRelation::Diverged(_))) {
            SyncHandoffState::Conflict
        } else if outgoing_pending {
            SyncHandoffState::PendingOutgoing
        } else if incoming_pending {
            SyncHandoffState::PendingIncoming
        } else if incoming.is_some() {
            SyncHandoffState::Applied
        } else {
            SyncHandoffState::Idle
        };

    let mut issues = Vec::new();
    if local_state.is_some() && !local_integrity_verified {
        issues.push(SyncHandoffIssue::LocalIntegrityUnverified);
    }
    if incoming_state.is_some() && !remote_integrity_verified {
        issues.push(SyncHandoffIssue::RemoteIntegrityUnverified);
    }
    if provenance_diverged {
        issues.push(SyncHandoffIssue::ProvenanceDiverged);
    }
    if local_state.is_some()
        && incoming_state.is_some()
        && shared_edit_id.is_none()
        && !matches!(relation.as_ref(), Some(SyncTransportRelation::Identical))
    {
        issues.push(SyncHandoffIssue::MissingSharedAncestor);
    }
    if incoming_pending {
        issues.push(SyncHandoffIssue::IncomingNotApplied);
    }
    if matches!(state, SyncHandoffState::Conflict) {
        issues.push(SyncHandoffIssue::ConflictRecorded);
    }

    SyncHandoffSummary {
        state,
        shared_edit_id,
        local_edit_id: local_state.and_then(|s| s.edit_id.clone()),
        remote_edit_id: incoming_state.and_then(|s| s.edit_id.clone()),
        continuity_verified,
        integrity_verified,
        score: sync_handoff_score(state, continuity_verified, integrity_verified, issues.len()),
        issues,
    }
}

fn shared_edit_id_between(
    local_envelope: Option<&SyncTransportEnvelope>,
    remote_envelope: Option<&SyncTransportEnvelope>,
    local_state: Option<&SyncRevisionState>,
    remote_state: Option<&SyncRevisionState>,
) -> Option<String> {
    local_envelope
        .and_then(|e| e.provenance.as_ref())
        .zip(remote_envelope.and_then(|e| e.provenance.as_ref()))
        .and_then(|(local, remote)| local.shared_ancestor_edit_id(remote))
        .or_else(|| fallback_shared_edit_id(local_state, remote_state))
}

fn fallback_shared_edit_id(
    local_state: Option<&SyncRevisionState>,
    remote_state: Option<&SyncRevisionState>,
) -> Option<String> {
    let local_edit = local_state.and_then(|s| s.edit_id.as_deref());
    let remote_edit = remote_state.and_then(|s| s.edit_id.as_deref());
    let local_parent = local_state.and_then(|s| s.parent_edit_id.as_deref());
    let remote_parent = remote_state.and_then(|s| s.parent_edit_id.as_deref());

    if local_edit == remote_edit {
        local_edit.map(str::to_string)
    } else if local_parent == remote_edit {
        remote_edit.map(str::to_string)
    } else if remote_parent == local_edit {
        local_edit.map(str::to_string)
    } else if local_parent == remote_parent {
        local_parent.map(str::to_string)
    } else {
        None
    }
}

fn sync_handoff_score(
    state: SyncHandoffState,
    continuity_verified: bool,
    integrity_verified: bool,
    issue_count: usize,
) -> u8 {
    let mut score = 100u8;
    if !continuity_verified {
        score = score.saturating_sub(25);
    }
    if !integrity_verified {
        score = score.saturating_sub(25);
    }
    score = match state {
        SyncHandoffState::Idle | SyncHandoffState::Applied => score,
        SyncHandoffState::PendingOutgoing | SyncHandoffState::PendingIncoming => {
            score.saturating_sub(10)
        },
        SyncHandoffState::Conflict => score.saturating_sub(35),
    };
    score.saturating_sub(
        u8::try_from(issue_count)
            .unwrap_or(u8::MAX)
            .saturating_mul(5),
    )
}

fn build_sync_transport_status(
    neuron_uuid: &str,
    snapshot: Option<SyncTransportEnvelope>,
    outgoing: Option<SyncTransportEnvelope>,
    incoming: Option<SyncTransportEnvelope>,
    mut conflict_paths: Vec<PathBuf>,
) -> Result<Option<SyncTransportStatus>> {
    if snapshot.is_none() && outgoing.is_none() && incoming.is_none() && conflict_paths.is_empty() {
        return Ok(None);
    }
    conflict_paths.sort();
    let snapshot_state = snapshot.as_ref().map(SyncRevisionState::from_envelope);
    let outgoing_state = outgoing.as_ref().map(SyncRevisionState::from_envelope);
    let incoming_state = incoming.as_ref().map(SyncRevisionState::from_envelope);
    let outgoing_pending = outgoing.is_some();
    let incoming_pending = incoming_requires_review(snapshot.as_ref(), incoming.as_ref());
    let handoff = build_sync_handoff_summary(
        snapshot.as_ref(),
        outgoing.as_ref(),
        incoming.as_ref(),
        snapshot_state.as_ref(),
        outgoing_state.as_ref(),
        incoming_state.as_ref(),
        outgoing_pending,
        incoming_pending,
        !conflict_paths.is_empty(),
    );
    Ok(Some(SyncTransportStatus {
        neuron_uuid: neuron_uuid.to_string(),
        snapshot: snapshot_state,
        outgoing: outgoing_state,
        incoming: incoming_state,
        conflict_paths,
        outgoing_pending,
        incoming_pending,
        handoff,
    }))
}

fn collect_transport_json_paths(base: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    let Ok(entries) = std::fs::read_dir(base) else {
        return Ok(paths);
    };

    for entry in entries {
        let path = entry?.path();
        if path.is_dir() {
            let Ok(files) = std::fs::read_dir(&path) else {
                continue;
            };
            for file in files {
                let file_path = file?.path();
                if file_path.extension().and_then(|ext| ext.to_str()) == Some("json") {
                    paths.push(file_path);
                }
            }
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("json") {
            paths.push(path);
        }
    }

    paths.sort();
    Ok(paths)
}

fn transport_uuid_from_path(path: &Path) -> Option<String> {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .map(|stem| stem.to_string())
}

fn bucketed_transport_path(base: &Path, neuron_uuid: &str) -> PathBuf {
    let neuron_uuid = safe_component(neuron_uuid);
    base.join(bucket_component(&neuron_uuid))
        .join(format!("{neuron_uuid}.json"))
}

fn safe_component(value: &str) -> String {
    let safe: String = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
                ch
            } else {
                '_'
            }
        })
        .collect();
    if safe.is_empty() {
        "unknown".to_string()
    } else {
        safe
    }
}

fn bucket_component(value: &str) -> String {
    let safe = safe_component(value);
    let mut chars = safe.chars();
    match (chars.next(), chars.next()) {
        (Some(a), Some(b)) => format!("{a}{b}"),
        (Some(a), None) => format!("{a}_"),
        (None, _) => "__".to_string(),
    }
}

fn revision_component(latest_edit_id: Option<&str>, content_hash: &str) -> String {
    let raw = latest_edit_id.unwrap_or(content_hash);
    safe_component(raw).chars().take(16).collect()
}
