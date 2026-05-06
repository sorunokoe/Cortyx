use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::neuron::provenance::{
    NeuronProvenance, ProvenanceIntegrityExpectation, ProvenanceIntegritySummary,
};

use super::SyncTransportEnvelope;

/// Minimal per-revision state extracted from a transport envelope for later status surfaces.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyncRevisionState {
    pub source_path: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub module: Option<String>,
    pub content_hash: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edit_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_edit_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edited_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub author_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub author_display: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_by_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_by_display: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance_fingerprint: Option<String>,
    #[serde(default)]
    pub revision_count: usize,
    #[serde(default)]
    pub author_count: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    #[serde(default)]
    pub integrity: ProvenanceIntegritySummary,
}

impl SyncRevisionState {
    pub fn from_envelope(envelope: &SyncTransportEnvelope) -> Self {
        let latest_edit = envelope
            .provenance
            .as_ref()
            .and_then(NeuronProvenance::latest_edit);
        let authorship = envelope
            .provenance
            .as_ref()
            .and_then(|provenance| provenance.authorship.as_ref());
        let integrity = envelope
            .provenance
            .as_ref()
            .map(|provenance| {
                provenance.integrity_summary(ProvenanceIntegrityExpectation {
                    neuron_uuid: Some(envelope.neuron_uuid()),
                    source_path: Some(envelope.syncable.headers.source_path.as_path()),
                    latest_edit_id: envelope.syncable.headers.latest_edit_id.as_deref(),
                    parent_edit_id: envelope.syncable.headers.parent_edit_id.as_deref(),
                    content_hash: Some(envelope.syncable.headers.body_hash.as_str()),
                })
            })
            .unwrap_or_else(ProvenanceIntegritySummary::missing);
        Self {
            source_path: envelope.syncable.headers.source_path.clone(),
            module: envelope.syncable.module.clone(),
            content_hash: envelope.syncable.headers.content_hash.clone(),
            edit_id: envelope
                .syncable
                .headers
                .latest_edit_id
                .clone()
                .or_else(|| latest_edit.map(|edit| edit.edit_id.clone())),
            parent_edit_id: envelope.syncable.headers.parent_edit_id.clone(),
            edited_at: latest_edit
                .map(|edit| edit.edited_at.clone())
                .or_else(|| Some(envelope.syncable.headers.last_updated.clone()))
                .filter(|value| !value.trim().is_empty()),
            author_id: latest_edit
                .and_then(|edit| edit.author.as_ref())
                .map(|author| author.author_id.clone()),
            author_display: latest_edit
                .and_then(|edit| edit.author.as_ref())
                .and_then(|author| author.display_name.clone()),
            created_by_id: authorship.map(|authorship| authorship.created_by.author_id.clone()),
            created_by_display: authorship
                .and_then(|authorship| authorship.created_by.display_name.clone()),
            provenance_fingerprint: envelope.syncable.headers.provenance_fingerprint.clone(),
            revision_count: envelope.syncable.headers.revision_count,
            author_count: envelope.syncable.headers.author_count,
            summary: latest_edit.and_then(|edit| edit.summary.clone()),
            integrity,
        }
    }

    pub fn author_id_or_created_by(&self) -> Option<&str> {
        self.author_id.as_deref().or(self.created_by_id.as_deref())
    }

    pub fn author_display_or_created_by(&self) -> Option<&str> {
        self.author_display
            .as_deref()
            .or(self.created_by_display.as_deref())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SyncHandoffState {
    #[default]
    Idle,
    PendingOutgoing,
    PendingIncoming,
    Applied,
    Conflict,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyncHandoffIssue {
    MissingSharedAncestor,
    ProvenanceDiverged,
    LocalIntegrityUnverified,
    RemoteIntegrityUnverified,
    IncomingNotApplied,
    ConflictRecorded,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct SyncHandoffSummary {
    #[serde(default)]
    pub state: SyncHandoffState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shared_edit_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_edit_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remote_edit_id: Option<String>,
    pub continuity_verified: bool,
    pub integrity_verified: bool,
    pub score: u8,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub issues: Vec<SyncHandoffIssue>,
}

impl SyncHandoffSummary {
    pub fn requires_attention(&self) -> bool {
        !self.integrity_verified
            || !self.continuity_verified
            || matches!(
                self.state,
                SyncHandoffState::PendingOutgoing
                    | SyncHandoffState::PendingIncoming
                    | SyncHandoffState::Conflict
            )
    }
}

/// Reusable sync status snapshot for one neuron UUID.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyncTransportStatus {
    pub neuron_uuid: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot: Option<SyncRevisionState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outgoing: Option<SyncRevisionState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub incoming: Option<SyncRevisionState>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub conflict_paths: Vec<PathBuf>,
    #[serde(default)]
    pub outgoing_pending: bool,
    #[serde(default)]
    pub incoming_pending: bool,
    #[serde(default)]
    pub handoff: SyncHandoffSummary,
}

impl SyncTransportStatus {
    pub fn pending_outgoing(&self) -> bool {
        self.outgoing_pending
    }
    pub fn pending_incoming(&self) -> bool {
        self.incoming_pending
    }
    pub fn conflict_count(&self) -> usize {
        self.conflict_paths.len()
    }

    pub fn module(&self) -> Option<&str> {
        self.primary_revision().and_then(|r| r.module.as_deref())
    }

    pub fn source_path(&self) -> Option<&Path> {
        self.primary_revision().map(|r| r.source_path.as_path())
    }

    pub fn latest_edit_id(&self) -> Option<&str> {
        self.latest_revision().and_then(|r| r.edit_id.as_deref())
    }

    pub fn latest_activity_at(&self) -> Option<&str> {
        self.latest_revision().and_then(|r| r.edited_at.as_deref())
    }

    pub fn latest_author_id(&self) -> Option<&str> {
        self.latest_revision()
            .and_then(SyncRevisionState::author_id_or_created_by)
    }

    pub fn latest_author_display(&self) -> Option<&str> {
        self.latest_revision()
            .and_then(SyncRevisionState::author_display_or_created_by)
    }

    pub fn latest_summary(&self) -> Option<&str> {
        self.latest_revision().and_then(|r| r.summary.as_deref())
    }

    pub fn handoff_shared_edit_id(&self) -> Option<&str> {
        self.handoff.shared_edit_id.as_deref()
    }

    pub fn integrity_issue_count(&self) -> usize {
        self.unique_revisions()
            .into_iter()
            .map(|r| r.integrity.issues.len())
            .sum()
    }

    pub fn verified_revision_count(&self) -> usize {
        self.unique_revisions()
            .into_iter()
            .filter(|r| r.integrity.trusted)
            .count()
    }

    pub fn fully_verified(&self) -> bool {
        let revisions = self.unique_revisions();
        !revisions.is_empty()
            && revisions.iter().all(|r| r.integrity.trusted)
            && self.handoff.integrity_verified
            && self.handoff.continuity_verified
            && !matches!(self.handoff.state, SyncHandoffState::Conflict)
    }

    pub fn trust_score(&self) -> f32 {
        let revisions = self.unique_revisions();
        if revisions.is_empty() {
            return self.handoff.score as f32;
        }
        let avg = revisions
            .iter()
            .map(|r| r.integrity.score as f32)
            .sum::<f32>()
            / revisions.len() as f32;
        avg * 0.7 + self.handoff.score as f32 * 0.3
    }

    pub fn requires_trust_attention(&self) -> bool {
        self.handoff.requires_attention() || self.integrity_issue_count() > 0
    }

    pub fn primary_revision(&self) -> Option<&SyncRevisionState> {
        self.snapshot
            .as_ref()
            .or(self.outgoing.as_ref())
            .or(self.incoming.as_ref())
    }

    pub fn latest_revision(&self) -> Option<&SyncRevisionState> {
        let mut best = None;
        for revision in [
            self.outgoing.as_ref(),
            self.snapshot.as_ref(),
            self.incoming.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            let replace = best
                .as_ref()
                .map(|current: &&SyncRevisionState| {
                    revision.edited_at.as_deref() > current.edited_at.as_deref()
                })
                .unwrap_or(true);
            if replace {
                best = Some(revision);
            }
        }
        best.or_else(|| self.primary_revision())
    }

    pub(super) fn unique_revisions(&self) -> Vec<&SyncRevisionState> {
        let mut seen = BTreeSet::new();
        let mut revisions = Vec::new();
        for revision in [
            self.snapshot.as_ref(),
            self.outgoing.as_ref(),
            self.incoming.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            let key = (
                revision.content_hash.clone(),
                revision.edit_id.clone().unwrap_or_default(),
                revision.provenance_fingerprint.clone().unwrap_or_default(),
            );
            if seen.insert(key) {
                revisions.push(revision);
            }
        }
        revisions
    }
}

/// Aggregate trust/integrity metrics for a set of sync statuses.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SyncTrustMetrics {
    pub neuron_count: usize,
    pub revision_count: usize,
    pub trusted_revision_count: usize,
    pub fingerprinted_revision_count: usize,
    pub fully_verified_neuron_count: usize,
    pub continuity_verified_count: usize,
    pub integrity_verified_count: usize,
    pub shared_ancestor_count: usize,
    pub pending_outgoing_count: usize,
    pub pending_incoming_count: usize,
    pub conflict_count: usize,
    pub integrity_issue_count: usize,
    pub trust_attention_count: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub average_trust_score: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub average_handoff_score: Option<f32>,
}

pub fn summarize_sync_trust(statuses: &[SyncTransportStatus]) -> SyncTrustMetrics {
    let mut metrics = SyncTrustMetrics {
        neuron_count: statuses.len(),
        ..Default::default()
    };
    let mut trust_scores = Vec::new();
    let mut handoff_scores = Vec::new();

    for status in statuses {
        let revisions = status.unique_revisions();
        metrics.revision_count += revisions.len();
        metrics.trusted_revision_count += revisions.iter().filter(|r| r.integrity.trusted).count();
        metrics.fingerprinted_revision_count += revisions
            .iter()
            .filter(|r| {
                r.provenance_fingerprint
                    .as_deref()
                    .map(str::trim)
                    .filter(|f| !f.is_empty())
                    .is_some()
            })
            .count();
        metrics.fully_verified_neuron_count += status.fully_verified() as usize;
        metrics.continuity_verified_count += status.handoff.continuity_verified as usize;
        metrics.integrity_verified_count += status.handoff.integrity_verified as usize;
        metrics.shared_ancestor_count += status.handoff_shared_edit_id().is_some() as usize;
        metrics.pending_outgoing_count += status.pending_outgoing() as usize;
        metrics.pending_incoming_count += status.pending_incoming() as usize;
        metrics.conflict_count += status.conflict_count();
        metrics.integrity_issue_count += status.integrity_issue_count();
        metrics.trust_attention_count += status.requires_trust_attention() as usize;
        trust_scores.push(status.trust_score());
        handoff_scores.push(status.handoff.score as f32);
    }

    metrics.average_trust_score = average_scores(&trust_scores);
    metrics.average_handoff_score = average_scores(&handoff_scores);
    metrics
}

fn average_scores(values: &[f32]) -> Option<f32> {
    (!values.is_empty()).then_some(values.iter().sum::<f32>() / values.len() as f32)
}
