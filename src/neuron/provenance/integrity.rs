use std::path::Path;

use serde::{Deserialize, Serialize};

use super::edit::ProvenanceEditRecord;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ProvenanceIntegrityExpectation<'a> {
    pub neuron_uuid: Option<&'a str>,
    pub source_path: Option<&'a Path>,
    pub latest_edit_id: Option<&'a str>,
    pub parent_edit_id: Option<&'a str>,
    pub content_hash: Option<&'a str>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProvenanceIntegrityIssue {
    MissingProvenance,
    MissingAuthorship,
    MissingLatestAuthor,
    MissingLatestContentHash,
    ContentHashMismatch,
    BrokenParentChain,
    TimestampRegression,
    LatestEditMismatch,
    ParentEditMismatch,
    NeuronUuidMismatch,
    SourcePathMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ProvenanceIntegritySummary {
    pub trusted: bool,
    pub score: u8,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fingerprint: Option<String>,
    pub revision_count: usize,
    pub author_count: usize,
    pub authorship_present: bool,
    pub latest_author_present: bool,
    pub identity_verified: bool,
    pub content_verified: bool,
    pub chain_verified: bool,
    pub timestamps_monotonic: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub issues: Vec<ProvenanceIntegrityIssue>,
}

impl ProvenanceIntegritySummary {
    #[must_use]
    pub fn missing() -> Self {
        Self {
            trusted: false,
            score: 0,
            issues: vec![ProvenanceIntegrityIssue::MissingProvenance],
            ..Default::default()
        }
    }
}

// ─── Verification helpers ─────────────────────────────────────────────────────

pub(super) fn verify_provenance_identity(
    neuron_uuid: Option<&str>,
    source_path: Option<&Path>,
    expectation: ProvenanceIntegrityExpectation<'_>,
    issues: &mut Vec<ProvenanceIntegrityIssue>,
) -> bool {
    let expected_uuid = expectation
        .neuron_uuid
        .map(str::trim)
        .filter(|uuid| !uuid.is_empty());
    let expected_source_path = expectation.source_path.map(path_to_string);
    let mut ok = true;

    if let Some(expected_uuid) = expected_uuid {
        if neuron_uuid != Some(expected_uuid) {
            ok = false;
            issues.push(ProvenanceIntegrityIssue::NeuronUuidMismatch);
        }
    }
    if let Some(expected_source_path) = expected_source_path {
        if source_path.map(path_to_string) != Some(expected_source_path) {
            ok = false;
            issues.push(ProvenanceIntegrityIssue::SourcePathMismatch);
        }
    }

    ok
}

pub(super) fn verify_provenance_chain(
    history: &[ProvenanceEditRecord],
    issues: &mut Vec<ProvenanceIntegrityIssue>,
) -> (bool, bool) {
    let mut chain_verified = true;
    let mut timestamps_monotonic = true;

    for pair in history.windows(2) {
        let previous = &pair[0];
        let current = &pair[1];
        if current.parent_edit_id.as_deref() != Some(previous.edit_id.as_str()) {
            chain_verified = false;
            if !issues.contains(&ProvenanceIntegrityIssue::BrokenParentChain) {
                issues.push(ProvenanceIntegrityIssue::BrokenParentChain);
            }
        }
        if current.edited_at < previous.edited_at {
            timestamps_monotonic = false;
            if !issues.contains(&ProvenanceIntegrityIssue::TimestampRegression) {
                issues.push(ProvenanceIntegrityIssue::TimestampRegression);
            }
        }
    }

    (chain_verified, timestamps_monotonic)
}

pub(super) fn verify_provenance_content(
    latest_edit: Option<&ProvenanceEditRecord>,
    expected_content_hash: Option<&str>,
    issues: &mut Vec<ProvenanceIntegrityIssue>,
) -> bool {
    let expected = expected_content_hash
        .map(str::trim)
        .filter(|h| !h.is_empty());
    let actual = latest_edit
        .and_then(|edit| edit.content_hash.as_deref())
        .map(str::trim)
        .filter(|h| !h.is_empty());

    let Some(expected) = expected else {
        return actual.is_some();
    };
    let Some(actual) = actual else {
        if latest_edit.is_some() {
            issues.push(ProvenanceIntegrityIssue::MissingLatestContentHash);
        }
        return false;
    };
    if actual != expected {
        issues.push(ProvenanceIntegrityIssue::ContentHashMismatch);
        return false;
    }
    true
}

pub(super) fn verify_provenance_header_links(
    latest: Option<&ProvenanceEditRecord>,
    expectation: ProvenanceIntegrityExpectation<'_>,
    issues: &mut Vec<ProvenanceIntegrityIssue>,
) {
    let expected_latest = expectation
        .latest_edit_id
        .map(str::trim)
        .filter(|id| !id.is_empty());
    if let Some(expected_latest) = expected_latest {
        if latest.map(|edit| edit.edit_id.as_str()) != Some(expected_latest) {
            issues.push(ProvenanceIntegrityIssue::LatestEditMismatch);
        }
    }

    let expected_parent = expectation
        .parent_edit_id
        .map(str::trim)
        .filter(|id| !id.is_empty());
    if let Some(expected_parent) = expected_parent {
        if latest.and_then(|edit| edit.parent_edit_id.as_deref()) != Some(expected_parent) {
            issues.push(ProvenanceIntegrityIssue::ParentEditMismatch);
        }
    }
}

pub(super) fn provenance_integrity_score(
    has_latest_edit: bool,
    authorship_present: bool,
    latest_author_present: bool,
    identity_verified: bool,
    content_verified: bool,
    chain_verified: bool,
    timestamps_monotonic: bool,
    has_history: bool,
    has_fingerprint: bool,
) -> u8 {
    let content_score = if has_latest_edit {
        if content_verified {
            20
        } else {
            0
        }
    } else {
        10
    };
    let history_score = if has_history { 5 } else { 0 };

    (identity_verified as u8) * 20
        + (chain_verified as u8) * 20
        + (timestamps_monotonic as u8) * 10
        + content_score
        + (authorship_present as u8) * 10
        + ((!has_latest_edit || latest_author_present) as u8) * 10
        + (has_fingerprint as u8) * 5
        + history_score
}

// Kept private to this module — used only by the verify_ functions above.
fn path_to_string(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}
