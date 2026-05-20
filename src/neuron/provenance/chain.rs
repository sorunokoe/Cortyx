use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use super::super::meta::NeuronMeta;
use super::edit::{
    AuthorshipRecord, ProvenanceAuthor, ProvenanceEdit, ProvenanceEditRecord, ProvenanceOperation,
    ProvenanceSource,
};
use super::integrity::{
    provenance_integrity_score, verify_provenance_chain, verify_provenance_content,
    verify_provenance_header_links, verify_provenance_identity, ProvenanceIntegrityExpectation,
    ProvenanceIntegrityIssue, ProvenanceIntegritySummary,
};

static PROVENANCE_EDIT_COUNTER: AtomicU64 = AtomicU64::new(0);

pub const PROVENANCE_VERSION: u32 = 1;
pub const PROVENANCE_HISTORY_LIMIT: usize = 64;
const PROVENANCE_FINGERPRINT_LEN: usize = 32;

fn default_provenance_version() -> u32 {
    PROVENANCE_VERSION
}

/// Additive provenance sidecar kept separate from the retrieval/index hot path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeuronProvenance {
    #[serde(default = "default_provenance_version")]
    pub version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub neuron_uuid: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_path: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authorship: Option<AuthorshipRecord>,
    #[serde(default, alias = "history", skip_serializing_if = "Vec::is_empty")]
    pub edit_history: Vec<ProvenanceEditRecord>,
}

impl Default for NeuronProvenance {
    fn default() -> Self {
        Self {
            version: PROVENANCE_VERSION,
            neuron_uuid: None,
            source_path: None,
            authorship: None,
            edit_history: Vec::new(),
        }
    }
}

impl NeuronProvenance {
    #[must_use]
    pub fn from_meta(meta: &NeuronMeta) -> Self {
        let mut provenance = Self::default();
        provenance.sync_from_meta(meta);
        provenance
    }

    pub fn sync_from_meta(&mut self, meta: &NeuronMeta) {
        self.version = PROVENANCE_VERSION;
        if meta.uuid.is_some() {
            self.neuron_uuid = meta.uuid.clone();
        }
        self.source_path = Some(meta.source_path.clone());
    }

    #[must_use]
    pub fn latest_edit(&self) -> Option<&ProvenanceEditRecord> {
        self.edit_history.last()
    }

    #[must_use]
    pub fn author_count(&self) -> usize {
        let mut authors = BTreeSet::new();
        if let Some(authorship) = &self.authorship {
            let id = authorship.created_by.author_id.trim();
            if !id.is_empty() {
                authors.insert(id.to_string());
            }
        }
        for edit in &self.edit_history {
            if let Some(id) = edit
                .author
                .as_ref()
                .map(|a| a.author_id.trim())
                .filter(|id| !id.is_empty())
            {
                authors.insert(id.to_string());
            }
        }
        authors.len()
    }

    pub fn fingerprint(&self) -> String {
        let input = ProvenanceFingerprintInput {
            version: self.version,
            neuron_uuid: self.neuron_uuid.as_deref(),
            source_path: self.source_path.as_deref().map(path_to_string),
            authorship: self
                .authorship
                .as_ref()
                .map(ProvenanceFingerprintAuthorship::from),
            edit_history: self
                .edit_history
                .iter()
                .map(ProvenanceFingerprintEdit::from)
                .collect(),
        };
        blake3::hash(&serde_json::to_vec(&input).unwrap_or_default()).to_hex()
            [..PROVENANCE_FINGERPRINT_LEN]
            .to_string()
    }

    #[must_use]
    pub fn shared_ancestor_edit_id(&self, other: &Self) -> Option<String> {
        let other_ids: BTreeSet<&str> = other
            .edit_history
            .iter()
            .map(|e| e.edit_id.as_str())
            .collect();
        self.edit_history
            .iter()
            .rev()
            .find(|e| other_ids.contains(e.edit_id.as_str()))
            .map(|e| e.edit_id.clone())
    }

    #[must_use]
    pub fn integrity_summary(
        &self,
        expectation: ProvenanceIntegrityExpectation<'_>,
    ) -> ProvenanceIntegritySummary {
        let latest = self.latest_edit();
        let authorship_present = self.authorship.is_some();
        let latest_author_present = latest.and_then(|e| e.author.as_ref()).is_some();
        let mut issues = Vec::new();

        let identity_verified = verify_provenance_identity(
            self.neuron_uuid.as_deref(),
            self.source_path.as_deref(),
            expectation,
            &mut issues,
        );
        let (chain_verified, timestamps_monotonic) =
            verify_provenance_chain(&self.edit_history, &mut issues);
        let content_verified =
            verify_provenance_content(latest, expectation.content_hash, &mut issues);
        verify_provenance_header_links(latest, expectation, &mut issues);

        if !authorship_present {
            issues.push(ProvenanceIntegrityIssue::MissingAuthorship);
        }
        if latest.is_some() && !latest_author_present {
            issues.push(ProvenanceIntegrityIssue::MissingLatestAuthor);
        }

        let revision_count = self.edit_history.len();
        let fingerprint = Some(self.fingerprint());
        let score = provenance_integrity_score(
            latest.is_some(),
            authorship_present,
            latest_author_present,
            identity_verified,
            content_verified,
            chain_verified,
            timestamps_monotonic,
            revision_count > 0,
            fingerprint.is_some(),
        );
        let trusted = fingerprint.is_some()
            && identity_verified
            && chain_verified
            && timestamps_monotonic
            && (latest.is_none() || content_verified)
            && issues.iter().all(|issue| {
                !matches!(
                    issue,
                    ProvenanceIntegrityIssue::MissingProvenance
                        | ProvenanceIntegrityIssue::MissingLatestContentHash
                        | ProvenanceIntegrityIssue::ContentHashMismatch
                        | ProvenanceIntegrityIssue::BrokenParentChain
                        | ProvenanceIntegrityIssue::TimestampRegression
                        | ProvenanceIntegrityIssue::LatestEditMismatch
                        | ProvenanceIntegrityIssue::ParentEditMismatch
                        | ProvenanceIntegrityIssue::NeuronUuidMismatch
                        | ProvenanceIntegrityIssue::SourcePathMismatch
                )
            })
            && (revision_count > 0 || authorship_present);

        ProvenanceIntegritySummary {
            trusted,
            score,
            fingerprint,
            revision_count,
            author_count: self.author_count(),
            authorship_present,
            latest_author_present,
            identity_verified,
            content_verified,
            chain_verified,
            timestamps_monotonic,
            issues,
        }
    }

    pub fn append_edit(&mut self, edit: ProvenanceEdit) -> &ProvenanceEditRecord {
        let ProvenanceEdit {
            operation,
            source,
            author,
            section,
            summary,
            content_hash,
            edited_at,
        } = edit;

        let edited_at = edited_at.unwrap_or_else(super::super::util::now_iso8601);
        if self.authorship.is_none() && matches!(operation, ProvenanceOperation::Create) {
            if let Some(created_by) = author.clone() {
                self.authorship = Some(AuthorshipRecord {
                    created_by,
                    created_at: edited_at.clone(),
                });
            }
        }

        let parent_edit_id = self.edit_history.last().map(|e| e.edit_id.clone());
        let edit_id = generate_edit_id(
            self.neuron_uuid.as_deref(),
            &operation,
            &source,
            &edited_at,
            author.as_ref(),
            section.as_deref(),
            summary.as_deref(),
            content_hash.as_deref(),
            parent_edit_id.as_deref(),
        );
        self.edit_history.push(ProvenanceEditRecord {
            edit_id,
            parent_edit_id,
            operation,
            source,
            edited_at,
            author,
            section,
            summary,
            content_hash,
        });
        if self.edit_history.len() > PROVENANCE_HISTORY_LIMIT {
            let excess = self.edit_history.len() - PROVENANCE_HISTORY_LIMIT;
            self.edit_history.drain(0..excess);
        }
        &self.edit_history[self.edit_history.len() - 1]
    }
}

// ─── Fingerprint helpers ──────────────────────────────────────────────────────

#[derive(Serialize)]
struct ProvenanceFingerprintInput<'a> {
    version: u32,
    neuron_uuid: Option<&'a str>,
    source_path: Option<String>,
    authorship: Option<ProvenanceFingerprintAuthorship<'a>>,
    edit_history: Vec<ProvenanceFingerprintEdit<'a>>,
}

#[derive(Serialize)]
struct ProvenanceFingerprintAuthorship<'a> {
    created_by: ProvenanceFingerprintAuthor<'a>,
    created_at: &'a str,
}

impl<'a> From<&'a AuthorshipRecord> for ProvenanceFingerprintAuthorship<'a> {
    fn from(a: &'a AuthorshipRecord) -> Self {
        Self {
            created_by: ProvenanceFingerprintAuthor::from(&a.created_by),
            created_at: &a.created_at,
        }
    }
}

#[derive(Serialize)]
struct ProvenanceFingerprintAuthor<'a> {
    author_id: &'a str,
    display_name: Option<&'a str>,
    device_id: Option<&'a str>,
}

impl<'a> From<&'a ProvenanceAuthor> for ProvenanceFingerprintAuthor<'a> {
    fn from(a: &'a ProvenanceAuthor) -> Self {
        Self {
            author_id: &a.author_id,
            display_name: a.display_name.as_deref(),
            device_id: a.device_id.as_deref(),
        }
    }
}

#[derive(Serialize)]
struct ProvenanceFingerprintEdit<'a> {
    edit_id: &'a str,
    parent_edit_id: Option<&'a str>,
    operation: &'a ProvenanceOperation,
    source: &'a ProvenanceSource,
    edited_at: &'a str,
    author: Option<ProvenanceFingerprintAuthor<'a>>,
    section: Option<&'a str>,
    summary: Option<&'a str>,
    content_hash: Option<&'a str>,
}

impl<'a> From<&'a ProvenanceEditRecord> for ProvenanceFingerprintEdit<'a> {
    fn from(e: &'a ProvenanceEditRecord) -> Self {
        Self {
            edit_id: &e.edit_id,
            parent_edit_id: e.parent_edit_id.as_deref(),
            operation: &e.operation,
            source: &e.source,
            edited_at: &e.edited_at,
            author: e.author.as_ref().map(ProvenanceFingerprintAuthor::from),
            section: e.section.as_deref(),
            summary: e.summary.as_deref(),
            content_hash: e.content_hash.as_deref(),
        }
    }
}

fn path_to_string(path: &std::path::Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn generate_edit_id(
    neuron_uuid: Option<&str>,
    operation: &ProvenanceOperation,
    source: &ProvenanceSource,
    edited_at: &str,
    author: Option<&ProvenanceAuthor>,
    section: Option<&str>,
    summary: Option<&str>,
    content_hash: Option<&str>,
    parent_edit_id: Option<&str>,
) -> String {
    let nonce = PROVENANCE_EDIT_COUNTER.fetch_add(1, Ordering::Relaxed);
    let author_id = author.map(|a| a.author_id.as_str()).unwrap_or("");
    let input = format!(
        "{}:{}:{}:{}:{}:{}:{}:{}:{}:{}",
        neuron_uuid.unwrap_or(""),
        operation.as_str(),
        source.as_str(),
        edited_at,
        author_id,
        section.unwrap_or(""),
        summary.unwrap_or(""),
        content_hash.unwrap_or(""),
        parent_edit_id.unwrap_or(""),
        nonce
    );
    blake3::hash(input.as_bytes()).to_hex()[..32].to_string()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::super::kind::NeuronKind;
    use super::*;
    use std::path::Path;

    fn test_meta(source_path: &str) -> NeuronMeta {
        let mut meta = NeuronMeta::new_stub(Path::new(source_path), NeuronKind::Core);
        meta.uuid = Some("uuid-1234".to_string());
        meta
    }

    fn test_author() -> ProvenanceAuthor {
        ProvenanceAuthor {
            author_id: "local:alice@macbook".to_string(),
            display_name: Some("Alice".to_string()),
            device_id: Some("macbook".to_string()),
        }
    }

    fn provenance_with_history(meta: &NeuronMeta) -> (NeuronProvenance, String, String) {
        let author = test_author();
        let mut provenance = NeuronProvenance::from_meta(meta);
        provenance.authorship = Some(AuthorshipRecord {
            created_by: author.clone(),
            created_at: "2026-01-02T03:04:05Z".to_string(),
        });
        let first = provenance
            .append_edit(ProvenanceEdit {
                operation: ProvenanceOperation::Create,
                source: ProvenanceSource::Local,
                author: Some(author.clone()),
                summary: Some("bootstrap neuron".to_string()),
                content_hash: Some("ctx-1".to_string()),
                edited_at: Some("2026-01-02T03:04:05Z".to_string()),
                ..Default::default()
            })
            .edit_id
            .clone();
        let second = provenance
            .append_edit(ProvenanceEdit {
                operation: ProvenanceOperation::Update,
                source: ProvenanceSource::Local,
                author: Some(author),
                summary: Some("refresh neuron".to_string()),
                content_hash: Some("ctx-2".to_string()),
                edited_at: Some("2026-01-02T03:04:06Z".to_string()),
                ..Default::default()
            })
            .edit_id
            .clone();
        (provenance, first, second)
    }

    #[test]
    fn append_edit_caps_history() {
        let meta = test_meta("src/lib.rs");
        let mut provenance = NeuronProvenance::from_meta(&meta);

        for idx in 0..(PROVENANCE_HISTORY_LIMIT + 2) {
            provenance.append_edit(ProvenanceEdit {
                operation: ProvenanceOperation::Update,
                summary: Some(format!("edit {idx}")),
                edited_at: Some(format!("2026-01-02T03:04:{idx:02}Z")),
                ..Default::default()
            });
        }

        let expected_last = format!("edit {}", PROVENANCE_HISTORY_LIMIT + 1);
        assert_eq!(provenance.edit_history.len(), PROVENANCE_HISTORY_LIMIT);
        assert_eq!(
            provenance.edit_history[0].summary.as_deref(),
            Some("edit 2")
        );
        assert_eq!(
            provenance
                .edit_history
                .last()
                .and_then(|e| e.summary.as_deref()),
            Some(expected_last.as_str())
        );
    }

    #[test]
    fn provenance_defaults_legacy_payloads() {
        let provenance: NeuronProvenance =
            serde_json::from_str(r#"{ "source_path": "src/lib.rs" }"#).unwrap();
        assert_eq!(provenance.version, PROVENANCE_VERSION);
        assert!(provenance.edit_history.is_empty());
        assert!(provenance.authorship.is_none());
    }

    #[test]
    fn provenance_integrity_summary_surfaces_trusted_chain() {
        let meta = test_meta("src/engine.rs");
        let (provenance, first_edit_id, second_edit_id) = provenance_with_history(&meta);

        let summary = provenance.integrity_summary(ProvenanceIntegrityExpectation {
            neuron_uuid: meta.uuid.as_deref(),
            source_path: Some(meta.source_path.as_path()),
            latest_edit_id: Some(second_edit_id.as_str()),
            parent_edit_id: Some(first_edit_id.as_str()),
            content_hash: Some("ctx-2"),
        });

        assert!(summary.trusted);
        assert_eq!(summary.score, 100);
        assert_eq!(summary.revision_count, 2);
        assert_eq!(summary.author_count, 1);
        assert!(summary.fingerprint.is_some());
        assert!(summary.issues.is_empty());

        let mut remote = provenance.clone();
        remote.append_edit(ProvenanceEdit {
            operation: ProvenanceOperation::Merge,
            source: ProvenanceSource::Sync,
            author: Some(test_author()),
            summary: Some("merge remote changes".to_string()),
            content_hash: Some("ctx-3".to_string()),
            edited_at: Some("2026-01-02T03:04:07Z".to_string()),
            ..Default::default()
        });
        assert_eq!(
            provenance.shared_ancestor_edit_id(&remote).as_deref(),
            Some(second_edit_id.as_str())
        );
    }

    #[test]
    fn provenance_integrity_summary_flags_broken_chain() {
        let meta = test_meta("src/engine.rs");
        let (mut provenance, _, _) = provenance_with_history(&meta);
        provenance.source_path = Some(PathBuf::from("src/other.rs"));
        provenance.edit_history[1].parent_edit_id = Some("other-base".to_string());
        provenance.edit_history[1].content_hash = Some("ctx-other".to_string());
        provenance.edit_history[1].edited_at = "2026-01-02T03:04:04Z".to_string();

        let summary = provenance.integrity_summary(ProvenanceIntegrityExpectation {
            neuron_uuid: meta.uuid.as_deref(),
            source_path: Some(meta.source_path.as_path()),
            latest_edit_id: Some(provenance.edit_history[1].edit_id.as_str()),
            parent_edit_id: Some("wrong-parent"),
            content_hash: Some("ctx-2"),
        });

        assert!(!summary.trusted);
        assert!(summary
            .issues
            .contains(&ProvenanceIntegrityIssue::SourcePathMismatch));
        assert!(summary
            .issues
            .contains(&ProvenanceIntegrityIssue::BrokenParentChain));
        assert!(summary
            .issues
            .contains(&ProvenanceIntegrityIssue::TimestampRegression));
        assert!(summary
            .issues
            .contains(&ProvenanceIntegrityIssue::ContentHashMismatch));
        assert!(summary
            .issues
            .contains(&ProvenanceIntegrityIssue::ParentEditMismatch));
    }
}
