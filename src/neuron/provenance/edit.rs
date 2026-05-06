use serde::{Deserialize, Serialize};

/// Stable author identity used for local-first authorship and future sync merges.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvenanceAuthor {
    pub author_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_id: Option<String>,
}

impl ProvenanceAuthor {
    pub fn new(author_id: impl Into<String>) -> Self {
        Self {
            author_id: author_id.into(),
            display_name: None,
            device_id: None,
        }
    }
}

/// Original authorship for a neuron when known.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorshipRecord {
    pub created_by: ProvenanceAuthor,
    pub created_at: String,
}

/// High-level mutation kind recorded in the provenance log.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ProvenanceOperation {
    #[default]
    Create,
    Update,
    SectionUpdate,
    Rollback,
    Import,
    Merge,
}

impl ProvenanceOperation {
    pub(super) fn as_str(&self) -> &'static str {
        match self {
            Self::Create => "create",
            Self::Update => "update",
            Self::SectionUpdate => "section_update",
            Self::Rollback => "rollback",
            Self::Import => "import",
            Self::Merge => "merge",
        }
    }
}

/// Where the mutation originated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ProvenanceSource {
    #[default]
    Local,
    Sync,
    Import,
    Migration,
}

impl ProvenanceSource {
    pub(super) fn as_str(&self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Sync => "sync",
            Self::Import => "import",
            Self::Migration => "migration",
        }
    }
}

/// A single persisted edit/revision entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvenanceEditRecord {
    pub edit_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_edit_id: Option<String>,
    pub operation: ProvenanceOperation,
    #[serde(default)]
    pub source: ProvenanceSource,
    pub edited_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub author: Option<ProvenanceAuthor>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub section: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_hash: Option<String>,
}

/// Builder-style input for appending a provenance edit.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ProvenanceEdit {
    pub operation: ProvenanceOperation,
    pub source: ProvenanceSource,
    pub author: Option<ProvenanceAuthor>,
    pub section: Option<String>,
    pub summary: Option<String>,
    pub content_hash: Option<String>,
    pub edited_at: Option<String>,
}
