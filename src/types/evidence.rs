//! Structured evidence types for typed fact extraction.
//!
//! An `EvidenceFact` is a typed triple (entity, predicate, value) extracted from
//! neuron content at mine time. Evidence facts are stored in the `## evidence_surface`
//! section of Verbatim neurons and returned via the `cortyx_get_evidence` MCP tool.
//!
//! The 8 `EvidenceFamily` variants generalize the pattern families from LME-500 into
//! domain-agnostic categories usable for any conversation corpus.

use serde::{Deserialize, Serialize};

/// A typed (entity, predicate, value) evidence triple extracted from neuron content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceFact {
    /// The subject of the fact (e.g. "user", "Alice", "project").
    pub entity: String,
    /// The relationship or attribute (e.g. "job", "visited", "prefers").
    pub predicate: String,
    /// The extracted value (e.g. "software engineer", "2023-06-15", "dark mode").
    pub value: String,
    /// Extraction confidence in `[0, 1]`.
    pub confidence: f32,
    /// Typed evidence family.
    pub family: EvidenceFamily,
    /// ISO date or relative expression anchoring this fact in time, if available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temporal_anchor: Option<String>,
    /// Zero-based turn index within the source neuron, if identifiable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_turn: Option<usize>,
}

/// Typed evidence family — the 8 categories that cover the space of
/// conversational memory facts needed for question answering.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceFamily {
    /// When something happened, elapsed intervals, before/after relationships.
    TemporalInterval,
    /// Facts about entities: job, home, pet, degree, relationship, contact.
    EntityFact,
    /// A fact that supersedes a prior value (user changed jobs, moved cities, etc.).
    KnowledgeUpdate,
    /// Preferences: likes, dislikes, favorites, recommendations.
    Preference,
    /// Explicit negations or confirmed absences ("has never been to").
    Absence,
    /// Facts that require joining two or more source turns to answer.
    MultiHop,
    /// Something the assistant explicitly stated or was told.
    AssistantStated,
    /// Aggregate counts: "how many times", "how often", totals.
    AggregateCount,
}
