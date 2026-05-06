use serde::{Deserialize, Serialize};

/// The role a neuron plays in the knowledge graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum NeuronKind {
    #[default]
    /// Per-file AI-curated context stub — the primary neuron type.
    Core,
    /// Proven task-specific chunk extracted from a raw source.
    UseCase,
    /// Raw conversation turn mined verbatim (no LLM curation needed).
    Verbatim,
    /// Cross-file synthesized concept (e.g. "JWT auth flow").
    Concept,
    /// One per project — top-level overview neuron.
    Project,
    /// Mine-time cross-session aggregate: pre-computed count and context snippets
    /// for entities/topics mentioned in ≥3 distinct sessions.
    Aggregate,
}

/// Lifecycle state of a neuron.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum NeuronStatus {
    #[default]
    /// Just created — content is a TODO placeholder, not yet useful for retrieval.
    Stub,
    /// Content has been set by the LLM — ready for activation.
    Fresh,
    /// Source file changed — content may be outdated; re-evolve recommended.
    Stale,
}
