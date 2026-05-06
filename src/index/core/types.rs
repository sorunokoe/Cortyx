//! Core index types - summary and metadata structures.

use crate::neuron::NeuronKind;
use std::path::PathBuf;

/// Summary of a module as returned by `list_modules()`.
#[derive(Debug, Clone)]
pub struct ModuleSummary {
    pub name: String,
    pub neuron_count: usize,
    pub avg_hit_rate: f32,
    /// True when name starts with `@` (person/project scope).
    pub is_person_scope: bool,
}

/// Summary of a single neuron as returned by `list_neurons()`.
#[derive(Debug, Clone)]
pub struct NeuronSummary {
    pub path: PathBuf,
    pub kind: NeuronKind,
    pub staleness_multiplier: f32,
    pub hit_rate: f32,
    pub use_count: u32,
}

/// Share-ready neuron summary for the git-federated concept library.
#[derive(Debug, Clone)]
pub struct PublishReadySummary {
    pub path: PathBuf,
    pub kind: NeuronKind,
    pub use_count: u32,
    pub hit_rate: f32,
    pub quality_score: f32,
}

/// Lightweight metadata for explainable answer/provenance rendering.
#[derive(Debug, Clone)]
pub struct ContextMetadata {
    pub kind: NeuronKind,
    pub module: Option<String>,
    pub summary: String,
    pub timestamp_secs: Option<i64>,
    pub tokens: usize,
    pub use_count: u32,
    pub hit_count: u32,
    pub hit_rate: f32,
}
