//! BM25 entry type - per-neuron indexing data.

use crate::index::core::config::{DEFAULT_QUALITY_SCORE, DEFAULT_STALENESS};
use crate::neuron::{NeuronKind, Synapse, DEFAULT_CONFIDENCE};
use crate::types::TermFrequency;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

fn default_confidence() -> f32 {
    DEFAULT_CONFIDENCE
}

fn default_staleness() -> f32 {
    DEFAULT_STALENESS
}

fn default_quality_score() -> f32 {
    DEFAULT_QUALITY_SCORE
}

/// Per-neuron data stored in the in-memory BM25 index.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub(crate) struct BM25Entry {
    pub neuron_path: PathBuf,
    pub kind: NeuronKind,

    /// Term → raw frequency within this document
    pub term_freq: HashMap<String, TermFrequency>,

    /// Total number of terms in this document (for BM25 length normalization)
    pub term_count: usize,

    /// LLM token estimate (from `estimate_tokens`), used for budget trimming
    pub tokens: usize,

    /// Tokenized task pattern (use-case neurons only)
    pub task_pattern_terms: Vec<String>,

    /// Parent core neuron path (use-case neurons only)
    pub parent: Option<PathBuf>,

    /// Typed synapse edges — persisted so weights survive restarts
    pub synapses: Vec<Synapse>,

    /// Source files synthesized by this Concept neuron
    pub source_files: Vec<PathBuf>,

    /// Optional module/namespace tag for namespace-filtered queries
    pub module: Option<String>,

    /// Git-derived confidence score applied as a mild BM25 multiplier.
    /// 1.0 = committed + unmodified (neutral). 0.9 = locally modified. 0.85 = untracked.
    #[serde(default = "default_confidence")]
    pub confidence_score: f32,

    /// Activation count — incremented each time this neuron is returned by get_contexts.
    #[serde(default)]
    pub use_count: u32,

    /// Citation count — incremented by cortyx_record_hit when the LLM confirms it used the neuron.
    #[serde(default)]
    pub hit_count: u32,

    /// Staleness multiplier (1.0 = fresh, 0.5 = stale). Demotes rather than evicts stale neurons
    /// so context is preserved; stale neurons can still activate for niche queries.
    #[serde(default = "default_staleness")]
    pub staleness_multiplier: f32,

    /// Concept cloud: union of significant identifier terms from this neuron's 1-hop
    /// structural neighbours (Calls, Imports, Implements edges). Built by `build_concept_clouds()`
    /// during `rebuild_derived()`. At query time, used as a graph-aware semantic thesaurus
    /// for zero/low-confidence BM25 queries — no external model required (TRIZ R12-S1).
    /// Not persisted: rebuilt from the live synapse graph on every load.
    #[serde(skip)]
    pub concept_cloud: Vec<String>,

    /// B2: Synonym cloud — terms that have co-activated with this neuron ≥30 times.
    ///
    /// Populated by `record_coactivation()`. Persisted so the signal accumulates across
    /// sessions. At query time, query terms are expanded through synonym clouds before
    /// BM25 scoring — improving recall for semantically related but lexically distant queries.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub synonym_cloud: Vec<String>,

    /// S-II (R16): 64-bit SimHash fingerprint of this neuron's TF-IDF term weights.
    ///
    /// R17 Sol4 upgrade: 1024-bit SimHash ensemble via 16 independent seeds.
    /// This is an empirical locality-sensitive fallback, not a Johnson-Lindenstrauss guarantee.
    /// LSH match: ANY of the 16 fingerprint pairs within Hamming ≤ 14.
    /// Migration from v7: old `lsh_fingerprint: u64` loaded via serde → replicated to [0].
    #[serde(default)]
    pub lsh_fingerprints: [u64; 4],

    /// S-III (R16): Self-quality score — fraction of neuron terms that overlap with
    /// the corresponding source file's AST terms.
    ///
    /// Computed at `index_neuron` time: `|neuron_terms ∩ source_ast_terms| / |neuron_terms|`.
    /// When `quality_score < 0.4`, a ×0.7 BM25 penalty is applied to demote stale neurons
    /// without evicting them. Surfaced in `cortyx status` as "needs curation" count.
    /// Defaults to 1.0 (neutral/unknown) when no source file is available.
    #[serde(default = "default_quality_score")]
    pub quality_score: f32,

    /// S-I (R16): Tier-1 summary for multi-resolution emission.
    ///
    /// Extracted from `## purpose` + first line of `## pitfalls` at `index_neuron` time.
    /// Emitted instead of full content when BM25 score is in the 1.5–5.0 range (Tier 1).
    /// ~50 tokens; avoids a disk read at query time. Not persisted (rebuilt from neuron file).
    #[serde(skip)]
    pub summary: String,

    /// Unix epoch seconds parsed from `NeuronMeta.timestamp` (Verbatim neurons only).
    ///
    /// Stored at `index_neuron` time so temporal query routing can apply a recency boost
    /// without any disk I/O at query time. Code neurons (no ISO 8601 timestamp) leave
    /// this as `None` — they are unaffected by temporal scoring.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp_secs: Option<i64>,

    /// Stored at index time so named-person relocation queries do not read neuron files
    /// from disk in the retrieval hot path.
    #[serde(default)]
    pub has_move_residence_evidence: bool,

    /// R21 T6: Session identifier for session-level grouping.
    ///
    /// Derived from the neuron filename stem (e.g., "lme_0060" from
    /// "lme_0060_0_user.verbatim.md"). Empty for non-Verbatim neurons.
    /// Used at retrieval time: when a neuron enters the top-3, its session
    /// siblings are injected as overflow candidates.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub session_id: String,
}
