//! BM25 entry type - per-neuron indexing data.

use super::interner::TermInterner;
use crate::index::core::config::{DEFAULT_QUALITY_SCORE, DEFAULT_STALENESS};
use crate::neuron::{NeuronKind, Synapse, DEFAULT_CONFIDENCE};
use crate::types::TermFrequency;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

fn default_confidence() -> f32 {
    DEFAULT_CONFIDENCE
}

fn default_staleness() -> f32 {
    DEFAULT_STALENESS
}

fn default_quality_score() -> f32 {
    DEFAULT_QUALITY_SCORE
}

pub(crate) fn serialize_atomic_u32<S>(value: &AtomicU32, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_u32(value.load(Ordering::Relaxed))
}

pub(crate) fn deserialize_atomic_u32<'de, D>(deserializer: D) -> Result<AtomicU32, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(AtomicU32::new(u32::deserialize(deserializer)?))
}

/// Per-neuron data stored in the in-memory BM25 index.
#[derive(Debug, Default, Serialize, Deserialize)]
pub(crate) struct BM25Entry {
    pub neuron_path: PathBuf,
    pub kind: NeuronKind,

    /// Term → raw frequency within this document
    pub term_freq: HashMap<String, TermFrequency>,

    /// Cache-friendly `u32`-keyed term frequencies, rebuilt from `term_freq` at load time.
    /// Never serialized — purely in-memory for hot-path BM25 scoring.
    #[serde(skip)]
    pub hot_term_freq: HashMap<u32, TermFrequency>,

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

    /// Runtime query counter. Authoritative source of truth during a live session.
    /// At load time, this value is overwritten from [`crate::neuron::NeuronMeta`]; see `persistence.rs`.
    #[serde(
        default,
        serialize_with = "serialize_atomic_u32",
        deserialize_with = "deserialize_atomic_u32"
    )]
    pub use_count: AtomicU32,

    /// Runtime citation counter. Authoritative source of truth during a live session.
    /// At load time, this value is overwritten from [`crate::neuron::NeuronMeta`]; see `persistence.rs`.
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

    /// S-II (R16): 256-bit SimHash ensemble via 4 independent hash seeds.
    ///
    /// Four 64-bit SimHash planes (256 effective bits) provide locality-sensitive
    /// approximate-nearest-neighbour fallback when BM25 returns < 2 candidates.
    /// The 4 seeds are drawn from `LSH_SEEDS[..4]`; the remaining 12 seeds are reserved.
    ///
    /// **Note on independence:** The 16 entries in `LSH_SEEDS` vary only the FNV-1a
    /// offset basis while sharing the multiplier (`0x100000001b3`). This produces
    /// correlated rather than fully independent hash functions. The ensemble is used
    /// only as a coarse fallback bridge — false-positive candidates are filtered by
    /// downstream BM25 re-ranking before reaching the caller.
    ///
    /// Migration from v7: old `lsh_fingerprint: u64` is loaded via serde and replicated to `[0]`.
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

impl Clone for BM25Entry {
    fn clone(&self) -> Self {
        Self {
            neuron_path: self.neuron_path.clone(),
            kind: self.kind.clone(),
            term_freq: self.term_freq.clone(),
            hot_term_freq: self.hot_term_freq.clone(),
            term_count: self.term_count,
            tokens: self.tokens,
            task_pattern_terms: self.task_pattern_terms.clone(),
            parent: self.parent.clone(),
            synapses: self.synapses.clone(),
            source_files: self.source_files.clone(),
            module: self.module.clone(),
            confidence_score: self.confidence_score,
            use_count: AtomicU32::new(self.use_count.load(Ordering::Relaxed)),
            hit_count: self.hit_count,
            staleness_multiplier: self.staleness_multiplier,
            concept_cloud: self.concept_cloud.clone(),
            synonym_cloud: self.synonym_cloud.clone(),
            lsh_fingerprints: self.lsh_fingerprints,
            quality_score: self.quality_score,
            summary: self.summary.clone(),
            timestamp_secs: self.timestamp_secs,
            has_move_residence_evidence: self.has_move_residence_evidence,
            session_id: self.session_id.clone(),
        }
    }
}

impl BM25Entry {
    pub fn build_hot_terms(&mut self, interner: &mut TermInterner) {
        self.hot_term_freq.clear();
        self.hot_term_freq.reserve(self.term_freq.len());
        for (term, freq) in &self.term_freq {
            let term_id = interner.intern(term);
            self.hot_term_freq.insert(term_id, *freq);
        }
    }

    pub fn increment_use_count(&self) -> u32 {
        self.use_count.fetch_add(1, Ordering::Relaxed) + 1
    }
}

#[cfg(test)]
mod tests {
    use super::BM25Entry;
    use std::sync::atomic::Ordering;
    use std::thread;

    #[test]
    fn atomic_use_count_increments_correctly() {
        let entry = BM25Entry::default();
        let threads = 8;
        let increments = 1_000;

        thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(|| {
                    for _ in 0..increments {
                        entry.increment_use_count();
                    }
                });
            }
        });

        assert_eq!(
            entry.use_count.load(Ordering::Relaxed),
            threads * increments
        );
    }

    #[test]
    fn atomic_use_count_serializes_as_u32() {
        let entry = BM25Entry {
            use_count: std::sync::atomic::AtomicU32::new(7),
            ..Default::default()
        };

        let json = serde_json::to_value(&entry).expect("serialize BM25Entry");
        assert_eq!(json.get("use_count"), Some(&serde_json::json!(7)));
        assert!(json["use_count"].is_number());

        let round_trip: BM25Entry = serde_json::from_value(json).expect("deserialize BM25Entry");
        assert_eq!(round_trip.use_count.load(Ordering::Relaxed), 7);
    }
}
