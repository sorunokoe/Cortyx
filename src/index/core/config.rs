// ─── Activation tuning constants ─────────────────────────────────────────────

/// Maximum core neurons returned in Phase 1 of activation.
/// At 5: covers most real queries without exceeding the typical 3K-token context budget.
/// Raise to 7–8 for richer contexts; lower to 3 for speed-critical agents.
/// Validated: LME-500 R@5 unchanged between 4–7.
pub const MAX_CORE_NEURONS: usize = 5;

/// Maximum use-case neurons attached per core in Phase 2 (synapse traversal).
/// 2 per core keeps total output bounded at MAX_CORE_NEURONS * 2 = 10 traversal candidates.
pub const MAX_USE_CASE_PER_CORE: usize = 2;

/// Minimum BM25 relevance ratio (relative to the top score) for synapse traversal to
/// include a neighbor. At 0.25: a neighbor scoring less than 25% of the best match is
/// considered off-topic and skipped. Range 0.1–0.4 tested; below 0.15 introduces noise,
/// above 0.35 over-prunes useful lateral context.
pub const SYNAPSE_RELEVANCE_THRESHOLD: f32 = 0.25;

/// BM25 score ratio above which a neuron triggers 2-hop graph traversal.
/// At 0.6: only neurons scoring ≥60% of the top hit expand into their neighborhood.
/// Prevents runaway traversal on low-confidence matches.
pub const HIGH_ACTIVATION_THRESHOLD: f32 = 0.6;

/// Minimum term length kept by the tokenizer. Filters stop-words like "a", "I".
pub(super) const MIN_TERM_LEN: usize = 2;

/// Okapi BM25 term-frequency saturation parameter. Standard value; k1=1.2 is the
/// broadly accepted default for short-to-medium documents (neurons are ~150–400 tokens).
/// Higher k1 (e.g. 2.0) rewards repeated terms more; lower (e.g. 0.5) flattens TF.
pub(super) const BM25_K1: f32 = 1.2;

/// Okapi BM25 length normalization parameter. Standard value for mixed-length docs.
/// b=0.75 is canonical; 0.65 slightly favors shorter neurons, empirically better for
/// code neurons where dense short-form beats verbose prose.
pub(super) const BM25_B: f32 = 0.65;

/// Index schema version. Migrations apply the chain from stored version to INDEX_VERSION,
/// preserving all user-curated `use_count`, `hit_count`, and `staleness_multiplier` data.
pub(super) const INDEX_VERSION: u32 = 8;

/// Minimum activations before quarantine decisions are made. Below 5 the Wilson CI is
/// too wide to be trustworthy (z=1.0 CI lower bound touches 0 even for 4/5 hit rates).
/// Adaptive CI (S4) uses z=1.0 for 5–19 samples, escalating to z=1.645 at 20–99.
#[allow(dead_code)]
pub(super) const QUARANTINE_MIN_SAMPLES: u32 = 5;

/// Wilson score lower bound threshold for the 20–99 sample tier (90% CI, z=1.645).
/// Kept for test assertions; runtime uses `adaptive_quarantine_params` directly.
#[allow(dead_code)]
pub(super) const QUARANTINE_WILSON_THRESHOLD: f32 = 0.05;

/// Wilson score lower bound above which a quarantined neuron is rehabilitated.
/// 0.15 is ~1.5× the quarantine entry threshold (0.10), so a neuron must show
/// meaningfully improved citation rate before exiting quarantine — noise spikes don't
/// lift it. Range 0.12–0.20 is stable; below 0.12 causes oscillation.
pub(super) const QUARANTINE_RECOVERY_THRESHOLD: f32 = 0.15;

/// Minimum activations before the learned hit-rate multiplier is applied.
/// Below this the multiplier stays 1.0 (cold-start neutral) to avoid penalizing new neurons.
pub(super) const MIN_SAMPLE_SIZE: u32 = 5;

/// Hit-rate floor used only by the legacy quarantine path in tests.
/// Runtime now uses adaptive Wilson bounds via `adaptive_quarantine_params`.
#[allow(dead_code)]
pub(super) const QUARANTINE_THRESHOLD: f32 = 0.10;

/// Estimated token cost per synapse-traversed neuron for dynamic budget allocation.
/// 200 is conservative (most neurons are ~150 tokens); overhead accounts for section
/// headers and MCP formatting. Raise if context overflows; lower to admit more neurons.
pub(super) const AVG_SYNAPSE_TOKEN_COST: usize = 200;

/// BM25 top-score above which retrieval is considered high-confidence and dense
/// re-ranking is skipped (no wasted compute for clear keyword matches).
/// 8.0 corresponds to a strong multi-term exact match; empirically derived from the
/// score distribution on LME-500 where scores >8 were never wrong at R@1.
pub(super) const HIGH_CONFIDENCE_THRESHOLD: f32 = 8.0;

/// BM25 top-score below which dense re-ranking (embed feature) is activated.
/// 4.0 marks genuinely ambiguous queries where semantic similarity adds recall.
/// Gap between LOW (4.0) and HIGH (8.0) is the hybrid zone where both paths run.
pub(super) const LOW_CONFIDENCE_THRESHOLD: f32 = 4.0;

/// Minimum public functions in a source file to trigger UseCase sub-neuron splitting (S3).
///
/// Files with fewer functions keep a single Core neuron (low overhead).
/// Files at or above this threshold get one UseCase sub-neuron per function,
/// enabling per-function BM25 retrieval precision without inflating the Core.
pub(super) const SUBNEURON_SPLIT_THRESHOLD: usize = 6;

/// Maximum sub-neurons generated per source file (caps index growth on huge files).
pub(super) const MAX_SUBNEURONS_PER_FILE: usize = 20;
