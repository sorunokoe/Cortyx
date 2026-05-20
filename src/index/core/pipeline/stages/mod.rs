mod bm25_scoring;
mod coactivation;
mod coreturn_boost;
mod counting_augment;
mod morpheme_bridge;
mod pmi_expansion;
mod session_cluster;
mod session_tf_decay;
mod staleness_decay;
mod synapse_traversal;
mod temporal_proximity;
mod use_case_augment;
mod vocab_bridge;

use super::super::{QueryContext, ScoredCandidate};
use crate::index::core::config::SESSION_SAME_SCORE_DECAY;

pub use bm25_scoring::Bm25ScoringStage;
pub use coactivation::CoactivationStage;
pub use coreturn_boost::CoreturnBoostStage;
pub use counting_augment::CountingAugmentStage;
pub use morpheme_bridge::MorphemeBridgeStage;
pub use pmi_expansion::PmiExpansionStage;
pub use session_cluster::SessionClusterStage;
pub use session_tf_decay::SessionTfDecayStage;
pub use staleness_decay::StalenessDecayStage;
pub use synapse_traversal::SynapseTraversalStage;
pub use temporal_proximity::TemporalProximityStage;
pub use use_case_augment::UseCaseAugmentStage;
pub use vocab_bridge::VocabBridgeStage;

pub(super) fn sort_candidates(candidates: &mut [ScoredCandidate]) {
    candidates.sort_unstable_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.entry_idx.cmp(&b.entry_idx))
    });
}

pub(super) fn upsert_candidate(
    candidates: &mut Vec<ScoredCandidate>,
    entry_idx: usize,
    score: f32,
    tokens: usize,
) {
    if let Some(existing) = candidates
        .iter_mut()
        .find(|candidate| candidate.entry_idx == entry_idx)
    {
        existing.score = existing.score.max(score);
        existing.tokens = tokens;
    } else {
        candidates.push(ScoredCandidate::new(entry_idx, score, tokens));
    }
}

pub(super) fn apply_preceding_decay(ctx: &QueryContext<'_>, idx: usize, score: f32) -> f32 {
    let entry = ctx.entry(idx);
    let mut adjusted = score * entry.staleness_multiplier;
    if ctx.session_id == Some(entry.session_id.as_str()) {
        adjusted *= SESSION_SAME_SCORE_DECAY;
    }
    adjusted
}
