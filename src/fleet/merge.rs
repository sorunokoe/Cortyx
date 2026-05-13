//! Merge helpers for combining local and fleet-supplemented context.

use std::collections::HashMap;

use super::{FleetNodeId, FleetQueryResult};

const RRF_K: usize = 60;

/// A rendered fleet result eligible for inclusion in merged output.
#[derive(Debug, Clone, PartialEq)]
pub struct MergedFleetResult {
    pub node_id: FleetNodeId,
    pub node_alias: String,
    pub context_text: String,
    pub top_score: f32,
}

#[derive(Debug, Clone)]
enum MergeSource {
    Local { text: String, top_score: f32 },
    Fleet(FleetQueryResult),
}

#[derive(Debug, Clone)]
struct RankedMergeSource {
    dedupe_key: String,
    weighted_rrf: f32,
    top_score: f32,
    source: MergeSource,
}

fn rrf_score(rank: usize) -> f32 {
    1.0 / (RRF_K + rank) as f32
}

/// Merge local context with supplementary fleet context.
pub fn rrf_merge(
    local_text: &str,
    local_score: f32,
    fleet_results: Vec<FleetQueryResult>,
    local_weight: f32,
    fleet_weight: f32,
) -> String {
    let mut fleet_ranked: Vec<FleetQueryResult> = fleet_results
        .into_iter()
        .filter(|result| result.top_score > 0.0)
        .collect();
    fleet_ranked.sort_by(|left, right| {
        right
            .top_score
            .total_cmp(&left.top_score)
            .then_with(|| left.node_alias.cmp(&right.node_alias))
    });

    if fleet_ranked.is_empty() {
        return local_text.to_string();
    }

    let mut ranked = Vec::new();
    if !local_text.trim().is_empty() || local_score > 0.0 {
        ranked.push(RankedMergeSource {
            dedupe_key: "__local__".to_string(),
            weighted_rrf: local_weight * rrf_score(1),
            top_score: local_score,
            source: MergeSource::Local {
                text: local_text.to_string(),
                top_score: local_score,
            },
        });
    }

    ranked.extend(
        fleet_ranked
            .into_iter()
            .enumerate()
            .map(|(rank, result)| RankedMergeSource {
                dedupe_key: format!("fleet:{}", result.node_id),
                weighted_rrf: fleet_weight * rrf_score(rank + 1),
                top_score: result.top_score,
                source: MergeSource::Fleet(result),
            }),
    );

    let mut deduped: HashMap<String, RankedMergeSource> = HashMap::new();
    for item in ranked {
        let replace_existing = deduped.get(&item.dedupe_key).is_none_or(|current| {
            match item.weighted_rrf.total_cmp(&current.weighted_rrf) {
                std::cmp::Ordering::Greater => true,
                std::cmp::Ordering::Equal => item.top_score > current.top_score,
                std::cmp::Ordering::Less => false,
            }
        });
        if replace_existing {
            deduped.insert(item.dedupe_key.clone(), item);
        }
    }

    let mut merged_sources: Vec<RankedMergeSource> = deduped.into_values().collect();
    merged_sources.sort_by(|left, right| {
        right
            .weighted_rrf
            .total_cmp(&left.weighted_rrf)
            .then_with(|| right.top_score.total_cmp(&left.top_score))
            .then_with(|| left.dedupe_key.cmp(&right.dedupe_key))
    });

    let fleet_count = merged_sources
        .iter()
        .filter(|item| matches!(item.source, MergeSource::Fleet(_)))
        .count();
    if fleet_count == 0 {
        return local_text.to_string();
    }

    let mut merged = String::new();
    merged.push_str(&format!(
        "<!-- Fleet context from {} additional nodes -->\n\n",
        fleet_count
    ));

    let mut first_block = true;
    for item in merged_sources {
        match item.source {
            MergeSource::Local { text, top_score } => {
                if text.trim().is_empty() {
                    continue;
                }
                if !first_block {
                    merged.push_str("\n\n");
                }
                merged.push_str(&format!(
                    "<!-- Local context (score: {:.2}) -->\n\n{}",
                    top_score, text
                ));
            },
            MergeSource::Fleet(result) => {
                if !first_block {
                    merged.push_str("\n\n");
                }
                merged.push_str(&format!(
                    "<!-- From fleet node: {} (score: {:.2}) -->\n\n{}",
                    result.node_alias, result.top_score, result.contexts
                ));
            },
        }
        first_block = false;
    }

    merged
}
