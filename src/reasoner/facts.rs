//! Build reasoned facts from KG entities and node scores.

use std::collections::HashMap;
use std::path::PathBuf;

use crate::kg::KgEntity;

use super::graph_ops::REVERSE_EDGE_WEIGHT_FACTOR;
use super::types::{ReasonedFact, ReasonedNode};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FactKey {
    path: PathBuf,
    predicate: String,
    value: String,
    valid_from: String,
    ended: String,
}

/// Build reasoned facts from nodes with KG entities
pub(super) fn build_reasoned_facts(
    nodes: &[ReasonedNode],
    kg_entities: &HashMap<PathBuf, KgEntity>,
    include_inactive_facts: bool,
) -> Vec<ReasonedFact> {
    let mut merged: HashMap<FactKey, ReasonedFact> = HashMap::new();

    for node in nodes {
        let Some(entity) = kg_entities.get(&node.path) else {
            continue;
        };

        let facts: Vec<_> = if include_inactive_facts {
            entity.facts.iter().collect()
        } else {
            entity.active_facts(None)
        };

        for fact in facts {
            if fact.value.trim().is_empty() {
                continue;
            }

            let key = FactKey {
                path: entity.path.clone(),
                predicate: fact.predicate.clone(),
                value: fact.value.clone(),
                valid_from: fact.valid_from.clone(),
                ended: fact.ended.clone(),
            };
            let fact_score = if fact.ended.is_empty() {
                node.score
            } else {
                node.score * REVERSE_EDGE_WEIGHT_FACTOR
            };

            let entry = merged.entry(key).or_insert_with(|| ReasonedFact::new(
                entity.path.clone(),
                entity.entity.clone(),
                fact.predicate.clone(),
                fact.value.clone(),
                fact_score,
                node.supporting.clone(),
                fact.ended.is_empty(),
                fact.valid_from.clone(),
                fact.ended.clone(),
            ));

            entry.score = entry.score.max(fact_score);
            merge_supporting_paths(&mut entry.supporting, &node.supporting);
        }
    }

    let mut facts: Vec<ReasonedFact> = merged.into_values().collect();
    facts.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.subject.cmp(&b.subject))
            .then_with(|| a.relation.cmp(&b.relation))
            .then_with(|| a.object.cmp(&b.object))
    });
    facts
}

fn merge_supporting_paths(target: &mut Vec<PathBuf>, new_paths: &[PathBuf]) {
    for path in new_paths {
        if !target.iter().any(|existing| existing == path) {
            target.push(path.clone());
        }
    }
    target.sort();
}
