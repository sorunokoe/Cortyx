//! Graph building and scoring operations for the reasoner.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::kg::KgEntity;
use crate::neuron::{NeuronKind, Synapse};

use super::types::{ReasonedNode, ReasonedStep, ReasonerNeuron};

pub(super) const CORROBORATION_FACTOR: f32 = 0.25;
pub(super) const EPSILON: f32 = 1e-6;
pub(super) const REVERSE_EDGE_WEIGHT_FACTOR: f32 = 0.7;

#[derive(Debug, Clone)]
pub(super) struct TraversalEdge {
    pub synapse: Synapse,
    pub reverse: bool,
}

#[derive(Debug, Clone)]
pub(super) struct NodeContribution {
    pub score: f32,
    pub depth: u8,
    pub strongest_step: Option<ReasonedStep>,
}

/// Build adjacency list from neuron synapses, including reverse edges
pub(super) fn build_adjacency(
    neurons: &HashMap<PathBuf, ReasonerNeuron>,
) -> HashMap<PathBuf, Vec<TraversalEdge>> {
    let mut adjacency = HashMap::new();

    for neuron in neurons.values() {
        for synapse in &neuron.meta.synapses {
            push_edge(
                &mut adjacency,
                neuron.path.clone(),
                TraversalEdge {
                    synapse: synapse.clone(),
                    reverse: false,
                },
            );

            push_edge(
                &mut adjacency,
                synapse.target.clone(),
                TraversalEdge {
                    synapse: Synapse {
                        target: neuron.path.clone(),
                        edge_type: synapse.edge_type.inverse(),
                        weight: synapse.weight * REVERSE_EDGE_WEIGHT_FACTOR,
                        reason: format!("← {}", synapse.reason),
                        learned_weight: 0.0,
                        traversal_count: 0,
                        last_co_activation_day: 0,
                    },
                    reverse: true,
                },
            );
        }
    }

    adjacency
}

fn push_edge(
    adjacency: &mut HashMap<PathBuf, Vec<TraversalEdge>>,
    source: PathBuf,
    edge: TraversalEdge,
) {
    let edges = adjacency.entry(source).or_default();
    if edges.iter().any(|existing| {
        existing.reverse == edge.reverse
            && existing.synapse.target == edge.synapse.target
            && existing.synapse.edge_type == edge.synapse.edge_type
            && existing.synapse.reason == edge.synapse.reason
    }) {
        return;
    }
    edges.push(edge);
}

/// Update contribution tracking, returning true if this is a new/better contribution
pub(super) fn upsert_contribution(
    contributions: &mut HashMap<PathBuf, HashMap<PathBuf, NodeContribution>>,
    target: PathBuf,
    origin: PathBuf,
    score: f32,
    depth: u8,
    strongest_step: Option<ReasonedStep>,
) -> bool {
    let entry = contributions.entry(target).or_default();
    let should_replace = match entry.get(&origin) {
        Some(existing) => {
            score > existing.score + EPSILON
                || ((score - existing.score).abs() <= EPSILON && depth < existing.depth)
        },
        None => true,
    };

    if should_replace {
        entry.insert(
            origin,
            NodeContribution {
                score,
                depth,
                strongest_step,
            },
        );
    }

    should_replace
}

/// Build reasoned nodes from contribution tracking
pub(super) fn build_reasoned_nodes(
    contributions: &HashMap<PathBuf, HashMap<PathBuf, NodeContribution>>,
    neurons: &HashMap<PathBuf, ReasonerNeuron>,
    kg_entities: &HashMap<PathBuf, KgEntity>,
    seed_set: &std::collections::HashSet<PathBuf>,
) -> Vec<ReasonedNode> {
    contributions
        .iter()
        .map(|(path, by_origin)| {
            let mut origin_contributions: Vec<(&PathBuf, &NodeContribution)> =
                by_origin.iter().collect();
            origin_contributions.sort_by(|a, b| {
                b.1.score
                    .total_cmp(&a.1.score)
                    .then_with(|| a.1.depth.cmp(&b.1.depth))
                    .then_with(|| a.0.cmp(b.0))
            });

            let best_score = origin_contributions
                .first()
                .map(|(_, contribution)| contribution.score)
                .unwrap_or(0.0);
            let corroboration = origin_contributions
                .iter()
                .skip(1)
                .map(|(_, contribution)| contribution.score)
                .sum::<f32>()
                * CORROBORATION_FACTOR;
            let score = best_score + corroboration;
            let depth = origin_contributions
                .iter()
                .map(|(_, contribution)| contribution.depth)
                .min()
                .unwrap_or(0);
            let strongest_step = origin_contributions
                .first()
                .and_then(|(_, contribution)| contribution.strongest_step.clone());
            let mut supporting: Vec<PathBuf> = by_origin.keys().cloned().collect();
            supporting.sort();

            let (kind, module, summary) = if let Some(neuron) = neurons.get(path) {
                (
                    Some(neuron.meta.kind.clone()),
                    neuron.meta.module.clone(),
                    neuron.summary.clone(),
                )
            } else if let Some(entity) = kg_entities.get(path) {
                (
                    Some(NeuronKind::Concept),
                    None,
                    Some(format!("KG facts for {}", entity.entity.replace('_', " "))),
                )
            } else {
                (None, None, None)
            };

            ReasonedNode {
                path: path.clone(),
                score,
                depth,
                kind,
                module,
                summary,
                supporting,
                strongest_step,
                is_seed: seed_set.contains(path),
                is_kg_entity: kg_entities.contains_key(path),
            }
        })
        .collect()
}

pub(super) fn ordered_pair(a: &Path, b: &Path) -> (PathBuf, PathBuf) {
    if a < b {
        (a.to_path_buf(), b.to_path_buf())
    } else {
        (b.to_path_buf(), a.to_path_buf())
    }
}
