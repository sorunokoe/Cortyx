//! Core graph traversal engine for the reasoner.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;

use crate::kg::KgEntity;
use crate::neuron::SynapseType;

use super::graph_ops::{
    build_adjacency, build_reasoned_nodes, ordered_pair, upsert_contribution, NodeContribution,
    TraversalEdge, EPSILON,
};
use super::types::{
    ReasonedStep, ReasonerConflict, ReasonerNeuron, ReasonerSeed, ReasoningReport, TraversalOptions,
};

#[derive(Debug, Clone)]
struct WorkItem {
    path: PathBuf,
    origin: PathBuf,
    score: f32,
    depth: u8,
}

/// Graph reasoner that performs multi-hop traversal over neuron synapse edges.
pub struct GraphReasoner {
    neurons: HashMap<PathBuf, ReasonerNeuron>,
    adjacency: HashMap<PathBuf, Vec<TraversalEdge>>,
    kg_entities: HashMap<PathBuf, KgEntity>,
}

impl GraphReasoner {
    /// Create a graph reasoner from mined neurons and KG entities.
    ///
    /// `neurons` provides the traversal graph inputs, while `kg_entities`
    /// provides fact-bearing entity nodes that can be surfaced in the final
    /// [`ReasoningReport`]. The adjacency list is built once up front.
    pub fn new<I, J>(neurons: I, kg_entities: J) -> Self
    where
        I: IntoIterator<Item = ReasonerNeuron>,
        J: IntoIterator<Item = KgEntity>,
    {
        let neurons: HashMap<PathBuf, ReasonerNeuron> = neurons
            .into_iter()
            .map(|neuron| (neuron.path.clone(), neuron))
            .collect();
        let kg_entities: HashMap<PathBuf, KgEntity> = kg_entities
            .into_iter()
            .map(|entity| (entity.path.clone(), entity))
            .collect();
        let adjacency = build_adjacency(&neurons);

        Self {
            neurons,
            adjacency,
            kg_entities,
        }
    }

    /// Traverse a support graph from the provided seed evidence.
    ///
    /// `seeds` is the starting evidence set; scores are normalized relative to
    /// the strongest seed so callers can pass any positive scoring system.
    /// `options` controls hop depth, expansion limits, reverse-edge traversal,
    /// and fact filtering for the resulting [`ReasoningReport`].
    pub fn trace(&self, seeds: &[ReasonerSeed], options: TraversalOptions) -> ReasoningReport {
        let mut deduped: HashMap<PathBuf, f32> = HashMap::new();
        for seed in seeds {
            deduped
                .entry(seed.path.clone())
                .and_modify(|score| *score = score.max(seed.score))
                .or_insert(seed.score);
        }

        let strongest_seed = deduped.values().copied().fold(0.0_f32, f32::max);
        if strongest_seed <= 0.0 {
            return ReasoningReport::default();
        }

        let mut ordered_seeds: Vec<(PathBuf, f32)> = deduped
            .into_iter()
            .map(|(path, score)| (path, score / strongest_seed))
            .collect();
        ordered_seeds.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let seed_set: HashSet<PathBuf> =
            ordered_seeds.iter().map(|(path, _)| path.clone()).collect();
        let mut contributions: HashMap<PathBuf, HashMap<PathBuf, NodeContribution>> =
            HashMap::new();
        let mut queue = VecDeque::new();

        for (path, score) in &ordered_seeds {
            upsert_contribution(
                &mut contributions,
                path.clone(),
                path.clone(),
                *score,
                0,
                None,
            );
            queue.push_back(WorkItem {
                path: path.clone(),
                origin: path.clone(),
                score: *score,
                depth: 0,
            });
        }

        let mut expansions = 0usize;
        let mut conflicts = Vec::new();
        let mut seen_conflicts = HashSet::new();
        let mut nodes_by_depth: HashMap<u8, usize> = HashMap::new();
        let mut max_depth_reached = 0u8;

        while let Some(work) = queue.pop_front() {
            if work.depth >= options.max_hops || expansions >= options.max_expansions {
                continue;
            }

            let Some(edges) = self.adjacency.get(&work.path) else {
                continue;
            };

            for edge in edges {
                if expansions >= options.max_expansions {
                    break;
                }
                if edge.reverse && !options.include_reverse_edges {
                    continue;
                }

                let synapse = &edge.synapse;
                if synapse.edge_type == SynapseType::Contradicts {
                    if seed_set.contains(&synapse.target)
                        || contributions.contains_key(&synapse.target)
                    {
                        let pair = ordered_pair(&work.path, &synapse.target);
                        if seen_conflicts.insert(pair) {
                            conflicts.push(ReasonerConflict {
                                source: work.path.clone(),
                                target: synapse.target.clone(),
                                edge_type: SynapseType::Contradicts,
                                reason: synapse.reason.clone(),
                            });
                        }
                    }
                    continue;
                }

                let propagated = work.score * synapse.weight.get() * synapse.effective_weight();
                let tier_floor = synapse.edge_type.confidence_tier().min_propagated_score();
                let effective_floor = tier_floor.max(options.min_propagated_score);
                if propagated + EPSILON < effective_floor {
                    continue;
                }

                let next_depth = work.depth + 1;
                let strongest_step = Some(ReasonedStep {
                    from: work.path.clone(),
                    edge_type: synapse.edge_type,
                    reason: synapse.reason.clone(),
                });

                if upsert_contribution(
                    &mut contributions,
                    synapse.target.clone(),
                    work.origin.clone(),
                    propagated,
                    next_depth,
                    strongest_step,
                ) {
                    expansions += 1;
                    *nodes_by_depth.entry(next_depth).or_insert(0) += 1;
                    max_depth_reached = max_depth_reached.max(next_depth);
                    queue.push_back(WorkItem {
                        path: synapse.target.clone(),
                        origin: work.origin.clone(),
                        score: propagated,
                        depth: next_depth,
                    });
                }
            }
        }
        // Converged = BFS queue was naturally exhausted (not cut short by max_expansions).
        let converged = expansions < options.max_expansions;

        let max_possible_depth = options.max_hops;
        let mut nodes_by_depth_vec = vec![0usize; (max_possible_depth as usize) + 1];
        // Seed nodes are at depth 0.
        nodes_by_depth_vec[0] = ordered_seeds.len();
        for (depth, count) in &nodes_by_depth {
            let d = *depth as usize;
            if d < nodes_by_depth_vec.len() {
                nodes_by_depth_vec[d] = *count;
            }
        }

        let traversal_stats = super::types::TraversalStats {
            nodes_by_depth: nodes_by_depth_vec,
            max_depth_reached,
            converged,
            total_expansions: expansions,
        };

        let mut nodes =
            build_reasoned_nodes(&contributions, &self.neurons, &self.kg_entities, &seed_set);
        let facts = super::facts::build_reasoned_facts(
            &nodes,
            &self.kg_entities,
            options.include_inactive_facts,
        );

        nodes.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.depth.cmp(&b.depth))
                .then_with(|| a.path.cmp(&b.path))
        });

        ReasoningReport {
            nodes,
            facts,
            conflicts,
            traversal_stats,
        }
    }
}
