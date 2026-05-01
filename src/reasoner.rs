//! Reasoning engine for traversing neuron graphs and extracting facts.
//!
//! Performs multi-hop reasoning across synapse edges to answer complex queries.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};

use crate::kg::KgEntity;
use crate::neuron::{NeuronKind, NeuronMeta, Synapse, SynapseType};

const CORROBORATION_FACTOR: f32 = 0.25;
const EPSILON: f32 = 1e-6;
const REVERSE_EDGE_WEIGHT_FACTOR: f32 = 0.7;

/// Minimal neuron view for graph reasoning.
///
/// The reasoner works on existing neuron metadata instead of maintaining a parallel graph model,
/// while allowing callers to optionally provide a short summary for render-time explanations.
#[derive(Debug, Clone)]
pub struct ReasonerNeuron {
    pub path: PathBuf,
    pub meta: NeuronMeta,
    pub summary: Option<String>,
}

impl ReasonerNeuron {
    pub fn new(path: impl Into<PathBuf>, meta: NeuronMeta) -> Self {
        Self {
            path: path.into(),
            meta,
            summary: None,
        }
    }

    pub fn with_summary(mut self, summary: impl Into<String>) -> Self {
        let summary = summary.into();
        if !summary.trim().is_empty() {
            self.summary = Some(summary);
        }
        self
    }
}

/// Seed evidence supplied by a caller (for example: retrieval hits, explicit focus nodes, etc.).
#[derive(Debug, Clone)]
pub struct ReasonerSeed {
    pub path: PathBuf,
    pub score: f32,
}

impl ReasonerSeed {
    pub fn new(path: impl Into<PathBuf>, score: f32) -> Self {
        Self {
            path: path.into(),
            score: score.max(0.0),
        }
    }
}

/// Conservative traversal knobs for the graph-reasoner core.
#[derive(Debug, Clone, Copy)]
pub struct TraversalOptions {
    /// Maximum number of hops from a seed node.
    pub max_hops: u8,
    /// Hard cap on enqueued expansions so wide graphs cannot explode.
    pub max_expansions: usize,
    /// Minimum normalized propagated score required to keep traversing an edge.
    pub min_propagated_score: f32,
    /// Whether reverse/inferred edges should be traversed.
    pub include_reverse_edges: bool,
    /// Whether ended KG facts should be surfaced alongside active facts.
    pub include_inactive_facts: bool,
}

impl Default for TraversalOptions {
    fn default() -> Self {
        Self {
            max_hops: 2,
            max_expansions: 64,
            min_propagated_score: 0.12,
            include_reverse_edges: true,
            include_inactive_facts: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasonedStep {
    pub from: PathBuf,
    pub edge_type: SynapseType,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct ReasonedNode {
    pub path: PathBuf,
    pub score: f32,
    pub depth: u8,
    pub kind: Option<NeuronKind>,
    pub module: Option<String>,
    pub summary: Option<String>,
    /// Seed/supporting origins that independently reached this node.
    pub supporting_paths: Vec<PathBuf>,
    pub strongest_step: Option<ReasonedStep>,
    pub is_seed: bool,
    pub is_kg_entity: bool,
}

#[derive(Debug, Clone)]
pub struct ReasonedFact {
    pub entity_path: PathBuf,
    pub entity: String,
    pub predicate: String,
    pub value: String,
    pub score: f32,
    pub valid_from: String,
    pub ended: String,
    pub supporting_paths: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasonerConflict {
    pub source: PathBuf,
    pub target: PathBuf,
    pub edge_type: SynapseType,
    pub reason: String,
}

#[derive(Debug, Clone, Default)]
pub struct ReasoningReport {
    pub nodes: Vec<ReasonedNode>,
    pub facts: Vec<ReasonedFact>,
    pub conflicts: Vec<ReasonerConflict>,
}

impl ReasoningReport {
    /// Render short, deterministic evidence lines suitable for later answer/context plumbing.
    pub fn summary_lines(&self, max_nodes: usize, max_facts: usize) -> Vec<String> {
        let mut lines = Vec::new();
        let mut fact_entity_paths = HashSet::new();

        for fact in self.facts.iter().take(max_facts) {
            fact_entity_paths.insert(fact.entity_path.clone());
            lines.push(format!(
                "fact {}.{} = {} (score {:.2}, support {})",
                fact.entity,
                fact.predicate,
                fact.value,
                fact.score,
                fact.supporting_paths.len()
            ));
        }

        for node in self
            .nodes
            .iter()
            .filter(|node| !(node.is_kg_entity && fact_entity_paths.contains(&node.path)))
            .take(max_nodes)
        {
            let mut line = format!(
                "node {} (score {:.2}, depth {})",
                short_path(&node.path),
                node.score,
                node.depth
            );

            if let Some(step) = &node.strongest_step {
                line.push_str(&format!(
                    " via {} from {}",
                    synapse_label(&step.edge_type),
                    short_path(&step.from)
                ));
            }

            if let Some(summary) = node.summary.as_deref().map(compact_summary) {
                if !summary.is_empty() {
                    line.push_str(&format!(": {summary}"));
                }
            }

            lines.push(line);
        }

        lines
    }
}

#[derive(Debug, Default)]
pub struct GraphReasoner {
    neurons: HashMap<PathBuf, ReasonerNeuron>,
    adjacency: HashMap<PathBuf, Vec<TraversalEdge>>,
    kg_entities: HashMap<PathBuf, KgEntity>,
}

impl GraphReasoner {
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

    /// Traverse a small support graph from the provided seed evidence.
    ///
    /// Seed scores are normalized relative to the strongest seed so callers can pass any
    /// positive scoring system (BM25, heuristic scores, etc.) without calibrating this core.
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

                let propagated =
                    work.score * synapse.weight.clamp(0.0, 1.0) * synapse.effective_weight();
                if propagated + EPSILON < options.min_propagated_score {
                    continue;
                }

                let next_depth = work.depth + 1;
                let strongest_step = Some(ReasonedStep {
                    from: work.path.clone(),
                    edge_type: synapse.edge_type.clone(),
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
                    queue.push_back(WorkItem {
                        path: synapse.target.clone(),
                        origin: work.origin.clone(),
                        score: propagated,
                        depth: next_depth,
                    });
                }
            }
        }

        let mut nodes =
            build_reasoned_nodes(&contributions, &self.neurons, &self.kg_entities, &seed_set);
        let facts = build_reasoned_facts(&nodes, &self.kg_entities, options.include_inactive_facts);

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
        }
    }
}

#[derive(Debug, Clone)]
struct TraversalEdge {
    synapse: Synapse,
    reverse: bool,
}

#[derive(Debug, Clone)]
struct WorkItem {
    path: PathBuf,
    origin: PathBuf,
    score: f32,
    depth: u8,
}

#[derive(Debug, Clone)]
struct NodeContribution {
    score: f32,
    depth: u8,
    strongest_step: Option<ReasonedStep>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FactKey {
    entity_path: PathBuf,
    predicate: String,
    value: String,
    valid_from: String,
    ended: String,
}

fn build_adjacency(
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

fn upsert_contribution(
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
        }
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

fn build_reasoned_nodes(
    contributions: &HashMap<PathBuf, HashMap<PathBuf, NodeContribution>>,
    neurons: &HashMap<PathBuf, ReasonerNeuron>,
    kg_entities: &HashMap<PathBuf, KgEntity>,
    seed_set: &HashSet<PathBuf>,
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
            let mut supporting_paths: Vec<PathBuf> = by_origin.keys().cloned().collect();
            supporting_paths.sort();

            let is_kg_entity = kg_entities.contains_key(path);
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
                supporting_paths,
                strongest_step,
                is_seed: seed_set.contains(path),
                is_kg_entity,
            }
        })
        .collect()
}

fn build_reasoned_facts(
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
                entity_path: entity.path.clone(),
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

            let entry = merged.entry(key).or_insert_with(|| ReasonedFact {
                entity_path: entity.path.clone(),
                entity: entity.entity.clone(),
                predicate: fact.predicate.clone(),
                value: fact.value.clone(),
                score: fact_score,
                valid_from: fact.valid_from.clone(),
                ended: fact.ended.clone(),
                supporting_paths: node.supporting_paths.clone(),
            });

            entry.score = entry.score.max(fact_score);
            merge_supporting_paths(&mut entry.supporting_paths, &node.supporting_paths);
        }
    }

    let mut facts: Vec<ReasonedFact> = merged.into_values().collect();
    facts.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.entity.cmp(&b.entity))
            .then_with(|| a.predicate.cmp(&b.predicate))
            .then_with(|| a.value.cmp(&b.value))
            .then_with(|| a.valid_from.cmp(&b.valid_from))
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

fn ordered_pair(a: &Path, b: &Path) -> (PathBuf, PathBuf) {
    if a <= b {
        (a.to_path_buf(), b.to_path_buf())
    } else {
        (b.to_path_buf(), a.to_path_buf())
    }
}

fn compact_summary(summary: &str) -> String {
    summary
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .unwrap_or_default()
}

fn short_path(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| path.display().to_string())
}

fn synapse_label(edge_type: &SynapseType) -> &'static str {
    match edge_type {
        SynapseType::SemanticRelated => "semantic_related",
        SynapseType::Imports => "imports",
        SynapseType::Calls => "calls",
        SynapseType::Implements => "implements",
        SynapseType::ImplementedBy => "implemented_by",
        SynapseType::CalledBy => "called_by",
        SynapseType::Contradicts => "contradicts",
        SynapseType::TemporalFollows => "temporal_follows",
        SynapseType::Derived => "derived",
        SynapseType::ConceptExpands => "concept_expands",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kg::KgFact;
    use crate::neuron::{NeuronKind, NeuronStatus};

    fn make_neuron(path: &str) -> ReasonerNeuron {
        let mut meta = NeuronMeta::new_stub(Path::new(path), NeuronKind::Core);
        meta.status = NeuronStatus::Fresh;
        ReasonerNeuron::new(PathBuf::from(path), meta)
    }

    fn make_synapse(target: &str, edge_type: SynapseType, reason: &str) -> Synapse {
        let mut synapse = Synapse::new(PathBuf::from(target), edge_type, reason.to_string());
        synapse.weight = 1.0;
        synapse
    }

    #[test]
    fn reverse_edges_allow_conservative_backtracking() {
        let a = "neurons/a.context.md";
        let b = "neurons/b.context.md";

        let mut left = make_neuron(a).with_summary("Primary implementation node.");
        left.meta.synapses.push(make_synapse(
            b,
            SynapseType::Implements,
            "implements the shared contract",
        ));
        let right = make_neuron(b).with_summary("Shared contract node.");

        let report = GraphReasoner::new(vec![left, right], Vec::<KgEntity>::new())
            .trace(&[ReasonerSeed::new(b, 3.0)], TraversalOptions::default());

        let reached = report
            .nodes
            .iter()
            .find(|node| node.path == PathBuf::from(a))
            .expect("reverse edge should reach the implementation node");
        assert_eq!(reached.depth, 1);
        assert!(reached.score > 0.5);
        assert_eq!(reached.supporting_paths, vec![PathBuf::from(b)]);
        assert_eq!(
            reached
                .strongest_step
                .as_ref()
                .map(|step| step.edge_type.clone()),
            Some(SynapseType::ImplementedBy)
        );
    }

    #[test]
    fn corroborating_seeds_raise_neighbor_score() {
        let a = "neurons/a.context.md";
        let b = "neurons/b.context.md";
        let c = "neurons/c.context.md";

        let mut left = make_neuron(a).with_summary("Imports the shared auth layer.");
        left.meta.synapses.push(make_synapse(
            b,
            SynapseType::Imports,
            "imports auth helpers",
        ));

        let mut right = make_neuron(c).with_summary("Calls into the same auth layer.");
        right.meta.synapses.push(make_synapse(
            b,
            SynapseType::Calls,
            "calls shared validation",
        ));

        let middle = make_neuron(b).with_summary("Auth helpers and validation logic.");

        let report = GraphReasoner::new(vec![left, middle, right], Vec::<KgEntity>::new()).trace(
            &[ReasonerSeed::new(a, 10.0), ReasonerSeed::new(c, 8.0)],
            TraversalOptions::default(),
        );

        let node = report
            .nodes
            .iter()
            .find(|node| node.path == PathBuf::from(b))
            .expect("shared node should be reached");
        assert_eq!(
            node.supporting_paths,
            vec![PathBuf::from(a), PathBuf::from(c)]
        );
        assert!(
            node.score > 0.9,
            "expected corroboration boost, got {}",
            node.score
        );
        assert_eq!(
            node.strongest_step
                .as_ref()
                .map(|step| step.edge_type.clone()),
            Some(SynapseType::Imports)
        );
    }

    #[test]
    fn reached_kg_entities_surface_active_facts_only() {
        let task = "neurons/task.context.md";
        let kg_path = PathBuf::from("neurons/_kg_user.context.md");

        let mut task_neuron = make_neuron(task).with_summary("User profile reasoning seed.");
        task_neuron.meta.synapses.push(make_synapse(
            kg_path.to_str().unwrap(),
            SynapseType::ConceptExpands,
            "expands to user facts",
        ));

        let entity = KgEntity {
            entity: "user".to_string(),
            path: kg_path.clone(),
            facts: vec![
                KgFact {
                    predicate: "status".to_string(),
                    value: "blocked".to_string(),
                    valid_from: "2026-04-01".to_string(),
                    ended: "2026-05-01".to_string(),
                },
                KgFact {
                    predicate: "status".to_string(),
                    value: "done".to_string(),
                    valid_from: "2026-05-01".to_string(),
                    ended: String::new(),
                },
                KgFact {
                    predicate: "related_entity".to_string(),
                    value: "agent_reviewer".to_string(),
                    valid_from: "2026-05-01".to_string(),
                    ended: String::new(),
                },
            ],
        };

        let report = GraphReasoner::new(vec![task_neuron], vec![entity])
            .trace(&[ReasonerSeed::new(task, 1.0)], TraversalOptions::default());

        assert!(report.facts.iter().any(|fact| {
            fact.entity == "user" && fact.predicate == "status" && fact.value == "done"
        }));
        assert!(report.facts.iter().any(|fact| {
            fact.entity == "user"
                && fact.predicate == "related_entity"
                && fact.value == "agent_reviewer"
        }));
        assert!(!report.facts.iter().any(|fact| {
            fact.entity == "user" && fact.predicate == "status" && fact.value == "blocked"
        }));

        let summary = report.summary_lines(2, 2).join("\n");
        assert!(summary.contains("fact user.status = done"));
    }

    #[test]
    fn contradiction_edges_are_reported_without_traversal() {
        let a = "neurons/a.context.md";
        let b = "neurons/b.context.md";
        let c = "neurons/c.context.md";

        let mut first = make_neuron(a).with_summary("Seed node.");
        first
            .meta
            .synapses
            .push(make_synapse(b, SynapseType::Calls, "calls intermediate"));

        let mut middle = make_neuron(b).with_summary("Intermediate node.");
        middle.meta.synapses.push(make_synapse(
            c,
            SynapseType::Contradicts,
            "conflicts with the target state",
        ));

        let last = make_neuron(c).with_summary("Conflicting target node.");

        let report = GraphReasoner::new(vec![first, middle, last], Vec::<KgEntity>::new()).trace(
            &[ReasonerSeed::new(a, 1.0), ReasonerSeed::new(c, 0.95)],
            TraversalOptions::default(),
        );

        assert_eq!(report.conflicts.len(), 1);
        assert_eq!(report.conflicts[0].edge_type, SynapseType::Contradicts);
        assert_eq!(
            ordered_pair(&report.conflicts[0].source, &report.conflicts[0].target),
            (PathBuf::from(b), PathBuf::from(c))
        );
    }
}
