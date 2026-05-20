//! Core types for the reasoning engine.

use std::path::{Path, PathBuf};

use crate::neuron::{NeuronKind, NeuronMeta, SynapseType};

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
    pub supporting: Vec<PathBuf>,
    pub strongest_step: Option<ReasonedStep>,
    pub is_seed: bool,
    pub is_kg_entity: bool,
}

#[derive(Debug, Clone)]
pub struct ReasonedFact {
    pub path: PathBuf,
    pub subject: String,
    pub relation: String,
    pub object: String,
    pub score: f32,
    pub supporting: Vec<PathBuf>,
    pub active: bool,
    // Legacy fields for backward compatibility
    pub entity: String,
    pub predicate: String,
    pub value: String,
    pub entity_path: PathBuf,
    pub supporting_paths: Vec<PathBuf>,
    pub valid_from: String,
    pub ended: String,
}

impl ReasonedFact {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        path: PathBuf,
        subject: String,
        relation: String,
        object: String,
        score: f32,
        supporting: Vec<PathBuf>,
        active: bool,
        valid_from: String,
        ended: String,
    ) -> Self {
        Self {
            entity: subject.clone(),
            predicate: relation.clone(),
            value: object.clone(),
            entity_path: path.clone(),
            supporting_paths: supporting.clone(),
            path,
            subject,
            relation,
            object,
            score,
            supporting,
            active,
            valid_from,
            ended,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReasonerConflict {
    pub source: PathBuf,
    pub target: PathBuf,
    pub edge_type: SynapseType,
    pub reason: String,
}

/// Statistics from a single graph traversal pass.
///
/// Captures depth distribution, convergence status, and coverage metrics so that
/// graph reasoning quality can be benchmarked and proven (moving from "smoke" to
/// "proven" on the benchmark scorecard).
#[derive(Debug, Clone, Default)]
pub struct TraversalStats {
    /// Number of nodes discovered at each traversal depth (index = depth level).
    pub nodes_by_depth: Vec<usize>,
    /// Deepest depth that was reached during traversal (0 = seeds only).
    pub max_depth_reached: u8,
    /// True if the traversal exhausted the BFS queue naturally (all reachable
    /// nodes visited). False if cut short by `max_expansions`.
    pub converged: bool,
    /// Total number of edge expansions performed.
    pub total_expansions: usize,
}

impl TraversalStats {
    /// Coverage ratio: fraction of max possible depth that was explored
    /// (simple proxy for traversal completeness).
    #[must_use]
    pub fn depth_coverage(&self, max_hops: u8) -> f32 {
        if max_hops == 0 {
            return 1.0;
        }
        self.max_depth_reached as f32 / max_hops as f32
    }

    /// Total nodes discovered across all depths.
    #[must_use]
    pub fn total_nodes(&self) -> usize {
        self.nodes_by_depth.iter().sum()
    }
}

/// Output of a reasoning pass over a neuron graph.
#[derive(Debug, Clone, Default)]
pub struct ReasoningReport {
    pub nodes: Vec<ReasonedNode>,
    pub facts: Vec<ReasonedFact>,
    pub conflicts: Vec<ReasonerConflict>,
    /// Per-depth coverage metrics from the traversal.
    pub traversal_stats: TraversalStats,
}

impl ReasoningReport {
    #[must_use]
    pub fn total_facts(&self) -> usize {
        self.facts.len()
    }

    #[must_use]
    pub fn active_facts(&self) -> Vec<&ReasonedFact> {
        self.facts.iter().filter(|f| f.active).collect()
    }

    #[must_use]
    pub fn ended_facts(&self) -> Vec<&ReasonedFact> {
        self.facts.iter().filter(|f| !f.active).collect()
    }

    #[must_use]
    pub fn conflicting_paths(&self) -> Vec<(&PathBuf, &PathBuf)> {
        self.conflicts
            .iter()
            .map(|c| (&c.source, &c.target))
            .collect()
    }

    #[must_use]
    pub fn top_nodes(&self, n: usize) -> &[ReasonedNode] {
        &self.nodes[..n.min(self.nodes.len())]
    }

    #[must_use]
    pub fn top_facts(&self, n: usize) -> Vec<&ReasonedFact> {
        let mut facts: Vec<&ReasonedFact> = self.active_facts();
        facts.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.path.cmp(&b.path))
        });
        facts.into_iter().take(n).collect()
    }

    #[must_use]
    pub fn node_paths(&self) -> Vec<&PathBuf> {
        self.nodes.iter().map(|n| &n.path).collect()
    }

    #[must_use]
    pub fn seed_nodes(&self) -> Vec<&ReasonedNode> {
        self.nodes.iter().filter(|n| n.depth == 0).collect()
    }

    #[must_use]
    pub fn hop1_nodes(&self) -> Vec<&ReasonedNode> {
        self.nodes.iter().filter(|n| n.depth == 1).collect()
    }

    #[must_use]
    pub fn hop2_nodes(&self) -> Vec<&ReasonedNode> {
        self.nodes.iter().filter(|n| n.depth == 2).collect()
    }

    /// Render short, deterministic evidence lines suitable for later answer/context plumbing.
    pub fn summary_lines(&self, max_nodes: usize, max_facts: usize) -> Vec<String> {
        use std::collections::HashSet;
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

    /// Reconstruct traversal chains (seed → hop1 → hop2 …) and render them as
    /// human-readable strings.
    ///
    /// Each chain is a path from a seed node to a leaf node, traced backwards
    /// through `strongest_step.from`. Returns at most `max_chains` chains,
    /// sorted by the leaf node's score (descending) for determinism.
    #[must_use]
    pub fn chain_lines(&self, max_chains: usize) -> Vec<String> {
        if self.nodes.is_empty() || max_chains == 0 {
            return Vec::new();
        }

        // Index nodes by path for fast lookup.
        let by_path: std::collections::HashMap<&PathBuf, &ReasonedNode> =
            self.nodes.iter().map(|n| (&n.path, n)).collect();

        // Find leaf nodes: nodes whose path is not any other node's strongest_step parent.
        let parent_paths: std::collections::HashSet<&PathBuf> = self
            .nodes
            .iter()
            .filter_map(|n| n.strongest_step.as_ref().map(|s| &s.from))
            .collect();
        let mut leaves: Vec<&ReasonedNode> = self
            .nodes
            .iter()
            .filter(|n| !n.is_seed && !parent_paths.contains(&n.path))
            .collect();

        // Sort leaves by score descending, then path ascending for determinism.
        leaves.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.path.cmp(&b.path))
        });

        let mut lines = Vec::new();
        for leaf in leaves.into_iter().take(max_chains) {
            // Walk backwards from leaf to seed.
            let mut chain: Vec<String> = Vec::new();
            let mut current: &ReasonedNode = leaf;
            loop {
                chain.push(short_path(&current.path));
                match current
                    .strongest_step
                    .as_ref()
                    .and_then(|s| by_path.get(&s.from))
                {
                    Some(parent) => current = parent,
                    None => break,
                }
            }
            chain.reverse();
            if chain.len() >= 2 {
                lines.push(format!("[{}] score {:.2}", chain.join(" → "), leaf.score));
            }
        }

        lines
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
