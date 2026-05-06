//! Reasoning engine for traversing neuron graphs and extracting facts.
//!
//! Performs multi-hop reasoning across synapse edges to answer complex queries.

mod adaptive;
mod facts;
mod graph_ops;
mod traversal;
mod types;

// Re-export public API
pub use adaptive::{AdaptiveReasoner, IterationStats};
pub use traversal::GraphReasoner;
pub use types::{
    ReasonedFact, ReasonedNode, ReasonedStep, ReasonerConflict, ReasonerNeuron, ReasonerSeed,
    ReasoningReport, TraversalOptions, TraversalStats,
};
