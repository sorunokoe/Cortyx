use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// The semantic type of a connection between two neurons.
///
/// Each type has an associated relevance multiplier applied during graph
/// traversal — structural edges (Imports, Implements) carry more weight
/// than loose semantic associations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SynapseType {
    #[default]
    /// General content similarity — weakest traversal signal (×0.50)
    SemanticRelated,
    /// A imports / depends on B (×0.80)
    Imports,
    /// A calls functions defined in B (×0.70)
    Calls,
    /// A implements an interface / trait from B (×0.90)
    Implements,
    /// B is the concrete implementation of A's interface (×0.80, reverse of Implements)
    ImplementedBy,
    /// B defines functions that A calls (×0.65, reverse of Calls)
    CalledBy,
    /// A and B hold conflicting information — excluded from co-activation (×0.40)
    Contradicts,
    /// B is the next session / event after A (×0.60)
    TemporalFollows,
    /// B's knowledge was derived from A (×0.70)
    Derived,
    /// Concept neuron → its constituent source files (×1.00, always propagates)
    ConceptExpands,
}

impl SynapseType {
    /// Weight multiplier applied during graph traversal.
    pub fn type_multiplier(&self) -> f32 {
        match self {
            Self::SemanticRelated => 0.50,
            Self::Imports => 0.80,
            Self::Calls => 0.70,
            Self::Implements => 0.90,
            Self::ImplementedBy => 0.80,
            Self::CalledBy => 0.65,
            Self::Contradicts => 0.40,
            Self::TemporalFollows => 0.60,
            Self::Derived => 0.70,
            Self::ConceptExpands => 1.00,
        }
    }

    /// Return the semantic inverse of this edge type for reverse graph construction.
    pub fn inverse(&self) -> SynapseType {
        match self {
            Self::Implements => Self::ImplementedBy,
            Self::ImplementedBy => Self::Implements,
            Self::Calls => Self::CalledBy,
            Self::CalledBy => Self::Calls,
            Self::Contradicts => Self::Contradicts,
            _ => Self::SemanticRelated,
        }
    }
}

/// A directed, typed, weighted edge in the neuron knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Synapse {
    /// Target neuron path (absolute or relative, must exist in the index).
    pub target: PathBuf,
    /// Semantic type — controls traversal multiplier and directionality.
    pub edge_type: SynapseType,
    /// Relevance weight in [0, 1]. Starts at 0.5; can be set manually via create_synapse.
    pub weight: f32,
    /// Human-readable reason written by the LLM.
    pub reason: String,
    /// Learned traversal weight — starts at `edge_type.type_multiplier()` and updates
    /// via EMA (α = 0.1) from citation signals in `record_hit`. After 10+ traversals,
    /// this weight encodes the actual helpfulness of this specific synapse edge.
    ///
    /// `#[serde(default)]` ensures backward compatibility: old index.json files that
    /// lack this field will deserialize to 0.0, then `effective_weight()` falls back
    /// to `type_multiplier()` so behaviour is identical before any learning occurs.
    #[serde(default)]
    pub learned_weight: f32,
    /// Number of times this synapse was evaluated — used to decide when the
    /// learned_weight has enough signal to trust.
    #[serde(default)]
    pub traversal_count: u32,
    /// Unix day of the last co-activation of source + target.
    /// Used by S-VII synapse temporal decay.
    #[serde(default)]
    pub last_co_activation_day: u32,
}

impl Synapse {
    pub fn new(target: PathBuf, edge_type: SynapseType, reason: String) -> Self {
        Self {
            target,
            edge_type,
            weight: 0.5,
            reason,
            learned_weight: 0.0,
            traversal_count: 0,
            last_co_activation_day: 0,
        }
    }

    /// Effective traversal weight, blending the static type multiplier with the
    /// learned weight once enough signal has accumulated.
    ///
    /// Cold-start (traversal_count < 10 or learned_weight == 0.0):
    ///   returns `type_multiplier()` — identical to old behaviour.
    /// Warm (traversal_count ≥ 10):
    ///   blends 50% static + 50% learned, clamped to [0.1, 1.0].
    pub fn effective_weight(&self) -> f32 {
        let base = self.edge_type.type_multiplier();
        if self.traversal_count < 10 || self.learned_weight <= 0.0 {
            return base;
        }
        let blend = 0.5_f32.min(self.traversal_count as f32 / 100.0);
        ((1.0 - blend) * base + blend * self.learned_weight).clamp(0.1, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synapse_type_multipliers_ordered() {
        assert!(
            SynapseType::Implements.type_multiplier()
                > SynapseType::SemanticRelated.type_multiplier()
        );
        assert_eq!(SynapseType::ConceptExpands.type_multiplier(), 1.0);
        assert!(
            SynapseType::Contradicts.type_multiplier()
                < SynapseType::SemanticRelated.type_multiplier()
        );
    }

    #[test]
    fn synapse_has_correct_defaults() {
        let s = Synapse::new(PathBuf::from("a.md"), SynapseType::Imports, "test".into());
        assert_eq!(s.weight, 0.5);
        assert_eq!(s.edge_type, SynapseType::Imports);
        assert_eq!(s.learned_weight, 0.0);
        assert_eq!(s.traversal_count, 0);
    }

    #[test]
    fn effective_weight_cold_start_returns_type_multiplier() {
        let s = Synapse::new(PathBuf::from("a.md"), SynapseType::Imports, "test".into());
        assert_eq!(s.effective_weight(), SynapseType::Imports.type_multiplier());
    }

    #[test]
    fn synapse_type_inverse_is_symmetric() {
        assert_eq!(
            SynapseType::Implements.inverse(),
            SynapseType::ImplementedBy
        );
        assert_eq!(
            SynapseType::ImplementedBy.inverse(),
            SynapseType::Implements
        );
        assert_eq!(SynapseType::Contradicts.inverse(), SynapseType::Contradicts);
    }
}
