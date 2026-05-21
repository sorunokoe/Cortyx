use crate::types::SynapseWeight;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Confidence tier of a synapse edge type — controls the per-tier minimum propagated
/// score floor during graph traversal.
///
/// Tier floors are applied as an additional constraint on top of
/// [`crate::reasoner::TraversalOptions::min_propagated_score`] (the global floor):
/// effective floor = `tier_floor.max(global_floor)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SynapseConfidenceTier {
    /// AST-derived structural edges (Imports, Calls, Implements, ImplementedBy, CalledBy,
    /// ConceptExpands). High confidence — lenient tier floor so structural signal propagates
    /// widely; the global floor (0.12) governs in practice.
    Structural,
    /// Learned or inferred edges (SemanticRelated, TemporalFollows, Derived).
    /// Medium confidence — tier floor (0.20) exceeds the global default, tightening the gate.
    Semantic,
    /// Future speculative / early-Hebbian edges. Low confidence — strict floor.
    Speculative,
}

impl SynapseConfidenceTier {
    /// Minimum normalized propagated score required to traverse an edge of this tier.
    ///
    /// Applied as `tier_floor.max(TraversalOptions::min_propagated_score)` so these
    /// floors only restrict edges that are MORE speculative than the global default.
    #[must_use]
    pub fn min_propagated_score(self) -> f32 {
        match self {
            Self::Structural => 0.08,
            Self::Semantic => 0.20,
            Self::Speculative => 0.45,
        }
    }
}

/// The semantic type of a connection between two neurons.
///
/// Each type has an associated relevance multiplier applied during graph
/// traversal — structural edges (Imports, Implements) carry more weight
/// than loose semantic associations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, JsonSchema)]
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
    #[must_use]
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
    ///
    /// Types with a proper directed inverse (`Implements`/`Calls`) return that inverse.
    /// Symmetric types (`Contradicts`) return themselves.
    /// Types without a defined reverse (`Imports`, `Derived`, `TemporalFollows`,
    /// `ConceptExpands`) fall back to `SemanticRelated` — the weakest symmetric edge.
    #[must_use]
    pub fn inverse(self) -> SynapseType {
        match self {
            Self::Implements => Self::ImplementedBy,
            Self::ImplementedBy => Self::Implements,
            Self::Calls => Self::CalledBy,
            Self::CalledBy => Self::Calls,
            Self::Contradicts => Self::Contradicts,
            // Imports, Derived, TemporalFollows, ConceptExpands have no declared
            // reverse variant; fall back to the weakest symmetric edge type.
            _ => Self::SemanticRelated,
        }
    }

    /// Confidence tier of this synapse type, used to derive per-tier traversal floors.
    ///
    /// `Contradicts` is gated out before the floor check in traversal, so its tier
    /// is moot — `Structural` is used as a safe default.
    #[must_use]
    pub fn confidence_tier(&self) -> SynapseConfidenceTier {
        match self {
            Self::Imports
            | Self::Calls
            | Self::Implements
            | Self::ImplementedBy
            | Self::CalledBy
            | Self::ConceptExpands => SynapseConfidenceTier::Structural,
            Self::SemanticRelated | Self::TemporalFollows | Self::Derived => {
                SynapseConfidenceTier::Semantic
            },
            Self::Contradicts => SynapseConfidenceTier::Structural,
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
    pub weight: SynapseWeight,
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
    pub learned_weight: SynapseWeight,
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
    #[must_use]
    pub fn new(target: PathBuf, edge_type: SynapseType, reason: String) -> Self {
        Self {
            target,
            edge_type,
            weight: SynapseWeight::new(0.5),
            reason,
            learned_weight: SynapseWeight::ZERO,
            traversal_count: 0,
            last_co_activation_day: 0,
        }
    }

    /// Effective traversal weight, blending the static type multiplier with the
    /// learned weight once enough signal has accumulated.
    ///
    /// Blend schedule (blend = min(0.5, traversal_count / 100)):
    /// - Cold-start (`traversal_count` < 10 or `learned_weight` == 0.0):
    ///   returns `type_multiplier()` — identical to old behaviour.
    /// - Warm (`traversal_count` ≥ 10):
    ///   graduated blend from 10 % learned (at count=10) up to 50 % learned
    ///   (at count ≥ 100), clamped to [0.1, 1.0]. The static type multiplier
    ///   always contributes at least 50 % so domain knowledge is never
    ///   fully replaced by empirical signal.
    #[must_use]
    pub fn effective_weight(&self) -> f32 {
        let base = self.edge_type.type_multiplier();
        if self.traversal_count < 10 || self.learned_weight.is_zero() {
            return base;
        }
        let blend = 0.5_f32.min(self.traversal_count as f32 / 100.0);
        ((1.0 - blend) * base + blend * self.learned_weight.get()).clamp(0.1, 1.0)
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
        assert_eq!(s.weight, SynapseWeight::new(0.5));
        assert_eq!(s.edge_type, SynapseType::Imports);
        assert_eq!(s.learned_weight, SynapseWeight::ZERO);
        assert_eq!(s.traversal_count, 0);
    }

    #[test]
    fn effective_weight_cold_start_returns_type_multiplier() {
        let s = Synapse::new(PathBuf::from("a.md"), SynapseType::Imports, "test".into());
        assert_eq!(s.effective_weight(), SynapseType::Imports.type_multiplier());
    }

    #[test]
    fn effective_weight_blend_schedule() {
        // blend = min(traversal_count / 100, 0.5)
        let mut s = Synapse::new(PathBuf::from("a.md"), SynapseType::Imports, "t".into());
        s.learned_weight = SynapseWeight::new(1.0);
        let base = SynapseType::Imports.type_multiplier(); // 0.80

        // At count=10: blend=0.10 → 90% prior + 10% learned
        s.traversal_count = 10;
        let expected10 = 0.90 * base + 0.10 * 1.0;
        assert!(
            (s.effective_weight() - expected10).abs() < 1e-5,
            "at count=10 expected {expected10:.4}, got {:.4}",
            s.effective_weight()
        );

        // At count=25: blend=0.25 → 75% prior + 25% learned
        s.traversal_count = 25;
        let expected25 = 0.75 * base + 0.25 * 1.0;
        assert!(
            (s.effective_weight() - expected25).abs() < 1e-5,
            "at count=25 expected {expected25:.4}, got {:.4}",
            s.effective_weight()
        );

        // At count≥50: blend=0.50 (cap) → 50% prior + 50% learned
        s.traversal_count = 50;
        let expected_cap = 0.50 * base + 0.50 * 1.0;
        assert!(
            (s.effective_weight() - expected_cap).abs() < 1e-5,
            "at count=50 expected {expected_cap:.4} (cap), got {:.4}",
            s.effective_weight()
        );

        // count=100 same as 50 (blend clamped at 0.5)
        s.traversal_count = 100;
        assert!(
            (s.effective_weight() - expected_cap).abs() < 1e-5,
            "at count=100 should equal count=50 (blend capped at 0.5)"
        );
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

    #[test]
    fn synapse_tier_structural_types() {
        for ty in [
            SynapseType::Imports,
            SynapseType::Calls,
            SynapseType::Implements,
            SynapseType::ImplementedBy,
            SynapseType::CalledBy,
            SynapseType::ConceptExpands,
        ] {
            assert_eq!(
                ty.confidence_tier(),
                SynapseConfidenceTier::Structural,
                "{ty:?} should be Structural"
            );
        }
    }

    #[test]
    fn synapse_tier_semantic_types() {
        for ty in [
            SynapseType::SemanticRelated,
            SynapseType::TemporalFollows,
            SynapseType::Derived,
        ] {
            assert_eq!(
                ty.confidence_tier(),
                SynapseConfidenceTier::Semantic,
                "{ty:?} should be Semantic"
            );
        }
    }

    #[test]
    fn tier_min_scores_ordered() {
        assert!(
            SynapseConfidenceTier::Structural.min_propagated_score()
                < SynapseConfidenceTier::Semantic.min_propagated_score()
        );
        assert!(
            SynapseConfidenceTier::Semantic.min_propagated_score()
                < SynapseConfidenceTier::Speculative.min_propagated_score()
        );
    }

    #[test]
    fn tier_floor_additive_with_global_floor() {
        // Structural tier floor (0.08) < global default (0.12) → global wins.
        let global = crate::reasoner::TraversalOptions::default().min_propagated_score;
        assert!(
            SynapseConfidenceTier::Structural.min_propagated_score() < global,
            "Structural tier is deliberately lenient; global floor governs"
        );
        // Semantic floor (0.20) > global (0.12) → tier wins.
        assert!(
            SynapseConfidenceTier::Semantic.min_propagated_score() > global,
            "Semantic tier tightens the propagation floor"
        );
    }
}
