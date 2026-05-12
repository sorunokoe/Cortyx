//! Typed score values — prevents passing the wrong score type to scoring functions.
//!
//! Each type has different invariants:
//! - `SynapseWeight`, `ConfidenceScore`, `QualityScore` — clamped to `[0.0, 1.0]`
//! - `StalenessMultiplier` — clamped to `(0.0, 1.0]`; zero is invalid (would suppress permanently)
//! - `BM25Score` — unbounded non-negative; raw BM25 output should not be confused with normalised scores

use std::fmt;
use std::ops::{Add, Mul};

use serde::{Deserialize, Serialize};

// ─── SynapseWeight ───────────────────────────────────────────────────────────

/// Relevance weight of a synapse edge, in `[0.0, 1.0]`.
///
/// Combines the static `type_multiplier` (from `SynapseType`) and the
/// EMA-learned `learned_weight`. Values are clamped on construction.
#[derive(Debug, Clone, Copy, Default, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct SynapseWeight(f32);

impl SynapseWeight {
    pub const ZERO: Self = Self(0.0);
    pub const ONE: Self = Self(1.0);

    /// Clamp `v` to `[0.0, 1.0]`.
    pub fn new(v: f32) -> Self {
        Self(v.clamp(0.0, 1.0))
    }

    pub fn get(self) -> f32 {
        self.0
    }

    pub fn is_zero(self) -> bool {
        self.0 == 0.0
    }

    /// Blend two weights: `self * (1 - alpha) + other * alpha`.
    pub fn ema(self, other: Self, alpha: f32) -> Self {
        let alpha = alpha.clamp(0.0, 1.0);
        Self::new(self.0 * (1.0 - alpha) + other.0 * alpha)
    }
}

impl Mul for SynapseWeight {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::new(self.0 * rhs.0)
    }
}

impl fmt::Display for SynapseWeight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.3}", self.0)
    }
}

// ─── ConfidenceScore ─────────────────────────────────────────────────────────

/// General-purpose confidence score, in `[0.0, 1.0]`.
///
/// Used for neuron git-state confidence and answer surface confidence.
/// Different from `BM25Score` which is raw and unbounded.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct ConfidenceScore(f32);

impl ConfidenceScore {
    pub const FULL: Self = Self(1.0);
    pub const ZERO: Self = Self(0.0);

    pub fn new(v: f32) -> Self {
        Self(v.clamp(0.0, 1.0))
    }

    pub fn get(self) -> f32 {
        self.0
    }

    pub fn is_high(self) -> bool {
        self.0 >= 0.8
    }

    pub fn is_low(self) -> bool {
        self.0 < 0.5
    }
}

impl fmt::Display for ConfidenceScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.3}", self.0)
    }
}

// ─── QualityScore ────────────────────────────────────────────────────────────

/// AST-derived content quality score, in `[0.0, 1.0]`.
///
/// Computed from term overlap between the neuron vocabulary and source AST.
/// Below `0.4` a penalty multiplier is applied during activation.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct QualityScore(f32);

impl QualityScore {
    pub const FULL: Self = Self(1.0);

    pub fn new(v: f32) -> Self {
        Self(v.clamp(0.0, 1.0))
    }

    pub fn get(self) -> f32 {
        self.0
    }

    /// Returns `true` if quality is below the penalty threshold (0.4).
    pub fn is_below_penalty_threshold(self) -> bool {
        self.0 < 0.4
    }
}

// ─── BM25Score ───────────────────────────────────────────────────────────────

/// Raw BM25 retrieval score, in `[0.0, ∞)`.
///
/// Not normalised to `[0.0, 1.0]`. Never mix with `ConfidenceScore` or
/// `SynapseWeight`; use `BM25Score` only for ranking within a single query result.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct BM25Score(f32);

impl BM25Score {
    pub const ZERO: Self = Self(0.0);

    /// Construct from a raw BM25 output. Negative values are clamped to zero.
    pub fn new(v: f32) -> Self {
        Self(v.max(0.0))
    }

    pub fn get(self) -> f32 {
        self.0
    }

    pub fn is_zero(self) -> bool {
        self.0 == 0.0
    }

    /// Score exceeds the high-confidence threshold (8.0 by default).
    pub fn is_high_confidence(self, threshold: f32) -> bool {
        self.0 >= threshold
    }

    /// Score is below the low-confidence threshold (4.0 by default).
    pub fn is_low_confidence(self, threshold: f32) -> bool {
        self.0 < threshold
    }
}

impl Add for BM25Score {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.0 + rhs.0)
    }
}

impl fmt::Display for BM25Score {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4}", self.0)
    }
}

// ─── StalenessMultiplier ─────────────────────────────────────────────────────

/// Activation score multiplier encoding source-file staleness.
///
/// `1.0` means the neuron is current; lower values demote stale neurons.
/// The minimum is `f32::EPSILON` (never zero — zero would permanently suppress).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct StalenessMultiplier(f32);

impl StalenessMultiplier {
    /// A fresh (up-to-date) neuron.
    pub const FRESH: Self = Self(1.0);
    /// A stale neuron (source file changed).
    pub const STALE: Self = Self(0.5);

    pub fn new(v: f32) -> Self {
        // Clamp to (0.0, 1.0] — never zero.
        Self(v.clamp(f32::EPSILON, 1.0))
    }

    pub fn get(self) -> f32 {
        self.0
    }

    pub fn is_fresh(self) -> bool {
        self.0 >= 1.0
    }
}

impl Mul<BM25Score> for StalenessMultiplier {
    type Output = BM25Score;
    fn mul(self, rhs: BM25Score) -> BM25Score {
        BM25Score::new(self.0 * rhs.0)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synapse_weight_clamps() {
        assert_eq!(SynapseWeight::new(-1.0).get(), 0.0);
        assert_eq!(SynapseWeight::new(2.0).get(), 1.0);
        assert_eq!(SynapseWeight::new(0.5).get(), 0.5);
    }

    #[test]
    fn synapse_weight_ema() {
        let a = SynapseWeight::new(0.8);
        let b = SynapseWeight::new(0.2);
        let blended = a.ema(b, 0.1);
        // 0.8 * 0.9 + 0.2 * 0.1 = 0.72 + 0.02 = 0.74
        assert!((blended.get() - 0.74).abs() < 1e-5);
    }

    #[test]
    fn bm25_score_never_negative() {
        assert_eq!(BM25Score::new(-5.0).get(), 0.0);
    }

    #[test]
    fn staleness_multiplier_never_zero() {
        assert!(StalenessMultiplier::new(0.0).get() > 0.0);
    }

    #[test]
    fn staleness_mul_bm25() {
        let s = StalenessMultiplier::STALE;
        let score = BM25Score::new(4.0);
        let result = s * score;
        assert!((result.get() - 2.0).abs() < 1e-5);
    }

    #[test]
    fn serde_round_trip() {
        let w = SynapseWeight::new(0.75);
        let json = serde_json::to_string(&w).unwrap();
        let back: SynapseWeight = serde_json::from_str(&json).unwrap();
        assert_eq!(w, back);
    }
}
