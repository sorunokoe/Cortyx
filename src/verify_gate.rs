//! ECS (Epistemic Confidence Score) verification gate — PureReason integration.
//!
//! When compiled with `--features verify`, each gate function calls
//! `VerifierService` from `pure-reason-verifier` and returns an [`EcsVerdict`].
//!
//! When the feature is absent every function is a zero-cost no-op stub that
//! always returns a passing verdict, preserving identical behaviour for users
//! who don't opt in.
//!
//! # Risk thresholds
//!
//! | `risk_score` range | Meaning | Default action |
//! |---|---|---|
//! | 0.0 – 0.35 | Low — clean content | Pass silently |
//! | 0.35 – 0.60 | Medium — uncertain / hedged | Write with quarantine tag |
//! | 0.60 – 1.0 | High — likely hallucination | Reject write |

/// The outcome of a single ECS verification call.
#[derive(Debug, Clone)]
pub struct EcsVerdict {
    /// `true` when content is safe to persist (risk below the caller's threshold).
    pub passed: bool,
    /// Normalised risk in \[0.0, 1.0\]. Higher = more risky / likely hallucination.
    pub risk_score: f64,
    /// Human-readable one-line summary of findings, if any.
    pub summary: Option<String>,
    /// PureReason-rewritten regulated form of the text, when the pipeline softened
    /// overconfident claims. `None` when nothing was changed.
    pub regulated_text: Option<String>,
}

impl EcsVerdict {
    /// Convenience: ECS score on a 0–100 scale (mirrors PureReason's public API).
    #[must_use]
    pub fn ecs_score(&self) -> u8 {
        // ECS = 100 − risk_score×100, clamped to [0, 100]
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let score = ((1.0 - self.risk_score) * 100.0).clamp(0.0, 100.0) as u8;
        score
    }

    /// Returns a short annotation string for neuron sidecar metadata when the
    /// verdict is a medium-risk quarantine (not a hard block).
    #[must_use]
    pub fn quarantine_annotation(&self) -> Option<String> {
        if self.risk_score > 0.35 && self.risk_score <= 0.60 {
            Some(format!("ecs_risk:medium ecs_score:{}", self.ecs_score()))
        } else {
            None
        }
    }
}

/// Default risk threshold above which content is **blocked** from being persisted.
pub const DEFAULT_BLOCK_THRESHOLD: f64 = 0.60;

/// Default risk threshold above which content is **quarantined** (written with a
/// reduced staleness multiplier and a risk annotation in the sidecar JSON).
pub const DEFAULT_QUARANTINE_THRESHOLD: f64 = 0.35;

// ── No-op stubs (ECS/PureReason integration removed; feature was private) ────

mod inner {
    use super::EcsVerdict;

    /// No-op stub — always returns a passing verdict with zero overhead.
    #[inline(always)]
    pub fn check(_content: &str) -> EcsVerdict {
        EcsVerdict {
            passed: true,
            risk_score: 0.0,
            summary: None,
            regulated_text: None,
        }
    }
}

// ── Public re-export ──────────────────────────────────────────────────────────

/// Run an ECS verification check on `content`.
///
/// Always returns a passing verdict — the PureReason `verify` feature has been
/// removed from the published crate (it required a private sibling-crate dep).
pub use inner::check;

// ── Semantic contradiction detection ─────────────────────────────────────────

/// A pair of semantically contradicting claim strings extracted from neuron bodies.
pub type ContradictionPair = (String, String);

mod semantic_inner {
    use super::ContradictionPair;

    #[inline(always)]
    pub fn find_semantic_contradictions(_bodies: &[&str]) -> Vec<ContradictionPair> {
        Vec::new()
    }
}

pub use semantic_inner::find_semantic_contradictions;
