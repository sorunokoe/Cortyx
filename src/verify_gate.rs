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

// ── Feature-enabled implementation ───────────────────────────────────────────

#[cfg(feature = "verify")]
mod inner {
    use super::EcsVerdict;
    use pure_reason_verifier::{ArtifactKind, VerificationRequest, VerifierService};

    /// Thread-local singleton so we pay `VerifierService::new()` only once per
    /// thread (stateless pipeline — cheap, but avoids repeated allocations on
    /// hot paths).
    std::thread_local! {
        static VERIFIER: VerifierService = VerifierService::new();
    }

    /// Run the full KantianPipeline on `content` and return an [`EcsVerdict`].
    ///
    /// # Errors
    ///
    /// Returns `None` on unexpected verifier panics (treated as passing verdict to
    /// avoid blocking writes due to verifier bugs).
    pub fn check(content: &str) -> EcsVerdict {
        VERIFIER.with(|svc| {
            let req = VerificationRequest {
                content: content.to_owned(),
                kind: ArtifactKind::Text,
                trace_id: None,
            };
            match svc.verify(req) {
                Ok(result) => EcsVerdict {
                    passed: result.verdict.passed,
                    risk_score: result.verdict.risk_score,
                    summary: Some(result.verdict.summary),
                    regulated_text: result.regulated_text,
                },
                Err(_) => {
                    // Fail open: a verifier error must never silently block writes.
                    EcsVerdict {
                        passed: true,
                        risk_score: 0.0,
                        summary: None,
                        regulated_text: None,
                    }
                },
            }
        })
    }
}

// ── No-op stubs when `verify` feature is absent ───────────────────────────────

#[cfg(not(feature = "verify"))]
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
/// With `--features verify`: calls PureReason's `VerifierService` and returns
/// a scored verdict.
///
/// Without `--features verify`: always returns a passing verdict with zero
/// overhead — no behaviour change for default builds.
pub use inner::check;

// ── Semantic contradiction detection ─────────────────────────────────────────

/// A pair of semantically contradicting claim strings extracted from neuron bodies.
pub type ContradictionPair = (String, String);

#[cfg(feature = "verify")]
mod semantic_inner {
    use super::ContradictionPair;
    use pure_reason_core::contradiction_detector::{extract_claims, find_contradictions};

    /// Extract claims from each text body and find logical contradictions across them.
    ///
    /// Intended for `cortyx_check_consistency` to surface semantic conflicts that have
    /// no explicit `Contradicts` synapse edge. Only the cross-body pairs are returned
    /// (within-body self-contradictions are not surfaced here).
    ///
    /// Returns at most 20 pairs to avoid flooding the output.
    pub fn find_semantic_contradictions(bodies: &[&str]) -> Vec<ContradictionPair> {
        let all_claims: Vec<String> = bodies.iter().flat_map(|b| extract_claims(b)).collect();
        if all_claims.len() < 2 {
            return Vec::new();
        }
        let analysis = find_contradictions(&all_claims);
        analysis
            .contradictions
            .into_iter()
            .map(|pair| (pair.claim_a, pair.claim_b))
            .take(20)
            .collect()
    }
}

#[cfg(not(feature = "verify"))]
mod semantic_inner {
    use super::ContradictionPair;

    #[inline(always)]
    pub fn find_semantic_contradictions(_bodies: &[&str]) -> Vec<ContradictionPair> {
        Vec::new()
    }
}

pub use semantic_inner::find_semantic_contradictions;
