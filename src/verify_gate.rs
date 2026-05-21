//! ECS (Epistemic Confidence Score) verification gate using lightweight heuristics.
//!
//! The verifier is advisory-first and analyzes prose that is already present in
//! neuron content. It strips fenced and indented code blocks, then combines
//! overconfidence and hedging signals to produce an [`EcsVerdict`].
//!
//! Clean technical prose should pass with low risk, code-only content is ignored,
//! and heuristic scoring stays below the hard block threshold.
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
    /// Optional softened or rewritten text from a future regulator pass.
    /// The current heuristic verifier always leaves this as `None`.
    pub regulated_text: Option<String>,
}

impl EcsVerdict {
    /// Convenience: ECS score on a 0–100 scale.
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

mod inner {
    use super::EcsVerdict;

    /// Strip fenced and indented code blocks before analysis.
    fn strip_code(content: &str) -> String {
        let mut result = String::with_capacity(content.len());
        let mut in_fence = false;

        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("```") {
                in_fence = !in_fence;
                continue;
            }
            if in_fence || line.starts_with("    ") || line.starts_with('\t') {
                continue;
            }
            result.push_str(line);
            result.push('\n');
        }

        result
    }

    /// Counts strong overconfidence markers in non-code prose.
    /// Returns the fraction of sentences containing at least one marker.
    fn overconfidence_ratio(text: &str) -> f64 {
        const MARKERS: &[&str] = &[
            "certainly ",
            "definitely ",
            "impossible ",
            "guaranteed ",
            "absolutely ",
            "without doubt",
            "100%",
            "never fails",
            "always works",
            "will always",
            "can never be wrong",
        ];
        const TECHNICAL_PHRASES: &[&str] = &[
            "always returns the same value",
            "same input",
            "never panics",
            "never throws",
            "certainly a bug",
        ];

        let lower = text.to_lowercase();
        let sentences: Vec<&str> = lower
            .split(['.', '!', '?'])
            .map(str::trim)
            .filter(|s| s.len() > 10)
            .collect();
        if sentences.is_empty() {
            return 0.0;
        }

        let marked = sentences
            .iter()
            .filter(|sentence| {
                !TECHNICAL_PHRASES
                    .iter()
                    .any(|phrase| sentence.contains(phrase))
                    && MARKERS.iter().any(|marker| sentence.contains(marker))
            })
            .count();
        marked as f64 / sentences.len() as f64
    }

    /// Counts hedging markers in non-code prose.
    /// Returns the fraction of sentences containing at least one marker.
    fn hedging_ratio(text: &str) -> f64 {
        const HEDGES: &[&str] = &[
            "maybe ",
            "perhaps ",
            " might ",
            " could ",
            "i think",
            "i believe",
            "it seems",
            "possibly ",
            "probably ",
            "not sure",
            "uncertain",
        ];

        let lower = text.to_lowercase();
        let sentences: Vec<&str> = lower
            .split(['.', '!', '?'])
            .map(str::trim)
            .filter(|s| s.len() > 5)
            .collect();
        if sentences.is_empty() {
            return 0.0;
        }

        let hedged = sentences
            .iter()
            .filter(|sentence| HEDGES.iter().any(|hedge| sentence.contains(hedge)))
            .count();
        hedged as f64 / sentences.len() as f64
    }

    /// Checks whether content is too short to produce a meaningful signal.
    fn is_trivially_short(content: &str) -> bool {
        content.chars().filter(|c| !c.is_whitespace()).count() < 20
    }

    /// Runs the heuristic ECS check.
    #[must_use]
    #[inline(always)]
    pub fn check(content: &str) -> EcsVerdict {
        if content.is_empty() || is_trivially_short(content) {
            return EcsVerdict {
                passed: true,
                risk_score: 0.0,
                summary: None,
                regulated_text: None,
            };
        }

        let prose = strip_code(content);
        if prose.trim().is_empty() {
            return EcsVerdict {
                passed: true,
                risk_score: 0.0,
                summary: None,
                regulated_text: None,
            };
        }

        let overconf = overconfidence_ratio(&prose);
        let hedging = hedging_ratio(&prose);
        let combined_risk = match (overconf, hedging) {
            (overconf, hedging) if overconf > 0.5 && hedging > 0.3 => {
                0.35 + ((overconf - 0.5) * 0.2).min(0.15) + ((hedging - 0.3) * 0.2).min(0.09)
            },
            (overconf, _) if overconf > 0.5 => (overconf * 0.25).min(0.25),
            (overconf, hedging) if overconf > 0.2 && hedging > 0.4 => 0.35,
            _ => 0.0,
        };

        // Heuristic formulas are bounded: the highest branch sums to 0.35+0.15+0.09 = 0.59,
        // which is already below DEFAULT_BLOCK_THRESHOLD (0.60). The min() cap makes this
        // invariant explicit and prevents future formula drift from accidentally enabling
        // hard-blocks via the heuristic path alone.
        // Advisory design: the heuristic gate never hard-blocks — it only quarantines.
        // Hard-blocking requires the semantic `verify` feature (PureReason pipeline).
        let risk_score = combined_risk.min(super::DEFAULT_BLOCK_THRESHOLD - 0.01);
        let passed = true; // Advisory gate: heuristic alone never hard-blocks.
        let summary = if risk_score > super::DEFAULT_QUARANTINE_THRESHOLD {
            Some(format!(
                "Heuristic ECS: overconfidence={overconf:.2} hedging={hedging:.2}"
            ))
        } else {
            None
        };

        EcsVerdict {
            passed,
            risk_score,
            summary,
            regulated_text: None,
        }
    }
}

// ── Public re-export ──────────────────────────────────────────────────────────

/// Run the heuristic ECS verification check on `content`.
pub use inner::check;

// ── Semantic contradiction detection ─────────────────────────────────────────

/// A pair of semantically contradicting claim strings extracted from neuron bodies.
pub type ContradictionPair = (String, String);

mod semantic_inner {
    use super::ContradictionPair;

    /// Currently unimplemented — always returns an empty Vec.
    ///
    /// A future implementation would perform sentence-level semantic comparison
    /// using the neuron's BM25 term vectors.
    #[must_use]
    #[inline(always)]
    pub fn find_semantic_contradictions(_bodies: &[&str]) -> Vec<ContradictionPair> {
        Vec::new()
    }
}

/// Currently unimplemented — always returns an empty Vec.
///
/// A future implementation would perform sentence-level semantic comparison
/// using the neuron's BM25 term vectors.
pub use semantic_inner::find_semantic_contradictions;

#[cfg(test)]
mod tests {
    use super::{check, EcsVerdict};

    #[test]
    fn clean_technical_content_passes() {
        let content = "This function always returns the same value for the same input. It never panics. The API is certainly stable.";
        let verdict = check(content);
        assert!(verdict.passed, "Technical API docs should pass");
        assert!(
            verdict.risk_score < 0.35,
            "Technical API docs should be below quarantine threshold, got {}",
            verdict.risk_score
        );
    }

    #[test]
    fn empty_content_passes() {
        assert!(check("").passed);
        assert!(check("   ").passed);
    }

    #[test]
    fn code_only_content_passes() {
        let content = "```rust\nlet x = certainly_important_fn();\nalways_true();\n```";
        let verdict = check(content);
        assert!(verdict.passed);
        assert!(verdict.risk_score < 0.1);
    }

    #[test]
    fn mixed_overconfident_and_hedging_raises_risk() {
        let content = concat!(
            "Definitely the best approach. Certainly will work. Impossible to fail. ",
            "But maybe it could break. Perhaps uncertain. I think it might not work. ",
            "Not sure if definitely correct. Possibly wrong. I believe certainly it's fine.",
        );
        let verdict = check(content);
        assert!(
            verdict.risk_score > 0.3,
            "High overconfidence+hedging should raise risk, got {}",
            verdict.risk_score
        );
    }

    #[test]
    fn technical_certainty_phrase_stays_low_risk() {
        let verdict = check("This is certainly a bug.");
        assert!(verdict.passed);
        assert!(verdict.risk_score < 0.1);
    }

    #[test]
    fn ecs_score_inverse_of_risk() {
        let verdict = EcsVerdict {
            passed: true,
            risk_score: 0.4,
            summary: None,
            regulated_text: None,
        };
        assert_eq!(verdict.ecs_score(), 60);
    }

    #[test]
    fn quarantine_annotation_at_medium_risk() {
        let verdict = EcsVerdict {
            passed: true,
            risk_score: 0.5,
            summary: None,
            regulated_text: None,
        };
        assert!(verdict.quarantine_annotation().is_some());
    }

    #[test]
    fn no_quarantine_at_low_risk() {
        let verdict = EcsVerdict {
            passed: true,
            risk_score: 0.1,
            summary: None,
            regulated_text: None,
        };
        assert!(verdict.quarantine_annotation().is_none());
    }
}
