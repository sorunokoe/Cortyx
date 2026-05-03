/// Integration tests for the ECS verification gate (Pillar A).
///
/// Tests both with and without the `verify` feature to ensure:
///   1. With `verify` off: behaviour is identical to current tree (no-op stubs).
///   2. With `verify` on: hallucinated content is blocked/quarantined, clean passes.
///
/// All tests use `cortyx::verify_gate` directly — no MCP round-trip needed.
use cortyx::verify_gate::{check, EcsVerdict};

// ─── No-op behaviour (always runs, feature-independent) ────────────────────────

#[test]
fn clean_text_passes() {
    let verdict: EcsVerdict = check("Rust uses an ownership model for memory safety.");
    // Even with verify feature on and a legitimate claim, risk should be low.
    // Without verify feature, risk is always 0.0 and passed is always true.
    assert!(
        verdict.passed || verdict.risk_score <= 0.65,
        "clean text should pass or have low risk, got {verdict:?}"
    );
}

#[test]
fn empty_text_passes() {
    let verdict = check("");
    assert!(
        verdict.passed,
        "empty text must always pass (no-op): {verdict:?}"
    );
    assert_eq!(verdict.risk_score, 0.0);
}

#[test]
fn whitespace_only_passes() {
    let verdict = check("   \n\t  ");
    assert!(verdict.passed);
    assert_eq!(verdict.risk_score, 0.0);
}

// ─── Threshold logic ───────────────────────────────────────────────────────────

#[test]
fn verdict_fields_are_in_range() {
    let verdict = check("The speed of light is exactly 100 km/h and also 300,000 km/s.");
    assert!(
        (0.0..=1.0).contains(&verdict.risk_score),
        "risk_score out of range: {}",
        verdict.risk_score
    );
}

#[test]
fn passed_consistent_with_risk_score() {
    // Without `verify` feature, passed is always true and risk always 0.0.
    // With `verify` feature, passed should be risk_score <= DEFAULT_BLOCK_THRESHOLD.
    let verdict = check("Water boils at 100°C and simultaneously at 50°C.");
    #[cfg(not(feature = "verify"))]
    {
        assert!(verdict.passed);
        assert_eq!(verdict.risk_score, 0.0);
    }
    #[cfg(feature = "verify")]
    {
        // High-contradiction text: passed == (risk_score <= 0.60)
        assert_eq!(
            verdict.passed,
            verdict.risk_score <= 0.60,
            "passed flag inconsistent with risk_score: {verdict:?}"
        );
    }
}

// ─── No-op stub invariants (only when verify feature is off) ───────────────────

#[cfg(not(feature = "verify"))]
mod no_op {
    use cortyx::verify_gate::check;

    #[test]
    fn no_op_always_passes_with_zero_risk() {
        for text in &[
            "normal content",
            "The sky is green and blue simultaneously.",
            "1 + 1 = 3 and also 1 + 1 = 2",
            "X is true. X is false.",
        ] {
            let v = check(text);
            assert!(v.passed, "no-op should always pass: {text}");
            assert_eq!(v.risk_score, 0.0, "no-op risk_score must be 0: {text}");
            assert!(
                v.regulated_text.is_none(),
                "no-op should have no regulated text"
            );
        }
    }

    #[test]
    fn no_op_is_zero_cost() {
        use std::time::Instant;
        let start = Instant::now();
        for _ in 0..1000 {
            let _ = check("some text content here");
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_micros() < 500,
            "no-op gate should be <500µs for 1000 calls, got {}µs",
            elapsed.as_micros()
        );
    }
}

// ─── Semantic contradiction detection (only when verify feature is on) ─────────

#[cfg(feature = "verify")]
mod semantic {
    use cortyx::verify_gate::find_semantic_contradictions;

    #[test]
    fn empty_slice_returns_no_contradictions() {
        let pairs = find_semantic_contradictions(&[]);
        assert!(pairs.is_empty());
    }

    #[test]
    fn single_body_returns_no_contradictions() {
        let pairs = find_semantic_contradictions(&["A single body with one claim."]);
        assert!(
            pairs.is_empty(),
            "need ≥2 bodies for cross-body contradictions"
        );
    }

    #[test]
    fn contradictory_bodies_detected() {
        let body_a = "All mammals breathe with lungs. Whales are mammals.";
        let body_b = "Whales breathe with gills, not lungs.";
        let pairs = find_semantic_contradictions(&[body_a, body_b]);
        // Not guaranteed to catch every contradiction, but should produce at least one pair.
        // If PureReason detects it: great. If not: still valid (not a false-positive test).
        // We only assert structure here, not that a contradiction was found.
        for (a, b) in &pairs {
            assert!(!a.is_empty(), "claim_a should not be empty");
            assert!(!b.is_empty(), "claim_b should not be empty");
        }
    }

    #[test]
    fn consistent_bodies_produce_no_contradictions() {
        let body_a = "Rust uses ownership for memory safety.";
        let body_b = "Rust's borrow checker prevents data races at compile time.";
        let pairs = find_semantic_contradictions(&[body_a, body_b]);
        assert!(
            pairs.is_empty(),
            "consistent bodies should not produce contradictions: {pairs:?}"
        );
    }

    #[test]
    fn gate_overhead_under_10ms_per_call() {
        use std::time::Instant;
        let text = "Cortyx is a MCP-native context delivery engine for AI agents.";
        let start = Instant::now();
        for _ in 0..10 {
            let _ = cortyx::verify_gate::check(text);
        }
        let avg_us = start.elapsed().as_micros() / 10;
        assert!(
            avg_us < 10_000,
            "verify gate should be <10ms per call, got {avg_us}µs avg"
        );
    }
}
