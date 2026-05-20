use super::super::{ActivationStage, QueryContext, ScoredCandidate};
use super::sort_candidates;
use std::collections::HashMap;

pub(super) const MIN_CORETURN_COUNT: u32 = 5;
pub(super) const CORETURN_BOOST_STEP: f32 = 0.05;
pub(super) const MAX_CORETURN_BONUS_STEPS: u32 = 3;

/// Boosts pairs of top-ranked candidates that have a strong co-return history.
pub struct CoreturnBoostStage;

impl ActivationStage for CoreturnBoostStage {
    fn name(&self) -> &'static str {
        "coreturn_boost"
    }

    fn apply(&self, ctx: &QueryContext<'_>, candidates: &mut Vec<ScoredCandidate>) {
        if candidates.len() < 2 {
            return;
        }
        sort_candidates(candidates);
        let snapshot = candidates.iter().take(10).cloned().collect::<Vec<_>>();
        let Ok(counts) = ctx.feedback.co_return_counts.lock() else {
            return;
        };

        let mut boosts: HashMap<usize, f32> = HashMap::new();
        for i in 0..snapshot.len() {
            for j in (i + 1)..snapshot.len() {
                let a = snapshot[i].entry_idx.min(snapshot[j].entry_idx);
                let b = snapshot[i].entry_idx.max(snapshot[j].entry_idx);
                let Some(&count) = counts.get(&(a, b)) else {
                    continue;
                };
                if count < MIN_CORETURN_COUNT {
                    continue;
                }
                let bonus_steps = count
                    .saturating_sub(MIN_CORETURN_COUNT - 1)
                    .min(MAX_CORETURN_BONUS_STEPS);
                let boost = 1.0 + (bonus_steps as f32 * CORETURN_BOOST_STEP);
                boosts
                    .entry(snapshot[i].entry_idx)
                    .and_modify(|value| *value *= boost)
                    .or_insert(boost);
                boosts
                    .entry(snapshot[j].entry_idx)
                    .and_modify(|value| *value *= boost)
                    .or_insert(boost);
            }
        }
        drop(counts);

        for candidate in candidates.iter_mut() {
            if let Some(boost) = boosts.get(&candidate.entry_idx) {
                candidate.score *= *boost;
            }
        }
        sort_candidates(candidates);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::core::pipeline::types::{test_entry, QueryContextFixture};
    use crate::neuron::NeuronKind;

    #[test]
    fn name_returns_expected_string() {
        assert_eq!(CoreturnBoostStage.name(), "coreturn_boost");
    }

    #[test]
    fn empty_candidates_is_passthrough() {
        let fixture = QueryContextFixture::new(vec![]);
        let ctx = fixture.ctx("anything");
        let mut candidates = Vec::new();

        CoreturnBoostStage.apply(&ctx, &mut candidates);

        assert!(candidates.is_empty());
    }

    #[test]
    fn qualifying_pair_boosts_both_candidates() {
        let first = test_entry("a.md", NeuronKind::Verbatim, &[("foo", 1.0)]);
        let second = test_entry("b.md", NeuronKind::Verbatim, &[("foo", 1.0)]);
        let fixture = QueryContextFixture::new(vec![first, second]);
        let mut counts = fixture.co_return_counts.lock().unwrap();
        counts.insert((0, 1), 5);
        drop(counts);
        let ctx = fixture.ctx("foo");
        let mut candidates = vec![
            ScoredCandidate::new(0, 1.0, 10),
            ScoredCandidate::new(1, 0.9, 10),
        ];

        CoreturnBoostStage.apply(&ctx, &mut candidates);

        assert!(candidates[0].score > 1.0);
        assert!(candidates[1].score > 0.9);
    }

    #[test]
    fn below_threshold_pair_is_ignored() {
        let first = test_entry("a.md", NeuronKind::Verbatim, &[("foo", 1.0)]);
        let second = test_entry("b.md", NeuronKind::Verbatim, &[("foo", 1.0)]);
        let fixture = QueryContextFixture::new(vec![first, second]);
        let mut counts = fixture.co_return_counts.lock().unwrap();
        counts.insert((0, 1), 4);
        drop(counts);
        let ctx = fixture.ctx("foo");
        let mut candidates = vec![
            ScoredCandidate::new(0, 1.0, 10),
            ScoredCandidate::new(1, 0.9, 10),
        ];

        CoreturnBoostStage.apply(&ctx, &mut candidates);

        assert_eq!(candidates[0].score, 1.0);
        assert_eq!(candidates[1].score, 0.9);
    }

    #[test]
    fn qualifying_pair_can_change_ordering() {
        let first = test_entry("a.md", NeuronKind::Verbatim, &[("foo", 1.0)]);
        let second = test_entry("b.md", NeuronKind::Verbatim, &[("foo", 1.0)]);
        let third = test_entry("c.md", NeuronKind::Verbatim, &[("foo", 1.0)]);
        let fixture = QueryContextFixture::new(vec![first, second, third]);
        let mut counts = fixture.co_return_counts.lock().unwrap();
        counts.insert((1, 2), 6);
        drop(counts);
        let ctx = fixture.ctx("foo");
        let mut candidates = vec![
            ScoredCandidate::new(0, 1.0, 10),
            ScoredCandidate::new(1, 0.95, 10),
            ScoredCandidate::new(2, 0.94, 10),
        ];

        CoreturnBoostStage.apply(&ctx, &mut candidates);

        assert!(candidates[0].entry_idx != 0);
    }

    #[test]
    fn caps_bonus_steps_for_large_counts() {
        let first = test_entry("a.md", NeuronKind::Verbatim, &[("foo", 1.0)]);
        let second = test_entry("b.md", NeuronKind::Verbatim, &[("foo", 1.0)]);
        let fixture = QueryContextFixture::new(vec![first, second]);
        let mut counts = fixture.co_return_counts.lock().unwrap();
        counts.insert((0, 1), 100);
        drop(counts);
        let ctx = fixture.ctx("foo");
        let mut candidates = vec![
            ScoredCandidate::new(0, 1.0, 10),
            ScoredCandidate::new(1, 1.0, 10),
        ];

        CoreturnBoostStage.apply(&ctx, &mut candidates);

        let expected = 1.0 + (MAX_CORETURN_BONUS_STEPS as f32 * CORETURN_BOOST_STEP);
        assert!((candidates[0].score - expected).abs() < 0.0001);
    }

    #[test]
    fn only_top_ten_candidates_are_considered() {
        let entries = (0..11)
            .map(|idx| test_entry(&format!("{idx}.md"), NeuronKind::Verbatim, &[("foo", 1.0)]))
            .collect::<Vec<_>>();
        let fixture = QueryContextFixture::new(entries);
        let mut counts = fixture.co_return_counts.lock().unwrap();
        counts.insert((9, 10), 6);
        drop(counts);
        let ctx = fixture.ctx("foo");
        let mut candidates = (0..11)
            .map(|idx| ScoredCandidate::new(idx, 20.0 - idx as f32, 10))
            .collect::<Vec<_>>();

        CoreturnBoostStage.apply(&ctx, &mut candidates);

        let eleventh = candidates
            .iter()
            .find(|candidate| candidate.entry_idx == 10)
            .unwrap();
        assert!((eleventh.score - 10.0).abs() < 0.0001);
    }
}
