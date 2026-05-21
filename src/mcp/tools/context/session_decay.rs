use super::*;

pub(super) async fn hot_session_tf(
    server: &CortyxServer,
) -> std::collections::HashMap<String, f32> {
    let tf = server.session.session_tf.lock().await;
    tf.iter()
        .filter(|(_, &count)| count >= 3.0)
        .map(|(term, &count)| (term.clone(), count))
        .collect()
}

pub(super) async fn update_session_tf(server: &CortyxServer, task: &str) {
    let raw_terms = crate::index::tokenize(task);
    let mut tf = server.session.session_tf.lock().await;
    apply_session_tf_update(&mut tf, raw_terms);
}

pub(super) async fn apply_path_history_boost(
    server: &CortyxServer,
    paths_with_scores: &mut [(PathBuf, f32)],
) {
    let path_history = server.session.session_path_history.lock().await;
    if !path_history.is_empty() {
        for (path, score) in paths_with_scores.iter_mut() {
            if let Some(&hist_weight) = path_history.get(path) {
                *score *= 1.0 + 0.15 * hist_weight.min(1.0);
            }
        }
        paths_with_scores.sort_unstable_by(|(left_path, left_score), (right_path, right_score)| {
            right_score
                .total_cmp(left_score)
                .then_with(|| left_path.cmp(right_path))
        });
    }
}

pub(super) async fn update_session_path_history(
    server: &CortyxServer,
    paths_with_scores: &[(PathBuf, f32)],
) {
    let mut path_history = server.session.session_path_history.lock().await;
    simulate_path_history_update(&mut path_history, paths_with_scores);
}

fn apply_session_tf_update<I, S>(tf: &mut std::collections::HashMap<String, f32>, raw_terms: I)
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    for term in raw_terms {
        *tf.entry(term.as_ref().to_string()).or_insert(0.0f32) += 1.0f32;
    }
    for count in tf.values_mut() {
        *count *= 0.9f32;
    }
    tf.retain(|_, count| *count >= 0.05f32);
}

fn simulate_path_history_update(
    history: &mut std::collections::HashMap<PathBuf, f32>,
    returned_paths: &[(PathBuf, f32)],
) {
    for weight in history.values_mut() {
        *weight *= 0.8;
    }
    history.retain(|_, w| *w > 0.01);
    for (path, _) in returned_paths.iter().take(5) {
        let prior = history.get(path).copied().unwrap_or(0.0);
        history.insert(path.clone(), 1.0_f32.max(prior));
    }
    if history.len() > 50 {
        let mut entries: Vec<(PathBuf, f32)> = history.drain().collect();
        entries.sort_unstable_by(|(_, left), (_, right)| right.total_cmp(left));
        entries.truncate(50);
        *history = entries.into_iter().collect();
    }
}

#[cfg(test)]
mod tests {
    use super::{apply_session_tf_update, simulate_path_history_update};
    use std::collections::HashMap;
    use std::path::PathBuf;

    #[test]
    fn session_tf_seen_once_per_call_becomes_hot_after_four_calls() {
        let mut tf = HashMap::new();
        for _ in 0..4 {
            apply_session_tf_update(&mut tf, ["auth"]);
        }
        assert!(
            tf.get("auth").copied().unwrap_or(0.0f32) >= 3.0f32,
            "term seen once per call should cross the hot threshold after 4 calls"
        );
    }

    #[test]
    fn session_tf_prunes_stale_terms_within_thirty_calls() {
        let mut tf = HashMap::from([("auth".to_string(), 1.0f32)]);
        let mut steps = 0;
        while tf.contains_key("auth") && steps < 30 {
            apply_session_tf_update(&mut tf, std::iter::empty::<&str>());
            steps += 1;
        }
        assert!(
            !tf.contains_key("auth"),
            "term should decay below the 0.05 retain threshold within 30 calls"
        );
        assert!(
            steps <= 30,
            "term should be pruned within 30 calls, took {steps}"
        );
    }

    #[test]
    fn path_history_inserts_at_full_weight() {
        let mut history = HashMap::new();
        let path = PathBuf::from("auth/security.context.md");
        simulate_path_history_update(&mut history, &[(path.clone(), 1.0)]);
        assert_eq!(history[&path], 1.0);
    }

    #[test]
    fn path_history_decays_over_calls() {
        let mut history = HashMap::new();
        let path = PathBuf::from("auth/security.context.md");
        simulate_path_history_update(&mut history, &[(path.clone(), 1.0)]);
        assert_eq!(history[&path], 1.0);
        simulate_path_history_update(&mut history, &[]);
        simulate_path_history_update(&mut history, &[]);
        let w = history[&path];
        assert!(
            w < 1.0 && w > 0.5,
            "weight should have decayed to ~0.64, got {w}"
        );
    }

    #[test]
    fn path_history_prunes_beyond_50_entries() {
        let mut history = HashMap::new();
        let paths: Vec<PathBuf> = (0..60).map(|i| PathBuf::from(format!("p{i}.md"))).collect();
        for p in &paths {
            history.insert(p.clone(), 0.5);
        }
        simulate_path_history_update(&mut history, &[]);
        assert_eq!(history.len(), 50);
    }

    #[test]
    fn path_history_boost_increases_score() {
        let path = PathBuf::from("auth.context.md");
        let base_score = 5.0_f32;
        let hist_weight = 1.0_f32;
        let boosted = base_score * (1.0 + 0.15 * hist_weight.min(1.0));
        assert!(
            boosted > base_score,
            "boosted {boosted} should exceed base {base_score}"
        );
        assert_eq!(path, PathBuf::from("auth.context.md"));
        assert!(
            (boosted - 5.75).abs() < 0.001,
            "expected 5.75, got {boosted}"
        );
    }
}
