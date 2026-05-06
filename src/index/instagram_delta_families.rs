use super::conversation_scan_support::session_score;
use super::instagram_delta_extractors::{
    extract_instagram_direct_delta_candidate, extract_instagram_growth_point,
    parse_instagram_delta_query, InstagramDeltaCandidate, InstagramDeltaQuery,
    InstagramGrowthPoint, InstagramGrowthRole,
};
use super::*;

impl NeuronIndex {
    pub(super) fn synthetic_instagram_delta_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        let query = parse_instagram_delta_query(task_lower)?;
        let candidates =
            self.ranked_numeric_aggregate_sessions(task, &query.required_terms, |line, lower| {
                extract_instagram_direct_delta_candidate(line, lower, &query).is_some()
                    || extract_instagram_growth_point(line, lower, &query).is_some()
            });
        let answer = best_same_session_instagram_delta(self, &candidates, &query)
            .or_else(|| best_grouped_fallback_instagram_delta(self, &query))?;
        self.write_synthetic_answer(
            "instagram-follower-increase",
            task,
            &answer.delta.to_string(),
            &answer.evidence,
        )
    }
}

fn best_same_session_instagram_delta(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &InstagramDeltaQuery,
) -> Option<InstagramDeltaCandidate> {
    candidates
        .iter()
        .filter_map(|(session_id, session_rank)| {
            let answer = resolve_instagram_delta(
                idx.session_answer_candidate_lines(session_id, usize::MAX),
                query,
            )?;
            Some((
                session_score(*session_rank, answer.score) + answer.delta.max(0) as usize,
                answer,
            ))
        })
        .max_by_key(|(score, _)| *score)
        .map(|(_, answer)| answer)
}

fn best_grouped_fallback_instagram_delta(
    idx: &NeuronIndex,
    query: &InstagramDeltaQuery,
) -> Option<InstagramDeltaCandidate> {
    let mut grouped: HashMap<String, Vec<(String, bool)>> = HashMap::new();
    for entry in idx
        .entries
        .iter()
        .filter(|entry| matches!(entry.kind, NeuronKind::Verbatim))
    {
        let Ok(content) = std::fs::read_to_string(&entry.neuron_path) else {
            continue;
        };
        let group_key = if entry.session_id.is_empty() {
            entry.neuron_path.to_string_lossy().to_string()
        } else {
            entry.session_id.clone()
        };
        let is_summary = is_session_summary_path(&entry.neuron_path);
        let lines = grouped.entry(group_key).or_default();
        for raw_line in strip_query_surface_section(&content).lines() {
            let line = raw_line.trim();
            if !line.is_empty() && is_session_answer_candidate_line(line) {
                lines.push((line.to_string(), is_summary));
            }
        }
    }
    grouped
        .into_values()
        .filter_map(|lines| resolve_instagram_delta(lines, query))
        .max_by_key(|answer| answer.score + answer.delta.max(0) as usize)
}

fn resolve_instagram_delta(
    lines: Vec<(String, bool)>,
    query: &InstagramDeltaQuery,
) -> Option<InstagramDeltaCandidate> {
    let mut best_direct = None;
    let mut points = Vec::new();

    for (position, (line, is_summary)) in lines.into_iter().enumerate() {
        let lower = line.to_ascii_lowercase();
        if let Some(mut direct) = extract_instagram_direct_delta_candidate(&line, &lower, query) {
            if is_summary {
                direct.score += 3;
            }
            upsert_best_delta_candidate(&mut best_direct, direct);
        }
        let Some(mut point) = extract_instagram_growth_point(&line, &lower, query) else {
            continue;
        };
        if is_summary {
            point.score += 3;
        }
        points.push((position, point));
    }

    let pair = best_instagram_delta_pair(&points, query);
    match (best_direct, pair) {
        (Some(direct), Some(pair)) => Some(if pair.score > direct.score {
            pair
        } else {
            direct
        }),
        (Some(direct), None) => Some(direct),
        (None, Some(pair)) => Some(pair),
        (None, None) => None,
    }
}

fn best_instagram_delta_pair(
    points: &[(usize, InstagramGrowthPoint)],
    query: &InstagramDeltaQuery,
) -> Option<InstagramDeltaCandidate> {
    let mut best = None;
    for (end_position, end_point) in points.iter().filter(|(_, point)| {
        matches!(point.role, InstagramGrowthRole::WindowEnd)
            || (query.window_days.is_none() && matches!(point.role, InstagramGrowthRole::Neutral))
    }) {
        let Some((start_position, start_point)) = points
            .iter()
            .filter(|(start_position, start_point)| {
                *start_position < *end_position
                    && !matches!(start_point.role, InstagramGrowthRole::WindowEnd)
            })
            .max_by_key(|(start_position, start_point)| {
                baseline_pair_score(*start_position, start_point, *end_position)
            })
        else {
            continue;
        };
        let delta = end_point.value - start_point.value;
        if delta <= 0 {
            continue;
        }
        let candidate = InstagramDeltaCandidate {
            delta,
            score: start_point.score
                + end_point.score
                + 12usize.saturating_sub(end_position.saturating_sub(*start_position).min(12)),
            evidence: dedupe_instagram_evidence([
                start_point.evidence.clone(),
                end_point.evidence.clone(),
            ]),
        };
        upsert_best_delta_candidate(&mut best, candidate);
    }
    best
}

fn baseline_pair_score(
    start_position: usize,
    start_point: &InstagramGrowthPoint,
    end_position: usize,
) -> usize {
    let proximity_bonus =
        12usize.saturating_sub(end_position.saturating_sub(start_position).min(12));
    start_point.score
        + proximity_bonus
        + usize::from(matches!(start_point.role, InstagramGrowthRole::Baseline)) * 6
}

fn upsert_best_delta_candidate(
    slot: &mut Option<InstagramDeltaCandidate>,
    candidate: InstagramDeltaCandidate,
) {
    let should_replace = slot
        .as_ref()
        .map(|best| {
            candidate.score > best.score
                || (candidate.score == best.score && candidate.delta > best.delta)
        })
        .unwrap_or(true);
    if should_replace {
        *slot = Some(candidate);
    }
}

fn dedupe_instagram_evidence<I>(evidence: I) -> Vec<String>
where
    I: IntoIterator<Item = String>,
{
    let mut deduped = Vec::new();
    let mut seen = HashSet::new();
    for line in evidence {
        if seen.insert(line.clone()) {
            deduped.push(line);
        }
    }
    deduped
}
