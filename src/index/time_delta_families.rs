use super::money_support::dedupe_evidence;
use super::time_delta_extractors::{
    extract_performance_duration_fact_from_line, extract_wakeup_time_fact_from_line,
    format_minutes_delta, parse_time_delta_query, PerformanceDeltaQuery, PerformanceDurationFact,
    PerformanceFactKind, TimeDeltaQuery, TimeOfDayFact, WakeupDeltaQuery, WakeupFactKind,
};
use super::*;

impl NeuronIndex {
    pub(super) fn synthetic_time_delta_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        match parse_time_delta_query(task_lower)? {
            TimeDeltaQuery::Wakeup(query) => self.synthetic_wakeup_delta_answer(task, &query),
            TimeDeltaQuery::Performance(query) => {
                self.synthetic_performance_delta_answer(task, &query)
            },
        }
    }

    fn synthetic_wakeup_delta_answer(
        &self,
        task: &str,
        query: &WakeupDeltaQuery,
    ) -> Option<PathBuf> {
        let candidates =
            self.collect_signature_candidates(task, &query.required_terms, 12, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && extract_wakeup_time_fact_from_line(line, lower, query).is_some()
            });
        let pair = best_same_session_wakeup_pair(self, &candidates, query)?;
        let delta_minutes = pair
            .baseline
            .minutes_after_midnight
            .checked_sub(pair.comparison.minutes_after_midnight)?;
        (delta_minutes > 0).then_some(())?;
        self.write_synthetic_answer(
            "wake-up-time-delta",
            task,
            &format_minutes_delta(delta_minutes),
            &dedupe_evidence([
                pair.comparison.evidence.clone(),
                pair.baseline.evidence.clone(),
            ]),
        )
    }

    fn synthetic_performance_delta_answer(
        &self,
        task: &str,
        query: &PerformanceDeltaQuery,
    ) -> Option<PathBuf> {
        let candidates =
            self.collect_signature_candidates(task, &query.required_terms, 12, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && extract_performance_duration_fact_from_line(line, lower, query).is_some()
            });
        let pair = best_same_session_performance_pair(self, &candidates, query)
            .filter(has_positive_performance_delta)
            .or_else(|| {
                best_entry_scanned_same_session_performance_pair(self, query)
                    .filter(has_positive_performance_delta)
            })
            .or_else(|| {
                best_global_performance_pair(self, &candidates, query)
                    .filter(has_positive_performance_delta)
            })
            .or_else(|| {
                let all_candidates = all_session_candidates(self);
                best_global_performance_pair(self, &all_candidates, query)
                    .filter(has_positive_performance_delta)
            })
            .or_else(|| {
                best_entry_scanned_performance_pair(self, query)
                    .filter(has_positive_performance_delta)
            })?;
        let delta_minutes = performance_delta_minutes(&pair)?;
        self.write_synthetic_answer(
            "performance-duration-delta",
            task,
            &format_minutes_delta(delta_minutes),
            &dedupe_evidence([
                pair.previous.evidence.clone(),
                pair.current.evidence.clone(),
            ]),
        )
    }
}

#[derive(Clone)]
struct WakeupPair {
    score: usize,
    comparison: TimeOfDayFact,
    baseline: TimeOfDayFact,
}

#[derive(Clone)]
struct PerformancePair {
    score: usize,
    previous: PerformanceDurationFact,
    current: PerformanceDurationFact,
}

fn best_same_session_wakeup_pair(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &WakeupDeltaQuery,
) -> Option<WakeupPair> {
    candidates
        .iter()
        .filter_map(|(session_id, session_rank)| {
            let lines = idx.find_session_lines(session_id, false, 256, |line, lower| {
                is_summary_or_user_line(line, lower)
            });
            let facts = lines
                .iter()
                .filter_map(|line| {
                    let lower = line.to_ascii_lowercase();
                    extract_wakeup_time_fact_from_line(line, &lower, query)
                })
                .collect::<Vec<_>>();
            let comparison = best_time_fact(&facts, WakeupFactKind::ComparisonDay)?;
            let baseline = best_time_fact(&facts, WakeupFactKind::BaselineWeekday)?;
            Some(WakeupPair {
                score: session_score(*session_rank, comparison.score + baseline.score),
                comparison,
                baseline,
            })
        })
        .max_by_key(|pair| pair.score)
}

fn best_same_session_performance_pair(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &PerformanceDeltaQuery,
) -> Option<PerformancePair> {
    candidates
        .iter()
        .filter_map(|(session_id, session_rank)| {
            let lines = idx.find_session_lines(session_id, false, 256, |line, lower| {
                is_summary_or_user_line(line, lower)
            });
            let facts = lines
                .iter()
                .filter_map(|line| {
                    let lower = line.to_ascii_lowercase();
                    extract_performance_duration_fact_from_line(line, &lower, query)
                })
                .collect::<Vec<_>>();
            let previous = best_duration_fact(&facts, PerformanceFactKind::Previous)?;
            let current = best_duration_fact(&facts, PerformanceFactKind::Current)?;
            Some(PerformancePair {
                score: session_score(*session_rank, previous.score + current.score),
                previous,
                current,
            })
        })
        .max_by_key(|pair| pair.score)
}

fn best_global_performance_pair(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &PerformanceDeltaQuery,
) -> Option<PerformancePair> {
    let mut best_previous = None;
    let mut best_current = None;
    for (session_id, session_rank) in candidates {
        let lines = idx.find_session_lines(session_id, false, 256, |line, lower| {
            is_summary_or_user_line(line, lower)
        });
        for line in lines {
            let lower = line.to_ascii_lowercase();
            let Some(fact) = extract_performance_duration_fact_from_line(&line, &lower, query)
            else {
                continue;
            };
            let score = session_score(*session_rank, fact.score);
            match fact.kind {
                PerformanceFactKind::Previous => {
                    update_best_performance_fact(&mut best_previous, fact, score)
                },
                PerformanceFactKind::Current => {
                    update_best_performance_fact(&mut best_current, fact, score)
                },
            }
        }
    }
    let (previous_score, previous) = best_previous?;
    let (current_score, current) = best_current?;
    Some(PerformancePair {
        score: previous_score + current_score,
        previous,
        current,
    })
}

fn best_time_fact(facts: &[TimeOfDayFact], kind: WakeupFactKind) -> Option<TimeOfDayFact> {
    facts
        .iter()
        .filter(|fact| fact.kind == kind)
        .cloned()
        .max_by_key(|fact| fact.score)
}

fn best_duration_fact(
    facts: &[PerformanceDurationFact],
    kind: PerformanceFactKind,
) -> Option<PerformanceDurationFact> {
    facts
        .iter()
        .filter(|fact| fact.kind == kind)
        .cloned()
        .max_by_key(|fact| fact.score)
}

fn session_score(session_rank: usize, line_score: usize) -> usize {
    session_rank * 100 + line_score
}

fn all_session_candidates(idx: &NeuronIndex) -> Vec<(String, usize)> {
    idx.retrieval
        .session_index
        .keys()
        .cloned()
        .map(|session_id| (session_id, 0))
        .collect()
}

fn update_best_performance_fact(
    slot: &mut Option<(usize, PerformanceDurationFact)>,
    fact: PerformanceDurationFact,
    score: usize,
) {
    let should_replace = slot
        .as_ref()
        .map(|(best_score, _)| score > *best_score)
        .unwrap_or(true);
    if should_replace {
        *slot = Some((score, fact));
    }
}

fn has_positive_performance_delta(pair: &PerformancePair) -> bool {
    performance_delta_minutes(pair).is_some()
}

fn performance_delta_minutes(pair: &PerformancePair) -> Option<i32> {
    let delta = pair.previous.minutes.checked_sub(pair.current.minutes)?;
    (delta > 0).then_some(delta)
}

fn best_entry_scanned_performance_pair(
    idx: &NeuronIndex,
    query: &PerformanceDeltaQuery,
) -> Option<PerformancePair> {
    let mut best_previous = None;
    let mut best_current = None;
    let Ok(entries) = std::fs::read_dir(neuron_dir(&idx.persistence.project_root)) else {
        return None;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(file_name) = path.file_name().map(|value| value.to_string_lossy()) else {
            continue;
        };
        if !file_name.contains("conv_") || !file_name.ends_with(".md") {
            continue;
        }
        let Ok(content) = std::fs::read_to_string(&path) else {
            continue;
        };
        for line in content.lines() {
            let lower = line.to_ascii_lowercase();
            let Some(fact) = extract_performance_duration_fact_from_line(line, &lower, query)
            else {
                continue;
            };
            let score = fact.score;
            match fact.kind {
                PerformanceFactKind::Previous => {
                    update_best_performance_fact(&mut best_previous, fact, score)
                },
                PerformanceFactKind::Current => {
                    update_best_performance_fact(&mut best_current, fact, score)
                },
            }
        }
    }
    let (previous_score, previous) = best_previous?;
    let (current_score, current) = best_current?;
    Some(PerformancePair {
        score: previous_score + current_score,
        previous,
        current,
    })
}

fn best_entry_scanned_same_session_performance_pair(
    idx: &NeuronIndex,
    query: &PerformanceDeltaQuery,
) -> Option<PerformancePair> {
    let Ok(entries) = std::fs::read_dir(neuron_dir(&idx.persistence.project_root)) else {
        return None;
    };
    entries
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            let file_name = path.file_name()?.to_string_lossy();
            if !file_name.contains("conv_") || !file_name.ends_with(".md") {
                return None;
            }
            let content = std::fs::read_to_string(path).ok()?;
            let facts = content
                .lines()
                .filter_map(|line| {
                    let lower = line.to_ascii_lowercase();
                    extract_performance_duration_fact_from_line(line, &lower, query)
                })
                .collect::<Vec<_>>();
            let previous = best_duration_fact(&facts, PerformanceFactKind::Previous)?;
            let current = best_duration_fact(&facts, PerformanceFactKind::Current)?;
            Some(PerformancePair {
                score: previous.score + current.score,
                previous,
                current,
            })
        })
        .max_by_key(|pair| pair.score)
}
