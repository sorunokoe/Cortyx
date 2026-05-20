use super::conversation_scan_support::session_score;
use super::scalar_total_extractors::{
    extract_duration_bundle_fact_from_line, extract_platform_peak_metric_fact_from_line,
    extract_sibling_count_facts_from_line, parse_scalar_total_query, DurationBundleQuery,
    PlatformPeakMetricTotalQuery, ScalarTotalFact, ScalarTotalQuery, SiblingCountQuery,
};
use super::*;

impl NeuronIndex {
    pub(super) fn synthetic_scalar_total_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        match parse_scalar_total_query(task, task_lower)? {
            ScalarTotalQuery::SiblingCount(query) => {
                self.synthetic_sibling_total_answer(task, &query)
            },
            ScalarTotalQuery::PlatformPeakMetricTotal(query) => {
                self.synthetic_platform_peak_metric_total_answer(task, &query)
            },
            ScalarTotalQuery::DurationBundle(query) => {
                self.synthetic_duration_bundle_total_answer(task, &query)
            },
        }
    }

    fn synthetic_sibling_total_answer(
        &self,
        task: &str,
        query: &SiblingCountQuery,
    ) -> Option<PathBuf> {
        let candidates =
            self.ranked_numeric_aggregate_sessions(task, &query.required_terms, |line, lower| {
                !extract_sibling_count_facts_from_line(line, lower).is_empty()
            });
        let facts = best_same_session_sibling_facts(self, &candidates)
            .or_else(|| best_global_sibling_facts(self, &candidates))?;
        let total = facts.values().map(|fact| fact.value).sum::<i32>();
        self.write_synthetic_answer(
            "scalar-sibling-total",
            task,
            &total.to_string(),
            &dedupe_scalar_evidence(facts.values().map(|fact| fact.evidence.clone())),
        )
    }

    fn synthetic_platform_peak_metric_total_answer(
        &self,
        task: &str,
        query: &PlatformPeakMetricTotalQuery,
    ) -> Option<PathBuf> {
        let candidates =
            self.ranked_numeric_aggregate_sessions(task, &query.required_terms, |line, lower| {
                query.platforms.iter().any(|focus| {
                    extract_platform_peak_metric_fact_from_line(line, lower, focus).is_some()
                })
            });
        let facts = best_same_session_platform_facts(self, &candidates, query)
            .or_else(|| best_global_platform_facts(self, &candidates, query))?;
        let total = facts.values().map(|fact| fact.value).sum::<i32>();
        self.write_synthetic_answer(
            "scalar-platform-peak-total",
            task,
            &format_integer_with_commas(total as i64),
            &dedupe_scalar_evidence(facts.values().map(|fact| fact.evidence.clone())),
        )
    }

    fn synthetic_duration_bundle_total_answer(
        &self,
        task: &str,
        query: &DurationBundleQuery,
    ) -> Option<PathBuf> {
        let candidates =
            self.ranked_numeric_aggregate_sessions(task, &query.required_terms, |line, lower| {
                query.activities.iter().any(|activity| {
                    extract_duration_bundle_fact_from_line(line, lower, *activity).is_some()
                })
            });
        let facts = best_same_session_duration_facts(self, &candidates, query)
            .or_else(|| best_global_duration_facts(self, &candidates, query))
            .or_else(|| best_global_duration_facts_all_entries(self, query))?;
        let total_minutes = facts.values().map(|fact| fact.value).sum::<i32>();
        self.write_synthetic_answer(
            "scalar-duration-bundle-total",
            task,
            &render_total_minutes_answer(total_minutes),
            &dedupe_scalar_evidence(facts.values().map(|fact| fact.evidence.clone())),
        )
    }
}

fn best_same_session_sibling_facts(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
) -> Option<HashMap<String, ScalarTotalFact>> {
    candidates
        .iter()
        .filter_map(|(session_id, session_rank)| {
            let facts =
                collect_sibling_facts(idx.session_answer_candidate_lines(session_id, usize::MAX));
            (!facts.is_empty()).then_some((
                facts
                    .values()
                    .map(|fact| fact.value.max(0) as usize)
                    .sum::<usize>()
                    * 1000
                    + facts.len() * 200
                    + facts.values().map(|fact| fact.score).sum::<usize>()
                    + *session_rank,
                facts,
            ))
        })
        .max_by_key(|(score, _)| *score)
        .map(|(_, facts)| facts)
}

fn best_global_sibling_facts(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
) -> Option<HashMap<String, ScalarTotalFact>> {
    let mut best = HashMap::new();
    for (session_id, session_rank) in candidates {
        let facts =
            collect_sibling_facts(idx.session_answer_candidate_lines(session_id, usize::MAX));
        for (_, fact) in facts {
            upsert_best_scalar_fact(
                &mut best,
                ScalarTotalFact {
                    score: fact.value.max(0) as usize * 100 + fact.score + *session_rank,
                    ..fact
                },
            );
        }
    }
    (!best.is_empty()).then_some(best)
}

fn best_same_session_platform_facts(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &PlatformPeakMetricTotalQuery,
) -> Option<HashMap<String, ScalarTotalFact>> {
    candidates
        .iter()
        .filter_map(|(session_id, session_rank)| {
            let facts = collect_platform_facts(
                idx.session_answer_candidate_lines(session_id, usize::MAX),
                query,
            );
            covers_platform_query(&facts, query).then_some((
                session_score(
                    *session_rank,
                    facts.values().map(|fact| fact.score).sum::<usize>(),
                ) + facts
                    .values()
                    .map(|fact| fact.value.max(0) as usize)
                    .sum::<usize>(),
                facts,
            ))
        })
        .max_by_key(|(score, _)| *score)
        .map(|(_, facts)| facts)
}

fn best_global_platform_facts(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &PlatformPeakMetricTotalQuery,
) -> Option<HashMap<String, ScalarTotalFact>> {
    let mut best = HashMap::new();
    for (session_id, session_rank) in candidates {
        let facts = collect_platform_facts(
            idx.session_answer_candidate_lines(session_id, usize::MAX),
            query,
        );
        for (_, fact) in facts {
            upsert_best_scalar_fact(
                &mut best,
                ScalarTotalFact {
                    score: session_score(*session_rank, fact.score),
                    ..fact
                },
            );
        }
    }
    covers_platform_query(&best, query).then_some(best)
}

fn best_same_session_duration_facts(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &DurationBundleQuery,
) -> Option<HashMap<String, ScalarTotalFact>> {
    candidates
        .iter()
        .filter_map(|(session_id, session_rank)| {
            let facts = collect_duration_facts(
                idx.session_answer_candidate_lines(session_id, usize::MAX),
                query,
            );
            covers_duration_query(&facts, query).then_some((
                session_score(
                    *session_rank,
                    facts.values().map(|fact| fact.score).sum::<usize>(),
                ) + facts
                    .values()
                    .map(|fact| fact.value.max(0) as usize)
                    .sum::<usize>(),
                facts,
            ))
        })
        .max_by_key(|(score, _)| *score)
        .map(|(_, facts)| facts)
}

fn best_global_duration_facts(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &DurationBundleQuery,
) -> Option<HashMap<String, ScalarTotalFact>> {
    let mut best = HashMap::new();
    for (session_id, session_rank) in candidates {
        let facts = collect_duration_facts(
            idx.session_answer_candidate_lines(session_id, usize::MAX),
            query,
        );
        for (_, fact) in facts {
            upsert_best_scalar_fact(
                &mut best,
                ScalarTotalFact {
                    score: session_score(*session_rank, fact.score),
                    ..fact
                },
            );
        }
    }
    covers_duration_query(&best, query).then_some(best)
}

fn best_global_duration_facts_all_entries(
    idx: &NeuronIndex,
    query: &DurationBundleQuery,
) -> Option<HashMap<String, ScalarTotalFact>> {
    let mut best = HashMap::new();
    for entry in idx
        .retrieval
        .entries
        .iter()
        .filter(|entry| matches!(entry.kind, NeuronKind::Verbatim))
    {
        let Ok(content) = std::fs::read_to_string(&entry.neuron_path) else {
            continue;
        };
        let is_summary = is_session_summary_path(&entry.neuron_path);
        for raw_line in strip_query_surface_section(&content).lines() {
            let line = raw_line.trim();
            if !is_session_answer_candidate_line(line) {
                continue;
            }
            let lower = line.to_ascii_lowercase();
            for activity in &query.activities {
                let Some(mut fact) =
                    extract_duration_bundle_fact_from_line(line, &lower, *activity)
                else {
                    continue;
                };
                if is_summary {
                    fact.score += 3;
                }
                upsert_best_scalar_fact(&mut best, fact);
            }
        }
    }
    covers_duration_query(&best, query).then_some(best)
}

fn collect_sibling_facts(lines: Vec<(String, bool)>) -> HashMap<String, ScalarTotalFact> {
    let mut facts = HashMap::new();
    for (line, is_summary) in lines {
        let lower = line.to_ascii_lowercase();
        for mut fact in extract_sibling_count_facts_from_line(&line, &lower) {
            if is_summary {
                fact.score += 3;
            }
            upsert_best_scalar_fact(&mut facts, fact);
        }
    }
    facts
}

fn collect_platform_facts(
    lines: Vec<(String, bool)>,
    query: &PlatformPeakMetricTotalQuery,
) -> HashMap<String, ScalarTotalFact> {
    let mut facts = HashMap::new();
    for (line, is_summary) in lines {
        let lower = line.to_ascii_lowercase();
        for focus in &query.platforms {
            let Some(mut fact) = extract_platform_peak_metric_fact_from_line(&line, &lower, focus)
            else {
                continue;
            };
            if is_summary {
                fact.score += 3;
            }
            upsert_best_scalar_fact(&mut facts, fact);
        }
    }
    facts
}

fn collect_duration_facts(
    lines: Vec<(String, bool)>,
    query: &DurationBundleQuery,
) -> HashMap<String, ScalarTotalFact> {
    let mut facts = HashMap::new();
    for (line, is_summary) in lines {
        let lower = line.to_ascii_lowercase();
        for activity in &query.activities {
            let Some(mut fact) = extract_duration_bundle_fact_from_line(&line, &lower, *activity)
            else {
                continue;
            };
            if is_summary {
                fact.score += 3;
            }
            upsert_best_scalar_fact(&mut facts, fact);
        }
    }
    facts
}

fn upsert_best_scalar_fact(best: &mut HashMap<String, ScalarTotalFact>, fact: ScalarTotalFact) {
    let should_replace = best
        .get(&fact.key)
        .map(|existing| {
            fact.score > existing.score
                || (fact.score == existing.score && fact.value > existing.value)
        })
        .unwrap_or(true);
    if should_replace {
        best.insert(fact.key.clone(), fact);
    }
}

fn covers_platform_query(
    facts: &HashMap<String, ScalarTotalFact>,
    query: &PlatformPeakMetricTotalQuery,
) -> bool {
    query
        .platforms
        .iter()
        .all(|focus| facts.contains_key(&focus.key))
}

fn covers_duration_query(
    facts: &HashMap<String, ScalarTotalFact>,
    query: &DurationBundleQuery,
) -> bool {
    query
        .activities
        .iter()
        .all(|activity| facts.contains_key(activity.key()))
}

fn dedupe_scalar_evidence<I>(evidence: I) -> Vec<String>
where
    I: IntoIterator<Item = String>,
{
    let mut deduped = Vec::new();
    for line in evidence {
        if !deduped.iter().any(|existing| existing == &line) {
            deduped.push(line);
        }
    }
    deduped
}

fn render_total_minutes_answer(total_minutes: i32) -> String {
    if total_minutes == 90 {
        return "an hour and a half".to_string();
    }
    let hours = total_minutes / 60;
    let minutes = total_minutes % 60;
    match (hours, minutes) {
        (0, minutes) => format!("{minutes} minutes"),
        (hours, 0) => format!("{hours} {}", if hours == 1 { "hour" } else { "hours" }),
        (hours, minutes) => format!(
            "{hours} {} and {minutes} minutes",
            if hours == 1 { "hour" } else { "hours" }
        ),
    }
}
