use super::conversation_scan_support::session_score;
use super::count_total_extractors::{
    extract_meal_count_fact_from_line, extract_metric_count_fact_from_line,
    extract_online_course_completion_facts_from_line, parse_count_total_query, CountTotalFact,
    CountTotalQuery, MealBundleQuery, MetricBundleQuery, OnlineCourseTotalQuery,
};
use super::*;

impl NeuronIndex {
    pub(super) fn synthetic_count_total_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        match parse_count_total_query(task, task_lower)? {
            CountTotalQuery::MetricBundle(query) => {
                self.synthetic_metric_bundle_total_answer(task, &query)
            },
            CountTotalQuery::MealBundle(query) => {
                self.synthetic_meal_bundle_total_answer(task, &query)
            },
            CountTotalQuery::OnlineCourseTotal(query) => {
                self.synthetic_online_course_bundle_total_answer(task, &query)
            },
        }
    }

    fn synthetic_metric_bundle_total_answer(
        &self,
        task: &str,
        query: &MetricBundleQuery,
    ) -> Option<PathBuf> {
        let candidates =
            self.ranked_numeric_aggregate_sessions(task, &query.required_terms, |line, lower| {
                query.metrics.iter().any(|metric| {
                    extract_metric_count_fact_from_line(line, lower, *metric, &query.anchor_terms)
                        .is_some()
                })
            });
        let facts = best_same_session_metric_facts(self, &candidates, query)
            .or_else(|| best_global_metric_facts(self, &candidates, query))?;
        let total = facts.values().map(|fact| fact.count).sum::<i32>();
        self.write_synthetic_answer(
            "count-metric-bundle-total",
            task,
            &total.to_string(),
            &dedupe_count_evidence(facts.values().map(|fact| fact.evidence.clone())),
        )
    }

    fn synthetic_meal_bundle_total_answer(
        &self,
        task: &str,
        query: &MealBundleQuery,
    ) -> Option<PathBuf> {
        let candidates =
            self.ranked_numeric_aggregate_sessions(task, &query.required_terms, |line, lower| {
                query
                    .items
                    .iter()
                    .any(|focus| extract_meal_count_fact_from_line(line, lower, focus).is_some())
            });
        let facts = best_same_session_meal_facts(self, &candidates, query)
            .or_else(|| best_global_meal_facts(self, &candidates, query))?;
        let total = facts.values().map(|fact| fact.count).sum::<i32>();
        self.write_synthetic_answer(
            "count-meal-bundle-total",
            task,
            &format!("{total} meals"),
            &dedupe_count_evidence(facts.values().map(|fact| fact.evidence.clone())),
        )
    }

    fn synthetic_online_course_bundle_total_answer(
        &self,
        task: &str,
        query: &OnlineCourseTotalQuery,
    ) -> Option<PathBuf> {
        let candidates =
            self.ranked_numeric_aggregate_sessions(task, &query.required_terms, |line, lower| {
                !extract_online_course_completion_facts_from_line(line, lower).is_empty()
            });
        let facts = best_same_session_course_facts(self, &candidates)
            .or_else(|| best_global_course_facts(self, &candidates))?;
        let total = facts.values().map(|fact| fact.count).sum::<i32>();
        self.write_synthetic_answer(
            "count-online-course-total",
            task,
            &total.to_string(),
            &dedupe_count_evidence(facts.values().map(|fact| fact.evidence.clone())),
        )
    }
}

fn best_same_session_metric_facts(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &MetricBundleQuery,
) -> Option<HashMap<String, CountTotalFact>> {
    candidates
        .iter()
        .filter_map(|(session_id, session_rank)| {
            let facts = collect_metric_facts(
                idx.session_verbatim_answer_candidate_lines(session_id, usize::MAX),
                query,
            );
            covers_metric_query(&facts, query).then_some((
                session_score(
                    *session_rank,
                    facts.values().map(|fact| fact.score).sum::<usize>(),
                ) + facts
                    .values()
                    .map(|fact| fact.count.max(0) as usize)
                    .sum::<usize>()
                    * 10,
                facts,
            ))
        })
        .max_by_key(|(score, _)| *score)
        .map(|(_, facts)| facts)
}

fn best_global_metric_facts(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &MetricBundleQuery,
) -> Option<HashMap<String, CountTotalFact>> {
    let mut best = HashMap::new();
    for (session_id, session_rank) in candidates {
        let facts = collect_metric_facts(
            idx.session_verbatim_answer_candidate_lines(session_id, usize::MAX),
            query,
        );
        for (_, fact) in facts {
            upsert_best_fact(
                &mut best,
                CountTotalFact {
                    score: session_score(*session_rank, fact.score),
                    ..fact
                },
            );
        }
    }
    covers_metric_query(&best, query).then_some(best)
}

fn best_same_session_meal_facts(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &MealBundleQuery,
) -> Option<HashMap<String, CountTotalFact>> {
    candidates
        .iter()
        .filter_map(|(session_id, session_rank)| {
            let facts = collect_meal_facts(
                idx.session_verbatim_answer_candidate_lines(session_id, usize::MAX),
                query,
            );
            covers_meal_query(&facts, query).then_some((
                session_score(
                    *session_rank,
                    facts.values().map(|fact| fact.score).sum::<usize>(),
                ) + facts
                    .values()
                    .map(|fact| fact.count.max(0) as usize)
                    .sum::<usize>()
                    * 10,
                facts,
            ))
        })
        .max_by_key(|(score, _)| *score)
        .map(|(_, facts)| facts)
}

fn best_global_meal_facts(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &MealBundleQuery,
) -> Option<HashMap<String, CountTotalFact>> {
    let mut best = HashMap::new();
    for (session_id, session_rank) in candidates {
        let facts = collect_meal_facts(
            idx.session_verbatim_answer_candidate_lines(session_id, usize::MAX),
            query,
        );
        for (_, fact) in facts {
            upsert_best_fact(
                &mut best,
                CountTotalFact {
                    score: session_score(*session_rank, fact.score),
                    ..fact
                },
            );
        }
    }
    covers_meal_query(&best, query).then_some(best)
}

fn best_same_session_course_facts(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
) -> Option<HashMap<String, CountTotalFact>> {
    candidates
        .iter()
        .filter_map(|(session_id, session_rank)| {
            let facts =
                collect_course_facts(idx.session_answer_candidate_lines(session_id, usize::MAX));
            let total = facts
                .values()
                .map(|fact| fact.count.max(0) as usize)
                .sum::<usize>();
            (!facts.is_empty()).then_some((
                total * 1000
                    + facts.len() * 200
                    + facts.values().map(|fact| fact.score).sum::<usize>()
                    + *session_rank,
                facts,
            ))
        })
        .max_by_key(|(score, _)| *score)
        .map(|(_, facts)| facts)
}

fn best_global_course_facts(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
) -> Option<HashMap<String, CountTotalFact>> {
    let mut best = HashMap::new();
    for (session_id, session_rank) in candidates {
        let facts =
            collect_course_facts(idx.session_answer_candidate_lines(session_id, usize::MAX));
        for (_, fact) in facts {
            upsert_best_fact(
                &mut best,
                CountTotalFact {
                    score: fact.count.max(0) as usize * 100 + fact.score + *session_rank,
                    ..fact
                },
            );
        }
    }
    (!best.is_empty()).then_some(best)
}

fn collect_metric_facts(
    lines: Vec<String>,
    query: &MetricBundleQuery,
) -> HashMap<String, CountTotalFact> {
    let mut facts = HashMap::new();
    for line in lines {
        let lower = line.to_ascii_lowercase();
        for metric in &query.metrics {
            let Some(fact) =
                extract_metric_count_fact_from_line(&line, &lower, *metric, &query.anchor_terms)
            else {
                continue;
            };
            upsert_best_fact(&mut facts, fact);
        }
    }
    facts
}

fn collect_meal_facts(
    lines: Vec<String>,
    query: &MealBundleQuery,
) -> HashMap<String, CountTotalFact> {
    let mut facts = HashMap::new();
    for line in lines {
        let lower = line.to_ascii_lowercase();
        for focus in &query.items {
            let Some(fact) = extract_meal_count_fact_from_line(&line, &lower, focus) else {
                continue;
            };
            upsert_best_fact(&mut facts, fact);
        }
    }
    facts
}

fn collect_course_facts(lines: Vec<(String, bool)>) -> HashMap<String, CountTotalFact> {
    let mut facts = HashMap::new();
    for (line, is_summary) in lines {
        let lower = line.to_ascii_lowercase();
        for mut fact in extract_online_course_completion_facts_from_line(&line, &lower) {
            if is_summary {
                fact.score += 4;
            }
            upsert_best_fact(&mut facts, fact);
        }
    }
    facts
}

fn upsert_best_fact(best: &mut HashMap<String, CountTotalFact>, fact: CountTotalFact) {
    let should_replace = best
        .get(&fact.key)
        .map(|existing| {
            fact.score > existing.score
                || (fact.score == existing.score && fact.count > existing.count)
        })
        .unwrap_or(true);
    if should_replace {
        best.insert(fact.key.clone(), fact);
    }
}

fn covers_metric_query(facts: &HashMap<String, CountTotalFact>, query: &MetricBundleQuery) -> bool {
    query
        .metrics
        .iter()
        .all(|metric| facts.contains_key(metric.key()))
}

fn covers_meal_query(facts: &HashMap<String, CountTotalFact>, query: &MealBundleQuery) -> bool {
    query
        .items
        .iter()
        .all(|focus| facts.contains_key(&focus.key))
}

fn dedupe_count_evidence<I>(evidence: I) -> Vec<String>
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
