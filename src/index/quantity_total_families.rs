use super::conversation_scan_support::session_score;
use super::quantity_total_extractors::{
    extract_road_trip_distance_fact_from_line, extract_stay_days_fact_from_line,
    extract_weekend_hike_distance_fact_from_line, format_distance_total_answer,
    parse_quantity_total_query, ConsecutiveWeekendHikeDistanceQuery, QuantityTotalFact,
    QuantityTotalQuery, RoadTripDistanceQuery, StayDaysQuery,
};
use super::*;

impl NeuronIndex {
    pub(super) fn synthetic_quantity_total_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        match parse_quantity_total_query(task_lower)? {
            QuantityTotalQuery::RoadTripDistance(query) => {
                self.synthetic_road_trip_distance_total_answer(task, &query)
            },
            QuantityTotalQuery::ConsecutiveWeekendHikeDistance(query) => {
                self.synthetic_weekend_hike_distance_total_answer(task, &query)
            },
            QuantityTotalQuery::StayDays(query) => {
                self.synthetic_stay_days_total_answer(task, &query)
            },
        }
    }

    fn synthetic_road_trip_distance_total_answer(
        &self,
        task: &str,
        query: &RoadTripDistanceQuery,
    ) -> Option<PathBuf> {
        let candidates =
            self.ranked_numeric_aggregate_sessions(task, &query.required_terms, |line, lower| {
                extract_road_trip_distance_fact_from_line(line, lower).is_some()
            });
        let facts = best_same_session_road_trip_facts(self, &candidates, query)
            .or_else(|| best_global_road_trip_facts(self, &candidates, query))?;
        let total_miles = facts.values().map(|fact| fact.value).sum::<f32>();
        self.write_synthetic_answer(
            "quantity-road-trip-distance-total",
            task,
            &format_distance_total_answer(total_miles),
            &dedupe_quantity_evidence(facts.values().map(|fact| fact.evidence.clone())),
        )
    }

    fn synthetic_weekend_hike_distance_total_answer(
        &self,
        task: &str,
        query: &ConsecutiveWeekendHikeDistanceQuery,
    ) -> Option<PathBuf> {
        let candidates =
            self.ranked_numeric_aggregate_sessions(task, &query.required_terms, |line, lower| {
                extract_weekend_hike_distance_fact_from_line(line, lower).is_some()
            });
        let facts = best_same_session_weekend_hike_facts(self, &candidates)
            .or_else(|| best_global_weekend_hike_facts(self, &candidates))?;
        let total_miles = facts.values().map(|fact| fact.value).sum::<f32>();
        self.write_synthetic_answer(
            "quantity-weekend-hike-distance-total",
            task,
            &format_distance_total_answer(total_miles),
            &dedupe_quantity_evidence(facts.values().map(|fact| fact.evidence.clone())),
        )
    }

    fn synthetic_stay_days_total_answer(
        &self,
        task: &str,
        query: &StayDaysQuery,
    ) -> Option<PathBuf> {
        let candidates =
            self.ranked_numeric_aggregate_sessions(task, &query.required_terms, |line, lower| {
                extract_stay_days_fact_from_line(line, lower, query).is_some()
            });
        let facts = best_same_session_stay_day_facts(self, &candidates, query)
            .or_else(|| best_global_stay_day_facts(self, &candidates, query))?;
        let total_days = facts.values().map(|fact| fact.value).sum::<f32>();
        let ambiguous_facts = facts
            .values()
            .filter(|fact| fact.alternate_value.is_some())
            .collect::<Vec<_>>();
        let alternate_total_days = (!ambiguous_facts.is_empty())
            .then(|| {
                facts
                    .values()
                    .map(|fact| fact.alternate_value.unwrap_or(fact.value))
                    .sum::<f32>()
            })
            .filter(|alternate| (*alternate - total_days).abs() > 0.01);
        let answer = match (alternate_total_days, ambiguous_facts.as_slice()) {
            (Some(alternate_total_days), [fact]) => format!(
                "{} (or {}, if {} is considered as {})",
                format_aggregate_duration_answer(total_days, "day"),
                format_aggregate_duration_answer(alternate_total_days, "day"),
                fact.alternate_reason.as_deref().unwrap_or("the date range"),
                format_aggregate_duration_answer(fact.alternate_value.unwrap_or(fact.value), "day"),
            ),
            (Some(alternate_total_days), _) => format!(
                "{} (or {} if the date ranges are counted inclusively)",
                format_aggregate_duration_answer(total_days, "day"),
                format_aggregate_duration_answer(alternate_total_days, "day"),
            ),
            (None, _) => format_aggregate_duration_answer(total_days, "day"),
        };
        self.write_synthetic_answer(
            "quantity-stay-days-total",
            task,
            &answer,
            &dedupe_quantity_evidence(facts.values().map(|fact| fact.evidence.clone())),
        )
    }
}

fn best_same_session_road_trip_facts(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &RoadTripDistanceQuery,
) -> Option<HashMap<String, QuantityTotalFact>> {
    candidates
        .iter()
        .filter_map(|(session_id, session_rank)| {
            let facts = collect_road_trip_facts(
                idx.session_verbatim_answer_candidate_lines(session_id, usize::MAX),
            );
            let total_count = facts.values().map(|fact| fact.item_count).sum::<usize>();
            (total_count >= query.expected_trip_count).then_some((
                session_score(
                    *session_rank,
                    facts.values().map(|fact| fact.score).sum::<usize>(),
                ) + total_count * 10,
                facts,
            ))
        })
        .max_by_key(|(score, _)| *score)
        .map(|(_, facts)| facts)
}

fn best_global_road_trip_facts(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &RoadTripDistanceQuery,
) -> Option<HashMap<String, QuantityTotalFact>> {
    let mut best = HashMap::new();
    for (session_id, session_rank) in candidates {
        let facts = collect_road_trip_facts(
            idx.session_verbatim_answer_candidate_lines(session_id, usize::MAX),
        );
        for (_, fact) in facts {
            upsert_best_fact(
                &mut best,
                QuantityTotalFact {
                    score: session_score(*session_rank, fact.score),
                    ..fact
                },
            );
        }
    }
    (best.values().map(|fact| fact.item_count).sum::<usize>() >= query.expected_trip_count)
        .then_some(best)
}

fn best_same_session_weekend_hike_facts(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
) -> Option<HashMap<String, QuantityTotalFact>> {
    candidates
        .iter()
        .filter_map(|(session_id, session_rank)| {
            let facts = collect_weekend_hike_facts(
                idx.session_verbatim_answer_candidate_lines(session_id, usize::MAX),
            );
            has_consecutive_weekend_pair(&facts).then_some((
                session_score(
                    *session_rank,
                    facts.values().map(|fact| fact.score).sum::<usize>(),
                ),
                facts,
            ))
        })
        .max_by_key(|(score, _)| *score)
        .map(|(_, facts)| facts)
}

fn best_global_weekend_hike_facts(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
) -> Option<HashMap<String, QuantityTotalFact>> {
    let mut best = HashMap::new();
    for (session_id, session_rank) in candidates {
        let facts = collect_weekend_hike_facts(
            idx.session_verbatim_answer_candidate_lines(session_id, usize::MAX),
        );
        for (_, fact) in facts {
            upsert_best_fact(
                &mut best,
                QuantityTotalFact {
                    score: session_score(*session_rank, fact.score),
                    ..fact
                },
            );
        }
    }
    has_consecutive_weekend_pair(&best).then_some(best)
}

fn best_same_session_stay_day_facts(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &StayDaysQuery,
) -> Option<HashMap<String, QuantityTotalFact>> {
    candidates
        .iter()
        .filter_map(|(session_id, session_rank)| {
            let facts = collect_stay_day_facts(
                idx.session_verbatim_answer_candidate_lines(session_id, usize::MAX),
                query,
            );
            stay_days_cover_places(&facts, query).then_some((
                session_score(
                    *session_rank,
                    facts.values().map(|fact| fact.score).sum::<usize>(),
                ),
                facts,
            ))
        })
        .max_by_key(|(score, _)| *score)
        .map(|(_, facts)| facts)
}

fn best_global_stay_day_facts(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &StayDaysQuery,
) -> Option<HashMap<String, QuantityTotalFact>> {
    let mut best = HashMap::new();
    for (session_id, session_rank) in candidates {
        let facts = collect_stay_day_facts(
            idx.session_verbatim_answer_candidate_lines(session_id, usize::MAX),
            query,
        );
        for (_, fact) in facts {
            upsert_best_fact(
                &mut best,
                QuantityTotalFact {
                    score: session_score(*session_rank, fact.score),
                    ..fact
                },
            );
        }
    }
    stay_days_cover_places(&best, query).then_some(best)
}

fn collect_road_trip_facts(lines: Vec<String>) -> HashMap<String, QuantityTotalFact> {
    let mut facts = HashMap::new();
    for line in lines {
        let lower = line.to_ascii_lowercase();
        let Some(fact) = extract_road_trip_distance_fact_from_line(&line, &lower) else {
            continue;
        };
        upsert_best_fact(&mut facts, fact);
    }
    facts
}

fn collect_weekend_hike_facts(lines: Vec<String>) -> HashMap<String, QuantityTotalFact> {
    let mut facts = HashMap::new();
    for line in lines {
        let lower = line.to_ascii_lowercase();
        let Some((_, fact)) = extract_weekend_hike_distance_fact_from_line(&line, &lower) else {
            continue;
        };
        upsert_best_fact(&mut facts, fact);
    }
    facts
}

fn collect_stay_day_facts(
    lines: Vec<String>,
    query: &StayDaysQuery,
) -> HashMap<String, QuantityTotalFact> {
    let mut facts = HashMap::new();
    for line in lines {
        let lower = line.to_ascii_lowercase();
        let Some(fact) = extract_stay_days_fact_from_line(&line, &lower, query) else {
            continue;
        };
        upsert_best_fact(&mut facts, fact);
    }
    facts
}

fn upsert_best_fact(best: &mut HashMap<String, QuantityTotalFact>, fact: QuantityTotalFact) {
    let should_replace = best
        .get(&fact.key)
        .map(|existing| fact.score > existing.score)
        .unwrap_or(true);
    if should_replace {
        best.insert(fact.key.clone(), fact);
    }
}

fn has_consecutive_weekend_pair(facts: &HashMap<String, QuantityTotalFact>) -> bool {
    facts.contains_key("weekend-lastweekend") && facts.contains_key("weekend-twoweekendsago")
}

fn stay_days_cover_places(
    facts: &HashMap<String, QuantityTotalFact>,
    query: &StayDaysQuery,
) -> bool {
    query.places.iter().all(|place| facts.contains_key(place))
}

fn dedupe_quantity_evidence<I>(evidence: I) -> Vec<String>
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
