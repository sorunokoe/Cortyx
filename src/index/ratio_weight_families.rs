use super::conversation_scan_support::session_score;
use super::ratio_weight_extractors::{
    extract_percentage_part_fact_from_line, extract_percentage_whole_fact_from_line,
    extract_weight_purchase_fact_from_line, format_percentage_answer, parse_ratio_weight_query,
    PercentageQuery, RatioWeightFact, RatioWeightQuery, WeightTotalQuery,
};
use super::*;

impl NeuronIndex {
    pub(super) fn synthetic_ratio_weight_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        match parse_ratio_weight_query(task, task_lower)? {
            RatioWeightQuery::WeightTotal(query) => {
                self.synthetic_weight_total_answer(task, &query)
            },
            RatioWeightQuery::Percentage(query) => self.synthetic_percentage_answer(task, &query),
        }
    }

    fn synthetic_weight_total_answer(
        &self,
        task: &str,
        query: &WeightTotalQuery,
    ) -> Option<PathBuf> {
        let candidates =
            self.ranked_numeric_aggregate_sessions(task, &query.required_terms, |line, lower| {
                extract_weight_purchase_fact_from_line(line, lower, query).is_some()
            });
        let aggregate = best_same_session_weight_aggregate(self, &candidates, query)
            .or_else(|| best_global_weight_aggregate(self, &candidates, query))?;
        self.write_synthetic_answer(
            "ratio-weight-total",
            task,
            &format!("{} {}", aggregate.total, aggregate.unit),
            &aggregate.evidence,
        )
    }

    fn synthetic_percentage_answer(&self, task: &str, query: &PercentageQuery) -> Option<PathBuf> {
        let candidates =
            self.ranked_numeric_aggregate_sessions(task, &query.required_terms, |line, lower| {
                extract_percentage_part_fact_from_line(line, lower, query).is_some()
                    || extract_percentage_whole_fact_from_line(line, lower, query).is_some()
            });
        let facts = best_same_session_percentage_facts(self, &candidates, query)
            .or_else(|| best_global_percentage_facts(self, &candidates, query))
            .or_else(|| best_global_percentage_facts_all_entries(self, query))?;
        let answer = format_percentage_answer(facts.part.value, facts.whole.value)?;
        self.write_synthetic_answer(
            "ratio-percentage-total",
            task,
            &answer,
            &dedupe_ratio_weight_evidence([
                facts.part.evidence.clone(),
                facts.whole.evidence.clone(),
            ]),
        )
    }
}

#[derive(Clone, Debug)]
struct WeightAggregate {
    total: i64,
    unit: String,
    evidence: Vec<String>,
    score: usize,
}

#[derive(Clone, Debug)]
struct PercentageFacts {
    part: RatioWeightFact,
    whole: RatioWeightFact,
}

fn best_same_session_weight_aggregate(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &WeightTotalQuery,
) -> Option<WeightAggregate> {
    candidates
        .iter()
        .filter_map(|(session_id, session_rank)| {
            let aggregate = summarize_weight_facts(collect_weight_facts(
                idx.session_answer_candidate_lines(session_id, usize::MAX),
                query,
            ))?;
            Some((
                session_score(*session_rank, aggregate.score)
                    + usize::try_from(aggregate.total).unwrap_or(0),
                aggregate,
            ))
        })
        .max_by_key(|(score, _)| *score)
        .map(|(_, aggregate)| aggregate)
}

fn best_global_weight_aggregate(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &WeightTotalQuery,
) -> Option<WeightAggregate> {
    let mut facts = Vec::new();
    for (session_id, _) in candidates {
        facts.extend(collect_weight_facts(
            idx.session_answer_candidate_lines(session_id, usize::MAX),
            query,
        ));
    }
    summarize_weight_facts(facts)
}

fn best_same_session_percentage_facts(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &PercentageQuery,
) -> Option<PercentageFacts> {
    candidates
        .iter()
        .filter_map(|(session_id, session_rank)| {
            let facts = collect_percentage_facts(
                idx.session_answer_candidate_lines(session_id, usize::MAX),
                query,
            )?;
            Some((
                session_score(*session_rank, facts.part.score + facts.whole.score),
                facts,
            ))
        })
        .max_by_key(|(score, _)| *score)
        .map(|(_, facts)| facts)
}

fn best_global_percentage_facts(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &PercentageQuery,
) -> Option<PercentageFacts> {
    let mut best_part: Option<RatioWeightFact> = None;
    let mut best_whole: Option<RatioWeightFact> = None;
    for (session_id, session_rank) in candidates {
        let Some(facts) = collect_percentage_facts(
            idx.session_answer_candidate_lines(session_id, usize::MAX),
            query,
        ) else {
            continue;
        };
        upsert_best_ratio_fact(
            &mut best_part,
            RatioWeightFact {
                score: session_score(*session_rank, facts.part.score),
                ..facts.part
            },
        );
        upsert_best_ratio_fact(
            &mut best_whole,
            RatioWeightFact {
                score: session_score(*session_rank, facts.whole.score),
                ..facts.whole
            },
        );
    }
    Some(PercentageFacts {
        part: best_part?,
        whole: best_whole?,
    })
}

fn best_global_percentage_facts_all_entries(
    idx: &NeuronIndex,
    query: &PercentageQuery,
) -> Option<PercentageFacts> {
    let mut best_part = None;
    let mut best_whole = None;
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
        let Some(facts) = collect_percentage_facts(
            strip_query_surface_section(&content)
                .lines()
                .filter_map(|raw_line| {
                    let line = raw_line.trim();
                    (!line.is_empty() && is_session_answer_candidate_line(line))
                        .then_some((line.to_string(), is_summary))
                })
                .collect(),
            query,
        ) else {
            continue;
        };
        upsert_best_ratio_fact(&mut best_part, facts.part);
        upsert_best_ratio_fact(&mut best_whole, facts.whole);
    }
    Some(PercentageFacts {
        part: best_part?,
        whole: best_whole?,
    })
}

fn collect_weight_facts(
    lines: Vec<(String, bool)>,
    query: &WeightTotalQuery,
) -> Vec<RatioWeightFact> {
    let mut facts = Vec::new();
    for (line, is_summary) in lines {
        let lower = line.to_ascii_lowercase();
        let Some(mut fact) = extract_weight_purchase_fact_from_line(&line, &lower, query) else {
            continue;
        };
        if is_summary {
            fact.score += 3;
        }
        facts.push(fact);
    }
    facts
}

fn collect_percentage_facts(
    lines: Vec<(String, bool)>,
    query: &PercentageQuery,
) -> Option<PercentageFacts> {
    let mut best_part = None;
    let mut best_whole = None;
    for (line, is_summary) in lines {
        let lower = line.to_ascii_lowercase();
        if let Some(mut fact) = extract_percentage_part_fact_from_line(&line, &lower, query) {
            if is_summary {
                fact.score += 3;
            }
            upsert_best_ratio_fact(&mut best_part, fact);
        }
        if let Some(mut fact) = extract_percentage_whole_fact_from_line(&line, &lower, query) {
            if is_summary {
                fact.score += 3;
            }
            upsert_best_ratio_fact(&mut best_whole, fact);
        }
    }
    Some(PercentageFacts {
        part: best_part?,
        whole: best_whole?,
    })
}

fn summarize_weight_facts(facts: Vec<RatioWeightFact>) -> Option<WeightAggregate> {
    let mut grouped: HashMap<String, WeightAggregate> = HashMap::new();
    let mut seen_keys = HashSet::new();
    for fact in facts {
        if !seen_keys.insert(fact.key.clone()) {
            continue;
        }
        let unit = fact.unit.clone()?;
        let entry = grouped
            .entry(unit.clone())
            .or_insert_with(|| WeightAggregate {
                total: 0,
                unit,
                evidence: Vec::new(),
                score: 0,
            });
        entry.total += fact.value;
        entry.score += fact.score;
        if !entry
            .evidence
            .iter()
            .any(|existing| existing == &fact.evidence)
        {
            entry.evidence.push(fact.evidence);
        }
    }
    grouped
        .into_values()
        .max_by_key(|aggregate| (aggregate.score, aggregate.total))
}

fn upsert_best_ratio_fact(best: &mut Option<RatioWeightFact>, fact: RatioWeightFact) {
    let replace = best
        .as_ref()
        .map(|existing| {
            fact.score > existing.score
                || (fact.score == existing.score && fact.value > existing.value)
        })
        .unwrap_or(true);
    if replace {
        *best = Some(fact);
    }
}

fn dedupe_ratio_weight_evidence<I>(evidence: I) -> Vec<String>
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
