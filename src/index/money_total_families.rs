use super::money_extractors::{
    extract_raised_total_fact_from_line, extract_revenue_quantity_fact_from_line,
    extract_revenue_unit_price_fact_from_line,
};
use super::money_support::{
    dedupe_evidence, format_money_cents, KeyedMoneyAmountFact, QuantityFact, RaisedTotalQuery,
    RevenueQuery, UnitPriceFact,
};
use super::*;

impl NeuronIndex {
    pub(super) fn synthetic_raised_total_answer(
        &self,
        task: &str,
        query: &RaisedTotalQuery,
    ) -> Option<PathBuf> {
        let candidates =
            self.collect_signature_candidates(task, &query.required_terms, 12, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && extract_raised_total_fact_from_line(line, lower, query).is_some()
            });
        let facts = choose_best_raised_fact_set(
            best_same_session_raised_facts(self, &candidates, query),
            best_entry_scanned_raised_facts(self, query),
            query,
        )?;
        let total_cents = facts.values().map(|fact| fact.amount_cents).sum::<i64>();
        (total_cents > 0).then_some(())?;
        self.write_synthetic_answer(
            "money-raised-total",
            task,
            &format_money_cents(total_cents),
            &dedupe_evidence(
                facts
                    .values()
                    .map(|fact| fact.evidence.clone())
                    .collect::<Vec<_>>(),
            ),
        )
    }

    pub(super) fn synthetic_revenue_answer(
        &self,
        task: &str,
        query: &RevenueQuery,
    ) -> Option<PathBuf> {
        let candidates =
            self.collect_signature_candidates(task, &query.required_terms, 12, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && (extract_revenue_quantity_fact_from_line(line, lower, query).is_some()
                        || extract_revenue_unit_price_fact_from_line(line, lower, query).is_some())
            });
        let pair = best_same_session_revenue_pair(self, &candidates, query)?;
        let total_cents = pair
            .quantity
            .quantity_units
            .checked_mul(pair.price.unit_price_cents)?;
        (total_cents > 0).then_some(())?;
        self.write_synthetic_answer(
            "money-sales-revenue",
            task,
            &format_money_cents(total_cents),
            &dedupe_evidence([pair.quantity.evidence.clone(), pair.price.evidence.clone()]),
        )
    }
}

#[derive(Clone)]
struct RevenuePair {
    score: usize,
    quantity: QuantityFact,
    price: UnitPriceFact,
}

fn best_same_session_raised_facts(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &RaisedTotalQuery,
) -> Option<HashMap<String, KeyedMoneyAmountFact>> {
    candidates
        .iter()
        .filter_map(|(session_id, session_rank)| {
            let lines = idx.find_session_lines(session_id, false, 256, |line, lower| {
                is_summary_or_user_line(line, lower)
            });
            let facts = collect_raised_facts(lines, query);
            (!facts.is_empty()).then_some((
                raised_session_total_score(*session_rank, &facts, query),
                facts,
            ))
        })
        .max_by_key(|(score, facts)| (*score, facts.len()))
        .map(|(_, facts)| facts)
}

fn best_same_session_revenue_pair(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &RevenueQuery,
) -> Option<RevenuePair> {
    candidates
        .iter()
        .filter_map(|(session_id, session_rank)| {
            let lines = idx.find_session_lines(session_id, false, 256, |line, lower| {
                is_summary_or_user_line(line, lower)
            });
            let (quantities, prices) = collect_revenue_facts(lines, *session_rank, query);
            quantities
                .into_iter()
                .filter_map(|(unit_key, quantity)| {
                    let price = prices.get(&unit_key)?.clone();
                    Some(RevenuePair {
                        score: session_total_score(*session_rank, quantity.score + price.score),
                        quantity,
                        price,
                    })
                })
                .max_by_key(|pair| pair.score)
        })
        .max_by_key(|pair| pair.score)
}

fn collect_raised_facts(
    lines: Vec<String>,
    query: &RaisedTotalQuery,
) -> HashMap<String, KeyedMoneyAmountFact> {
    let mut best = HashMap::new();
    for line in lines {
        let lower = line.to_ascii_lowercase();
        let Some(fact) = extract_raised_total_fact_from_line(&line, &lower, query) else {
            continue;
        };
        let should_replace = best
            .get(&fact.key)
            .map(|existing: &KeyedMoneyAmountFact| {
                fact.score > existing.score
                    || (fact.score == existing.score && fact.amount_cents > existing.amount_cents)
            })
            .unwrap_or(true);
        if should_replace {
            best.insert(fact.key.clone(), fact);
        }
    }
    best
}

fn collect_revenue_facts(
    lines: Vec<String>,
    session_rank: usize,
    query: &RevenueQuery,
) -> (
    HashMap<String, QuantityFact>,
    HashMap<String, UnitPriceFact>,
) {
    let mut quantities = HashMap::new();
    let mut prices = HashMap::new();
    for line in lines {
        let lower = line.to_ascii_lowercase();
        if let Some(fact) = extract_revenue_quantity_fact_from_line(&line, &lower, query) {
            let scored_fact = QuantityFact {
                score: session_total_score(session_rank, fact.score),
                ..fact
            };
            let should_replace = quantities
                .get(&scored_fact.unit_key)
                .map(|existing: &QuantityFact| scored_fact.score > existing.score)
                .unwrap_or(true);
            if should_replace {
                quantities.insert(scored_fact.unit_key.clone(), scored_fact);
            }
        }
        if let Some(fact) = extract_revenue_unit_price_fact_from_line(&line, &lower, query) {
            let scored_fact = UnitPriceFact {
                score: session_total_score(session_rank, fact.score),
                ..fact
            };
            let should_replace = prices
                .get(&scored_fact.unit_key)
                .map(|existing: &UnitPriceFact| scored_fact.score > existing.score)
                .unwrap_or(true);
            if should_replace {
                prices.insert(scored_fact.unit_key.clone(), scored_fact);
            }
        }
    }
    (quantities, prices)
}

fn session_total_score(session_rank: usize, line_score: usize) -> usize {
    session_rank * 100 + line_score
}

fn raised_session_total_score(
    session_rank: usize,
    facts: &HashMap<String, KeyedMoneyAmountFact>,
    query: &RaisedTotalQuery,
) -> usize {
    raised_fact_set_score(facts, query) * 100 + session_rank
}

fn raised_fact_set_score(
    facts: &HashMap<String, KeyedMoneyAmountFact>,
    query: &RaisedTotalQuery,
) -> usize {
    let line_score = facts.values().map(|fact| fact.score).sum::<usize>();
    let overlap_score = facts
        .values()
        .map(|fact| raised_query_overlap_score(&fact.evidence, query))
        .sum::<usize>();
    line_score * 100 + overlap_score * 10 + facts.len() * 10
}

fn raised_query_overlap_score(evidence: &str, query: &RaisedTotalQuery) -> usize {
    let lower = evidence.to_ascii_lowercase();
    query
        .required_terms
        .iter()
        .filter(|term| lower.contains(term.as_str()))
        .count()
}

fn best_entry_scanned_raised_facts(
    idx: &NeuronIndex,
    query: &RaisedTotalQuery,
) -> Option<HashMap<String, KeyedMoneyAmountFact>> {
    let Ok(entries) = std::fs::read_dir(neuron_dir(&idx.project_root)) else {
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
            let facts = collect_raised_facts(
                content.lines().map(str::to_string).collect::<Vec<_>>(),
                query,
            );
            (!facts.is_empty()).then_some((raised_fact_set_score(&facts, query), facts))
        })
        .max_by_key(|(score, facts)| (*score, facts.len()))
        .map(|(_, facts)| facts)
}

fn choose_best_raised_fact_set(
    primary: Option<HashMap<String, KeyedMoneyAmountFact>>,
    fallback: Option<HashMap<String, KeyedMoneyAmountFact>>,
    query: &RaisedTotalQuery,
) -> Option<HashMap<String, KeyedMoneyAmountFact>> {
    match (primary, fallback) {
        (Some(primary), Some(fallback)) => {
            let primary_score = raised_fact_set_score(&primary, query);
            let fallback_score = raised_fact_set_score(&fallback, query);
            if fallback_score > primary_score
                || (fallback_score == primary_score && fallback.len() > primary.len())
            {
                Some(fallback)
            } else {
                Some(primary)
            }
        },
        (Some(primary), None) => Some(primary),
        (None, Some(fallback)) => Some(fallback),
        (None, None) => None,
    }
}
