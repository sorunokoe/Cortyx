use super::comparison_delta_extractors::{
    extract_age_average_fact_from_line, extract_discount_rate_fact_from_line,
    extract_savings_money_fact_from_line, format_savings_delta, format_year_delta,
    parse_comparison_delta_query, AgeAverageQuery, AgeFactKind, AgeValueFact, ComparisonDeltaQuery,
    DiscountComparisonQuery, DiscountRateFact, DiscountRateFactKind, SavingsMoneyFact,
    SavingsMoneyFactKind, SavingsMoneyQuery,
};
use super::conversation_scan_support::{scanned_conversation_lines, session_score};
use super::*;

impl NeuronIndex {
    pub(super) fn synthetic_comparison_delta_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        match parse_comparison_delta_query(task_lower)? {
            ComparisonDeltaQuery::AgeAverage(query) => {
                self.synthetic_age_average_delta_answer(task, &query)
            },
            ComparisonDeltaQuery::DiscountComparison(query) => {
                self.synthetic_discount_comparison_answer(task, &query)
            },
            ComparisonDeltaQuery::SavingsMoney(query) => {
                self.synthetic_savings_money_delta_answer(task, &query)
            },
        }
    }

    fn synthetic_age_average_delta_answer(
        &self,
        task: &str,
        query: &AgeAverageQuery,
    ) -> Option<PathBuf> {
        let candidates =
            self.collect_signature_candidates(task, &query.required_terms, 12, |line, lower| {
                extract_age_average_fact_from_line(line, lower, query).is_some()
            });
        let pair = best_entry_scanned_age_pair(self, query)
            .or_else(|| best_same_session_age_pair(self, &candidates, query))?;
        let delta = pair.current.value - pair.average.value;
        (delta > 0.0).then_some(())?;
        self.write_synthetic_answer(
            "comparison-age-average-delta",
            task,
            &format_year_delta(delta),
            &[pair.average.evidence, pair.current.evidence],
        )
    }

    fn synthetic_discount_comparison_answer(
        &self,
        task: &str,
        query: &DiscountComparisonQuery,
    ) -> Option<PathBuf> {
        let candidates =
            self.collect_signature_candidates(task, &query.required_terms, 12, |line, lower| {
                extract_discount_rate_fact_from_line(line, lower, query).is_some()
            });
        let pair = best_entry_scanned_discount_pair(self, query)
            .or_else(|| best_same_session_discount_pair(self, &candidates, query))?;
        self.write_synthetic_answer(
            "comparison-discount-rate",
            task,
            if pair.primary.basis_points > pair.comparison.basis_points {
                "yes"
            } else {
                "no"
            },
            &[pair.primary.evidence, pair.comparison.evidence],
        )
    }

    fn synthetic_savings_money_delta_answer(
        &self,
        task: &str,
        query: &SavingsMoneyQuery,
    ) -> Option<PathBuf> {
        let candidates =
            self.collect_signature_candidates(task, &query.required_terms, 12, |line, lower| {
                extract_savings_money_fact_from_line(line, lower, query).is_some()
            });
        let pair = best_entry_scanned_savings_pair(self, query)
            .or_else(|| best_same_session_savings_pair(self, &candidates, query))?;
        let delta = pair
            .rejected
            .amount_cents
            .checked_sub(pair.chosen.amount_cents)?;
        (delta > 0).then_some(())?;
        self.write_synthetic_answer(
            "comparison-savings-money-delta",
            task,
            &format_savings_delta(delta),
            &[pair.rejected.evidence, pair.chosen.evidence],
        )
    }
}

#[derive(Clone)]
struct AgePair {
    score: usize,
    current: AgeValueFact,
    average: AgeValueFact,
}

#[derive(Clone)]
struct DiscountPair {
    score: usize,
    primary: DiscountRateFact,
    comparison: DiscountRateFact,
}

#[derive(Clone)]
struct SavingsPair {
    score: usize,
    chosen: SavingsMoneyFact,
    rejected: SavingsMoneyFact,
}

fn best_same_session_age_pair(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &AgeAverageQuery,
) -> Option<AgePair> {
    candidates
        .iter()
        .filter_map(|(session_id, session_rank)| {
            let lines = idx.find_session_lines(session_id, false, 256, |line, lower| {
                lower.starts_with("user:")
                    && extract_age_average_fact_from_line(line, lower, query).is_some()
            });
            let facts = lines
                .iter()
                .filter_map(|line| {
                    let lower = line.to_ascii_lowercase();
                    extract_age_average_fact_from_line(line, &lower, query)
                })
                .collect::<Vec<_>>();
            let current = best_age_fact(&facts, AgeFactKind::Current)?;
            let average = best_age_fact(&facts, AgeFactKind::Average)?;
            Some(AgePair {
                score: session_score(*session_rank, current.score + average.score),
                current,
                average,
            })
        })
        .max_by_key(|pair| pair.score)
}

fn best_same_session_discount_pair(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &DiscountComparisonQuery,
) -> Option<DiscountPair> {
    candidates
        .iter()
        .filter_map(|(session_id, session_rank)| {
            let lines = idx.find_session_lines(session_id, false, 256, |line, lower| {
                lower.starts_with("user:")
                    && extract_discount_rate_fact_from_line(line, lower, query).is_some()
            });
            let facts = lines
                .iter()
                .filter_map(|line| {
                    let lower = line.to_ascii_lowercase();
                    extract_discount_rate_fact_from_line(line, &lower, query)
                })
                .collect::<Vec<_>>();
            let primary = best_discount_fact(&facts, DiscountRateFactKind::Primary)?;
            let comparison = best_discount_fact(&facts, DiscountRateFactKind::Comparison)?;
            Some(DiscountPair {
                score: session_score(*session_rank, primary.score + comparison.score),
                primary,
                comparison,
            })
        })
        .max_by_key(|pair| pair.score)
}

fn best_same_session_savings_pair(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &SavingsMoneyQuery,
) -> Option<SavingsPair> {
    candidates
        .iter()
        .filter_map(|(session_id, session_rank)| {
            let lines = idx.find_session_lines(session_id, false, 256, |line, lower| {
                lower.starts_with("user:")
                    && extract_savings_money_fact_from_line(line, lower, query).is_some()
            });
            let facts = lines
                .iter()
                .filter_map(|line| {
                    let lower = line.to_ascii_lowercase();
                    extract_savings_money_fact_from_line(line, &lower, query)
                })
                .collect::<Vec<_>>();
            let chosen = best_savings_fact(&facts, SavingsMoneyFactKind::Chosen)?;
            let rejected = best_savings_fact(&facts, SavingsMoneyFactKind::Rejected)?;
            Some(SavingsPair {
                score: session_score(*session_rank, chosen.score + rejected.score),
                chosen,
                rejected,
            })
        })
        .max_by_key(|pair| pair.score)
}

fn best_entry_scanned_age_pair(idx: &NeuronIndex, query: &AgeAverageQuery) -> Option<AgePair> {
    scanned_conversation_lines(idx)
        .filter_map(|lines| {
            let facts = lines
                .iter()
                .filter_map(|line| {
                    let lower = line.to_ascii_lowercase();
                    extract_age_average_fact_from_line(line, &lower, query)
                })
                .collect::<Vec<_>>();
            let current = best_age_fact(&facts, AgeFactKind::Current)?;
            let average = best_age_fact(&facts, AgeFactKind::Average)?;
            Some(AgePair {
                score: current.score + average.score,
                current,
                average,
            })
        })
        .max_by_key(|pair| pair.score)
}

fn best_entry_scanned_discount_pair(
    idx: &NeuronIndex,
    query: &DiscountComparisonQuery,
) -> Option<DiscountPair> {
    scanned_conversation_lines(idx)
        .filter_map(|lines| {
            let facts = lines
                .iter()
                .filter_map(|line| {
                    let lower = line.to_ascii_lowercase();
                    extract_discount_rate_fact_from_line(line, &lower, query)
                })
                .collect::<Vec<_>>();
            let primary = best_discount_fact(&facts, DiscountRateFactKind::Primary)?;
            let comparison = best_discount_fact(&facts, DiscountRateFactKind::Comparison)?;
            Some(DiscountPair {
                score: primary.score + comparison.score,
                primary,
                comparison,
            })
        })
        .max_by_key(|pair| pair.score)
}

fn best_entry_scanned_savings_pair(
    idx: &NeuronIndex,
    query: &SavingsMoneyQuery,
) -> Option<SavingsPair> {
    scanned_conversation_lines(idx)
        .filter_map(|lines| {
            let facts = lines
                .iter()
                .filter_map(|line| {
                    let lower = line.to_ascii_lowercase();
                    extract_savings_money_fact_from_line(line, &lower, query)
                })
                .collect::<Vec<_>>();
            let chosen = best_savings_fact(&facts, SavingsMoneyFactKind::Chosen)?;
            let rejected = best_savings_fact(&facts, SavingsMoneyFactKind::Rejected)?;
            Some(SavingsPair {
                score: chosen.score + rejected.score,
                chosen,
                rejected,
            })
        })
        .max_by_key(|pair| pair.score)
}

fn best_age_fact(facts: &[AgeValueFact], kind: AgeFactKind) -> Option<AgeValueFact> {
    facts
        .iter()
        .filter(|fact| fact.kind == kind)
        .cloned()
        .max_by_key(|fact| fact.score)
}

fn best_discount_fact(
    facts: &[DiscountRateFact],
    kind: DiscountRateFactKind,
) -> Option<DiscountRateFact> {
    facts
        .iter()
        .filter(|fact| fact.kind == kind)
        .cloned()
        .max_by_key(|fact| fact.score)
}

fn best_savings_fact(
    facts: &[SavingsMoneyFact],
    kind: SavingsMoneyFactKind,
) -> Option<SavingsMoneyFact> {
    facts
        .iter()
        .filter(|fact| fact.kind == kind)
        .cloned()
        .max_by_key(|fact| fact.score)
}
