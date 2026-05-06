use super::conversation_scan_support::scanned_conversation_lines;
use super::money_extractors::{
    extract_cashback_purchase_fact_from_line, extract_cashback_rate_fact_from_line,
    extract_original_price_fact_from_line, extract_paid_price_fact_from_line,
};
use super::money_queries::parse_money_query;
use super::money_support::{
    dedupe_evidence, format_money_cents, CashbackQuery, CashbackRateFact, MoneyAmountFact,
    MoneyQuery, SavingsQuery,
};
use super::*;

impl NeuronIndex {
    pub(super) fn synthetic_money_computation_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        match parse_money_query(task, task_lower)? {
            MoneyQuery::Cashback(query) => self.synthetic_cashback_earned_answer(task, &query),
            MoneyQuery::DiscountPercent(query) => {
                self.synthetic_discount_percentage_answer(task, &query)
            },
            MoneyQuery::Savings(query) => self.synthetic_savings_delta_answer(task, &query),
            MoneyQuery::RaisedTotal(query) => self.synthetic_raised_total_answer(task, &query),
            MoneyQuery::Revenue(query) => self.synthetic_revenue_answer(task, &query),
            MoneyQuery::RecipientGiftTotal(_)
            | MoneyQuery::SpendSum(_)
            | MoneyQuery::SaleMinimum(_) => None,
        }
    }

    fn synthetic_cashback_earned_answer(
        &self,
        task: &str,
        query: &CashbackQuery,
    ) -> Option<PathBuf> {
        let candidates =
            self.collect_signature_candidates(task, &query.required_terms, 12, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && (extract_cashback_purchase_fact_from_line(line, lower, query).is_some()
                        || extract_cashback_rate_fact_from_line(line, lower, query).is_some())
            });
        let best_pair = best_same_session_cashback_pair(self, &candidates, query)
            .or_else(|| best_global_cashback_pair(self, &candidates, query))?;
        let cashback_cents =
            ((best_pair.purchase.amount_cents * best_pair.rate.basis_points) + 5_000) / 10_000;
        self.write_synthetic_answer(
            "cashback-earned",
            task,
            &format_money_cents(cashback_cents),
            &dedupe_evidence([
                best_pair.purchase.evidence.clone(),
                best_pair.rate.evidence.clone(),
            ]),
        )
    }

    fn synthetic_savings_delta_answer(&self, task: &str, query: &SavingsQuery) -> Option<PathBuf> {
        let candidates =
            self.collect_signature_candidates(task, &query.required_terms, 12, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && (extract_paid_price_fact_from_line(line, lower, query).is_some()
                        || extract_original_price_fact_from_line(line, lower, query).is_some())
            });
        let best_pair = best_entry_scanned_savings_pair(self, query)
            .or_else(|| best_same_session_savings_pair(self, &candidates, query))
            .or_else(|| best_global_savings_pair(self, &candidates, query))?;
        let savings_cents = best_pair
            .original
            .amount_cents
            .checked_sub(best_pair.paid.amount_cents)?;
        (savings_cents > 0).then_some(())?;
        self.write_synthetic_answer(
            "money-savings-delta",
            task,
            &format_money_cents(savings_cents),
            &dedupe_evidence([
                best_pair.paid.evidence.clone(),
                best_pair.original.evidence.clone(),
            ]),
        )
    }

    fn synthetic_discount_percentage_answer(
        &self,
        task: &str,
        query: &SavingsQuery,
    ) -> Option<PathBuf> {
        let candidates =
            self.collect_signature_candidates(task, &query.required_terms, 12, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && (extract_paid_price_fact_from_line(line, lower, query).is_some()
                        || extract_original_price_fact_from_line(line, lower, query).is_some())
            });
        let best_pair = best_entry_scanned_savings_pair(self, query)
            .or_else(|| best_same_session_savings_pair(self, &candidates, query))
            .or_else(|| best_global_savings_pair(self, &candidates, query))?;
        let savings_cents = best_pair
            .original
            .amount_cents
            .checked_sub(best_pair.paid.amount_cents)?;
        (savings_cents > 0).then_some(())?;
        let basis_points = ((savings_cents * 10_000) + (best_pair.original.amount_cents / 2))
            / best_pair.original.amount_cents;
        self.write_synthetic_answer(
            "money-discount-percent",
            task,
            &format_basis_points_percent(basis_points),
            &dedupe_evidence([
                best_pair.paid.evidence.clone(),
                best_pair.original.evidence.clone(),
            ]),
        )
    }
}

#[derive(Clone)]
struct CashbackPair {
    score: usize,
    purchase: MoneyAmountFact,
    rate: CashbackRateFact,
}

#[derive(Clone)]
struct SavingsPair {
    score: usize,
    paid: MoneyAmountFact,
    original: MoneyAmountFact,
}

fn best_same_session_cashback_pair(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &CashbackQuery,
) -> Option<CashbackPair> {
    candidates
        .iter()
        .filter_map(|(session_id, session_rank)| {
            let lines = idx.find_session_lines(session_id, false, 256, |line, lower| {
                is_summary_or_user_line(line, lower)
            });
            let purchase = best_amount_fact(
                lines
                    .iter()
                    .filter_map(|line| {
                        let lower = line.to_ascii_lowercase();
                        extract_cashback_purchase_fact_from_line(line, &lower, query)
                    })
                    .collect(),
                *session_rank,
            )?;
            let rate = best_rate_fact(
                lines
                    .iter()
                    .filter_map(|line| {
                        let lower = line.to_ascii_lowercase();
                        extract_cashback_rate_fact_from_line(line, &lower, query)
                    })
                    .collect(),
                *session_rank,
            )?;
            Some(CashbackPair {
                score: purchase.score + rate.score,
                purchase,
                rate,
            })
        })
        .max_by_key(|pair| pair.score)
}

fn best_global_cashback_pair(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &CashbackQuery,
) -> Option<CashbackPair> {
    let mut best_purchase = None;
    let mut best_rate = None;
    for (session_id, session_rank) in candidates {
        let lines = idx.find_session_lines(session_id, false, 256, |line, lower| {
            is_summary_or_user_line(line, lower)
        });
        for line in lines {
            let lower = line.to_ascii_lowercase();
            if let Some(fact) = extract_cashback_purchase_fact_from_line(&line, &lower, query) {
                update_best_amount_fact(&mut best_purchase, fact, *session_rank);
            }
            if let Some(fact) = extract_cashback_rate_fact_from_line(&line, &lower, query) {
                update_best_rate_fact(&mut best_rate, fact, *session_rank);
            }
        }
    }
    let (purchase_score, purchase) = best_purchase?;
    let (rate_score, rate) = best_rate?;
    Some(CashbackPair {
        score: purchase_score + rate_score,
        purchase,
        rate,
    })
}

fn best_entry_scanned_savings_pair(idx: &NeuronIndex, query: &SavingsQuery) -> Option<SavingsPair> {
    scanned_conversation_lines(idx)
        .filter_map(|lines| {
            let paid = best_amount_fact(
                lines
                    .iter()
                    .filter_map(|line| {
                        let lower = line.to_ascii_lowercase();
                        extract_paid_price_fact_from_line(line, &lower, query)
                    })
                    .collect(),
                0,
            )?;
            let original = best_amount_fact(
                lines
                    .iter()
                    .filter_map(|line| {
                        let lower = line.to_ascii_lowercase();
                        extract_original_price_fact_from_line(line, &lower, query)
                    })
                    .collect(),
                0,
            )?;
            (original.amount_cents > paid.amount_cents).then_some(SavingsPair {
                score: paid.score + original.score,
                paid,
                original,
            })
        })
        .max_by_key(|pair| pair.score)
}

fn best_same_session_savings_pair(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &SavingsQuery,
) -> Option<SavingsPair> {
    candidates
        .iter()
        .filter_map(|(session_id, session_rank)| {
            let lines = idx.find_session_lines(session_id, false, 256, |line, lower| {
                is_summary_or_user_line(line, lower)
            });
            let paid = best_amount_fact(
                lines
                    .iter()
                    .filter_map(|line| {
                        let lower = line.to_ascii_lowercase();
                        extract_paid_price_fact_from_line(line, &lower, query)
                    })
                    .collect(),
                *session_rank,
            )?;
            let original = best_amount_fact(
                lines
                    .iter()
                    .filter_map(|line| {
                        let lower = line.to_ascii_lowercase();
                        extract_original_price_fact_from_line(line, &lower, query)
                    })
                    .collect(),
                *session_rank,
            )?;
            (original.amount_cents > paid.amount_cents).then_some(SavingsPair {
                score: paid.score + original.score,
                paid,
                original,
            })
        })
        .max_by_key(|pair| pair.score)
}

fn best_global_savings_pair(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &SavingsQuery,
) -> Option<SavingsPair> {
    let mut best_paid = None;
    let mut best_original = None;
    for (session_id, session_rank) in candidates {
        let lines = idx.find_session_lines(session_id, false, 256, |line, lower| {
            is_summary_or_user_line(line, lower)
        });
        for line in lines {
            let lower = line.to_ascii_lowercase();
            if let Some(fact) = extract_paid_price_fact_from_line(&line, &lower, query) {
                update_best_amount_fact(&mut best_paid, fact, *session_rank);
            }
            if let Some(fact) = extract_original_price_fact_from_line(&line, &lower, query) {
                update_best_amount_fact(&mut best_original, fact, *session_rank);
            }
        }
    }
    let (paid_score, paid) = best_paid?;
    let (original_score, original) = best_original?;
    (original.amount_cents > paid.amount_cents).then_some(SavingsPair {
        score: paid_score + original_score,
        paid,
        original,
    })
}

fn best_amount_fact(facts: Vec<MoneyAmountFact>, session_rank: usize) -> Option<MoneyAmountFact> {
    facts
        .into_iter()
        .max_by_key(|fact| session_rank * 100 + fact.score)
        .map(|fact| MoneyAmountFact {
            score: session_rank * 100 + fact.score,
            ..fact
        })
}

fn best_rate_fact(facts: Vec<CashbackRateFact>, session_rank: usize) -> Option<CashbackRateFact> {
    facts
        .into_iter()
        .max_by_key(|fact| session_rank * 100 + fact.score)
        .map(|fact| CashbackRateFact {
            score: session_rank * 100 + fact.score,
            ..fact
        })
}

fn update_best_amount_fact(
    slot: &mut Option<(usize, MoneyAmountFact)>,
    fact: MoneyAmountFact,
    session_rank: usize,
) {
    let score = session_rank * 100 + fact.score;
    let should_replace = slot
        .as_ref()
        .map(|(best_score, _)| score > *best_score)
        .unwrap_or(true);
    if should_replace {
        *slot = Some((score, fact));
    }
}

fn update_best_rate_fact(
    slot: &mut Option<(usize, CashbackRateFact)>,
    fact: CashbackRateFact,
    session_rank: usize,
) {
    let score = session_rank * 100 + fact.score;
    let should_replace = slot
        .as_ref()
        .map(|(best_score, _)| score > *best_score)
        .unwrap_or(true);
    if should_replace {
        *slot = Some((score, fact));
    }
}

fn format_basis_points_percent(basis_points: i64) -> String {
    if basis_points % 100 == 0 {
        return format!("{}%", basis_points / 100);
    }
    let whole = basis_points / 100;
    let remainder = (basis_points % 100).abs();
    format!("{whole}.{remainder:02}%")
}
