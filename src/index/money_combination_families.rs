use super::money_combination_extractors::{
    extract_contextual_spend_followup_fact_from_line, extract_keyed_gift_recipient_fact_from_line,
    extract_sale_value_fact_from_line, extract_spend_focus_fact_from_line,
};
use super::money_queries::parse_money_query;
use super::money_support::{
    dedupe_evidence, format_money_cents, line_matches_spend_focus, MoneyAmountFact, MoneyQuery,
    RecipientGiftTotalQuery, SaleMinimumQuery, SpendFocus, SpendFocusKind, SpendSumQuery,
};
use super::*;

impl NeuronIndex {
    pub(super) fn synthetic_money_combination_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        match parse_money_query(task, task_lower)? {
            MoneyQuery::RecipientGiftTotal(query) => {
                self.synthetic_single_recipient_gift_total_answer(task, &query)
            },
            MoneyQuery::SpendSum(query) => self.synthetic_spend_sum_answer(task, &query),
            MoneyQuery::SaleMinimum(query) => self.synthetic_sale_minimum_answer(task, &query),
            _ => None,
        }
    }

    fn synthetic_single_recipient_gift_total_answer(
        &self,
        task: &str,
        query: &RecipientGiftTotalQuery,
    ) -> Option<PathBuf> {
        let candidates =
            self.collect_signature_candidates(task, &query.required_terms, 12, |line, lower| {
                lower.starts_with("user:")
                    && extract_spend_focus_fact_from_line(line, lower, &query.focus).is_some()
            });
        let facts = best_same_session_single_recipient_gift_facts(self, &candidates, query)
            .or_else(|| best_global_single_recipient_gift_facts(self, &candidates, query))?;
        let total_cents = facts.values().map(|fact| fact.amount_cents).sum::<i64>();
        self.write_synthetic_answer(
            "money-single-recipient-gift-total",
            task,
            &format_money_cents(total_cents),
            &dedupe_evidence(facts.into_values().map(|fact| fact.evidence)),
        )
    }

    fn synthetic_spend_sum_answer(&self, task: &str, query: &SpendSumQuery) -> Option<PathBuf> {
        let candidates =
            self.collect_signature_candidates(task, &query.required_terms, 12, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && query.focuses.iter().any(|focus| {
                        extract_spend_focus_fact_from_line(line, lower, focus).is_some()
                    })
            });
        let facts = best_same_session_spend_facts(self, &candidates, query)
            .or_else(|| best_global_spend_facts(self, &candidates, query))?;
        let total_cents = facts.values().map(|fact| fact.amount_cents).sum::<i64>();
        self.write_synthetic_answer(
            "money-spend-sum",
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

    fn synthetic_sale_minimum_answer(
        &self,
        task: &str,
        query: &SaleMinimumQuery,
    ) -> Option<PathBuf> {
        let candidates =
            self.collect_signature_candidates(task, &query.required_terms, 12, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && query.focuses.iter().any(|focus| {
                        extract_sale_value_fact_from_line(line, lower, focus).is_some()
                    })
            });
        let facts = best_same_session_sale_facts(self, &candidates, query)
            .or_else(|| best_global_sale_facts(self, &candidates, query))?;
        let total_cents = facts.values().map(|fact| fact.amount_cents).sum::<i64>();
        self.write_synthetic_answer(
            "money-sale-minimum",
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
}

fn best_same_session_single_recipient_gift_facts(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &RecipientGiftTotalQuery,
) -> Option<HashMap<String, MoneyAmountFact>> {
    candidates
        .iter()
        .filter_map(|(session_id, session_rank)| {
            let lines = idx.find_session_lines(session_id, false, 256, |line, lower| {
                lower.starts_with("user:")
                    && extract_spend_focus_fact_from_line(line, lower, &query.focus).is_some()
            });
            let facts = collect_single_recipient_gift_facts(lines, &query.focus, *session_rank);
            (!facts.is_empty()).then_some((
                facts
                    .values()
                    .map(|fact| usize::try_from(fact.amount_cents.max(0)).unwrap_or(0))
                    .sum::<usize>()
                    * 100
                    + facts.len() * 20
                    + facts.values().map(|fact| fact.score).sum::<usize>(),
                facts,
            ))
        })
        .max_by_key(|(score, facts)| (*score, facts.len()))
        .map(|(_, facts)| facts)
}

fn best_global_single_recipient_gift_facts(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &RecipientGiftTotalQuery,
) -> Option<HashMap<String, MoneyAmountFact>> {
    let mut best = HashMap::new();
    for (session_id, session_rank) in candidates {
        let lines = idx.find_session_lines(session_id, false, 256, |line, lower| {
            lower.starts_with("user:")
                && extract_spend_focus_fact_from_line(line, lower, &query.focus).is_some()
        });
        let facts = collect_single_recipient_gift_facts(lines, &query.focus, *session_rank);
        for (key, fact) in facts {
            let should_replace = best
                .get(&key)
                .map(|existing: &MoneyAmountFact| fact.score > existing.score)
                .unwrap_or(true);
            if should_replace {
                best.insert(key, fact);
            }
        }
    }
    (!best.is_empty()).then_some(best)
}

fn best_same_session_spend_facts(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &SpendSumQuery,
) -> Option<HashMap<String, MoneyAmountFact>> {
    candidates
        .iter()
        .filter_map(|(session_id, session_rank)| {
            let lines = idx.find_session_lines(session_id, false, 256, |line, lower| {
                is_summary_or_user_line(line, lower)
            });
            let facts = collect_spend_facts(lines, &query.focuses, *session_rank);
            (facts.len() == query.focuses.len()).then_some(facts)
        })
        .max_by_key(|facts| facts.values().map(|fact| fact.score).sum::<usize>())
}

fn best_global_spend_facts(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &SpendSumQuery,
) -> Option<HashMap<String, MoneyAmountFact>> {
    let mut best = HashMap::new();
    for (session_id, session_rank) in candidates {
        let lines = idx.find_session_lines(session_id, false, 256, |line, lower| {
            is_summary_or_user_line(line, lower)
        });
        let facts = collect_spend_facts(lines, &query.focuses, *session_rank);
        for (key, fact) in facts {
            let should_replace = best
                .get(&key)
                .map(|existing: &MoneyAmountFact| fact.score > existing.score)
                .unwrap_or(true);
            if should_replace {
                best.insert(key, fact);
            }
        }
    }
    (best.len() == query.focuses.len()).then_some(best)
}

fn best_same_session_sale_facts(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &SaleMinimumQuery,
) -> Option<HashMap<String, MoneyAmountFact>> {
    candidates
        .iter()
        .filter_map(|(session_id, session_rank)| {
            let lines = idx.find_session_lines(session_id, false, 256, |line, lower| {
                is_summary_or_user_line(line, lower)
            });
            let facts = collect_sale_facts(lines, &query.focuses, *session_rank);
            (facts.len() == query.focuses.len()).then_some(facts)
        })
        .max_by_key(|facts| facts.values().map(|fact| fact.score).sum::<usize>())
}

fn best_global_sale_facts(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &SaleMinimumQuery,
) -> Option<HashMap<String, MoneyAmountFact>> {
    let mut best = HashMap::new();
    for (session_id, session_rank) in candidates {
        let lines = idx.find_session_lines(session_id, false, 256, |line, lower| {
            is_summary_or_user_line(line, lower)
        });
        let facts = collect_sale_facts(lines, &query.focuses, *session_rank);
        for (key, fact) in facts {
            let should_replace = best
                .get(&key)
                .map(|existing: &MoneyAmountFact| {
                    fact.score > existing.score
                        || (fact.score == existing.score
                            && fact.amount_cents < existing.amount_cents)
                })
                .unwrap_or(true);
            if should_replace {
                best.insert(key, fact);
            }
        }
    }
    (best.len() == query.focuses.len()).then_some(best)
}

fn collect_single_recipient_gift_facts(
    lines: Vec<String>,
    focus: &SpendFocus,
    session_rank: usize,
) -> HashMap<String, MoneyAmountFact> {
    let mut best = HashMap::new();
    for line in lines {
        let lower = line.to_ascii_lowercase();
        let Some(fact) = extract_keyed_gift_recipient_fact_from_line(&line, &lower, focus) else {
            continue;
        };
        let scored_fact = MoneyAmountFact {
            score: session_rank * 100 + fact.score,
            amount_cents: fact.amount_cents,
            evidence: fact.evidence,
        };
        upsert_fact_by_key(&mut best, fact.key, scored_fact);
    }
    best
}

fn collect_spend_facts(
    lines: Vec<String>,
    focuses: &[SpendFocus],
    session_rank: usize,
) -> HashMap<String, MoneyAmountFact> {
    let mut best = HashMap::new();
    let mut pending_contexts: HashMap<String, usize> = HashMap::new();
    for (line_idx, line) in lines.into_iter().enumerate() {
        let lower = line.to_ascii_lowercase();
        let mut matched_focus = false;
        for focus in focuses {
            if line_matches_spend_focus(&lower, focus)
                && matches!(focus.kind, SpendFocusKind::GiftRecipient)
                && extract_spend_focus_fact_from_line(&line, &lower, focus).is_none()
            {
                pending_contexts.insert(focus.key.clone(), line_idx);
            }

            let Some(fact) = extract_spend_focus_fact_from_line(&line, &lower, focus) else {
                continue;
            };
            matched_focus = true;
            let scored_fact = MoneyAmountFact {
                score: session_rank * 100 + fact.score,
                ..fact
            };
            let should_replace = best
                .get(&focus.key)
                .map(|existing: &MoneyAmountFact| scored_fact.score > existing.score)
                .unwrap_or(true);
            if should_replace {
                best.insert(focus.key.clone(), scored_fact);
            }
        }
        if matched_focus {
            continue;
        }

        let Some(followup_fact) = extract_contextual_spend_followup_fact_from_line(&line, &lower)
        else {
            continue;
        };
        let Some(focus) = focuses
            .iter()
            .filter(|focus| matches!(focus.kind, SpendFocusKind::GiftRecipient))
            .filter_map(|focus| {
                pending_contexts
                    .get(&focus.key)
                    .copied()
                    .map(|context_idx| (focus, context_idx))
            })
            .filter(|(_, context_idx)| line_idx.saturating_sub(*context_idx) <= 2)
            .max_by_key(|(_, context_idx)| *context_idx)
            .map(|(focus, _)| focus)
        else {
            continue;
        };
        let scored_fact = MoneyAmountFact {
            score: session_rank * 100 + followup_fact.score,
            ..followup_fact
        };
        let should_replace = best
            .get(&focus.key)
            .map(|existing: &MoneyAmountFact| scored_fact.score > existing.score)
            .unwrap_or(true);
        if should_replace {
            best.insert(focus.key.clone(), scored_fact);
        }
        pending_contexts.remove(&focus.key);
    }
    best
}

fn upsert_fact_by_key(
    best: &mut HashMap<String, MoneyAmountFact>,
    key: String,
    fact: MoneyAmountFact,
) {
    let should_replace = best
        .get(&key)
        .map(|existing: &MoneyAmountFact| fact.score > existing.score)
        .unwrap_or(true);
    if should_replace {
        best.insert(key, fact);
    }
}

fn collect_sale_facts(
    lines: Vec<String>,
    focuses: &[SpendFocus],
    session_rank: usize,
) -> HashMap<String, MoneyAmountFact> {
    let mut best = HashMap::new();
    for line in lines {
        let lower = line.to_ascii_lowercase();
        for focus in focuses {
            let Some(fact) = extract_sale_value_fact_from_line(&line, &lower, focus) else {
                continue;
            };
            let scored_fact = MoneyAmountFact {
                score: session_rank * 100 + fact.score,
                ..fact
            };
            let should_replace = best
                .get(&focus.key)
                .map(|existing: &MoneyAmountFact| {
                    scored_fact.score > existing.score
                        || (scored_fact.score == existing.score
                            && scored_fact.amount_cents < existing.amount_cents)
                })
                .unwrap_or(true);
            if should_replace {
                best.insert(focus.key.clone(), scored_fact);
            }
        }
    }
    best
}
