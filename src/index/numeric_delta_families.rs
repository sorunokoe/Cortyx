use super::conversation_scan_support::{scanned_conversation_lines, session_score};
use super::numeric_delta_extractors::{
    extract_anchored_money_fact_from_line, extract_goal_money_fact_from_line,
    extract_metric_delta_fact_from_line, format_money_delta, format_numeric_delta,
    parse_numeric_delta_query, AnchoredMoneyDeltaQuery, GoalMoneyDeltaQuery, MetricDeltaQuery,
    MetricFactKind, MetricValueFact, MoneyFactKind, MoneyValueFact, NumericDeltaQuery,
};
use super::*;

impl NeuronIndex {
    pub(super) fn synthetic_numeric_delta_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        match parse_numeric_delta_query(task_lower)? {
            NumericDeltaQuery::Metric(query) => self.synthetic_metric_delta_answer(task, &query),
            NumericDeltaQuery::GoalMoney(query) => {
                self.synthetic_goal_money_delta_answer(task, &query)
            },
            NumericDeltaQuery::AnchoredMoney(query) => {
                self.synthetic_anchored_money_delta_answer(task, &query)
            },
        }
    }

    fn synthetic_metric_delta_answer(
        &self,
        task: &str,
        query: &MetricDeltaQuery,
    ) -> Option<PathBuf> {
        let candidates =
            self.collect_signature_candidates(task, &query.required_terms, 12, |line, lower| {
                extract_metric_delta_fact_from_line(line, lower, query).is_some()
            });
        let pair = best_entry_scanned_metric_pair(self, query)
            .or_else(|| best_same_session_metric_pair(self, &candidates, query))?;
        let delta = pair.previous.value - pair.current.value;
        (delta > 0.0).then_some(())?;
        self.write_synthetic_answer(
            "numeric-metric-delta",
            task,
            &format_numeric_delta(delta),
            &[pair.previous.evidence, pair.current.evidence],
        )
    }

    fn synthetic_goal_money_delta_answer(
        &self,
        task: &str,
        query: &GoalMoneyDeltaQuery,
    ) -> Option<PathBuf> {
        let candidates =
            self.collect_signature_candidates(task, &query.required_terms, 12, |line, lower| {
                extract_goal_money_fact_from_line(line, lower, query).is_some()
            });
        let pair = best_entry_scanned_goal_money_pair(self, query)
            .or_else(|| best_same_session_goal_money_pair(self, &candidates, query))?;
        let delta = pair
            .actual
            .amount_cents
            .checked_sub(pair.goal.amount_cents)?;
        (delta > 0).then_some(())?;
        self.write_synthetic_answer(
            "numeric-goal-money-delta",
            task,
            &format_money_delta(delta),
            &[pair.goal.evidence, pair.actual.evidence],
        )
    }

    fn synthetic_anchored_money_delta_answer(
        &self,
        task: &str,
        query: &AnchoredMoneyDeltaQuery,
    ) -> Option<PathBuf> {
        let candidates =
            self.collect_signature_candidates(task, &query.required_terms, 12, |line, lower| {
                extract_anchored_money_fact_from_line(line, lower, query).is_some()
            });
        let pair = best_entry_scanned_anchored_money_pair(self, query)
            .or_else(|| best_same_session_anchored_money_pair(self, &candidates, query))?;
        let delta = pair
            .left
            .amount_cents
            .checked_sub(pair.right.amount_cents)?;
        (delta > 0).then_some(())?;
        self.write_synthetic_answer(
            "numeric-anchored-money-delta",
            task,
            &format_money_delta(delta),
            &[pair.left.evidence, pair.right.evidence],
        )
    }
}

#[derive(Clone)]
struct MetricPair {
    score: usize,
    previous: MetricValueFact,
    current: MetricValueFact,
}

#[derive(Clone)]
struct GoalMoneyPair {
    score: usize,
    goal: MoneyValueFact,
    actual: MoneyValueFact,
}

#[derive(Clone)]
struct AnchoredMoneyPair {
    score: usize,
    left: MoneyValueFact,
    right: MoneyValueFact,
}

fn best_same_session_metric_pair(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &MetricDeltaQuery,
) -> Option<MetricPair> {
    candidates
        .iter()
        .filter_map(|(session_id, session_rank)| {
            let lines = idx.find_session_lines(session_id, false, 256, |line, lower| {
                lower.starts_with("user:")
                    && extract_metric_delta_fact_from_line(line, lower, query).is_some()
            });
            let facts = lines
                .iter()
                .filter_map(|line| {
                    let lower = line.to_ascii_lowercase();
                    extract_metric_delta_fact_from_line(line, &lower, query)
                })
                .collect::<Vec<_>>();
            let previous = best_metric_fact(&facts, MetricFactKind::Previous)?;
            let current = best_metric_fact(&facts, MetricFactKind::Current)?;
            Some(MetricPair {
                score: session_score(*session_rank, previous.score + current.score),
                previous,
                current,
            })
        })
        .max_by_key(|pair| pair.score)
}

fn best_same_session_goal_money_pair(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &GoalMoneyDeltaQuery,
) -> Option<GoalMoneyPair> {
    candidates
        .iter()
        .filter_map(|(session_id, session_rank)| {
            let lines = idx.find_session_lines(session_id, false, 256, |line, lower| {
                lower.starts_with("user:")
                    && extract_goal_money_fact_from_line(line, lower, query).is_some()
            });
            let facts = lines
                .iter()
                .filter_map(|line| {
                    let lower = line.to_ascii_lowercase();
                    extract_goal_money_fact_from_line(line, &lower, query)
                })
                .collect::<Vec<_>>();
            let goal = best_money_fact(&facts, MoneyFactKind::Goal)?;
            let actual = best_money_fact(&facts, MoneyFactKind::Actual)?;
            Some(GoalMoneyPair {
                score: session_score(*session_rank, goal.score + actual.score),
                goal,
                actual,
            })
        })
        .max_by_key(|pair| pair.score)
}

fn best_same_session_anchored_money_pair(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &AnchoredMoneyDeltaQuery,
) -> Option<AnchoredMoneyPair> {
    candidates
        .iter()
        .filter_map(|(session_id, session_rank)| {
            let lines = idx.find_session_lines(session_id, false, 256, |line, lower| {
                lower.starts_with("user:")
                    && extract_anchored_money_fact_from_line(line, lower, query).is_some()
            });
            let facts = lines
                .iter()
                .filter_map(|line| {
                    let lower = line.to_ascii_lowercase();
                    extract_anchored_money_fact_from_line(line, &lower, query)
                })
                .collect::<Vec<_>>();
            let left = best_money_fact(&facts, MoneyFactKind::Left)?;
            let right = best_money_fact(&facts, MoneyFactKind::Right)?;
            Some(AnchoredMoneyPair {
                score: session_score(*session_rank, left.score + right.score),
                left,
                right,
            })
        })
        .max_by_key(|pair| pair.score)
}

fn best_entry_scanned_metric_pair(
    idx: &NeuronIndex,
    query: &MetricDeltaQuery,
) -> Option<MetricPair> {
    scanned_conversation_lines(idx)
        .filter_map(|lines| {
            let facts = lines
                .iter()
                .filter_map(|line| {
                    let lower = line.to_ascii_lowercase();
                    extract_metric_delta_fact_from_line(line, &lower, query)
                })
                .collect::<Vec<_>>();
            let previous = best_metric_fact(&facts, MetricFactKind::Previous)?;
            let current = best_metric_fact(&facts, MetricFactKind::Current)?;
            Some(MetricPair {
                score: previous.score + current.score,
                previous,
                current,
            })
        })
        .max_by_key(|pair| pair.score)
}

fn best_entry_scanned_goal_money_pair(
    idx: &NeuronIndex,
    query: &GoalMoneyDeltaQuery,
) -> Option<GoalMoneyPair> {
    scanned_conversation_lines(idx)
        .filter_map(|lines| {
            let facts = lines
                .iter()
                .filter_map(|line| {
                    let lower = line.to_ascii_lowercase();
                    extract_goal_money_fact_from_line(line, &lower, query)
                })
                .collect::<Vec<_>>();
            let goal = best_money_fact(&facts, MoneyFactKind::Goal)?;
            let actual = best_money_fact(&facts, MoneyFactKind::Actual)?;
            Some(GoalMoneyPair {
                score: goal.score + actual.score,
                goal,
                actual,
            })
        })
        .max_by_key(|pair| pair.score)
}

fn best_entry_scanned_anchored_money_pair(
    idx: &NeuronIndex,
    query: &AnchoredMoneyDeltaQuery,
) -> Option<AnchoredMoneyPair> {
    scanned_conversation_lines(idx)
        .filter_map(|lines| {
            let facts = lines
                .iter()
                .filter_map(|line| {
                    let lower = line.to_ascii_lowercase();
                    extract_anchored_money_fact_from_line(line, &lower, query)
                })
                .collect::<Vec<_>>();
            let left = best_money_fact(&facts, MoneyFactKind::Left)?;
            let right = best_money_fact(&facts, MoneyFactKind::Right)?;
            Some(AnchoredMoneyPair {
                score: left.score + right.score,
                left,
                right,
            })
        })
        .max_by_key(|pair| pair.score)
}

fn best_metric_fact(facts: &[MetricValueFact], kind: MetricFactKind) -> Option<MetricValueFact> {
    facts
        .iter()
        .filter(|fact| fact.kind == kind)
        .cloned()
        .max_by_key(|fact| fact.score)
}

fn best_money_fact(facts: &[MoneyValueFact], kind: MoneyFactKind) -> Option<MoneyValueFact> {
    facts
        .iter()
        .filter(|fact| fact.kind == kind)
        .cloned()
        .max_by_key(|fact| fact.score)
}
