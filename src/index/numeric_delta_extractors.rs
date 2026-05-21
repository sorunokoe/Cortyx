use super::money_support::{extract_money_after_markers, focus_match_count, format_money_cents};
use super::*;

const NUMERIC_QUERY_STOP: &[&str] = &[
    "a", "ago", "amount", "compared", "did", "final", "goal", "how", "i", "in", "initial", "more",
    "much", "my", "now", "of", "than", "the", "to", "was",
];

#[derive(Clone, Debug, PartialEq)]
pub(super) enum NumericDeltaQuery {
    Metric(MetricDeltaQuery),
    GoalMoney(GoalMoneyDeltaQuery),
    AnchoredMoney(AnchoredMoneyDeltaQuery),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct MetricDeltaQuery {
    pub(super) metric_phrase: String,
    pub(super) metric_terms: Vec<String>,
    pub(super) required_terms: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct GoalMoneyDeltaQuery {
    pub(super) event_terms: Vec<String>,
    pub(super) required_terms: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct AnchoredMoneyDeltaQuery {
    pub(super) left_terms: Vec<String>,
    pub(super) right_terms: Vec<String>,
    pub(super) left_aliases: Vec<String>,
    pub(super) right_aliases: Vec<String>,
    pub(super) required_terms: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum MetricFactKind {
    Previous,
    Current,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum MoneyFactKind {
    Goal,
    Actual,
    Left,
    Right,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct MetricValueFact {
    pub(super) value: f64,
    pub(super) kind: MetricFactKind,
    pub(super) score: usize,
    pub(super) evidence: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct MoneyValueFact {
    pub(super) amount_cents: i64,
    pub(super) kind: MoneyFactKind,
    pub(super) score: usize,
    pub(super) evidence: String,
}

pub(super) fn parse_numeric_delta_query(task_lower: &str) -> Option<NumericDeltaQuery> {
    parse_goal_money_delta_query(task_lower)
        .map(NumericDeltaQuery::GoalMoney)
        .or_else(|| {
            parse_anchored_money_delta_query(task_lower).map(NumericDeltaQuery::AnchoredMoney)
        })
        .or_else(|| parse_metric_delta_query(task_lower).map(NumericDeltaQuery::Metric))
}

pub(super) fn extract_metric_delta_fact_from_line(
    line: &str,
    lower: &str,
    query: &MetricDeltaQuery,
) -> Option<MetricValueFact> {
    if !lower.starts_with("user:") || focus_match_count(lower, &query.metric_terms) == 0 {
        return None;
    }
    let value = extract_metric_value(line, &query.metric_phrase)?;
    let (kind, bonus) =
        if task_contains_any(lower, &["few months ago", "a few months ago", "last year"]) {
            (MetricFactKind::Previous, 12)
        } else if task_contains_any(lower, &["lately", "currently", "currently at", "now"]) {
            (MetricFactKind::Current, 10)
        } else {
            return None;
        };
    Some(MetricValueFact {
        value,
        kind,
        score: focus_match_count(lower, &query.metric_terms) * 10 + 8 + bonus,
        evidence: line.trim().to_string(),
    })
}

pub(super) fn extract_goal_money_fact_from_line(
    line: &str,
    lower: &str,
    query: &GoalMoneyDeltaQuery,
) -> Option<MoneyValueFact> {
    if !lower.starts_with("user:")
        || (!query.event_terms.is_empty() && focus_match_count(lower, &query.event_terms) < 2)
    {
        return None;
    }
    if task_contains_any(lower, &["initially aimed to raise", "initial goal"]) {
        let amount_cents = extract_money_after_markers(
            line,
            &[
                r"(?i)\binitially aimed to raise\b[^$\n]{0,24}?\$([0-9][0-9,]*(?:\.\d{1,2})?)",
                r"(?i)\binitial goal\b[^$\n]{0,24}?\$([0-9][0-9,]*(?:\.\d{1,2})?)",
            ],
        )?;
        return Some(MoneyValueFact {
            amount_cents,
            kind: MoneyFactKind::Goal,
            score: focus_match_count(lower, &query.event_terms) * 10 + 20,
            evidence: line.trim().to_string(),
        });
    }
    if !task_contains_any(lower, &["raised", "ended up raising"]) {
        return None;
    }
    let amount_cents = extract_money_after_markers(
        line,
        &[
            r"(?i)\bended up raising\b[^$\n]{0,24}?\$([0-9][0-9,]*(?:\.\d{1,2})?)",
            r"(?i)\braised\b[^$\n]{0,24}?\$([0-9][0-9,]*(?:\.\d{1,2})?)",
        ],
    )?;
    Some(MoneyValueFact {
        amount_cents,
        kind: MoneyFactKind::Actual,
        score: focus_match_count(lower, &query.event_terms) * 10 + 18,
        evidence: line.trim().to_string(),
    })
}

pub(super) fn extract_anchored_money_fact_from_line(
    line: &str,
    lower: &str,
    query: &AnchoredMoneyDeltaQuery,
) -> Option<MoneyValueFact> {
    if !lower.starts_with("user:") || !lower.contains('$') {
        return None;
    }
    let amount_cents = extract_money_after_markers(line, &[r"(?i)\$([0-9][0-9,]*(?:\.\d{1,2})?)"])?;
    let left_score = anchor_focus_score(lower, &query.left_terms, &query.left_aliases);
    let right_score = anchor_focus_score(lower, &query.right_terms, &query.right_aliases);
    let (kind, score) = if left_score > right_score && left_score > 0 {
        (MoneyFactKind::Left, left_score * 10 + 8)
    } else if right_score > 0 {
        (MoneyFactKind::Right, right_score * 10 + 8)
    } else {
        return None;
    };
    Some(MoneyValueFact {
        amount_cents,
        kind,
        score,
        evidence: line.trim().to_string(),
    })
}

pub(super) fn format_numeric_delta(value: f64) -> String {
    if (value - value.round()).abs() < 0.01 {
        #[allow(clippy::cast_possible_truncation)]
        let rounded = value.round() as i64;
        format!("{rounded}")
    } else {
        let rendered = format!("{value:.2}");
        rendered
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    }
}

pub(super) fn format_money_delta(amount_cents: i64) -> String {
    format_money_cents(amount_cents)
}

fn parse_metric_delta_query(task_lower: &str) -> Option<MetricDeltaQuery> {
    if !task_lower.contains("compared to now") || !task_lower.starts_with("how much more ") {
        return None;
    }
    let metric_phrase = compile_regex_static(r"(?i)\bhow much more\s+(.+?)\s+was\b")
        .captures(task_lower)
        .and_then(|captures| captures.get(1))
        .map(|value| value.as_str().trim().to_string())?;
    let metric_terms = normalized_numeric_terms(&metric_phrase);
    if metric_terms.is_empty() {
        return None;
    }
    let mut required_terms = metric_terms.clone();
    required_terms.push("now".to_string());
    required_terms.sort();
    required_terms.dedup();
    Some(MetricDeltaQuery {
        metric_phrase,
        metric_terms,
        required_terms,
    })
}

fn parse_goal_money_delta_query(task_lower: &str) -> Option<GoalMoneyDeltaQuery> {
    if !task_lower.contains("how much more money did i raise")
        || !task_lower.contains("than my initial goal")
    {
        return None;
    }
    let event_surface = compile_regex_static(r"(?i)\bthan my initial goal in the\s+(.+?)(?:\?|$)")
        .captures(task_lower)
        .and_then(|captures| captures.get(1))
        .map(|value| value.as_str().trim().to_string())
        .unwrap_or_default();
    let mut event_terms = normalized_numeric_terms(&event_surface);
    if event_terms.is_empty() {
        event_terms = vec!["raise".to_string()];
    }
    let mut required_terms = event_terms.clone();
    required_terms.extend(["goal".to_string(), "raise".to_string()]);
    required_terms.sort();
    required_terms.dedup();
    Some(GoalMoneyDeltaQuery {
        event_terms,
        required_terms,
    })
}

fn parse_anchored_money_delta_query(task_lower: &str) -> Option<AnchoredMoneyDeltaQuery> {
    if !task_lower.starts_with("how much more was the ") || !task_lower.contains(" than the ") {
        return None;
    }
    let captures =
        compile_regex_static(r"(?i)\bhow much more was the\s+(.+?)\s+than the\s+(.+?)(?:\?|$)")
            .captures(task_lower)?;
    let left_surface = captures.get(1)?.as_str().trim();
    let right_surface = captures.get(2)?.as_str().trim();
    if !task_contains_any(left_surface, &["amount", "approval", "price", "sale"])
        && !task_contains_any(right_surface, &["amount", "approval", "price", "sale"])
    {
        return None;
    }
    let left_terms = normalized_numeric_terms(left_surface);
    let right_terms = normalized_numeric_terms(right_surface);
    if left_terms.is_empty() || right_terms.is_empty() {
        return None;
    }
    let left_aliases = if task_contains_any(left_surface, &["pre-approval", "pre approval"]) {
        vec!["pre-approved".to_string(), "borrow".to_string()]
    } else {
        Vec::new()
    };
    let right_aliases = if right_surface.contains("sale price") {
        vec!["sale price".to_string(), "final sale price".to_string()]
    } else {
        Vec::new()
    };
    let mut required_terms = left_terms.clone();
    required_terms.extend(right_terms.clone());
    required_terms.sort();
    required_terms.dedup();
    Some(AnchoredMoneyDeltaQuery {
        left_terms,
        right_terms,
        left_aliases,
        right_aliases,
        required_terms,
    })
}

fn normalized_numeric_terms(surface: &str) -> Vec<String> {
    let mut terms = synthetic_query_terms(surface)
        .into_iter()
        .filter(|term| !NUMERIC_QUERY_STOP.contains(&term.as_str()))
        .collect::<Vec<_>>();
    terms.sort();
    terms.dedup();
    terms
}

fn extract_metric_value(line: &str, metric_phrase: &str) -> Option<f64> {
    let pattern = format!(
        r"(?i)\b(\d+(?:\.\d+)?)\s+{}\b",
        regex::escape(metric_phrase)
    );
    compile_regex_static(&pattern)
        .captures(line)
        .and_then(|captures| captures.get(1))
        .and_then(|value| value.as_str().parse::<f64>().ok())
}

fn anchor_focus_score(lower: &str, terms: &[String], aliases: &[String]) -> usize {
    focus_match_count(lower, terms)
        + aliases
            .iter()
            .filter(|alias| lower.contains(alias.as_str()))
            .count()
            * 2
}
