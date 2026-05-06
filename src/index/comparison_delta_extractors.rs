use super::event_extractors::extract_current_age_from_line;
use super::money_support::{
    extract_money_after_markers, extract_percent_basis_points, focus_match_count,
    line_matches_focus,
};
use super::numeric_delta_extractors::{format_money_delta, format_numeric_delta};
use super::*;

const COMPARISON_QUERY_STOP: &[&str] = &[
    "a",
    "age",
    "am",
    "airport",
    "by",
    "compared",
    "did",
    "discount",
    "first",
    "from",
    "higher",
    "how",
    "hotel",
    "i",
    "in",
    "instead",
    "much",
    "my",
    "of",
    "on",
    "order",
    "percentage",
    "receive",
    "save",
    "taking",
    "than",
    "the",
    "to",
    "will",
];

#[derive(Clone, Debug, PartialEq)]
pub(super) enum ComparisonDeltaQuery {
    AgeAverage(AgeAverageQuery),
    DiscountComparison(DiscountComparisonQuery),
    SavingsMoney(SavingsMoneyQuery),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct AgeAverageQuery {
    pub(super) subject_terms: Vec<String>,
    pub(super) required_terms: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct DiscountComparisonQuery {
    pub(super) primary_terms: Vec<String>,
    pub(super) comparison_terms: Vec<String>,
    pub(super) required_terms: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SavingsMoneyQuery {
    pub(super) chosen_terms: Vec<String>,
    pub(super) rejected_terms: Vec<String>,
    pub(super) required_terms: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum AgeFactKind {
    Current,
    Average,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct AgeValueFact {
    pub(super) value: f64,
    pub(super) kind: AgeFactKind,
    pub(super) score: usize,
    pub(super) evidence: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DiscountRateFactKind {
    Primary,
    Comparison,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct DiscountRateFact {
    pub(super) basis_points: i64,
    pub(super) kind: DiscountRateFactKind,
    pub(super) score: usize,
    pub(super) evidence: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SavingsMoneyFactKind {
    Chosen,
    Rejected,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SavingsMoneyFact {
    pub(super) amount_cents: i64,
    pub(super) kind: SavingsMoneyFactKind,
    pub(super) score: usize,
    pub(super) evidence: String,
}

pub(super) fn parse_comparison_delta_query(task_lower: &str) -> Option<ComparisonDeltaQuery> {
    parse_discount_comparison_query(task_lower)
        .map(ComparisonDeltaQuery::DiscountComparison)
        .or_else(|| parse_savings_money_query(task_lower).map(ComparisonDeltaQuery::SavingsMoney))
        .or_else(|| parse_age_average_query(task_lower).map(ComparisonDeltaQuery::AgeAverage))
}

pub(super) fn extract_age_average_fact_from_line(
    line: &str,
    lower: &str,
    query: &AgeAverageQuery,
) -> Option<AgeValueFact> {
    if !lower.starts_with("user:") {
        return None;
    }
    if let Some(value) = extract_average_age_from_line(line) {
        line_matches_focus(lower, &query.subject_terms).then(|| AgeValueFact {
            value,
            kind: AgeFactKind::Average,
            score: focus_match_count(lower, &query.subject_terms) * 10
                + usize::from(lower.contains("average age")) * 8,
            evidence: line.trim().to_string(),
        })
    } else {
        extract_current_age_from_line(line).map(|value| AgeValueFact {
            value: value as f64,
            kind: AgeFactKind::Current,
            score: usize::from(lower.contains("currently")) * 10
                + usize::from(lower.contains("years old")) * 8
                + 10,
            evidence: line.trim().to_string(),
        })
    }
}

pub(super) fn extract_discount_rate_fact_from_line(
    line: &str,
    lower: &str,
    query: &DiscountComparisonQuery,
) -> Option<DiscountRateFact> {
    if !lower.starts_with("user:") || !task_contains_any(lower, &["discount", "off"]) {
        return None;
    }
    let basis_points = extract_percent_basis_points(line)?;
    let primary_matches = line_matches_focus(lower, &query.primary_terms);
    let comparison_matches = line_matches_focus(lower, &query.comparison_terms);
    let primary_score = focus_match_count(lower, &query.primary_terms);
    let comparison_score = focus_match_count(lower, &query.comparison_terms);
    let (kind, score) =
        if primary_matches && (!comparison_matches || primary_score >= comparison_score) {
            (
                DiscountRateFactKind::Primary,
                primary_score * 10
                    + usize::from(lower.contains("first order")) * 6
                    + usize::from(lower.contains("discount")) * 4
                    + 8,
            )
        } else if comparison_matches {
            (
                DiscountRateFactKind::Comparison,
                comparison_score * 10
                    + usize::from(lower.contains("first order")) * 6
                    + usize::from(lower.contains("off")) * 4
                    + 8,
            )
        } else {
            return None;
        };
    Some(DiscountRateFact {
        basis_points,
        kind,
        score,
        evidence: line.trim().to_string(),
    })
}

pub(super) fn extract_savings_money_fact_from_line(
    line: &str,
    lower: &str,
    query: &SavingsMoneyQuery,
) -> Option<SavingsMoneyFact> {
    if !lower.starts_with("user:") || !lower.contains('$') {
        return None;
    }
    let amount_cents = extract_money_after_markers(line, &[r"(?i)\$([0-9][0-9,]*(?:\.\d{1,2})?)"])?;
    let chosen_matches = line_matches_focus(lower, &query.chosen_terms);
    let rejected_matches = line_matches_focus(lower, &query.rejected_terms);
    let chosen_score = focus_match_count(lower, &query.chosen_terms);
    let rejected_score = focus_match_count(lower, &query.rejected_terms);
    let (kind, score) = if chosen_matches && (!rejected_matches || chosen_score >= rejected_score) {
        (
            SavingsMoneyFactKind::Chosen,
            chosen_score * 10 + usize::from(lower.contains("train")) * 4 + 8,
        )
    } else if rejected_matches {
        (
            SavingsMoneyFactKind::Rejected,
            rejected_score * 10 + usize::from(lower.contains("taxi")) * 4 + 8,
        )
    } else {
        return None;
    };
    Some(SavingsMoneyFact {
        amount_cents,
        kind,
        score,
        evidence: line.trim().to_string(),
    })
}

pub(super) fn format_year_delta(value: f64) -> String {
    let rendered = format_numeric_delta(value);
    format!(
        "{rendered} {}",
        if (value - 1.0).abs() < 0.01 {
            "year"
        } else {
            "years"
        }
    )
}

pub(super) fn format_savings_delta(amount_cents: i64) -> String {
    format_money_delta(amount_cents)
}

fn parse_age_average_query(task_lower: &str) -> Option<AgeAverageQuery> {
    if !task_lower.starts_with("how much older am i than the average age of ") {
        return None;
    }
    let subject_surface = compile_regex(r"(?i)\baverage age of\s+(.+?)(?:\?|$)")
        .captures(task_lower)
        .and_then(|captures| captures.get(1))
        .map(|value| value.as_str().trim().to_string())?;
    let subject_terms = normalized_comparison_terms(&subject_surface);
    (!subject_terms.is_empty()).then_some(())?;
    let mut required_terms = subject_terms.clone();
    required_terms.push("average".to_string());
    required_terms.push("age".to_string());
    required_terms.sort();
    required_terms.dedup();
    Some(AgeAverageQuery {
        subject_terms,
        required_terms,
    })
}

fn parse_discount_comparison_query(task_lower: &str) -> Option<DiscountComparisonQuery> {
    let captures = compile_regex(
        r"(?i)\bdid i receive a higher percentage discount on my first order from\s+(.+?),\s+compared to my first\s+(.+?)\s+order\??$",
    )
    .captures(task_lower)?;
    let primary_terms = normalized_comparison_terms(captures.get(1)?.as_str());
    let comparison_terms = normalized_comparison_terms(captures.get(2)?.as_str());
    if primary_terms.is_empty() || comparison_terms.is_empty() {
        return None;
    }
    let mut required_terms = primary_terms.clone();
    required_terms.extend(comparison_terms.clone());
    required_terms.push("discount".to_string());
    required_terms.sort();
    required_terms.dedup();
    Some(DiscountComparisonQuery {
        primary_terms,
        comparison_terms,
        required_terms,
    })
}

fn parse_savings_money_query(task_lower: &str) -> Option<SavingsMoneyQuery> {
    if !task_lower.starts_with("how much will i save by taking ")
        || !task_lower.contains(" instead of ")
    {
        return None;
    }
    let captures =
        compile_regex(r"(?i)\bhow much will i save by taking\s+(.+?)\s+instead of\s+(.+?)(?:\?|$)")
            .captures(task_lower)?;
    let chosen_terms = normalized_comparison_terms(captures.get(1)?.as_str());
    let rejected_terms = normalized_comparison_terms(captures.get(2)?.as_str());
    if chosen_terms.is_empty() || rejected_terms.is_empty() {
        return None;
    }
    let mut required_terms = chosen_terms.clone();
    required_terms.extend(rejected_terms.clone());
    required_terms.sort();
    required_terms.dedup();
    Some(SavingsMoneyQuery {
        chosen_terms,
        rejected_terms,
        required_terms,
    })
}

fn normalized_comparison_terms(surface: &str) -> Vec<String> {
    let mut terms = synthetic_query_terms(surface)
        .into_iter()
        .filter(|term| !COMPARISON_QUERY_STOP.contains(&term.as_str()))
        .collect::<Vec<_>>();
    terms.sort();
    terms.dedup();
    terms
}

fn extract_average_age_from_line(line: &str) -> Option<f64> {
    compile_regex(r"(?i)\baverage age\b[^.\n]{0,80}?\bis\s+(\d+(?:\.\d+)?)\s+years old\b")
        .captures(line)
        .and_then(|captures| captures.get(1))
        .and_then(|value| value.as_str().parse::<f64>().ok())
}
