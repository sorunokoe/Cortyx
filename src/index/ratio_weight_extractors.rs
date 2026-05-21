use super::money_support::{extract_money_after_markers, parse_money_cents};
use super::*;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum RatioWeightQuery {
    WeightTotal(WeightTotalQuery),
    Percentage(PercentageQuery),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct WeightTotalQuery {
    pub(super) focus: QueryFocus,
    pub(super) required_terms: Vec<String>,
    pub(super) is_feed_like: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct PercentageQuery {
    pub(super) numerator: QueryFocus,
    pub(super) denominator: QueryFocus,
    pub(super) kind: PercentageKind,
    pub(super) required_terms: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct QueryFocus {
    pub(super) key: String,
    pub(super) required_terms: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PercentageKind {
    Count,
    Money,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct RatioWeightFact {
    pub(super) key: String,
    pub(super) value: i64,
    pub(super) unit: Option<String>,
    pub(super) score: usize,
    pub(super) evidence: String,
}

pub(super) fn parse_ratio_weight_query(task: &str, task_lower: &str) -> Option<RatioWeightQuery> {
    parse_weight_total_query(task_lower)
        .map(RatioWeightQuery::WeightTotal)
        .or_else(|| parse_percentage_query(task, task_lower).map(RatioWeightQuery::Percentage))
}

pub(super) fn extract_weight_purchase_fact_from_line(
    line: &str,
    lower: &str,
    query: &WeightTotalQuery,
) -> Option<RatioWeightFact> {
    if !is_summary_or_user_line(line, lower)
        || !task_contains_any(lower, &["purchased", "bought", "got"])
        || !weight_focus_matches(lower, query)
    {
        return None;
    }
    let captures =
        compile_regex_static(r"(?i)\b(\d+)\s*(?:-| )?(pounds?|lbs?|kilograms?|kgs?|kg)\b")
            .captures(line)?;
    let value = captures.get(1)?.as_str().parse::<i64>().ok()?;
    let unit = normalize_weight_unit(captures.get(2)?.as_str())?;
    let descriptor = weight_descriptor_key(lower, query).unwrap_or_else(|| query.focus.key.clone());
    Some(RatioWeightFact {
        key: format!("weight:{unit}:{value}:{descriptor}"),
        value,
        unit: Some(unit),
        score: 18
            + usize::try_from(value.max(0)).unwrap_or(0)
            + focus_overlap(lower, &query.focus) * 6
            + usize::from(lower.starts_with("user:")) * 6,
        evidence: line.trim().to_string(),
    })
}

pub(super) fn extract_percentage_whole_fact_from_line(
    line: &str,
    lower: &str,
    query: &PercentageQuery,
) -> Option<RatioWeightFact> {
    if !is_summary_or_user_line(line, lower) {
        return None;
    }
    match query.kind {
        PercentageKind::Count => {
            if focus_overlap(lower, &query.denominator) == 0
                || !task_contains_any(lower, &["total", "across", "overall"])
            {
                return None;
            }
            let value = extract_line_numbers(line).into_iter().max()? as i64;
            Some(RatioWeightFact {
                key: "whole".to_string(),
                value,
                unit: None,
                score: 20
                    + focus_overlap(lower, &query.denominator) * 8
                    + usize::from(lower.starts_with("user:")) * 6,
                evidence: line.trim().to_string(),
            })
        },
        PercentageKind::Money => {
            if focus_overlap(lower, &query.denominator) == 0
                || !task_contains_any(lower, &["listed", "price", "worth", "valued"])
            {
                return None;
            }
            let value = extract_money_after_markers(
                line,
                &[
                    r"(?i)\b(?:listed at|listed for|price(?:d)? at|worth|valued at)\b[^$\n]{0,24}?\$([0-9][0-9,]*(?:\.\d{1,2})?)",
                    r"(?i)\$([0-9][0-9,]*(?:\.\d{1,2})?)",
                ],
            )?;
            Some(RatioWeightFact {
                key: "whole".to_string(),
                value,
                unit: Some("cents".to_string()),
                score: 20
                    + focus_overlap(lower, &query.denominator) * 8
                    + usize::from(lower.starts_with("user:")) * 6,
                evidence: line.trim().to_string(),
            })
        },
    }
}

pub(super) fn extract_percentage_part_fact_from_line(
    line: &str,
    lower: &str,
    query: &PercentageQuery,
) -> Option<RatioWeightFact> {
    if !is_summary_or_user_line(line, lower) {
        return None;
    }
    match query.kind {
        PercentageKind::Count => {
            if focus_overlap(lower, &query.numerator) == 0
                || focus_overlap(lower, &query.denominator) == 0
            {
                return None;
            }
            let value = compile_regex_static(r"(?i)\b(\d+)\s+of\s+the\b")
                .captures(line)
                .and_then(|captures| captures.get(1))
                .and_then(|value| value.as_str().parse::<i64>().ok())
                .or_else(|| {
                    extract_line_numbers(line)
                        .into_iter()
                        .max()
                        .map(|value| value as i64)
                })?;
            Some(RatioWeightFact {
                key: "part".to_string(),
                value,
                unit: None,
                score: 22
                    + focus_overlap(lower, &query.numerator) * 8
                    + focus_overlap(lower, &query.denominator) * 4
                    + usize::from(lower.starts_with("user:")) * 6,
                evidence: line.trim().to_string(),
            })
        },
        PercentageKind::Money => {
            if focus_overlap(lower, &query.numerator) == 0
                || !task_contains_any(lower, &["cost", "budget", "estimate", "renovation"])
            {
                return None;
            }
            let value = extract_money_after_markers(
                line,
                &[
                    r"(?i)\b(?:cost(?: around| approximately)?|estimate(?: will cost|d at)?|budget(?:ed)?(?: at)?)\b[^$\n]{0,32}?\$([0-9][0-9,]*(?:\.\d{1,2})?)",
                    r"(?i)\$([0-9][0-9,]*(?:\.\d{1,2})?)",
                ],
            )?;
            Some(RatioWeightFact {
                key: "part".to_string(),
                value,
                unit: Some("cents".to_string()),
                score: 22
                    + focus_overlap(lower, &query.numerator) * 8
                    + usize::from(lower.starts_with("user:")) * 6,
                evidence: line.trim().to_string(),
            })
        },
    }
}

pub(super) fn format_percentage_answer(part: i64, whole: i64) -> Option<String> {
    if part < 0 || whole <= 0 {
        return None;
    }
    let basis_points = ((part * 10_000) + (whole / 2)) / whole;
    if basis_points % 100 == 0 {
        return Some(format!("{}%", basis_points / 100));
    }
    if basis_points % 10 == 0 {
        return Some(format!("{:.1}%", basis_points as f64 / 100.0));
    }
    Some(format!("{:.2}%", basis_points as f64 / 100.0))
}

fn parse_weight_total_query(task_lower: &str) -> Option<WeightTotalQuery> {
    if !task_lower.contains("total weight")
        || !task_contains_any(task_lower, &["purchased", "bought", "got"])
    {
        return None;
    }
    let captures = compile_regex_static(
        r"(?i)total weight of (?:the )?(.+?)\s+i\s+(?:purchased|bought|got)\b",
    )
    .captures(task_lower)?;
    let focus = build_query_focus(captures.get(1)?.as_str())?;
    let is_feed_like = task_contains_any(captures.get(1)?.as_str(), &["feed", "grain", "grains"]);
    let mut required_terms = focus.required_terms.clone();
    required_terms.extend(
        [
            "weight",
            "pound",
            "pounds",
            "kg",
            "purchased",
            "bought",
            "got",
        ]
        .into_iter()
        .map(str::to_string),
    );
    if is_feed_like {
        required_terms.extend(
            ["feed", "grain", "grains", "scratch", "layer", "chickens"]
                .into_iter()
                .map(str::to_string),
        );
    }
    required_terms.sort();
    required_terms.dedup();
    Some(WeightTotalQuery {
        focus,
        required_terms,
        is_feed_like,
    })
}

fn parse_percentage_query(task: &str, task_lower: &str) -> Option<PercentageQuery> {
    if !task_lower.starts_with("what percentage of ") {
        return None;
    }
    let surface = task_lower
        .strip_prefix("what percentage of ")?
        .trim_end_matches('?');
    let (denominator_surface, numerator_surface) =
        if let Some((left, right)) = surface.split_once(" is ") {
            (left, right)
        } else if let Some((left, right)) = surface.split_once(" do ") {
            (left, trim_percentage_tail(right)?)
        } else {
            let (left, right) = surface.split_once(" does ")?;
            (left, trim_percentage_tail(right)?)
        };
    let denominator = build_query_focus(denominator_surface)?;
    let numerator = build_query_focus(numerator_surface)?;
    let mut required_terms = denominator.required_terms.clone();
    required_terms.extend(numerator.required_terms.iter().cloned());
    required_terms.push("percentage".to_string());
    required_terms.sort();
    required_terms.dedup();
    Some(PercentageQuery {
        numerator,
        denominator,
        kind: infer_percentage_kind(task),
        required_terms,
    })
}

fn trim_percentage_tail(surface: &str) -> Option<&str> {
    [
        " hold",
        " have",
        " occupy",
        " occupies",
        " represent",
        " represents",
    ]
    .into_iter()
    .filter_map(|marker| surface.split_once(marker).map(|(left, _)| left.trim()))
    .find(|value| !value.is_empty())
}

fn build_query_focus(surface: &str) -> Option<QueryFocus> {
    let cleaned = surface
        .trim()
        .trim_start_matches("the ")
        .trim_start_matches("my ")
        .trim();
    let required_terms = synthetic_query_terms(cleaned);
    (!required_terms.is_empty()).then_some(QueryFocus {
        key: normalized_synthetic_phrase_key(cleaned),
        required_terms,
    })
}

fn infer_percentage_kind(task: &str) -> PercentageKind {
    let lower = task.to_ascii_lowercase();
    if task_contains_any(&lower, &["price", "cost", "budget", "value"]) {
        PercentageKind::Money
    } else {
        PercentageKind::Count
    }
}

fn focus_overlap(lower: &str, focus: &QueryFocus) -> usize {
    term_overlap_count(
        lower,
        &focus
            .required_terms
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
    )
}

fn is_summary_or_user_line(line: &str, lower: &str) -> bool {
    lower.starts_with("user:") || line.trim_start().starts_with('-')
}

fn normalize_weight_unit(raw: &str) -> Option<String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "pound" | "pounds" | "lb" | "lbs" => Some("pounds".to_string()),
        "kilogram" | "kilograms" | "kg" | "kgs" => Some("kg".to_string()),
        _ => None,
    }
}

fn weight_focus_matches(lower: &str, query: &WeightTotalQuery) -> bool {
    focus_overlap(lower, &query.focus) > 0
        || (query.is_feed_like && task_contains_any(lower, &["feed", "grain", "grains", "scratch"]))
}

fn weight_descriptor_key(lower: &str, query: &WeightTotalQuery) -> Option<String> {
    let captured = compile_regex_static(
        r"(?i)\b\d+\s*(?:-| )?(?:pounds?|lbs?|kilograms?|kgs?|kg)\s+of\s+([a-z][a-z\s-]{0,48})",
    )
    .captures(lower)
    .and_then(|captures| captures.get(1))
    .map(|value| value.as_str().to_string());
    let descriptor = captured
        .as_deref()
        .unwrap_or_else(|| {
            if lower.contains("layer feed") {
                "layer feed"
            } else if lower.contains("feed") {
                "feed"
            } else {
                ""
            }
        })
        .split([',', '.', ';'])
        .next()
        .unwrap_or_default()
        .split(" for ")
        .next()
        .unwrap_or_default()
        .split(" recently")
        .next()
        .unwrap_or_default()
        .trim()
        .to_string();
    if !descriptor.is_empty() {
        return Some(normalized_synthetic_phrase_key(&descriptor));
    }
    query.is_feed_like.then_some(query.focus.key.clone())
}
