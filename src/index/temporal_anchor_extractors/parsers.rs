//! Query parsers for interval and elapsed gap queries.

use super::types::*;
use super::super::*;

pub(super) fn parse_elapsed_before_event_query(task_lower: &str) -> Option<ElapsedBeforeEventQuery> {
    let trimmed = task_lower.trim().trim_end_matches('?');
    let (subject_phrase, event_phrase) =
        if let Some(rest) = trimmed.strip_prefix("how long had i been ") {
            let (subject, event) = rest.split_once(" when ")?;
            (subject.trim().to_string(), event.trim().to_string())
        } else {
            let rest = trimmed.strip_prefix("how long did i ")?;
            let (subject, event) = rest.split_once(" before i ")?;
            (subject.trim().to_string(), format!("i {}", event.trim()))
        };
    let required_terms = build_required_terms([subject_phrase.as_str(), event_phrase.as_str()]);
    (!required_terms.is_empty()).then_some(ElapsedBeforeEventQuery {
        subject_phrase,
        event_phrase,
        required_terms,
    })
}

pub(super) fn parse_temporal_interval_query(task_lower: &str) -> Option<TemporalIntervalQuery> {
    parse_days_between_query(task_lower)
        .or_else(|| parse_days_after_query(task_lower))
        .or_else(|| parse_days_before_query(task_lower))
}

pub(super) fn parse_temporal_elapsed_gap_query(task_lower: &str) -> Option<TemporalElapsedGapQuery> {
    parse_elapsed_since_when_query(task_lower)
        .or_else(|| parse_elapsed_after_query(task_lower))
        .or_else(|| parse_elapsed_have_i_been_query(task_lower))
}

fn parse_days_between_query(task_lower: &str) -> Option<TemporalIntervalQuery> {
    let trimmed = task_lower.trim().trim_end_matches('?');
    let rest = trimmed
        .strip_prefix("how many days had passed between ")
        .or_else(|| trimmed.strip_prefix("how many days passed between "))?;
    let (start_phrase, end_phrase) = rest.split_once(" and ")?;
    let required_terms = build_required_terms([start_phrase.trim(), end_phrase.trim()]);
    (!required_terms.is_empty()).then_some(TemporalIntervalQuery {
        start_phrase: start_phrase.trim().to_string(),
        end_phrase: end_phrase.trim().to_string(),
        required_terms,
    })
}

fn parse_days_after_query(task_lower: &str) -> Option<TemporalIntervalQuery> {
    let trimmed = task_lower.trim().trim_end_matches('?');
    let rest = trimmed.strip_prefix("how many days did it take ")?;
    let (end_phrase, start_phrase) = rest.split_once(" after ")?;
    let end_phrase = end_phrase
        .strip_prefix("for me to ")
        .or_else(|| end_phrase.strip_prefix("me to "))
        .unwrap_or(end_phrase)
        .trim()
        .to_string();
    let start_phrase = start_phrase.trim().to_string();
    let required_terms = build_required_terms([end_phrase.as_str(), start_phrase.as_str()]);
    (!required_terms.is_empty()).then_some(TemporalIntervalQuery {
        start_phrase,
        end_phrase,
        required_terms,
    })
}

fn parse_days_before_query(task_lower: &str) -> Option<TemporalIntervalQuery> {
    let trimmed = task_lower.trim().trim_end_matches('?');
    let rest = trimmed.strip_prefix("how many days before ")?;
    let (end_phrase, start_phrase) = rest.split_once(" did i ")?;
    let required_terms = build_required_terms([end_phrase.trim(), start_phrase.trim()]);
    (!required_terms.is_empty()).then_some(TemporalIntervalQuery {
        start_phrase: start_phrase.trim().to_string(),
        end_phrase: end_phrase.trim().to_string(),
        required_terms,
    })
}

fn parse_elapsed_since_when_query(task_lower: &str) -> Option<TemporalElapsedGapQuery> {
    let trimmed = strip_temporal_reference_prefix(task_lower)
        .trim()
        .trim_end_matches('?');
    for unit in ["day", "week"] {
        let Some(rest) =
            strip_prefix_case_insensitive(trimmed, &format!("How many {unit}s had passed since "))
                .or_else(|| {
                    strip_prefix_case_insensitive(
                        trimmed,
                        &format!("How many {unit} had passed since "),
                    )
                })
        else {
            continue;
        };
        let (start_phrase, end_phrase) = split_once_case_insensitive(rest, " when ")?;
        let required_terms = build_required_terms([start_phrase.trim(), end_phrase.trim()]);
        if !required_terms.is_empty() {
            return Some(TemporalElapsedGapQuery {
                start_phrase: start_phrase.trim().to_string(),
                end_phrase: end_phrase.trim().to_string(),
                unit: unit.to_string(),
                required_terms,
            });
        }
    }
    None
}

fn parse_elapsed_after_query(task_lower: &str) -> Option<TemporalElapsedGapQuery> {
    let trimmed = strip_temporal_reference_prefix(task_lower)
        .trim()
        .trim_end_matches('?');
    for unit in ["day", "week"] {
        let Some(rest) =
            strip_prefix_case_insensitive(trimmed, &format!("How many {unit}s did it take "))
                .or_else(|| {
                    strip_prefix_case_insensitive(trimmed, &format!("How many {unit} did it take "))
                })
        else {
            continue;
        };
        let (end_phrase, start_phrase) = split_once_case_insensitive(rest, " after ")?;
        let end_phrase = strip_prefix_case_insensitive(end_phrase, "for me to ")
            .or_else(|| strip_prefix_case_insensitive(end_phrase, "me to "))
            .unwrap_or(end_phrase)
            .trim()
            .to_string();
        let start_phrase = start_phrase.trim().to_string();
        let required_terms = build_required_terms([start_phrase.as_str(), end_phrase.as_str()]);
        if !required_terms.is_empty() {
            return Some(TemporalElapsedGapQuery {
                start_phrase,
                end_phrase,
                unit: unit.to_string(),
                required_terms,
            });
        }
    }
    None
}

fn parse_elapsed_have_i_been_query(task_lower: &str) -> Option<TemporalElapsedGapQuery> {
    let trimmed = strip_temporal_reference_prefix(task_lower)
        .trim()
        .trim_end_matches('?');
    for unit in ["day", "week"] {
        let Some(rest) =
            strip_prefix_case_insensitive(trimmed, &format!("How many {unit}s have I been "))
                .or_else(|| {
                    strip_prefix_case_insensitive(trimmed, &format!("How many {unit} have I been "))
                })
                .or_else(|| {
                    strip_prefix_case_insensitive(trimmed, &format!("How many {unit}s had I been "))
                })
                .or_else(|| {
                    strip_prefix_case_insensitive(trimmed, &format!("How many {unit} had I been "))
                })
        else {
            continue;
        };
        let (start_phrase, end_phrase) = split_once_case_insensitive(rest, " when ")?;
        let required_terms = build_required_terms([start_phrase.trim(), end_phrase.trim()]);
        if !required_terms.is_empty() {
            return Some(TemporalElapsedGapQuery {
                start_phrase: start_phrase.trim().to_string(),
                end_phrase: end_phrase.trim().to_string(),
                unit: unit.to_string(),
                required_terms,
            });
        }
    }
    None
}

fn build_required_terms<'a, I>(phrases: I) -> Vec<String>
where
    I: IntoIterator<Item = &'a str>,
{
    let mut terms = Vec::new();
    for phrase in phrases {
        terms.extend(synthetic_query_terms(phrase));
    }
    terms.sort();
    terms.dedup();
    terms
}

fn split_once_case_insensitive<'a>(text: &'a str, needle: &str) -> Option<(&'a str, &'a str)> {
    let lower = text.to_ascii_lowercase();
    let needle_lower = needle.to_ascii_lowercase();
    let idx = lower.find(&needle_lower)?;
    Some((&text[..idx], &text[idx + needle.len()..]))
}

fn strip_prefix_case_insensitive<'a>(text: &'a str, prefix: &str) -> Option<&'a str> {
    if text.len() < prefix.len() {
        return None;
    }
    text[..prefix.len()]
        .eq_ignore_ascii_case(prefix)
        .then_some(&text[prefix.len()..])
}
