//! Duration extraction and rendering: parse, calculate, and format temporal durations.
//!
//! This module handles extraction of durations from text, computation from date pairs,
//! and natural language formatting of duration values.

use super::*;

pub(crate) fn extract_duration_months(text: &str) -> Option<i32> {
    let years = extract_count_before_unit(text, "year").unwrap_or(0);
    let months = extract_count_before_unit(text, "month").unwrap_or(0);
    ((years > 0) || (months > 0)).then_some(years * 12 + months)
}

pub(crate) fn extract_duration_months_near_phrases(text: &str, phrases: &[&str]) -> Option<i32> {
    let tokens = duration_candidate_tokens(text);
    let spans = extract_duration_month_spans(&tokens);
    if spans.is_empty() {
        return None;
    }

    let mut positions = phrases
        .iter()
        .flat_map(|phrase| phrase_token_positions(&tokens, phrase))
        .collect::<Vec<_>>();
    positions.sort();
    positions.dedup();
    if positions.is_empty() {
        return extract_duration_months(text);
    }

    positions
        .into_iter()
        .flat_map(|position| {
            spans.iter().map(move |(start, end, months)| {
                let distance = if *start >= position {
                    *start - position
                } else {
                    position.saturating_sub(*end)
                };
                (distance, *start, *months)
            })
        })
        .min_by(|left, right| left.cmp(right))
        .map(|(_, _, months)| months)
        .or_else(|| extract_duration_months(text))
}

pub(crate) fn extract_duration_days(text: &str) -> Option<i32> {
    let tokens = duration_candidate_tokens(text);
    extract_duration_day_spans(&tokens)
        .first()
        .map(|(_, _, days)| *days)
}

pub(crate) fn extract_duration_days_near_phrases(text: &str, phrases: &[&str]) -> Option<i32> {
    let tokens = duration_candidate_tokens(text);
    let spans = extract_duration_day_spans(&tokens);
    if spans.is_empty() {
        return None;
    }

    let mut positions = phrases
        .iter()
        .flat_map(|phrase| phrase_token_positions(&tokens, phrase))
        .collect::<Vec<_>>();
    positions.sort();
    positions.dedup();
    if positions.is_empty() {
        return extract_duration_days(text);
    }

    positions
        .into_iter()
        .flat_map(|position| {
            spans.iter().map(move |(start, end, days)| {
                let distance = if *start >= position {
                    *start - position
                } else {
                    position.saturating_sub(*end)
                };
                (distance, *start, *days)
            })
        })
        .min_by(|left, right| left.cmp(right))
        .map(|(_, _, days)| days)
        .or_else(|| extract_duration_days(text))
}

fn duration_candidate_tokens(text: &str) -> Vec<String> {
    text.to_ascii_lowercase()
        .split_whitespace()
        .map(|token| token.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '+'))
        .filter(|token| !token.is_empty())
        .map(|token| token.to_string())
        .collect()
}

fn extract_duration_month_spans(tokens: &[String]) -> Vec<(usize, usize, i32)> {
    let mut spans = Vec::new();
    let mut index = 0usize;
    while index + 1 < tokens.len() {
        let Some(amount) = parse_count_token(tokens[index].as_str()) else {
            index += 1;
            continue;
        };
        let unit = tokens[index + 1].as_str();
        if matches!(unit, "month" | "months") {
            spans.push((index, index + 2, amount));
            index += 2;
            continue;
        }
        if !matches!(unit, "year" | "years") {
            index += 1;
            continue;
        }

        let mut months = amount * 12;
        let mut end = index + 2;
        let month_start = if tokens.get(end).map(|token| token.as_str()) == Some("and") {
            end + 1
        } else {
            end
        };
        if let (Some(month_amount), Some(month_unit)) =
            (tokens.get(month_start), tokens.get(month_start + 1))
        {
            if matches!(month_unit.as_str(), "month" | "months") {
                if let Some(extra_months) = parse_count_token(month_amount) {
                    months += extra_months;
                    end = month_start + 2;
                }
            }
        }
        spans.push((index, end, months));
        index = end;
    }
    spans
}

fn duration_unit_days(unit: &str) -> Option<i32> {
    match unit {
        "day" | "days" => Some(1),
        "week" | "weeks" => Some(7),
        "month" | "months" => Some(30),
        "year" | "years" => Some(365),
        _ => None,
    }
}

fn extract_duration_day_spans(tokens: &[String]) -> Vec<(usize, usize, i32)> {
    let mut spans = Vec::new();
    let mut index = 0usize;
    while index + 1 < tokens.len() {
        let Some(amount) = parse_count_token(tokens[index].as_str()) else {
            index += 1;
            continue;
        };
        let Some(scale) = duration_unit_days(tokens[index + 1].as_str()) else {
            index += 1;
            continue;
        };

        let mut total_days = amount * scale;
        let mut end = index + 2;
        let next_start = if tokens.get(end).map(|token| token.as_str()) == Some("and") {
            end + 1
        } else {
            end
        };
        if let (Some(extra_amount), Some(extra_unit)) =
            (tokens.get(next_start), tokens.get(next_start + 1))
        {
            if let (Some(extra_count), Some(extra_scale)) = (
                parse_count_token(extra_amount.as_str()),
                duration_unit_days(extra_unit.as_str()),
            ) {
                total_days += extra_count * extra_scale;
                end = next_start + 2;
            }
        }

        spans.push((index, end, total_days));
        index = end;
    }
    spans
}

fn phrase_token_positions(tokens: &[String], phrase: &str) -> Vec<usize> {
    let phrase_tokens = duration_candidate_tokens(phrase);
    if phrase_tokens.is_empty() || phrase_tokens.len() > tokens.len() {
        return Vec::new();
    }

    (0..=tokens.len() - phrase_tokens.len())
        .filter(|start| tokens[*start..*start + phrase_tokens.len()] == phrase_tokens[..])
        .collect()
}

pub(crate) fn merge_duration_max(slot: &mut Option<i32>, candidate: Option<i32>) {
    if let Some(value) = candidate {
        *slot = Some(slot.map_or(value, |existing| existing.max(value)));
    }
}

fn extract_count_before_unit(text: &str, unit: &str) -> Option<i32> {
    let lower = text.to_ascii_lowercase();
    let unit_plural = format!("{unit}s");
    let tokens = lower
        .split_whitespace()
        .map(|token| token.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '+'))
        .collect::<Vec<_>>();
    for idx in 1..tokens.len() {
        if tokens[idx] == unit || tokens[idx] == unit_plural {
            if let Some(value) = parse_count_token(tokens[idx - 1]) {
                return Some(value);
            }
        }
    }
    None
}

pub(crate) fn format_duration_months(total_months: i32) -> String {
    let years = total_months / 12;
    let months = total_months % 12;
    match (years, months) {
        (0, months) => format!("{months} months"),
        (years, 0) => format!("{years} years"),
        (years, months) => format!("{years} years and {months} months"),
    }
}

pub(crate) fn render_temporal_gap_answer(
    days: i32,
    style: &TemporalGapAnswerStyle,
) -> Option<String> {
    match style {
        TemporalGapAnswerStyle::FixedUnit { unit } => {
            let amount = convert_days_to_gap_unit(days, unit)?;
            let suffix = if amount == 1 {
                unit.as_str()
            } else {
                match unit.as_str() {
                    "day" => "days",
                    "week" => "weeks",
                    "month" => "months",
                    "year" => "years",
                    _ => return None,
                }
            };
            Some(format!("{amount} {suffix}"))
        },
        TemporalGapAnswerStyle::NaturalLanguage => Some(render_natural_duration(days)),
    }
}

fn convert_days_to_gap_unit(days: i32, unit: &str) -> Option<i32> {
    let amount = match unit {
        "day" => days,
        "week" => (days + 3) / 7,
        "month" => (days + 15) / 30,
        "year" => (days + 182) / 365,
        _ => return None,
    };
    Some(amount.max(0))
}

pub(crate) fn render_natural_duration(days: i32) -> String {
    if days >= 365 {
        let years = days / 365;
        let months = ((days % 365) + 15) / 30;
        if months == 0 {
            return render_small_duration_quantity(years, "year");
        }
        return format!(
            "{} and {}",
            render_small_duration_quantity(years, "year"),
            render_small_duration_quantity(months, "month")
        );
    }

    if days >= 45 {
        return render_small_duration_quantity(((days + 15) / 30).max(1), "month");
    }

    if days >= 7 {
        return render_small_duration_quantity(((days + 3) / 7).max(1), "week");
    }

    render_small_duration_quantity(days.max(1), "day")
}

fn render_small_duration_quantity(amount: i32, unit: &str) -> String {
    let quantity = small_number_word(amount).unwrap_or_else(|| amount.to_string());
    let suffix = if amount == 1 {
        unit.to_string()
    } else {
        format!("{unit}s")
    };
    format!("{quantity} {suffix}")
}

pub(crate) fn elapsed_days_since_anchor(anchor_rank: Option<i32>, rank: i32) -> Option<i32> {
    if rank < 0 {
        return Some(-rank);
    }
    Some((anchor_rank? - rank).abs())
}

pub(crate) fn convert_days_to_elapsed_unit(days: i32, unit: &str) -> Option<i32> {
    let amount = match unit {
        "day" => days,
        "week" => (days + 3) / 7,
        "month" => (days + 15) / 30,
        "year" => (days + 182) / 365,
        _ => return None,
    };
    Some(amount.max(1))
}

pub(crate) fn render_relative_elapsed(unit: &str, amount: i32) -> String {
    let quantity = small_number_word(amount).unwrap_or_else(|| amount.to_string());
    let suffix = if amount == 1 {
        unit.to_string()
    } else {
        format!("{unit}s")
    };
    format!("{quantity} {suffix} ago")
}

fn small_number_word(value: i32) -> Option<String> {
    let word = match value {
        0 => "zero",
        1 => "one",
        2 => "two",
        3 => "three",
        4 => "four",
        5 => "five",
        6 => "six",
        7 => "seven",
        8 => "eight",
        9 => "nine",
        10 => "ten",
        11 => "eleven",
        12 => "twelve",
        _ => return None,
    };
    Some(word.to_string())
}
