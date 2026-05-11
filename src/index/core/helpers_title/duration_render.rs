use super::*;
use crate::index::compile_regex;

pub(in crate::index) fn extract_title_duration_value(
    line: &str,
    title_lower: &str,
) -> Option<SyntheticDurationValue> {
    let lower = line.to_ascii_lowercase();
    if !lower.contains(title_lower) {
        return None;
    }
    for marker in ["which took me ", "took me ", "took "] {
        let Some(idx) = lower.find(marker) else {
            continue;
        };
        let tail = &lower[idx + marker.len()..];
        if let Some(value) = parse_leading_duration_value(tail) {
            return Some(value);
        }
    }
    None
}

pub(in crate::index) fn parse_leading_duration_value(text: &str) -> Option<SyntheticDurationValue> {
    let regex = compile_regex(
        r"(?i)^\s*(?:about\s+|around\s+)?(a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|couple|few|\d+(?:\.\d+)?)(\s+and\s+a\s+half)?\s+(day|days|week|weeks|month|months|year|years)\b",
    );
    let caps = regex.captures(text)?;
    let mut amount =
        caps.get(1)
            .and_then(|value| match value.as_str().to_ascii_lowercase().as_str() {
                "a" | "an" | "one" => Some(1.0),
                "two" => Some(2.0),
                "three" => Some(3.0),
                "four" => Some(4.0),
                "five" => Some(5.0),
                "six" => Some(6.0),
                "seven" => Some(7.0),
                "eight" => Some(8.0),
                "nine" => Some(9.0),
                "ten" => Some(10.0),
                "eleven" => Some(11.0),
                "twelve" => Some(12.0),
                "couple" => Some(2.0),
                "few" => Some(3.0),
                value => value.parse::<f32>().ok(),
            })?;
    if caps.get(2).is_some() {
        amount += 0.5;
    }
    let unit = caps.get(3)?.as_str().to_ascii_lowercase();
    let days = amount
        * match unit.as_str() {
            "day" | "days" => 1.0,
            "week" | "weeks" => 7.0,
            "month" | "months" => 30.0,
            "year" | "years" => 365.0,
            _ => return None,
        };
    Some(SyntheticDurationValue {
        amount,
        days,
        unit: match unit.as_str() {
            "day" | "days" => "day",
            "week" | "weeks" => "week",
            "month" | "months" => "month",
            "year" | "years" => "year",
            _ => return None,
        },
    })
}

pub(in crate::index) fn render_duration_unit(unit: &'static str, amount: f32) -> &'static str {
    if (amount - 1.0).abs() < f32::EPSILON {
        unit
    } else {
        match unit {
            "day" => "days",
            "week" => "weeks",
            "month" => "months",
            "year" => "years",
            _ => unit,
        }
    }
}

pub(in crate::index) fn render_elapsed_duration_answer(days: i32) -> String {
    if days % 30 == 0 {
        return render_small_duration(days / 30, "month");
    }
    if days % 7 == 0 {
        return render_small_duration(days / 7, "week");
    }
    if (7..=10).contains(&days) {
        return "one week".to_string();
    }
    render_small_duration(days, "day")
}

pub(in crate::index) fn render_elapsed_from_now_answer(
    days: i32,
    unit: SyntheticElapsedFromNowUnit,
    append_ago: bool,
) -> String {
    let answer = match unit {
        SyntheticElapsedFromNowUnit::Day => render_small_duration(days, "day"),
        SyntheticElapsedFromNowUnit::Week => (((days as f32) / 7.0).round() as i32).to_string(),
        SyntheticElapsedFromNowUnit::Month => (((days as f32) / 30.0).round() as i32).to_string(),
        SyntheticElapsedFromNowUnit::Year => (((days as f32) / 365.0).round() as i32).to_string(),
    };
    if append_ago {
        format!("{answer} ago")
    } else {
        answer
    }
}

pub(in crate::index) fn render_small_duration(amount: i32, unit: &str) -> String {
    let amount_text = match amount {
        1 => "one".to_string(),
        2 => "two".to_string(),
        3 => "three".to_string(),
        4 => "four".to_string(),
        5 => "five".to_string(),
        6 => "six".to_string(),
        7 => "seven".to_string(),
        8 => "eight".to_string(),
        9 => "nine".to_string(),
        10 => "ten".to_string(),
        11 => "eleven".to_string(),
        12 => "twelve".to_string(),
        _ => amount.to_string(),
    };
    let rendered_unit = if amount == 1 {
        unit
    } else {
        match unit {
            "day" => "days",
            "week" => "weeks",
            "month" => "months",
            "year" => "years",
            _ => unit,
        }
    };
    format!("{amount_text} {rendered_unit}")
}
