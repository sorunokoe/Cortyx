use super::*;
use crate::index::compile_regex;

pub(in crate::index) fn extract_current_role_total_months_from_line(
    line: &str,
    lower: &str,
) -> Option<i32> {
    let has_total_marker = task_contains_any(
        lower,
        &[
            "experience in the company",
            "experience at the company",
            "with the company",
            "at the company",
            "been at ",
            "been with ",
            "working at ",
        ],
    );
    if !has_total_marker {
        return None;
    }
    extract_duration_months_from_text(line)
}

pub(in crate::index) fn extract_current_role_offset_months_from_line(
    line: &str,
    lower: &str,
) -> Option<i32> {
    if !task_contains_any(
        lower,
        &[
            "worked my way up to ",
            "promoted to ",
            "promotion to ",
            "moved into ",
            "became ",
        ],
    ) {
        return None;
    }
    let (_, tail) = lower.split_once(" after ")?;
    extract_duration_months_from_text(tail).or_else(|| extract_duration_months_from_text(line))
}

pub(in crate::index) fn extract_current_role_title_from_transition_line(
    line: &str,
    lower: &str,
) -> Option<String> {
    for marker in [
        "worked my way up to ",
        "promoted to ",
        "promotion to ",
        "moved into ",
        "became ",
    ] {
        let Some(start) = lower.find(marker) else {
            continue;
        };
        let tail = &line[start + marker.len()..];
        let title = [" after ", ",", "."]
            .iter()
            .filter_map(|delimiter| tail.find(delimiter))
            .min()
            .map(|end| &tail[..end])
            .unwrap_or(tail)
            .trim()
            .trim_matches(|c: char| matches!(c, '"' | '\'' | ',' | '.' | ';' | ':'));
        if !title.is_empty() {
            return Some(title.to_ascii_lowercase());
        }
    }
    None
}

pub(in crate::index) fn render_month_span(total_months: i32) -> String {
    let years = total_months / 12;
    let months = total_months % 12;
    match (years, months) {
        (0, months) => format!("{months} {}", if months == 1 { "month" } else { "months" }),
        (years, 0) => format!("{years} {}", if years == 1 { "year" } else { "years" }),
        (years, months) => format!(
            "{years} {} and {months} {}",
            if years == 1 { "year" } else { "years" },
            if months == 1 { "month" } else { "months" }
        ),
    }
}

pub(in crate::index) fn extract_explicit_date_rank(line: &str) -> Option<i32> {
    let numeric = compile_regex(r"(?i)\b(\d{1,2})/(\d{1,2})(?:/(\d{4}))?\b");
    if let Some(caps) = numeric.captures(line) {
        let month = caps.get(1)?.as_str().parse::<u32>().ok()?;
        let day = caps.get(2)?.as_str().parse::<u32>().ok()?;
        if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
            return None;
        }
        let year = caps
            .get(3)
            .and_then(|value| value.as_str().parse::<i32>().ok())
            .unwrap_or(2023);
        return Some(ymd_to_days(year, month, day));
    }

    let month_day = compile_regex(
        r"(?i)\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(\d{4}))?\b",
    );
    if let Some(caps) = month_day.captures(line) {
        let month = named_month_to_number(caps.get(1)?.as_str())?;
        let day = caps.get(2)?.as_str().parse::<u32>().ok()?;
        let year = caps
            .get(3)
            .and_then(|value| value.as_str().parse::<i32>().ok())
            .unwrap_or(2023);
        return Some(ymd_to_days(year, month, day));
    }

    let day_month_named = compile_regex(
        r"(?i)\b(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December)(?:,\s*(\d{4}))?\b",
    );
    if let Some(caps) = day_month_named.captures(line) {
        let day = caps.get(1)?.as_str().parse::<u32>().ok()?;
        let month = named_month_to_number(caps.get(2)?.as_str())?;
        let year = caps
            .get(3)
            .and_then(|value| value.as_str().parse::<i32>().ok())
            .unwrap_or(2023);
        return Some(ymd_to_days(year, month, day));
    }

    let day_month = compile_regex(
        r"(?i)\b(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)?\s+of\s+(January|February|March|April|May|June|July|August|September|October|November|December)(?:,\s*(\d{4}))?\b",
    );
    if let Some(caps) = day_month.captures(line) {
        let day = caps.get(1)?.as_str().parse::<u32>().ok()?;
        let month = named_month_to_number(caps.get(2)?.as_str())?;
        let year = caps
            .get(3)
            .and_then(|value| value.as_str().parse::<i32>().ok())
            .unwrap_or(2023);
        return Some(ymd_to_days(year, month, day));
    }

    let fuzzy_month = compile_regex(
        r"(?i)\b(?:(early|mid|late)[-\s]+)?(January|February|March|April|May|June|July|August|September|October|November|December)(?:,\s*|\s+)?(\d{4})?\b",
    );
    let caps = fuzzy_month.captures(line)?;
    let month = named_month_to_number(caps.get(2)?.as_str())?;
    let day = match caps
        .get(1)
        .map(|value| value.as_str().to_ascii_lowercase())
        .as_deref()
    {
        Some("early") => 5,
        Some("late") => 25,
        _ => 15,
    };
    let year = caps
        .get(3)
        .and_then(|value| value.as_str().parse::<i32>().ok())
        .unwrap_or(2023);
    Some(ymd_to_days(year, month, day))
}

pub(in crate::index) fn named_month_to_number(month: &str) -> Option<u32> {
    match &month.to_ascii_lowercase()[..] {
        "january" => Some(1),
        "february" => Some(2),
        "march" => Some(3),
        "april" => Some(4),
        "may" => Some(5),
        "june" => Some(6),
        "july" => Some(7),
        "august" => Some(8),
        "september" => Some(9),
        "october" => Some(10),
        "november" => Some(11),
        "december" => Some(12),
        _ => None,
    }
}

pub(in crate::index) fn ymd_to_days(year: i32, month: u32, day: u32) -> i32 {
    const MONTH_START_DAYS: [i32; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let leap_years = {
        let y = year - 1;
        y / 4 - y / 100 + y / 400 - (1969 / 4 - 1969 / 100 + 1969 / 400)
    };
    (year - 1970) * 365 + leap_years + MONTH_START_DAYS[(month - 1) as usize] + day as i32 - 1
}
