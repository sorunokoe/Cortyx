use super::*;

pub(super) fn month_number(raw: &str) -> Option<u32> {
    match raw.to_ascii_lowercase().as_str() {
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

pub(super) fn ordinal_day(month: u32, day: u32) -> u32 {
    const DAYS_BEFORE_MONTH: [u32; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    DAYS_BEFORE_MONTH
        .get(month.saturating_sub(1) as usize)
        .copied()
        .unwrap_or(0)
        + day
}

pub(super) fn extract_count_before_phrase(lower: &str, phrase: &str) -> Option<usize> {
    let pattern = compile_regex(&format!(
        r"(?i)\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+{}\b",
        regex::escape(phrase)
    ))
    .unwrap_or_else(|err| panic!("escaped quantity regex failed to compile: {err}"));
    let captures = pattern.captures(lower)?;
    parse_count_token(captures.get(1)?.as_str())
}

fn parse_count_token(raw: &str) -> Option<usize> {
    match raw.to_ascii_lowercase().as_str() {
        "one" => Some(1),
        "two" => Some(2),
        "three" => Some(3),
        "four" => Some(4),
        "five" => Some(5),
        "six" => Some(6),
        "seven" => Some(7),
        "eight" => Some(8),
        "nine" => Some(9),
        "ten" => Some(10),
        value => value.parse::<usize>().ok(),
    }
}
