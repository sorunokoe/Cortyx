//! Relative temporal recall query parsing.

use super::types::*;
use super::super::*;

pub(super) fn parse_relative_temporal_recall_query(task_lower: &str) -> Option<RelativeTemporalRecallQuery> {
    let trimmed = task_lower.trim().trim_end_matches('?');
    let captures = Regex::new(r"^as of\s+(.+?\d{4}),\s*(.+)$")
        .unwrap()
        .captures(trimmed)?;
    let as_of_text = captures.get(1)?.as_str();
    let prompt_body = captures.get(2)?.as_str().trim().to_string();
    let (year, month, day) = parse_relative_recall_date(as_of_text)?;
    let target_day = relative_recall_target_day(&prompt_body, year, month, day)?;
    let focus_terms = relative_recall_focus_terms(&prompt_body);
    Some(RelativeTemporalRecallQuery {
        target_day,
        prompt_body: prompt_body.clone(),
        focus_terms,
        answer_kind: relative_recall_answer_kind(&prompt_body),
    })
}

fn parse_relative_recall_date(text: &str) -> Option<(i32, u32, u32)> {
    let captures = Regex::new(
        r"(?i)\b(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december),\s*(\d{4})\b",
    )
    .unwrap()
    .captures(text)?;
    let day = captures.get(1)?.as_str().parse::<u32>().ok()?;
    let month = named_month_to_number(captures.get(2)?.as_str())?;
    let year = captures.get(3)?.as_str().parse::<i32>().ok()?;
    Some((year, month, day))
}

fn relative_recall_target_day(body: &str, year: i32, month: u32, day: u32) -> Option<i32> {
    if let Some(weekday) = parse_last_weekday_marker(body) {
        let as_of_day = ymd_to_days(year, month, day);
        let current = weekday_number(year, month, day);
        let delta = (current - weekday + 7) % 7;
        return Some(as_of_day - if delta == 0 { 7 } else { delta });
    }

    let (amount, unit) = parse_relative_unit_ago(body)?;
    match unit.as_str() {
        "day" => Some(ymd_to_days(year, month, day) - amount),
        "week" => Some(ymd_to_days(year, month, day) - amount * 7),
        "month" => {
            let (target_year, target_month) = shift_months(year, month, -amount)?;
            Some(ymd_to_days(
                target_year,
                target_month,
                day.min(days_in_month(target_year, target_month)),
            ))
        },
        "year" => {
            let target_year = year - amount;
            Some(ymd_to_days(
                target_year,
                month,
                day.min(days_in_month(target_year, month)),
            ))
        },
        _ => None,
    }
}

fn parse_last_weekday_marker(body: &str) -> Option<i32> {
    let weekday =
        Regex::new(r"(?i)\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b")
            .unwrap()
            .captures(body)?
            .get(1)?
            .as_str()
            .to_ascii_lowercase();
    Some(match weekday.as_str() {
        "monday" => 1,
        "tuesday" => 2,
        "wednesday" => 3,
        "thursday" => 4,
        "friday" => 5,
        "saturday" => 6,
        "sunday" => 0,
        _ => return None,
    })
}

fn parse_relative_unit_ago(body: &str) -> Option<(i32, String)> {
    let lower = body.to_ascii_lowercase();
    if lower.contains("a couple of days ago") {
        return Some((2, "day".to_string()));
    }
    if lower.contains("a few days ago") {
        return Some((3, "day".to_string()));
    }
    let captures = Regex::new(
        r"(?i)\b(a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+(day|days|week|weeks|month|months|year|years)\s+ago\b",
    )
    .unwrap()
    .captures(&lower)?;
    let amount = match captures.get(1)?.as_str() {
        "a" | "an" | "one" => 1,
        "two" => 2,
        "three" => 3,
        "four" => 4,
        "five" => 5,
        "six" => 6,
        "seven" => 7,
        "eight" => 8,
        "nine" => 9,
        "ten" => 10,
        "eleven" => 11,
        "twelve" => 12,
        value => value.parse::<i32>().ok()?,
    };
    let unit = captures.get(2)?.as_str().trim_end_matches('s');
    Some((amount, unit.to_string()))
}

fn relative_recall_focus_terms(body: &str) -> Vec<String> {
    let mut terms: Vec<String> = synthetic_query_terms(body)
        .into_iter()
        .map(|term| match term.as_str() {
            "buisiness" => "business".to_string(),
            _ => term,
        })
        .filter(|term| {
            !matches!(
                term.as_str(),
                "as" | "of"
                    | "what"
                    | "which"
                    | "who"
                    | "whom"
                    | "where"
                    | "when"
                    | "something"
                    | "mentioned"
                    | "significant"
                    | "milestone"
                    | "did"
                    | "do"
                    | "does"
                    | "was"
                    | "were"
                    | "is"
                    | "i"
                    | "me"
                    | "my"
                    | "ago"
                    | "last"
                    | "day"
                    | "days"
                    | "week"
                    | "weeks"
                    | "month"
                    | "months"
                    | "year"
                    | "years"
                    | "couple"
                    | "few"
                    | "one"
                    | "two"
                    | "three"
                    | "four"
                    | "five"
                    | "six"
                    | "seven"
                    | "eight"
                    | "nine"
                    | "ten"
                    | "eleven"
                    | "twelve"
                    | "monday"
                    | "tuesday"
                    | "wednesday"
                    | "thursday"
                    | "friday"
                    | "saturday"
                    | "sunday"
            )
        })
        .collect();
    if terms.iter().any(|term| term == "business") {
        terms.extend(
            ["client", "contract", "freelance"]
                .into_iter()
                .map(str::to_string),
        );
    }
    terms.sort();
    terms.dedup();
    terms
}

fn relative_recall_answer_kind(body: &str) -> RelativeTemporalRecallAnswerKind {
    let lower = body.to_ascii_lowercase();
    if lower.contains("from whom") {
        RelativeTemporalRecallAnswerKind::SourcePerson
    } else if lower.starts_with("which book") || lower.starts_with("what book") {
        RelativeTemporalRecallAnswerKind::BookTitle
    } else if lower.contains("what was it") {
        RelativeTemporalRecallAnswerKind::DirectObject
    } else {
        RelativeTemporalRecallAnswerKind::EventClause
    }
}

fn shift_months(year: i32, month: u32, delta: i32) -> Option<(i32, u32)> {
    let month_index = year
        .checked_mul(12)?
        .checked_add(month as i32 - 1)?
        .checked_add(delta)?;
    let target_year = month_index.div_euclid(12);
    let target_month = month_index.rem_euclid(12) as u32 + 1;
    Some((target_year, target_month))
}

fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 => 29,
        2 => 28,
        _ => 30,
    }
}

fn weekday_number(year: i32, month: u32, day: u32) -> i32 {
    (ymd_to_days(year, month, day) + 4).rem_euclid(7)
}
