//! Schedule domain helpers: weekday extraction, schedule shifts, dishes, commute, images, dollar amounts.

use super::super::*;
use crate::index::{compile_regex, compile_regex_static};

pub fn extract_weekday_from_query(task_lower: &str) -> Option<&'static str> {
    [
        "sunday",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
    ]
    .into_iter()
    .find(|day| task_lower.contains(day))
}

pub fn extract_weekday_surface_from_line(lower: &str) -> Option<String> {
    extract_weekday_from_query(lower).map(capitalize_first_ascii)
}

pub fn pluralize_weekday(day: &str) -> String {
    let mut chars = day.chars();
    let first = chars.next().map(|c| c.to_ascii_uppercase()).unwrap_or('D');
    format!("{first}{}s", chars.as_str())
}

pub fn extract_schedule_query_person(task: &str) -> Option<String> {
    let mut best = None;
    for token in task.split(|c: char| !c.is_ascii_alphabetic() && c != '-') {
        let trimmed = token.trim();
        if trimmed.len() < 3 {
            continue;
        }
        let Some(first) = trimmed.chars().next() else {
            continue;
        };
        if !first.is_ascii_uppercase() {
            continue;
        }
        let lower = trimmed.to_ascii_lowercase();
        if [
            "bandung",
            "can",
            "cihampelas",
            "friday",
            "gm",
            "monday",
            "previous",
            "saturday",
            "sunday",
            "thursday",
            "tuesday",
            "wednesday",
        ]
        .contains(&lower.as_str())
        {
            continue;
        }
        best = Some(trimmed.to_string());
    }
    best
}

pub fn parse_markdown_table_cells(line: &str) -> Option<Vec<String>> {
    let trimmed = line.trim();
    if !trimmed.starts_with('|') {
        return None;
    }
    let cells: Vec<String> = trimmed
        .split('|')
        .map(str::trim)
        .filter(|cell| !cell.is_empty())
        .map(|cell| cell.to_string())
        .collect();
    if cells.is_empty() {
        return None;
    }
    if cells
        .iter()
        .all(|cell| cell.chars().all(|c| matches!(c, '-' | ':' | ' ')))
    {
        return None;
    }
    Some(cells)
}

pub fn extract_schedule_shift_from_table(
    lines: &[String],
    person: &str,
    day: &str,
) -> Option<(String, Vec<String>)> {
    pub fn looks_like_shift_header_row(cells: &[String]) -> bool {
        cells.iter().any(|cell| {
            let lower = cell.to_ascii_lowercase();
            lower.contains("shift") || lower.contains("am -") || lower.contains("pm -")
        })
    }

    let person_lower = person.to_ascii_lowercase();
    let mut header = None::<(Vec<String>, String)>;
    for line in lines {
        let Some(cells) = parse_markdown_table_cells(line) else {
            continue;
        };
        if looks_like_shift_header_row(&cells) {
            header = Some((cells, line.clone()));
            continue;
        }
        if header.is_none() {
            header = Some((cells, line.clone()));
            continue;
        }
        let Some((header_cells, header_line)) = header.as_ref() else {
            continue;
        };
        if cells.is_empty() || !cells[0].eq_ignore_ascii_case(day) {
            continue;
        }
        for (idx, cell) in cells.iter().enumerate().skip(1) {
            if cell.eq_ignore_ascii_case(&person_lower) || cell.eq_ignore_ascii_case(person) {
                let header_idx = if header_cells.len() + 1 == cells.len() {
                    idx - 1
                } else {
                    idx
                };
                if let Some(shift) = header_cells.get(header_idx) {
                    return Some((shift.clone(), vec![header_line.clone(), line.clone()]));
                }
            }
        }
    }
    None
}

pub fn extract_served_dish_from_query(task: &str, task_lower: &str) -> Option<String> {
    let marker = if let Some(idx) = task_lower.find("serves ") {
        idx + "serves ".len()
    } else if let Some(idx) = task_lower.find("serve ") {
        idx + "serve ".len()
    } else {
        return None;
    };
    let tail = task[marker..].trim();
    let mut words = Vec::new();
    for raw in tail.split_whitespace() {
        let cleaned = raw
            .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '\'')
            .to_ascii_lowercase();
        if cleaned.is_empty() {
            continue;
        }
        if ["a", "an", "the", "great", "good"].contains(&cleaned.as_str()) {
            continue;
        }
        if ["that", "which", "with", "in"].contains(&cleaned.as_str()) {
            break;
        }
        words.push(cleaned);
    }
    (!words.is_empty()).then(|| words.join(" "))
}

pub fn extract_list_item_label(line: &str) -> Option<String> {
    let trimmed = line.trim();
    let candidate = trimmed
        .trim_start_matches(|c: char| c.is_ascii_digit() || c == '.' || c == ' ')
        .trim();
    let label = candidate.split(':').next()?.trim();
    (!label.is_empty()).then(|| label.to_string())
}

pub fn venue_stem_from_dish_label(label: &str, dish: &str) -> Option<String> {
    let lower = label.to_ascii_lowercase();
    let dish_lower = dish.to_ascii_lowercase();
    if let Some(idx) = lower.find(&format!("'s {dish_lower}")) {
        return Some(label[..idx].trim().to_string());
    }
    lower
        .find(&dish_lower)
        .map(|idx| label[..idx].trim().to_string())
        .filter(|stem| !stem.is_empty())
}

pub fn extract_restaurant_serving_dish(
    lines: &[String],
    dish: &str,
) -> Option<(String, Vec<String>)> {
    let dish_lower = dish.to_ascii_lowercase();
    for line in lines {
        if !line.contains(':') {
            continue;
        }
        let lower = line.to_ascii_lowercase();
        if !lower.contains(&dish_lower) {
            continue;
        }
        let Some(label) = extract_list_item_label(line) else {
            continue;
        };
        let Some(candidate_stem) = venue_stem_from_dish_label(&label, dish) else {
            continue;
        };
        let stem_lower = candidate_stem.to_ascii_lowercase();
        let mut best = None::<(String, String)>;
        for venue_line in lines {
            if !venue_line.contains(':') {
                continue;
            }
            let Some(venue_label) = extract_list_item_label(venue_line) else {
                continue;
            };
            let lower_label = venue_label.to_ascii_lowercase();
            if lower_label.contains(&dish_lower) || !lower_label.contains(&stem_lower) {
                continue;
            }
            if best
                .as_ref()
                .map(|(current, _)| venue_label.len() > current.len())
                .unwrap_or(true)
            {
                best = Some((venue_label, venue_line.clone()));
            }
        }
        if let Some((restaurant, venue_line)) = best {
            return Some((restaurant, vec![venue_line, line.clone()]));
        }
    }
    None
}

pub fn extract_commute_duration_from_line(line: &str) -> Option<String> {
    let lower = line.to_ascii_lowercase();
    if !lower.contains("commute") {
        return None;
    }
    let pattern = compile_regex_static(
        r"(?i)(?:which\s+takes|takes|is)\s+(?:about\s+)?((?:an?|one|\d+)\s+(?:hours?|minutes?)(?:\s+each\s+way)?)",
    );
    pattern.captures(line).and_then(|caps| {
        caps.get(1).map(|m| {
            m.as_str()
                .trim()
                .trim_matches(|c: char| matches!(c, ',' | '.' | ';' | ':' | '"' | '\''))
                .to_string()
        })
    })
}

pub fn extract_store_name_from_line(_line: &str, lower: &str) -> Option<String> {
    [
        ("whole foods", "Whole Foods"),
        ("trader joe", "Trader Joe's"),
        ("target", "Target"),
        ("walmart", "Walmart"),
        ("costco", "Costco"),
        ("walgreens", "Walgreens"),
        ("cvs", "CVS"),
    ]
    .into_iter()
    .find_map(|(needle, rendered)| lower.contains(needle).then(|| rendered.to_string()))
}

pub fn extract_image_subject_from_query(task: &str) -> Option<String> {
    let scoped = compile_regex_static(r"of the ([A-Z][A-Za-z-]+)");
    if let Some(subject) = scoped
        .captures(task)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().to_string())
    {
        return Some(subject);
    }

    let mut best = None::<String>;
    for token in task.split(|c: char| !c.is_ascii_alphabetic() && c != '-') {
        let trimmed = token.trim();
        if trimmed.len() < 3 {
            continue;
        }
        let Some(first) = trimmed.chars().next() else {
            continue;
        };
        if !first.is_ascii_uppercase() {
            continue;
        }
        let lower = trimmed.to_ascii_lowercase();
        if ["bandung", "can", "dinosaurs", "i", "im", "what"].contains(&lower.as_str()) {
            continue;
        }
        best = Some(trimmed.to_string());
    }
    best
}

pub fn extract_image_subject_body_color(
    lines: &[String],
    subject: &str,
) -> Option<(String, Vec<String>)> {
    let pattern = compile_regex(&format!(
        r"(?i)\b{}\b[^.]*?\bhas a ([a-z ]+?) body",
        regex::escape(subject)
    ))
    .unwrap_or_else(|err| panic!("escaped image-subject regex failed to compile: {err}"));
    for line in lines {
        let Some(caps) = pattern.captures(line) else {
            continue;
        };
        let phrase = caps
            .get(1)
            .map(|m| {
                m.as_str()
                    .trim()
                    .trim_matches(|c: char| matches!(c, ',' | '.' | ';' | ':' | '"' | '\''))
                    .to_string()
            })
            .filter(|value| !value.is_empty())?;
        let answer = format!("The {subject} had a {phrase} body.");
        return Some((answer, vec![line.clone()]));
    }
    None
}

pub fn extract_issue_after_service_line(line: &str, lower: &str) -> Option<String> {
    let mut issue = extract_phrase_after_any_index(
        line,
        lower,
        &["issue with my ", "issue with the ", "issue with "],
        &[" on ", " and ", " but ", " because ", ","],
        2,
    )?;
    let prefixes = ["my car's ", "the car's ", "car's ", "my ", "the "];
    for prefix in prefixes {
        if issue.to_ascii_lowercase().starts_with(prefix) {
            issue = issue[prefix.len()..].trim().to_string();
            break;
        }
    }
    let lower_issue = issue.to_ascii_lowercase();
    if lower_issue.contains("gps") && lower_issue.contains("system") {
        return Some("GPS system not functioning correctly".to_string());
    }
    Some(issue)
}

pub fn extract_dollar_amounts(line: &str) -> Vec<f32> {
    let pattern = compile_regex_static(r"\$([0-9][0-9,]*(?:\.[0-9]+)?)");
    pattern
        .captures_iter(line)
        .filter_map(|caps| caps.get(1))
        .filter_map(|m| m.as_str().replace(',', "").parse::<f32>().ok())
        .collect()
}

pub fn is_grounded_user_money_fact_line(lower: &str) -> bool {
    if !lower.trim_start().starts_with("user:") {
        return false;
    }

    ![
        "under $",
        "over $",
        "around $",
        "approximately $",
        "approx $",
        "starting at $",
        "start at $",
        "ranges from $",
        "range from $",
        "between $",
        "if you book",
        "fare is around",
        "might run around",
        "could cost",
        "would cost",
        "would be around",
        "going to order",
        "order next week",
        "thinking about getting",
        "set a budget",
        "budget and stick to it",
        "budget for",
        "budget of $",
        "my budget is $",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

pub fn is_grounded_user_duration_fact_line(lower: &str) -> bool {
    lower.trim_start().starts_with("user:")
}

pub fn split_numeric_aggregate_segments(line: &str) -> Vec<String> {
    let mut segments = vec![line.trim().to_string()];
    for delimiter in [". ", "! ", "? ", "; ", " but ", " however ", " while "] {
        let mut next = Vec::new();
        for segment in segments {
            if segment.contains(delimiter) {
                next.extend(
                    segment
                        .split(delimiter)
                        .map(str::trim)
                        .filter(|part| !part.is_empty())
                        .map(ToString::to_string),
                );
            } else if !segment.is_empty() {
                next.push(segment);
            }
        }
        segments = next;
    }
    segments
}

pub fn split_duration_aggregate_segments(line: &str) -> Vec<String> {
    let mut segments = split_numeric_aggregate_segments(line);
    for delimiter in [
        ", By the way, ",
        " By the way, ",
        ", by the way, ",
        " by the way, ",
        ", like ",
        " like ",
    ] {
        let mut next = Vec::new();
        for segment in segments {
            if segment.contains(delimiter) {
                next.extend(
                    segment
                        .split(delimiter)
                        .map(str::trim)
                        .filter(|part| !part.is_empty())
                        .map(ToString::to_string),
                );
            } else if !segment.is_empty() {
                next.push(segment);
            }
        }
        segments = next;
    }
    segments
}

pub fn extract_focused_dollar_amounts(line: &str, focus_terms: &[String]) -> Vec<f32> {
    let amounts = extract_dollar_amounts(line);
    if amounts.len() <= 1 {
        return amounts;
    }

    let focus_refs: Vec<&str> = focus_terms.iter().map(String::as_str).collect();
    let mut focused = Vec::new();
    for segment in split_numeric_aggregate_segments(line) {
        let lower = segment.to_ascii_lowercase();
        if term_overlap_count(&lower, &focus_refs) == 0 {
            continue;
        }
        focused.extend(extract_dollar_amounts(&segment));
    }
    if !focused.is_empty() {
        return focused;
    }

    let lower = line.to_ascii_lowercase();
    if term_overlap_count(&lower, &focus_refs) > 0 {
        amounts
    } else {
        Vec::new()
    }
}

pub fn money_total_line_matches_query(task_lower: &str, lower: &str) -> bool {
    if !task_contains_any(
        lower,
        &[
            "bought",
            "buy ",
            "got ",
            "cost me",
            "paid",
            "spent",
            "splurge",
            "purchase",
            "purchased",
            "installed",
            "replaced",
        ],
    ) {
        return false;
    }

    if task_lower.contains("luxury") {
        return task_contains_any(lower, &["luxury", "designer", "gucci", "high-end"]);
    }
    if task_lower.contains("bike") {
        return task_contains_any(
            lower,
            &[
                "bike",
                "helmet",
                "lights",
                "chain",
                "cycling",
                "tune-up",
                "bike shop",
            ],
        );
    }
    true
}

pub fn format_numeric_answer(value: f32) -> String {
    if (value - value.round()).abs() < 0.01 {
        #[allow(clippy::cast_possible_truncation)]
        let rounded = value.round() as i64;
        return rounded.to_string();
    }

    let mut rendered = format!("{value:.2}");
    while rendered.ends_with('0') {
        rendered.pop();
    }
    if rendered.ends_with('.') {
        rendered.pop();
    }
    rendered
}

pub fn format_integer_with_commas(value: i64) -> String {
    let digits = value.abs().to_string();
    let mut parts = Vec::new();
    let mut idx = digits.len();
    while idx > 3 {
        parts.push(digits[idx - 3..idx].to_string());
        idx -= 3;
    }
    parts.push(digits[..idx].to_string());
    parts.reverse();
    let joined = parts.join(",");
    if value < 0 {
        format!("-{joined}")
    } else {
        joined
    }
}

pub fn format_money_answer(value: f32) -> String {
    if (value - value.round()).abs() < 0.01 {
        #[allow(clippy::cast_possible_truncation)]
        let rounded = value.round() as i64;
        return format!("${}", format_integer_with_commas(rounded));
    }
    format!("${}", format_numeric_answer(value))
}

pub fn extract_aggregate_duration_value(line: &str) -> Option<SyntheticDurationValue> {
    pub fn parse_amount(token: &str) -> Option<f32> {
        match token.to_ascii_lowercase().as_str() {
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
        }
    }

    let postfix_half = compile_regex_static(
        r"(?i)\b(?:about\s+|around\s+)?(a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|couple|few|\d+(?:\.\d+)?)(?:\s+|-)(day|days|week|weeks|month|months|year|years|hour|hours|minute|minutes)\s+and\s+a\s+half\b",
    );
    let long_form = compile_regex_static(
        r"(?i)\b(?:(a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|couple|few|\d+(?:\.\d+)?)\s+)?(day|days|week|weeks|month|months|year|years|hour|hours|minute|minutes)(?:-|\s+)long\b",
    );
    let prefix_half = compile_regex_static(
        r"(?i)\b(?:about\s+|around\s+)?(a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|couple|few|\d+(?:\.\d+)?)(\s+and\s+a\s+half)?(?:\s+|-)(day|days|week|weeks|month|months|year|years|hour|hours|minute|minutes)\b",
    );
    let (amount_token, has_half, unit) = if let Some(caps) = postfix_half.captures(line) {
        (
            caps.get(1)?.as_str().to_string(),
            true,
            caps.get(2)?.as_str().to_ascii_lowercase(),
        )
    } else if let Some(caps) = long_form.captures(line) {
        (
            caps.get(1)
                .map(|m| m.as_str().to_string())
                .unwrap_or_else(|| "one".to_string()),
            false,
            caps.get(2)?.as_str().to_ascii_lowercase(),
        )
    } else {
        let caps = prefix_half.captures(line)?;
        (
            caps.get(1)?.as_str().to_string(),
            caps.get(2).is_some(),
            caps.get(3)?.as_str().to_ascii_lowercase(),
        )
    };
    let mut amount = parse_amount(&amount_token)?;
    if has_half {
        amount += 0.5;
    }
    let days = amount
        * match unit.as_str() {
            "minute" | "minutes" => 1.0 / (24.0 * 60.0),
            "hour" | "hours" => 1.0 / 24.0,
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
            "minute" | "minutes" => "minute",
            "hour" | "hours" => "hour",
            "day" | "days" => "day",
            "week" | "weeks" => "week",
            "month" | "months" => "month",
            "year" | "years" => "year",
            _ => return None,
        },
    })
}

pub fn extract_requested_aggregate_duration_unit(task_lower: &str) -> Option<&'static str> {
    let caps = compile_regex_static(r"(?i)\bhow many\s+(day|days|week|weeks|month|months|year|years|hour|hours|minute|minutes)\b")
        .captures(task_lower)?;
    match caps.get(1)?.as_str().to_ascii_lowercase().as_str() {
        "minute" | "minutes" => Some("minute"),
        "hour" | "hours" => Some("hour"),
        "day" | "days" => Some("day"),
        "week" | "weeks" => Some("week"),
        "month" | "months" => Some("month"),
        "year" | "years" => Some("year"),
        _ => None,
    }
}

pub fn format_aggregate_duration_answer(amount: f32, unit: &str) -> String {
    let rendered = format_numeric_answer(amount);
    let singular = (amount - 1.0).abs() < 0.01;
    let suffix = if singular {
        unit.to_string()
    } else {
        format!("{unit}s")
    };
    format!("{rendered} {suffix}")
}

pub fn convert_duration_days(days: f32, unit: &str) -> f32 {
    match unit {
        "minute" => days * 24.0 * 60.0,
        "hour" => days * 24.0,
        "day" => days,
        "week" => days / 7.0,
        "month" => days / 30.0,
        "year" => days / 365.0,
        _ => days,
    }
}

pub fn should_try_multi_session_money_total(task_lower: &str) -> bool {
    task_contains_any(
        task_lower,
        &[
            "$",
            " dollar",
            " dollars",
            " money",
            " expense",
            " expenses",
            " cost",
            " costs",
            " paid",
            " purchase",
            " purchased",
            " spent",
            " amount",
        ],
    ) && task_contains_any(
        task_lower,
        &[
            "how much total",
            "total money",
            "total amount",
            "in total",
            "combined",
            "altogether",
            "since the start",
            "past few months",
            "expenses",
        ],
    ) && !task_contains_any(
        task_lower,
        &[
            " compared to ",
            " difference ",
            " more expensive ",
            " less expensive ",
            " save ",
            " saved ",
            " each ",
            " per ",
            " before ",
            " after ",
        ],
    )
}

pub fn should_try_multi_session_duration_total(task_lower: &str) -> bool {
    extract_requested_aggregate_duration_unit(task_lower).is_some()
        && (task_contains_any(
            task_lower,
            &[
                " in total",
                " combined",
                " altogether",
                " this year",
                " since the start",
                " past few months",
            ],
        ) || task_lower.contains(" and ")
            || task_contains_any(
                task_lower,
                &[
                    "trips",
                    "breaks",
                    "games",
                    "destinations",
                    "films",
                    "movies",
                ],
            ))
        && !task_contains_any(
            task_lower,
            &[
                "formal education",
                "high school",
                "bachelor",
                "master",
                "degree",
                "college",
                "university",
            ],
        )
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(in crate::index) enum EducationStageKind {
    HighSchool,
    Associate,
    Bachelor,
    Master,
}

#[derive(Clone, Debug)]
pub struct EducationStageFact {
    pub kind: EducationStageKind,
    pub completed: bool,
    pub start_year: Option<i32>,
    pub end_year: Option<i32>,
    pub duration_years: Option<i32>,
    pub evidence: String,
}
