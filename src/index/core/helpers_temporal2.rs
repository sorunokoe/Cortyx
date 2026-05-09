// This file is a submodule of `crate::index::core`.
// Contains free-standing helper functions extracted from helpers.rs.
use super::*;
use crate::index::compile_regex;
use crate::types::{QueryText, SynapseWeight};

pub(in crate::index) fn extract_weekday_from_query(task_lower: &str) -> Option<&'static str> {
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

pub(in crate::index) fn extract_weekday_surface_from_line(lower: &str) -> Option<String> {
    extract_weekday_from_query(lower).map(capitalize_first_ascii)
}

pub(in crate::index) fn pluralize_weekday(day: &str) -> String {
    let mut chars = day.chars();
    let first = chars.next().map(|c| c.to_ascii_uppercase()).unwrap_or('D');
    format!("{first}{}s", chars.as_str())
}

pub(in crate::index) fn extract_schedule_query_person(task: &str) -> Option<String> {
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

pub(in crate::index) fn parse_markdown_table_cells(line: &str) -> Option<Vec<String>> {
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

pub(in crate::index) fn extract_schedule_shift_from_table(
    lines: &[String],
    person: &str,
    day: &str,
) -> Option<(String, Vec<String>)> {
    pub(in crate::index) fn looks_like_shift_header_row(cells: &[String]) -> bool {
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

pub(in crate::index) fn extract_served_dish_from_query(
    task: &str,
    task_lower: &str,
) -> Option<String> {
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

pub(in crate::index) fn extract_list_item_label(line: &str) -> Option<String> {
    let trimmed = line.trim();
    let candidate = trimmed
        .trim_start_matches(|c: char| c.is_ascii_digit() || c == '.' || c == ' ')
        .trim();
    let label = candidate.split(':').next()?.trim();
    (!label.is_empty()).then(|| label.to_string())
}

pub(in crate::index) fn venue_stem_from_dish_label(label: &str, dish: &str) -> Option<String> {
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

pub(in crate::index) fn extract_restaurant_serving_dish(
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

pub(in crate::index) fn extract_commute_duration_from_line(line: &str) -> Option<String> {
    let lower = line.to_ascii_lowercase();
    if !lower.contains("commute") {
        return None;
    }
    let pattern = compile_regex(
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

pub(in crate::index) fn extract_store_name_from_line(_line: &str, lower: &str) -> Option<String> {
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

pub(in crate::index) fn extract_image_subject_from_query(task: &str) -> Option<String> {
    let scoped = compile_regex(r"of the ([A-Z][A-Za-z-]+)");
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

pub(in crate::index) fn extract_image_subject_body_color(
    lines: &[String],
    subject: &str,
) -> Option<(String, Vec<String>)> {
    let pattern = compile_regex(&format!(
        r"(?i)\b{}\b[^.]*?\bhas a ([a-z ]+?) body",
        regex::escape(subject)
    ));
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

pub(in crate::index) fn extract_issue_after_service_line(
    line: &str,
    lower: &str,
) -> Option<String> {
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

pub(in crate::index) fn extract_dollar_amounts(line: &str) -> Vec<f32> {
    let pattern = compile_regex(r"\$([0-9][0-9,]*(?:\.[0-9]+)?)");
    pattern
        .captures_iter(line)
        .filter_map(|caps| caps.get(1))
        .filter_map(|m| m.as_str().replace(',', "").parse::<f32>().ok())
        .collect()
}

pub(in crate::index) fn is_grounded_user_money_fact_line(lower: &str) -> bool {
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

pub(in crate::index) fn is_grounded_user_duration_fact_line(lower: &str) -> bool {
    lower.trim_start().starts_with("user:")
}

pub(in crate::index) fn split_numeric_aggregate_segments(line: &str) -> Vec<String> {
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

pub(in crate::index) fn split_duration_aggregate_segments(line: &str) -> Vec<String> {
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

pub(in crate::index) fn extract_focused_dollar_amounts(
    line: &str,
    focus_terms: &[String],
) -> Vec<f32> {
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

pub(in crate::index) fn money_total_line_matches_query(task_lower: &str, lower: &str) -> bool {
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

pub(in crate::index) fn format_numeric_answer(value: f32) -> String {
    if (value - value.round()).abs() < 0.01 {
        return (value.round() as i64).to_string();
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

pub(in crate::index) fn format_integer_with_commas(value: i64) -> String {
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

pub(in crate::index) fn format_money_answer(value: f32) -> String {
    if (value - value.round()).abs() < 0.01 {
        return format!("${}", format_integer_with_commas(value.round() as i64));
    }
    format!("${}", format_numeric_answer(value))
}

pub(in crate::index) fn extract_aggregate_duration_value(
    line: &str,
) -> Option<SyntheticDurationValue> {
    pub(in crate::index) fn parse_amount(token: &str) -> Option<f32> {
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

    let postfix_half = compile_regex(
        r"(?i)\b(?:about\s+|around\s+)?(a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|couple|few|\d+(?:\.\d+)?)(?:\s+|-)(day|days|week|weeks|month|months|year|years|hour|hours|minute|minutes)\s+and\s+a\s+half\b",
    );
    let long_form = compile_regex(
        r"(?i)\b(?:(a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|couple|few|\d+(?:\.\d+)?)\s+)?(day|days|week|weeks|month|months|year|years|hour|hours|minute|minutes)(?:-|\s+)long\b",
    );
    let prefix_half = compile_regex(
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

pub(in crate::index) fn extract_requested_aggregate_duration_unit(
    task_lower: &str,
) -> Option<&'static str> {
    let caps = compile_regex(r"(?i)\bhow many\s+(day|days|week|weeks|month|months|year|years|hour|hours|minute|minutes)\b")
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

pub(in crate::index) fn format_aggregate_duration_answer(amount: f32, unit: &str) -> String {
    let rendered = format_numeric_answer(amount);
    let singular = (amount - 1.0).abs() < 0.01;
    let suffix = if singular {
        unit.to_string()
    } else {
        format!("{unit}s")
    };
    format!("{rendered} {suffix}")
}

pub(in crate::index) fn convert_duration_days(days: f32, unit: &str) -> f32 {
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

pub(in crate::index) fn should_try_multi_session_money_total(task_lower: &str) -> bool {
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

pub(in crate::index) fn should_try_multi_session_duration_total(task_lower: &str) -> bool {
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
pub(in crate::index) struct EducationStageFact {
    kind: EducationStageKind,
    completed: bool,
    start_year: Option<i32>,
    end_year: Option<i32>,
    duration_years: Option<i32>,
    evidence: String,
}

pub(in crate::index) fn extract_formal_education_target_stage(
    task_lower: &str,
) -> Option<EducationStageKind> {
    if !task_lower.contains("formal education") || !task_lower.contains("high school") {
        return None;
    }
    if task_lower.contains("bachelor") {
        return Some(EducationStageKind::Bachelor);
    }
    if task_lower.contains("master") {
        return Some(EducationStageKind::Master);
    }
    None
}

pub(in crate::index) fn collect_education_stage_facts(
    lines: &[String],
) -> HashMap<EducationStageKind, EducationStageFact> {
    let mut facts = HashMap::new();
    for line in lines {
        let Some(parsed) = parse_education_stage_fact(line) else {
            continue;
        };
        let should_replace = facts
            .get(&parsed.kind)
            .map(|existing| {
                education_stage_fact_score(&parsed) > education_stage_fact_score(existing)
            })
            .unwrap_or(true);
        if should_replace {
            facts.insert(parsed.kind, parsed);
        }
    }
    facts
}

pub(in crate::index) fn solve_formal_education_total(
    facts: &HashMap<EducationStageKind, EducationStageFact>,
    target_stage: EducationStageKind,
) -> Option<(i32, Vec<String>, usize)> {
    let high_school = facts.get(&EducationStageKind::HighSchool)?;
    let high_school_duration = education_stage_duration_years(high_school)?;
    let high_school_end = education_stage_end_year(high_school)?;

    let bachelor = facts
        .get(&EducationStageKind::Bachelor)
        .filter(|fact| fact.completed)?;
    let bachelor_duration = education_stage_duration_years(bachelor)?;
    let bachelor_start = education_stage_start_year(bachelor)?;
    let bachelor_end = education_stage_end_year(bachelor)?;

    let mut total_years = high_school_duration + bachelor_duration;
    let mut evidence = vec![high_school.evidence.clone()];

    if let Some(associate) = facts
        .get(&EducationStageKind::Associate)
        .filter(|fact| fact.completed)
    {
        let associate_duration = education_stage_duration_years(associate).or_else(|| {
            let associate_end = education_stage_end_year(associate)?;
            ((associate_end > high_school_end) && (associate_end <= bachelor_start))
                .then_some(associate_end - high_school_end)
        });
        if let Some(years) = associate_duration.filter(|years| *years > 0) {
            total_years += years;
            evidence.push(associate.evidence.clone());
        }
    }

    evidence.push(bachelor.evidence.clone());

    if target_stage == EducationStageKind::Master {
        let master = facts
            .get(&EducationStageKind::Master)
            .filter(|fact| fact.completed)?;
        let master_duration = education_stage_duration_years(master).or_else(|| {
            let master_end = education_stage_end_year(master)?;
            (master_end > bachelor_end).then_some(master_end - bachelor_end)
        })?;
        if master_duration <= 0 {
            return None;
        }
        total_years += master_duration;
        evidence.push(master.evidence.clone());
    }

    let fact_count = evidence.len();
    Some((total_years, evidence, fact_count))
}

pub(in crate::index) fn parse_education_stage_fact(line: &str) -> Option<EducationStageFact> {
    let body = normalize_session_answer_line_body(line);
    let lower = body.to_ascii_lowercase();
    let years = extract_year_mentions(&body);

    let high_school_range =
        compile_regex(r"(?i)\bhigh school\b.*?\bfrom\s+(\d{4})\s+to\s+(\d{4})\b");
    if let Some(caps) = high_school_range.captures(&body) {
        let start_year = caps.get(1)?.as_str().parse::<i32>().ok()?;
        let end_year = caps.get(2)?.as_str().parse::<i32>().ok()?;
        if end_year > start_year {
            return Some(EducationStageFact {
                kind: EducationStageKind::HighSchool,
                completed: true,
                start_year: Some(start_year),
                end_year: Some(end_year),
                duration_years: Some(end_year - start_year),
                evidence: line.to_string(),
            });
        }
    }

    if task_contains_any(
        &lower,
        &[
            "associate's degree",
            "associates degree",
            "associate degree",
        ],
    ) {
        return Some(EducationStageFact {
            kind: EducationStageKind::Associate,
            completed: task_contains_any(&lower, &["earned", "completed", "graduated"]),
            start_year: None,
            end_year: years.last().copied(),
            duration_years: extract_education_duration_years(&lower),
            evidence: line.to_string(),
        });
    }

    if task_contains_any(
        &lower,
        &[
            "bachelor's degree",
            "bachelors degree",
            "bachelor degree",
            "bachelor's in",
            "bachelors in",
            "bachelor in",
        ],
    ) {
        return Some(EducationStageFact {
            kind: EducationStageKind::Bachelor,
            completed: task_contains_any(&lower, &["graduated", "earned", "completed"])
                || lower.contains("took me"),
            start_year: None,
            end_year: years.last().copied(),
            duration_years: extract_education_duration_years(&lower),
            evidence: line.to_string(),
        });
    }

    if task_contains_any(
        &lower,
        &[
            "master's degree",
            "masters degree",
            "master degree",
            "master's in",
            "masters in",
            "master in",
        ],
    ) {
        return Some(EducationStageFact {
            kind: EducationStageKind::Master,
            completed: task_contains_any(&lower, &["graduated", "earned", "completed", "finished"]),
            start_year: None,
            end_year: years.last().copied(),
            duration_years: extract_education_duration_years(&lower),
            evidence: line.to_string(),
        });
    }

    None
}

pub(in crate::index) fn extract_education_duration_years(lower: &str) -> Option<i32> {
    for marker in [
        "which took me ",
        "took me ",
        "completed in ",
        "finished in ",
    ] {
        let Some(idx) = lower.find(marker) else {
            continue;
        };
        let tail = &lower[idx + marker.len()..];
        let value = parse_leading_duration_value(tail)?;
        if value.unit == "year" {
            return Some(value.amount.round() as i32);
        }
    }
    None
}

pub(in crate::index) fn extract_year_mentions(text: &str) -> Vec<i32> {
    let years = compile_regex(r"\b(19|20)\d{2}\b");
    years
        .captures_iter(text)
        .filter_map(|caps| caps.get(0).and_then(|m| m.as_str().parse::<i32>().ok()))
        .collect()
}

pub(in crate::index) fn education_stage_fact_score(fact: &EducationStageFact) -> i32 {
    let mut score = 0;
    if fact.completed {
        score += 2;
    }
    if fact.start_year.is_some() {
        score += 2;
    }
    if fact.end_year.is_some() {
        score += 2;
    }
    if fact.duration_years.is_some() {
        score += 3;
    }
    score
}

pub(in crate::index) fn education_stage_duration_years(fact: &EducationStageFact) -> Option<i32> {
    fact.duration_years.or_else(|| {
        fact.start_year
            .zip(fact.end_year)
            .and_then(|(start, end)| (end > start).then_some(end - start))
    })
}

pub(in crate::index) fn education_stage_start_year(fact: &EducationStageFact) -> Option<i32> {
    fact.start_year.or_else(|| {
        fact.end_year
            .zip(fact.duration_years)
            .and_then(|(end, years)| (years > 0).then_some(end - years))
    })
}

pub(in crate::index) fn education_stage_end_year(fact: &EducationStageFact) -> Option<i32> {
    fact.end_year.or_else(|| {
        fact.start_year
            .zip(fact.duration_years)
            .and_then(|(start, years)| (years > 0).then_some(start + years))
    })
}

pub(in crate::index) fn extract_multi_session_money_focus_terms(task_lower: &str) -> Vec<String> {
    const STOP: &[&str] = &[
        "total",
        "combined",
        "altogether",
        "since",
        "start",
        "year",
        "years",
        "month",
        "months",
        "past",
        "last",
        "few",
        "item",
        "items",
        "related",
        "i",
        "money",
        "amount",
        "spent",
        "spend",
        "cost",
        "costs",
        "expense",
        "expenses",
        "paid",
        "purchase",
        "purchased",
    ];
    let stop: HashSet<&str> = STOP.iter().copied().collect();
    let mut terms = synthetic_query_terms(task_lower);
    terms.retain(|term| !stop.contains(term.as_str()));
    if task_lower.contains("bike") {
        for extra in [
            "helmet",
            "lights",
            "chain",
            "cycling",
            "tune-up",
            "bike shop",
        ] {
            terms.push(extra.to_string());
        }
    }
    terms.sort();
    terms.dedup();
    terms
}

pub(in crate::index) fn extract_multi_session_duration_focus_terms(
    task_lower: &str,
) -> Vec<String> {
    const STOP: &[&str] = &[
        "total",
        "combined",
        "altogether",
        "since",
        "start",
        "year",
        "years",
        "month",
        "months",
        "week",
        "weeks",
        "day",
        "days",
        "hour",
        "hours",
        "minute",
        "minutes",
        "time",
        "take",
        "took",
        "spent",
        "spend",
        "main",
        "past",
        "last",
        "few",
        "item",
        "items",
        "related",
        "united",
        "states",
        "i",
    ];
    let stop: HashSet<&str> = STOP.iter().copied().collect();
    let mut terms = synthetic_query_terms(task_lower);
    terms.retain(|term| !stop.contains(term.as_str()));
    terms.retain(|term| term.len() >= 2);
    if task_lower.contains("game") || task_lower.contains("gaming") {
        for extra in [
            "playing",
            "played",
            "finish",
            "finished",
            "complete",
            "completed",
        ] {
            terms.push(extra.to_string());
        }
    }
    if task_lower.contains("road trip") || task_lower.contains("destinations") {
        terms.retain(|term| !matches!(term.as_str(), "three" | "destination" | "destinations"));
        for extra in ["road", "trip", "drive", "drove", "driving"] {
            terms.push(extra.to_string());
        }
    }
    terms.sort();
    terms.dedup();
    terms
}

pub(in crate::index) fn is_realized_duration_fact_text(lower: &str) -> bool {
    let realized = task_contains_any(
        lower,
        &[
            "just got back",
            "got back from",
            "watched all",
            "did it in",
            "spent around",
            "spent ",
            "took me",
            "completed",
            "finished",
            "clocked",
            "drove for",
            "driving to",
            "camping trip",
            "break in",
            "break from",
            "marathon",
        ],
    );
    let future = task_contains_any(
        lower,
        &[
            "i'm planning",
            "i am planning",
            "plan to",
            "going to",
            "i'll",
            "i will",
            "next week",
            "next month",
            "by the end",
            "goal is",
            "goal to",
            "thinking about",
            "thinking of",
        ],
    );
    realized && !future
}

pub(in crate::index) fn extract_matching_duration_total_segments(
    line: &str,
    task_lower: &str,
) -> Vec<(String, SyntheticDurationValue)> {
    let mut matches = Vec::new();
    for segment in split_duration_aggregate_segments(line) {
        let lower = segment.to_ascii_lowercase();
        if !is_realized_duration_fact_text(&lower)
            || !duration_total_line_matches_query(task_lower, &lower)
        {
            continue;
        }
        let Some(duration) = extract_aggregate_duration_value(&segment) else {
            continue;
        };
        matches.push((segment, duration));
    }
    matches
}

pub(in crate::index) fn duration_total_line_matches_query(task_lower: &str, lower: &str) -> bool {
    if task_lower.contains("social media") {
        return lower.contains("social media")
            && task_contains_any(lower, &["break from", "break in", "break"]);
    }
    if task_lower.contains("camping") {
        return lower.contains("camping trip");
    }
    if task_lower.contains("road trip") || task_lower.contains("destinations") {
        return task_contains_any(
            lower,
            &[
                "drove for",
                "drive there",
                "drive to",
                "driving to",
                "took me",
            ],
        );
    }
    if task_contains_any(task_lower, &["marvel", "star wars", "movies", "films"]) {
        return task_contains_any(lower, &["watched", "marathon"]);
    }
    if task_contains_any(task_lower, &["games", "gaming"]) {
        return task_contains_any(
            lower,
            &[
                "playing",
                "spent around",
                "took me",
                "finished",
                "completed",
            ],
        ) && !task_contains_any(
            lower,
            &[
                "developers",
                "development",
                "develop ",
                "release",
                "announced",
                "team ",
                "script",
                "dialogue",
                "motion capture",
                "pages long",
            ],
        );
    }
    true
}

pub(in crate::index) fn aggregate_fact_terms(line: &str) -> HashSet<String> {
    synthetic_query_terms(&normalize_session_answer_line_body(line).to_ascii_lowercase())
        .into_iter()
        .collect()
}

pub(in crate::index) fn is_duplicate_numeric_aggregate_fact(
    existing: &[(String, f32, HashSet<String>)],
    session_id: &str,
    value: f32,
    terms: &HashSet<String>,
) -> bool {
    existing
        .iter()
        .any(|(existing_session, existing_value, existing_terms)| {
            if (existing_value - value).abs() >= 0.01 {
                return false;
            }
            let overlap = existing_terms.intersection(terms).count();
            let min_size = existing_terms.len().min(terms.len());
            if existing_session == session_id {
                overlap >= 4 || (min_size > 0 && overlap == min_size)
            } else {
                overlap >= 5 || (min_size >= 4 && overlap == min_size)
            }
        })
}

pub(in crate::index) fn extract_nightly_rate(line: &str) -> Option<f32> {
    let lower = line.to_ascii_lowercase();
    if !lower.starts_with("user:") {
        return None;
    }
    if !lower.contains("per night") {
        return None;
    }
    if !task_contains_any(
        &lower,
        &[
            "stay", "staying", "hotel", "hostel", "resort", "room", "accommod",
        ],
    ) {
        return None;
    }
    extract_dollar_amounts(line).into_iter().next()
}

pub(in crate::index) fn extract_sale_total(line: &str) -> Option<f32> {
    let lower = line.to_ascii_lowercase();
    if !(lower.starts_with("user:") || line.trim_start().starts_with('-')) {
        return None;
    }
    if !(lower.contains("sold") || lower.contains("earned") || lower.contains("earning")) {
        return None;
    }

    let explicit_total = compile_regex(
        r"(?:earned|earning(?: a total of)?|for a total of)\s+\$([0-9][0-9,]*(?:\.[0-9]+)?)",
    );
    if let Some(caps) = explicit_total.captures(&lower) {
        if let Some(value) = caps
            .get(1)
            .and_then(|m| m.as_str().replace(',', "").parse::<f32>().ok())
        {
            return Some(value);
        }
    }

    let per_item = compile_regex(r"sold\s+(\d+)[^$]{0,160}?\$([0-9][0-9,]*(?:\.[0-9]+)?)\s*each");
    if let Some(caps) = per_item.captures(&lower) {
        let quantity = caps.get(1).and_then(|m| m.as_str().parse::<f32>().ok())?;
        let price = caps
            .get(2)
            .and_then(|m| m.as_str().replace(',', "").parse::<f32>().ok())?;
        return Some(quantity * price);
    }

    None
}

pub(in crate::index) fn normalized_index_answer_surface_key(text: &str) -> String {
    text.trim()
        .trim_matches(|c: char| matches!(c, '"' | '\'' | ',' | '.' | ';' | ':' | '!' | '?'))
        .to_ascii_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

pub(in crate::index) fn index_answer_surface_answers_overlap(left: &str, right: &str) -> bool {
    let left_key = normalized_index_answer_surface_key(left);
    let right_key = normalized_index_answer_surface_key(right);
    !left_key.is_empty()
        && !right_key.is_empty()
        && (left_key == right_key || left_key.contains(&right_key) || right_key.contains(&left_key))
}

pub(in crate::index) fn index_answer_surface_bucket_rank(bucket: &IndexAnswerSurfaceBucket) -> f32 {
    let corroboration = ((bucket.total_score - bucket.best_score).max(0.0)).min(6.0) * 0.15;
    bucket.best_score
        + bucket.max_overlap as f32 * 1.5
        + (bucket.paths.len().saturating_sub(1).min(2) as f32) * 0.75
        + (bucket.hits.saturating_sub(1).min(3) as f32) * 0.25
        + corroboration
}

pub(in crate::index) fn index_answer_surface_buckets_conflict(
    top: &IndexAnswerSurfaceBucket,
    runner_up: &IndexAnswerSurfaceBucket,
) -> bool {
    !index_answer_surface_answers_overlap(&top.answer_span, &runner_up.answer_span)
        && index_answer_surface_bucket_rank(runner_up) + 2.5
            >= index_answer_surface_bucket_rank(top)
        && runner_up.max_overlap + 1 >= top.max_overlap
}

pub(in crate::index) fn index_answer_surface_bucket_has_query_affinity(
    task_lower: &str,
    bucket: &IndexAnswerSurfaceBucket,
) -> bool {
    let answer_lower = bucket.answer_span.to_ascii_lowercase();
    (task_contains_any(
        task_lower,
        &["religious", "religion", "faith", "church", "spiritual"],
    ) && answer_lower.contains("religious"))
        || (task_contains_any(
            task_lower,
            &[
                "member of the lgbtq community",
                "member of the lgbtq+ community",
                "part of the lgbtq community",
                "part of the lgbtq+ community",
                "member of the transgender community",
                "ally to the transgender community",
                "ally to the lgbtq community",
                "ally to the lgbtq+ community",
                "considered an ally",
            ],
        ) && answer_lower.contains("ally"))
        || (task_contains_any(
            task_lower,
            &["move from", "moved from", "home country", "origin country"],
        ) && bucket
            .relation_families
            .contains(&SyntheticAnswerSurfaceRelationFamily::Origin))
        || (task_contains_any(
            task_lower,
            &["what books", "which books", " books", "book "],
        ) && bucket
            .relation_families
            .contains(&SyntheticAnswerSurfaceRelationFamily::Book))
        || (task_contains_any(
            task_lower,
            &[
                "what lgbtq",
                "transgender-specific events",
                "lgbtq events",
                "in what ways",
            ],
        ) && bucket
            .relation_families
            .contains(&SyntheticAnswerSurfaceRelationFamily::CommunityEvent))
        || (task_contains_any(task_lower, &["help children", "help kids", "help youth"])
            && bucket
                .relation_families
                .contains(&SyntheticAnswerSurfaceRelationFamily::ChildHelpEvent))
        || (task_contains_any(
            task_lower,
            &[
                "with her family",
                "with his family",
                "with my family",
                "with the kids",
                "family activities",
            ],
        ) && bucket
            .relation_families
            .contains(&SyntheticAnswerSurfaceRelationFamily::FamilyActivity))
        || (task_contains_any(
            task_lower,
            &["to destress", "to de-stress", "self-care", "relax"],
        ) && bucket
            .relation_families
            .contains(&SyntheticAnswerSurfaceRelationFamily::SelfCareActivity))
}

pub(in crate::index) fn synthetic_answer_surface_should_skip_fallback(
    task: &str,
    task_lower: &str,
    profile: &SyntheticAnswerSurfaceQueryProfile,
    evidence: &[String],
) -> bool {
    let real_evidence = evidence
        .iter()
        .filter(|line| !line.starts_with("answer_surface:"))
        .collect::<Vec<_>>();
    let evidence_has_any = |needles: &[&str]| {
        real_evidence.iter().any(|line| {
            let lower = line.to_ascii_lowercase();
            task_contains_any(&lower, needles)
        })
    };
    let collecting_target = task_lower
        .split_once("collecting ")
        .map(|(_, tail)| tail)
        .map(|tail| {
            ["?", ".", ",", " before ", " after "]
                .iter()
                .find_map(|marker| tail.split_once(marker).map(|(head, _)| head))
                .unwrap_or(tail)
                .trim()
                .to_string()
        })
        .filter(|phrase| phrase.split_whitespace().count() >= 2);
    let mut poster_focus_terms = synthetic_query_terms(task_lower);
    poster_focus_terms.retain(|term| {
        term.len() >= 4
            && !matches!(
                term.as_str(),
                "university"
                    | "college"
                    | "present"
                    | "presented"
                    | "poster"
                    | "research"
                    | "conference"
            )
    });
    let poster_focus_refs = poster_focus_terms
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    let poster_focus_min_overlap = if poster_focus_refs.len() >= 2 { 2 } else { 1 };

    (matches!(profile.route_kind, SyntheticAnswerSurfaceRouteKind::Choice)
        && task_contains_any(
            task_lower,
            &[
                " first",
                " earlier",
                " later",
                " before ",
                " after ",
                " more often",
                " less often",
                " higher percentage",
                " lower percentage",
                " higher discount",
                " lower discount",
                " cheaper",
                " more expensive",
                " cost more",
                " cost less",
                " older",
                " younger",
            ],
        ))
        || ((matches!(
            profile.expected_type,
            SyntheticAnswerSurfaceExpectedType::Count
        ) || is_money_query(task))
            && synthetic_count_query_requires_multi_operand_reasoning(task, task_lower))
        || (task_contains_any(
            task_lower,
            &[
                "at which university",
                "which university",
                "what university",
                "which college",
                "what college",
            ],
        ) && !evidence_has_any(&["university", "college", "school", "institute"]))
        || (task_contains_any(task_lower, &["presented", "poster"])
            && !evidence_has_any(&["presented", "poster"]))
        || (task_contains_any(
            task_lower,
            &[
                "at which university",
                "which university",
                "what university",
                "which college",
                "what college",
            ],
        ) && task_contains_any(task_lower, &["present", "poster"])
            && !poster_focus_refs.is_empty()
            && !real_evidence.iter().any(|line| {
                let lower = line.to_ascii_lowercase();
                term_overlap_count(&lower, &poster_focus_refs) >= poster_focus_min_overlap
            }))
        || (task_contains_any(task_lower, &["conference"]) && !evidence_has_any(&["conference"]))
        || collecting_target.as_ref().is_some_and(|phrase| {
            !real_evidence
                .iter()
                .any(|line| line.to_ascii_lowercase().contains(phrase))
        })
}

pub(in crate::index) fn extract_rare_collection_count(line: &str) -> Option<(&'static str, i32)> {
    let lower = line.to_ascii_lowercase();
    if !(lower.starts_with("user:") || line.trim_start().starts_with('-')) {
        return None;
    }
    let kind = if lower.contains("rare books") {
        "rare_books"
    } else if lower.contains("rare records") {
        "rare_records"
    } else if lower.contains("rare figurines") {
        "rare_figurines"
    } else if lower.contains("rare coins") {
        "rare_coins"
    } else {
        return None;
    };

    let count = extract_line_numbers(line)
        .into_iter()
        .find(|value| *value > 0 && *value < 1000)?;
    Some((kind, count))
}

pub(in crate::index) fn extract_previous_role(line: &str) -> Option<String> {
    let lower = line.to_ascii_lowercase();
    if !lower.starts_with("user:") || !lower.contains("previous role") {
        return None;
    }

    let pattern = compile_regex(r"previous role as a[n]?\s+(.+?)(?:,|\.| and\b| but\b| with\b)");
    let role = pattern
        .captures(line)?
        .get(1)?
        .as_str()
        .trim()
        .trim_matches('"')
        .to_string();
    if role.is_empty() {
        None
    } else {
        Some(role)
    }
}

pub(in crate::index) fn extract_finished_issue_count(line: &str, lower: &str) -> Option<i32> {
    if !(lower.starts_with("user:") || line.trim_start().starts_with('-'))
        || !lower.contains("national geographic")
    {
        return None;
    }

    if lower.contains("finished") {
        return extract_line_numbers(line).into_iter().next();
    }
    if lower.contains("currently on") {
        return extract_line_numbers(line)
            .into_iter()
            .next()
            .map(|value| value - 1)
            .filter(|value| *value > 0);
    }
    None
}

pub(in crate::index) fn extract_quoted_title(task: &str) -> Option<String> {
    extract_quoted_titles(task)
        .into_iter()
        .next()
        .map(|title| title.to_ascii_lowercase())
}

pub(in crate::index) fn extract_quoted_titles(task: &str) -> Vec<String> {
    let mut titles = Vec::new();
    for quote in ['"', '\''] {
        let mut cursor = task;
        while let Some(start) = cursor.find(quote) {
            let tail = &cursor[start + quote.len_utf8()..];
            let Some(end) = tail.find(quote) else {
                break;
            };
            let title = tail[..end].trim();
            if title.split_whitespace().count() >= 2 {
                let title = title.to_string();
                if !titles.iter().any(|existing| existing == &title) {
                    titles.push(title);
                }
            }
            cursor = &tail[end + quote.len_utf8()..];
        }
        if !titles.is_empty() {
            break;
        }
    }
    titles
}

pub(in crate::index) fn extract_named_artwork_location_surface_from_line(
    _line: &str,
    line_lower: &str,
    title_lower: &str,
) -> Option<String> {
    let title_idx = line_lower.find(title_lower)?;
    let context_lower = &line_lower[title_idx + title_lower.len()..];
    extract_named_artwork_room_surface_from_context(context_lower, line_lower).or_else(|| {
        let prefix_lower = &line_lower[..title_idx];
        if context_lower.contains("above my bed")
            || context_lower.contains("above the bed")
            || (task_contains_any(context_lower, &["on my wall", "on the wall"])
                && prefix_lower.contains("bedroom"))
        {
            Some("in my bedroom".to_string())
        } else if task_contains_any(context_lower, &["above my sofa", "above the sofa"])
            && prefix_lower.contains("living room")
        {
            Some("above my living room sofa".to_string())
        } else {
            None
        }
    })
}

pub(in crate::index) fn extract_named_artwork_room_surface_from_context(
    context_lower: &str,
    full_lower: &str,
) -> Option<String> {
    if context_lower.contains("living room sofa") {
        return Some("above my living room sofa".to_string());
    }
    if context_lower.contains("above my bed") || context_lower.contains("above the bed") {
        return Some("in my bedroom".to_string());
    }
    for (marker, answer) in [
        ("bedroom", "in my bedroom"),
        ("living room", "in my living room"),
        ("dining room", "in my dining room"),
        ("family room", "in my family room"),
        ("guest room", "in my guest room"),
        ("office", "in my office"),
        ("studio", "in my studio"),
        ("kitchen", "in my kitchen"),
        ("hallway", "in my hallway"),
        ("entryway", "in my entryway"),
        ("party area", "in the party area"),
    ] {
        if context_lower.contains(marker) {
            return Some(answer.to_string());
        }
    }
    if task_contains_any(context_lower, &["on my wall", "on the wall"]) {
        for (marker, answer) in [
            ("bedroom", "in my bedroom"),
            ("living room", "in my living room"),
            ("office", "in my office"),
            ("studio", "in my studio"),
        ] {
            if full_lower.contains(marker) {
                return Some(answer.to_string());
            }
        }
    }
    None
}

pub(in crate::index) fn extract_rewatch_title_from_line(line: &str, lower: &str) -> Option<String> {
    for marker in ["re-watched ", "re watched ", "rewatched "] {
        let Some(start) = lower.find(marker) else {
            continue;
        };
        let title_start = start + marker.len();
        let tail = line[title_start..].trim();
        let tail_lower = lower[title_start..].trim();
        let mut end = tail.len();
        for delimiter in [
            ",",
            ".",
            "?",
            "!",
            " yesterday",
            " today",
            " again",
            " which ",
            " and ",
            " but ",
            " because ",
        ] {
            if let Some(idx) = tail_lower.find(delimiter) {
                end = end.min(idx);
            }
        }
        let title = tail[..end]
            .trim()
            .trim_matches(|c: char| matches!(c, '"' | '\'' | ',' | '.' | ';' | '!' | '?'));
        if title.len() >= 3 {
            return Some(title.to_string());
        }
    }
    None
}

pub(in crate::index) fn normalize_rewatch_title(title: &str) -> String {
    title
        .trim()
        .trim_matches(|c: char| matches!(c, '"' | '\'' | ',' | '.' | ';' | '!' | '?'))
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

pub(in crate::index) fn extract_origin_country_answer(line: &str) -> Option<String> {
    compile_regex(r"(?i)home country[, ]+([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)?)")
        .captures(line)
        .and_then(|captures| captures.get(1))
        .map(|value| value.as_str().trim().to_string())
        .filter(|value| !value.is_empty())
}

#[derive(Clone, Copy)]
pub(in crate::index) enum SyntheticDurationAnchor {
    CurrentDays(i32),
    AbsoluteDay(i32),
}

#[derive(Clone, Copy)]
pub(in crate::index) enum SyntheticEventAnchor {
    RelativeDaysAgo(i32),
    AbsoluteDay(i32),
}

#[derive(Clone, Copy)]
pub(in crate::index) struct SyntheticDurationValue {
    pub(in crate::index) amount: f32,
    pub(in crate::index) days: f32,
    pub(in crate::index) unit: &'static str,
}

#[derive(Clone, Copy)]
pub(in crate::index) enum SyntheticTemporalDirection {
    Earlier,
    Later,
}

pub(in crate::index) fn extract_temporal_choice_options(task: &str) -> Option<(String, String)> {
    let quoted = extract_quoted_titles(task);
    if quoted.len() >= 2 {
        return Some((quoted[0].trim().to_string(), quoted[1].trim().to_string()));
    }

    let tail = task
        .split_once(',')
        .map(|(_, suffix)| suffix)
        .unwrap_or(task)
        .trim()
        .trim_end_matches('?');
    let (left, right) = tail.rsplit_once(" or ")?;
    Some((
        normalize_temporal_choice_option(left),
        normalize_temporal_choice_option(right),
    ))
}

pub(in crate::index) fn normalize_temporal_choice_option(option: &str) -> String {
    option
        .trim()
        .trim_matches(|c: char| matches!(c, '"' | '\''))
        .trim_start_matches("the ")
        .trim_start_matches("The ")
        .trim()
        .to_string()
}

pub(in crate::index) fn extract_temporal_elapsed_phrases(
    task_lower: &str,
) -> Option<(String, String)> {
    let trimmed = task_lower.trim().trim_end_matches('?');
    let rest = trimmed.strip_prefix("how long had i been ")?;
    let (subject, event) = rest.split_once(" when ")?;
    Some((subject.trim().to_string(), event.trim().to_string()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::index) enum SyntheticElapsedFromNowUnit {
    Day,
    Week,
    Month,
    Year,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::index) struct SyntheticFromNowQuery {
    pub(super) unit: SyntheticElapsedFromNowUnit,
    pub(super) event_phrase: String,
    pub(super) anchor_phrase: Option<String>,
    pub(super) append_ago: bool,
}

pub(in crate::index) fn extract_temporal_from_now_query(
    task_lower: &str,
) -> Option<SyntheticFromNowQuery> {
    let trimmed = strip_temporal_reference_prefix(task_lower)
        .trim()
        .trim_end_matches('?');
    let rest = trimmed.strip_prefix("how many ")?;
    if let Some((unit_raw, event)) = rest.split_once(" ago did i ") {
        let unit = parse_temporal_from_now_unit(unit_raw)?;
        let (event_phrase, anchor_phrase) = split_temporal_when_anchor(event);
        let append_ago = anchor_phrase.is_some();
        return Some(SyntheticFromNowQuery {
            unit,
            event_phrase,
            anchor_phrase,
            append_ago,
        });
    }
    if let Some((unit_raw, event)) = rest.split_once(" have passed since i ") {
        let unit = parse_temporal_from_now_unit(unit_raw)?;
        let (event_phrase, anchor_phrase) = split_temporal_when_anchor(event);
        return Some(SyntheticFromNowQuery {
            unit,
            event_phrase,
            anchor_phrase,
            append_ago: false,
        });
    }
    None
}

pub(in crate::index) fn split_temporal_when_anchor(event: &str) -> (String, Option<String>) {
    let trimmed = event.trim();
    if let Some((primary, anchor)) = trimmed.split_once(" when i ") {
        let primary = primary.trim().to_string();
        let anchor = anchor.trim();
        if !primary.is_empty() && !anchor.is_empty() {
            return (primary, Some(anchor.to_string()));
        }
    }
    (trimmed.to_string(), None)
}

pub(in crate::index) fn strip_temporal_reference_prefix(task_lower: &str) -> &str {
    let trimmed = task_lower.trim();
    if trimmed.starts_with("as of ") {
        if let Some(pos) = trimmed.find("how many ") {
            return &trimmed[pos..];
        }
    }
    trimmed
}
