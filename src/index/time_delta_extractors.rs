use super::*;

const TIME_QUERY_STOP: &[&str] = &[
    "a", "compared", "did", "earlier", "faster", "finish", "how", "i", "my", "other", "previous",
    "run", "than", "the", "time", "to", "up", "wake", "weekdays", "year", "years",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum TimeDeltaQuery {
    Wakeup(WakeupDeltaQuery),
    Performance(PerformanceDeltaQuery),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct WakeupDeltaQuery {
    pub(super) comparison_day: String,
    pub(super) required_terms: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct PerformanceDeltaQuery {
    pub(super) activity_terms: Vec<String>,
    pub(super) required_terms: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum WakeupFactKind {
    ComparisonDay,
    BaselineWeekday,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PerformanceFactKind {
    Previous,
    Current,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct TimeOfDayFact {
    pub(super) minutes_after_midnight: i32,
    pub(super) kind: WakeupFactKind,
    pub(super) score: usize,
    pub(super) evidence: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct PerformanceDurationFact {
    pub(super) minutes: i32,
    pub(super) kind: PerformanceFactKind,
    pub(super) score: usize,
    pub(super) evidence: String,
}

pub(super) fn parse_time_delta_query(task_lower: &str) -> Option<TimeDeltaQuery> {
    if let Some(query) = parse_wakeup_delta_query(task_lower) {
        return Some(TimeDeltaQuery::Wakeup(query));
    }
    parse_performance_delta_query(task_lower).map(TimeDeltaQuery::Performance)
}

pub(super) fn extract_wakeup_time_fact_from_line(
    line: &str,
    lower: &str,
    query: &WakeupDeltaQuery,
) -> Option<TimeOfDayFact> {
    if !is_summary_or_user_line(line, lower) || !task_contains_any(lower, &["wake up", "waking up"])
    {
        return None;
    }
    let time = extract_time_answer_from_line(line)?;
    let minutes_after_midnight = parse_meridiem_time_minutes(&time)?;
    let comparison_surface = format!("{}s", query.comparison_day);
    let (kind, score) =
        if lower.contains(query.comparison_day.as_str()) || lower.contains(&comparison_surface) {
            (
                WakeupFactKind::ComparisonDay,
                usize::from(lower.starts_with("user:")) * 8 + 12,
            )
        } else if lower.contains("weekdays") || lower.contains("monday to friday") {
            (
                WakeupFactKind::BaselineWeekday,
                usize::from(lower.starts_with("user:")) * 8 + 10,
            )
        } else {
            return None;
        };
    Some(TimeOfDayFact {
        minutes_after_midnight,
        kind,
        score,
        evidence: line.trim().to_string(),
    })
}

pub(super) fn extract_performance_duration_fact_from_line(
    line: &str,
    lower: &str,
    query: &PerformanceDeltaQuery,
) -> Option<PerformanceDurationFact> {
    if !is_summary_or_user_line(line, lower) || performance_focus_match_count(lower, query) == 0 {
        return None;
    }
    let minutes = extract_duration_minutes_from_line(line)?;
    (minutes > 0).then_some(())?;
    let (kind, score) = if task_contains_any(lower, &["last year", "previous year", "previous"]) {
        (
            PerformanceFactKind::Previous,
            performance_focus_match_count(lower, query) * 10
                + usize::from(lower.starts_with("user:")) * 8
                + 12,
        )
    } else if task_contains_any(
        lower,
        &[
            "recently finished",
            "just finished",
            "completed",
            "finished a",
        ],
    ) {
        (
            PerformanceFactKind::Current,
            performance_focus_match_count(lower, query) * 10
                + usize::from(lower.starts_with("user:")) * 8
                + 10,
        )
    } else {
        return None;
    };
    Some(PerformanceDurationFact {
        minutes,
        kind,
        score,
        evidence: line.trim().to_string(),
    })
}

pub(super) fn format_minutes_delta(minutes: i32) -> String {
    let unit = if minutes == 1 { "minute" } else { "minutes" };
    format!("{minutes} {unit}")
}

fn parse_wakeup_delta_query(task_lower: &str) -> Option<WakeupDeltaQuery> {
    if !task_contains_any(task_lower, &["how much earlier", "how much later"])
        || !task_lower.contains("wake up")
        || !task_lower.contains("weekday")
    {
        return None;
    }
    let comparison_day = compile_regex_static(
        r"(?i)\bon\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?\b",
    )
    .captures(task_lower)
    .and_then(|captures| captures.get(1))
    .map(|value| value.as_str().to_string())?;
    Some(WakeupDeltaQuery {
        comparison_day: comparison_day.clone(),
        required_terms: vec![comparison_day, "wake".to_string(), "weekdays".to_string()],
    })
}

fn parse_performance_delta_query(task_lower: &str) -> Option<PerformanceDeltaQuery> {
    if !task_contains_any(task_lower, &["how much faster", "how much slower"])
        || !task_lower.contains("compared to")
    {
        return None;
    }
    let activity_phrase =
        compile_regex_static(r"(?i)\bfinish(?:ed)?\s+(?:the\s+)?(.+?)\s+compared to\b")
            .captures(task_lower)
            .and_then(|captures| captures.get(1))
            .map(|value| value.as_str().trim().to_string())?;
    let activity_terms = normalized_time_terms(&activity_phrase);
    if activity_terms.is_empty() {
        return None;
    }
    let mut required_terms = activity_terms.clone();
    required_terms.push("previous".to_string());
    required_terms.sort();
    required_terms.dedup();
    Some(PerformanceDeltaQuery {
        activity_terms,
        required_terms,
    })
}

fn normalized_time_terms(surface: &str) -> Vec<String> {
    let mut terms = synthetic_query_terms(surface)
        .into_iter()
        .filter(|term| !TIME_QUERY_STOP.contains(&term.as_str()))
        .collect::<Vec<_>>();
    terms.sort();
    terms.dedup();
    terms
}

fn performance_focus_match_count(lower: &str, query: &PerformanceDeltaQuery) -> usize {
    query
        .activity_terms
        .iter()
        .filter(|term| lower.contains(term.as_str()))
        .count()
}

fn parse_meridiem_time_minutes(surface: &str) -> Option<i32> {
    let captures =
        compile_regex_static(r"(?i)\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b").captures(surface)?;
    let mut hour = captures.get(1)?.as_str().parse::<i32>().ok()?;
    let minute = captures
        .get(2)
        .map(|value| value.as_str().parse::<i32>().ok())
        .unwrap_or(Some(0))?;
    let meridiem = captures.get(3)?.as_str().to_ascii_lowercase();
    if meridiem == "am" {
        if hour == 12 {
            hour = 0;
        }
    } else if hour != 12 {
        hour += 12;
    }
    Some(hour * 60 + minute)
}

fn extract_duration_minutes_from_line(line: &str) -> Option<i32> {
    if let Some(total) = extract_hour_minute_total_from_text(line) {
        return Some(total);
    }
    let duration = normalize_current_duration_answer(&extract_duration_answer_from_line(line)?);
    let captures = compile_regex_static(
        r"(?i)\b(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(hour|minute)s?\b",
    )
    .captures(&duration)?;
    let quantity = parse_small_duration_quantity(captures.get(1)?.as_str())?;
    let minutes = match captures.get(2)?.as_str().to_ascii_lowercase().as_str() {
        "hour" => quantity * 60.0,
        "minute" => quantity,
        _ => return None,
    };
    #[allow(clippy::cast_possible_truncation)]
    let rounded = minutes.round() as i32;
    Some(rounded)
}

fn parse_small_duration_quantity(raw: &str) -> Option<f32> {
    Some(match raw.to_ascii_lowercase().as_str() {
        "one" => 1.0,
        "two" => 2.0,
        "three" => 3.0,
        "four" => 4.0,
        "five" => 5.0,
        "six" => 6.0,
        "seven" => 7.0,
        "eight" => 8.0,
        "nine" => 9.0,
        "ten" => 10.0,
        "eleven" => 11.0,
        "twelve" => 12.0,
        value => value.parse::<f32>().ok()?,
    })
}
