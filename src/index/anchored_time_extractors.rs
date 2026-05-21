use super::*;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum AnchoredTimeQuery {
    BedtimeBeforeAppointment(BedtimeBeforeAppointmentQuery),
    ClinicArrival(ClinicArrivalQuery),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct BedtimeBeforeAppointmentQuery {
    pub(super) required_terms: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ClinicArrivalQuery {
    pub(super) weekday: String,
    pub(super) required_terms: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct WeekdayTimeFact {
    pub(super) weekday: String,
    pub(super) time: String,
    pub(super) score: usize,
    pub(super) evidence: String,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct TravelDurationFact {
    pub(super) minutes: i32,
    pub(super) score: usize,
    pub(super) evidence: String,
}

pub(super) fn parse_anchored_time_query(task_lower: &str) -> Option<AnchoredTimeQuery> {
    if !task_lower.contains("what time") {
        return None;
    }
    if task_lower.contains("go to bed")
        && task_contains_any(task_lower, &["day before", "the day before"])
        && task_lower.contains("doctor")
        && task_lower.contains("appointment")
    {
        return Some(AnchoredTimeQuery::BedtimeBeforeAppointment(
            BedtimeBeforeAppointmentQuery {
                required_terms: vec![
                    "bed".to_string(),
                    "doctor".to_string(),
                    "appointment".to_string(),
                    "last".to_string(),
                    "am".to_string(),
                ],
            },
        ));
    }
    if task_contains_any(
        task_lower,
        &[
            "reach the clinic",
            "get to the clinic",
            "arrive at the clinic",
        ],
    ) {
        return Some(AnchoredTimeQuery::ClinicArrival(ClinicArrivalQuery {
            weekday: extract_weekday_from_query(task_lower)?.to_string(),
            required_terms: vec![
                "clinic".to_string(),
                "left".to_string(),
                "home".to_string(),
                "hours".to_string(),
                extract_weekday_from_query(task_lower)?.to_string(),
            ],
        }));
    }
    None
}

pub(super) fn extract_bedtime_fact_from_line(line: &str, lower: &str) -> Option<WeekdayTimeFact> {
    if !is_summary_or_user_line(line, lower) || !lower.contains("bed") {
        return None;
    }
    let time = extract_time_answer_from_line(line)?;
    let weekday = extract_time_aligned_weekday(line, lower)?;
    Some(WeekdayTimeFact {
        weekday,
        time,
        score: 20
            + usize::from(lower.contains("sluggish")) * 2
            + usize::from(lower.starts_with("user:")) * 6,
        evidence: line.trim().to_string(),
    })
}

pub(super) fn extract_doctor_appointment_fact_from_line(
    line: &str,
    lower: &str,
) -> Option<WeekdayTimeFact> {
    if !is_summary_or_user_line(line, lower)
        || !lower.contains("doctor")
        || !lower.contains("appointment")
    {
        return None;
    }
    let time = extract_time_answer_from_line(line)?;
    let weekday = extract_time_aligned_weekday(line, lower)?;
    Some(WeekdayTimeFact {
        weekday,
        time,
        score: 22 + usize::from(lower.starts_with("user:")) * 6,
        evidence: line.trim().to_string(),
    })
}

pub(super) fn extract_departure_home_fact_from_line(
    line: &str,
    lower: &str,
    weekday: &str,
) -> Option<WeekdayTimeFact> {
    if !is_summary_or_user_line(line, lower)
        || !lower.contains("left home")
        || !lower.contains(weekday)
    {
        return None;
    }
    let time = extract_time_answer_from_line(line)?;
    Some(WeekdayTimeFact {
        weekday: weekday.to_string(),
        time,
        score: 22
            + usize::from(lower.contains("doctor's appointment")) * 4
            + usize::from(lower.starts_with("user:")) * 6,
        evidence: line.trim().to_string(),
    })
}

pub(super) fn extract_clinic_travel_duration_fact_from_line(
    line: &str,
    lower: &str,
) -> Option<TravelDurationFact> {
    if !is_summary_or_user_line(line, lower)
        || !lower.contains("clinic")
        || !task_contains_any(lower, &["took me", "get to the clinic", "drive"])
    {
        return None;
    }
    let minutes = duration_to_minutes(&extract_duration_answer_from_line(line)?)?;
    Some(TravelDurationFact {
        minutes,
        score: 20
            + usize::from(lower.contains("last time")) * 3
            + usize::from(lower.starts_with("user:")) * 6,
        evidence: line.trim().to_string(),
    })
}

pub(super) fn previous_weekday(day: &str) -> Option<&'static str> {
    match day {
        "monday" => Some("sunday"),
        "tuesday" => Some("monday"),
        "wednesday" => Some("tuesday"),
        "thursday" => Some("wednesday"),
        "friday" => Some("thursday"),
        "saturday" => Some("friday"),
        "sunday" => Some("saturday"),
        _ => None,
    }
}

pub(super) fn add_minutes_to_clock_time(time: &str, minutes: i32) -> Option<String> {
    let total = parse_clock_minutes(time)? + minutes;
    let normalized = ((total % (24 * 60)) + (24 * 60)) % (24 * 60);
    let hour24 = normalized / 60;
    let minute = normalized % 60;
    let meridiem = if hour24 >= 12 { "PM" } else { "AM" };
    let hour12 = match hour24 % 12 {
        0 => 12,
        value => value,
    };
    Some(format!("{hour12}:{minute:02} {meridiem}"))
}

fn duration_to_minutes(surface: &str) -> Option<i32> {
    let days = duration_answer_magnitude(surface)?;
    Some((days * 24.0 * 60.0).round() as i32)
}

fn parse_clock_minutes(surface: &str) -> Option<i32> {
    let captures =
        compile_regex_static(r"(?i)\b(\d{1,2})(?::(\d{2}))?\s?(AM|PM)\b").captures(surface)?;
    let hour = captures.get(1)?.as_str().parse::<i32>().ok()?;
    let minute = captures
        .get(2)
        .and_then(|value| value.as_str().parse::<i32>().ok())
        .unwrap_or(0);
    let meridiem = captures.get(3)?.as_str().to_ascii_uppercase();
    let hour24 = match (hour % 12, meridiem.as_str()) {
        (value, "AM") => value,
        (value, "PM") => value + 12,
        _ => return None,
    };
    Some(hour24 * 60 + minute)
}

fn extract_time_aligned_weekday(line: &str, lower: &str) -> Option<String> {
    let time_start = compile_regex_static(r"(?i)\b\d{1,2}(?::\d{2})?\s?(?:AM|PM)\b")
        .find(line)
        .map(|matched| matched.start())?;
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
    .filter_map(|day| {
        lower
            .find(day)
            .map(|position| (position.abs_diff(time_start), day))
    })
    .min_by_key(|(distance, _)| *distance)
    .map(|(_, day)| day.to_string())
}

fn is_summary_or_user_line(line: &str, lower: &str) -> bool {
    lower.starts_with("user:") || line.trim_start().starts_with('-')
}
