use super::*;

pub(crate) fn generate_temporal_turn_answer_surface_rows(turn: &Turn) -> Vec<AnswerSurfaceRow> {
    let Some(timestamp) = turn.timestamp.as_deref() else {
        return Vec::new();
    };
    let Some(answer_span) = extract_temporal_event_date_surface_value(&turn.text, timestamp) else {
        return Vec::new();
    };
    let Some(event_pattern) = temporal_event_question_pattern(&turn.text, turn.speaker.as_deref())
    else {
        return Vec::new();
    };
    vec![AnswerSurfaceRow {
        question_pattern: event_pattern,
        answer_span,
        confidence: 0.9,
    }]
}

fn temporal_event_question_pattern(text: &str, speaker: Option<&str>) -> Option<String> {
    let lower = text.to_ascii_lowercase();
    let event = extract_fact_after_any(
        text,
        &lower,
        &["went to ", "attended ", "visited ", "joined "],
        &[
            "yesterday",
            "today",
            "tonight",
            "last",
            "this",
            "on",
            "and",
            "but",
            "because",
            "with",
            "after",
        ],
        8,
    )?;
    let mut terms = vec![
        "when".to_string(),
        "date".to_string(),
        "day".to_string(),
        "go".to_string(),
        "went".to_string(),
        "attend".to_string(),
        "attended".to_string(),
        "visit".to_string(),
        "visited".to_string(),
        "join".to_string(),
        "joined".to_string(),
    ];
    terms.extend(
        event
            .split(|c: char| !c.is_ascii_alphanumeric() && c != '-' && c != '\'')
            .filter_map(|term| {
                let clean = term.trim().to_ascii_lowercase();
                (clean.len() >= 2).then_some(clean)
            }),
    );
    if let Some(scoped) = scoped_question_pattern(&terms.join(" "), speaker) {
        return Some(scoped);
    }
    terms.sort();
    terms.dedup();
    Some(terms.join(" "))
}

fn extract_temporal_event_date_surface_value(text: &str, timestamp: &str) -> Option<String> {
    if let Some(value) = extract_explicit_calendar_date(text) {
        return Some(value);
    }

    let lower = text.to_ascii_lowercase();
    let offset_days = if lower.contains("yesterday") || lower.contains("last night") {
        -1
    } else if lower.contains("today")
        || lower.contains("tonight")
        || lower.contains("this morning")
        || lower.contains("this afternoon")
        || lower.contains("this evening")
    {
        0
    } else {
        return None;
    };

    let shifted = shift_iso_date_by_days(timestamp, offset_days)?;
    let (year, month, day) = parse_iso_date_parts(&shifted)?;
    Some(render_human_date(year, month, day))
}

fn extract_explicit_calendar_date(text: &str) -> Option<String> {
    let dmy = compile_regex(r"(?i)\b(?:on\s+)?(\d{1,2})\s+([a-z]+),?\s+(\d{4})\b");
    if let Some(captures) = dmy.captures(text) {
        let day = captures.get(1)?.as_str().parse::<u32>().ok()?;
        let month = month_name_to_number(captures.get(2)?.as_str())?;
        let year = captures.get(3)?.as_str().parse::<u32>().ok()?;
        return Some(render_human_date(year, month, day));
    }

    let mdy = compile_regex(r"(?i)\b(?:on\s+)?([a-z]+)\s+(\d{1,2}),\s*(\d{4})\b");
    let captures = mdy.captures(text)?;
    let month = month_name_to_number(captures.get(1)?.as_str())?;
    let day = captures.get(2)?.as_str().parse::<u32>().ok()?;
    let year = captures.get(3)?.as_str().parse::<u32>().ok()?;
    Some(render_human_date(year, month, day))
}

fn shift_iso_date_by_days(timestamp: &str, delta_days: i32) -> Option<String> {
    let (year, month, day) = parse_iso_date_parts(timestamp)?;
    let absolute_days =
        days_from_civil(year as i32, month as i32, day as i32).checked_add(delta_days)?;
    let (shifted_year, shifted_month, shifted_day) = civil_from_days(absolute_days);
    Some(format!(
        "{shifted_year:04}-{shifted_month:02}-{shifted_day:02}T00:00:00Z"
    ))
}

fn parse_iso_date_parts(timestamp: &str) -> Option<(u32, u32, u32)> {
    let date = timestamp.get(..10)?;
    let mut parts = date.split('-');
    Some((
        parts.next()?.parse::<u32>().ok()?,
        parts.next()?.parse::<u32>().ok()?,
        parts.next()?.parse::<u32>().ok()?,
    ))
}

fn render_human_date(year: u32, month: u32, day: u32) -> String {
    const MONTHS: [&str; 12] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    format!(
        "{day} {} {year}",
        MONTHS[(month.saturating_sub(1)) as usize]
    )
}

pub(super) fn month_name_to_number(name: &str) -> Option<u32> {
    match name.trim().to_ascii_lowercase().as_str() {
        "jan" | "january" => Some(1),
        "feb" | "february" => Some(2),
        "mar" | "march" => Some(3),
        "apr" | "april" => Some(4),
        "may" => Some(5),
        "jun" | "june" => Some(6),
        "jul" | "july" => Some(7),
        "aug" | "august" => Some(8),
        "sep" | "sept" | "september" => Some(9),
        "oct" | "october" => Some(10),
        "nov" | "november" => Some(11),
        "dec" | "december" => Some(12),
        _ => None,
    }
}

fn days_from_civil(year: i32, month: i32, day: i32) -> i32 {
    let year = year - i32::from(month <= 2);
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let yoe = year - era * 400;
    let mp = month + if month > 2 { -3 } else { 9 };
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn civil_from_days(days: i32) -> (i32, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    (year + i32::from(month <= 2), month as u32, day as u32)
}
