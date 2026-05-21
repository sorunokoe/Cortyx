use super::quantity_total_support::{extract_count_before_phrase, month_number, ordinal_day};
use super::*;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum QuantityTotalQuery {
    RoadTripDistance(RoadTripDistanceQuery),
    ConsecutiveWeekendHikeDistance(ConsecutiveWeekendHikeDistanceQuery),
    StayDays(StayDaysQuery),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct RoadTripDistanceQuery {
    pub(super) expected_trip_count: usize,
    pub(super) required_terms: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ConsecutiveWeekendHikeDistanceQuery {
    pub(super) required_terms: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct StayDaysQuery {
    pub(super) places: Vec<String>,
    pub(super) required_terms: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) enum WeekendMarker {
    LastWeekend,
    TwoWeekendsAgo,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct QuantityTotalFact {
    pub(super) key: String,
    pub(super) value: f32,
    pub(super) alternate_value: Option<f32>,
    pub(super) alternate_reason: Option<String>,
    pub(super) item_count: usize,
    pub(super) score: usize,
    pub(super) evidence: String,
}

pub(super) fn parse_quantity_total_query(task_lower: &str) -> Option<QuantityTotalQuery> {
    if task_lower.contains("total distance") {
        if task_lower.contains("road trip") {
            return parse_road_trip_distance_query(task_lower)
                .map(QuantityTotalQuery::RoadTripDistance);
        }
        if task_lower.contains("hike") && task_lower.contains("weekend") {
            return Some(QuantityTotalQuery::ConsecutiveWeekendHikeDistance(
                ConsecutiveWeekendHikeDistanceQuery {
                    required_terms: vec![
                        "hike".to_string(),
                        "trail".to_string(),
                        "weekend".to_string(),
                        "mile".to_string(),
                    ],
                },
            ));
        }
    }

    if task_lower.contains("total number of days") && task_lower.contains("spent in") {
        return parse_stay_days_query(task_lower).map(QuantityTotalQuery::StayDays);
    }

    None
}

pub(super) fn extract_road_trip_distance_fact_from_line(
    line: &str,
    lower: &str,
) -> Option<QuantityTotalFact> {
    if !lower.starts_with("user:")
        || !task_contains_any(lower, &["covered a total of", "covered total of"])
        || !task_contains_any(lower, &["road trip", "road trips", " trip "])
        || !is_realized_quantity_fact_line(lower)
    {
        return None;
    }

    let miles = extract_miles_value(
        line,
        &[
            r"(?i)\bcovered\s+(?:a\s+total\s+of\s+)?([0-9][0-9,]*(?:\.\d+)?)\s+miles\b",
            r"(?i)\b([0-9][0-9,]*(?:\.\d+)?)\s+miles\b",
        ],
    )?;
    let trip_count = extract_count_before_phrase(lower, "road trip")
        .or_else(|| extract_count_before_phrase(lower, "road trips"))
        .unwrap_or(1);
    Some(QuantityTotalFact {
        key: normalized_synthetic_phrase_key(line),
        value: miles,
        alternate_value: None,
        alternate_reason: None,
        item_count: trip_count,
        score: 18
            + trip_count * 8
            + usize::from(lower.contains("covered a total of")) * 8
            + usize::from(lower.contains("road trips")) * 6,
        evidence: line.trim().to_string(),
    })
}

pub(super) fn extract_weekend_hike_distance_fact_from_line(
    line: &str,
    lower: &str,
) -> Option<(WeekendMarker, QuantityTotalFact)> {
    if !lower.starts_with("user:")
        || !task_contains_any(lower, &["hike", "trail", "loop"])
        || !is_realized_quantity_fact_line(lower)
    {
        return None;
    }

    let marker = if lower.contains("two weekends ago") || lower.contains("weekend before last") {
        WeekendMarker::TwoWeekendsAgo
    } else if lower.contains("last weekend") {
        WeekendMarker::LastWeekend
    } else {
        return None;
    };
    let miles = extract_miles_value(
        line,
        &[
            r"(?i)\b([0-9][0-9,]*(?:\.\d+)?)\s*-\s*mile\b",
            r"(?i)\b([0-9][0-9,]*(?:\.\d+)?)\s+miles?\b",
        ],
    )?;
    Some((
        marker,
        QuantityTotalFact {
            key: format!("weekend-{marker:?}").to_ascii_lowercase(),
            value: miles,
            alternate_value: None,
            alternate_reason: None,
            item_count: 1,
            score: 20
                + usize::from(matches!(marker, WeekendMarker::LastWeekend)) * 6
                + usize::from(lower.contains("hike")) * 6
                + usize::from(lower.contains("trail")) * 4,
            evidence: line.trim().to_string(),
        },
    ))
}

pub(super) fn extract_stay_days_fact_from_line(
    line: &str,
    lower: &str,
    query: &StayDaysQuery,
) -> Option<QuantityTotalFact> {
    if !lower.starts_with("user:") || !is_realized_quantity_fact_line(lower) {
        return None;
    }

    let place = query
        .places
        .iter()
        .find(|place| lower.contains(place.as_str()))?
        .to_string();
    let days = extract_stay_days_from_line(line)?;
    Some(QuantityTotalFact {
        key: place.clone(),
        value: days.primary_days,
        alternate_value: days.alternate_days,
        alternate_reason: days.alternate_reason,
        item_count: 1,
        score: 20
            + usize::from(lower.contains("trip")) * 6
            + usize::from(lower.contains(&place)) * 8
            + usize::from(lower.contains("from ")) * 4,
        evidence: line.trim().to_string(),
    })
}

pub(super) fn format_distance_total_answer(miles: f32) -> String {
    let suffix = if (miles - 1.0).abs() < 0.01 {
        "mile"
    } else {
        "miles"
    };
    if (miles - miles.round()).abs() < 0.01 {
        format!(
            "{} {suffix}",
            format_integer_with_commas(miles.round() as i64)
        )
    } else {
        format!("{} {suffix}", format_numeric_answer(miles))
    }
}

fn parse_road_trip_distance_query(task_lower: &str) -> Option<RoadTripDistanceQuery> {
    let expected_trip_count = extract_count_before_phrase(task_lower, "road trip")
        .or_else(|| extract_count_before_phrase(task_lower, "road trips"))
        .unwrap_or(2);
    Some(RoadTripDistanceQuery {
        expected_trip_count,
        required_terms: vec![
            "road".to_string(),
            "trip".to_string(),
            "distance".to_string(),
            "miles".to_string(),
            "covered".to_string(),
        ],
    })
}

fn parse_stay_days_query(task_lower: &str) -> Option<StayDaysQuery> {
    let tail = compile_regex_static(r"(?i)\bspent in\s+(.+?)\??$")
        .captures(task_lower)
        .and_then(|captures| captures.get(1))
        .map(|value| value.as_str().trim())?;
    let places = split_bundle_items(tail)
        .into_iter()
        .map(|place| normalize_place_surface(&place))
        .filter(|place| !place.is_empty())
        .collect::<Vec<_>>();
    (places.len() >= 2).then_some(StayDaysQuery {
        required_terms: places
            .iter()
            .cloned()
            .chain(["trip".to_string(), "days".to_string()])
            .collect(),
        places,
    })
}

fn normalize_place_surface(raw: &str) -> String {
    raw.trim()
        .trim_start_matches("the ")
        .trim()
        .to_ascii_lowercase()
}

fn split_bundle_items(surface: &str) -> Vec<String> {
    let normalized = surface.trim().replace(", and ", ", ");
    if normalized.contains(',') {
        return normalized
            .split(',')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .map(ToString::to_string)
            .collect();
    }
    normalized
        .split(" and ")
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn is_realized_quantity_fact_line(lower: &str) -> bool {
    if task_contains_any(
        lower,
        &[
            "just got back",
            "got back from",
            "i went to",
            "i went",
            "i did a",
            "i did an",
            "i've covered",
            "i covered",
            "during my last",
            "last weekend",
            "two weekends ago",
            "before from",
        ],
    ) {
        return true;
    }

    !task_contains_any(
        lower,
        &[
            "i'm planning",
            "i am planning",
            "planning another",
            "planning a",
            "thinking about taking",
            "thinking of going",
            "would like to know",
            "can you recommend",
            "can you help",
            "best route",
            "good routes",
        ],
    )
}

fn extract_miles_value(line: &str, patterns: &[&str]) -> Option<f32> {
    patterns.iter().find_map(|pattern| {
        compile_regex_static(pattern)
            .captures(line)
            .and_then(|captures| captures.get(1))
            .and_then(|value| value.as_str().replace(',', "").parse::<f32>().ok())
    })
}

#[derive(Clone, Debug, PartialEq)]
struct StayDaysExtraction {
    primary_days: f32,
    alternate_days: Option<f32>,
    alternate_reason: Option<String>,
}

fn extract_stay_days_from_line(line: &str) -> Option<StayDaysExtraction> {
    if let Some(duration) =
        extract_aggregate_duration_value(line).filter(|value| value.unit == "day")
    {
        return Some(StayDaysExtraction {
            primary_days: duration.amount,
            alternate_days: None,
            alternate_reason: None,
        });
    }
    extract_day_range_days(line)
}

fn extract_day_range_days(line: &str) -> Option<StayDaysExtraction> {
    let same_month = compile_regex_static(
        r"(?i)\bfrom\s+([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?\s+to\s+(\d{1,2})(?:st|nd|rd|th)?\b",
    );
    if let Some(caps) = same_month.captures(line) {
        let month = month_number(caps.get(1)?.as_str())?;
        let start = caps.get(2)?.as_str().parse::<u32>().ok()?;
        let end = caps.get(3)?.as_str().parse::<u32>().ok()?;
        let reason = caps
            .get(0)?
            .as_str()
            .trim()
            .trim_start_matches("from ")
            .to_string();
        let primary_days = ordinal_day(month, end)
            .checked_sub(ordinal_day(month, start))
            .filter(|days| *days > 0)
            .map(|days| days as f32)?;
        return Some(StayDaysExtraction {
            primary_days,
            alternate_days: Some(primary_days + 1.0),
            alternate_reason: Some(reason),
        });
    }

    let cross_month = compile_regex_static(
        r"(?i)\bfrom\s+([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?\s+to\s+([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?\b",
    );
    let caps = cross_month.captures(line)?;
    let start_month = month_number(caps.get(1)?.as_str())?;
    let start_day = caps.get(2)?.as_str().parse::<u32>().ok()?;
    let end_month = month_number(caps.get(3)?.as_str())?;
    let end_day = caps.get(4)?.as_str().parse::<u32>().ok()?;
    let reason = caps
        .get(0)?
        .as_str()
        .trim()
        .trim_start_matches("from ")
        .to_string();
    let primary_days = ordinal_day(end_month, end_day)
        .checked_sub(ordinal_day(start_month, start_day))
        .filter(|days| *days > 0)
        .map(|days| days as f32)?;
    Some(StayDaysExtraction {
        primary_days,
        alternate_days: Some(primary_days + 1.0),
        alternate_reason: Some(reason),
    })
}
