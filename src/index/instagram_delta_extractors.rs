use super::*;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct InstagramDeltaQuery {
    pub(super) required_terms: Vec<String>,
    pub(super) window_days: Option<i32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct InstagramDeltaCandidate {
    pub(super) delta: i32,
    pub(super) score: usize,
    pub(super) evidence: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum InstagramGrowthRole {
    Baseline,
    WindowEnd,
    Neutral,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct InstagramGrowthPoint {
    pub(super) value: i32,
    pub(super) role: InstagramGrowthRole,
    pub(super) score: usize,
    pub(super) evidence: String,
}

pub(super) fn parse_instagram_delta_query(task_lower: &str) -> Option<InstagramDeltaQuery> {
    if !task_lower.contains("instagram")
        || !task_lower.contains("follower")
        || !task_contains_any(
            task_lower,
            &[
                "increase",
                "increased",
                "gain",
                "gained",
                "difference",
                "grew",
                "growth",
            ],
        )
    {
        return None;
    }
    let mut required_terms = vec!["instagram".to_string(), "follower".to_string()];
    if task_lower.contains("week") {
        required_terms.push("week".to_string());
    }
    if task_lower.contains("month") {
        required_terms.push("month".to_string());
    }
    if task_lower.contains("day") {
        required_terms.push("day".to_string());
    }
    Some(InstagramDeltaQuery {
        required_terms,
        window_days: extract_surface_window_days(task_lower),
    })
}

pub(super) fn extract_instagram_direct_delta_candidate(
    line: &str,
    lower: &str,
    query: &InstagramDeltaQuery,
) -> Option<InstagramDeltaCandidate> {
    if !is_instagram_growth_line(line, lower) {
        return None;
    }
    if let Some(days) = query.window_days {
        if !surface_window_matches(line, days) {
            return None;
        }
    }
    let lower = lower.trim();
    if let Some(captures) =
        compile_regex(r"(?i)\bfrom\s+(\d{1,7})\s+followers?\b.*?\bto\s+(\d{1,7})\s+followers?\b")
            .captures(line)
    {
        let start = captures.get(1)?.as_str().parse::<i32>().ok()?;
        let end = captures.get(2)?.as_str().parse::<i32>().ok()?;
        let delta = end.checked_sub(start)?;
        if delta > 0 {
            return Some(InstagramDeltaCandidate {
                delta,
                score: 34
                    + usize::from(query.window_days.is_some()) * 6
                    + usize::from(lower.starts_with("user:")) * 6,
                evidence: vec![line.trim().to_string()],
            });
        }
    }
    let delta = compile_regex(r"(?i)\b(?:grew|increased|gained)\s+by\s+(\d{1,7})\s+followers?\b")
        .captures(line)
        .and_then(|captures| captures.get(1))
        .and_then(|value| value.as_str().parse::<i32>().ok())?;
    (delta > 0).then_some(InstagramDeltaCandidate {
        delta,
        score: 30
            + usize::from(query.window_days.is_some()) * 6
            + usize::from(lower.starts_with("user:")) * 6,
        evidence: vec![line.trim().to_string()],
    })
}

pub(super) fn extract_instagram_growth_point(
    line: &str,
    lower: &str,
    query: &InstagramDeltaQuery,
) -> Option<InstagramGrowthPoint> {
    if !is_instagram_growth_line(line, lower) {
        return None;
    }
    let value = *extract_instagram_follower_counts(line).last()?;
    let role = if let Some(days) = query.window_days {
        if surface_window_matches(line, days) {
            if task_contains_any(lower, &["ago", "before"]) {
                InstagramGrowthRole::Baseline
            } else {
                InstagramGrowthRole::WindowEnd
            }
        } else if line_has_start_marker(lower) {
            InstagramGrowthRole::Baseline
        } else {
            InstagramGrowthRole::Neutral
        }
    } else if line_has_current_count_marker(lower) {
        InstagramGrowthRole::WindowEnd
    } else if line_has_start_marker(lower) {
        InstagramGrowthRole::Baseline
    } else {
        InstagramGrowthRole::Neutral
    };
    let score = 12
        + value.max(0) as usize / 100
        + usize::from(line_has_start_marker(lower)) * 6
        + usize::from(matches!(role, InstagramGrowthRole::WindowEnd)) * 8
        + usize::from(lower.starts_with("user:")) * 6
        + usize::from(task_contains_any(
            lower,
            &["around ", "about ", "approximately", "approx "],
        )) * 2;
    Some(InstagramGrowthPoint {
        value,
        role,
        score,
        evidence: line.trim().to_string(),
    })
}

fn is_instagram_growth_line(line: &str, lower: &str) -> bool {
    is_summary_or_user_line(line, lower)
        && lower.contains("instagram")
        && lower.contains("follower")
        && !task_contains_any(
            lower,
            &["facebook", "twitter", "tiktok", "youtube", "linkedin"],
        )
        && !extract_instagram_follower_counts(line).is_empty()
}

fn extract_instagram_follower_counts(line: &str) -> Vec<i32> {
    compile_regex(r"(?i)\b(\d{1,7})\s+followers?\b")
        .captures_iter(line)
        .filter_map(|captures| captures.get(1))
        .filter_map(|value| value.as_str().parse::<i32>().ok())
        .filter(|value| *value >= 10)
        .collect()
}

fn extract_surface_window_days(surface: &str) -> Option<i32> {
    duration_answer_magnitude(&normalize_current_duration_answer(
        &extract_duration_answer_from_line(surface)?,
    ))
    .map(|days| days.round() as i32)
}

fn surface_window_matches(surface: &str, expected_days: i32) -> bool {
    extract_surface_window_days(surface)
        .map(|days| days == expected_days)
        .unwrap_or(false)
}

fn line_has_start_marker(lower: &str) -> bool {
    task_contains_any(
        lower,
        &[
            "start of the year",
            "since the start of the year",
            "started the year",
            "began the year",
            "started with",
            "began with",
            "at the beginning",
            "initially",
            "at first",
        ],
    )
}
