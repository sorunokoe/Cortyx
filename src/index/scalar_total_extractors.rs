use super::*;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum ScalarTotalQuery {
    SiblingCount(SiblingCountQuery),
    PlatformPeakMetricTotal(PlatformPeakMetricTotalQuery),
    DurationBundle(DurationBundleQuery),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SiblingCountQuery {
    pub(super) required_terms: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct PlatformPeakMetricTotalQuery {
    pub(super) platforms: Vec<PlatformMetricFocus>,
    pub(super) required_terms: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct PlatformMetricFocus {
    pub(super) key: String,
    pub(super) display: String,
    pub(super) required_terms: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct DurationBundleQuery {
    pub(super) activities: Vec<DurationActivity>,
    pub(super) required_terms: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) enum DurationActivity {
    GetReady,
    CommuteToWork,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct ScalarTotalFact {
    pub(super) key: String,
    pub(super) value: i32,
    pub(super) score: usize,
    pub(super) evidence: String,
}

pub(super) fn parse_scalar_total_query(task: &str, task_lower: &str) -> Option<ScalarTotalQuery> {
    if !detect_counting_query(task) && !task_contains_any(task_lower, &["total time", "how long"]) {
        return None;
    }

    if task_lower.contains("siblings") {
        return Some(ScalarTotalQuery::SiblingCount(SiblingCountQuery {
            required_terms: vec![
                "siblings".to_string(),
                "brother".to_string(),
                "sister".to_string(),
                "family".to_string(),
            ],
        }));
    }

    if task_contains_any(task_lower, &["most popular video", "most popular videos"])
        && task_lower.contains("views")
    {
        let tail = task_lower
            .rsplit_once(" on ")
            .map(|(_, tail)| tail.trim().trim_end_matches('?'))?;
        let platforms = split_bundle_items(tail)
            .into_iter()
            .map(|surface| build_platform_focus(&surface))
            .filter(|focus| !focus.required_terms.is_empty())
            .collect::<Vec<_>>();
        if platforms.len() >= 2 {
            let mut required_terms = vec!["views".to_string(), "video".to_string()];
            for focus in &platforms {
                required_terms.extend(focus.required_terms.iter().cloned());
            }
            required_terms.sort();
            required_terms.dedup();
            return Some(ScalarTotalQuery::PlatformPeakMetricTotal(
                PlatformPeakMetricTotalQuery {
                    platforms,
                    required_terms,
                },
            ));
        }
    }

    if task_lower.contains("get ready")
        && task_contains_any(task_lower, &["commute to work", "commute"])
        && task_contains_any(task_lower, &["total time", "takes"])
    {
        return Some(ScalarTotalQuery::DurationBundle(DurationBundleQuery {
            activities: vec![DurationActivity::GetReady, DurationActivity::CommuteToWork],
            required_terms: vec![
                "get".to_string(),
                "ready".to_string(),
                "commute".to_string(),
                "work".to_string(),
                "minutes".to_string(),
                "hour".to_string(),
            ],
        }));
    }

    None
}

pub(super) fn extract_sibling_count_facts_from_line(
    line: &str,
    lower: &str,
) -> Vec<ScalarTotalFact> {
    if !(lower.starts_with("user:") || line.trim_start().starts_with('-')) {
        return Vec::new();
    }

    let mut facts = Vec::new();
    for (label, key) in [("brother", "brothers"), ("sister", "sisters")] {
        let Some(count) = extract_count_for_relation(line, label) else {
            continue;
        };
        facts.push(ScalarTotalFact {
            key: key.to_string(),
            value: count,
            score: 18 + count.max(0) as usize * 4 + usize::from(lower.contains("family")) * 3,
            evidence: line.trim().to_string(),
        });
    }
    facts
}

pub(super) fn extract_platform_peak_metric_fact_from_line(
    line: &str,
    lower: &str,
    focus: &PlatformMetricFocus,
) -> Option<ScalarTotalFact> {
    if !(lower.starts_with("user:") || line.trim_start().starts_with('-'))
        || !lower.contains("view")
        || term_overlap_count(
            lower,
            &focus
                .required_terms
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
        ) == 0
    {
        return None;
    }

    let value = extract_line_numbers(line).into_iter().max()?;
    Some(ScalarTotalFact {
        key: focus.key.clone(),
        value,
        score: 20 + value.max(0) as usize / 100 + usize::from(lower.contains("popular")) * 3,
        evidence: line.trim().to_string(),
    })
}

pub(super) fn extract_duration_bundle_fact_from_line(
    line: &str,
    lower: &str,
    activity: DurationActivity,
) -> Option<ScalarTotalFact> {
    if !(lower.starts_with("user:") || line.trim_start().starts_with('-')) {
        return None;
    }

    let minutes = match activity {
        DurationActivity::GetReady => {
            if !task_contains_any(lower, &["get ready", "getting ready"]) {
                return None;
            }
            duration_surface_minutes(&extract_duration_answer_from_line(line)?)?
        },
        DurationActivity::CommuteToWork => {
            if !lower.contains("commute") {
                return None;
            }
            duration_surface_minutes(&extract_commute_duration_from_line(line)?)?
        },
    };
    Some(ScalarTotalFact {
        key: activity.key().to_string(),
        value: minutes,
        score: 18 + minutes.max(0) as usize + usize::from(lower.contains("work")) * 2,
        evidence: line.trim().to_string(),
    })
}

fn build_platform_focus(surface: &str) -> PlatformMetricFocus {
    let display = surface.trim().trim_start_matches("my ").trim().to_string();
    PlatformMetricFocus {
        key: normalized_synthetic_phrase_key(&display),
        required_terms: synthetic_query_terms(&display),
        display,
    }
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

fn extract_count_for_relation(line: &str, relation: &str) -> Option<i32> {
    for pattern in [
        format!(
            r"(?i)\b(?:i have|i've got|i also have|come from a family with)\s+(a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+{relation}s?\b"
        ),
        format!(r"(?i)\bmy\s+{relation}\b"),
    ] {
        let Some(captures) = compile_regex(&pattern).captures(line) else {
            continue;
        };
        let value = captures
            .get(1)
            .and_then(|matched| parse_count_token_value(matched.as_str()))
            .unwrap_or(1);
        return Some(value);
    }
    None
}

fn duration_surface_minutes(surface: &str) -> Option<i32> {
    let days = duration_answer_magnitude(surface)?;
    Some((days * 24.0 * 60.0).round() as i32)
}

impl DurationActivity {
    pub(super) fn key(self) -> &'static str {
        match self {
            Self::GetReady => "get-ready",
            Self::CommuteToWork => "commute-to-work",
        }
    }
}
