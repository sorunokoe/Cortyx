use super::*;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum CountTotalQuery {
    MetricBundle(MetricBundleQuery),
    MealBundle(MealBundleQuery),
    OnlineCourseTotal(OnlineCourseTotalQuery),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct MetricBundleQuery {
    pub(super) metrics: Vec<MetricKind>,
    pub(super) anchor_terms: Vec<String>,
    pub(super) required_terms: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) enum MetricKind {
    Goals,
    Assists,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct MealBundleQuery {
    pub(super) items: Vec<CountItemFocus>,
    pub(super) required_terms: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct CountItemFocus {
    pub(super) key: String,
    pub(super) display: String,
    pub(super) required_terms: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct OnlineCourseTotalQuery {
    pub(super) required_terms: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct CountTotalFact {
    pub(super) key: String,
    pub(super) count: i32,
    pub(super) score: usize,
    pub(super) evidence: String,
}

pub(super) fn parse_count_total_query(task: &str, task_lower: &str) -> Option<CountTotalQuery> {
    if !detect_counting_query(task) {
        return None;
    }

    if let Some(query) = parse_metric_bundle_query(task_lower) {
        return Some(CountTotalQuery::MetricBundle(query));
    }
    if let Some(query) = parse_meal_bundle_query(task_lower) {
        return Some(CountTotalQuery::MealBundle(query));
    }
    if task_contains_any(
        task_lower,
        &[
            "total number of online course",
            "total number of online courses",
        ],
    ) && task_contains_any(task_lower, &["completed", "finished"])
    {
        return Some(CountTotalQuery::OnlineCourseTotal(OnlineCourseTotalQuery {
            required_terms: vec![
                "online".to_string(),
                "course".to_string(),
                "courses".to_string(),
                "completed".to_string(),
                "coursera".to_string(),
                "edx".to_string(),
            ],
        }));
    }

    None
}

pub(super) fn extract_metric_count_fact_from_line(
    line: &str,
    lower: &str,
    metric: MetricKind,
    anchor_terms: &[String],
) -> Option<CountTotalFact> {
    if !lower.starts_with("user:")
        || !line_matches_anchor_terms(lower, anchor_terms)
        || task_contains_any(lower, &["goal of", "goals for this season", "assist with"])
    {
        return None;
    }

    let count = match metric {
        MetricKind::Goals => extract_count_from_patterns(
            line,
            &[
                r"(?i)\bscored\s+([A-Za-z0-9,-]+)\s+goals?\b",
                r"(?i)\bhave\s+([A-Za-z0-9,-]+)\s+goals?\b",
                r"(?i)\b([A-Za-z0-9,-]+)\s+goals?\s+so far\b",
            ],
        )?,
        MetricKind::Assists => extract_count_from_patterns(
            line,
            &[
                r"(?i)\bhave\s+(?:had\s+)?([A-Za-z0-9,-]+)\s+assists?\b",
                r"(?i)\bgot\s+([A-Za-z0-9,-]+)\s+assists?\b",
                r"(?i)\b([A-Za-z0-9,-]+)\s+assists?\s+in the league\b",
            ],
        )?,
    };
    Some(CountTotalFact {
        key: metric.key().to_string(),
        count,
        score: 20
            + count.max(0) as usize * 4
            + term_overlap_count(
                lower,
                &anchor_terms.iter().map(String::as_str).collect::<Vec<_>>(),
            ) * 6,
        evidence: line.trim().to_string(),
    })
}

pub(super) fn extract_meal_count_fact_from_line(
    line: &str,
    lower: &str,
    focus: &CountItemFocus,
) -> Option<CountTotalFact> {
    if !lower.starts_with("user:") || line_matches_focus_terms(lower, &focus.required_terms) == 0 {
        return None;
    }

    let count = extract_count_from_patterns(
        line,
        &[
            r"(?i)\b(?:this is|it's|it was|the)\s+(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|\d+)\s+(?:meal|lunch)\b",
            r"(?i)\b(?:lasted me for|gave me|yielded)\s+(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+(?:meals?|lunches?)\b",
            r"(?i)\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+(?:meals?|lunches?)\b",
        ],
    )?;
    if task_contains_any(
        lower,
        &["looking for", "do you have", "recipe ideas", "suggestions"],
    ) && !task_contains_any(
        lower,
        &["this is", "lasted me for", "got from", "finished off"],
    ) {
        return None;
    }
    Some(CountTotalFact {
        key: focus.key.clone(),
        count,
        score: 18
            + count.max(0) as usize * 4
            + line_matches_focus_terms(lower, &focus.required_terms) * 6
            + usize::from(lower.contains("lunch")) * 4,
        evidence: line.trim().to_string(),
    })
}

pub(super) fn extract_online_course_completion_facts_from_line(
    line: &str,
    lower: &str,
) -> Vec<CountTotalFact> {
    if !(lower.starts_with("user:") || line.trim_start().starts_with('-'))
        || !lower.contains("course")
        || task_contains_any(
            lower,
            &[
                "looking for courses",
                "recommend some online resources or courses",
                "started an online course",
                "starting an online course",
            ],
        )
    {
        return Vec::new();
    }

    const PLATFORM_KEYS: &[(&str, &str)] = &[
        ("coursera", "Coursera"),
        ("edx", "edX"),
        ("udemy", "Udemy"),
        ("datacamp", "DataCamp"),
        ("kaggle", "Kaggle"),
        ("fast.ai", "Fast.ai"),
    ];

    let mut facts = Vec::new();
    for (platform_key, platform_name) in PLATFORM_KEYS {
        if !lower.contains(platform_key) {
            continue;
        }
        let platform_pattern = regex::escape(platform_key);
        let count = extract_count_from_patterns(
            line,
            &[
                &format!(
                    r"(?i)\bcompleted\s+([A-Za-z0-9,-]+)\s+courses?\s+on\s+{platform_pattern}\b"
                ),
                &format!(r"(?i)\bcompleted\s+([A-Za-z0-9,-]+)\s+{platform_pattern}\s+courses?\b"),
                &format!(
                    r"(?i)\b(?:previous|prior)\s+([A-Za-z0-9,-]+)\s+{platform_pattern}\s+courses?\b"
                ),
                &format!(r"(?i)\b([A-Za-z0-9,-]+)\s+courses?\s+on\s+{platform_pattern}\b"),
                &format!(r"(?i)\b([A-Za-z0-9,-]+)\s+{platform_pattern}\s+courses?\b"),
            ],
        );
        let Some(count) = count else {
            continue;
        };
        if !line_supports_completed_course_total(lower) {
            continue;
        }
        facts.push(CountTotalFact {
            key: platform_name.to_string(),
            count,
            score: 22
                + count.max(0) as usize * 3
                + usize::from(lower.contains("completed")) * 8
                + usize::from(lower.contains("previous")) * 6
                + usize::from(lower.contains("prior")) * 6,
            evidence: line.trim().to_string(),
        });
    }
    facts
}

fn parse_metric_bundle_query(task_lower: &str) -> Option<MetricBundleQuery> {
    let captures =
        compile_regex(r"(?i)\btotal number of (.+?) i have in (.+?)\??$").captures(task_lower)?;
    let metrics = split_bundle_items(captures.get(1)?.as_str())
        .into_iter()
        .map(|surface| parse_metric_kind(&surface))
        .collect::<Option<Vec<_>>>()?;
    if metrics.len() < 2 {
        return None;
    }
    let anchor_terms = synthetic_query_terms(captures.get(2)?.as_str());
    let mut required_terms = anchor_terms.clone();
    required_terms.extend(metrics.iter().map(|metric| metric.query_term().to_string()));
    required_terms.sort();
    required_terms.dedup();
    Some(MetricBundleQuery {
        metrics,
        anchor_terms,
        required_terms,
    })
}

fn parse_meal_bundle_query(task_lower: &str) -> Option<MealBundleQuery> {
    if !task_contains_any(task_lower, &["meal", "meals", "lunch", "lunches"])
        || !task_contains_any(task_lower, &[" from "])
    {
        return None;
    }

    let tail = task_lower
        .split_once(" from ")
        .map(|(_, tail)| tail.trim().trim_end_matches('?'))?;
    let items = split_bundle_items(tail)
        .into_iter()
        .map(|item| build_item_focus(&item))
        .filter(|focus| !focus.required_terms.is_empty())
        .collect::<Vec<_>>();
    if items.len() < 2 {
        return None;
    }

    let mut required_terms = vec![
        "meal".to_string(),
        "meals".to_string(),
        "lunch".to_string(),
        "lunches".to_string(),
    ];
    for focus in &items {
        required_terms.extend(focus.required_terms.iter().cloned());
    }
    required_terms.sort();
    required_terms.dedup();
    Some(MealBundleQuery {
        items,
        required_terms,
    })
}

fn parse_metric_kind(surface: &str) -> Option<MetricKind> {
    match surface.trim().trim_start_matches("the ").trim() {
        "goal" | "goals" => Some(MetricKind::Goals),
        "assist" | "assists" => Some(MetricKind::Assists),
        _ => None,
    }
}

fn build_item_focus(surface: &str) -> CountItemFocus {
    let display = surface.trim().trim_start_matches("the ").trim().to_string();
    CountItemFocus {
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

fn extract_count_from_patterns(line: &str, patterns: &[&str]) -> Option<i32> {
    patterns.iter().find_map(|pattern| {
        compile_regex(pattern)
            .captures(line)
            .and_then(|captures| captures.get(1))
            .and_then(|value| parse_count_token_value(value.as_str()))
    })
}

fn line_matches_anchor_terms(lower: &str, anchor_terms: &[String]) -> bool {
    if anchor_terms.is_empty() {
        return true;
    }
    let refs = anchor_terms.iter().map(String::as_str).collect::<Vec<_>>();
    term_overlap_count(lower, &refs) >= refs.len().min(2)
}

fn line_matches_focus_terms(lower: &str, focus_terms: &[String]) -> usize {
    let refs = focus_terms.iter().map(String::as_str).collect::<Vec<_>>();
    if refs.is_empty() {
        0
    } else {
        term_overlap_count(lower, &refs)
    }
}

fn line_supports_completed_course_total(lower: &str) -> bool {
    lower.contains("completed")
        || lower.contains("previous")
        || lower.contains("prior")
        || lower.contains("solid foundation")
        || lower.contains("already")
}

impl MetricKind {
    pub(super) fn key(self) -> &'static str {
        match self {
            Self::Goals => "goals",
            Self::Assists => "assists",
        }
    }

    fn query_term(self) -> &'static str {
        match self {
            Self::Goals => "goal",
            Self::Assists => "assist",
        }
    }
}
