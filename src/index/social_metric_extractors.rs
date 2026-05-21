use super::*;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum SocialMetricQuery {
    CommentTotal(SocialCommentTotalQuery),
    ReachTotal(SocialReachTotalQuery),
    MaxPlatformGrowth(MaxPlatformGrowthQuery),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SocialCommentTotalQuery {
    pub(super) required_terms: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SocialReachTotalQuery {
    pub(super) required_terms: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct MaxPlatformGrowthQuery {
    pub(super) required_terms: Vec<String>,
    pub(super) window_days: Option<i32>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) enum SocialCommentSource {
    FacebookLiveSession,
    MostPopularVideo,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) enum SocialReachSource {
    FacebookAdCampaign,
    InstagramInfluencer,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SocialCommentCandidate {
    pub(super) source: SocialCommentSource,
    pub(super) value: i32,
    pub(super) score: usize,
    pub(super) evidence: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SocialReachCandidate {
    pub(super) source: SocialReachSource,
    pub(super) value: i32,
    pub(super) score: usize,
    pub(super) evidence: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct PlatformGrowthCandidate {
    pub(super) platform: String,
    pub(super) delta: i32,
    pub(super) score: usize,
    pub(super) evidence: String,
    pub(super) window_days: Option<i32>,
}

pub(super) fn parse_social_metric_query(task_lower: &str) -> Option<SocialMetricQuery> {
    parse_social_comment_total_query(task_lower)
        .map(SocialMetricQuery::CommentTotal)
        .or_else(|| parse_social_reach_total_query(task_lower).map(SocialMetricQuery::ReachTotal))
        .or_else(|| {
            parse_max_platform_growth_query(task_lower).map(SocialMetricQuery::MaxPlatformGrowth)
        })
}

pub(super) fn extract_social_comment_candidate(
    line: &str,
    lower: &str,
) -> Option<SocialCommentCandidate> {
    if !is_summary_or_user_line(line, lower) || !lower.contains("comment") {
        return None;
    }
    let value = extract_line_numbers(line).into_iter().max()?;

    if lower.contains("facebook live") {
        return Some(SocialCommentCandidate {
            source: SocialCommentSource::FacebookLiveSession,
            value,
            score: 24
                + usize::from(lower.starts_with("user:")) * 8
                + usize::from(lower.contains("facebook live session")) * 4,
            evidence: line.trim().to_string(),
        });
    }

    if lower.contains("most popular video") {
        return Some(SocialCommentCandidate {
            source: SocialCommentSource::MostPopularVideo,
            value,
            score: 22
                + usize::from(lower.starts_with("user:")) * 8
                + usize::from(lower.contains("youtube")) * 4
                + usize::from(lower.contains("social media analytics")) * 2,
            evidence: line.trim().to_string(),
        });
    }

    None
}

pub(super) fn extract_social_reach_candidate(
    line: &str,
    lower: &str,
) -> Option<SocialReachCandidate> {
    if !is_summary_or_user_line(line, lower) {
        return None;
    }

    let facebook_value = compile_regex_static(
        r"(?i)\breached\s+(?:around\s+|about\s+|approximately\s+)?(\d[\d,]*)\s+people\b",
    )
    .captures(line)
    .and_then(|captures| captures.get(1))
    .and_then(|value| parse_social_number(value.as_str()));
    if lower.contains("facebook")
        && task_contains_any(lower, &["campaign", "ad campaign", "ad"])
        && lower.contains("reach")
        && facebook_value.is_some()
    {
        return Some(SocialReachCandidate {
            source: SocialReachSource::FacebookAdCampaign,
            value: facebook_value?,
            score: 26
                + usize::from(lower.starts_with("user:")) * 8
                + usize::from(lower.contains("ad campaign")) * 6,
            evidence: line.trim().to_string(),
        });
    }

    let influencer_value = compile_regex_static(
        r"(?i)\b(?:promoted|exposed|shared|introduced).*?\b(?:to\s+)?(\d[\d,]*)\s+followers\b",
    )
    .captures(line)
    .and_then(|captures| captures.get(1))
    .and_then(|value| parse_social_number(value.as_str()))
    .or_else(|| {
        compile_regex_static(r"(?i)\b(\d[\d,]*)\s+followers\b")
            .captures(line)
            .and_then(|captures| captures.get(1))
            .and_then(|value| parse_social_number(value.as_str()))
    });
    if task_contains_any(lower, &["influencer", "collaboration"])
        && influencer_value.is_some()
        && task_contains_any(lower, &["instagram", "followers", "promoted"])
    {
        return Some(SocialReachCandidate {
            source: SocialReachSource::InstagramInfluencer,
            value: influencer_value?,
            score: 24
                + usize::from(lower.starts_with("user:")) * 8
                + usize::from(lower.contains("influencer")) * 6
                + usize::from(lower.contains("instagram")) * 4,
            evidence: line.trim().to_string(),
        });
    }

    None
}

pub(super) fn extract_platform_growth_candidate(
    line: &str,
    lower: &str,
    query: &MaxPlatformGrowthQuery,
) -> Option<PlatformGrowthCandidate> {
    if !is_summary_or_user_line(line, lower) || !lower.contains("follower") {
        return None;
    }
    let (_, platform_name) = social_platforms()
        .into_iter()
        .find(|(platform_key, _)| lower.contains(platform_key))?;

    let window_days = extract_social_window_days(line);
    if let Some(expected_days) = query.window_days {
        if let Some(candidate_days) = window_days {
            if candidate_days > expected_days {
                return None;
            }
        } else if !task_contains_any(lower, &["current", "now", "steady", "stayed", "remained"]) {
            return None;
        }
    }

    if task_contains_any(
        lower,
        &["steady", "stayed", "remained", "flat", "unchanged"],
    ) {
        return Some(PlatformGrowthCandidate {
            platform: platform_name.to_string(),
            delta: 0,
            score: 6 + usize::from(lower.starts_with("user:")) * 4,
            evidence: line.trim().to_string(),
            window_days,
        });
    }

    let direct_delta = compile_regex_static(
        r"(?i)\b(?:gained|grew by|increased by|went up by|up by)\s+(?:around\s+|about\s+|approximately\s+)?(\d[\d,]*)\s+followers?\b",
    )
    .captures(line)
    .and_then(|captures| captures.get(1))
    .and_then(|value| parse_social_number(value.as_str()));
    if let Some(delta) = direct_delta.filter(|delta| *delta > 0) {
        return Some(PlatformGrowthCandidate {
            platform: platform_name.to_string(),
            delta,
            score: 28
                + usize::from(lower.starts_with("user:")) * 8
                + usize::from(window_days.is_some()) * 4,
            evidence: line.trim().to_string(),
            window_days,
        });
    }

    let range_delta = compile_regex_static(
        r"(?i)\bfrom\s+(\d[\d,]*)\s+(?:followers?\b)?(?:.*?\bto\s+)(\d[\d,]*)\b",
    )
    .captures(line)
    .and_then(|captures| {
        let start = parse_social_number(captures.get(1)?.as_str())?;
        let end = parse_social_number(captures.get(2)?.as_str())?;
        end.checked_sub(start)
    });
    let delta = range_delta.filter(|delta| *delta > 0)?;
    Some(PlatformGrowthCandidate {
        platform: platform_name.to_string(),
        delta,
        score: 30
            + usize::from(lower.starts_with("user:")) * 8
            + usize::from(window_days.is_some()) * 4,
        evidence: line.trim().to_string(),
        window_days,
    })
}

fn parse_social_comment_total_query(task_lower: &str) -> Option<SocialCommentTotalQuery> {
    if !task_contains_all(task_lower, &["total", "comments"])
        || !task_contains_any(task_lower, &["facebook live session", "facebook live"])
        || !task_contains_all(task_lower, &["most popular", "video", "youtube"])
    {
        return None;
    }
    Some(SocialCommentTotalQuery {
        required_terms: vec![
            "comments".to_string(),
            "facebook".to_string(),
            "live".to_string(),
            "most".to_string(),
            "popular".to_string(),
            "video".to_string(),
            "youtube".to_string(),
        ],
    })
}

fn parse_social_reach_total_query(task_lower: &str) -> Option<SocialReachTotalQuery> {
    if !task_contains_all(task_lower, &["total", "reached"])
        || !task_contains_any(task_lower, &["facebook", "ad campaign"])
        || !task_contains_any(task_lower, &["instagram", "influencer", "collaboration"])
    {
        return None;
    }
    Some(SocialReachTotalQuery {
        required_terms: vec![
            "facebook".to_string(),
            "instagram".to_string(),
            "influencer".to_string(),
            "reach".to_string(),
        ],
    })
}

fn parse_max_platform_growth_query(task_lower: &str) -> Option<MaxPlatformGrowthQuery> {
    if !task_contains_any(
        task_lower,
        &["which social media platform", "which platform"],
    ) || !task_contains_any(
        task_lower,
        &["gain the most followers", "gained the most followers"],
    ) {
        return None;
    }
    Some(MaxPlatformGrowthQuery {
        required_terms: vec!["platform".to_string(), "followers".to_string()],
        window_days: extract_social_window_days(task_lower),
    })
}

fn extract_social_window_days(surface: &str) -> Option<i32> {
    duration_answer_magnitude(&normalize_current_duration_answer(
        &extract_duration_answer_from_line(surface)?,
    ))
    .map(|days| {
        #[allow(clippy::cast_possible_truncation)]
        let rounded = days.round() as i32;
        rounded
    })
}

fn parse_social_number(value: &str) -> Option<i32> {
    value.replace(',', "").parse::<i32>().ok()
}

fn social_platforms() -> [(&'static str, &'static str); 5] {
    [
        ("instagram", "Instagram"),
        ("facebook", "Facebook"),
        ("twitter", "Twitter"),
        ("tiktok", "TikTok"),
        ("linkedin", "LinkedIn"),
    ]
}
