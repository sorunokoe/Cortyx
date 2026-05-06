use super::count_support::SignatureDetail;
use super::*;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct DinnerPartyCountQuery {
    pub(super) required_terms: Vec<String>,
}

pub(super) fn parse_dinner_party_count_query(
    task: &str,
    task_lower: &str,
) -> Option<DinnerPartyCountQuery> {
    if !detect_counting_query(task)
        || !task_lower.contains("dinner part")
        || !task_contains_any(task_lower, &["attended", "have i attended", "past month"])
    {
        return None;
    }

    Some(DinnerPartyCountQuery {
        required_terms: vec![
            "dinner".to_string(),
            "party".to_string(),
            "place".to_string(),
            "bbq".to_string(),
            "potluck".to_string(),
            "feast".to_string(),
            "recently".to_string(),
        ],
    })
}

pub(super) fn extract_dinner_party_attendance_details(
    line: &str,
    lower: &str,
) -> Vec<SignatureDetail> {
    if !lower.starts_with("user:")
        || !contains_recent_marker(lower)
        || !contains_gathering_cue(lower)
    {
        return Vec::new();
    }

    compile_regex(r"\bat\s+([A-Z][a-z]+)'s place\b")
        .captures_iter(line)
        .filter_map(|captures| captures.get(1).map(|value| value.as_str()))
        .map(|name| {
            SignatureDetail::new(
                normalized_synthetic_phrase_key(name),
                format!("{name}'s place"),
            )
        })
        .collect()
}

fn contains_recent_marker(lower: &str) -> bool {
    task_contains_any(
        lower,
        &[
            "yesterday",
            "last week",
            "last weekend",
            "two weeks ago",
            "recently",
        ],
    )
}

fn contains_gathering_cue(lower: &str) -> bool {
    task_contains_any(
        lower,
        &[
            "dinner party",
            "bbq",
            "potluck",
            "feast",
            "board game",
            "board games",
        ],
    )
}
