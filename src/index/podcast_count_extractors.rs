use super::*;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct PodcastEpisodeTotalQuery {
    pub(super) titles: Vec<PodcastTitleFocus>,
    pub(super) required_terms: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct PodcastTitleFocus {
    pub(super) key: String,
    pub(super) display: String,
    pub(super) required_terms: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct PodcastEpisodeFact {
    pub(super) key: String,
    pub(super) count: i32,
    pub(super) score: usize,
    pub(super) evidence: String,
}

pub(super) fn parse_podcast_episode_total_query(
    task: &str,
    task_lower: &str,
) -> Option<PodcastEpisodeTotalQuery> {
    if !detect_counting_query(task)
        || !task_contains_any(task_lower, &["episode", "episodes"])
        || !task_contains_any(task_lower, &["listened"])
    {
        return None;
    }

    let titles = extract_quoted_titles(task)
        .into_iter()
        .map(|title| build_title_focus(&title))
        .filter(|focus| !focus.key.is_empty() && !focus.required_terms.is_empty())
        .collect::<Vec<_>>();
    if titles.len() < 2 {
        return None;
    }

    let mut required_terms = vec![
        "episode".to_string(),
        "episodes".to_string(),
        "listened".to_string(),
        "podcast".to_string(),
    ];
    for focus in &titles {
        required_terms.extend(focus.required_terms.iter().cloned());
    }
    required_terms.sort();
    required_terms.dedup();

    Some(PodcastEpisodeTotalQuery {
        titles,
        required_terms,
    })
}

pub(super) fn extract_podcast_episode_fact_from_line(
    line: &str,
    lower: &str,
    focus: &PodcastTitleFocus,
) -> Option<PodcastEpisodeFact> {
    if !lower.starts_with("user:") {
        return None;
    }

    let overlap = line_matches_focus_terms(lower, &focus.required_terms);
    let min_overlap = if focus.required_terms.len() >= 2 {
        2
    } else {
        1
    };
    if overlap < min_overlap {
        return None;
    }

    let count = extract_episode_count(line)?;
    Some(PodcastEpisodeFact {
        key: focus.key.clone(),
        count,
        score: 20
            + count.max(0) as usize * 4
            + overlap * 8
            + usize::from(lower.contains("finished")) * 6
            + usize::from(lower.contains("listening")) * 4
            + usize::from(lower.contains("episode")) * 3,
        evidence: line.trim().to_string(),
    })
}

fn build_title_focus(title: &str) -> PodcastTitleFocus {
    const TITLE_STOP: &[&str] = &["a", "an", "i", "my", "of", "the", "this"];
    let display = title.trim().to_string();
    let lower = display.to_ascii_lowercase();
    let mut required_terms = synthetic_query_terms(&lower)
        .into_iter()
        .filter(|term| !TITLE_STOP.contains(&term.as_str()))
        .collect::<Vec<_>>();
    if required_terms.is_empty() {
        required_terms = lower
            .split_whitespace()
            .filter(|term| !term.is_empty())
            .map(ToString::to_string)
            .collect();
    }
    required_terms.sort();
    required_terms.dedup();
    PodcastTitleFocus {
        key: normalized_synthetic_phrase_key(&display),
        display,
        required_terms,
    }
}

fn extract_quoted_titles(text: &str) -> Vec<String> {
    let mut titles = Vec::new();
    let mut seen = HashSet::new();
    for captures in
        compile_regex(r#"(?:"([^"]+)"|(?:^|[^A-Za-z0-9])'([^']+)')"#).captures_iter(text)
    {
        let Some(value) = captures
            .get(1)
            .or_else(|| captures.get(2))
            .map(|value| value.as_str().trim())
        else {
            continue;
        };
        if value.is_empty() {
            continue;
        }
        let key = normalized_synthetic_phrase_key(value);
        if !key.is_empty() && seen.insert(key) {
            titles.push(value.to_string());
        }
    }
    titles
}

fn extract_episode_count(line: &str) -> Option<i32> {
    [
        r"(?i)\bfinished\s+(?:around\s+)?([A-Za-z0-9,-]+)\s+episodes?\b",
        r"(?i)\b(?:around\s+)?([A-Za-z0-9,-]+)\s+episodes?\s+so far\b",
        r"(?i)\blistened to\s+(?:around\s+)?([A-Za-z0-9,-]+)\s+episodes?\b",
        r"(?i)\bepisode\s+([A-Za-z0-9,-]+)\b",
    ]
    .into_iter()
    .find_map(|pattern| {
        compile_regex(pattern)
            .captures(line)
            .and_then(|captures| captures.get(1))
            .and_then(|value| parse_count_token_value(value.as_str()))
    })
}

fn line_matches_focus_terms(lower: &str, focus_terms: &[String]) -> usize {
    let refs = focus_terms.iter().map(String::as_str).collect::<Vec<_>>();
    if refs.is_empty() {
        0
    } else {
        term_overlap_count(lower, &refs)
    }
}
