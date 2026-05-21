use super::*;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct PaperSubmissionQuery {
    pub(super) topic_terms: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct PaperVenueCandidate {
    pub(super) venue: String,
    pub(super) score: usize,
    pub(super) evidence: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct VenueDateCandidate {
    pub(super) date: String,
    pub(super) score: usize,
    pub(super) evidence: String,
}

pub(super) fn parse_paper_submission_query(task_lower: &str) -> Option<PaperSubmissionQuery> {
    if !task_lower.starts_with("when did i submit") || !task_lower.contains("paper") {
        return None;
    }
    let topic_terms = topic_terms_after(task_lower, "paper on ")
        .or_else(|| topic_terms_after(task_lower, "research paper on "))
        .filter(|terms| !terms.is_empty())?;
    Some(PaperSubmissionQuery { topic_terms })
}

pub(super) fn extract_paper_submission_venue(
    line: &str,
    lower: &str,
    query: &PaperSubmissionQuery,
) -> Option<PaperVenueCandidate> {
    if !lower.starts_with("user:")
        || !lower.contains("submitted")
        || !lower.contains("paper")
        || synthetic_answer_surface_overlap_count(
            &synthetic_answer_surface_term_key_set(&synthetic_query_terms(lower)),
            &synthetic_answer_surface_term_key_set(&query.topic_terms),
        ) == 0
    {
        return None;
    }
    let venue =
        compile_regex_static(r"(?i)\bsubmitted(?:\s+\w+){0,6}\s+to\s+([A-Za-z][A-Za-z0-9.\-]+)\b")
            .captures(line)
            .and_then(|captures| captures.get(1))
            .map(|value| value.as_str().trim().to_string())?;
    Some(PaperVenueCandidate {
        venue,
        score: 28 + query.topic_terms.len() * 3,
        evidence: line.trim().to_string(),
    })
}

pub(super) fn extract_venue_submission_date(
    line: &str,
    lower: &str,
    venue: &str,
) -> Option<VenueDateCandidate> {
    if !is_summary_or_user_line(line, lower) || !lower.contains(&venue.to_ascii_lowercase()) {
        return None;
    }
    if !task_contains_any(lower, &["submission date", "deadline", "due", "submitted"]) {
        return None;
    }
    let date = extract_date_or_time_answer_from_line(line)?;
    Some(VenueDateCandidate {
        date,
        score: 22
            + usize::from(lower.starts_with("user:")) * 8
            + usize::from(lower.contains("submission date")) * 6,
        evidence: line.trim().to_string(),
    })
}

fn topic_terms_after(task_lower: &str, marker: &str) -> Option<Vec<String>> {
    let topic = task_lower.split_once(marker)?.1;
    let stop_at = topic.find('?').unwrap_or(topic.len());
    let trimmed = topic[..stop_at].trim();
    let terms = synthetic_query_terms(trimmed)
        .into_iter()
        .filter(|term| term.len() >= 4)
        .filter(|term| !["paper", "research", "submit", "submitted"].contains(&term.as_str()))
        .collect::<Vec<_>>();
    Some(terms)
}
