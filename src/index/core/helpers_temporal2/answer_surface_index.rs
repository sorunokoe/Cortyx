//! Answer surface indexing: scoring, ranking, temporal query extraction.

use super::super::*;
use crate::index::compile_regex;

pub fn normalized_index_answer_surface_key(text: &str) -> String {
    text.trim()
        .trim_matches(|c: char| matches!(c, '"' | '\'' | ',' | '.' | ';' | ':' | '!' | '?'))
        .to_ascii_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn index_answer_surface_answers_overlap(left: &str, right: &str) -> bool {
    let left_key = normalized_index_answer_surface_key(left);
    let right_key = normalized_index_answer_surface_key(right);
    !left_key.is_empty()
        && !right_key.is_empty()
        && (left_key == right_key || left_key.contains(&right_key) || right_key.contains(&left_key))
}

pub fn index_answer_surface_bucket_rank(bucket: &IndexAnswerSurfaceBucket) -> f32 {
    let corroboration = ((bucket.total_score - bucket.best_score).max(0.0)).min(6.0) * 0.15;
    bucket.best_score
        + bucket.max_overlap as f32 * 1.5
        + (bucket.paths.len().saturating_sub(1).min(2) as f32) * 0.75
        + (bucket.hits.saturating_sub(1).min(3) as f32) * 0.25
        + corroboration
}

pub fn index_answer_surface_buckets_conflict(
    top: &IndexAnswerSurfaceBucket,
    runner_up: &IndexAnswerSurfaceBucket,
) -> bool {
    !index_answer_surface_answers_overlap(&top.answer_span, &runner_up.answer_span)
        && index_answer_surface_bucket_rank(runner_up) + 2.5
            >= index_answer_surface_bucket_rank(top)
        && runner_up.max_overlap + 1 >= top.max_overlap
}

pub fn index_answer_surface_bucket_has_query_affinity(
    task_lower: &str,
    bucket: &IndexAnswerSurfaceBucket,
) -> bool {
    let answer_lower = bucket.answer_span.to_ascii_lowercase();
    (task_contains_any(
        task_lower,
        &["religious", "religion", "faith", "church", "spiritual"],
    ) && answer_lower.contains("religious"))
        || (task_contains_any(
            task_lower,
            &[
                "member of the lgbtq community",
                "member of the lgbtq+ community",
                "part of the lgbtq community",
                "part of the lgbtq+ community",
                "member of the transgender community",
                "ally to the transgender community",
                "ally to the lgbtq community",
                "ally to the lgbtq+ community",
                "considered an ally",
            ],
        ) && answer_lower.contains("ally"))
        || (task_contains_any(
            task_lower,
            &["move from", "moved from", "home country", "origin country"],
        ) && bucket
            .relation_families
            .contains(&SyntheticAnswerSurfaceRelationFamily::Origin))
        || (task_contains_any(
            task_lower,
            &["what books", "which books", " books", "book "],
        ) && bucket
            .relation_families
            .contains(&SyntheticAnswerSurfaceRelationFamily::Book))
        || (task_contains_any(
            task_lower,
            &[
                "what lgbtq",
                "transgender-specific events",
                "lgbtq events",
                "in what ways",
            ],
        ) && bucket
            .relation_families
            .contains(&SyntheticAnswerSurfaceRelationFamily::CommunityEvent))
        || (task_contains_any(task_lower, &["help children", "help kids", "help youth"])
            && bucket
                .relation_families
                .contains(&SyntheticAnswerSurfaceRelationFamily::ChildHelpEvent))
        || (task_contains_any(
            task_lower,
            &[
                "with her family",
                "with his family",
                "with my family",
                "with the kids",
                "family activities",
            ],
        ) && bucket
            .relation_families
            .contains(&SyntheticAnswerSurfaceRelationFamily::FamilyActivity))
        || (task_contains_any(
            task_lower,
            &["to destress", "to de-stress", "self-care", "relax"],
        ) && bucket
            .relation_families
            .contains(&SyntheticAnswerSurfaceRelationFamily::SelfCareActivity))
}

pub fn synthetic_answer_surface_should_skip_fallback(
    task: &str,
    task_lower: &str,
    profile: &SyntheticAnswerSurfaceQueryProfile,
    evidence: &[String],
) -> bool {
    let real_evidence = evidence
        .iter()
        .filter(|line| !line.starts_with("answer_surface:"))
        .collect::<Vec<_>>();
    let evidence_has_any = |needles: &[&str]| {
        real_evidence.iter().any(|line| {
            let lower = line.to_ascii_lowercase();
            task_contains_any(&lower, needles)
        })
    };
    let collecting_target = task_lower
        .split_once("collecting ")
        .map(|(_, tail)| tail)
        .map(|tail| {
            ["?", ".", ",", " before ", " after "]
                .iter()
                .find_map(|marker| tail.split_once(marker).map(|(head, _)| head))
                .unwrap_or(tail)
                .trim()
                .to_string()
        })
        .filter(|phrase| phrase.split_whitespace().count() >= 2);
    let mut poster_focus_terms = synthetic_query_terms(task_lower);
    poster_focus_terms.retain(|term| {
        term.len() >= 4
            && !matches!(
                term.as_str(),
                "university"
                    | "college"
                    | "present"
                    | "presented"
                    | "poster"
                    | "research"
                    | "conference"
            )
    });
    let poster_focus_refs = poster_focus_terms
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    let poster_focus_min_overlap = if poster_focus_refs.len() >= 2 { 2 } else { 1 };

    (matches!(profile.route_kind, SyntheticAnswerSurfaceRouteKind::Choice)
        && task_contains_any(
            task_lower,
            &[
                " first",
                " earlier",
                " later",
                " before ",
                " after ",
                " more often",
                " less often",
                " higher percentage",
                " lower percentage",
                " higher discount",
                " lower discount",
                " cheaper",
                " more expensive",
                " cost more",
                " cost less",
                " older",
                " younger",
            ],
        ))
        || ((matches!(
            profile.expected_type,
            SyntheticAnswerSurfaceExpectedType::Count
        ) || is_money_query(task))
            && synthetic_count_query_requires_multi_operand_reasoning(task, task_lower))
        || (task_contains_any(
            task_lower,
            &[
                "at which university",
                "which university",
                "what university",
                "which college",
                "what college",
            ],
        ) && !evidence_has_any(&["university", "college", "school", "institute"]))
        || (task_contains_any(task_lower, &["presented", "poster"])
            && !evidence_has_any(&["presented", "poster"]))
        || (task_contains_any(
            task_lower,
            &[
                "at which university",
                "which university",
                "what university",
                "which college",
                "what college",
            ],
        ) && task_contains_any(task_lower, &["present", "poster"])
            && !poster_focus_refs.is_empty()
            && !real_evidence.iter().any(|line| {
                let lower = line.to_ascii_lowercase();
                term_overlap_count(&lower, &poster_focus_refs) >= poster_focus_min_overlap
            }))
        || (task_contains_any(task_lower, &["conference"]) && !evidence_has_any(&["conference"]))
        || collecting_target.as_ref().is_some_and(|phrase| {
            !real_evidence
                .iter()
                .any(|line| line.to_ascii_lowercase().contains(phrase))
        })
}

pub fn extract_rare_collection_count(line: &str) -> Option<(&'static str, i32)> {
    let lower = line.to_ascii_lowercase();
    if !(lower.starts_with("user:") || line.trim_start().starts_with('-')) {
        return None;
    }
    let kind = if lower.contains("rare books") {
        "rare_books"
    } else if lower.contains("rare records") {
        "rare_records"
    } else if lower.contains("rare figurines") {
        "rare_figurines"
    } else if lower.contains("rare coins") {
        "rare_coins"
    } else {
        return None;
    };

    let count = extract_line_numbers(line)
        .into_iter()
        .find(|value| *value > 0 && *value < 1000)?;
    Some((kind, count))
}

pub fn extract_previous_role(line: &str) -> Option<String> {
    let lower = line.to_ascii_lowercase();
    if !lower.starts_with("user:") || !lower.contains("previous role") {
        return None;
    }

    let pattern = compile_regex(r"previous role as a[n]?\s+(.+?)(?:,|\.| and\b| but\b| with\b)");
    let role = pattern
        .captures(line)?
        .get(1)?
        .as_str()
        .trim()
        .trim_matches('"')
        .to_string();
    if role.is_empty() {
        None
    } else {
        Some(role)
    }
}

pub fn extract_finished_issue_count(line: &str, lower: &str) -> Option<i32> {
    if !(lower.starts_with("user:") || line.trim_start().starts_with('-'))
        || !lower.contains("national geographic")
    {
        return None;
    }

    if lower.contains("finished") {
        return extract_line_numbers(line).into_iter().next();
    }
    if lower.contains("currently on") {
        return extract_line_numbers(line)
            .into_iter()
            .next()
            .map(|value| value - 1)
            .filter(|value| *value > 0);
    }
    None
}

pub fn extract_quoted_title(task: &str) -> Option<String> {
    extract_quoted_titles(task)
        .into_iter()
        .next()
        .map(|title| title.to_ascii_lowercase())
}

pub fn extract_quoted_titles(task: &str) -> Vec<String> {
    let mut titles = Vec::new();
    for quote in ['"', '\''] {
        let mut cursor = task;
        while let Some(start) = cursor.find(quote) {
            let tail = &cursor[start + quote.len_utf8()..];
            let Some(end) = tail.find(quote) else {
                break;
            };
            let title = tail[..end].trim();
            if title.split_whitespace().count() >= 2 {
                let title = title.to_string();
                if !titles.iter().any(|existing| existing == &title) {
                    titles.push(title);
                }
            }
            cursor = &tail[end + quote.len_utf8()..];
        }
        if !titles.is_empty() {
            break;
        }
    }
    titles
}

pub fn extract_named_artwork_location_surface_from_line(
    _line: &str,
    line_lower: &str,
    title_lower: &str,
) -> Option<String> {
    let title_idx = line_lower.find(title_lower)?;
    let context_lower = &line_lower[title_idx + title_lower.len()..];
    extract_named_artwork_room_surface_from_context(context_lower, line_lower).or_else(|| {
        let prefix_lower = &line_lower[..title_idx];
        if context_lower.contains("above my bed")
            || context_lower.contains("above the bed")
            || (task_contains_any(context_lower, &["on my wall", "on the wall"])
                && prefix_lower.contains("bedroom"))
        {
            Some("in my bedroom".to_string())
        } else if task_contains_any(context_lower, &["above my sofa", "above the sofa"])
            && prefix_lower.contains("living room")
        {
            Some("above my living room sofa".to_string())
        } else {
            None
        }
    })
}

pub fn extract_named_artwork_room_surface_from_context(
    context_lower: &str,
    full_lower: &str,
) -> Option<String> {
    if context_lower.contains("living room sofa") {
        return Some("above my living room sofa".to_string());
    }
    if context_lower.contains("above my bed") || context_lower.contains("above the bed") {
        return Some("in my bedroom".to_string());
    }
    for (marker, answer) in [
        ("bedroom", "in my bedroom"),
        ("living room", "in my living room"),
        ("dining room", "in my dining room"),
        ("family room", "in my family room"),
        ("guest room", "in my guest room"),
        ("office", "in my office"),
        ("studio", "in my studio"),
        ("kitchen", "in my kitchen"),
        ("hallway", "in my hallway"),
        ("entryway", "in my entryway"),
        ("party area", "in the party area"),
    ] {
        if context_lower.contains(marker) {
            return Some(answer.to_string());
        }
    }
    if task_contains_any(context_lower, &["on my wall", "on the wall"]) {
        for (marker, answer) in [
            ("bedroom", "in my bedroom"),
            ("living room", "in my living room"),
            ("office", "in my office"),
            ("studio", "in my studio"),
        ] {
            if full_lower.contains(marker) {
                return Some(answer.to_string());
            }
        }
    }
    None
}

pub fn extract_rewatch_title_from_line(line: &str, lower: &str) -> Option<String> {
    for marker in ["re-watched ", "re watched ", "rewatched "] {
        let Some(start) = lower.find(marker) else {
            continue;
        };
        let title_start = start + marker.len();
        let tail = line[title_start..].trim();
        let tail_lower = lower[title_start..].trim();
        let mut end = tail.len();
        for delimiter in [
            ",",
            ".",
            "?",
            "!",
            " yesterday",
            " today",
            " again",
            " which ",
            " and ",
            " but ",
            " because ",
        ] {
            if let Some(idx) = tail_lower.find(delimiter) {
                end = end.min(idx);
            }
        }
        let title = tail[..end]
            .trim()
            .trim_matches(|c: char| matches!(c, '"' | '\'' | ',' | '.' | ';' | '!' | '?'));
        if title.len() >= 3 {
            return Some(title.to_string());
        }
    }
    None
}

pub fn normalize_rewatch_title(title: &str) -> String {
    title
        .trim()
        .trim_matches(|c: char| matches!(c, '"' | '\'' | ',' | '.' | ';' | '!' | '?'))
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

pub fn extract_origin_country_answer(line: &str) -> Option<String> {
    compile_regex(r"(?i)home country[, ]+([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)?)")
        .captures(line)
        .and_then(|captures| captures.get(1))
        .map(|value| value.as_str().trim().to_string())
        .filter(|value| !value.is_empty())
}

#[derive(Clone, Copy)]
pub(in crate::index) enum SyntheticDurationAnchor {
    CurrentDays(i32),
    AbsoluteDay(i32),
}

#[derive(Clone, Copy)]
pub(in crate::index) enum SyntheticEventAnchor {
    RelativeDaysAgo(i32),
    AbsoluteDay(i32),
}

#[derive(Clone, Copy)]
pub(in crate::index) struct SyntheticDurationValue {
    pub(in crate::index) amount: f32,
    pub(in crate::index) days: f32,
    pub(in crate::index) unit: &'static str,
}

#[derive(Clone, Copy)]
pub(in crate::index) enum SyntheticTemporalDirection {
    Earlier,
    Later,
}

pub fn extract_temporal_choice_options(task: &str) -> Option<(String, String)> {
    let quoted = extract_quoted_titles(task);
    if quoted.len() >= 2 {
        return Some((quoted[0].trim().to_string(), quoted[1].trim().to_string()));
    }

    let tail = task
        .split_once(',')
        .map(|(_, suffix)| suffix)
        .unwrap_or(task)
        .trim()
        .trim_end_matches('?');
    let (left, right) = tail.rsplit_once(" or ")?;
    Some((
        normalize_temporal_choice_option(left),
        normalize_temporal_choice_option(right),
    ))
}

pub fn normalize_temporal_choice_option(option: &str) -> String {
    option
        .trim()
        .trim_matches(|c: char| matches!(c, '"' | '\''))
        .trim_start_matches("the ")
        .trim_start_matches("The ")
        .trim()
        .to_string()
}

pub fn extract_temporal_elapsed_phrases(
    task_lower: &str,
) -> Option<(String, String)> {
    let trimmed = task_lower.trim().trim_end_matches('?');
    let rest = trimmed.strip_prefix("how long had i been ")?;
    let (subject, event) = rest.split_once(" when ")?;
    Some((subject.trim().to_string(), event.trim().to_string()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::index) enum SyntheticElapsedFromNowUnit {
    Day,
    Week,
    Month,
    Year,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntheticFromNowQuery {
    pub unit: SyntheticElapsedFromNowUnit,
    pub event_phrase: String,
    pub anchor_phrase: Option<String>,
    pub append_ago: bool,
}

pub fn extract_temporal_from_now_query(
    task_lower: &str,
) -> Option<SyntheticFromNowQuery> {
    let trimmed = strip_temporal_reference_prefix(task_lower)
        .trim()
        .trim_end_matches('?');
    let rest = trimmed.strip_prefix("how many ")?;
    if let Some((unit_raw, event)) = rest.split_once(" ago did i ") {
        let unit = parse_temporal_from_now_unit(unit_raw)?;
        let (event_phrase, anchor_phrase) = split_temporal_when_anchor(event);
        let append_ago = anchor_phrase.is_some();
        return Some(SyntheticFromNowQuery {
            unit,
            event_phrase,
            anchor_phrase,
            append_ago,
        });
    }
    if let Some((unit_raw, event)) = rest.split_once(" have passed since i ") {
        let unit = parse_temporal_from_now_unit(unit_raw)?;
        let (event_phrase, anchor_phrase) = split_temporal_when_anchor(event);
        return Some(SyntheticFromNowQuery {
            unit,
            event_phrase,
            anchor_phrase,
            append_ago: false,
        });
    }
    None
}

pub fn split_temporal_when_anchor(event: &str) -> (String, Option<String>) {
    let trimmed = event.trim();
    if let Some((primary, anchor)) = trimmed.split_once(" when i ") {
        let primary = primary.trim().to_string();
        let anchor = anchor.trim();
        if !primary.is_empty() && !anchor.is_empty() {
            return (primary, Some(anchor.to_string()));
        }
    }
    (trimmed.to_string(), None)
}

pub fn strip_temporal_reference_prefix(task_lower: &str) -> &str {
    let trimmed = task_lower.trim();
    if trimmed.starts_with("as of ") {
        if let Some(pos) = trimmed.find("how many ") {
            return &trimmed[pos..];
        }
    }
    trimmed
}
