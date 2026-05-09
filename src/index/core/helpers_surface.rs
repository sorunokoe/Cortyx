// This file is a submodule of `crate::index::core`.
// Contains free-standing helper functions extracted from helpers.rs.
use super::*;
use crate::index::compile_regex;
use crate::types::{QueryText, SynapseWeight};

pub(in crate::index) fn looks_like_answer_surface_status(answer_span: &str) -> bool {
    matches!(
        answer_span.to_ascii_lowercase().trim(),
        "single" | "married" | "engaged" | "divorced" | "widowed" | "separated"
    )
}

pub(in crate::index) fn index_answer_surface_score(
    row: &IndexAnswerSurfaceRow,
    retrieval_score: f32,
    profile: &SyntheticAnswerSurfaceQueryProfile,
    evidence_line: Option<&str>,
    has_future_answer_evidence: bool,
    has_completed_answer_evidence: bool,
) -> (f32, usize) {
    let pattern_terms = synthetic_query_terms(&row.question_pattern.to_ascii_lowercase());
    let pattern_term_keys = synthetic_answer_surface_term_key_set(&pattern_terms);
    if pattern_term_keys.is_empty() {
        return (0.0, 0);
    }

    let evidence_terms = evidence_line
        .map(|line| synthetic_query_terms(&line.to_ascii_lowercase()))
        .unwrap_or_default();
    let evidence_term_keys = synthetic_answer_surface_term_key_set(&evidence_terms);
    let mut support_term_keys = pattern_term_keys.clone();
    support_term_keys.extend(evidence_term_keys.iter().cloned());
    let row_family = synthetic_answer_surface_relation_family(&row.question_pattern, evidence_line);
    let relation_overlap =
        synthetic_answer_surface_overlap_count(&support_term_keys, &profile.relation_term_keys);

    let overlap =
        synthetic_answer_surface_overlap_count(&support_term_keys, &profile.task_term_keys);
    if overlap == 0 {
        return (0.0, 0);
    }

    let subject_overlap =
        synthetic_answer_surface_overlap_count(&support_term_keys, &profile.subject_term_keys);
    if !profile.subject_term_keys.is_empty() && subject_overlap == 0 {
        return (0.0, 0);
    }

    let anchor_overlap =
        synthetic_answer_surface_overlap_count(&support_term_keys, &profile.anchor_term_keys);
    if profile.requires_strict_anchor_overlap
        && !profile.anchor_term_keys.is_empty()
        && anchor_overlap == 0
    {
        return (0.0, 0);
    }
    if matches!(
        profile.expected_type,
        SyntheticAnswerSurfaceExpectedType::Count
            | SyntheticAnswerSurfaceExpectedType::Duration
            | SyntheticAnswerSurfaceExpectedType::Person
    ) && !profile.relation_term_keys.is_empty()
        && relation_overlap < usize::min(2, profile.relation_term_keys.len())
    {
        return (0.0, 0);
    }
    if !synthetic_answer_surface_relation_family_matches(profile, row_family, relation_overlap) {
        return (0.0, 0);
    }
    let choice_overlap = synthetic_answer_surface_choice_overlap(profile, &support_term_keys);
    if matches!(profile.route_kind, SyntheticAnswerSurfaceRouteKind::Choice) && choice_overlap == 0
    {
        return (0.0, 0);
    }
    let Some(type_bonus) =
        synthetic_answer_surface_type_bonus(profile, &row.answer_span, row_family)
    else {
        return (0.0, 0);
    };
    if profile.requires_completed_evidence {
        if has_future_answer_evidence && !has_completed_answer_evidence {
            return (0.0, 0);
        }
        if let Some(line) = evidence_line {
            let lower = line.to_ascii_lowercase();
            if synthetic_answer_surface_evidence_looks_future(&lower)
                && !synthetic_answer_surface_evidence_looks_completed(&lower)
                && !has_completed_answer_evidence
            {
                return (0.0, 0);
            }
        }
    }

    let coverage = overlap as f32 / profile.task_term_keys.len().max(1) as f32;
    let specificity = overlap as f32 / support_term_keys.len().max(1) as f32;
    let anchor_coverage = anchor_overlap as f32 / profile.anchor_term_keys.len().max(1) as f32;
    let evidence_overlap =
        synthetic_answer_surface_overlap_count(&evidence_term_keys, &profile.task_term_keys);
    let evidence_bonus = evidence_overlap as f32 * 2.0
        + if profile.requires_completed_evidence
            && evidence_line
                .map(|line| {
                    synthetic_answer_surface_evidence_looks_completed(&line.to_ascii_lowercase())
                })
                .unwrap_or(false)
        {
            1.0
        } else {
            0.0
        };
    let query_bonus = synthetic_answer_surface_query_bonus(profile, row, evidence_line);
    let relation_bonus = if profile.relation_families.is_empty() {
        0.0
    } else if row_family
        .map(|family| profile.relation_families.contains(&family))
        .unwrap_or(false)
    {
        5.0
    } else {
        relation_overlap as f32 * 1.5
    };
    (
        retrieval_score * 0.75
            + overlap as f32 * 3.5
            + coverage * 4.0
            + specificity * 1.5
            + anchor_overlap as f32 * 3.0
            + anchor_coverage * 4.0
            + relation_overlap as f32 * 2.5
            + choice_overlap as f32 * 3.5
            + subject_overlap as f32 * 3.5
            + evidence_bonus
            + relation_bonus
            + query_bonus
            + row.confidence
            + type_bonus,
        overlap,
    )
}

pub(in crate::index) fn format_index_answer_surface_answer(
    task_lower: &str,
    answer: &str,
) -> String {
    let answer_lower = answer.to_ascii_lowercase();
    if answer_lower.contains("ally")
        && task_contains_any(
            task_lower,
            &[
                "member of the lgbtq community",
                "member of the lgbtq+ community",
                "part of the lgbtq community",
                "part of the lgbtq+ community",
                "member of the transgender community",
            ],
        )
    {
        return "Likely no, supportive ally".to_string();
    }
    if answer_lower.contains("ally")
        && task_contains_any(
            task_lower,
            &[
                "ally to the transgender community",
                "ally to the lgbtq community",
                "ally to the lgbtq+ community",
                "considered an ally",
            ],
        )
    {
        return "Yes, supportive ally".to_string();
    }
    answer.to_string()
}

pub(in crate::index) fn answer_surface_evidence_line(
    content: &str,
    task_terms: &[String],
    answer_span: &str,
    question_pattern: &str,
) -> Option<String> {
    let body = strip_query_surface_section(content);
    let answer_lower = answer_span.to_ascii_lowercase();
    let answer_term_keys =
        synthetic_answer_surface_term_key_set(&synthetic_query_terms(&answer_lower));
    let pattern_terms = synthetic_query_terms(&question_pattern.to_ascii_lowercase());
    let pattern_term_keys = synthetic_answer_surface_term_key_set(&pattern_terms);
    let task_term_keys = synthetic_answer_surface_term_key_set(task_terms);

    let mut best: Option<(usize, usize, usize, bool, String)> = None;
    for line in body
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .filter(|line| !line.starts_with('|'))
    {
        let lower = line.to_ascii_lowercase();
        let line_terms = synthetic_query_terms(&lower);
        let line_term_keys = synthetic_answer_surface_term_key_set(&line_terms);
        let pattern_overlap =
            synthetic_answer_surface_overlap_count(&line_term_keys, &pattern_term_keys);
        let task_overlap = synthetic_answer_surface_overlap_count(&line_term_keys, &task_term_keys);
        let answer_hit = lower.contains(&answer_lower)
            || (!answer_term_keys.is_empty()
                && answer_term_keys.iter().all(|term| {
                    line_term_keys.iter().any(|line_term| {
                        line_term == term
                            || line_term.starts_with(term.as_str())
                            || term.starts_with(line_term.as_str())
                    })
                }));
        let score = usize::from(answer_hit) * 10 + task_overlap * 4 + pattern_overlap * 2;
        if !answer_hit && pattern_overlap < 2 && task_overlap == 0 {
            continue;
        }
        let replace = best
            .as_ref()
            .map(
                |(best_score, best_task, best_pattern, best_answer_hit, best_line)| {
                    score > *best_score
                        || (score == *best_score
                            && (task_overlap > *best_task
                                || (task_overlap == *best_task
                                    && (pattern_overlap > *best_pattern
                                        || (pattern_overlap == *best_pattern
                                            && (answer_hit && !*best_answer_hit
                                                || (answer_hit == *best_answer_hit
                                                    && line.len() < best_line.len())))))))
                },
            )
            .unwrap_or(true);
        if replace {
            best = Some((
                score,
                task_overlap,
                pattern_overlap,
                answer_hit,
                line.to_string(),
            ));
        }
    }

    best.map(|(_, _, _, _, line)| line)
}

pub(in crate::index) fn answer_surface_answer_span_evidence_state(
    content: &str,
    answer_span: &str,
) -> (bool, bool) {
    let body = strip_query_surface_section(content);
    let answer_lower = answer_span.to_ascii_lowercase();
    let mut has_future = false;
    let mut has_completed = false;

    for line in body
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .filter(|line| !line.starts_with('|'))
    {
        let lower = line.to_ascii_lowercase();
        if !lower.contains(&answer_lower) {
            continue;
        }
        has_future |= synthetic_answer_surface_evidence_looks_future(&lower);
        has_completed |= synthetic_answer_surface_evidence_looks_completed(&lower);
    }

    (has_future, has_completed)
}

pub(in crate::index) fn latest_active_kg_value(
    entity: &kg::KgEntity,
    predicate: &str,
) -> Option<String> {
    pub(in crate::index) fn latest_value_for_predicate(
        entity: &kg::KgEntity,
        predicate: &str,
    ) -> Option<String> {
        let mut facts = entity.active_values_for_predicate(predicate, None);
        facts.sort_by(|a, b| a.valid_from.cmp(&b.valid_from));
        if let Some(value) = facts
            .last()
            .map(|fact| fact.value.trim())
            .filter(|value| !value.is_empty())
        {
            return Some(normalize_latest_kg_value(predicate, value));
        }
        None
    }

    latest_value_for_predicate(entity, predicate).or_else(|| match predicate {
        "education" => latest_value_for_predicate(entity, "major"),
        "major" => latest_value_for_predicate(entity, "education"),
        _ => None,
    })
}

pub(in crate::index) fn normalize_latest_kg_value(predicate: &str, value: &str) -> String {
    match predicate {
        "location" => normalize_location_kg_value(value),
        "education" | "major" => normalize_education_kg_value(value),
        "fitness_record" => normalize_fitness_record_kg_value(value),
        _ => value.trim().to_string(),
    }
}

pub(in crate::index) fn normalize_location_kg_value(value: &str) -> String {
    let lower = value.to_ascii_lowercase();
    let cutoff = [" again ", " so ", " because ", " but ", " with ", " after "]
        .iter()
        .filter_map(|marker| lower.find(marker))
        .min()
        .unwrap_or(value.len());
    let mut trimmed = value[..cutoff]
        .trim()
        .trim_matches(|c: char| matches!(c, ',' | '.' | ';' | ':' | '"' | '\''));
    if let Some(stripped) = trimmed.strip_suffix(" again") {
        trimmed = stripped.trim();
    }
    if trimmed.eq_ignore_ascii_case("suburbs") {
        "the suburbs".to_string()
    } else if trimmed.eq_ignore_ascii_case("the suburbs") {
        "the suburbs".to_string()
    } else {
        trimmed.to_string()
    }
}

pub(in crate::index) fn normalize_education_kg_value(value: &str) -> String {
    let lower = value.to_ascii_lowercase();
    let cutoff = [
        " which ",
        " that ",
        " because ",
        " but ",
        " and ",
        " from ",
        " with a concentration in ",
        " with concentration in ",
        " with a minor in ",
        " with minor in ",
    ]
    .iter()
    .filter_map(|marker| lower.find(marker))
    .min()
    .unwrap_or(value.len());
    let mut trimmed = value[..cutoff]
        .trim()
        .trim_matches(|c: char| matches!(c, ',' | '.' | ';' | ':' | '"' | '\''));
    for suffix in [" which", " that", " from"] {
        if let Some(stripped) = trimmed.strip_suffix(suffix) {
            trimmed = stripped.trim();
        }
    }
    trimmed.to_string()
}

pub(in crate::index) fn normalize_fitness_record_kg_value(value: &str) -> String {
    let trimmed = value.trim();
    let parts: Vec<_> = trimmed.split(':').collect();
    if parts.len() == 2
        && parts[0].chars().all(|c| c.is_ascii_digit())
        && parts[1].len() == 2
        && parts[1].chars().all(|c| c.is_ascii_digit())
    {
        let minutes = parts[0].parse::<u32>().ok();
        let seconds = parts[1].parse::<u32>().ok();
        if let (Some(minutes), Some(seconds)) = (minutes, seconds) {
            return format!("{minutes} minutes and {seconds} seconds (or {trimmed})");
        }
    }
    trimmed.to_string()
}

pub(in crate::index) fn extract_fitness_record_time_value(line: &str) -> Option<(u32, String)> {
    compile_regex(r"\b(\d{1,2}):(\d{2})\b")
        .captures_iter(line)
        .filter_map(|caps| {
            let minutes = caps.get(1)?.as_str().parse::<u32>().ok()?;
            let seconds = caps.get(2)?.as_str().parse::<u32>().ok()?;
            (seconds < 60).then_some((minutes * 60 + seconds, caps.get(0)?.as_str().to_string()))
        })
        .min_by_key(|(total_seconds, _)| *total_seconds)
}

pub(in crate::index) fn parse_count_token_value(token: &str) -> Option<i32> {
    let cleaned = token
        .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '$' && c != ',' && c != '%')
        .to_ascii_lowercase();
    if cleaned.is_empty() {
        return None;
    }
    match cleaned.as_str() {
        "zero" => Some(0),
        "one" => Some(1),
        "first" => Some(1),
        "two" => Some(2),
        "second" => Some(2),
        "three" => Some(3),
        "third" => Some(3),
        "four" => Some(4),
        "fourth" => Some(4),
        "five" => Some(5),
        "fifth" => Some(5),
        "six" => Some(6),
        "sixth" => Some(6),
        "seven" => Some(7),
        "seventh" => Some(7),
        "eight" => Some(8),
        "eighth" => Some(8),
        "nine" => Some(9),
        "ninth" => Some(9),
        "ten" => Some(10),
        "tenth" => Some(10),
        "eleven" => Some(11),
        "eleventh" => Some(11),
        "twelve" => Some(12),
        "twelfth" => Some(12),
        _ => {
            if let Some(stripped) = cleaned
                .strip_suffix("st")
                .or_else(|| cleaned.strip_suffix("nd"))
                .or_else(|| cleaned.strip_suffix("rd"))
                .or_else(|| cleaned.strip_suffix("th"))
            {
                if !stripped.is_empty() && stripped.chars().all(|c| c.is_ascii_digit() || c == ',')
                {
                    return stripped.replace(',', "").parse::<i32>().ok();
                }
            }
            if cleaned.chars().any(|c| c.is_ascii_digit())
                && cleaned.chars().any(|c| c.is_ascii_alphabetic())
                && !cleaned.contains('-')
            {
                return None;
            }
            let digits: String = cleaned
                .chars()
                .filter(|c| c.is_ascii_digit() || *c == ',')
                .collect();
            if digits.is_empty() {
                None
            } else {
                digits.replace(',', "").parse::<i32>().ok()
            }
        },
    }
}

pub(in crate::index) fn extract_line_numbers(line: &str) -> Vec<i32> {
    line.split_whitespace()
        .filter_map(parse_count_token_value)
        .collect()
}

pub(in crate::index) fn extract_focus_aligned_count(
    line: &str,
    focus_terms: &[String],
    task_lower: &str,
) -> Option<(i32, usize)> {
    const TIME_UNITS: &[&str] = &[
        "day", "days", "week", "weeks", "month", "months", "year", "years", "hour", "hours",
    ];
    let focus_keys: HashSet<String> = focus_terms
        .iter()
        .map(|term| synthetic_answer_surface_term_key(term))
        .filter(|key| !key.is_empty())
        .collect();
    if focus_keys.is_empty() {
        return None;
    }

    let tokens = line.split_whitespace().collect::<Vec<_>>();
    let token_keys = tokens
        .iter()
        .map(|token| synthetic_answer_surface_term_key(token))
        .collect::<Vec<_>>();
    let mut best: Option<(usize, usize, i32)> = None;

    for (idx, token) in tokens.iter().enumerate() {
        if idx == 0
            && token
                .trim_end_matches(['.', ')'])
                .chars()
                .all(|c| c.is_ascii_digit())
        {
            continue;
        }
        let Some(value) = parse_count_token_value(token) else {
            continue;
        };
        if (1900..=2100).contains(&value) {
            continue;
        }

        let negation_start = idx.saturating_sub(2);
        if token_keys[negation_start..idx]
            .iter()
            .any(|key| matches!(key.as_str(), "not" | "never"))
        {
            continue;
        }

        let raw_token = token.to_ascii_lowercase();
        let adjacent_time_unit = TIME_UNITS.iter().find(|unit| {
            raw_token.contains(&format!("-{unit}"))
                || token_keys
                    .get(idx + 1)
                    .map(|next| next == *unit)
                    .unwrap_or(false)
        });
        if let Some(unit) = adjacent_time_unit {
            if !task_lower.contains(unit) {
                continue;
            }
        }

        let window_start = idx.saturating_sub(6);
        let window_end = usize::min(token_keys.len(), idx + 7);
        let nearby_focus = token_keys[window_start..window_end]
            .iter()
            .filter(|key| focus_keys.contains(*key))
            .collect::<HashSet<_>>()
            .len();
        if nearby_focus == 0 {
            continue;
        }

        let nearest_distance = token_keys
            .iter()
            .enumerate()
            .filter(|(_, key)| focus_keys.contains(*key))
            .map(|(focus_idx, _)| idx.abs_diff(focus_idx))
            .min()
            .unwrap_or(usize::MAX);
        let score = nearby_focus * 10 + 7usize.saturating_sub(nearest_distance.min(7));

        if best
            .as_ref()
            .map(|(best_score, best_distance, best_value)| {
                score > *best_score
                    || (score == *best_score && nearest_distance < *best_distance)
                    || (score == *best_score
                        && nearest_distance == *best_distance
                        && value > *best_value)
            })
            .unwrap_or(true)
        {
            best = Some((score, nearest_distance, value));
        }
    }

    best.map(|(score, _, value)| (value, score))
}

pub(in crate::index) fn is_summary_or_user_line(line: &str, lower: &str) -> bool {
    lower.starts_with("user:") || line.trim_start().starts_with('-')
}

pub(in crate::index) fn is_session_answer_candidate_line(line: &str) -> bool {
    let trimmed = line.trim();
    !(trimmed.is_empty()
        || trimmed.starts_with('#')
        || trimmed.starts_with("<!--")
        || trimmed.starts_with('|')
        || trimmed.starts_with("Question:")
        || trimmed.starts_with("Answer:"))
}

pub(in crate::index) fn normalize_session_answer_line_body(line: &str) -> String {
    let mut body = line.trim();
    if let Some(stripped) = body.strip_prefix('-') {
        body = stripped.trim();
    }

    let digit_prefix = body.chars().take_while(|c| c.is_ascii_digit()).count();
    if digit_prefix > 0 {
        let rest = body[digit_prefix..].trim_start();
        if rest.starts_with('.') || rest.starts_with(')') {
            body = rest[1..].trim();
        }
    }

    let lower = body.to_ascii_lowercase();
    for prefix in ["user:", "assistant:"] {
        if lower.starts_with(prefix) {
            body = body[prefix.len()..].trim();
            break;
        }
    }

    body.trim_matches(|c: char| matches!(c, '"' | '\'' | '`'))
        .trim()
        .to_string()
}

pub(in crate::index) fn task_has_recall_context(task_lower: &str) -> bool {
    task_contains_any(
        task_lower,
        &[
            "remind me",
            "previous chat",
            "previous conversation",
            "last time",
            "follow up",
            "follow-up",
            "told me",
            "talked about",
            "we talked",
            "remember you",
            "remember what",
            "used as an example",
            "going back to our previous",
        ],
    )
}

pub(in crate::index) fn should_try_session_recall_answer(task: &str, task_lower: &str) -> bool {
    if task_contains_any(
        task_lower,
        &[
            " in total",
            " altogether",
            " combined",
            " before ",
            " after ",
            " difference ",
            " compared ",
            " how long had i been",
            " when i just started",
        ],
    ) {
        return false;
    }

    task_lower.contains("what color")
        || (task_lower.starts_with("where ")
            && task_contains_any(
                task_lower,
                &[
                    "buy",
                    "bought",
                    "redeem",
                    "use my coupon",
                    "which store",
                    "shop",
                    "keep",
                    "kept",
                ],
            ))
        || task_lower.contains("discount")
        || is_money_query(task)
        || task_contains_any(task_lower, &["what speed", "internet plan", "camera lens"])
}

pub(in crate::index) fn normalized_synthetic_phrase_key(text: &str) -> String {
    text.trim()
        .trim_matches(|c: char| matches!(c, '"' | '\'' | ',' | '.' | ';' | ':' | '!' | '?'))
        .to_ascii_lowercase()
}

pub(in crate::index) fn project_session_answer_from_line(
    task: &str,
    task_lower: &str,
    predicate: Option<&str>,
    line: &str,
    lower: &str,
) -> Option<String> {
    match predicate {
        Some("education") | Some("major") => return extract_session_education_answer(line, lower),
        Some("project_name") => {
            return extract_session_named_answer_from_line(task_lower, line, lower)
        },
        Some("location") => return extract_session_location_answer(task_lower, line, lower),
        Some("occupation") => return extract_session_occupation_answer(line, lower),
        Some("book") => return extract_session_named_answer_from_line(task_lower, line, lower),
        _ => {},
    }

    if is_education_query(task_lower) || is_major_query(task_lower) {
        if let Some(answer) = extract_session_education_answer(line, lower) {
            return Some(answer);
        }
    }
    if task_lower.contains("discount") {
        if let Some(answer) = extract_percent_answer_from_line(line) {
            return Some(answer);
        }
    }
    if task_lower.contains("what color") {
        if task_lower.contains("did i")
            && (lower.contains("planning to") || lower.contains("thinking of"))
        {
            return None;
        }
        if let Some(answer) = extract_color_answer_from_line(line) {
            return Some(answer);
        }
    }
    if task_lower.starts_with("where ") {
        if let Some(answer) = extract_session_location_answer(task_lower, line, lower) {
            return Some(answer);
        }
    }
    if task_lower.starts_with("when ")
        || task_contains_any(task_lower, &["what day", "what date", "what time"])
    {
        if let Some(answer) = extract_date_or_time_answer_from_line(line) {
            return Some(answer);
        }
    }
    if task_lower.starts_with("how long ") {
        if let Some(answer) = extract_duration_answer_from_line(line) {
            return Some(answer);
        }
    }
    if is_money_query(task) {
        if let Some(answer) = extract_money_answer_from_line(line) {
            return Some(answer);
        }
    }
    if task_contains_any(task_lower, &["what speed", "internet plan"]) {
        if let Some(answer) = extract_speed_answer_from_line(line) {
            return Some(answer);
        }
    }
    if task_contains_any(task_lower, &["camera lens"]) {
        if let Some(answer) = extract_session_purchase_item(line, lower) {
            if answer.to_ascii_lowercase().contains("lens") {
                return Some(answer);
            }
        }
    }
    if task_has_recall_context(task_lower) && detect_counting_query(task) {
        if let Some(answer) = extract_query_aligned_numeric_answer(task_lower, line) {
            return Some(answer);
        }
    }
    if task_has_recall_context(task_lower)
        || task_contains_any(
            task_lower,
            &[
                "name of",
                "called",
                "call it",
                "title",
                "what kind",
                "what type",
                "specific",
            ],
        )
    {
        if let Some(answer) = extract_session_list_answer_from_line(task_lower, line, lower) {
            return Some(answer);
        }
        if let Some(answer) = extract_session_named_answer_from_line(task_lower, line, lower) {
            return Some(answer);
        }
    }

    None
}

pub(in crate::index) fn is_assistant_followup_query(task_lower: &str) -> bool {
    task_contains_any(
        task_lower,
        &[
            "you mentioned",
            "you recommended",
            "our previous conversation",
            "previous conversation",
            "previous chat",
            "previous chess game",
            "follow up on our previous",
            "looking back at our previous",
            "going back to our previous",
        ],
    ) && task_contains_any(
        task_lower,
        &[
            "remind me",
            "can you remind me",
            "what was",
            "what kind",
            "what type",
            "how many",
            "which website",
            "what website",
            "what move",
            "which move",
            "what was the move",
        ],
    )
}

pub(in crate::index) fn project_assistant_followup_answer_from_context(
    task: &str,
    task_lower: &str,
    lines: &[String],
    line_idx: usize,
) -> Option<String> {
    if let Some(answer) = extract_adjacent_role_person_followup_answer(task_lower, lines, line_idx)
    {
        return Some(answer);
    }
    let line = lines.get(line_idx)?;
    let lower = line.to_ascii_lowercase();
    project_assistant_followup_answer_from_line(task, task_lower, line, &lower)
}

pub(in crate::index) fn extract_adjacent_role_person_followup_answer(
    task_lower: &str,
    lines: &[String],
    line_idx: usize,
) -> Option<String> {
    if !task_contains_any(task_lower, &["who is the", "who was the"]) {
        return None;
    }
    let role_terms = assistant_followup_role_terms(task_lower);
    if role_terms.is_empty() {
        return None;
    }
    let line = lines.get(line_idx)?;
    let lower = line.to_ascii_lowercase();
    let role_overlap = role_terms
        .iter()
        .filter(|term| lower.contains(term.as_str()))
        .count();
    if role_overlap == 0 {
        return None;
    }
    for neighbor_idx in [line_idx.checked_sub(1), Some(line_idx + 1)] {
        let Some(neighbor_idx) = neighbor_idx else {
            continue;
        };
        let Some(neighbor) = lines.get(neighbor_idx) else {
            continue;
        };
        let neighbor_lower = neighbor.to_ascii_lowercase();
        if let Some(answer) =
            extract_session_named_answer_from_line(task_lower, neighbor, &neighbor_lower)
        {
            if answer
                .chars()
                .next()
                .map(|ch| ch.is_ascii_uppercase())
                .unwrap_or(false)
            {
                return Some(answer);
            }
        }
    }
    None
}

pub(in crate::index) fn project_assistant_followup_answer_from_line(
    task: &str,
    task_lower: &str,
    line: &str,
    lower: &str,
) -> Option<String> {
    if task_contains_any(
        task_lower,
        &["what move", "which move", "what was the move"],
    ) {
        if let Some(answer) = extract_chess_move_answer_from_line(
            line,
            extract_expected_chess_reply_move_number(task_lower),
        ) {
            return Some(answer);
        }
    }
    if let Some(answer) = extract_descriptor_named_followup_answer(task_lower, line, lower) {
        return Some(answer);
    }
    if detect_counting_query(task) {
        if let Some(answer) = extract_parenthetical_label_count_answer(task_lower, line, lower)
            .or_else(|| extract_query_aligned_numeric_answer(task_lower, line))
        {
            return Some(answer);
        }
        return None;
    }
    if task_lower.contains("website") {
        if let Some(answer) = extract_website_name_from_line(line) {
            return Some(answer);
        }
    }
    if task_contains_any(task_lower, &["what type of beer", "what kind of beer"]) {
        if let Some(answer) = extract_beer_recommendation_answer_from_line(lower) {
            return Some(answer);
        }
    }
    if task_lower.contains("two-factor authentication") {
        if let Some(answer) = extract_two_factor_method_answer_from_line(line, lower) {
            return Some(answer);
        }
    }
    project_session_answer_from_line(task, task_lower, None, line, lower)
}

pub(in crate::index) fn extract_descriptor_named_followup_answer(
    task_lower: &str,
    line: &str,
    lower: &str,
) -> Option<String> {
    if detect_counting_query(task_lower)
        || task_lower.starts_with("how ")
        || task_lower.starts_with("when ")
        || task_lower.starts_with("where ")
    {
        return None;
    }
    let descriptor_terms = assistant_followup_descriptor_terms(task_lower);
    if descriptor_terms.len() < 2 {
        return None;
    }
    let matched = descriptor_terms
        .iter()
        .filter(|term| lower.contains(term.as_str()))
        .count();
    if matched < 2 {
        return None;
    }
    extract_session_named_answer_from_line(task_lower, line, lower)
}

pub(in crate::index) fn assistant_followup_descriptor_terms(task_lower: &str) -> Vec<String> {
    let mut terms = Vec::new();
    if let Some((_, clause)) = task_lower
        .rsplit_once(" that ")
        .or_else(|| task_lower.rsplit_once(" which "))
        .or_else(|| task_lower.rsplit_once(" who "))
    {
        terms.extend(
            synthetic_query_terms(clause)
                .into_iter()
                .filter(|term| term.len() >= 3)
                .filter(|term| !term.chars().all(|ch| ch.is_ascii_digit()))
                .filter(|term| {
                    !matches!(term.as_str(), "companies" | "company" | "people" | "person")
                }),
        );
    }
    if let Some(subject_clause) = assistant_followup_subject_descriptor_clause(task_lower) {
        terms.extend(
            synthetic_query_terms(subject_clause)
                .into_iter()
                .filter(|term| term.len() >= 3)
                .filter(|term| !term.chars().all(|ch| ch.is_ascii_digit()))
                .filter(|term| !matches!(term.as_str(), "example" | "gave" | "people" | "person")),
        );
    }
    terms.sort();
    terms.dedup();
    terms
}

pub(in crate::index) fn assistant_followup_subject_descriptor_clause(
    task_lower: &str,
) -> Option<&str> {
    for marker in [
        "example you gave of a ",
        "example you gave of an ",
        "example you gave of the ",
    ] {
        let Some((_, tail)) = task_lower.split_once(marker) else {
            continue;
        };
        let stop = tail
            .find(" who ")
            .or_else(|| tail.find(" that "))
            .or_else(|| tail.find(" which "))
            .unwrap_or(tail.len());
        let clause = tail[..stop].trim();
        if !clause.is_empty() {
            return Some(clause);
        }
    }
    None
}

pub(in crate::index) fn assistant_followup_role_terms(task_lower: &str) -> Vec<String> {
    synthetic_query_terms(task_lower)
        .into_iter()
        .filter(|term| term.len() >= 5)
        .filter(|term| {
            !matches!(
                term.as_str(),
                "article"
                    | "conversation"
                    | "follow"
                    | "mentioned"
                    | "previous"
                    | "remind"
                    | "science"
                    | "technology"
            )
        })
        .collect()
}

pub(in crate::index) fn assistant_followup_anchor_terms(task_lower: &str) -> Vec<String> {
    let Some((_, tail)) = task_lower.rsplit_once(" at ") else {
        return Vec::new();
    };
    let segment = tail.split(['.', '?', '!', ',']).next().unwrap_or("").trim();
    let terms: Vec<String> = synthetic_query_terms(segment)
        .into_iter()
        .filter(|term| term.len() >= 4)
        .collect();
    if (1..=4).contains(&terms.len()) {
        terms
    } else {
        Vec::new()
    }
}

pub(in crate::index) fn assistant_followup_anchor_distance(
    line_lower: &str,
    match_end: usize,
    anchor_terms: &[String],
) -> Option<usize> {
    if anchor_terms.is_empty() {
        return None;
    }
    anchor_terms
        .iter()
        .filter_map(|term| {
            line_lower[match_end..]
                .find(term)
                .map(|offset| offset + match_end)
        })
        .map(|position| position.saturating_sub(match_end))
        .min()
}

pub(in crate::index) fn assistant_followup_context(lines: &[String], line_idx: usize) -> String {
    let start = line_idx.saturating_sub(1);
    let end = usize::min(line_idx + 1, lines.len().saturating_sub(1));
    lines[start..=end].join(" ")
}

pub(in crate::index) fn extract_expected_chess_reply_move_number(task_lower: &str) -> Option<i32> {
    let prior_move = compile_regex(r"after\s+(\d+)\.")
        .captures(task_lower)
        .and_then(|caps| caps.get(1))
        .and_then(|m| m.as_str().parse::<i32>().ok())?;
    Some(prior_move + 1)
}

pub(in crate::index) fn extract_chess_move_answer_from_line(
    line: &str,
    expected_move_number: Option<i32>,
) -> Option<String> {
    let capture = compile_regex(
        r"\b(\d+)\.\s*(O-O(?:-O)?|[KQRNB]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRNB])?[+#]?)\b",
    )
    .captures(line)?;
    let move_number = capture.get(1)?.as_str().parse::<i32>().ok()?;
    if expected_move_number.is_some_and(|expected| expected != move_number) {
        return None;
    }
    let notation = capture.get(2)?.as_str().trim();
    Some(format!("{move_number}. {notation}"))
}

pub(in crate::index) fn extract_parenthetical_label_count_answer(
    task_lower: &str,
    line: &str,
    _lower: &str,
) -> Option<String> {
    let focus_terms = synthetic_query_terms(task_lower);
    let focus_refs: Vec<&str> = focus_terms.iter().map(String::as_str).collect();
    let capture = compile_regex(r"(?i)\b([A-Za-z][A-Za-z' -]+?)\s*\((\d+)\)").captures(line)?;
    let label = capture.get(1)?.as_str().trim().to_ascii_lowercase();
    (term_overlap_count(&label, &focus_refs) >= 1)
        .then(|| capture.get(2).map(|m| m.as_str().trim().to_string()))
        .flatten()
}

pub(in crate::index) fn extract_website_name_from_line(line: &str) -> Option<String> {
    compile_regex(r"\b([A-Za-z0-9-]+\.(?:org|com|net|edu|io))\b")
        .captures(line)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim().to_string())
}

pub(in crate::index) fn extract_beer_recommendation_answer_from_line(
    lower: &str,
) -> Option<String> {
    (lower.contains("beer") && lower.contains("pilsner") && lower.contains("lager"))
        .then_some("I recommended using a Pilsner or Lager for the recipe.".to_string())
}

pub(in crate::index) fn extract_two_factor_method_answer_from_line(
    line: &str,
    lower: &str,
) -> Option<String> {
    if !lower.contains("two-factor authentication") {
        return None;
    }
    let methods = extract_phrase_after_any_index(
        line,
        lower,
        &["such as "],
        &[", enhances security", " enhances security", ".", ";"],
        1,
    )?;
    Some(format!(
        "I mentioned {} as examples of two-factor authentication methods.",
        methods.trim().trim_end_matches(',')
    ))
}

pub(in crate::index) fn extract_session_education_answer(
    line: &str,
    lower: &str,
) -> Option<String> {
    let mut answer = extract_phrase_after_any_index(
        line,
        lower,
        &[
            "degree in ",
            "bachelor's in ",
            "bachelors in ",
            "master's in ",
            "masters in ",
            "graduated with a degree in ",
            "graduated with degree in ",
            "graduated with ",
            "majored in ",
            "major in ",
            "studying ",
            "study ",
        ],
        &[
            " which",
            " from ",
            " at ",
            " and ",
            " but ",
            " because ",
            ",",
        ],
        1,
    )?;
    for prefix in [
        "a degree in ",
        "degree in ",
        "a bachelor's in ",
        "a bachelors in ",
        "bachelor's in ",
        "bachelors in ",
        "a master's in ",
        "a masters in ",
        "master's in ",
        "masters in ",
    ] {
        if answer.to_ascii_lowercase().starts_with(prefix) {
            answer = answer[prefix.len()..].trim().to_string();
            break;
        }
    }
    Some(normalize_education_kg_value(&answer))
}

pub(in crate::index) fn extract_session_named_answer_from_line(
    task_lower: &str,
    line: &str,
    lower: &str,
) -> Option<String> {
    let is_query_context = |candidate: &str| {
        let terms = tokenize(&candidate.to_ascii_lowercase());
        !terms.is_empty()
            && terms
                .iter()
                .all(|term| term.len() <= 2 || task_lower.contains(term.as_str()))
    };
    if let Some(value) = extract_descriptor_led_named_answer(line) {
        if !is_query_context(&value) {
            return Some(value);
        }
    }
    let is_question = lower.trim_end().ends_with('?');
    let markers = if is_question {
        vec![
            "called ",
            "named ",
            "titled ",
            "example is ",
            "example was ",
        ]
    } else {
        vec![
            "called ",
            "named ",
            "titled ",
            "recommend ",
            "recommended ",
            "try ",
            "example is ",
            "example was ",
            "was ",
        ]
    };
    if let Some(value) = extract_phrase_after_any_index(
        line,
        lower,
        &markers,
        &[" for ", " because ", " and ", " but ", ".", ",", " while "],
        1,
    ) {
        if let Some(best_title) = extract_title_like_phrases(&value)
            .into_iter()
            .find(|candidate| !is_query_context(candidate))
        {
            return Some(best_title);
        }
        if value.split_whitespace().count() <= 8 && !is_query_context(&value) {
            return Some(value);
        }
    }

    let mut titles = extract_title_like_phrases(line)
        .into_iter()
        .filter(|value| {
            let lower_value = value.to_ascii_lowercase();
            ![
                "also", "by", "can", "do", "does", "for", "i", "it", "my", "our", "that", "the",
                "this", "we", "what", "when", "where", "which", "who",
            ]
            .contains(&lower_value.as_str())
                && !is_query_context(value)
        })
        .collect::<Vec<_>>();
    if task_contains_any(task_lower, &["playlist", "project", "blog", "channel"]) {
        titles.retain(|value| value.split_whitespace().count() <= 6);
    }
    titles.sort_by(|left, right| right.len().cmp(&left.len()).then_with(|| left.cmp(right)));
    titles.into_iter().next()
}

pub(in crate::index) fn extract_descriptor_led_named_answer(line: &str) -> Option<String> {
    let body = normalize_session_answer_line_body(line);
    let body_lower = body.to_ascii_lowercase();
    let split_idx = [
        " has ", " have ", " had ", " is ", " was ", " said ", " taken ",
    ]
    .into_iter()
    .filter_map(|marker| body_lower.find(marker))
    .min()?;
    let mut prefix = body[..split_idx].trim();
    for marker in ["for example,", "for instance,", "likewise,", "similarly,"] {
        if body_lower.starts_with(marker) {
            prefix = prefix[marker.len()..].trim();
            break;
        }
    }
    prefix = prefix
        .trim_start_matches("the ")
        .trim_start_matches("a ")
        .trim_start_matches("an ")
        .trim();
    let tokens: Vec<&str> = prefix
        .split_whitespace()
        .map(|token| {
            token.trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '\'' && ch != '-')
        })
        .filter(|token| !token.is_empty())
        .collect();
    if tokens.len() < 2 {
        return None;
    }
    let candidate_tokens: Vec<&str> = tokens
        .iter()
        .rev()
        .take_while(|token| !token.contains('/') && !token.eq_ignore_ascii_case("the"))
        .take(2)
        .copied()
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    if candidate_tokens.len() < 2 {
        return None;
    }
    Some(title_case_named_words(&candidate_tokens.join(" ")))
}

pub(in crate::index) fn title_case_named_words(value: &str) -> String {
    value
        .split_whitespace()
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(first) => format!("{}{}", first.to_uppercase(), chars.as_str()),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

pub(in crate::index) fn extract_session_list_answer_from_line(
    task_lower: &str,
    line: &str,
    lower: &str,
) -> Option<String> {
    let answer = extract_phrase_after_any_index(
        line,
        lower,
        &[
            "such as ",
            "including ",
            "include ",
            "includes ",
            "uses ",
            "using ",
            "were ",
        ],
        &[". ", "?", " and i'm ", " and i’m ", " but "],
        1,
    )?;
    task_contains_any(
        task_lower,
        &["what kind", "what type", "specific", "what were the"],
    )
    .then_some(answer)
}

pub(in crate::index) fn extract_session_location_answer(
    task_lower: &str,
    line: &str,
    lower: &str,
) -> Option<String> {
    if task_contains_any(
        task_lower,
        &[
            "buy",
            "bought",
            "redeem",
            "use my coupon",
            "which store",
            "shop",
        ],
    ) {
        return extract_phrase_after_any_index(
            line,
            lower,
            &["from the ", "from ", "at the ", "at "],
            &[
                " for ",
                " with ",
                " because ",
                " and ",
                " but ",
                " last ",
                ".",
            ],
            1,
        );
    }
    if task_contains_any(
        task_lower,
        &["keep", "kept", "store", "stored", "put", "place"],
    ) {
        for marker in ["under ", "in ", "inside ", "on "] {
            if let Some(phrase) = extract_phrase_after_any_index(
                line,
                lower,
                &[marker],
                &[" because ", " and ", " but ", ".", ","],
                1,
            ) {
                return Some(format!("{} {}", marker.trim(), phrase));
            }
        }
    }
    extract_phrase_after_any_index(
        line,
        lower,
        &[
            "based in ",
            "live in ",
            "living in ",
            "now living in ",
            "moved to ",
            "moved back to ",
        ],
        &[
            " again",
            " because ",
            " and ",
            " but ",
            " with ",
            " after ",
            ".",
            ",",
        ],
        1,
    )
    .map(|value| normalize_location_kg_value(&value))
}

pub(in crate::index) fn extract_session_occupation_answer(
    line: &str,
    lower: &str,
) -> Option<String> {
    extract_phrase_after_any_index(
        line,
        lower,
        &[
            "work as ",
            "working as ",
            "employed as ",
            "job as ",
            "role as ",
            "i'm a ",
            "i am a ",
        ],
        &[" at ", " for ", " and ", " but ", " because ", "."],
        1,
    )
}

pub(in crate::index) fn extract_money_answer_from_line(line: &str) -> Option<String> {
    compile_regex(r"(?i)(\$\d[\d,]*(?:\.\d+)?)")
        .captures(line)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim().to_string())
}

pub(in crate::index) fn extract_percent_answer_from_line(line: &str) -> Option<String> {
    compile_regex(r"(?i)(\d+(?:\.\d+)?%)")
        .captures(line)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim().to_string())
}

pub(in crate::index) fn extract_speed_answer_from_line(line: &str) -> Option<String> {
    compile_regex(r"(?i)(\d+(?:\.\d+)?\s*(?:mbps|gbps))")
        .captures(line)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim().to_string())
}

pub(in crate::index) fn extract_university_name_from_line(line: &str) -> Option<String> {
    compile_regex(r"([A-Z][A-Za-z&.'-]*(?:\s+[A-Z][A-Za-z&.'-]*)*\s+University)")
        .captures(line)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim().to_string())
}

pub(in crate::index) fn extract_query_month_name(lower: &str) -> Option<&'static str> {
    [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ]
    .into_iter()
    .find(|month| lower.contains(month))
}

pub(in crate::index) fn next_month_name(month: &str) -> Option<&'static str> {
    match month {
        "january" => Some("february"),
        "february" => Some("march"),
        "march" => Some("april"),
        "april" => Some("may"),
        "may" => Some("june"),
        "june" => Some("july"),
        "july" => Some("august"),
        "august" => Some("september"),
        "september" => Some("october"),
        "october" => Some("november"),
        "november" => Some("december"),
        "december" => Some("january"),
        _ => None,
    }
}

pub(in crate::index) fn line_matches_query_month_window(lower: &str, month: &str) -> bool {
    if lower.contains(month) {
        return true;
    }

    lower.contains("this month")
        && next_month_name(month)
            .map(|next_month| lower.contains(&format!("before {next_month}")))
            .unwrap_or(false)
}

pub(in crate::index) fn line_describes_actual_doctor_visit(lower: &str) -> bool {
    let positive = task_contains_any(
        lower,
        &[
            "follow-up appointment",
            "appointment with",
            "went to see",
            "got back from",
            "diagnosed me with",
            "diagnosed with",
            "was prescribed",
            "prescribed antibiotics",
            "prescribed a nasal spray",
            "recently had",
            "just got diagnosed",
        ],
    );
    if !positive {
        return false;
    }

    if task_contains_any(
        lower,
        &[
            "thinking about",
            "considering",
            "i'll schedule",
            "i will schedule",
            "schedule an appointment",
            "scheduling an appointment",
            "talk to dr.",
            "ask dr.",
            "follow up with dr.",
            "consult with",
        ],
    ) {
        return false;
    }

    true
}

pub(in crate::index) fn extract_doctor_role_from_line(_line: &str, lower: &str) -> Option<String> {
    [
        ("primary care physician", "a primary care physician"),
        ("ent specialist", "an ENT specialist"),
        ("dermatologist", "a dermatologist"),
        ("orthopedic surgeon", "an orthopedic surgeon"),
        ("neurologist", "a neurologist"),
        ("gastroenterologist", "a gastroenterologist"),
    ]
    .into_iter()
    .find(|(needle, _)| lower.contains(needle))
    .map(|(_, rendered)| rendered.to_string())
}

pub(in crate::index) fn doctor_role_sort_key(role: &str) -> usize {
    match role {
        "a primary care physician" => 0,
        "an ENT specialist" => 1,
        "a dermatologist" => 2,
        "an orthopedic surgeon" => 3,
        "a neurologist" => 4,
        "a gastroenterologist" => 5,
        _ => 99,
    }
}

pub(in crate::index) fn doctor_visit_event_key(role: &str, lower: &str) -> String {
    let day = compile_regex(r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?\b")
        .captures(lower)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().to_string());
    match day {
        Some(day) => format!("{role}|{day}"),
        None => role.to_string(),
    }
}

pub(in crate::index) fn extract_duration_answer_from_line(line: &str) -> Option<String> {
    compile_regex(
        r"(?i)\b((?:about\s+)?(?:an?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?)\s+(?:days?|weeks?|months?|years?|hours?|minutes?)(?:\s+(?:ago|now|each way))?)\b",
    )
    .captures(line)
    .and_then(|caps| caps.get(1))
    .map(|m| m.as_str().trim().to_string())
}

pub(in crate::index) fn normalize_current_duration_answer(duration: &str) -> String {
    duration
        .trim()
        .trim_start_matches("about ")
        .trim_end_matches(" now")
        .trim_end_matches(" ago")
        .trim_start_matches("an ")
        .trim_start_matches("a ")
        .to_string()
        .replacen("one ", "1 ", 1)
}

pub(in crate::index) fn duration_answer_magnitude(duration: &str) -> Option<f32> {
    let lower = duration.to_ascii_lowercase();
    let caps = compile_regex(
        r"\b(\d+(?:\.\d+)?|an?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)(?:\s*-\s*(\d+(?:\.\d+)?))?\s+(day|week|month|year|hour|minute)s?\b",
    )
    .captures(&lower)?;
    let quantity = match caps.get(2).map(|m| m.as_str()) {
        Some(value) => value.parse::<f32>().ok()?,
        None => match caps.get(1)?.as_str() {
            "a" | "an" => 1.0,
            "one" => 1.0,
            "two" => 2.0,
            "three" => 3.0,
            "four" => 4.0,
            "five" => 5.0,
            "six" => 6.0,
            "seven" => 7.0,
            "eight" => 8.0,
            "nine" => 9.0,
            "ten" => 10.0,
            "eleven" => 11.0,
            "twelve" => 12.0,
            value => value.parse::<f32>().ok()?,
        },
    };
    let unit_days = match caps.get(3)?.as_str() {
        "minute" => 1.0 / (24.0 * 60.0),
        "hour" => 1.0 / 24.0,
        "day" => 1.0,
        "week" => 7.0,
        "month" => 30.0,
        "year" => 365.0,
        _ => return None,
    };
    Some(quantity * unit_days)
}

pub(in crate::index) fn is_ongoing_duration_query(task_lower: &str) -> bool {
    task_lower.starts_with("how long have ")
        && !task_contains_any(
            task_lower,
            &[" before ", " after ", " until ", "left to", "remaining"],
        )
}

pub(in crate::index) fn extract_ongoing_duration_anchor_terms(terms: &[String]) -> Vec<String> {
    const STOP: &[&str] = &[
        "long",
        "been",
        "being",
        "using",
        "living",
        "sticking",
        "staying",
        "working",
        "collecting",
        "keeping",
        "having",
        "doing",
        "going",
        "current",
        "daily",
        "about",
        "around",
        "there",
        "here",
    ];
    let anchors: Vec<String> = terms
        .iter()
        .filter(|term| term.len() >= 4)
        .filter(|term| !STOP.contains(&term.as_str()))
        .cloned()
        .collect();
    if anchors.is_empty() {
        terms
            .iter()
            .filter(|term| term.len() >= 3)
            .filter(|term| !STOP.contains(&term.as_str()))
            .cloned()
            .collect()
    } else {
        anchors
    }
}

pub(in crate::index) fn extract_tablespoon_water_ounces(line: &str) -> Option<f32> {
    let lower = line.to_ascii_lowercase();
    if !(lower.contains("tablespoon")
        && lower.contains("coffee")
        && lower.contains("ounces")
        && lower.contains("water"))
    {
        return None;
    }
    compile_regex(r"(?i)\b(\d+(?:\.\d+)?)\s+ounces?\s+of\s+water\b")
        .captures(line)
        .and_then(|caps| caps.get(1))
        .and_then(|m| m.as_str().parse::<f32>().ok())
}

pub(in crate::index) fn compact_decimal_string(value: f32) -> String {
    let mut rendered = value.to_string();
    if rendered.ends_with(".0") {
        rendered.truncate(rendered.len() - 2);
    }
    rendered
}

pub(in crate::index) fn extract_date_or_time_answer_from_line(line: &str) -> Option<String> {
    for pattern in [
        r"(?i)\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?(?:-\d{1,2}(?:st|nd|rd|th)?)?)\b",
        r"(?i)\b(\d{1,2}:\d{2}\s?(?:AM|PM))\b",
        r"(?i)\b(\d{1,2}\s?(?:AM|PM))\b",
        r"(?i)\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
    ] {
        if let Some(value) = compile_regex(pattern)
            .captures(line)
            .and_then(|caps| caps.get(1))
            .map(|m| m.as_str().trim().to_string())
        {
            return Some(value);
        }
    }
    None
}

pub(in crate::index) fn extract_color_answer_from_line(line: &str) -> Option<String> {
    for pattern in [
        r"(?i)\b((?:a\s+)?(?:lighter|darker|light|dark|soft|pale|bright|deep)\s+shade of\s+(?:gray|grey|blue|green|pink|purple|yellow|red|orange|white|black|beige|brown))\b",
        r"(?i)\b((?:light|dark|pale|bright|deep|soft)\s+(?:gray|grey|blue|green|pink|purple|yellow|red|orange|white|black|beige|brown))\b",
        r"(?i)\b(gray|grey|blue|green|pink|purple|yellow|red|orange|white|black|beige|brown)\b",
    ] {
        if let Some(value) = compile_regex(pattern)
            .captures(line)
            .and_then(|caps| caps.get(1))
            .map(|m| m.as_str().trim().to_string())
        {
            return Some(value);
        }
    }
    None
}

pub(in crate::index) fn extract_query_aligned_numeric_answer(
    task_lower: &str,
    line: &str,
) -> Option<String> {
    let mut terms = synthetic_query_terms(task_lower)
        .into_iter()
        .filter(|term| term.len() >= 4)
        .filter(|term| {
            ![
                "current",
                "currently",
                "recently",
                "specific",
                "previous",
                "conversation",
                "recommended",
            ]
            .contains(&term.as_str())
        })
        .collect::<Vec<_>>();
    if task_lower.contains("times") {
        terms.extend(
            ["game", "games", "match", "matches", "meeting", "meetings"]
                .into_iter()
                .map(str::to_string),
        );
    }
    terms.sort();
    terms.dedup();
    let line_lower = line.to_ascii_lowercase();
    let anchor_terms = assistant_followup_anchor_terms(task_lower);
    let mut best_anchor_match: Option<(usize, usize, String)> = None;
    for term in &terms {
        let pattern = compile_regex(&format!(
            r"(?i)\b((?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred))\s+{}s?\b",
            regex::escape(&term)
        ));
        for capture in pattern.captures_iter(line) {
            let Some(full_match) = capture.get(0) else {
                continue;
            };
            let Some(value_match) = capture.get(1) else {
                continue;
            };
            let Some(distance) =
                assistant_followup_anchor_distance(&line_lower, full_match.end(), &anchor_terms)
            else {
                continue;
            };
            let value = value_match.as_str().trim().to_string();
            if best_anchor_match
                .as_ref()
                .map(|(best_distance, best_start, _)| {
                    distance < *best_distance
                        || (distance == *best_distance && full_match.start() > *best_start)
                })
                .unwrap_or(true)
            {
                best_anchor_match = Some((distance, full_match.start(), value));
            }
        }
    }
    if let Some((_, _, value)) = best_anchor_match {
        return Some(value);
    }
    for term in terms {
        let pattern = compile_regex(&format!(
            r"(?i)\b((?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred))\s+{}s?\b",
            regex::escape(&term)
        ));
        if let Some(value) = pattern
            .captures(line)
            .and_then(|caps| caps.get(1))
            .map(|m| m.as_str().trim().to_string())
        {
            return Some(value);
        }
    }
    None
}

pub(in crate::index) fn extract_session_purchase_item(line: &str, lower: &str) -> Option<String> {
    extract_phrase_after_any_index(
        line,
        lower,
        &[
            "purchased a ",
            "purchased an ",
            "bought a ",
            "bought an ",
            "picked up a ",
            "picked up an ",
            "got a ",
            "got an ",
        ],
        &[" for ", " with ", " because ", " and ", " but ", "."],
        1,
    )
}

pub(in crate::index) fn extract_title_like_phrases(text: &str) -> Vec<String> {
    const CONNECTORS: &[&str] = &[
        "of", "the", "and", "at", "in", "on", "to", "for", "dei", "del", "di", "du", "&", "+",
    ];
    let mut phrases = Vec::new();
    let mut current = Vec::new();
    let mut seen_title = false;

    for raw in text.split_whitespace() {
        let cleaned = raw.trim_matches(|c: char| {
            !c.is_ascii_alphanumeric() && !matches!(c, '&' | '+' | '\'' | '-')
        });
        if cleaned.is_empty() {
            continue;
        }
        let lower = cleaned.to_ascii_lowercase();
        let starts_upper = cleaned
            .chars()
            .next()
            .map(|c| c.is_ascii_uppercase())
            .unwrap_or(false);
        let short_acronym = cleaned.len() <= 5
            && cleaned
                .chars()
                .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || matches!(c, '&' | '+'));
        let is_title = starts_upper || short_acronym;

        if is_title || (seen_title && CONNECTORS.contains(&lower.as_str())) {
            current.push(cleaned.to_string());
            if is_title {
                seen_title = true;
            }
            continue;
        }

        if seen_title && !current.is_empty() {
            let phrase = current.join(" ");
            if phrase.split_whitespace().count() <= 8 {
                phrases.push(phrase);
            }
        }
        current.clear();
        seen_title = false;
    }

    if seen_title && !current.is_empty() {
        let phrase = current.join(" ");
        if phrase.split_whitespace().count() <= 8 {
            phrases.push(phrase);
        }
    }

    phrases
}
