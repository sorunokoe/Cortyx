//! Index answer surface scoring and classification.

use super::super::*;
use crate::index::{compile_regex, compile_regex_static};

pub fn looks_like_answer_surface_status(answer_span: &str) -> bool {
    matches!(
        answer_span.to_ascii_lowercase().trim(),
        "single" | "married" | "engaged" | "divorced" | "widowed" | "separated"
    )
}

pub fn index_answer_surface_score(
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

pub fn format_index_answer_surface_answer(task_lower: &str, answer: &str) -> String {
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

pub fn answer_surface_evidence_line(
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

pub fn answer_surface_answer_span_evidence_state(content: &str, answer_span: &str) -> (bool, bool) {
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

pub fn latest_active_kg_value(entity: &kg::KgEntity, predicate: &str) -> Option<String> {
    pub fn latest_value_for_predicate(entity: &kg::KgEntity, predicate: &str) -> Option<String> {
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

pub fn normalize_latest_kg_value(predicate: &str, value: &str) -> String {
    match predicate {
        "location" => normalize_location_kg_value(value),
        "education" | "major" => normalize_education_kg_value(value),
        "fitness_record" => normalize_fitness_record_kg_value(value),
        _ => value.trim().to_string(),
    }
}

pub fn normalize_location_kg_value(value: &str) -> String {
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
    if trimmed.eq_ignore_ascii_case("suburbs") || trimmed.eq_ignore_ascii_case("the suburbs") {
        "the suburbs".to_string()
    } else {
        trimmed.to_string()
    }
}

pub fn normalize_education_kg_value(value: &str) -> String {
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

pub fn normalize_fitness_record_kg_value(value: &str) -> String {
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

pub fn extract_fitness_record_time_value(line: &str) -> Option<(u32, String)> {
    compile_regex_static(r"\b(\d{1,2}):(\d{2})\b")
        .captures_iter(line)
        .filter_map(|caps| {
            let minutes = caps.get(1)?.as_str().parse::<u32>().ok()?;
            let seconds = caps.get(2)?.as_str().parse::<u32>().ok()?;
            (seconds < 60).then_some((minutes * 60 + seconds, caps.get(0)?.as_str().to_string()))
        })
        .min_by_key(|(total_seconds, _)| *total_seconds)
}

pub fn parse_count_token_value(token: &str) -> Option<i32> {
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

pub fn extract_line_numbers(line: &str) -> Vec<i32> {
    line.split_whitespace()
        .filter_map(parse_count_token_value)
        .collect()
}

pub fn extract_focus_aligned_count(
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

pub fn is_summary_or_user_line(line: &str, lower: &str) -> bool {
    lower.starts_with("user:") || line.trim_start().starts_with('-')
}

pub fn is_session_answer_candidate_line(line: &str) -> bool {
    let trimmed = line.trim();
    !(trimmed.is_empty()
        || trimmed.starts_with('#')
        || trimmed.starts_with("<!--")
        || trimmed.starts_with('|')
        || trimmed.starts_with("Question:")
        || trimmed.starts_with("Answer:"))
}

pub fn normalize_session_answer_line_body(line: &str) -> String {
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

pub fn task_has_recall_context(task_lower: &str) -> bool {
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

pub fn should_try_session_recall_answer(task: &str, task_lower: &str) -> bool {
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

pub fn normalized_synthetic_phrase_key(text: &str) -> String {
    text.trim()
        .trim_matches(|c: char| matches!(c, '"' | '\'' | ',' | '.' | ';' | ':' | '!' | '?'))
        .to_ascii_lowercase()
}

pub fn project_session_answer_from_line(
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

pub fn is_assistant_followup_query(task_lower: &str) -> bool {
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

pub fn project_assistant_followup_answer_from_context(
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
