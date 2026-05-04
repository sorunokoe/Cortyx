use super::*;

pub(super) fn update_best_answer(best: &mut Option<(f32, String)>, score: f32, answer: String) {
    if best
        .as_ref()
        .map(|(best_score, _)| score > *best_score)
        .unwrap_or(true)
    {
        *best = Some((score, answer));
    }
}

pub(super) fn extract_subject_hints(task: &str) -> Vec<String> {
    let mut hints = Vec::new();
    for token in task.split(|c: char| !c.is_ascii_alphabetic() && c != '-') {
        let trimmed = token.trim();
        if trimmed.len() < 3 {
            continue;
        }
        let Some(first) = trimmed.chars().next() else {
            continue;
        };
        if !first.is_ascii_uppercase() {
            continue;
        }
        let lower = trimmed.to_ascii_lowercase();
        if ENTITY_STOPWORDS.contains(&lower.as_str()) {
            continue;
        }
        hints.push(lower);
    }
    hints.sort();
    hints.dedup();
    hints
}

pub(super) fn dialogue_focus_terms(
    task: &str,
    task_terms: &[String],
    subject_hints: &[String],
) -> Vec<String> {
    let lower = task.to_ascii_lowercase();
    let mut focus = task_terms
        .iter()
        .filter(|term| !subject_hints.iter().any(|hint| hint == *term))
        .cloned()
        .collect::<Vec<_>>();
    focus.retain(|term| !matches!(term.as_str(), "likely" | "current" | "currently"));

    if is_education_field_query(&lower) {
        focus.extend(
            [
                "job",
                "jobs",
                "career",
                "work",
                "working",
                "study",
                "studying",
                "education",
                "school",
            ]
            .iter()
            .map(|term| (*term).to_string()),
        );
    }

    if lower.contains("research") || lower.contains("looking into") || lower.contains("look into") {
        focus.extend(
            [
                "research",
                "researching",
                "looking",
                "into",
                "check",
                "checking",
            ]
            .iter()
            .map(|term| (*term).to_string()),
        );
    }

    if lower.contains("support group") {
        focus.extend(["support", "group"].iter().map(|term| (*term).to_string()));
    }

    focus.sort();
    focus.dedup();
    focus
}

pub(super) fn turn_matches_subject(turn: &DialogueTurn, subject_hints: &[String]) -> bool {
    if subject_hints.is_empty() {
        return true;
    }
    speaker_match_bonus(turn.speaker.as_deref(), subject_hints) > 0.0
        || task_overlap_count(&turn.text, subject_hints) > 0
}

pub(super) fn normalize_match_term(term: &str) -> &str {
    term.strip_suffix("'s")
        .or_else(|| term.strip_suffix("s'"))
        .unwrap_or(term)
}

pub(super) fn rough_match_term(term: &str) -> &str {
    term.strip_suffix("ing")
        .or_else(|| term.strip_suffix("ed"))
        .or_else(|| term.strip_suffix("es"))
        .or_else(|| term.strip_suffix('s'))
        .filter(|value| value.len() >= 4)
        .unwrap_or(term)
}

pub(super) fn common_prefix_len(left: &str, right: &str) -> usize {
    left.chars()
        .zip(right.chars())
        .take_while(|(l, r)| l == r)
        .count()
}

pub(super) fn within_edit_distance_one(left: &str, right: &str) -> bool {
    let left_chars = left.chars().collect::<Vec<_>>();
    let right_chars = right.chars().collect::<Vec<_>>();
    let left_len = left_chars.len();
    let right_len = right_chars.len();
    if left_len.abs_diff(right_len) > 1 {
        return false;
    }

    let mut left_idx = 0usize;
    let mut right_idx = 0usize;
    let mut seen_edit = false;
    while left_idx < left_len && right_idx < right_len {
        if left_chars[left_idx] == right_chars[right_idx] {
            left_idx += 1;
            right_idx += 1;
            continue;
        }
        if seen_edit {
            return false;
        }
        seen_edit = true;
        if left_len > right_len {
            left_idx += 1;
        } else if right_len > left_len {
            right_idx += 1;
        } else {
            left_idx += 1;
            right_idx += 1;
        }
    }
    true
}

pub(super) fn query_term_matches_token(term: &str, token: &str) -> bool {
    let left = rough_match_term(normalize_match_term(term));
    let right = rough_match_term(normalize_match_term(token));
    if left.is_empty() || right.is_empty() {
        return false;
    }
    if left == right {
        return true;
    }
    if left.len() >= 4 && right.starts_with(left) {
        return true;
    }
    if right.len() >= 5 && left.starts_with(right) {
        return true;
    }
    left.len() >= 6
        && right.len() >= 6
        && common_prefix_len(left, right) >= 4
        && within_edit_distance_one(left, right)
}

pub(super) fn term_list_overlap_count(left: &[String], right: &[String]) -> usize {
    left.iter()
        .filter(|term| {
            right
                .iter()
                .any(|candidate| query_term_matches_token(term, candidate))
        })
        .count()
}

pub(super) fn speaker_match_bonus(speaker: Option<&str>, subject_hints: &[String]) -> f32 {
    let Some(speaker) = speaker else {
        return 0.0;
    };
    let lower = speaker.to_ascii_lowercase();
    if subject_hints.iter().any(|hint| hint == &lower) {
        14.0
    } else {
        0.0
    }
}

pub(super) fn dialogue_match_score(text: &str, task_terms: &[String]) -> f32 {
    let overlap = task_overlap_count(text, task_terms) as f32;
    candidate_weight(text, task_terms, 0.0, false) + overlap * 6.0
}

pub(super) fn extract_turn_answer(task: &str, text: &str, task_terms: &[String]) -> Option<String> {
    let clean = sanitize_answer_text(text);
    if clean.is_empty() {
        return None;
    }

    if is_reason_query(task) {
        if let Some(reason) = extract_reason_answer(&clean) {
            return Some(reason);
        }
    }

    if let Some(answer) = extract_relation_answer(task, &clean, task_terms) {
        return Some(answer);
    }

    if let Some(compact) = compact_answer(task, &clean, task_terms) {
        if is_informative_compact_answer(&compact) {
            return Some(compact);
        }
    }

    if task.to_ascii_lowercase().contains("research")
        && clean.to_ascii_lowercase().contains("research")
    {
        return None;
    }

    Some(summarize_turn_text(&clean, task_terms))
}

pub(super) fn is_reason_query(task: &str) -> bool {
    let lower = task.to_ascii_lowercase();
    lower.starts_with("why ")
        || lower.contains(" motivated ")
        || lower.contains("motivate")
        || lower.contains("inspired")
        || lower.contains(" inspire ")
        || lower.contains("what made")
        || lower.contains("what pushed")
        || lower.contains("what gave")
}

pub(super) fn extract_reason_answer(text: &str) -> Option<String> {
    let clean = sanitize_inline(text);
    if clean.is_empty() {
        return None;
    }

    let lower = clean.to_ascii_lowercase();
    for marker in ["because ", "since ", "after ", "from "] {
        if let Some(idx) = lower.find(marker) {
            let phrase = trim_answer_tail(&clean[idx + marker.len()..], false);
            if phrase.split_whitespace().count() >= 3 {
                return Some(phrase);
            }
        }
    }

    let mut first_clause = clean
        .split(['.', '!', '?', ';'])
        .map(str::trim)
        .find(|clause| clause.split_whitespace().count() >= 4)?
        .to_string();
    let lower_clause = first_clause.to_ascii_lowercase();
    for boundary in [", and i ", " and i ", ", but i ", " but i "] {
        if let Some(idx) = lower_clause.find(boundary) {
            let head = first_clause[..idx].trim();
            if head.split_whitespace().count() >= 4 {
                first_clause = head.to_string();
                break;
            }
        }
    }
    Some(sanitize_inline(&first_clause))
}

pub(super) fn relation_answer_markers(lower_task: &str) -> &'static [&'static str] {
    if lower_task.contains("raise awareness") {
        &["awareness for "]
    } else if lower_task.contains("work with") {
        &["worked with ", "working with ", "collaborated with "]
    } else if lower_task.contains("blog") || lower_task.contains("topic") {
        &["blogging about ", "writing about ", "posting about "]
    } else if lower_task.contains("fan of") {
        &["fan of "]
    } else if lower_task.contains("screenplay") {
        &[
            "screenplay about ",
            "screenplay explores ",
            "screenplay is about ",
            "movie about ",
            "story about ",
        ]
    } else if lower_task.contains("letter about") {
        &["letter about ", "wrote me a letter about "]
    } else if lower_task.contains("share") {
        &["shared ", "share "]
    } else if lower_task.contains("play") || lower_task.contains("game convention") {
        &["played ", "playing "]
    } else if lower_task.contains("feel") {
        &["felt ", "feeling "]
    } else if lower_task.contains("plan") || lower_task.contains("later on") {
        &["planned to ", "planning to "]
    } else if lower_task.contains("opening") || lower_task.contains("working on opening") {
        &["working on opening ", "opening ", "working on "]
    } else if lower_task.contains("join") || lower_task.contains("group") {
        &["joined a ", "joined an ", "joined "]
    } else if lower_task.contains("teach") && lower_task.contains("kids") {
        &[
            "teach my kids ",
            "teach his kids ",
            "teach her kids ",
            "teach our kids ",
        ]
    } else {
        &[]
    }
}

pub(super) fn extract_relation_answer(
    task: &str,
    text: &str,
    task_terms: &[String],
) -> Option<String> {
    let lower_task = task.to_ascii_lowercase();
    if lower_task.contains("ingredient") || lower_task.contains("recipe") {
        if let Some(list) = extract_ingredient_list(text) {
            return Some(list);
        }
    }

    let markers = relation_answer_markers(&lower_task);

    for marker in markers {
        if let Some(answer) = extract_after_marker(task, text, marker, task_terms) {
            if lower_task.contains("group") && !answer.to_ascii_lowercase().contains("group") {
                continue;
            }
            return Some(answer);
        }
    }
    None
}

pub(super) fn extract_after_marker(
    task: &str,
    text: &str,
    marker: &str,
    task_terms: &[String],
) -> Option<String> {
    let lower_text = text.to_ascii_lowercase();
    let idx = lower_text.find(marker)?;
    let tail = &text[idx + marker.len()..];
    let mut phrase = trim_answer_tail(tail, true);
    if let Some(to_idx) = phrase.to_ascii_lowercase().find(" to ") {
        let head = phrase[..to_idx].trim();
        if head.split_whitespace().count() >= 2 {
            phrase = head.to_string();
        }
    }
    if marker.starts_with("shared") || marker.starts_with("share ") {
        if let Some(with_idx) = phrase.to_ascii_lowercase().find(" with ") {
            let head = phrase[..with_idx].trim();
            if head.split_whitespace().count() >= 2 {
                phrase = head.to_string();
            }
        }
    }
    if marker.starts_with("played") || marker.starts_with("playing") {
        if let Some(at_idx) = phrase.to_ascii_lowercase().find(" at ") {
            let head = phrase[..at_idx].trim();
            if head.split_whitespace().count() >= 1 {
                phrase = head.to_string();
            }
        }
    }
    if marker.starts_with("planned") || marker.starts_with("planning") {
        if let Some(later_idx) = phrase.to_ascii_lowercase().find(" later ") {
            let head = phrase[..later_idx].trim();
            if head.split_whitespace().count() >= 2 {
                phrase = head.to_string();
            }
        }
    }
    is_plausible_compact_answer(task, &phrase, task_terms)
        .then_some(phrase)
        .filter(|answer| is_informative_compact_answer(answer))
}

pub(super) fn extract_ingredient_list(text: &str) -> Option<String> {
    if text.contains('?') {
        return None;
    }

    let clean = sanitize_answer_text(text);
    if clean.is_empty() || !clean.contains(',') {
        return None;
    }

    let normalized = clean.replace(" and ", ", ");
    let mut parts = normalized
        .split(',')
        .map(str::trim)
        .filter(|part| {
            let words = part.split_whitespace().count();
            words >= 1
                && words <= 4
                && !part.eq_ignore_ascii_case("and")
                && !part.eq_ignore_ascii_case("or")
        })
        .map(sanitize_inline)
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    parts.sort();
    parts.dedup();
    (parts.len() >= 3).then(|| parts.into_iter().take(4).collect::<Vec<_>>().join(", "))
}

pub(super) fn summarize_turn_text(text: &str, task_terms: &[String]) -> String {
    let mut best = sanitize_inline(text);
    let mut best_score = candidate_weight(&best, task_terms, 0.0, false);

    for fragment in split_candidate_fragments(text) {
        let clean = sanitize_inline(&fragment);
        if clean.is_empty() {
            continue;
        }
        let score = candidate_weight(&clean, task_terms, 0.0, false);
        if score > best_score {
            best_score = score;
            best = clean;
        }
    }

    best.split_whitespace()
        .take(24)
        .collect::<Vec<_>>()
        .join(" ")
}

pub(super) fn is_informative_compact_answer(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    let words = lower.split_whitespace().collect::<Vec<_>>();
    if words.is_empty() {
        return false;
    }
    if words.len() == 1 {
        let token = words[0];
        if matches!(
            token,
            "good" | "time" | "part" | "customers" | "topic" | "topics" | "again" | "them"
        ) {
            return false;
        }
        if token.chars().all(|c| c.is_ascii_lowercase()) && token.len() <= 4 {
            return false;
        }
    }
    if words.len() == 2
        && matches!(
            lower.as_str(),
            "a good"
                | "the cause"
                | "the convention"
                | "my favorite"
                | "my customers"
                | "your rock"
                | "to chat"
                | "those topics"
                | "having them"
        )
    {
        return false;
    }
    true
}

pub(super) fn extract_derived_answer(content: &str) -> Option<String> {
    for line in content.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("Answer:") {
            let clean = sanitize_answer_text(rest);
            if !clean.is_empty() {
                return Some(clean);
            }
        }
    }
    None
}

pub(super) fn derived_answer_is_explicit_abstention(answer: &str) -> bool {
    let lower = answer.trim().to_ascii_lowercase();
    lower.starts_with("the information provided is not enough")
        || lower.starts_with("you did not mention")
        || lower.starts_with("the information provided doesn't say")
}

pub(super) fn candidate_weight(
    text: &str,
    task_terms: &[String],
    retrieval_score: f32,
    from_summary: bool,
) -> f32 {
    let lower = text.to_lowercase();
    let overlap = task_overlap_count(text, task_terms) as f32;
    let raw_token_count = text.split_whitespace().count();
    let token_count = raw_token_count.min(16) as f32;
    let has_number = lower.chars().any(|c| c.is_ascii_digit());
    let has_month = [
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
    .iter()
    .any(|month| lower.contains(month));
    let density_bonus = if has_number || has_month { 4.0 } else { 0.0 };
    let summary_bonus = if from_summary { 3.0 } else { 0.0 };
    let informational_bonus = token_count * 0.2;
    let concision_bonus = match raw_token_count {
        1..=3 => 0.5,
        4..=16 => 1.5,
        17..=28 => 0.0,
        _ => -2.0,
    };
    retrieval_score * 10.0
        + overlap * 15.0
        + density_bonus
        + summary_bonus
        + informational_bonus
        + concision_bonus
}

pub(super) fn task_overlap_count(text: &str, task_terms: &[String]) -> usize {
    let lower = text.to_ascii_lowercase();
    let tokens = lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>();
    task_terms
        .iter()
        .filter(|term| {
            tokens
                .iter()
                .any(|token| query_term_matches_token(term, token))
        })
        .count()
}

pub(super) fn max_task_overlap<'a>(
    texts: impl IntoIterator<Item = &'a str>,
    task_terms: &[String],
) -> usize {
    texts
        .into_iter()
        .map(|text| task_overlap_count(text, task_terms))
        .max()
        .unwrap_or(0)
}

pub(super) fn candidate_has_required_anchor_support(task: &str, candidate: &CandidateLine) -> bool {
    if !looks_like_typed_open_qa_query(task)
        && !looks_like_relation_query(task)
        && parse_binary_choice(task).is_none()
        && parse_open_qa_choice_options(task).is_empty()
    {
        return true;
    }
    let task_terms = salient_query_terms(task);
    let subject_hints = extract_subject_hints(task);
    let anchor_terms = task_anchor_terms(task, &task_terms, &subject_hints);
    if institution_query_expected(task) {
        let specific_anchor_terms = institution_specific_anchor_terms(task);
        if !specific_anchor_terms.is_empty() {
            let min_overlap = if specific_anchor_terms.len() >= 2 {
                2
            } else {
                1
            };
            return candidate.specific_anchor_overlap >= min_overlap;
        }
    }
    anchor_terms.is_empty() || candidate.anchor_overlap > 0
}

pub(super) fn validate_selected_answer(
    task: &str,
    answer: Option<String>,
    min_answer_confidence: Option<f32>,
) -> Option<String> {
    answer.filter(|answer| answer_meets_form_gate(task, answer, min_answer_confidence))
}

pub(super) fn is_reading_progress_pages_left_query(task: &str) -> bool {
    task.to_ascii_lowercase().contains("pages do i have left")
}

pub(super) fn answer_meets_form_gate(
    task: &str,
    text: &str,
    min_answer_confidence: Option<f32>,
) -> bool {
    let task_terms = salient_query_terms(task);
    let confidence = answer_form_confidence(task, text, &task_terms);
    confidence > 0.0
        && min_answer_confidence
            .map(|threshold| confidence >= threshold)
            .unwrap_or(true)
}

pub(super) fn salient_query_terms(task: &str) -> Vec<String> {
    let mut terms: Vec<String> = task
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter_map(|term| {
            let lower = term
                .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '\'')
                .to_lowercase();
            if lower.len() < 3 || QUESTION_STOPWORDS.contains(&lower.as_str()) {
                None
            } else {
                Some(lower)
            }
        })
        .collect();
    terms.sort();
    terms.dedup();
    terms
}

pub(super) fn is_enumerative_query(task: &str) -> bool {
    let lower = task.to_lowercase();
    if lower.contains(" or ")
        || lower.contains(" first")
        || lower.contains(" second")
        || lower.contains(" earlier")
        || lower.contains(" later")
        || lower.contains(" before ")
        || lower.contains(" after ")
    {
        return false;
    }

    lower.contains("list ")
        || lower.contains("what are")
        || lower.contains("who are")
        || lower.contains("which are")
        || lower.contains("which ones")
        || lower.contains("which people")
        || lower.contains("which items")
        || lower.contains("which activities")
        || lower.contains("which topics")
        || lower.contains("which books")
        || lower.contains("which movies")
        || lower.contains("which events were")
}

pub(super) fn sanitize_answer_text(text: &str) -> String {
    let mut line = text.trim().trim_start_matches("- ").trim().to_string();
    if let Some((prefix, rest)) = line.split_once(": ") {
        let words = prefix.split_whitespace().count();
        let alpha_like = prefix
            .chars()
            .all(|c| c.is_alphabetic() || c == ' ' || c == '-' || c == '\'');
        if words <= 3 && alpha_like {
            line = rest.to_string();
        }
    }
    collapse_inline_whitespace(&line)
}

pub(super) fn sanitize_inline(text: &str) -> String {
    collapse_inline_whitespace(text).chars().take(240).collect()
}

pub(super) fn collapse_inline_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub(super) fn parse_binary_choice(task: &str) -> Option<(Vec<ChoiceOption>, TemporalDirection)> {
    let lower = task.to_ascii_lowercase();
    let direction = if lower.contains(" first")
        || lower.contains(" earlier")
        || lower.contains(" before ")
        || lower.contains(" oldest")
    {
        TemporalDirection::Earlier
    } else if lower.contains(" later")
        || lower.contains(" last")
        || lower.contains(" after ")
        || lower.contains(" newest")
        || lower.contains(" most recent")
    {
        TemporalDirection::Later
    } else {
        return None;
    };

    let tail = task
        .rsplit_once(',')
        .map(|(_, rest)| rest.trim())
        .unwrap_or(task.trim())
        .trim_end_matches('?')
        .trim();
    if !tail.to_ascii_lowercase().contains(" or ") {
        return None;
    }
    let mut parts = tail.splitn(2, " or ");
    let left = parts.next()?.trim();
    let right = parts.next()?.trim();
    let options = [left, right]
        .into_iter()
        .map(|raw| {
            let display = raw
                .trim()
                .trim_start_matches("the ")
                .trim_start_matches("a ")
                .trim_start_matches("an ")
                .trim_matches(|c: char| c == '?' || c == ',' || c == '.')
                .to_string();
            let tokens = display
                .split(|c: char| !c.is_alphanumeric())
                .filter_map(|token| {
                    let lower = token.to_ascii_lowercase();
                    if lower.len() < 2
                        || QUESTION_STOPWORDS.contains(&lower.as_str())
                        || parse_count_token(&lower).is_some()
                    {
                        None
                    } else {
                        Some(lower)
                    }
                })
                .collect::<Vec<_>>();
            ChoiceOption { display, tokens }
        })
        .filter(|option| !option.display.is_empty() && !option.tokens.is_empty())
        .collect::<Vec<_>>();
    if options.len() == 2 {
        Some((options, direction))
    } else {
        None
    }
}

pub(super) fn extract_session_base_date(content: &str) -> Option<(i32, u32, u32)> {
    content
        .lines()
        .take(8)
        .find_map(|line| extract_explicit_date(line, None))
}

pub(super) fn extract_temporal_rank(line: &str, base_date: Option<(i32, u32, u32)>) -> Option<i32> {
    if let Some(date) = extract_explicit_date(line, base_date) {
        return Some(ymd_to_days(date.0, date.1, date.2));
    }
    if let Some(days_ago) = extract_relative_days(line) {
        if let Some(base) = base_date {
            let base_days = ymd_to_days(base.0, base.1, base.2);
            Some(base_days - days_ago)
        } else {
            Some(-days_ago)
        }
    } else {
        None
    }
}

pub(super) fn extract_explicit_date(
    text: &str,
    base_date: Option<(i32, u32, u32)>,
) -> Option<(i32, u32, u32)> {
    let lower = text.to_ascii_lowercase();
    let year_hint = base_date.map(|(year, _, _)| year);
    if let Some(date) = extract_numeric_slash_date(text, year_hint) {
        return Some(date);
    }
    for (month_idx, month) in [
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
    .iter()
    .enumerate()
    {
        if let Some(pos) = lower.find(month) {
            let before = &lower[..pos];
            let after = &lower[pos + month.len()..];
            let day = extract_nearest_day(before, after, &lower, pos).unwrap_or_else(|| {
                if before.ends_with("mid-") || before.ends_with("mid ") {
                    15
                } else if before.ends_with("early-") || before.ends_with("early ") {
                    5
                } else if before.ends_with("late-") || before.ends_with("late ") {
                    25
                } else {
                    15
                }
            });
            let year = extract_year_near(after).or(year_hint).unwrap_or(2023);
            return Some((year, (month_idx + 1) as u32, day));
        }
    }
    if let Some(date) = extract_named_holiday_date(&lower, year_hint) {
        return Some(date);
    }
    None
}

pub(super) fn extract_numeric_slash_date(
    text: &str,
    year_hint: Option<i32>,
) -> Option<(i32, u32, u32)> {
    for raw in text.split_whitespace() {
        let clean = raw.trim_matches(|c: char| !c.is_ascii_digit() && c != '/');
        if clean.len() < 3 || !clean.contains('/') {
            continue;
        }
        let parts = clean
            .split('/')
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>();
        if parts.len() < 2 || parts.len() > 3 {
            continue;
        }
        let Some(month) = parts[0].parse::<u32>().ok() else {
            continue;
        };
        let Some(day) = parts[1].parse::<u32>().ok() else {
            continue;
        };
        if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
            continue;
        }
        let year = parts
            .get(2)
            .and_then(|part| {
                if part.len() == 4 {
                    part.parse::<i32>().ok()
                } else {
                    None
                }
            })
            .or(year_hint)
            .unwrap_or(2023);
        return Some((year, month, day));
    }
    None
}

pub(super) fn extract_named_holiday_date(
    lower: &str,
    year_hint: Option<i32>,
) -> Option<(i32, u32, u32)> {
    let year = year_hint.unwrap_or(2023);
    if lower.contains("black friday") {
        return Some(black_friday_date(year));
    }
    if lower.contains("thanksgiving") {
        return Some(thanksgiving_date(year));
    }
    if lower.contains("christmas eve") {
        return Some((year, 12, 24));
    }
    if lower.contains("christmas") {
        return Some((year, 12, 25));
    }
    if lower.contains("maundy thursday") {
        return Some(shift_date_by_days(easter_sunday_date(year), -3));
    }
    if lower.contains("good friday") {
        return Some(shift_date_by_days(easter_sunday_date(year), -2));
    }
    if lower.contains("ash wednesday") {
        return Some(shift_date_by_days(easter_sunday_date(year), -46));
    }
    if lower.contains("easter monday") {
        return Some(shift_date_by_days(easter_sunday_date(year), 1));
    }
    if lower.contains("easter sunday") || contains_standalone_token(lower, "easter") {
        return Some(easter_sunday_date(year));
    }
    if lower.contains("holi") {
        return Some(match year {
            2023 => (2023, 3, 8),
            2024 => (2024, 3, 25),
            2025 => (2025, 3, 14),
            2026 => (2026, 3, 3),
            _ => (year, 3, 8),
        });
    }
    None
}

pub(super) fn thanksgiving_date(year: i32) -> (i32, u32, u32) {
    let november_first = ymd_to_days(year, 11, 1);
    let november_first_weekday = (4 + november_first).rem_euclid(7);
    let days_until_thursday = (4 - november_first_weekday).rem_euclid(7);
    let thanksgiving_day = 1 + days_until_thursday as u32 + 21;
    (year, 11, thanksgiving_day)
}

pub(super) fn black_friday_date(year: i32) -> (i32, u32, u32) {
    shift_date_by_days(thanksgiving_date(year), 1)
}

pub(super) fn easter_sunday_date(year: i32) -> (i32, u32, u32) {
    let a = year % 19;
    let b = year / 100;
    let c = year % 100;
    let d = b / 4;
    let e = b % 4;
    let f = (b + 8) / 25;
    let g = (b - f + 1) / 3;
    let h = (19 * a + b - d - g + 15) % 30;
    let i = c / 4;
    let k = c % 4;
    let l = (32 + 2 * e + 2 * i - h - k) % 7;
    let m = (a + 11 * h + 22 * l) / 451;
    let month = (h + l - 7 * m + 114) / 31;
    let day = ((h + l - 7 * m + 114) % 31) + 1;
    (year, month as u32, day as u32)
}

pub(super) fn extract_nearest_day(
    before: &str,
    after: &str,
    lower: &str,
    month_pos: usize,
) -> Option<u32> {
    extract_last_number(before)
        .or_else(|| extract_first_number(after))
        .and_then(|value| (1..=31).contains(&value).then_some(value as u32))
        .or_else(|| {
            let around = safe_slice(
                lower,
                month_pos.saturating_sub(8),
                (month_pos + 20).min(lower.len()),
            );
            if around.contains("mid-") || around.contains("mid ") {
                Some(15)
            } else if around.contains("early-") || around.contains("early ") {
                Some(5)
            } else if around.contains("late-") || around.contains("late ") {
                Some(25)
            } else {
                None
            }
        })
}

pub(super) fn extract_year_near(after: &str) -> Option<i32> {
    after
        .split(|c: char| !c.is_ascii_digit())
        .find_map(|token| {
            if token.len() == 4 {
                token.parse::<i32>().ok()
            } else {
                None
            }
        })
}

pub(super) fn extract_last_number(text: &str) -> Option<i32> {
    text.split(|c: char| !c.is_ascii_digit())
        .filter(|token| !token.is_empty())
        .filter_map(|token| token.parse::<i32>().ok())
        .last()
}

pub(super) fn extract_first_number(text: &str) -> Option<i32> {
    text.split(|c: char| !c.is_ascii_digit()).find_map(|token| {
        (!token.is_empty())
            .then(|| token.parse::<i32>().ok())
            .flatten()
    })
}

pub(super) fn extract_relative_days(text: &str) -> Option<i32> {
    let lower = text.to_ascii_lowercase();
    if lower.contains("yesterday") {
        return Some(1);
    }
    if lower.contains("a couple of days ago") {
        return Some(2);
    }
    if lower.contains("a few days ago") {
        return Some(3);
    }
    if lower.contains("last week") {
        return Some(7);
    }
    if lower.contains("last month") {
        return Some(30);
    }
    if [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ]
    .iter()
    .any(|day| lower.contains(&format!("last {day}")))
    {
        return Some(7);
    }

    for unit in ["day", "week", "month"] {
        for marker in [format!("{unit} ago"), format!("{unit}s ago")] {
            if !lower.contains(&marker) {
                continue;
            }
            if let Some(prefix) = lower.split(&marker).next() {
                if let Some(amount) = extract_trailing_count(prefix) {
                    let scale = match unit {
                        "day" => 1,
                        "week" => 7,
                        "month" => 30,
                        _ => 1,
                    };
                    return Some(amount * scale);
                }
            }
        }
    }
    None
}

pub(super) fn extract_trailing_count(prefix: &str) -> Option<i32> {
    let token = prefix
        .split_whitespace()
        .rev()
        .find(|token| !token.is_empty())?;
    parse_count_token(token)
}

pub(super) fn parse_count_token(token: &str) -> Option<i32> {
    let clean = token
        .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '+')
        .trim_end_matches('+');
    if let Ok(value) = clean.parse::<i32>() {
        return Some(value);
    }
    match clean {
        "a" | "an" | "one" => Some(1),
        "two" => Some(2),
        "three" => Some(3),
        "four" => Some(4),
        "five" => Some(5),
        "six" => Some(6),
        "seven" => Some(7),
        "eight" => Some(8),
        "nine" => Some(9),
        "ten" => Some(10),
        "eleven" => Some(11),
        "twelve" => Some(12),
        "couple" => Some(2),
        "few" => Some(3),
        _ => None,
    }
}

pub(super) fn ymd_to_days(year: i32, month: u32, day: u32) -> i32 {
    const MONTH_START_DAYS: [i32; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let leap_years = {
        let y = year - 1;
        y / 4 - y / 100 + y / 400 - (1969 / 4 - 1969 / 100 + 1969 / 400)
    };
    (year - 1970) * 365 + leap_years + MONTH_START_DAYS[(month - 1) as usize] + day as i32 - 1
}

pub(super) fn compact_answer(task: &str, text: &str, task_terms: &[String]) -> Option<String> {
    let lower_task = task.to_ascii_lowercase();

    if let Some(answer) = extract_after_action_marker(task, text, &lower_task, task_terms) {
        return Some(answer);
    }

    if let Some(answer) = extract_after_preposition(task, text, &lower_task, task_terms) {
        return Some(answer);
    }

    if let Some(answer) = extract_after_anchor_copula(task, text, task_terms) {
        return Some(answer);
    }

    None
}

pub(super) fn extract_after_action_marker(
    task: &str,
    text: &str,
    lower_task: &str,
    task_terms: &[String],
) -> Option<String> {
    let lower_text = text.to_ascii_lowercase();
    let markers: &[&str] = if lower_task.contains("blog") || lower_task.contains("topic") {
        &["blogging about ", "writing about ", "posting about "]
    } else if lower_task.contains("research") {
        &[
            "researched ",
            "researching ",
            "looking into ",
            "look into ",
            "checking out ",
            "check out ",
        ]
    } else if lower_task.contains("join") || lower_task.contains("group") {
        &["joined ", "join "]
    } else if lower_task.contains("open")
        || lower_task.contains("working on")
        || lower_task.contains("start")
        || lower_task.contains("business")
    {
        &[
            "starting ",
            "opening ",
            "building ",
            "launching ",
            "working on ",
            "planning ",
            "creating ",
        ]
    } else if is_education_field_query(lower_task) {
        &[
            "keen on ",
            "interested in ",
            "thinking of ",
            "thinking about ",
            "working in ",
            "looking into ",
            "look into ",
        ]
    } else {
        &[]
    };

    for marker in markers {
        if let Some(idx) = lower_text.find(marker) {
            let tail = &text[idx + marker.len()..];
            let mut phrase = trim_answer_tail(tail, true);
            if is_education_field_query(lower_task) {
                if let Some((head, rest)) = split_once_case_insensitive(&phrase, " or working in ")
                {
                    phrase = format!("{} or {}", head.trim(), rest.trim());
                } else if let Some(rest) = phrase.strip_prefix("working in ") {
                    phrase = rest.trim().to_string();
                }
            }
            if is_plausible_compact_answer(task, &phrase, task_terms) {
                return Some(phrase);
            }
        }
    }
    None
}

pub(super) fn extract_after_preposition(
    task: &str,
    text: &str,
    lower_task: &str,
    task_terms: &[String],
) -> Option<String> {
    let lower_text = text.to_ascii_lowercase();
    for prep in PREPOSITION_HINTS {
        let prep_marker = format!("{prep} ");
        if !contains_standalone_token(lower_task, prep) {
            continue;
        }
        let mut search_start = 0usize;
        let mut best: Option<(usize, String)> = None;
        while let Some(rel_idx) = lower_text[search_start..].find(&prep_marker) {
            let idx = search_start + rel_idx;
            let tail = &text[idx + prep_marker.len()..];
            let phrase = trim_answer_tail(tail, true);
            if is_plausible_compact_answer(task, &phrase, task_terms) {
                let window_start = idx.saturating_sub(96);
                let context = safe_slice(&lower_text, window_start, idx);
                let overlap = task_terms
                    .iter()
                    .filter(|term| context.contains(term.as_str()))
                    .count();
                let score = overlap * 10 + phrase.split_whitespace().count().min(8);
                if best
                    .as_ref()
                    .map(|(best_score, _)| score > *best_score)
                    .unwrap_or(true)
                {
                    best = Some((score, phrase));
                }
            }
            search_start = idx + prep_marker.len();
        }
        if let Some((_, phrase)) = best {
            return Some(phrase);
        }
    }
    None
}

pub(super) fn extract_after_anchor_copula(
    task: &str,
    text: &str,
    task_terms: &[String],
) -> Option<String> {
    let lower_text = text.to_ascii_lowercase();
    let mut anchors: Vec<&str> = task_terms.iter().map(String::as_str).collect();
    anchors.sort_by_key(|term| std::cmp::Reverse(term.len()));

    for anchor in anchors {
        if let Some(anchor_idx) = lower_text.find(anchor) {
            let after_anchor = &lower_text[anchor_idx + anchor.len()..];
            for marker in [" is ", " was ", " are ", " were ", ": "] {
                if let Some(marker_idx) = after_anchor.find(marker) {
                    let raw_tail = &text[anchor_idx + anchor.len() + marker_idx + marker.len()..];
                    let phrase = trim_answer_tail(raw_tail, marker != ": ");
                    if is_plausible_compact_answer(task, &phrase, task_terms) {
                        return Some(phrase);
                    }
                }
            }
        }
    }
    None
}

pub(super) fn trim_answer_tail(tail: &str, stop_on_copula: bool) -> String {
    let mut cleaned = sanitize_inline(tail);
    let lower = cleaned.to_ascii_lowercase();
    let mut cut = cleaned.len();

    for boundary in TAIL_BOUNDARIES {
        if let Some(idx) = lower.find(boundary) {
            cut = cut.min(idx);
        }
    }
    if stop_on_copula {
        for boundary in COPULA_BOUNDARIES {
            if let Some(idx) = lower.find(boundary) {
                cut = cut.min(idx);
            }
        }
    }
    cleaned.truncate(cut);

    cleaned = cleaned
        .trim_matches(|c: char| {
            matches!(
                c,
                '"' | '\'' | '(' | ')' | '[' | ']' | '{' | '}' | ',' | '.' | ';' | ':' | '!' | '?'
            )
        })
        .trim()
        .to_string();

    for prefix in ["the ", "a ", "an "] {
        if cleaned.to_ascii_lowercase().starts_with(prefix)
            && cleaned.split_whitespace().count() > 2
        {
            cleaned = cleaned[prefix.len()..].trim().to_string();
            break;
        }
    }

    cleaned
}

pub(super) fn contains_standalone_token(text: &str, token: &str) -> bool {
    text.split(|c: char| !c.is_ascii_alphanumeric())
        .any(|part| !part.is_empty() && part == token)
}

pub(super) fn safe_slice(text: &str, start: usize, end: usize) -> &str {
    fn floor_char_boundary(text: &str, mut idx: usize) -> usize {
        idx = idx.min(text.len());
        while idx > 0 && !text.is_char_boundary(idx) {
            idx -= 1;
        }
        idx
    }

    fn ceil_char_boundary(text: &str, mut idx: usize) -> usize {
        idx = idx.min(text.len());
        while idx < text.len() && !text.is_char_boundary(idx) {
            idx += 1;
        }
        idx
    }

    let start = floor_char_boundary(text, start);
    let end = ceil_char_boundary(text, end);
    if start >= end {
        ""
    } else {
        &text[start..end]
    }
}

pub(super) fn split_candidate_fragments(line: &str) -> Vec<String> {
    let mut fragments = vec![line.to_string()];
    for separator in ['.', '!', '?', ';'] {
        fragments = fragments
            .into_iter()
            .flat_map(|fragment| {
                fragment
                    .split(separator)
                    .map(str::trim)
                    .filter(|part| part.split_whitespace().count() >= 3)
                    .map(|part| part.to_string())
                    .collect::<Vec<_>>()
            })
            .collect();
    }

    let mut expanded = Vec::new();
    for fragment in fragments {
        expanded.push(fragment.clone());
        let discourse = strip_temporal_discourse_prefix(&fragment);
        if discourse != fragment && discourse.split_whitespace().count() >= 3 {
            expanded.push(discourse);
        }
        for marker in [
            " and got ",
            " and bought ",
            " and ordered ",
            " and attended ",
            " and joined ",
            " and redeemed ",
            " and signed up ",
            " and used ",
            " and received ",
            " and started ",
            " and finished ",
            " and discovered ",
            " and found ",
            " and took ",
            " and realized ",
        ] {
            if let Some((_, tail)) = split_once_case_insensitive(&fragment, marker) {
                let head = marker.trim().trim_start_matches("and ").to_string();
                let clause = format!("{head} {tail}").trim().to_string();
                if clause.split_whitespace().count() >= 3 {
                    expanded.push(clause);
                }
            }
        }
        for marker in [" - ", " — "] {
            for part in fragment.split(marker).map(str::trim) {
                if part.split_whitespace().count() >= 3 {
                    expanded.push(part.to_string());
                }
            }
        }
    }
    expanded.sort();
    expanded.dedup();
    expanded
}

pub(super) fn strip_temporal_discourse_prefix(text: &str) -> String {
    let mut clean = sanitize_inline(text);
    loop {
        let lower = clean.to_ascii_lowercase();
        if lower.starts_with("by the way, ") {
            clean = clean["by the way, ".len()..].trim().to_string();
            continue;
        }
        if lower.starts_with("by the way ") {
            clean = clean["by the way ".len()..].trim().to_string();
            continue;
        }
        if lower.starts_with("and by the way, ") {
            clean = clean["and by the way, ".len()..].trim().to_string();
            continue;
        }
        if lower.starts_with("oh, and by the way, ") {
            clean = clean["oh, and by the way, ".len()..].trim().to_string();
            continue;
        }
        if lower.starts_with("speaking of ") {
            if let Some((_, rest)) = clean.split_once(',') {
                clean = rest.trim().to_string();
                continue;
            }
        }
        if lower.starts_with("also, ") {
            clean = clean["also, ".len()..].trim().to_string();
            continue;
        }
        if lower.starts_with("oh, ") {
            clean = clean["oh, ".len()..].trim().to_string();
            continue;
        }
        break;
    }
    clean
}

pub(super) fn is_plausible_compact_answer(task: &str, text: &str, task_terms: &[String]) -> bool {
    if text.is_empty() {
        return false;
    }
    let word_count = text.split_whitespace().count();
    if word_count == 0 || word_count > 8 {
        return false;
    }
    if !text.chars().any(|c| c.is_alphanumeric()) {
        return false;
    }
    let lower = normalized_validation_text(text).to_ascii_lowercase();
    if !task.is_empty()
        && !is_temporal_reasoning_query(task)
        && answer_form_confidence(task, text, task_terms) <= 0.0
    {
        return false;
    }
    let overlap = task_terms
        .iter()
        .filter(|term| task_overlap_count(&lower, &[(*term).clone()]) > 0)
        .count();
    if overlap < task_terms.len().min(2) {
        return true;
    }

    let novel_tokens = lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|token| token.len() >= 3)
        .filter(|token| {
            !task_terms
                .iter()
                .any(|term| query_term_matches_token(term, token))
        })
        .count();
    novel_tokens > 0
}

pub(super) fn fallback_snippet(path: &Path) -> String {
    path.file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned()
}

pub(super) fn format_provenance_line(item: &EvidenceItem) -> String {
    let mut parts = Vec::new();
    parts.push(format!("{}", item.path.display()));
    parts.push(format!("score={:.1}", item.score));
    if let Some(metadata) = item.metadata.as_ref() {
        parts.push(format!("kind={}", kind_label(&metadata.kind)));
        if let Some(module) = metadata.module.as_deref() {
            parts.push(format!("module={module}"));
        }
        if let Some(ts) = metadata.timestamp_secs {
            parts.push(format!("time={}", format_timestamp(ts)));
        }
        parts.push(format!("tokens={}", metadata.tokens));
        if metadata.use_count > 0 {
            parts.push(format!(
                "hits={}/{}",
                metadata.hit_count, metadata.use_count
            ));
            parts.push(format!(
                "hit_rate={:.0}%",
                (metadata.hit_rate * 100.0).clamp(0.0, 100.0)
            ));
        }
    }
    format!("{} — {}", parts.join(", "), item.snippet)
}

pub(super) fn kind_label(kind: &NeuronKind) -> &'static str {
    match kind {
        NeuronKind::Core => "core",
        NeuronKind::Project => "project",
        NeuronKind::UseCase => "use_case",
        NeuronKind::Concept => "concept",
        NeuronKind::Verbatim => "verbatim",
        NeuronKind::Aggregate => "aggregate",
    }
}

pub(super) fn format_timestamp(timestamp_secs: i64) -> String {
    if timestamp_secs < 0 {
        return timestamp_secs.to_string();
    }
    let (y, mo, d, h, mi, s) = unix_secs_to_datetime(timestamp_secs as u64);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}
