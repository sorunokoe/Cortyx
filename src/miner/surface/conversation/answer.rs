use super::*;

pub(crate) fn generate_dialogue_answer_surface_rows(
    turns: &[Turn],
    index: usize,
) -> Vec<AnswerSurfaceRow> {
    if index == 0 {
        return Vec::new();
    }

    let question = &turns[index - 1];
    let answer = &turns[index];
    if question.speaker == answer.speaker {
        return Vec::new();
    }

    let mut rows = extract_dialogue_name_answer_surface_rows(question, answer);
    let Some(answer_span) = mine_dialogue_answer_surface_span(&question.text, &answer.text) else {
        return rows;
    };
    let mut answer_span = canonicalize_dialogue_answer_surface_span(&answer_span);
    if let Some(prefix) = extract_dialogue_statement_prefix(&answer.text) {
        let prefix_lower = prefix.to_ascii_lowercase();
        let answer_lower = answer_span.to_ascii_lowercase();
        if dialogue_answer_looks_like_question(&answer_span)
            || (answer.text.contains('?')
                && !answer_lower.is_empty()
                && !prefix_lower.contains(&answer_lower))
        {
            answer_span = canonicalize_dialogue_answer_surface_span(&prefix);
        }
    }
    if answer_span.is_empty() {
        return rows;
    }
    let question_patterns =
        dialogue_answer_surface_patterns(&question.text, answer.speaker.as_deref());
    if question_patterns.is_empty() {
        return rows;
    }
    let base_confidence = if answer_span.split_whitespace().count() <= 12 {
        0.88
    } else {
        0.8
    };
    rows.extend(question_patterns.into_iter().enumerate().map(
        |(pattern_index, question_pattern)| AnswerSurfaceRow {
            question_pattern,
            answer_span: answer_span.clone(),
            confidence: if pattern_index == 0 {
                base_confidence
            } else {
                (base_confidence + 0.04).min(0.94)
            },
        },
    ));
    rows
}

pub(crate) fn generate_cross_chunk_dialogue_answer_surface_rows(
    turns: &[Turn],
    index: usize,
) -> Vec<AnswerSurfaceRow> {
    if index == 0 {
        return Vec::new();
    }

    let previous_turns = parse_embedded_dialogue_turns(&turns[index - 1].text);
    let current_turns = parse_embedded_dialogue_turns(&turns[index].text);
    let (Some(question), Some(answer)) = (previous_turns.last(), current_turns.first()) else {
        return Vec::new();
    };
    if question.speaker == answer.speaker {
        return Vec::new();
    }

    let bridge_turns = vec![question.clone(), answer.clone()];
    generate_dialogue_answer_surface_rows(&bridge_turns, 1)
}

fn extract_dialogue_name_answer_surface_rows(
    question: &Turn,
    answer: &Turn,
) -> Vec<AnswerSurfaceRow> {
    let lower_question = question.text.to_ascii_lowercase();
    if !(lower_question.contains(" name") || lower_question.contains(" names")) {
        return Vec::new();
    }
    let Some(answer_span) = extract_dialogue_name_surface_value(&answer.text) else {
        return Vec::new();
    };
    let mut patterns = dialogue_answer_surface_patterns(&question.text, answer.speaker.as_deref());
    if let Some(base_pattern) = patterns.first().cloned() {
        if let Some(scoped_pattern) =
            scoped_question_pattern(&base_pattern, answer.speaker.as_deref())
        {
            if !patterns.iter().any(|pattern| pattern == &scoped_pattern) {
                patterns.push(scoped_pattern);
            }
        }
    }
    if patterns.is_empty() {
        return Vec::new();
    }
    patterns
        .into_iter()
        .enumerate()
        .map(|(pattern_index, question_pattern)| AnswerSurfaceRow {
            question_pattern,
            answer_span: answer_span.clone(),
            confidence: if pattern_index == 0 { 0.92 } else { 0.95 },
        })
        .collect()
}

fn extract_dialogue_name_surface_value(answer: &str) -> Option<String> {
    let prefix = answer
        .split(['!', '?', '.'])
        .next()
        .unwrap_or(answer)
        .trim()
        .trim_matches(|c: char| matches!(c, '"' | '\'' | ',' | ';' | ':' | '-'));
    if prefix.is_empty() || prefix.split_whitespace().count() > 6 {
        return None;
    }
    let tokens = prefix.split_whitespace().collect::<Vec<_>>();
    let capitalized = tokens
        .iter()
        .filter(|token| {
            let clean =
                token.trim_matches(|c: char| !c.is_ascii_alphabetic() && c != '\'' && c != '-');
            clean
                .chars()
                .next()
                .map(|c| c.is_ascii_uppercase())
                .unwrap_or(false)
        })
        .count();
    let valid = tokens.iter().all(|token| {
        let clean = token.trim_matches(|c: char| !c.is_ascii_alphabetic() && c != '\'' && c != '-');
        if clean.is_empty() {
            return false;
        }
        let lower = clean.to_ascii_lowercase();
        matches!(lower.as_str(), "and" | "the" | "of" | "a" | "an" | "&")
            || clean
                .chars()
                .next()
                .map(|c| c.is_ascii_uppercase())
                .unwrap_or(false)
    });
    (capitalized >= 1 && valid).then(|| normalize_answer_surface_span(prefix))
}

fn canonicalize_dialogue_answer_surface_span(answer_span: &str) -> String {
    const FOLLOWUP_MARKERS: &[&str] = &[
        ". what ",
        ". what's",
        ". why ",
        ". how ",
        ". who's",
        ". who ",
        ". where's",
        ". where ",
        ". when's",
        ". when ",
        ". which ",
        ". do ",
        ". does ",
        ". did ",
        ". have ",
        ". has ",
        ". can ",
        ". could ",
        ". would ",
        ". is ",
        ". are ",
        ". was ",
        ". were ",
        "? what ",
        "? what's",
        "? why ",
        "? how ",
        "? who ",
        "? where ",
        "? when ",
        "? which ",
        "? do ",
        "? does ",
        "? did ",
        "? have ",
        "? has ",
        "? can ",
        "? could ",
        "? would ",
        "? is ",
        "? are ",
        "? was ",
        "? were ",
    ];
    let lower = answer_span.to_ascii_lowercase();
    let cutoff = FOLLOWUP_MARKERS
        .iter()
        .filter_map(|marker| lower.find(marker))
        .min();
    let trimmed = cutoff
        .map(|idx| &answer_span[..idx])
        .unwrap_or(answer_span)
        .trim_end_matches(['.', '!', '?', ';', ',', ':']);
    normalize_answer_surface_span(trimmed)
}

fn extract_dialogue_statement_prefix(answer: &str) -> Option<String> {
    const FOLLOWUP_MARKERS: &[&str] = &[
        ". What ",
        ". What's",
        ". Why ",
        ". How ",
        ". Who ",
        ". Who's",
        ". Where ",
        ". Where's",
        ". When ",
        ". When's",
        ". Which ",
        ". Do ",
        ". Does ",
        ". Did ",
        ". Have ",
        ". Has ",
        ". Can ",
        ". Could ",
        ". Would ",
        ". Is ",
        ". Are ",
        ". Was ",
        ". Were ",
        "? ",
    ];
    let cutoff = FOLLOWUP_MARKERS
        .iter()
        .filter_map(|marker| answer.find(marker))
        .min()
        .or_else(|| answer.find('?'));
    let trimmed = cutoff
        .map(|idx| &answer[..idx])
        .unwrap_or(answer)
        .trim_end_matches(['.', '!', '?', ';', ',', ':']);
    let normalized = normalize_answer_surface_span(trimmed);
    (!normalized.is_empty() && !dialogue_answer_looks_like_question(&normalized))
        .then_some(normalized)
}

fn dialogue_answer_looks_like_question(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }
    let lower = trimmed.to_ascii_lowercase();
    trimmed.ends_with('?')
        || [
            "what ", "why ", "how ", "who ", "where ", "when ", "which ", "do ", "does ", "did ",
            "have ", "has ", "can ", "could ", "would ", "is ", "are ", "was ", "were ",
        ]
        .iter()
        .any(|prefix| lower.starts_with(prefix))
}

fn dialogue_answer_surface_patterns(question: &str, answer_speaker: Option<&str>) -> Vec<String> {
    let Some(base_pattern) = mine_dialogue_question_pattern(question) else {
        return Vec::new();
    };
    let mut patterns = vec![base_pattern.clone()];
    if let Some(scoped_pattern) =
        speaker_scoped_question_pattern(&base_pattern, question, answer_speaker)
    {
        if scoped_pattern != base_pattern {
            patterns.push(scoped_pattern);
        }
    }
    patterns
}

fn speaker_scoped_question_pattern(
    base_pattern: &str,
    question: &str,
    answer_speaker: Option<&str>,
) -> Option<String> {
    let speaker_terms = speaker_scope_terms(answer_speaker?);
    if speaker_terms.is_empty() || question_mentions_other_named_subject(question, &speaker_terms) {
        return None;
    }
    scoped_question_pattern(base_pattern, answer_speaker)
}

fn speaker_scope_terms(speaker: &str) -> Vec<String> {
    let lower = speaker.trim().to_ascii_lowercase();
    if matches!(
        lower.as_str(),
        "user" | "assistant" | "human" | "ai" | "system"
    ) {
        return Vec::new();
    }
    let mut terms = speaker
        .split(|c: char| !c.is_ascii_alphabetic() && c != '-')
        .filter_map(|term| {
            let clean = term.trim().to_ascii_lowercase();
            (clean.len() >= 3).then_some(clean)
        })
        .collect::<Vec<_>>();
    terms.sort();
    terms.dedup();
    terms
}

pub(in super::super) fn scoped_question_pattern(
    question_pattern: &str,
    speaker: Option<&str>,
) -> Option<String> {
    let mut terms = question_pattern
        .split_whitespace()
        .map(|term| term.to_string())
        .collect::<Vec<_>>();
    let speaker_terms = speaker_scope_terms(speaker?);
    if speaker_terms.is_empty() {
        return None;
    }
    terms.extend(speaker_terms);
    terms.sort();
    terms.dedup();
    Some(terms.join(" "))
}

fn question_mentions_other_named_subject(question: &str, speaker_terms: &[String]) -> bool {
    let lower = question.to_ascii_lowercase();
    if lower.starts_with("you ")
        || lower.starts_with("your ")
        || lower.contains(" you ")
        || lower.contains(" your ")
    {
        return false;
    }

    let named_subjects = question
        .split(|c: char| !c.is_ascii_alphabetic() && c != '-')
        .filter_map(|token| {
            let trimmed = token.trim();
            let first = trimmed.chars().next()?;
            if trimmed.len() < 3 || !first.is_ascii_uppercase() {
                return None;
            }
            let lower = trimmed.to_ascii_lowercase();
            if matches!(
                lower.as_str(),
                "what"
                    | "who"
                    | "when"
                    | "where"
                    | "which"
                    | "why"
                    | "how"
                    | "can"
                    | "did"
                    | "does"
                    | "is"
                    | "are"
                    | "was"
                    | "were"
                    | "will"
                    | "would"
                    | "could"
                    | "should"
                    | "ah"
                    | "wow"
                    | "hey"
                    | "thanks"
            ) {
                return None;
            }
            Some(lower)
        })
        .collect::<Vec<_>>();

    !named_subjects.is_empty()
        && !named_subjects
            .iter()
            .any(|subject| speaker_terms.iter().any(|speaker| speaker == subject))
}
