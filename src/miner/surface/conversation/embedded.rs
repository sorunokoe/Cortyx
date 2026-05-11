use super::*;

pub(in super::super) fn generate_embedded_dialogue_answer_surface_rows(
    text: &str,
) -> Vec<AnswerSurfaceRow> {
    let turns = parse_embedded_dialogue_turns(text);
    if turns.is_empty() {
        return Vec::new();
    }

    let mut rows = Vec::new();
    for turn in &turns {
        for row in generate_temporal_turn_answer_surface_rows(turn) {
            push_answer_surface_row(
                &mut rows,
                &row.question_pattern,
                Some(row.answer_span),
                row.confidence,
            );
        }
        for row in generate_dialogue_bridge_surface_rows(turn) {
            push_answer_surface_row(
                &mut rows,
                &row.question_pattern,
                Some(row.answer_span),
                row.confidence,
            );
        }
    }
    for index in 1..turns.len() {
        for row in generate_dialogue_answer_surface_rows(&turns, index) {
            push_answer_surface_row(
                &mut rows,
                &row.question_pattern,
                Some(row.answer_span),
                row.confidence,
            );
        }
    }
    rows
}

pub(super) fn parse_embedded_dialogue_turns(content: &str) -> Vec<Turn> {
    let mut turns = Vec::new();
    let mut current: Option<Turn> = None;
    let mut session_timestamp: Option<String> = None;

    for raw_line in content.lines() {
        let trimmed = raw_line.trim();
        if trimmed.is_empty()
            || trimmed.starts_with("<!--")
            || trimmed.starts_with("##")
            || trimmed.starts_with('#')
            || trimmed.starts_with("===")
        {
            continue;
        }

        if let Some(timestamp) = parse_embedded_session_timestamp(trimmed) {
            if let Some(turn) = current.take() {
                if !turn.text.is_empty() {
                    turns.push(turn);
                }
            }
            session_timestamp = Some(timestamp);
            continue;
        }

        if let Some((speaker, text)) = parse_embedded_dialogue_line(trimmed) {
            if let Some(turn) = current.take() {
                if !turn.text.is_empty() {
                    turns.push(turn);
                }
            }
            current = Some(Turn {
                speaker: Some(speaker.to_string()),
                text: text.to_string(),
                timestamp: session_timestamp.clone(),
            });
            continue;
        }

        if let Some(turn) = current.as_mut() {
            if !turn.text.is_empty() {
                turn.text.push(' ');
            }
            turn.text.push_str(trimmed);
        }
    }

    if let Some(turn) = current {
        if !turn.text.is_empty() {
            turns.push(turn);
        }
    }

    turns
}

pub(crate) fn parse_embedded_dialogue_line(line: &str) -> Option<(&str, &str)> {
    let (speaker, rest) = line.split_once(':')?;
    if !is_dialogue_speaker(speaker) {
        return None;
    }
    let rest = rest.trim();
    (!rest.is_empty()).then_some((speaker.trim(), rest))
}

pub(crate) fn is_dialogue_speaker(prefix: &str) -> bool {
    let trimmed = prefix.trim();
    if trimmed.is_empty() || trimmed.len() > 32 {
        return false;
    }
    let lower = trimmed.to_ascii_lowercase();
    if lower.starts_with("speaker ") {
        return false;
    }
    trimmed
        .chars()
        .all(|c| c.is_ascii_alphabetic() || c == ' ' || c == '-' || c == '\'')
}

pub(crate) fn normalize_dialogue_speaker_label(speaker: &str) -> String {
    let trimmed = speaker.trim();
    let lower = trimmed.to_ascii_lowercase();
    if matches!(
        lower.as_str(),
        "user" | "assistant" | "human" | "ai" | "system"
    ) {
        lower
    } else {
        trimmed.to_string()
    }
}

pub(crate) fn parse_embedded_session_timestamp(line: &str) -> Option<String> {
    let captures = compile_regex(
        r"(?i)^\[session\s+\d+\s+[—-]\s+(?:\d{1,2}:\d{2}\s*[ap]m\s+on\s+)?(\d{1,2})\s+([a-z]+),\s*(\d{4})\]$",
    )
    .captures(line)?;
    let day = captures.get(1)?.as_str().parse::<u32>().ok()?;
    let month = month_name_to_number(captures.get(2)?.as_str())?;
    let year = captures.get(3)?.as_str().parse::<u32>().ok()?;
    Some(format!("{year:04}-{month:02}-{day:02}T00:00:00Z"))
}
