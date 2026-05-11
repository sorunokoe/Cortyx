use super::*;

pub(in super::super) fn compile_regex(pattern: &str) -> Regex {
    match Regex::new(pattern) {
        Ok(regex) => regex,
        Err(err) => {
            tracing::error!("invalid miner regex {pattern:?}: {err}");
            match Regex::new(r"$^") {
                Ok(fallback) => fallback,
                Err(_) => unreachable!("fallback regex must compile"),
            }
        },
    }
}

pub(crate) fn append_answer_surface_section(
    content: &mut String,
    text: &str,
    extra_rows: &[AnswerSurfaceRow],
    note: &str,
) {
    let mut rows = generate_answer_surface_rows(text);
    for row in generate_embedded_dialogue_answer_surface_rows(text) {
        push_answer_surface_row(
            &mut rows,
            &row.question_pattern,
            Some(row.answer_span),
            row.confidence,
        );
    }
    for row in extra_rows {
        push_answer_surface_row(
            &mut rows,
            &row.question_pattern,
            Some(row.answer_span.clone()),
            row.confidence,
        );
    }
    if rows.is_empty() {
        return;
    }
    content.push_str("\n## answer_surface\n");
    content.push_str(&format!("<!-- {note} -->\n"));
    content.push_str("<!-- SECTION: answer_surface -->\n");
    content.push_str("| question_pattern | answer_span | confidence |\n");
    content.push_str("| --- | --- | --- |\n");
    for row in rows {
        content.push_str(&format!(
            "| {} | {} | {:.2} |\n",
            sanitize_answer_surface_cell(&row.question_pattern),
            sanitize_answer_surface_cell(&row.answer_span),
            row.confidence
        ));
    }
    content.push_str("<!-- /SECTION -->\n");
}

fn sanitize_answer_surface_cell(value: &str) -> String {
    value.replace('|', "/")
}

pub(in super::super) fn normalize_answer_surface_span(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub(in super::super) fn push_answer_surface_row(
    rows: &mut Vec<AnswerSurfaceRow>,
    question_pattern: &str,
    answer_span: Option<String>,
    confidence: f32,
) {
    let Some(answer_span) = answer_span else {
        return;
    };
    let answer_span = normalize_answer_surface_span(&answer_span);
    if answer_span.is_empty() {
        return;
    }
    if rows.iter().any(|row| {
        row.question_pattern == question_pattern
            && row.answer_span.eq_ignore_ascii_case(&answer_span)
    }) {
        return;
    }
    rows.push(AnswerSurfaceRow {
        question_pattern: question_pattern.to_string(),
        answer_span,
        confidence,
    });
}
