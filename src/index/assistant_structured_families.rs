use super::assistant_structured_extractors::{
    described_entity_line_bonus, extract_described_entity_from_line,
    extract_examples_list_from_line, parse_assistant_structured_query,
    render_described_entity_answer, AssistantStructuredQuery, DescribedEntityQuery,
    ExampleListQuery, StructuredRecallSource,
};
use super::conversation_scan_support::session_score;
use super::*;

impl NeuronIndex {
    pub(super) fn synthetic_assistant_structured_recall_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        let query = parse_assistant_structured_query(task, task_lower)?;
        let candidates = assistant_structured_candidates(self, task, query.focus_terms());
        let answer = match &query {
            AssistantStructuredQuery::DescribedEntity(query) => {
                resolve_described_entity(self, &candidates, query)?
            },
            AssistantStructuredQuery::ExampleList(query) => {
                resolve_example_list(self, &candidates, query)?
            },
        };
        self.write_synthetic_answer(
            "assistant-structured-recall",
            task,
            &answer.answer,
            &answer.evidence,
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct StructuredAnswer {
    answer: String,
    score: usize,
    evidence: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct StructuredSourceLine {
    line: String,
    from_assistant: bool,
    order: usize,
}

fn assistant_structured_candidates(
    idx: &NeuronIndex,
    task: &str,
    focus_terms: &[String],
) -> Vec<(String, usize)> {
    let required_terms: Vec<&str> = focus_terms.iter().map(String::as_str).collect();
    let mut candidate_scores: HashMap<String, usize> = HashMap::new();
    for (rank, session_id) in idx
        .candidate_session_ids(task, &required_terms, 6)
        .into_iter()
        .enumerate()
    {
        *candidate_scores.entry(session_id).or_insert(0) += 60usize.saturating_sub(rank * 10);
    }
    for (session_id, score) in idx.candidate_session_ids_by_line_overlap(focus_terms, 8) {
        *candidate_scores.entry(session_id).or_insert(0) += score;
    }
    let mut candidates = candidate_scores.into_iter().collect::<Vec<_>>();
    candidates.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    candidates
}

fn resolve_example_list(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &ExampleListQuery,
) -> Option<StructuredAnswer> {
    let sessions = assistant_structured_session_pool(idx, candidates);
    let focus_refs: Vec<&str> = query.focus_terms.iter().map(String::as_str).collect();
    let mut best = None;
    for (session_id, session_rank) in &sessions {
        for line in idx.find_session_assistant_lines(session_id, 256, |line, lower| {
            extract_examples_list_from_line(line, lower, query).is_some()
        }) {
            let lower = line.to_ascii_lowercase();
            let overlap = term_overlap_count(&lower, &focus_refs);
            let Some(answer) = extract_examples_list_from_line(&line, &lower, query) else {
                continue;
            };
            upsert_structured_answer(
                &mut best,
                StructuredAnswer {
                    answer,
                    score: session_score(*session_rank, 18 + overlap * 5),
                    evidence: vec![line],
                },
            );
        }
    }
    best
}

fn resolve_described_entity(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &DescribedEntityQuery,
) -> Option<StructuredAnswer> {
    let sessions = assistant_structured_session_pool(idx, candidates);
    let focus_refs: Vec<&str> = query.focus_terms.iter().map(String::as_str).collect();
    let mut best = None;
    for (session_id, session_rank) in &sessions {
        let source_lines = structured_source_lines(idx, session_id, query.source);
        let total_lines = source_lines.len();
        for source_line in source_lines {
            let line = source_line.line;
            let from_assistant = source_line.from_assistant;
            let lower = line.to_ascii_lowercase();
            if !query.required_terms.is_empty()
                && !query.required_terms.iter().all(|term| lower.contains(term))
            {
                continue;
            }
            let overlap = term_overlap_count(&lower, &focus_refs);
            let Some(entity) = extract_decided_entity_candidate(&line, &lower)
                .or_else(|| extract_described_entity_from_line(&line, &lower))
            else {
                continue;
            };
            if !is_valid_described_entity_candidate(&entity) {
                continue;
            }
            let decision_bonus = described_entity_decision_bonus(
                query,
                &line,
                &lower,
                &entity,
                from_assistant,
                source_line.order,
                total_lines,
            );
            if query.prefer_latest {
                if overlap == 0 && decision_bonus == 0 {
                    continue;
                }
            } else if overlap < 2 {
                continue;
            }
            upsert_structured_answer(
                &mut best,
                StructuredAnswer {
                    answer: render_described_entity_answer(query, &entity),
                    score: session_score(
                        *session_rank,
                        14 + overlap * 5
                            + described_entity_line_bonus(&line, &lower)
                            + usize::from(from_assistant) * 2
                            + decision_bonus,
                    ),
                    evidence: vec![line],
                },
            );
        }
    }
    best
}

fn structured_source_lines(
    idx: &NeuronIndex,
    session_id: &str,
    source: StructuredRecallSource,
) -> Vec<StructuredSourceLine> {
    match source {
        StructuredRecallSource::Assistant => idx
            .find_session_assistant_lines(session_id, 256, |line, _| !line.trim().is_empty())
            .into_iter()
            .enumerate()
            .map(|(order, line)| StructuredSourceLine {
                line,
                from_assistant: true,
                order,
            })
            .collect(),
        StructuredRecallSource::AssistantOrUser => {
            let mut entries: Vec<_> = idx
                .entries
                .iter()
                .filter(|entry| {
                    matches!(entry.kind, NeuronKind::Verbatim)
                        && entry.session_id == session_id
                        && !is_session_summary_path(&entry.neuron_path)
                })
                .collect();
            entries.sort_by(|a, b| a.neuron_path.cmp(&b.neuron_path));

            let mut lines = Vec::new();
            let mut order = 0;
            for entry in entries {
                let Ok(content) = std::fs::read_to_string(&entry.neuron_path) else {
                    continue;
                };
                let mut active_role = None;
                for raw_line in strip_query_surface_section(&content).lines() {
                    let line = raw_line.trim();
                    if line.is_empty() {
                        continue;
                    }
                    let lower = line.to_ascii_lowercase();
                    if let Some(body) = lower
                        .starts_with("assistant:")
                        .then(|| line["Assistant:".len()..].trim())
                    {
                        active_role = Some(true);
                        if !body.is_empty() {
                            lines.push(StructuredSourceLine {
                                line: body.to_string(),
                                from_assistant: true,
                                order,
                            });
                            order += 1;
                        }
                        continue;
                    }
                    if let Some(body) = lower
                        .starts_with("user:")
                        .then(|| line["User:".len()..].trim())
                    {
                        active_role = Some(false);
                        if !body.is_empty() {
                            lines.push(StructuredSourceLine {
                                line: body.to_string(),
                                from_assistant: false,
                                order,
                            });
                            order += 1;
                        }
                        continue;
                    }
                    let Some(from_assistant) = active_role else {
                        continue;
                    };
                    lines.push(StructuredSourceLine {
                        line: line.to_string(),
                        from_assistant,
                        order,
                    });
                    order += 1;
                }
            }
            lines
        },
    }
}

fn assistant_structured_session_pool(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
) -> Vec<(String, usize)> {
    let mut sessions = Vec::new();
    let mut seen = HashSet::new();
    for (session_id, score) in candidates {
        if seen.insert(session_id.clone()) {
            sessions.push((session_id.clone(), *score));
        }
    }
    for entry in idx
        .entries
        .iter()
        .filter(|entry| matches!(entry.kind, NeuronKind::Verbatim) && !entry.session_id.is_empty())
    {
        if seen.insert(entry.session_id.clone()) {
            sessions.push((entry.session_id.clone(), 1));
        }
    }
    sessions
}

fn upsert_structured_answer(slot: &mut Option<StructuredAnswer>, candidate: StructuredAnswer) {
    let should_replace = slot
        .as_ref()
        .map(|best| candidate.score > best.score)
        .unwrap_or(true);
    if should_replace {
        *slot = Some(candidate);
    }
}

fn extract_decided_entity_candidate(line: &str, lower: &str) -> Option<String> {
    compile_regex(
        r"(?i)^(?:the\s+)?([A-Z][A-Za-z0-9-]+(?:\s+[A-Z][A-Za-z0-9-]+){0,3})\s+(?:is|was|could|would|has|have)\b",
    )
    .captures(line)
    .and_then(|captures| captures.get(1))
    .map(|value| trim_decided_entity(value.as_str()))
    .or_else(|| {
        compile_regex(r"(?i)\bfor the ([A-Z][A-Za-z0-9-]+(?:\s+[A-Z][A-Za-z0-9-]+){0,3})\b")
            .captures(line)
            .and_then(|captures| captures.get(1))
            .map(|value| trim_decided_entity(value.as_str()))
    })
    .filter(|value| {
        !value.is_empty() && !lower.starts_with("i ") && is_valid_described_entity_candidate(value)
    })
}

fn described_entity_decision_bonus(
    query: &DescribedEntityQuery,
    line: &str,
    lower: &str,
    entity: &str,
    from_assistant: bool,
    order: usize,
    total_lines: usize,
) -> usize {
    if !query.prefer_latest {
        return 0;
    }
    let entity_lower = entity.to_ascii_lowercase();
    let mut bonus = 0i32;
    if !from_assistant {
        bonus += 12;
    }
    if lower.starts_with(&format!("{entity_lower} is"))
        || lower.starts_with(&format!("{entity_lower} was"))
    {
        bonus += 18;
    }
    if lower.contains(&format!("the {entity_lower}")) {
        bonus += 14;
    }
    if task_contains_any(
        lower,
        &[
            "i love",
            "i really like",
            "i like",
            "really cool",
            "great idea",
            "love the idea",
        ],
    ) {
        bonus += 24;
    }
    if extract_numbered_list_item(line).is_some() {
        bonus -= 8;
    }
    if task_contains_any(
        lower,
        &[
            "how about",
            "potential one-word names",
            "here are some potential",
            "some potential",
        ],
    ) {
        bonus -= 24;
    }
    bonus += ((order + 1) * 2) as i32;
    if total_lines > 0 && order + 1 == total_lines {
        bonus += 6;
    }
    bonus.max(0) as usize
}

fn trim_decided_entity(value: &str) -> String {
    value
        .trim()
        .trim_start_matches("the ")
        .trim()
        .trim_matches(|ch: char| matches!(ch, '"' | '\'' | '*' | '`' | ',' | ';' | ':'))
        .to_string()
}

fn is_valid_described_entity_candidate(value: &str) -> bool {
    let trimmed = value.trim().trim_end_matches(['.', '!', '?']).trim();
    if trimmed.len() <= 1 {
        return false;
    }
    !matches!(
        trimmed.to_ascii_lowercase().as_str(),
        "he" | "i"
            | "idea"
            | "it"
            | "she"
            | "that"
            | "the"
            | "their"
            | "there"
            | "these"
            | "they"
            | "this"
            | "those"
            | "we"
            | "yes"
            | "you"
    )
}
