use super::assistant_fact_extractors::{
    parse_assistant_fact_query, render_assistant_fact_answer, AssistantFactQuery, EntityKind,
    EntityRecallQuery, ListKind, ListRecallQuery, QuoteRecallQuery, ValueKind, ValueRecallQuery,
};
use super::assistant_fact_support::{
    entity_line_bonus, extract_entity_candidate, extract_label_list_item,
    extract_objective_list_item, extract_quote_candidate, extract_structured_fact_label,
    extract_value_candidate, render_list_answer, value_line_bonus,
};
use super::*;

impl NeuronIndex {
    pub(super) fn synthetic_assistant_fact_recall_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        let query = parse_assistant_fact_query(task, task_lower)?;
        let required_terms = match &query {
            AssistantFactQuery::Entity(query) => query.required_terms.as_slice(),
            AssistantFactQuery::Value(query) => query.required_terms.as_slice(),
            AssistantFactQuery::List(query) => query.required_terms.as_slice(),
            AssistantFactQuery::Quote(query) => query.required_terms.as_slice(),
        };
        let candidates = assistant_fact_candidates(self, task, query.focus_terms(), required_terms);
        let answer = match &query {
            AssistantFactQuery::Entity(query) => resolve_entity_fact(self, &candidates, query)?,
            AssistantFactQuery::Value(query) => resolve_value_fact(self, &candidates, query)?,
            AssistantFactQuery::List(query) => resolve_list_fact(self, &candidates, query)?,
            AssistantFactQuery::Quote(query) => resolve_quote_fact(self, &candidates, query)?,
        };
        self.write_synthetic_answer(
            "assistant-fact-recall",
            task,
            &answer.answer,
            &answer.evidence,
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct FactAnswer {
    answer: String,
    score: usize,
    evidence: Vec<String>,
}

fn assistant_fact_candidates(
    idx: &NeuronIndex,
    task: &str,
    focus_terms: &[String],
    required_terms: &[String],
) -> Vec<(String, usize)> {
    let focus_refs: Vec<&str> = focus_terms.iter().map(String::as_str).collect();
    let mut candidate_scores: HashMap<String, usize> = HashMap::new();
    for (rank, session_id) in idx
        .candidate_session_ids(task, &focus_refs, 8)
        .into_iter()
        .enumerate()
    {
        *candidate_scores.entry(session_id).or_insert(0) += 70usize.saturating_sub(rank * 10);
    }
    for (session_id, score) in idx.candidate_session_ids_by_line_overlap(focus_terms, 10) {
        *candidate_scores.entry(session_id).or_insert(0) += score;
    }
    let min_required_matches = assistant_fact_required_min(required_terms);
    if min_required_matches > 0 {
        let mut seen = HashSet::new();
        for entry in idx.retrieval.entries.iter().filter(|entry| {
            matches!(entry.kind, NeuronKind::Verbatim) && !entry.session_id.is_empty()
        }) {
            if !seen.insert(entry.session_id.clone()) {
                continue;
            }
            let best_required_matches = assistant_fact_lines(idx, &entry.session_id)
                .into_iter()
                .map(|line| {
                    assistant_fact_required_match_count(&line.to_ascii_lowercase(), required_terms)
                })
                .max()
                .unwrap_or(0);
            if best_required_matches >= min_required_matches {
                *candidate_scores
                    .entry(entry.session_id.clone())
                    .or_insert(0) += best_required_matches * 20;
            }
        }
    }
    let mut candidates = candidate_scores.into_iter().collect::<Vec<_>>();
    candidates.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    candidates
}

fn resolve_entity_fact(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &EntityRecallQuery,
) -> Option<FactAnswer> {
    let sessions = assistant_fact_session_pool(idx, candidates);
    let focus_refs: Vec<&str> = query.focus_terms.iter().map(String::as_str).collect();
    let mut best = None;
    for (session_id, session_rank) in &sessions {
        let lines = assistant_fact_lines(idx, session_id);
        for (line_idx, line) in lines.iter().enumerate() {
            let context = assistant_fact_context(&lines, line_idx);
            let context_lower = context.to_ascii_lowercase();
            let score_text =
                assistant_fact_entity_score_text(query, &lines, line_idx, line, &context_lower);
            let required_matches =
                assistant_fact_required_match_count(&score_text, &query.required_terms);
            if !matches!(
                query.kind,
                EntityKind::Wearing | EntityKind::ImplementedAlgorithm
            ) && required_matches < assistant_fact_required_min(&query.required_terms)
            {
                continue;
            }
            let overlap = term_overlap_count(&score_text, &focus_refs);
            if overlap == 0 {
                continue;
            }
            let lower = line.to_ascii_lowercase();
            let (value, evidence, detail_bonus) = if let Some((value, evidence)) =
                extract_adjacent_role_person_candidate(query, &lines, line_idx)
            {
                (value, evidence, entity_line_bonus(query, line, &lower) + 20)
            } else {
                let Some(value) = extract_entity_candidate(query, line, &lower) else {
                    continue;
                };
                (
                    value,
                    vec![line.clone()],
                    entity_line_bonus(query, line, &lower),
                )
            };
            upsert_fact_answer(
                &mut best,
                FactAnswer {
                    answer: render_assistant_fact_answer(&query.render, &value),
                    score: assistant_fact_score(
                        *session_rank,
                        required_matches,
                        overlap,
                        detail_bonus,
                    ),
                    evidence,
                },
            );
        }
    }
    best
}

fn resolve_value_fact(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &ValueRecallQuery,
) -> Option<FactAnswer> {
    let sessions = assistant_fact_session_pool(idx, candidates);
    let focus_refs: Vec<&str> = query.focus_terms.iter().map(String::as_str).collect();
    let mut best = None;
    for (session_id, session_rank) in &sessions {
        if !query.topic_terms.is_empty() {
            let session_text = assistant_fact_session_text(idx, session_id);
            let topic_matches = assistant_fact_required_match_count(
                &session_text.to_ascii_lowercase(),
                &query.topic_terms,
            );
            if topic_matches < assistant_fact_topic_min(&query.topic_terms) {
                continue;
            }
        }
        let lines = assistant_fact_value_lines(idx, session_id, query.kind);
        for (line_idx, line) in lines.iter().enumerate() {
            let context = assistant_fact_context(&lines, line_idx);
            let context_lower = context.to_ascii_lowercase();
            let line_lower = line.to_ascii_lowercase();
            let score_text = if matches!(query.kind, ValueKind::Count) {
                &line_lower
            } else {
                &context_lower
            };
            let required_matches =
                assistant_fact_required_match_count(score_text, &query.required_terms);
            if required_matches < assistant_fact_required_min(&query.required_terms) {
                continue;
            }
            let overlap = term_overlap_count(score_text, &focus_refs);
            if overlap == 0 {
                continue;
            }
            let Some(value) = extract_value_candidate(query, line) else {
                continue;
            };
            upsert_fact_answer(
                &mut best,
                FactAnswer {
                    answer: render_assistant_fact_answer(&query.render, &value),
                    score: assistant_fact_score(
                        *session_rank,
                        required_matches,
                        overlap,
                        value_line_bonus(query, line),
                    ),
                    evidence: vec![line.clone()],
                },
            );
        }
    }
    best
}

fn resolve_list_fact(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &ListRecallQuery,
) -> Option<FactAnswer> {
    let sessions = assistant_fact_session_pool(idx, candidates);
    let mut best = None;
    let list_required_min = usize::from(!query.required_terms.is_empty());
    for (session_id, session_rank) in &sessions {
        let lines = assistant_fact_lines(idx, session_id);
        let mut items = Vec::new();
        let mut evidence = Vec::new();
        let mut seen = HashSet::new();
        let mut in_objectives = false;
        for (line_idx, line) in lines.iter().enumerate() {
            let context = assistant_fact_context(&lines, line_idx);
            let context_lower = context.to_ascii_lowercase();
            let required_matches =
                assistant_fact_required_match_count(&context_lower, &query.required_terms);
            if query.kind == ListKind::Objectives {
                let body = normalize_session_answer_line_body(line);
                if body.eq_ignore_ascii_case("Objectives:") {
                    in_objectives = true;
                    evidence.push(line.clone());
                    continue;
                }
                if in_objectives {
                    if let Some(item) = extract_objective_list_item(line) {
                        let key = item.to_ascii_lowercase();
                        if seen.insert(key) {
                            items.push(item);
                            evidence.push(line.clone());
                        }
                        continue;
                    }
                    if !items.is_empty() && body.ends_with(':') {
                        break;
                    }
                }
            } else {
                if required_matches < list_required_min {
                    continue;
                }
                let Some(item) = extract_label_list_item(line) else {
                    continue;
                };
                let item_lower = item.to_ascii_lowercase();
                if query
                    .exclude_terms
                    .iter()
                    .any(|term| item_lower.contains(&term.to_ascii_lowercase()))
                {
                    continue;
                }
                if seen.insert(item_lower) {
                    items.push(item);
                    evidence.push(line.clone());
                }
            }
        }
        if items.is_empty() {
            continue;
        }
        if let Some(limit) = query.expected_count {
            items.truncate(limit);
        }
        upsert_fact_answer(
            &mut best,
            FactAnswer {
                answer: render_list_answer(&query.render, &items),
                score: assistant_fact_score(*session_rank, list_required_min, items.len(), 0),
                evidence,
            },
        );
    }
    best
}

fn resolve_quote_fact(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &QuoteRecallQuery,
) -> Option<FactAnswer> {
    let sessions = assistant_fact_session_pool(idx, candidates);
    let focus_refs: Vec<&str> = query.focus_terms.iter().map(String::as_str).collect();
    let mut best = None;
    for (session_id, session_rank) in &sessions {
        let lines = assistant_fact_lines(idx, session_id);
        for (line_idx, line) in lines.iter().enumerate() {
            let context = assistant_fact_context(&lines, line_idx);
            let context_lower = context.to_ascii_lowercase();
            let required_matches =
                assistant_fact_required_match_count(&context_lower, &query.required_terms);
            if required_matches < assistant_fact_required_min(&query.required_terms) {
                continue;
            }
            let overlap = term_overlap_count(&context_lower, &focus_refs);
            if overlap == 0 {
                continue;
            }
            let Some(value) = extract_quote_candidate(line) else {
                continue;
            };
            upsert_fact_answer(
                &mut best,
                FactAnswer {
                    answer: render_assistant_fact_answer(&query.render, &value),
                    score: assistant_fact_score(
                        *session_rank,
                        required_matches,
                        overlap,
                        usize::from(line.contains('"')) * 6,
                    ),
                    evidence: vec![line.clone()],
                },
            );
        }
    }
    best
}

fn assistant_fact_session_pool(
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
    for entry in
        idx.retrieval.entries.iter().filter(|entry| {
            matches!(entry.kind, NeuronKind::Verbatim) && !entry.session_id.is_empty()
        })
    {
        if seen.insert(entry.session_id.clone()) {
            sessions.push((entry.session_id.clone(), 1));
        }
    }
    sessions
}

fn assistant_fact_lines(idx: &NeuronIndex, session_id: &str) -> Vec<String> {
    idx.find_session_assistant_lines(session_id, 256, |line, _| !line.trim().is_empty())
}

fn assistant_fact_value_lines(idx: &NeuronIndex, session_id: &str, kind: ValueKind) -> Vec<String> {
    if matches!(kind, ValueKind::Handle | ValueKind::Year) {
        return idx.find_session_lines(session_id, false, 256, |line, _| !line.trim().is_empty());
    }
    assistant_fact_lines(idx, session_id)
}

fn assistant_fact_session_text(idx: &NeuronIndex, session_id: &str) -> String {
    idx.find_session_lines(session_id, false, 512, |line, _| !line.trim().is_empty())
        .join(" ")
}

fn assistant_fact_context(lines: &[String], line_idx: usize) -> String {
    let start = line_idx.saturating_sub(2);
    let end = usize::min(lines.len(), line_idx + 2);
    lines[start..end].join(" ")
}

fn assistant_fact_entity_score_text(
    query: &EntityRecallQuery,
    lines: &[String],
    line_idx: usize,
    line: &str,
    context_lower: &str,
) -> String {
    if query.kind == EntityKind::NamedThing {
        if extract_structured_fact_label(line).is_some() {
            return assistant_fact_structured_item_context(lines, line_idx).to_ascii_lowercase();
        }
        return line.to_ascii_lowercase();
    }
    context_lower.to_string()
}

fn assistant_fact_structured_item_context(lines: &[String], line_idx: usize) -> String {
    let mut item_lines = vec![lines[line_idx].clone()];
    for next in lines.iter().skip(line_idx + 1) {
        if extract_numbered_list_item(next).is_some() {
            break;
        }
        if !next.trim().is_empty() {
            item_lines.push(next.clone());
        }
        if item_lines.len() >= 3 {
            break;
        }
    }
    item_lines.join(" ")
}

fn assistant_fact_required_match_count(text_lower: &str, required_terms: &[String]) -> usize {
    required_terms
        .iter()
        .filter(|term| text_lower.contains(term.as_str()))
        .count()
}

fn assistant_fact_required_min(required_terms: &[String]) -> usize {
    match required_terms.len() {
        0 => 0,
        1..=3 => 1,
        _ => 2,
    }
}

fn assistant_fact_topic_min(topic_terms: &[String]) -> usize {
    match topic_terms.len() {
        0 => 0,
        1..=2 => 1,
        _ => 2,
    }
}

fn extract_adjacent_role_person_candidate(
    query: &EntityRecallQuery,
    lines: &[String],
    line_idx: usize,
) -> Option<(String, Vec<String>)> {
    if query.kind != EntityKind::PersonByRole {
        return None;
    }
    let line = lines.get(line_idx)?;
    let line_lower = line.to_ascii_lowercase();
    if assistant_fact_required_match_count(&line_lower, &query.required_terms) == 0 {
        return None;
    }
    for neighbor_idx in [line_idx.checked_sub(1), Some(line_idx + 1)] {
        let Some(neighbor_idx) = neighbor_idx else {
            continue;
        };
        let Some(neighbor) = lines.get(neighbor_idx) else {
            continue;
        };
        let neighbor_body = normalize_session_answer_line_body(neighbor);
        let Some(value) = extract_title_like_phrases(&neighbor_body)
            .into_iter()
            .find(|phrase| phrase.split_whitespace().count() <= 4)
        else {
            continue;
        };
        if value
            .chars()
            .next()
            .map(|ch| ch.is_ascii_uppercase())
            .unwrap_or(false)
        {
            return Some((value, vec![line.clone(), neighbor.clone()]));
        }
    }
    None
}

fn assistant_fact_score(
    session_signal: usize,
    required_matches: usize,
    overlap: usize,
    detail_bonus: usize,
) -> usize {
    required_matches * 1000 + overlap * 100 + detail_bonus * 10 + usize::min(session_signal, 200)
}

fn upsert_fact_answer(slot: &mut Option<FactAnswer>, candidate: FactAnswer) {
    let should_replace = slot
        .as_ref()
        .map(|best| candidate.score > best.score)
        .unwrap_or(true);
    if should_replace {
        *slot = Some(candidate);
    }
}
