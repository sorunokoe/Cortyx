use super::assistant_recall_extractors::{
    extract_example_title_from_line, extract_metric_value_from_line,
    extract_recommended_name_from_line, extract_section_heading_from_line,
    extract_section_item_label, parse_assistant_recall_query, render_numeric_recall_answer,
    render_ordinal_recall_answer, AssistantRecallQuery, ExampleTitleQuery, NumericValueQuery,
    OrdinalListItemQuery, RecallSource, RecommendedNameQuery, SectionItemsQuery,
};
use super::conversation_scan_support::session_score;
use super::*;

impl NeuronIndex {
    pub(super) fn synthetic_typed_assistant_recall_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        let query = parse_assistant_recall_query(task, task_lower)?;
        let candidates = assistant_recall_candidates(self, task, query.focus_terms());
        let answer = match &query {
            AssistantRecallQuery::RecommendedName(query) => {
                resolve_recommended_name(self, &candidates, query)?
            },
            AssistantRecallQuery::OrdinalListItem(query) => {
                resolve_ordinal_list_item(self, &candidates, query)?
            },
            AssistantRecallQuery::SectionItems(query) => {
                resolve_section_items(self, &candidates, query)?
            },
            AssistantRecallQuery::NumericValue(query) => {
                resolve_numeric_value(self, &candidates, query)?
            },
            AssistantRecallQuery::ExampleTitle(query) => {
                resolve_example_title(self, &candidates, query)?
            },
        };
        self.write_synthetic_answer("assistant-recall", task, &answer.answer, &answer.evidence)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RecallAnswer {
    answer: String,
    score: usize,
    evidence: Vec<String>,
}

fn assistant_recall_candidates(
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
    for (session_id, score) in idx.candidate_session_ids_by_line_overlap(focus_terms, 6) {
        *candidate_scores.entry(session_id).or_insert(0) += score;
    }
    let mut candidates = candidate_scores.into_iter().collect::<Vec<_>>();
    candidates.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    candidates
}

fn resolve_recommended_name(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &RecommendedNameQuery,
) -> Option<RecallAnswer> {
    let focus_refs: Vec<&str> = query.focus_terms.iter().map(String::as_str).collect();
    let mut best = None;
    for (session_id, session_rank) in candidates {
        for line in recall_source_lines(idx, session_id, RecallSource::Assistant) {
            let lower = line.to_ascii_lowercase();
            let Some(answer) = extract_recommended_name_from_line(&line, &lower) else {
                continue;
            };
            let overlap = term_overlap_count(&lower, &focus_refs);
            if overlap == 0 {
                continue;
            }
            let score = session_score(
                *session_rank,
                overlap * 6 + usize::from(lower.contains("romantic")) * 4,
            );
            upsert_best_answer(
                &mut best,
                RecallAnswer {
                    answer,
                    score,
                    evidence: vec![line],
                },
            );
        }
    }
    best
}

fn resolve_ordinal_list_item(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &OrdinalListItemQuery,
) -> Option<RecallAnswer> {
    let focus_refs: Vec<&str> = query.focus_terms.iter().map(String::as_str).collect();
    let mut best = None;
    for (session_id, session_rank) in candidates {
        let lines = recall_source_lines(idx, session_id, RecallSource::Assistant);
        for (line_idx, line) in lines.iter().enumerate() {
            let Some((ordinal, value)) = extract_numbered_list_item(&line) else {
                continue;
            };
            if ordinal != query.ordinal {
                continue;
            }
            let contextual_lower = lines[line_idx.saturating_sub(2)..=line_idx]
                .iter()
                .map(|entry| entry.to_ascii_lowercase())
                .collect::<Vec<_>>()
                .join(" ");
            let overlap = term_overlap_count(&contextual_lower, &focus_refs);
            let score = session_score(*session_rank, overlap * 4 + 20);
            upsert_best_answer(
                &mut best,
                RecallAnswer {
                    answer: render_ordinal_recall_answer(query, &value),
                    score,
                    evidence: vec![line.clone()],
                },
            );
        }
    }
    best
}

fn resolve_section_items(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &SectionItemsQuery,
) -> Option<RecallAnswer> {
    let focus_refs: Vec<&str> = query.focus_terms.iter().map(String::as_str).collect();
    let mut best = None;
    for (session_id, session_rank) in candidates {
        let lines = recall_source_lines(idx, session_id, RecallSource::Assistant);
        let mut active = false;
        let mut items = Vec::new();
        let mut evidence = Vec::new();
        for line in lines {
            if let Some(heading) = extract_section_heading_from_line(&line) {
                let heading_lower = heading.to_ascii_lowercase();
                if active && !heading_lower.contains(&query.section_label) && !items.is_empty() {
                    break;
                }
                active = heading_lower.contains(&query.section_label);
                if active {
                    evidence.push(line.clone());
                }
                continue;
            }
            if !active {
                continue;
            }
            let Some(label) = extract_section_item_label(&line) else {
                continue;
            };
            items.push(label);
            evidence.push(line.clone());
        }
        if items.is_empty() {
            continue;
        }
        let local_overlap = evidence
            .iter()
            .map(|line| term_overlap_count(&line.to_ascii_lowercase(), &focus_refs))
            .max()
            .unwrap_or(0);
        upsert_best_answer(
            &mut best,
            RecallAnswer {
                answer: join_recall_items(&items),
                score: session_score(*session_rank, 18 + local_overlap * 4 + items.len() * 3),
                evidence,
            },
        );
    }
    best
}

fn resolve_numeric_value(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &NumericValueQuery,
) -> Option<RecallAnswer> {
    let focus_refs: Vec<&str> = query.focus_terms.iter().map(String::as_str).collect();
    let mut best = None;
    for (session_id, session_rank) in candidates {
        for line in recall_source_lines(idx, session_id, RecallSource::Assistant) {
            let lower = line.to_ascii_lowercase();
            let overlap = term_overlap_count(&lower, &focus_refs);
            if overlap < 2 {
                continue;
            }
            let Some(value) = extract_metric_value_from_line(&line, &lower) else {
                continue;
            };
            upsert_best_answer(
                &mut best,
                RecallAnswer {
                    answer: render_numeric_recall_answer(query, &value),
                    score: session_score(*session_rank, 16 + overlap * 5),
                    evidence: vec![line],
                },
            );
        }
    }
    best
}

fn resolve_example_title(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &ExampleTitleQuery,
) -> Option<RecallAnswer> {
    let focus_refs: Vec<&str> = query.focus_terms.iter().map(String::as_str).collect();
    let mut best = None;
    for (session_id, session_rank) in candidates {
        for line in recall_source_lines(idx, session_id, RecallSource::User) {
            let lower = line.to_ascii_lowercase();
            if !lower.contains("example") {
                continue;
            }
            let Some(answer) = extract_example_title_from_line(&line, &lower) else {
                continue;
            };
            let overlap = term_overlap_count(&lower, &focus_refs);
            upsert_best_answer(
                &mut best,
                RecallAnswer {
                    answer,
                    score: session_score(
                        *session_rank,
                        14 + overlap * 4 + usize::from(lower.contains("last season")) * 4,
                    ),
                    evidence: vec![line],
                },
            );
        }
    }
    best
}

fn recall_source_lines(idx: &NeuronIndex, session_id: &str, source: RecallSource) -> Vec<String> {
    match source {
        RecallSource::Assistant => {
            idx.find_session_assistant_lines(session_id, 256, |line, _| !line.trim().is_empty())
        },
        RecallSource::User => idx
            .find_session_lines(session_id, false, 256, |_line, lower| {
                lower.starts_with("user:")
            })
            .into_iter()
            .map(|line| {
                line.strip_prefix("User:")
                    .map(str::trim)
                    .unwrap_or_else(|| line.trim())
                    .to_string()
            })
            .collect(),
    }
}

fn join_recall_items(items: &[String]) -> String {
    match items {
        [] => String::new(),
        [only] => format!("{only}."),
        [first, second] => format!("{first} and {second}."),
        _ => {
            let mut rendered = items[..items.len() - 1].join(", ");
            rendered.push_str(", and ");
            rendered.push_str(&items[items.len() - 1]);
            rendered.push('.');
            rendered
        },
    }
}

fn upsert_best_answer(slot: &mut Option<RecallAnswer>, candidate: RecallAnswer) {
    let should_replace = slot
        .as_ref()
        .map(|best| candidate.score > best.score)
        .unwrap_or(true);
    if should_replace {
        *slot = Some(candidate);
    }
}
