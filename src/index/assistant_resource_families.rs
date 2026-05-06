use super::assistant_resource_extractors::{
    extract_duration_recall_answer, extract_example_entity_from_line, extract_specific_list_answer,
    extract_video_recall_answer, extract_website_recall_answer, looks_like_website_label,
    parse_assistant_resource_query, website_query_matches_line, AssistantResourceQuery,
    DurationRecallQuery, ExampleEntityQuery, SpecificListQuery, VideoRecallQuery,
    WebsiteRecallQuery,
};
use super::conversation_scan_support::session_score;
use super::*;

impl NeuronIndex {
    pub(super) fn synthetic_assistant_resource_recall_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        let query = parse_assistant_resource_query(task, task_lower)?;
        let candidates = assistant_resource_candidates(self, task, query.focus_terms());
        let answer = match &query {
            AssistantResourceQuery::Video(query) => resolve_video_recall(self, &candidates, query)?,
            AssistantResourceQuery::Website(query) => {
                resolve_website_recall(self, &candidates, query)?
            },
            AssistantResourceQuery::ExampleEntity(query) => {
                resolve_example_entity(self, &candidates, query)?
            },
            AssistantResourceQuery::SpecificList(query) => {
                resolve_specific_list(self, &candidates, query)?
            },
            AssistantResourceQuery::Duration(query) => {
                resolve_duration_recall(self, &candidates, query)?
            },
        };
        self.write_synthetic_answer(
            "assistant-resource-recall",
            task,
            &answer.answer,
            &answer.evidence,
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ResourceAnswer {
    answer: String,
    score: usize,
    evidence: Vec<String>,
}

fn assistant_resource_candidates(
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

fn resolve_video_recall(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &VideoRecallQuery,
) -> Option<ResourceAnswer> {
    let sessions = assistant_resource_session_pool(idx, candidates);
    let focus_refs: Vec<&str> = query.focus_terms.iter().map(String::as_str).collect();
    let mut best = None;
    for (session_id, session_rank) in &sessions {
        for line in assistant_lines(idx, session_id) {
            let lower = line.to_ascii_lowercase();
            let overlap = term_overlap_count(&lower, &focus_refs);
            if overlap == 0 {
                continue;
            }
            let Some(answer) =
                extract_video_recall_answer(&line, &lower, query.source_hint.as_deref())
            else {
                continue;
            };
            upsert_resource_answer(
                &mut best,
                ResourceAnswer {
                    answer,
                    score: session_score(*session_rank, 18 + overlap * 5),
                    evidence: vec![line],
                },
            );
        }
    }
    best
}

fn resolve_website_recall(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &WebsiteRecallQuery,
) -> Option<ResourceAnswer> {
    let sessions = assistant_resource_session_pool(idx, candidates);
    let focus_refs: Vec<&str> = query.focus_terms.iter().map(String::as_str).collect();
    let mut best_domain = None;
    let mut best_fallback = None;
    for (session_id, session_rank) in &sessions {
        for line in assistant_lines(idx, session_id) {
            let lower = line.to_ascii_lowercase();
            let overlap = term_overlap_count(&lower, &focus_refs);
            if overlap < 2 {
                continue;
            }
            if !website_query_matches_line(query, &lower) {
                continue;
            }
            let Some(answer) = extract_website_recall_answer(&line, &lower) else {
                continue;
            };
            let candidate = ResourceAnswer {
                score: session_score(
                    *session_rank,
                    16 + overlap * 4 + usize::from(looks_like_website_label(&answer)) * 12,
                ),
                answer,
                evidence: vec![line],
            };
            if looks_like_website_label(&candidate.answer) {
                upsert_resource_answer(&mut best_domain, candidate);
            } else {
                upsert_resource_answer(&mut best_fallback, candidate);
            }
        }
    }
    best_domain.or(best_fallback)
}

fn resolve_example_entity(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &ExampleEntityQuery,
) -> Option<ResourceAnswer> {
    let sessions = assistant_resource_session_pool(idx, candidates);
    let focus_refs: Vec<&str> = query.focus_terms.iter().map(String::as_str).collect();
    let mut best = None;
    for (session_id, session_rank) in &sessions {
        for line in assistant_lines(idx, session_id) {
            let lower = line.to_ascii_lowercase();
            let overlap = term_overlap_count(&lower, &focus_refs);
            if overlap < 2 {
                continue;
            }
            let Some(answer) = extract_example_entity_from_line(&line, &lower) else {
                continue;
            };
            upsert_resource_answer(
                &mut best,
                ResourceAnswer {
                    answer,
                    score: session_score(*session_rank, 14 + overlap * 4),
                    evidence: vec![line],
                },
            );
        }
    }
    best
}

fn resolve_specific_list(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &SpecificListQuery,
) -> Option<ResourceAnswer> {
    let sessions = assistant_resource_session_pool(idx, candidates);
    let focus_refs: Vec<&str> = query.focus_terms.iter().map(String::as_str).collect();
    let mut best = None;
    for (session_id, session_rank) in &sessions {
        for line in assistant_lines(idx, session_id) {
            let lower = line.to_ascii_lowercase();
            let overlap = term_overlap_count(&lower, &focus_refs);
            if overlap < 2 {
                continue;
            }
            let Some(answer) = extract_specific_list_answer(&line, &lower, query) else {
                continue;
            };
            upsert_resource_answer(
                &mut best,
                ResourceAnswer {
                    answer,
                    score: session_score(*session_rank, 16 + overlap * 4),
                    evidence: vec![line],
                },
            );
        }
    }
    best
}

fn resolve_duration_recall(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &DurationRecallQuery,
) -> Option<ResourceAnswer> {
    let sessions = assistant_resource_session_pool(idx, candidates);
    let focus_refs: Vec<&str> = query.focus_terms.iter().map(String::as_str).collect();
    let mut best = None;
    for (session_id, session_rank) in &sessions {
        for line in assistant_lines(idx, session_id) {
            let lower = line.to_ascii_lowercase();
            if !query.required_terms.iter().all(|term| lower.contains(term)) {
                continue;
            }
            let overlap = term_overlap_count(&lower, &focus_refs);
            if overlap < 2 {
                continue;
            }
            let Some(answer) = extract_duration_recall_answer(&line, &lower) else {
                continue;
            };
            upsert_resource_answer(
                &mut best,
                ResourceAnswer {
                    answer,
                    score: session_score(*session_rank, 14 + overlap * 4),
                    evidence: vec![line],
                },
            );
        }
    }
    best
}

fn assistant_lines(idx: &NeuronIndex, session_id: &str) -> Vec<String> {
    idx.find_session_assistant_lines(session_id, 256, |line, _| !line.trim().is_empty())
}

fn assistant_resource_session_pool(
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

fn upsert_resource_answer(slot: &mut Option<ResourceAnswer>, candidate: ResourceAnswer) {
    let should_replace = slot
        .as_ref()
        .map(|best| candidate.score > best.score)
        .unwrap_or(true);
    if should_replace {
        *slot = Some(candidate);
    }
}
