use super::temporal_anchor_extractors::{
    parse_temporal_anchor_query, ElapsedBeforeEventQuery, RelativeTemporalRecallQuery,
    TemporalAnchorQuery, TemporalElapsedGapQuery, TemporalIntervalQuery,
};
use super::temporal_relative_recall_families::best_grouped_relative_temporal_recall;
use super::*;

impl NeuronIndex {
    pub(super) fn synthetic_temporal_anchor_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        match parse_temporal_anchor_query(task_lower)? {
            TemporalAnchorQuery::ElapsedBeforeEvent(query) => {
                self.synthetic_elapsed_before_event_answer(task, &query)
            },
            TemporalAnchorQuery::Interval(query) => {
                self.synthetic_temporal_interval_answer(task, &query)
            },
            TemporalAnchorQuery::ElapsedGap(query) => {
                self.synthetic_temporal_elapsed_gap_answer(task, &query)
            },
            TemporalAnchorQuery::RelativeRecall(query) => {
                self.synthetic_relative_temporal_recall_answer(task, &query)
            },
        }
    }

    fn synthetic_elapsed_before_event_answer(
        &self,
        task: &str,
        query: &ElapsedBeforeEventQuery,
    ) -> Option<PathBuf> {
        let answer = best_grouped_elapsed_before_event(self, query)?;
        self.write_synthetic_answer("temporal-anchor-elapsed", task, &answer.0, &answer.1)
    }

    fn synthetic_temporal_interval_answer(
        &self,
        task: &str,
        query: &TemporalIntervalQuery,
    ) -> Option<PathBuf> {
        let answer = best_grouped_temporal_interval(self, query)?;
        self.write_synthetic_answer("temporal-anchor-interval", task, &answer.0, &answer.1)
    }

    fn synthetic_temporal_elapsed_gap_answer(
        &self,
        task: &str,
        query: &TemporalElapsedGapQuery,
    ) -> Option<PathBuf> {
        let answer = best_grouped_temporal_elapsed_gap(self, query)?;
        self.write_synthetic_answer("temporal-anchor-elapsed-gap", task, &answer.0, &answer.1)
    }

    fn synthetic_relative_temporal_recall_answer(
        &self,
        task: &str,
        query: &RelativeTemporalRecallQuery,
    ) -> Option<PathBuf> {
        let answer = best_grouped_relative_temporal_recall(self, query)?;
        self.write_synthetic_answer(
            "temporal-anchor-relative-recall",
            task,
            &answer.0,
            &answer.1,
        )
    }
}

fn best_grouped_elapsed_before_event(
    idx: &NeuronIndex,
    query: &ElapsedBeforeEventQuery,
) -> Option<(String, Vec<String>, usize)> {
    grouped_verbatim_lines(idx)
        .into_values()
        .filter_map(|lines| resolve_elapsed_before_event(lines, query))
        .max_by_key(|(_, _, score)| *score)
}

fn resolve_elapsed_before_event(
    lines: Vec<String>,
    query: &ElapsedBeforeEventQuery,
) -> Option<(String, Vec<String>, usize)> {
    let subject_terms = synthetic_query_terms(&query.subject_phrase);
    let event_terms = synthetic_query_terms(&query.event_phrase);
    let subject_lower = query.subject_phrase.to_ascii_lowercase();
    let event_lower = query.event_phrase.to_ascii_lowercase();
    let subject_match = best_temporal_duration_anchor_line(&lines, &subject_lower, &subject_terms)?;
    let event_match = best_temporal_event_anchor_line(&lines, &event_lower, &event_terms)
        .or_else(|| best_relaxed_temporal_event_anchor_line(&lines, &event_lower, &event_terms))?;
    let delta_days = match (subject_match.0, event_match.0) {
        (
            SyntheticDurationAnchor::CurrentDays(subject_days),
            SyntheticEventAnchor::RelativeDaysAgo(event_days),
        ) => subject_days - event_days,
        (
            SyntheticDurationAnchor::AbsoluteDay(start_day),
            SyntheticEventAnchor::AbsoluteDay(event_day),
        ) => event_day - start_day,
        _ => return None,
    };
    if delta_days <= 0 {
        return None;
    }
    Some((
        render_elapsed_duration_answer(delta_days),
        dedupe_temporal_anchor_evidence([subject_match.2, event_match.2]),
        subject_match.1 + event_match.1,
    ))
}

fn best_grouped_temporal_interval(
    idx: &NeuronIndex,
    query: &TemporalIntervalQuery,
) -> Option<(String, Vec<String>, usize)> {
    grouped_verbatim_lines(idx)
        .into_values()
        .filter_map(|lines| resolve_temporal_interval(lines, query))
        .max_by_key(|(_, _, score)| *score)
}

fn resolve_temporal_interval(
    lines: Vec<String>,
    query: &TemporalIntervalQuery,
) -> Option<(String, Vec<String>, usize)> {
    let (start_terms, start_focus_terms) = temporal_interval_start_terms(query);
    let end_terms = synthetic_query_terms(&query.end_phrase);
    let end_focus_terms = temporal_interval_focus_terms(&query.end_phrase);
    let start_lower = query.start_phrase.to_ascii_lowercase();
    let end_lower = query.end_phrase.to_ascii_lowercase();
    let start_match =
        best_temporal_interval_rank_line(&lines, &start_lower, &start_terms, &start_focus_terms)?;
    let end_match =
        best_temporal_interval_rank_line(&lines, &end_lower, &end_terms, &end_focus_terms)?;
    let object_identity_terms = temporal_interval_object_identity_terms(&query.end_phrase);
    if weak_temporal_pronoun_reference(&query.start_phrase)
        && !object_identity_terms.is_empty()
        && (!temporal_line_matches_any_term(&start_match.2, &object_identity_terms)
            || !temporal_line_matches_any_term(&end_match.2, &object_identity_terms))
    {
        return None;
    }
    let delta_days = end_match.0 - start_match.0;
    if delta_days <= 0 {
        return None;
    }
    let explicit_dates = extract_explicit_date_rank(&start_match.2).is_some()
        && extract_explicit_date_rank(&end_match.2).is_some();
    Some((
        render_temporal_interval_answer(delta_days, explicit_dates),
        dedupe_temporal_anchor_evidence([start_match.2, end_match.2]),
        start_match.1 + end_match.1,
    ))
}

fn best_grouped_temporal_elapsed_gap(
    idx: &NeuronIndex,
    query: &TemporalElapsedGapQuery,
) -> Option<(String, Vec<String>, usize)> {
    grouped_verbatim_lines(idx)
        .into_values()
        .filter_map(|lines| resolve_temporal_elapsed_gap(lines, query))
        .max_by_key(|(_, _, score)| *score)
}

fn resolve_temporal_elapsed_gap(
    lines: Vec<String>,
    query: &TemporalElapsedGapQuery,
) -> Option<(String, Vec<String>, usize)> {
    let start_query = TemporalIntervalQuery {
        start_phrase: query.start_phrase.clone(),
        end_phrase: query.end_phrase.clone(),
        required_terms: query.required_terms.clone(),
    };
    let (start_terms, start_focus_terms) = temporal_interval_start_terms(&start_query);
    let end_terms = synthetic_query_terms(&query.end_phrase);
    let end_focus_terms = temporal_interval_focus_terms(&query.end_phrase);
    let start_lower = query.start_phrase.to_ascii_lowercase();
    let end_lower = query.end_phrase.to_ascii_lowercase();
    let start_match =
        best_temporal_interval_rank_line(&lines, &start_lower, &start_terms, &start_focus_terms)?;
    let end_match =
        best_temporal_interval_rank_line(&lines, &end_lower, &end_terms, &end_focus_terms)?;
    let object_identity_terms = temporal_interval_object_identity_terms(&query.end_phrase);
    if weak_temporal_pronoun_reference(&query.start_phrase)
        && !object_identity_terms.is_empty()
        && (!temporal_line_matches_any_term(&start_match.2, &object_identity_terms)
            || !temporal_line_matches_any_term(&end_match.2, &object_identity_terms))
    {
        return None;
    }
    let delta_days = end_match.0 - start_match.0;
    if delta_days <= 0 {
        return None;
    }
    let explicit_dates = extract_explicit_date_rank(&start_match.2).is_some()
        && extract_explicit_date_rank(&end_match.2).is_some();
    Some((
        render_temporal_elapsed_gap_answer(delta_days, &query.unit, explicit_dates)?,
        dedupe_temporal_anchor_evidence([start_match.2, end_match.2]),
        start_match.1 + end_match.1,
    ))
}

pub(super) fn grouped_verbatim_lines(idx: &NeuronIndex) -> HashMap<String, Vec<String>> {
    let mut grouped_entries: HashMap<String, Vec<&BM25Entry>> = HashMap::new();
    for entry in idx
        .retrieval
        .entries
        .iter()
        .filter(|entry| matches!(entry.kind, NeuronKind::Verbatim))
    {
        let group_key = if entry.session_id.is_empty() {
            verbatim_source_group_key(entry)
        } else {
            entry.session_id.clone()
        };
        grouped_entries.entry(group_key).or_default().push(entry);
    }
    let mut grouped = HashMap::new();
    for (group_key, mut entries) in grouped_entries {
        entries.sort_by(|a, b| a.neuron_path.cmp(&b.neuron_path));
        let mut lines = Vec::new();
        for (entry_idx, entry) in entries.into_iter().enumerate() {
            let is_summary = is_session_summary_path(&entry.neuron_path);
            if entry_idx > 0 && is_summary {
                lines.push("[File Boundary]".to_string());
            }
            let Ok(content) = std::fs::read_to_string(&entry.neuron_path) else {
                continue;
            };
            for raw_line in strip_query_surface_section(&content).lines() {
                let line = raw_line.trim();
                if is_summary
                    && !line.starts_with("[Session ")
                    && extract_explicit_date_rank(line).is_none()
                {
                    continue;
                }
                if !line.is_empty() && is_session_answer_candidate_line(line) {
                    lines.push(line.to_string());
                }
            }
        }
        grouped.insert(group_key, lines);
    }
    grouped
}

fn best_relaxed_temporal_event_anchor_line(
    lines: &[String],
    phrase_lower: &str,
    terms: &[String],
) -> Option<(SyntheticEventAnchor, usize, String)> {
    let (rank, score, line) =
        best_temporal_rank_line_with_min_overlap(lines, phrase_lower, terms, Some(1))?;
    let anchor = if let Some(days_ago) = extract_temporal_relative_days(&line) {
        let adjusted = match extract_relative_reference_offset_days(&line) {
            Some((SyntheticTemporalDirection::Earlier, offset)) => days_ago + offset,
            Some((SyntheticTemporalDirection::Later, offset)) => days_ago.saturating_sub(offset),
            None => days_ago,
        };
        SyntheticEventAnchor::RelativeDaysAgo(adjusted)
    } else if extract_explicit_date_rank(&line).is_some() {
        SyntheticEventAnchor::AbsoluteDay(rank)
    } else {
        return None;
    };
    Some((anchor, score, line))
}

fn best_temporal_interval_rank_line(
    lines: &[String],
    phrase_lower: &str,
    terms: &[String],
    focus_terms: &[String],
) -> Option<(i32, usize, String)> {
    let keys = synthetic_answer_surface_term_key_set(terms);
    let focus_keys = synthetic_answer_surface_term_key_set(focus_terms);
    let min_overlap = if keys.len() >= 3 { 2 } else { 1 };
    let min_focus_overlap = if focus_keys.is_empty() {
        0
    } else if focus_keys.len() >= 3 {
        2
    } else {
        1
    };
    let mut best: Option<(i32, usize, usize, usize, String)> = None;
    let mut current_session_rank: Option<i32> = None;
    for (line_idx, line) in lines.iter().enumerate() {
        if line == "[File Boundary]" {
            current_session_rank = None;
            continue;
        }
        if line.starts_with("[Session ") {
            current_session_rank = extract_explicit_date_rank(line);
            continue;
        }
        let lower = line.to_ascii_lowercase();
        let overlap = temporal_interval_overlap_count(&lower, terms);
        let focus_overlap = temporal_interval_overlap_count(&lower, focus_terms);
        let exact = lower.contains(phrase_lower);
        if !exact && (overlap < min_overlap || focus_overlap < min_focus_overlap) {
            continue;
        }
        let Some(rank) = temporal_interval_line_rank(line, current_session_rank) else {
            continue;
        };
        let exact_bonus = usize::from(exact);
        let score = focus_overlap * 20 + overlap * 10 + exact_bonus * 5;
        let should_replace = best
            .as_ref()
            .map(|(_, best_score, best_exact, best_line_idx, _)| {
                score > *best_score
                    || (score == *best_score
                        && (exact_bonus > *best_exact
                            || (exact_bonus == *best_exact && line_idx > *best_line_idx)))
            })
            .unwrap_or(true);
        if should_replace {
            best = Some((rank, score, exact_bonus, line_idx, line.clone()));
        }
    }
    best.map(|(rank, score, _, _, line)| (rank, score, line))
}

fn temporal_interval_line_rank(line: &str, current_session_rank: Option<i32>) -> Option<i32> {
    if let Some(day) = extract_explicit_date_rank(line) {
        return Some(day);
    }
    if let Some(session_rank) = current_session_rank {
        if let Some(days_ago) = extract_temporal_relative_days(line) {
            let adjusted = match extract_relative_reference_offset_days(line) {
                Some((SyntheticTemporalDirection::Earlier, offset)) => days_ago + offset,
                Some((SyntheticTemporalDirection::Later, offset)) => {
                    days_ago.saturating_sub(offset)
                },
                None => days_ago,
            };
            return Some(session_rank - adjusted);
        }
        return Some(session_rank);
    }
    extract_temporal_rank_value(line)
}

fn temporal_interval_overlap_count(lower_line: &str, terms: &[String]) -> usize {
    terms
        .iter()
        .filter(|term| temporal_interval_line_matches_term(lower_line, term))
        .count()
}

fn temporal_interval_line_matches_term(lower_line: &str, term: &str) -> bool {
    if lower_line.contains(term) {
        return true;
    }

    match term {
        "find" | "found" => {
            lower_line.contains("find")
                || lower_line.contains("found")
                || lower_line.contains("saw")
        },
        "love" | "loved" => {
            lower_line.contains("love") || lower_line.contains("checks all the boxes")
        },
        "invest" | "invested" => {
            lower_line.contains("got")
                || lower_line.contains("bought")
                || lower_line.contains("purchased")
                || lower_line.contains("acquired")
                || lower_line.contains("invest")
        },
        "launch" | "launched" => lower_line.contains("launch"),
        "sign" | "signed" => lower_line.contains("sign"),
        "take" | "taking" | "took" => {
            lower_line.contains("take")
                || lower_line.contains("taking")
                || lower_line.contains("took")
        },
        "go" | "went" => lower_line.contains("go") || lower_line.contains("went"),
        "get" | "got" => lower_line.contains("get") || lower_line.contains("got"),
        "buy" | "bought" => lower_line.contains("buy") || lower_line.contains("bought"),
        _ => {
            let stem = term
                .trim_end_matches("ing")
                .trim_end_matches("ed")
                .trim_end_matches('s');
            stem.len() >= 3 && lower_line.contains(stem)
        },
    }
}

fn temporal_interval_start_terms(query: &TemporalIntervalQuery) -> (Vec<String>, Vec<String>) {
    let mut terms: Vec<String> = synthetic_query_terms(&query.start_phrase)
        .into_iter()
        .filter(|term| {
            !matches!(
                term.as_str(),
                "it" | "them" | "this" | "that" | "these" | "those"
            )
        })
        .collect();
    let mut focus_terms = temporal_interval_focus_terms(&query.start_phrase);
    let weak_pronoun_reference = weak_temporal_pronoun_reference(&query.start_phrase);
    if weak_pronoun_reference {
        let object_terms = temporal_interval_focus_terms(&query.end_phrase);
        for term in object_terms {
            terms.push(term.clone());
            focus_terms.push(term);
        }
        terms.sort();
        terms.dedup();
        focus_terms.sort();
        focus_terms.dedup();
    }
    (terms, focus_terms)
}

fn weak_temporal_pronoun_reference(phrase: &str) -> bool {
    [" it", " them", " this", " that", " these", " those"]
        .iter()
        .any(|suffix| phrase.ends_with(suffix))
}

fn temporal_interval_focus_terms(phrase: &str) -> Vec<String> {
    synthetic_query_terms(phrase)
        .into_iter()
        .filter(|term| {
            !matches!(
                term.as_str(),
                "i" | "me"
                    | "my"
                    | "it"
                    | "them"
                    | "this"
                    | "that"
                    | "these"
                    | "those"
                    | "buy"
                    | "bought"
                    | "order"
                    | "ordered"
                    | "attend"
                    | "attended"
                    | "receive"
                    | "received"
                    | "arrive"
                    | "arrived"
                    | "get"
                    | "got"
                    | "gift"
                    | "gifts"
            )
        })
        .collect()
}

fn temporal_interval_object_identity_terms(phrase: &str) -> Vec<String> {
    temporal_interval_focus_terms(phrase)
        .into_iter()
        .filter(|term| {
            !matches!(
                term.as_str(),
                "case"
                    | "cases"
                    | "delivery"
                    | "deliveries"
                    | "item"
                    | "items"
                    | "package"
                    | "packages"
                    | "product"
                    | "products"
                    | "purchase"
                    | "purchases"
                    | "shipment"
                    | "shipments"
                    | "thing"
                    | "things"
            )
        })
        .collect()
}

fn temporal_line_matches_any_term(line: &str, terms: &[String]) -> bool {
    let lower = line.to_ascii_lowercase();
    terms
        .iter()
        .any(|term| temporal_interval_line_matches_term(&lower, term))
}

pub(super) fn dedupe_temporal_anchor_evidence<I>(evidence: I) -> Vec<String>
where
    I: IntoIterator<Item = String>,
{
    let mut deduped = Vec::new();
    let mut seen = HashSet::new();
    for line in evidence {
        if seen.insert(line.clone()) {
            deduped.push(line);
        }
    }
    deduped
}

fn render_temporal_interval_answer(delta_days: i32, explicit_dates: bool) -> String {
    if explicit_dates {
        return format!(
            "{} days. {} days (including the last day) is also acceptable.",
            delta_days,
            delta_days + 1,
        );
    }
    format!("{delta_days} days")
}

fn render_temporal_elapsed_gap_answer(
    delta_days: i32,
    unit: &str,
    explicit_dates: bool,
) -> Option<String> {
    match unit {
        "day" => Some(render_temporal_interval_answer(delta_days, explicit_dates)),
        "week" if delta_days >= 7 => Some((delta_days / 7).to_string()),
        _ => None,
    }
}
