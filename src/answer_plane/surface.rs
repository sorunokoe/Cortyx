use super::*;

pub(super) fn select_answer_surface(task: &str, evidence: &[EvidenceItem]) -> Option<String> {
    let task_terms = salient_query_terms(task);
    let subject_hints = extract_subject_hints(task);
    let focus_terms = dialogue_focus_terms(task, &task_terms, &subject_hints);
    let typed_open_qa = looks_like_typed_open_qa_query(task);
    let anchor_terms = task_anchor_terms(task, &task_terms, &subject_hints);
    if task_terms.is_empty() {
        return None;
    }

    let mut buckets: HashMap<String, AnswerSurfaceBucket> = HashMap::new();
    for item in evidence {
        let Some(content) = read_context_text(&item.path, "answer surface lookup") else {
            continue;
        };
        for row in parse_answer_surface_rows(&content) {
            let overlap = answer_surface_overlap(&row, &task_terms);
            if overlap == 0 {
                continue;
            }
            let anchor_overlap = if anchor_terms.is_empty() {
                0
            } else {
                answer_surface_overlap(&row, &anchor_terms)
            };
            if !anchor_terms.is_empty() && anchor_overlap == 0 {
                continue;
            }
            let subject_overlap = if subject_hints.is_empty() {
                0
            } else {
                answer_surface_overlap(&row, &subject_hints)
            };
            let focus_overlap = if focus_terms.is_empty() {
                0
            } else {
                answer_surface_overlap(&row, &focus_terms)
            };
            if !focus_terms.is_empty() && focus_overlap == 0 && subject_overlap > 0 {
                continue;
            }
            let score = answer_surface_score(&row, &task_terms, item.score)
                + focus_overlap as f32 * 5.0
                + subject_overlap as f32 * 2.0;
            if score <= 0.0 {
                continue;
            }
            let key = normalized_answer_key(&row.answer_span);
            if key.is_empty() {
                continue;
            }
            let bucket = buckets.entry(key).or_insert_with(|| AnswerSurfaceBucket {
                answer_span: row.answer_span.clone(),
                best_score: score,
                total_score: 0.0,
                best_confidence: row.confidence,
                max_overlap: 0,
                max_anchor_overlap: anchor_overlap,
                paths: HashSet::new(),
                hits: 0,
            });
            if score > bucket.best_score
                || ((score - bucket.best_score).abs() < 0.01
                    && row.answer_span.len() < bucket.answer_span.len())
            {
                bucket.answer_span = row.answer_span.clone();
                bucket.best_score = score;
                bucket.best_confidence = row.confidence;
            }
            bucket.total_score += score;
            bucket.max_overlap = bucket.max_overlap.max(overlap.max(focus_overlap));
            bucket.max_anchor_overlap = bucket.max_anchor_overlap.max(anchor_overlap);
            bucket.paths.insert(item.path.clone());
            bucket.hits += 1;
        }
    }

    let mut buckets = buckets
        .into_values()
        .filter(|bucket| answer_meets_form_gate(task, &bucket.answer_span, None))
        .collect::<Vec<_>>();
    buckets.sort_by(|left, right| {
        answer_surface_bucket_rank(right)
            .total_cmp(&answer_surface_bucket_rank(left))
            .then_with(|| right.max_anchor_overlap.cmp(&left.max_anchor_overlap))
            .then_with(|| right.max_overlap.cmp(&left.max_overlap))
            .then_with(|| right.paths.len().cmp(&left.paths.len()))
            .then_with(|| left.answer_span.len().cmp(&right.answer_span.len()))
            .then_with(|| left.answer_span.cmp(&right.answer_span))
    });
    let top = buckets.first()?;
    if let Some(next) = buckets.get(1) {
        if answer_surface_buckets_conflict(top, next) {
            return None;
        }
    }
    Some(if typed_open_qa {
        format_open_qa_answer_surface_answer(task, &top.answer_span)
    } else {
        top.answer_span.clone()
    })
}

fn answer_surface_overlap(row: &AnswerSurfaceRow, task_terms: &[String]) -> usize {
    let pattern_terms = salient_query_terms(&row.question_pattern);
    if pattern_terms.is_empty() {
        return 0;
    }
    term_list_overlap_count(task_terms, &pattern_terms)
}

fn answer_surface_bucket_rank(bucket: &AnswerSurfaceBucket) -> f32 {
    bucket.total_score
        + bucket.max_overlap as f32
        + bucket.max_anchor_overlap as f32 * 6.0
        + (bucket.paths.len().saturating_sub(1) as f32) * 2.5
        + (bucket.hits.saturating_sub(1) as f32) * 0.5
        + bucket.best_confidence * 2.0
}

fn answer_surface_buckets_conflict(
    top: &AnswerSurfaceBucket,
    runner_up: &AnswerSurfaceBucket,
) -> bool {
    !answer_items_overlap(&top.answer_span, &runner_up.answer_span)
        && answer_surface_bucket_rank(runner_up) + 2.5 >= answer_surface_bucket_rank(top)
        && runner_up.max_overlap + 1 >= top.max_overlap
}

pub(super) fn parse_answer_surface_rows(content: &str) -> Vec<AnswerSurfaceRow> {
    let sections = parse_sections(content);
    let Some(table) = sections.get("answer_surface") else {
        return Vec::new();
    };

    table
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if !trimmed.starts_with('|') {
                return None;
            }
            let columns = trimmed
                .trim_matches('|')
                .split('|')
                .map(str::trim)
                .collect::<Vec<_>>();
            if columns.len() != 3 {
                return None;
            }
            if columns[0].eq_ignore_ascii_case("question_pattern")
                || columns[0].chars().all(|c| c == '-' || c == ' ')
            {
                return None;
            }
            let confidence = columns[2].parse::<f32>().unwrap_or(0.0);
            let answer_span = sanitize_answer_text(columns[1]);
            if answer_span.is_empty() {
                return None;
            }
            Some(AnswerSurfaceRow {
                question_pattern: columns[0].to_string(),
                answer_span,
                confidence,
            })
        })
        .collect()
}

pub(super) fn answer_surface_score(
    row: &AnswerSurfaceRow,
    task_terms: &[String],
    retrieval_score: f32,
) -> f32 {
    let pattern_terms = salient_query_terms(&row.question_pattern);
    if pattern_terms.is_empty() {
        return 0.0;
    }
    let overlap = term_list_overlap_count(task_terms, &pattern_terms);
    if overlap == 0 {
        return 0.0;
    }

    let coverage = overlap as f32 / task_terms.len().max(1) as f32;
    let specificity = overlap as f32 / pattern_terms.len().max(1) as f32;
    retrieval_score + overlap as f32 * 4.0 + coverage * 6.0 + specificity * 2.0 + row.confidence
}
