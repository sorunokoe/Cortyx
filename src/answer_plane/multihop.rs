use super::*;

pub(super) fn looks_like_multi_hop_list_query(task: &str) -> bool {
    let lower = task.to_ascii_lowercase();
    lower.starts_with("what groups")
        || lower.starts_with("which groups")
        || lower.starts_with("what instruments")
        || lower.starts_with("which instruments")
        || lower.starts_with("what events")
        || lower.starts_with("which events")
        || lower.starts_with("what books")
        || lower.starts_with("which books")
        || lower.starts_with("what movies")
        || lower.starts_with("which movies")
        || lower.starts_with("what files")
        || lower.starts_with("which files")
        || lower.starts_with("what modules")
        || lower.starts_with("which modules")
        || lower.starts_with("what topics")
        || lower.starts_with("which topics")
        || lower.starts_with("what activities")
        || lower.starts_with("which activities")
        || lower.starts_with("where has ")
        || lower.starts_with("where have ")
        || lower.starts_with("who supports ")
        || lower.starts_with("who supported ")
}

pub(super) fn collect_answer_candidates(
    task: &str,
    evidence: &[EvidenceItem],
) -> Vec<CandidateLine> {
    let task_terms = salient_query_terms(task);
    let subject_hints = extract_subject_hints(task);
    let focus_terms = dialogue_focus_terms(task, &task_terms, &subject_hints);
    let anchor_terms = task_anchor_terms(task, &task_terms, &subject_hints);
    let required_tail_terms = required_tail_anchor_tokens(task);
    let enumerative = is_enumerative_query(task);
    let mut candidates = Vec::new();

    for item in evidence {
        let Some(content) = read_context_text(&item.path, "answer candidate extraction") else {
            continue;
        };
        for line in answer_candidate_lines(&content) {
            let clean = sanitize_answer_text(&line);
            if clean.is_empty() || looks_like_question_turn(&clean) {
                continue;
            }
            let subject_overlap = if subject_hints.is_empty() {
                0
            } else {
                task_overlap_count(&line, &subject_hints)
                    .max(task_overlap_count(&clean, &subject_hints))
            };
            if !subject_hints.is_empty() && subject_overlap == 0 {
                continue;
            }
            let focus_overlap = if focus_terms.is_empty() {
                0
            } else {
                task_overlap_count(&line, &focus_terms)
                    .max(task_overlap_count(&clean, &focus_terms))
            };
            if !focus_terms.is_empty()
                && focus_overlap == 0
                && subject_overlap > 0
                && !is_reason_query(task)
            {
                continue;
            }
            let support_overlap = task_overlap_count(&line, &task_terms)
                .max(task_overlap_count(&clean, &task_terms))
                .max(focus_overlap);
            let anchor_overlap = if anchor_terms.is_empty() {
                0
            } else {
                max_task_overlap([line.as_str(), clean.as_str()], &anchor_terms)
            };
            if !anchor_terms.is_empty() && anchor_overlap == 0 {
                continue;
            }
            if !required_tail_terms.is_empty()
                && max_task_overlap([line.as_str(), clean.as_str()], &required_tail_terms)
                    < required_tail_terms.len()
            {
                continue;
            }
            let weight = candidate_weight(&clean, &task_terms, item.score, false)
                + focus_overlap as f32 * 6.0;
            if weight > 0.0 && support_overlap > 0 {
                candidates.push(CandidateLine {
                    path: item.path.clone(),
                    text: clean.clone(),
                    weight,
                    retrieval_score: item.score,
                    support_overlap,
                    anchor_overlap,
                    specific_anchor_overlap: 0,
                });
                if !enumerative {
                    if let Some(compact) = compact_answer(task, &clean, &task_terms) {
                        if compact != clean
                            && is_informative_compact_answer(&compact)
                            && answer_meets_form_gate(task, &compact, None)
                        {
                            let compact_bonus =
                                4.0 + answer_form_confidence(task, &compact, &task_terms) * 6.0;
                            candidates.push(CandidateLine {
                                path: item.path.clone(),
                                text: compact,
                                weight: weight + compact_bonus,
                                retrieval_score: item.score,
                                support_overlap,
                                anchor_overlap,
                                specific_anchor_overlap: 0,
                            });
                        }
                    }
                }
            }
        }

        for turn in parse_dialogue_turns(&content) {
            let candidate = extract_relation_answer(task, &turn.text, &task_terms)
                .or_else(|| compact_answer(task, &turn.text, &task_terms));
            let Some(candidate) = candidate else {
                continue;
            };
            let clean = sanitize_answer_text(&candidate);
            if clean.is_empty() {
                continue;
            }
            if !subject_hints.is_empty() && !turn_matches_subject(&turn, &subject_hints) {
                continue;
            }
            let focus_overlap = if focus_terms.is_empty() {
                0
            } else {
                task_overlap_count(&turn.text, &focus_terms)
                    .max(task_overlap_count(&clean, &focus_terms))
            };
            if !focus_terms.is_empty() && focus_overlap == 0 {
                continue;
            }
            let support_overlap = task_overlap_count(&turn.text, &task_terms)
                .max(task_overlap_count(&clean, &task_terms))
                .max(focus_overlap)
                .max(1);
            let anchor_overlap = if anchor_terms.is_empty() {
                0
            } else {
                max_task_overlap([turn.text.as_str(), clean.as_str()], &anchor_terms)
            };
            if !anchor_terms.is_empty() && anchor_overlap == 0 {
                continue;
            }
            if !required_tail_terms.is_empty()
                && max_task_overlap([turn.text.as_str(), clean.as_str()], &required_tail_terms)
                    < required_tail_terms.len()
            {
                continue;
            }
            let weight = item.score * 10.0
                + dialogue_match_score(&turn.text, &task_terms)
                + speaker_match_bonus(turn.speaker.as_deref(), &subject_hints)
                + focus_overlap as f32 * 8.0
                + 10.0;
            candidates.push(CandidateLine {
                path: item.path.clone(),
                text: clean,
                weight,
                retrieval_score: item.score,
                support_overlap,
                anchor_overlap,
                specific_anchor_overlap: 0,
            });
        }
    }

    candidates.sort_by(|a, b| {
        b.weight
            .total_cmp(&a.weight)
            .then_with(|| a.path.cmp(&b.path))
            .then_with(|| a.text.cmp(&b.text))
    });
    candidates
}

pub(super) fn select_multi_item_answer_from_candidates(
    task: &str,
    candidates: &[CandidateLine],
    _min_answer_confidence: Option<f32>,
) -> Option<String> {
    if !looks_like_multi_hop_list_query(task) || candidates.is_empty() {
        return None;
    }

    let top_weight = candidates.first()?.weight;
    let mut chosen = Vec::new();
    let mut seen_paths = HashSet::new();
    let mut seen_keys = HashSet::new();

    for prefer_new_path in [true, false] {
        for candidate in candidates {
            if candidate.weight + 14.0 < top_weight {
                break;
            }
            if !candidate_has_required_anchor_support(task, candidate) {
                continue;
            }
            if !is_composeable_multi_item_candidate(candidate) {
                continue;
            }
            if prefer_new_path && !seen_paths.insert(candidate.path.clone()) {
                continue;
            }

            let mut added_any = false;
            for item in split_composable_answer_items(&candidate.text) {
                let key = normalized_answer_key(&item);
                if key.is_empty()
                    || !seen_keys.insert(key)
                    || chosen
                        .iter()
                        .any(|existing: &String| answer_items_overlap(existing.as_str(), &item))
                {
                    continue;
                }
                chosen.push(item);
                added_any = true;
                if chosen.len() >= 3 {
                    break;
                }
            }

            if prefer_new_path && !added_any {
                seen_paths.remove(&candidate.path);
            }
            if chosen.len() >= 3 {
                break;
            }
        }
        if chosen.len() >= 2 {
            break;
        }
    }

    (chosen.len() >= 2).then(|| format_answer_list(&chosen))
}

fn is_composeable_multi_item_candidate(candidate: &CandidateLine) -> bool {
    let word_count = candidate.text.split_whitespace().count();
    word_count > 0
        && word_count <= 8
        && !candidate.text.contains('?')
        && !candidate.text.contains(" because ")
        && !candidate.text.contains(". ")
        && (is_informative_compact_answer(&candidate.text) || candidate.text.contains(','))
}

fn split_composable_answer_items(text: &str) -> Vec<String> {
    let clean = sanitize_answer_text(text);
    if clean.contains(',') {
        let parts = clean
            .replace(", and ", ", ")
            .split(',')
            .map(str::trim)
            .map(sanitize_inline)
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>();
        if parts.len() >= 2 {
            return parts;
        }
    }
    vec![clean]
}

pub(super) fn normalized_answer_key(text: &str) -> String {
    sanitize_inline(
        &text
            .trim()
            .trim_matches(|c: char| matches!(c, '"' | '\'' | ',' | '.' | ';' | ':' | '!' | '?'))
            .to_ascii_lowercase(),
    )
}

pub(super) fn answer_items_overlap(left: &str, right: &str) -> bool {
    let left_key = normalized_answer_key(left);
    let right_key = normalized_answer_key(right);
    !left_key.is_empty()
        && !right_key.is_empty()
        && (left_key == right_key || left_key.contains(&right_key) || right_key.contains(&left_key))
}

pub(super) fn format_answer_list(items: &[String]) -> String {
    match items {
        [] => String::new(),
        [item] => item.clone(),
        [left, right] => format!("{left} and {right}"),
        _ => {
            let mut out = items[..items.len() - 1].join(", ");
            out.push_str(", and ");
            out.push_str(items.last().unwrap_or(&String::new()));
            out
        },
    }
}

pub(super) fn decompose_multi_hop_subquestions(task: &str) -> Option<Vec<String>> {
    let trimmed = task.trim().trim_end_matches('?').trim();
    if trimmed.is_empty() {
        return None;
    }

    decompose_explicit_question_clauses(trimmed)
        .or_else(|| decompose_shared_prefix_question(trimmed))
}

fn decompose_explicit_question_clauses(task: &str) -> Option<Vec<String>> {
    for question_word in ["what", "who", "where", "when", "why", "how", "which"] {
        for marker in [
            format!(", and {question_word} "),
            format!(" and {question_word} "),
        ] {
            let Some((left, right)) = split_once_case_insensitive(task, &marker) else {
                continue;
            };
            return Some(vec![
                ensure_question_suffix(left),
                ensure_question_suffix(&format!("{question_word} {}", right.trim())),
            ]);
        }
    }
    None
}

fn decompose_shared_prefix_question(task: &str) -> Option<Vec<String>> {
    let lower = task.to_ascii_lowercase();
    let opener = ["what is ", "what are ", "what was ", "what were "]
        .into_iter()
        .find(|prefix| lower.starts_with(prefix))?;
    let tail = task[opener.len()..].trim();
    if !tail.contains(" and ") {
        return None;
    }

    let normalized_tail = tail.replace(", and ", ", ").replace(" and ", ", ");
    let segments = normalized_tail
        .split(',')
        .map(str::trim)
        .filter(|segment| !segment.is_empty())
        .collect::<Vec<_>>();
    if !(2..=3).contains(&segments.len()) {
        return None;
    }

    let first_field = trailing_multihop_field(segments.first()?)?;
    let first_segment = segments.first()?.trim();
    let shared_prefix = first_segment[..first_segment.len() - first_field.len()]
        .trim()
        .to_string();
    if shared_prefix.is_empty() {
        return None;
    }

    let mut out = Vec::new();
    for segment in segments {
        let field = trailing_multihop_field(segment)
            .or_else(|| segment_field_name(segment))
            .or_else(|| infer_subanswer_label(segment).map(str::to_string))?;
        let body = if segment
            .to_ascii_lowercase()
            .starts_with(&shared_prefix.to_ascii_lowercase())
        {
            segment.to_string()
        } else {
            format!("{shared_prefix} {field}")
        };
        out.push(ensure_question_suffix(&format!("{opener}{body}")));
    }
    Some(out)
}

fn trailing_multihop_field(text: &str) -> Option<String> {
    let lower = text.trim().to_ascii_lowercase();
    MULTIHOP_BUNDLE_FIELDS
        .iter()
        .filter(|field| lower.ends_with(**field))
        .max_by_key(|field| field.len())
        .map(|field| (*field).to_string())
}

fn segment_field_name(text: &str) -> Option<String> {
    let clean = text.trim();
    MULTIHOP_BUNDLE_FIELDS
        .iter()
        .find(|field| clean.eq_ignore_ascii_case(field))
        .map(|field| (*field).to_string())
}

fn ensure_question_suffix(text: &str) -> String {
    let mut clean = sanitize_inline(text.trim().trim_matches(','));
    if !clean.ends_with('?') {
        clean.push('?');
    }
    clean
}

pub(super) fn infer_subanswer_label(task: &str) -> Option<&'static str> {
    let lower = task.to_ascii_lowercase();
    if lower.contains("next step") || lower.contains("next action") {
        Some("next step")
    } else if structured_diary_blocker_query(&lower) {
        Some("blocker")
    } else if lower.contains("find") || lower.contains("found") || lower.contains("discover") {
        Some("found")
    } else if structured_diary_status_query(&lower) {
        Some("status")
    } else if structured_diary_goal_query(&lower) {
        Some("goal")
    } else if structured_diary_dependencies_query(&lower) {
        Some("dependencies")
    } else if structured_diary_entities_query(&lower) {
        Some("entities")
    } else if structured_diary_action_query(&lower) {
        Some("action")
    } else if structured_diary_title_query(&lower) {
        Some("title")
    } else if lower.contains("where ")
        || lower.contains(" location")
        || lower.contains(" live")
        || lower.contains(" residence")
        || lower.contains(" city")
        || lower.contains(" home")
    {
        Some("location")
    } else if lower.contains("job")
        || lower.contains("occupation")
        || lower.contains("career")
        || lower.contains(" role")
    {
        Some("job")
    } else {
        None
    }
}
