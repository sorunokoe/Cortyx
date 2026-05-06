use super::*;

pub(super) fn resolve_relation_answer(
    task: &str,
    evidence: &[EvidenceItem],
    min_answer_confidence: Option<f32>,
) -> Option<RelationResolution> {
    if looks_like_multi_hop_list_query(task)
        || is_enumerative_query(task)
        || !looks_like_relation_query(task)
    {
        return None;
    }

    let mut candidates = aggregate_answer_candidates(collect_relation_candidates(task, evidence));
    candidates.retain(|candidate| {
        candidate_has_required_anchor_support(task, candidate)
            && answer_meets_form_gate(task, &candidate.text, min_answer_confidence)
    });

    if let Some(kg_support) = best_relation_kg_support(task, evidence) {
        if let Some(candidate) = candidates
            .into_iter()
            .find(|candidate| relation_candidate_matches_kg(&candidate.text, &kg_support.values))
        {
            return Some(RelationResolution::Answer(candidate.text));
        }

        let answer = format_answer_list(&kg_support.values);
        return if answer.is_empty() {
            Some(RelationResolution::Suppress)
        } else {
            Some(RelationResolution::Answer(answer))
        };
    }

    let top = candidates.first()?.clone();
    if candidates.iter().skip(1).any(|candidate| {
        candidate.weight + 12.0 >= top.weight && !answer_items_overlap(&candidate.text, &top.text)
    }) {
        return Some(RelationResolution::Suppress);
    }

    Some(RelationResolution::Answer(top.text))
}

pub(super) fn looks_like_relation_query(task: &str) -> bool {
    let lower = task.to_ascii_lowercase();
    if lower.contains("ingredient") || lower.contains("recipe") {
        return true;
    }
    if !relation_answer_markers(&lower).is_empty() {
        return true;
    }
    [
        "job",
        "occupation",
        "career",
        "role",
        "live",
        "location",
        "residence",
        "city",
        "home",
        "based",
        "partner",
        "husband",
        "wife",
        "boyfriend",
        "girlfriend",
        "spouse",
        "degree",
        "education",
        "major",
        "study",
        "studied",
        "school",
        "pet",
        "dog",
        "cat",
        "phone",
        "number",
        "book",
        "reading",
        "project",
        "playlist",
        "blog",
        "channel",
        "called",
        "name",
        "vehicle",
        "car",
        "truck",
        "model",
        "commute",
        "diet",
        "allergy",
        "allergic",
        "group",
        "joined",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn collect_relation_candidates(task: &str, evidence: &[EvidenceItem]) -> Vec<CandidateLine> {
    let task_terms = salient_query_terms(task);
    let subject_hints = extract_subject_hints(task);
    let focus_terms = dialogue_focus_terms(task, &task_terms, &subject_hints);
    let anchor_terms = task_anchor_terms(task, &task_terms, &subject_hints);
    let institution_anchor_terms = institution_specific_anchor_terms(task);
    if task_terms.is_empty() {
        return Vec::new();
    }

    let mut candidates = Vec::new();
    for item in evidence {
        let Some(content) = read_context_text(&item.path, "relation candidate collection") else {
            continue;
        };

        for row in parse_answer_surface_rows(&content) {
            let score = answer_surface_score(&row, &task_terms, item.score);
            if score <= 0.0 {
                continue;
            }
            let support_overlap = task_overlap_count(&row.question_pattern, &task_terms)
                .max(task_overlap_count(&row.answer_span, &task_terms))
                .max(1);
            let anchor_overlap = if anchor_terms.is_empty() {
                0
            } else {
                max_task_overlap(
                    [row.question_pattern.as_str(), row.answer_span.as_str()],
                    &anchor_terms,
                )
            };
            if !anchor_terms.is_empty() && anchor_overlap == 0 {
                continue;
            }
            let specific_anchor_overlap = if institution_anchor_terms.is_empty() {
                0
            } else {
                max_task_overlap(
                    [row.question_pattern.as_str(), row.answer_span.as_str()],
                    &institution_anchor_terms,
                )
            };
            candidates.push(CandidateLine {
                path: item.path.clone(),
                text: row.answer_span,
                weight: item.score * 10.0 + score * 2.0,
                retrieval_score: item.score,
                support_overlap,
                anchor_overlap,
                specific_anchor_overlap,
            });
        }

        let turns = parse_dialogue_turns(&content);
        for turn in &turns {
            let base_score = dialogue_match_score(&turn.text, &task_terms);
            let speaker_bonus = speaker_match_bonus(turn.speaker.as_deref(), &subject_hints);
            let Some(candidate) = extract_relation_answer(task, &turn.text, &task_terms) else {
                continue;
            };
            if !subject_hints.is_empty() && !turn_matches_subject(turn, &subject_hints) {
                continue;
            }
            let clean = sanitize_answer_text(&candidate);
            if clean.is_empty() {
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
            let total_score = base_score + speaker_bonus + focus_overlap as f32 * 8.0 + 10.0;
            if total_score < 24.0 {
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
            let specific_anchor_overlap = if institution_anchor_terms.is_empty() {
                0
            } else {
                max_task_overlap(
                    [turn.text.as_str(), clean.as_str()],
                    &institution_anchor_terms,
                )
            };
            candidates.push(CandidateLine {
                path: item.path.clone(),
                text: clean,
                weight: item.score * 10.0 + total_score,
                retrieval_score: item.score,
                support_overlap,
                anchor_overlap,
                specific_anchor_overlap,
            });
        }

        if turns.len() < 2 {
            continue;
        }
        let requires_reason = is_reason_query(task);
        for idx in 0..turns.len() - 1 {
            let question = &turns[idx];
            let answer = &turns[idx + 1];
            if !looks_like_question_turn(&question.text) {
                continue;
            }
            if question.speaker.is_some() && question.speaker == answer.speaker {
                continue;
            }

            let mut context = question.text.clone();
            if idx > 0 {
                context = format!("{} {}", turns[idx - 1].text, context);
            }
            let subject_overlap = if subject_hints.is_empty() {
                0
            } else {
                task_overlap_count(&context, &subject_hints)
                    .max(task_overlap_count(&answer.text, &subject_hints))
            };
            if !subject_hints.is_empty() && subject_overlap == 0 {
                continue;
            }
            let question_focus_overlap = if focus_terms.is_empty() {
                0
            } else {
                task_overlap_count(&context, &focus_terms)
            };
            let answer_focus_overlap = if focus_terms.is_empty() {
                0
            } else {
                task_overlap_count(&answer.text, &focus_terms)
            };
            if !focus_terms.is_empty() && question_focus_overlap == 0 && answer_focus_overlap == 0 {
                continue;
            }
            let question_score = dialogue_match_score(&context, &task_terms);
            let speaker_bonus = speaker_match_bonus(answer.speaker.as_deref(), &subject_hints);
            let total_score = question_score
                + speaker_bonus
                + question_focus_overlap as f32 * 8.0
                + answer_focus_overlap as f32 * 10.0;
            let threshold = if requires_reason { 26.0 } else { 20.0 };
            if total_score < threshold {
                continue;
            }

            let Some(candidate) = extract_turn_answer(task, &answer.text, &task_terms) else {
                continue;
            };
            let clean = sanitize_answer_text(&candidate);
            if clean.is_empty() {
                continue;
            }
            let candidate_focus_overlap = if focus_terms.is_empty() {
                0
            } else {
                task_overlap_count(&clean, &focus_terms)
            };
            if !focus_terms.is_empty()
                && candidate_focus_overlap == 0
                && answer_focus_overlap == 0
                && !requires_reason
            {
                continue;
            }
            let support_overlap = task_overlap_count(&context, &task_terms)
                .max(task_overlap_count(&clean, &task_terms))
                .max(candidate_focus_overlap)
                .max(1);
            let anchor_overlap = if anchor_terms.is_empty() {
                0
            } else {
                max_task_overlap(
                    [context.as_str(), answer.text.as_str(), clean.as_str()],
                    &anchor_terms,
                )
            };
            if !anchor_terms.is_empty() && anchor_overlap == 0 {
                continue;
            }
            let specific_anchor_overlap = if institution_anchor_terms.is_empty() {
                0
            } else {
                max_task_overlap(
                    [context.as_str(), answer.text.as_str(), clean.as_str()],
                    &institution_anchor_terms,
                )
            };
            candidates.push(CandidateLine {
                path: item.path.clone(),
                text: clean,
                weight: item.score * 10.0
                    + total_score
                    + candidate_focus_overlap as f32 * 8.0
                    + subject_overlap as f32 * 4.0
                    + 6.0,
                retrieval_score: item.score,
                support_overlap,
                anchor_overlap,
                specific_anchor_overlap,
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

pub(super) fn aggregate_answer_candidates(candidates: Vec<CandidateLine>) -> Vec<CandidateLine> {
    let mut buckets: HashMap<String, RelationCandidateBucket> = HashMap::new();
    for candidate in candidates {
        let key = normalized_answer_key(&candidate.text);
        if key.is_empty() {
            continue;
        }
        let bucket = buckets
            .entry(key)
            .or_insert_with(|| RelationCandidateBucket {
                best_candidate: candidate.clone(),
                best_single_weight: candidate.weight,
                total_weight: 0.0,
                max_retrieval_score: candidate.retrieval_score,
                max_support_overlap: candidate.support_overlap,
                max_anchor_overlap: candidate.anchor_overlap,
                max_specific_anchor_overlap: candidate.specific_anchor_overlap,
                paths: HashSet::new(),
                hits: 0,
            });
        if candidate.weight > bucket.best_single_weight {
            bucket.best_candidate = candidate.clone();
            bucket.best_single_weight = candidate.weight;
        }
        bucket.total_weight += candidate.weight;
        bucket.max_retrieval_score = bucket.max_retrieval_score.max(candidate.retrieval_score);
        bucket.max_support_overlap = bucket.max_support_overlap.max(candidate.support_overlap);
        bucket.max_anchor_overlap = bucket.max_anchor_overlap.max(candidate.anchor_overlap);
        bucket.max_specific_anchor_overlap = bucket
            .max_specific_anchor_overlap
            .max(candidate.specific_anchor_overlap);
        bucket.paths.insert(candidate.path.clone());
        bucket.hits += 1;
    }

    let mut aggregated = buckets
        .into_values()
        .map(|bucket| {
            let mut candidate = bucket.best_candidate;
            candidate.weight = bucket.total_weight
                + (bucket.paths.len().saturating_sub(1) as f32) * 8.0
                + (bucket.hits.saturating_sub(1) as f32) * 4.0;
            candidate.retrieval_score = bucket.max_retrieval_score;
            candidate.support_overlap = bucket.max_support_overlap;
            candidate.anchor_overlap = bucket.max_anchor_overlap;
            candidate.specific_anchor_overlap = bucket.max_specific_anchor_overlap;
            candidate
        })
        .collect::<Vec<_>>();
    aggregated.sort_by(|a, b| {
        b.weight
            .total_cmp(&a.weight)
            .then_with(|| a.path.cmp(&b.path))
            .then_with(|| a.text.cmp(&b.text))
    });
    aggregated
}

fn best_relation_kg_support(task: &str, evidence: &[EvidenceItem]) -> Option<RelationKgSupport> {
    let task_terms = salient_query_terms(task);
    if task_terms.is_empty() {
        return None;
    }

    let mut candidates = Vec::new();
    for item in evidence {
        if !path_looks_like_kg_neuron(&item.path) {
            continue;
        }
        let Ok(entity) = kg::KgEntity::load(&item.path) else {
            continue;
        };
        if entity.facts.is_empty() {
            continue;
        }

        let mut predicates = entity
            .facts
            .iter()
            .map(|fact| fact.predicate.clone())
            .collect::<Vec<_>>();
        predicates.sort();
        predicates.dedup();

        for predicate in predicates {
            if !is_relation_kg_predicate(&predicate) {
                continue;
            }
            let values = current_kg_values(&entity, &predicate);
            if values.is_empty() {
                continue;
            }
            let score =
                relation_kg_candidate_score(task, &task_terms, item.score, &entity, &predicate);
            if score <= 0.0 {
                continue;
            }
            candidates.push((score, values));
        }
    }

    candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
    let (top_score, top_values) = candidates.first()?.clone();
    if top_score < 14.0 {
        return None;
    }
    if candidates.iter().skip(1).any(|(score, values)| {
        *score + 4.0 >= top_score && !kg_value_sets_overlap(values, &top_values)
    }) {
        return None;
    }

    Some(RelationKgSupport { values: top_values })
}

fn path_looks_like_kg_neuron(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.starts_with("_kg_") && name.ends_with(".context.md"))
        .unwrap_or(false)
}

fn is_relation_kg_predicate(predicate: &str) -> bool {
    matches!(
        predicate,
        "occupation"
            | "location"
            | "partner"
            | "phone"
            | "education"
            | "major"
            | "school"
            | "studying"
            | "pet"
            | "book"
            | "project_name"
            | "commute_time"
            | "diet"
            | "allergy"
            | "vehicle_model"
            | "family_trip_location"
    )
}

fn relation_kg_candidate_score(
    task: &str,
    task_terms: &[String],
    retrieval_score: f32,
    entity: &kg::KgEntity,
    predicate: &str,
) -> f32 {
    let predicate_context = kg_predicate_query_terms(predicate).join(" ");
    let predicate_overlap = task_overlap_count(&predicate_context, task_terms);
    if predicate_overlap == 0 {
        return 0.0;
    }

    let entity_context = kg_entity_query_terms(&entity.entity).join(" ");
    let entity_overlap = if entity_context.is_empty() {
        0
    } else {
        task_overlap_count(&entity_context, task_terms)
    };
    let lower = task.to_ascii_lowercase();
    let entity_bonus = if entity_overlap > 0 {
        entity_overlap as f32 * 8.0
    } else if (entity.entity == "user" || entity.entity.starts_with("agent_"))
        && query_targets_primary_entity(&lower)
    {
        4.0
    } else {
        0.0
    };

    predicate_overlap as f32 * 10.0
        + entity_bonus
        + relation_predicate_query_bonus(&lower, predicate)
        + retrieval_score
}

fn query_targets_primary_entity(task_lower: &str) -> bool {
    task_lower.contains(" my ")
        || task_lower.starts_with("my ")
        || task_lower.starts_with("what is my ")
        || task_lower.starts_with("what's my ")
        || task_lower.starts_with("where do i ")
        || task_lower.starts_with("where am i ")
        || task_lower.starts_with("who is my ")
        || task_lower.starts_with("what is the reviewer")
}

fn relation_predicate_query_bonus(task_lower: &str, predicate: &str) -> f32 {
    match predicate {
        "occupation"
            if ["job", "occupation", "career", "role", "work"]
                .iter()
                .any(|needle| task_lower.contains(needle)) =>
        {
            6.0
        },
        "location"
            if task_lower.starts_with("where ")
                || [
                    "live",
                    "location",
                    "residence",
                    "city",
                    "home",
                    "based",
                    "moved",
                ]
                .iter()
                .any(|needle| task_lower.contains(needle)) =>
        {
            6.0
        },
        "partner"
            if task_lower.starts_with("who ")
                || [
                    "partner",
                    "husband",
                    "wife",
                    "boyfriend",
                    "girlfriend",
                    "spouse",
                ]
                .iter()
                .any(|needle| task_lower.contains(needle)) =>
        {
            6.0
        },
        "major"
            if ["major", "field"]
                .iter()
                .any(|needle| task_lower.contains(needle)) =>
        {
            6.0
        },
        "education"
            if ["study", "studied", "degree", "education", "graduated"]
                .iter()
                .any(|needle| task_lower.contains(needle)) =>
        {
            5.0
        },
        "school"
            if ["school", "college", "university"]
                .iter()
                .any(|needle| task_lower.contains(needle)) =>
        {
            6.0
        },
        "pet"
            if ["pet", "dog", "cat", "name", "called"]
                .iter()
                .any(|needle| task_lower.contains(needle)) =>
        {
            6.0
        },
        "phone"
            if ["phone", "number", "call"]
                .iter()
                .any(|needle| task_lower.contains(needle)) =>
        {
            6.0
        },
        "book"
            if ["book", "read", "reading", "novel"]
                .iter()
                .any(|needle| task_lower.contains(needle)) =>
        {
            6.0
        },
        "project_name"
            if ["project", "playlist", "blog", "channel", "called", "name"]
                .iter()
                .any(|needle| task_lower.contains(needle)) =>
        {
            6.0
        },
        "vehicle_model"
            if ["vehicle", "car", "truck", "model", "drive"]
                .iter()
                .any(|needle| task_lower.contains(needle)) =>
        {
            6.0
        },
        "family_trip_location"
            if task_lower.starts_with("where ")
                || ["family", "trip", "vacation", "travel", "destination"]
                    .iter()
                    .any(|needle| task_lower.contains(needle)) =>
        {
            5.0
        },
        "commute_time"
            if ["commute", "travel", "minutes", "time"]
                .iter()
                .any(|needle| task_lower.contains(needle)) =>
        {
            5.0
        },
        "diet"
            if ["diet", "vegan", "vegetarian", "pescatarian", "keto"]
                .iter()
                .any(|needle| task_lower.contains(needle)) =>
        {
            5.0
        },
        "allergy"
            if ["allergy", "allergic"]
                .iter()
                .any(|needle| task_lower.contains(needle)) =>
        {
            5.0
        },
        "studying"
            if ["study", "studying"]
                .iter()
                .any(|needle| task_lower.contains(needle)) =>
        {
            4.0
        },
        _ => 0.0,
    }
}

fn relation_candidate_matches_kg(candidate: &str, values: &[String]) -> bool {
    values
        .iter()
        .any(|value| answer_items_overlap(candidate, value))
}

fn kg_value_sets_overlap(left: &[String], right: &[String]) -> bool {
    left.iter().any(|left_value| {
        right
            .iter()
            .any(|right_value| answer_items_overlap(left_value, right_value))
    })
}
