//! Answer surface composition, ranking, and list formatting.

use super::super::*;

impl NeuronIndex {
    pub fn synthetic_answer_surface_answer(&self, task: &str, task_lower: &str) -> Option<PathBuf> {
        let task_terms = synthetic_query_terms(task_lower);
        if task_terms.len() < 2 {
            return None;
        }
        let compose_list_answer = Self::synthetic_answer_surface_is_list_query(task_lower);
        let query_profile = synthetic_answer_surface_query_profile(
            task,
            task_lower,
            &task_terms,
            compose_list_answer,
        );

        let mut buckets: HashMap<String, IndexAnswerSurfaceBucket> = HashMap::new();
        let mut candidates: Vec<_> = self
            .entries
            .iter()
            .filter(|entry| matches!(entry.kind, NeuronKind::Verbatim))
            .map(|entry| (entry, self.bm25_score(&task_terms, entry)))
            .collect();
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let has_positive_candidates = candidates
            .iter()
            .any(|(_, retrieval_score)| *retrieval_score > 0.0);
        if has_positive_candidates {
            candidates.retain(|(_, retrieval_score)| *retrieval_score > 0.0);
        }
        let min_overlap = if matches!(
            query_profile.route_kind,
            SyntheticAnswerSurfaceRouteKind::Choice
        ) {
            1
        } else if task_terms.len() >= 6 {
            3
        } else {
            2
        };

        let candidate_limit = if has_positive_candidates {
            usize::min(candidates.len(), if compose_list_answer { 96 } else { 32 })
        } else {
            candidates.len()
        };

        for (entry, retrieval_score) in candidates.into_iter().take(candidate_limit) {
            let Ok(content) = std::fs::read_to_string(&entry.neuron_path) else {
                continue;
            };
            let rows = parse_index_answer_surface_rows(&content);
            if rows.is_empty() {
                continue;
            }

            for row in rows {
                let evidence_line = answer_surface_evidence_line(
                    &content,
                    &task_terms,
                    &row.answer_span,
                    &row.question_pattern,
                );
                let (has_future_answer_evidence, has_completed_answer_evidence) =
                    answer_surface_answer_span_evidence_state(&content, &row.answer_span);
                let (score, overlap) = index_answer_surface_score(
                    &row,
                    retrieval_score,
                    &query_profile,
                    evidence_line.as_deref(),
                    has_future_answer_evidence,
                    has_completed_answer_evidence,
                );
                if overlap < min_overlap || score < 8.0 {
                    continue;
                }

                let Some(projected_answer) = synthetic_answer_surface_project_answer(
                    &query_profile,
                    &row,
                    evidence_line.as_deref(),
                ) else {
                    continue;
                };
                let row_family = synthetic_answer_surface_relation_family(
                    &row.question_pattern,
                    evidence_line.as_deref(),
                );
                let key = normalized_index_answer_surface_key(&projected_answer);
                if key.is_empty() {
                    continue;
                }

                let mut evidence = Vec::new();
                if let Some(line) = evidence_line {
                    evidence.push(line);
                }
                evidence.push(format!(
                    "answer_surface: {} -> {}",
                    row.question_pattern, row.answer_span
                ));

                let bucket = buckets
                    .entry(key)
                    .or_insert_with(|| IndexAnswerSurfaceBucket {
                        answer_span: projected_answer.clone(),
                        best_score: score,
                        total_score: 0.0,
                        max_overlap: 0,
                        paths: HashSet::new(),
                        hits: 0,
                        evidence: Vec::new(),
                        relation_families: HashSet::new(),
                    });
                if score > bucket.best_score
                    || ((score - bucket.best_score).abs() < 0.01
                        && projected_answer.len() < bucket.answer_span.len())
                {
                    bucket.answer_span = projected_answer;
                    bucket.best_score = score;
                }
                bucket.total_score += score;
                bucket.max_overlap = bucket.max_overlap.max(overlap);
                bucket.paths.insert(entry.neuron_path.clone());
                bucket.hits += 1;
                if let Some(row_family) = row_family {
                    bucket.relation_families.insert(row_family);
                }
                for line in evidence {
                    if bucket.evidence.len() >= 3 {
                        break;
                    }
                    if !bucket.evidence.iter().any(|existing| existing == &line) {
                        bucket.evidence.push(line);
                    }
                }
            }
        }

        let mut buckets = buckets.into_values().collect::<Vec<_>>();
        buckets.sort_by(|left, right| {
            index_answer_surface_bucket_rank(right)
                .total_cmp(&index_answer_surface_bucket_rank(left))
                .then_with(|| right.max_overlap.cmp(&left.max_overlap))
                .then_with(|| right.paths.len().cmp(&left.paths.len()))
                .then_with(|| left.answer_span.len().cmp(&right.answer_span.len()))
                .then_with(|| left.answer_span.cmp(&right.answer_span))
        });
        if compose_list_answer {
            if let Some((items, evidence)) =
                Self::compose_index_answer_surface_answer(task_lower, &query_profile, &buckets)
            {
                let answer = if matches!(
                    query_profile.expected_type,
                    SyntheticAnswerSurfaceExpectedType::Count
                ) {
                    items.len().to_string()
                } else {
                    Self::format_index_answer_surface_list(&items)
                };
                return self.write_synthetic_answer(
                    "answer-surface-compose",
                    task,
                    &answer,
                    &evidence,
                );
            }
        }
        let top = buckets.first()?;
        if let Some(next) = buckets.get(1) {
            if index_answer_surface_buckets_conflict(top, next)
                && !index_answer_surface_bucket_has_query_affinity(task_lower, top)
            {
                return None;
            }
        }
        if synthetic_answer_surface_should_skip_fallback(
            task,
            task_lower,
            &query_profile,
            &top.evidence,
        ) {
            return None;
        }
        let answer = format_index_answer_surface_answer(task_lower, &top.answer_span);
        self.write_synthetic_answer("answer-surface-fallback", task, &answer, &top.evidence)
    }

    pub fn synthetic_answer_surface_is_list_query(task_lower: &str) -> bool {
        task_lower.contains(" activities")
            || task_lower.contains(" books")
            || task_lower.contains(" events")
            || task_lower.contains(" fields")
            || task_lower.contains(" names")
            || task_lower.starts_with("where has ")
            || task_lower.starts_with("where have ")
            || task_lower.starts_with("what places")
            || task_lower.starts_with("which places")
            || task_lower.starts_with("in what ways")
            || task_lower.contains(" to destress")
            || task_lower.contains(" to de-stress")
            || task_lower.contains("self-care")
    }

    pub fn synthetic_answer_surface_target_items(task_lower: &str) -> usize {
        if task_lower.contains(" activities") {
            6
        } else if task_lower.starts_with("where has ") || task_lower.starts_with("where have ") {
            4
        } else if task_lower.contains(" names") {
            4
        } else if task_lower.contains(" books") {
            4
        } else if task_lower.contains(" events") || task_lower.starts_with("in what ways") {
            4
        } else {
            3
        }
    }

    pub(crate) fn compose_index_answer_surface_answer(
        task_lower: &str,
        profile: &SyntheticAnswerSurfaceQueryProfile,
        buckets: &[IndexAnswerSurfaceBucket],
    ) -> Option<(Vec<String>, Vec<String>)> {
        if buckets.is_empty() || !Self::synthetic_answer_surface_is_list_query(task_lower) {
            return None;
        }

        let mut ranked = buckets
            .iter()
            .filter(|bucket| {
                synthetic_answer_surface_bucket_matches_relation_profile(profile, bucket)
            })
            .cloned()
            .collect::<Vec<_>>();
        if ranked.is_empty() {
            return None;
        }
        ranked.sort_by(|left, right| {
            Self::index_answer_surface_composition_rank(right)
                .total_cmp(&Self::index_answer_surface_composition_rank(left))
                .then_with(|| {
                    index_answer_surface_bucket_rank(right)
                        .total_cmp(&index_answer_surface_bucket_rank(left))
                })
                .then_with(|| right.max_overlap.cmp(&left.max_overlap))
                .then_with(|| left.answer_span.cmp(&right.answer_span))
        });

        let top_rank = Self::index_answer_surface_composition_rank(ranked.first()?);
        let counting_query = matches!(
            profile.expected_type,
            SyntheticAnswerSurfaceExpectedType::Count
        );
        let target_items = if counting_query {
            usize::max(Self::synthetic_answer_surface_target_items(task_lower), 8)
        } else {
            Self::synthetic_answer_surface_target_items(task_lower)
        };
        let min_items = if counting_query { 1 } else { 2 };
        let margin = if task_lower.contains(" activities") {
            10.0
        } else {
            8.0
        };

        let mut chosen = Vec::new();
        let mut evidence = Vec::new();
        let mut seen_keys = HashSet::new();
        let mut seen_paths = HashSet::new();

        'passes: for prefer_new_path in [true, false] {
            for bucket in &ranked {
                if Self::index_answer_surface_composition_rank(bucket) + margin < top_rank {
                    break;
                }
                if !Self::is_composeable_index_answer_surface_bucket(bucket) {
                    continue;
                }
                if prefer_new_path && bucket.paths.iter().all(|path| seen_paths.contains(path)) {
                    continue;
                }

                let mut added_any = false;
                for item in Self::split_index_answer_surface_items(&bucket.answer_span) {
                    let key = normalized_index_answer_surface_key(&item);
                    if key.is_empty()
                        || !seen_keys.insert(key)
                        || chosen.iter().any(|existing: &String| {
                            index_answer_surface_answers_overlap(existing, &item)
                        })
                    {
                        continue;
                    }
                    chosen.push(item);
                    added_any = true;
                    if chosen.len() >= target_items {
                        break;
                    }
                }

                if added_any {
                    for path in &bucket.paths {
                        seen_paths.insert(path.clone());
                    }
                    for line in &bucket.evidence {
                        if evidence.len() >= 6 {
                            break;
                        }
                        if !evidence.iter().any(|existing| existing == line) {
                            evidence.push(line.clone());
                        }
                    }
                }

                if chosen.len() >= target_items {
                    break 'passes;
                }
            }
        }

        (chosen.len() >= min_items).then_some((chosen, evidence))
    }

    pub(crate) fn index_answer_surface_composition_rank(bucket: &IndexAnswerSurfaceBucket) -> f32 {
        bucket.best_score
            + bucket.max_overlap as f32 * 2.0
            + (bucket.paths.len().saturating_sub(1) as f32) * 1.5
    }

    pub(crate) fn is_composeable_index_answer_surface_bucket(
        bucket: &IndexAnswerSurfaceBucket,
    ) -> bool {
        let word_count = bucket.answer_span.split_whitespace().count();
        word_count > 0
            && word_count <= 8
            && !bucket.answer_span.contains('?')
            && !bucket.answer_span.contains(". ")
            && !bucket.answer_span.contains(" because ")
    }

    pub fn split_index_answer_surface_items(text: &str) -> Vec<String> {
        let clean = text
            .trim()
            .replace(", and ", ", ")
            .replace(" and ", ", ")
            .replace(" or ", ", ");
        let parts = clean
            .split(',')
            .map(str::trim)
            .map(|part| {
                part.trim_matches(|c: char| {
                    matches!(c, '"' | '\'' | ',' | '.' | ';' | ':' | '!' | '?')
                })
                .to_string()
            })
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>();
        if parts.len() >= 2 {
            parts
        } else {
            vec![clean.trim().to_string()]
        }
    }

    pub fn format_index_answer_surface_list(items: &[String]) -> String {
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

    pub fn synthetic_kg_personal_fact_answer(&self, task: &str) -> Option<PathBuf> {
        let predicate = detect_personal_fact_query(task)?;
        let task_lower = task.to_ascii_lowercase();
        if predicate == "rare_items_total" {
            return None;
        }
        if predicate == "instagram_followers"
            && task_contains_any(
                &task_lower,
                &[
                    "increase",
                    "increased",
                    "gain",
                    "gained",
                    "difference",
                    "grew",
                ],
            )
        {
            return None;
        }
        if let Some(path) =
            self.synthetic_session_personal_fact_answer(task, &task_lower, predicate)
        {
            return Some(path);
        }
        let entity = detect_personal_fact_entity(task)?;
        let kg_path = kg::kg_neuron_path(&self.project_root, &entity);
        let kg_entity = kg::KgEntity::load(&kg_path).ok()?;
        let answer = latest_active_kg_value(&kg_entity, predicate)?;
        self.write_synthetic_answer(
            &format!("kg-{}", predicate.replace('_', "-")),
            task,
            &answer,
            &[format!("kg: {entity}.{predicate} = {answer}")],
        )
    }
}
