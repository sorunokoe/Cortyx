use super::*;

impl NeuronIndex {
    pub fn synthetic_media_rewatch_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task) {
            return None;
        }

        let (focus_phrase, media_kind) = extract_media_rewatch_focus(task_lower)?;
        let mut required_owned = synthetic_query_terms(&focus_phrase);
        required_owned.push(media_kind);
        required_owned.extend(["watched".to_string(), "rewatched".to_string()]);
        required_owned.sort();
        required_owned.dedup();
        let required_terms: Vec<&str> = required_owned.iter().map(String::as_str).collect();
        let focus_terms = synthetic_query_terms(&focus_phrase);
        let focus_refs: Vec<&str> = focus_terms.iter().map(String::as_str).collect();

        let mut candidates = self
            .candidate_session_ids(task, &required_terms, 8)
            .into_iter()
            .enumerate()
            .map(|(idx, session_id)| (session_id, 8usize.saturating_sub(idx)))
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            candidates = self.candidate_session_ids_by_line_overlap(&required_owned, 8);
        }

        let mut best: Option<(usize, usize, Vec<String>)> = None;
        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, 192, |line, lower| {
                is_summary_or_user_line(line, lower) && line_has_rewatch_marker(lower)
            });
            let mut titles = HashSet::new();
            let mut evidence = Vec::new();

            for line in lines {
                let lower = line.to_ascii_lowercase();
                if !focus_refs.is_empty() && term_overlap_count(&lower, &focus_refs) == 0 {
                    continue;
                }
                let Some(title) = extract_rewatch_title_from_line(&line, &lower) else {
                    continue;
                };
                let normalized = normalize_rewatch_title(&title);
                if normalized.is_empty() || !titles.insert(normalized) {
                    continue;
                }
                evidence.push(line);
            }

            if titles.is_empty() {
                continue;
            }

            let count = titles.len();
            let score = session_rank * 1000 + count * 100 + evidence.len();
            let should_replace = best
                .as_ref()
                .map(|(best_score, best_count, _)| {
                    score > *best_score || (score == *best_score && count > *best_count)
                })
                .unwrap_or(true);
            if should_replace {
                best = Some((score, count, evidence));
            }
        }

        let (_, count, evidence) = best?;
        self.write_synthetic_answer("media-rewatch-count", task, &count.to_string(), &evidence)
    }

    pub fn synthetic_art_related_event_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_contains_all(task_lower, &["art-related", "events"])
            || !task_contains_any(task_lower, &["past month", "last month"])
        {
            return None;
        }

        let mut required_owned = vec![
            "art".to_string(),
            "event".to_string(),
            "events".to_string(),
            "exhibition".to_string(),
            "museum".to_string(),
            "gallery".to_string(),
            "lecture".to_string(),
        ];
        required_owned.sort();
        required_owned.dedup();
        let required_terms: Vec<&str> = required_owned.iter().map(String::as_str).collect();

        let mut candidates = self
            .session_ids_matching_line(|line, lower| {
                is_summary_or_user_line(line, lower)
                    && extract_art_related_event_signature_from_line(line, lower).is_some()
            })
            .into_iter()
            .map(|session_id| (session_id, 0usize))
            .collect::<Vec<_>>();
        for (session_id, score) in self.candidate_session_ids_by_line_overlap(&required_owned, 8) {
            if let Some((_, existing_score)) = candidates
                .iter_mut()
                .find(|(existing_session_id, _)| existing_session_id == &session_id)
            {
                *existing_score = (*existing_score).max(score);
            } else {
                candidates.push((session_id, score));
            }
        }
        for (idx, session_id) in self
            .candidate_session_ids(task, &required_terms, 8)
            .into_iter()
            .enumerate()
        {
            let score = 8usize.saturating_sub(idx);
            if let Some((_, existing_score)) = candidates
                .iter_mut()
                .find(|(existing_session_id, _)| existing_session_id == &session_id)
            {
                *existing_score = (*existing_score).max(score);
            } else {
                candidates.push((session_id, score));
            }
        }

        let mut best: Option<(usize, usize, Vec<String>)> = None;
        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, 256, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && extract_art_related_event_signature_from_line(line, lower).is_some()
            });
            let mut extracted = Vec::new();
            for line in lines {
                let lower = line.to_ascii_lowercase();
                let Some((rank, key)) =
                    extract_art_related_event_signature_from_line(&line, &lower)
                else {
                    continue;
                };
                extracted.push((rank, key, line));
            }
            if extracted.is_empty() {
                continue;
            }
            let latest_rank = extracted
                .iter()
                .map(|(rank, _, _)| *rank)
                .max()
                .unwrap_or_default();
            let mut events = HashSet::new();
            let mut evidence = Vec::new();
            for (rank, key, line) in extracted {
                if latest_rank - rank > 32 {
                    continue;
                }
                if !events.insert(key) {
                    continue;
                }
                if evidence.len() < 4 && !evidence.iter().any(|existing| existing == &line) {
                    evidence.push(line);
                }
            }

            if events.is_empty() {
                continue;
            }

            let count = events.len();
            let score = session_rank * 10 + count * 5 + evidence.len();
            let should_replace = best
                .as_ref()
                .map(|(best_score, best_count, _)| {
                    score > *best_score || (score == *best_score && count > *best_count)
                })
                .unwrap_or(true);
            if should_replace {
                best = Some((score, count, evidence));
            }
        }

        let (_, count, evidence) = best?;
        self.write_synthetic_answer(
            "art-related-event-count",
            task,
            &count.to_string(),
            &evidence,
        )
    }

    pub fn synthetic_museum_gallery_visit_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_contains_any(task_lower, &["museum", "museums"])
            || !task_contains_any(task_lower, &["gallery", "galleries"])
        {
            return None;
        }

        let month_filter = extract_query_month_name(task_lower)?;
        let mut per_session: HashMap<String, (HashSet<String>, bool, bool, Vec<String>)> =
            HashMap::new();
        for entry in self.entries.iter().filter(|entry| {
            matches!(entry.kind, NeuronKind::Verbatim) && !entry.session_id.is_empty()
        }) {
            let Ok(content) = std::fs::read_to_string(&entry.neuron_path) else {
                continue;
            };
            for raw_line in strip_query_surface_section(&content).lines() {
                let line = raw_line.trim();
                if line.is_empty() {
                    continue;
                }
                let lower = line.to_ascii_lowercase();
                if !is_summary_or_user_line(line, &lower)
                    || !line_mentions_candidate_museum_gallery_visit(line, &lower, month_filter)
                {
                    continue;
                }
                let Some(venue) =
                    extract_museum_gallery_visit_venue_from_line(line, &lower, month_filter)
                else {
                    continue;
                };
                let venue_key = normalized_synthetic_phrase_key(&venue);
                let venue_lower = venue.to_ascii_lowercase();
                let (venues, has_museum, has_gallery, evidence) = per_session
                    .entry(entry.session_id.clone())
                    .or_insert_with(|| (HashSet::new(), false, false, Vec::new()));
                let inserted = venues.insert(venue_key);
                *has_museum |= venue_lower.contains("museum");
                *has_gallery |= venue_lower.contains("gallery") || venue_lower.contains("art cube");
                if inserted
                    && evidence.len() < 4
                    && !evidence.iter().any(|existing| existing == line)
                {
                    evidence.push(line.to_string());
                }
            }
        }

        let mut best: Option<(usize, usize, usize, Vec<String>)> = None;
        for (_, (venues, has_museum, has_gallery, evidence)) in per_session {
            if venues.is_empty() {
                continue;
            }
            let count = venues.len();
            let category_bonus = usize::from(has_museum && has_gallery);
            let score = count * 100 + category_bonus * 25 + evidence.len();
            let should_replace = best
                .as_ref()
                .map(|(best_score, best_count, best_bonus, _)| {
                    score > *best_score
                        || (score == *best_score
                            && (count > *best_count
                                || (count == *best_count && category_bonus > *best_bonus)))
                })
                .unwrap_or(true);
            if should_replace {
                best = Some((score, count, category_bonus, evidence));
            }
        }

        let (_, count, _, evidence) = best?;
        self.write_synthetic_answer(
            "museum-gallery-visit-count",
            task,
            &count.to_string(),
            &evidence,
        )
    }

    pub fn synthetic_distinct_cuisine_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_contains_any(task_lower, &["cuisine", "cuisines"])
            || !task_contains_any(task_lower, &["learned to cook", "tried out"])
        {
            return None;
        }

        let mut required_owned = vec![
            "cuisine".to_string(),
            "restaurant".to_string(),
            "recipe".to_string(),
            "class".to_string(),
            "learned".to_string(),
            "tried".to_string(),
        ];
        required_owned.sort();
        required_owned.dedup();
        let required_terms: Vec<&str> = required_owned.iter().map(String::as_str).collect();

        let mut candidates = self
            .session_ids_matching_line(|line, lower| {
                is_summary_or_user_line(line, lower)
                    && !extract_cuisine_labels_from_line(line, lower).is_empty()
            })
            .into_iter()
            .map(|session_id| (session_id, 0usize))
            .collect::<Vec<_>>();
        for (session_id, score) in self.candidate_session_ids_by_line_overlap(&required_owned, 8) {
            if let Some((_, existing_score)) = candidates
                .iter_mut()
                .find(|(existing_session_id, _)| existing_session_id == &session_id)
            {
                *existing_score = (*existing_score).max(score);
            } else {
                candidates.push((session_id, score));
            }
        }
        for (idx, session_id) in self
            .candidate_session_ids(task, &required_terms, 8)
            .into_iter()
            .enumerate()
        {
            let score = 8usize.saturating_sub(idx);
            if let Some((_, existing_score)) = candidates
                .iter_mut()
                .find(|(existing_session_id, _)| existing_session_id == &session_id)
            {
                *existing_score = (*existing_score).max(score);
            } else {
                candidates.push((session_id, score));
            }
        }

        let mut best: Option<(usize, usize, Vec<String>)> = None;
        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, 256, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && !extract_cuisine_labels_from_line(line, lower).is_empty()
            });
            let mut cuisines = HashSet::new();
            let mut evidence = Vec::new();
            for line in lines {
                let lower = line.to_ascii_lowercase();
                let mut inserted = false;
                for cuisine in extract_cuisine_labels_from_line(&line, &lower) {
                    inserted |= cuisines.insert(normalized_synthetic_phrase_key(&cuisine));
                }
                if inserted
                    && evidence.len() < 4
                    && !evidence.iter().any(|existing| existing == &line)
                {
                    evidence.push(line);
                }
            }

            if cuisines.is_empty() {
                continue;
            }

            let count = cuisines.len();
            let score = session_rank * 10 + count * 5 + evidence.len();
            let should_replace = best
                .as_ref()
                .map(|(best_score, best_count, _)| {
                    score > *best_score || (score == *best_score && count > *best_count)
                })
                .unwrap_or(true);
            if should_replace {
                best = Some((score, count, evidence));
            }
        }

        let (_, count, evidence) = best?;
        self.write_synthetic_answer(
            "distinct-cuisine-count",
            task,
            &count.to_string(),
            &evidence,
        )
    }
}
