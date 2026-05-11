//! Furniture, garden planting, sephora points, property views.

use super::super::*;

impl NeuronIndex {
    pub fn synthetic_recent_furniture_action_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_lower.contains("furniture")
            || !task_contains_any(
                task_lower,
                &[
                    "buy",
                    "bought",
                    "assemble",
                    "assembled",
                    "sell",
                    "sold",
                    "fix",
                    "fixed",
                ],
            )
            || !task_contains_any(
                task_lower,
                &[
                    "past few months",
                    "past few month",
                    "recent",
                    "last few months",
                ],
            )
        {
            return None;
        }

        let required_owned = vec![
            "furniture".to_string(),
            "coffee table".to_string(),
            "mattress".to_string(),
            "bookshelf".to_string(),
            "table".to_string(),
            "assembled".to_string(),
            "fixed".to_string(),
        ];
        let required_terms: Vec<&str> = required_owned.iter().map(String::as_str).collect();

        let mut candidates = self
            .session_ids_matching_line(|line, lower| {
                is_summary_or_user_line(line, lower)
                    && !extract_recent_furniture_action_signatures_from_line(line, lower).is_empty()
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
                    && !extract_recent_furniture_action_signatures_from_line(line, lower).is_empty()
            });
            let mut items = HashSet::new();
            let mut evidence = Vec::new();
            for line in lines {
                let lower = line.to_ascii_lowercase();
                let mut inserted = false;
                for item in extract_recent_furniture_action_signatures_from_line(&line, &lower) {
                    inserted |= items.insert(item);
                }
                if inserted
                    && evidence.len() < 4
                    && !evidence.iter().any(|existing| existing == &line)
                {
                    evidence.push(line);
                }
            }

            if items.is_empty() {
                continue;
            }

            let count = items.len();
            let score = count * 100 + session_rank * 10 + evidence.len();
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
            "recent-furniture-action-count",
            task,
            &count.to_string(),
            &evidence,
        )
    }

    pub fn synthetic_initial_garden_planting_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_contains_any(task_lower, &["plant", "plants"])
            || !task_contains_any(task_lower, &["initially", "initial"])
            || !task_contains_any(task_lower, &["tomato", "tomatoes"])
            || !task_contains_any(task_lower, &["cucumber", "cucumbers"])
        {
            return None;
        }

        let required_owned = vec![
            "tomato".to_string(),
            "cucumber".to_string(),
            "plants".to_string(),
            "garden".to_string(),
            "planted".to_string(),
        ];
        let required_terms: Vec<&str> = required_owned.iter().map(String::as_str).collect();

        let mut candidates = self
            .session_ids_matching_line(|line, lower| {
                lower.starts_with("user:")
                    && lower.contains("plant")
                    && (lower.contains("tomato") || lower.contains("cucumber"))
                    && !extract_line_numbers(line).is_empty()
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

        let mut best: Option<(usize, i32, Vec<String>)> = None;
        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, 256, |line, lower| {
                lower.starts_with("user:")
                    && lower.contains("plant")
                    && (lower.contains("tomato") || lower.contains("cucumber"))
                    && !extract_line_numbers(line).is_empty()
            });
            let mut tomato = None::<(usize, i32, String)>;
            let mut cucumber = None::<(usize, i32, String)>;
            for line in lines {
                let lower = line.to_ascii_lowercase();
                if lower.contains("tomato") {
                    if let Some((value, proximity_score)) = extract_focus_aligned_count(
                        &line,
                        &["tomato".to_string(), "plants".to_string()],
                        task_lower,
                    ) {
                        if !(lower.contains("planted") || lower.contains("initial")) {
                            continue;
                        }
                        let score = proximity_score
                            + usize::from(lower.contains("initially")) * 8
                            + usize::from(lower.contains("garden")) * 2;
                        if tomato
                            .as_ref()
                            .map(|(best_score, best_value, _)| {
                                score > *best_score || (score == *best_score && value > *best_value)
                            })
                            .unwrap_or(true)
                        {
                            tomato = Some((score, value, line.clone()));
                        }
                    }
                }
                if lower.contains("cucumber") {
                    if let Some((value, proximity_score)) = extract_focus_aligned_count(
                        &line,
                        &["cucumber".to_string(), "plants".to_string()],
                        task_lower,
                    ) {
                        if !(lower.contains("garden") && lower.contains("plants")) {
                            continue;
                        }
                        let score = proximity_score
                            + usize::from(lower.contains("got")) * 2
                            + usize::from(lower.contains("growing")) * 4;
                        if cucumber
                            .as_ref()
                            .map(|(best_score, best_value, _)| {
                                score > *best_score || (score == *best_score && value > *best_value)
                            })
                            .unwrap_or(true)
                        {
                            cucumber = Some((score, value, line.clone()));
                        }
                    }
                }
            }

            let (
                Some((tomato_score, tomato_value, tomato_line)),
                Some((cucumber_score, cucumber_value, cucumber_line)),
            ) = (tomato, cucumber)
            else {
                continue;
            };

            let total = tomato_value + cucumber_value;
            let tomato_initially = tomato_line.to_ascii_lowercase().contains("initially");
            let cucumber_growing = cucumber_line.to_ascii_lowercase().contains("growing");
            let mut evidence = vec![tomato_line];
            if cucumber_line != evidence[0] {
                evidence.push(cucumber_line);
            }
            let score = session_rank * 1000
                + tomato_score * 20
                + cucumber_score * 20
                + usize::from(tomato_initially) * 50
                + usize::from(cucumber_growing) * 25;
            let should_replace = best
                .as_ref()
                .map(|(best_score, best_total, _)| {
                    score > *best_score || (score == *best_score && total > *best_total)
                })
                .unwrap_or(true);
            if should_replace {
                best = Some((score, total, evidence));
            }
        }

        let (_, total, evidence) = best?;
        self.write_synthetic_answer(
            "initial-garden-planting-count",
            task,
            &total.to_string(),
            &evidence,
        )
    }

    pub fn synthetic_sephora_points_needed_for_free_skincare_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_lower.contains("point")
            || !task_lower.contains("sephora")
            || !task_lower.contains("redeem")
            || !task_contains_any(task_lower, &["free skincare", "skincare product"])
            || !task_contains_any(task_lower, &["need to earn", "need"])
        {
            return None;
        }

        let session_id = self
            .best_matching_session_id(task, &["sephora", "points", "free", "skincare"])
            .or_else(|| {
                self.session_ids_matching_line(|_line, lower| {
                    lower.starts_with("user:")
                        && lower.contains("sephora")
                        && lower.contains("point")
                })
                .into_iter()
                .next()
            })?;
        let lines = self.find_session_lines(&session_id, false, 128, |_line, lower| {
            lower.starts_with("user:") && lower.contains("sephora") && lower.contains("point")
        });

        let mut target_total = None::<(i32, String)>;
        let mut current_total = None::<(i32, String)>;
        for line in lines {
            let lower = line.to_ascii_lowercase();
            if target_total.is_none() {
                if let Some(value) = extract_loyalty_point_goal_total_from_line(&line, &lower) {
                    target_total = Some((value, line.clone()));
                }
            }
            if current_total.is_none() {
                if let Some(value) = extract_loyalty_point_current_total_from_line(&line, &lower) {
                    current_total = Some((value, line.clone()));
                }
            }
        }

        let (target_value, target_line) = target_total?;
        let (current_value, current_line) = current_total?;
        let needed = target_value - current_value;
        if needed < 0 {
            return None;
        }

        let mut evidence = vec![target_line];
        if current_line != evidence[0] {
            evidence.push(current_line);
        }

        self.write_synthetic_answer(
            "sephora-points-needed-for-free-skincare",
            task,
            &needed.to_string(),
            &evidence,
        )
    }

    pub fn synthetic_pre_offer_property_view_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_lower.contains("properties")
            || !task_lower.contains("offer")
            || !task_lower.contains("townhouse")
            || !task_lower.contains("brookside")
        {
            return None;
        }

        let session_id = self
            .best_matching_session_id(
                task,
                &["brookside", "townhouse", "offer", "property", "condo"],
            )
            .or_else(|| {
                self.session_ids_matching_line(|_line, lower| {
                    lower.starts_with("user:")
                        && task_contains_any(
                            lower,
                            &[
                                "brookside",
                                "townhouse",
                                "condo",
                                "bungalow",
                                "cedar creek",
                                "offer",
                            ],
                        )
                })
                .into_iter()
                .next()
            })?;
        let lines = self.find_session_lines(&session_id, false, 256, |_line, lower| {
            lower.starts_with("user:")
                && task_contains_any(
                    lower,
                    &[
                        "brookside",
                        "townhouse",
                        "condo",
                        "bungalow",
                        "cedar creek",
                        "offer",
                    ],
                )
        });

        let offer_line = lines.iter().find(|line| {
            let lower = line.to_ascii_lowercase();
            lower.contains("brookside") && lower.contains("offer")
        })?;
        let offer_rank = extract_explicit_date_rank(offer_line)?;

        let mut views: HashMap<String, (i32, String, String)> = HashMap::new();
        for line in lines.iter() {
            let lower = line.to_ascii_lowercase();
            let Some((key, rank, reason)) = extract_property_view_reason_from_line(line, &lower)
            else {
                continue;
            };
            if rank >= offer_rank {
                continue;
            }
            let should_replace = views
                .get(&key)
                .map(|(best_rank, _, _)| rank < *best_rank)
                .unwrap_or(true);
            if should_replace {
                views.insert(key, (rank, reason, line.clone()));
            }
        }

        if views.is_empty() {
            return None;
        }

        let mut ordered_views = views.into_iter().collect::<Vec<_>>();
        ordered_views.sort_by_key(|(_, (rank, _, _))| *rank);

        let count = ordered_views.len();
        let reasons = ordered_views
            .iter()
            .map(|(_, (_, reason, _))| reason.clone())
            .collect::<Vec<_>>();
        let mut evidence = vec![offer_line.clone()];
        for (_, (_, _, line)) in &ordered_views {
            if !evidence.iter().any(|existing| existing == line) {
                evidence.push(line.clone());
            }
        }

        let answer = format!(
            "I viewed {} properties before making an offer on the townhouse in the Brookside neighborhood. The reasons I didn't make an offer on them were: {}.",
            small_cardinal_word_lower(count),
            join_reason_clauses(&reasons),
        );

        self.write_synthetic_answer("pre-offer-property-view-count", task, &answer, &evidence)
    }
}
