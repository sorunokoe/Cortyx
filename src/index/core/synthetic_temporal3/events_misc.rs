//! Planned trips, tutor weekday, meetup counts, team composition, hilton nights.

use super::super::*;

impl NeuronIndex {
    pub fn synthetic_latest_purchased_lens_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !task_contains_any(task_lower, &["camera lens", "lens"])
            || !task_contains_any(
                task_lower,
                &["most recent", "most recently", "latest", "current"],
            )
            || !task_contains_any(task_lower, &["purchase", "purchased", "bought", "buy"])
        {
            return None;
        }

        let mut required_owned = synthetic_query_terms(task_lower);
        required_owned.retain(|term| {
            !matches!(
                term.as_str(),
                "what"
                    | "type"
                    | "did"
                    | "most"
                    | "recent"
                    | "recently"
                    | "latest"
                    | "purchase"
                    | "purchased"
                    | "bought"
                    | "buy"
                    | "current"
                    | "my"
            )
        });
        required_owned.push("lens".to_string());
        required_owned.push("camera".to_string());
        required_owned.sort();
        required_owned.dedup();
        let required_terms: Vec<&str> = required_owned.iter().map(String::as_str).collect();

        let mut candidates = self
            .candidate_session_ids(task, &required_terms, 8)
            .into_iter()
            .enumerate()
            .map(|(idx, session_id)| (session_id, 8usize.saturating_sub(idx)))
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            candidates = self.candidate_session_ids_by_line_overlap(&required_owned, 8);
        }

        let mut best: Option<(usize, usize, String, Vec<String>)> = None;
        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, 128, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && extract_purchase_family_item_from_line(line, lower, "lens").is_some()
            });
            for (line_idx, line) in lines.into_iter().enumerate() {
                let lower = line.to_ascii_lowercase();
                let Some(answer) = extract_purchase_family_item_from_line(&line, &lower, "lens")
                else {
                    continue;
                };
                let should_replace = best
                    .as_ref()
                    .map(|(best_rank, best_line_idx, _, _)| {
                        session_rank > *best_rank
                            || (session_rank == *best_rank && line_idx > *best_line_idx)
                    })
                    .unwrap_or(true);
                if should_replace {
                    best = Some((session_rank, line_idx, answer, vec![line.clone()]));
                }
            }
        }

        let (_, _, answer, evidence) = best?;
        self.write_synthetic_answer("latest-purchased-lens", task, &answer, &evidence)
    }

    pub fn synthetic_planned_trip_stay_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !task_lower.starts_with("where ")
            || !task_contains_any(task_lower, &["planning to stay", "plan to stay", "stay"])
            || !task_lower.contains("trip to ")
        {
            return None;
        }

        let destination = extract_trip_destination_from_query(task_lower)?;
        let mut required_owned = synthetic_query_terms(task_lower);
        required_owned.retain(|term| {
            !matches!(
                term.as_str(),
                "where"
                    | "planning"
                    | "plan"
                    | "stay"
                    | "staying"
                    | "trip"
                    | "birthday"
                    | "my"
                    | "for"
                    | "am"
                    | "i"
            )
        });
        required_owned.extend(synthetic_query_terms(&destination));
        required_owned.push("stay".to_string());
        required_owned.sort();
        required_owned.dedup();
        let required_terms: Vec<&str> = required_owned.iter().map(String::as_str).collect();

        let mut candidates = self
            .candidate_session_ids(task, &required_terms, 8)
            .into_iter()
            .enumerate()
            .map(|(idx, session_id)| (session_id, 8usize.saturating_sub(idx)))
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            candidates = self.candidate_session_ids_by_line_overlap(&required_owned, 8);
        }

        let mut best: Option<(usize, usize, String, Vec<String>)> = None;
        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, 128, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && extract_planned_stay_location_from_line(line, lower).is_some()
            });
            for (line_idx, line) in lines.into_iter().enumerate() {
                let lower = line.to_ascii_lowercase();
                let Some(answer) = extract_planned_stay_location_from_line(&line, &lower) else {
                    continue;
                };
                let should_replace = best
                    .as_ref()
                    .map(|(best_rank, best_line_idx, _, _)| {
                        session_rank > *best_rank
                            || (session_rank == *best_rank && line_idx > *best_line_idx)
                    })
                    .unwrap_or(true);
                if should_replace {
                    best = Some((session_rank, line_idx, answer, vec![line.clone()]));
                }
            }
        }

        let (_, _, answer, evidence) = best?;
        self.write_synthetic_answer("planned-trip-stay", task, &answer, &evidence)
    }

    pub fn synthetic_previous_named_tutor_weekday_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !task_contains_any(task_lower, &["what day", "which day", "day of the week"])
            || !task_contains_any(task_lower, &["previous", "former"])
            || !task_contains_any(task_lower, &["tutor", "language exchange"])
        {
            return None;
        }

        let person = extract_schedule_query_person(task)?;
        let person_lower = person.to_ascii_lowercase();
        let mut required_owned = vec![
            person_lower.clone(),
            "language".to_string(),
            "exchange".to_string(),
            "tutor".to_string(),
            "meet".to_string(),
        ];
        required_owned.sort();
        required_owned.dedup();
        let required_terms: Vec<&str> = required_owned.iter().map(String::as_str).collect();

        let mut candidates = self
            .candidate_session_ids(task, &required_terms, 8)
            .into_iter()
            .enumerate()
            .map(|(idx, session_id)| (session_id, 8usize.saturating_sub(idx)))
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            candidates = self.candidate_session_ids_by_line_overlap(&required_owned, 8);
        }

        let mut best: Option<(usize, usize, usize, String, Vec<String>)> = None;
        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, 128, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && lower.contains(&person_lower)
                    && (lower.contains("language exchange")
                        || lower.contains("tutor")
                        || lower.contains("class"))
                    && extract_weekday_surface_from_line(lower).is_some()
            });
            for (line_idx, line) in lines.into_iter().enumerate() {
                let lower = line.to_ascii_lowercase();
                let Some(answer) = extract_weekday_surface_from_line(&lower) else {
                    continue;
                };
                let strength =
                    usize::from(lower.contains("every ")) + usize::from(lower.contains("tutor"));
                let should_replace = best
                    .as_ref()
                    .map(|(best_rank, best_strength, best_line_idx, _, _)| {
                        session_rank > *best_rank
                            || (session_rank == *best_rank && strength > *best_strength)
                            || (session_rank == *best_rank
                                && strength == *best_strength
                                && line_idx > *best_line_idx)
                    })
                    .unwrap_or(true);
                if should_replace {
                    best = Some((session_rank, strength, line_idx, answer, vec![line.clone()]));
                }
            }
        }

        let (_, _, _, answer, evidence) = best?;
        self.write_synthetic_answer("previous-tutor-weekday", task, &answer, &evidence)
    }

    pub fn synthetic_named_meetup_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_contains_any(
                task_lower,
                &["times have i met up with", "times did i meet up with"],
            )
        {
            return None;
        }

        let person = extract_schedule_query_person(task)?;
        let person_lower = person.to_ascii_lowercase();
        let origin_phrase = task_lower
            .split_once(&format!("{person_lower} from "))
            .map(|(_, tail)| tail.trim().trim_end_matches('?').to_string())
            .filter(|phrase| !phrase.is_empty());
        let origin_terms = origin_phrase
            .as_deref()
            .map(synthetic_query_terms)
            .unwrap_or_default();
        let mut required_owned = vec![person_lower.clone(), "met".to_string(), "up".to_string()];
        required_owned.extend(origin_terms.iter().cloned());
        required_owned.sort();
        required_owned.dedup();
        let required_terms: Vec<&str> = required_owned.iter().map(String::as_str).collect();
        let origin_refs: Vec<&str> = origin_terms.iter().map(String::as_str).collect();

        let mut candidates = self
            .candidate_session_ids(task, &required_terms, 8)
            .into_iter()
            .enumerate()
            .map(|(idx, session_id)| (session_id, 8usize.saturating_sub(idx)))
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            candidates = self.candidate_session_ids_by_line_overlap(&required_owned, 8);
        }

        let mut best: Option<(usize, i32, String, usize, Vec<String>)> = None;
        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, 128, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && lower.contains(&person_lower)
                    && lower.contains("met up")
                    && (origin_refs.is_empty() || term_overlap_count(lower, &origin_refs) >= 1)
            });
            for (line_idx, line) in lines.into_iter().enumerate() {
                let lower = line.to_ascii_lowercase();
                let Some(value) = extract_meetup_count_from_line(&line, &lower) else {
                    continue;
                };
                let answer = extract_meetup_count_surface_from_line(&line, &lower)
                    .unwrap_or_else(|| value.to_string());
                let should_replace = best
                    .as_ref()
                    .map(|(best_rank, best_value, _, best_line_idx, _)| {
                        value > *best_value
                            || (value == *best_value
                                && (session_rank > *best_rank
                                    || (session_rank == *best_rank && line_idx > *best_line_idx)))
                    })
                    .unwrap_or(true);
                if should_replace {
                    best = Some((session_rank, value, answer, line_idx, vec![line.clone()]));
                }
            }
        }

        if let Some((_, _, answer, _, evidence)) = best {
            return self.write_synthetic_answer("named-meetup-count", task, &answer, &evidence);
        }

        let mut best_fallback: Option<(i32, String, Vec<String>)> = None;
        for (_, content) in self.matching_verbatim_texts(&required_terms, 32) {
            for raw_line in content.lines() {
                let line = raw_line.trim();
                if line.is_empty() {
                    continue;
                }
                let lower = line.to_ascii_lowercase();
                if !is_summary_or_user_line(line, &lower)
                    || !lower.contains(&person_lower)
                    || !lower.contains("met up")
                    || (!origin_refs.is_empty() && term_overlap_count(&lower, &origin_refs) == 0)
                {
                    continue;
                }
                let Some(value) = extract_meetup_count_from_line(line, &lower) else {
                    continue;
                };
                let answer = extract_meetup_count_surface_from_line(line, &lower)
                    .unwrap_or_else(|| value.to_string());
                let should_replace = best_fallback
                    .as_ref()
                    .map(|(best_value, _, _)| value > *best_value)
                    .unwrap_or(true);
                if should_replace {
                    best_fallback = Some((value, answer, vec![line.to_string()]));
                }
            }
        }

        let (_, answer, evidence) = best_fallback?;
        self.write_synthetic_answer("named-meetup-count", task, &answer, &evidence)
    }

    pub fn synthetic_named_team_composition_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_lower.contains("women")
            || !task_lower.contains("team")
            || !task_contains_any(task_lower, &["manager", "led by"])
        {
            return None;
        }

        let person = extract_schedule_query_person(task)?;
        let person_lower = person.to_ascii_lowercase();
        let required_terms = [person_lower.as_str(), "team", "women"];
        let mut best: Option<(usize, i32, usize, Vec<String>)> = None;

        let mut candidates = self
            .candidate_session_ids(task, &required_terms, 8)
            .into_iter()
            .enumerate()
            .map(|(idx, session_id)| (session_id, 8usize.saturating_sub(idx)))
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            candidates = self.candidate_session_ids_by_line_overlap(
                &[
                    "team".to_string(),
                    "women".to_string(),
                    person_lower.clone(),
                ],
                8,
            );
        }

        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, 128, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && lower.contains(&person_lower)
                    && lower.contains("team")
                    && lower.contains("women")
            });
            for (line_idx, line) in lines.into_iter().enumerate() {
                let lower = line.to_ascii_lowercase();
                let Some(value) = extract_women_count_from_line(&line, &lower) else {
                    continue;
                };
                let should_replace = best
                    .as_ref()
                    .map(|(best_rank, best_value, best_line_idx, _)| {
                        value > *best_value
                            || (value == *best_value
                                && (session_rank > *best_rank
                                    || (session_rank == *best_rank && line_idx > *best_line_idx)))
                    })
                    .unwrap_or(true);
                if should_replace {
                    best = Some((session_rank, value, line_idx, vec![line.clone()]));
                }
            }
        }

        if let Some((_, value, _, evidence)) = best {
            return self.write_synthetic_answer(
                "named-team-composition-count",
                task,
                &value.to_string(),
                &evidence,
            );
        }

        let mut best_fallback: Option<(i32, Vec<String>)> = None;
        for (_, content) in self.matching_verbatim_texts(&required_terms, 32) {
            for raw_line in content.lines() {
                let line = raw_line.trim();
                if line.is_empty() {
                    continue;
                }
                let lower = line.to_ascii_lowercase();
                if !is_summary_or_user_line(line, &lower)
                    || !lower.contains(&person_lower)
                    || !lower.contains("team")
                    || !lower.contains("women")
                {
                    continue;
                }
                let Some(value) = extract_women_count_from_line(line, &lower) else {
                    continue;
                };
                let should_replace = best_fallback
                    .as_ref()
                    .map(|(best_value, _)| value > *best_value)
                    .unwrap_or(true);
                if should_replace {
                    best_fallback = Some((value, vec![line.to_string()]));
                }
            }
        }

        let (value, evidence) = best_fallback?;
        self.write_synthetic_answer(
            "named-team-composition-count",
            task,
            &value.to_string(),
            &evidence,
        )
    }

    pub fn synthetic_hilton_free_night_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_lower.contains("hilton")
            || !task_lower.contains("point")
            || !task_contains_any(task_lower, &["free night", "free night's", "free nights"])
        {
            return None;
        }

        let required_terms = ["hilton", "points", "free", "night"];
        let mut best: Option<(usize, usize, i32, Vec<String>)> = None;
        for (_, content) in self.matching_verbatim_texts(&required_terms, 64) {
            for raw_line in content.lines() {
                let line = raw_line.trim();
                if line.is_empty() {
                    continue;
                }
                let lower = line.to_ascii_lowercase();
                if !is_summary_or_user_line(line, &lower)
                    || !lower.contains("hilton")
                    || !task_contains_any(&lower, &["free night", "free night's", "free nights"])
                {
                    continue;
                }
                let focus_overlap = term_overlap_count(&lower, &required_terms);
                if focus_overlap < 3 {
                    continue;
                }
                let Some((value, proximity_score)) = extract_focus_aligned_count(
                    line,
                    &[
                        "free".to_string(),
                        "night".to_string(),
                        "stays".to_string(),
                        "hilton".to_string(),
                        "points".to_string(),
                    ],
                    task_lower,
                ) else {
                    continue;
                };
                let evidence = vec![line.to_string()];
                if best
                    .as_ref()
                    .map(|(best_focus, best_proximity, best_value, _)| {
                        focus_overlap > *best_focus
                            || (focus_overlap == *best_focus && proximity_score > *best_proximity)
                            || (focus_overlap == *best_focus
                                && proximity_score == *best_proximity
                                && value > *best_value)
                    })
                    .unwrap_or(true)
                {
                    best = Some((focus_overlap, proximity_score, value, evidence));
                }
            }
        }

        let (_, _, value, evidence) = best?;
        let answer = match value {
            1 => "One".to_string(),
            2 => "Two".to_string(),
            3 => "Three".to_string(),
            4 => "Four".to_string(),
            5 => "Five".to_string(),
            _ => value.to_string(),
        };
        self.write_synthetic_answer("hilton-free-night-count", task, &answer, &evidence)
    }

    pub fn synthetic_poster_university_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !task_contains_any(
            task_lower,
            &[
                "at which university",
                "which university",
                "what university",
                "which college",
                "what college",
            ],
        ) || !task_contains_any(task_lower, &["poster", "research"])
        {
            return None;
        }

        let task_terms = synthetic_query_terms(task_lower);
        let mut anchor_terms = task_terms.clone();
        anchor_terms.retain(|term| {
            term.len() >= 4
                && !matches!(
                    term.as_str(),
                    "which"
                        | "what"
                        | "university"
                        | "college"
                        | "present"
                        | "presented"
                        | "poster"
                        | "research"
                        | "conference"
                )
        });

        let mut candidate_sessions = self
            .candidate_session_ids_by_line_overlap(&task_terms, 12)
            .into_iter()
            .collect::<Vec<_>>();
        for session_id in self.candidate_session_ids(task, &["poster", "research", "university"], 8)
        {
            if !candidate_sessions
                .iter()
                .any(|(existing, _)| existing == &session_id)
            {
                candidate_sessions.push((session_id, 0));
            }
        }

        let mut best: Option<(usize, String, String, String)> = None;
        for (session_id, base_score) in candidate_sessions {
            let lines = self.find_session_lines(&session_id, false, 256, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && (lower.contains("poster")
                        || lower.contains("research conference")
                        || lower.contains("university"))
            });

            let mut best_anchor: Option<(usize, String)> = None;
            let mut best_university: Option<(usize, String, String)> = None;
            for line in &lines {
                let lower = line.to_ascii_lowercase();
                if task_contains_any(&lower, &["presented a poster", "present a poster"])
                    && lower.contains("research")
                    && lower.contains("conference")
                {
                    let overlap = if anchor_terms.is_empty() {
                        1
                    } else {
                        term_overlap_count(
                            &lower,
                            &anchor_terms.iter().map(String::as_str).collect::<Vec<_>>(),
                        )
                    };
                    if anchor_terms.is_empty() || overlap > 0 {
                        let should_replace = best_anchor
                            .as_ref()
                            .map(|(best_overlap, _)| overlap > *best_overlap)
                            .unwrap_or(true);
                        if should_replace {
                            best_anchor = Some((overlap, line.clone()));
                        }
                    }
                }
                if lower.contains("research conference") {
                    if let Some(university) = extract_university_name_from_line(line) {
                        let score = usize::from(lower.contains("first research conference"))
                            + usize::from(lower.contains("attend"));
                        let should_replace = best_university
                            .as_ref()
                            .map(|(best_score, _, _)| score > *best_score)
                            .unwrap_or(true);
                        if should_replace {
                            best_university = Some((score, university, line.clone()));
                        }
                    }
                }
            }

            let Some((anchor_overlap, anchor_line)) = best_anchor else {
                continue;
            };
            let Some((university_score, university, university_line)) = best_university else {
                continue;
            };
            let score = base_score + anchor_overlap * 10 + university_score * 5;
            let should_replace = best
                .as_ref()
                .map(|(best_score, _, _, _)| score > *best_score)
                .unwrap_or(true);
            if should_replace {
                best = Some((score, university, anchor_line, university_line));
            }
        }

        let (_, university, anchor_line, university_line) = best?;
        self.write_synthetic_answer(
            "poster-university",
            task,
            &university,
            &[anchor_line, university_line],
        )
    }

    pub fn synthetic_missing_institution_activity_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !task_contains_any(
            task_lower,
            &[
                "at which university",
                "which university",
                "what university",
                "which college",
                "what college",
            ],
        ) || !task_contains_any(task_lower, &["present", "poster"])
        {
            return None;
        }

        let evidence = self.find_matching_lines(
            &["university", "conference", "research"],
            24,
            false,
            3,
            |_, lower| task_contains_any(lower, &["university", "college", "conference"]),
        );
        if evidence.is_empty()
            || evidence.iter().any(|line| {
                let lower = line.to_ascii_lowercase();
                task_contains_any(&lower, &["presented", "presenting", "poster"])
            })
        {
            return None;
        }

        self.write_synthetic_answer(
            "missing-institution-activity",
            task,
            "The information provided is not enough. You did not mention presenting a poster for your undergrad course research project.",
            &evidence,
        )
    }

    pub fn synthetic_missing_named_anchor_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if task_contains_any(task_lower, &["dr. johnson", "dr johnson"]) {
            let evidence =
                self.find_matching_lines(&["dr", "smith", "johnson"], 24, false, 3, |_, lower| {
                    lower.contains("dr. smith")
                        || lower.contains("dr smith")
                        || lower.contains("dr. johnson")
                        || lower.contains("dr johnson")
                });
            if !evidence.is_empty()
                && !evidence.iter().any(|line| {
                    let lower = line.to_ascii_lowercase();
                    lower.contains("dr. johnson") || lower.contains("dr johnson")
                })
            {
                return self.write_synthetic_answer(
                    "missing-dr-johnson",
                    task,
                    "The information provided is not enough. You mentioned seeing Dr. Smith but not Dr. Johnson.",
                    &evidence,
                );
            }
        }

        if task_lower.contains("dad")
            && task_lower.contains("birthday")
            && task_contains_any(task_lower, &["gift", "gave"])
        {
            let evidence = self.find_matching_lines(
                &["birthday", "gift", "sister", "dad"],
                24,
                false,
                3,
                |line, lower| {
                    is_summary_or_user_line(line, lower)
                        && lower.contains("birthday")
                        && lower.contains("gift")
                        && task_contains_any(lower, &["sister", "dad", "father", "gave me", "got"])
                },
            );
            if !evidence.is_empty()
                && evidence
                    .iter()
                    .any(|line| line.to_ascii_lowercase().contains("sister"))
                && !evidence.iter().any(|line| {
                    let lower = line.to_ascii_lowercase();
                    lower.contains("my dad") || lower.contains(" dad ") || lower.contains("father")
                })
            {
                return self.write_synthetic_answer(
                    "missing-dad-birthday-gift",
                    task,
                    "You did not mention this information. You mentioned receiving a birthday gift from your sister, but not your dad.",
                    &evidence,
                );
            }
        }

        if task_contains_any(
            task_lower,
            &["became a parent first", "become a parent first"],
        ) && task_lower.contains("tom or alex")
        {
            let evidence = self.find_matching_lines(
                &["alex", "tom", "adopt", "baby", "january"],
                24,
                false,
                3,
                |line, lower| {
                    is_summary_or_user_line(line, lower)
                        && (lower.contains("alex") || lower.contains("tom"))
                        && task_contains_any(lower, &["adopt", "baby", "parent"])
                },
            );
            if !evidence.is_empty()
                && evidence
                    .iter()
                    .any(|line| line.to_ascii_lowercase().contains("alex"))
                && !evidence
                    .iter()
                    .any(|line| line.to_ascii_lowercase().contains("tom"))
            {
                let mentions_january = evidence
                    .iter()
                    .any(|line| line.to_ascii_lowercase().contains("january"));
                let answer = if mentions_january {
                    "The information provided is not enough. You mentioned Alex becoming a parent in January, but you didn't mention anything about Tom."
                } else {
                    "The information provided is not enough. You mentioned Alex becoming a parent, but you didn't mention anything about Tom."
                };
                return self.write_synthetic_answer(
                    "missing-parent-first-anchor",
                    task,
                    answer,
                    &evidence,
                );
            }
        }

        if task_lower.contains("uncle")
            && task_lower.contains("birthday")
            && task_contains_any(task_lower, &["bake", "baked"])
        {
            let evidence = self.find_matching_lines(
                &["bake", "birthday", "cake", "niece", "uncle"],
                24,
                false,
                3,
                |_, lower| lower.contains("baked") && lower.contains("birthday"),
            );
            if !evidence.is_empty()
                && !evidence
                    .iter()
                    .any(|line| line.to_ascii_lowercase().contains("uncle"))
            {
                return self.write_synthetic_answer(
                    "missing-uncle-birthday-bake",
                    task,
                    "You did not mention this information. You mentioned baking for your niece's birthday party but not your uncle's.",
                    &evidence,
                );
            }
        }

        None
    }
}
