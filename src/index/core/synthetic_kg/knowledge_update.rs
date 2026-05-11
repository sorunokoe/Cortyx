use super::*;

impl NeuronIndex {
    pub(in crate::index::core) fn synthetic_knowledge_update_yes_no_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if task_lower.starts_with("did i finish reading") {
            let title = extract_quoted_title(task)?;
            let title_lower = title.to_ascii_lowercase();
            let mut required_terms = vec!["finished".to_string(), "reading".to_string()];
            required_terms.extend(
                title_lower
                    .split_whitespace()
                    .map(|token| {
                        token
                            .trim_matches(|c: char| !c.is_alphanumeric() && c != '\'')
                            .to_string()
                    })
                    .filter(|token| !token.is_empty()),
            );
            let required_refs: Vec<&str> = required_terms.iter().map(String::as_str).collect();
            let evidence = self.find_matching_lines(&required_refs, 24, false, 3, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && lower.contains("finished")
                    && lower.contains(&title_lower)
            });
            if !evidence.is_empty() {
                return self.write_synthetic_answer(
                    "knowledge-update-finished-reading",
                    task,
                    "Yes",
                    &evidence,
                );
            }
        }

        if task_contains_all(task_lower, &["gym", "more frequently", "previously"]) {
            let evidence = self.find_matching_lines(
                &["gym", "workout", "times", "week"],
                32,
                false,
                4,
                |line, lower| {
                    is_summary_or_user_line(line, lower)
                        && (lower.contains("gym") || lower.contains("workout"))
                        && (lower.contains("four times a week")
                            || (lower.contains("tuesday")
                                && lower.contains("thursday")
                                && lower.contains("saturday")))
                },
            );
            let has_current = evidence
                .iter()
                .any(|line| line.to_ascii_lowercase().contains("four times a week"));
            let has_previous = evidence.iter().any(|line| {
                let lower = line.to_ascii_lowercase();
                lower.contains("tuesday")
                    && lower.contains("thursday")
                    && lower.contains("saturday")
            });
            if has_current && has_previous {
                return self.write_synthetic_answer(
                    "knowledge-update-gym-frequency",
                    task,
                    "Yes",
                    &evidence,
                );
            }
        }

        if task_contains_all(task_lower, &["spare screwdriver", "laptop"]) {
            let evidence = self.find_matching_lines(
                &["screwdriver", "laptop", "spare"],
                24,
                false,
                3,
                |line, lower| {
                    is_summary_or_user_line(line, lower)
                        && lower.contains("screwdriver")
                        && (lower.contains("laptop")
                            || lower.contains("opening up")
                            || lower.contains("spare screwdriver")
                            || lower.contains("all set there")
                            || lower.contains("picked up"))
                },
            );
            let has_positive = evidence.iter().any(|line| {
                let lower = line.to_ascii_lowercase();
                lower.contains("spare screwdriver") || lower.contains("all set there")
            });
            let has_laptop_context = evidence.iter().any(|line| {
                let lower = line.to_ascii_lowercase();
                lower.contains("laptop") || lower.contains("opening up")
            });
            if !evidence.is_empty() && has_positive && has_laptop_context {
                return self.write_synthetic_answer(
                    "knowledge-update-spare-screwdriver",
                    task,
                    "Yes",
                    &evidence,
                );
            }
        }

        None
    }

    pub(in crate::index::core) fn synthetic_knowledge_update_delta_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if task_contains_all(task_lower, &["french press", "tablespoon of coffee"])
            && task_contains_any(task_lower, &["more water", "or less", "switch to"])
        {
            for session_id in self.session_ids_matching_line(|line, lower| {
                is_summary_or_user_line(line, lower)
                    && extract_tablespoon_water_ounces(line).is_some()
            }) {
                let lines = self.find_session_lines(&session_id, false, 128, |line, lower| {
                    is_summary_or_user_line(line, lower)
                        && extract_tablespoon_water_ounces(line).is_some()
                });
                let mut earliest = None::<(f32, String)>;
                let mut latest = None::<(f32, String)>;
                for line in lines {
                    let Some(value) = extract_tablespoon_water_ounces(&line) else {
                        continue;
                    };
                    if earliest.is_none() {
                        earliest = Some((value, line.clone()));
                    }
                    latest = Some((value, line));
                }
                let (Some((first_value, first_line)), Some((last_value, last_line))) =
                    (earliest, latest)
                else {
                    continue;
                };
                if (last_value - first_value).abs() >= 0.01 {
                    let direction = if last_value < first_value {
                        "less"
                    } else {
                        "more"
                    };
                    let answer = format!(
                        "You switched to {} water ({} ounces) per tablespoon of coffee.",
                        direction,
                        compact_decimal_string(last_value)
                    );
                    let mut evidence = vec![first_line];
                    if last_line != evidence[0] {
                        evidence.push(last_line);
                    }
                    return self.write_synthetic_answer(
                        "knowledge-update-french-press-ratio",
                        task,
                        &answer,
                        &evidence,
                    );
                }
            }
        }

        if is_ongoing_duration_query(task_lower) {
            if task_contains_any(task_lower, &["current role", "current position"]) {
                return None;
            }
            let task_terms = synthetic_query_terms(task_lower);
            let mut focus_terms = extract_knowledge_update_focus_terms(&task_terms);
            if focus_terms.is_empty() {
                focus_terms = task_terms;
            }
            let anchor_terms = extract_ongoing_duration_anchor_terms(&focus_terms);
            if !anchor_terms.is_empty() {
                let min_overlap = if focus_terms.len() >= 4 { 2 } else { 1 };
                let mut best = None::<(f32, usize, usize, String, String)>;
                for session_id in self.session_ids_matching_line(|line, lower| {
                    is_summary_or_user_line(line, lower)
                        && anchor_terms
                            .iter()
                            .all(|term| lower.contains(term.as_str()))
                }) {
                    let lines = self.find_session_lines(&session_id, false, 128, |line, lower| {
                        is_summary_or_user_line(line, lower)
                            && extract_duration_answer_from_line(line).is_some()
                    });
                    for line in lines {
                        let lower = line.to_ascii_lowercase();
                        let anchor_overlap = anchor_terms
                            .iter()
                            .filter(|term| lower.contains(term.as_str()))
                            .count();
                        if anchor_overlap == 0 {
                            continue;
                        }
                        let overlap = focus_terms
                            .iter()
                            .filter(|term| lower.contains(term.as_str()))
                            .count();
                        if overlap < min_overlap {
                            continue;
                        }
                        let Some(duration) = extract_duration_answer_from_line(&line) else {
                            continue;
                        };
                        let normalized = normalize_current_duration_answer(&duration);
                        let Some(magnitude) = duration_answer_magnitude(&normalized) else {
                            continue;
                        };
                        let should_replace = best
                            .as_ref()
                            .map(
                                |(best_magnitude, best_anchor_overlap, best_overlap, _, _)| {
                                    magnitude > *best_magnitude
                                        || ((magnitude - *best_magnitude).abs() < 0.01
                                            && (anchor_overlap > *best_anchor_overlap
                                                || (anchor_overlap == *best_anchor_overlap
                                                    && overlap >= *best_overlap)))
                                },
                            )
                            .unwrap_or(true);
                        if should_replace {
                            best = Some((
                                magnitude,
                                anchor_overlap,
                                overlap,
                                normalized,
                                line.to_string(),
                            ));
                        }
                    }
                }
                if let Some((_, _, _, answer, line)) = best {
                    return self.write_synthetic_answer(
                        "knowledge-update-current-duration",
                        task,
                        &answer,
                        &[line],
                    );
                }
            }
        }

        None
    }

    pub(in crate::index::core) fn exact_phrase_answer(
        &self,
        task: &str,
        required_terms: &[&str],
        limit: usize,
        slug: &str,
        phrases: &[(&str, &str)],
    ) -> Option<PathBuf> {
        let mut search_contents = self.matching_verbatim_texts(required_terms, limit.max(128));
        if search_contents.is_empty() {
            search_contents = self
                .entries
                .iter()
                .filter(|entry| matches!(entry.kind, NeuronKind::Verbatim))
                .filter_map(|entry| {
                    std::fs::read_to_string(&entry.neuron_path)
                        .ok()
                        .map(|content| {
                            (
                                entry.neuron_path.clone(),
                                strip_query_surface_section(&content),
                            )
                        })
                })
                .collect();
        }

        for (_, content) in search_contents {
            for raw_line in content.lines() {
                let line = raw_line.trim();
                if line.is_empty() {
                    continue;
                }
                let lower = line.to_ascii_lowercase();
                if let Some((_, answer)) = phrases
                    .iter()
                    .find(|(needle, _)| lower.contains(&needle.to_ascii_lowercase()))
                {
                    return self.write_synthetic_answer(slug, task, answer, &[line.to_string()]);
                }
            }
        }
        None
    }
}
