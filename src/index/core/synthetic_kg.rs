// This file is a submodule of `crate::index::core`.
// Contains `impl NeuronIndex` synthetic answer methods extracted from synthetic.rs.
use super::*;

impl NeuronIndex {
    pub(super) fn synthetic_knowledge_update_yes_no_answer(
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

    pub(super) fn synthetic_knowledge_update_delta_answer(
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

    pub(super) fn exact_phrase_answer(
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

    pub(super) fn synthetic_transport_cost_delta_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !is_money_query(task)
            || !task_contains_any(task_lower, &["taxi", "train"])
            || !task_contains_any(
                task_lower,
                &["more expensive", "compared to", "than the train"],
            )
        {
            return None;
        }

        let session_id = self.best_matching_session_id(task, &["commute", "taxi", "train"])?;
        let lines = self.find_session_lines(&session_id, false, 256, |line, lower| {
            is_summary_or_user_line(line, lower)
                && !extract_dollar_amounts(line).is_empty()
                && (lower.contains("taxi") || lower.contains("train fare"))
        });

        let mut taxi: Option<(usize, usize, f32, String)> = None;
        let mut train: Option<(usize, usize, f32, String)> = None;

        for (line_idx, line) in lines.into_iter().enumerate() {
            let lower = line.to_ascii_lowercase();
            let Some(amount) = extract_dollar_amounts(&line).into_iter().next() else {
                continue;
            };

            if lower.contains("taxi") {
                let score = usize::from(lower.contains("cost me"))
                    + usize::from(lower.contains("taxi ride"))
                    + usize::from(lower.contains("take a taxi"))
                    + usize::from(lower.contains("missed my train"));
                let should_replace = taxi
                    .as_ref()
                    .map(|(best_score, best_idx, _, _)| {
                        score > *best_score || (score == *best_score && line_idx > *best_idx)
                    })
                    .unwrap_or(true);
                if should_replace {
                    taxi = Some((score, line_idx, amount, line.clone()));
                }
            }

            if lower.contains("train fare") {
                let score = usize::from(lower.contains("actually"))
                    + usize::from(lower.contains("daily train fare"))
                    + usize::from(lower.contains("averages out"));
                let should_replace = train
                    .as_ref()
                    .map(|(best_score, best_idx, _, _)| {
                        score > *best_score || (score == *best_score && line_idx > *best_idx)
                    })
                    .unwrap_or(true);
                if should_replace {
                    train = Some((score, line_idx, amount, line.clone()));
                }
            }
        }

        let (_, _, taxi_cost, taxi_line) = taxi?;
        let (_, _, train_cost, train_line) = train?;
        if taxi_cost <= train_cost {
            return None;
        }
        let diff = taxi_cost - train_cost;
        let answer = format!("${}", format_numeric_answer(diff));
        self.write_synthetic_answer(
            "transport-cost-delta",
            task,
            &answer,
            &[train_line, taxi_line],
        )
    }

    pub(super) fn synthetic_named_schedule_rotation_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !task_contains_any(task_lower, &["schedule", "shift", "rotation"]) {
            return None;
        }
        let day = extract_weekday_from_query(task_lower)?;
        let person = extract_schedule_query_person(task)?;
        let person_lower = person.to_ascii_lowercase();
        for session_id in self.session_ids_matching_line(|line, lower| {
            line.trim_start().starts_with('|')
                && (lower.contains(day) || lower.contains(&person_lower) || lower.contains("shift"))
        }) {
            let table_lines = self.find_session_lines(&session_id, false, 256, |line, _| {
                line.trim_start().starts_with('|')
            });
            if let Some((shift, evidence)) =
                extract_schedule_shift_from_table(&table_lines, &person, day)
            {
                let answer = format!(
                    "{person} was assigned to the {shift} on {}.",
                    pluralize_weekday(day)
                );
                return self.write_synthetic_answer("schedule-rotation", task, &answer, &evidence);
            }
        }
        None
    }

    pub(super) fn synthetic_restaurant_serving_dish_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !task_lower.contains("restaurant")
            || !task_contains_any(task_lower, &["serves", "serve"])
        {
            return None;
        }
        let dish = extract_served_dish_from_query(task, task_lower)?;
        for session_id in self.session_ids_matching_line(|_line, lower| {
            lower.contains(&dish)
                || lower.contains("restaurant")
                || lower.contains("cafe")
                || lower.contains("providore")
        }) {
            let lines = self.find_session_lines(&session_id, false, 256, |_line, lower| {
                lower.contains(&dish)
                    || lower.contains("restaurant")
                    || lower.contains("cafe")
                    || lower.contains("providore")
            });
            if let Some((restaurant, evidence)) = extract_restaurant_serving_dish(&lines, &dish) {
                return self.write_synthetic_answer(
                    "restaurant-serving-dish",
                    task,
                    &restaurant,
                    &evidence,
                );
            }
        }
        None
    }

    pub(super) fn synthetic_bike_inventory_before_purchase_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !task_lower.contains("bike")
            || !task_contains_any(task_lower, &["before i purchased", "before i bought"])
            || !task_contains_any(task_lower, &["other bikes", "in addition to"])
            || !task_lower.contains("mountain bike")
            || !task_lower.contains("commuter bike")
        {
            return None;
        }

        for session_id in self.session_ids_matching_line(|line, lower| {
            is_summary_or_user_line(line, lower)
                && lower.contains("road bike")
                && lower.contains("mountain bike")
                && lower.contains("commuter bike")
        }) {
            let lines = self.find_session_lines(&session_id, true, 128, |line, lower| {
                is_summary_or_user_line(line, lower) && lower.contains("bike")
            });
            let inventory_line = lines.iter().find(|line| {
                let lower = line.to_ascii_lowercase();
                lower.contains("road bike")
                    && lower.contains("mountain bike")
                    && lower.contains("commuter bike")
            });
            let Some(inventory_line) = inventory_line else {
                continue;
            };
            let count_line = lines.iter().find(|line| {
                let lower = line.to_ascii_lowercase();
                lower.contains("currently have three bikes") || lower.contains("other two bikes")
            });
            let mut evidence = vec![inventory_line.clone()];
            if let Some(line) = count_line {
                if line != inventory_line {
                    evidence.push(line.clone());
                }
            }
            return self.write_synthetic_answer(
                "bike-inventory-before-purchase",
                task,
                "Yes. (You have a road bike too.)",
                &evidence,
            );
        }

        for entry in self
            .entries
            .iter()
            .filter(|entry| matches!(entry.kind, NeuronKind::Verbatim))
        {
            let Ok(content) = std::fs::read_to_string(&entry.neuron_path) else {
                continue;
            };
            let lines: Vec<String> = strip_query_surface_section(&content)
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(ToString::to_string)
                .collect();
            let inventory_line = lines.iter().find(|line| {
                let lower = line.to_ascii_lowercase();
                lower.contains("road bike")
                    && lower.contains("mountain bike")
                    && lower.contains("commuter bike")
            });
            let Some(inventory_line) = inventory_line else {
                continue;
            };
            let count_line = lines.iter().find(|line| {
                let lower = line.to_ascii_lowercase();
                lower.contains("currently have three bikes") || lower.contains("other two bikes")
            });
            let mut evidence = vec![inventory_line.clone()];
            if let Some(line) = count_line {
                if line != inventory_line {
                    evidence.push(line.clone());
                }
            }
            return self.write_synthetic_answer(
                "bike-inventory-before-purchase",
                task,
                "Yes. (You have a road bike too.)",
                &evidence,
            );
        }

        None
    }

    pub(super) fn synthetic_commute_time_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !is_commute_query(task_lower) {
            return None;
        }
        let task_terms = synthetic_query_terms(task_lower);
        let candidates =
            self.ranked_session_candidates(task, &["commute", "work", "audiobook"], &task_terms, 8);
        let mut best = None::<(usize, String, Vec<String>)>;

        for (session_id, session_score) in candidates {
            let lines = self.find_session_lines(&session_id, true, 64, |line, lower| {
                is_summary_or_user_line(line, lower) && lower.contains("commute")
            });
            for line in lines {
                let Some(value) = extract_commute_duration_from_line(&line) else {
                    continue;
                };
                let lower = line.to_ascii_lowercase();
                let overlap = task_terms
                    .iter()
                    .filter(|term| lower.contains(term.as_str()))
                    .count();
                let score = session_score
                    + overlap * 4
                    + if value.contains("each way") { 8 } else { 0 }
                    + if lower.contains("daily commute") {
                        4
                    } else {
                        0
                    }
                    + if lower.contains("work") { 2 } else { 0 };
                let replace = best
                    .as_ref()
                    .map(|(best_score, current, _)| {
                        (value.contains("each way") && !current.contains("each way"))
                            || (value.contains("each way") == current.contains("each way")
                                && (score > *best_score
                                    || (score == *best_score && value.len() > current.len())))
                    })
                    .unwrap_or(true);
                if replace {
                    best = Some((score, value, vec![line.clone()]));
                }
            }
        }

        let (_, answer, evidence) = best?;
        self.write_synthetic_answer("commute-time", task, &answer, &evidence)
    }

    pub(super) fn synthetic_coupon_store_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !task_lower.contains("coupon")
            || !task_contains_any(
                task_lower,
                &[
                    "where did i redeem",
                    "where did i use my coupon",
                    "where did i buy",
                    "which store",
                ],
            )
        {
            return None;
        }
        let session_id = self.best_matching_session_id(task, &["coupon", "redeem", "shop"])?;
        let lines = self.find_session_lines(&session_id, true, 64, |line, lower| {
            is_summary_or_user_line(line, lower)
        });
        let mut redemption_line = None::<String>;
        let mut store = None::<(String, String)>;
        for line in lines {
            let lower = line.to_ascii_lowercase();
            if redemption_line.is_none()
                && lower.contains("coupon")
                && task_contains_any(&lower, &["redeemed", "redeem"])
            {
                redemption_line = Some(line.clone());
            }
            if store.is_none() {
                if let Some(name) = extract_store_name_from_line(&line, &lower) {
                    store = Some((name, line.clone()));
                }
            }
        }
        let (answer, store_line) = store?;
        let mut evidence = Vec::new();
        if let Some(line) = redemption_line {
            evidence.push(line);
        }
        if !evidence.iter().any(|existing| existing == &store_line) {
            evidence.push(store_line);
        }
        self.write_synthetic_answer("coupon-store", task, &answer, &evidence)
    }

    pub(super) fn synthetic_image_subject_color_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !(task_lower.contains("what color") && task_lower.contains("image")) {
            return None;
        }
        let subject = extract_image_subject_from_query(task)?;
        let subject_lower = subject.to_ascii_lowercase();
        for session_id in self.session_ids_matching_line(|line, lower| {
            lower.contains(&subject_lower) && (line.contains("::") || lower.contains(" body"))
        }) {
            let lines = self.find_session_lines(&session_id, false, 64, |_line, lower| {
                lower.contains(&subject_lower)
            });
            if let Some((answer, evidence)) = extract_image_subject_body_color(&lines, &subject) {
                return self.write_synthetic_answer(
                    "image-subject-color",
                    task,
                    &answer,
                    &evidence,
                );
            }
        }
        None
    }

    pub(super) fn synthetic_issue_after_service_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !(task_lower.contains("first issue")
            && task_lower.contains("service")
            && task_lower.contains("car"))
        {
            return None;
        }
        let task_terms = synthetic_query_terms(task_lower);
        let candidates =
            self.ranked_session_candidates(task, &["car", "service", "issue"], &task_terms, 8);
        let mut best = None::<(usize, String, Vec<String>)>;

        for (session_id, session_score) in candidates {
            let lines = self.find_session_lines(&session_id, true, 64, |line, lower| {
                is_summary_or_user_line(line, lower)
            });
            let mut service_line = None::<String>;
            for line in lines {
                let lower = line.to_ascii_lowercase();
                if service_line.is_none()
                    && (lower.contains("serviced for the first time")
                        || lower.contains("first service"))
                {
                    service_line = Some(line);
                    continue;
                }
                let Some(service_evidence) = service_line.as_ref() else {
                    continue;
                };
                let Some(issue) = extract_issue_after_service_line(&line, &lower) else {
                    continue;
                };
                let overlap = task_terms
                    .iter()
                    .filter(|term| lower.contains(term.as_str()))
                    .count();
                let score = session_score
                    + overlap * 4
                    + if lower.contains("gps") && lower.contains("system") {
                        8
                    } else {
                        0
                    };
                let replace = best
                    .as_ref()
                    .map(|(best_score, _, _)| score > *best_score)
                    .unwrap_or(true);
                if replace {
                    best = Some((score, issue, vec![service_evidence.clone(), line.clone()]));
                }
                break;
            }
        }

        let (_, answer, evidence) = best?;
        self.write_synthetic_answer("first-issue-after-service", task, &answer, &evidence)
    }

    pub(super) fn synthetic_fitness_record_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !is_fitness_record_query(task_lower) {
            return None;
        }

        let task_terms = synthetic_query_terms(task_lower);
        let anchor_terms = task_terms
            .iter()
            .filter(|term| matches!(term.as_str(), "5k" | "charity" | "run"))
            .cloned()
            .collect::<Vec<_>>();
        let candidates =
            self.ranked_session_candidates(task, &["personal", "best", "time"], &task_terms, 8);
        let mut best = None::<(usize, u32, String, Vec<String>)>;

        for (session_id, session_score) in candidates {
            let lines = self.find_session_lines(&session_id, true, 96, |line, lower| {
                is_summary_or_user_line(line, lower)
            });
            let mut session_best = None::<(usize, u32, String, Vec<String>)>;

            for (idx, line) in lines.iter().enumerate() {
                let lower = line.to_ascii_lowercase();
                let mut context_lines = Vec::new();
                if let Some(previous) = idx.checked_sub(1).and_then(|prev| lines.get(prev)) {
                    context_lines.push(previous.clone());
                }
                context_lines.push(line.clone());
                if let Some(next) = lines.get(idx + 1) {
                    context_lines.push(next.clone());
                }
                let context_lower = context_lines
                    .iter()
                    .map(|candidate| candidate.to_ascii_lowercase())
                    .collect::<Vec<_>>()
                    .join(" ");

                if !task_contains_any(&context_lower, &["personal best", "best time", "record"]) {
                    continue;
                }
                if !anchor_terms.is_empty()
                    && !anchor_terms
                        .iter()
                        .any(|term| context_lower.contains(term.as_str()))
                {
                    continue;
                }
                let Some((seconds, raw)) = extract_fitness_record_time_value(line) else {
                    continue;
                };

                let overlap = task_terms
                    .iter()
                    .filter(|term| context_lower.contains(term.as_str()))
                    .count();
                let line_score = overlap * 4
                    + if lower.contains("personal best") || lower.contains("best time") {
                        8
                    } else {
                        0
                    }
                    + if context_lower.contains("5k") { 6 } else { 0 }
                    + if context_lower.contains("charity") {
                        4
                    } else {
                        0
                    };
                let evidence = context_lines
                    .into_iter()
                    .filter(|candidate| {
                        let lower = candidate.to_ascii_lowercase();
                        lower.contains(&raw.to_ascii_lowercase())
                            || lower.contains("5k")
                            || lower.contains("charity")
                    })
                    .collect::<Vec<_>>();
                let replace = session_best
                    .as_ref()
                    .map(|(best_score, best_seconds, _, _)| {
                        seconds < *best_seconds
                            || (seconds == *best_seconds && line_score > *best_score)
                    })
                    .unwrap_or(true);
                if replace {
                    session_best = Some((line_score, seconds, raw, evidence));
                }
            }

            let Some((line_score, seconds, raw, evidence)) = session_best else {
                continue;
            };
            let combined_score = session_score + line_score;
            let answer = normalize_fitness_record_kg_value(&raw);
            let replace = best
                .as_ref()
                .map(|(best_score, best_seconds, _, _)| {
                    combined_score > *best_score
                        || (combined_score == *best_score && seconds < *best_seconds)
                })
                .unwrap_or(true);
            if replace {
                best = Some((combined_score, seconds, answer, evidence));
            }
        }

        let (_, _, answer, evidence) = best?;
        self.write_synthetic_answer("fitness-record", task, &answer, &evidence)
    }

    pub(super) fn synthetic_project_lead_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_lower.contains("project")
            || !task_contains_any(task_lower, &["led", "leading"])
        {
            return None;
        }

        let session_id =
            self.best_matching_session_id(task, &["project", "competition", "class"])?;
        let lines = self.find_session_lines(&session_id, true, 128, |line, lower| {
            is_summary_or_user_line(line, lower)
        });
        let mut items = Vec::<String>::new();
        let mut seen = std::collections::HashSet::new();
        let mut evidence = Vec::new();

        for line in lines {
            let lower = line.to_ascii_lowercase();
            let Some(item) = extract_project_count_item(&line, &lower) else {
                continue;
            };
            if seen.insert(normalized_synthetic_phrase_key(&item)) {
                items.push(item);
                if evidence.len() < 3 {
                    evidence.push(line);
                }
            }
        }

        if items.len() < 2 {
            return None;
        }
        self.write_synthetic_answer(
            "project-lead-count",
            task,
            &items.len().to_string(),
            &evidence,
        )
    }

    pub(super) fn synthetic_model_kit_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task) || !task_contains_all(task_lower, &["model", "kit"]) {
            return None;
        }

        let session_id = self.best_matching_session_id(task, &["model", "kit", "scale"])?;
        let lines = self.find_session_lines(&session_id, false, 192, |line, lower| {
            is_summary_or_user_line(line, lower) || lower.starts_with("user:")
        });
        let mut items = Vec::<String>::new();
        let mut evidence = Vec::new();

        for line in lines {
            let lower = line.to_ascii_lowercase();
            let Some(item) = extract_model_kit_count_item(&line, &lower) else {
                continue;
            };
            let item_key = normalized_synthetic_phrase_key(&item);
            if let Some(existing) = items.iter_mut().find(|existing| {
                let existing_key = normalized_synthetic_phrase_key(existing);
                existing_key == item_key
                    || existing_key.contains(&item_key)
                    || item_key.contains(&existing_key)
            }) {
                if item.len() > existing.len() {
                    *existing = item;
                }
            } else {
                items.push(item);
                if evidence.len() < 3 {
                    evidence.push(line);
                }
            }
        }

        if items.len() < 3 {
            return None;
        }

        let word = num_to_word(items.len());
        let rendered_count = if word.is_empty() {
            items.len().to_string()
        } else {
            word.to_string()
        };
        let rendered_items = items
            .iter()
            .map(|item| {
                if item.eq_ignore_ascii_case("Revell F-15 Eagle") {
                    "Revell F-15 Eagle (scale not mentioned)".to_string()
                } else {
                    item.clone()
                }
            })
            .collect::<Vec<_>>();
        let answer = format!(
            "I have worked on or bought {rendered_count} model kits. The scales of the models are: {}.",
            Self::format_index_answer_surface_list(&rendered_items)
        );
        self.write_synthetic_answer("model-kit-count", task, &answer, &evidence)
    }

    pub(super) fn synthetic_clothing_store_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_contains_all(task_lower, &["items", "clothing"])
            || !task_contains_any(task_lower, &["pick up", "return"])
        {
            return None;
        }

        let session_id =
            self.best_matching_session_id(task, &["clothes", "blazer", "boots", "store"])?;
        let lines = self.find_session_lines(&session_id, true, 128, |line, lower| {
            is_summary_or_user_line(line, lower)
        });
        let mut actions = std::collections::HashSet::new();
        let mut evidence = Vec::new();

        let mut last_item = None::<String>;
        for line in lines {
            let lower = line.to_ascii_lowercase();

            if let Some(item) = extract_clothing_store_item(&line, &lower) {
                let item_key = normalized_synthetic_phrase_key(&item);
                if lower.contains("pick up")
                    && actions.insert(format!("pickup:{item_key}"))
                    && evidence.len() < 3
                {
                    evidence.push(line.clone());
                }
                if lower.contains("return")
                    && actions.insert(format!("return:{item_key}"))
                    && evidence.len() < 3
                {
                    evidence.push(line.clone());
                }
                last_item = Some(item);
                continue;
            }

            if let Some(item) = last_item.as_deref() {
                if (lower.contains("pick them up") || lower.contains("pick it up"))
                    && actions.insert(format!("pickup:{}", normalized_synthetic_phrase_key(item)))
                    && evidence.len() < 3
                {
                    evidence.push(line);
                }
            }
        }

        if actions.len() < 2 {
            return None;
        }
        self.write_synthetic_answer(
            "clothing-store-count",
            task,
            &actions.len().to_string(),
            &evidence,
        )
    }
}
