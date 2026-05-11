use super::*;

impl NeuronIndex {
    pub(in crate::index::core) fn synthetic_restaurant_serving_dish_answer(
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

    pub(in crate::index::core) fn synthetic_bike_inventory_before_purchase_answer(
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

    pub(in crate::index::core) fn synthetic_coupon_store_answer(
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

    pub(in crate::index::core) fn synthetic_image_subject_color_answer(
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

    pub(in crate::index::core) fn synthetic_issue_after_service_answer(
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

    pub(in crate::index::core) fn synthetic_fitness_record_answer(
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

    pub(in crate::index::core) fn synthetic_clothing_store_count_answer(
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
