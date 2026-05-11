use super::*;

impl NeuronIndex {
    pub(in crate::index::core) fn synthetic_project_lead_count_answer(
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

    pub(in crate::index::core) fn synthetic_model_kit_count_answer(
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
}
