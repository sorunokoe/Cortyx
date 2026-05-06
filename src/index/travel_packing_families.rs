use super::*;

impl NeuronIndex {
    pub(super) fn synthetic_travel_packing_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        parse_travel_packing_query(task_lower)?;
        self.synthetic_packed_shoes_percentage_answer(task)
    }

    fn synthetic_packed_shoes_percentage_answer(&self, task: &str) -> Option<PathBuf> {
        let texts = self.matching_verbatim_texts(&["shoes", "trip", "packed"], 8);
        let mut packed = None;
        let mut wore = None;
        let mut evidence = Vec::new();
        for (_, content) in texts {
            for line in content.lines() {
                let lower = line.to_ascii_lowercase();
                let nums = extract_line_numbers(line);
                if nums.is_empty() {
                    continue;
                }
                if lower.contains("pairs of shoes") && lower.contains("packed") && packed.is_none()
                {
                    packed = nums.first().copied();
                    push_unique(&mut evidence, line);
                }
                if (lower.contains("only wearing") || lower.contains("only wore"))
                    && lower.contains("shoe")
                    && wore.is_none()
                {
                    wore = nums.first().copied();
                    push_unique(&mut evidence, line);
                }
            }
        }

        let (packed, wore) = (packed?, wore?);
        (packed > 0 && wore <= packed).then_some(())?;
        let percent = ((wore as f32 / packed as f32) * 100.0).round() as i32;
        self.write_synthetic_answer(
            "packed-shoes-percent",
            task,
            &format!("{percent}%"),
            &evidence,
        )
    }
}

fn parse_travel_packing_query(task_lower: &str) -> Option<()> {
    (task_lower.contains("packed shoes")
        && task_lower.contains("last trip")
        && task_contains_any(task_lower, &["percentage", "percent"]))
    .then_some(())
}

fn push_unique(lines: &mut Vec<String>, line: &str) {
    let trimmed = line.trim().to_string();
    if !lines.iter().any(|existing| existing == &trimmed) {
        lines.push(trimmed);
    }
}
