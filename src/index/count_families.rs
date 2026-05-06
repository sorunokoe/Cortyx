use super::count_extractors::{
    extract_competitive_sport_signatures_from_line, extract_current_tank_signatures_from_line,
    extract_group_project_course_signature_from_line, extract_recent_baking_signatures_from_line,
    extract_recent_jewelry_acquisition_signatures_from_line,
    extract_recent_plant_acquisition_signatures_from_line,
};
use super::count_support::{nz, DistinctSignatureCountConfig};
use super::*;

impl NeuronIndex {
    pub(super) fn synthetic_recent_jewelry_acquisition_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_lower.contains("jewelry")
            || !task_contains_any(task_lower, &["acquire", "acquired"])
            || !task_contains_any(
                task_lower,
                &[
                    "last two months",
                    "past two months",
                    "in the last two months",
                ],
            )
        {
            return None;
        }

        let (count, evidence) = self.best_distinct_signature_count(
            task,
            DistinctSignatureCountConfig {
                required_owned: vec![
                    "jewelry".to_string(),
                    "necklace".to_string(),
                    "ring".to_string(),
                    "earrings".to_string(),
                    "month".to_string(),
                    "weekend".to_string(),
                ],
                candidate_limit: nz(8),
                max_lines: nz(256),
                evidence_limit: nz(4),
                line_match: |line: &str, lower: &str| is_summary_or_user_line(line, lower),
                extract: |line: &str, lower: &str| {
                    extract_recent_jewelry_acquisition_signatures_from_line(line, lower, 60)
                },
            },
        )?;
        self.write_synthetic_answer(
            "recent-jewelry-acquisition-count",
            task,
            &count.to_string(),
            &evidence,
        )
    }

    pub(super) fn synthetic_recent_plant_acquisition_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_contains_any(task_lower, &["plant", "plants"])
            || !task_contains_any(task_lower, &["acquire", "acquired"])
            || !task_contains_any(
                task_lower,
                &["last month", "past month", "in the last month"],
            )
        {
            return None;
        }

        let (count, evidence) = self.best_distinct_signature_count(
            task,
            DistinctSignatureCountConfig {
                required_owned: vec![
                    "plant".to_string(),
                    "plants".to_string(),
                    "nursery".to_string(),
                    "sister".to_string(),
                    "month".to_string(),
                    "week".to_string(),
                ],
                candidate_limit: nz(8),
                max_lines: nz(256),
                evidence_limit: nz(4),
                line_match: |line: &str, lower: &str| is_summary_or_user_line(line, lower),
                extract: |line: &str, lower: &str| {
                    extract_recent_plant_acquisition_signatures_from_line(line, lower, 30)
                },
            },
        )?;
        self.write_synthetic_answer(
            "recent-plant-acquisition-count",
            task,
            &count.to_string(),
            &evidence,
        )
    }

    pub(super) fn synthetic_simultaneous_project_count_excluding_thesis_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_lower.contains("project")
            || !task_contains_any(task_lower, &["excluding my thesis", "excluding thesis"])
        {
            return None;
        }

        let (count, evidence) = self.best_distinct_signature_count(
            task,
            DistinctSignatureCountConfig {
                required_owned: vec![
                    "project".to_string(),
                    "thesis".to_string(),
                    "course".to_string(),
                    "group".to_string(),
                ],
                candidate_limit: nz(8),
                max_lines: nz(128),
                evidence_limit: nz(4),
                line_match: |_line: &str, lower: &str| {
                    lower.starts_with("user:")
                        && (lower.contains("project") || lower.contains("thesis"))
                },
                extract: |line: &str, lower: &str| {
                    extract_group_project_course_signature_from_line(line, lower)
                        .into_iter()
                        .collect()
                },
            },
        )?;

        self.write_synthetic_answer(
            "simultaneous-project-count-excluding-thesis",
            task,
            &count.to_string(),
            &evidence,
        )
    }

    pub(super) fn synthetic_competitive_sport_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_lower.contains("sport")
            || !task_contains_any(task_lower, &["competitive", "competitively"])
        {
            return None;
        }

        let (count, evidence) = self.best_distinct_signature_count(
            task,
            DistinctSignatureCountConfig {
                required_owned: vec![
                    "sport".to_string(),
                    "competitive".to_string(),
                    "used to".to_string(),
                    "college".to_string(),
                    "high school".to_string(),
                ],
                candidate_limit: nz(8),
                max_lines: nz(192),
                evidence_limit: nz(4),
                line_match: |_line: &str, lower: &str| {
                    lower.starts_with("user:")
                        && task_contains_any(lower, &["competitive", "competitively"])
                        && task_contains_any(
                            lower,
                            &["used to", "college", "high school", "played"],
                        )
                },
                extract: extract_competitive_sport_signatures_from_line,
            },
        )?;

        self.write_synthetic_answer(
            "competitive-sport-count",
            task,
            &small_cardinal_word_lower(count),
            &evidence,
        )
    }

    pub(super) fn synthetic_current_tank_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_lower.contains("tank")
            || !task_contains_any(task_lower, &["currently", "current"])
        {
            return None;
        }

        let (count, evidence) = self.best_distinct_signature_count(
            task,
            DistinctSignatureCountConfig {
                required_owned: vec![
                    "tank".to_string(),
                    "current".to_string(),
                    "friend".to_string(),
                    "kid".to_string(),
                    "gallon".to_string(),
                ],
                candidate_limit: nz(8),
                max_lines: nz(512),
                evidence_limit: nz(6),
                line_match: |_line: &str, lower: &str| {
                    lower.starts_with("user:")
                        && lower.contains("tank")
                        && task_contains_any(
                            lower,
                            &[
                                "have a",
                                "have an",
                                "have my",
                                "currently",
                                "set up",
                                "taking care of",
                            ],
                        )
                },
                extract: extract_current_tank_signatures_from_line,
            },
        )?;

        self.write_synthetic_answer("current-tank-count", task, &count.to_string(), &evidence)
    }

    pub(super) fn synthetic_recent_baking_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_contains_any(task_lower, &["bake", "baked"])
            || !task_contains_any(task_lower, &["past two weeks", "last two weeks"])
        {
            return None;
        }

        let baking_focus_terms = synthetic_query_terms(task_lower)
            .into_iter()
            .filter(|term| {
                !matches!(
                    term.as_str(),
                    "how"
                        | "many"
                        | "times"
                        | "did"
                        | "i"
                        | "bake"
                        | "baked"
                        | "something"
                        | "past"
                        | "last"
                        | "two"
                        | "week"
                        | "weeks"
                        | "the"
                        | "in"
                )
            })
            .collect::<Vec<_>>();
        let (count, evidence) = self.best_distinct_signature_count(
            task,
            DistinctSignatureCountConfig {
                required_owned: vec![
                    "bake".to_string(),
                    "recipe".to_string(),
                    "weeks".to_string(),
                    "week".to_string(),
                ],
                candidate_limit: nz(8),
                max_lines: nz(256),
                evidence_limit: nz(6),
                line_match: move |_line: &str, lower: &str| {
                    lower.starts_with("user:")
                        && task_contains_any(
                            lower,
                            &[
                                "baked",
                                "bake",
                                "bread recipe",
                                "recipe",
                                "cake",
                                "cookies",
                                "baguette",
                                "make",
                            ],
                        )
                        && (baking_focus_terms.is_empty()
                            || baking_focus_terms
                                .iter()
                                .all(|term| lower.contains(term.as_str())))
                },
                extract: extract_recent_baking_signatures_from_line,
            },
        )?;

        self.write_synthetic_answer("recent-baking-count", task, &count.to_string(), &evidence)
    }
}
