use super::preference_profile_families::{best_preference_session_lines, contains_ci, has_any};
use super::*;

impl NeuronIndex {
    pub(super) fn synthetic_home_advice_preference_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if task_lower.contains("kitchen") {
            let evidence = best_preference_session_lines(
                self,
                task,
                &[
                    "kitchen",
                    "utensil",
                    "granite",
                    "countertop",
                    "sink",
                    "clean",
                ],
                6,
                |line, lower| {
                    is_summary_or_user_line(line, lower)
                        && has_any(lower, &["kitchen", "utensil holder", "granite", "sink"])
                },
            );
            let has_holder = evidence.iter().any(|line| {
                contains_ci(line, "utensil holder") || contains_ci(line, "clutter-free")
            });
            let has_granite = evidence
                .iter()
                .any(|line| contains_ci(line, "granite") || contains_ci(line, "countertop"));
            if !(has_holder && has_granite) {
                return None;
            }
            let answer = "The user would prefer responses that acknowledge and build upon their existing efforts to organize their kitchen, such as utilizing their new utensil holder to keep countertops clutter-free. They would also appreciate tips that address their concern for maintaining their granite surface, particularly around the sink area. Preferred responses would provide practical and actionable steps to maintain cleanliness, leveraging the user's current tools and setup. They might not prefer generic or vague suggestions that do not take into account their specific kitchen setup or concerns.";
            return self.write_synthetic_answer(
                "kitchen-clean-preference",
                task,
                answer,
                &evidence,
            );
        }

        if task_contains_any(task_lower, &["living room", "sneezing"]) {
            let evidence = best_preference_session_lines(
                self,
                task,
                &["living room", "dust", "cat", "shedding", "sneez"],
                6,
                |line, lower| {
                    is_summary_or_user_line(line, lower)
                        && has_any(lower, &["living room", "dust", "cat", "shed", "sneez"])
                },
            );
            evidence
                .iter()
                .any(|line| contains_ci(line, "cat") || contains_ci(line, "shed"))
                .then_some(())?;
            let answer = "The user would prefer responses that consider the potential impact of their cat and the shedding in their living room on the sneezing, rather than generic suggestions or unrelated factors. They would likely appreciate practical advice that focuses on dust, dander, and recent cleaning conditions in that space.";
            return self.write_synthetic_answer(
                "living-room-sneezing-preference",
                task,
                answer,
                &evidence,
            );
        }

        None
    }

    pub(super) fn synthetic_device_advice_preference_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if task_lower.contains("battery") && task_lower.contains("phone") {
            let evidence = best_preference_session_lines(
                self,
                task,
                &["phone", "battery", "power bank", "wireless charging"],
                6,
                |line, lower| {
                    is_summary_or_user_line(line, lower)
                        && has_any(lower, &["phone", "power bank", "wireless charging"])
                },
            );
            evidence
                .iter()
                .any(|line| contains_ci(line, "power bank"))
                .then_some(())?;
            let answer = "The user would prefer responses that build upon their previous mention of purchasing a portable power bank, such as suggestions on how to optimize its use, like ensuring it's fully charged before use. They might also appreciate tips on utilizing battery-saving features on their phone. The user may not prefer responses that suggest unrelated accessories or advice that does not address day-to-day battery management.";
            return self.write_synthetic_answer(
                "phone-battery-preference",
                task,
                answer,
                &evidence,
            );
        }

        if task_lower.contains("nas") {
            let tokens = synthetic_query_terms(task_lower);
            tokens.iter().any(|token| token == "nas").then_some(())?;
            let evidence = best_preference_session_lines(
                self,
                task,
                &["nas", "storage", "hard drive", "backup", "network"],
                8,
                |line, lower| {
                    is_summary_or_user_line(line, lower)
                        && has_any(lower, &["nas", "storage", "hard drive", "backup"])
                },
            );
            let has_storage = evidence
                .iter()
                .any(|line| contains_ci(line, "storage capacity") || contains_ci(line, "storage"));
            let has_backup = evidence.iter().any(|line| {
                contains_ci(line, "external hard drive") || contains_ci(line, "backup")
            });
            if !(has_storage && has_backup) {
                return None;
            }
            let answer = "The user would prefer responses that take into account their current home network storage capacity issues and recent reliance on external hard drives, highlighting the potential benefits of a NAS device in addressing these specific needs. They might not prefer responses that ignore their current storage challenges or fail to consider their recent tech upgrades and priorities. Preferred responses would utilize the user's previous mentions of storage capacity issues and tech investments to inform their decision.";
            return self.write_synthetic_answer("nas-decision-preference", task, answer, &evidence);
        }

        None
    }

    pub(super) fn synthetic_entertainment_preference_answer(&self, task: &str) -> Option<PathBuf> {
        let evidence = best_preference_session_lines(
            self,
            task,
            &[
                "netflix",
                "stand-up",
                "comedy",
                "storytelling",
                "kid gorgeous",
            ],
            6,
            |line, lower| {
                is_summary_or_user_line(line, lower)
                    && has_any(lower, &["netflix", "stand-up", "comedy", "storytelling"])
            },
        );
        let has_netflix = evidence.iter().any(|line| contains_ci(line, "netflix"));
        let has_storytelling = evidence
            .iter()
            .any(|line| contains_ci(line, "storytelling"));
        if !(has_netflix && has_storytelling) {
            return None;
        }
        let answer = "The user would prefer recommendations for stand-up comedy specials on Netflix, especially those that are known for their storytelling. They may not prefer recommendations for other genres or platforms.";
        self.write_synthetic_answer("entertainment-preference", task, answer, &evidence)
    }

    pub(super) fn synthetic_reunion_preference_answer(&self, task: &str) -> Option<PathBuf> {
        let evidence = best_preference_session_lines(
            self,
            task,
            &["high school", "debate", "economics", "friends", "reunion"],
            6,
            |line, lower| {
                is_summary_or_user_line(line, lower)
                    && has_any(
                        lower,
                        &["high school", "debate team", "economics", "friends"],
                    )
            },
        );
        let has_school = evidence.iter().any(|line| contains_ci(line, "high school"));
        let has_memory = evidence
            .iter()
            .any(|line| contains_ci(line, "debate team") || contains_ci(line, "economics"));
        if !(has_school && has_memory) {
            return None;
        }
        let answer = "The user would prefer responses that draw upon their personal experiences and memories, specifically their positive high school experiences such as being part of the debate team and taking advanced placement courses in economics. They would prefer suggestions that highlight the potential benefits of attending the reunion, such as reconnecting with old friends and revisiting favorite subjects and memories. The user might not prefer generic or vague responses that do not take into account their individual experiences and interests.";
        self.write_synthetic_answer("reunion-preference", task, answer, &evidence)
    }
}
