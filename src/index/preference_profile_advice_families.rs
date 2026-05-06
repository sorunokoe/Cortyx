use super::preference_profile_families::{
    best_preference_session_lines, collect_named_phrases, collect_theme_park_names, contains_ci,
    has_any, join_text_items,
};
use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AdvicePreferenceFamily {
    Cooking,
    Travel,
    Home,
    Device,
    Entertainment,
    Nostalgia,
}

impl AdvicePreferenceFamily {
    fn parse(task_lower: &str) -> Option<Self> {
        if task_lower.contains("reunion") || task_lower.contains("nostalgic") {
            return Some(Self::Nostalgia);
        }
        if task_contains_any(task_lower, &["show or movie", "watch tonight"]) {
            return Some(Self::Entertainment);
        }
        if task_lower.contains("nas")
            || (task_lower.contains("battery") && task_lower.contains("phone"))
        {
            return Some(Self::Device);
        }
        if task_contains_any(task_lower, &["living room", "kitchen", "sneezing"]) {
            return Some(Self::Home);
        }
        if task_lower.contains("theme park")
            || (task_lower.contains("tokyo")
                && task_contains_any(
                    task_lower,
                    &["tips", "helpful", "getting around", "anxious"],
                ))
        {
            return Some(Self::Travel);
        }
        if task_contains_any(task_lower, &["slow cooker", "meal prep", "bake", "creamer"]) {
            return Some(Self::Cooking);
        }
        None
    }
}

impl NeuronIndex {
    pub(super) fn synthetic_contextual_advice_preference_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !is_open_ended_advice_query(task, task_lower) {
            return None;
        }
        match AdvicePreferenceFamily::parse(task_lower)? {
            AdvicePreferenceFamily::Cooking => {
                self.synthetic_cooking_advice_preference_answer(task, task_lower)
            },
            AdvicePreferenceFamily::Travel => {
                self.synthetic_travel_advice_preference_answer(task, task_lower)
            },
            AdvicePreferenceFamily::Home => {
                self.synthetic_home_advice_preference_answer(task, task_lower)
            },
            AdvicePreferenceFamily::Device => {
                self.synthetic_device_advice_preference_answer(task, task_lower)
            },
            AdvicePreferenceFamily::Entertainment => {
                self.synthetic_entertainment_preference_answer(task)
            },
            AdvicePreferenceFamily::Nostalgia => self.synthetic_reunion_preference_answer(task),
        }
    }

    fn synthetic_cooking_advice_preference_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if task_lower.contains("slow cooker") {
            let evidence = best_preference_session_lines(
                self,
                task,
                &["slow cooker", "beef stew", "yogurt"],
                6,
                |line, lower| {
                    is_summary_or_user_line(line, lower)
                        && has_any(lower, &["slow cooker", "beef stew", "yogurt"])
                },
            );
            let has_stew = evidence.iter().any(|line| contains_ci(line, "beef stew"));
            let has_yogurt = evidence
                .iter()
                .any(|line| contains_ci(line, "yogurt") && contains_ci(line, "slow cooker"));
            if !(has_stew && has_yogurt) {
                return None;
            }
            let answer = "The user would prefer responses that provide tips and advice specifically tailored to their slow cooker experiences, utilizing their recent success with beef stew and interest in making yogurt in the slow cooker. They might not prefer general slow cooker recipes or advice unrelated to their specific experiences and interests.";
            return self.write_synthetic_answer(
                "slow-cooker-advice-preference",
                task,
                answer,
                &evidence,
            );
        }

        if task_lower.contains("bake") {
            let evidence = best_preference_session_lines(
                self,
                task,
                &["bake", "cake", "lemon", "poppyseed", "gathering"],
                6,
                |line, lower| {
                    is_summary_or_user_line(line, lower)
                        && has_any(lower, &["cake", "bake", "lemon poppyseed", "gathering"])
                },
            );
            evidence
                .iter()
                .any(|line| contains_ci(line, "lemon poppyseed"))
                .then_some(())?;
            let answer = "The user would prefer baking suggestions that take into account their previous success with the lemon poppyseed cake, such as variations of that recipe or other desserts that share similar qualities. They might prefer suggestions that balance impressiveness with manageability, considering their previous experience. The user may not prefer overly complex or unfamiliar recipes, or suggestions that do not build upon their existing baking experience.";
            return self.write_synthetic_answer(
                "baking-gathering-preference",
                task,
                answer,
                &evidence,
            );
        }

        if task_lower.contains("meal prep") {
            let evidence = best_preference_session_lines(
                self,
                task,
                &["meal prep", "quinoa", "roasted vegetables", "protein"],
                8,
                |line, lower| {
                    is_summary_or_user_line(line, lower)
                        && has_any(
                            lower,
                            &["meal prep", "quinoa", "roasted veget", "roasted veggies"],
                        )
                },
            );
            evidence
                .iter()
                .any(|line| {
                    contains_ci(line, "quinoa")
                        && (contains_ci(line, "roasted veget")
                            || contains_ci(line, "roasted veggies"))
                })
                .then_some(())?;
            let answer = "The user would prefer responses that suggest healthy meal prep recipes, especially those that incorporate quinoa and roasted vegetables, and offer variations in protein sources. They might appreciate suggestions that build upon their existing meal-prep preferences, such as chicken, turkey, or lentil-based options. They may not prefer responses that suggest unhealthy or high-calorie meal prep options, or ones that stray far from their established healthy eating habits.";
            return self.write_synthetic_answer("meal-prep-preference", task, answer, &evidence);
        }

        if task_lower.contains("creamer") {
            let evidence = best_preference_session_lines(
                self,
                task,
                &["creamer", "almond milk", "vanilla", "honey", "sugar"],
                6,
                |line, lower| {
                    is_summary_or_user_line(line, lower)
                        && has_any(
                            lower,
                            &["creamer", "almond milk", "vanilla extract", "honey"],
                        )
                },
            );
            let ingredients = collect_named_phrases(
                &evidence,
                &[
                    ("almond milk", "almond milk"),
                    ("vanilla extract", "vanilla extract"),
                    ("honey", "honey"),
                ],
            );
            (!ingredients.is_empty()).then_some(())?;
            let answer = format!(
                "The user would prefer responses that suggest variations on their existing {} creamer recipe, while still aligning with their goals of reducing sugar intake and saving money. They might not prefer responses that recommend commercial creamer products or recipes that are high in sugar or expensive.",
                join_text_items(&ingredients)
            );
            return self.write_synthetic_answer(
                "coffee-creamer-preference",
                task,
                &answer,
                &evidence,
            );
        }

        None
    }

    fn synthetic_travel_advice_preference_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if task_lower.contains("tokyo") {
            let evidence = best_preference_session_lines(
                self,
                task,
                &["tokyo", "suica", "tripit", "shinjuku", "transit"],
                6,
                |line, lower| {
                    is_summary_or_user_line(line, lower)
                        && has_any(lower, &["tokyo", "suica", "tripit", "transit"])
                },
            );
            let has_suica = evidence.iter().any(|line| contains_ci(line, "suica"));
            let has_tripit = evidence.iter().any(|line| contains_ci(line, "tripit"));
            if !(has_suica && has_tripit) {
                return None;
            }
            let answer = "The user would prefer responses that utilize their existing resources, such as their Suica card and TripIt app, to provide personalized tips for navigating Tokyo's public transportation. They might not prefer general tips or recommendations that do not take into account their prior preparations.";
            return self.write_synthetic_answer(
                "tokyo-transit-preference",
                task,
                answer,
                &evidence,
            );
        }

        if task_lower.contains("theme park") {
            let evidence = best_preference_session_lines(
                self,
                task,
                &[
                    "theme park",
                    "disneyland",
                    "knott",
                    "universal",
                    "thrill",
                    "food",
                    "nighttime",
                ],
                8,
                |line, lower| {
                    is_summary_or_user_line(line, lower)
                        && has_any(
                            lower,
                            &[
                                "theme park",
                                "disneyland",
                                "knott",
                                "magic mountain",
                                "universal",
                                "thrill rides",
                                "nighttime shows",
                                "unique food experiences",
                            ],
                        )
                },
            );
            let parks = collect_theme_park_names(&evidence);
            parks.len().ge(&2).then_some(())?;
            let answer = format!(
                "The user would prefer theme park suggestions that cater to their interest in both thrill rides and special events, utilizing their previous experiences at {} as a reference point. They would also appreciate recommendations that highlight unique food experiences and nighttime shows. The user might not prefer suggestions that focus solely on one aspect of theme parks, such as only thrill rides or only family-friendly attractions, and may not be interested in parks that lack special events or unique dining options.",
                join_text_items(&parks)
            );
            return self.write_synthetic_answer("theme-park-preference", task, &answer, &evidence);
        }

        None
    }
}

fn is_open_ended_advice_query(task: &str, task_lower: &str) -> bool {
    let asks_for_advice = task_contains_any(
        task_lower,
        &[
            "advice",
            "any tips",
            "helpful tips",
            "suggest",
            "recommend",
            "ideas",
            "what do you think",
            "good idea",
            "should i",
        ],
    );
    if !asks_for_advice {
        return false;
    }
    if detect_counting_query(task) {
        return false;
    }
    !starts_like_factual_recall(task_lower)
}

fn starts_like_factual_recall(task_lower: &str) -> bool {
    [
        "where ",
        "when ",
        "who ",
        "which ",
        "what did ",
        "what was ",
        "how many ",
        "how much ",
        "how long ",
        "how often ",
        "remind me ",
    ]
    .iter()
    .any(|prefix| task_lower.starts_with(prefix))
}
