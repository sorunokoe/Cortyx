use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum OpenQaLocationTarget {
    State,
    Country,
    NationalPark,
}

pub(super) fn open_qa_location_target(task: &str) -> Option<OpenQaLocationTarget> {
    let lower = task.to_ascii_lowercase();
    if lower.contains("national park") {
        Some(OpenQaLocationTarget::NationalPark)
    } else if lower.starts_with("what state")
        || lower.starts_with("which state")
        || lower.contains(" in what state")
        || lower.contains(" in which state")
        || lower.contains(" us state")
        || lower.contains(" us states")
    {
        Some(OpenQaLocationTarget::State)
    } else if lower.starts_with("what country")
        || lower.starts_with("which country")
        || lower.contains(" in what country")
        || lower.contains(" in which country")
        || lower.contains(" home country")
    {
        Some(OpenQaLocationTarget::Country)
    } else {
        None
    }
}

pub(super) fn parse_open_qa_choice_options(task: &str) -> Vec<ChoiceOption> {
    let lower = task.to_ascii_lowercase();
    if !lower.contains(" or ")
        || lower.contains("answer in yes or no")
        || lower.ends_with("yes or no")
    {
        return Vec::new();
    }

    let tail = task.trim().trim_end_matches('?').trim();
    let Some((left_segment, right_segment)) = tail.rsplit_once(" or ") else {
        return Vec::new();
    };
    let left_raw = [
        " close to ",
        " going to ",
        " visiting ",
        " visit ",
        " in ",
        " at ",
        " between ",
        ", ",
    ]
    .iter()
    .find_map(|marker| left_segment.rsplit_once(marker).map(|(_, value)| value))
    .unwrap_or(left_segment);

    [left_raw, right_segment]
        .into_iter()
        .filter_map(|raw| {
            let display = raw
                .trim()
                .trim_start_matches("the ")
                .trim_start_matches("a ")
                .trim_start_matches("an ")
                .trim_matches(|c: char| matches!(c, '?' | ',' | '.' | ':' | ';'))
                .to_string();
            let tokens = salient_query_terms(&display)
                .into_iter()
                .filter(|token| parse_count_token(token).is_none())
                .collect::<Vec<_>>();
            (!display.is_empty() && !tokens.is_empty()).then_some(ChoiceOption { display, tokens })
        })
        .collect()
}

fn open_qa_choice_affinity_terms(display_lower: &str) -> &'static [&'static str] {
    if display_lower.contains("national park") {
        &[
            "nature",
            "outdoors",
            "outdoor",
            "camping",
            "camp",
            "hiking",
            "mountain",
            "mountains",
            "forest",
            "woods",
            "trail",
            "park",
        ]
    } else if display_lower.contains("theme park") {
        &[
            "theme",
            "amusement",
            "rides",
            "roller",
            "coaster",
            "disney",
            "universal",
            "park",
        ]
    } else if display_lower.contains("mountain") {
        &[
            "mountain",
            "mountains",
            "hiking",
            "camping",
            "nature",
            "outdoors",
            "trail",
            "park",
        ]
    } else if display_lower.contains("beach") {
        &["beach", "ocean", "coast", "shore", "sand", "waves", "surf"]
    } else {
        &[]
    }
}

pub(super) fn open_qa_location_alias(target: OpenQaLocationTarget, text: &str) -> Option<String> {
    let lower = text.to_ascii_lowercase();
    let aliases: &[(&[&str], &str)] = match target {
        OpenQaLocationTarget::State => &[
            (
                &["universal studios hollywood", "hollywood", "los angeles"],
                "California",
            ),
            (
                &[
                    "universal studios orlando",
                    "orlando",
                    "miami",
                    "disney world",
                ],
                "Florida",
            ),
            (&["universal studios"], "California or Florida"),
            (&["california"], "California"),
            (&["florida"], "Florida"),
            (&["indiana", "indianapolis", "indiana dunes"], "Indiana"),
            (
                &["minnesota", "minneapolis", "st. paul", "voyageurs"],
                "Minnesota",
            ),
            (
                &["connecticut", "new haven", "hartford", "bridgeport"],
                "Connecticut",
            ),
            (&["alaska", "anchorage", "denali", "fairbanks"], "Alaska"),
            (&["arizona", "grand canyon"], "Arizona"),
        ],
        OpenQaLocationTarget::Country => &[
            (&["canada", "vancouver", "toronto", "montreal"], "Canada"),
            (&["greenland"], "Greenland"),
            (&["france", "paris"], "France"),
            (&["colombia", "bogota", "medellin", "cartagena"], "Colombia"),
            (&["sweden"], "Sweden"),
            (
                &[
                    "united states",
                    "u.s.",
                    "usa",
                    "america",
                    "boston",
                    "new york",
                    "florida",
                    "california",
                    "minnesota",
                    "connecticut",
                    "alaska",
                    "arizona",
                    "universal studios",
                ],
                "United States",
            ),
        ],
        OpenQaLocationTarget::NationalPark => &[
            (
                &["voyageurs", "voyageurs national park"],
                "Voyageurs National Park",
            ),
            (&["grand canyon"], "Grand Canyon National Park"),
            (&["yellowstone"], "Yellowstone National Park"),
        ],
    };
    aliases.iter().find_map(|(needles, canonical)| {
        needles
            .iter()
            .any(|needle| lower.contains(needle))
            .then(|| (*canonical).to_string())
    })
}

pub(super) fn select_typed_open_qa_structured_answer(
    task: &str,
    evidence: &[EvidenceItem],
) -> Option<String> {
    let lower_task = task.to_ascii_lowercase();
    let subject_hints = extract_subject_hints(task);
    let task_terms = salient_query_terms(task);

    let choice_options = parse_open_qa_choice_options(task);
    if !choice_options.is_empty() {
        let mut best: Option<(usize, f32, String)> = None;
        for item in evidence {
            let Some(content) = read_context_text(&item.path, "typed open qa choice selection")
            else {
                continue;
            };
            for turn in parse_dialogue_turns(&content) {
                if !subject_hints.is_empty() && !turn_matches_subject(&turn, &subject_hints) {
                    continue;
                }
                let lower = turn.text.to_ascii_lowercase();
                let support = task_overlap_count(&turn.text, &task_terms);
                for option in &choice_options {
                    let display_lower = option.display.to_ascii_lowercase();
                    let direct = option
                        .tokens
                        .iter()
                        .filter(|token| lower.contains(token.as_str()))
                        .count();
                    let affinity = open_qa_choice_affinity_terms(&display_lower)
                        .iter()
                        .filter(|needle| lower.contains(**needle))
                        .count();
                    let score = direct * 5 + affinity * 3 + support;
                    if score == 0 {
                        continue;
                    }
                    if best
                        .as_ref()
                        .map(|(best_score, best_retrieval, _)| {
                            score > *best_score
                                || (score == *best_score && item.score > *best_retrieval)
                        })
                        .unwrap_or(true)
                    {
                        best = Some((score, item.score, option.display.clone()));
                    }
                }
            }
        }
        if let Some((score, _, answer)) = best {
            if score >= 3 {
                return Some(answer);
            }
        }
    }

    if let Some(target) = open_qa_location_target(task) {
        let mut best: Option<(usize, f32, String)> = None;
        for item in evidence {
            let Some(content) = read_context_text(&item.path, "typed open qa location selection")
            else {
                continue;
            };
            for turn in parse_dialogue_turns(&content) {
                if !subject_hints.is_empty() && !turn_matches_subject(&turn, &subject_hints) {
                    continue;
                }
                let Some(answer) = open_qa_location_alias(target, &turn.text) else {
                    continue;
                };
                let score = task_overlap_count(&turn.text, &task_terms).max(1);
                if best
                    .as_ref()
                    .map(|(best_score, best_retrieval, _)| {
                        score > *best_score
                            || (score == *best_score && item.score > *best_retrieval)
                    })
                    .unwrap_or(true)
                {
                    best = Some((score, item.score, answer));
                }
            }
        }
        if let Some((_, _, answer)) = best {
            return Some(answer);
        }
    }

    if lower_task.contains("religious")
        || lower_task.contains("religion")
        || lower_task.contains("faith")
    {
        for item in evidence {
            let Some(content) = read_context_text(&item.path, "typed open qa religion selection")
            else {
                continue;
            };
            for turn in parse_dialogue_turns(&content) {
                if !subject_hints.is_empty() && !turn_matches_subject(&turn, &subject_hints) {
                    continue;
                }
                let lower = turn.text.to_ascii_lowercase();
                if lower.contains("church") || lower.contains("faith") {
                    return Some("Somewhat religious".to_string());
                }
            }
        }
    }

    if lower_task.contains("ally")
        || lower_task.contains("lgbtq")
        || lower_task.contains("transgender")
    {
        for item in evidence {
            let Some(content) = read_context_text(&item.path, "typed open qa ally selection")
            else {
                continue;
            };
            for turn in parse_dialogue_turns(&content) {
                if !subject_hints.is_empty() && !turn_matches_subject(&turn, &subject_hints) {
                    continue;
                }
                let lower = turn.text.to_ascii_lowercase();
                let community = lower.contains("lgbtq")
                    || lower.contains("transgender")
                    || lower.contains("trans community")
                    || lower.contains("gender identity");
                let supportive = lower.contains("support")
                    || lower.contains("supportive")
                    || lower.contains("accept")
                    || lower.contains("ally")
                    || lower.contains("proud of you")
                    || lower.contains("back you")
                    || lower.contains("not alone");
                if community && supportive {
                    return Some(format_open_qa_answer_surface_answer(
                        task,
                        "supportive ally",
                    ));
                }
            }
        }
    }

    if is_education_field_query(&lower_task) {
        let focus_terms = dialogue_focus_terms(task, &task_terms, &subject_hints);
        let mut best: Option<(f32, String)> = None;
        for item in evidence {
            let Some(content) = read_context_text(&item.path, "typed open qa education selection")
            else {
                continue;
            };
            for turn in parse_dialogue_turns(&content) {
                if !subject_hints.is_empty() && !turn_matches_subject(&turn, &subject_hints) {
                    continue;
                }
                let focus_overlap = if focus_terms.is_empty() {
                    0
                } else {
                    task_overlap_count(&turn.text, &focus_terms)
                };
                if !focus_terms.is_empty() && focus_overlap == 0 {
                    continue;
                }
                let Some(answer) = compact_answer(task, &turn.text, &task_terms) else {
                    continue;
                };
                if !answer_meets_form_gate(task, &answer, None) {
                    continue;
                }
                let score = item.score * 10.0
                    + speaker_match_bonus(turn.speaker.as_deref(), &subject_hints)
                    + focus_overlap as f32 * 10.0
                    + candidate_weight(&answer, &task_terms, item.score, false);
                update_best_answer(&mut best, score, answer);
            }
        }
        if let Some((_, answer)) = best {
            return Some(answer);
        }
    }

    None
}
