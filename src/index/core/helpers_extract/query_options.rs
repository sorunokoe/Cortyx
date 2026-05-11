use super::*;

pub(in crate::index) fn synthetic_answer_surface_choice_options(
    task: &str,
) -> Vec<SyntheticAnswerSurfaceChoiceOption> {
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
        " answer in ",
        ", ",
    ]
    .iter()
    .find_map(|marker| left_segment.rsplit_once(marker).map(|(_, value)| value))
    .unwrap_or(left_segment);

    [left_raw, right_segment]
        .into_iter()
        .map(synthetic_answer_surface_choice_option)
        .collect::<Option<Vec<_>>>()
        .unwrap_or_default()
}

pub(in crate::index) fn synthetic_conjoined_choice_options(
    task: &str,
) -> Vec<SyntheticAnswerSurfaceChoiceOption> {
    let lower = task.to_ascii_lowercase();
    if !lower.contains(" and ") {
        return Vec::new();
    }

    let tail = task.trim().trim_end_matches('?').trim();
    let Some((left_segment, right_segment)) = tail.rsplit_once(" and ") else {
        return Vec::new();
    };
    let left_raw = [
        " on both the ",
        " on both ",
        " both the ",
        " both ",
        " of ",
        " for ",
        " between ",
        ", ",
    ]
    .iter()
    .find_map(|marker| left_segment.rsplit_once(marker).map(|(_, value)| value))
    .unwrap_or(left_segment);

    [left_raw, right_segment]
        .into_iter()
        .map(synthetic_answer_surface_choice_option)
        .collect::<Option<Vec<_>>>()
        .unwrap_or_default()
}

pub(in crate::index) fn synthetic_answer_surface_choice_option(
    raw: &str,
) -> Option<SyntheticAnswerSurfaceChoiceOption> {
    let display = raw
        .trim()
        .trim_start_matches("the ")
        .trim_start_matches("a ")
        .trim_start_matches("an ")
        .trim_matches(|c: char| matches!(c, '?' | ',' | '.' | ':' | ';'))
        .to_string();
    if display.is_empty() {
        return None;
    }

    let display_lower = display.to_ascii_lowercase();
    let term_keys = synthetic_answer_surface_term_key_set(&synthetic_query_terms(&display_lower));
    if term_keys.is_empty() {
        return None;
    }
    let mut affinity_terms = synthetic_query_terms(&display_lower);
    affinity_terms.extend(
        synthetic_answer_surface_choice_affinity_terms(&display_lower)
            .iter()
            .map(|term| (*term).to_string()),
    );
    let affinity_term_keys = synthetic_answer_surface_term_key_set(&affinity_terms);
    Some(SyntheticAnswerSurfaceChoiceOption {
        display,
        term_keys,
        affinity_term_keys,
    })
}

pub(in crate::index) fn missing_operand_display_phrase(display: &str) -> String {
    let mut phrase = display.trim().to_string();
    loop {
        let lower = phrase.to_ascii_lowercase();
        let mut stripped = false;
        for prefix in [
            "my ",
            "our ",
            "his ",
            "her ",
            "their ",
            "recently ",
            "recent ",
            "new ",
        ] {
            if lower.starts_with(prefix) {
                phrase = phrase[prefix.len()..].trim().to_string();
                stripped = true;
                break;
            }
        }
        if !stripped {
            break;
        }
    }
    phrase
}

pub(in crate::index) fn synthetic_answer_surface_choice_affinity_terms(
    display_lower: &str,
) -> &'static [&'static str] {
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
            "lake",
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
    } else if display_lower == "yes" {
        &["yes", "true", "correct"]
    } else if display_lower == "no" {
        &["no", "not", "never", "false"]
    } else {
        &[]
    }
}

pub(in crate::index) fn synthetic_answer_surface_subject_terms(task: &str) -> Vec<String> {
    const STOP: &[&str] = &[
        "what", "when", "where", "which", "who", "whom", "whose", "why", "how", "does", "did",
        "do", "is", "are", "was", "were", "has", "have", "would", "could", "should", "may",
        "might", "can", "will", "the", "a", "an", "and", "or", "for", "from", "with", "about",
        "into", "after", "before", "between", "around", "through", "this", "that", "these",
        "those",
    ];
    const MONTHS: &[&str] = &[
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ];

    let mut terms = task
        .split(|c: char| !c.is_ascii_alphabetic() && c != '-' && c != '\'')
        .filter_map(|token| {
            let trimmed = token.trim();
            let first = trimmed.chars().next()?;
            if trimmed.len() < 3 || !first.is_ascii_uppercase() {
                return None;
            }
            let lower = trimmed.to_ascii_lowercase();
            if STOP.contains(&lower.as_str()) || MONTHS.contains(&lower.as_str()) {
                return None;
            }
            Some(lower)
        })
        .collect::<Vec<_>>();
    terms.sort();
    terms.dedup();
    terms
}

pub(in crate::index) fn synthetic_answer_surface_expected_type(
    task_lower: &str,
    compose_list_answer: bool,
) -> SyntheticAnswerSurfaceExpectedType {
    if task_lower.starts_with("how long ") || task_lower.contains("how long ago") {
        SyntheticAnswerSurfaceExpectedType::Duration
    } else if task_lower.starts_with("when ")
        || task_contains_any(
            task_lower,
            &[
                "what date",
                "what day",
                "which day",
                "which month",
                "what month",
                "what year",
                "around which",
            ],
        )
    {
        SyntheticAnswerSurfaceExpectedType::Date
    } else if task_lower.starts_with("how many ") || task_lower.starts_with("how much ") {
        SyntheticAnswerSurfaceExpectedType::Count
    } else if task_lower.starts_with("who ") || task_lower.contains(" who ") {
        SyntheticAnswerSurfaceExpectedType::Person
    } else if task_lower.contains("relationship status") {
        SyntheticAnswerSurfaceExpectedType::Status
    } else if task_lower.starts_with("where ")
        || task_contains_any(
            task_lower,
            &[
                " which state",
                " which country",
                " which city",
                " in what country",
                " in which state",
                " in which country",
                " live close to ",
                " close to a beach",
                " close to the mountains",
                " national park",
            ],
        )
    {
        SyntheticAnswerSurfaceExpectedType::Location
    } else if compose_list_answer
        && !task_lower.contains(" name")
        && !task_lower.contains(" names")
        && !task_contains_any(task_lower, &["book", "books", " called "])
    {
        SyntheticAnswerSurfaceExpectedType::ListItem
    } else if compose_list_answer
        || task_lower.contains(" name")
        || task_lower.contains(" names")
        || task_contains_any(task_lower, &["book", "books", " called "])
    {
        SyntheticAnswerSurfaceExpectedType::NameLike
    } else {
        SyntheticAnswerSurfaceExpectedType::Generic
    }
}

pub(in crate::index) fn synthetic_answer_surface_requires_completed_evidence(
    task_lower: &str,
) -> bool {
    task_lower.starts_with("where has ")
        || task_lower.starts_with("where did ")
        || task_lower.starts_with("what did ")
        || task_contains_any(
            task_lower,
            &[
                " participated in",
                " has participated",
                " have participated",
                " attended ",
                " joined ",
                " camped",
                " books has ",
                " books have ",
                " what books",
                " has read",
                " have read",
                " researched",
                " research",
                " tried ",
                " been on ",
                " gone on ",
            ],
        )
}
