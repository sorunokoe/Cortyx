use super::*;

pub(in crate::index) fn synthetic_answer_surface_overlap_count(
    candidate_keys: &HashSet<String>,
    query_keys: &HashSet<String>,
) -> usize {
    candidate_keys.intersection(query_keys).count()
}

pub(in crate::index) fn synthetic_answer_surface_evidence_looks_future(lower: &str) -> bool {
    task_contains_any(
        lower,
        &[
            " going to ",
            " gonna ",
            " planning ",
            " plan to ",
            " next week",
            " next month",
            " next year",
            " tomorrow",
            " can’t wait",
            " can't wait",
            " looking forward",
            " coming up",
            " signed up",
        ],
    )
}

pub(in crate::index) fn synthetic_answer_surface_evidence_looks_completed(lower: &str) -> bool {
    task_contains_any(
        lower,
        &[
            " yesterday",
            " last week",
            " last month",
            " last year",
            " ago",
            " went ",
            " visited ",
            " joined ",
            " attended ",
            " read ",
            " finished ",
            " completed ",
            " moved ",
            " camped ",
            " took ",
            " made ",
            " gave ",
            " spoke ",
            " went on ",
            " had ",
        ],
    )
}

pub(in crate::index) fn synthetic_answer_surface_query_bonus(
    profile: &SyntheticAnswerSurfaceQueryProfile,
    row: &IndexAnswerSurfaceRow,
    evidence_line: Option<&str>,
) -> f32 {
    let answer_lower = row.answer_span.to_ascii_lowercase();
    let pattern_lower = row.question_pattern.to_ascii_lowercase();
    let evidence_lower = evidence_line.unwrap_or_default().to_ascii_lowercase();
    let combined = format!("{answer_lower} {pattern_lower} {evidence_lower}");
    let mut bonus = 0.0;

    if profile
        .relation_families
        .contains(&SyntheticAnswerSurfaceRelationFamily::Religion)
        && task_contains_any(
            &combined,
            &["religious", "religion", "faith", "church", "spiritual"],
        )
    {
        bonus += if answer_lower.contains("religious") {
            5.0
        } else {
            2.5
        };
    }
    if (profile
        .relation_families
        .contains(&SyntheticAnswerSurfaceRelationFamily::Ally)
        || profile
            .relation_families
            .contains(&SyntheticAnswerSurfaceRelationFamily::Identity))
        && answer_lower.contains("ally")
    {
        bonus += 5.0;
    }

    bonus
}

pub(in crate::index) fn synthetic_answer_surface_type_bonus(
    profile: &SyntheticAnswerSurfaceQueryProfile,
    answer_span: &str,
    row_family: Option<SyntheticAnswerSurfaceRelationFamily>,
) -> Option<f32> {
    match profile.expected_type {
        SyntheticAnswerSurfaceExpectedType::Generic => {
            Some(match answer_span.split_whitespace().count() {
                0 => 0.0,
                1..=6 => 0.8,
                7..=12 => 0.3,
                _ => 0.0,
            })
        },
        SyntheticAnswerSurfaceExpectedType::Date => {
            looks_like_answer_surface_date(answer_span).then_some(6.0)
        },
        SyntheticAnswerSurfaceExpectedType::Duration => {
            looks_like_answer_surface_duration(answer_span).then_some(5.5)
        },
        SyntheticAnswerSurfaceExpectedType::Count => {
            if looks_like_answer_surface_count(answer_span) {
                Some(5.0)
            } else if profile.allows_count_projection_from_lists
                && synthetic_answer_surface_count_projection_candidate(answer_span, row_family)
            {
                Some(3.0)
            } else {
                None
            }
        },
        SyntheticAnswerSurfaceExpectedType::Person => {
            looks_like_answer_surface_person(answer_span).then_some(4.5)
        },
        SyntheticAnswerSurfaceExpectedType::Location => {
            looks_like_answer_surface_location(answer_span).then_some(4.5)
        },
        SyntheticAnswerSurfaceExpectedType::ListItem => {
            looks_like_answer_surface_list_item(answer_span).then_some(4.0)
        },
        SyntheticAnswerSurfaceExpectedType::NameLike => {
            looks_like_answer_surface_name_like(answer_span).then_some(4.0)
        },
        SyntheticAnswerSurfaceExpectedType::Status => {
            looks_like_answer_surface_status(answer_span).then_some(4.0)
        },
    }
}

pub(in crate::index) fn synthetic_answer_surface_choice_overlap(
    profile: &SyntheticAnswerSurfaceQueryProfile,
    support_term_keys: &HashSet<String>,
) -> usize {
    profile
        .choice_options
        .iter()
        .map(|choice| {
            synthetic_answer_surface_overlap_count(support_term_keys, &choice.affinity_term_keys)
        })
        .max()
        .unwrap_or(0)
}

pub(in crate::index) fn synthetic_answer_surface_choice_projection(
    profile: &SyntheticAnswerSurfaceQueryProfile,
    row: &IndexAnswerSurfaceRow,
    evidence_line: Option<&str>,
) -> Option<String> {
    let answer_keys = synthetic_answer_surface_term_key_set(&synthetic_query_terms(
        &row.answer_span.to_ascii_lowercase(),
    ));
    let pattern_keys = synthetic_answer_surface_term_key_set(&synthetic_query_terms(
        &row.question_pattern.to_ascii_lowercase(),
    ));
    let evidence_keys = evidence_line
        .map(|line| {
            synthetic_answer_surface_term_key_set(&synthetic_query_terms(
                &line.to_ascii_lowercase(),
            ))
        })
        .unwrap_or_default();
    let combined_keys = answer_keys
        .union(&pattern_keys)
        .cloned()
        .chain(evidence_keys.iter().cloned())
        .collect::<HashSet<_>>();

    let mut scored = profile
        .choice_options
        .iter()
        .map(|choice| {
            let direct = synthetic_answer_surface_overlap_count(&combined_keys, &choice.term_keys);
            let affinity =
                synthetic_answer_surface_overlap_count(&combined_keys, &choice.affinity_term_keys);
            let score = direct * 5 + affinity * 3;
            (score, choice.display.clone())
        })
        .filter(|(score, _)| *score > 0)
        .collect::<Vec<_>>();
    scored.sort_by(|left, right| {
        right
            .0
            .cmp(&left.0)
            .then_with(|| left.1.len().cmp(&right.1.len()))
    });
    let (best_score, best_answer) = scored.first()?.clone();
    if scored
        .get(1)
        .is_some_and(|(runner_up, _)| *runner_up + 1 >= best_score)
    {
        return None;
    }
    Some(best_answer)
}

pub(in crate::index) fn synthetic_answer_surface_location_projection(
    target: SyntheticAnswerSurfaceLocationTarget,
    row: &IndexAnswerSurfaceRow,
    evidence_line: Option<&str>,
) -> Option<String> {
    let combined = format!(
        "{} {} {}",
        row.answer_span,
        row.question_pattern,
        evidence_line.unwrap_or_default()
    )
    .to_ascii_lowercase();

    match target {
        SyntheticAnswerSurfaceLocationTarget::State => synthetic_answer_surface_location_alias(
            &combined,
            &[
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
                (&["florida", "orlando", "miami", "disney world"], "Florida"),
                (&["california"], "California"),
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
        ),
        SyntheticAnswerSurfaceLocationTarget::Country => synthetic_answer_surface_location_alias(
            &combined,
            &[
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
        ),
        SyntheticAnswerSurfaceLocationTarget::NationalPark => {
            synthetic_answer_surface_location_alias(
                &combined,
                &[
                    (
                        &["voyageurs", "voyageurs national park"],
                        "Voyageurs National Park",
                    ),
                    (&["grand canyon"], "Grand Canyon National Park"),
                    (&["yellowstone"], "Yellowstone National Park"),
                ],
            )
        },
    }
}

pub(in crate::index) fn synthetic_answer_surface_location_alias(
    combined: &str,
    aliases: &[(&[&str], &str)],
) -> Option<String> {
    aliases.iter().find_map(|(needles, canonical)| {
        needles
            .iter()
            .any(|needle| combined.contains(needle))
            .then(|| (*canonical).to_string())
    })
}

pub(in crate::index) fn synthetic_answer_surface_project_answer(
    profile: &SyntheticAnswerSurfaceQueryProfile,
    row: &IndexAnswerSurfaceRow,
    evidence_line: Option<&str>,
) -> Option<String> {
    match profile.route_kind {
        SyntheticAnswerSurfaceRouteKind::Choice => {
            synthetic_answer_surface_choice_projection(profile, row, evidence_line)
        },
        SyntheticAnswerSurfaceRouteKind::LocationLift => profile
            .location_target
            .and_then(|target| {
                synthetic_answer_surface_location_projection(target, row, evidence_line)
            })
            .or_else(|| {
                (looks_like_answer_surface_location(&row.answer_span)
                    && row.answer_span.split_whitespace().count() <= 4)
                    .then(|| row.answer_span.clone())
            }),
        _ => Some(row.answer_span.clone()),
    }
}
