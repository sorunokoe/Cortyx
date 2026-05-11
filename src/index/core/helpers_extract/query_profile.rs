use super::*;

pub(in crate::index) fn synthetic_answer_surface_query_profile(
    task: &str,
    task_lower: &str,
    task_terms: &[String],
    compose_list_answer: bool,
) -> SyntheticAnswerSurfaceQueryProfile {
    const OPEN_QA_FILLER: &[&str] = &[
        "would",
        "could",
        "should",
        "can",
        "will",
        "may",
        "might",
        "likely",
        "probably",
        "possibly",
        "potentially",
        "considered",
        "still",
        "more",
        "most",
        "less",
        "least",
        "another",
        "kind",
        "sort",
        "thing",
        "things",
        "personality",
        "trait",
        "traits",
        "additional",
        "alternative",
        "popular",
        "based",
        "around",
    ];
    let subject_terms = synthetic_answer_surface_subject_terms(task);
    let subject_term_keys = synthetic_answer_surface_term_key_set(&subject_terms);
    let choice_options = synthetic_answer_surface_choice_options(task);
    let location_target = synthetic_answer_surface_location_target(task_lower);
    let route_kind = if !choice_options.is_empty() {
        SyntheticAnswerSurfaceRouteKind::Choice
    } else if location_target.is_some() {
        SyntheticAnswerSurfaceRouteKind::LocationLift
    } else if synthetic_answer_surface_is_typed_open_qa_query(task_lower) {
        SyntheticAnswerSurfaceRouteKind::YesNo
    } else {
        SyntheticAnswerSurfaceRouteKind::Default
    };
    let mut anchor_terms = task_terms
        .iter()
        .filter(|term| {
            !OPEN_QA_FILLER.contains(&term.as_str())
                && !choice_options.iter().any(|option| {
                    option
                        .term_keys
                        .contains(&synthetic_answer_surface_term_key(term))
                })
                && (subject_terms.iter().any(|subject| subject == *term)
                    || term.len() >= 4
                    || term.chars().any(|c| c.is_ascii_digit()))
        })
        .cloned()
        .collect::<Vec<_>>();
    if anchor_terms.is_empty() {
        anchor_terms = task_terms
            .iter()
            .filter(|term| !OPEN_QA_FILLER.contains(&term.as_str()))
            .cloned()
            .collect();
    }
    if anchor_terms.is_empty() {
        anchor_terms = task_terms.to_vec();
    }
    anchor_terms.sort();
    anchor_terms.dedup();
    let anchor_term_keys = synthetic_answer_surface_term_key_set(&anchor_terms);
    let relation_term_keys = anchor_term_keys
        .difference(&subject_term_keys)
        .cloned()
        .collect::<HashSet<_>>();
    let expected_type = synthetic_answer_surface_expected_type(task_lower, compose_list_answer);
    let (relation_families, strict_relation_family_match) =
        synthetic_answer_surface_query_relation_families(task_lower);

    SyntheticAnswerSurfaceQueryProfile {
        task_term_keys: synthetic_answer_surface_term_key_set(task_terms),
        subject_term_keys,
        anchor_term_keys,
        relation_term_keys,
        expected_type,
        route_kind,
        choice_options,
        location_target,
        requires_strict_anchor_overlap: !matches!(
            route_kind,
            SyntheticAnswerSurfaceRouteKind::Choice
        ),
        requires_completed_evidence: synthetic_answer_surface_requires_completed_evidence(
            task_lower,
        ),
        strict_relation_family_match,
        relation_families,
        allows_count_projection_from_lists: matches!(
            expected_type,
            SyntheticAnswerSurfaceExpectedType::Count
        ) && compose_list_answer,
    }
}

pub(in crate::index) fn synthetic_answer_surface_query_relation_families(
    task_lower: &str,
) -> (HashSet<SyntheticAnswerSurfaceRelationFamily>, bool) {
    let mut families = HashSet::new();
    let mut strict = false;

    let mut push_strict = |family| {
        families.insert(family);
        strict = true;
    };

    if task_contains_any(
        task_lower,
        &["move from", "moved from", "home country", "origin country"],
    ) {
        push_strict(SyntheticAnswerSurfaceRelationFamily::Origin);
    } else if task_lower.starts_with("how long ")
        && task_contains_any(task_lower, &["group of friends", "support system"])
    {
        push_strict(SyntheticAnswerSurfaceRelationFamily::FriendGroupDuration);
    } else if task_lower.starts_with("who ")
        && task_contains_any(
            task_lower,
            &[
                "support",
                "supports",
                "support system",
                "negative experience",
                "my rocks",
            ],
        )
    {
        push_strict(SyntheticAnswerSurfaceRelationFamily::SupportNetwork);
    } else if task_contains_any(
        task_lower,
        &[
            "research",
            "researched",
            "researching",
            "looking into",
            "investigating",
        ],
    ) {
        push_strict(SyntheticAnswerSurfaceRelationFamily::Research);
    } else if task_contains_any(
        task_lower,
        &[
            "career path",
            "career",
            " fields",
            " field",
            "education",
            "pursue",
            "study",
            "job",
            "work in",
        ],
    ) {
        push_strict(SyntheticAnswerSurfaceRelationFamily::Career);
    } else if task_contains_any(
        task_lower,
        &["what books", "which books", " books", "book "],
    ) && task_contains_any(task_lower, &[" read", "reading", "bookshelf", "book"])
    {
        push_strict(SyntheticAnswerSurfaceRelationFamily::Book);
    } else if task_contains_any(
        task_lower,
        &[
            "what events has",
            "which events",
            "events has",
            "events have",
            "events did",
            "in what ways",
        ],
    ) && task_contains_any(
        task_lower,
        &[
            "help children",
            "help kids",
            "help youth",
            "children",
            "kids",
            "youth",
        ],
    ) {
        push_strict(SyntheticAnswerSurfaceRelationFamily::ChildHelpEvent);
    } else if task_contains_any(
        task_lower,
        &[
            "lgbtq",
            "lgbtq+",
            "transgender-specific",
            "transgender community",
            "lgbtq community",
        ],
    ) && task_contains_any(
        task_lower,
        &[
            "event",
            "events",
            "participat",
            "attend",
            "joined",
            "join ",
            "in what ways",
        ],
    ) {
        push_strict(SyntheticAnswerSurfaceRelationFamily::CommunityEvent);
    } else if task_contains_any(task_lower, &["where has ", "where have ", " camped"])
        && task_contains_any(task_lower, &["camp", "camped", "camping"])
    {
        push_strict(SyntheticAnswerSurfaceRelationFamily::CampLocation);
    } else if task_contains_any(
        task_lower,
        &[
            "to destress",
            "to de-stress",
            "self-care",
            "stay distracted",
            "relax",
        ],
    ) {
        push_strict(SyntheticAnswerSurfaceRelationFamily::SelfCareActivity);
    } else if task_contains_any(
        task_lower,
        &[" activities", " activity", "hobbies", "hobby"],
    ) {
        if task_contains_any(
            task_lower,
            &[
                "with her family",
                "with his family",
                "with my family",
                "with their family",
                "with the kids",
                "with my kids",
                "family",
                "kids",
                "children",
                "together",
            ],
        ) {
            push_strict(SyntheticAnswerSurfaceRelationFamily::FamilyActivity);
        } else {
            families.insert(SyntheticAnswerSurfaceRelationFamily::Activity);
            families.insert(SyntheticAnswerSurfaceRelationFamily::FamilyActivity);
            families.insert(SyntheticAnswerSurfaceRelationFamily::SelfCareActivity);
        }
    } else if task_contains_any(
        task_lower,
        &["kids like", "children like", "what do", "what does"],
    ) && task_contains_any(task_lower, &["kids", "children"])
    {
        push_strict(SyntheticAnswerSurfaceRelationFamily::KidsPreference);
    } else if task_contains_any(task_lower, &["paint", "painting", "art does"]) {
        push_strict(SyntheticAnswerSurfaceRelationFamily::PaintSubject);
    } else if task_contains_any(
        task_lower,
        &[
            "member of the lgbtq community",
            "member of the transgender community",
            "ally",
        ],
    ) {
        push_strict(SyntheticAnswerSurfaceRelationFamily::Ally);
    } else if task_contains_any(
        task_lower,
        &["religious", "religion", "faith", "church", "spiritual"],
    ) {
        push_strict(SyntheticAnswerSurfaceRelationFamily::Religion);
    } else if task_lower.contains("relationship status") {
        push_strict(SyntheticAnswerSurfaceRelationFamily::Relationship);
    } else if task_contains_any(
        task_lower,
        &["identity", "transgender woman", "transgender man"],
    ) {
        push_strict(SyntheticAnswerSurfaceRelationFamily::Identity);
    }

    (families, strict)
}

pub(in crate::index) fn synthetic_answer_surface_is_typed_open_qa_query(task_lower: &str) -> bool {
    task_lower.starts_with("would ")
        || task_lower.starts_with("could ")
        || task_lower.starts_with("should ")
        || task_lower.starts_with("can ")
        || task_lower.starts_with("will ")
        || task_lower.starts_with("may ")
        || task_lower.starts_with("might ")
        || task_lower.starts_with("is ")
        || task_lower.starts_with("are ")
        || task_lower.starts_with("was ")
        || task_lower.starts_with("were ")
        || task_lower.starts_with("does ")
        || task_lower.starts_with("do ")
        || task_lower.starts_with("did ")
        || task_lower.starts_with("has ")
        || task_lower.starts_with("have ")
        || task_lower.starts_with("had ")
        || task_lower.starts_with("which ")
        || task_lower.starts_with("what might ")
        || task_lower.starts_with("what would ")
        || task_lower.contains(" likely ")
        || task_lower.contains(" likely be ")
        || task_lower.contains(" considered ")
}

pub(in crate::index) fn synthetic_answer_surface_location_target(
    task_lower: &str,
) -> Option<SyntheticAnswerSurfaceLocationTarget> {
    if task_contains_any(task_lower, &["national park", "which park"]) {
        Some(SyntheticAnswerSurfaceLocationTarget::NationalPark)
    } else if task_lower.starts_with("what state")
        || task_lower.starts_with("which state")
        || task_contains_any(
            task_lower,
            &[
                " in what state",
                " in which state",
                " us state",
                " us states",
            ],
        )
    {
        Some(SyntheticAnswerSurfaceLocationTarget::State)
    } else if task_lower.starts_with("what country")
        || task_lower.starts_with("which country")
        || task_contains_any(
            task_lower,
            &[
                " in what country",
                " in which country",
                " home country",
                "move from",
                "moved from",
                "origin country",
            ],
        )
    {
        Some(SyntheticAnswerSurfaceLocationTarget::Country)
    } else {
        None
    }
}
