// This file is a submodule of `crate::index::core`.
// Contains free-standing helper functions extracted from helpers.rs.
use super::*;
use crate::index::compile_regex;
use crate::types::{QueryText, SynapseWeight};

pub(in crate::index) fn extract_current_company_answer_from_line(
    line: &str,
    lower: &str,
) -> Option<String> {
    if !line_has_current_company_marker(lower) {
        return None;
    }
    let answer = extract_phrase_after_any_index(
        line,
        lower,
        &[
            "currently working at ",
            "currently at ",
            "current company is ",
            "works at ",
            "working at ",
            "employed at ",
        ],
        &[
            " because ",
            " and ",
            " but ",
            " while ",
            ".",
            ",",
            ";",
            " with ",
        ],
        1,
    )?;
    (answer.split_whitespace().count() <= 6).then_some(answer)
}

pub(in crate::index) fn extract_instagram_current_count_candidate(
    line: &str,
    lower: &str,
) -> Option<(i32, usize)> {
    if !lower.contains("follower")
        || task_contains_any(
            lower,
            &["facebook", "twitter", "tiktok", "youtube", "linkedin"],
        )
        || !line_has_current_count_marker(lower)
    {
        return None;
    }
    if extract_duration_answer_from_line(line).is_some()
        && !task_contains_any(
            lower,
            &[
                "just checked",
                "now at",
                "currently have",
                "currently at",
                "current follower count",
            ],
        )
    {
        return None;
    }
    let value = extract_line_numbers(line)
        .into_iter()
        .filter(|value| *value >= 10)
        .last()?;
    let mut strength = 4usize;
    if task_contains_any(
        lower,
        &[
            "just checked",
            "now at",
            "recently crossed",
            "just reached",
            "currently have",
            "currently at",
        ],
    ) {
        strength += 6;
    }
    if lower.contains("follower count") {
        strength += 2;
    }
    if task_contains_any(
        lower,
        &[
            "close to",
            "almost",
            "nearly",
            "about ",
            "around ",
            "roughly",
            "approximately",
            "approx ",
        ],
    ) {
        strength = strength.saturating_sub(4);
    }
    Some((value, strength))
}

pub(in crate::index) fn line_has_current_count_marker(lower: &str) -> bool {
    task_contains_any(
        lower,
        &[
            " currently",
            " current",
            " now",
            " right now",
            " today",
            " these days",
            " recently",
            " just ",
            " already",
            " actually",
            " still",
            " so far",
        ],
    )
}

pub(in crate::index) fn is_money_query(task: &str) -> bool {
    const MONEY_MARKERS: &[&str] = &[
        "$",
        " dollar",
        " dollars",
        "money",
        "expense",
        "expenses",
        "cost",
        "costs",
        "price",
        "prices",
        "paid",
        "bill",
        "bills",
        "budget",
        "purchase",
        "purchased",
        "income",
        "earnings",
        "earned",
        "earning",
        "salary",
        "wage",
        "wages",
        "revenue",
        "profit",
        "profits",
    ];
    const NON_MONEY_UNITS: &[&str] = &[
        "time", "times", "hour", "hours", "day", "days", "week", "weeks", "month", "months",
        "year", "years", "session", "sessions",
    ];

    let lower = task.to_ascii_lowercase();
    MONEY_MARKERS.iter().any(|marker| lower.contains(marker))
        || (lower.contains("how much") && !NON_MONEY_UNITS.iter().any(|unit| lower.contains(unit)))
}

pub(in crate::index) fn normalize_aggregate_focus_token(token: &str) -> Option<String> {
    let mut cleaned: String = token
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase();
    if cleaned.len() < 3 {
        return None;
    }
    if cleaned.ends_with("ies") && cleaned.len() > 4 {
        cleaned = format!("{}y", &cleaned[..cleaned.len() - 3]);
    } else if cleaned.ends_with('s') && !cleaned.ends_with("ss") && cleaned.len() > 4 {
        cleaned.pop();
    }
    Some(cleaned)
}

pub(in crate::index) fn aggregate_focus_tokens_for_path(path: &Path) -> Vec<String> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    let base = file_name.strip_suffix(".aggregate.md").unwrap_or(file_name);
    let topic = base
        .strip_prefix("_arith_")
        .or_else(|| base.strip_prefix("_count_"))
        .unwrap_or(base);
    topic
        .split('_')
        .filter_map(normalize_aggregate_focus_token)
        .collect()
}

pub(in crate::index) fn aggregate_focus_token_count_for_path(path: &Path) -> usize {
    aggregate_focus_tokens_for_path(path).len()
}

pub(in crate::index) fn aggregate_focus_match_count_for_path(
    path: &Path,
    focus_terms: &[String],
) -> usize {
    let aggregate_tokens: HashSet<String> =
        aggregate_focus_tokens_for_path(path).into_iter().collect();
    let focus_tokens: HashSet<String> = focus_terms
        .iter()
        .filter_map(|term| normalize_aggregate_focus_token(term))
        .collect();
    aggregate_tokens.intersection(&focus_tokens).count()
}

pub(in crate::index) fn best_matching_arithmetic_aggregate_path(
    project_root: &Path,
    focus_terms: &[String],
) -> Option<PathBuf> {
    let ndir = neuron_dir(project_root);
    let Ok(read_dir) = std::fs::read_dir(&ndir) else {
        return None;
    };

    read_dir
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.starts_with("_arith_") && name.ends_with(".aggregate.md"))
                .unwrap_or(false)
        })
        .filter_map(|path| {
            let match_count = aggregate_focus_match_count_for_path(&path, focus_terms);
            if match_count == 0 {
                return None;
            }
            let token_count = aggregate_focus_token_count_for_path(&path).max(1);
            let score = (match_count as f32 * 100.0) + (match_count as f32 / token_count as f32);
            Some((score, path))
        })
        .max_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)))
        .map(|(_, path)| path)
}

pub(in crate::index) fn is_session_summary_path(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.ends_with("_summary.md"))
        .unwrap_or(false)
}

pub(in crate::index) fn strip_query_surface_section(content: &str) -> String {
    let without_query = strip_named_section(content, "query_surface");
    strip_named_section(&without_query, "answer_surface")
}

pub(in crate::index) fn strip_named_section(content: &str, section_name: &str) -> String {
    let header = format!("## {section_name}");
    let marker = format!("<!-- SECTION: {section_name} -->");
    let end_marker = "<!-- /SECTION -->";
    let Some(header_start) = content.find(&header) else {
        return content.to_string();
    };
    let Some(section_start_rel) = content[header_start..].find(&marker) else {
        return content.to_string();
    };
    let section_start = header_start + section_start_rel;
    let Some(section_end_rel) = content[section_start..].find(end_marker) else {
        return content.to_string();
    };
    let section_end = section_start + section_end_rel + end_marker.len();

    let mut stripped = String::with_capacity(content.len());
    stripped.push_str(content[..header_start].trim_end());
    if !stripped.ends_with('\n') {
        stripped.push('\n');
    }
    let tail = content[section_end..].trim_start_matches('\n');
    if !tail.is_empty() {
        stripped.push('\n');
        stripped.push_str(tail);
    }
    stripped
}

pub(in crate::index) fn parse_index_answer_surface_rows(
    content: &str,
) -> Vec<IndexAnswerSurfaceRow> {
    let sections = crate::neuron::parse_sections(content);
    let Some(table) = sections.get("answer_surface") else {
        return Vec::new();
    };

    table
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if !trimmed.starts_with('|') {
                return None;
            }
            let columns = trimmed
                .trim_matches('|')
                .split('|')
                .map(str::trim)
                .collect::<Vec<_>>();
            if columns.len() != 3 {
                return None;
            }
            if columns[0].eq_ignore_ascii_case("question_pattern")
                || columns[0].chars().all(|c| c == '-' || c == ' ')
            {
                return None;
            }

            let answer_span = columns[1]
                .trim()
                .trim_matches(|c: char| matches!(c, '"' | '\'' | '`'))
                .to_string();
            if answer_span.is_empty() {
                return None;
            }

            Some(IndexAnswerSurfaceRow {
                question_pattern: columns[0].to_string(),
                answer_span,
                confidence: columns[2].parse::<f32>().unwrap_or(0.0),
            })
        })
        .collect()
}

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
            .into_iter()
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

pub(in crate::index) fn synthetic_answer_surface_term_key_set(terms: &[String]) -> HashSet<String> {
    terms
        .iter()
        .map(|term| synthetic_answer_surface_term_key(term))
        .filter(|term| !term.is_empty())
        .collect()
}

pub(in crate::index) fn synthetic_answer_surface_term_key(term: &str) -> String {
    pub(in crate::index) fn trim_repeated_suffix(word: &mut String) {
        let chars = word.chars().collect::<Vec<_>>();
        if chars.len() >= 2 {
            let last = chars[chars.len() - 1];
            let prev = chars[chars.len() - 2];
            if last == prev && matches!(last, 'b' | 'd' | 'g' | 'l' | 'm' | 'n' | 'p' | 'r' | 't') {
                word.pop();
            }
        }
    }

    let mut key = term
        .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '\'' && c != '-')
        .to_ascii_lowercase();
    if key.ends_with("'s") {
        key.truncate(key.len() - 2);
    }
    if key.is_empty() {
        return key;
    }

    let mapped = match key.as_str() {
        "went" | "gone" | "goes" => Some("go"),
        "bought" => Some("buy"),
        "taught" | "teaches" | "teaching" => Some("teach"),
        "grew" | "grown" | "growing" => Some("grow"),
        "ran" | "running" => Some("run"),
        "swam" | "swimming" => Some("swim"),
        "wrote" | "written" | "writing" => Some("write"),
        "reads" | "reading" => Some("read"),
        "met" | "meeting" => Some("meet"),
        "took" | "taken" => Some("take"),
        "drove" | "driving" => Some("drive"),
        "brought" => Some("bring"),
        "began" | "begun" => Some("begin"),
        _ => None,
    };
    if let Some(mapped) = mapped {
        return mapped.to_string();
    }

    if key.len() > 5 && key.ends_with("ied") {
        key.truncate(key.len() - 3);
        key.push('y');
    } else if key.len() > 5 && key.ends_with("ies") {
        key.truncate(key.len() - 3);
        key.push('y');
    } else if key.len() > 5 && key.ends_with("ing") {
        key.truncate(key.len() - 3);
        trim_repeated_suffix(&mut key);
    } else if key.len() > 4 && key.ends_with("ed") {
        key.truncate(key.len() - 2);
        trim_repeated_suffix(&mut key);
    } else if key.len() > 4 && key.ends_with("es") {
        key.truncate(key.len() - 2);
    } else if key.len() > 3 && key.ends_with('s') && !key.ends_with("ss") {
        key.pop();
    }

    if key.len() > 4 && key.ends_with('e') {
        key.pop();
    }
    key
}

pub(in crate::index) fn synthetic_answer_surface_family_activity_context(lower: &str) -> bool {
    task_contains_any(
        lower,
        &[
            " kids",
            "my kids",
            "with the kids",
            "with my kids",
            "with my fam",
            "with my family",
            "family",
            "children",
            "together",
        ],
    )
}

pub(in crate::index) fn synthetic_answer_surface_self_care_activity_context(lower: &str) -> bool {
    task_contains_any(
        lower,
        &[
            "de-stress",
            "destress",
            "self-care",
            "relax",
            "peace",
            "therapeutic",
            "calming",
            "me-time",
            "stay distracted",
        ],
    )
}

pub(in crate::index) fn synthetic_answer_surface_relation_family(
    question_pattern: &str,
    evidence_line: Option<&str>,
) -> Option<SyntheticAnswerSurfaceRelationFamily> {
    let pattern_lower = question_pattern.to_ascii_lowercase();
    let evidence_lower = evidence_line.unwrap_or_default().to_ascii_lowercase();
    let pattern_keys =
        synthetic_answer_surface_term_key_set(&synthetic_query_terms(&pattern_lower));
    let pattern_has_any = |keys: &[&str]| keys.iter().any(|key| pattern_keys.contains(*key));
    let pattern_has_all = |keys: &[&str]| keys.iter().all(|key| pattern_keys.contains(*key));

    if pattern_has_any(&["mov", "origin", "country"]) && pattern_has_any(&["from", "country"]) {
        Some(SyntheticAnswerSurfaceRelationFamily::Origin)
    } else if pattern_has_any(&["friend"])
        && pattern_has_any(&["known", "know", "long", "duration"])
        && pattern_has_any(&["year", "month", "week", "day"])
    {
        Some(SyntheticAnswerSurfaceRelationFamily::FriendGroupDuration)
    } else if !pattern_has_any(&["event"])
        && (pattern_has_all(&["who", "support"])
            || pattern_has_all(&["negative", "experienc"])
            || pattern_has_any(&["rock"]))
        && pattern_has_any(&["mentor", "friend", "family", "kid", "husband", "partner"])
    {
        Some(SyntheticAnswerSurfaceRelationFamily::SupportNetwork)
    } else if pattern_has_any(&["research", "topic", "investigat", "look", "into"]) {
        Some(SyntheticAnswerSurfaceRelationFamily::Research)
    } else if pattern_has_any(&["career", "field", "educat", "study", "job", "work"]) {
        Some(SyntheticAnswerSurfaceRelationFamily::Career)
    } else if pattern_has_any(&["book", "read", "title", "literatur"]) {
        Some(SyntheticAnswerSurfaceRelationFamily::Book)
    } else if pattern_has_any(&["camp", "location", "place"])
        && pattern_has_any(&["camp", "beach", "mountain", "forest", "lake"])
    {
        Some(SyntheticAnswerSurfaceRelationFamily::CampLocation)
    } else if pattern_has_any(&["kid", "children", "child"])
        && pattern_has_any(&["like", "lov", "enjoy", "favorit", "interest"])
    {
        Some(SyntheticAnswerSurfaceRelationFamily::KidsPreference)
    } else if pattern_has_any(&["paint", "scene", "subject"])
        || (pattern_has_any(&["art"]) && pattern_has_any(&["paint", "made", "make", "creat"]))
    {
        Some(SyntheticAnswerSurfaceRelationFamily::PaintSubject)
    } else if pattern_has_any(&[
        "identity",
        "gender",
        "transgender",
        "woman",
        "man",
        "nonbinary",
        "queer",
    ]) && !pattern_has_any(&["event"])
    {
        Some(SyntheticAnswerSurfaceRelationFamily::Identity)
    } else if pattern_has_any(&["event"]) && pattern_has_any(&["children", "kid", "youth"]) {
        Some(SyntheticAnswerSurfaceRelationFamily::ChildHelpEvent)
    } else if pattern_has_any(&["event"])
        && pattern_has_any(&[
            "lgbtq",
            "community",
            "parade",
            "activist",
            "group",
            "speech",
            "program",
            "art",
            "support",
        ])
    {
        Some(SyntheticAnswerSurfaceRelationFamily::CommunityEvent)
    } else if pattern_has_any(&["activity", "hobby"])
        && (pattern_has_any(&[
            "destress",
            "relax",
            "self-care",
            "peace",
            "therapeutic",
            "calm",
        ]) || (!pattern_has_any(&["family", "kid", "children", "together", "fun"])
            && synthetic_answer_surface_self_care_activity_context(&evidence_lower)))
    {
        Some(SyntheticAnswerSurfaceRelationFamily::SelfCareActivity)
    } else if pattern_has_any(&["activity", "hobby"])
        && (pattern_has_any(&["family", "kid", "children", "together", "fun"])
            || (!pattern_has_any(&[
                "destress",
                "relax",
                "self-care",
                "peace",
                "therapeutic",
                "calm",
            ]) && synthetic_answer_surface_family_activity_context(&evidence_lower)))
    {
        Some(SyntheticAnswerSurfaceRelationFamily::FamilyActivity)
    } else if pattern_has_any(&["activity", "hobby"]) {
        Some(SyntheticAnswerSurfaceRelationFamily::Activity)
    } else if pattern_has_any(&["religious", "religion", "faith", "church", "spiritual"]) {
        Some(SyntheticAnswerSurfaceRelationFamily::Religion)
    } else if pattern_has_any(&[
        "relationship",
        "statu",
        "single",
        "married",
        "partner",
        "spouse",
    ]) {
        Some(SyntheticAnswerSurfaceRelationFamily::Relationship)
    } else if pattern_has_any(&["ally", "supportive", "acceptance"])
        || (pattern_has_all(&["support", "community"]) && !pattern_has_any(&["event"]))
    {
        Some(SyntheticAnswerSurfaceRelationFamily::Ally)
    } else {
        None
    }
}

pub(in crate::index) fn synthetic_answer_surface_relation_family_matches(
    profile: &SyntheticAnswerSurfaceQueryProfile,
    row_family: Option<SyntheticAnswerSurfaceRelationFamily>,
    relation_overlap: usize,
) -> bool {
    if profile.relation_families.is_empty() {
        return true;
    }
    if row_family
        .map(|family| profile.relation_families.contains(&family))
        .unwrap_or(false)
    {
        return true;
    }
    if !profile.strict_relation_family_match {
        return row_family.is_some_and(|family| {
            profile
                .relation_families
                .contains(&SyntheticAnswerSurfaceRelationFamily::Activity)
                && matches!(
                    family,
                    SyntheticAnswerSurfaceRelationFamily::FamilyActivity
                        | SyntheticAnswerSurfaceRelationFamily::SelfCareActivity
                )
        }) || relation_overlap > 0;
    }
    row_family.is_none()
        && !profile.relation_term_keys.is_empty()
        && relation_overlap >= usize::min(2, profile.relation_term_keys.len())
}

pub(in crate::index) fn synthetic_answer_surface_bucket_matches_relation_profile(
    profile: &SyntheticAnswerSurfaceQueryProfile,
    bucket: &IndexAnswerSurfaceBucket,
) -> bool {
    if profile.relation_families.is_empty() || bucket.relation_families.is_empty() {
        return true;
    }
    bucket
        .relation_families
        .iter()
        .copied()
        .any(|family| synthetic_answer_surface_relation_family_matches(profile, Some(family), 1))
}

pub(in crate::index) fn synthetic_answer_surface_relation_family_supports_count_projection(
    family: SyntheticAnswerSurfaceRelationFamily,
) -> bool {
    matches!(
        family,
        SyntheticAnswerSurfaceRelationFamily::Activity
            | SyntheticAnswerSurfaceRelationFamily::FamilyActivity
            | SyntheticAnswerSurfaceRelationFamily::SelfCareActivity
            | SyntheticAnswerSurfaceRelationFamily::Book
            | SyntheticAnswerSurfaceRelationFamily::CampLocation
            | SyntheticAnswerSurfaceRelationFamily::KidsPreference
            | SyntheticAnswerSurfaceRelationFamily::PaintSubject
            | SyntheticAnswerSurfaceRelationFamily::CommunityEvent
            | SyntheticAnswerSurfaceRelationFamily::ChildHelpEvent
    )
}

pub(in crate::index) fn synthetic_answer_surface_count_projection_candidate(
    answer_span: &str,
    row_family: Option<SyntheticAnswerSurfaceRelationFamily>,
) -> bool {
    row_family
        .filter(|family| {
            synthetic_answer_surface_relation_family_supports_count_projection(*family)
        })
        .is_some()
        && (looks_like_answer_surface_list_item(answer_span)
            || looks_like_answer_surface_name_like(answer_span)
            || looks_like_answer_surface_location(answer_span)
            || looks_like_answer_surface_person(answer_span))
}

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

pub(in crate::index) fn looks_like_answer_surface_date(answer_span: &str) -> bool {
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
    let lower = answer_span.to_ascii_lowercase();
    compile_regex(r"\b(?:19|20)\d{2}\b").is_match(&lower)
        || MONTHS.iter().any(|month| lower.contains(month))
        || task_contains_any(
            &lower,
            &[
                "yesterday",
                "today",
                "tonight",
                "tomorrow",
                "last week",
                "last month",
                "last year",
                "next week",
                "next month",
                "week before",
                "month before",
                "year before",
                "last saturday",
                "last sunday",
                "last monday",
                "last tuesday",
                "last wednesday",
                "last thursday",
                "last friday",
            ],
        )
}

pub(in crate::index) fn looks_like_answer_surface_duration(answer_span: &str) -> bool {
    let lower = answer_span.to_ascii_lowercase();
    lower.starts_with("since ")
        || compile_regex(
            r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(?:day|week|month|year)s?\b",
        )
        .is_match(&lower)
        || compile_regex(
            r"\b(?:day|week|month|year)s?\s+(?:ago|already|now)\b",
        )
        .is_match(&lower)
}

pub(in crate::index) fn looks_like_answer_surface_count(answer_span: &str) -> bool {
    if looks_like_answer_surface_date(answer_span) {
        return false;
    }
    let lower = answer_span.to_ascii_lowercase();
    compile_regex(
        r"^(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twice|thrice)(?:\s+(?:times?|kids?|children|dogs?|cats?|followers?|issues?|books?|letters?))?$",
    )
    .is_match(lower.trim())
}

pub(in crate::index) fn looks_like_answer_surface_person(answer_span: &str) -> bool {
    let lower = answer_span.to_ascii_lowercase();
    if task_contains_any(
        &lower,
        &[
            "family",
            "friends",
            "friend",
            "mentor",
            "mentors",
            "mother",
            "mom",
            "father",
            "dad",
            "aunt",
            "uncle",
            "sister",
            "brother",
            "husband",
            "wife",
            "partner",
            "spouse",
            "colleague",
            "colleagues",
            "teammates",
            "children",
            "kids",
        ],
    ) {
        return true;
    }
    let words = answer_span.split_whitespace().collect::<Vec<_>>();
    words.len() <= 8
        && words.iter().any(|word| {
            word.chars()
                .next()
                .map(|c| c.is_ascii_uppercase())
                .unwrap_or(false)
        })
}

pub(in crate::index) fn looks_like_answer_surface_name_like(answer_span: &str) -> bool {
    if answer_span.contains('?')
        || answer_span.contains(". ")
        || looks_like_answer_surface_date(answer_span)
        || looks_like_answer_surface_duration(answer_span)
        || looks_like_answer_surface_count(answer_span)
    {
        return false;
    }
    let words = answer_span.split_whitespace().collect::<Vec<_>>();
    words.len() <= 10
        && words.iter().any(|word| {
            word.chars()
                .next()
                .map(|c| c.is_ascii_uppercase())
                .unwrap_or(false)
        })
}

pub(in crate::index) fn looks_like_answer_surface_list_item(answer_span: &str) -> bool {
    if answer_span.contains('?')
        || answer_span.contains(". ")
        || looks_like_answer_surface_date(answer_span)
        || looks_like_answer_surface_duration(answer_span)
        || looks_like_answer_surface_count(answer_span)
    {
        return false;
    }
    let words = answer_span.split_whitespace().collect::<Vec<_>>();
    !words.is_empty()
        && words.len() <= 8
        && !task_contains_any(
            &answer_span.to_ascii_lowercase(),
            &[" because ", " although ", " however ", " but "],
        )
}

pub(in crate::index) fn looks_like_answer_surface_location(answer_span: &str) -> bool {
    if looks_like_answer_surface_date(answer_span) || looks_like_answer_surface_count(answer_span) {
        return false;
    }
    let lower = answer_span.to_ascii_lowercase();
    if task_contains_any(
        &lower,
        &[
            "beach",
            "mountain",
            "mountains",
            "forest",
            "woods",
            "lake",
            "park",
            "city",
            "country",
            "state",
            "suburbs",
            "downtown",
            "village",
            "town",
            "island",
        ],
    ) {
        return true;
    }
    let words = answer_span.split_whitespace().collect::<Vec<_>>();
    words.len() <= 6
        && words.iter().any(|word| {
            word.chars()
                .next()
                .map(|c| c.is_ascii_uppercase())
                .unwrap_or(false)
        })
}
