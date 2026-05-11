use super::*;

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

    if key.len() > 5 && (key.ends_with("ied") || key.ends_with("ies")) {
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
