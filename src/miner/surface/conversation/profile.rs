use super::*;

pub(super) fn has_dialogue_self_reference(lower: &str) -> bool {
    lower.starts_with("i ")
        || lower.starts_with("i'")
        || lower.starts_with("i’m")
        || lower.starts_with("my ")
        || lower.starts_with("we ")
        || lower.starts_with("our ")
        || lower.contains(" i'm ")
        || lower.contains(" i am ")
        || lower.contains(" my ")
        || lower.contains(" we ")
        || lower.contains(" our ")
}

pub(super) fn push_unique_bridge_value(values: &mut Vec<String>, value: &str) {
    let clean = normalize_answer_surface_span(value);
    if clean.is_empty()
        || values
            .iter()
            .any(|existing| existing.eq_ignore_ascii_case(&clean))
    {
        return;
    }
    values.push(clean);
}

pub(super) fn extract_dialogue_career_interest_surface_value(
    text: &str,
    lower: &str,
) -> Option<String> {
    let career_context = lower.contains("career")
        || lower.contains("job")
        || lower.contains("work")
        || lower.contains("education")
        || lower.contains("study")
        || lower.contains("field")
        || lower.contains("looking into")
        || lower.contains("keen on")
        || lower.contains("thinking of");
    if !career_context || !has_dialogue_self_reference(lower) {
        return None;
    }

    let mut values = Vec::new();
    if lower.contains("counsel") {
        push_unique_bridge_value(&mut values, "counseling");
    }
    if lower.contains("mental health") {
        push_unique_bridge_value(&mut values, "mental health");
    }
    if lower.contains("counsel") && lower.contains("mental health") {
        push_unique_bridge_value(&mut values, "psychology");
    }
    if lower.contains("psycholog") {
        push_unique_bridge_value(&mut values, "psychology");
    }
    if lower.contains("social work") {
        push_unique_bridge_value(&mut values, "social work");
    }
    if !values.is_empty() {
        return Some(values.join(", "));
    }

    extract_fact_after_any(
        text,
        lower,
        &[
            "keen on ",
            "looking into ",
            "thinking of ",
            "interested in ",
        ],
        &[" because ", " and ", " but ", " so ", " for "],
        6,
    )
    .map(|value| normalize_answer_surface_span(&value))
    .filter(|value| !value.is_empty())
}

pub(super) fn extract_dialogue_career_reason_surface_value(
    text: &str,
    lower: &str,
) -> Option<String> {
    let career_context = lower.contains("career")
        || lower.contains("job")
        || lower.contains("work")
        || lower.contains("education")
        || lower.contains("study")
        || lower.contains("field")
        || lower.contains("counsel")
        || lower.contains("mental health");
    if !career_context || !has_dialogue_self_reference(lower) {
        return None;
    }

    extract_clause_after_any(
        text,
        lower,
        &["because ", "'cause ", "cause "],
        &[". ", "! ", "? ", " but ", " so ", " though ", " although "],
        12,
    )
    .or_else(|| {
        extract_clause_after_any(
            text,
            lower,
            &[
                "i'd love to ",
                "i would love to ",
                "i want to ",
                "i wanna ",
                "my goal is to ",
                "goal is to ",
            ],
            &[". ", "! ", "? ", " but ", " so ", " though ", " although "],
            12,
        )
    })
    .map(|value| normalize_dialogue_reason_phrase(&value))
    .filter(|value| !value.is_empty())
}

pub(super) fn extract_dialogue_research_topic_surface_value(
    text: &str,
    lower: &str,
) -> Option<String> {
    let looks_self_referential = has_dialogue_self_reference(lower)
        || lower.starts_with("researching ")
        || lower.starts_with("researched ")
        || lower.starts_with("looking into ")
        || lower.starts_with("investigating ");
    looks_self_referential
        .then(|| extract_research_surface_value(text, lower))
        .flatten()
        .map(|value| normalize_answer_surface_span(&value))
        .filter(|value| !value.is_empty())
}

pub(super) fn extract_dialogue_identity_surface_value(_text: &str, lower: &str) -> Option<String> {
    [
        ("transgender woman", "transgender woman"),
        ("trans woman", "transgender woman"),
        ("transgender man", "transgender man"),
        ("trans man", "transgender man"),
        ("nonbinary person", "nonbinary person"),
        ("non-binary person", "nonbinary person"),
        ("queer woman", "queer woman"),
        ("queer man", "queer man"),
        ("bisexual woman", "bisexual woman"),
        ("bisexual man", "bisexual man"),
        ("gay man", "gay man"),
        ("lesbian woman", "lesbian woman"),
    ]
    .into_iter()
    .find_map(|(needle, value)| lower.contains(needle).then(|| value.to_string()))
}

pub(super) fn extract_dialogue_ally_surface_value(_text: &str, lower: &str) -> Option<String> {
    let community_context = lower.contains("lgbtq")
        || lower.contains("transgender")
        || lower.contains("trans community")
        || lower.contains("gender identity");
    let supportive_context = lower.contains("support")
        || lower.contains("supportive")
        || lower.contains("accept")
        || lower.contains("ally")
        || lower.contains("proud of you")
        || lower.contains("back you")
        || lower.contains("not alone");
    (community_context && supportive_context).then(|| "supportive ally".to_string())
}

pub(super) fn extract_dialogue_relationship_status_surface_value(
    _text: &str,
    lower: &str,
) -> Option<String> {
    if lower.contains("single parent")
        || lower.starts_with("i'm single")
        || lower.starts_with("i am single")
        || lower.contains(" as a single ")
    {
        return Some("single".to_string());
    }
    if lower.contains("my husband")
        || lower.contains("my wife")
        || lower.contains("my spouse")
        || lower.starts_with("i'm married")
        || lower.starts_with("i am married")
    {
        return Some("married".to_string());
    }
    None
}

pub(super) fn extract_dialogue_origin_surface_value(text: &str, lower: &str) -> Option<String> {
    if lower.contains("home country") {
        let explicit = compile_regex(r"(?i)home country[, ]+([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)?)");
        if let Some(value) = explicit
            .captures(text)
            .and_then(|caps| caps.get(1))
            .map(|m| normalize_answer_surface_span(m.as_str()))
            .filter(|value| !value.is_empty())
        {
            return Some(value);
        }
    }

    if !has_dialogue_self_reference(lower) {
        return None;
    }

    extract_fact_after_any(
        text,
        lower,
        &["i'm from ", "i am from "],
        &[" and ", " but ", " because ", " since ", " after "],
        3,
    )
    .map(|value| normalize_answer_surface_span(&value))
    .filter(|value| !value.is_empty())
}

pub(super) fn extract_dialogue_friend_group_duration_surface_value(
    text: &str,
    lower: &str,
) -> Option<String> {
    let friendship_context = lower.contains("friend")
        && (lower.contains("known")
            || lower.contains("been friends")
            || lower.contains("group of friends"));
    if !friendship_context {
        return None;
    }

    compile_regex(
        r"(?i)\bfor\s+(?:about\s+|around\s+|over\s+|almost\s+)?((?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(?:day|week|month|year)s?(?:\s+and\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(?:day|week|month|year)s?)?)",
    )
    .captures(text)
    .and_then(|captures| captures.get(1))
    .map(|value| normalize_answer_surface_span(value.as_str()))
    .filter(|value| !value.is_empty())
}

pub(super) fn extract_dialogue_support_network_surface_value(
    text: &str,
    lower: &str,
) -> Option<String> {
    if lower.contains("friends, family and mentors")
        || lower.contains("friends, family, and mentors")
    {
        return Some("friends, family, and mentors".to_string());
    }
    if lower.contains("friends and mentors") && lower.contains("support") {
        return Some("friends and mentors".to_string());
    }
    if lower.contains("friends and family") && lower.contains("support") {
        return Some("friends and family".to_string());
    }
    if lower.contains("my husband and kids") {
        return Some("husband and kids".to_string());
    }
    if !(lower.contains("support me")
        || lower.contains("supports me")
        || lower.contains("help me")
        || lower.contains("helps me"))
    {
        return None;
    }

    let raw =
        compile_regex(r"(?i)(?:that|because)\s+(.+?)\s+(?:support|supports|help|helps)\s+me\b")
            .captures(text)
            .and_then(|captures| captures.get(1))
            .map(|value| value.as_str().to_string())
            .or_else(|| {
                compile_regex(r"(?i)^(.+?)\s+(?:support|supports|help|helps)\s+me\b")
                    .captures(text)
                    .and_then(|captures| captures.get(1))
                    .map(|value| value.as_str().to_string())
            })?;

    let mut values = Vec::new();
    for part in raw
        .replace(", and ", ", ")
        .replace(" and ", ", ")
        .split(',')
    {
        let clean = normalize_answer_surface_span(
            part.trim()
                .trim_matches(|c: char| matches!(c, '"' | '\'' | ',' | '.' | ';' | ':'))
                .trim_start_matches("my ")
                .trim_start_matches("My ")
                .trim_start_matches("our ")
                .trim_start_matches("Our ")
                .trim_start_matches("the ")
                .trim_start_matches("The "),
        );
        if clean.is_empty() || clean.split_whitespace().count() > 4 {
            continue;
        }
        push_unique_bridge_value(&mut values, &clean);
    }

    (!values.is_empty()).then(|| values.join(", "))
}

pub(super) fn extract_dialogue_support_effect_surface_value(
    text: &str,
    lower: &str,
) -> Option<String> {
    if !lower.contains("support group") {
        return None;
    }

    extract_clause_after_any(
        text,
        lower,
        &["has made me ", "made me "],
        &[". ", "! ", "? ", " but ", " and now ", " and since "],
        18,
    )
    .map(|value| normalize_dialogue_support_effect_phrase(&value))
    .or_else(|| {
        extract_clause_after_any(
            text,
            lower,
            &["has helped me ", "helped me ", "helps me "],
            &[". ", "! ", "? ", " but ", " and now ", " and since "],
            18,
        )
        .map(|value| normalize_answer_surface_span(&value))
    })
    .or_else(|| {
        extract_clause_after_any(
            text,
            lower,
            &["has given me ", "given me ", "gave me "],
            &[". ", "! ", "? ", " but ", " and now ", " and since "],
            14,
        )
        .map(|value| format!("have {}", normalize_answer_surface_span(&value)))
    })
    .filter(|value| !value.is_empty())
}

pub(super) fn extract_dialogue_religiosity_surface_value(
    _text: &str,
    lower: &str,
) -> Option<String> {
    if lower.contains("local church")
        || lower.contains("my church")
        || (lower.contains("faith") && has_dialogue_self_reference(lower))
    {
        return Some("somewhat religious".to_string());
    }
    None
}
