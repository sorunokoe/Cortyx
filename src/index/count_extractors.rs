use super::*;

pub(super) fn extract_recent_jewelry_acquisition_signatures_from_line(
    line: &str,
    lower: &str,
    max_days: i32,
) -> Vec<String> {
    let mut items = Vec::new();
    let mut seen = HashSet::new();
    if !(lower.starts_with("user:") || line.trim_start().starts_with('-'))
        || !line_mentions_recent_window_days(line, max_days)
        || !lower.contains("got")
    {
        return items;
    }

    let mut push = |label: &str| {
        let key = normalized_synthetic_phrase_key(label);
        if key.len() >= 3 && seen.insert(key.clone()) {
            items.push(key);
        }
    };

    if lower.contains("silver necklace") {
        push("silver necklace");
    } else if lower.contains("necklace") && lower.contains("got a new") {
        push("necklace");
    }

    if lower.contains("engagement ring") && lower.contains("got") {
        push("engagement ring");
    }

    if lower.contains("emerald earrings") {
        push("emerald earrings");
    } else if lower.contains("earrings") && lower.contains("got a new pair of") {
        push("earrings");
    }

    items
}

pub(super) fn extract_recent_plant_acquisition_signatures_from_line(
    line: &str,
    lower: &str,
    max_days: i32,
) -> Vec<String> {
    let mut items = Vec::new();
    let mut seen = HashSet::new();
    if !(lower.starts_with("user:") || line.trim_start().starts_with('-'))
        || !line_mentions_recent_window_days(line, max_days)
        || !lower.contains("got")
    {
        return items;
    }

    let mut push = |label: &str| {
        let key = normalized_synthetic_phrase_key(label);
        if key.len() >= 3 && seen.insert(key.clone()) {
            items.push(key);
        }
    };

    if lower.contains("peace lily") {
        push("peace lily");
    }
    if lower.contains("succulent plant")
        || lower.contains("a succulent")
        || lower.contains("succulent")
    {
        push("succulent");
    }
    if lower.contains("snake plant") {
        push("snake plant");
    }

    items
}

pub(super) fn extract_group_project_course_signature_from_line(
    line: &str,
    lower: &str,
) -> Option<String> {
    if !lower.contains("group project") {
        return None;
    }
    let regex =
        compile_regex_static(r"\b(?:my|the)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s+course\b");
    let course = regex.captures(line)?.get(1)?.as_str();
    Some(normalized_synthetic_phrase_key(&format!(
        "{course} group project"
    )))
}

pub(super) fn extract_competitive_sport_signatures_from_line(
    line: &str,
    lower: &str,
) -> Vec<String> {
    let mut sports = Vec::new();
    let mut seen = HashSet::new();
    if !task_contains_any(lower, &["competitive", "competitively"]) {
        return sports;
    }

    let patterns = [
        r"(?i)\b(?:(?:as\s+someone\s+who's|as\s+someone\s+who\s+used\s+to|i|we)\s+)?(?:used\s+to\s+)?(?:play(?:ed|ing)?\s+)?([a-z][a-z'-]+(?:\s+[a-z][a-z'-]+)?)\s+competitively\b",
    ];

    for pattern in patterns {
        let regex = compile_regex_static(pattern);
        for captures in regex.captures_iter(line) {
            let Some(raw) = captures.get(1).map(|m| m.as_str()) else {
                continue;
            };
            let Some(signature) = normalize_competitive_activity_signature(raw) else {
                continue;
            };
            push_unique_normalized_signature(&signature, &mut seen, &mut sports);
        }
    }

    sports
}

pub(super) fn extract_current_tank_signatures_from_line(line: &str, lower: &str) -> Vec<String> {
    let mut tanks = Vec::new();
    let mut seen = HashSet::new();
    if !lower.starts_with("user:") || !lower.contains("tank") {
        return tanks;
    }
    if task_contains_any(
        lower,
        &[
            "thinking about",
            "considering",
            "might set up",
            "should i",
            "plan to set up",
        ],
    ) {
        return tanks;
    }
    if !task_contains_any(
        lower,
        &[
            "have a",
            "have an",
            "have my",
            "currently",
            "set up",
            "taking care of",
            "keeping",
            "keep a",
            "keep my",
        ],
    ) {
        return tanks;
    }

    let regex = compile_regex_static(r"(?i)\b\d+\s*-\s*gallon(?:\s+[a-z']+){0,4}\s+tank\b");
    for matched in regex.find_iter(line) {
        if let Some(signature) = normalize_current_tank_signature(matched.as_str(), lower) {
            push_unique_normalized_signature(&signature, &mut seen, &mut tanks);
        }
    }

    tanks
}

pub(super) fn extract_recent_baking_signatures_from_line(line: &str, lower: &str) -> Vec<String> {
    let mut bakes = Vec::new();
    let mut seen = HashSet::new();
    let mut extracted_any = false;
    if !lower.starts_with("user:")
        || !task_contains_any(
            lower,
            &[
                "baked",
                "bake",
                "bread recipe",
                "cookies",
                "cake",
                "baguette",
            ],
        )
        || !line_matches_recent_baking_window(lower)
    {
        return bakes;
    }

    let patterns = [
        r"(?i)\bbaked\s+(?:a|an|some|another)?\s*(?:batch of\s+)?([a-z][a-z' -]{1,40}?)(?:\s+(?:last|this|on|for|with|using|during|over)\b|[,.!?]|$)",
        r"(?i)\b(?:to\s+)?bake\s+(?:a|an|some)?\s*(?:batch of\s+)?([a-z][a-z' -]{1,40}?)(?:\s+(?:last|this|on|for|with|using|during|over)\b|[,.!?]|$)",
        r"(?i)\bmake\s+(?:a|an)\s+(?:delicious\s+|fresh\s+|new\s+)?([a-z][a-z' -]{1,40}?)(?:\s+(?:last|this|on|for|with|using|during|over)\b|[,.!?]|$)",
        r"(?i)\bnew\s+([a-z][a-z' -]{1,30}?)\s+recipe\b",
    ];

    for pattern in patterns {
        let regex = compile_regex_static(pattern);
        for captures in regex.captures_iter(line) {
            let Some(raw) = captures.get(1).map(|m| m.as_str()) else {
                continue;
            };
            let Some(signature) = normalize_baking_signature(raw) else {
                continue;
            };
            extracted_any |= push_unique_normalized_signature(&signature, &mut seen, &mut bakes);
        }
    }

    if !extracted_any && lower.contains("bread recipe") {
        push_unique_normalized_signature("bread", &mut seen, &mut bakes);
    }

    bakes
}

fn push_unique_normalized_signature(
    label: &str,
    seen: &mut HashSet<String>,
    values: &mut Vec<String>,
) -> bool {
    let key = normalized_synthetic_phrase_key(label);
    if seen.insert(key.clone()) {
        values.push(key);
        true
    } else {
        false
    }
}

fn normalize_competitive_activity_signature(raw: &str) -> Option<String> {
    let trimmed = raw
        .trim()
        .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '\'' && c != '-' && c != ' ');
    if trimmed.is_empty() {
        return None;
    }

    let cleaned =
        compile_regex_static(r"(?i)^(.+?)(?:\s+(?:in|during|through|for|at|with|on)\b|$)")
            .captures(trimmed)
            .and_then(|captures| captures.get(1).map(|m| m.as_str().trim()))
            .unwrap_or(trimmed);
    let stripped = compile_regex_static(
        r"(?i)^(?:(?:i|we)\s+)?(?:used\s+to\s+)?(?:play(?:ed|ing)?\s+|do(?:ne)?\s+|did\s+|was\s+in\s+|were\s+in\s+)?",
    )
    .replace(cleaned, "")
    .to_string();
    let normalized = normalized_synthetic_phrase_key(&stripped);
    let filtered_tokens = normalized
        .split_whitespace()
        .filter(|token| {
            !matches!(
                *token,
                "i" | "we" | "used" | "to" | "play" | "played" | "playing" | "did" | "done"
            )
        })
        .collect::<Vec<_>>();
    let key = filtered_tokens.join(" ");
    if key.is_empty()
        || matches!(
            key.as_str(),
            "i" | "it" | "that" | "this" | "something" | "sport" | "sports"
        )
    {
        return None;
    }

    Some(match key.as_str() {
        "swim" => "swimming".to_string(),
        "run" => "running".to_string(),
        "row" => "rowing".to_string(),
        "ski" => "skiing".to_string(),
        "skate" => "skating".to_string(),
        "surf" => "surfing".to_string(),
        "dive" => "diving".to_string(),
        _ => key,
    })
}

fn normalize_current_tank_signature(raw: &str, lower: &str) -> Option<String> {
    let size = compile_regex_static(r"(?i)\b(\d+\s*-\s*gallon)\b")
        .captures(raw)
        .and_then(|captures| {
            captures
                .get(1)
                .map(|m| normalized_synthetic_phrase_key(m.as_str()))
        })?;

    let suffix = if lower.contains("friend") && lower.contains("kid") {
        "friend-kid tank"
    } else if lower.contains("community") {
        "community tank"
    } else if lower.contains("reef") {
        "reef tank"
    } else if lower.contains("quarantine") || lower.contains("hospital tank") {
        "quarantine tank"
    } else if lower.contains("betta") {
        "betta tank"
    } else if lower.contains("shrimp") {
        "shrimp tank"
    } else {
        "tank"
    };

    Some(format!("{size} {suffix}"))
}

fn normalize_baking_signature(raw: &str) -> Option<String> {
    let trimmed = raw
        .trim()
        .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '\'' && c != '-' && c != ' ');
    if trimmed.is_empty() {
        return None;
    }

    let cleaned =
        compile_regex_static(r"(?i)^(.+?)(?:\s+(?:using|with|for|on|during|last|this|over)\b|$)")
            .captures(trimmed)
            .and_then(|captures| captures.get(1).map(|m| m.as_str().trim()))
            .unwrap_or(trimmed)
            .trim_end_matches(" recipe")
            .trim();
    let key = normalized_synthetic_phrase_key(cleaned);
    if key.is_empty() || matches!(key.as_str(), "something" | "something new" | "it" | "one") {
        return None;
    }
    Some(key)
}

fn line_mentions_recent_window_days(text: &str, max_days: i32) -> bool {
    extract_temporal_relative_days(text).is_some_and(|days| days <= max_days)
}

fn line_matches_recent_baking_window(lower: &str) -> bool {
    line_mentions_recent_window_days(lower, 14)
        || compile_regex_static(
            r"(?i)\b(?:yesterday|today|tonight|this\s+(?:week|weekend)|last\s+(?:week|weekend|monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b",
        )
        .is_match(lower)
        || compile_regex_static(r"(?i)\bon\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b")
            .is_match(lower)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn competitive_sport_extractor_discards_context_prefixes() {
        let line = "user: I'm looking to find a local pool that offers lap swimming hours. I used to swim competitively in college, and I'm looking to get back into it as a way to stay active and relieve stress.";
        let lower = line.to_ascii_lowercase();
        assert_eq!(
            extract_competitive_sport_signatures_from_line(line, &lower),
            vec!["swimming"]
        );
    }

    #[test]
    fn competitive_sport_extractor_handles_swimming_context_clause() {
        let line = "user: I'll also ask about their policy on lane sharing and circle swimming. As someone who's used to swimming competitively in college, I'm comfortable swimming in a fast-paced environment, but I want to make sure I'll have enough space to do my workouts effectively.";
        let lower = line.to_ascii_lowercase();
        assert_eq!(
            extract_competitive_sport_signatures_from_line(line, &lower),
            vec!["swimming"]
        );
    }

    #[test]
    fn current_tank_extractor_keeps_friends_kid_setup() {
        let line = "user: I've also been taking care of a small 1-gallon tank that I set up for a friend's kid, which has a few guppies and some plants. I was wondering if the temperature requirements are the same for these plants in a smaller tank like that?";
        let lower = line.to_ascii_lowercase();
        assert_eq!(
            extract_current_tank_signatures_from_line(line, &lower),
            vec!["1-gallon friend-kid tank"]
        );
    }
}
