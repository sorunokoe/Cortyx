use super::Turn;

// ─── Disclosure segment helpers ───────────────────────────────────────────────

pub(super) fn user_disclosure_segments(turn: &Turn) -> Vec<String> {
    match turn.speaker.as_deref().map(|s| s.to_ascii_lowercase()) {
        Some(role) if role == "user" || role == "human" => vec![turn.text.clone()],
        Some(role) if role == "assistant" || role == "ai" || role == "system" => Vec::new(),
        _ => {
            let mut segments = Vec::new();
            let mut current: Vec<String> = Vec::new();
            let mut in_user = false;
            for line in turn.text.lines() {
                let trimmed = line.trim();
                let lower = trimmed.to_ascii_lowercase();
                if let Some((label, rest)) = trimmed.split_once(':') {
                    let label_lower = label.trim().to_ascii_lowercase();
                    if label_lower == "user" || label_lower == "human" {
                        if in_user && !current.is_empty() {
                            segments.push(current.join("\n").trim().to_string());
                            current.clear();
                        }
                        in_user = true;
                        if !rest.trim().is_empty() {
                            current.push(rest.trim().to_string());
                        }
                        continue;
                    }
                    if label_lower == "assistant" || label_lower == "ai" || label_lower == "system"
                    {
                        if in_user && !current.is_empty() {
                            segments.push(current.join("\n").trim().to_string());
                            current.clear();
                        }
                        in_user = false;
                        continue;
                    }
                }
                if in_user && !lower.is_empty() {
                    current.push(trimmed.to_string());
                }
            }
            if in_user && !current.is_empty() {
                segments.push(current.join("\n").trim().to_string());
            }
            segments.into_iter().filter(|s| !s.is_empty()).collect()
        },
    }
}

pub(super) fn assistant_segments(turn: &Turn) -> Vec<String> {
    match turn.speaker.as_deref().map(|s| s.to_ascii_lowercase()) {
        Some(role) if role == "assistant" || role == "ai" || role == "system" => {
            vec![turn.text.clone()]
        },
        Some(role) if role == "user" || role == "human" => Vec::new(),
        _ => {
            let mut segments = Vec::new();
            let mut current: Vec<String> = Vec::new();
            let mut in_assistant = false;
            for line in turn.text.lines() {
                let trimmed = line.trim();
                if let Some((label, rest)) = trimmed.split_once(':') {
                    let label_lower = label.trim().to_ascii_lowercase();
                    if label_lower == "assistant" || label_lower == "ai" || label_lower == "system"
                    {
                        if in_assistant && !current.is_empty() {
                            segments.push(current.join("\n").trim().to_string());
                            current.clear();
                        }
                        in_assistant = true;
                        if !rest.trim().is_empty() {
                            current.push(rest.trim().to_string());
                        }
                        continue;
                    }
                    if label_lower == "user" || label_lower == "human" {
                        if in_assistant && !current.is_empty() {
                            segments.push(current.join("\n").trim().to_string());
                            current.clear();
                        }
                        in_assistant = false;
                        continue;
                    }
                }
                if in_assistant && !trimmed.is_empty() {
                    current.push(trimmed.to_string());
                }
            }
            if in_assistant && !current.is_empty() {
                segments.push(current.join("\n").trim().to_string());
            }
            segments.into_iter().filter(|s| !s.is_empty()).collect()
        },
    }
}

// ─── Entity + value extractors ────────────────────────────────────────────────

pub(super) fn extract_fact_entity(text: &str, trigger_pos: usize) -> String {
    const STOPWORDS: &[&str] = &[
        "The",
        "And",
        "But",
        "For",
        "Are",
        "Was",
        "Has",
        "Had",
        "She",
        "Her",
        "His",
        "Him",
        "They",
        "Them",
        "Our",
        "You",
        "Your",
        "This",
        "That",
        "With",
        "From",
        "Have",
        "Will",
        "Been",
        "Just",
        "Can",
        "Yeah",
        "There",
        "Here",
        "Its",
        "It's",
        "That's",
        "I've",
        "I'm",
        "I'll",
        "I'd",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Fridays",
        "Saturday",
        "Sunday",
        "January",
        "February",
        "March",
        "April",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    const RELATIONS: &[(&str, &str)] = &[
        ("my mom", "mom"),
        ("my mother", "mom"),
        ("my dad", "dad"),
        ("my father", "dad"),
        ("my sister", "sister"),
        ("my brother", "brother"),
        ("my wife", "wife"),
        ("my husband", "husband"),
        ("my partner", "partner"),
        ("my boyfriend", "boyfriend"),
        ("my girlfriend", "girlfriend"),
        ("my friend", "friend"),
        ("my coworker", "coworker"),
        ("my colleague", "coworker"),
        ("my boss", "boss"),
        ("my manager", "manager"),
        ("my neighbor", "neighbor"),
    ];

    let before = &text[..trigger_pos.min(text.len())];
    let mut last_named: Option<String> = None;
    for word in before.split_whitespace() {
        let clean = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'');
        if clean.len() >= 3
            && clean.chars().next().is_some_and(|c| c.is_uppercase())
            && !STOPWORDS.contains(&clean)
        {
            last_named = Some(crate::kg::slugify(clean));
        }
    }

    if let Some(named) = last_named {
        return named;
    }

    let before_lower = before.to_ascii_lowercase();
    for (needle, entity) in RELATIONS {
        if before_lower.contains(needle) {
            return (*entity).to_string();
        }
    }

    "user".to_string()
}

pub(super) fn extract_numeric_fact_value(after: &str) -> Option<String> {
    after
        .split_whitespace()
        .map(|word| word.trim_matches(|c: char| !c.is_alphanumeric()))
        .find(|word| !word.is_empty())
        .map(|word| word.to_string())
}

pub(super) fn extract_count_fact_value(after: &str) -> Option<String> {
    fn is_count_token(token: &str) -> bool {
        matches!(
            token,
            "zero"
                | "one"
                | "two"
                | "three"
                | "four"
                | "five"
                | "six"
                | "seven"
                | "eight"
                | "nine"
                | "ten"
                | "eleven"
                | "twelve"
                | "thirteen"
                | "fourteen"
                | "fifteen"
                | "sixteen"
                | "seventeen"
                | "eighteen"
                | "nineteen"
                | "twenty"
                | "first"
                | "second"
                | "third"
                | "fourth"
                | "fifth"
                | "sixth"
                | "seventh"
                | "eighth"
                | "ninth"
                | "tenth"
        )
    }

    after
        .split_whitespace()
        .map(|word| word.trim_matches(|c: char| !c.is_alphanumeric()))
        .find_map(|word| {
            if word.is_empty() {
                return None;
            }
            let lower = word.to_ascii_lowercase();
            if lower.chars().all(|c| c.is_ascii_digit()) || is_count_token(&lower) {
                Some(lower)
            } else {
                None
            }
        })
}

pub(super) fn extract_dollar_amount(text: &str) -> Option<String> {
    text.split_whitespace()
        .map(|word| word.trim_matches(|c: char| !c.is_alphanumeric() && c != '$'))
        .find_map(|word| word.strip_prefix('$').filter(|amount| !amount.is_empty()))
        .map(|amount| amount.to_string())
}

pub(super) fn extract_phrase_fact_value(
    after: &str,
    stop_words: &[&str],
    max_words: usize,
) -> Option<String> {
    let mut words = Vec::new();
    for raw in after.split_whitespace() {
        let cleaned =
            raw.trim_matches(|c: char| !c.is_alphanumeric() && c != '-' && c != '&' && c != '\'');
        if cleaned.is_empty() {
            continue;
        }
        if stop_words.contains(&cleaned.to_ascii_lowercase().as_str()) {
            break;
        }
        words.push(cleaned.to_string());
        if words.len() >= max_words {
            break;
        }
    }
    if words.is_empty() {
        None
    } else {
        Some(words.join(" "))
    }
}

pub(super) fn sum_active_numeric_predicates(
    kg_entity: &crate::kg::KgEntity,
    predicates: &[&str],
) -> i64 {
    kg_entity
        .facts
        .iter()
        .filter(|fact| fact.ended.is_empty() && predicates.contains(&fact.predicate.as_str()))
        .filter_map(|fact| fact.value.trim_start_matches('$').parse::<i64>().ok())
        .sum()
}
