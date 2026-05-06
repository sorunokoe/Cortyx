use super::*;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct OnlineCommunityHobbyQuery {
    pub(super) max_items: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct HobbySignal {
    pub(super) hobby: String,
    pub(super) score: usize,
    pub(super) evidence: String,
}

pub(super) fn parse_online_community_hobby_query(
    task_lower: &str,
) -> Option<OnlineCommunityHobbyQuery> {
    (task_lower.contains("hobbies")
        && task_lower.contains("join online communit")
        && task_lower.contains("led me"))
    .then_some(OnlineCommunityHobbyQuery { max_items: 2 })
}

pub(super) fn is_online_community_participation_line(lower: &str) -> bool {
    lower.starts_with("user:")
        && (lower.contains("online communities")
            || lower.contains("online community")
            || lower.contains("online forums")
            || lower.contains("online forum"))
        && (lower.contains("joined")
            || lower.contains("learned from")
            || lower.contains("feedback")
            || lower.contains("discussions")
            || lower.contains("share my")
            || lower.contains("share their")
            || lower.contains("posts"))
}

pub(super) fn extract_hobby_signals_from_line(line: &str, lower: &str) -> Vec<HobbySignal> {
    if !lower.starts_with("user:") {
        return Vec::new();
    }
    let mut signals = Vec::new();
    for (marker, score) in [
        ("online communities related to ", 30),
        ("online community related to ", 30),
        ("interested in ", 18),
        ("really into ", 18),
        ("looking for some ", 14),
        ("focus on ", 12),
        ("sharing my ", 10),
        ("share my ", 10),
        ("about ", 6),
    ] {
        let Some(phrase) = extract_phrase_after_any_index(
            line,
            lower,
            &[marker],
            &[
                " and ", " but ", " which ", " where ", " that ", " to ", ",", ".", "?",
            ],
            1,
        ) else {
            continue;
        };
        let Some(hobby) = canonicalize_hobby_phrase(&phrase) else {
            continue;
        };
        signals.push(HobbySignal {
            hobby,
            score,
            evidence: line.trim().to_string(),
        });
    }

    if lower.contains("photography") || lower.contains("astrophotography") {
        signals.push(HobbySignal {
            hobby: "photography".to_string(),
            score: 12 + usize::from(lower.contains("interested in")) * 6,
            evidence: line.trim().to_string(),
        });
    }
    if lower.contains("cooking") || lower.contains("recipe") || lower.contains("food-related") {
        signals.push(HobbySignal {
            hobby: "cooking".to_string(),
            score: 12 + usize::from(lower.contains("related to cooking")) * 10,
            evidence: line.trim().to_string(),
        });
    }

    dedupe_hobby_signals(signals)
}

fn canonicalize_hobby_phrase(phrase: &str) -> Option<String> {
    let tokens = phrase
        .split_whitespace()
        .map(|token| {
            token
                .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-')
                .to_ascii_lowercase()
        })
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>();
    if tokens.is_empty() {
        return None;
    }
    if tokens.iter().any(|token| token.ends_with("photography")) {
        return Some("photography".to_string());
    }
    if tokens
        .iter()
        .any(|token| matches!(token.as_str(), "cooking" | "recipe" | "recipes" | "food"))
    {
        return Some("cooking".to_string());
    }
    const STOP: &[&str] = &[
        "online",
        "communities",
        "community",
        "forums",
        "forum",
        "tips",
        "tip",
        "inspiration",
        "resources",
        "tutorials",
        "tutorial",
        "work",
        "feedback",
        "others",
        "posts",
        "post",
        "discussions",
        "discussion",
        "thoughts",
        "techniques",
        "technique",
        "photos",
        "editing",
        "settings",
        "gear",
        "blog",
        "blogs",
        "website",
        "websites",
        "advice",
        "process",
        "videos",
        "video",
        "with",
        "from",
        "about",
        "into",
        "some",
        "more",
        "good",
        "share",
        "sharing",
        "learn",
        "learning",
        "helpful",
        "new",
        "their",
        "my",
    ];
    let best = tokens
        .iter()
        .find(|token| !STOP.contains(&token.as_str()) && token.ends_with("ing"))
        .or_else(|| tokens.iter().find(|token| !STOP.contains(&token.as_str())))?;
    Some(best.to_string())
}

fn dedupe_hobby_signals(signals: Vec<HobbySignal>) -> Vec<HobbySignal> {
    let mut best = HashMap::<String, HobbySignal>::new();
    for signal in signals {
        let key = normalized_synthetic_phrase_key(&signal.hobby);
        let should_replace = best
            .get(&key)
            .map(|existing| signal.score > existing.score)
            .unwrap_or(true);
        if should_replace {
            best.insert(key, signal);
        }
    }
    best.into_values().collect()
}
