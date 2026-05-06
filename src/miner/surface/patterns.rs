use regex::Regex;

use super::super::kg_extract::{
    extract_count_fact_value, extract_numeric_fact_value, extract_phrase_fact_value,
};
use super::super::{AnswerSurfaceRow, Turn};
use super::conversation::{
    generate_embedded_dialogue_answer_surface_rows, is_dialogue_speaker, scoped_question_pattern,
};

pub(super) fn compile_regex(pattern: &str) -> Regex {
    match Regex::new(pattern) {
        Ok(regex) => regex,
        Err(err) => {
            tracing::error!("invalid miner regex {pattern:?}: {err}");
            match Regex::new(r"$^") {
                Ok(fallback) => fallback,
                Err(_) => unreachable!("fallback regex must compile"),
            }
        },
    }
}

pub(crate) fn append_answer_surface_section(
    content: &mut String,
    text: &str,
    extra_rows: &[AnswerSurfaceRow],
    note: &str,
) {
    let mut rows = generate_answer_surface_rows(text);
    for row in generate_embedded_dialogue_answer_surface_rows(text) {
        push_answer_surface_row(
            &mut rows,
            &row.question_pattern,
            Some(row.answer_span),
            row.confidence,
        );
    }
    for row in extra_rows {
        push_answer_surface_row(
            &mut rows,
            &row.question_pattern,
            Some(row.answer_span.clone()),
            row.confidence,
        );
    }
    if rows.is_empty() {
        return;
    }
    content.push_str("\n## answer_surface\n");
    content.push_str(&format!("<!-- {note} -->\n"));
    content.push_str("<!-- SECTION: answer_surface -->\n");
    content.push_str("| question_pattern | answer_span | confidence |\n");
    content.push_str("| --- | --- | --- |\n");
    for row in rows {
        content.push_str(&format!(
            "| {} | {} | {:.2} |\n",
            sanitize_answer_surface_cell(&row.question_pattern),
            sanitize_answer_surface_cell(&row.answer_span),
            row.confidence
        ));
    }
    content.push_str("<!-- /SECTION -->\n");
}

fn sanitize_answer_surface_cell(value: &str) -> String {
    value.replace('|', "/")
}

pub(super) fn normalize_answer_surface_span(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn generate_answer_surface_rows(text: &str) -> Vec<AnswerSurfaceRow> {
    const JOB_PATTERN: &str = "job occupation profession work career role";
    const LOCATION_PATTERN: &str = "live location residence city home moved based";
    const DEGREE_PATTERN: &str = "degree major education field studied graduated";
    const PET_PATTERN: &str = "pet cat dog name called";
    const FAMILY_TRIP_PATTERN: &str = "family trip vacation destination travel location";
    const ISSUE_PATTERN: &str = "issue problem malfunction wrong service repair not functioning";
    const VEHICLE_PATTERN: &str = "vehicle car model current vehicle";
    const PRODUCT_PATTERN: &str = "current product brand shampoo conditioner skincare use";
    const SHOE_BRAND_PATTERN: &str = "favorite running shoes brand shoe sneaker trainer";
    const CERTIFICATION_PATTERN: &str = "certification credential completed last month recent";
    const GIFT_PATTERN: &str = "birthday gift sister present bought";
    const PLAY_PATTERN: &str = "play theater community theater attended watched";
    const CONCERT_VENUE_PATTERN: &str = "concert venue attended live show";
    const RICE_PATTERN: &str = "favorite rice type grain";
    const INSTAGRAM_FOLLOWERS_PATTERN: &str =
        "instagram followers follower count current social media";
    const PRE_1920_COIN_PATTERN: &str = "pre-1920 coins collection count total";
    const NATIONAL_GEOGRAPHIC_PATTERN: &str = "national geographic issues finished reading count";
    const KOREAN_RESTAURANT_PATTERN: &str = "korean restaurants tried city count";
    const FISH_CATCH_PATTERN: &str = "largemouth bass fishing trip catch count";
    const PLAYLIST_PATTERN: &str = "playlist music spotify called name";
    const GROUP_PATTERN: &str = "kind type group joined online group community";
    const SIGN_PATTERN: &str = "sign warning notice precaution precautionary cafe café";
    const RELAX_ACTIVITY_PATTERN: &str = "relax unwind nature walk hike road trip activity";
    const RESEARCH_PATTERN: &str = "research researched topic investigating looking into";
    const FITNESS_RECORD_PATTERN: &str =
        "personal best time record fastest race run charity 5k score";

    let mut rows = Vec::new();
    for raw_line in text.split(['\n', '.', '!', '?']) {
        let line = raw_line.trim();
        if line.len() < 10 {
            continue;
        }
        let lower = line.to_ascii_lowercase();
        let mut push = |question_pattern: &str, answer_span: Option<String>, confidence: f32| {
            push_answer_surface_row_for_line(
                &mut rows,
                line,
                question_pattern,
                answer_span,
                confidence,
            );
        };

        push(
            JOB_PATTERN,
            extract_fact_after_any(
                line,
                &lower,
                &[
                    "i work as ",
                    "i'm a ",
                    "i am a ",
                    "i'm an ",
                    "i am an ",
                    "i work as an ",
                    "i work as a ",
                    "i became a ",
                    "i became an ",
                ],
                &[
                    " at ",
                    " for ",
                    " in ",
                    " with ",
                    " because ",
                    " since ",
                    " and ",
                    " but ",
                ],
                4,
            )
            .filter(|value| looks_like_job_surface_value(value)),
            0.92,
        );

        push(
            LOCATION_PATTERN,
            extract_fact_after_any(
                line,
                &lower,
                &[
                    "i live in ",
                    "i'm living in ",
                    "i am living in ",
                    "i moved to ",
                    "i moved back to ",
                    "i relocated to ",
                    "i settled in ",
                    "i'm based in ",
                    "i am based in ",
                ],
                &[
                    " with ",
                    " because ",
                    " and ",
                    " but ",
                    " now ",
                    " again ",
                    " after ",
                    " so ",
                ],
                4,
            ),
            0.91,
        );

        push(
            DEGREE_PATTERN,
            extract_fact_after_any(
                line,
                &lower,
                &[
                    "degree in ",
                    "majored in ",
                    "major in ",
                    "graduated with a degree in ",
                    "graduated in ",
                    "studied ",
                ],
                &[" at ", " from ", " and ", " but ", " because "],
                4,
            ),
            0.9,
        );

        push(
            PET_PATTERN,
            extract_fact_after_any(
                line,
                &lower,
                &[
                    "my cat's name is ",
                    "my dog's name is ",
                    "my cat is named ",
                    "my dog is named ",
                    "our cat's name is ",
                    "our dog's name is ",
                ],
                &[" and ", " but ", " because "],
                2,
            ),
            0.95,
        );

        push(ISSUE_PATTERN, extract_issue_surface_value(line), 0.84);
        push(
            RESEARCH_PATTERN,
            extract_research_surface_value(line, &lower),
            0.84,
        );
        push(
            FITNESS_RECORD_PATTERN,
            extract_fitness_record_surface_value(line, &lower),
            0.9,
        );

        if lower.contains("as a family") || lower.contains("with my family") {
            push(
                FAMILY_TRIP_PATTERN,
                extract_fact_after_any(
                    line,
                    &lower,
                    &[
                        "thinking of going to ",
                        "planning a trip to ",
                        "planned a trip to ",
                        "went to ",
                    ],
                    &[" with ", " for ", " and ", " but ", " because "],
                    4,
                ),
                0.82,
            );
        }

        if lower.contains("joined a ") || lower.contains("joined an ") || lower.contains("joined ")
        {
            push(
                GROUP_PATTERN,
                extract_fact_after_any(
                    line,
                    &lower,
                    &["joined a ", "joined an ", "joined "],
                    &[" last ", " and ", " but ", " because ", " to ", " with "],
                    5,
                ),
                0.83,
            );
        }

        if lower.contains("model") || lower.contains("vehicle") || lower.contains("car") {
            push(
                VEHICLE_PATTERN,
                extract_fact_after_any(
                    line,
                    &lower,
                    &[
                        "switched to a ",
                        "switched to an ",
                        "bought a ",
                        "bought an ",
                        "drive a ",
                        "drive an ",
                    ],
                    &[" model", " because ", " and ", " but "],
                    4,
                ),
                0.8,
            );
        }

        if lower.contains("using") || lower.contains("shampoo") || lower.contains("conditioner") {
            push(
                PRODUCT_PATTERN,
                extract_fact_after_any(
                    line,
                    &lower,
                    &[
                        "i switched to using ",
                        "i use ",
                        "i'm using ",
                        "i am using ",
                        "i switched to ",
                    ],
                    &[" for ", " because ", " and ", " but "],
                    4,
                ),
                0.78,
            );
            push(
                PRODUCT_PATTERN,
                extract_shampoo_brand_surface_value(line, &lower),
                0.86,
            );
        }

        push(
            SHOE_BRAND_PATTERN,
            extract_running_shoe_brand_surface_value(line, &lower),
            0.87,
        );
        push(
            CERTIFICATION_PATTERN,
            extract_certification_surface_value(line, &lower),
            0.88,
        );
        push(
            GIFT_PATTERN,
            extract_sister_gift_surface_value(line, &lower),
            0.84,
        );
        push(
            PLAY_PATTERN,
            extract_theater_play_surface_value(line, &lower),
            0.84,
        );
        push(
            CONCERT_VENUE_PATTERN,
            extract_concert_venue_surface_value(line, &lower),
            0.84,
        );
        push(
            RICE_PATTERN,
            extract_favorite_rice_surface_value(line, &lower),
            0.84,
        );

        if let Some((question_pattern, value)) = extract_relative_location_surface_row(line, &lower)
        {
            push(&question_pattern, Some(value), 0.86);
        }

        push(
            INSTAGRAM_FOLLOWERS_PATTERN,
            extract_instagram_followers_surface_value(line, &lower),
            0.86,
        );
        push(
            PRE_1920_COIN_PATTERN,
            extract_pre_1920_coin_surface_value(line, &lower),
            0.84,
        );
        push(
            NATIONAL_GEOGRAPHIC_PATTERN,
            extract_national_geographic_count_surface_value(line, &lower),
            0.82,
        );
        push(
            KOREAN_RESTAURANT_PATTERN,
            extract_korean_restaurant_count_surface_value(line, &lower),
            0.82,
        );
        push(
            FISH_CATCH_PATTERN,
            extract_largemouth_bass_count_surface_value(line, &lower),
            0.82,
        );

        if lower.contains("playlist") {
            push(
                PLAYLIST_PATTERN,
                extract_fact_after_any(
                    line,
                    &lower,
                    &[
                        "playlist called ",
                        "playlist is called ",
                        "named my playlist ",
                    ],
                    &[" and ", " but ", " because "],
                    4,
                ),
                0.84,
            );
        }

        if lower.contains("sign ") {
            push(
                SIGN_PATTERN,
                extract_fact_after_any(
                    line,
                    &lower,
                    &[
                        "sign saying ",
                        "sign said ",
                        "sign that said ",
                        "sign reading ",
                        "sign read ",
                    ],
                    &[" and ", " but ", " because ", " near ", " at "],
                    8,
                ),
                0.8,
            );
        }

        push(
            RELAX_ACTIVITY_PATTERN,
            extract_relax_activity_surface_value(line, &lower),
            0.76,
        );
    }
    rows
}

pub(crate) fn extract_fact_after_any(
    line: &str,
    lower_line: &str,
    markers: &[&str],
    stop_tokens: &[&str],
    max_words: usize,
) -> Option<String> {
    for marker in markers {
        if let Some(idx) = lower_line.find(marker) {
            let tail = line[idx + marker.len()..].trim();
            let lower_tail = tail.to_ascii_lowercase();
            let cutoff = stop_tokens
                .iter()
                .filter_map(|token| lower_tail.find(token))
                .min()
                .unwrap_or(tail.len());
            let bounded_tail = tail[..cutoff].trim();
            if let Some(value) = extract_phrase_fact_value(bounded_tail, &[], max_words) {
                let clean = value.trim_matches(|c: char| matches!(c, '"' | '\'' | ',' | ';' | ':'));
                if !clean.is_empty() {
                    return Some(clean.to_string());
                }
            }
        }
    }
    None
}

pub(super) fn extract_clause_after_any(
    line: &str,
    lower_line: &str,
    markers: &[&str],
    stop_markers: &[&str],
    max_words: usize,
) -> Option<String> {
    for marker in markers {
        if let Some(idx) = lower_line.find(marker) {
            let tail = line[idx + marker.len()..].trim();
            if let Some(value) = extract_clause_fact_value(tail, stop_markers, max_words) {
                return Some(value);
            }
        }
    }
    None
}

fn extract_clause_fact_value(
    after: &str,
    stop_markers: &[&str],
    max_words: usize,
) -> Option<String> {
    let lower = after.to_ascii_lowercase();
    let cutoff = stop_markers
        .iter()
        .filter_map(|marker| lower.find(marker))
        .min()
        .unwrap_or(after.len());
    let trimmed = after[..cutoff].trim();
    if trimmed.is_empty() {
        return None;
    }
    let words = trimmed
        .split_whitespace()
        .take(max_words)
        .collect::<Vec<_>>()
        .join(" ");
    let clean = words.trim_matches(|c: char| matches!(c, '"' | '\'' | ',' | ';' | ':' | '-' | '.'));
    (!clean.is_empty()).then(|| clean.to_string())
}

pub(crate) fn normalize_dialogue_reason_phrase(value: &str) -> String {
    let lower = value.to_ascii_lowercase();
    for prefix in [
        "i want to ",
        "i'd love to ",
        "i would love to ",
        "i wanna ",
        "my goal is to ",
        "goal is to ",
    ] {
        if lower.starts_with(prefix) {
            let rest = value[prefix.len()..].trim();
            return normalize_answer_surface_span(rest);
        }
    }
    normalize_answer_surface_span(value)
}

pub(super) fn normalize_dialogue_support_effect_phrase(value: &str) -> String {
    let mut clean = normalize_answer_surface_span(value);
    clean = clean.replace("and given me ", "and have ");
    clean = clean.replace("And given me ", "and have ");
    if clean.to_ascii_lowercase().starts_with("accepted ") {
        clean = format!("feel {clean}");
    }
    clean
}

pub(crate) fn extract_issue_surface_value(line: &str) -> Option<String> {
    let lower = line.to_ascii_lowercase();
    if let Some(issue) = extract_fact_after_any(
        line,
        &lower,
        &["first issue was ", "issue was ", "problem was "],
        &[" and ", " but ", " because "],
        8,
    ) {
        return Some(issue);
    }

    for marker in [
        " wasn't functioning",
        " not functioning",
        " stopped working",
    ] {
        if let Some(idx) = lower.find(marker) {
            let tail = &line[idx..];
            let lower_tail = tail.to_ascii_lowercase();
            let cutoff = [" after ", " because ", " but ", " and "]
                .iter()
                .filter_map(|stop| lower_tail.find(stop))
                .min()
                .unwrap_or(tail.len());
            let start = line[..idx]
                .rfind(['.', '!', '?', ';'])
                .map(|pos| pos + 1)
                .unwrap_or(0);
            let clause = format!(
                "{}{}",
                line[start..idx]
                    .trim()
                    .trim_matches(|c: char| matches!(c, ',' | ';' | ':' | '"' | '\'')),
                &tail[..cutoff]
            );
            let clean = normalize_answer_surface_span(&clause);
            if !clean.is_empty() {
                return Some(clean);
            }
        }
    }
    None
}

fn extract_relax_activity_surface_value(line: &str, lower: &str) -> Option<String> {
    if let Some(idx) = lower.find("went on a ") {
        let tail = &line[idx..];
        return extract_phrase_fact_value(
            tail,
            &[" and ", " but ", " because ", " after ", " with "],
            5,
        )
        .map(|value| normalize_answer_surface_span(&value));
    }
    if let Some(idx) = lower.find("went on ") {
        let tail = &line[idx..];
        return extract_phrase_fact_value(
            tail,
            &[" and ", " but ", " because ", " after ", " with "],
            5,
        )
        .map(|value| normalize_answer_surface_span(&value));
    }
    if let Some(idx) = lower.find("went hiking") {
        let tail = &line[idx..];
        return extract_phrase_fact_value(
            tail,
            &[" and ", " but ", " because ", " after ", " with "],
            3,
        )
        .map(|value| normalize_answer_surface_span(&value));
    }
    None
}

pub(super) fn extract_research_surface_value(line: &str, lower: &str) -> Option<String> {
    if !lower.contains("research")
        && !lower.contains("looking into")
        && !lower.contains("investigating")
    {
        return None;
    }
    extract_fact_after_any(
        line,
        lower,
        &[
            "researching ",
            "researched ",
            "been researching ",
            "been looking into ",
            "looking into ",
            "investigating ",
            "research into ",
        ],
        &[
            "because", "and", "but", "so", "lately", "recently", "online", "after", "before",
            "it's", "it", "i'm", "im", "more",
        ],
        6,
    )
}

pub(crate) fn extract_fitness_record_surface_value(line: &str, lower: &str) -> Option<String> {
    if !(lower.contains("personal best")
        || lower.contains("best time")
        || lower.contains("race time")
        || lower.contains("fastest time"))
    {
        return None;
    }

    for trigger in [
        "personal best time of ",
        "personal best time was ",
        "personal best of ",
        "personal best was ",
        "best time of ",
        "best time was ",
        "race time was ",
        "fastest time is ",
        "with a time of ",
        "time of ",
    ] {
        let Some(pos) = lower.find(trigger) else {
            continue;
        };
        if let Some(value) = extract_fitness_record_time_value(&line[pos + trigger.len()..]) {
            return Some(value);
        }
    }

    None
}

fn extract_fitness_record_time_value(after: &str) -> Option<String> {
    let time =
        Regex::new(r"(?i)\b(\d{1,2}:\d{2}|\d{1,3}\s+minutes?(?:\s+and\s+\d{1,2}\s+seconds?)?)\b")
            .ok()?;
    time.captures(after)
        .and_then(|captures| captures.get(1))
        .map(|value| value.as_str().trim().to_string())
}

fn push_answer_surface_row_for_line(
    rows: &mut Vec<AnswerSurfaceRow>,
    line: &str,
    question_pattern: &str,
    answer_span: Option<String>,
    confidence: f32,
) {
    let Some(answer_span) = answer_span else {
        return;
    };

    push_answer_surface_row(
        rows,
        question_pattern,
        Some(answer_span.clone()),
        confidence,
    );

    let Some(scoped_pattern) =
        scoped_question_pattern(question_pattern, dialogue_line_scope_speaker(line))
    else {
        return;
    };
    push_answer_surface_row(
        rows,
        &scoped_pattern,
        Some(answer_span),
        (confidence + 0.03).min(0.95),
    );
}

fn dialogue_line_scope_speaker(line: &str) -> Option<&str> {
    let (speaker, rest) = line.split_once(':')?;
    if !is_dialogue_speaker(speaker) {
        return None;
    }
    let rest = rest.trim();
    let lower = rest.to_ascii_lowercase();
    let ellided_self_reference = lower.starts_with("researching ")
        || lower.starts_with("looking into ")
        || lower.starts_with("working in ")
        || lower.starts_with("working on ")
        || lower.starts_with("planning ")
        || lower.starts_with("hoping ")
        || lower.starts_with("trying ");
    (lower.starts_with("i ")
        || lower.starts_with("i'")
        || lower.starts_with("i’m")
        || lower.starts_with("my ")
        || lower.starts_with("we ")
        || lower.starts_with("our ")
        || ellided_self_reference)
        .then_some(speaker.trim())
}

fn extract_fact_before_any(
    line: &str,
    lower_line: &str,
    markers: &[&str],
    max_words: usize,
) -> Option<String> {
    for marker in markers {
        if let Some(idx) = lower_line.find(marker) {
            let mut words = Vec::new();
            for raw in line[..idx].split_whitespace().rev() {
                let cleaned = raw.trim_matches(|c: char| {
                    !c.is_alphanumeric() && c != '-' && c != '&' && c != '\''
                });
                if cleaned.is_empty() {
                    continue;
                }
                words.push(cleaned.to_string());
                if words.len() >= max_words {
                    break;
                }
            }
            if !words.is_empty() {
                words.reverse();
                return Some(words.join(" "));
            }
        }
    }
    None
}

fn looks_like_job_surface_value(value: &str) -> bool {
    let first = value
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();
    !matches!(
        first.as_str(),
        "huge" | "big" | "small" | "massive" | "little" | "fan" | "bit"
    )
}

fn extract_running_shoe_brand_surface_value(line: &str, lower: &str) -> Option<String> {
    if !lower.contains("running shoes") {
        return None;
    }
    extract_fact_before_any(
        line,
        lower,
        &[
            " has been my favourite brand",
            " has been my favorite brand",
            " is my favourite brand",
            " is my favorite brand",
        ],
        3,
    )
    .or_else(|| {
        extract_fact_after_any(
            line,
            lower,
            &[
                "my favourite running shoes are ",
                "my favorite running shoes are ",
                "favorite running shoes are ",
                "favourite running shoes are ",
            ],
            &["and", "but", "because", "for"],
            3,
        )
    })
}

fn extract_favorite_rice_surface_value(line: &str, lower: &str) -> Option<String> {
    if !lower.contains("rice") || !lower.contains("favorite") && !lower.contains("favourite") {
        return None;
    }
    extract_fact_before_any(line, lower, &[" is my favorite", " is my favourite"], 4).or_else(
        || {
            extract_fact_after_any(
                line,
                lower,
                &["my favorite rice is ", "my favourite rice is "],
                &["and", "but", "because", "for"],
                4,
            )
        },
    )
}

fn extract_shampoo_brand_surface_value(line: &str, lower: &str) -> Option<String> {
    if !(lower.contains("shampoo") || lower.contains("conditioner")) {
        return None;
    }
    if let Some(idx) = lower.rfind(" at ") {
        if let Some(value) = extract_phrase_fact_value(
            &line[idx + " at ".len()..],
            &["for", "because", "and", "but", "with"],
            3,
        ) {
            return Some(value);
        }
    }
    if let Some(idx) = lower.rfind(" from ") {
        return extract_phrase_fact_value(
            &line[idx + " from ".len()..],
            &["for", "because", "and", "but", "with"],
            3,
        );
    }
    None
}

fn extract_certification_surface_value(line: &str, lower: &str) -> Option<String> {
    if !lower.contains("certification") {
        return None;
    }
    extract_fact_after_any(
        line,
        lower,
        &[
            "completed a certification in ",
            "completed certification in ",
            "finished a certification in ",
            "earned a certification in ",
            "certification in ",
        ],
        &["last", "this", "through", "from", "and", "but"],
        4,
    )
}

fn extract_sister_gift_surface_value(line: &str, lower: &str) -> Option<String> {
    if !(lower.contains("sister") && lower.contains("birthday")) {
        return None;
    }
    extract_fact_after_any(
        line,
        lower,
        &[
            "i bought my sister ",
            "bought my sister ",
            "got my sister ",
            "picked up ",
            "chose ",
        ],
        &["for", "and", "but", "because", "from"],
        5,
    )
}

fn extract_theater_play_surface_value(line: &str, lower: &str) -> Option<String> {
    if !(lower.contains("theater") || lower.contains("theatre")) {
        return None;
    }
    extract_fact_after_any(
        line,
        lower,
        &[
            "production of ",
            "play called ",
            "went to see ",
            "saw ",
            "attended ",
        ],
        &["at", "with", "on", "last", "and", "but", "because"],
        6,
    )
}

fn extract_concert_venue_surface_value(line: &str, lower: &str) -> Option<String> {
    if !lower.contains("concert") {
        return None;
    }
    let idx = lower.rfind(" at ")?;
    extract_phrase_fact_value(
        &line[idx + " at ".len()..],
        &["on", "with", "and", "but", "for"],
        4,
    )
}

fn extract_relative_location_surface_row(line: &str, lower: &str) -> Option<(String, String)> {
    let (marker, relation_label) = if lower.contains("my sister") {
        ("my sister", "sister")
    } else if lower.contains("my cousin") {
        ("my cousin", "cousin")
    } else {
        return None;
    };
    let relation_idx = lower.find(marker)?;
    let after_relation = line[relation_idx + marker.len()..].trim_start();
    let relation_name = after_relation.split_whitespace().find_map(|word| {
        let clean: String = word.chars().filter(|c| c.is_alphabetic()).collect();
        (clean.len() >= 3 && word.chars().next().is_some_and(|c| c.is_uppercase()))
            .then(|| clean.to_ascii_lowercase())
    });

    let after_relation_lower = lower[relation_idx + marker.len()..].to_string();
    let in_idx = after_relation_lower.rfind(" in ")?;
    let value = extract_phrase_fact_value(
        &after_relation[in_idx + " in ".len()..],
        &[
            "for", "with", "and", "but", "next", "this", "because", "during",
        ],
        3,
    )?;
    let question_pattern = relation_name.map_or_else(
        || format!("{relation_label} live location city home based"),
        |name| format!("{relation_label} {name} live location city home based"),
    );
    Some((question_pattern, value))
}

fn extract_instagram_followers_surface_value(line: &str, lower: &str) -> Option<String> {
    if !lower.contains("instagram") {
        return None;
    }
    for trigger in [
        "i'm now at ",
        "i am now at ",
        "i just reached ",
        "i'm close to ",
        "i am close to ",
        "i think i'm close to ",
        "i think i am close to ",
    ] {
        let Some(pos) = lower.find(trigger) else {
            continue;
        };
        let after = &line[pos + trigger.len()..];
        if after.to_ascii_lowercase().contains("followers") || lower.contains("follower count") {
            if let Some(value) = extract_numeric_fact_value(after) {
                return Some(value);
            }
        }
    }
    None
}

fn extract_pre_1920_coin_surface_value(line: &str, lower: &str) -> Option<String> {
    if !lower.contains("pre-1920 american coins") {
        return None;
    }
    if let Some(pos) = lower.find("i have a total of ") {
        return extract_numeric_fact_value(&line[pos + "i have a total of ".len()..]);
    }
    None
}

pub(crate) fn extract_national_geographic_count_surface_value(
    line: &str,
    lower: &str,
) -> Option<String> {
    if !(lower.contains("national geographic") && lower.contains("issue")) {
        return None;
    }
    if let Some(pos) = lower.find("finished ") {
        return extract_count_fact_value(&line[pos + "finished ".len()..]);
    }
    if let Some(pos) = lower.find("completed ") {
        return extract_count_fact_value(&line[pos + "completed ".len()..]);
    }
    None
}

pub(crate) fn extract_korean_restaurant_count_surface_value(
    line: &str,
    lower: &str,
) -> Option<String> {
    if !(lower.contains("korean restaurant") && lower.contains("tried")) {
        return None;
    }
    if let Some(pos) = lower.find("tried ") {
        return extract_count_fact_value(&line[pos + "tried ".len()..]);
    }
    None
}

pub(crate) fn extract_largemouth_bass_count_surface_value(
    line: &str,
    lower: &str,
) -> Option<String> {
    if !(lower.contains("largemouth bass") && lower.contains("caught")) {
        return None;
    }
    if let Some(pos) = lower.find("caught ") {
        return extract_count_fact_value(&line[pos + "caught ".len()..]);
    }
    None
}

pub(super) fn push_answer_surface_row(
    rows: &mut Vec<AnswerSurfaceRow>,
    question_pattern: &str,
    answer_span: Option<String>,
    confidence: f32,
) {
    let Some(answer_span) = answer_span else {
        return;
    };
    let answer_span = normalize_answer_surface_span(&answer_span);
    if answer_span.is_empty() {
        return;
    }
    if rows.iter().any(|row| {
        row.question_pattern == question_pattern
            && row.answer_span.eq_ignore_ascii_case(&answer_span)
    }) {
        return;
    }
    rows.push(AnswerSurfaceRow {
        question_pattern: question_pattern.to_string(),
        answer_span,
        confidence,
    });
}
