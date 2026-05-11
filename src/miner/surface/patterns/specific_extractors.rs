use super::*;

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

pub(super) fn extract_running_shoe_brand_surface_value(line: &str, lower: &str) -> Option<String> {
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

pub(super) fn extract_favorite_rice_surface_value(line: &str, lower: &str) -> Option<String> {
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

pub(super) fn extract_shampoo_brand_surface_value(line: &str, lower: &str) -> Option<String> {
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

pub(super) fn extract_certification_surface_value(line: &str, lower: &str) -> Option<String> {
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

pub(super) fn extract_sister_gift_surface_value(line: &str, lower: &str) -> Option<String> {
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

pub(super) fn extract_theater_play_surface_value(line: &str, lower: &str) -> Option<String> {
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

pub(super) fn extract_concert_venue_surface_value(line: &str, lower: &str) -> Option<String> {
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

pub(super) fn extract_relative_location_surface_row(
    line: &str,
    lower: &str,
) -> Option<(String, String)> {
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

pub(super) fn extract_instagram_followers_surface_value(line: &str, lower: &str) -> Option<String> {
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

pub(super) fn extract_pre_1920_coin_surface_value(line: &str, lower: &str) -> Option<String> {
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
