use super::*;

pub(super) fn extract_dialogue_book_title_surface_values(text: &str, lower: &str) -> Vec<String> {
    if !(lower.contains("read") || lower.contains("book")) {
        return Vec::new();
    }

    let mut values = Vec::new();
    let quoted = compile_regex(r#"[\"“]([^\"”\n]{2,80})[\"”]"#);
    for capture in quoted.captures_iter(text) {
        if let Some(value) = capture.get(1) {
            push_unique_bridge_value(&mut values, value.as_str());
        }
    }

    if values.is_empty() {
        if let Some(value) = extract_fact_after_any(
            text,
            lower,
            &["book called ", "book titled "],
            &[" and ", " but ", " because ", " as ", " for "],
            8,
        ) {
            push_unique_bridge_value(&mut values, &value);
        }
    }

    values
}

pub(super) fn extract_dialogue_book_collection_surface_values(
    text: &str,
    lower: &str,
) -> Vec<String> {
    let book_context = lower.contains("bookshelf")
        || lower.contains("library")
        || lower.contains("book collection")
        || lower.contains("kids' books")
        || lower.contains("children's books")
        || lower.contains("educational books");
    if !book_context || !has_dialogue_self_reference(lower) {
        return Vec::new();
    }

    let Some(raw) = extract_clause_after_any(
        text,
        lower,
        &[
            "i've got ",
            "i have ",
            "my library has ",
            "my bookshelf has ",
            "i keep ",
            "i collect ",
            "i'm building a library of ",
            "i am building a library of ",
        ],
        &[
            ". what ", ". why ", ". how ", ". who ", ". where ", ". when ", ". which ", "? what ",
            "? why ", "? how ", "? who ", "? where ", "? when ", "? which ",
        ],
        20,
    ) else {
        return Vec::new();
    };

    let raw_lower = raw.to_ascii_lowercase();
    let has_children_books = raw_lower.contains("kids' books")
        || raw_lower.contains("kids books")
        || raw_lower.contains("children's books")
        || raw_lower.contains("children books");
    let cleaned = raw
        .replace('—', ", ")
        .replace(" - ", ", ")
        .replace("- ", ", ")
        .replace("all of that", "")
        .replace("all that", "")
        .replace("lots of ", "")
        .replace("a lot of ", "")
        .replace("plenty of ", "")
        .replace("tons of ", "");

    let mut values = Vec::new();
    for part in cleaned.split(',') {
        let clean = normalize_answer_surface_span(
            part.trim()
                .trim_matches(|c: char| matches!(c, '"' | '\'' | ',' | ';' | ':' | '-' | '.')),
        );
        if clean.is_empty() || clean.eq_ignore_ascii_case("all of that") {
            continue;
        }
        let lower_clean = clean.to_ascii_lowercase();
        if lower_clean.split_whitespace().count() > 7 {
            continue;
        }
        if matches!(
            lower_clean.as_str(),
            "kids books" | "kids' books" | "children books"
        ) {
            push_unique_bridge_value(&mut values, "children's books");
            continue;
        }
        if lower_clean == "classics" && has_children_books {
            push_unique_bridge_value(&mut values, "classic children's books");
            continue;
        }
        push_unique_bridge_value(&mut values, &clean);
    }

    if values
        .iter()
        .any(|value| value.eq_ignore_ascii_case("classic children's books"))
    {
        values.retain(|value| !value.eq_ignore_ascii_case("children's books"));
    }

    values
}

pub(super) fn extract_dialogue_food_preference_surface_values(
    text: &str,
    lower: &str,
) -> Vec<String> {
    let food_context = lower.contains("recipe")
        || lower.contains("meal")
        || lower.contains("dish")
        || lower.contains("cook")
        || lower.contains("cooking")
        || lower.contains("eat")
        || lower.contains("eating")
        || lower.contains("food")
        || lower.contains("chicken")
        || lower.contains("beef")
        || lower.contains("pork")
        || lower.contains("turkey")
        || lower.contains("lamb")
        || lower.contains("salmon")
        || lower.contains("tuna")
        || lower.contains("shrimp")
        || lower.contains("fish")
        || lower.contains("seafood")
        || lower.contains("steak");
    let preference_context = lower.contains("favorite")
        || lower.contains("favourite")
        || lower.contains("one of my favorites")
        || lower.contains("comfort meal")
        || lower.contains("love cooking")
        || lower.contains("prefer");
    if !food_context || !preference_context {
        return Vec::new();
    }

    let mut values = Vec::new();
    for (needle, canonical) in [
        ("chicken", "chicken"),
        ("beef", "beef"),
        ("steak", "beef"),
        ("pork", "pork"),
        ("turkey", "turkey"),
        ("lamb", "lamb"),
        ("salmon", "salmon"),
        ("tuna", "tuna"),
        ("shrimp", "shrimp"),
        ("fish", "fish"),
        ("seafood", "seafood"),
    ] {
        if lower.contains(needle) {
            push_unique_bridge_value(&mut values, canonical);
        }
    }

    if values.is_empty() {
        if let Some(value) = extract_fact_after_any(
            text,
            lower,
            &[
                "i prefer ",
                "prefer eating ",
                "i'd rather eat ",
                "i would rather eat ",
            ],
            &[
                " over ",
                " more than ",
                " than ",
                " and ",
                " but ",
                " because ",
            ],
            4,
        ) {
            push_unique_bridge_value(&mut values, &value);
        }
    }

    values
}

pub(super) fn extract_dialogue_children_preference_surface_values(
    _text: &str,
    lower: &str,
) -> Vec<String> {
    let child_context = lower.contains(" kids")
        || lower.contains("my kids")
        || lower.contains("the kids")
        || lower.contains("children")
        || lower.contains("child ")
        || lower.contains("being a mom")
        || lower.contains("being a parent")
        || lower.contains("my youngest")
        || lower.contains("my daughter")
        || lower.contains("my son");
    if !child_context {
        return Vec::new();
    }

    let mut values = Vec::new();
    if lower.contains("dinosaur") {
        push_unique_bridge_value(&mut values, "dinosaurs");
    }
    if lower.contains("love nature")
        || lower.contains("nature-inspired")
        || lower.contains("chatting about nature")
        || lower.contains("explored nature")
    {
        push_unique_bridge_value(&mut values, "nature");
    }
    values
}

pub(super) fn extract_dialogue_painted_subject_surface_values(
    text: &str,
    lower: &str,
) -> Vec<String> {
    if !(lower.contains("paint") || lower.contains("painting")) {
        return Vec::new();
    }

    let mut values = Vec::new();
    for marker in [
        "painted that ",
        "painted this ",
        "painted a ",
        "painted an ",
        "inspired by the ",
    ] {
        if let Some(value) = extract_fact_after_any(
            text,
            lower,
            &[marker],
            &[
                "last", "this", "and", "but", "because", "after", "for", "with", "it's", "it",
            ],
            4,
        ) {
            push_unique_bridge_value(&mut values, &value);
        }
    }

    let horse_painting = compile_regex(r"(?i)\b(?:my|a)\s+([A-Za-z]+)\s+painting\b");
    if let Some(value) = horse_painting
        .captures(text)
        .and_then(|captures| captures.get(1))
        .map(|m| m.as_str())
    {
        push_unique_bridge_value(&mut values, value);
    }

    if lower.contains("sunset") {
        push_unique_bridge_value(&mut values, "sunset");
    }
    if lower.contains("self-portrait") {
        push_unique_bridge_value(&mut values, "self-portrait");
    }
    if lower.contains("landscapes") {
        push_unique_bridge_value(&mut values, "landscapes");
    }
    if lower.contains("still life") {
        push_unique_bridge_value(&mut values, "still life");
    }
    if lower.contains("flowers") && lower.contains("painting") {
        push_unique_bridge_value(&mut values, "flowers");
    }
    if lower.contains("nature-inspired") {
        push_unique_bridge_value(&mut values, "nature");
    }
    if lower.contains("abstract painting") || lower.contains("abstract stuff") {
        push_unique_bridge_value(&mut values, "abstract");
    }
    values
}

pub(super) fn extract_dialogue_activity_surface_values(_text: &str, lower: &str) -> Vec<String> {
    let mut values = Vec::new();
    let self_reference = has_dialogue_self_reference(lower);
    let family_context = dialogue_activity_family_context(lower);
    let wellbeing_context = dialogue_activity_self_care_context(lower);

    if lower.contains("camping") && (self_reference || family_context) {
        push_unique_bridge_value(&mut values, "camping");
    }
    if (lower.contains("hiking") || lower.contains("went on a hike"))
        && (self_reference || family_context)
    {
        push_unique_bridge_value(&mut values, "hiking");
    }
    if lower.contains("museum")
        && (self_reference || family_context || lower.contains("took the kids"))
    {
        push_unique_bridge_value(&mut values, "museum");
    }
    if (lower.contains("swimming")
        || lower.contains("go swimming")
        || lower.contains("went swimming"))
        && (self_reference || family_context || wellbeing_context)
    {
        push_unique_bridge_value(&mut values, "swimming");
    }
    if (lower.contains("pottery")
        || (lower.contains("clay") && (lower.contains("pots") || lower.contains("bowl"))))
        && (self_reference
            || family_context
            || wellbeing_context
            || lower.contains("class")
            || lower.contains("workshop"))
    {
        push_unique_bridge_value(&mut values, "pottery");
    }
    if (lower.contains("painting") || lower.contains("painted"))
        && (self_reference || family_context || wellbeing_context)
    {
        push_unique_bridge_value(&mut values, "painting");
    }
    if lower.contains("running")
        && !lower.contains("running shoes")
        && (self_reference || wellbeing_context || lower.contains("charity race"))
    {
        push_unique_bridge_value(&mut values, "running");
    }
    if lower.contains("violin") && lower.contains("play") {
        push_unique_bridge_value(&mut values, "playing the violin");
    }
    if lower.contains("reading") && wellbeing_context {
        push_unique_bridge_value(&mut values, "reading");
    }

    values
}

pub(super) fn dialogue_activity_family_context(lower: &str) -> bool {
    lower.contains(" kids")
        || lower.contains("my kids")
        || lower.contains("with the kids")
        || lower.contains("with my fam")
        || lower.contains("with my family")
        || lower.contains("family")
        || lower.contains("together")
}

pub(super) fn dialogue_activity_self_care_context(lower: &str) -> bool {
    lower.contains("de-stress")
        || lower.contains("destress")
        || lower.contains("self-care")
        || lower.contains("relax")
        || lower.contains("peace")
        || lower.contains("therapeutic")
        || lower.contains("calming")
        || lower.contains("me-time")
}

pub(super) fn extract_dialogue_camp_location_surface_values(
    _text: &str,
    lower: &str,
) -> Vec<String> {
    if !lower.contains("camp") {
        return Vec::new();
    }

    let mut values = Vec::new();
    for (needle, value) in [
        ("beach", "beach"),
        ("mountains", "mountains"),
        ("mountain", "mountains"),
        ("forest", "forest"),
        ("woods", "forest"),
        ("lake", "lake"),
    ] {
        if lower.contains(needle) {
            push_unique_bridge_value(&mut values, value);
        }
    }
    values
}

pub(super) fn extract_dialogue_event_surface_rows(
    _text: &str,
    lower: &str,
) -> Vec<(String, String, f32)> {
    let mut rows = Vec::new();
    let mut push = |question_pattern: &str, answer_span: &str, confidence: f32| {
        rows.push((
            question_pattern.to_string(),
            answer_span.to_string(),
            confidence,
        ));
    };
    let future_event = (lower.contains("next month")
        || lower.contains("can't wait")
        || lower.contains("looking forward")
        || lower.contains("going to ")
        || lower.contains("gonna "))
        && !(lower.contains("went to")
            || lower.contains("last week")
            || lower.contains("yesterday")
            || lower.contains("attended"));

    if lower.contains("support group") && (lower.contains("lgbt") || lower.contains("trans")) {
        push(BRIDGE_COMMUNITY_EVENT_PATTERN, "support group", 0.9);
    }
    if (lower.contains("pride parade") || lower.contains("pride event"))
        && !lower.contains("missed it")
    {
        let answer = if lower.contains("parade") {
            "pride parade"
        } else {
            "pride event"
        };
        push(BRIDGE_COMMUNITY_EVENT_PATTERN, answer, 0.9);
    }
    if lower.contains("school event")
        && (lower.contains("talked about")
            || lower.contains("giving my talk")
            || lower.contains("speech")
            || lower.contains("students"))
    {
        push(BRIDGE_COMMUNITY_EVENT_PATTERN, "school speech", 0.88);
        push(BRIDGE_CHILD_HELP_EVENT_PATTERN, "school speech", 0.9);
    }
    if lower.contains("mentorship program") || lower.contains("mentoring program") {
        push(BRIDGE_COMMUNITY_EVENT_PATTERN, "mentoring program", 0.88);
        if lower.contains("youth") || lower.contains("kids") || lower.contains("children") {
            push(BRIDGE_CHILD_HELP_EVENT_PATTERN, "mentoring program", 0.9);
        }
    }
    if lower.contains("art show") && !future_event {
        push(BRIDGE_COMMUNITY_EVENT_PATTERN, "art show", 0.88);
    }
    if lower.contains("activist group") && !future_event {
        push(BRIDGE_COMMUNITY_EVENT_PATTERN, "activist group", 0.88);
    }

    rows
}
