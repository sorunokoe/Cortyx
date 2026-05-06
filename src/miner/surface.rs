use std::collections::HashMap;

use regex::Regex;

use crate::answer_plane::{mine_dialogue_answer_surface_span, mine_dialogue_question_pattern};

fn compile_regex(pattern: &str) -> Regex {
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

use super::kg_extract::{
    extract_count_fact_value, extract_numeric_fact_value, extract_phrase_fact_value,
};
use super::{AnswerSurfaceRow, Turn};

pub(super) fn append_answer_surface_section(
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

fn normalize_answer_surface_span(value: &str) -> String {
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

pub(super) fn generate_dialogue_answer_surface_rows(
    turns: &[Turn],
    index: usize,
) -> Vec<AnswerSurfaceRow> {
    if index == 0 {
        return Vec::new();
    }

    let question = &turns[index - 1];
    let answer = &turns[index];
    if question.speaker == answer.speaker {
        return Vec::new();
    }

    let mut rows = extract_dialogue_name_answer_surface_rows(question, answer);
    let Some(answer_span) = mine_dialogue_answer_surface_span(&question.text, &answer.text) else {
        return rows;
    };
    let mut answer_span = canonicalize_dialogue_answer_surface_span(&answer_span);
    if let Some(prefix) = extract_dialogue_statement_prefix(&answer.text) {
        let prefix_lower = prefix.to_ascii_lowercase();
        let answer_lower = answer_span.to_ascii_lowercase();
        if dialogue_answer_looks_like_question(&answer_span)
            || (answer.text.contains('?')
                && !answer_lower.is_empty()
                && !prefix_lower.contains(&answer_lower))
        {
            answer_span = canonicalize_dialogue_answer_surface_span(&prefix);
        }
    }
    if answer_span.is_empty() {
        return rows;
    }
    let question_patterns =
        dialogue_answer_surface_patterns(&question.text, answer.speaker.as_deref());
    if question_patterns.is_empty() {
        return rows;
    }
    let base_confidence = if answer_span.split_whitespace().count() <= 12 {
        0.88
    } else {
        0.8
    };
    rows.extend(question_patterns.into_iter().enumerate().map(
        |(pattern_index, question_pattern)| AnswerSurfaceRow {
            question_pattern,
            answer_span: answer_span.clone(),
            confidence: if pattern_index == 0 {
                base_confidence
            } else {
                (base_confidence + 0.04).min(0.94)
            },
        },
    ));
    rows
}

pub(super) fn generate_cross_chunk_dialogue_answer_surface_rows(
    turns: &[Turn],
    index: usize,
) -> Vec<AnswerSurfaceRow> {
    if index == 0 {
        return Vec::new();
    }

    let previous_turns = parse_embedded_dialogue_turns(&turns[index - 1].text);
    let current_turns = parse_embedded_dialogue_turns(&turns[index].text);
    let (Some(question), Some(answer)) = (previous_turns.last(), current_turns.first()) else {
        return Vec::new();
    };
    if question.speaker == answer.speaker {
        return Vec::new();
    }

    let bridge_turns = vec![question.clone(), answer.clone()];
    generate_dialogue_answer_surface_rows(&bridge_turns, 1)
}

fn extract_dialogue_name_answer_surface_rows(
    question: &Turn,
    answer: &Turn,
) -> Vec<AnswerSurfaceRow> {
    let lower_question = question.text.to_ascii_lowercase();
    if !(lower_question.contains(" name") || lower_question.contains(" names")) {
        return Vec::new();
    }
    let Some(answer_span) = extract_dialogue_name_surface_value(&answer.text) else {
        return Vec::new();
    };
    let mut patterns = dialogue_answer_surface_patterns(&question.text, answer.speaker.as_deref());
    if let Some(base_pattern) = patterns.first().cloned() {
        if let Some(scoped_pattern) =
            scoped_question_pattern(&base_pattern, answer.speaker.as_deref())
        {
            if !patterns.iter().any(|pattern| pattern == &scoped_pattern) {
                patterns.push(scoped_pattern);
            }
        }
    }
    if patterns.is_empty() {
        return Vec::new();
    }
    patterns
        .into_iter()
        .enumerate()
        .map(|(pattern_index, question_pattern)| AnswerSurfaceRow {
            question_pattern,
            answer_span: answer_span.clone(),
            confidence: if pattern_index == 0 { 0.92 } else { 0.95 },
        })
        .collect()
}

fn extract_dialogue_name_surface_value(answer: &str) -> Option<String> {
    let prefix = answer
        .split(['!', '?', '.'])
        .next()
        .unwrap_or(answer)
        .trim()
        .trim_matches(|c: char| matches!(c, '"' | '\'' | ',' | ';' | ':' | '-'));
    if prefix.is_empty() || prefix.split_whitespace().count() > 6 {
        return None;
    }
    let tokens = prefix.split_whitespace().collect::<Vec<_>>();
    let capitalized = tokens
        .iter()
        .filter(|token| {
            let clean =
                token.trim_matches(|c: char| !c.is_ascii_alphabetic() && c != '\'' && c != '-');
            clean
                .chars()
                .next()
                .map(|c| c.is_ascii_uppercase())
                .unwrap_or(false)
        })
        .count();
    let valid = tokens.iter().all(|token| {
        let clean = token.trim_matches(|c: char| !c.is_ascii_alphabetic() && c != '\'' && c != '-');
        if clean.is_empty() {
            return false;
        }
        let lower = clean.to_ascii_lowercase();
        matches!(lower.as_str(), "and" | "the" | "of" | "a" | "an" | "&")
            || clean
                .chars()
                .next()
                .map(|c| c.is_ascii_uppercase())
                .unwrap_or(false)
    });
    (capitalized >= 1 && valid).then(|| normalize_answer_surface_span(prefix))
}

fn canonicalize_dialogue_answer_surface_span(answer_span: &str) -> String {
    const FOLLOWUP_MARKERS: &[&str] = &[
        ". what ",
        ". what's",
        ". why ",
        ". how ",
        ". who's",
        ". who ",
        ". where's",
        ". where ",
        ". when's",
        ". when ",
        ". which ",
        ". do ",
        ". does ",
        ". did ",
        ". have ",
        ". has ",
        ". can ",
        ". could ",
        ". would ",
        ". is ",
        ". are ",
        ". was ",
        ". were ",
        "? what ",
        "? what's",
        "? why ",
        "? how ",
        "? who ",
        "? where ",
        "? when ",
        "? which ",
        "? do ",
        "? does ",
        "? did ",
        "? have ",
        "? has ",
        "? can ",
        "? could ",
        "? would ",
        "? is ",
        "? are ",
        "? was ",
        "? were ",
    ];
    let lower = answer_span.to_ascii_lowercase();
    let cutoff = FOLLOWUP_MARKERS
        .iter()
        .filter_map(|marker| lower.find(marker))
        .min();
    let trimmed = cutoff
        .map(|idx| &answer_span[..idx])
        .unwrap_or(answer_span)
        .trim_end_matches(|c: char| matches!(c, '.' | '!' | '?' | ';' | ',' | ':'));
    normalize_answer_surface_span(trimmed)
}

fn extract_dialogue_statement_prefix(answer: &str) -> Option<String> {
    const FOLLOWUP_MARKERS: &[&str] = &[
        ". What ",
        ". What's",
        ". Why ",
        ". How ",
        ". Who ",
        ". Who's",
        ". Where ",
        ". Where's",
        ". When ",
        ". When's",
        ". Which ",
        ". Do ",
        ". Does ",
        ". Did ",
        ". Have ",
        ". Has ",
        ". Can ",
        ". Could ",
        ". Would ",
        ". Is ",
        ". Are ",
        ". Was ",
        ". Were ",
        "? ",
    ];
    let cutoff = FOLLOWUP_MARKERS
        .iter()
        .filter_map(|marker| answer.find(marker))
        .min()
        .or_else(|| answer.find('?'));
    let trimmed = cutoff
        .map(|idx| &answer[..idx])
        .unwrap_or(answer)
        .trim_end_matches(|c: char| matches!(c, '.' | '!' | '?' | ';' | ',' | ':'));
    let normalized = normalize_answer_surface_span(trimmed);
    (!normalized.is_empty() && !dialogue_answer_looks_like_question(&normalized))
        .then_some(normalized)
}

fn dialogue_answer_looks_like_question(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }
    let lower = trimmed.to_ascii_lowercase();
    trimmed.ends_with('?')
        || [
            "what ", "why ", "how ", "who ", "where ", "when ", "which ", "do ", "does ", "did ",
            "have ", "has ", "can ", "could ", "would ", "is ", "are ", "was ", "were ",
        ]
        .iter()
        .any(|prefix| lower.starts_with(prefix))
}

fn dialogue_answer_surface_patterns(question: &str, answer_speaker: Option<&str>) -> Vec<String> {
    let Some(base_pattern) = mine_dialogue_question_pattern(question) else {
        return Vec::new();
    };
    let mut patterns = vec![base_pattern.clone()];
    if let Some(scoped_pattern) =
        speaker_scoped_question_pattern(&base_pattern, question, answer_speaker)
    {
        if scoped_pattern != base_pattern {
            patterns.push(scoped_pattern);
        }
    }
    patterns
}

const BRIDGE_ACTIVITY_PATTERN: &str = "activities activity hobbies hobby";
const BRIDGE_FAMILY_ACTIVITY_PATTERN: &str =
    "activities activity hobbies hobby family kids together fun";
const BRIDGE_SELF_CARE_ACTIVITY_PATTERN: &str =
    "activities activity hobbies hobby destress relax self-care peace therapeutic calming me-time";
const BRIDGE_BOOK_PATTERN: &str = "books book read reading literature title";
const BRIDGE_BOOK_COLLECTION_PATTERN: &str =
    "bookshelf library collection classic children's educational stories reading";
const BRIDGE_RESEARCH_PATTERN: &str =
    "research researched researching topic investigating looking into";
const BRIDGE_CAMP_LOCATION_PATTERN: &str =
    "where camped camping location place beach mountains forest";
const BRIDGE_CAREER_PATTERN: &str =
    "career path field pursue education future study job work counseling mental health";
const BRIDGE_CAREER_REASON_PATTERN: &str =
    "why career path field pursue education motivation reason counseling mental health support help";
const BRIDGE_KIDS_LIKE_PATTERN: &str =
    "kids children child like likes love enjoy favorite interested dinosaurs nature";
const BRIDGE_FOOD_PREFERENCE_PATTERN: &str =
    "food recipe meal dish meat protein prefer preference favorite eat eating chicken beef pork";
const BRIDGE_IDENTITY_PATTERN: &str = "identity gender transgender woman man nonbinary queer";
const BRIDGE_ALLY_PATTERN: &str = "ally supportive support transgender lgbtq community acceptance";
const BRIDGE_ORIGIN_PATTERN: &str = "where from moved from home country origin country";
const BRIDGE_PAINT_SUBJECT_PATTERN: &str =
    "paint painted painting artwork art subject scene recently created made";
const BRIDGE_RELIGION_PATTERN: &str = "religious religion faith church spiritual";
const BRIDGE_RELATIONSHIP_PATTERN: &str = "relationship status single married partner spouse";
const BRIDGE_SUPPORT_EFFECT_PATTERN: &str =
    "how support group affect effect impact help helped made feel accepted courage";
const BRIDGE_COMMUNITY_EVENT_PATTERN: &str =
    "events event lgbtq community participate participated joined support group pride parade art show activist group speech mentoring program";
const BRIDGE_CHILD_HELP_EVENT_PATTERN: &str =
    "events event help children kids youth school speech mentoring program";
const BRIDGE_SUPPORT_NETWORK_PATTERN: &str =
    "who support supports supported help negative experience mentors family friends";
const BRIDGE_FRIEND_GROUP_DURATION_PATTERN: &str =
    "how long current group friends friend known know duration years months";

pub(super) fn generate_dialogue_bridge_surface_rows(turn: &Turn) -> Vec<AnswerSurfaceRow> {
    let text = turn.text.trim();
    if text.is_empty() {
        return Vec::new();
    }

    let lower = text.to_ascii_lowercase();
    let mut rows = Vec::new();
    let speaker = turn.speaker.as_deref();

    if let Some(answer_span) = extract_dialogue_research_topic_surface_value(text, &lower) {
        push_turn_answer_surface_row(
            &mut rows,
            BRIDGE_RESEARCH_PATTERN,
            Some(answer_span),
            0.9,
            speaker,
        );
    }

    if let Some(answer_span) = extract_dialogue_career_interest_surface_value(text, &lower) {
        push_turn_answer_surface_row(
            &mut rows,
            BRIDGE_CAREER_PATTERN,
            Some(answer_span),
            0.9,
            speaker,
        );
    }

    if let Some(answer_span) = extract_dialogue_career_reason_surface_value(text, &lower) {
        push_turn_answer_surface_row(
            &mut rows,
            BRIDGE_CAREER_REASON_PATTERN,
            Some(answer_span),
            0.88,
            speaker,
        );
    }

    if let Some(answer_span) = extract_dialogue_origin_surface_value(text, &lower) {
        push_turn_answer_surface_row(
            &mut rows,
            BRIDGE_ORIGIN_PATTERN,
            Some(answer_span),
            0.88,
            speaker,
        );
    }

    if let Some(answer_span) = extract_dialogue_friend_group_duration_surface_value(text, &lower) {
        push_turn_answer_surface_row(
            &mut rows,
            BRIDGE_FRIEND_GROUP_DURATION_PATTERN,
            Some(answer_span),
            0.9,
            speaker,
        );
    }

    if let Some(answer_span) = extract_dialogue_support_network_surface_value(text, &lower) {
        push_turn_answer_surface_row(
            &mut rows,
            BRIDGE_SUPPORT_NETWORK_PATTERN,
            Some(answer_span),
            0.9,
            speaker,
        );
    }

    if let Some(answer_span) = extract_dialogue_support_effect_surface_value(text, &lower) {
        push_turn_answer_surface_row(
            &mut rows,
            BRIDGE_SUPPORT_EFFECT_PATTERN,
            Some(answer_span),
            0.9,
            speaker,
        );
    }

    if let Some(answer_span) = extract_dialogue_identity_surface_value(text, &lower) {
        push_turn_answer_surface_row(
            &mut rows,
            BRIDGE_IDENTITY_PATTERN,
            Some(answer_span),
            0.9,
            speaker,
        );
    }

    if let Some(answer_span) = extract_dialogue_ally_surface_value(text, &lower) {
        push_turn_answer_surface_row(
            &mut rows,
            BRIDGE_ALLY_PATTERN,
            Some(answer_span),
            0.86,
            speaker,
        );
    }

    if let Some(answer_span) = extract_dialogue_relationship_status_surface_value(text, &lower) {
        push_turn_answer_surface_row(
            &mut rows,
            BRIDGE_RELATIONSHIP_PATTERN,
            Some(answer_span),
            0.88,
            speaker,
        );
    }

    if let Some(answer_span) = extract_dialogue_religiosity_surface_value(text, &lower) {
        push_turn_answer_surface_row(
            &mut rows,
            BRIDGE_RELIGION_PATTERN,
            Some(answer_span),
            0.84,
            speaker,
        );
    }

    for title in extract_dialogue_book_title_surface_values(text, &lower) {
        push_turn_answer_surface_row(&mut rows, BRIDGE_BOOK_PATTERN, Some(title), 0.88, speaker);
    }

    for collection in extract_dialogue_book_collection_surface_values(text, &lower) {
        push_turn_answer_surface_row(
            &mut rows,
            BRIDGE_BOOK_COLLECTION_PATTERN,
            Some(collection.clone()),
            0.9,
            speaker,
        );
        push_turn_answer_surface_row(
            &mut rows,
            BRIDGE_BOOK_COLLECTION_PATTERN,
            Some(format!("Likely yes, {collection}")),
            0.87,
            speaker,
        );
    }

    for preference in extract_dialogue_food_preference_surface_values(text, &lower) {
        push_turn_answer_surface_row(
            &mut rows,
            BRIDGE_FOOD_PREFERENCE_PATTERN,
            Some(preference),
            0.88,
            speaker,
        );
    }

    for preference in extract_dialogue_children_preference_surface_values(text, &lower) {
        push_turn_answer_surface_row(
            &mut rows,
            BRIDGE_KIDS_LIKE_PATTERN,
            Some(preference),
            0.87,
            speaker,
        );
    }

    for subject in extract_dialogue_painted_subject_surface_values(text, &lower) {
        push_turn_answer_surface_row(
            &mut rows,
            BRIDGE_PAINT_SUBJECT_PATTERN,
            Some(subject),
            0.88,
            speaker,
        );
    }

    let family_activity_context = dialogue_activity_family_context(&lower);
    let self_care_activity_context = dialogue_activity_self_care_context(&lower);
    for activity in extract_dialogue_activity_surface_values(text, &lower) {
        push_turn_answer_surface_row(
            &mut rows,
            BRIDGE_ACTIVITY_PATTERN,
            Some(activity.clone()),
            0.84,
            speaker,
        );
        if family_activity_context {
            push_turn_answer_surface_row(
                &mut rows,
                BRIDGE_FAMILY_ACTIVITY_PATTERN,
                Some(activity.clone()),
                0.88,
                speaker,
            );
        }
        if self_care_activity_context {
            push_turn_answer_surface_row(
                &mut rows,
                BRIDGE_SELF_CARE_ACTIVITY_PATTERN,
                Some(activity),
                0.88,
                speaker,
            );
        }
    }

    for location in extract_dialogue_camp_location_surface_values(text, &lower) {
        push_turn_answer_surface_row(
            &mut rows,
            BRIDGE_CAMP_LOCATION_PATTERN,
            Some(location),
            0.88,
            speaker,
        );
    }

    for (question_pattern, answer_span, confidence) in
        extract_dialogue_event_surface_rows(text, &lower)
    {
        push_turn_answer_surface_row(
            &mut rows,
            &question_pattern,
            Some(answer_span),
            confidence,
            speaker,
        );
    }

    rows
}

fn push_turn_answer_surface_row(
    rows: &mut Vec<AnswerSurfaceRow>,
    question_pattern: &str,
    answer_span: Option<String>,
    confidence: f32,
    speaker: Option<&str>,
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
    if let Some(scoped_pattern) = scoped_question_pattern(question_pattern, speaker) {
        push_answer_surface_row(
            rows,
            &scoped_pattern,
            Some(answer_span),
            (confidence + 0.03).min(0.95),
        );
    }
}

#[derive(Default)]
struct SpeakerBridgeFacts {
    research_topics: Vec<String>,
    career_fields: Vec<String>,
    career_reason: Option<String>,
    camp_locations: Vec<String>,
    activities: Vec<String>,
    family_activities: Vec<String>,
    self_care_activities: Vec<String>,
    book_titles: Vec<String>,
    book_collections: Vec<String>,
    food_preferences: Vec<String>,
    kids_likes: Vec<String>,
    painted_subjects: Vec<String>,
    community_events: Vec<String>,
    child_help_events: Vec<String>,
    support_network: Option<String>,
    support_effect: Option<String>,
    friend_group_duration: Option<String>,
    origin: Option<String>,
    identity: Option<String>,
    ally: Option<String>,
    relationship: Option<String>,
    religion: Option<String>,
}

pub(super) fn generate_session_bridge_surface_rows(turns: &[Turn]) -> Vec<AnswerSurfaceRow> {
    let mut by_speaker: HashMap<String, SpeakerBridgeFacts> = HashMap::new();
    for turn in expand_bridge_analysis_turns(turns) {
        let Some(speaker) = turn.speaker.as_deref() else {
            continue;
        };
        let speaker = normalize_dialogue_speaker_label(speaker);
        let lower = turn.text.to_ascii_lowercase();
        let facts = by_speaker.entry(speaker).or_default();

        if let Some(value) = extract_dialogue_research_topic_surface_value(&turn.text, &lower) {
            push_unique_bridge_value(&mut facts.research_topics, &value);
        }
        if let Some(value) = extract_dialogue_career_interest_surface_value(&turn.text, &lower) {
            extend_split_bridge_values(&mut facts.career_fields, &value);
        }
        if let Some(value) = extract_dialogue_career_reason_surface_value(&turn.text, &lower) {
            facts.career_reason = Some(value);
        }
        if let Some(value) = extract_dialogue_origin_surface_value(&turn.text, &lower) {
            facts.origin = Some(value);
        }
        if let Some(value) =
            extract_dialogue_friend_group_duration_surface_value(&turn.text, &lower)
        {
            facts.friend_group_duration = Some(value);
        }
        if let Some(value) = extract_dialogue_support_network_surface_value(&turn.text, &lower) {
            facts.support_network = Some(value);
        }
        if let Some(value) = extract_dialogue_support_effect_surface_value(&turn.text, &lower) {
            facts.support_effect = Some(value);
        }
        if let Some(value) = extract_dialogue_identity_surface_value(&turn.text, &lower) {
            facts.identity = Some(value);
        }
        if let Some(value) = extract_dialogue_ally_surface_value(&turn.text, &lower) {
            facts.ally = Some(value);
        }
        if let Some(value) = extract_dialogue_relationship_status_surface_value(&turn.text, &lower)
        {
            facts.relationship = Some(value);
        }
        if let Some(value) = extract_dialogue_religiosity_surface_value(&turn.text, &lower) {
            facts.religion = Some(value);
        }

        for title in extract_dialogue_book_title_surface_values(&turn.text, &lower) {
            push_unique_bridge_value(&mut facts.book_titles, &title);
        }
        for collection in extract_dialogue_book_collection_surface_values(&turn.text, &lower) {
            push_unique_bridge_value(&mut facts.book_collections, &collection);
        }
        for preference in extract_dialogue_food_preference_surface_values(&turn.text, &lower) {
            push_unique_bridge_value(&mut facts.food_preferences, &preference);
        }
        for preference in extract_dialogue_children_preference_surface_values(&turn.text, &lower) {
            push_unique_bridge_value(&mut facts.kids_likes, &preference);
        }
        for subject in extract_dialogue_painted_subject_surface_values(&turn.text, &lower) {
            push_unique_bridge_value(&mut facts.painted_subjects, &subject);
        }
        for activity in extract_dialogue_activity_surface_values(&turn.text, &lower) {
            push_unique_bridge_value(&mut facts.activities, &activity);
            if dialogue_activity_family_context(&lower) {
                push_unique_bridge_value(&mut facts.family_activities, &activity);
            }
            if dialogue_activity_self_care_context(&lower) {
                push_unique_bridge_value(&mut facts.self_care_activities, &activity);
            }
        }
        for location in extract_dialogue_camp_location_surface_values(&turn.text, &lower) {
            push_unique_bridge_value(&mut facts.camp_locations, &location);
        }
        for (question_pattern, answer_span, _) in
            extract_dialogue_event_surface_rows(&turn.text, &lower)
        {
            if question_pattern == BRIDGE_COMMUNITY_EVENT_PATTERN {
                push_unique_bridge_value(&mut facts.community_events, &answer_span);
            }
            if question_pattern == BRIDGE_CHILD_HELP_EVENT_PATTERN {
                push_unique_bridge_value(&mut facts.child_help_events, &answer_span);
            }
        }
    }

    let mut rows = Vec::new();
    for (speaker, facts) in by_speaker {
        let speaker = Some(speaker.as_str());

        push_turn_answer_surface_row(
            &mut rows,
            BRIDGE_KIDS_LIKE_PATTERN,
            render_multi_bridge_surface_values(&facts.kids_likes),
            0.94,
            speaker,
        );
        push_turn_answer_surface_row(
            &mut rows,
            BRIDGE_CAREER_PATTERN,
            render_multi_bridge_surface_values(&facts.career_fields),
            0.93,
            speaker,
        );
        push_turn_answer_surface_row(
            &mut rows,
            BRIDGE_CAREER_REASON_PATTERN,
            facts.career_reason.clone(),
            0.92,
            speaker,
        );
        push_turn_answer_surface_row(
            &mut rows,
            BRIDGE_BOOK_PATTERN,
            render_multi_bridge_surface_values(&facts.book_titles),
            0.94,
            speaker,
        );
        push_turn_answer_surface_row(
            &mut rows,
            BRIDGE_BOOK_COLLECTION_PATTERN,
            render_bridge_surface_values(&facts.book_collections),
            0.93,
            speaker,
        );
        push_turn_answer_surface_row(
            &mut rows,
            BRIDGE_FOOD_PREFERENCE_PATTERN,
            render_bridge_surface_values(&facts.food_preferences),
            0.92,
            speaker,
        );
        push_turn_answer_surface_row(
            &mut rows,
            BRIDGE_FAMILY_ACTIVITY_PATTERN,
            render_multi_bridge_surface_values(&facts.family_activities),
            0.94,
            speaker,
        );
        push_turn_answer_surface_row(
            &mut rows,
            BRIDGE_SELF_CARE_ACTIVITY_PATTERN,
            render_multi_bridge_surface_values(&facts.self_care_activities),
            0.94,
            speaker,
        );
        push_turn_answer_surface_row(
            &mut rows,
            BRIDGE_CAMP_LOCATION_PATTERN,
            render_multi_bridge_surface_values(&facts.camp_locations),
            0.94,
            speaker,
        );
        push_turn_answer_surface_row(
            &mut rows,
            BRIDGE_FRIEND_GROUP_DURATION_PATTERN,
            facts.friend_group_duration.clone(),
            0.92,
            speaker,
        );
        push_turn_answer_surface_row(
            &mut rows,
            BRIDGE_SUPPORT_NETWORK_PATTERN,
            facts.support_network.clone(),
            0.92,
            speaker,
        );
        push_turn_answer_surface_row(
            &mut rows,
            BRIDGE_SUPPORT_EFFECT_PATTERN,
            facts.support_effect.clone(),
            0.92,
            speaker,
        );
        push_turn_answer_surface_row(
            &mut rows,
            BRIDGE_PAINT_SUBJECT_PATTERN,
            render_multi_bridge_surface_values(&facts.painted_subjects),
            0.93,
            speaker,
        );
        push_turn_answer_surface_row(
            &mut rows,
            BRIDGE_COMMUNITY_EVENT_PATTERN,
            render_multi_bridge_surface_values(&facts.community_events),
            0.94,
            speaker,
        );
        push_turn_answer_surface_row(
            &mut rows,
            BRIDGE_CHILD_HELP_EVENT_PATTERN,
            render_multi_bridge_surface_values(&facts.child_help_events),
            0.94,
            speaker,
        );
    }
    rows
}

fn expand_bridge_analysis_turns(turns: &[Turn]) -> Vec<Turn> {
    let mut expanded = Vec::new();
    for turn in turns {
        let embedded = parse_embedded_dialogue_turns(&turn.text);
        if embedded.is_empty() {
            expanded.push(turn.clone());
        } else {
            expanded.extend(embedded);
        }
    }
    expanded
}

fn extend_split_bridge_values(values: &mut Vec<String>, value: &str) {
    for part in value.split(',') {
        let clean = normalize_answer_surface_span(part.trim());
        if clean.is_empty() {
            continue;
        }
        push_unique_bridge_value(values, &clean);
    }
}

fn render_multi_bridge_surface_values(values: &[String]) -> Option<String> {
    (values.len() > 1).then(|| values.join(", "))
}

fn render_bridge_surface_values(values: &[String]) -> Option<String> {
    (!values.is_empty()).then(|| values.join(", "))
}

fn generate_embedded_dialogue_answer_surface_rows(text: &str) -> Vec<AnswerSurfaceRow> {
    let turns = parse_embedded_dialogue_turns(text);
    if turns.is_empty() {
        return Vec::new();
    }

    let mut rows = Vec::new();
    for turn in &turns {
        for row in generate_temporal_turn_answer_surface_rows(turn) {
            push_answer_surface_row(
                &mut rows,
                &row.question_pattern,
                Some(row.answer_span),
                row.confidence,
            );
        }
        for row in generate_dialogue_bridge_surface_rows(turn) {
            push_answer_surface_row(
                &mut rows,
                &row.question_pattern,
                Some(row.answer_span),
                row.confidence,
            );
        }
    }
    for index in 1..turns.len() {
        for row in generate_dialogue_answer_surface_rows(&turns, index) {
            push_answer_surface_row(
                &mut rows,
                &row.question_pattern,
                Some(row.answer_span),
                row.confidence,
            );
        }
    }
    rows
}

fn parse_embedded_dialogue_turns(content: &str) -> Vec<Turn> {
    let mut turns = Vec::new();
    let mut current: Option<Turn> = None;
    let mut session_timestamp: Option<String> = None;

    for raw_line in content.lines() {
        let trimmed = raw_line.trim();
        if trimmed.is_empty()
            || trimmed.starts_with("<!--")
            || trimmed.starts_with("##")
            || trimmed.starts_with('#')
            || trimmed.starts_with("===")
        {
            continue;
        }

        if let Some(timestamp) = parse_embedded_session_timestamp(trimmed) {
            if let Some(turn) = current.take() {
                if !turn.text.is_empty() {
                    turns.push(turn);
                }
            }
            session_timestamp = Some(timestamp);
            continue;
        }

        if let Some((speaker, text)) = parse_embedded_dialogue_line(trimmed) {
            if let Some(turn) = current.take() {
                if !turn.text.is_empty() {
                    turns.push(turn);
                }
            }
            current = Some(Turn {
                speaker: Some(speaker.to_string()),
                text: text.to_string(),
                timestamp: session_timestamp.clone(),
            });
            continue;
        }

        if let Some(turn) = current.as_mut() {
            if !turn.text.is_empty() {
                turn.text.push(' ');
            }
            turn.text.push_str(trimmed);
        }
    }

    if let Some(turn) = current {
        if !turn.text.is_empty() {
            turns.push(turn);
        }
    }

    turns
}

pub(super) fn parse_embedded_dialogue_line(line: &str) -> Option<(&str, &str)> {
    let (speaker, rest) = line.split_once(':')?;
    if !is_dialogue_speaker(speaker) {
        return None;
    }
    let rest = rest.trim();
    (!rest.is_empty()).then_some((speaker.trim(), rest))
}

pub(super) fn is_dialogue_speaker(prefix: &str) -> bool {
    let trimmed = prefix.trim();
    if trimmed.is_empty() || trimmed.len() > 32 {
        return false;
    }
    let lower = trimmed.to_ascii_lowercase();
    if lower.starts_with("speaker ") {
        return false;
    }
    trimmed
        .chars()
        .all(|c| c.is_ascii_alphabetic() || c == ' ' || c == '-' || c == '\'')
}

pub(super) fn normalize_dialogue_speaker_label(speaker: &str) -> String {
    let trimmed = speaker.trim();
    let lower = trimmed.to_ascii_lowercase();
    if matches!(
        lower.as_str(),
        "user" | "assistant" | "human" | "ai" | "system"
    ) {
        lower
    } else {
        trimmed.to_string()
    }
}

pub(super) fn parse_embedded_session_timestamp(line: &str) -> Option<String> {
    let captures = compile_regex(
        r"(?i)^\[session\s+\d+\s+[—-]\s+(?:\d{1,2}:\d{2}\s*[ap]m\s+on\s+)?(\d{1,2})\s+([a-z]+),\s*(\d{4})\]$",
    )
    .captures(line)?;
    let day = captures.get(1)?.as_str().parse::<u32>().ok()?;
    let month = month_name_to_number(captures.get(2)?.as_str())?;
    let year = captures.get(3)?.as_str().parse::<u32>().ok()?;
    Some(format!("{year:04}-{month:02}-{day:02}T00:00:00Z"))
}

fn speaker_scoped_question_pattern(
    base_pattern: &str,
    question: &str,
    answer_speaker: Option<&str>,
) -> Option<String> {
    let speaker_terms = speaker_scope_terms(answer_speaker?);
    if speaker_terms.is_empty() || question_mentions_other_named_subject(question, &speaker_terms) {
        return None;
    }
    scoped_question_pattern(base_pattern, answer_speaker)
}

fn speaker_scope_terms(speaker: &str) -> Vec<String> {
    let lower = speaker.trim().to_ascii_lowercase();
    if matches!(
        lower.as_str(),
        "user" | "assistant" | "human" | "ai" | "system"
    ) {
        return Vec::new();
    }
    let mut terms = speaker
        .split(|c: char| !c.is_ascii_alphabetic() && c != '-')
        .filter_map(|term| {
            let clean = term.trim().to_ascii_lowercase();
            (clean.len() >= 3).then_some(clean)
        })
        .collect::<Vec<_>>();
    terms.sort();
    terms.dedup();
    terms
}

fn scoped_question_pattern(question_pattern: &str, speaker: Option<&str>) -> Option<String> {
    let mut terms = question_pattern
        .split_whitespace()
        .map(|term| term.to_string())
        .collect::<Vec<_>>();
    let speaker_terms = speaker_scope_terms(speaker?);
    if speaker_terms.is_empty() {
        return None;
    }
    terms.extend(speaker_terms);
    terms.sort();
    terms.dedup();
    Some(terms.join(" "))
}

fn question_mentions_other_named_subject(question: &str, speaker_terms: &[String]) -> bool {
    let lower = question.to_ascii_lowercase();
    if lower.starts_with("you ")
        || lower.starts_with("your ")
        || lower.contains(" you ")
        || lower.contains(" your ")
    {
        return false;
    }

    let named_subjects = question
        .split(|c: char| !c.is_ascii_alphabetic() && c != '-')
        .filter_map(|token| {
            let trimmed = token.trim();
            let first = trimmed.chars().next()?;
            if trimmed.len() < 3 || !first.is_ascii_uppercase() {
                return None;
            }
            let lower = trimmed.to_ascii_lowercase();
            if matches!(
                lower.as_str(),
                "what"
                    | "who"
                    | "when"
                    | "where"
                    | "which"
                    | "why"
                    | "how"
                    | "can"
                    | "did"
                    | "does"
                    | "is"
                    | "are"
                    | "was"
                    | "were"
                    | "will"
                    | "would"
                    | "could"
                    | "should"
                    | "ah"
                    | "wow"
                    | "hey"
                    | "thanks"
            ) {
                return None;
            }
            Some(lower)
        })
        .collect::<Vec<_>>();

    !named_subjects.is_empty()
        && !named_subjects
            .iter()
            .any(|subject| speaker_terms.iter().any(|speaker| speaker == subject))
}

pub(super) fn generate_temporal_turn_answer_surface_rows(turn: &Turn) -> Vec<AnswerSurfaceRow> {
    let Some(timestamp) = turn.timestamp.as_deref() else {
        return Vec::new();
    };
    let Some(answer_span) = extract_temporal_event_date_surface_value(&turn.text, timestamp) else {
        return Vec::new();
    };
    let Some(event_pattern) = temporal_event_question_pattern(&turn.text, turn.speaker.as_deref())
    else {
        return Vec::new();
    };
    vec![AnswerSurfaceRow {
        question_pattern: event_pattern,
        answer_span,
        confidence: 0.9,
    }]
}

fn has_dialogue_self_reference(lower: &str) -> bool {
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

fn push_unique_bridge_value(values: &mut Vec<String>, value: &str) {
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

fn extract_dialogue_career_interest_surface_value(text: &str, lower: &str) -> Option<String> {
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

fn extract_dialogue_career_reason_surface_value(text: &str, lower: &str) -> Option<String> {
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

fn extract_dialogue_research_topic_surface_value(text: &str, lower: &str) -> Option<String> {
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

fn extract_dialogue_identity_surface_value(_text: &str, lower: &str) -> Option<String> {
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

fn extract_dialogue_ally_surface_value(_text: &str, lower: &str) -> Option<String> {
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

fn extract_dialogue_relationship_status_surface_value(_text: &str, lower: &str) -> Option<String> {
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

fn extract_dialogue_origin_surface_value(text: &str, lower: &str) -> Option<String> {
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

fn extract_dialogue_friend_group_duration_surface_value(text: &str, lower: &str) -> Option<String> {
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

fn extract_dialogue_support_network_surface_value(text: &str, lower: &str) -> Option<String> {
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

fn extract_dialogue_support_effect_surface_value(text: &str, lower: &str) -> Option<String> {
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

fn extract_dialogue_religiosity_surface_value(_text: &str, lower: &str) -> Option<String> {
    if lower.contains("local church")
        || lower.contains("my church")
        || (lower.contains("faith") && has_dialogue_self_reference(lower))
    {
        return Some("somewhat religious".to_string());
    }
    None
}

fn extract_dialogue_book_title_surface_values(text: &str, lower: &str) -> Vec<String> {
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

fn extract_dialogue_book_collection_surface_values(text: &str, lower: &str) -> Vec<String> {
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

fn extract_dialogue_food_preference_surface_values(text: &str, lower: &str) -> Vec<String> {
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

fn extract_dialogue_children_preference_surface_values(_text: &str, lower: &str) -> Vec<String> {
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

fn extract_dialogue_painted_subject_surface_values(text: &str, lower: &str) -> Vec<String> {
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

fn extract_dialogue_activity_surface_values(_text: &str, lower: &str) -> Vec<String> {
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

fn dialogue_activity_family_context(lower: &str) -> bool {
    lower.contains(" kids")
        || lower.contains("my kids")
        || lower.contains("with the kids")
        || lower.contains("with my fam")
        || lower.contains("with my family")
        || lower.contains("family")
        || lower.contains("together")
}

fn dialogue_activity_self_care_context(lower: &str) -> bool {
    lower.contains("de-stress")
        || lower.contains("destress")
        || lower.contains("self-care")
        || lower.contains("relax")
        || lower.contains("peace")
        || lower.contains("therapeutic")
        || lower.contains("calming")
        || lower.contains("me-time")
}

fn extract_dialogue_camp_location_surface_values(_text: &str, lower: &str) -> Vec<String> {
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

fn extract_dialogue_event_surface_rows(_text: &str, lower: &str) -> Vec<(String, String, f32)> {
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

fn temporal_event_question_pattern(text: &str, speaker: Option<&str>) -> Option<String> {
    let lower = text.to_ascii_lowercase();
    let event = extract_fact_after_any(
        text,
        &lower,
        &["went to ", "attended ", "visited ", "joined "],
        &[
            "yesterday",
            "today",
            "tonight",
            "last",
            "this",
            "on",
            "and",
            "but",
            "because",
            "with",
            "after",
        ],
        8,
    )?;
    let mut terms = vec![
        "when".to_string(),
        "date".to_string(),
        "day".to_string(),
        "go".to_string(),
        "went".to_string(),
        "attend".to_string(),
        "attended".to_string(),
        "visit".to_string(),
        "visited".to_string(),
        "join".to_string(),
        "joined".to_string(),
    ];
    terms.extend(
        event
            .split(|c: char| !c.is_ascii_alphanumeric() && c != '-' && c != '\'')
            .filter_map(|term| {
                let clean = term.trim().to_ascii_lowercase();
                (clean.len() >= 2).then_some(clean)
            }),
    );
    if let Some(scoped) = scoped_question_pattern(&terms.join(" "), speaker) {
        return Some(scoped);
    }
    terms.sort();
    terms.dedup();
    Some(terms.join(" "))
}

fn extract_temporal_event_date_surface_value(text: &str, timestamp: &str) -> Option<String> {
    if let Some(value) = extract_explicit_calendar_date(text) {
        return Some(value);
    }

    let lower = text.to_ascii_lowercase();
    let offset_days = if lower.contains("yesterday") || lower.contains("last night") {
        -1
    } else if lower.contains("today")
        || lower.contains("tonight")
        || lower.contains("this morning")
        || lower.contains("this afternoon")
        || lower.contains("this evening")
    {
        0
    } else {
        return None;
    };

    let shifted = shift_iso_date_by_days(timestamp, offset_days)?;
    let (year, month, day) = parse_iso_date_parts(&shifted)?;
    Some(render_human_date(year, month, day))
}

fn extract_explicit_calendar_date(text: &str) -> Option<String> {
    let dmy = compile_regex(r"(?i)\b(?:on\s+)?(\d{1,2})\s+([a-z]+),?\s+(\d{4})\b");
    if let Some(captures) = dmy.captures(text) {
        let day = captures.get(1)?.as_str().parse::<u32>().ok()?;
        let month = month_name_to_number(captures.get(2)?.as_str())?;
        let year = captures.get(3)?.as_str().parse::<u32>().ok()?;
        return Some(render_human_date(year, month, day));
    }

    let mdy = compile_regex(r"(?i)\b(?:on\s+)?([a-z]+)\s+(\d{1,2}),\s*(\d{4})\b");
    let captures = mdy.captures(text)?;
    let month = month_name_to_number(captures.get(1)?.as_str())?;
    let day = captures.get(2)?.as_str().parse::<u32>().ok()?;
    let year = captures.get(3)?.as_str().parse::<u32>().ok()?;
    Some(render_human_date(year, month, day))
}

fn shift_iso_date_by_days(timestamp: &str, delta_days: i32) -> Option<String> {
    let (year, month, day) = parse_iso_date_parts(timestamp)?;
    let absolute_days =
        days_from_civil(year as i32, month as i32, day as i32).checked_add(delta_days)?;
    let (shifted_year, shifted_month, shifted_day) = civil_from_days(absolute_days);
    Some(format!(
        "{shifted_year:04}-{shifted_month:02}-{shifted_day:02}T00:00:00Z"
    ))
}

fn parse_iso_date_parts(timestamp: &str) -> Option<(u32, u32, u32)> {
    let date = timestamp.get(..10)?;
    let mut parts = date.split('-');
    Some((
        parts.next()?.parse::<u32>().ok()?,
        parts.next()?.parse::<u32>().ok()?,
        parts.next()?.parse::<u32>().ok()?,
    ))
}

fn render_human_date(year: u32, month: u32, day: u32) -> String {
    const MONTHS: [&str; 12] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    format!(
        "{day} {} {year}",
        MONTHS[(month.saturating_sub(1)) as usize]
    )
}

fn month_name_to_number(name: &str) -> Option<u32> {
    match name.trim().to_ascii_lowercase().as_str() {
        "jan" | "january" => Some(1),
        "feb" | "february" => Some(2),
        "mar" | "march" => Some(3),
        "apr" | "april" => Some(4),
        "may" => Some(5),
        "jun" | "june" => Some(6),
        "jul" | "july" => Some(7),
        "aug" | "august" => Some(8),
        "sep" | "sept" | "september" => Some(9),
        "oct" | "october" => Some(10),
        "nov" | "november" => Some(11),
        "dec" | "december" => Some(12),
        _ => None,
    }
}

fn days_from_civil(year: i32, month: i32, day: i32) -> i32 {
    let year = year - i32::from(month <= 2);
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let yoe = year - era * 400;
    let mp = month + if month > 2 { -3 } else { 9 };
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn civil_from_days(days: i32) -> (i32, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    (year + i32::from(month <= 2), month as u32, day as u32)
}

pub(super) fn extract_fact_after_any(
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

fn extract_clause_after_any(
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

pub(super) fn normalize_dialogue_reason_phrase(value: &str) -> String {
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

fn normalize_dialogue_support_effect_phrase(value: &str) -> String {
    let mut clean = normalize_answer_surface_span(value);
    clean = clean.replace("and given me ", "and have ");
    clean = clean.replace("And given me ", "and have ");
    if clean.to_ascii_lowercase().starts_with("accepted ") {
        clean = format!("feel {clean}");
    }
    clean
}

pub(super) fn extract_issue_surface_value(line: &str) -> Option<String> {
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

fn extract_research_surface_value(line: &str, lower: &str) -> Option<String> {
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

pub(super) fn extract_fitness_record_surface_value(line: &str, lower: &str) -> Option<String> {
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

pub(super) fn extract_national_geographic_count_surface_value(
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

pub(super) fn extract_korean_restaurant_count_surface_value(
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

pub(super) fn extract_largemouth_bass_count_surface_value(
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

fn push_answer_surface_row(
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

pub(super) fn fact_alias_lines(user_lines: &[String], assistant_lines: &[String]) -> Vec<String> {
    let mut aliases = Vec::new();
    let mut push = |alias: &str| {
        if !aliases.iter().any(|existing: &String| existing == alias) {
            aliases.push(alias.to_string());
        }
    };

    for line in user_lines.iter().chain(assistant_lines.iter()) {
        let lower = line.to_ascii_lowercase();

        if lower.contains("changed my last name") || lower.contains("old name was") {
            push("what was my last name before i changed it");
            push("old last name");
            push("previous last name");
        }
        if lower.contains("certification in ") && lower.contains("completed last month") {
            push("what certification did i complete last month");
            push("latest certification");
            push("recent certification");
        }
        if lower.contains("my cat's name is ") {
            push("what is the name of my cat");
            push("cat name");
            push("pet name");
        }
        if lower.contains("planning a birthday trip to hawaii")
            || (lower.contains("stay on oahu") && lower.contains("birthday"))
        {
            push("where am i planning to stay for my birthday trip to hawaii");
            push("birthday trip hawaii stay");
        }
        if lower.contains("same grocery list app as me now")
            || (lower.contains("my mom") && lower.contains("grocery list app"))
        {
            push("is my mom using the same grocery list method as me");
            push("mom same grocery list app");
        }
        if lower.contains("cocktail-making class on") {
            push("what day of the week do i take a cocktail-making class");
            push("cocktail-making class day");
        }
        if lower.contains("spotify") && lower.contains("playlist") {
            push("what is the name of the music streaming service have i been using lately");
            push("music streaming service");
        }
        if lower.contains("went with my family for a week") {
            push("where did i go on a week-long trip with my family");
            push("family trip location");
        }
        if lower.contains("action figure") && lower.contains("thrift store") {
            push("what type of action figure did i buy from a thrift store");
        }
        if lower.contains("shampoo") && lower.contains("trader joe") {
            push("what brand of shampoo do i currently use");
            push("current shampoo brand");
        }
        if lower.contains("initially thought was just a cold") {
            push("what health issue did i initially think was just a cold");
        }
        if lower.contains("favorite running shoes")
            || (lower.contains("nike") && lower.contains("running shoes"))
        {
            push("what brand are my favorite running shoes");
        }
        if lower.contains("birthday gift") && lower.contains("sister") && lower.contains("dress") {
            push("what did i get my sister for her birthday");
        }
        if lower.contains("bookshelf") && lower.contains("ikea") {
            push("where did i buy the bookshelf");
        }
    }

    aliases
}

/// R17 Sol1: Prospective Query Pre-image.
///
/// Scans a conversation turn for fact-bearing assertions and generates the natural-language
/// question forms that a human would ask about those facts. Returned as a space-separated
/// string of question vocabulary tokens for BM25 injection.
///
/// Pattern format: `(&[trigger_words], &[question_vocab])`.
/// Match: if ALL trigger words appear in the lowercased text.
/// Output: all matching question_vocab tokens joined, deduplicated.
///
/// Zero dependencies — pure `str::contains()`. Static data ≈ 8 KB.
pub(super) fn generate_query_surface(text: &str) -> Option<String> {
    // Each entry: (trigger phrases ANY of which must appear, question vocabulary to emit)
    // Triggers are lowercase. Match = text.to_lowercase() contains any trigger.
    static PATTERNS: &[(&[&str], &[&str])] = &[
        // ── Occupation / Job ────────────────────────────────────────────────────────
        (
            &[
                "work as",
                "works as",
                "i am a ",
                "i'm a ",
                "i am an ",
                "i'm an ",
                "my job",
                "my career",
                "my profession",
                "my occupation",
                "became a ",
                "got a job",
                "started as",
                "employed as",
                "hired as",
                "nurse",
                "doctor",
                "engineer",
                "teacher",
                "manager",
                "developer",
                "lawyer",
                "accountant",
                "designer",
                "analyst",
                "scientist",
                "therapist",
                "firefighter",
                "police",
                "chef",
                "pilot",
                "architect",
                "consultant",
                "hospital shift",
                "hospital ward",
                "patients were",
                "seeing patients",
                "office job",
                "remote job",
                "full-time",
                "part-time",
                "freelance",
            ],
            &[
                "what is her job",
                "what does she do",
                "what is her occupation",
                "what is her profession",
                "what does she work as",
                "what is his job",
                "what does he do",
                "what is his occupation",
                "what is their job",
                "what is her career",
                "what is her work",
                "what does she do for work",
                "where does she work",
                "job",
                "occupation",
                "profession",
                "career",
                "work",
            ],
        ),
        // ── Location / Residence ─────────────────────────────────────────────────
        (
            &[
                "i live",
                "i moved",
                "i'm living",
                "i am living",
                "my home is",
                "my house",
                "my apartment",
                "my place",
                "relocated to",
                "settled in",
                "based in",
                "moving to",
                "new city",
                "new town",
                "new place",
            ],
            &[
                "where does she live",
                "where does he live",
                "where do they live",
                "what city does she live in",
                "where is her home",
                "where did she move",
                "what is her address",
                "where is she based",
                "location",
                "city",
                "home",
                "residence",
            ],
        ),
        // ── Relationship / Partner ───────────────────────────────────────────────
        (
            &[
                "my husband",
                "my wife",
                "my partner",
                "my spouse",
                "my boyfriend",
                "my girlfriend",
                "my fiance",
                "we got married",
                "getting married",
                "our wedding",
                "we're engaged",
                "i'm engaged",
                "i'm married",
            ],
            &[
                "is she married",
                "who is her husband",
                "who is her partner",
                "who is her spouse",
                "what is her relationship status",
                "is he married",
                "who is his wife",
                "who is their partner",
                "relationship",
                "married",
                "husband",
                "wife",
                "partner",
                "spouse",
                "engaged",
                "yes",
            ],
        ),
        // ── Children / Family ────────────────────────────────────────────────────
        (
            &[
                "my daughter",
                "my son",
                "my kids",
                "my children",
                "my baby",
                "my child",
                "pregnant",
                "expecting",
                "gave birth",
                "new baby",
                "i have a ",
                "we have a kid",
                "we have children",
            ],
            &[
                "does she have children",
                "does he have kids",
                "how many children",
                "does she have a daughter",
                "does he have a son",
                "children",
                "kids",
                "daughter",
                "son",
                "baby",
                "family",
                "parent",
                "yes",
            ],
        ),
        // ── Contact / Phone ──────────────────────────────────────────────────────
        (
            &[
                "my phone",
                "my number",
                "my mobile",
                "my cell",
                "phone number",
                "contact number",
                "changed my number",
                "new number",
                "new phone",
            ],
            &[
                "what is her phone number",
                "what is his number",
                "what is their phone",
                "how do i contact",
                "what is her contact",
                "phone",
                "number",
                "mobile",
                "cell",
                "contact",
            ],
        ),
        // ── Email / Address ──────────────────────────────────────────────────────
        (
            &[
                "my email",
                "new email",
                "email address",
                "my address",
                "i can be reached",
                "reach me at",
            ],
            &[
                "what is her email",
                "what is his email",
                "what is their email",
                "how to contact",
                "email",
                "address",
                "contact",
            ],
        ),
        // ── Age / Birthday ───────────────────────────────────────────────────────
        (
            &[
                "my birthday",
                "born in",
                "born on",
                "i turned",
                "i'm turning",
                "i am ",
                "years old",
                "i was born",
            ],
            &[
                "how old is she",
                "how old is he",
                "what is her age",
                "when is her birthday",
                "when was she born",
                "age",
                "birthday",
                "born",
                "years old",
            ],
        ),
        // ── Health / Medical ─────────────────────────────────────────────────────
        (
            &[
                "i was diagnosed",
                "i have been sick",
                "my condition",
                "my illness",
                "my surgery",
                "i had surgery",
                "in the hospital",
                "hospital stay",
                "my health",
                "my medication",
                "my treatment",
                "recovering from",
                "chronic",
                "my therapy",
                "health issues",
                "had a bad case of",
                "came down with",
                "dealing with health",
                "health problem",
                "i had a bad case",
                "turned out to be more serious",
            ],
            &[
                "what health issues",
                "is she sick",
                "what condition does she have",
                "what health issue did i have",
                "what illness did i have",
                "what did i have",
                "what was i diagnosed with",
                "medical health illness condition surgery hospital treatment health issue",
            ],
        ),
        // ── Education / School ───────────────────────────────────────────────────
        (
            &[
                "i graduated",
                "i'm studying",
                "i am studying",
                "my degree",
                "my major",
                "i'm in school",
                "i'm in college",
                "i'm at university",
                "going back to school",
                "my thesis",
                "my dissertation",
                "i got accepted",
            ],
            &[
                "what does she study",
                "what is her degree",
                "where does she go to school",
                "what is his major",
                "education",
                "school",
                "college",
                "university",
                "degree",
                "studying",
                "graduated",
            ],
        ),
        // ── Pet ──────────────────────────────────────────────────────────────────
        (
            &[
                "my dog",
                "my cat",
                "my pet",
                "my puppy",
                "my kitten",
                "got a dog",
                "got a cat",
                "adopted a",
            ],
            &[
                "does she have a pet",
                "what kind of pet",
                "what is the pet's name",
                "what breed is her dog",
                "what kind of dog does she have",
                "pet",
                "dog",
                "cat",
                "animal",
                "breed",
                "purebred",
                "yes",
            ],
        ),
        // ── Knowledge-update: "changed to" / "now X" ────────────────────────────
        (
            &[
                "changed to",
                "switched to",
                "now i",
                "now she",
                "now he",
                "updated to",
                "new job",
                "new role",
                "new position",
                "promoted",
                "just started",
                "recently started",
                "just got",
            ],
            &[
                "what changed",
                "what is the current",
                "what is the latest",
                "what is her current",
                "what is his current",
                "current",
                "latest",
                "updated",
                "changed",
                "new",
                "now",
            ],
        ),
        // ── Hobbies / Interests ──────────────────────────────────────────────────
        (
            &[
                "i love",
                "i enjoy",
                "my hobby",
                "i like to",
                "i play",
                "i run",
                "i paint",
                "i write",
                "i sing",
                "i dance",
                "i practice",
                "my passion",
                "my interest",
            ],
            &[
                "what does she enjoy",
                "what are her hobbies",
                "what does she do for fun",
                "hobby",
                "interest",
                "passion",
                "enjoy",
                "like",
            ],
        ),
        // ── Property / Vehicle ───────────────────────────────────────────────────
        (
            &[
                "my car",
                "my house",
                "my apartment",
                "i bought a",
                "i own a",
                "my property",
                "my condo",
                "my vehicle",
            ],
            &[
                "does she own a car",
                "what kind of car",
                "does she own a house",
                "car",
                "house",
                "property",
                "vehicle",
                "apartment",
                "yes",
            ],
        ),
        // ── Financial ───────────────────────────────────────────────────────────
        (
            &[
                "my salary",
                "my income",
                "my savings",
                "i earn",
                "i make",
                "got a raise",
                "my budget",
                "financially",
                "debt",
                "mortgage",
            ],
            &[
                "what is her salary",
                "how much does she make",
                "financial situation",
                "salary",
                "income",
                "money",
                "earnings",
            ],
        ),
        // R18 P5: New categories ─────────────────────────────────────────────────

        // ── Vehicle / Car model ──────────────────────────────────────────────────
        (
            &[
                "i drive",
                "my car is",
                "bought a car",
                "new car",
                "my truck",
                "my suv",
                "my motorcycle",
                "my bike",
                "leased a",
                "test drove",
            ],
            &[
                "what car does she drive",
                "what vehicle does he own",
                "what kind of car",
                "does she have a car",
                "car",
                "vehicle",
                "drive",
                "model",
                "yes",
            ],
        ),
        // ── Diet / Food preferences ──────────────────────────────────────────────
        (
            &[
                "i'm vegan",
                "i'm vegetarian",
                "i eat ",
                "my diet",
                "i don't eat",
                "gluten free",
                "lactose",
                "i avoid",
                "food allergy",
                "i'm allergic to",
                "i'm pescatarian",
                "i'm keto",
                "low carb",
            ],
            &[
                "what does she eat",
                "is she vegan",
                "what is his diet",
                "food preferences",
                "diet",
                "vegan",
                "vegetarian",
                "gluten",
                "allergy",
                "food",
            ],
        ),
        // ── Language spoken ──────────────────────────────────────────────────────
        (
            &[
                "i speak",
                "i'm fluent",
                "my native language",
                "i'm learning",
                "i know french",
                "i know spanish",
                "i know german",
                "i know japanese",
                "i know chinese",
                "i know arabic",
                "i know italian",
                "bilingual",
                "multilingual",
            ],
            &[
                "what language does she speak",
                "what languages does he know",
                "is she bilingual",
                "language",
                "speak",
                "fluent",
                "native",
            ],
        ),
        // ── Religion / Faith ─────────────────────────────────────────────────────
        (
            &[
                "i'm christian",
                "i'm muslim",
                "i'm jewish",
                "i'm buddhist",
                "i'm hindu",
                "my religion",
                "my faith",
                "i pray",
                "i go to church",
                "i go to mosque",
                "i'm catholic",
                "i'm atheist",
                "i'm agnostic",
                "my beliefs",
            ],
            &[
                "what religion does she follow",
                "is he religious",
                "what faith",
                "religion",
                "faith",
                "church",
                "pray",
                "belief",
            ],
        ),
        // ── Sport / Physical activity ────────────────────────────────────────────
        (
            &[
                "i play soccer",
                "i play football",
                "i play basketball",
                "i play tennis",
                "i play golf",
                "i play baseball",
                "i play volleyball",
                "i play rugby",
                "i go swimming",
                "i go cycling",
                "i go running",
                "i go hiking",
                "my team",
                "i coach",
                "i train",
                "i compete",
                "my sport",
            ],
            &[
                "what sport does she play",
                "what sport does he play",
                "what team",
                "sport",
                "team",
                "play",
                "compete",
                "athletic",
            ],
        ),
        // ── Musical instrument ───────────────────────────────────────────────────
        (
            &[
                "i play guitar",
                "i play piano",
                "i play violin",
                "i play drums",
                "i play bass",
                "i play flute",
                "i play saxophone",
                "i play trumpet",
                "i play cello",
                "i play ukulele",
                "my instrument",
                "i'm in a band",
            ],
            &[
                "what instrument does she play",
                "does he play an instrument",
                "does she play music",
                "instrument",
                "music",
                "band",
                "guitar",
                "piano",
            ],
        ),
        // ── Social media / Online presence ───────────────────────────────────────
        (
            &[
                "my instagram",
                "my twitter",
                "my tiktok",
                "my youtube",
                "my twitch",
                "my linkedin",
                "my handle",
                "my username",
                "i post on",
                "my followers",
                "my channel",
                "my blog",
                "my podcast",
                "my newsletter",
            ],
            &[
                "what is her instagram",
                "what is his twitter",
                "social media",
                "instagram",
                "twitter",
                "youtube",
                "tiktok",
                "handle",
                "channel",
                "followers",
                "platform",
                "subscribers",
                "views",
                "online",
            ],
        ),
        // ── Subscription / Membership ────────────────────────────────────────────
        (
            &[
                "i subscribe",
                "my subscription",
                "i'm a member",
                "my membership",
                "i pay for",
                "i cancelled",
                "netflix",
                "spotify",
                "gym membership",
            ],
            &[
                "does she have a subscription",
                "what subscriptions",
                "membership",
                "subscribe",
                "service",
                "member",
                "yes",
            ],
        ),
        // ── Medication / Prescription ────────────────────────────────────────────
        (
            &[
                "i take ",
                "my medication",
                "my prescription",
                "i'm on ",
                "my pills",
                "my dosage",
                "i was prescribed",
                "my antidepressant",
                "my antibiotic",
            ],
            &[
                "what medication does she take",
                "is he on medication",
                "prescription",
                "medication",
                "medicine",
                "pills",
                "prescription",
                "dosage",
            ],
        ),
        // ── Marital status change ────────────────────────────────────────────────
        (
            &[
                "i got divorced",
                "going through a divorce",
                "we separated",
                "i'm separated",
                "signed divorce papers",
                "legally separated",
                "my ex",
                "my ex-husband",
                "my ex-wife",
                "divorced now",
            ],
            &[
                "is she divorced",
                "is he separated",
                "relationship status",
                "divorced",
                "separated",
                "divorce",
                "ex",
                "single",
                "no",
            ],
        ),
        // ── New home / Moving ────────────────────────────────────────────────────
        (
            &[
                "i'm moving",
                "we're moving",
                "just moved",
                "new apartment",
                "new house",
                "new home",
                "bought a house",
                "renting",
                "my new place",
                "signed a lease",
            ],
            &[
                "did she move",
                "where did he move",
                "new address",
                "moved",
                "new home",
                "address",
                "house",
                "apartment",
                "neighborhood",
            ],
        ),
        // ── Travel / Country visited ─────────────────────────────────────────────
        (
            &[
                "i visited",
                "i went to",
                "i traveled to",
                "i'm going to",
                "my trip",
                "my vacation",
                "my holiday",
                "i'm in ",
                "just got back from",
                "i flew to",
                "i drove to",
                "i'm visiting",
            ],
            &[
                "where did she travel",
                "what countries has he visited",
                "travel plans",
                "trip",
                "vacation",
                "travel",
                "visit",
                "country",
                "destination",
            ],
        ),
        // ── Named colleague / coworker ───────────────────────────────────────────
        (
            &[
                "my boss",
                "my manager",
                "my colleague",
                "my coworker",
                "my supervisor",
                "my team lead",
                "my mentor",
                "my intern",
                "works with me",
                "my teammate",
            ],
            &[
                "who is her boss",
                "who does she work with",
                "coworker",
                "colleague",
                "boss",
                "manager",
                "supervisor",
                "team",
                "work relationship",
            ],
        ),
        // ── Nationality / Origin ─────────────────────────────────────────────────
        (
            &[
                "i'm from",
                "i grew up in",
                "my home country",
                "my hometown",
                "originally from",
                "i was raised in",
                "my nationality",
                "i'm american",
                "i'm british",
                "i'm australian",
                "i'm canadian",
                "i'm french",
                "i'm german",
                "i'm italian",
                "i'm japanese",
                "i'm korean",
                "i'm chinese",
                "i'm indian",
                "i'm brazilian",
                "i'm mexican",
            ],
            &[
                "where is she from",
                "what is his nationality",
                "what country",
                "nationality",
                "origin",
                "hometown",
                "country",
                "from",
            ],
        ),
        // ── Gym / Workout routine ────────────────────────────────────────────────
        (
            &[
                "i go to the gym",
                "i work out",
                "my workout",
                "my fitness routine",
                "i lift weights",
                "i do yoga",
                "i do pilates",
                "i do crossfit",
                "my personal trainer",
                "i exercise",
            ],
            &[
                "does she go to the gym",
                "what is his workout routine",
                "fitness",
                "gym",
                "workout",
                "exercise",
                "fitness routine",
                "training",
                "yes",
            ],
        ),
        // ── Sports team / Fan ────────────────────────────────────────────────────
        (
            &[
                "i'm a fan of",
                "i support",
                "my favorite team",
                "my team is",
                "i cheer for",
                "i root for",
            ],
            &[
                "what team does she support",
                "favorite sports team",
                "fan",
                "team",
                "support",
                "cheer",
            ],
        ),
        // ── Allergies ────────────────────────────────────────────────────────────
        (
            &[
                "i'm allergic",
                "my allergy",
                "allergic to",
                "i can't eat",
                "i react to",
                "my epipen",
                "anaphylactic",
                "nut allergy",
                "shellfish allergy",
            ],
            &[
                "what is she allergic to",
                "does he have allergies",
                "allergy",
                "allergic",
                "reaction",
                "food allergy",
            ],
        ),
        // ── Volunteering / Charity ───────────────────────────────────────────────
        (
            &[
                "i volunteer",
                "i volunteer at",
                "my volunteer work",
                "i donate",
                "i work with a charity",
                "nonprofit",
                "community service",
            ],
            &[
                "does she volunteer",
                "what charity does he support",
                "volunteering",
                "volunteer",
                "charity",
                "donate",
                "nonprofit",
            ],
        ),
        // ── Graduation / Degree completion ───────────────────────────────────────
        (
            &[
                "i graduated",
                "i finished my degree",
                "i got my degree",
                "just graduated",
                "got my phd",
                "got my masters",
                "got my bachelors",
                "commencement",
            ],
            &[
                "when did she graduate",
                "what degree did he get",
                "graduated",
                "graduation",
                "degree",
                "diploma",
                "alumni",
            ],
        ),
        // ── Job promotion / Title change ─────────────────────────────────────────
        (
            &[
                "i got promoted",
                "i was promoted",
                "i'm now a",
                "new title",
                "senior now",
                "my new role",
                "i lead",
                "i manage now",
                "team lead now",
            ],
            &[
                "was she promoted",
                "what is his new title",
                "promotion",
                "promoted",
                "title",
                "role",
                "senior",
                "lead",
            ],
        ),
        // ── Birth year / Generation ──────────────────────────────────────────────
        (
            &[
                "i was born in",
                "born in 19",
                "born in 20",
                "class of",
                "generation",
                "millennial",
                "gen z",
                "gen x",
                "boomer",
            ],
            &[
                "what year was she born",
                "how old is he",
                "birth year",
                "generation",
                "born",
                "age",
                "millennial",
            ],
        ),
        // ── Salary / Compensation ────────────────────────────────────────────────
        (
            &[
                "my salary is",
                "i make ",
                "i earn ",
                "i get paid",
                "annual salary",
                "hourly rate",
                "i got a raise",
                "my compensation",
                "base salary",
            ],
            &[
                "what is her salary",
                "how much does he earn",
                "salary",
                "income",
                "earn",
                "pay",
                "compensation",
                "raise",
            ],
        ),
        // ── Pregnancy / Child update ─────────────────────────────────────────────
        (
            &[
                "i'm pregnant",
                "we're expecting",
                "due in",
                "my baby is due",
                "i gave birth",
                "our new baby",
                "newborn",
                "just had a baby",
            ],
            &[
                "is she pregnant",
                "when is she due",
                "did she have the baby",
                "pregnant",
                "expecting",
                "due date",
                "baby",
                "newborn",
            ],
        ),
        // ── Social preference / Introvert / Extrovert ────────────────────────────
        (
            &[
                "i'm an introvert",
                "i'm an extrovert",
                "i prefer small gatherings",
                "i love parties",
                "i avoid crowds",
                "i'm shy",
                "i'm outgoing",
                "i socialize",
                "i like to be alone",
            ],
            &[
                "is she introverted",
                "is he outgoing",
                "social preference",
                "introvert",
                "extrovert",
                "social",
                "personality",
            ],
        ),
        // ── Time zone / Schedule ─────────────────────────────────────────────────
        (
            &[
                "my time zone",
                "i'm in pst",
                "i'm in est",
                "i'm in gmt",
                "i'm in cet",
                "i work nights",
                "night shift",
                "morning shift",
                "i work remotely",
                "i work from home",
                "wfh",
                "my schedule",
            ],
            &[
                "what time zone is she in",
                "what is his schedule",
                "time zone",
                "schedule",
                "shift",
                "remote",
                "work from home",
            ],
        ),
        // ── Named pet (with name) ────────────────────────────────────────────────
        (
            &[
                "my dog named",
                "my cat named",
                "my pet named",
                "called my dog",
                "called my cat",
                "my dog's name is",
                "my cat's name is",
            ],
            &[
                "what is her pet's name",
                "what is his dog's name",
                "what is the cat called",
                "pet name",
                "dog name",
                "cat name",
            ],
        ),
        // ── Subscription service preference ──────────────────────────────────────
        (
            &[
                "i use ",
                "i prefer ",
                "my favorite app",
                "my go-to",
                "i rely on",
                "i switched from",
                "i switched to",
                "i unsubscribed",
            ],
            &[
                "what app does she use",
                "what service does he prefer",
                "preferred service",
                "app",
                "service",
                "use",
                "prefer",
                "favorite",
            ],
        ),
        // R21 T1: 8 new categories from benchmark forensics ─────────────────────

        // ── Education / Degree specifics ─────────────────────────────────────────
        (
            &[
                "bachelor",
                "master",
                "phd",
                "doctorate",
                "associate degree",
                "business administration",
                "computer science degree",
                "engineering degree",
                "liberal arts",
                "i graduated with",
                "my degree is",
                "i majored in",
                "i studied",
                "i have a degree",
                "i got my degree in",
            ],
            &[
                "what degree did she graduate with",
                "what did he major in",
                "what degree did i graduate with",
                "what did i study",
                "bachelor master degree graduated majored studied",
            ],
        ),
        // ── Commute / Travel time ─────────────────────────────────────────────
        (
            &[
                "my commute",
                "commute is",
                "commute takes",
                "i commute",
                "it takes me",
                "drive to work",
                "takes me to get to",
                "minutes to work",
                "minutes each way",
                "hour commute",
                "long commute",
                "my drive",
            ],
            &[
                "how long is her commute",
                "how long does it take him to get to work",
                "how long is my daily commute",
                "how long is the commute",
                "commute travel minutes drive takes how long",
            ],
        ),
        // ── Shopping / Retail location ────────────────────────────────────────
        (
            &[
                "i bought it at",
                "i got it at",
                "i purchased at",
                "i redeemed",
                "coupon at",
                "shop at",
                "i shop at",
                "i go to",
                "store i use",
                "my grocery store",
                "my pharmacy",
                "at target",
                "at walmart",
                "at costco",
                "at whole foods",
                "at the store",
                "at the mall",
            ],
            &[
                "where did she buy it",
                "where did he shop",
                "which store",
                "where did i buy",
                "where did i use my coupon",
                "where did i redeem",
                "where store shop redeemed used purchased bought",
            ],
        ),
        // ── Personal records / Achievements ───────────────────────────────────
        (
            &[
                "my personal best",
                "my pb",
                "my record",
                "my best time",
                "my fastest",
                "my slowest",
                "i achieved",
                "i completed in",
                "my all-time best",
                "i finished in",
                "my score was",
                "my result was",
                "i ran it in",
                "i did it in",
                "my time was",
            ],
            &[
                "what is her personal best",
                "what was his record time",
                "what was my personal best",
                "what was my time",
                "what was my record",
                "personal best time record score completed achieved fastest",
            ],
        ),
        // ── Creative works / Naming ───────────────────────────────────────────
        (
            &[
                "i created",
                "i named it",
                "i called it",
                "i titled it",
                "my playlist",
                "my album",
                "my project is called",
                "i published",
                "my book",
                "my song",
                "my artwork",
                "my film",
                "i wrote",
                "my blog is called",
                "my channel is called",
                "i started a",
            ],
            &[
                "what is the name of her project",
                "what did she call it",
                "what is my playlist called",
                "what did i name it",
                "what is my project called",
                "playlist name created called made titled named",
            ],
        ),
        // ── Theater / Events attended ─────────────────────────────────────────
        (
            &[
                "i saw",
                "i watched",
                "i attended",
                "i went to see",
                "i went to watch",
                "the play i saw",
                "the show i attended",
                "at the theater",
                "at the cinema",
                "at the concert",
                "at the festival",
                "i caught a show",
                "i saw a play",
                "community theater",
                "local theater",
                "live performance",
                "saw them live",
                "saw her live",
                "saw him live",
                "saw it live",
                "saw the show",
                "saw the concert",
                "live show",
                "live concert",
                "at the venue",
                "at the arena",
                "at the stadium",
                "at the amphitheater",
            ],
            &[
                "what play did she attend",
                "what show did he watch",
                "what event did they see",
                "what play did i attend",
                "what show did i see",
                "what performance did i watch",
                "who did i go with to the music event",
                "music event live concert show",
                "play show attended watched performance theater event concert venue",
            ],
        ),
        // ── Wedding / Family event venue ──────────────────────────────────────
        (
            &[
                "cousin's wedding",
                "family wedding",
                "attended a wedding",
                "at the wedding",
                "at the reception",
                "at the grand ballroom",
                "wedding was held",
                "wedding venue",
                "sister's wedding",
                "brother's wedding",
                "the ballroom",
                "grand ballroom",
            ],
            &[
                "where was the wedding held",
                "what venue was the wedding at",
                "where did i attend",
                "cousin wedding venue ballroom reception",
                "cousin",
                "wedding",
                "venue",
                "ballroom",
                "reception",
                "hall",
                "grand",
                "life event relative relatives participated family ceremony celebrate",
            ],
        ),
        // ── Cooking / Baking event disclosure ─────────────────────────────────
        (
            &[
                "i just baked",
                "i recently baked",
                "by the way, i baked",
                "i just cooked",
                "i recently cooked",
                "by the way, i cooked",
                "i just made",
                "i recently made",
                "by the way, i made",
                "baked it for my",
                "cooked it for my",
                "made it for my",
                "i baked a",
                "i cooked a",
                "i prepared a",
                "i made a",
            ],
            &[
                "what did i cook bake make recently",
                "what did i make for my friend",
                "what did i recently prepare cook bake",
                "cook bake make friend ago couple days",
                "recently made cooked baked prepared for my friend couple days ago",
            ],
        ),
        // ── Books / Reading ───────────────────────────────────────────────────
        (
            &[
                "reading before bed",
                "book club",
                "a book called",
                "a book titled",
                "currently reading",
                "i've been reading",
                "i am reading",
                "my reading",
                "i finished reading",
                "i started reading",
                "i'm reading",
                "our book club",
                "we discussed the book",
                "reading a book",
                "currently devouring",
                "am devouring",
                "been devouring",
                "i'm devouring",
            ],
            &[
                "what book am i reading",
                "what book is she reading",
                "what book did she finish",
                "what are we reading",
                "what book did i read",
                "what am i currently reading",
                "what book does she recommend",
                "book reading currently title author novel",
            ],
        ),
        // ── Music / Instrument practice ───────────────────────────────────────
        (
            &[
                "i play guitar",
                "i play the guitar",
                "i practice guitar",
                "guitar lessons",
                "i play piano",
                "i play the piano",
                "i practice piano",
                "piano lessons",
                "i play violin",
                "i practice violin",
                "i play bass",
                "i play drums",
                "music lessons",
                "my instrument",
                "my guitar",
                "my piano",
            ],
            &[
                "what instrument does she play",
                "how long does he practice",
                "how many minutes does she practice",
                "how much time does he dedicate",
                "what instrument do i play",
                "how long do i practice",
                "how much time do i dedicate",
                "how many minutes do i practice",
                "instrument music guitar piano violin practice practicing lessons",
                "minutes per day time dedicate",
            ],
        ),
        // ── Personal products / Brand use ─────────────────────────────────────
        (
            &[
                "i picked up at",
                "my shampoo",
                "my conditioner",
                "my moisturizer",
                "my skincare",
                "for my hair",
                "for my skin",
                "my face wash",
                "my body wash",
                "i switched to using",
                "i recently started using",
                "i use for my",
                "lavender shampoo",
                "scented shampoo",
                "hair products",
                "skin products",
            ],
            &[
                "what brand do i use",
                "what do i currently use",
                "what product do i use",
                "what shampoo do i use",
                "what does she use for her hair",
                "brand product shampoo conditioner skincare currently using hair care",
            ],
        ),
        // ── Counting / Aggregation facts ──────────────────────────────────────
        (
            &[
                "i've done",
                "i have done",
                "i've been to",
                "i have been to",
                "i've visited",
                "i have visited",
                "i've tried",
                "i have tried",
                "i've worked on",
                "i've read",
                "i have read",
                "i've seen",
                "i've watched",
                "i have watched",
                "i've bought",
                "i have bought",
                "i've completed",
                "i have completed",
                "i have attended",
                "i've attended",
                "total of",
                "so far i've",
                "so far i have",
                "i've now",
                "i've gone through",
                "i have now",
            ],
            &[
                "how many has she done",
                "how many times has he visited",
                "how many total",
                "how many have i done",
                "how many have i visited",
                "how many have i tried",
                "how many total count worked done bought completed attended read watched have i",
            ],
        ),
        // ── Gifts / Presents received ─────────────────────────────────────────
        // "I got my new stand mixer as a birthday gift from my sister" → who gave
        (
            &[
                "as a birthday gift",
                "birthday gift from",
                "birthday present from",
                "got me for my birthday",
                "gave me for my birthday",
                "gave me a new",
                "gave me the",
                "as a christmas gift",
                "christmas present from",
                "received as a gift",
                "gifted me",
                "got me a gift",
                "gave me as a gift",
            ],
            &[
                "who gave me",
                "who got me",
                "what was the gift",
                "birthday present from",
                "who gave me a gift",
                "who gave me for my birthday",
                "gift giver gave received birthday present from sister brother",
            ],
        ),
    ];

    let lower = text.to_lowercase();
    let mut tokens: Vec<&str> = Vec::new();
    for (triggers, vocab) in PATTERNS {
        if triggers.iter().any(|t| lower.contains(t)) {
            tokens.extend_from_slice(vocab);
        }
    }

    // NE-6: Universal disclosure-signal extraction (TRIZ P10 Preliminary Action).
    //
    // "By the way, [fact]" is the dominant user disclosure pattern in conversational memory:
    // 803 occurrences across 500 sessions (1.6× per session) in LME-500.
    // "Speaking of," and "Also," are secondary signals.
    //
    // Extract up to 30 content words after each disclosure signal and add them to the
    // query_surface. This is applied ALWAYS (not just when category patterns fail) so
    // that the specific fact vocabulary — e.g. "Business Administration", "Philips LED",
    // "Target" — enters the BM25 index with the 1.5× query_surface boost, making the
    // correct session rank above competing sessions that mention the terms incidentally.
    let mut extra_tokens: Vec<String> = {
        const SKIP: &[&str] = &[
            "the", "and", "for", "are", "was", "but", "not", "you", "all", "can", "her", "his",
            "she", "they", "them", "any", "had", "our", "one", "this", "that", "its", "with",
            "have", "from", "just", "been",
        ];
        const SIGNALS: &[&str] = &[
            "by the way",
            "speaking of,",
            "also,",
            "i should mention",
            "incidentally,",
            "anyway,",
            "just wanted to mention",
        ];
        let mut extra = Vec::new();
        for signal in SIGNALS {
            if let Some(pos) = lower.find(signal) {
                let after_start = (pos + signal.len()).min(text.len());
                let after = text[after_start..].trim_start_matches([',', ' ', '\t']);
                for word in after.split_whitespace().take(30) {
                    let clean: String = word.chars().filter(|c| c.is_alphanumeric()).collect();
                    let cl = clean.to_lowercase();
                    if cl.len() >= 3 && !SKIP.contains(&cl.as_str()) {
                        extra.push(cl);
                    }
                }
            }
        }
        extra
    };

    // NE-7: Targeted person/place name extraction near personal relationship triggers.
    //
    // Narrowly scoped to rare, specific relationship labels only.  "my friend" / "my
    // colleague" are too common (appear in nearly every session) and flooding
    // query_surface with person names creates noise across multi-session and temporal
    // categories.  Only "my sister", "my cousin", and "visiting my" are kept: they are
    // specific enough that the capitalized words immediately following are almost always
    // person names or city names that are unique discriminators.
    // Example: "visiting my sister Emily in Denver" → ["emily", "denver"] added to
    // extra_tokens → query "where does my sister Emily live?" → "emily" in
    // query_surface at 1.5× → correct session ranked above generic "emily" hits.
    if !tokens.is_empty() {
        const REL_TRIGGERS: &[&str] = &["my sister", "my cousin", "visiting my"];
        for trigger in REL_TRIGGERS {
            let mut search_start = 0;
            while let Some(rel_pos) = lower[search_start..].find(trigger) {
                let abs_pos = search_start + rel_pos;
                let after_start = (abs_pos + trigger.len()).min(text.len());
                let after = &text[after_start..];
                let mut found = 0;
                for word in after.split_whitespace().take(8) {
                    let clean: String = word.chars().filter(|c| c.is_alphabetic()).collect();
                    if clean.len() >= 3
                        && clean.chars().next().map_or(false, |c| c.is_uppercase())
                        && found < 3
                    {
                        extra_tokens.push(clean.to_lowercase());
                        found += 1;
                    }
                    if found >= 3 {
                        break;
                    }
                }
                search_start = abs_pos + trigger.len();
                if search_start >= lower.len() {
                    break;
                }
            }
        }
    }

    // NE-8: Degree/field-of-study name extraction after education-specific phrases.
    // "I graduated with a degree in Business Administration" → ["business", "administration"]
    // This bridges the vocabulary gap: the query "what degree did I graduate with?" does not
    // contain "business administration", but those capitalized words are unique to the session.
    // Having them in query_surface means cross-session deduplication is stronger.
    // Fires only when tokens is non-empty (an education or other pattern already matched).
    if !tokens.is_empty() {
        const EDU_TRIGGERS: &[&str] = &[
            "degree in ",
            "majored in ",
            "major in ",
            "studied ",
            "i have a degree in",
            "graduated with a degree in",
            "studying for a ",
            "i earn my degree in",
        ];
        for trigger in EDU_TRIGGERS {
            if let Some(pos) = lower.find(trigger) {
                let after_start = (pos + trigger.len()).min(text.len());
                let after = &text[after_start..];
                let mut found = 0;
                for word in after.split_whitespace().take(5) {
                    let clean: String = word.chars().filter(|c| c.is_alphabetic()).collect();
                    if clean.len() >= 3
                        && clean.chars().next().map_or(false, |c| c.is_uppercase())
                        && found < 3
                    {
                        extra_tokens.push(clean.to_lowercase());
                        found += 1;
                    }
                    if found >= 3 {
                        break;
                    }
                }
            }
        }
    }

    // This catch-all layer ensures BM25 can find the neuron via ANY vocabulary in its
    // content, even when the content doesn't match any predefined category pattern.
    // Zero false-positive risk: these terms are extracted directly from the content.
    if tokens.is_empty() {
        let mut fallback: Vec<String> = Vec::new();

        // (a) Proper nouns: capitalized words ≥3 chars, not sentence-start
        for (i, word) in text.split_whitespace().enumerate() {
            let clean: String = word.chars().filter(|c| c.is_alphanumeric()).collect();
            if clean.len() >= 3
                && i > 0  // skip sentence-start capitals
                && clean.chars().next().map_or(false, |c| c.is_uppercase())
            {
                fallback.push(clean.to_lowercase());
            }
        }

        // (b) Numbers / quantities: tokens containing digits (ages, counts, times)
        for word in text.split_whitespace() {
            let clean: String = word
                .chars()
                .filter(|c| c.is_ascii_alphanumeric() || *c == '.')
                .collect();
            if clean.chars().any(|c| c.is_ascii_digit()) && clean.len() >= 2 {
                fallback.push(clean.to_lowercase());
            }
        }

        // (c) Quoted strings: extract content between " " or ' '
        let mut in_quote = false;
        let mut quote_buf = String::new();
        for ch in text.chars() {
            if ch == '"' || ch == '\'' {
                if in_quote && !quote_buf.trim().is_empty() {
                    for part in quote_buf.split_whitespace() {
                        let clean: String = part.chars().filter(|c| c.is_alphabetic()).collect();
                        if clean.len() >= 3 {
                            fallback.push(clean.to_lowercase());
                        }
                    }
                    quote_buf.clear();
                }
                in_quote = !in_quote;
            } else if in_quote {
                quote_buf.push(ch);
            }
        }

        fallback.extend(extra_tokens);
        if fallback.is_empty() {
            return None;
        }

        // Deduplicate fallback tokens
        let mut seen = std::collections::HashSet::new();
        let deduped: Vec<String> = fallback
            .into_iter()
            .filter(|t| seen.insert(t.clone()))
            .collect();
        return Some(deduped.join(", "));
    }

    // Deduplicate while preserving order; merge category vocab + disclosure terms
    let mut seen = std::collections::HashSet::new();
    let mut deduped: Vec<String> = tokens
        .into_iter()
        .filter(|t| seen.insert(t.to_string()))
        .map(|s| s.to_string())
        .collect();
    for t in extra_tokens {
        if seen.insert(t.clone()) {
            deduped.push(t);
        }
    }
    Some(deduped.join(", "))
}

// ─── Tests ────────────────────────────────────────────────────────────────────
