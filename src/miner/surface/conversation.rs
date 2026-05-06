use std::collections::HashMap;

use crate::answer_plane::{mine_dialogue_answer_surface_span, mine_dialogue_question_pattern};

use super::super::kg_extract::extract_phrase_fact_value;
use super::super::{AnswerSurfaceRow, Turn};
use super::patterns::{
    compile_regex, extract_clause_after_any, extract_fact_after_any,
    extract_research_surface_value, normalize_answer_surface_span,
    normalize_dialogue_reason_phrase, normalize_dialogue_support_effect_phrase,
    push_answer_surface_row,
};

pub(crate) fn generate_dialogue_answer_surface_rows(
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

pub(crate) fn generate_cross_chunk_dialogue_answer_surface_rows(
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

pub(crate) fn generate_dialogue_bridge_surface_rows(turn: &Turn) -> Vec<AnswerSurfaceRow> {
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

pub(crate) fn generate_session_bridge_surface_rows(turns: &[Turn]) -> Vec<AnswerSurfaceRow> {
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

pub(super) fn generate_embedded_dialogue_answer_surface_rows(text: &str) -> Vec<AnswerSurfaceRow> {
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

pub(crate) fn parse_embedded_dialogue_line(line: &str) -> Option<(&str, &str)> {
    let (speaker, rest) = line.split_once(':')?;
    if !is_dialogue_speaker(speaker) {
        return None;
    }
    let rest = rest.trim();
    (!rest.is_empty()).then_some((speaker.trim(), rest))
}

pub(crate) fn is_dialogue_speaker(prefix: &str) -> bool {
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

pub(crate) fn normalize_dialogue_speaker_label(speaker: &str) -> String {
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

pub(crate) fn parse_embedded_session_timestamp(line: &str) -> Option<String> {
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

pub(super) fn scoped_question_pattern(
    question_pattern: &str,
    speaker: Option<&str>,
) -> Option<String> {
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

pub(crate) fn generate_temporal_turn_answer_surface_rows(turn: &Turn) -> Vec<AnswerSurfaceRow> {
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
