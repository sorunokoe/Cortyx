use super::*;

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
pub(super) const BRIDGE_COMMUNITY_EVENT_PATTERN: &str =
    "events event lgbtq community participate participated joined support group pride parade art show activist group speech mentoring program";
pub(super) const BRIDGE_CHILD_HELP_EVENT_PATTERN: &str =
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
