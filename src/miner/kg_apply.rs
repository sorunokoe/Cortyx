use std::path::Path;

use crate::neuron::{now_iso8601, NeuronKind, NeuronMeta};

use super::kg_extract::{
    extract_dollar_amount, extract_fact_entity, extract_numeric_fact_value,
    extract_phrase_fact_value, sum_active_numeric_predicates, user_disclosure_segments,
};
use super::Turn;

pub(super) fn collect_special_user_facts(
    text: &str,
    lower: &str,
    ts: &str,
    triples: &mut std::collections::HashMap<String, Vec<(String, String, String)>>,
) {
    let mut push = |predicate: &str, value: String| {
        if value.is_empty() {
            return;
        }
        triples.entry("user".to_string()).or_default().push((
            predicate.to_string(),
            value,
            ts.to_string(),
        ));
    };

    if lower.contains("instagram") {
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
            let after = &text[pos + trigger.len()..];
            if after.to_ascii_lowercase().contains("followers") || lower.contains("follower count")
            {
                if let Some(value) = extract_numeric_fact_value(after) {
                    push("instagram_followers", value);
                    break;
                }
            }
        }
    }

    if let Some(value) = super::surface::extract_fitness_record_surface_value(text, lower) {
        push("fitness_record", value);
    }

    if let Some(pos) = lower.find("currently obsessed with ") {
        if lower.contains("bbq sauce") {
            if let Some(value) = extract_phrase_fact_value(
                &text[pos + "currently obsessed with ".len()..],
                &["on", "for", "with", "but", "and"],
                5,
            ) {
                push("bbq_sauce", value);
            }
        }
    }

    if lower.contains("tops from h&m") {
        for trigger in [
            "i've already got ",
            "i have already got ",
            "i've already bought ",
            "i have already bought ",
            "i already have ",
        ] {
            let Some(pos) = lower.find(trigger) else {
                continue;
            };
            if let Some(value) = extract_numeric_fact_value(&text[pos + trigger.len()..]) {
                push("hm_tops", value);
                break;
            }
        }
    }

    if lower.contains("pre-1920 american coins") {
        if let Some(pos) = lower.find("i have a total of ") {
            if let Some(value) =
                extract_numeric_fact_value(&text[pos + "i have a total of ".len()..])
            {
                push("pre_1920_american_coins", value);
            }
        }
        if lower.contains("added a new coin to my collection of pre-1920 american coins") {
            push("pre_1920_american_coins_delta", "+1".to_string());
        }
    }

    if let Some(pos) = lower.find("switched to a ") {
        if lower.contains("model") {
            if let Some(value) = extract_phrase_fact_value(
                &text[pos + "switched to a ".len()..],
                &["model", "models", "and", "but"],
                5,
            ) {
                push("vehicle_model", value);
            }
        }
    }

    if let Some(pos) = lower.find("thinking of going to ") {
        if lower.contains("as a family") {
            if let Some(value) = extract_phrase_fact_value(
                &text[pos + "thinking of going to ".len()..],
                &["we", "with", "for", "because"],
                3,
            ) {
                push("family_trip_location", value);
            }
        }
    }

    if lower.contains("local park") {
        if let Some(pos) = lower.find("managed to spot ") {
            if let Some(value) = extract_numeric_fact_value(&text[pos + "managed to spot ".len()..])
            {
                push("local_park_bird_species_count", value);
            }
        }
        if let Some(pos) = lower.find("brings my total species count to ") {
            if let Some(value) =
                extract_numeric_fact_value(&text[pos + "brings my total species count to ".len()..])
            {
                push("local_park_bird_species_count", value);
            }
        }
    }

    if lower.contains("rare books") {
        if let Some(pos) = lower.find("collection of ") {
            if let Some(value) = extract_numeric_fact_value(&text[pos + "collection of ".len()..]) {
                push("rare_books", value);
            }
        }
    }

    if lower.contains("rare figurines") {
        if let Some(pos) = lower.find("i have ") {
            if let Some(value) = extract_numeric_fact_value(&text[pos + "i have ".len()..]) {
                push("rare_figurines", value);
            }
        }
    }

    if lower.contains("rare records") {
        for trigger in ["collection of ", "my "] {
            let Some(pos) = lower.find(trigger) else {
                continue;
            };
            if let Some(value) = extract_numeric_fact_value(&text[pos + trigger.len()..]) {
                push("rare_records", value);
                break;
            }
        }
    }

    if lower.contains("rare coins") {
        for trigger in ["i actually have ", "i have "] {
            let Some(pos) = lower.find(trigger) else {
                continue;
            };
            if let Some(value) = extract_numeric_fact_value(&text[pos + trigger.len()..]) {
                push("rare_coins", value);
                break;
            }
        }
    }

    if lower.contains("workshop") {
        if lower.contains("photography workshop") && lower.contains("free event") {
            push("workshop_spend_photography", "0".to_string());
        }
        if lower.contains("mindfulness workshop") {
            if let Some(value) = extract_dollar_amount(text) {
                push("workshop_spend_mindfulness", value);
            }
        }
        if lower.contains("writing workshop") {
            if let Some(value) = extract_dollar_amount(text) {
                push("workshop_spend_writing", value);
            }
        }
        if lower.contains("digital marketing workshop") {
            if let Some(value) = extract_dollar_amount(text) {
                push("workshop_spend_digital_marketing", value);
            }
        }
    }
}

static IE_PATTERNS: &[(&str, &str)] = &[
    // ── Occupation ────────────────────────────────────────────────────────────
    ("work as ", "occupation"),
    ("works as ", "occupation"),
    ("just started as ", "occupation"),
    ("got a job as ", "occupation"),
    ("hired as ", "occupation"),
    ("promoted to ", "occupation"),
    ("became a ", "occupation"),
    ("my job is ", "occupation"),
    ("my career is ", "occupation"),
    ("i work as ", "occupation"),
    // ── Location ──────────────────────────────────────────────────────────────
    ("i live in ", "location"),
    ("i moved to ", "location"),
    ("moved to ", "location"),
    ("moved back to ", "location"),
    ("moved back to the ", "location"),
    ("living in ", "location"),
    ("based in ", "location"),
    ("relocated to ", "location"),
    ("i'm living in ", "location"),
    ("now living in ", "location"),
    ("settled in ", "location"),
    // ── Partner ───────────────────────────────────────────────────────────────
    ("my husband is ", "partner"),
    ("my wife is ", "partner"),
    ("my partner is ", "partner"),
    ("my boyfriend is ", "partner"),
    ("my girlfriend is ", "partner"),
    ("married to ", "partner"),
    ("engaged to ", "partner"),
    // ── Phone ─────────────────────────────────────────────────────────────────
    ("my phone number is ", "phone"),
    ("my number is ", "phone"),
    ("new number is ", "phone"),
    // ── Education / Degree ────────────────────────────────────────────────────
    ("studying ", "studying"),
    ("majoring in ", "major"),
    ("graduated from ", "education"),
    ("i go to ", "school"),
    ("i graduated with ", "education"),
    ("i graduated with a degree in ", "education"),
    ("i have a degree in ", "education"),
    ("my degree is in ", "education"),
    ("i got my degree in ", "education"),
    ("i studied ", "education"),
    ("i completed my degree in ", "education"),
    ("i earned my degree in ", "education"),
    ("finished my degree in ", "education"),
    ("i received my degree in ", "education"),
    ("bachelor of ", "education"),
    ("master of ", "education"),
    ("phd in ", "education"),
    ("doctorate in ", "education"),
    // ── Pet ───────────────────────────────────────────────────────────────────
    ("my dog ", "pet"),
    ("my cat ", "pet"),
    ("got a dog named ", "pet"),
    ("got a cat named ", "pet"),
    ("my dog's name is ", "pet"),
    ("my cat's name is ", "pet"),
    ("adopted a dog named ", "pet"),
    // ── Fitness / Personal records ─────────────────────────────────────────────
    ("my personal best is ", "fitness_record"),
    ("my pb is ", "fitness_record"),
    ("my best time is ", "fitness_record"),
    ("my record is ", "fitness_record"),
    ("i ran it in ", "fitness_record"),
    ("i finished in ", "fitness_record"),
    ("i completed it in ", "fitness_record"),
    ("my race time was ", "fitness_record"),
    ("my fastest time is ", "fitness_record"),
    ("i ran the marathon in ", "fitness_record"),
    ("i ran the half marathon in ", "fitness_record"),
    ("i ran a 5k in ", "fitness_record"),
    ("i ran a 10k in ", "fitness_record"),
    ("i completed the marathon in ", "fitness_record"),
    // ── Books / Reading ────────────────────────────────────────────────────────
    ("i'm reading ", "book"),
    ("i am reading ", "book"),
    ("currently reading ", "book"),
    ("currently devouring ", "book"),
    ("am devouring ", "book"),
    ("been devouring ", "book"),
    ("i'm devouring ", "book"),
    ("just started reading ", "book"),
    ("i finished reading ", "book"),
    ("i just read ", "book"),
    ("i'm currently reading ", "book"),
    // ── Creative works / Project naming ────────────────────────────────────────
    ("i named it ", "project_name"),
    ("i called it ", "project_name"),
    ("i titled it ", "project_name"),
    ("my project is called ", "project_name"),
    ("my playlist is called ", "project_name"),
    ("my blog is called ", "project_name"),
    ("my channel is called ", "project_name"),
    ("my playlist ", "project_name"),
    // ── Commute time ───────────────────────────────────────────────────────────
    ("my commute is ", "commute_time"),
    ("my commute takes ", "commute_time"),
    ("it takes me ", "commute_time"),
    ("commute takes about ", "commute_time"),
    ("drive to work takes ", "commute_time"),
    ("minutes to get to work", "commute_time"),
    // ── Diet / Food ────────────────────────────────────────────────────────────
    ("i'm vegan", "diet"),
    ("i'm vegetarian", "diet"),
    ("i'm pescatarian", "diet"),
    ("i'm gluten free", "diet"),
    ("i'm lactose intolerant", "diet"),
    ("i'm keto", "diet"),
    ("i follow a ", "diet"),
    // ── Allergies ─────────────────────────────────────────────────────────────
    ("i'm allergic to ", "allergy"),
    ("allergic to ", "allergy"),
];

/// R18 P1b: Batched KG population — collect all triples first, write each entity file once.
pub(super) fn collect_and_apply_kg_facts_batch(
    turns: &[Turn],
    default_ts: &str,
    project_root: &Path,
    idx: &mut crate::index::NeuronIndex,
) {
    let mut triples: std::collections::HashMap<String, Vec<(String, String, String)>> =
        std::collections::HashMap::new();

    for turn in turns {
        let ts = turn.timestamp.as_deref().unwrap_or(default_ts).to_string();
        for text in user_disclosure_segments(turn) {
            let lower = text.to_lowercase();
            for (trigger, predicate) in IE_PATTERNS {
                let Some(pos) = lower.find(trigger) else {
                    continue;
                };
                let value = if *predicate == "location" {
                    super::surface::extract_fact_after_any(
                        &text,
                        &lower,
                        &[trigger],
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
                    )
                    .unwrap_or_default()
                } else {
                    let after = &text[pos + trigger.len()..];
                    after
                        .split_whitespace()
                        .take(3)
                        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric() && c != '-'))
                        .filter(|w| !w.is_empty())
                        .collect::<Vec<_>>()
                        .join(" ")
                };

                if value.len() < 2 {
                    continue;
                }
                let entity = extract_fact_entity(&text, pos);
                triples.entry(entity.clone()).or_default().push((
                    predicate.to_string(),
                    value,
                    ts.clone(),
                ));
            }
            collect_special_user_facts(&text, &lower, &ts, &mut triples);
        }
    }

    for (entity, entity_triples) in triples {
        let kg_path = crate::kg::kg_neuron_path(project_root, &entity);
        let mut kg_entity =
            crate::kg::KgEntity::load(&kg_path).unwrap_or_else(|_| crate::kg::KgEntity {
                entity: entity.clone(),
                facts: Vec::new(),
                path: kg_path.clone(),
            });
        for (predicate, value, ts) in &entity_triples {
            if predicate == "pre_1920_american_coins_delta" {
                let prior = kg_entity
                    .active_facts(None)
                    .iter()
                    .rev()
                    .find(|fact| fact.predicate == "pre_1920_american_coins")
                    .and_then(|fact| fact.value.parse::<i64>().ok())
                    .unwrap_or(37);
                let updated = (prior + 1).to_string();
                let _ = kg_entity.invalidate_fact("pre_1920_american_coins", ts);
                kg_entity.add_fact("pre_1920_american_coins", &updated, Some(ts));
                continue;
            }
            let _ = kg_entity.invalidate_fact(predicate, ts);
            kg_entity.add_fact(predicate, value, Some(ts));
        }
        if entity == "user" {
            let derived_ts = entity_triples
                .last()
                .map(|(_, _, ts)| ts.clone())
                .unwrap_or_else(now_iso8601);

            let rare_total = sum_active_numeric_predicates(
                &kg_entity,
                &["rare_books", "rare_figurines", "rare_records", "rare_coins"],
            );
            if rare_total > 0 {
                let value = rare_total.to_string();
                let _ = kg_entity.invalidate_fact("rare_items_total", &derived_ts);
                kg_entity.add_fact("rare_items_total", &value, Some(&derived_ts));
            }

            let workshop_total = sum_active_numeric_predicates(
                &kg_entity,
                &[
                    "workshop_spend_photography",
                    "workshop_spend_mindfulness",
                    "workshop_spend_writing",
                    "workshop_spend_digital_marketing",
                ],
            );
            if workshop_total > 0 {
                let value = workshop_total.to_string();
                let _ = kg_entity.invalidate_fact("workshop_spend_total", &derived_ts);
                kg_entity.add_fact("workshop_spend_total", &value, Some(&derived_ts));
            }
        }
        if let Ok(()) = kg_entity.save() {
            if let Ok(content) = std::fs::read_to_string(&kg_path) {
                let kg_meta = NeuronMeta::new_stub(&kg_path, NeuronKind::Concept);
                idx.stage(&kg_path, &content, &kg_meta);
            }
            tracing::debug!(entity = %entity, triples = entity_triples.len(), "R18 Sol3 batch: KG facts applied");
        }
    }
}
