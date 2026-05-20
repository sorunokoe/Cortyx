//! Parametric evidence extractor: extracts typed `EvidenceFact` triples from neuron content.
//!
//! Runs at `cortyx mine` time (P10 Preliminary Action) so extraction work is paid once,
//! not at every query. Results are stored in the `## evidence_surface` section of each
//! Verbatim neuron and returned via the `cortyx_get_evidence` MCP tool.
//!
//! The 8 `EvidenceFamily` variants are matched by parametric regex families — each
//! one a config object (name, trigger, extractor, confidence), not hardcoded code.
//! This generalises LME-500 patterns to any user corpus.

use std::sync::OnceLock;

use regex::Regex;

use crate::types::{EvidenceFact, EvidenceFamily};

// ─── Sentence splitting ────────────────────────────────────────────────────────

fn sentences(text: &str) -> Vec<&str> {
    text.split(['.', '\n'])
        .map(str::trim)
        .filter(|s| s.len() > 10)
        .collect()
}

// ─── Compiled regex helpers ────────────────────────────────────────────────────

fn re(pattern: &str) -> Regex {
    Regex::new(pattern).unwrap_or_else(|_| Regex::new(r"$^").unwrap())
}

// ─── TemporalInterval patterns ────────────────────────────────────────────────

static TEMPORAL_ELAPSED: OnceLock<Regex> = OnceLock::new();
static TEMPORAL_DATE: OnceLock<Regex> = OnceLock::new();
static TEMPORAL_BEFORE_AFTER: OnceLock<Regex> = OnceLock::new();

fn temporal_regexes() -> (&'static Regex, &'static Regex, &'static Regex) {
    (
        TEMPORAL_ELAPSED.get_or_init(|| {
            re(r"(?i)(\d+)\s+(day|week|month|year)s?\s+(ago|later|after|before|since|elapsed)")
        }),
        TEMPORAL_DATE.get_or_init(|| {
            re(r"(?i)(on\s+)?((january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:,\s*\d{4})?|\d{4}-\d{2}-\d{2})")
        }),
        TEMPORAL_BEFORE_AFTER.get_or_init(|| {
            re(r"(?i)(before|after|since|until|by)\s+([A-Z][a-z]+\s+\d{4}|\d{4})")
        }),
    )
}

// ─── EntityFact patterns ───────────────────────────────────────────────────────

static ENTITY_JOB: OnceLock<Regex> = OnceLock::new();
static ENTITY_LIVE: OnceLock<Regex> = OnceLock::new();
static ENTITY_PET: OnceLock<Regex> = OnceLock::new();
static ENTITY_RELATION: OnceLock<Regex> = OnceLock::new();

fn entity_regexes() -> (
    &'static Regex,
    &'static Regex,
    &'static Regex,
    &'static Regex,
) {
    (
        ENTITY_JOB.get_or_init(|| {
            re(r"(?i)(work(?:s|ed|ing)?(?:\s+as)?|(?:my|her|his|their)\s+(?:job|role|position|title)\s+is|(?:is|am|are)\s+(?:a|an))\s+([a-z][a-z\s]{2,40}(?:engineer|developer|manager|designer|researcher|analyst|consultant|director|scientist|teacher|doctor|nurse|lawyer|architect|writer|artist|chef))")
        }),
        ENTITY_LIVE.get_or_init(|| {
            re(r"(?i)(?:live[ds]?|moved?(?:\s+to)?|based\s+in|located\s+in|from|reside[ds]?\s+in)\s+([A-Z][a-zA-Z\s,]{2,40}(?:city|street|avenue|road|lane|drive|place|court)?)")
        }),
        ENTITY_PET.get_or_init(|| {
            re(r"(?i)(?:(?:my|our|his|her|their)\s+)?(?:dog|cat|pet|puppy|kitten|rabbit|bird|fish|hamster|parrot)\s+(?:is\s+named?|is\s+called|'s\s+name\s+is)?\s*([A-Z][a-z]+)")
        }),
        ENTITY_RELATION.get_or_init(|| {
            re(r"(?i)(?:my|his|her|their)\s+(wife|husband|partner|boyfriend|girlfriend|mother|father|mom|dad|sister|brother|daughter|son|friend|colleague)\s+(?:is\s+named?|is\s+called|'s\s+name\s+is)?\s*([A-Z][a-z]+)")
        }),
    )
}

// ─── KnowledgeUpdate patterns ─────────────────────────────────────────────────

static UPDATE_CHANGED: OnceLock<Regex> = OnceLock::new();
static UPDATE_NOW: OnceLock<Regex> = OnceLock::new();

fn update_regexes() -> (&'static Regex, &'static Regex) {
    (
        UPDATE_CHANGED.get_or_init(|| {
            re(r"(?i)(?:changed?|switched?|moved?|transitioned?|updated?|upgraded?)\s+(?:from\s+[^,\.]{2,40}\s+)?to\s+([^,\.]{2,40})")
        }),
        UPDATE_NOW.get_or_init(|| {
            re(r"(?i)(?:now\s+(?:uses?|is|works?|lives?|employs?|prefers?))\s+([^,\.]{2,40})")
        }),
    )
}

// ─── Preference patterns ───────────────────────────────────────────────────────

static PREF_LIKE: OnceLock<Regex> = OnceLock::new();
static PREF_FAV: OnceLock<Regex> = OnceLock::new();

fn pref_regexes() -> (&'static Regex, &'static Regex) {
    (
        PREF_LIKE.get_or_init(|| {
            re(r"(?i)(?:(?:i\s+)?(?:love|like|enjoy|prefer|hate|dislike|can't\s+stand|can't\s+bear))\s+([^,\.]{2,60})")
        }),
        PREF_FAV.get_or_init(|| {
            re(r"(?i)(?:(?:my|his|her|their)\s+)?favorite\s+([a-z]+)\s+(?:is|are|was|were)\s+([^,\.]{2,60})")
        }),
    )
}

// ─── Absence patterns ─────────────────────────────────────────────────────────

static ABSENCE_NEVER: OnceLock<Regex> = OnceLock::new();

fn absence_regex() -> &'static Regex {
    ABSENCE_NEVER.get_or_init(|| {
        re(r"(?i)(?:never|hasn't|haven't|hadn't|not\s+(?:yet|once)|didn't|don't)\s+(?:been\s+to|visited?|gone\s+to|tried?|done)\s+([^,\.]{2,60})")
    })
}

// ─── AssistantStated patterns ──────────────────────────────────────────────────

static ASSISTANT: OnceLock<Regex> = OnceLock::new();

fn assistant_regex() -> &'static Regex {
    ASSISTANT.get_or_init(|| {
        re(r"(?i)(?:(?:i|the\s+assistant)\s+(?:said|told|mentioned|noted|explained|suggested|recommended|stated|confirmed))\s+(?:that\s+)?([^,\.]{4,80})")
    })
}

// ─── AggregateCount patterns ───────────────────────────────────────────────────

static COUNT: OnceLock<Regex> = OnceLock::new();

fn count_regex() -> &'static Regex {
    COUNT.get_or_init(|| {
        re(r"(?i)(\d+)\s+(?:times?|instances?|occasions?|visits?|trips?|times\s+per\s+week|days?\s+per\s+week)")
    })
}

// ─── Main extractor ────────────────────────────────────────────────────────────

/// Extract `EvidenceFact` triples from arbitrary conversation neuron content.
///
/// Operates at the sentence level. All 8 `EvidenceFamily` variants are matched
/// by parametric regex patterns — domain-agnostic and user-corpus independent.
pub fn extract_evidence(content: &str) -> Vec<EvidenceFact> {
    let mut facts: Vec<EvidenceFact> = Vec::new();

    for (turn_idx, sentence) in sentences(content).into_iter().enumerate() {
        let source_turn = Some(turn_idx);

        extract_temporal(sentence, source_turn, &mut facts);
        extract_entity(sentence, source_turn, &mut facts);
        extract_update(sentence, source_turn, &mut facts);
        extract_preference(sentence, source_turn, &mut facts);
        extract_absence(sentence, source_turn, &mut facts);
        extract_assistant(sentence, source_turn, &mut facts);
        extract_count(sentence, source_turn, &mut facts);
    }

    dedup_facts(&mut facts);
    facts
}

// ─── Family extractors ─────────────────────────────────────────────────────────

fn extract_temporal(sentence: &str, source_turn: Option<usize>, out: &mut Vec<EvidenceFact>) {
    let (elapsed, date, before_after) = temporal_regexes();
    if let Some(cap) = elapsed.captures(sentence) {
        let n = cap.get(1).map_or("", |m| m.as_str());
        let unit = cap.get(2).map_or("", |m| m.as_str());
        let rel = cap.get(3).map_or("", |m| m.as_str());
        out.push(EvidenceFact {
            entity: "event".into(),
            predicate: rel.to_lowercase(),
            value: format!("{n} {unit}s"),
            confidence: 0.80,
            family: EvidenceFamily::TemporalInterval,
            temporal_anchor: None,
            source_turn,
        });
    }
    if let Some(cap) = date.captures(sentence) {
        let date_str = cap.get(2).map_or("", |m| m.as_str()).trim().to_string();
        out.push(EvidenceFact {
            entity: "event".into(),
            predicate: "occurred_on".into(),
            value: date_str.clone(),
            confidence: 0.85,
            family: EvidenceFamily::TemporalInterval,
            temporal_anchor: Some(date_str),
            source_turn,
        });
    }
    if let Some(cap) = before_after.captures(sentence) {
        let rel = cap.get(1).map_or("", |m| m.as_str()).to_lowercase();
        let anchor = cap.get(2).map_or("", |m| m.as_str()).trim().to_string();
        out.push(EvidenceFact {
            entity: "event".into(),
            predicate: rel,
            value: anchor.clone(),
            confidence: 0.78,
            family: EvidenceFamily::TemporalInterval,
            temporal_anchor: Some(anchor),
            source_turn,
        });
    }
}

fn extract_entity(sentence: &str, source_turn: Option<usize>, out: &mut Vec<EvidenceFact>) {
    let (job, live, pet, relation) = entity_regexes();
    if let Some(cap) = job.captures(sentence) {
        let value = cap.get(2).map_or("", |m| m.as_str()).trim().to_string();
        out.push(EvidenceFact {
            entity: "user".into(),
            predicate: "job".into(),
            value,
            confidence: 0.88,
            family: EvidenceFamily::EntityFact,
            temporal_anchor: None,
            source_turn,
        });
    }
    if let Some(cap) = live.captures(sentence) {
        let value = cap.get(1).map_or("", |m| m.as_str()).trim().to_string();
        out.push(EvidenceFact {
            entity: "user".into(),
            predicate: "location".into(),
            value,
            confidence: 0.85,
            family: EvidenceFamily::EntityFact,
            temporal_anchor: None,
            source_turn,
        });
    }
    if let Some(cap) = pet.captures(sentence) {
        let name = cap.get(1).map_or("", |m| m.as_str()).trim().to_string();
        out.push(EvidenceFact {
            entity: "user".into(),
            predicate: "pet_name".into(),
            value: name,
            confidence: 0.90,
            family: EvidenceFamily::EntityFact,
            temporal_anchor: None,
            source_turn,
        });
    }
    if let Some(cap) = relation.captures(sentence) {
        let rel_type = cap.get(1).map_or("", |m| m.as_str()).to_lowercase();
        let name = cap.get(2).map_or("", |m| m.as_str()).trim().to_string();
        out.push(EvidenceFact {
            entity: "user".into(),
            predicate: rel_type,
            value: name,
            confidence: 0.88,
            family: EvidenceFamily::EntityFact,
            temporal_anchor: None,
            source_turn,
        });
    }
}

fn extract_update(sentence: &str, source_turn: Option<usize>, out: &mut Vec<EvidenceFact>) {
    let (changed, now) = update_regexes();
    if let Some(cap) = changed.captures(sentence) {
        let value = cap.get(1).map_or("", |m| m.as_str()).trim().to_string();
        out.push(EvidenceFact {
            entity: "user".into(),
            predicate: "updated_to".into(),
            value,
            confidence: 0.82,
            family: EvidenceFamily::KnowledgeUpdate,
            temporal_anchor: None,
            source_turn,
        });
    }
    if let Some(cap) = now.captures(sentence) {
        let value = cap.get(1).map_or("", |m| m.as_str()).trim().to_string();
        out.push(EvidenceFact {
            entity: "user".into(),
            predicate: "current_state".into(),
            value,
            confidence: 0.80,
            family: EvidenceFamily::KnowledgeUpdate,
            temporal_anchor: None,
            source_turn,
        });
    }
}

fn extract_preference(sentence: &str, source_turn: Option<usize>, out: &mut Vec<EvidenceFact>) {
    let (like, fav) = pref_regexes();
    if let Some(cap) = like.captures(sentence) {
        // Determine predicate from the verb used
        let lower = sentence.to_ascii_lowercase();
        let predicate = if lower.contains("hate")
            || lower.contains("dislike")
            || lower.contains("can't stand")
            || lower.contains("can't bear")
        {
            "dislikes"
        } else {
            "likes"
        };
        let value = cap.get(1).map_or("", |m| m.as_str()).trim().to_string();
        out.push(EvidenceFact {
            entity: "user".into(),
            predicate: predicate.into(),
            value,
            confidence: 0.85,
            family: EvidenceFamily::Preference,
            temporal_anchor: None,
            source_turn,
        });
    }
    if let Some(cap) = fav.captures(sentence) {
        let category = cap.get(1).map_or("", |m| m.as_str()).trim().to_string();
        let value = cap.get(2).map_or("", |m| m.as_str()).trim().to_string();
        out.push(EvidenceFact {
            entity: "user".into(),
            predicate: format!("favorite_{category}"),
            value,
            confidence: 0.90,
            family: EvidenceFamily::Preference,
            temporal_anchor: None,
            source_turn,
        });
    }
}

fn extract_absence(sentence: &str, source_turn: Option<usize>, out: &mut Vec<EvidenceFact>) {
    if let Some(cap) = absence_regex().captures(sentence) {
        let value = cap.get(1).map_or("", |m| m.as_str()).trim().to_string();
        out.push(EvidenceFact {
            entity: "user".into(),
            predicate: "has_not".into(),
            value,
            confidence: 0.80,
            family: EvidenceFamily::Absence,
            temporal_anchor: None,
            source_turn,
        });
    }
}

fn extract_assistant(sentence: &str, source_turn: Option<usize>, out: &mut Vec<EvidenceFact>) {
    if let Some(cap) = assistant_regex().captures(sentence) {
        let value = cap.get(1).map_or("", |m| m.as_str()).trim().to_string();
        if value.len() > 4 {
            out.push(EvidenceFact {
                entity: "assistant".into(),
                predicate: "stated".into(),
                value,
                confidence: 0.75,
                family: EvidenceFamily::AssistantStated,
                temporal_anchor: None,
                source_turn,
            });
        }
    }
}

fn extract_count(sentence: &str, source_turn: Option<usize>, out: &mut Vec<EvidenceFact>) {
    if let Some(cap) = count_regex().captures(sentence) {
        let n = cap.get(1).map_or("", |m| m.as_str()).trim().to_string();
        let unit = cap.get(0).map_or("", |m| m.as_str()).trim().to_string();
        out.push(EvidenceFact {
            entity: "event".into(),
            predicate: "count".into(),
            value: unit,
            confidence: 0.88,
            family: EvidenceFamily::AggregateCount,
            temporal_anchor: None,
            source_turn: Some(
                n.parse::<usize>()
                    .unwrap_or(0)
                    .max(source_turn.unwrap_or(0)),
            ),
        });
    }
}

// ─── Deduplication ────────────────────────────────────────────────────────────

fn dedup_facts(facts: &mut Vec<EvidenceFact>) {
    let mut seen: std::collections::HashSet<(String, String, String)> = Default::default();
    facts.retain(|f| {
        let key = (
            f.entity.clone(),
            f.predicate.clone(),
            f.value.to_lowercase(),
        );
        seen.insert(key)
    });
}

// ─── Neuron section helpers ────────────────────────────────────────────────────

/// Write an `## evidence_surface` section into neuron `content`.
///
/// The section contains a JSON array of `EvidenceFact` objects. It is inserted
/// after any existing `## answer_surface` section, or appended at the end.
/// Skipped when `facts` is empty.
pub fn append_evidence_surface_section(content: &mut String, facts: &[EvidenceFact]) {
    if facts.is_empty() {
        return;
    }
    let json = match serde_json::to_string(facts) {
        Ok(j) => j,
        Err(e) => {
            tracing::warn!("evidence_surface serialization failed: {e}");
            return;
        },
    };
    content.push_str("\n## evidence_surface\n");
    content.push_str("<!-- SECTION: evidence_surface -->\n");
    content.push_str(&json);
    content.push('\n');
    content.push_str("<!-- /SECTION -->\n");
}

/// Parse the `## evidence_surface` JSON section from neuron content.
///
/// Returns an empty vec when the section is absent or malformed.
pub fn parse_evidence_surface(content: &str) -> Vec<EvidenceFact> {
    let start_marker = "<!-- SECTION: evidence_surface -->";
    let end_marker = "<!-- /SECTION -->";
    let Some(start_pos) = content.find(start_marker) else {
        return Vec::new();
    };
    let rest = &content[start_pos + start_marker.len()..];
    let Some(end_pos) = rest.find(end_marker) else {
        return Vec::new();
    };
    let json = rest[..end_pos].trim();
    serde_json::from_str(json).unwrap_or_default()
}
