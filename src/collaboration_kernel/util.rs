use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::agent_memory::StructuredDiaryEntry;
use crate::kg::{self, KgEntity};
use crate::reasoner::ReasoningReport;

use super::CollaborationDiaryRecord;

pub(super) fn modules_for_diary_entry(
    entry: &StructuredDiaryEntry,
    module_lookup: &BTreeMap<String, String>,
) -> Vec<String> {
    let mut modules = BTreeSet::new();
    for value in entry.entities.iter().chain(entry.depends_on.iter()) {
        if let Some(module) = module_lookup.get(&normalized_label(value)) {
            modules.insert(module.clone());
        }
    }
    modules.into_iter().collect()
}

pub(super) fn is_collaboration_module(module: &str) -> bool {
    let module = module.trim();
    !module.is_empty() && !module.starts_with('@')
}

pub(super) fn merge_unique<I>(target: &mut Vec<String>, values: I)
where
    I: IntoIterator<Item = String>,
{
    let mut seen: BTreeSet<String> = target.iter().map(|value| normalized_label(value)).collect();
    for value in values {
        let clean = value.trim();
        if clean.is_empty() {
            continue;
        }
        let key = normalized_label(clean);
        if key.is_empty() || !seen.insert(key) {
            continue;
        }
        target.push(clean.to_string());
    }
}

pub(super) fn normalize_values(values: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    merge_unique(&mut out, values.iter().cloned());
    out
}

pub(super) fn push_unique_fact(
    target: &mut Vec<String>,
    seen: &mut BTreeSet<String>,
    value: String,
    limit: usize,
) {
    if target.len() >= limit {
        return;
    }
    let key = normalize_summary_key(&value);
    if seen.insert(key) {
        target.push(value);
    }
}

pub(super) fn normalize_summary_key(value: &str) -> String {
    value
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

pub(super) fn collaborator_key(value: &str) -> String {
    normalized_label(value)
}

pub(super) fn normalized_label(value: &str) -> String {
    kg::slugify(value)
}

pub(super) fn collaborator_from_agent_entity(entity: &str) -> Option<String> {
    entity
        .strip_prefix("agent_")
        .map(|value| value.replace('_', "-"))
        .filter(|value| !value.is_empty())
}

pub(super) fn identity_tokens(value: &str) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut tokens = Vec::new();
    for token in value
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .map(str::trim)
        .filter(|token| !token.is_empty())
    {
        let lower = token.to_ascii_lowercase();
        if lower.len() < 2
            || super::IDENTITY_STOPWORDS.contains(&lower.as_str())
            || !seen.insert(lower.clone())
        {
            continue;
        }
        tokens.push(lower);
    }
    tokens
}

pub(super) fn max_optional_strings<I>(values: I) -> Option<String>
where
    I: IntoIterator<Item = Option<String>>,
{
    values.into_iter().flatten().max()
}

pub(super) fn max_optional_pair(left: Option<String>, right: Option<String>) -> Option<String> {
    max_optional_strings([left, right])
}

pub(super) fn latest_diary_record<'a>(
    diary_records: &'a [&'a CollaborationDiaryRecord],
) -> Option<&'a CollaborationDiaryRecord> {
    use crate::agent_memory::summarize_structured_diary_entry;
    diary_records.iter().copied().max_by(|left, right| {
        left.when
            .as_deref()
            .cmp(&right.when.as_deref())
            .then_with(|| {
                summarize_structured_diary_entry(&left.entry)
                    .cmp(&summarize_structured_diary_entry(&right.entry))
            })
    })
}

pub(super) fn collect_supporting_facts(
    agent_kg: Option<&KgEntity>,
    entity_names: &[String],
    kg_by_entity: &HashMap<String, &KgEntity>,
    reasoning: Option<&ReasoningReport>,
) -> (Vec<String>, usize, usize) {
    let mut facts = Vec::new();
    let mut seen = BTreeSet::new();
    let mut matched_entities = BTreeSet::new();
    if let Some(agent_kg) = agent_kg {
        matched_entities.insert(agent_kg.entity.clone());
        for predicate in super::DIRECT_AGENT_FACT_PREDICATES {
            if let Some(value) = agent_kg.latest_active_value(predicate) {
                push_unique_fact(
                    &mut facts,
                    &mut seen,
                    format!("{}.{} = {}", agent_kg.entity, predicate, value),
                    6,
                );
            }
        }
    }

    for name in entity_names {
        let slug = normalized_label(name);
        if !slug.is_empty() && kg_by_entity.contains_key(&slug) {
            matched_entities.insert(slug);
        }
    }

    let mut matched_reasoned_facts = Vec::new();
    if let Some(reasoning) = reasoning {
        let entity_slugs: BTreeSet<String> = matched_entities.iter().cloned().collect();
        matched_reasoned_facts.extend(
            reasoning
                .facts
                .iter()
                .filter(|fact| {
                    entity_slugs.contains(&normalized_label(&fact.entity))
                        || entity_slugs.contains(&normalized_label(&fact.value))
                })
                .collect::<Vec<_>>(),
        );
        matched_reasoned_facts.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.entity.cmp(&right.entity))
                .then_with(|| left.predicate.cmp(&right.predicate))
        });
        for fact in matched_reasoned_facts.iter().take(6) {
            push_unique_fact(
                &mut facts,
                &mut seen,
                format!(
                    "{}.{} = {} (score {:.2})",
                    fact.entity, fact.predicate, fact.value, fact.score
                ),
                6,
            );
        }
    }

    let mut kg_fact_count = 0usize;
    for entity_name in matched_entities {
        let Some(entity) = kg_by_entity.get(&entity_name).copied().or_else(|| {
            agent_kg.filter(|candidate| candidate.entity.as_str() == entity_name.as_str())
        }) else {
            continue;
        };
        let active = entity.active_facts(None);
        kg_fact_count += active.len();
        for fact in active {
            push_unique_fact(
                &mut facts,
                &mut seen,
                format!("{}.{} = {}", entity.entity, fact.predicate, fact.value),
                6,
            );
        }
    }

    (facts, matched_reasoned_facts.len(), kg_fact_count)
}
