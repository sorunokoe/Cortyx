use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::error::Result;

use crate::neuron::atomic_write;

use super::fact::KgFact;
use super::parse::parse_facts_table;
use super::render::entity_slug_from_path;

/// An in-memory representation of one KG entity file.
#[derive(Debug, Clone)]
pub struct KgEntity {
    /// Normalised entity slug (lower-snake-case). Used in the filename.
    pub entity: String,
    pub facts: Vec<KgFact>,
    /// Absolute path to the `.context.md` neuron file.
    pub path: PathBuf,
}

impl KgEntity {
    // ─── I/O ─────────────────────────────────────────────────────────────

    /// Load (or create empty) a `KgEntity` from the neuron path.
    pub fn load(path: &Path) -> Result<Self> {
        let entity = entity_slug_from_path(path);
        if !path.exists() {
            return Ok(KgEntity {
                entity,
                facts: Vec::new(),
                path: path.to_path_buf(),
            });
        }
        let content = std::fs::read_to_string(path)
            .map_err(|e| crate::cortyx_err!("read KG neuron {}: {e}", path.display()))?;
        let facts = parse_facts_table(&content);
        Ok(KgEntity {
            entity,
            facts,
            path: path.to_path_buf(),
        })
    }

    /// Persist the entity back to its neuron file, replacing the `## facts` table.
    pub fn save(&self) -> Result<()> {
        let content = self.render();
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        atomic_write(&self.path, content.as_bytes())
            .map_err(|e| crate::cortyx_err!("write KG neuron {}: {e}", self.path.display()))?;
        Ok(())
    }

    // ─── Mutations ───────────────────────────────────────────────────────

    /// Add a new fact. If a fact with the same predicate and no `ended` date already
    /// exists, it is NOT automatically invalidated — call `invalidate_fact` first.
    pub fn add_fact(&mut self, predicate: &str, value: &str, valid_from: Option<&str>) {
        self.facts.push(KgFact {
            predicate: predicate.to_string(),
            value: value.to_string(),
            valid_from: valid_from.unwrap_or("").to_string(),
            ended: String::new(),
        });
    }

    /// Set the `ended` date on the first active fact matching `predicate`.
    pub fn invalidate_fact(&mut self, predicate: &str, ended: &str) -> Result<()> {
        let hit = self
            .facts
            .iter_mut()
            .find(|f| f.predicate == predicate && f.ended.is_empty());
        match hit {
            Some(f) => {
                f.ended = ended.to_string();
                Ok(())
            },
            None => crate::cortyx_bail!(
                "No active fact with predicate {:?} on entity {:?}",
                predicate,
                self.entity
            ),
        }
    }

    /// Replace the currently active value for a predicate with `value`.
    ///
    /// If the same value is already the only active one, this is a no-op.
    /// If `value` is empty, all active values for the predicate are ended.
    pub fn replace_active_fact(
        &mut self,
        predicate: &str,
        value: &str,
        effective_from: &str,
    ) -> bool {
        let desired = value.trim();
        let mut changed = false;
        let mut kept_desired = false;

        for fact in self
            .facts
            .iter_mut()
            .filter(|fact| fact.predicate == predicate && fact.ended.is_empty())
        {
            if !desired.is_empty() && fact.value == desired && !kept_desired {
                kept_desired = true;
                continue;
            }
            fact.ended = effective_from.to_string();
            changed = true;
        }

        if !desired.is_empty() && !kept_desired {
            self.add_fact(predicate, desired, Some(effective_from));
            changed = true;
        }

        changed
    }

    /// Synchronize the active value set for a predicate.
    ///
    /// Active facts not present in `values` are ended at `effective_from`.
    /// New desired values are inserted with `valid_from = effective_from`.
    pub fn sync_active_values(
        &mut self,
        predicate: &str,
        values: &[String],
        effective_from: &str,
    ) -> bool {
        let mut desired = Vec::new();
        for value in values {
            let clean = value.trim();
            if clean.is_empty() {
                continue;
            }
            if !desired.iter().any(|existing: &String| existing == clean) {
                desired.push(clean.to_string());
            }
        }

        let desired_set: HashSet<String> = desired.iter().cloned().collect();
        let mut kept: HashSet<String> = HashSet::new();
        let mut changed = false;

        for fact in self
            .facts
            .iter_mut()
            .filter(|fact| fact.predicate == predicate && fact.ended.is_empty())
        {
            if desired_set.contains(&fact.value) && kept.insert(fact.value.clone()) {
                continue;
            }
            fact.ended = effective_from.to_string();
            changed = true;
        }

        for value in desired {
            if kept.insert(value.clone()) {
                self.add_fact(predicate, &value, Some(effective_from));
                changed = true;
            }
        }

        changed
    }

    // ─── Queries ─────────────────────────────────────────────────────────

    /// Return facts active as of `as_of` (or all active facts if `None`).
    pub fn active_facts(&self, as_of: Option<&str>) -> Vec<&KgFact> {
        self.facts.iter().filter(|f| f.is_active(as_of)).collect()
    }

    /// Return active facts for a single predicate.
    pub fn active_values_for_predicate(
        &self,
        predicate: &str,
        as_of: Option<&str>,
    ) -> Vec<&KgFact> {
        self.active_facts(as_of)
            .into_iter()
            .filter(|fact| fact.predicate == predicate)
            .collect()
    }

    /// Return the full temporal timeline for a predicate, sorted by `valid_from`.
    pub fn timeline_for(&self, predicate: &str) -> Vec<&KgFact> {
        let mut v: Vec<&KgFact> = self
            .facts
            .iter()
            .filter(|f| f.predicate == predicate)
            .collect();
        v.sort_by(|a, b| a.valid_from.cmp(&b.valid_from));
        v
    }

    /// Count the number of distinct active values for a predicate.
    pub fn count_active_values_for_predicate(&self, predicate: &str) -> usize {
        let mut seen: HashSet<&str> = HashSet::new();
        for f in self.active_facts(None) {
            if f.predicate == predicate && !f.value.is_empty() {
                seen.insert(f.value.as_str());
            }
        }
        seen.len()
    }

    /// Return the latest active value for a predicate, preferring the newest `valid_from`.
    pub fn latest_active_value(&self, predicate: &str) -> Option<String> {
        self.active_values_for_predicate(predicate, None)
            .into_iter()
            .max_by(|left, right| {
                left.valid_from
                    .cmp(&right.valid_from)
                    .then_with(|| left.value.cmp(&right.value))
            })
            .map(|fact| fact.value.clone())
    }

    /// Return the distinct active values for a predicate in deterministic order.
    pub fn active_value_strings(&self, predicate: &str) -> Vec<String> {
        let mut values: Vec<String> = self
            .active_values_for_predicate(predicate, None)
            .into_iter()
            .map(|fact| fact.value.clone())
            .collect();
        values.sort();
        values.dedup();
        values
    }

    /// Return the latest `valid_from` timestamp among currently active facts.
    pub fn latest_active_timestamp(&self) -> Option<String> {
        self.active_facts(None)
            .into_iter()
            .filter_map(|fact| (!fact.valid_from.is_empty()).then_some(fact.valid_from.clone()))
            .max()
    }

    // ─── Rendering ───────────────────────────────────────────────────────

    fn render(&self) -> String {
        let mut out = format!(
            "# KG: {entity}\n\n\
             <!-- AUTO-GENERATED KG NEURON — edit facts table or run cortyx_kg_add -->\n\n\
             ## purpose\n\
             Temporal knowledge graph entity for `{entity}`. \
             Managed by the `cortyx_kg_*` MCP tools.\n\n\
             ## facts\n\
             | predicate | value | valid_from | ended |\n\
             |---|---|---|---|\n",
            entity = self.entity,
        );
        for fact in &self.facts {
            out.push_str(&fact.to_string());
            out.push('\n');
        }
        out.push_str("\n## context\n_Evolve with additional free-form notes about this entity._\n");
        out
    }
}

#[cfg(test)]
#[path = "entity_tests.rs"]
mod tests;
