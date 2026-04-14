//! S4 (NE3): Temporal Knowledge Graph stored as Markdown pipe-tables.
//!
//! KG entities live as Concept neurons at `.cortyx/neurons/_kg_{entity}.context.md`.
//! Each file contains a `## facts` section with a pipe-delimited table:
//!
//! ```markdown
//! | predicate | value | valid_from | ended |
//! |---|---|---|---|
//! | language | Rust | 2024-01-01 | |
//! | lead | Alice | 2024-01-01 | 2024-06-01 |
//! ```
//!
//! Benefits over a SQLite KG:
//! - Git-trackable (diff, history, blame)
//! - Human-readable and hand-editable
//! - BM25-indexed by Cortyx — KG facts are searchable alongside code context
//! - Zero new dependencies

use std::fmt;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

// ─── Data structures ─────────────────────────────────────────────────────────

/// A single fact triple with optional temporal validity window.
#[derive(Debug, Clone, PartialEq)]
pub struct KgFact {
    pub predicate: String,
    pub value: String,
    /// ISO-8601 date string (e.g. "2024-01-15") or empty string if unknown.
    pub valid_from: String,
    /// ISO-8601 date string when this fact was superseded/ended, or empty.
    pub ended: String,
}

impl KgFact {
    /// Returns `true` if this fact is active as of `as_of` (ISO-8601 date string).
    ///
    /// Rules:
    /// - If `ended` is non-empty and `as_of >= ended`, the fact is inactive.
    /// - Otherwise the fact is considered active (open-ended or no temporal bounds).
    pub fn is_active(&self, as_of: Option<&str>) -> bool {
        if self.ended.is_empty() {
            return true;
        }
        match as_of {
            Some(d) => d < self.ended.as_str(),
            None => true, // no date supplied → treat as "now" but unknown, default active
        }
    }
}

impl fmt::Display for KgFact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "| {} | {} | {} | {} |",
            self.predicate, self.value, self.valid_from, self.ended
        )
    }
}

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
            return Ok(KgEntity { entity, facts: Vec::new(), path: path.to_path_buf() });
        }
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("read KG neuron: {}", path.display()))?;
        let facts = parse_facts_table(&content);
        Ok(KgEntity { entity, facts, path: path.to_path_buf() })
    }

    /// Persist the entity back to its neuron file, replacing the `## facts` table.
    pub fn save(&self) -> Result<()> {
        let content = self.render();
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&self.path, &content)
            .with_context(|| format!("write KG neuron: {}", self.path.display()))?;
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
    ///
    /// Returns `Err` if no matching active fact is found.
    pub fn invalidate_fact(&mut self, predicate: &str, ended: &str) -> Result<()> {
        let hit = self
            .facts
            .iter_mut()
            .find(|f| f.predicate == predicate && f.ended.is_empty());
        match hit {
            Some(f) => {
                f.ended = ended.to_string();
                Ok(())
            }
            None => bail!("No active fact with predicate {:?} on entity {:?}", predicate, self.entity),
        }
    }

    // ─── Queries ─────────────────────────────────────────────────────────

    /// Return facts active as of `as_of` (or all active facts if `None`).
    pub fn active_facts(&self, as_of: Option<&str>) -> Vec<&KgFact> {
        self.facts.iter().filter(|f| f.is_active(as_of)).collect()
    }

    /// Return the full temporal timeline for a predicate, sorted by `valid_from`.
    pub fn timeline_for(&self, predicate: &str) -> Vec<&KgFact> {
        let mut v: Vec<&KgFact> = self.facts.iter().filter(|f| f.predicate == predicate).collect();
        v.sort_by(|a, b| a.valid_from.cmp(&b.valid_from));
        v
    }

    /// Count the number of distinct active values for a predicate.
    ///
    /// Used by the multi-session counting router to answer "how many X have I had?".
    /// Returns the count of unique non-empty values (deduplication handles duplicates).
    pub fn count_active_values_for_predicate(&self, predicate: &str) -> usize {
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for f in self.active_facts(None) {
            if f.predicate == predicate && !f.value.is_empty() {
                seen.insert(f.value.as_str());
            }
        }
        seen.len()
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

// ─── Path helpers ────────────────────────────────────────────────────────────

/// Derive the KG neuron path from a project root and entity slug.
pub fn kg_neuron_path(project_root: &Path, entity: &str) -> PathBuf {
    project_root
        .join(".cortyx")
        .join("neurons")
        .join(format!("_kg_{}.context.md", slugify(entity)))
}

/// Collect all KG entity paths under a project root.
pub fn list_kg_paths(project_root: &Path) -> Vec<PathBuf> {
    let ndir = project_root.join(".cortyx").join("neurons");
    let Ok(rd) = std::fs::read_dir(&ndir) else { return Vec::new() };
    rd.flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("_kg_") && n.ends_with(".context.md"))
                .unwrap_or(false)
        })
        .collect()
}

// ─── Markdown parsing ────────────────────────────────────────────────────────

/// Parse the `## facts` pipe-table section from neuron markdown.
fn parse_facts_table(content: &str) -> Vec<KgFact> {
    let mut in_facts = false;
    let mut facts = Vec::new();

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed == "## facts" {
            in_facts = true;
            continue;
        }
        if in_facts {
            // Stop on the next `##` heading
            if trimmed.starts_with("## ") {
                break;
            }
            // Skip the header row and divider
            if !trimmed.starts_with('|') {
                continue;
            }
            let cols: Vec<&str> = trimmed
                .trim_matches('|')
                .split('|')
                .map(str::trim)
                .collect();
            if cols.len() < 4 {
                continue;
            }
            // Skip header row
            if cols[0] == "predicate" {
                continue;
            }
            // Skip divider rows (contain only `-`)
            if cols.iter().all(|c| c.chars().all(|ch| ch == '-')) {
                continue;
            }
            facts.push(KgFact {
                predicate: cols[0].to_string(),
                value: cols[1].to_string(),
                valid_from: cols[2].to_string(),
                ended: cols[3].to_string(),
            });
        }
    }
    facts
}

// ─── String helpers ──────────────────────────────────────────────────────────

/// Normalise an entity name to a lower-snake-case slug safe for filenames.
pub fn slugify(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_alphanumeric() { c.to_ascii_lowercase() } else { '_' })
        .collect::<String>()
        .split('_')
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join("_")
}

fn entity_slug_from_path(path: &Path) -> String {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    // Strip ".context" from stem if present
    let stem = stem.strip_suffix(".context").unwrap_or(stem);
    // Strip "_kg_" prefix
    stem.strip_prefix("_kg_").unwrap_or(stem).to_string()
}

// ─── KG stats ────────────────────────────────────────────────────────────────

/// Aggregate statistics across all KG entities in a project.
#[derive(Debug, Default, serde::Serialize)]
pub struct KgStats {
    pub entity_count: usize,
    pub total_facts: usize,
    pub active_facts: usize,
    pub ended_facts: usize,
}

/// Compute KG stats for a project.
pub fn compute_stats(project_root: &Path) -> KgStats {
    let mut stats = KgStats::default();
    for path in list_kg_paths(project_root) {
        if let Ok(entity) = KgEntity::load(&path) {
            stats.entity_count += 1;
            stats.total_facts += entity.facts.len();
            let active = entity.active_facts(None).len();
            stats.active_facts += active;
            stats.ended_facts += entity.facts.len() - active;
        }
    }
    stats
}

// ─── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn kg_roundtrip_save_load() {
        let dir = TempDir::new().unwrap();
        let path = kg_neuron_path(dir.path(), "project_meta");

        let mut entity = KgEntity::load(&path).unwrap();
        entity.add_fact("language", "Rust", Some("2024-01-01"));
        entity.add_fact("version", "0.2.0", Some("2024-06-01"));
        entity.save().unwrap();

        let loaded = KgEntity::load(&path).unwrap();
        assert_eq!(loaded.facts.len(), 2);
        assert_eq!(loaded.facts[0].predicate, "language");
        assert_eq!(loaded.facts[0].value, "Rust");
        assert_eq!(loaded.facts[1].value, "0.2.0");
    }

    #[test]
    fn kg_is_active_temporal_window() {
        let fact = KgFact {
            predicate: "lead".into(),
            value: "Alice".into(),
            valid_from: "2024-01-01".into(),
            ended: "2024-06-01".into(),
        };
        assert!(fact.is_active(Some("2024-03-01")));  // within window
        assert!(!fact.is_active(Some("2024-06-01"))); // on end date → inactive
        assert!(!fact.is_active(Some("2025-01-01"))); // after → inactive
    }

    #[test]
    fn kg_invalidate_fact() {
        let dir = TempDir::new().unwrap();
        let path = kg_neuron_path(dir.path(), "team");

        let mut entity = KgEntity::load(&path).unwrap();
        entity.add_fact("lead", "Alice", Some("2024-01-01"));
        entity.invalidate_fact("lead", "2024-06-01").unwrap();
        assert_eq!(entity.facts[0].ended, "2024-06-01");

        // No active fact to invalidate again
        assert!(entity.invalidate_fact("lead", "2025-01-01").is_err());
    }

    #[test]
    fn kg_timeline_sorted() {
        let dir = TempDir::new().unwrap();
        let path = kg_neuron_path(dir.path(), "deps");

        let mut entity = KgEntity::load(&path).unwrap();
        entity.add_fact("db", "postgres", Some("2023-01-01"));
        entity.invalidate_fact("db", "2024-01-01").unwrap();
        entity.add_fact("db", "sqlite", Some("2024-01-01"));

        let tl = entity.timeline_for("db");
        assert_eq!(tl.len(), 2);
        assert_eq!(tl[0].value, "postgres");
        assert_eq!(tl[1].value, "sqlite");
    }

    #[test]
    fn slugify_normalises() {
        assert_eq!(slugify("My Project!"), "my_project");
        assert_eq!(slugify("hello-world"), "hello_world");
        assert_eq!(slugify("camelCase"), "camelcase");
    }

    #[test]
    fn kg_stats_aggregates() {
        let dir = TempDir::new().unwrap();

        for name in &["entity_a", "entity_b"] {
            let path = kg_neuron_path(dir.path(), name);
            let mut e = KgEntity::load(&path).unwrap();
            e.add_fact("key", "val", None);
            e.save().unwrap();
        }
        let stats = compute_stats(dir.path());
        assert_eq!(stats.entity_count, 2);
        assert_eq!(stats.total_facts, 2);
        assert_eq!(stats.active_facts, 2);
    }
}
