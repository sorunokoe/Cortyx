//! User-extensible pattern registry for typed evidence extraction.
//!
//! The `PatternRegistry` separates the general evidence extraction engine from the
//! LME-500 eval harness (P1 Segmentation). Users can add domain-specific patterns
//! without touching Cortyx source code (P35 Parameter Changes).
//!
//! # Built-in patterns
//! The registry ships with 8 built-in pattern families — the same ones in
//! `miner/evidence.rs`. These generalise the LME-500 extraction rules to any corpus.
//!
//! # User patterns
//! Place `.toml` files in `.cortyx/patterns/` to extend the registry:
//!
//! ```toml
//! [[pattern]]
//! family = "EntityFact"
//! name = "programming_language"
//! trigger = "(favorite|preferred|primary|main)\s+(programming\s+)?language"
//! confidence = 0.88
//! ```
//!
//! Run `cortyx patterns list` to see all loaded patterns.
//! Run `cortyx patterns add` to scaffold a new TOML pattern file.

use std::path::Path;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::types::EvidenceFamily;

/// A single extractable pattern entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    /// Human-readable name (used in `cortyx patterns list` output).
    pub name: String,
    /// Evidence family this pattern belongs to.
    pub family: EvidenceFamily,
    /// Regex trigger — if it matches a sentence, the pattern fires.
    pub trigger: String,
    /// Default extraction confidence for facts from this pattern.
    #[serde(default = "default_confidence")]
    pub confidence: f32,
    /// Optional human-readable description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

fn default_confidence() -> f32 {
    0.80
}

/// Compiled pattern entry (trigger regex pre-compiled for fast matching).
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CompiledPattern {
    pub name: String,
    pub family: EvidenceFamily,
    pub trigger: Regex,
    pub confidence: f32,
    pub description: Option<String>,
    /// Whether this pattern ships with Cortyx (true) or was user-defined (false).
    pub builtin: bool,
}

/// TOML file format for user-defined pattern files.
#[derive(Debug, Deserialize)]
struct PatternFile {
    #[serde(default, rename = "pattern")]
    patterns: Vec<Pattern>,
}

/// The pattern registry — all compiled patterns (built-in + user-defined).
#[derive(Debug, Default)]
pub struct PatternRegistry {
    pub patterns: Vec<CompiledPattern>,
}

impl PatternRegistry {
    /// Load the built-in patterns and any user patterns from `.cortyx/patterns/*.toml`.
    pub fn load(project_root: &Path) -> Self {
        let mut registry = PatternRegistry::default();
        registry.load_builtins();
        registry.load_user_patterns(project_root);
        registry
    }

    /// Load built-in pattern set (8 evidence families, generalised across domains).
    fn load_builtins(&mut self) {
        let builtins: &[(&str, EvidenceFamily, &str, f32, &str)] = &[
            // TemporalInterval
            (
                "elapsed_duration",
                EvidenceFamily::TemporalInterval,
                r"(?i)\d+\s+(day|week|month|year)s?\s+(ago|later|after|before|since|elapsed)",
                0.80,
                "Elapsed time expressions (e.g. '3 months ago')",
            ),
            (
                "calendar_date",
                EvidenceFamily::TemporalInterval,
                r"(?i)(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:,\s*\d{4})?|\d{4}-\d{2}-\d{2}",
                0.85,
                "Specific calendar dates in ISO or written form",
            ),
            (
                "before_after_anchor",
                EvidenceFamily::TemporalInterval,
                r"(?i)(before|after|since|until|by)\s+([A-Z][a-z]+\s+\d{4}|\d{4})",
                0.78,
                "Temporal anchors relative to a named month/year",
            ),
            // EntityFact
            (
                "job_role",
                EvidenceFamily::EntityFact,
                r"(?i)(work(?:s|ed|ing)?\s+as|(?:my|her|his|their)\s+(?:job|role|position|title)\s+is|(?:is|am|are)\s+(?:a|an))\s+[a-z][a-z\s]{2,40}(?:engineer|developer|manager|designer|researcher|analyst|consultant|director|scientist|teacher|doctor|nurse|lawyer|architect|writer|artist|chef)",
                0.88,
                "Job title or professional role",
            ),
            (
                "location",
                EvidenceFamily::EntityFact,
                r"(?i)(?:live[ds]?|moved?(?:\s+to)?|based\s+in|located\s+in|from|reside[ds]?\s+in)\s+[A-Z][a-zA-Z\s,]{2,40}",
                0.85,
                "Location: city, region, or country",
            ),
            (
                "pet_name",
                EvidenceFamily::EntityFact,
                r"(?i)(?:my|our)\s+(?:dog|cat|pet|puppy|kitten|rabbit|bird|fish|hamster|parrot)\s+(?:is\s+named?|is\s+called|'s\s+name\s+is)?\s*[A-Z][a-z]+",
                0.90,
                "Pet name",
            ),
            (
                "family_member",
                EvidenceFamily::EntityFact,
                r"(?i)(?:my|his|her|their)\s+(wife|husband|partner|boyfriend|girlfriend|mother|father|mom|dad|sister|brother|daughter|son|friend|colleague)\s+(?:is\s+named?|is\s+called|'s\s+name\s+is)?\s*[A-Z][a-z]+",
                0.88,
                "Family member or close relationship name",
            ),
            // KnowledgeUpdate
            (
                "changed_to",
                EvidenceFamily::KnowledgeUpdate,
                r"(?i)(?:changed?|switched?|moved?|transitioned?|updated?|upgraded?)\s+(?:from\s+[^,\.]{2,40}\s+)?to\s+[^,\.]{2,40}",
                0.82,
                "Value change: X switched/changed to Y",
            ),
            (
                "now_uses",
                EvidenceFamily::KnowledgeUpdate,
                r"(?i)(?:now\s+(?:uses?|is|works?|lives?|employs?|prefers?))\s+[^,\.]{2,40}",
                0.80,
                "Current-state update introduced by 'now'",
            ),
            // Preference
            (
                "likes_dislikes",
                EvidenceFamily::Preference,
                r"(?i)(?:(?:i\s+)?(?:love|like|enjoy|prefer|hate|dislike|can't\s+stand))\s+[^,\.]{2,60}",
                0.85,
                "Preference: positive or negative",
            ),
            (
                "favorite",
                EvidenceFamily::Preference,
                r"(?i)(?:(?:my|his|her|their)\s+)?favorite\s+[a-z]+\s+(?:is|are|was|were)\s+[^,\.]{2,60}",
                0.90,
                "Explicit 'favorite X is Y' statement",
            ),
            // Absence
            (
                "never_done",
                EvidenceFamily::Absence,
                r"(?i)(?:never|hasn't|haven't|hadn't|not\s+(?:yet|once)|didn't|don't)\s+(?:been\s+to|visited?|gone\s+to|tried?|done)\s+[^,\.]{2,60}",
                0.80,
                "Explicit negation or confirmed absence",
            ),
            // AssistantStated
            (
                "assistant_said",
                EvidenceFamily::AssistantStated,
                r"(?i)(?:(?:i|the\s+assistant)\s+(?:said|told|mentioned|noted|explained|suggested|recommended|stated|confirmed))\s+(?:that\s+)?[^,\.]{4,80}",
                0.75,
                "Something the assistant explicitly stated",
            ),
            // AggregateCount
            (
                "count_times",
                EvidenceFamily::AggregateCount,
                r"(?i)\d+\s+(?:times?|instances?|occasions?|visits?|trips?|times\s+per\s+week|days?\s+per\s+week)",
                0.88,
                "Aggregate count: N times/instances/occasions",
            ),
        ];

        for (name, family, trigger, confidence, desc) in builtins {
            let compiled = match Regex::new(trigger) {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!(pattern = name, "Built-in pattern regex failed: {e}");
                    continue;
                },
            };
            self.patterns.push(CompiledPattern {
                name: name.to_string(),
                family: family.clone(),
                trigger: compiled,
                confidence: *confidence,
                description: Some(desc.to_string()),
                builtin: true,
            });
        }
    }

    /// Load user-defined patterns from `.cortyx/patterns/*.toml`.
    fn load_user_patterns(&mut self, project_root: &Path) {
        let pattern_dir = project_root.join(".cortyx").join("patterns");
        let entries = match std::fs::read_dir(&pattern_dir) {
            Ok(e) => e,
            Err(_) => return, // directory absent — no user patterns
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("toml") {
                continue;
            }
            let text = match std::fs::read_to_string(&path) {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!(path = %path.display(), "Failed to read pattern file: {e}");
                    continue;
                },
            };
            let file: PatternFile = match toml::from_str(&text) {
                Ok(f) => f,
                Err(e) => {
                    tracing::warn!(path = %path.display(), "Failed to parse pattern file: {e}");
                    continue;
                },
            };
            for p in file.patterns {
                let compiled = match Regex::new(&p.trigger) {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::warn!(pattern = %p.name, "User pattern regex failed: {e}");
                        continue;
                    },
                };
                self.patterns.push(CompiledPattern {
                    name: p.name,
                    family: p.family,
                    trigger: compiled,
                    confidence: p.confidence,
                    description: p.description,
                    builtin: false,
                });
            }
        }
    }

    /// Count of loaded patterns by source.
    pub fn stats(&self) -> (usize, usize) {
        let builtin = self.patterns.iter().filter(|p| p.builtin).count();
        let user = self.patterns.len() - builtin;
        (builtin, user)
    }

    /// Check whether any loaded pattern trigger matches the given text.
    #[allow(dead_code)]
    pub fn any_matches(&self, text: &str) -> bool {
        self.patterns.iter().any(|p| p.trigger.is_match(text))
    }

    /// Return all patterns that match the given text.
    #[allow(dead_code)]
    pub fn matching_patterns<'a>(&'a self, text: &str) -> Vec<&'a CompiledPattern> {
        self.patterns
            .iter()
            .filter(|p| p.trigger.is_match(text))
            .collect()
    }
}

/// TOML template written by `cortyx patterns add`.
pub const PATTERN_TOML_TEMPLATE: &str = r#"# Cortyx user pattern file — add entries to extend evidence extraction.
# See: https://github.com/sorunokoe/Cortyx/blob/main/ARCHITECTURE.md

[[pattern]]
name = "my_pattern_name"
family = "EntityFact"    # EntityFact | TemporalInterval | KnowledgeUpdate |
                         # Preference | Absence | MultiHop | AssistantStated | AggregateCount
trigger = "(?i)your regex trigger here"
confidence = 0.85
description = "What this pattern detects"
"#;
