use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::kind::{NeuronKind, NeuronStatus};
use super::synapse::Synapse;
use super::util::{estimate_context_tokens, generate_neuron_uuid, now_iso8601};

/// Sidecar metadata stored beside every `.context.md` neuron.
///
/// Persisted as `<stem>.context.json` adjacent to the Markdown file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronMeta {
    /// Absolute path of the original source file this neuron describes.
    pub source_path: PathBuf,
    pub kind: NeuronKind,
    pub status: NeuronStatus,
    pub source_hash: String,
    /// BLAKE3 hash of the AST signature string (sorted function/type names only).
    /// Compared against `source_hash` to distinguish cosmetic edits from semantic changes.
    #[serde(default)]
    pub sig_hash: Option<String>,
    pub tokens: usize,
    pub last_updated: String,
    pub use_count: u32,
    /// Number of times the LLM confirmed this neuron was actually cited.
    #[serde(default)]
    pub hit_count: u32,
    pub synapses: Vec<Synapse>,
    /// Task pattern phrase (UseCase neurons only).
    pub task_pattern: Option<String>,
    /// Parent Core neuron (UseCase neurons only).
    pub parent: Option<PathBuf>,
    /// Optional project/module tag — used for namespace filtering.
    pub module: Option<String>,
    /// Source files synthesized by this Concept neuron (Concept kind only).
    pub source_files: Vec<PathBuf>,
    /// Speaker label (Verbatim neurons from conversation mining).
    pub speaker: Option<String>,
    /// ISO 8601 timestamp (Verbatim neurons).
    pub timestamp: Option<String>,
    /// Git-derived confidence score (1.0 = committed + unmodified, 0.85 = untracked/WIP).
    /// Applied as a mild BM25 multiplier.
    #[serde(
        default = "default_confidence",
        deserialize_with = "deserialize_confidence"
    )]
    pub confidence_score: f32,
    /// E2: Section shadow history, stored before evolve_* calls.
    /// Key `"_full"` holds prior full neuron bodies; other keys hold prior section bodies.
    ///
    /// # Best-effort semantics — not crash-safe
    /// Shadow sections are persisted by serializing the parent `NeuronMeta` to disk after
    /// a write operation. If the process is killed between the content write and the
    /// subsequent meta write, the shadow will be stale (pointing to an earlier version
    /// of the content). Rollback after an interrupted write will restore to an older state,
    /// not the pre-interrupted state.
    ///
    /// This is intentional: shadow sections are an undo-convenience feature for
    /// interactive use, not a transactional rollback log. Users who need crash-safe
    /// rollback should rely on their VCS (git) history.
    #[serde(
        default,
        skip_serializing_if = "HashMap::is_empty",
        deserialize_with = "deserialize_shadow_sections"
    )]
    pub shadow_sections: HashMap<String, Vec<String>>,
    /// S-XI (R16): Stable UUID — rename-resilient identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uuid: Option<String>,
}

pub const DEFAULT_CONFIDENCE: f32 = 1.0;

fn default_confidence() -> f32 {
    DEFAULT_CONFIDENCE
}

fn deserialize_confidence<'de, D: serde::Deserializer<'de>>(d: D) -> Result<f32, D::Error> {
    let v = f32::deserialize(d)?;
    Ok(v.clamp(0.0, 1.0))
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ShadowHistoryValue {
    Single(String),
    History(Vec<String>),
}

fn deserialize_shadow_sections<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<HashMap<String, Vec<String>>, D::Error> {
    let raw = Option::<HashMap<String, ShadowHistoryValue>>::deserialize(d)?.unwrap_or_default();
    Ok(raw
        .into_iter()
        .map(|(key, value)| {
            let history = match value {
                ShadowHistoryValue::Single(single) => vec![single],
                ShadowHistoryValue::History(history) => history,
            };
            (key, history)
        })
        .collect())
}

const SHADOW_HISTORY_LIMIT: usize = 3;

pub fn push_shadow(shadows: &mut HashMap<String, Vec<String>>, key: &str, value: String) {
    let history = shadows.entry(key.to_string()).or_default();
    if history.last() == Some(&value) {
        return;
    }
    history.push(value);
    if history.len() > SHADOW_HISTORY_LIMIT {
        let excess = history.len() - SHADOW_HISTORY_LIMIT;
        history.drain(0..excess);
    }
}

pub fn latest_shadow<'a>(shadows: &'a HashMap<String, Vec<String>>, key: &str) -> Option<&'a str> {
    shadows
        .get(key)
        .and_then(|history| history.last().map(String::as_str))
}

pub fn pop_shadow(shadows: &mut HashMap<String, Vec<String>>, key: &str) -> Option<String> {
    let history = shadows.get_mut(key)?;
    let value = history.pop();
    if history.is_empty() {
        shadows.remove(key);
    }
    value
}

impl NeuronMeta {
    pub fn new_stub(source: &Path, kind: NeuronKind) -> Self {
        Self {
            source_path: source.to_path_buf(),
            kind,
            status: NeuronStatus::Stub,
            source_hash: String::new(),
            sig_hash: None,
            tokens: 0,
            last_updated: now_iso8601(),
            use_count: 0,
            hit_count: 0,
            synapses: Vec::new(),
            task_pattern: None,
            parent: None,
            module: None,
            source_files: Vec::new(),
            speaker: None,
            timestamp: None,
            confidence_score: DEFAULT_CONFIDENCE,
            shadow_sections: HashMap::new(),
            uuid: Some(generate_neuron_uuid(source)),
        }
    }

    pub fn new_verbatim_chunk(
        neuron_path: &Path,
        speaker: Option<String>,
        text: &str,
        timestamp: Option<String>,
        module: Option<String>,
    ) -> Self {
        Self {
            source_path: neuron_path.to_path_buf(),
            kind: NeuronKind::Verbatim,
            status: NeuronStatus::Fresh,
            source_hash: String::new(),
            sig_hash: None,
            tokens: estimate_context_tokens(text).get(),
            last_updated: timestamp.clone().unwrap_or_default(),
            use_count: 0,
            hit_count: 0,
            synapses: Vec::new(),
            task_pattern: None,
            parent: None,
            module,
            source_files: Vec::new(),
            speaker,
            timestamp,
            confidence_score: 1.0,
            shadow_sections: HashMap::new(),
            uuid: Some(generate_neuron_uuid(neuron_path)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shadow_history_keeps_recent_entries() {
        let mut shadows = HashMap::new();
        push_shadow(&mut shadows, "purpose", "one".to_string());
        push_shadow(&mut shadows, "purpose", "two".to_string());
        push_shadow(&mut shadows, "purpose", "three".to_string());
        push_shadow(&mut shadows, "purpose", "four".to_string());
        assert_eq!(
            shadows.get("purpose").cloned().unwrap(),
            vec!["two".to_string(), "three".to_string(), "four".to_string()]
        );
        assert_eq!(latest_shadow(&shadows, "purpose"), Some("four"));
        assert_eq!(
            pop_shadow(&mut shadows, "purpose"),
            Some("four".to_string())
        );
        assert_eq!(latest_shadow(&shadows, "purpose"), Some("three"));
    }

    #[test]
    fn shadow_history_deserializes_legacy_single_string() {
        let meta: NeuronMeta = serde_json::from_str(
            r#"{
                "source_path": "src/lib.rs",
                "kind": "core",
                "status": "stub",
                "source_hash": "",
                "tokens": 1,
                "last_updated": "",
                "use_count": 0,
                "hit_count": 0,
                "synapses": [],
                "task_pattern": null,
                "parent": null,
                "module": null,
                "source_files": [],
                "speaker": null,
                "timestamp": null,
                "confidence_score": 1.0,
                "shadow_sections": { "purpose": "old body" }
            }"#,
        )
        .unwrap();
        assert_eq!(
            meta.shadow_sections.get("purpose"),
            Some(&vec!["old body".to_string()])
        );
    }

    #[test]
    fn shadow_history_skips_duplicate_value() {
        let mut shadows = HashMap::new();
        push_shadow(&mut shadows, "api", "same".to_string());
        push_shadow(&mut shadows, "api", "same".to_string());
        assert_eq!(shadows.get("api").map(|v| v.len()), Some(1));
    }
}
