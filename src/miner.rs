/// Conversation mining — parses dialogue files into Verbatim neurons.
///
/// Supported formats (auto-detected by content structure):
/// - ChatGPT `conversations.json` export (mapping tree)
/// - Claude markdown export (`## Human` / `## Assistant` headings)
/// - LongMemEval JSON (`session_history` turn arrays)
/// - Generic markdown (any `##` headings as chunk boundaries)
///
/// Each parsed turn becomes a `NeuronKind::Verbatim` neuron.  Consecutive
/// turns in the same file get `SynapseType::TemporalFollows` edges so the
/// graph captures temporal order without structural pre-knowledge.
use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::neuron::{NeuronMeta, Synapse, SynapseType, atomic_write, atomic_write_json, neuron_dir, now_iso8601, unix_secs_to_datetime};
use crate::index::NeuronIndex;

// ─── Public API ───────────────────────────────────────────────────────────────

/// A single extracted conversation turn ready to be written as a Verbatim neuron.
#[derive(Debug, Clone)]
pub struct Turn {
    pub speaker: Option<String>,
    pub text: String,
    pub timestamp: Option<String>,
}

/// Parse a file (or all `.json`/`.md` files in a directory) into turns,
/// write them as Verbatim neurons, and upsert into the index.
///
/// Returns the number of neurons created.
pub fn mine_path(
    path: &Path,
    project_root: &Path,
    idx: &mut NeuronIndex,
    module: Option<&str>,
) -> Result<usize> {
    if path.is_dir() {
        let mut total = 0;
        for entry in walkdir::WalkDir::new(path).min_depth(1).into_iter().filter_map(|e| e.ok()) {
            if entry.file_type().is_file() {
                let ext = entry.path().extension().and_then(|e| e.to_str()).unwrap_or("");
                if matches!(ext, "json" | "md" | "txt") {
                    total += mine_file(entry.path(), project_root, idx, module)
                        .unwrap_or_else(|e| { tracing::warn!("Skipping {}: {e}", entry.path().display()); 0 });
                }
            }
        }
        Ok(total)
    } else {
        mine_file(path, project_root, idx, module)
    }
}

/// Mine a single file. Auto-detects format and writes Verbatim neurons.
pub fn mine_file(
    path: &Path,
    project_root: &Path,
    idx: &mut NeuronIndex,
    module: Option<&str>,
) -> Result<usize> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("Cannot read {}", path.display()))?;

    let turns = detect_and_parse(&raw)
        .with_context(|| format!("Failed to parse {}", path.display()))?;

    write_verbatim_neurons(&turns, path, project_root, idx, module)
}

/// Mine a raw string (called from the MCP tool with inline content).
pub fn mine_text(
    content: &str,
    source_hint: &str,
    project_root: &Path,
    idx: &mut NeuronIndex,
    module: Option<&str>,
    speaker: Option<&str>,
    timestamp: Option<&str>,
) -> Result<usize> {
    let turns = if speaker.is_some() || timestamp.is_some() {
        // Single explicit turn
        vec![Turn {
            speaker: speaker.map(|s| s.to_string()),
            text: content.to_string(),
            timestamp: timestamp.map(|s| s.to_string()),
        }]
    } else {
        detect_and_parse(content).unwrap_or_else(|_| {
            vec![Turn { speaker: None, text: content.to_string(), timestamp: None }]
        })
    };

    let fake_path = PathBuf::from(source_hint);
    write_verbatim_neurons(&turns, &fake_path, project_root, idx, module)
}

// ─── Format detection ─────────────────────────────────────────────────────────

fn detect_and_parse(raw: &str) -> Result<Vec<Turn>> {
    let trimmed = raw.trim_start();

    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        // Try LongMemEval first (has `session_history`)
        if trimmed.contains("session_history") {
            if let Ok(turns) = parse_longmemeval(raw) {
                return Ok(turns);
            }
        }
        // Try ChatGPT export (has `mapping`)
        if trimmed.contains("\"mapping\"") {
            if let Ok(turns) = parse_chatgpt(raw) {
                return Ok(turns);
            }
        }
        // Fallback: generic JSON array of turn objects
        if let Ok(turns) = parse_generic_json(raw) {
            return Ok(turns);
        }
    }

    // Markdown format
    if raw.contains("## Human") || raw.contains("## Assistant") {
        return parse_claude_md(raw);
    }

    parse_generic_md(raw)
}

// ─── Format parsers ───────────────────────────────────────────────────────────

// -- LongMemEval -------------------------------------------------------------

/// LongMemEval dataset: array of sessions, each with `session_history` array.
/// See: huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
#[derive(Deserialize)]
struct LmeSession {
    #[allow(dead_code)]
    #[serde(default)]
    session_id: String,
    #[serde(default)]
    session_history: Vec<LmeTurn>,
}

#[derive(Deserialize)]
struct LmeTurn {
    role: String,
    content: String,
    #[serde(default)]
    timestamp: Option<String>,
}

fn parse_longmemeval(raw: &str) -> Result<Vec<Turn>> {
    // Try as array of sessions
    if let Ok(sessions) = serde_json::from_str::<Vec<LmeSession>>(raw) {
        let turns = sessions.into_iter().flat_map(|s| {
            s.session_history.into_iter().map(move |t| Turn {
                speaker: Some(t.role.clone()),
                text: t.content.clone(),
                timestamp: t.timestamp.clone(),
            })
        }).filter(|t| !t.text.trim().is_empty()).collect();
        return Ok(turns);
    }
    // Try as single session object
    if let Ok(session) = serde_json::from_str::<LmeSession>(raw) {
        let turns = session.session_history.into_iter().map(|t| Turn {
            speaker: Some(t.role.clone()),
            text: t.content.clone(),
            timestamp: t.timestamp.clone(),
        }).filter(|t| !t.text.trim().is_empty()).collect();
        return Ok(turns);
    }
    anyhow::bail!("Not LongMemEval format")
}

// -- ChatGPT conversations.json ---------------------------------------------

/// Subset of the ChatGPT export format needed for mining.
#[allow(dead_code)]
#[derive(Deserialize)]
struct ChatGptExport {
    #[serde(default)]
    title: String,
    mapping: std::collections::HashMap<String, ChatGptNode>,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct ChatGptNode {
    id: String,
    message: Option<ChatGptMessage>,
    parent: Option<String>,
    children: Vec<String>,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct ChatGptMessage {
    author: ChatGptAuthor,
    content: ChatGptContent,
    #[serde(default)]
    create_time: Option<f64>,
}

#[derive(Deserialize)]
struct ChatGptAuthor {
    role: String,
}

#[derive(Deserialize)]
struct ChatGptContent {
    #[serde(default)]
    parts: Vec<serde_json::Value>,
}

fn parse_chatgpt(raw: &str) -> Result<Vec<Turn>> {
    // May be a single conversation or an array
    let conversations: Vec<ChatGptExport> = if raw.trim_start().starts_with('[') {
        serde_json::from_str(raw)?
    } else {
        vec![serde_json::from_str(raw)?]
    };

    let mut turns = Vec::new();
    for conv in conversations {
        // BFS walk to collect all reachable nodes in conversation order.
        // A visited set prevents duplicates if the export contains cycles or
        // a node appears as a child of multiple parents.
        let root = conv.mapping.values().find(|n| n.parent.is_none());
        let mut queue: std::collections::VecDeque<&str> =
            root.map(|r| std::iter::once(r.id.as_str()).collect())
                .unwrap_or_default();
        let mut visited: HashSet<&str> = queue.iter().copied().collect();
        while let Some(id) = queue.pop_front() {
            if let Some(node) = conv.mapping.get(id) {
                if let Some(msg) = &node.message {
                    let role = msg.author.role.clone();
                    if role == "user" || role == "assistant" {
                        let text: String = msg.content.parts.iter()
                            .filter_map(|p| p.as_str().map(|s| s.to_string()))
                            .collect::<Vec<_>>()
                            .join("\n");
                        if !text.trim().is_empty() {
                            let ts = msg.create_time.map(|t| {
                                let (y, mo, d, ..) = unix_secs_to_datetime(t as u64);
                                format!("{y:04}-{mo:02}-{d:02}T00:00:00Z")
                            });
                            turns.push(Turn { speaker: Some(role), text, timestamp: ts });
                        }
                    }
                }
                for child_id in &node.children {
                    if visited.insert(child_id.as_str()) {
                        queue.push_back(child_id.as_str());
                    }
                }
            }
        }
    }
    Ok(turns)
}

// -- Claude markdown export -------------------------------------------------

fn parse_claude_md(raw: &str) -> Result<Vec<Turn>> {
    // Detect `## Human` / `## Assistant` (and variants) explicitly.
    // We do NOT use parse_headed_md here because the full marker text
    // becomes the role in that helper — leaving an empty string.
    let mut turns = Vec::new();
    let mut current_speaker: Option<String> = None;
    let mut current_lines: Vec<&str> = Vec::new();

    for line in raw.lines() {
        if line.starts_with("## ") {
            let heading = line.trim_start_matches("## ").trim().to_lowercase();
            // Only treat known roles as turn boundaries; other ## headings are content
            if matches!(heading.as_str(), "human" | "assistant" | "user" | "ai" | "system") {
                if !current_lines.is_empty() {
                    let text = current_lines.join("\n").trim().to_string();
                    if !text.is_empty() {
                        turns.push(Turn { speaker: current_speaker.clone(), text, timestamp: None });
                    }
                    current_lines.clear();
                }
                current_speaker = Some(heading);
            } else {
                current_lines.push(line);
            }
        } else {
            current_lines.push(line);
        }
    }
    if !current_lines.is_empty() {
        let text = current_lines.join("\n").trim().to_string();
        if !text.is_empty() {
            turns.push(Turn { speaker: current_speaker.clone(), text, timestamp: None });
        }
    }
    Ok(turns)
}

// -- Generic markdown -------------------------------------------------------

fn parse_generic_md(raw: &str) -> Result<Vec<Turn>> {
    // Split on ## or ### headings — treat each section as a chunk
    let turns = parse_headed_md(raw, &["## ", "### "])?;
    if !turns.is_empty() {
        return Ok(turns);
    }
    // No headings: treat whole file as a single chunk
    Ok(vec![Turn { speaker: None, text: raw.to_string(), timestamp: None }])
}

fn parse_headed_md(raw: &str, markers: &[&str]) -> Result<Vec<Turn>> {
    let mut turns = Vec::new();
    let mut current_speaker: Option<String> = None;
    let mut current_lines: Vec<&str> = Vec::new();

    for line in raw.lines() {
        if let Some(marker) = markers.iter().find(|&&m| line.starts_with(m)) {
            // Flush previous chunk
            if !current_lines.is_empty() {
                let text = current_lines.join("\n").trim().to_string();
                if !text.is_empty() {
                    turns.push(Turn { speaker: current_speaker.clone(), text, timestamp: None });
                }
                current_lines.clear();
            }
            // New speaker from heading
            let role = line.trim_start_matches(marker).trim().to_lowercase();
            current_speaker = if role.is_empty() { None } else { Some(role) };
        } else {
            current_lines.push(line);
        }
    }
    // Flush final chunk
    if !current_lines.is_empty() {
        let text = current_lines.join("\n").trim().to_string();
        if !text.is_empty() {
            turns.push(Turn { speaker: current_speaker.clone(), text, timestamp: None });
        }
    }
    Ok(turns)
}

// -- Generic JSON array of turn objects -------------------------------------

#[derive(Deserialize)]
struct GenericTurn {
    #[serde(alias = "role", alias = "speaker", alias = "author", default)]
    speaker: Option<String>,
    #[serde(alias = "text", alias = "message", alias = "content")]
    content: String,
    #[serde(alias = "ts", alias = "time", default)]
    timestamp: Option<String>,
}

fn parse_generic_json(raw: &str) -> Result<Vec<Turn>> {
    let items: Vec<GenericTurn> = serde_json::from_str(raw)?;
    Ok(items.into_iter()
        .filter(|t| !t.content.trim().is_empty())
        .map(|t| Turn { speaker: t.speaker, text: t.content, timestamp: t.timestamp })
        .collect())
}

// ─── Neuron writer ────────────────────────────────────────────────────────────

/// Write a sequence of turns as Verbatim neurons with TemporalFollows synapses.
pub fn write_verbatim_neurons(
    turns: &[Turn],
    source: &Path,
    project_root: &Path,
    idx: &mut NeuronIndex,
    module: Option<&str>,
) -> Result<usize> {
    if turns.is_empty() {
        return Ok(0);
    }

    let ndir = neuron_dir(project_root);
    std::fs::create_dir_all(&ndir)?;

    // Derive a stable base name from the source file
    let base = source.file_stem()
        .map(|s| s.to_string_lossy().replace(|c: char| !c.is_alphanumeric() && c != '_', "_"))
        .unwrap_or_else(|| "chat".to_string());
    let base = base.trim_matches('_').to_string();

    let now = now_iso8601();
    let mut neuron_paths: Vec<PathBuf> = Vec::new();

    for (i, turn) in turns.iter().enumerate() {
        let speaker_slug = turn.speaker.as_deref()
            .map(|s| s.replace(' ', "_"))
            .unwrap_or_else(|| "chunk".to_string());
        let file_name = format!("{base}_{i:04}_{speaker_slug}.verbatim.md");
        let neuron_path = ndir.join(&file_name);

        let content = format_verbatim_neuron(turn, source, i, turns.len(), &now);
        atomic_write(&neuron_path, content.as_bytes())?;

        let meta = NeuronMeta::new_verbatim_chunk(
            &neuron_path,
            turn.speaker.clone(),
            &content,
            turn.timestamp.clone().or_else(|| Some(now.clone())),
            module.map(|s| s.to_string()),
        );
        let meta_path = crate::neuron::meta_path(&neuron_path);
        atomic_write_json(&meta_path, &meta)?;

        idx.stage(&neuron_path, &content, &meta);
        neuron_paths.push(neuron_path);
    }

    // Wire consecutive turns with TemporalFollows synapses
    let count = neuron_paths.len();
    for i in 0..count.saturating_sub(1) {
        let meta_path = crate::neuron::meta_path(&neuron_paths[i]);
        if let Ok(data) = std::fs::read_to_string(&meta_path) {
            if let Ok(mut meta) = serde_json::from_str::<NeuronMeta>(&data) {
                meta.synapses.push(Synapse {
                    target: neuron_paths[i + 1].clone(),
                    edge_type: SynapseType::TemporalFollows,
                    weight: 0.6,
                    reason: "consecutive turn".to_string(),
                });
                atomic_write_json(&meta_path, &meta)?;
                let content = std::fs::read_to_string(&neuron_paths[i]).unwrap_or_default();
                idx.stage(&neuron_paths[i], &content, &meta);
            }
        }
    }

    // One rebuild + save for all neurons (instead of one per upsert)
    idx.commit()?;
    Ok(count)
}

/// Format a single verbatim turn as a neuron document.
fn format_verbatim_neuron(turn: &Turn, source: &Path, index: usize, total: usize, now: &str) -> String {
    let speaker = turn.speaker.as_deref().unwrap_or("unknown");
    let ts = turn.timestamp.as_deref().unwrap_or(now);
    format!(
        "<!-- VERBATIM CHUNK {index}/{total} — source: {} -->\n\
         <!-- speaker: {speaker} | timestamp: {ts} -->\n\n\
         {}",
        source.file_name().unwrap_or_default().to_string_lossy(),
        turn.text
    )
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_claude_md_basic() {
        let md = "## Human\nHow does auth work?\n\n## Assistant\nAuth uses JWT tokens.\n";
        let turns = parse_claude_md(md).unwrap();
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].speaker.as_deref(), Some("human"));
        assert!(turns[0].text.contains("auth"));
        assert_eq!(turns[1].speaker.as_deref(), Some("assistant"));
        assert!(turns[1].text.contains("JWT"));
    }

    #[test]
    fn parse_generic_md_single_chunk() {
        let md = "No headings here.\nJust some content.";
        let turns = parse_generic_md(md).unwrap();
        assert_eq!(turns.len(), 1);
        assert!(turns[0].text.contains("No headings"));
    }

    #[test]
    fn parse_generic_md_with_headings() {
        let md = "## Section A\nContent A\n\n## Section B\nContent B\n";
        let turns = parse_generic_md(md).unwrap();
        assert_eq!(turns.len(), 2);
        assert!(turns[0].text.contains("Content A"));
    }

    #[test]
    fn parse_longmemeval_session_array() {
        let json = r#"[{"session_id":"s1","session_history":[
            {"role":"user","content":"What time is it?"},
            {"role":"assistant","content":"It is noon."}
        ]}]"#;
        let turns = parse_longmemeval(json).unwrap();
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].speaker.as_deref(), Some("user"));
        assert!(turns[1].text.contains("noon"));
    }

    #[test]
    fn parse_longmemeval_single_session() {
        let json = r#"{"session_id":"s1","session_history":[
            {"role":"user","content":"Hello"},
            {"role":"assistant","content":"Hi there"}
        ]}"#;
        let turns = parse_longmemeval(json).unwrap();
        assert_eq!(turns.len(), 2);
    }

    #[test]
    fn parse_generic_json_turns() {
        let json = r#"[
            {"role":"user","content":"What is rust?"},
            {"role":"assistant","content":"A systems language."}
        ]"#;
        let turns = parse_generic_json(json).unwrap();
        assert_eq!(turns.len(), 2);
        assert!(turns[0].text.contains("rust"));
    }

    #[test]
    fn detect_longmemeval_format() {
        let json = r#"[{"session_id":"s1","session_history":[
            {"role":"user","content":"Test question"}
        ]}]"#;
        let turns = detect_and_parse(json).unwrap();
        assert_eq!(turns.len(), 1);
    }

    #[test]
    fn detect_claude_md_format() {
        let md = "## Human\nHello\n## Assistant\nHi\n";
        let turns = detect_and_parse(md).unwrap();
        assert_eq!(turns.len(), 2);
    }

    #[test]
    fn format_epoch_secs_known_date() {
        // 2025-01-01 00:00:00 UTC = 1735689600 seconds
        // Delegates to neuron::unix_secs_to_datetime — same Hinnant algorithm, single source of truth.
        let (y, mo, d, ..) = unix_secs_to_datetime(1_735_689_600);
        assert_eq!(format!("{y:04}-{mo:02}-{d:02}T00:00:00Z"), "2025-01-01T00:00:00Z");
    }

    #[test]
    fn mine_file_creates_verbatim_neurons() {
        use tempfile::TempDir;
        let dir = TempDir::new().unwrap();

        // Write a Claude markdown conversation
        let conv = dir.path().join("chat.md");
        std::fs::write(&conv, "## Human\nHow do I use serde?\n\n## Assistant\nImport serde and derive Serialize.\n").unwrap();

        let mut idx = crate::index::NeuronIndex::load_or_create(dir.path()).unwrap();
        let count = mine_file(&conv, dir.path(), &mut idx, Some("rust")).unwrap();
        assert_eq!(count, 2, "Should create one Verbatim neuron per turn");

        let ndir = dir.path().join(".cortyx").join("neurons");
        let verbatim: Vec<_> = std::fs::read_dir(&ndir).unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".verbatim.md"))
            .collect();
        assert_eq!(verbatim.len(), 2);
    }

    #[test]
    fn mine_file_creates_temporal_follows_synapses() {
        use tempfile::TempDir;
        let dir = TempDir::new().unwrap();
        let conv = dir.path().join("chat.md");
        std::fs::write(&conv, "## Human\nFirst message\n\n## Assistant\nSecond message\n").unwrap();

        let mut idx = crate::index::NeuronIndex::load_or_create(dir.path()).unwrap();
        mine_file(&conv, dir.path(), &mut idx, None).unwrap();

        // First neuron should have a TemporalFollows synapse to the second
        let ndir = dir.path().join(".cortyx").join("neurons");
        let mut jsons: Vec<_> = std::fs::read_dir(&ndir).unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                let n = e.file_name().to_string_lossy().to_string();
                n.ends_with(".json") && n.contains("verbatim")
            })
            .collect();
        jsons.sort_by_key(|e| e.file_name());
        assert!(!jsons.is_empty());
        let data = std::fs::read_to_string(jsons[0].path()).unwrap();
        assert!(data.contains("temporal_follows"), "First turn should have TemporalFollows synapse: {data}");
    }

    #[test]
    fn mine_text_single_turn() {
        use tempfile::TempDir;
        let dir = TempDir::new().unwrap();
        let mut idx = crate::index::NeuronIndex::load_or_create(dir.path()).unwrap();
        let count = mine_text(
            "The cache eviction policy uses LRU.",
            "inline",
            dir.path(),
            &mut idx,
            Some("cache"),
            Some("user"),
            None,
        ).unwrap();
        assert_eq!(count, 1);
    }
}
