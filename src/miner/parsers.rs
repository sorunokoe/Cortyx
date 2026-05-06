use std::collections::HashSet;

use crate::error::Result;
use serde::Deserialize;

use crate::neuron::unix_secs_to_datetime;

use super::{surface, Turn};

// ─── LongMemEval ─────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub(super) struct LmeSession {
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

pub(super) fn parse_longmemeval(raw: &str) -> Result<Vec<Turn>> {
    if let Ok(sessions) = serde_json::from_str::<Vec<LmeSession>>(raw) {
        let turns = sessions
            .into_iter()
            .flat_map(|s| {
                s.session_history.into_iter().map(move |t| Turn {
                    speaker: Some(t.role.clone()),
                    text: t.content.clone(),
                    timestamp: t.timestamp.clone(),
                })
            })
            .filter(|t| !t.text.trim().is_empty())
            .collect();
        return Ok(turns);
    }
    if let Ok(session) = serde_json::from_str::<LmeSession>(raw) {
        let turns = session
            .session_history
            .into_iter()
            .map(|t| Turn {
                speaker: Some(t.role.clone()),
                text: t.content.clone(),
                timestamp: t.timestamp.clone(),
            })
            .filter(|t| !t.text.trim().is_empty())
            .collect();
        return Ok(turns);
    }
    crate::cortyx_bail!("Not LongMemEval format")
}

// ─── ChatGPT conversations.json ───────────────────────────────────────────────

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

pub(super) fn parse_chatgpt(raw: &str) -> Result<Vec<Turn>> {
    let conversations: Vec<ChatGptExport> = if raw.trim_start().starts_with('[') {
        serde_json::from_str(raw)?
    } else {
        vec![serde_json::from_str(raw)?]
    };

    let mut turns = Vec::new();
    for conv in conversations {
        let root = conv.mapping.values().find(|n| n.parent.is_none());
        let mut queue: std::collections::VecDeque<&str> = root
            .map(|r| std::iter::once(r.id.as_str()).collect())
            .unwrap_or_default();
        let mut visited: HashSet<&str> = queue.iter().copied().collect();
        while let Some(id) = queue.pop_front() {
            if let Some(node) = conv.mapping.get(id) {
                if let Some(msg) = &node.message {
                    let role = msg.author.role.clone();
                    if role == "user" || role == "assistant" {
                        let text: String = msg
                            .content
                            .parts
                            .iter()
                            .filter_map(|p| p.as_str().map(|s| s.to_string()))
                            .collect::<Vec<_>>()
                            .join("\n");
                        if !text.trim().is_empty() {
                            let ts = msg.create_time.map(|t| {
                                let (y, mo, d, ..) = unix_secs_to_datetime(t as u64);
                                format!("{y:04}-{mo:02}-{d:02}T00:00:00Z")
                            });
                            turns.push(Turn {
                                speaker: Some(role),
                                text,
                                timestamp: ts,
                            });
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

// ─── Claude markdown export ───────────────────────────────────────────────────

pub(super) fn parse_claude_md(raw: &str) -> Result<Vec<Turn>> {
    let mut turns = Vec::new();
    let mut current_speaker: Option<String> = None;
    let mut current_lines: Vec<&str> = Vec::new();

    for line in raw.lines() {
        if line.starts_with("## ") {
            let heading = line.trim_start_matches("## ").trim().to_lowercase();
            if matches!(
                heading.as_str(),
                "human" | "assistant" | "user" | "ai" | "system"
            ) {
                if !current_lines.is_empty() {
                    let text = current_lines.join("\n").trim().to_string();
                    if !text.is_empty() {
                        turns.push(Turn {
                            speaker: current_speaker.clone(),
                            text,
                            timestamp: None,
                        });
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
            turns.push(Turn {
                speaker: current_speaker.clone(),
                text,
                timestamp: None,
            });
        }
    }
    Ok(turns)
}

// ─── Generic markdown ─────────────────────────────────────────────────────────

pub(super) fn parse_generic_md(raw: &str) -> Result<Vec<Turn>> {
    if raw
        .lines()
        .any(|line| line.starts_with("## ") || line.starts_with("### "))
    {
        let turns = parse_headed_md(raw, &["## ", "### "])?;
        if !turns.is_empty() {
            return Ok(turns);
        }
    }
    let turns = parse_dialog_md(raw);
    if !turns.is_empty() {
        return Ok(turns);
    }
    Ok(vec![Turn {
        speaker: None,
        text: raw.to_string(),
        timestamp: None,
    }])
}

/// Splits speaker-labelled dialog into size-capped chunks for BM25 length calibration.
///
/// TRIZ Principle 1 (Segmentation): per-turn chunks so BM25 length normalisation
/// operates at turn granularity rather than penalising large sessions.
fn parse_dialog_md(raw: &str) -> Vec<Turn> {
    const DIALOG_CHUNK_MAX_CHARS: usize = 10_000;

    let mut raw_turns: Vec<(Option<String>, String)> = Vec::new();
    let mut current_speaker: Option<String> = None;
    let mut current_lines: Vec<&str> = Vec::new();

    for line in raw.lines() {
        let trimmed = line.trim();
        if surface::parse_embedded_session_timestamp(trimmed).is_some() {
            let text = current_lines.join("\n").trim().to_string();
            if !text.is_empty() {
                raw_turns.push((current_speaker.clone(), text));
            }
            current_lines.clear();
            current_speaker = None;
            current_lines.push(trimmed);
            continue;
        }
        if let Some((speaker, rest)) = surface::parse_embedded_dialogue_line(trimmed) {
            let text = current_lines.join("\n").trim().to_string();
            if !text.is_empty() {
                raw_turns.push((current_speaker.clone(), text));
            }
            current_lines.clear();
            current_speaker = Some(surface::normalize_dialogue_speaker_label(speaker));
            if !rest.is_empty() {
                current_lines.push(rest);
            }
        } else {
            current_lines.push(line);
        }
    }
    let text = current_lines.join("\n").trim().to_string();
    if !text.is_empty() {
        raw_turns.push((current_speaker, text));
    }

    if raw_turns.is_empty() {
        return vec![];
    }

    let mut chunks: Vec<Turn> = Vec::new();
    let mut chunk_parts: Vec<String> = Vec::new();
    let mut chunk_len: usize = 0;

    for (speaker, text) in &raw_turns {
        let part = speaker
            .as_deref()
            .map(|label| format!("{label}: {text}"))
            .unwrap_or_else(|| text.clone());
        let part_len = part.len();

        if chunk_len + part_len > DIALOG_CHUNK_MAX_CHARS && chunk_len > 0 {
            chunks.push(Turn {
                speaker: None,
                text: chunk_parts.join("\n\n"),
                timestamp: None,
            });
            chunk_parts.clear();
            chunk_len = 0;
        }

        chunk_parts.push(part);
        chunk_len += part_len;
    }
    if !chunk_parts.is_empty() {
        chunks.push(Turn {
            speaker: None,
            text: chunk_parts.join("\n\n"),
            timestamp: None,
        });
    }

    chunks
}

fn parse_headed_md(raw: &str, markers: &[&str]) -> Result<Vec<Turn>> {
    let mut turns = Vec::new();
    let mut current_speaker: Option<String> = None;
    let mut current_lines: Vec<&str> = Vec::new();

    for line in raw.lines() {
        if let Some(marker) = markers.iter().find(|&&m| line.starts_with(m)) {
            if !current_lines.is_empty() {
                let text = current_lines.join("\n").trim().to_string();
                if !text.is_empty() {
                    turns.push(Turn {
                        speaker: current_speaker.clone(),
                        text,
                        timestamp: None,
                    });
                }
                current_lines.clear();
            }
            let role = line.trim_start_matches(marker).trim().to_lowercase();
            current_speaker = if role.is_empty() { None } else { Some(role) };
        } else {
            current_lines.push(line);
        }
    }
    if !current_lines.is_empty() {
        let text = current_lines.join("\n").trim().to_string();
        if !text.is_empty() {
            turns.push(Turn {
                speaker: current_speaker.clone(),
                text,
                timestamp: None,
            });
        }
    }
    Ok(turns)
}

// ─── Generic JSON ─────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct GenericTurn {
    #[serde(alias = "role", alias = "speaker", alias = "author", default)]
    speaker: Option<String>,
    #[serde(alias = "text", alias = "message", alias = "content")]
    content: String,
    #[serde(alias = "ts", alias = "time", default)]
    timestamp: Option<String>,
}

pub(super) fn parse_generic_json(raw: &str) -> Result<Vec<Turn>> {
    let items: Vec<GenericTurn> = serde_json::from_str(raw)?;
    Ok(items
        .into_iter()
        .filter(|t| !t.content.trim().is_empty())
        .map(|t| Turn {
            speaker: t.speaker,
            text: t.content,
            timestamp: t.timestamp,
        })
        .collect())
}
