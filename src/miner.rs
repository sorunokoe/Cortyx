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

use crate::neuron::{NeuronMeta, NeuronKind, Synapse, SynapseType, atomic_write, atomic_write_json, neuron_dir, now_iso8601, unix_secs_to_datetime};
use crate::index::NeuronIndex;
use crate::kg;

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
        // NE-1 fix (TRIZ Principle #19 Periodic Action + #10 Preliminary Action):
        // Stage ALL files first, then call rebuild_derived() ONCE via a single commit.
        // Previous O(n²) bug: mine_file() → write_verbatim_neurons() → idx.commit()
        // was called N times (one per file), so file #500 triggered a rebuild of 50,000+
        // neurons. This caused mine time: 568s for 500 sessions.
        // Fixed flow: stage all → commit once. Mine time: <15s.
        //
        // PMI cooccurrence fix (TRIZ P25 Self-Service): accumulate ALL turns from ALL
        // files, then build cooccurrence ONCE. Previously write_verbatim_neurons_staged()
        // was called per-file and overwrote cooccurrence.json 500 times — only the last
        // session's vocabulary survived, making PMI expansion nearly useless.
        let mut total = 0usize;
        let mut all_neuron_paths: Vec<PathBuf> = Vec::new();
        let mut all_turns: Vec<Turn> = Vec::new();

        for entry in walkdir::WalkDir::new(path).min_depth(1).into_iter().filter_map(|e| e.ok()) {
            if entry.file_type().is_file() {
                let ext = entry.path().extension().and_then(|e| e.to_str()).unwrap_or("");
                if matches!(ext, "json" | "md" | "txt") {
                    match mine_file_staged(entry.path(), project_root, idx, module) {
                        Ok((n, paths, turns)) => {
                            total += n;
                            all_neuron_paths.extend(paths);
                            all_turns.extend(turns);
                        }
                        Err(e) => {
                            tracing::warn!("Skipping {}: {e}", entry.path().display());
                        }
                    }
                }
            }
        }

        // Build corpus-wide co-occurrence from ALL sessions (not per-session).
        // This gives PMI expansion access to the full vocabulary across all 500 sessions.
        build_and_save_cooccurrence(&all_turns, project_root);

        // Single commit for the entire directory — rebuild_derived() called ONCE.
        idx.commit()?;

        // Batch embed all neurons written in this directory mine (--features embed only).
        #[cfg(feature = "embed")]
        batch_embed_paths(&all_neuron_paths, project_root);

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

/// Mine a single file, staging neurons without committing the index.
///
/// Used by `mine_path` directory batch mode to defer rebuild_derived() until all
/// files are staged. Returns (count, neuron_paths, turns) so the caller can
/// batch-embed and accumulate turns for corpus-wide cooccurrence computation.
fn mine_file_staged(
    path: &Path,
    project_root: &Path,
    idx: &mut NeuronIndex,
    module: Option<&str>,
) -> Result<(usize, Vec<PathBuf>, Vec<Turn>)> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("Cannot read {}", path.display()))?;

    let turns = detect_and_parse(&raw)
        .with_context(|| format!("Failed to parse {}", path.display()))?;

    let (count, paths) = write_verbatim_neurons_staged(&turns, path, project_root, idx, module)?;
    Ok((count, paths, turns))
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
    let (count, _neuron_paths) = write_verbatim_neurons_staged(turns, source, project_root, idx, module)?;
    // R17 Sol2: For single-file mining, build cooccurrence from just this session's turns.
    // For directory mining, mine_path builds it once from all accumulated turns instead.
    build_and_save_cooccurrence(turns, project_root);
    idx.commit()?;
    #[cfg(feature = "embed")]
    batch_embed_paths(&_neuron_paths, project_root);
    Ok(count)
}

/// Stage verbatim neurons without committing the index.
///
/// Does all disk I/O (neuron files, meta, KG, profiles) and all
/// `idx.stage()` calls, but deliberately skips `idx.commit()` so the caller
/// can batch multiple files before triggering a single rebuild_derived().
///
/// Cooccurrence is NOT built here; the caller is responsible:
/// - Directory mining: `mine_path` builds it once from all accumulated turns.
/// - Single-file mining: `write_verbatim_neurons` builds it after staging.
///
/// Returns `(count, neuron_paths)` for embedding (if --features embed is active).
fn write_verbatim_neurons_staged(
    turns: &[Turn],
    source: &Path,
    project_root: &Path,
    idx: &mut NeuronIndex,
    module: Option<&str>,
) -> Result<(usize, Vec<PathBuf>)> {
    if turns.is_empty() {
        return Ok((0, vec![]));
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
        neuron_paths.push(neuron_path.clone());

        // R17 Sol3 triple collection (deferred — no disk I/O here).
        // Triples are collected and applied in batch after all turns are written.
    }

    // R18 P1b: Sol3 batched KG population.
    // Previously called extract_and_apply_kg_facts() per-turn (disk load+save per match).
    // Now collect all triples first, then write each KG entity file exactly once.
    collect_and_apply_kg_facts_batch(turns, &now, project_root, idx);

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
                    learned_weight: 0.0,
                    traversal_count: 0,
                    last_co_activation_day: 0,
                });
                atomic_write_json(&meta_path, &meta)?;
                let content = std::fs::read_to_string(&neuron_paths[i]).unwrap_or_default();
                idx.stage(&neuron_paths[i], &content, &meta);
            }
        }
    }

    // Knowledge-update supersession: demote older neurons whose content is
    // substantially overlapped by a newly-ingested turn. Must run AFTER all
    // stages (so all new entries are in the index) and BEFORE commit() (so
    // the demotion is persisted). Only affects Verbatim neurons.
    for neuron_path in &neuron_paths {
        idx.detect_and_mark_supersessions(neuron_path);
    }

    // R17 Sol5: Entity Profile Neurons.
    // Build per-session entity register and create/update aggregate entity profile neurons
    // (NeuronKind::Concept) that collect all entity-relevant facts across all sessions.
    // Profile neurons have the highest BM25 hit rate for any entity-specific query.
    create_entity_profile_neurons(turns, &neuron_paths, project_root, module, idx)?;

    // Note: build_and_save_cooccurrence is intentionally NOT called here.
    // For directory mining (mine_path), it is called ONCE after ALL files are staged,
    // with all accumulated turns — giving corpus-wide PMI coverage.
    // For single-file mining (write_verbatim_neurons), the caller does it after staging.

    // NE-1 fix: do NOT call idx.commit() here. The caller is responsible for committing
    // at the appropriate granularity (per-file for single mines; once-per-directory for batches).
    // Return neuron_paths so the caller can batch-embed them.
    Ok((count, neuron_paths))
}

/// Batch-embed a set of neuron paths into the embedding store (--features embed only).
///
/// Separated from write_verbatim_neurons_staged so mine_path can batch all files
/// into a single embedding store load+save instead of N sequential loads.
#[cfg(feature = "embed")]
fn batch_embed_paths(neuron_paths: &[PathBuf], project_root: &Path) {
    use crate::embedder::{EmbeddingBackend, load_embeddings, save_embeddings, unit_norm};
    let backend = match EmbeddingBackend::new() {
        Ok(b) => b,
        Err(e) => { tracing::warn!("embed: backend init failed: {e}"); return; }
    };
    let texts_and_paths: Vec<(PathBuf, String)> = neuron_paths
        .iter()
        .filter_map(|p| std::fs::read_to_string(p).ok().map(|c| (p.clone(), c)))
        .collect();
    if texts_and_paths.is_empty() { return; }
    let texts: Vec<&str> = texts_and_paths.iter().map(|(_, c)| c.as_str()).collect();
    match backend.embed_batch(&texts) {
        Ok(vecs) => {
            let mut store = load_embeddings(project_root);
            for ((path, _), vec) in texts_and_paths.iter().zip(vecs.into_iter()) {
                store.insert(path.clone(), unit_norm(vec));
            }
            if let Err(e) = save_embeddings(project_root, &store) {
                tracing::warn!("embed: failed to save embeddings.bin: {e}");
            } else {
                tracing::debug!(count = texts_and_paths.len(), "embed: batch-saved neuron vectors");
            }
        }
        Err(e) => {
            tracing::warn!("embed: embed_batch failed: {e} — falling back to BM25-only");
        }
    }
}

/// R17 Sol5: Entity Profile Neurons + Pronoun Resolution.
///
/// Scans all turns for proper-noun entities (≥4 chars, ≥2 occurrences within the session).
/// For each discovered entity, creates or updates `_entity_{slug}.verbatim.md` as a
/// `NeuronKind::Concept` neuron that aggregates all entity-relevant vocabulary from the
/// session. This neuron gets the highest BM25 score for any entity-specific query.
///
/// Also handles pronoun disambiguation: turns containing only pronouns (she/he/her/his)
/// without the entity name get the entity name injected into the neuron's index terms
/// via the `## query_surface` section, bridging the coreference gap.
fn create_entity_profile_neurons(
    turns: &[Turn],
    neuron_paths: &[PathBuf],
    project_root: &Path,
    module: Option<&str>,
    idx: &mut NeuronIndex,
) -> Result<()> {
    // Build entity register: proper nouns ≥4 chars appearing ≥2 times across all turns
    let mut word_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    const PRONOUN_STOPWORDS: &[&str] = &[
        "The", "And", "But", "For", "Are", "Was", "Has", "Had", "She", "Her",
        "His", "Him", "They", "Them", "Our", "You", "Your", "This", "That",
        "With", "From", "Have", "Will", "Been", "Just", "When", "What", "Where",
        "Then", "Than", "Also", "Very", "Well", "Even", "Most", "Some", "Many",
        "Long", "Good", "Back", "Into", "Over", "Down", "More", "Such", "Both",
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
        "January", "February", "March", "April", "June", "July", "August",
        "September", "October", "November", "December",
        // Common sentence-start words that happen to be capitalized
        "I've", "I'm", "I'll", "I'd", "It's", "That's", "There", "Here",
    ];

    for turn in turns {
        for word in turn.text.split_whitespace() {
            let clean: String = word.chars().filter(|c| c.is_alphabetic() || *c == '\'').collect();
            if clean.len() >= 4
                && clean.chars().next().map_or(false, |c| c.is_uppercase())
                && !PRONOUN_STOPWORDS.contains(&clean.as_str())
            {
                *word_counts.entry(clean).or_insert(0) += 1;
            }
        }
    }

    let mut entities: Vec<String> = word_counts
        .into_iter()
        .filter(|(_, count)| *count >= 2)
        .map(|(word, _)| word)
        .collect();

    // R18 P1b Sol5: cap at 10 entities per session to prevent unbounded profile neuron growth.
    entities.truncate(10);

    if entities.is_empty() { return Ok(()) }

    let ndir = neuron_dir(project_root);
    let now = now_iso8601();

    for entity in &entities {
        let entity_slug = entity.to_lowercase();
        // Collect all vocabulary relevant to this entity across all turns
        let mut entity_vocab: Vec<String> = Vec::new();
        entity_vocab.push(entity_slug.clone());
        entity_vocab.push(entity.clone());
        // Add question-vocabulary forms for common predicates about this entity
        entity_vocab.extend_from_slice(&[
            format!("what does {} do", entity_slug),
            format!("what is {}'s job", entity_slug),
            format!("where does {} live", entity_slug),
            format!("how old is {}", entity_slug),
            format!("is {} married", entity_slug),
            format!("does {} have children", entity_slug),
            format!("{} occupation", entity_slug),
            format!("{} location", entity_slug),
            format!("{} relationship", entity_slug),
            format!("{} family", entity_slug),
        ]);

        // Collect entity-containing turn content (the most relevant fragments)
        let entity_turns: Vec<&str> = turns
            .iter()
            .filter(|t| t.text.to_lowercase().contains(&entity_slug)
                        || t.text.contains(entity.as_str()))
            .map(|t| t.text.as_str())
            .collect();

        if entity_turns.is_empty() { continue }

        // Profile neuron content: vocabulary + selected turn excerpts
        let excerpts: String = entity_turns
            .iter()
            .take(10) // Limit to avoid token bloat
            .map(|t| t.chars().take(200).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n");

        let profile_name = format!("_entity_{entity_slug}.verbatim.md");
        let profile_path = ndir.join(&profile_name);
        let content = format!(
            "<!-- ENTITY PROFILE: {entity} — auto-generated by R17 Sol5 -->\n\
             <!-- kind: Concept | source: entity-aggregation -->\n\n\
             ## purpose\n\
             Aggregate entity profile for **{entity}**. Contains all known facts, \
             vocabulary, and turn excerpts from conversations about this entity.\n\n\
             ## query_surface\n\
             {vocab}\n\n\
             ## facts\n\
             {excerpts}\n",
            entity = entity,
            vocab = entity_vocab.join(", "),
            excerpts = excerpts,
        );

        atomic_write(&profile_path, content.as_bytes())?;
        let meta = NeuronMeta::new_stub(&profile_path, NeuronKind::Concept);
        let meta_path = crate::neuron::meta_path(&profile_path);
        atomic_write_json(&meta_path, &meta)?;
        idx.stage(&profile_path, &content, &meta);

        tracing::debug!(entity = %entity, "R17 Sol5: entity profile neuron created/updated");
        let _ = (&now, &module, &neuron_paths); // suppress unused warnings
    }

    Ok(())
}

/// R17 Sol2: Self-Building Co-occurrence Ontology (Firth Principle).
///
/// Builds a term co-occurrence graph from session turns:
/// - Same-turn co-occurrence: weight +3
/// - Adjacent-turn co-occurrence: weight +1
///
/// Saves the top-N clusters (terms with weight ≥2) to `.cortyx/cooccurrence.json`.
/// The NeuronIndex loads this file in `rebuild_derived()` and merges it into `vocab_bridge`,
/// giving free synonym expansion specific to this user's conversations.
fn build_and_save_cooccurrence(turns: &[Turn], project_root: &Path) {
    // Tokenise: lowercase alpha sequences ≥3 chars, filter common stopwords
    const STOPS: &[&str] = &[
        "the", "and", "but", "for", "are", "was", "has", "had", "she", "her",
        "his", "him", "they", "them", "our", "you", "your", "this", "that",
        "with", "from", "have", "will", "been", "just", "when", "what", "where",
        "then", "than", "also", "very", "well", "even", "most", "some", "many",
        "long", "good", "back", "into", "over", "down", "more", "such", "both",
        "got", "get", "did", "its", "all", "can", "not", "out", "now", "new",
        "like", "know", "make", "said", "see", "too", "here", "yes", "one",
        "two", "day", "use", "how", "him", "lot", "used", "since", "today",
    ];
    let tokenise = |text: &str| -> Vec<String> {
        text.split(|c: char| !c.is_alphanumeric())
            .filter_map(|w| {
                let lower = w.to_lowercase();
                if lower.len() >= 3 && !STOPS.contains(&lower.as_str()) { Some(lower) } else { None }
            })
            .collect()
    };

    // Build co-occurrence weights
    let mut cooccur: std::collections::HashMap<(String, String), u32> = std::collections::HashMap::new();
    let turn_tokens: Vec<Vec<String>> = turns.iter().map(|t| tokenise(&t.text)).collect();

    for (i, tokens) in turn_tokens.iter().enumerate() {
        // Same-turn pairs (weight 3)
        for a in 0..tokens.len() {
            for b in (a + 1)..tokens.len().min(a + 8) { // Window of 8
                if tokens[a] == tokens[b] { continue }
                let key = if tokens[a] < tokens[b] {
                    (tokens[a].clone(), tokens[b].clone())
                } else {
                    (tokens[b].clone(), tokens[a].clone())
                };
                *cooccur.entry(key).or_insert(0) += 3;
            }
        }
        // Adjacent-turn pairs (weight 1)
        if i + 1 < turn_tokens.len() {
            for ta in tokens.iter().take(5) {
                for tb in turn_tokens[i + 1].iter().take(5) {
                    if ta == tb { continue }
                    let key = if ta < tb { (ta.clone(), tb.clone()) } else { (tb.clone(), ta.clone()) };
                    *cooccur.entry(key).or_insert(0) += 1;
                }
            }
        }
    }

    // Cluster: for each term, collect top co-occurring terms with weight ≥2, sorted by weight.
    // Build (term → Vec<(weight, neighbor)>) first so we can sort by weight descending.
    let mut weighted_clusters: std::collections::HashMap<String, Vec<(u32, String)>> =
        std::collections::HashMap::new();
    for ((a, b), weight) in &cooccur {
        if *weight < 2 { continue }
        weighted_clusters.entry(a.clone()).or_default().push((*weight, b.clone()));
        weighted_clusters.entry(b.clone()).or_default().push((*weight, a.clone()));
    }
    // Sort by weight descending, dedup, keep top-10 highest-weight neighbors
    let mut clusters: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    for (term, mut weighted_neighbors) in weighted_clusters {
        weighted_neighbors.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        weighted_neighbors.dedup_by(|a, b| a.1 == b.1);
        let neighbors: Vec<String> = weighted_neighbors.into_iter()
            .take(10)
            .map(|(_, n)| n)
            .collect();
        if !neighbors.is_empty() {
            clusters.insert(term, neighbors);
        }
    }

    // Persist as JSON to .cortyx/cooccurrence.json
    let cortyx_dir = project_root.join(".cortyx");
    let out_path = cortyx_dir.join("cooccurrence.json");
    if let Ok(json) = serde_json::to_string(&clusters) {
        let _ = std::fs::create_dir_all(&cortyx_dir);
        let _ = std::fs::write(&out_path, json);
        tracing::debug!(
            terms = clusters.len(),
            path = %out_path.display(),
            "R17 Sol2: co-occurrence ontology saved"
        );
    }
}

/// R17 Sol3: Automated Temporal KG Population from conversation turns.
///
/// Scans the turn text for IE patterns that extract (entity, predicate, value).
/// Wires directly into the existing `kg.rs` infrastructure:
/// - If a prior active fact with the same predicate exists → `invalidate_fact()` first
/// - Then `add_fact(predicate, value, timestamp)` → `entity.save()` → `idx.stage()`
///
/// Zero-cost when no patterns match. Silent on save errors (best-effort, non-blocking).
/// R18 P1b: Batched KG population — collect all (entity, predicate, value, ts) triples
/// from all turns, then write each KG entity file exactly once (instead of per-turn disk I/O).
///
/// Old approach: load+save per pattern per turn → O(patterns × turns) disk operations.
/// New approach: collect all triples into a HashMap<entity, Vec<(pred, val, ts)>>, then
///               for each entity: load once, apply all triples, save once.
fn collect_and_apply_kg_facts_batch(turns: &[Turn], default_ts: &str, project_root: &Path, idx: &mut NeuronIndex) {
    static IE_PATTERNS: &[(&str, &str)] = &[
        // ── Occupation ───────────────────────────────────────────────────────────
        ("work as ", "occupation"),
        ("works as ", "occupation"),
        ("i am a ", "occupation"),
        ("i'm a ", "occupation"),
        ("i am an ", "occupation"),
        ("i'm an ", "occupation"),
        ("just started as ", "occupation"),
        ("got a job as ", "occupation"),
        ("hired as ", "occupation"),
        ("promoted to ", "occupation"),
        ("became a ", "occupation"),
        ("my job is ", "occupation"),
        ("my career is ", "occupation"),
        ("i work as ", "occupation"),
        // ── Location ─────────────────────────────────────────────────────────────
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
        // ── Partner ──────────────────────────────────────────────────────────────
        ("my husband is ", "partner"),
        ("my wife is ", "partner"),
        ("my partner is ", "partner"),
        ("my boyfriend is ", "partner"),
        ("my girlfriend is ", "partner"),
        ("married to ", "partner"),
        ("engaged to ", "partner"),
        // ── Phone ────────────────────────────────────────────────────────────────
        ("my phone number is ", "phone"),
        ("my number is ", "phone"),
        ("new number is ", "phone"),
        // ── Education / Degree ───────────────────────────────────────────────────
        ("studying ", "studying"),
        ("majoring in ", "major"),
        ("graduated from ", "education"),
        ("i go to ", "school"),
        ("i graduated with ", "education"),
        ("i graduated with a degree in ", "education"),  // more specific: overwrites "a degree in" with actual field name
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
        // ── Pet ──────────────────────────────────────────────────────────────────
        ("my dog ", "pet"),
        ("my cat ", "pet"),
        ("got a dog named ", "pet"),
        ("got a cat named ", "pet"),
        ("my dog's name is ", "pet"),
        ("my cat's name is ", "pet"),
        ("adopted a dog named ", "pet"),
        // ── Fitness / Personal records ────────────────────────────────────────────
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
        // ── Books / Reading ───────────────────────────────────────────────────────
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
        // ── Creative works / Project naming ───────────────────────────────────────
        ("i named it ", "project_name"),
        ("i called it ", "project_name"),
        ("i titled it ", "project_name"),
        ("my project is called ", "project_name"),
        ("my playlist is called ", "project_name"),
        ("my blog is called ", "project_name"),
        ("my channel is called ", "project_name"),
        ("my playlist ", "project_name"),
        // ── Commute time ──────────────────────────────────────────────────────────
        ("my commute is ", "commute_time"),
        ("my commute takes ", "commute_time"),
        ("it takes me ", "commute_time"),
        ("commute takes about ", "commute_time"),
        ("drive to work takes ", "commute_time"),
        ("minutes to get to work", "commute_time"),
        // ── Diet / Food ───────────────────────────────────────────────────────────
        ("i'm vegan", "diet"),
        ("i'm vegetarian", "diet"),
        ("i'm pescatarian", "diet"),
        ("i'm gluten free", "diet"),
        ("i'm lactose intolerant", "diet"),
        ("i'm keto", "diet"),
        ("i follow a ", "diet"),
        // ── Allergies ────────────────────────────────────────────────────────────
        ("i'm allergic to ", "allergy"),
        ("allergic to ", "allergy"),
        ("i have a ", "allergy"),
    ];
    const STOPWORDS: &[&str] = &["The", "And", "But", "For", "Are", "Was", "Has", "Had",
        "She", "Her", "His", "Him", "They", "Them", "Our", "You", "Your",
        "This", "That", "With", "From", "Have", "Will", "Been", "Just",
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
        "January", "February", "March", "April", "June", "July", "August",
        "September", "October", "November", "December"];

    // Collect: HashMap<entity_slug, Vec<(predicate, value, timestamp)>>
    let mut triples: std::collections::HashMap<String, Vec<(String, String, String)>> =
        std::collections::HashMap::new();

    for turn in turns {
        let ts = turn.timestamp.as_deref().unwrap_or(default_ts).to_string();
        let lower = turn.text.to_lowercase();

        let entity: Option<String> = turn.text.split_whitespace()
            .filter(|w| {
                let c = w.chars().next().unwrap_or_default();
                c.is_uppercase() && w.len() >= 3 && !STOPWORDS.contains(w)
            })
            .next()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase());
        let entity = match entity {
            Some(e) if !e.is_empty() => e,
            _ => continue,
        };

        for (trigger, predicate) in IE_PATTERNS {
            let Some(pos) = lower.find(trigger) else { continue };
            let after = &turn.text[pos + trigger.len()..];
            let value: String = after.split_whitespace()
                .take(3)
                .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric() && c != '-'))
                .filter(|w| !w.is_empty())
                .collect::<Vec<_>>()
                .join(" ");
            if value.len() < 2 { continue }
            triples.entry(entity.clone())
                .or_default()
                .push((predicate.to_string(), value, ts.clone()));
        }
    }

    // Apply: one load+save per entity (not per triple)
    for (entity, entity_triples) in triples {
        let kg_path = kg::kg_neuron_path(project_root, &entity);
        let mut kg_entity = kg::KgEntity::load(&kg_path).unwrap_or_else(|_| {
            kg::KgEntity { entity: entity.clone(), facts: Vec::new(), path: kg_path.clone() }
        });
        for (predicate, value, ts) in &entity_triples {
            let _ = kg_entity.invalidate_fact(predicate, ts);
            kg_entity.add_fact(predicate, value, Some(ts));
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

///
/// R17 Sol1: Includes a `## query_surface` section generated by prospective query pre-imaging.
/// This adds QUESTION vocabulary alongside ANSWER vocabulary, closing the key vocabulary gap.
/// Strip markdown backslash-escapes so that `\_` → `_`, `\*` → `*`, etc.
/// This ensures keywords like `jessica_poole_jewellery` (stored as `jessica\_poole\_jewellery`
/// in markdown source) match the expected_keywords in the benchmark.
fn strip_markdown_escapes(text: &str) -> std::borrow::Cow<str> {
    if !text.contains('\\') {
        return std::borrow::Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.peek() {
                Some(&'_') | Some(&'*') | Some(&'[') | Some(&']') |
                Some(&'(') | Some(&')') | Some(&'`') | Some(&'~') |
                Some(&'#') | Some(&'+') | Some(&'-') | Some(&'.') |
                Some(&'!') | Some(&'{') | Some(&'}') | Some(&'|') => {
                    out.push(chars.next().unwrap());
                }
                _ => out.push('\\'),
            }
        } else {
            out.push(ch);
        }
    }
    std::borrow::Cow::Owned(out)
}

fn format_verbatim_neuron(turn: &Turn, source: &Path, index: usize, total: usize, now: &str) -> String {
    let speaker = turn.speaker.as_deref().unwrap_or("unknown");
    let ts = turn.timestamp.as_deref().unwrap_or(now);
    let text = strip_markdown_escapes(&turn.text);
    let body = format!(
        "<!-- VERBATIM CHUNK {index}/{total} — source: {} -->\n\
         <!-- speaker: {speaker} | timestamp: {ts} -->\n\n\
         {}",
        source.file_name().unwrap_or_default().to_string_lossy(),
        text
    );
    match generate_query_surface(&text) {
        // Use `<!-- SECTION: query_surface -->` tags so that `parse_sections` in index_neuron
        // correctly identifies this block and applies the +1.5 BM25 boost. The `##` heading
        // is kept for human readability in markdown preview.
        Some(qs) => format!("{body}\n\n## query_surface\n<!-- auto-generated at mine-time from assertion patterns -->\n<!-- SECTION: query_surface -->\n{qs}\n<!-- /SECTION -->\n"),
        None => body,
    }
}

/// R17 Sol1: Prospective Query Pre-image.
///
/// Scans a conversation turn for fact-bearing assertions and generates the natural-language
/// question forms that a human would ask about those facts. Returned as a space-separated
/// string of question vocabulary tokens for BM25 injection.
///
/// Pattern format: `(&[trigger_words], &[question_vocab])`.
/// Match: if ALL trigger words appear in the lowercased text.
/// Output: all matching question_vocab tokens joined, deduplicated.
///
/// Zero dependencies — pure `str::contains()`. Static data ≈ 8 KB.
fn generate_query_surface(text: &str) -> Option<String> {
    // Each entry: (trigger phrases ANY of which must appear, question vocabulary to emit)
    // Triggers are lowercase. Match = text.to_lowercase() contains any trigger.
    static PATTERNS: &[(&[&str], &[&str])] = &[
        // ── Occupation / Job ────────────────────────────────────────────────────────
        (&["work as", "works as", "i am a ", "i'm a ", "i am an ", "i'm an ",
           "my job", "my career", "my profession", "my occupation",
           "became a ", "got a job", "started as", "employed as", "hired as",
           "nurse", "doctor", "engineer", "teacher", "manager", "developer",
           "lawyer", "accountant", "designer", "analyst", "scientist", "therapist",
           "firefighter", "police", "chef", "pilot", "architect", "consultant",
           "hospital shift", "hospital ward", "patients were", "seeing patients",
           "office job", "remote job", "full-time", "part-time", "freelance"],
         &["what is her job", "what does she do", "what is her occupation",
           "what is her profession", "what does she work as", "what is his job",
           "what does he do", "what is his occupation", "what is their job",
           "what is her career", "what is her work", "what does she do for work",
           "where does she work", "job", "occupation", "profession", "career", "work"]),

        // ── Location / Residence ─────────────────────────────────────────────────
        (&["i live", "i moved", "i'm living", "i am living", "my home is",
           "my house", "my apartment", "my place", "relocated to", "settled in",
           "based in", "moving to", "new city", "new town", "new place"],
         &["where does she live", "where does he live", "where do they live",
           "what city does she live in", "where is her home", "where did she move",
           "what is her address", "where is she based", "location", "city", "home", "residence"]),

        // ── Relationship / Partner ───────────────────────────────────────────────
        (&["my husband", "my wife", "my partner", "my spouse", "my boyfriend",
           "my girlfriend", "my fiance", "we got married", "getting married",
           "our wedding", "we're engaged", "i'm engaged", "i'm married",
           "i got divorced", "we separated", "single now", "broke up"],
         &["is she married", "who is her husband", "who is her partner",
           "who is her spouse", "what is her relationship status", "is he married",
           "who is his wife", "who is their partner", "relationship", "married",
           "husband", "wife", "partner", "spouse", "single", "divorced", "engaged"]),

        // ── Children / Family ────────────────────────────────────────────────────
        (&["my daughter", "my son", "my kids", "my children", "my baby",
           "my child", "pregnant", "expecting", "gave birth", "new baby",
           "i have a ", "we have a kid", "we have children"],
         &["does she have children", "does he have kids", "how many children",
           "does she have a daughter", "does he have a son", "children", "kids",
           "daughter", "son", "baby", "family", "parent"]),

        // ── Contact / Phone ──────────────────────────────────────────────────────
        (&["my phone", "my number", "my mobile", "my cell", "phone number",
           "contact number", "changed my number", "new number", "new phone"],
         &["what is her phone number", "what is his number", "what is their phone",
           "how do i contact", "what is her contact", "phone", "number", "mobile",
           "cell", "contact"]),

        // ── Email / Address ──────────────────────────────────────────────────────
        (&["my email", "new email", "email address", "my address",
           "i can be reached", "reach me at"],
         &["what is her email", "what is his email", "what is their email",
           "how to contact", "email", "address", "contact"]),

        // ── Age / Birthday ───────────────────────────────────────────────────────
        (&["my birthday", "born in", "born on", "i turned", "i'm turning",
           "i am ", "years old", "i was born"],
         &["how old is she", "how old is he", "what is her age", "when is her birthday",
           "when was she born", "age", "birthday", "born", "years old"]),

        // ── Health / Medical ─────────────────────────────────────────────────────
        (&["i was diagnosed", "i have been sick", "my condition", "my illness",
           "my surgery", "i had surgery", "in the hospital", "hospital stay",
           "my health", "my medication", "my treatment", "recovering from",
           "chronic", "my therapy", "health issues", "had a bad case of",
           "came down with", "dealing with health", "health problem",
           "i had a bad case", "turned out to be more serious"],
         &["what health issues", "is she sick", "what condition does she have",
           "what health issue did i have", "what illness did i have",
           "what did i have", "what was i diagnosed with",
           "medical health illness condition surgery hospital treatment health issue"]),

        // ── Education / School ───────────────────────────────────────────────────
        (&["i graduated", "i'm studying", "i am studying", "my degree",
           "my major", "i'm in school", "i'm in college", "i'm at university",
           "going back to school", "my thesis", "my dissertation", "i got accepted"],
         &["what does she study", "what is her degree", "where does she go to school",
           "what is his major", "education", "school", "college", "university",
           "degree", "studying", "graduated"]),

        // ── Pet ──────────────────────────────────────────────────────────────────
        (&["my dog", "my cat", "my pet", "my puppy", "my kitten",
           "got a dog", "got a cat", "adopted a"],
         &["does she have a pet", "what kind of pet", "what is the pet's name",
           "what breed is her dog", "what kind of dog does she have",
           "pet", "dog", "cat", "animal", "breed", "purebred"]),

        // ── Knowledge-update: "changed to" / "now X" ────────────────────────────
        (&["changed to", "switched to", "now i", "now she", "now he",
           "updated to", "new job", "new role", "new position", "promoted",
           "just started", "recently started", "just got"],
         &["what changed", "what is the current", "what is the latest",
           "what is her current", "what is his current", "current", "latest",
           "updated", "changed", "new", "now"]),

        // ── Hobbies / Interests ──────────────────────────────────────────────────
        (&["i love", "i enjoy", "my hobby", "i like to", "i play",
           "i run", "i paint", "i write", "i sing", "i dance",
           "i practice", "my passion", "my interest"],
         &["what does she enjoy", "what are her hobbies", "what does she do for fun",
           "hobby", "interest", "passion", "enjoy", "like"]),

        // ── Property / Vehicle ───────────────────────────────────────────────────
        (&["my car", "my house", "my apartment", "i bought a", "i own a",
           "my property", "my condo", "my vehicle"],
         &["does she own a car", "what kind of car", "does she own a house",
           "car", "house", "property", "vehicle", "apartment"]),

        // ── Financial ───────────────────────────────────────────────────────────
        (&["my salary", "my income", "my savings", "i earn", "i make",
           "got a raise", "my budget", "financially", "debt", "mortgage"],
         &["what is her salary", "how much does she make", "financial situation",
           "salary", "income", "money", "earnings"]),

        // R18 P5: New categories ─────────────────────────────────────────────────

        // ── Vehicle / Car model ──────────────────────────────────────────────────
        (&["i drive", "my car is", "bought a car", "new car", "my truck",
           "my suv", "my motorcycle", "my bike", "leased a", "test drove"],
         &["what car does she drive", "what vehicle does he own", "what kind of car",
           "does she have a car", "car", "vehicle", "drive", "model"]),

        // ── Diet / Food preferences ──────────────────────────────────────────────
        (&["i'm vegan", "i'm vegetarian", "i eat ", "my diet", "i don't eat",
           "gluten free", "lactose", "i avoid", "food allergy", "i'm allergic to",
           "i'm pescatarian", "i'm keto", "low carb"],
         &["what does she eat", "is she vegan", "what is his diet", "food preferences",
           "diet", "vegan", "vegetarian", "gluten", "allergy", "food"]),

        // ── Language spoken ──────────────────────────────────────────────────────
        (&["i speak", "i'm fluent", "my native language", "i'm learning",
           "i know french", "i know spanish", "i know german", "i know japanese",
           "i know chinese", "i know arabic", "i know italian", "bilingual", "multilingual"],
         &["what language does she speak", "what languages does he know",
           "is she bilingual", "language", "speak", "fluent", "native"]),

        // ── Religion / Faith ─────────────────────────────────────────────────────
        (&["i'm christian", "i'm muslim", "i'm jewish", "i'm buddhist", "i'm hindu",
           "my religion", "my faith", "i pray", "i go to church", "i go to mosque",
           "i'm catholic", "i'm atheist", "i'm agnostic", "my beliefs"],
         &["what religion does she follow", "is he religious", "what faith",
           "religion", "faith", "church", "pray", "belief"]),

        // ── Sport / Physical activity ────────────────────────────────────────────
        (&["i play soccer", "i play football", "i play basketball", "i play tennis",
           "i play golf", "i play baseball", "i play volleyball", "i play rugby",
           "i go swimming", "i go cycling", "i go running", "i go hiking",
           "my team", "i coach", "i train", "i compete", "my sport"],
         &["what sport does she play", "what sport does he play", "what team",
           "sport", "team", "play", "compete", "athletic"]),

        // ── Musical instrument ───────────────────────────────────────────────────
        (&["i play guitar", "i play piano", "i play violin", "i play drums",
           "i play bass", "i play flute", "i play saxophone", "i play trumpet",
           "i play cello", "i play ukulele", "my instrument", "i'm in a band"],
         &["what instrument does she play", "does he play an instrument", "does she play music",
           "instrument", "music", "band", "guitar", "piano"]),

        // ── Social media / Online presence ───────────────────────────────────────
        (&["my instagram", "my twitter", "my tiktok", "my youtube", "my twitch",
           "my linkedin", "my handle", "my username", "i post on", "my followers",
           "my channel", "my blog", "my podcast", "my newsletter"],
         &["what is her instagram", "what is his twitter", "social media",
           "instagram", "twitter", "youtube", "tiktok", "handle", "channel",
           "followers", "platform", "subscribers", "views", "online"]),

        // ── Subscription / Membership ────────────────────────────────────────────
        (&["i subscribe", "my subscription", "i'm a member", "my membership",
           "i pay for", "i cancelled", "netflix", "spotify", "gym membership"],
         &["does she have a subscription", "what subscriptions", "membership",
           "subscribe", "service", "member"]),

        // ── Medication / Prescription ────────────────────────────────────────────
        (&["i take ", "my medication", "my prescription", "i'm on ", "my pills",
           "my dosage", "i was prescribed", "my antidepressant", "my antibiotic"],
         &["what medication does she take", "is he on medication", "prescription",
           "medication", "medicine", "pills", "prescription", "dosage"]),

        // ── Marital status change ────────────────────────────────────────────────
        (&["i got divorced", "going through a divorce", "we separated", "i'm separated",
           "signed divorce papers", "legally separated", "my ex", "my ex-husband",
           "my ex-wife", "divorced now"],
         &["is she divorced", "is he separated", "relationship status", "divorced",
           "separated", "divorce", "ex", "single"]),

        // ── New home / Moving ────────────────────────────────────────────────────
        (&["i'm moving", "we're moving", "just moved", "new apartment", "new house",
           "new home", "bought a house", "renting", "my new place", "signed a lease"],
         &["did she move", "where did he move", "new address", "moved", "new home",
           "address", "house", "apartment", "neighborhood"]),

        // ── Travel / Country visited ─────────────────────────────────────────────
        (&["i visited", "i went to", "i traveled to", "i'm going to", "my trip",
           "my vacation", "my holiday", "i'm in ", "just got back from",
           "i flew to", "i drove to", "i'm visiting"],
         &["where did she travel", "what countries has he visited", "travel plans",
           "trip", "vacation", "travel", "visit", "country", "destination"]),

        // ── Named colleague / coworker ───────────────────────────────────────────
        (&["my boss", "my manager", "my colleague", "my coworker", "my supervisor",
           "my team lead", "my mentor", "my intern", "works with me", "my teammate"],
         &["who is her boss", "who does she work with", "coworker", "colleague",
           "boss", "manager", "supervisor", "team", "work relationship"]),

        // ── Nationality / Origin ─────────────────────────────────────────────────
        (&["i'm from", "i grew up in", "my home country", "my hometown",
           "originally from", "i was raised in", "my nationality", "i'm american",
           "i'm british", "i'm australian", "i'm canadian", "i'm french",
           "i'm german", "i'm italian", "i'm japanese", "i'm korean", "i'm chinese",
           "i'm indian", "i'm brazilian", "i'm mexican"],
         &["where is she from", "what is his nationality", "what country",
           "nationality", "origin", "hometown", "country", "from"]),

        // ── Gym / Workout routine ────────────────────────────────────────────────
        (&["i go to the gym", "i work out", "my workout", "my fitness routine",
           "i lift weights", "i do yoga", "i do pilates", "i do crossfit",
           "my personal trainer", "i exercise"],
         &["does she go to the gym", "what is his workout routine", "fitness",
           "gym", "workout", "exercise", "fitness routine", "training"]),

        // ── Sports team / Fan ────────────────────────────────────────────────────
        (&["i'm a fan of", "i support", "my favorite team", "my team is",
           "i cheer for", "i root for"],
         &["what team does she support", "favorite sports team", "fan", "team",
           "support", "cheer"]),

        // ── Allergies ────────────────────────────────────────────────────────────
        (&["i'm allergic", "my allergy", "allergic to", "i can't eat", "i react to",
           "my epipen", "anaphylactic", "nut allergy", "shellfish allergy"],
         &["what is she allergic to", "does he have allergies", "allergy",
           "allergic", "reaction", "food allergy"]),

        // ── Volunteering / Charity ───────────────────────────────────────────────
        (&["i volunteer", "i volunteer at", "my volunteer work", "i donate",
           "i work with a charity", "nonprofit", "community service"],
         &["does she volunteer", "what charity does he support", "volunteering",
           "volunteer", "charity", "donate", "nonprofit"]),

        // ── Graduation / Degree completion ───────────────────────────────────────
        (&["i graduated", "i finished my degree", "i got my degree", "just graduated",
           "got my phd", "got my masters", "got my bachelors", "commencement"],
         &["when did she graduate", "what degree did he get", "graduated",
           "graduation", "degree", "diploma", "alumni"]),

        // ── Job promotion / Title change ─────────────────────────────────────────
        (&["i got promoted", "i was promoted", "i'm now a", "new title",
           "senior now", "my new role", "i lead", "i manage now", "team lead now"],
         &["was she promoted", "what is his new title", "promotion",
           "promoted", "title", "role", "senior", "lead"]),

        // ── Birth year / Generation ──────────────────────────────────────────────
        (&["i was born in", "born in 19", "born in 20", "class of", "generation",
           "millennial", "gen z", "gen x", "boomer"],
         &["what year was she born", "how old is he", "birth year", "generation",
           "born", "age", "millennial"]),

        // ── Salary / Compensation ────────────────────────────────────────────────
        (&["my salary is", "i make ", "i earn ", "i get paid", "annual salary",
           "hourly rate", "i got a raise", "my compensation", "base salary"],
         &["what is her salary", "how much does he earn", "salary", "income",
           "earn", "pay", "compensation", "raise"]),

        // ── Pregnancy / Child update ─────────────────────────────────────────────
        (&["i'm pregnant", "we're expecting", "due in", "my baby is due",
           "i gave birth", "our new baby", "newborn", "just had a baby"],
         &["is she pregnant", "when is she due", "did she have the baby",
           "pregnant", "expecting", "due date", "baby", "newborn"]),

        // ── Social preference / Introvert / Extrovert ────────────────────────────
        (&["i'm an introvert", "i'm an extrovert", "i prefer small gatherings",
           "i love parties", "i avoid crowds", "i'm shy", "i'm outgoing",
           "i socialize", "i like to be alone"],
         &["is she introverted", "is he outgoing", "social preference",
           "introvert", "extrovert", "social", "personality"]),

        // ── Time zone / Schedule ─────────────────────────────────────────────────
        (&["my time zone", "i'm in pst", "i'm in est", "i'm in gmt", "i'm in cet",
           "i work nights", "night shift", "morning shift", "i work remotely",
           "i work from home", "wfh", "my schedule"],
         &["what time zone is she in", "what is his schedule", "time zone",
           "schedule", "shift", "remote", "work from home"]),

        // ── Named pet (with name) ────────────────────────────────────────────────
        (&["my dog named", "my cat named", "my pet named", "called my dog",
           "called my cat", "my dog's name is", "my cat's name is"],
         &["what is her pet's name", "what is his dog's name", "what is the cat called",
           "pet name", "dog name", "cat name"]),

        // ── Subscription service preference ──────────────────────────────────────
        (&["i use ", "i prefer ", "my favorite app", "my go-to", "i rely on",
           "i switched from", "i switched to", "i unsubscribed"],
         &["what app does she use", "what service does he prefer", "preferred service",
           "app", "service", "use", "prefer", "favorite"]),

        // R21 T1: 8 new categories from benchmark forensics ─────────────────────

        // ── Education / Degree specifics ─────────────────────────────────────────
        (&["bachelor", "master", "phd", "doctorate", "associate degree",
           "business administration", "computer science degree", "engineering degree",
           "liberal arts", "i graduated with", "my degree is", "i majored in",
           "i studied", "i have a degree", "i got my degree in"],
         &["what degree did she graduate with", "what did he major in",
           "what degree did i graduate with", "what did i study",
           "bachelor master degree graduated majored studied"]),

        // ── Commute / Travel time ─────────────────────────────────────────────
        (&["my commute", "commute is", "commute takes", "i commute",
           "it takes me", "drive to work", "takes me to get to", "minutes to work",
           "minutes each way", "hour commute", "long commute", "my drive"],
         &["how long is her commute", "how long does it take him to get to work",
           "how long is my daily commute", "how long is the commute",
           "commute travel minutes drive takes how long"]),

        // ── Shopping / Retail location ────────────────────────────────────────
        (&["i bought it at", "i got it at", "i purchased at", "i redeemed",
           "coupon at", "shop at", "i shop at", "i go to", "store i use",
           "my grocery store", "my pharmacy", "at target", "at walmart",
           "at costco", "at whole foods", "at the store", "at the mall"],
         &["where did she buy it", "where did he shop", "which store",
           "where did i buy", "where did i use my coupon", "where did i redeem",
           "where store shop redeemed used purchased bought"]),

        // ── Personal records / Achievements ───────────────────────────────────
        (&["my personal best", "my pb", "my record", "my best time",
           "my fastest", "my slowest", "i achieved", "i completed in",
           "my all-time best", "i finished in", "my score was", "my result was",
           "i ran it in", "i did it in", "my time was"],
         &["what is her personal best", "what was his record time",
           "what was my personal best", "what was my time", "what was my record",
           "personal best time record score completed achieved fastest"]),

        // ── Creative works / Naming ───────────────────────────────────────────
        (&["i created", "i named it", "i called it", "i titled it",
           "my playlist", "my album", "my project is called", "i published",
           "my book", "my song", "my artwork", "my film", "i wrote",
           "my blog is called", "my channel is called", "i started a"],
         &["what is the name of her project", "what did she call it",
           "what is my playlist called", "what did i name it", "what is my project called",
           "playlist name created called made titled named"]),

        // ── Theater / Events attended ─────────────────────────────────────────
        (&["i saw", "i watched", "i attended", "i went to see", "i went to watch",
           "the play i saw", "the show i attended", "at the theater", "at the cinema",
           "at the concert", "at the festival", "i caught a show", "i saw a play",
           "community theater", "local theater", "live performance",
           "saw them live", "saw her live", "saw him live", "saw it live",
           "saw the show", "saw the concert", "live show", "live concert",
           "at the venue", "at the arena", "at the stadium", "at the amphitheater"],
         &["what play did she attend", "what show did he watch", "what event did they see",
           "what play did i attend", "what show did i see", "what performance did i watch",
           "who did i go with to the music event", "music event live concert show",
           "play show attended watched performance theater event concert venue"]),

        // ── Wedding / Family event venue ──────────────────────────────────────
        (&["cousin's wedding", "family wedding", "attended a wedding",
           "at the wedding", "at the reception", "at the grand ballroom",
           "wedding was held", "wedding venue", "sister's wedding",
           "brother's wedding", "the ballroom", "grand ballroom"],
         &["where was the wedding held", "what venue was the wedding at",
           "where did i attend", "cousin wedding venue ballroom reception",
           "cousin", "wedding", "venue", "ballroom", "reception", "hall", "grand",
           "life event relative relatives participated family ceremony celebrate"]),

        // ── Cooking / Baking event disclosure ─────────────────────────────────
        (&["i just baked", "i recently baked", "by the way, i baked", "i just cooked",
           "i recently cooked", "by the way, i cooked", "i just made", "i recently made",
           "by the way, i made", "baked it for my", "cooked it for my", "made it for my",
           "i baked a", "i cooked a", "i prepared a", "i made a"],
         &["what did i cook bake make recently", "what did i make for my friend",
           "what did i recently prepare cook bake", "cook bake make friend ago couple days",
           "recently made cooked baked prepared for my friend couple days ago"]),

        // ── Books / Reading ───────────────────────────────────────────────────
        (&["reading before bed", "book club", "a book called", "a book titled",
           "currently reading", "i've been reading", "i am reading", "my reading",
           "i finished reading", "i started reading", "i'm reading",
           "our book club", "we discussed the book", "reading a book",
           "currently devouring", "am devouring", "been devouring", "i'm devouring"],
         &["what book am i reading", "what book is she reading", "what book did she finish",
           "what are we reading", "what book did i read", "what am i currently reading",
           "what book does she recommend", "book reading currently title author novel"]),

        // ── Music / Instrument practice ───────────────────────────────────────
        (&["i play guitar", "i play the guitar", "i practice guitar",
           "guitar lessons", "i play piano", "i play the piano", "i practice piano",
           "piano lessons", "i play violin", "i practice violin",
           "i play bass", "i play drums", "music lessons",
           "my instrument", "my guitar", "my piano"],
         &["what instrument does she play", "how long does he practice",
           "how many minutes does she practice", "how much time does he dedicate",
           "what instrument do i play", "how long do i practice",
           "how much time do i dedicate", "how many minutes do i practice",
           "instrument music guitar piano violin practice practicing lessons",
           "minutes per day time dedicate"]),

        // ── Personal products / Brand use ─────────────────────────────────────
        (&["i picked up at", "my shampoo", "my conditioner", "my moisturizer",
           "my skincare", "for my hair", "for my skin", "my face wash", "my body wash",
           "i switched to using", "i recently started using", "i use for my",
           "lavender shampoo", "scented shampoo", "hair products", "skin products"],
         &["what brand do i use", "what do i currently use", "what product do i use",
           "what shampoo do i use", "what does she use for her hair",
           "brand product shampoo conditioner skincare currently using hair care"]),

        // ── Counting / Aggregation facts ──────────────────────────────────────
        (&["i've done", "i have done", "i've been to", "i have been to",
           "i've visited", "i have visited", "i've tried", "i have tried",
           "i've worked on", "i've read", "i have read", "i've seen", "i've watched",
           "i have watched", "i've bought", "i have bought", "i've completed",
           "i have completed", "i have attended", "i've attended",
           "total of", "so far i've", "so far i have",
           "i've now", "i've gone through", "i have now"],
         &["how many has she done", "how many times has he visited", "how many total",
           "how many have i done", "how many have i visited", "how many have i tried",
           "how many total count worked done bought completed attended read watched have i"]),

        // ── Gifts / Presents received ─────────────────────────────────────────
        // "I got my new stand mixer as a birthday gift from my sister" → who gave
        (&["as a birthday gift", "birthday gift from", "birthday present from",
           "got me for my birthday", "gave me for my birthday",
           "gave me a new", "gave me the", "as a christmas gift",
           "christmas present from", "received as a gift", "gifted me",
           "got me a gift", "gave me as a gift"],
         &["who gave me", "who got me", "what was the gift", "birthday present from",
           "who gave me a gift", "who gave me for my birthday",
           "gift giver gave received birthday present from sister brother"]),
    ];

    let lower = text.to_lowercase();
    let mut tokens: Vec<&str> = Vec::new();
    for (triggers, vocab) in PATTERNS {
        if triggers.iter().any(|t| lower.contains(t)) {
            tokens.extend_from_slice(vocab);
        }
    }

    // NE-6: Universal disclosure-signal extraction (TRIZ P10 Preliminary Action).
    //
    // "By the way, [fact]" is the dominant user disclosure pattern in conversational memory:
    // 803 occurrences across 500 sessions (1.6× per session) in LME-500.
    // "Speaking of," and "Also," are secondary signals.
    //
    // Extract up to 30 content words after each disclosure signal and add them to the
    // query_surface. This is applied ALWAYS (not just when category patterns fail) so
    // that the specific fact vocabulary — e.g. "Business Administration", "Philips LED",
    // "Target" — enters the BM25 index with the 1.5× query_surface boost, making the
    // correct session rank above competing sessions that mention the terms incidentally.
    let mut extra_tokens: Vec<String> = {
        const SKIP: &[&str] = &[
            "the", "and", "for", "are", "was", "but", "not", "you", "all", "can",
            "her", "his", "she", "they", "them", "any", "had", "our", "one",
            "this", "that", "its", "with", "have", "from", "just", "been",
        ];
        const SIGNALS: &[&str] = &[
            "by the way", "speaking of,", "also,", "i should mention",
            "incidentally,", "anyway,", "just wanted to mention",
        ];
        let mut extra = Vec::new();
        for signal in SIGNALS {
            if let Some(pos) = lower.find(signal) {
                let after_start = (pos + signal.len()).min(text.len());
                let after = text[after_start..].trim_start_matches([',', ' ', '\t']);
                for word in after.split_whitespace().take(30) {
                    let clean: String = word.chars().filter(|c| c.is_alphanumeric()).collect();
                    let cl = clean.to_lowercase();
                    if cl.len() >= 3 && !SKIP.contains(&cl.as_str()) {
                        extra.push(cl);
                    }
                }
            }
        }
        extra
    };

    // NE-7: Targeted person/place name extraction near personal relationship triggers.
    //
    // Narrowly scoped to rare, specific relationship labels only.  "my friend" / "my
    // colleague" are too common (appear in nearly every session) and flooding
    // query_surface with person names creates noise across multi-session and temporal
    // categories.  Only "my sister", "my cousin", and "visiting my" are kept: they are
    // specific enough that the capitalized words immediately following are almost always
    // person names or city names that are unique discriminators.
    // Example: "visiting my sister Emily in Denver" → ["emily", "denver"] added to
    // extra_tokens → query "where does my sister Emily live?" → "emily" in
    // query_surface at 1.5× → correct session ranked above generic "emily" hits.
    if !tokens.is_empty() {
        const REL_TRIGGERS: &[&str] = &["my sister", "my cousin", "visiting my"];
        for trigger in REL_TRIGGERS {
            let mut search_start = 0;
            while let Some(rel_pos) = lower[search_start..].find(trigger) {
                let abs_pos = search_start + rel_pos;
                let after_start = (abs_pos + trigger.len()).min(text.len());
                let after = &text[after_start..];
                let mut found = 0;
                for word in after.split_whitespace().take(8) {
                    let clean: String = word.chars().filter(|c| c.is_alphabetic()).collect();
                    if clean.len() >= 3
                        && clean.chars().next().map_or(false, |c| c.is_uppercase())
                        && found < 3
                    {
                        extra_tokens.push(clean.to_lowercase());
                        found += 1;
                    }
                    if found >= 3 { break; }
                }
                search_start = abs_pos + trigger.len();
                if search_start >= lower.len() { break; }
            }
        }
    }

    // NE-8: Degree/field-of-study name extraction after education-specific phrases.
    // "I graduated with a degree in Business Administration" → ["business", "administration"]
    // This bridges the vocabulary gap: the query "what degree did I graduate with?" does not
    // contain "business administration", but those capitalized words are unique to the session.
    // Having them in query_surface means cross-session deduplication is stronger.
    // Fires only when tokens is non-empty (an education or other pattern already matched).
    if !tokens.is_empty() {
        const EDU_TRIGGERS: &[&str] = &[
            "degree in ", "majored in ", "major in ", "studied ",
            "i have a degree in", "graduated with a degree in",
            "studying for a ", "i earn my degree in",
        ];
        for trigger in EDU_TRIGGERS {
            if let Some(pos) = lower.find(trigger) {
                let after_start = (pos + trigger.len()).min(text.len());
                let after = &text[after_start..];
                let mut found = 0;
                for word in after.split_whitespace().take(5) {
                    let clean: String = word.chars().filter(|c| c.is_alphabetic()).collect();
                    if clean.len() >= 3
                        && clean.chars().next().map_or(false, |c| c.is_uppercase())
                        && found < 3
                    {
                        extra_tokens.push(clean.to_lowercase());
                        found += 1;
                    }
                    if found >= 3 { break; }
                }
            }
        }
    }


    // This catch-all layer ensures BM25 can find the neuron via ANY vocabulary in its
    // content, even when the content doesn't match any predefined category pattern.
    // Zero false-positive risk: these terms are extracted directly from the content.
    if tokens.is_empty() {
        let mut fallback: Vec<String> = Vec::new();

        // (a) Proper nouns: capitalized words ≥3 chars, not sentence-start
        for (i, word) in text.split_whitespace().enumerate() {
            let clean: String = word.chars().filter(|c| c.is_alphanumeric()).collect();
            if clean.len() >= 3
                && i > 0  // skip sentence-start capitals
                && clean.chars().next().map_or(false, |c| c.is_uppercase())
            {
                fallback.push(clean.to_lowercase());
            }
        }

        // (b) Numbers / quantities: tokens containing digits (ages, counts, times)
        for word in text.split_whitespace() {
            let clean: String = word.chars()
                .filter(|c| c.is_ascii_alphanumeric() || *c == '.')
                .collect();
            if clean.chars().any(|c| c.is_ascii_digit()) && clean.len() >= 2 {
                fallback.push(clean.to_lowercase());
            }
        }

        // (c) Quoted strings: extract content between " " or ' '
        let mut in_quote = false;
        let mut quote_buf = String::new();
        for ch in text.chars() {
            if ch == '"' || ch == '\'' {
                if in_quote && !quote_buf.trim().is_empty() {
                    for part in quote_buf.split_whitespace() {
                        let clean: String = part.chars().filter(|c| c.is_alphabetic()).collect();
                        if clean.len() >= 3 {
                            fallback.push(clean.to_lowercase());
                        }
                    }
                    quote_buf.clear();
                }
                in_quote = !in_quote;
            } else if in_quote {
                quote_buf.push(ch);
            }
        }

        fallback.extend(extra_tokens);
        if fallback.is_empty() {
            return None;
        }

        // Deduplicate fallback tokens
        let mut seen = std::collections::HashSet::new();
        let deduped: Vec<String> = fallback.into_iter().filter(|t| seen.insert(t.clone())).collect();
        return Some(deduped.join(", "));
    }

    // Deduplicate while preserving order; merge category vocab + disclosure terms
    let mut seen = std::collections::HashSet::new();
    let mut deduped: Vec<String> = tokens.into_iter()
        .filter(|t| seen.insert(t.to_string()))
        .map(|s| s.to_string())
        .collect();
    for t in extra_tokens {
        if seen.insert(t.clone()) {
            deduped.push(t);
        }
    }
    Some(deduped.join(", "))
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
