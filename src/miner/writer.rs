use std::path::{Path, PathBuf};

use crate::error::Result;

use crate::neuron::{
    atomic_write, atomic_write_json, meta_path, neuron_dir, now_iso8601, NeuronMeta, Synapse,
    SynapseType,
};

use super::{cooccurrence, evidence, kg_apply, summary, surface, AnswerSurfaceRow, Turn};

/// Write a sequence of turns as Verbatim neurons with TemporalFollows synapses.
///
/// # Errors
///
/// Returns an error if the underlying operation fails.
pub fn write_verbatim_neurons(
    turns: &[Turn],
    source: &Path,
    project_root: &Path,
    idx: &mut crate::index::NeuronIndex,
    module: Option<&str>,
) -> Result<usize> {
    let (count, _neuron_paths) =
        write_verbatim_neurons_staged(turns, source, project_root, idx, module)?;
    // For single-file mining: build cooccurrence from this session's turns.
    // For directory mining: mine_path builds it once from all accumulated turns.
    cooccurrence::build_and_save_cooccurrence(turns, project_root);
    idx.commit()?;
    #[cfg(feature = "embed")]
    batch_embed_paths(&_neuron_paths, project_root);
    Ok(count)
}

/// Stage verbatim neurons without committing the index.
///
/// Skips `idx.commit()` so the caller can batch multiple files before a single rebuild_derived().
/// Cooccurrence is NOT built here; the caller handles it.
/// Returns `(count, neuron_paths)` for optional batch embedding.
pub(super) fn write_verbatim_neurons_staged(
    turns: &[Turn],
    source: &Path,
    project_root: &Path,
    idx: &mut crate::index::NeuronIndex,
    module: Option<&str>,
) -> Result<(usize, Vec<PathBuf>)> {
    if turns.is_empty() {
        return Ok((0, vec![]));
    }

    let ndir = neuron_dir(project_root);
    std::fs::create_dir_all(&ndir)?;

    let base = source
        .file_stem()
        .map(|s| {
            s.to_string_lossy()
                .replace(|c: char| !c.is_alphanumeric() && c != '_', "_")
        })
        .unwrap_or_else(|| "chat".to_string());
    let base = base.trim_matches('_').to_string();

    let now = now_iso8601();
    let session_bridge_rows = surface::generate_session_bridge_surface_rows(turns);

    // Pre-compute all neuron paths (deterministic formula) so TemporalFollows synapses
    // can be injected inline during the write phase — no second pass over disk needed (S2).
    let turn_neuron_paths: Vec<PathBuf> = turns
        .iter()
        .enumerate()
        .map(|(i, turn)| {
            let speaker_slug = turn
                .speaker
                .as_deref()
                .map(|s| s.replace(' ', "_"))
                .unwrap_or_else(|| "chunk".to_string());
            ndir.join(format!("{base}_{i:04}_{speaker_slug}.verbatim.md"))
        })
        .collect();

    // Parallel write phase (S5): each turn's file I/O is independent.
    // idx.stage() requires &mut — it runs sequentially after this phase.
    use rayon::prelude::*;

    struct TurnWriteResult {
        neuron_path: PathBuf,
        content: String,
        meta: NeuronMeta,
    }

    let turn_results: Vec<TurnWriteResult> = turns
        .par_iter()
        .enumerate()
        .map(|(i, turn)| -> Result<TurnWriteResult> {
            let neuron_path = turn_neuron_paths[i].clone();

            let mut dialogue_rows = surface::generate_dialogue_answer_surface_rows(turns, i);
            dialogue_rows.extend(surface::generate_cross_chunk_dialogue_answer_surface_rows(
                turns, i,
            ));
            dialogue_rows.extend(surface::generate_temporal_turn_answer_surface_rows(turn));
            if i == 0 {
                dialogue_rows.extend(session_bridge_rows.clone());
            }
            let content =
                format_verbatim_neuron(turn, source, i, turns.len(), &now, &dialogue_rows);
            atomic_write(&neuron_path, content.as_bytes())?;

            let mut meta = NeuronMeta::new_verbatim_chunk(
                &neuron_path,
                turn.speaker.clone(),
                &content,
                turn.timestamp.clone().or_else(|| Some(now.clone())),
                module.map(|s| s.to_string()),
            );
            // Inject TemporalFollows inline — no re-read/re-write pass needed (S2).
            if i + 1 < turns.len() {
                meta.synapses.push(Synapse {
                    target: turn_neuron_paths[i + 1].clone(),
                    edge_type: SynapseType::TemporalFollows,
                    weight: crate::types::SynapseWeight::new(0.6),
                    reason: "consecutive turn".to_string(),
                    learned_weight: crate::types::SynapseWeight::ZERO,
                    traversal_count: 0,
                    last_co_activation_day: 0,
                });
            }
            atomic_write_json(&meta_path(&neuron_path), &meta)?;

            Ok(TurnWriteResult {
                neuron_path,
                content,
                meta,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    // Sequential staging: idx.stage() requires &mut NeuronIndex.
    let mut all_neuron_paths: Vec<PathBuf> = Vec::new();
    for result in &turn_results {
        idx.stage(&result.neuron_path, &result.content, &result.meta);
        all_neuron_paths.push(result.neuron_path.clone());
    }

    kg_apply::collect_and_apply_kg_facts_batch(turns, &now, project_root, idx);

    if let Some(content) = format_fact_summary_neuron(turns, source) {
        let summary_path = ndir.join(format!("{base}_summary.md"));
        atomic_write(&summary_path, content.as_bytes())?;
        let summary_ts = turns
            .iter()
            .rev()
            .find_map(|turn| turn.timestamp.clone())
            .or_else(|| Some(now.clone()));
        let summary_meta = NeuronMeta::new_verbatim_chunk(
            &summary_path,
            Some("summary".to_string()),
            &content,
            summary_ts,
            module.map(|s| s.to_string()),
        );
        let summary_meta_path = meta_path(&summary_path);
        atomic_write_json(&summary_meta_path, &summary_meta)?;
        idx.stage(&summary_path, &content, &summary_meta);
        all_neuron_paths.push(summary_path);
    }

    for result in &turn_results {
        idx.detect_and_mark_supersessions(&result.neuron_path);
    }

    Ok((turn_results.len(), all_neuron_paths))
}

#[cfg(feature = "embed")]
pub(super) fn batch_embed_paths(neuron_paths: &[PathBuf], project_root: &Path) {
    use crate::embedder::{load_embeddings, save_embeddings, unit_norm, EmbeddingBackend};
    let backend = match EmbeddingBackend::new() {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!("embed: backend init failed: {e}");
            return;
        },
    };
    let texts_and_paths: Vec<(PathBuf, String)> = neuron_paths
        .iter()
        .filter_map(|p| std::fs::read_to_string(p).ok().map(|c| (p.clone(), c)))
        .collect();
    if texts_and_paths.is_empty() {
        return;
    }
    let texts: Vec<&str> = texts_and_paths.iter().map(|(_, c)| c.as_str()).collect();
    match backend.embed_batch(&texts) {
        Ok(vecs) => {
            let mut store = load_embeddings(project_root);
            for ((path, _), vec) in texts_and_paths.iter().zip(vecs.into_iter()) {
                store.insert(path.clone(), unit_norm(vec));
            }
            if let Err(e) = save_embeddings(project_root, &store) {
                tracing::warn!("embed: failed to save embedding cache: {e}");
            } else {
                tracing::debug!(
                    count = texts_and_paths.len(),
                    "embed: batch-saved neuron vectors"
                );
            }
        },
        Err(e) => {
            tracing::warn!("embed: embed_batch failed: {e} — falling back to BM25-only");
        },
    }
}

// ─── Neuron formatters ────────────────────────────────────────────────────────

/// Strip markdown backslash-escapes so `\_` → `_`, `\*` → `*`, etc.
fn strip_markdown_escapes(text: &str) -> std::borrow::Cow<'_, str> {
    if !text.contains('\\') {
        return std::borrow::Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.peek() {
                Some(&'_') | Some(&'*') | Some(&'[') | Some(&']') | Some(&'(') | Some(&')')
                | Some(&'`') | Some(&'~') | Some(&'#') | Some(&'+') | Some(&'-') | Some(&'.')
                | Some(&'!') | Some(&'{') | Some(&'}') | Some(&'|') => {
                    if let Some(escaped) = chars.next() {
                        out.push(escaped);
                    }
                },
                _ => out.push('\\'),
            }
        } else {
            out.push(ch);
        }
    }
    std::borrow::Cow::Owned(out)
}

pub(super) fn format_verbatim_neuron(
    turn: &Turn,
    source: &Path,
    index: usize,
    total: usize,
    now: &str,
    extra_answer_rows: &[AnswerSurfaceRow],
) -> String {
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
    let mut content = body;
    if let Some(qs) = surface::generate_query_surface(&text) {
        content.push_str(
            "\n\n## query_surface\n<!-- auto-generated at mine-time from assertion patterns -->\n<!-- SECTION: query_surface -->\n",
        );
        content.push_str(&qs);
        content.push_str("\n<!-- /SECTION -->\n");
    }
    surface::append_answer_surface_section(
        &mut content,
        &text,
        extra_answer_rows,
        "mine-time extracted direct-answer spans",
    );
    let facts = evidence::extract_evidence(&content);
    evidence::append_evidence_surface_section(&mut content, &facts);
    content
}

pub(super) fn format_fact_summary_neuron(turns: &[Turn], source: &Path) -> Option<String> {
    let user_lines = summary::fact_summary_lines(turns);
    let mut assistant_lines = summary::assistant_numeric_summary_lines(turns);
    for line in summary::assistant_named_item_summary_lines(turns) {
        if !assistant_lines
            .iter()
            .any(|existing| existing.eq_ignore_ascii_case(&line))
        {
            assistant_lines.push(line);
        }
    }
    let alias_lines = surface::fact_alias_lines(&user_lines, &assistant_lines);
    if user_lines.is_empty() && assistant_lines.is_empty() {
        return None;
    }
    let extra_answer_rows = surface::generate_session_bridge_surface_rows(turns);

    let facts_block = user_lines
        .iter()
        .map(|line| format!("- {line}"))
        .collect::<Vec<_>>()
        .join("\n");
    let assistant_block = assistant_lines
        .iter()
        .map(|line| format!("- {line}"))
        .collect::<Vec<_>>()
        .join("\n");
    let mut joined_lines = user_lines.clone();
    joined_lines.extend(assistant_lines.clone());
    let joined = joined_lines.join(". ");

    let mut content = format!(
        "# Session facts: {}\n\n\
         ## purpose\n\
         Distilled user-fact summary for {}.\n\n\
         ## facts\n\
         {}\n",
        source.file_name().unwrap_or_default().to_string_lossy(),
        source.file_name().unwrap_or_default().to_string_lossy(),
        facts_block,
    );

    if !assistant_block.is_empty() {
        content.push_str("\n## assistant_evidence\n");
        content.push_str(&assistant_block);
        content.push('\n');
    }

    if !alias_lines.is_empty() {
        content.push_str(
            "\n## fact_aliases\n<!-- auto-generated from distilled fact patterns -->\n<!-- SECTION: fact_aliases -->\n",
        );
        content.push_str(&alias_lines.join("\n"));
        content.push_str("\n<!-- /SECTION -->\n");
    }

    if let Some(qs) = surface::generate_query_surface(&joined) {
        content.push_str(
            "\n## query_surface\n<!-- auto-generated from distilled user facts -->\n<!-- SECTION: query_surface -->\n",
        );
        content.push_str(&qs);
        content.push_str("\n<!-- /SECTION -->\n");
    }

    surface::append_answer_surface_section(
        &mut content,
        &joined,
        &extra_answer_rows,
        "mine-time extracted direct-answer spans from distilled facts",
    );
    let facts = evidence::extract_evidence(&content);
    evidence::append_evidence_surface_section(&mut content, &facts);

    Some(content)
}
