//! Neuron metadata I/O: loading, saving, and syncing structured diary entries to KG.

use super::super::*;
use crate::error::Result;
use crate::{cortyx_bail, cortyx_err};

/// Load existing metadata or create a fresh stub.
pub fn load_or_new_meta(meta_file: &Path, source: &Path, kind: NeuronKind) -> NeuronMeta {
    if let Ok(data) = std::fs::read_to_string(meta_file) {
        if let Ok(meta) = serde_json::from_str::<NeuronMeta>(&data) {
            return meta;
        }
    }
    NeuronMeta::new_stub(source, kind)
}

/// Serialize and write metadata to disk atomically.
pub fn save_meta(meta_file: &Path, meta: &NeuronMeta) -> Result<()> {
    Ok(atomic_write_json(meta_file, meta)?)
}

pub fn refresh_meta_after_content_write(meta: &mut NeuronMeta, content: &str) {
    if let Some(source_hash) = hash_file(&meta.source_path) {
        meta.source_hash = source_hash;
    }
    meta.tokens = estimate_context_tokens(content).get();
    meta.last_updated = now_iso8601();
    meta.status = NeuronStatus::Fresh;
    meta.synapses = parse_synapses_from_content(content);
}

pub fn record_mutation_provenance(
    neuron_path: &Path,
    meta: &NeuronMeta,
    content: &str,
    operation: ProvenanceOperation,
    source: ProvenanceSource,
    section: Option<String>,
    summary: Option<String>,
) -> Result<()> {
    Ok(record_content_provenance_edit(
        neuron_path,
        meta,
        content,
        ProvenanceEdit {
            operation,
            source,
            section,
            summary,
            ..Default::default()
        },
    )
    .map(|_| ())?)
}

pub fn finalize_mutation_message(message: String, provenance_result: Result<()>) -> String {
    match provenance_result {
        Ok(()) => message,
        Err(err) => format!("{message}\nWARNING: Failed to record provenance: {err}"),
    }
}

pub fn resolve_neuron_store_path(raw_path: &str, project_root: &Path) -> Result<PathBuf> {
    let neuron_root = neuron_dir(project_root)
        .canonicalize()
        .map_err(|err| cortyx_err!("cannot access neuron directory: {err}"))?;
    let candidate = Path::new(raw_path);
    let resolved = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        neuron_root.join(candidate)
    };
    let canonical = resolved
        .canonicalize()
        .map_err(|err| cortyx_err!("cannot access neuron path: {err}"))?;
    if !canonical.starts_with(&neuron_root) {
        cortyx_bail!(
            "path {} is outside neuron directory {}",
            canonical.display(),
            neuron_root.display()
        );
    }
    if canonical.is_dir() {
        cortyx_bail!(
            "path {} is a directory, not a neuron file",
            canonical.display()
        );
    }
    Ok(canonical)
}

pub fn build_augmented_task(index: &NeuronIndex, input: &GetContextsInput) -> String {
    let mut extra = String::new();

    if let Some(ref open_files) = input.open_files {
        if !open_files.is_empty() {
            let soft = index.soft_terms_for_editor_context(open_files, 8);
            if !soft.is_empty() {
                extra.push(' ');
                extra.push_str(&soft.join(" "));
                tracing::debug!(
                    files = open_files.len(),
                    soft_terms = soft.len(),
                    "S-V: editor context injected"
                );
            }
        }
    }

    if let Some(ref err_ctx) = input.error_context {
        if !err_ctx.is_empty() {
            let err_terms = tokenize(err_ctx);
            if !err_terms.is_empty() {
                extra.push(' ');
                extra.push_str(&err_terms.join(" "));
                tracing::debug!(err_terms = err_terms.len(), "S-V: error_context injected");
            }
        }
    }

    if extra.is_empty() {
        input.task.clone()
    } else {
        format!("{}{}", input.task, extra)
    }
}

pub fn index_kg_entity_path(index: &mut NeuronIndex, path: &Path) -> Result<()> {
    let content = std::fs::read_to_string(path)
        .map_err(|err| cortyx_err!("reload KG entity {}: {err}", path.display()))?;
    let mut meta = NeuronMeta::new_stub(path, NeuronKind::Concept);
    meta.module = Some("@kg".to_string());
    meta.tokens = estimate_context_tokens(&content).get();
    index.index_neuron(path, &content, &meta);
    Ok(())
}

pub fn sync_structured_diary_to_kg(
    project_root: &Path,
    index: &mut NeuronIndex,
    agent: &str,
    entry: &StructuredDiaryEntry,
    effective_from: &str,
) -> Result<()> {
    let path = kg::kg_neuron_path(project_root, &agent_entity_name(agent));
    let mut entity = kg::KgEntity::load(&path)?;
    entity.replace_active_fact(
        AGENT_FOCUS_PREDICATE,
        entry.title.as_deref().unwrap_or(""),
        effective_from,
    );
    entity.replace_active_fact(
        AGENT_STATUS_PREDICATE,
        entry.status.as_deref().unwrap_or(""),
        effective_from,
    );
    entity.replace_active_fact(
        AGENT_GOAL_PREDICATE,
        entry.goal.as_deref().unwrap_or(""),
        effective_from,
    );
    entity.replace_active_fact(
        AGENT_NEXT_STEP_PREDICATE,
        entry.next_step.as_deref().unwrap_or(""),
        effective_from,
    );
    entity.replace_active_fact(
        AGENT_BLOCKER_PREDICATE,
        entry.blocker.as_deref().unwrap_or(""),
        effective_from,
    );
    entity.replace_active_fact(
        AGENT_OUTCOME_PREDICATE,
        entry.outcome.as_deref().unwrap_or(""),
        effective_from,
    );
    entity.replace_active_fact(
        AGENT_ACTION_PREDICATE,
        entry.action.as_deref().unwrap_or(""),
        effective_from,
    );
    entity.sync_active_values(
        AGENT_RELATED_ENTITY_PREDICATE,
        &entry.entities,
        effective_from,
    );
    entity.sync_active_values(
        AGENT_DEPENDS_ON_PREDICATE,
        &entry.depends_on,
        effective_from,
    );
    entity.save()?;
    index_kg_entity_path(index, &path)
}
