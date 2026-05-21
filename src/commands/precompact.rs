//! Snapshot activated session context before compaction.
//!
//! Used by the Claude Code PreCompact hook to preserve high-value context in a
//! diary-style session snapshot before the conversation window is compacted.

use anyhow::Result;
use std::path::Path;
use std::time::UNIX_EPOCH;

use crate::agent_memory;
use crate::index::NeuronIndex;
use crate::miner;
use crate::neuron;

const RECENT_WINDOW_SECS: i64 = 2 * 60 * 60;
const SNAPSHOT_LIMIT: usize = 10;

#[derive(Debug, Clone)]
struct SnapshotNeuron {
    label: String,
    module: Option<String>,
    summary: String,
    use_count: u32,
    hit_rate: f32,
    activity_secs: i64,
}

/// Snapshot the current session's activated neurons before compaction.
pub fn snapshot_precompact(idx: &NeuronIndex, project_root: &Path) -> Result<String> {
    let cutoff = crate::commands::timeline::now_unix_secs().saturating_sub(RECENT_WINDOW_SECS);
    let mut neurons: Vec<SnapshotNeuron> = idx
        .list_neurons(None)
        .into_iter()
        .filter_map(|summary| {
            let metadata = idx.context_metadata_for(&summary.path)?;
            if metadata.module.as_deref() == Some("@session/precompact")
                || is_kg_path(&summary.path)
            {
                return None;
            }
            let activity_secs =
                snapshot_activity_secs(&summary.path, metadata.timestamp_secs).unwrap_or(0);
            if activity_secs < cutoff && summary.use_count == 0 {
                return None;
            }
            Some(SnapshotNeuron {
                label: display_label(project_root, &summary.path),
                module: metadata.module,
                summary: compact_summary(&metadata.summary),
                use_count: summary.use_count,
                hit_rate: metadata.hit_rate,
                activity_secs,
            })
        })
        .collect();

    neurons.sort_by(|left, right| {
        right
            .use_count
            .cmp(&left.use_count)
            .then_with(|| right.activity_secs.cmp(&left.activity_secs))
            .then_with(|| left.label.cmp(&right.label))
    });
    neurons.truncate(SNAPSHOT_LIMIT);

    let labels: Vec<String> = neurons.iter().map(|neuron| neuron.label.clone()).collect();
    let content = render_snapshot_entry(&neurons);
    let timestamp = neuron::now_iso8601();
    let mut write_idx = NeuronIndex::load_or_create(project_root)?;
    miner::mine_text(
        &content,
        "precompact-snapshot",
        project_root,
        &mut write_idx,
        Some("@session/precompact"),
        Some("precompact"),
        Some(timestamp.as_str()),
    )?;

    Ok(format!(
        "Snapshotted {} neurons: [{}]",
        labels.len(),
        labels.join(", ")
    ))
}

fn render_snapshot_entry(neurons: &[SnapshotNeuron]) -> String {
    let outcome = if neurons.is_empty() {
        "No recently active neurons met the snapshot threshold.".to_string()
    } else {
        neurons
            .iter()
            .map(|neuron| {
                let module_suffix = neuron
                    .module
                    .as_deref()
                    .map(|module| format!(", module: `{module}`"))
                    .unwrap_or_default();
                let summary_suffix = if neuron.summary.is_empty() {
                    String::new()
                } else {
                    format!("\n  summary: {}", neuron.summary)
                };
                format!(
                    "- `{}` — uses: {}, hit rate: {:.0}%, last activity: {}{}{}",
                    neuron.label,
                    neuron.use_count,
                    neuron.hit_rate * 100.0,
                    format_timestamp_secs(neuron.activity_secs),
                    module_suffix,
                    summary_suffix,
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    let entities: Vec<String> = neurons.iter().map(|neuron| neuron.label.clone()).collect();
    agent_memory::render_structured_diary_entry(
        "precompact",
        "Captured the highest-signal neurons before Claude Code compaction, prioritizing prior activations and fresh sidecar activity.",
        Some("Session snapshot before compaction"),
        Some("captured"),
        Some("Preserve useful local context across compaction."),
        Some("Rehydrate the listed neurons if the task still depends on them after compaction."),
        None,
        Some(outcome.as_str()),
        &entities,
        &[],
    )
}

fn display_label(project_root: &Path, path: &Path) -> String {
    path.strip_prefix(project_root)
        .unwrap_or(path)
        .display()
        .to_string()
}

fn compact_summary(summary: &str) -> String {
    let clean = summary.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut chars = clean.chars();
    let truncated: String = chars.by_ref().take(160).collect();
    if chars.next().is_some() {
        format!("{truncated}…")
    } else {
        truncated
    }
}

fn snapshot_activity_secs(path: &Path, timestamp_secs: Option<i64>) -> Option<i64> {
    [
        timestamp_secs,
        file_timestamp_secs(&neuron::meta_path(path)),
        file_timestamp_secs(path),
    ]
    .into_iter()
    .flatten()
    .max()
}

fn file_timestamp_secs(path: &Path) -> Option<i64> {
    std::fs::metadata(path)
        .ok()?
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| i64::try_from(duration.as_secs()).ok())
}

fn format_timestamp_secs(timestamp_secs: i64) -> String {
    let Ok(secs) = u64::try_from(timestamp_secs) else {
        return "unknown".to_string();
    };
    let (year, month, day, hour, minute, second) = neuron::unix_secs_to_datetime(secs);
    format!("{year:04}-{month:02}-{day:02} {hour:02}:{minute:02}:{second:02} UTC")
}

fn is_kg_path(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.starts_with("_kg_") && name.ends_with(".context.md"))
        .unwrap_or(false)
}
