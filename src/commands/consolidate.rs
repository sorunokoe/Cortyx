use crate::index::NeuronIndex;
use crate::miner;
use crate::neuron::{now_iso8601, unix_secs_to_datetime, NeuronKind};
use anyhow::Context;
use std::path::{Path, PathBuf};

#[derive(Debug)]
struct ConsolidationCandidate {
    path: PathBuf,
    module: String,
    promoted_module: String,
    use_count: u32,
    speaker: Option<String>,
    timestamp: String,
    content: String,
    source_hint: String,
}

/// # Errors
///
/// Returns an error if a qualifying diary entry cannot be read or if mining a
/// promoted consolidation entry fails.
pub fn consolidate_diary(
    idx: &mut NeuronIndex,
    min_refs: u32,
    dry_run: bool,
    project_root: &Path,
) -> anyhow::Result<String> {
    let mut candidates = idx
        .list_neurons(None)
        .into_iter()
        .filter(|summary| summary.kind == NeuronKind::Verbatim && summary.use_count >= min_refs)
        .filter_map(|summary| {
            let metadata = idx.context_metadata_for(&summary.path)?;
            let module = metadata.module?;
            if !is_diary_module(&module) {
                return None;
            }
            Some((summary, module, metadata.timestamp_secs))
        })
        .map(|(summary, module, timestamp_secs)| {
            let content = std::fs::read_to_string(&summary.path).with_context(|| {
                format!("failed to read diary entry {}", summary.path.display())
            })?;
            Ok(ConsolidationCandidate {
                source_hint: promoted_source_hint(&summary.path),
                promoted_module: format!("consolidated/{}", module.trim_start_matches('@')),
                speaker: module.strip_prefix("@agent/").map(ToOwned::to_owned),
                timestamp: iso8601_from_secs(timestamp_secs),
                path: summary.path,
                module,
                use_count: summary.use_count,
                content,
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    if candidates.is_empty() {
        return Ok(format!(
            "No diary entries met consolidation threshold (min_refs={min_refs})."
        ));
    }

    candidates.sort_by(|left, right| {
        right
            .use_count
            .cmp(&left.use_count)
            .then_with(|| left.path.cmp(&right.path))
    });

    if !dry_run {
        for candidate in &candidates {
            miner::mine_text(
                &candidate.content,
                &candidate.source_hint,
                project_root,
                idx,
                Some(&candidate.promoted_module),
                candidate.speaker.as_deref(),
                Some(&candidate.timestamp),
            )?;
        }
    }

    let lines = candidates
        .iter()
        .map(|candidate| {
            format!(
                "  - {} (use_count: {}, target: {})",
                candidate.module, candidate.use_count, candidate.promoted_module
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let noun = if candidates.len() == 1 {
        "entry"
    } else {
        "entries"
    };
    let prefix = if dry_run {
        format!("DRY RUN: Would promote {} diary {noun}:", candidates.len())
    } else {
        format!("Consolidated {} diary {noun}:", candidates.len())
    };

    Ok(format!("{prefix}\n{lines}"))
}

fn is_diary_module(module: &str) -> bool {
    module.starts_with("@agent/") || module.starts_with("@session/")
}

fn promoted_source_hint(path: &Path) -> String {
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .unwrap_or("diary");
    format!("consolidated_{stem}")
}

fn iso8601_from_secs(timestamp_secs: Option<i64>) -> String {
    let Some(secs) = timestamp_secs.and_then(|value| u64::try_from(value).ok()) else {
        return now_iso8601();
    };
    let (year, month, day, hour, minute, second) = unix_secs_to_datetime(secs);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}
