use super::*;
use std::collections::{BTreeMap, HashMap};

pub(super) fn scanned_conversation_lines(idx: &NeuronIndex) -> std::vec::IntoIter<Vec<String>> {
    let mut grouped = BTreeMap::<String, Vec<(String, Vec<String>)>>::new();
    let entries = std::fs::read_dir(neuron_dir(&idx.persistence.project_root))
        .ok()
        .into_iter()
        .flat_map(|entries| entries.flatten().collect::<Vec<_>>());
    for entry in entries {
        let path = entry.path();
        let Some(file_name) = path
            .file_name()
            .map(|value| value.to_string_lossy().to_string())
        else {
            continue;
        };
        if !file_name.contains("_conv_") || !file_name.ends_with(".md") {
            continue;
        }
        let Some(conversation_id) = file_name
            .split_once("_conv_")
            .map(|(conversation_id, _)| conversation_id.to_string())
        else {
            continue;
        };
        let Some(lines) = std::fs::read_to_string(&path)
            .ok()
            .map(|content| content.lines().map(str::to_string).collect::<Vec<_>>())
        else {
            continue;
        };
        grouped
            .entry(conversation_id)
            .or_default()
            .push((file_name, lines));
    }
    grouped
        .into_values()
        .map(|mut chunks| {
            chunks.sort_by(|left, right| left.0.cmp(&right.0));
            chunks
                .into_iter()
                .flat_map(|(_, lines)| lines)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
        .into_iter()
}

pub(super) fn session_score(session_rank: usize, fact_score: usize) -> usize {
    session_rank * 100 + fact_score
}

pub(super) fn grouped_verbatim_candidate_lines(
    idx: &NeuronIndex,
) -> HashMap<String, Vec<(String, bool)>> {
    let mut grouped: HashMap<String, Vec<(String, bool)>> = HashMap::new();
    for entry in idx
        .retrieval
        .entries
        .iter()
        .filter(|entry| matches!(entry.kind, NeuronKind::Verbatim))
    {
        let Ok(content) = std::fs::read_to_string(&entry.neuron_path) else {
            continue;
        };
        let group_key = if entry.session_id.is_empty() {
            entry.neuron_path.to_string_lossy().to_string()
        } else {
            entry.session_id.clone()
        };
        let is_summary = is_session_summary_path(&entry.neuron_path);
        let lines = grouped.entry(group_key).or_default();
        for raw_line in strip_query_surface_section(&content).lines() {
            let line = raw_line.trim();
            if !line.is_empty() && is_session_answer_candidate_line(line) {
                lines.push((line.to_string(), is_summary));
            }
        }
    }
    grouped
}
