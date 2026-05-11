use super::*;

pub(in crate::index) fn extract_task_reference_label(task: &str) -> Option<String> {
    let trimmed = task.trim();
    let lower = trimmed.to_ascii_lowercase();
    if !lower.starts_with("as of ") {
        return None;
    }
    let question_pos = lower.find("how many ")?;
    let candidate = trimmed[6..question_pos].trim().trim_end_matches(',').trim();
    if extract_explicit_date_rank(candidate).is_some() {
        return Some(candidate.to_string());
    }
    None
}

pub(in crate::index) fn verbatim_source_group_key(entry: &BM25Entry) -> String {
    if let Ok(content) = std::fs::read_to_string(&entry.neuron_path) {
        if let Some(line) = content.lines().next() {
            if let Some(source_idx) = line.find("source:") {
                let source = &line[source_idx + "source:".len()..];
                let source = source.trim();
                let source = source.strip_suffix("-->").unwrap_or(source).trim();
                if !source.is_empty() {
                    return source.to_string();
                }
            }
        }
    }

    let Some(name) = entry.neuron_path.file_name().and_then(|name| name.to_str()) else {
        return entry.neuron_path.display().to_string();
    };
    name.split('.').next().unwrap_or(name).to_string()
}

pub(in crate::index) fn parse_temporal_from_now_unit(
    raw: &str,
) -> Option<SyntheticElapsedFromNowUnit> {
    match raw.trim() {
        "day" | "days" => Some(SyntheticElapsedFromNowUnit::Day),
        "week" | "weeks" => Some(SyntheticElapsedFromNowUnit::Week),
        "month" | "months" => Some(SyntheticElapsedFromNowUnit::Month),
        "year" | "years" => Some(SyntheticElapsedFromNowUnit::Year),
        _ => None,
    }
}

pub(in crate::index) fn extract_temporal_interval_phrases(
    task_lower: &str,
) -> Option<(String, String)> {
    let trimmed = task_lower.trim().trim_end_matches('?');
    let (before_after, start_phrase) = trimmed.split_once(" after ")?;
    let end_phrase = before_after
        .strip_prefix("how many days did it take for me to ")
        .or_else(|| before_after.strip_prefix("how many days did it take me to "))?
        .trim();
    Some((end_phrase.to_string(), start_phrase.trim().to_string()))
}
