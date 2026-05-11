use super::*;

pub(crate) fn fallback_snippet(path: &Path) -> String {
    path.file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned()
}

pub(crate) fn format_provenance_line(item: &EvidenceItem) -> String {
    let mut parts = Vec::new();
    parts.push(format!("{}", item.path.display()));
    parts.push(format!("score={:.1}", item.score));
    if let Some(metadata) = item.metadata.as_ref() {
        parts.push(format!("kind={}", kind_label(&metadata.kind)));
        if let Some(module) = metadata.module.as_deref() {
            parts.push(format!("module={module}"));
        }
        if let Some(ts) = metadata.timestamp_secs {
            parts.push(format!("time={}", format_timestamp(ts)));
        }
        parts.push(format!("tokens={}", metadata.tokens));
        if metadata.use_count > 0 {
            parts.push(format!(
                "hits={}/{}",
                metadata.hit_count, metadata.use_count
            ));
            parts.push(format!(
                "hit_rate={:.0}%",
                (metadata.hit_rate * 100.0).clamp(0.0, 100.0)
            ));
        }
    }
    format!("{} — {}", parts.join(", "), item.snippet)
}

pub(crate) fn kind_label(kind: &NeuronKind) -> &'static str {
    match kind {
        NeuronKind::Core => "core",
        NeuronKind::Project => "project",
        NeuronKind::UseCase => "use_case",
        NeuronKind::Concept => "concept",
        NeuronKind::Verbatim => "verbatim",
        NeuronKind::Aggregate => "aggregate",
    }
}

pub(crate) fn format_timestamp(timestamp_secs: i64) -> String {
    if timestamp_secs < 0 {
        return timestamp_secs.to_string();
    }
    let (y, mo, d, h, mi, s) = unix_secs_to_datetime(timestamp_secs as u64);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}
