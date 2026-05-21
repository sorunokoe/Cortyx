use crate::index::NeuronIndex;
use crate::neuron::{io::meta_path, NeuronKind, NeuronMeta};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::timeline::{now_unix_secs, parse_iso8601_to_secs};

#[derive(Debug, Clone)]
struct InsightNeuron {
    path: PathBuf,
    kind: NeuronKind,
    module: String,
    use_count: u32,
    hit_count: u32,
    staleness_multiplier: f32,
    last_updated: String,
    activity_secs: Option<i64>,
}

#[derive(Debug, Default)]
struct ModuleStats {
    neuron_count: usize,
    total_use_count: u64,
    stale_count: usize,
}

/// Render a terminal-friendly neuron health report.
#[must_use]
pub fn render_insights(idx: &NeuronIndex, since_secs: i64, top_n: usize) -> String {
    let since_secs = since_secs.max(0);
    let cutoff = (since_secs > 0).then(|| now_unix_secs().saturating_sub(since_secs));
    let neurons = collect_neurons(idx, cutoff);

    let mut out = String::from("# Cortyx Insights Dashboard\n\n");
    if since_secs > 0 {
        out.push_str(&format!(
            "Filtered to neurons active within the last {since_secs} seconds.\n\n"
        ));
    }

    render_top_activated(&mut out, &neurons, top_n);
    render_stalest(&mut out, &neurons, top_n);
    render_module_health(&mut out, &neurons, top_n);
    render_index_summary(&mut out, &neurons);

    out
}

fn collect_neurons(idx: &NeuronIndex, cutoff: Option<i64>) -> Vec<InsightNeuron> {
    idx.list_neurons(None)
        .into_iter()
        .filter_map(|summary| {
            let metadata = idx.context_metadata_for(&summary.path);
            let (last_updated, last_updated_secs) = read_last_updated(&summary.path);
            let activity_secs =
                last_updated_secs.or_else(|| metadata.as_ref().and_then(|m| m.timestamp_secs));
            if cutoff.is_some_and(|cutoff| activity_secs.map_or(true, |secs| secs < cutoff)) {
                return None;
            }
            Some(InsightNeuron {
                path: summary.path,
                kind: summary.kind,
                module: metadata
                    .as_ref()
                    .and_then(|m| m.module.clone())
                    .unwrap_or_else(|| "(unscoped)".to_string()),
                use_count: summary.use_count,
                hit_count: metadata.as_ref().map_or(0, |m| m.hit_count),
                staleness_multiplier: summary.staleness_multiplier,
                last_updated,
                activity_secs,
            })
        })
        .collect()
}

fn read_last_updated(path: &Path) -> (String, Option<i64>) {
    let meta_path = meta_path(path);
    let Ok(data) = std::fs::read_to_string(meta_path) else {
        return ("unknown".to_string(), None);
    };
    let Ok(meta) = serde_json::from_str::<NeuronMeta>(&data) else {
        return ("unknown".to_string(), None);
    };
    let last_updated = meta.last_updated.trim();
    let display = if last_updated.is_empty() {
        "unknown".to_string()
    } else {
        last_updated.to_string()
    };
    let parsed = parse_iso8601_to_secs(&display)
        .or_else(|| meta.timestamp.as_deref().and_then(parse_iso8601_to_secs));
    (display, parsed)
}

fn render_top_activated(out: &mut String, neurons: &[InsightNeuron], top_n: usize) {
    let mut rows: Vec<&InsightNeuron> = neurons
        .iter()
        .filter(|neuron| neuron.use_count > 0)
        .collect();
    rows.sort_by(|left, right| {
        right
            .use_count
            .cmp(&left.use_count)
            .then_with(|| right.hit_count.cmp(&left.hit_count))
            .then_with(|| left.path.cmp(&right.path))
    });

    let table_rows: Vec<Vec<String>> = rows
        .into_iter()
        .take(top_n)
        .enumerate()
        .map(|(idx, neuron)| {
            vec![
                (idx + 1).to_string(),
                format!("`{}`", neuron.path.display()),
                format!("`{}`", neuron.module),
                neuron.use_count.to_string(),
                neuron.hit_count.to_string(),
            ]
        })
        .collect();

    render_table_section(
        out,
        "Top Activated Neurons",
        &["Rank", "Neuron path", "Module", "use_count", "hit_count"],
        &table_rows,
        "_No activated neurons found._",
    );
}

fn render_stalest(out: &mut String, neurons: &[InsightNeuron], top_n: usize) {
    let mut rows: Vec<&InsightNeuron> = neurons
        .iter()
        .filter(|neuron| neuron.staleness_multiplier < 1.0)
        .collect();
    rows.sort_by(|left, right| {
        left.staleness_multiplier
            .total_cmp(&right.staleness_multiplier)
            .then_with(|| left.path.cmp(&right.path))
    });

    let table_rows: Vec<Vec<String>> = rows
        .into_iter()
        .take(top_n)
        .enumerate()
        .map(|(idx, neuron)| {
            vec![
                (idx + 1).to_string(),
                format!("`{}`", neuron.path.display()),
                format!("`{}`", neuron.module),
                format!("{:.2}", neuron.staleness_multiplier),
                neuron.last_updated.clone(),
            ]
        })
        .collect();

    render_table_section(
        out,
        "Stalest Neurons",
        &[
            "Rank",
            "Neuron path",
            "Module",
            "Staleness",
            "Last modified",
        ],
        &table_rows,
        "_No stale neurons found._",
    );
}

fn render_module_health(out: &mut String, neurons: &[InsightNeuron], top_n: usize) {
    let mut by_module: HashMap<String, ModuleStats> = HashMap::new();
    for neuron in neurons {
        let stats = by_module.entry(neuron.module.clone()).or_default();
        stats.neuron_count += 1;
        stats.total_use_count = stats
            .total_use_count
            .saturating_add(u64::from(neuron.use_count));
        if neuron.staleness_multiplier < 0.7 {
            stats.stale_count += 1;
        }
    }

    let mut rows: Vec<(&String, &ModuleStats)> = by_module.iter().collect();
    rows.sort_by(|left, right| {
        right
            .1
            .stale_count
            .cmp(&left.1.stale_count)
            .then_with(|| right.1.neuron_count.cmp(&left.1.neuron_count))
            .then_with(|| left.0.cmp(right.0))
    });

    let table_rows: Vec<Vec<String>> = rows
        .into_iter()
        .take(top_n)
        .map(|(module, stats)| {
            let avg_use_count = if stats.neuron_count == 0 {
                0.0
            } else {
                stats.total_use_count as f32 / stats.neuron_count as f32
            };
            let stale_ratio = if stats.neuron_count == 0 {
                0.0
            } else {
                stats.stale_count as f32 / stats.neuron_count as f32
            };
            vec![
                format!("`{module}`"),
                stats.neuron_count.to_string(),
                format!("{avg_use_count:.1}"),
                stats.stale_count.to_string(),
                module_health(avg_use_count, stale_ratio).to_string(),
            ]
        })
        .collect();

    render_table_section(
        out,
        "Module Health",
        &["Module", "Neurons", "Avg uses", "Stale", "Health"],
        &table_rows,
        "_No module data available._",
    );
}

fn render_index_summary(out: &mut String, neurons: &[InsightNeuron]) {
    out.push_str("## Index Summary\n\n");
    out.push_str(&format!("- Total neurons in scope: {}\n", neurons.len()));
    out.push_str(&format!(
        "- Most recently active module: {}\n\n",
        most_recent_module(neurons)
    ));

    let mut counts: HashMap<&'static str, usize> = HashMap::new();
    for neuron in neurons {
        *counts.entry(kind_label(&neuron.kind)).or_default() += 1;
    }

    let table_rows: Vec<Vec<String>> = [
        "Core",
        "UseCase",
        "Verbatim",
        "Concept",
        "Project",
        "Aggregate",
    ]
    .into_iter()
    .map(|kind| {
        vec![
            kind.to_string(),
            counts.get(kind).copied().unwrap_or(0).to_string(),
        ]
    })
    .collect();

    render_table(out, &["Kind", "Count"], &table_rows);
    out.push('\n');
}

fn most_recent_module(neurons: &[InsightNeuron]) -> String {
    neurons
        .iter()
        .filter_map(|neuron| {
            neuron
                .activity_secs
                .map(|secs| (secs, neuron.module.as_str()))
        })
        .max_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(right.1)))
        .map(|(_, module)| format!("`{module}`"))
        .unwrap_or_else(|| "n/a".to_string())
}

fn kind_label(kind: &NeuronKind) -> &'static str {
    match kind {
        NeuronKind::Core => "Core",
        NeuronKind::UseCase => "UseCase",
        NeuronKind::Verbatim => "Verbatim",
        NeuronKind::Concept => "Concept",
        NeuronKind::Project => "Project",
        NeuronKind::Aggregate => "Aggregate",
    }
}

fn module_health(avg_use_count: f32, stale_ratio: f32) -> &'static str {
    if avg_use_count > 2.0 && stale_ratio < 0.2 {
        "✅ Good"
    } else if stale_ratio >= 0.5 || (stale_ratio >= 0.2 && avg_use_count < 1.0) {
        "❌ Poor"
    } else {
        "⚠️ Fair"
    }
}

fn render_table_section(
    out: &mut String,
    title: &str,
    headers: &[&str],
    rows: &[Vec<String>],
    empty_message: &str,
) {
    out.push_str(&format!("## {title}\n\n"));
    if rows.is_empty() {
        out.push_str(empty_message);
        out.push_str("\n\n");
        return;
    }
    render_table(out, headers, rows);
    out.push('\n');
}

fn render_table(out: &mut String, headers: &[&str], rows: &[Vec<String>]) {
    out.push_str("| ");
    out.push_str(&headers.join(" | "));
    out.push_str(" |\n| ");
    out.push_str(
        &headers
            .iter()
            .map(|_| "---")
            .collect::<Vec<_>>()
            .join(" | "),
    );
    out.push_str(" |\n");
    for row in rows {
        out.push_str("| ");
        out.push_str(&row.join(" | "));
        out.push_str(" |\n");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_insights_handles_empty_index() {
        let output = render_insights(&NeuronIndex::default(), 0, 10);
        assert!(output.contains("# Cortyx Insights Dashboard"));
        assert!(output.contains("## Top Activated Neurons"));
        assert!(output.contains("## Index Summary"));
    }
}
