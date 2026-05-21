use crate::agent_memory::{parse_structured_diary_entry, render_structured_diary_history_entry};
use crate::collaboration_kernel::agent_entity_name;
use crate::index::NeuronIndex;
use crate::kg::KgEntity;
use crate::neuron::{unix_secs_to_datetime, NeuronKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TimelineSection {
    Diary,
    Activated,
    Knowledge,
}

#[derive(Debug, Clone)]
struct TimelineItem {
    section: TimelineSection,
    timestamp_secs: i64,
    rendered: String,
}

/// Parse a duration string like "2h", "1d", "3d", or "1w" into seconds.
#[must_use]
pub fn parse_duration_secs(s: &str) -> i64 {
    let trimmed = s.trim();
    if let Some(n) = trimmed
        .strip_suffix('h')
        .and_then(|value| value.parse::<i64>().ok())
    {
        return n.saturating_mul(3_600).max(0);
    }
    if let Some(n) = trimmed
        .strip_suffix('d')
        .and_then(|value| value.parse::<i64>().ok())
    {
        return n.saturating_mul(86_400).max(0);
    }
    if let Some(n) = trimmed
        .strip_suffix('w')
        .and_then(|value| value.parse::<i64>().ok())
    {
        return n.saturating_mul(7).saturating_mul(86_400).max(0);
    }
    86_400
}

/// Return the current Unix timestamp in seconds.
#[must_use]
pub fn now_unix_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .and_then(|duration| i64::try_from(duration.as_secs()).ok())
        .unwrap_or(0)
}

/// Render a chronological session timeline across agent diaries, activated neurons, and KG facts.
#[must_use]
pub fn render_session_timeline(
    idx: &NeuronIndex,
    since_secs: i64,
    agent_filter: Option<&str>,
    limit: usize,
) -> String {
    if limit == 0 {
        return "# Cortyx Session Timeline\n\n_No items requested._\n".to_string();
    }

    let normalized_agent = agent_filter
        .map(str::trim)
        .filter(|agent| !agent.is_empty());
    let cutoff = now_unix_secs().saturating_sub(since_secs.max(0));

    let mut items = collect_diary_items(idx, cutoff, normalized_agent);
    items.extend(collect_activated_items(idx, cutoff));
    items.extend(collect_kg_items(idx, cutoff, normalized_agent));

    if items.is_empty() {
        return "# Cortyx Session Timeline\n\n_No recent session activity found._\n".to_string();
    }

    items.sort_by(|left, right| {
        right
            .timestamp_secs
            .cmp(&left.timestamp_secs)
            .then_with(|| left.rendered.cmp(&right.rendered))
    });
    items.truncate(limit);

    let mut diary = Vec::new();
    let mut activated = Vec::new();
    let mut knowledge = Vec::new();
    for item in items {
        match item.section {
            TimelineSection::Diary => diary.push(item.rendered),
            TimelineSection::Activated => activated.push(item.rendered),
            TimelineSection::Knowledge => knowledge.push(item.rendered),
        }
    }

    let mut out = String::from("# Cortyx Session Timeline\n\n");
    if let Some(agent) = normalized_agent {
        out.push_str(&format!("Filtered to agent `{agent}`.\n\n"));
    }
    render_section(
        &mut out,
        "Recent Diary Entries",
        &diary,
        "_No recent diary entries found._",
    );
    render_section(
        &mut out,
        "Recently Activated Neurons",
        &activated,
        "_No recently activated neurons found._",
    );
    render_section(
        &mut out,
        "Recent KG Facts",
        &knowledge,
        "_No recent KG facts found._",
    );
    out
}

fn collect_diary_items(
    idx: &NeuronIndex,
    cutoff: i64,
    agent_filter: Option<&str>,
) -> Vec<TimelineItem> {
    let wanted_module = agent_filter.map(|agent| format!("@agent/{agent}"));
    idx.list_neurons(None)
        .into_iter()
        .filter(|summary| summary.kind == NeuronKind::Verbatim)
        .filter_map(|summary| {
            let metadata = idx.context_metadata_for(&summary.path)?;
            let module = metadata.module.as_deref()?;
            if !module.starts_with("@agent/") {
                return None;
            }
            if wanted_module
                .as_deref()
                .is_some_and(|wanted| module != wanted)
            {
                return None;
            }
            let timestamp_secs = metadata.timestamp_secs?;
            if timestamp_secs < cutoff {
                return None;
            }
            let rendered = match std::fs::read_to_string(&summary.path) {
                Ok(content) => {
                    if let Some(entry) = parse_structured_diary_entry(&content) {
                        render_structured_diary_history_entry(&entry, Some(timestamp_secs))
                    } else {
                        let label = module.strip_prefix("@agent/").unwrap_or(module);
                        let body = content.trim();
                        format!(
                            "### {label} — {}\n\n{}\n\n",
                            format_timestamp_secs(timestamp_secs),
                            if body.is_empty() {
                                "(empty entry)"
                            } else {
                                body
                            }
                        )
                    }
                },
                Err(err) => format!("- {} — read error: {}\n", summary.path.display(), err),
            };
            Some(TimelineItem {
                section: TimelineSection::Diary,
                timestamp_secs,
                rendered,
            })
        })
        .collect()
}

fn collect_activated_items(idx: &NeuronIndex, cutoff: i64) -> Vec<TimelineItem> {
    idx.list_neurons(None)
        .into_iter()
        .filter(|summary| summary.use_count > 0)
        .filter_map(|summary| {
            let metadata = idx.context_metadata_for(&summary.path)?;
            let timestamp_secs = metadata.timestamp_secs?;
            if timestamp_secs < cutoff {
                return None;
            }
            if metadata
                .module
                .as_deref()
                .is_some_and(|module| module.starts_with("@agent/"))
                || is_kg_path(&summary.path)
            {
                return None;
            }
            let label = summary
                .path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("unknown");
            let summary_text = metadata.summary.trim();
            let module_suffix = metadata
                .module
                .as_deref()
                .map(|module| format!(", module: `{module}`"))
                .unwrap_or_default();
            let summary_suffix = if summary_text.is_empty() {
                String::new()
            } else {
                format!("\n  - {summary_text}")
            };
            Some(TimelineItem {
                section: TimelineSection::Activated,
                timestamp_secs,
                rendered: format!(
                    "- {} — `{label}` (uses: {}, hit rate: {:.0}%{module_suffix}){}\n",
                    format_timestamp_secs(timestamp_secs),
                    summary.use_count,
                    metadata.hit_rate * 100.0,
                    summary_suffix,
                ),
            })
        })
        .collect()
}

fn collect_kg_items(
    idx: &NeuronIndex,
    cutoff: i64,
    agent_filter: Option<&str>,
) -> Vec<TimelineItem> {
    let wanted_entity = agent_filter.map(agent_entity_name);
    idx.list_neurons(None)
        .into_iter()
        .filter(|summary| is_kg_path(&summary.path))
        .filter_map(|summary| KgEntity::load(&summary.path).ok())
        .filter(|entity| {
            wanted_entity
                .as_deref()
                .is_none_or(|wanted| entity.entity == wanted)
        })
        .flat_map(|entity| {
            let entity_name = entity.entity.clone();
            entity.facts.into_iter().filter_map(move |fact| {
                let timestamp_secs = parse_iso8601_to_secs(&fact.valid_from)?;
                if timestamp_secs < cutoff {
                    return None;
                }
                let ended_suffix = if fact.ended.trim().is_empty() {
                    String::new()
                } else {
                    format!(" (ended: {})", fact.ended.trim())
                };
                Some(TimelineItem {
                    section: TimelineSection::Knowledge,
                    timestamp_secs,
                    rendered: format!(
                        "- {} — `{}`: {} = {}{}\n",
                        format_timestamp_secs(timestamp_secs),
                        entity_name,
                        fact.predicate,
                        fact.value,
                        ended_suffix,
                    ),
                })
            })
        })
        .collect()
}

fn render_section(out: &mut String, title: &str, items: &[String], empty_message: &str) {
    out.push_str(&format!("## {title}\n\n"));
    if items.is_empty() {
        out.push_str(empty_message);
        out.push_str("\n\n");
        return;
    }
    for item in items {
        out.push_str(item);
        if !item.ends_with('\n') {
            out.push('\n');
        }
    }
    out.push('\n');
}

fn is_kg_path(path: &std::path::Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.starts_with("_kg_") && name.ends_with(".context.md"))
        .unwrap_or(false)
}

fn format_timestamp_secs(timestamp_secs: i64) -> String {
    let Ok(timestamp_secs) = u64::try_from(timestamp_secs) else {
        return timestamp_secs.to_string();
    };
    let (year, month, day, hour, minute, second) = unix_secs_to_datetime(timestamp_secs);
    format!("{year:04}-{month:02}-{day:02} {hour:02}:{minute:02}:{second:02} UTC")
}

pub(super) fn parse_iso8601_to_secs(value: &str) -> Option<i64> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }

    let date_part = trimmed.split(['T', ' ']).next()?;
    let parts: Vec<&str> = date_part.split('-').collect();
    if parts.len() < 3 {
        return None;
    }

    let year = parts[0].parse::<i64>().ok()?;
    let month = parts[1].parse::<i64>().ok()?;
    let day = parts[2].parse::<i64>().ok()?;
    if !(1970..=2200).contains(&year) || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }

    const MONTH_START_DAYS: [i64; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let leap_years = {
        let y = year - 1;
        y / 4 - y / 100 + y / 400 - (1969 / 4 - 1969 / 100 + 1969 / 400)
    };
    let month_index = usize::try_from(month - 1).unwrap_or(0);
    let mut total = ((year - 1970) * 365 + leap_years + MONTH_START_DAYS[month_index] + day - 1)
        .saturating_mul(86_400);

    if let Some(time_part) = trimmed.split(['T', ' ']).nth(1) {
        let bytes = time_part.as_bytes();
        if bytes.len() >= 8 && bytes[2] == b':' && bytes[5] == b':' {
            let hour = time_part.get(0..2)?.parse::<i64>().ok()?;
            let minute = time_part.get(3..5)?.parse::<i64>().ok()?;
            let second = time_part.get(6..8)?.parse::<i64>().ok()?;
            if !(0..=23).contains(&hour)
                || !(0..=59).contains(&minute)
                || !(0..=59).contains(&second)
            {
                return None;
            }
            total = total
                .saturating_add(hour.saturating_mul(3_600))
                .saturating_add(minute.saturating_mul(60))
                .saturating_add(second);
        }
    }

    Some(total)
}
