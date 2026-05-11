//! Context rendering: neuron emission, focused selection, code mining.

use super::super::*;
use super::collaboration::{
    build_collaboration_projection, format_timestamp_secs, matches_collaboration_filter,
    render_collaborator_status_report, summarize_plain_diary_content,
};
use std::collections::{HashMap, HashSet};

/// Convert a task pattern string to a URL-safe kebab-case identifier.
pub fn to_kebab(s: &str) -> String {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s: &&str| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}

/// Truncate a string to at most `max_chars` characters (byte boundary safe for ASCII).
pub fn truncate_str(s: &str, max_chars: usize) -> String {
    s.chars().take(max_chars).collect()
}

pub fn recent_module_paths(
    index: &NeuronIndex,
    module: &str,
    limit: usize,
    kind_filter: Option<NeuronKind>,
) -> Vec<PathBuf> {
    let mut items: Vec<(i64, PathBuf)> = index
        .list_neurons(Some(module))
        .into_iter()
        .filter(|summary| {
            kind_filter
                .as_ref()
                .map(|kind| summary.kind == *kind)
                .unwrap_or(true)
        })
        .map(|summary| {
            let timestamp = index
                .context_metadata_for(&summary.path)
                .and_then(|metadata| metadata.timestamp_secs)
                .unwrap_or(i64::MIN);
            (timestamp, summary.path)
        })
        .collect();
    items.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    items
        .into_iter()
        .take(limit)
        .map(|(_, path)| path)
        .collect()
}

/// Strip HTML comment delimiters and control characters from user-supplied strings
/// before embedding them in comment blocks, preventing comment breakout and prompt injection.
pub fn sanitize_comment(s: &str) -> String {
    let clean: String = s
        .chars()
        .map(|c| {
            if c.is_ascii_control() && c != '\t' {
                ' '
            } else {
                c
            }
        })
        .collect();
    let clean = clean.replace("-->", "—>").replace("<!--", "<—");
    clean.chars().take(500).collect()
}

pub fn render_recent_agent_memory_block(
    index: &NeuronIndex,
    agent: &str,
    limit: usize,
) -> Option<String> {
    let module = format!("@agent/{}", agent.trim());
    let paths = recent_module_paths(index, &module, limit, Some(NeuronKind::Verbatim));
    if paths.is_empty() {
        return None;
    }

    let mut out = format!(
        "<!-- CORTYX WAKE-UP: @agent/{} memories -->\n",
        agent.trim()
    );
    for path in paths {
        let timestamp_secs = index
            .context_metadata_for(&path)
            .and_then(|metadata| metadata.timestamp_secs);
        match std::fs::read_to_string(&path) {
            Ok(content) => {
                out.push_str(&render_agent_memory_summary(&content, timestamp_secs));
                out.push('\n');
            },
            Err(err) => {
                out.push_str(&format!(
                    "- {} — read error: {}\n",
                    path.display(),
                    sanitize_comment(&err.to_string())
                ));
            },
        }
    }
    Some(out)
}

pub fn render_agent_memory_summary(content: &str, timestamp_secs: Option<i64>) -> String {
    let timestamp = timestamp_secs
        .map(format_timestamp_secs)
        .unwrap_or_else(|| "unknown-time".to_string());
    if let Some(entry) = parse_structured_diary_entry(content) {
        format!(
            "- {timestamp} — {}",
            summarize_structured_diary_entry(&entry)
        )
    } else {
        format!(
            "- {timestamp} — {}",
            truncate_str(&summarize_plain_diary_content(content), 180)
        )
    }
}

pub fn render_structured_diary_history_entry(
    entry: &crate::agent_memory::StructuredDiaryEntry,
    timestamp_secs: Option<i64>,
) -> String {
    let timestamp = timestamp_secs
        .map(format_timestamp_secs)
        .unwrap_or_else(|| "unknown-time".to_string());
    let mut out = format!(
        "- {timestamp} — {}",
        summarize_structured_diary_entry(entry)
    );
    if let Some(action) = &entry.action {
        out.push_str(&format!(
            "\n  action: {}",
            truncate_str(&summarize_plain_diary_content(action), 200)
        ));
    }
    if let Some(goal) = &entry.goal {
        out.push_str(&format!("\n  goal: {}", truncate_str(goal, 200)));
    }
    if let Some(next_step) = &entry.next_step {
        out.push_str(&format!("\n  next step: {}", truncate_str(next_step, 200)));
    }
    if let Some(blocker) = &entry.blocker {
        out.push_str(&format!("\n  blocker: {}", truncate_str(blocker, 200)));
    }
    if let Some(outcome) = &entry.outcome {
        out.push_str(&format!(
            "\n  outcome: {}",
            truncate_str(&summarize_plain_diary_content(outcome), 200)
        ));
    }
    if !entry.depends_on.is_empty() {
        out.push_str(&format!("\n  depends on: {}", entry.depends_on.join(", ")));
    }
    out.push('\n');
    out
}

pub fn render_agent_status_report(
    index: &NeuronIndex,
    project_root: &Path,
    agent: &str,
    include_timeline: bool,
) -> Option<String> {
    let projection = build_collaboration_projection(index, project_root);
    let summary = projection
        .collaborators
        .iter()
        .find(|summary| matches_collaboration_filter(&summary.collaborator, agent))?;
    Some(render_collaborator_status_report(
        summary,
        &projection,
        include_timeline,
    ))
}

#[allow(dead_code)]
pub fn latest_active_kg_value(entity: &kg::KgEntity, predicate: &str) -> Option<String> {
    entity
        .active_values_for_predicate(predicate, None)
        .last()
        .map(|fact| fact.value.clone())
}

#[allow(dead_code)]
pub fn active_kg_values(entity: &kg::KgEntity, predicate: &str) -> Vec<String> {
    entity
        .active_values_for_predicate(predicate, None)
        .into_iter()
        .map(|fact| fact.value.clone())
        .collect()
}

pub fn fingerprint_rendered_context(rendered: &str) -> String {
    blake3::hash(rendered.as_bytes()).to_hex()[..16].to_string()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum EmissionTier {
    Full,
    Focused,
    Summary,
}

pub fn render_context_item(
    path: &Path,
    score: f32,
    task_terms: &[String],
    index: &NeuronIndex,
) -> RenderedContextItem {
    let rendered = match std::fs::read_to_string(path) {
        Ok(content) => {
            let content = strip_render_only_sections(&content);
            match select_emission_tier(score, &content) {
                EmissionTier::Full => format!(
                    "<!-- === NEURON: {} === -->\n{}\n\n",
                    path.display(),
                    content
                ),
                EmissionTier::Focused => {
                    let focused = build_focused_context(&content, task_terms);
                    format!(
                        "<!-- === NEURON (focused, score={:.1}): {} === -->\n{}\n\n",
                        score,
                        path.file_name().unwrap_or_default().to_string_lossy(),
                        focused
                    )
                },
                EmissionTier::Summary => {
                    let summary = index
                        .summary_for(path)
                        .filter(|s| !s.is_empty())
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| {
                            fallback_excerpt(&content, 3)
                                .lines()
                                .take(3)
                                .collect::<Vec<_>>()
                                .join("\n")
                        });
                    format!(
                        "<!-- === NEURON (summary, score={:.1}): {} === -->\n{}\n\n",
                        score,
                        path.file_name().unwrap_or_default().to_string_lossy(),
                        sanitize_comment(&summary),
                    )
                },
            }
        },
        Err(err) => {
            if score >= 5.0 {
                format!("<!-- NEURON {} — read error: {err} -->\n\n", path.display())
            } else {
                tracing::warn!(
                    "Failed to read {} while building summary context: {}",
                    path.display(),
                    err
                );
                format!(
                    "<!-- === NEURON (summary, score={:.1}): {} === -->\n{}\n\n",
                    score,
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    sanitize_comment(&format!("(read error: {err})")),
                )
            }
        },
    };

    RenderedContextItem {
        path: path.to_path_buf(),
        fingerprint: fingerprint_rendered_context(&rendered),
        rendered,
    }
}

pub(super) fn select_emission_tier(score: f32, content: &str) -> EmissionTier {
    let tokens = estimate_context_tokens(content).get();
    if score < 5.0 {
        EmissionTier::Summary
    } else if score >= 9.0 || tokens <= 160 {
        EmissionTier::Full
    } else {
        EmissionTier::Focused
    }
}

pub fn build_focused_context(content: &str, task_terms: &[String]) -> String {
    if let Some(sectioned) = render_focused_sections(content, task_terms) {
        return sectioned;
    }
    render_focused_excerpt(content, task_terms)
}

pub fn render_focused_sections(content: &str, task_terms: &[String]) -> Option<String> {
    let sections = parse_markdown_sections(content);
    if sections.len() < 2 {
        return None;
    }

    let focus_terms = significant_task_terms(task_terms);
    let debug_task = focus_terms.iter().any(|term| {
        matches!(
            term.as_str(),
            "fix" | "bug" | "error" | "errors" | "failing" | "failure" | "debug" | "issue"
        )
    });
    let guidance_task = focus_terms.iter().any(|term| {
        matches!(
            term.as_str(),
            "implement" | "implementation" | "how" | "why" | "use" | "usage" | "example"
        )
    });

    let mut selected = std::collections::BTreeSet::new();
    if let Some(idx) = sections
        .iter()
        .position(|(name, _)| name.eq_ignore_ascii_case("purpose"))
    {
        selected.insert(idx);
    }

    let mut scored: Vec<(i32, usize)> = sections
        .iter()
        .enumerate()
        .map(|(idx, (name, body))| {
            let lower_name = name.to_ascii_lowercase();
            let section_terms: std::collections::HashSet<String> =
                tokenize(body).into_iter().collect();
            let overlap = section_terms
                .iter()
                .filter(|term| focus_terms.contains(*term))
                .count() as i32;
            let mut score = overlap * 10;
            match lower_name.as_str() {
                "purpose" => score += 15,
                "api" => score += 12,
                "pitfalls" if debug_task => score += 14,
                "patterns" | "examples" if guidance_task => score += 12,
                "auto_evolved" => score += 6,
                "notes" => score += 2,
                _ => {},
            }
            (score, idx)
        })
        .collect();
    scored.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    for (_, idx) in scored {
        selected.insert(idx);
        if selected.len() >= 3 {
            break;
        }
    }

    if selected.is_empty() {
        return None;
    }

    let title = content
        .lines()
        .find(|line| line.trim_start().starts_with("# "))
        .map(str::trim)
        .filter(|line| !line.is_empty());
    let mut out = String::new();
    if let Some(title) = title {
        out.push_str(title);
        out.push_str("\n\n");
    }
    for idx in selected {
        let (name, body) = &sections[idx];
        let body = trim_body_lines(body, 8);
        if body.is_empty() {
            continue;
        }
        out.push_str("## ");
        out.push_str(name);
        out.push('\n');
        out.push_str(&body);
        out.push_str("\n\n");
    }

    let trimmed = out.trim().to_string();
    (!trimmed.is_empty()).then_some(trimmed)
}

pub fn render_focused_excerpt(content: &str, task_terms: &[String]) -> String {
    let focus_terms = significant_task_terms(task_terms);
    let lines: Vec<&str> = content.lines().collect();
    let mut scored: Vec<(usize, usize)> = lines
        .iter()
        .enumerate()
        .filter_map(|(idx, line)| {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                return None;
            }
            let overlap = tokenize(trimmed)
                .into_iter()
                .filter(|term| focus_terms.contains(term))
                .count();
            if overlap == 0 {
                return None;
            }
            let speaker_bonus =
                usize::from(trimmed.starts_with("User:") || trimmed.starts_with("Assistant:"));
            Some((overlap + speaker_bonus, idx))
        })
        .collect();

    if scored.is_empty() {
        return fallback_excerpt(content, 6);
    }

    scored.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    let mut chosen = std::collections::BTreeSet::new();
    for (_, idx) in scored.into_iter().take(3) {
        let start = idx.saturating_sub(1);
        let end = (idx + 1).min(lines.len().saturating_sub(1));
        for (line_idx, line) in lines.iter().enumerate().take(end + 1).skip(start) {
            if !line.trim().is_empty() {
                chosen.insert(line_idx);
            }
        }
    }

    let excerpt_lines: Vec<&str> = chosen
        .into_iter()
        .filter_map(|idx| lines.get(idx).copied())
        .filter(|line| !line.trim().is_empty())
        .take(10)
        .collect();
    if excerpt_lines.is_empty() {
        fallback_excerpt(content, 6)
    } else {
        excerpt_lines.join("\n")
    }
}

pub fn parse_markdown_sections(content: &str) -> Vec<(String, String)> {
    let mut sections = Vec::new();
    let mut current_name: Option<String> = None;
    let mut body_lines: Vec<&str> = Vec::new();

    for line in content.lines() {
        if let Some(name) = line.trim_start().strip_prefix("## ") {
            if let Some(prev) = current_name.take() {
                sections.push((prev, body_lines.join("\n").trim().to_string()));
                body_lines.clear();
            }
            current_name = Some(name.trim().to_string());
        } else if current_name.is_some() {
            body_lines.push(line);
        }
    }

    if let Some(name) = current_name {
        sections.push((name, body_lines.join("\n").trim().to_string()));
    }

    sections.retain(|(_, body)| !body.is_empty());
    sections
}

pub fn trim_body_lines(body: &str, max_nonempty_lines: usize) -> String {
    body.lines()
        .filter(|line| !line.trim().is_empty())
        .take(max_nonempty_lines)
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn fallback_excerpt(content: &str, max_nonempty_lines: usize) -> String {
    content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .take(max_nonempty_lines)
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn significant_task_terms(task_terms: &[String]) -> std::collections::HashSet<String> {
    const STOPWORDS: &[&str] = &[
        "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "for", "from", "have", "how",
        "i", "in", "into", "is", "it", "me", "my", "of", "on", "or", "our", "that", "the", "their",
        "them", "they", "this", "to", "was", "what", "when", "where", "which", "who", "why",
        "with", "you", "your",
    ];
    task_terms
        .iter()
        .map(|term| term.to_ascii_lowercase())
        .filter(|term| term.len() >= 3 && !STOPWORDS.contains(&term.as_str()))
        .collect()
}

pub fn strip_render_only_sections(content: &str) -> String {
    let without_query = strip_named_render_section(content, "query_surface");
    strip_named_render_section(&without_query, "answer_surface")
}

pub fn strip_named_render_section(content: &str, section_name: &str) -> String {
    let header = format!("## {section_name}");
    let marker = format!("<!-- SECTION: {section_name} -->");
    let end_marker = "<!-- /SECTION -->";
    let Some(header_start) = content.find(&header) else {
        return content.to_string();
    };
    let Some(section_start_rel) = content[header_start..].find(&marker) else {
        return content.to_string();
    };
    let section_start = header_start + section_start_rel;
    let Some(section_end_rel) = content[section_start..].find(end_marker) else {
        return content.to_string();
    };
    let section_end = section_start + section_end_rel + end_marker.len();

    let mut stripped = String::with_capacity(content.len());
    stripped.push_str(content[..header_start].trim_end());
    if !stripped.ends_with('\n') {
        stripped.push('\n');
    }
    let tail = content[section_end..].trim_start_matches('\n');
    if !tail.is_empty() {
        stripped.push('\n');
        stripped.push_str(tail);
    }
    stripped
}

pub fn render_overflow_item(path: &Path, headline: &str) -> RenderedContextItem {
    let rendered = format!(
        "<!-- NEURON (compressed): {} — {} -->\n",
        path.file_name().unwrap_or_default().to_string_lossy(),
        sanitize_comment(headline),
    );
    RenderedContextItem {
        path: path.to_path_buf(),
        fingerprint: fingerprint_rendered_context(&rendered),
        rendered,
    }
}

pub fn render_module_capsule(project_root: &Path, module: &str) -> Option<RenderedContextItem> {
    let path = module_capsule_path(project_root, module);
    let content = match std::fs::read_to_string(&path) {
        Ok(content) => content,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return None,
        Err(err) => {
            tracing::warn!(
                "Failed to read module capsule {} for {}: {}",
                path.display(),
                module,
                err
            );
            return Some(RenderedContextItem {
                path,
                fingerprint: fingerprint_rendered_context(&format!(
                    "<!-- MODULE CAPSULE {} — read error: {} -->\n\n",
                    sanitize_comment(module),
                    sanitize_comment(&err.to_string())
                )),
                rendered: format!(
                    "<!-- MODULE CAPSULE {} — read error: {} -->\n\n",
                    sanitize_comment(module),
                    sanitize_comment(&err.to_string())
                ),
            });
        },
    };
    let rendered = format!(
        "<!-- === MODULE CAPSULE: {} === -->\n{}\n\n",
        sanitize_comment(module),
        content
    );
    Some(RenderedContextItem {
        path,
        fingerprint: fingerprint_rendered_context(&rendered),
        rendered,
    })
}

pub fn build_path_module_map(
    paths_with_scores: &[(PathBuf, f32)],
    overflow: &[(PathBuf, String)],
    index: &NeuronIndex,
) -> HashMap<PathBuf, String> {
    let mut path_modules = HashMap::new();
    for (path, _) in paths_with_scores {
        if let Some(module) = index.module_for(path) {
            path_modules.insert(path.clone(), module.to_string());
        }
    }
    for (path, _) in overflow {
        if let Some(module) = index.module_for(path) {
            path_modules.insert(path.clone(), module.to_string());
        }
    }
    path_modules
}

pub fn select_capsule_modules(
    paths_with_scores: &[(PathBuf, f32)],
    explicit_module: Option<&str>,
    path_modules: &HashMap<PathBuf, String>,
) -> Vec<String> {
    if let Some(module) = explicit_module {
        return if is_capsule_module(module) {
            vec![module.to_string()]
        } else {
            Vec::new()
        };
    }

    let module_tagged_total = paths_with_scores
        .iter()
        .filter(|(path, _)| {
            path_modules
                .get(path)
                .is_some_and(|module| is_capsule_module(module))
        })
        .count();
    if module_tagged_total < 2 {
        return Vec::new();
    }

    let mut counts: HashMap<&str, usize> = HashMap::new();
    for (path, _) in paths_with_scores {
        let Some(module) = path_modules.get(path) else {
            continue;
        };
        if !is_capsule_module(module) {
            continue;
        }
        *counts.entry(module.as_str()).or_insert(0) += 1;
    }

    let mut ranked: Vec<(&str, usize)> = counts.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    ranked
        .into_iter()
        .find(|(_, count)| *count >= 2 && *count * 2 >= module_tagged_total)
        .map(|(module, _)| vec![module.to_string()])
        .unwrap_or_default()
}

pub fn select_capsule_anchor_paths(
    paths_with_scores: &[(PathBuf, f32)],
    capsule_modules: &HashSet<String>,
    path_modules: &HashMap<PathBuf, String>,
) -> HashSet<PathBuf> {
    const CAPSULE_DYNAMIC_NEURONS_PER_MODULE: usize = 2;
    const CAPSULE_FULL_BODY_SCORE_THRESHOLD: f32 = 5.0;

    let mut grouped: HashMap<&str, Vec<(&PathBuf, f32)>> = HashMap::new();
    for (path, score) in paths_with_scores {
        let Some(module) = path_modules.get(path) else {
            continue;
        };
        if capsule_modules.contains(module) {
            grouped
                .entry(module.as_str())
                .or_default()
                .push((path, *score));
        }
    }

    let mut keep = HashSet::new();
    for items in grouped.values_mut() {
        items.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        let mut kept = 0usize;
        for (path, score) in items.iter() {
            if *score >= CAPSULE_FULL_BODY_SCORE_THRESHOLD
                && kept < CAPSULE_DYNAMIC_NEURONS_PER_MODULE
            {
                keep.insert((*path).clone());
                kept += 1;
            }
        }
        if kept == 0 {
            if let Some((path, _)) = items.first() {
                keep.insert((*path).clone());
            }
        }
    }

    keep
}

pub fn select_delta_items(
    items: &[RenderedContextItem],
    previous: Option<&HashMap<PathBuf, String>>,
) -> DeltaSelection {
    let current_paths: std::collections::HashSet<&PathBuf> =
        items.iter().map(|item| &item.path).collect();

    let mut emitted = Vec::new();
    let mut unchanged = 0usize;
    for item in items {
        if previous
            .and_then(|snapshot| snapshot.get(&item.path))
            .is_some_and(|fingerprint| fingerprint == &item.fingerprint)
        {
            unchanged += 1;
        } else {
            emitted.push(item.clone());
        }
    }

    let removed = previous
        .map(|snapshot| {
            snapshot
                .keys()
                .filter(|path| !current_paths.contains(*path))
                .count()
        })
        .unwrap_or(0);

    DeltaSelection {
        emitted,
        unchanged,
        removed,
    }
}

/// S-VIII (R16): Auto-mine UseCase stubs from code blocks in an LLM response.
///
/// Scans `response_text` for fenced code blocks (``` ... ```) with ≥5 lines.
/// For each block, finds the cited neuron with the highest term overlap.
/// If overlap ≥ 60% of the neuron's own terms, writes a UseCase stub to
/// `.cortyx/neurons/{neuron}.usecase.auto-{hash}.md` with `status: Stub`.
///
/// Returns the count of stubs written.
pub fn auto_mine_code_blocks(
    response_text: &str,
    cited_paths: &[PathBuf],
    project_root: &Path,
    index: &NeuronIndex,
) -> usize {
    if cited_paths.is_empty() {
        return 0;
    }

    let mut blocks: Vec<String> = Vec::new();
    let mut in_block = false;
    let mut current_block = Vec::new();
    for line in response_text.lines() {
        let trimmed = line.trim();
        if !in_block && trimmed.starts_with("```") {
            in_block = true;
            current_block.clear();
        } else if in_block && trimmed.starts_with("```") {
            if current_block.len() >= 5 {
                blocks.push(current_block.join("\n"));
            }
            in_block = false;
            current_block.clear();
        } else if in_block {
            current_block.push(line.to_string());
        }
    }

    if blocks.is_empty() {
        return 0;
    }

    let ndir = neuron_dir(project_root);
    let mut written = 0usize;

    for block in &blocks {
        let block_terms: std::collections::HashSet<String> = tokenize(block).into_iter().collect();
        if block_terms.is_empty() {
            continue;
        }

        let best = cited_paths
            .iter()
            .filter_map(|path| {
                let overlap = index.term_freq_overlap(path, &block_terms);
                let total_neuron_terms = index.term_count_for(path);
                if total_neuron_terms == 0 {
                    return None;
                }
                let ratio = overlap as f32 / total_neuron_terms as f32;
                Some((ratio, path))
            })
            .max_by(|a, b| a.0.total_cmp(&b.0));

        let Some((ratio, best_path)) = best else {
            continue;
        };
        if ratio < 0.6 {
            continue;
        }

        let stem = best_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .trim_end_matches(".context")
            .to_string();

        let hash_bytes = blake3::hash(block.as_bytes());
        let short_hash = &hash_bytes.to_hex()[..8];
        let usecase_filename = format!("{stem}.usecase.auto-{short_hash}.md");
        let usecase_path = ndir.join(&usecase_filename);

        if usecase_path.exists() {
            continue;
        }

        let content = format!(
            "# {stem} — auto-mined UseCase\n\
             status: Stub\n\
             source: auto-mined from close_task\n\n\
             ## task\n\
             (edit: describe the task pattern this code solves)\n\n\
             ## example\n\
             ```\n{block}\n```\n"
        );
        if let Err(e) = std::fs::write(&usecase_path, &content) {
            tracing::warn!(
                "S-VIII: failed to write UseCase stub {:?}: {e}",
                usecase_path
            );
        } else {
            tracing::debug!(
                "S-VIII: wrote UseCase stub {:?} (ratio={ratio:.2})",
                usecase_path
            );
            written += 1;
        }
    }

    written
}
