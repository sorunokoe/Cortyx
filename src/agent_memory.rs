//! Agent memory persistence for Cortyx.
//!
//! Structured diary entries for tracking agent interactions and decisions.

use serde::{Deserialize, Serialize};

/// A structured record of an agent interaction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StructuredDiaryEntry {
    pub agent: Option<String>,
    pub title: Option<String>,
    pub status: Option<String>,
    pub goal: Option<String>,
    pub next_step: Option<String>,
    pub blocker: Option<String>,
    pub outcome: Option<String>,
    pub entities: Vec<String>,
    pub depends_on: Vec<String>,
    pub action: Option<String>,
    /// Heuristic-generated decomposition plan, populated by `refine_entry()`.
    /// Not persisted if None.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refined_plan: Option<String>,
}

/// Return `true` when any structured diary metadata is present after normalization.
///
/// This is the gate between storing a diary entry as raw free-form text versus
/// rendering the full structured `# Agent memory` format.
#[must_use]
pub fn has_structured_diary_fields(
    title: Option<&str>,
    status: Option<&str>,
    goal: Option<&str>,
    next_step: Option<&str>,
    blocker: Option<&str>,
    outcome: Option<&str>,
    entities: &[String],
    depends_on: &[String],
) -> bool {
    normalize_inline(title).is_some()
        || normalize_inline(status).is_some()
        || normalize_inline(goal).is_some()
        || normalize_inline(next_step).is_some()
        || normalize_inline(blocker).is_some()
        || normalize_multiline(outcome).is_some()
        || !normalize_list(entities).is_empty()
        || !normalize_list(depends_on).is_empty()
}

/// Render a diary entry in Cortyx's structured Markdown format.
///
/// The output starts with `# Agent memory`, followed by normalized `- key:
/// value` metadata lines and optional `## action` / `## outcome` sections. If no
/// structured fields survive normalization, the function returns the normalized
/// action text by itself.
#[must_use]
pub fn render_structured_diary_entry(
    agent: &str,
    action: &str,
    title: Option<&str>,
    status: Option<&str>,
    goal: Option<&str>,
    next_step: Option<&str>,
    blocker: Option<&str>,
    outcome: Option<&str>,
    entities: &[String],
    depends_on: &[String],
) -> String {
    let clean_agent = normalize_inline(Some(agent)).unwrap_or_else(|| "agent".to_string());
    let clean_action = normalize_multiline(Some(action));
    let clean_title = normalize_inline(title);
    let clean_status = normalize_inline(status);
    let clean_goal = normalize_inline(goal);
    let clean_next_step = normalize_inline(next_step);
    let clean_blocker = normalize_inline(blocker);
    let clean_outcome = normalize_multiline(outcome);
    let clean_entities = normalize_list(entities);
    let clean_depends_on = normalize_list(depends_on);

    if clean_title.is_none()
        && clean_status.is_none()
        && clean_goal.is_none()
        && clean_next_step.is_none()
        && clean_blocker.is_none()
        && clean_outcome.is_none()
        && clean_entities.is_empty()
        && clean_depends_on.is_empty()
    {
        return clean_action.unwrap_or_default();
    }

    let mut out = String::from("# Agent memory\n\n");
    out.push_str(&format!("- agent: {clean_agent}\n"));
    if let Some(title) = clean_title {
        out.push_str(&format!("- title: {title}\n"));
    }
    if let Some(status) = clean_status {
        out.push_str(&format!("- status: {status}\n"));
    }
    if let Some(goal) = clean_goal {
        out.push_str(&format!("- goal: {goal}\n"));
    }
    if let Some(next_step) = clean_next_step {
        out.push_str(&format!("- next_step: {next_step}\n"));
    }
    if let Some(blocker) = clean_blocker {
        out.push_str(&format!("- blocker: {blocker}\n"));
    }
    if !clean_entities.is_empty() {
        out.push_str(&format!("- entities: {}\n", clean_entities.join(", ")));
    }
    if !clean_depends_on.is_empty() {
        out.push_str(&format!("- depends_on: {}\n", clean_depends_on.join(", ")));
    }
    if let Some(action) = clean_action {
        out.push_str("\n## action\n");
        out.push_str(&action);
        out.push('\n');
    }
    if let Some(outcome) = clean_outcome {
        out.push_str("\n## outcome\n");
        out.push_str(&outcome);
        out.push('\n');
    }
    out.trim_end().to_string()
}

#[must_use]
pub fn render_structured_diary_entry_from_entry(entry: &StructuredDiaryEntry) -> String {
    let mut rendered = render_structured_diary_entry(
        entry.agent.as_deref().unwrap_or("agent"),
        entry.action.as_deref().unwrap_or(""),
        entry.title.as_deref(),
        entry.status.as_deref(),
        entry.goal.as_deref(),
        entry.next_step.as_deref(),
        entry.blocker.as_deref(),
        entry.outcome.as_deref(),
        &entry.entities,
        &entry.depends_on,
    );
    if let Some(refined_plan) = normalize_inline(entry.refined_plan.as_deref()) {
        let metadata = format!("- refined_plan: {refined_plan}\n");
        if let Some(section_pos) = rendered.find("\n## ") {
            rendered.insert_str(section_pos + 1, &metadata);
        } else if rendered.is_empty() {
            rendered = metadata.trim_end().to_string();
        } else if rendered.ends_with('\n') {
            rendered.push_str(&metadata);
        } else {
            rendered.push('\n');
            rendered.push_str(&metadata);
        }
    }
    rendered.trim_end().to_string()
}

/// Parse a structured diary entry from Markdown.
///
/// Recognizes the format emitted by [`render_structured_diary_entry`]: the
/// `# Agent memory` header, `- key: value` metadata lines, and `## action` /
/// `## outcome` sections. Field values are normalized as they are extracted,
/// and `None` is returned when no structured diary markers or fields are found.
#[must_use]
pub fn parse_structured_diary_entry(content: &str) -> Option<StructuredDiaryEntry> {
    #[derive(Clone, Copy)]
    enum Section {
        Action,
        Outcome,
    }

    let mut entry = StructuredDiaryEntry {
        agent: None,
        title: None,
        status: None,
        goal: None,
        next_step: None,
        blocker: None,
        outcome: None,
        entities: Vec::new(),
        depends_on: Vec::new(),
        action: None,
        refined_plan: None,
    };
    let mut current_section = None;
    let mut action_lines = Vec::new();
    let mut outcome_lines = Vec::new();
    let mut saw_structure = false;

    for raw_line in content.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with("<!--") {
            continue;
        }
        match line {
            "# Agent memory" => {
                saw_structure = true;
                current_section = None;
                continue;
            },
            "## action" => {
                saw_structure = true;
                current_section = Some(Section::Action);
                continue;
            },
            "## outcome" => {
                saw_structure = true;
                current_section = Some(Section::Outcome);
                continue;
            },
            _ => {},
        }
        if line.starts_with("## ") || line.starts_with('#') {
            current_section = None;
            continue;
        }
        if let Some((label, value)) = parse_metadata_line(line) {
            saw_structure = true;
            match label {
                "agent" => entry.agent = normalize_inline(Some(value)),
                "title" => entry.title = normalize_inline(Some(value)),
                "status" => entry.status = normalize_inline(Some(value)),
                "goal" => entry.goal = normalize_inline(Some(value)),
                "next_step" => entry.next_step = normalize_inline(Some(value)),
                "blocker" => entry.blocker = normalize_inline(Some(value)),
                "refined_plan" => entry.refined_plan = normalize_inline(Some(value)),
                "entities" => {
                    entry.entities = value
                        .split(',')
                        .filter_map(|part| normalize_inline(Some(part)))
                        .collect();
                },
                "depends_on" => {
                    entry.depends_on = value
                        .split(',')
                        .filter_map(|part| normalize_inline(Some(part)))
                        .collect();
                },
                _ => {},
            }
            continue;
        }

        match current_section {
            Some(Section::Action) => action_lines.push(line.to_string()),
            Some(Section::Outcome) => outcome_lines.push(line.to_string()),
            None => {},
        }
    }

    entry.action = normalize_multiline(Some(&action_lines.join("\n")));
    entry.outcome = normalize_multiline(Some(&outcome_lines.join("\n")));

    if saw_structure
        || entry.title.is_some()
        || entry.status.is_some()
        || entry.goal.is_some()
        || entry.next_step.is_some()
        || entry.blocker.is_some()
        || entry.outcome.is_some()
        || !entry.entities.is_empty()
        || !entry.depends_on.is_empty()
        || entry.action.is_some()
    {
        Some(entry)
    } else {
        None
    }
}

/// Build a one-line summary for a structured diary entry.
///
/// The summary prefers title/goal/action as the headline and truncates long
/// inline fields to 96 characters with an ellipsis so it stays compact in
/// timelines, history views, and collaboration summaries.
#[must_use]
pub fn entry_summary(entry: &StructuredDiaryEntry) -> String {
    let headline = entry
        .title
        .clone()
        .or_else(|| entry.goal.clone())
        .or_else(|| entry.action.as_deref().map(truncate_inline))
        .or_else(|| entry.outcome.as_deref().map(truncate_inline))
        .unwrap_or_else(|| "untitled action".to_string());

    let mut parts = vec![headline];
    if let Some(status) = &entry.status {
        parts.push(format!("status: {status}"));
    }
    if let Some(goal) = &entry.goal {
        if parts
            .first()
            .map(|headline| headline != goal)
            .unwrap_or(true)
        {
            parts.push(format!("goal: {}", truncate_inline(goal)));
        }
    }
    if let Some(blocker) = &entry.blocker {
        parts.push(format!("blocker: {}", truncate_inline(blocker)));
    }
    if let Some(next_step) = &entry.next_step {
        parts.push(format!("next: {}", truncate_inline(next_step)));
    }
    if let Some(outcome) = &entry.outcome {
        parts.push(format!("outcome: {}", truncate_inline(outcome)));
    }
    if !entry.depends_on.is_empty() {
        parts.push(format!("depends on: {}", entry.depends_on.join(", ")));
    }
    if !entry.entities.is_empty() {
        parts.push(format!("entities: {}", entry.entities.join(", ")));
    }
    if let Some(refined_plan) = &entry.refined_plan {
        parts.push(format!("refined_plan: {}", truncate_inline(refined_plan)));
    }
    parts.join(" — ")
}

/// Backward-compatible alias for [`entry_summary`].
#[must_use]
pub fn summarize_structured_diary_entry(entry: &StructuredDiaryEntry) -> String {
    entry_summary(entry)
}

/// Render a timestamped history-view summary for a structured diary entry.
///
/// Unlike [`render_structured_diary_entry`], this is a display-oriented output:
/// it starts with a timestamped one-line summary, then appends indented action,
/// goal, blocker, outcome, and dependency lines for recent-history views.
pub fn render_structured_diary_history_entry(
    entry: &StructuredDiaryEntry,
    timestamp_secs: Option<i64>,
) -> String {
    let timestamp = timestamp_secs
        .map(format_history_timestamp_secs)
        .unwrap_or_else(|| "unknown-time".to_string());
    let mut out = format!("- {timestamp} — {}", entry_summary(entry));
    if let Some(action) = &entry.action {
        out.push_str(&format!(
            "\n  action: {}",
            truncate_chars(&summarize_history_text(action), 200)
        ));
    }
    if let Some(goal) = &entry.goal {
        out.push_str(&format!("\n  goal: {}", truncate_chars(goal, 200)));
    }
    if let Some(next_step) = &entry.next_step {
        out.push_str(&format!(
            "\n  next step: {}",
            truncate_chars(next_step, 200)
        ));
    }
    if let Some(blocker) = &entry.blocker {
        out.push_str(&format!("\n  blocker: {}", truncate_chars(blocker, 200)));
    }
    if let Some(outcome) = &entry.outcome {
        out.push_str(&format!(
            "\n  outcome: {}",
            truncate_chars(&summarize_history_text(outcome), 200)
        ));
    }
    if !entry.depends_on.is_empty() {
        out.push_str(&format!("\n  depends on: {}", entry.depends_on.join(", ")));
    }
    out.push('\n');
    out
}

fn parse_metadata_line(line: &str) -> Option<(&str, &str)> {
    let rest = line.strip_prefix("- ")?;
    let (label, value) = rest.split_once(':')?;
    Some((label.trim(), value.trim()))
}

fn normalize_inline(value: Option<&str>) -> Option<String> {
    let value = value?.trim();
    if value.is_empty() {
        return None;
    }
    Some(value.split_whitespace().collect::<Vec<_>>().join(" "))
}

fn normalize_multiline(value: Option<&str>) -> Option<String> {
    let value = value?;
    let normalized = value
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    if normalized.is_empty() {
        None
    } else {
        Some(normalized)
    }
}

fn normalize_list(values: &[String]) -> Vec<String> {
    let mut normalized = Vec::new();
    for value in values {
        let Some(clean) = normalize_inline(Some(value.as_str())) else {
            continue;
        };
        if !normalized.contains(&clean) {
            normalized.push(clean);
        }
    }
    normalized
}

fn truncate_inline(value: &str) -> String {
    const MAX_CHARS: usize = 96;
    let inline = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if inline.chars().count() <= MAX_CHARS {
        return inline;
    }
    let mut truncated = inline.chars().take(MAX_CHARS - 1).collect::<String>();
    truncated.push('…');
    truncated
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

fn summarize_history_text(content: &str) -> String {
    content
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.starts_with("<!--") && !line.starts_with('#'))
        .unwrap_or("(empty diary entry)")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn format_history_timestamp_secs(timestamp_secs: i64) -> String {
    if timestamp_secs < 0 {
        return timestamp_secs.to_string();
    }
    let (y, mo, d, h, mi, s) = crate::neuron::unix_secs_to_datetime(timestamp_secs.cast_unsigned());
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

/// Populate `entry.refined_plan` with a heuristic decomposition suggestion when
/// the blocker, goal, or blocked status matches a vague or stuck pattern.
///
/// Pure heuristic — no LLM required. Returns `true` if a suggestion was generated.
pub fn refine_entry(entry: &mut StructuredDiaryEntry) -> bool {
    let blocker = entry.blocker.as_deref().unwrap_or("").to_lowercase();
    let status = entry.status.as_deref().unwrap_or("").to_lowercase();
    let goal = entry.goal.as_deref().unwrap_or("").to_lowercase();

    let suggestion = if matches_vague_pattern(&blocker) || matches_vague_pattern(&goal) {
        Some(
            "Break this into smaller sub-tasks: (1) List what you know, \
             (2) List what is unclear, (3) Identify the first concrete step \
             that does not require the unknown information."
                .to_string(),
        )
    } else if matches_too_large_pattern(&blocker) || matches_too_large_pattern(&goal) {
        Some(
            "Decompose into independent slices: (1) Identify the smallest \
             deliverable that provides value on its own, (2) Do that first, \
             (3) Reassess scope after each completed slice."
                .to_string(),
        )
    } else if matches_waiting_pattern(&blocker) {
        Some(
            "Unblock parallel work: (1) Capture the exact dependency \
             in depends_on, (2) Identify any parts of the task that can \
             proceed without the blocked input, (3) Schedule a follow-up \
             check-in."
                .to_string(),
        )
    } else if status.contains("blocked") && entry.blocker.is_none() {
        Some(
            "Status is blocked but no blocker is recorded. \
             Add a specific blocker description so the dependency can be tracked."
                .to_string(),
        )
    } else {
        None
    };

    if let Some(s) = suggestion {
        entry.refined_plan = Some(s);
        true
    } else {
        false
    }
}

fn matches_vague_pattern(text: &str) -> bool {
    const VAGUE: &[&str] = &[
        "unclear",
        "not sure",
        "unsure",
        "don't know",
        "dont know",
        "unknown",
        "not clear",
        "not certain",
        "uncertain",
    ];
    VAGUE.iter().any(|p| text.contains(p))
}

fn matches_too_large_pattern(text: &str) -> bool {
    const TOO_LARGE: &[&str] = &[
        "too large",
        "too big",
        "too complex",
        "too broad",
        "overwhelming",
        "scope creep",
        "sprawling",
        "hard to scope",
    ];
    TOO_LARGE.iter().any(|p| text.contains(p))
}

fn matches_waiting_pattern(text: &str) -> bool {
    const WAITING: &[&str] = &[
        "waiting on",
        "waiting for",
        "blocked by",
        "blocked on",
        "pending ",
        "depends on",
        "need approval",
        "needs approval",
    ];
    WAITING.iter().any(|p| text.contains(p))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_and_parse_structured_diary_round_trip() {
        let content = render_structured_diary_entry(
            "reviewer",
            "Investigated auth middleware coverage across the login path.",
            Some("Audit auth middleware"),
            Some("done"),
            Some("Close the auth bypass without regressing login flow."),
            Some("Patch the legacy REST route and update tests."),
            Some("Waiting on route ownership clarification."),
            Some("Found a legacy bypass in the old REST route."),
            &["auth".to_string(), "middleware".to_string()],
            &["router-owner".to_string(), "qa".to_string()],
        );
        let parsed = parse_structured_diary_entry(&content).unwrap();
        assert_eq!(parsed.agent.as_deref(), Some("reviewer"));
        assert_eq!(parsed.title.as_deref(), Some("Audit auth middleware"));
        assert_eq!(parsed.status.as_deref(), Some("done"));
        assert_eq!(
            parsed.goal.as_deref(),
            Some("Close the auth bypass without regressing login flow.")
        );
        assert_eq!(
            parsed.next_step.as_deref(),
            Some("Patch the legacy REST route and update tests.")
        );
        assert_eq!(
            parsed.blocker.as_deref(),
            Some("Waiting on route ownership clarification.")
        );
        assert_eq!(
            parsed.outcome.as_deref(),
            Some("Found a legacy bypass in the old REST route.")
        );
        assert_eq!(parsed.entities, vec!["auth", "middleware"]);
        assert_eq!(parsed.depends_on, vec!["router-owner", "qa"]);
        assert!(parsed
            .action
            .as_deref()
            .unwrap()
            .contains("Investigated auth middleware coverage"));
    }

    #[test]
    fn render_entry_from_entry_preserves_refined_plan() {
        let content = render_structured_diary_entry_from_entry(&StructuredDiaryEntry {
            agent: Some("reviewer".to_string()),
            title: Some("Audit auth middleware".to_string()),
            status: Some("blocked".to_string()),
            goal: Some("Close the auth bypass without regressing login flow.".to_string()),
            next_step: Some("Patch the legacy REST route and update tests.".to_string()),
            blocker: Some("Waiting on route ownership clarification.".to_string()),
            outcome: None,
            entities: vec!["auth".to_string()],
            depends_on: vec!["router-owner".to_string()],
            action: Some(
                "Investigated auth middleware coverage across the login path.".to_string(),
            ),
            refined_plan: Some("Break this into smaller sub-tasks first.".to_string()),
        });
        let parsed = parse_structured_diary_entry(&content).unwrap();
        assert_eq!(
            parsed.refined_plan.as_deref(),
            Some("Break this into smaller sub-tasks first.")
        );
    }

    #[test]
    fn summarize_structured_diary_prefers_title_status_and_outcome() {
        let entry = StructuredDiaryEntry {
            agent: Some("architect".to_string()),
            title: Some("Design auth refactor".to_string()),
            status: Some("blocked".to_string()),
            goal: Some("Unify auth entry points.".to_string()),
            next_step: Some("Confirm ownership of the legacy route.".to_string()),
            blocker: Some("Waiting on route ownership clarification.".to_string()),
            outcome: Some("Waiting on route ownership clarification.".to_string()),
            entities: vec!["auth".to_string(), "router".to_string()],
            depends_on: vec!["platform-team".to_string()],
            action: Some("Reviewed auth routing".to_string()),
            refined_plan: None,
        };
        let summary = summarize_structured_diary_entry(&entry);
        assert!(summary.contains("Design auth refactor"));
        assert!(summary.contains("status: blocked"));
        assert!(summary.contains("goal: Unify auth entry points."));
        assert!(summary.contains("blocker: Waiting on route ownership clarification."));
        assert!(summary.contains("next: Confirm ownership of the legacy route."));
        assert!(summary.contains("Waiting on route ownership clarification."));
        assert!(summary.contains("depends on: platform-team"));
        assert!(summary.contains("entities: auth, router"));
    }

    fn make_entry_with_blocker(blocker: &str) -> StructuredDiaryEntry {
        StructuredDiaryEntry {
            agent: None,
            title: None,
            status: None,
            goal: None,
            next_step: None,
            blocker: Some(blocker.to_string()),
            outcome: None,
            entities: Vec::new(),
            depends_on: Vec::new(),
            action: None,
            refined_plan: None,
        }
    }

    #[test]
    fn refine_entry_generates_plan_for_vague_blocker() {
        let mut entry = make_entry_with_blocker("unclear what to do next");
        let refined = refine_entry(&mut entry);
        assert!(refined, "Expected refine_entry to return true");
        let plan = entry.refined_plan.as_deref().unwrap();
        assert!(
            plan.contains("sub-task") || plan.contains("unclear") || plan.contains("smaller"),
            "Expected decomposition advice; got: {plan}"
        );
    }

    #[test]
    fn refine_entry_generates_plan_for_too_large_blocker() {
        let mut entry = make_entry_with_blocker("the task is too large to start");
        let refined = refine_entry(&mut entry);
        assert!(refined);
        let plan = entry.refined_plan.as_deref().unwrap();
        assert!(
            plan.contains("slice") || plan.contains("deliverable"),
            "Expected slicing advice; got: {plan}"
        );
    }

    #[test]
    fn refine_entry_generates_plan_for_waiting_blocker() {
        let mut entry = make_entry_with_blocker("waiting on approval from platform-team");
        let refined = refine_entry(&mut entry);
        assert!(refined);
        let plan = entry.refined_plan.as_deref().unwrap();
        assert!(
            plan.contains("parallel") || plan.contains("depends_on"),
            "Expected parallel-work advice; got: {plan}"
        );
    }

    #[test]
    fn refine_entry_generates_plan_for_blocked_status_no_blocker() {
        let mut entry = StructuredDiaryEntry {
            status: Some("blocked".to_string()),
            blocker: None,
            refined_plan: None,
            agent: None,
            title: None,
            goal: None,
            next_step: None,
            outcome: None,
            entities: Vec::new(),
            depends_on: Vec::new(),
            action: None,
        };
        assert!(refine_entry(&mut entry));
        assert!(entry.refined_plan.is_some());
    }

    #[test]
    fn refine_entry_leaves_clear_entries_unchanged() {
        let mut entry = make_entry_with_blocker("implementing the new cache layer");
        let refined = refine_entry(&mut entry);
        assert!(
            !refined,
            "Should not generate a plan for a clear, actionable blocker"
        );
        assert!(entry.refined_plan.is_none());
    }
}
