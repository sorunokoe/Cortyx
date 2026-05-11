use super::*;

pub(in crate::index) fn extract_current_company_answer_from_line(
    line: &str,
    lower: &str,
) -> Option<String> {
    if !line_has_current_company_marker(lower) {
        return None;
    }
    let answer = extract_phrase_after_any_index(
        line,
        lower,
        &[
            "currently working at ",
            "currently at ",
            "current company is ",
            "works at ",
            "working at ",
            "employed at ",
        ],
        &[
            " because ",
            " and ",
            " but ",
            " while ",
            ".",
            ",",
            ";",
            " with ",
        ],
        1,
    )?;
    (answer.split_whitespace().count() <= 6).then_some(answer)
}

pub(in crate::index) fn extract_instagram_current_count_candidate(
    line: &str,
    lower: &str,
) -> Option<(i32, usize)> {
    if !lower.contains("follower")
        || task_contains_any(
            lower,
            &["facebook", "twitter", "tiktok", "youtube", "linkedin"],
        )
        || !line_has_current_count_marker(lower)
    {
        return None;
    }
    if extract_duration_answer_from_line(line).is_some()
        && !task_contains_any(
            lower,
            &[
                "just checked",
                "now at",
                "currently have",
                "currently at",
                "current follower count",
            ],
        )
    {
        return None;
    }
    let value = extract_line_numbers(line)
        .into_iter().rfind(|value| *value >= 10)?;
    let mut strength = 4usize;
    if task_contains_any(
        lower,
        &[
            "just checked",
            "now at",
            "recently crossed",
            "just reached",
            "currently have",
            "currently at",
        ],
    ) {
        strength += 6;
    }
    if lower.contains("follower count") {
        strength += 2;
    }
    if task_contains_any(
        lower,
        &[
            "close to",
            "almost",
            "nearly",
            "about ",
            "around ",
            "roughly",
            "approximately",
            "approx ",
        ],
    ) {
        strength = strength.saturating_sub(4);
    }
    Some((value, strength))
}

pub(in crate::index) fn line_has_current_count_marker(lower: &str) -> bool {
    task_contains_any(
        lower,
        &[
            " currently",
            " current",
            " now",
            " right now",
            " today",
            " these days",
            " recently",
            " just ",
            " already",
            " actually",
            " still",
            " so far",
        ],
    )
}

pub(in crate::index) fn is_money_query(task: &str) -> bool {
    const MONEY_MARKERS: &[&str] = &[
        "$",
        " dollar",
        " dollars",
        "money",
        "expense",
        "expenses",
        "cost",
        "costs",
        "price",
        "prices",
        "paid",
        "bill",
        "bills",
        "budget",
        "purchase",
        "purchased",
        "income",
        "earnings",
        "earned",
        "earning",
        "salary",
        "wage",
        "wages",
        "revenue",
        "profit",
        "profits",
    ];
    const NON_MONEY_UNITS: &[&str] = &[
        "time", "times", "hour", "hours", "day", "days", "week", "weeks", "month", "months",
        "year", "years", "session", "sessions",
    ];

    let lower = task.to_ascii_lowercase();
    MONEY_MARKERS.iter().any(|marker| lower.contains(marker))
        || (lower.contains("how much") && !NON_MONEY_UNITS.iter().any(|unit| lower.contains(unit)))
}

pub(in crate::index) fn normalize_aggregate_focus_token(token: &str) -> Option<String> {
    let mut cleaned: String = token
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase();
    if cleaned.len() < 3 {
        return None;
    }
    if cleaned.ends_with("ies") && cleaned.len() > 4 {
        cleaned = format!("{}y", &cleaned[..cleaned.len() - 3]);
    } else if cleaned.ends_with('s') && !cleaned.ends_with("ss") && cleaned.len() > 4 {
        cleaned.pop();
    }
    Some(cleaned)
}

pub(in crate::index) fn aggregate_focus_tokens_for_path(path: &Path) -> Vec<String> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    let base = file_name.strip_suffix(".aggregate.md").unwrap_or(file_name);
    let topic = base
        .strip_prefix("_arith_")
        .or_else(|| base.strip_prefix("_count_"))
        .unwrap_or(base);
    topic
        .split('_')
        .filter_map(normalize_aggregate_focus_token)
        .collect()
}

pub(in crate::index) fn aggregate_focus_token_count_for_path(path: &Path) -> usize {
    aggregate_focus_tokens_for_path(path).len()
}

pub(in crate::index) fn aggregate_focus_match_count_for_path(
    path: &Path,
    focus_terms: &[String],
) -> usize {
    let aggregate_tokens: HashSet<String> =
        aggregate_focus_tokens_for_path(path).into_iter().collect();
    let focus_tokens: HashSet<String> = focus_terms
        .iter()
        .filter_map(|term| normalize_aggregate_focus_token(term))
        .collect();
    aggregate_tokens.intersection(&focus_tokens).count()
}

pub(in crate::index) fn best_matching_arithmetic_aggregate_path(
    project_root: &Path,
    focus_terms: &[String],
) -> Option<PathBuf> {
    let ndir = neuron_dir(project_root);
    let Ok(read_dir) = std::fs::read_dir(&ndir) else {
        return None;
    };

    read_dir
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.starts_with("_arith_") && name.ends_with(".aggregate.md"))
                .unwrap_or(false)
        })
        .filter_map(|path| {
            let match_count = aggregate_focus_match_count_for_path(&path, focus_terms);
            if match_count == 0 {
                return None;
            }
            let token_count = aggregate_focus_token_count_for_path(&path).max(1);
            let score = (match_count as f32 * 100.0) + (match_count as f32 / token_count as f32);
            Some((score, path))
        })
        .max_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)))
        .map(|(_, path)| path)
}

pub(in crate::index) fn is_session_summary_path(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.ends_with("_summary.md"))
        .unwrap_or(false)
}

pub(in crate::index) fn strip_query_surface_section(content: &str) -> String {
    let without_query = strip_named_section(content, "query_surface");
    strip_named_section(&without_query, "answer_surface")
}

pub(in crate::index) fn strip_named_section(content: &str, section_name: &str) -> String {
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

pub(in crate::index) fn parse_index_answer_surface_rows(
    content: &str,
) -> Vec<IndexAnswerSurfaceRow> {
    let sections = crate::neuron::parse_sections(content);
    let Some(table) = sections.get("answer_surface") else {
        return Vec::new();
    };

    table
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if !trimmed.starts_with('|') {
                return None;
            }
            let columns = trimmed
                .trim_matches('|')
                .split('|')
                .map(str::trim)
                .collect::<Vec<_>>();
            if columns.len() != 3 {
                return None;
            }
            if columns[0].eq_ignore_ascii_case("question_pattern")
                || columns[0].chars().all(|c| c == '-' || c == ' ')
            {
                return None;
            }

            let answer_span = columns[1]
                .trim()
                .trim_matches(|c: char| matches!(c, '"' | '\'' | '`'))
                .to_string();
            if answer_span.is_empty() {
                return None;
            }

            Some(IndexAnswerSurfaceRow {
                question_pattern: columns[0].to_string(),
                answer_span,
                confidence: columns[2].parse::<f32>().unwrap_or(0.0),
            })
        })
        .collect()
}
