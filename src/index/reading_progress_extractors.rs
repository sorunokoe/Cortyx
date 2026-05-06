use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum ReadingProgressQuery {
    PagesRead(ReadingTitleQuery),
    PagesLeft(ReadingTitleQuery),
    FinishedNovelPageTotal,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ReadingTitleQuery {
    pub title_variants: Vec<String>,
    pub required_terms: Vec<String>,
}

pub(super) fn parse_reading_progress_query(
    task: &str,
    task_lower: &str,
) -> Option<ReadingProgressQuery> {
    if !task_lower.contains("page") {
        return None;
    }
    if task_lower.contains("page count of the two novels") {
        return Some(ReadingProgressQuery::FinishedNovelPageTotal);
    }

    let title_query = build_reading_title_query(task, task_lower)?;
    if task_contains_any(
        task_lower,
        &["read so far", "have i read", "have i finished"],
    ) {
        return Some(ReadingProgressQuery::PagesRead(title_query));
    }
    if task_lower.contains("pages do i have left") {
        return Some(ReadingProgressQuery::PagesLeft(title_query));
    }
    None
}

pub(super) fn extract_current_page_for_title_variants(
    line: &str,
    title_variants: &[String],
) -> Option<i32> {
    let lower = line.to_ascii_lowercase();
    reading_progress_line_matches(line, &lower, title_variants).then_some(())?;
    compile_regex(r"on page\s+(\d{1,4})")
        .captures(&lower)?
        .get(1)?
        .as_str()
        .parse::<i32>()
        .ok()
}

pub(super) fn extract_total_pages_for_title_variants(
    line: &str,
    title_variants: &[String],
) -> Option<i32> {
    let lower = line.to_ascii_lowercase();
    reading_progress_line_matches(line, &lower, title_variants).then_some(())?;

    let hyphenated = compile_regex(r"(\d{2,4})-page");
    if let Some(value) = hyphenated
        .captures(&lower)
        .and_then(|caps| caps.get(1))
        .and_then(|m| m.as_str().parse::<i32>().ok())
    {
        return Some(value);
    }

    compile_regex(r"(\d{2,4})\s+pages")
        .captures(&lower)?
        .get(1)?
        .as_str()
        .parse::<i32>()
        .ok()
}

pub(super) fn extract_just_finished_page_count(line: &str) -> Option<i32> {
    let lower = line.to_ascii_lowercase();
    if !(lower.starts_with("user:") || line.trim_start().starts_with('-'))
        || !lower.contains("just finished")
    {
        return None;
    }

    let hyphenated = compile_regex(r"just finished(?: reading)?[^0-9\n]{0,120}?(\d{2,4})-page");
    if let Some(value) = hyphenated
        .captures(&lower)
        .and_then(|caps| caps.get(1))
        .and_then(|m| m.as_str().parse::<i32>().ok())
    {
        return Some(value);
    }

    compile_regex(r"just finished(?: reading)?[^\n]{0,200}?(\d{2,4})\s+pages")
        .captures(&lower)?
        .get(1)?
        .as_str()
        .parse::<i32>()
        .ok()
}

fn build_reading_title_query(task: &str, task_lower: &str) -> Option<ReadingTitleQuery> {
    let title = extract_quoted_title(task).or_else(|| extract_unquoted_title(task, task_lower))?;
    let canonical = normalize_reading_title(&title);
    if canonical.is_empty() {
        return None;
    }

    let mut required_terms = vec!["page".to_string()];
    for token in canonical.split_whitespace() {
        let cleaned = token
            .trim_matches(|c: char| !c.is_alphanumeric() && c != '\'')
            .to_ascii_lowercase();
        if cleaned.len() < 4 || required_terms.iter().any(|existing| existing == &cleaned) {
            continue;
        }
        required_terms.push(cleaned);
        if required_terms.len() >= 4 {
            break;
        }
    }

    let mut title_variants = vec![canonical.clone()];
    for article in ["the ", "a ", "an "] {
        if let Some(stripped) = canonical.strip_prefix(article) {
            let stripped = stripped.trim().to_string();
            if !stripped.is_empty() && !title_variants.iter().any(|existing| existing == &stripped)
            {
                title_variants.push(stripped);
            }
        }
    }

    Some(ReadingTitleQuery {
        title_variants,
        required_terms,
    })
}

fn extract_unquoted_title(task: &str, task_lower: &str) -> Option<String> {
    for marker in [
        "pages do i have left in ",
        "pages do i have left of ",
        "pages do i have left for ",
        "pages have i read in ",
        "pages have i read of ",
        "have i read in ",
        "have i read of ",
        "read so far in ",
        "read so far of ",
    ] {
        let Some(start) = task_lower.find(marker) else {
            continue;
        };
        let tail = task[start + marker.len()..]
            .trim()
            .trim_end_matches('?')
            .trim_end_matches('.')
            .trim();
        if !tail.is_empty() {
            return Some(tail.to_string());
        }
    }
    None
}

fn normalize_reading_title(title: &str) -> String {
    title
        .trim()
        .trim_matches(|c: char| matches!(c, '"' | '\'' | ',' | '.' | ';' | '!' | '?'))
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

fn reading_progress_line_matches(line: &str, lower: &str, title_variants: &[String]) -> bool {
    (lower.starts_with("user:") || line.trim_start().starts_with('-'))
        && title_variants
            .iter()
            .any(|title| !title.is_empty() && lower.contains(title))
}
