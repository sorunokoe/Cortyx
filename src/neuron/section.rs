use std::collections::HashMap;

/// Parse all `<!-- SECTION: name -->` … `<!-- /SECTION -->` blocks in a neuron.
///
/// Returns `section_name → body` (content between tags, whitespace-trimmed).
/// Handles unclosed sections (content captured until EOF or next open tag).
#[must_use]
pub fn parse_sections(content: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let mut current_name: Option<String> = None;
    let mut body_lines: Vec<&str> = Vec::new();

    for line in content.lines() {
        if let Some(name) = section_open_name(line) {
            if let Some(prev_name) = current_name.take() {
                map.insert(prev_name, body_lines.join("\n").trim().to_string());
                body_lines.clear();
            }
            current_name = Some(name.to_string());
        } else if line.contains("<!-- /SECTION -->") {
            if let Some(name) = current_name.take() {
                map.insert(name, body_lines.join("\n").trim().to_string());
                body_lines.clear();
            }
        } else if current_name.is_some() {
            body_lines.push(line);
        }
    }

    if let Some(name) = current_name {
        map.insert(name, body_lines.join("\n").trim().to_string());
    }

    map
}

/// Replace or append a named section in neuron markdown.
///
/// - If `<!-- SECTION: name -->` exists: replaces its body with `new_body`.
/// - If not found: appends the section at the end of the file.
#[must_use]
pub fn replace_section(content: &str, name: &str, new_body: &str) -> String {
    let mut result = String::with_capacity(content.len() + new_body.len() + 64);
    let mut in_section = false;
    let mut found = false;

    for line in content.lines() {
        let trimmed = line.trim();

        if !in_section {
            if let Some(open_name) = section_open_name(line) {
                if open_name == name {
                    result.push_str(line);
                    result.push('\n');
                    result.push_str(new_body.trim_end_matches('\n'));
                    result.push('\n');
                    in_section = true;
                    found = true;
                    continue;
                }
            }
            result.push_str(line);
            result.push('\n');
        } else if trimmed.contains("<!-- /SECTION -->") {
            in_section = false;
            result.push_str(line);
            result.push('\n');
        }
    }

    if !found {
        if !result.ends_with('\n') {
            result.push('\n');
        }
        result.push_str(&format!(
            "<!-- SECTION: {name} -->\n{}\n<!-- /SECTION -->\n",
            new_body.trim_end_matches('\n')
        ));
    }

    result
}

/// Update the fixed header comment lines of an existing neuron.
///
/// Patches `<!-- hash: … -->`, `<!-- last-updated: … -->`, and
/// `<!-- status: … -->` lines in-place, leaving all other content intact.
#[must_use]
pub fn update_neuron_header(content: &str, hash: &str, now: &str) -> String {
    let mut out = content
        .lines()
        .map(|line| {
            let t = line.trim_start();
            if t.starts_with("<!-- hash:") {
                format!("<!-- hash: {hash} -->")
            } else if t.starts_with("<!-- last-updated:") {
                format!("<!-- last-updated: {now} -->")
            } else if t.starts_with("<!-- status:") {
                "<!-- status: stale -->".to_string()
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out
}

/// Detect whether a line is a section open tag; return the section name if so.
pub(crate) fn section_open_name(line: &str) -> Option<&str> {
    let trimmed = line.trim();
    let rest = trimmed.strip_prefix("<!-- SECTION:")?;
    let name_part = rest.split("-->").next()?.trim();
    let name = name_part.split('|').next()?.trim();
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_sections_basic() {
        let content =
            "header\n<!-- SECTION: purpose -->\nsome purpose\n<!-- /SECTION -->\nfooter\n";
        let sections = parse_sections(content);
        assert_eq!(
            sections.get("purpose").map(String::as_str),
            Some("some purpose")
        );
    }

    #[test]
    fn parse_sections_multiple() {
        let content = "<!-- SECTION: api -->\nfn foo()\n<!-- /SECTION -->\n<!-- SECTION: pitfalls -->\nwatch out\n<!-- /SECTION -->\n";
        let sections = parse_sections(content);
        assert_eq!(sections.get("api").map(String::as_str), Some("fn foo()"));
        assert_eq!(
            sections.get("pitfalls").map(String::as_str),
            Some("watch out")
        );
    }

    #[test]
    fn parse_sections_empty_returns_empty_map() {
        assert!(parse_sections("no sections here").is_empty());
    }

    #[test]
    fn replace_section_existing() {
        let content = "pre\n<!-- SECTION: api -->\nold\n<!-- /SECTION -->\npost\n";
        let result = replace_section(content, "api", "new content");
        assert!(result.contains("new content"), "new: {result}");
        assert!(!result.contains("old"), "old body removed: {result}");
        assert!(result.contains("pre"), "prefix preserved");
        assert!(result.contains("post"), "suffix preserved");
        assert!(
            result.contains("<!-- SECTION: api -->"),
            "open tag preserved"
        );
        assert!(result.contains("<!-- /SECTION -->"), "close tag preserved");
    }

    #[test]
    fn replace_section_appends_if_missing() {
        let content = "existing content\n";
        let result = replace_section(content, "new_section", "body");
        assert!(result.contains("<!-- SECTION: new_section -->"));
        assert!(result.contains("body"));
        assert!(result.contains("existing content"), "original preserved");
    }

    #[test]
    fn replace_section_round_trip() {
        let content = "<!-- SECTION: purpose -->\noriginal\n<!-- /SECTION -->\n";
        let updated = replace_section(content, "purpose", "updated");
        let sections = parse_sections(&updated);
        assert_eq!(sections.get("purpose").map(String::as_str), Some("updated"));
    }
}
