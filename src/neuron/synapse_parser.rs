use std::path::PathBuf;

use super::filter::validate_synapse_path;
use super::synapse::{Synapse, SynapseType};

/// Parse the `## CROSS-REFERENCES (synapses)` section of a neuron file.
///
/// Supports the format:
/// ```markdown
/// - `path/to/other.context.md` → reason [imports]
/// ```
/// The `[type]` suffix is optional; defaults to `SemanticRelated`.
pub fn parse_synapses_from_content(content: &str) -> Vec<Synapse> {
    let mut in_section = false;
    let mut synapses = Vec::new();

    for line in content.lines() {
        if line.contains("## CROSS-REFERENCES") || line.contains("## SYNAPSES") {
            in_section = true;
            continue;
        }
        if in_section && line.starts_with("## ") {
            break;
        }
        if !in_section {
            continue;
        }
        let trimmed = line.trim_start();
        if !trimmed.starts_with("- ") && !trimmed.starts_with("* ") {
            continue;
        }

        let Some(bt_start) = line.find('`') else {
            continue;
        };
        let rest = &line[bt_start + 1..];
        let Some(bt_end) = rest.find('`') else {
            continue;
        };
        let path_str = &rest[..bt_end];
        if path_str.is_empty() {
            continue;
        }
        if validate_synapse_path(path_str).is_err() {
            tracing::warn!("Skipping unsafe synapse target in neuron content: {path_str}");
            continue;
        }
        let target = PathBuf::from(path_str);

        let raw_reason = if let Some(i) = line.find('→') {
            line[i + '→'.len_utf8()..].trim()
        } else if let Some(i) = line.find("->") {
            line[i + 2..].trim()
        } else {
            ""
        }
        .to_string();

        let (edge_type, reason) = extract_edge_type(&raw_reason);

        synapses.push(Synapse::new(target, edge_type, reason));
    }
    synapses
}

/// Detect synapse type from optional `[type]` suffix and keywords.
fn extract_edge_type(reason: &str) -> (SynapseType, String) {
    let lower = reason.to_lowercase();

    let kind = if lower.ends_with("[imports]") || lower.ends_with("[import]") {
        SynapseType::Imports
    } else if lower.ends_with("[calls]") || lower.ends_with("[call]") {
        SynapseType::Calls
    } else if lower.ends_with("[implements]") || lower.ends_with("[implement]") {
        SynapseType::Implements
    } else if lower.ends_with("[temporal]") || lower.ends_with("[follows]") {
        SynapseType::TemporalFollows
    } else if lower.ends_with("[contradicts]") || lower.ends_with("[contradict]") {
        SynapseType::Contradicts
    } else if lower.ends_with("[derived]") {
        SynapseType::Derived
    } else if lower.ends_with("[concept]") {
        SynapseType::ConceptExpands
    } else if contains_positive_reason_keyword(&lower, &["import", "imports", "depend", "depends"])
    {
        SynapseType::Imports
    } else if contains_positive_reason_keyword(
        &lower,
        &["call", "calls", "invoke", "invokes", "invoked"],
    ) {
        SynapseType::Calls
    } else if contains_positive_reason_keyword(&lower, &["implement", "implements", "implemented"])
    {
        SynapseType::Implements
    } else {
        SynapseType::SemanticRelated
    };

    let clean = if let Some(i) = reason.rfind('[') {
        reason[..i].trim().to_string()
    } else {
        reason.trim().to_string()
    };

    (kind, clean)
}

fn contains_positive_reason_keyword(reason: &str, variants: &[&str]) -> bool {
    const NEGATIONS: &[&str] = &["not", "no", "without", "never", "dont", "doesnt", "didnt"];
    let tokens: Vec<&str> = reason
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
        .collect();
    for (idx, token) in tokens.iter().enumerate() {
        if !variants.contains(token) {
            continue;
        }
        let negated = idx
            .checked_sub(1)
            .and_then(|prior| tokens.get(prior))
            .is_some_and(|prior| NEGATIONS.contains(prior))
            || idx
                .checked_sub(2)
                .and_then(|prior| tokens.get(prior))
                .is_some_and(|prior| NEGATIONS.contains(prior));
        if !negated {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_synapses_basic() {
        let content = "## CROSS-REFERENCES (synapses)\n\
                       - `.cortyx/neurons/auth_rs.context.md` → handles tokens [imports]\n";
        let synapses = parse_synapses_from_content(content);
        assert_eq!(synapses.len(), 1);
        assert_eq!(
            synapses[0].target,
            PathBuf::from(".cortyx/neurons/auth_rs.context.md")
        );
        assert_eq!(synapses[0].edge_type, SynapseType::Imports);
        assert_eq!(synapses[0].reason, "handles tokens");
    }

    #[test]
    fn parse_synapses_defaults_to_semantic() {
        let content = "## CROSS-REFERENCES (synapses)\n\
                       - `.cortyx/neurons/ui_rs.context.md` → related UI code\n";
        let synapses = parse_synapses_from_content(content);
        assert_eq!(synapses.len(), 1);
        assert_eq!(synapses[0].edge_type, SynapseType::SemanticRelated);
    }

    #[test]
    fn parse_synapses_no_section() {
        let content = "No cross references here";
        assert!(parse_synapses_from_content(content).is_empty());
    }

    #[test]
    fn extract_edge_type_respects_word_boundaries_and_negation() {
        let (edge, _) = extract_edge_type("the caller should be aware of caching");
        assert_eq!(edge, SynapseType::SemanticRelated);

        let (edge, _) = extract_edge_type("this module does not import anything from auth");
        assert_eq!(edge, SynapseType::SemanticRelated);

        let (edge, _) = extract_edge_type("module imports auth helpers");
        assert_eq!(edge, SynapseType::Imports);
    }

    #[test]
    fn parse_synapses_empty_section() {
        let content = "## CROSS-REFERENCES (synapses)\n[TODO]\n";
        assert!(parse_synapses_from_content(content).is_empty());
    }

    #[test]
    fn parse_synapses_ignores_malformed_lines() {
        let content = "## CROSS-REFERENCES (synapses)\n\
                       - no backticks here\n\
                       - `valid.context.md` → ok\n";
        let synapses = parse_synapses_from_content(content);
        assert_eq!(synapses.len(), 1);
    }

    #[test]
    fn parse_synapses_multiple_types() {
        let content = "## CROSS-REFERENCES (synapses)\n\
                       - `a.context.md` → provides types [implements]\n\
                       - `b.context.md` → next session [temporal]\n\
                       - `c.context.md` → loosely related\n";
        let synapses = parse_synapses_from_content(content);
        assert_eq!(synapses.len(), 3);
        assert_eq!(synapses[0].edge_type, SynapseType::Implements);
        assert_eq!(synapses[1].edge_type, SynapseType::TemporalFollows);
        assert_eq!(synapses[2].edge_type, SynapseType::SemanticRelated);
    }
}
