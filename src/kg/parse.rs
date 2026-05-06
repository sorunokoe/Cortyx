use super::fact::{split_table_row, KgFact};

/// Parse the `## facts` pipe-table section from neuron markdown.
pub(super) fn parse_facts_table(content: &str) -> Vec<KgFact> {
    let mut in_facts = false;
    let mut facts = Vec::new();

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed == "## facts" {
            in_facts = true;
            continue;
        }
        if in_facts {
            if trimmed.starts_with("## ") {
                break;
            }
            if !trimmed.starts_with('|') {
                continue;
            }
            let cols = split_table_row(trimmed);
            if cols.len() < 4 {
                continue;
            }
            if cols[0] == "predicate" {
                continue;
            }
            if cols.iter().all(|c| c.chars().all(|ch| ch == '-')) {
                continue;
            }
            facts.push(KgFact {
                predicate: cols[0].clone(),
                value: cols[1].clone(),
                valid_from: cols[2].clone(),
                ended: cols[3].clone(),
            });
        }
    }
    facts
}
