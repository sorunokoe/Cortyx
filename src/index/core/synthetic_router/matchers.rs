use super::*;

impl NeuronIndex {

    pub(in crate::index) fn matching_verbatim_texts(
        &self,
        required_terms: &[&str],
        limit: usize,
    ) -> Vec<(PathBuf, String)> {
        let mut matches: Vec<(usize, bool, PathBuf)> = self
            .entries
            .iter()
            .filter(|entry| matches!(entry.kind, NeuronKind::Verbatim))
            .filter_map(|entry| {
                let overlap = required_terms
                    .iter()
                    .filter(|term| entry.term_freq.contains_key(**term))
                    .count();
                if overlap == 0 {
                    return None;
                }
                Some((
                    overlap,
                    is_session_summary_path(&entry.neuron_path),
                    entry.neuron_path.clone(),
                ))
            })
            .collect();

        matches.sort_unstable_by(|a, b| {
            b.0.cmp(&a.0)
                .then_with(|| b.1.cmp(&a.1))
                .then_with(|| a.2.cmp(&b.2))
        });

        matches
            .into_iter()
            .take(limit)
            .filter_map(|(_, _, path)| {
                std::fs::read_to_string(&path)
                    .ok()
                    .map(|content| (path, strip_query_surface_section(&content)))
            })
            .collect()
    }

    pub(in crate::index) fn find_matching_lines<F>(
        &self,
        required_terms: &[&str],
        limit: usize,
        summary_only: bool,
        max_lines: usize,
        mut predicate: F,
    ) -> Vec<String>
    where
        F: FnMut(&str, &str) -> bool,
    {
        let mut lines = Vec::new();
        for (path, content) in self.matching_verbatim_texts(required_terms, limit) {
            if summary_only && !is_session_summary_path(&path) {
                continue;
            }
            for raw_line in content.lines() {
                let line = raw_line.trim();
                if line.is_empty() {
                    continue;
                }
                let lower = line.to_ascii_lowercase();
                if predicate(line, &lower) && !lines.iter().any(|existing| existing == line) {
                    lines.push(line.to_string());
                    if lines.len() >= max_lines {
                        return lines;
                    }
                }
            }
        }
        lines
    }
}
