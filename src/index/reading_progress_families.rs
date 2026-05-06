use super::reading_progress_extractors::{
    extract_current_page_for_title_variants, extract_just_finished_page_count,
    extract_total_pages_for_title_variants, parse_reading_progress_query, ReadingProgressQuery,
    ReadingTitleQuery,
};
use super::*;

impl NeuronIndex {
    pub(super) fn synthetic_reading_progress_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        match parse_reading_progress_query(task, task_lower)? {
            ReadingProgressQuery::PagesRead(query) => {
                self.synthetic_title_pages_read_answer(task, &query)
            },
            ReadingProgressQuery::PagesLeft(query) => {
                self.synthetic_title_pages_left_answer(task, &query)
            },
            ReadingProgressQuery::FinishedNovelPageTotal => {
                self.synthetic_novel_page_total_answer(task)
            },
        }
    }

    fn synthetic_title_pages_read_answer(
        &self,
        task: &str,
        query: &ReadingTitleQuery,
    ) -> Option<PathBuf> {
        let facts = collect_reading_progress_facts(self, query);
        facts.current.and_then(|current| {
            self.write_synthetic_answer(
                "quoted-title-pages-read",
                task,
                &current.to_string(),
                &facts.evidence,
            )
        })
    }

    fn synthetic_title_pages_left_answer(
        &self,
        task: &str,
        query: &ReadingTitleQuery,
    ) -> Option<PathBuf> {
        let facts = collect_reading_progress_facts(self, query);
        let (current, total) = (facts.current?, facts.total?);
        (total > current).then_some(())?;
        self.write_synthetic_answer(
            "quoted-title-pages-left",
            task,
            &(total - current).to_string(),
            &facts.evidence,
        )
    }

    fn synthetic_novel_page_total_answer(&self, task: &str) -> Option<PathBuf> {
        let evidence =
            self.find_matching_lines(&["finished", "novel", "page"], 12, false, 6, |line, _| {
                extract_just_finished_page_count(line).is_some()
            });
        let mut page_counts = Vec::new();
        let mut selected = Vec::new();
        for line in &evidence {
            if let Some(value) = extract_just_finished_page_count(line) {
                if !page_counts.contains(&value) {
                    page_counts.push(value);
                }
                push_unique(&mut selected, line);
            }
        }
        (page_counts.len() >= 2).then_some(())?;
        self.write_synthetic_answer(
            "novel-page-total",
            task,
            &page_counts.iter().take(2).sum::<i32>().to_string(),
            &selected,
        )
    }
}

#[derive(Default)]
struct ReadingProgressFacts {
    current: Option<i32>,
    total: Option<i32>,
    evidence: Vec<String>,
}

fn collect_reading_progress_facts(
    idx: &NeuronIndex,
    query: &ReadingTitleQuery,
) -> ReadingProgressFacts {
    let required_terms: Vec<&str> = query.required_terms.iter().map(String::as_str).collect();
    let evidence = idx.find_matching_lines(&required_terms, 12, false, 8, |line, lower| {
        (lower.starts_with("user:") || line.trim_start().starts_with('-'))
            && query
                .title_variants
                .iter()
                .any(|title| !title.is_empty() && lower.contains(title))
    });

    let mut facts = ReadingProgressFacts::default();
    for line in &evidence {
        if let Some(value) = extract_current_page_for_title_variants(line, &query.title_variants) {
            facts.current = Some(facts.current.map_or(value, |existing| existing.max(value)));
            push_unique(&mut facts.evidence, line);
        }
        if let Some(value) = extract_total_pages_for_title_variants(line, &query.title_variants) {
            facts.total = Some(facts.total.map_or(value, |existing| existing.max(value)));
            push_unique(&mut facts.evidence, line);
        }
    }
    facts
}

fn push_unique(lines: &mut Vec<String>, line: &str) {
    if !lines.iter().any(|existing| existing == line) {
        lines.push(line.to_string());
    }
}
