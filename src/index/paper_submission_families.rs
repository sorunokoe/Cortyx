use super::conversation_scan_support::grouped_verbatim_candidate_lines;
use super::paper_submission_extractors::{
    extract_paper_submission_venue, extract_venue_submission_date, parse_paper_submission_query,
    PaperSubmissionQuery, PaperVenueCandidate, VenueDateCandidate,
};
use super::*;

impl NeuronIndex {
    pub(super) fn synthetic_paper_submission_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        let query = parse_paper_submission_query(task_lower)?;
        let answer = best_grouped_paper_submission_date(self, &query)?;
        self.write_synthetic_answer(
            "paper-submission-date",
            task,
            &answer.date,
            &answer.evidence,
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ResolvedPaperSubmissionDate {
    date: String,
    score: usize,
    evidence: Vec<String>,
}

fn best_grouped_paper_submission_date(
    idx: &NeuronIndex,
    query: &PaperSubmissionQuery,
) -> Option<ResolvedPaperSubmissionDate> {
    grouped_verbatim_candidate_lines(idx)
        .into_values()
        .filter_map(|lines| resolve_paper_submission_date(lines, query))
        .max_by_key(|answer| answer.score)
}

fn resolve_paper_submission_date(
    lines: Vec<(String, bool)>,
    query: &PaperSubmissionQuery,
) -> Option<ResolvedPaperSubmissionDate> {
    let mut best_venue: Option<PaperVenueCandidate> = None;
    let mut best_date: Option<VenueDateCandidate> = None;

    for (line, is_summary) in &lines {
        let lower = line.to_ascii_lowercase();
        let Some(mut venue_candidate) = extract_paper_submission_venue(line, &lower, query) else {
            continue;
        };
        if *is_summary {
            venue_candidate.score += 2;
        }
        let should_replace = best_venue
            .as_ref()
            .map(|best| venue_candidate.score > best.score)
            .unwrap_or(true);
        if should_replace {
            best_venue = Some(venue_candidate);
        }
    }

    let venue = best_venue?;
    for (line, is_summary) in lines {
        let lower = line.to_ascii_lowercase();
        let Some(mut date_candidate) = extract_venue_submission_date(&line, &lower, &venue.venue)
        else {
            continue;
        };
        if is_summary {
            date_candidate.score += 2;
        }
        let should_replace = best_date
            .as_ref()
            .map(|best| date_candidate.score > best.score)
            .unwrap_or(true);
        if should_replace {
            best_date = Some(date_candidate);
        }
    }

    let date = best_date?;
    Some(ResolvedPaperSubmissionDate {
        date: date.date,
        score: venue.score + date.score + 8,
        evidence: dedupe_paper_evidence([venue.evidence, date.evidence]),
    })
}

fn dedupe_paper_evidence<I>(evidence: I) -> Vec<String>
where
    I: IntoIterator<Item = String>,
{
    let mut deduped = Vec::new();
    let mut seen = HashSet::new();
    for line in evidence {
        if seen.insert(line.clone()) {
            deduped.push(line);
        }
    }
    deduped
}
