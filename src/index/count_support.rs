use super::*;
use std::num::NonZeroUsize;

pub(super) struct DistinctSignatureCountConfig<FMatch, FExtract> {
    pub(super) required_owned: Vec<String>,
    pub(super) candidate_limit: NonZeroUsize,
    pub(super) max_lines: NonZeroUsize,
    pub(super) evidence_limit: NonZeroUsize,
    pub(super) line_match: FMatch,
    pub(super) extract: FExtract,
}

pub(super) struct DistinctSignatureDetailsConfig<FMatch, FExtract> {
    pub(super) required_owned: Vec<String>,
    pub(super) candidate_limit: NonZeroUsize,
    pub(super) max_lines: NonZeroUsize,
    pub(super) evidence_limit: NonZeroUsize,
    pub(super) line_match: FMatch,
    pub(super) extract: FExtract,
}

pub(super) struct SignatureQuantitySumConfig<FMatch, FExtract> {
    pub(super) required_owned: Vec<String>,
    pub(super) candidate_limit: NonZeroUsize,
    pub(super) max_lines: NonZeroUsize,
    pub(super) evidence_limit: NonZeroUsize,
    pub(super) line_match: FMatch,
    pub(super) extract: FExtract,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SignatureDetail {
    pub(super) key: String,
    pub(super) display: String,
}

impl SignatureDetail {
    pub(super) fn new(key: impl Into<String>, display: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            display: display.into(),
        }
    }
}

impl NeuronIndex {
    pub(super) fn best_distinct_signature_count<FMatch, FExtract>(
        &self,
        task: &str,
        config: DistinctSignatureCountConfig<FMatch, FExtract>,
    ) -> Option<(usize, Vec<String>)>
    where
        FMatch: for<'a> Fn(&'a str, &'a str) -> bool,
        FExtract: for<'a> Fn(&'a str, &'a str) -> Vec<String>,
    {
        let DistinctSignatureCountConfig {
            required_owned,
            candidate_limit,
            max_lines,
            evidence_limit,
            line_match,
            extract,
        } = config;
        let candidate_limit = candidate_limit.get();
        let max_lines = max_lines.get();
        let evidence_limit = evidence_limit.get();
        let candidates = self.collect_signature_candidates(
            task,
            &required_owned,
            candidate_limit,
            |line, lower| line_match(line, lower) && !extract(line, lower).is_empty(),
        );

        let mut best: Option<(usize, usize, Vec<String>)> = None;
        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, max_lines, |line, lower| {
                line_match(line, lower) && !extract(line, lower).is_empty()
            });
            let mut items = HashSet::new();
            let mut evidence = Vec::new();
            for line in lines {
                let lower = line.to_ascii_lowercase();
                let mut inserted = false;
                for item in extract(&line, &lower) {
                    inserted |= items.insert(item);
                }
                if inserted
                    && evidence.len() < evidence_limit
                    && !evidence.iter().any(|existing| existing == &line)
                {
                    evidence.push(line);
                }
            }
            if items.is_empty() {
                continue;
            }

            let count = items.len();
            let score = session_rank * 1000 + count * 100 + evidence.len();
            let should_replace = best
                .as_ref()
                .map(|(best_score, best_count, _)| {
                    score > *best_score || (score == *best_score && count > *best_count)
                })
                .unwrap_or(true);
            if should_replace {
                best = Some((score, count, evidence));
            }
        }

        best.map(|(_, count, evidence)| (count, evidence))
    }

    pub(super) fn best_distinct_signature_details<FMatch, FExtract>(
        &self,
        task: &str,
        config: DistinctSignatureDetailsConfig<FMatch, FExtract>,
    ) -> Option<(Vec<SignatureDetail>, Vec<String>)>
    where
        FMatch: for<'a> Fn(&'a str, &'a str) -> bool,
        FExtract: for<'a> Fn(&'a str, &'a str) -> Vec<SignatureDetail>,
    {
        let DistinctSignatureDetailsConfig {
            required_owned,
            candidate_limit,
            max_lines,
            evidence_limit,
            line_match,
            extract,
        } = config;
        let candidate_limit = candidate_limit.get();
        let max_lines = max_lines.get();
        let evidence_limit = evidence_limit.get();
        let candidates = self.collect_signature_candidates(
            task,
            &required_owned,
            candidate_limit,
            |line, lower| line_match(line, lower) && !extract(line, lower).is_empty(),
        );

        let mut best: Option<(usize, usize, Vec<SignatureDetail>, Vec<String>)> = None;
        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, max_lines, |line, lower| {
                line_match(line, lower) && !extract(line, lower).is_empty()
            });
            let mut displays_by_key: HashMap<String, String> = HashMap::new();
            let mut key_order = Vec::new();
            let mut evidence = Vec::new();
            for line in lines {
                let lower = line.to_ascii_lowercase();
                let mut inserted = false;
                for detail in extract(&line, &lower) {
                    if detail.key.is_empty() || detail.display.is_empty() {
                        continue;
                    }
                    if let Some(existing_display) = displays_by_key.get_mut(&detail.key) {
                        if better_signature_display(existing_display, &detail.display) {
                            *existing_display = detail.display;
                        }
                        continue;
                    }
                    inserted = true;
                    key_order.push(detail.key.clone());
                    displays_by_key.insert(detail.key, detail.display);
                }
                if inserted
                    && evidence.len() < evidence_limit
                    && !evidence.iter().any(|existing| existing == &line)
                {
                    evidence.push(line);
                }
            }
            if key_order.is_empty() {
                continue;
            }

            let details = key_order
                .into_iter()
                .filter_map(|key| {
                    displays_by_key
                        .get(&key)
                        .cloned()
                        .map(|display| SignatureDetail { key, display })
                })
                .collect::<Vec<_>>();
            let count = details.len();
            let score = session_rank * 1000 + count * 100 + evidence.len();
            let should_replace = best
                .as_ref()
                .map(|(best_score, best_count, _, _)| {
                    score > *best_score || (score == *best_score && count > *best_count)
                })
                .unwrap_or(true);
            if should_replace {
                best = Some((score, count, details, evidence));
            }
        }

        best.map(|(_, _, details, evidence)| (details, evidence))
    }

    pub(super) fn best_signature_quantity_sum<FMatch, FExtract>(
        &self,
        task: &str,
        config: SignatureQuantitySumConfig<FMatch, FExtract>,
    ) -> Option<(usize, Vec<String>)>
    where
        FMatch: for<'a> Fn(&'a str, &'a str) -> bool,
        FExtract: for<'a> Fn(&'a str, &'a str) -> Vec<(String, usize)>,
    {
        let SignatureQuantitySumConfig {
            required_owned,
            candidate_limit,
            max_lines,
            evidence_limit,
            line_match,
            extract,
        } = config;
        let candidate_limit = candidate_limit.get();
        let max_lines = max_lines.get();
        let evidence_limit = evidence_limit.get();
        let candidates = self.collect_signature_candidates(
            task,
            &required_owned,
            candidate_limit,
            |line, lower| line_match(line, lower) && !extract(line, lower).is_empty(),
        );

        let mut best: Option<(usize, usize, Vec<String>)> = None;
        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, max_lines, |line, lower| {
                line_match(line, lower) && !extract(line, lower).is_empty()
            });
            let mut quantities = HashMap::new();
            let mut evidence = Vec::new();
            for line in lines {
                let lower = line.to_ascii_lowercase();
                let mut inserted = false;
                for (key, quantity) in extract(&line, &lower) {
                    if key.is_empty() || quantity == 0 {
                        continue;
                    }
                    match quantities.get_mut(&key) {
                        Some(existing) if quantity > *existing => {
                            *existing = quantity;
                            inserted = true;
                        },
                        Some(_) => {},
                        None => {
                            quantities.insert(key, quantity);
                            inserted = true;
                        },
                    }
                }
                if inserted
                    && evidence.len() < evidence_limit
                    && !evidence.iter().any(|existing| existing == &line)
                {
                    evidence.push(line);
                }
            }
            if quantities.is_empty() {
                continue;
            }

            let total = quantities.values().sum::<usize>();
            let score = session_rank * 1000 + total * 100 + quantities.len() * 10 + evidence.len();
            let should_replace = best
                .as_ref()
                .map(|(best_score, best_total, _)| {
                    score > *best_score || (score == *best_score && total > *best_total)
                })
                .unwrap_or(true);
            if should_replace {
                best = Some((score, total, evidence));
            }
        }

        best.map(|(_, total, evidence)| (total, evidence))
    }

    pub(super) fn collect_signature_candidates<FProbe>(
        &self,
        task: &str,
        required_owned: &[String],
        candidate_limit: usize,
        has_match: FProbe,
    ) -> Vec<(String, usize)>
    where
        FProbe: for<'a> Fn(&'a str, &'a str) -> bool,
    {
        let required_terms: Vec<&str> = required_owned.iter().map(String::as_str).collect();
        let mut candidates = self
            .session_ids_matching_line(|line, lower| has_match(line, lower))
            .into_iter()
            .map(|session_id| (session_id, 0usize))
            .collect::<Vec<_>>();

        for (session_id, score) in
            self.candidate_session_ids_by_line_overlap(required_owned, candidate_limit)
        {
            upsert_candidate_score(&mut candidates, session_id, score);
        }
        for (idx, session_id) in self
            .candidate_session_ids(task, &required_terms, candidate_limit)
            .into_iter()
            .enumerate()
        {
            upsert_candidate_score(
                &mut candidates,
                session_id,
                candidate_limit.saturating_sub(idx),
            );
        }

        candidates
    }
}

pub(super) fn nz(value: usize) -> NonZeroUsize {
    debug_assert!(value > 0, "count-family limits must be non-zero");
    NonZeroUsize::new(value).unwrap_or(NonZeroUsize::MIN)
}

fn better_signature_display(existing: &str, candidate: &str) -> bool {
    let existing_has_pair = existing.contains(" and ");
    let candidate_has_pair = candidate.contains(" and ");
    candidate_has_pair && !existing_has_pair || candidate.len() > existing.len()
}

fn upsert_candidate_score(candidates: &mut Vec<(String, usize)>, session_id: String, score: usize) {
    if let Some((_, existing_score)) = candidates
        .iter_mut()
        .find(|(existing_session_id, _)| existing_session_id == &session_id)
    {
        *existing_score = (*existing_score).max(score);
    } else {
        candidates.push((session_id, score));
    }
}
