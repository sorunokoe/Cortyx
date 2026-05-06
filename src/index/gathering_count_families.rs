use super::conversation_scan_support::scanned_conversation_lines;
use super::count_support::SignatureDetail;
use super::gathering_count_extractors::{
    extract_dinner_party_attendance_details, parse_dinner_party_count_query,
};
use super::*;

impl NeuronIndex {
    pub(super) fn synthetic_dinner_party_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        let query = parse_dinner_party_count_query(task, task_lower)?;
        let candidates =
            self.collect_signature_candidates(task, &query.required_terms, 12, |line, lower| {
                !extract_dinner_party_attendance_details(line, lower).is_empty()
            });

        let best = best_scanned_dinner_party_details(self)
            .or_else(|| best_same_session_dinner_party_details(self, &candidates))?;
        self.write_synthetic_answer(
            "dinner-party-count",
            task,
            &small_cardinal_word_lower(best.details.len()),
            &best.evidence,
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DinnerPartyEvidence {
    details: Vec<SignatureDetail>,
    evidence: Vec<String>,
}

fn best_same_session_dinner_party_details(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
) -> Option<DinnerPartyEvidence> {
    candidates
        .iter()
        .filter_map(|(session_id, session_rank)| {
            let lines = idx.session_verbatim_answer_candidate_lines(session_id, usize::MAX);
            let collected = collect_dinner_party_details(lines);
            (!collected.details.is_empty()).then_some((
                session_rank * 1000 + collected.details.len() * 100 + collected.evidence.len(),
                collected,
            ))
        })
        .max_by_key(|(score, _)| *score)
        .map(|(_, collected)| collected)
}

fn best_scanned_dinner_party_details(idx: &NeuronIndex) -> Option<DinnerPartyEvidence> {
    scanned_conversation_lines(idx)
        .filter_map(|lines| {
            let collected = collect_dinner_party_details(lines);
            (!collected.details.is_empty()).then_some((
                collected.details.len() * 100 + collected.evidence.len(),
                collected,
            ))
        })
        .max_by_key(|(score, _)| *score)
        .map(|(_, collected)| collected)
}

fn collect_dinner_party_details(lines: impl IntoIterator<Item = String>) -> DinnerPartyEvidence {
    let mut displays_by_key = HashMap::new();
    let mut key_order = Vec::new();
    let mut evidence = Vec::new();

    for line in lines {
        let lower = line.to_ascii_lowercase();
        let details = extract_dinner_party_attendance_details(&line, &lower);
        let mut inserted = false;
        for detail in details {
            if detail.key.is_empty() || detail.display.is_empty() {
                continue;
            }
            if displays_by_key.contains_key(&detail.key) {
                continue;
            }
            inserted = true;
            key_order.push(detail.key.clone());
            displays_by_key.insert(detail.key, detail.display);
        }
        if inserted && !evidence.iter().any(|existing| existing == &line) {
            evidence.push(line);
        }
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
    DinnerPartyEvidence { details, evidence }
}
