use super::community_relation_extractors::{
    extract_hobby_signals_from_line, is_online_community_participation_line,
    parse_online_community_hobby_query,
};
use super::conversation_scan_support::scanned_conversation_lines;
use super::*;

#[derive(Clone, Debug, Eq, PartialEq)]
struct HobbyAggregate {
    hobby: String,
    score: usize,
    first_index: usize,
    evidence: Vec<String>,
}

impl NeuronIndex {
    pub(super) fn synthetic_online_community_hobby_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        let query = parse_online_community_hobby_query(task_lower)?;
        let hobbies = best_online_community_hobbies(self, query.max_items);
        (hobbies.len() == query.max_items).then_some(())?;
        let answer = match hobbies.as_slice() {
            [first, second] => format!("{} and {}", first.hobby, second.hobby),
            _ => return None,
        };
        let evidence = hobbies
            .iter()
            .flat_map(|aggregate| aggregate.evidence.clone())
            .collect::<Vec<_>>();
        self.write_synthetic_answer("online-community-hobbies", task, &answer, &evidence)
    }
}

fn best_online_community_hobbies(idx: &NeuronIndex, max_items: usize) -> Vec<HobbyAggregate> {
    let mut aggregates = HashMap::<String, HobbyAggregate>::new();
    for (conversation_index, lines) in scanned_conversation_lines(idx).enumerate() {
        let community_evidence = lines
            .iter()
            .filter_map(|line| {
                let lower = line.to_ascii_lowercase();
                is_online_community_participation_line(&lower).then_some(line.trim().to_string())
            })
            .collect::<Vec<_>>();
        if community_evidence.is_empty() {
            continue;
        }

        let mut best_in_conversation = HashMap::<String, HobbyAggregate>::new();
        for line in &lines {
            let lower = line.to_ascii_lowercase();
            for signal in extract_hobby_signals_from_line(line, &lower) {
                let key = normalized_synthetic_phrase_key(&signal.hobby);
                let entry = best_in_conversation.entry(key).or_insert(HobbyAggregate {
                    hobby: signal.hobby.clone(),
                    score: 0,
                    first_index: conversation_index,
                    evidence: Vec::new(),
                });
                entry.score += signal.score;
                push_evidence(&mut entry.evidence, &signal.evidence);
            }
        }

        for aggregate in best_in_conversation.values_mut() {
            aggregate.score += community_evidence.len() * 20;
            for evidence in &community_evidence {
                push_evidence(&mut aggregate.evidence, evidence);
            }
        }

        let Some(best) = best_in_conversation
            .into_values()
            .max_by_key(|aggregate| (aggregate.score, std::cmp::Reverse(aggregate.first_index)))
        else {
            continue;
        };

        let key = normalized_synthetic_phrase_key(&best.hobby);
        let entry = aggregates.entry(key).or_insert(HobbyAggregate {
            hobby: best.hobby.clone(),
            score: 0,
            first_index: conversation_index,
            evidence: Vec::new(),
        });
        entry.score += best.score;
        entry.first_index = entry.first_index.min(conversation_index);
        for evidence in best.evidence {
            push_evidence(&mut entry.evidence, &evidence);
        }
    }

    let mut best = aggregates.into_values().collect::<Vec<_>>();
    best.sort_by(|left, right| {
        right
            .score
            .cmp(&left.score)
            .then(left.first_index.cmp(&right.first_index))
            .then(left.hobby.cmp(&right.hobby))
    });
    best.truncate(max_items);
    best.sort_by_key(|left| left.first_index);
    best
}

fn push_evidence(out: &mut Vec<String>, evidence: &str) {
    if out.len() >= 4 || out.iter().any(|existing| existing == evidence) {
        return;
    }
    out.push(evidence.to_string());
}
