use super::conversation_scan_support::scanned_conversation_lines;
use super::podcast_count_extractors::{
    extract_podcast_episode_fact_from_line, parse_podcast_episode_total_query, PodcastEpisodeFact,
    PodcastEpisodeTotalQuery,
};
use super::*;

impl NeuronIndex {
    pub(super) fn synthetic_podcast_episode_total_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        let query = parse_podcast_episode_total_query(task, task_lower)?;
        let candidates =
            self.collect_signature_candidates(task, &query.required_terms, 12, |line, lower| {
                query.titles.iter().any(|focus| {
                    extract_podcast_episode_fact_from_line(line, lower, focus).is_some()
                })
            });

        let facts = best_scanned_podcast_facts(self, &query)
            .or_else(|| best_same_session_podcast_facts(self, &candidates, &query))?;
        let total = facts.values().map(|fact| fact.count).sum::<i32>();
        self.write_synthetic_answer(
            "podcast-episode-total",
            task,
            &total.to_string(),
            &ordered_podcast_evidence(&facts, &query),
        )
    }
}

fn best_same_session_podcast_facts(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &PodcastEpisodeTotalQuery,
) -> Option<HashMap<String, PodcastEpisodeFact>> {
    candidates
        .iter()
        .filter_map(|(session_id, session_rank)| {
            let facts = collect_podcast_facts(
                idx.session_verbatim_answer_candidate_lines(session_id, usize::MAX),
                query,
            );
            covers_podcast_query(&facts, query).then_some((
                session_rank * 1000
                    + facts.values().map(|fact| fact.score).sum::<usize>()
                    + facts
                        .values()
                        .map(|fact| fact.count.max(0) as usize)
                        .sum::<usize>()
                        * 10,
                facts,
            ))
        })
        .max_by_key(|(score, _)| *score)
        .map(|(_, facts)| facts)
}

fn best_scanned_podcast_facts(
    idx: &NeuronIndex,
    query: &PodcastEpisodeTotalQuery,
) -> Option<HashMap<String, PodcastEpisodeFact>> {
    scanned_conversation_lines(idx)
        .filter_map(|lines| {
            let facts = collect_podcast_facts(lines, query);
            covers_podcast_query(&facts, query).then_some((
                facts.values().map(|fact| fact.score).sum::<usize>()
                    + facts
                        .values()
                        .map(|fact| fact.count.max(0) as usize)
                        .sum::<usize>()
                        * 10,
                facts,
            ))
        })
        .max_by_key(|(score, _)| *score)
        .map(|(_, facts)| facts)
}

fn collect_podcast_facts(
    lines: impl IntoIterator<Item = String>,
    query: &PodcastEpisodeTotalQuery,
) -> HashMap<String, PodcastEpisodeFact> {
    let mut best = HashMap::new();
    for line in lines {
        let lower = line.to_ascii_lowercase();
        for focus in &query.titles {
            let Some(fact) = extract_podcast_episode_fact_from_line(&line, &lower, focus) else {
                continue;
            };
            upsert_podcast_fact(&mut best, fact);
        }
    }
    best
}

fn covers_podcast_query(
    facts: &HashMap<String, PodcastEpisodeFact>,
    query: &PodcastEpisodeTotalQuery,
) -> bool {
    query
        .titles
        .iter()
        .all(|focus| facts.contains_key(&focus.key))
}

fn upsert_podcast_fact(best: &mut HashMap<String, PodcastEpisodeFact>, fact: PodcastEpisodeFact) {
    match best.get(&fact.key) {
        Some(existing)
            if existing.score > fact.score
                || (existing.score == fact.score && existing.count >= fact.count) => {},
        _ => {
            best.insert(fact.key.clone(), fact);
        },
    }
}

fn ordered_podcast_evidence(
    facts: &HashMap<String, PodcastEpisodeFact>,
    query: &PodcastEpisodeTotalQuery,
) -> Vec<String> {
    let mut evidence = Vec::new();
    for focus in &query.titles {
        let Some(line) = facts.get(&focus.key).map(|fact| fact.evidence.clone()) else {
            continue;
        };
        if !evidence.iter().any(|existing| existing == &line) {
            evidence.push(line);
        }
    }
    evidence
}
