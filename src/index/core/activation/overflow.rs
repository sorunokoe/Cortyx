use super::*;

impl NeuronIndex {
    /// Like `get_contexts` but also returns compressed (headline-only) neurons that
    /// exceeded the token budget.
    ///
    /// Returns `(full_neurons, overflow_neurons)`.  `overflow_neurons` is a vec of
    /// `(path, headline)` pairs — the headline is the first content line of the
    /// `## purpose` section (or a stub fallback).  Callers can inject the headlines
    /// into the prompt as low-cost navigation hints without the full neuron body.
    ///
    /// `min_confidence`: when `Some(threshold)`, returns `([], [])` immediately if the
    /// top raw BM25 score for `task` is below `threshold`.  Use this to implement the
    /// LongMemEval *abstention* signal — the system should say "no relevant memory"
    /// rather than hallucinating a low-quality match.  Typical threshold: `0.5`
    /// (= `LOW_CONFIDENCE_THRESHOLD`).  Pass `None` to disable (default behaviour).
    pub fn get_contexts_with_overflow(
        &self,
        task: &str,
        max_tokens: usize,
        module: Option<&str>,
        kind: Option<&str>,
        min_confidence: Option<f32>,
        multi_hop: bool,
    ) -> (Vec<PathBuf>, Vec<(PathBuf, String)>) {
        self.get_contexts_with_overflow_and_temporal_bias(
            task,
            max_tokens,
            module,
            kind,
            min_confidence,
            multi_hop,
            None,
        )
    }

    pub(crate) fn get_contexts_with_overflow_and_temporal_bias(
        &self,
        task: &str,
        max_tokens: usize,
        module: Option<&str>,
        kind: Option<&str>,
        min_confidence: Option<f32>,
        multi_hop: bool,
        temporal_bias: Option<f32>,
    ) -> (Vec<PathBuf>, Vec<(PathBuf, String)>) {
        let Ok(query) = QueryText::new(task) else {
            return (Vec::new(), Vec::new());
        };
        // Abstention signal: if caller set a minimum confidence threshold and the
        // best BM25 score for this query is below it, return nothing immediately.
        // This is critical for LongMemEval "absent" questions (20% of the dataset),
        // where returning a low-relevance neuron counts as a false positive.
        if let Some(threshold) = min_confidence {
            if self.peek_max_bm25_score(query.as_str()) < threshold {
                tracing::debug!(
                    task = query.as_str(),
                    threshold,
                    "Abstention: top BM25 score below min_confidence — returning empty."
                );
                return (vec![], vec![]);
            }
        }

        // F1: Task Complexity Adaptive Budget
        //
        // Scale max_tokens by [0.5, 1.5] based on query complexity:
        //   - BM25 breadth: how many distinct terms have posting-list hits
        //   - Module spread: unique modules in top candidates
        //   - Synapse depth: whether candidates have outgoing synapses
        //
        // Simple queries (breadth=1, no synapses) → 0.5× budget (saves tokens)
        // Complex queries (broad match, cross-module) → 1.5× budget
        let terms = tokenize(query.as_str());
        let complexity = self.compute_task_complexity(&terms);
        // F2: apply session-history budget scale on top of F1 complexity scale
        let history_scale = self.adaptive_budget_scale();
        let adjusted_max = ((max_tokens as f32 * complexity * history_scale) as usize)
            .max(512)
            .min(8192.max(max_tokens * 2));
        tracing::debug!(
            task,
            complexity,
            history_scale,
            original_max = max_tokens,
            adjusted_max,
            "F1+F2: adaptive token budget"
        );

        let candidate_set: HashSet<usize> = {
            let mut s = HashSet::new();
            for term in &terms {
                if let Some(idxs) = self.retrieval.posting_list.get(term) {
                    s.extend(idxs);
                }
            }
            s
        };

        // Run the full activation pipeline via get_contexts with an enormous budget,
        // then re-split. Slightly wasteful but keeps logic DRY.
        //
        // Collected as Vec so the multi-hop block can reference the pre-budget-split
        // ranked order (all_ordered[..5]) without re-running the pipeline.
        let all_ordered: Vec<PathBuf> =
            self.get_contexts_with_temporal_bias(task, usize::MAX / 2, module, kind, temporal_bias);

        let mut full = Vec::new();
        let mut overflow = Vec::new();
        let mut used = 0usize;

        for path in all_ordered.iter().cloned() {
            let tokens = self.entry_by_path(&path).map(|e| e.tokens).unwrap_or(200);
            if used + tokens <= adjusted_max || full.is_empty() {
                used += tokens;
                full.push(path);
            } else {
                // Collect headline for overflow neuron
                let headline = neuron_headline_for(&path);
                overflow.push((path, headline));
            }
        }

        // Multi-hop retrieval: expand from the top-5 pre-budget-split retrieval hits
        // to discover neurons reachable via multiple semantic paths.
        //
        // Improvement over prior top-1 expansion: seeding from all top-5 hits captures
        // terms from multiple subtopics, improving recall for complex multi-hop queries
        // (recursiveMAS iterative deepening principle applied heuristically).
        //
        // All novel neurons go to overflow (lower-priority hints), so full results and
        // their ranking are unchanged — recall can only increase, not decrease.
        if multi_hop && !all_ordered.is_empty() {
            let seed_entries: Vec<&BM25Entry> = all_ordered
                .iter()
                .take(5)
                .filter_map(|p| self.entry_by_path(p))
                .collect();

            if !seed_entries.is_empty() {
                let mut hop_terms = terms.clone();

                for entry in &seed_entries {
                    // Sort clouds before truncation for determinism across runs.
                    let mut cloud: Vec<&String> = entry.concept_cloud.iter().collect();
                    cloud.sort();
                    hop_terms.extend(cloud.into_iter().take(5).cloned());

                    let mut syns: Vec<&String> = entry.synonym_cloud.iter().collect();
                    syns.sort();
                    hop_terms.extend(syns.into_iter().take(3).cloned());
                }

                // Gather TF-IDF terms from all seeds; deduplicate by keeping max freq per
                // term via BTreeMap (lexicographic key order → deterministic output).
                let already: HashSet<&str> = hop_terms.iter().map(|s| s.as_str()).collect();
                let mut tfidf_best: std::collections::BTreeMap<String, f32> =
                    std::collections::BTreeMap::new();
                for entry in &seed_entries {
                    for (t, f) in &entry.term_freq {
                        let f = f.get();
                        if t.len() >= 4 && !already.contains(t.as_str()) {
                            tfidf_best
                                .entry(t.clone())
                                .and_modify(|v| *v = v.max(f))
                                .or_insert(f);
                        }
                    }
                }
                // Sort by (freq DESC, term ASC) for stable ordering across runs.
                let mut tfidf: Vec<(f32, String)> =
                    tfidf_best.into_iter().map(|(t, f)| (f, t)).collect();
                tfidf.sort_unstable_by(|a, b| {
                    b.0.total_cmp(&a.0).then(a.1.as_str().cmp(b.1.as_str()))
                });
                hop_terms.extend(tfidf.into_iter().take(15).map(|(_, t)| t));

                hop_terms.sort();
                hop_terms.dedup();

                let expanded_task = hop_terms.join(" ");
                let second_pass = self.get_contexts_with_temporal_bias(
                    &expanded_task,
                    usize::MAX / 2,
                    module,
                    kind,
                    temporal_bias,
                );

                let already_included: HashSet<&PathBuf> =
                    full.iter().chain(overflow.iter().map(|(p, _)| p)).collect();
                // Cap novel overflow additions to avoid explosion on broad expanded queries.
                let novel: Vec<(PathBuf, String)> = second_pass
                    .into_iter()
                    .filter(|p| !already_included.contains(p))
                    .take(25)
                    .map(|p| {
                        let headline = neuron_headline_for(&p);
                        (p, headline)
                    })
                    .collect();

                if !novel.is_empty() {
                    tracing::debug!(
                        count = novel.len(),
                        seeds = seed_entries.len(),
                        "Multi-hop 2nd pass: injected additional candidate neurons \
                         (top-{} seed expansion)",
                        seed_entries.len()
                    );
                    overflow.extend(novel);
                }
            }
        }

        let _ = candidate_set; // suppress unused warning
        (full, overflow)
    }

    /// F1: Compute task complexity as a [0.5, 1.5] budget scale factor.
    ///
    /// Inputs:
    /// - BM25 breadth: fraction of query terms that hit the posting list (term coverage)
    /// - Module spread: unique module count in top-10 candidates (cross-module indicator)
    /// - Synapse depth: fraction of top candidates with outgoing synapses (graph richness)
    ///
    /// Formula: clamp(0.5 + breadth * 0.3 + spread * 0.4 + depth * 0.3, 0.5, 1.5)
    pub(in crate::index) fn compute_task_complexity(&self, terms: &[String]) -> f32 {
        if terms.is_empty() {
            return 1.0;
        }

        // Breadth: fraction of query terms with any posting-list hit
        let hit_terms = terms
            .iter()
            .filter(|t| self.retrieval.posting_list.contains_key(t.as_str()))
            .count();
        let breadth = hit_terms as f32 / terms.len() as f32;

        // Candidate set for spread/depth analysis
        let mut candidates: HashSet<usize> = HashSet::new();
        for t in terms {
            if let Some(idxs) = self.retrieval.posting_list.get(t.as_str()) {
                candidates.extend(idxs.iter().take(10));
            }
        }

        // Spread: unique modules among top candidates (normalized by 3)
        let unique_modules: HashSet<Option<&str>> = candidates
            .iter()
            .filter_map(|&i| self.retrieval.entries.get(i))
            .map(|e| e.module.as_deref())
            .collect();
        let spread = ((unique_modules.len() as f32 - 1.0) / 3.0).clamp(0.0, 1.0);

        // Depth: fraction of candidates that have outgoing synapses
        let with_synapses = candidates
            .iter()
            .filter_map(|&i| self.retrieval.entries.get(i))
            .filter(|e| !e.synapses.is_empty())
            .count();
        let depth = if candidates.is_empty() {
            0.0
        } else {
            with_synapses as f32 / candidates.len() as f32
        };

        (0.5 + breadth * 0.3 + spread * 0.4 + depth * 0.3).clamp(0.5, 1.5)
    }
}
