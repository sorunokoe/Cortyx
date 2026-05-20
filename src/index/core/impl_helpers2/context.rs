use super::*;

impl NeuronIndex {
    /// Build a bounded, read-only reasoning report around already-selected evidence paths.
    ///
    /// This intentionally operates after retrieval: callers provide the selected evidence
    /// seeds and the reasoner only explores a small adjacency neighborhood rooted at those
    /// seeds, leaving the BM25 hot path unchanged.
    pub fn reason_over_paths(
        &self,
        seeds: &[(PathBuf, f32)],
        options: TraversalOptions,
    ) -> ReasoningReport {
        let seeds: Vec<(PathBuf, f32)> = seeds
            .iter()
            .filter(|(_, score)| *score > 0.0)
            .map(|(path, score)| (path.clone(), *score))
            .collect();
        if seeds.is_empty() {
            return ReasoningReport::default();
        }

        let mut included = HashSet::new();
        let mut queue = VecDeque::new();
        for (path, _) in &seeds {
            if included.insert(path.clone()) {
                queue.push_back((path.clone(), 0_u8));
            }
        }

        while let Some((path, depth)) = queue.pop_front() {
            if depth >= options.max_hops {
                continue;
            }

            let Some(neighbors) = self.retrieval.adjacency.get(&path) else {
                continue;
            };
            for synapse in neighbors {
                if included.insert(synapse.target.clone()) {
                    queue.push_back((synapse.target.clone(), depth + 1));
                }
            }
        }

        let neurons = included
            .iter()
            .filter_map(|path| self.entry_by_path(path).map(reasoner_neuron_from_entry))
            .collect::<Vec<_>>();
        let kg_entities = included
            .iter()
            .filter(|path| looks_like_kg_neuron_path(path))
            .filter_map(|path| kg::KgEntity::load(path).ok())
            .collect::<Vec<_>>();

        if neurons.is_empty() && kg_entities.is_empty() {
            return ReasoningReport::default();
        }

        GraphReasoner::new(neurons, kg_entities).trace(
            &seeds
                .into_iter()
                .map(|(path, score)| ReasonerSeed::new(path, score))
                .collect::<Vec<_>>(),
            options,
        )
    }

    /// S-I (R16): Like `get_contexts_with_overflow` but returns BM25 scores for tiered emission.
    ///
    /// Returns:
    /// - `full`: `(path, bm25_score)` for neurons within budget
    /// - `overflow`: `(path, headline)` for budget-overflow neurons
    ///
    /// Tier mapping (by score):
    /// - `score ≥ 5.0` → Tier 2 (full body) — caller reads the file
    /// - `1.5 ≤ score < 5.0` → Tier 1 (summary only) — caller uses `summary_for()`
    /// - `score < 1.5` → Tier 0 (headline only, same as overflow) — already in overflow set
    #[allow(clippy::type_complexity)]
    pub fn get_contexts_with_scores_and_overflow(
        &self,
        task: &str,
        max_tokens: usize,
        module: Option<&str>,
        kind: Option<&str>,
        min_confidence: Option<f32>,
        multi_hop: bool,
        temporal_bias: Option<f32>,
    ) -> (Vec<(PathBuf, f32)>, Vec<(PathBuf, String)>) {
        let Ok(query) = QueryText::new(task) else {
            return (Vec::new(), Vec::new());
        };
        // Delegation: run the full pipeline then re-score the results for tier assignment.
        let (full_paths, overflow) = self.get_contexts_with_overflow_and_temporal_bias(
            task,
            max_tokens,
            module,
            kind,
            min_confidence,
            multi_hop,
            temporal_bias,
        );
        let terms = tokenize(query.as_str());
        let full_with_scores: Vec<(PathBuf, f32)> = full_paths
            .into_iter()
            .map(|path| {
                let score = self
                    .entry_by_path(&path)
                    .map(|e| self.bm25_score(&terms, e))
                    .unwrap_or(0.0);
                (path, score)
            })
            .collect();
        (full_with_scores, overflow)
    }

    ///
    /// For each file path in `open_files`, looks up the corresponding neuron entry
    /// and returns the top-N most frequent terms as soft expansion tokens.
    /// These are injected into the task string with a weight comment so BM25
    /// treats them at reduced significance relative to the direct task query.
    ///
    /// Lookup is O(k) where k = |open_files| — all data is already in the index.
    /// Returns a deduplicated list of terms (sorted by frequency descending).
    pub fn soft_terms_for_editor_context(
        &self,
        open_files: &[String],
        max_terms_per_file: usize,
    ) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();

        for file_path in open_files {
            // Match the open file path to an indexed neuron (suffix or substring match).
            let entry = self.retrieval.entries.iter().find(|e| {
                let ep = e.neuron_path.to_string_lossy();
                ep.ends_with(file_path.as_str()) || ep.contains(file_path.as_str())
            });

            if let Some(e) = entry {
                // Sort by term frequency descending, take top-N
                let mut term_freq_sorted: Vec<(&String, f32)> =
                    e.term_freq.iter().map(|(t, f)| (t, f.get())).collect();
                term_freq_sorted
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                for (term, _freq) in term_freq_sorted.iter().take(max_terms_per_file) {
                    if term.len() >= 3 && seen.insert((*term).clone()) {
                        result.push((*term).clone());
                    }
                }
            }
        }
        result
    }
}
