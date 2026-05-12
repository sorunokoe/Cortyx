use super::*;

impl NeuronIndex {
    /// Find all `Contradicts` edges between any pair of activated neurons.
    ///
    /// Used by `get_contexts` to append a warning block when conflicting neurons
    /// are simultaneously activated — alerting the LLM to verify which is current.
    ///
    /// Performance: O(n²) over the activated set. For typical n=5, this is 10 lookups
    /// into the adjacency HashMap — effectively O(1) at runtime.
    ///
    /// Returns: `(path_a, path_b, reason)` for each contradicting pair found.
    pub fn find_contradictions(&self, activated: &[PathBuf]) -> Vec<(PathBuf, PathBuf, String)> {
        let mut pairs = Vec::new();
        for i in 0..activated.len() {
            if let Some(syns) = self.adjacency.get(&activated[i]) {
                for syn in syns {
                    if syn.edge_type == SynapseType::Contradicts {
                        // Only report each pair once (i < j by index in activated)
                        if let Some(j) = activated[i + 1..].iter().position(|p| *p == syn.target) {
                            let j_abs = i + 1 + j;
                            pairs.push((
                                activated[i].clone(),
                                activated[j_abs].clone(),
                                syn.reason.trim_start_matches("← ").to_string(),
                            ));
                        }
                    }
                }
            }
        }
        pairs
    }

    /// Scan all neurons (or a single neuron if `path` is given) for `Contradicts` edges.
    ///
    /// Used by `cortyx_check_consistency` — a proactive scan before task execution.
    /// Returns all contradiction pairs in the index (or pairs involving `path`).
    pub fn all_contradictions(
        &self,
        path_filter: Option<&Path>,
    ) -> Vec<(PathBuf, PathBuf, String)> {
        let mut seen: std::collections::HashSet<(PathBuf, PathBuf)> = Default::default();
        let mut pairs = Vec::new();
        for (src, syns) in &self.adjacency {
            if let Some(pf) = path_filter {
                if src != pf {
                    continue;
                }
            }
            for syn in syns {
                if syn.edge_type != SynapseType::Contradicts {
                    continue;
                }
                let a = src.min(&syn.target).clone();
                let b = src.max(&syn.target).clone();
                if seen.insert((a.clone(), b.clone())) {
                    pairs.push((a, b, syn.reason.trim_start_matches("← ").to_string()));
                }
            }
        }
        pairs
    }

    /// Load neuron body text for semantic consistency checks.
    ///
    /// When `path_filter` is given, returns only that neuron's body (for single-neuron
    /// scans). Without a filter, returns up to `limit` neuron bodies ordered by hit-rate
    /// descending so the most-used neurons are checked first.
    ///
    /// Used by `cortyx_check_consistency` to feed PureReason's semantic contradiction
    /// detector with raw neuron text.
    pub fn neuron_bodies_for_consistency(
        &self,
        path_filter: Option<&Path>,
        limit: usize,
    ) -> Option<Vec<String>> {
        if let Some(pf) = path_filter {
            let body = std::fs::read_to_string(pf).ok()?;
            return Some(vec![body]);
        }
        let mut entries: Vec<&BM25Entry> = self.entries.iter().collect();
        entries.sort_by(|a, b| {
            b.hit_count
                .partial_cmp(&a.hit_count)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let bodies: Vec<String> = entries
            .into_iter()
            .take(limit)
            .filter_map(|e| std::fs::read_to_string(&e.neuron_path).ok())
            .collect();
        Some(bodies)
    }
}
