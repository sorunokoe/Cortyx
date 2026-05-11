use super::*;

impl NeuronIndex {
    /// Return the token count for a neuron path (for F2 budget tracking).
    pub fn tokens_for(&self, path: &Path) -> usize {
        self.entry_by_path(path).map(|e| e.tokens).unwrap_or(0)
    }


    /// S-III (R16): Count neurons with quality_score below the curation threshold.
    ///
    /// Used by `cortyx status` to surface "needs curation" count.
    pub fn low_quality_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.quality_score < 0.4)
            .count()
    }


    /// Return the number of distinct terms indexed for a neuron.
    ///
    /// Used by S-VIII auto-mine to compute code-block ∩ neuron term overlap ratio.
    pub fn term_count_for(&self, path: &Path) -> usize {
        self.entry_by_path(path)
            .map(|e| e.term_freq.len())
            .unwrap_or(0)
    }


    /// S-I (R16): Return the pre-computed Tier-1 summary for a neuron.
    ///
    /// Returns `None` if the neuron is not indexed or has no summary.
    pub fn summary_for(&self, path: &Path) -> Option<&str> {
        self.entry_by_path(path)
            .filter(|e| !e.summary.is_empty())
            .map(|e| e.summary.as_str())
    }


    pub fn module_for(&self, path: &Path) -> Option<&str> {
        self.entry_by_path(path)
            .and_then(|entry| entry.module.as_deref())
    }


    pub fn context_metadata_for(&self, path: &Path) -> Option<ContextMetadata> {
        self.entry_by_path(path).map(|entry| {
            let hit_rate = if entry.use_count == 0 {
                0.0
            } else {
                entry.hit_count as f32 / entry.use_count as f32
            };
            ContextMetadata {
                kind: entry.kind.clone(),
                module: entry.module.clone(),
                summary: entry.summary.clone(),
                timestamp_secs: entry.timestamp_secs,
                tokens: entry.tokens,
                use_count: entry.use_count,
                hit_count: entry.hit_count,
                hit_rate,
            }
        })
    }


    pub fn derived_answer_path_for_task(&self, task: &str) -> Option<PathBuf> {
        let query = QueryText::new(task).ok()?;
        self.synthetic_answer_path(query.as_str())
    }
}
