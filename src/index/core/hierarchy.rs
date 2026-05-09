// This file is a submodule of `crate::index::core`.
// It contains `impl NeuronIndex` methods extracted from helpers.rs.
// All visibility is relative to `crate::index` (the parent of `core`).
use super::*;

impl NeuronIndex {
    // ── Hierarchy navigation (TRIZ R13-G2) ───────────────────────────────────

    /// List all modules with their neuron count and average hit rate.
    /// Includes `@person` scoped modules alongside directory modules.
    /// Returns entries sorted by name for deterministic output.
    pub fn list_modules(&self) -> Vec<ModuleSummary> {
        let mut map: HashMap<String, (usize, f32)> = HashMap::new();
        for entry in &self.entries {
            if let Some(m) = entry.module.as_deref() {
                let e = map.entry(m.to_string()).or_default();
                e.0 += 1;
                let rate = if entry.use_count > 0 {
                    entry.hit_count as f32 / entry.use_count as f32
                } else {
                    0.0
                };
                e.1 += rate;
            }
        }
        let mut result: Vec<ModuleSummary> = map
            .into_iter()
            .map(|(name, (count, rate_sum))| ModuleSummary {
                name: name.clone(),
                neuron_count: count,
                avg_hit_rate: if count > 0 {
                    rate_sum / count as f32
                } else {
                    0.0
                },
                is_person_scope: name.starts_with('@'),
            })
            .collect();
        result.sort_by(|a, b| a.name.cmp(&b.name));
        result
    }

    /// List neurons in a module (or all neurons if `module` is None).
    /// Returns a summary of each neuron's path, kind, staleness, and hit rate.
    pub fn list_neurons(&self, module: Option<&str>) -> Vec<NeuronSummary> {
        let indices: Vec<usize> = if let Some(m) = module {
            self.module_index.get(m).cloned().unwrap_or_default()
        } else {
            (0..self.entries.len()).collect()
        };
        let mut result: Vec<NeuronSummary> = indices
            .into_iter()
            .map(|i| {
                let e = &self.entries[i];
                let hit_rate = if e.use_count > 0 {
                    e.hit_count as f32 / e.use_count as f32
                } else {
                    0.0
                };
                NeuronSummary {
                    path: e.neuron_path.clone(),
                    kind: e.kind.clone(),
                    staleness_multiplier: e.staleness_multiplier,
                    hit_rate,
                    use_count: e.use_count,
                }
            })
            .collect();
        result.sort_by(|a, b| a.path.cmp(&b.path));
        result
    }

    /// Return the most recent Verbatim neurons that mention "current moment" markers.
    ///
    /// This stays index-only: it uses precomputed timestamps plus token presence to cheaply
    /// surface likely `today` / `currently` / `this week` sessions for downstream temporal
    /// reasoning without scanning the full corpus at query time.
    pub fn recent_verbatim_paths_with_current_markers(
        &self,
        module: Option<&str>,
        limit: usize,
    ) -> Vec<PathBuf> {
        if limit == 0 {
            return Vec::new();
        }

        let has_current_marker_terms = |terms: &HashMap<String, f32>| {
            terms.contains_key("today")
                || terms.contains_key("currently")
                || terms.contains_key("now")
                || (terms.contains_key("this")
                    && (terms.contains_key("week")
                        || terms.contains_key("month")
                        || terms.contains_key("year")))
        };

        let mut ranked = self
            .entries
            .iter()
            .filter(|entry| matches!(entry.kind, NeuronKind::Verbatim))
            .filter(|entry| {
                module.is_none_or(|scope| entry.module.as_deref() == Some(scope))
                    && has_current_marker_terms(&entry.term_freq)
            })
            .filter_map(|entry| Some((entry.timestamp_secs?, entry.neuron_path.clone())))
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)));
        ranked
            .into_iter()
            .take(limit)
            .map(|(_, path)| path)
            .collect()
    }

    /// Return neurons that are strong candidates for the shared concept library.
    ///
    /// Candidates must:
    /// - be Core or Concept neurons
    /// - meet the minimum use_count / hit_rate / quality thresholds
    /// - be sorted by strongest observed utility first
    pub fn publish_ready_candidates(
        &self,
        min_use: u32,
        min_hit_rate: f32,
        min_quality: f32,
        limit: usize,
    ) -> Vec<PublishReadySummary> {
        let mut result: Vec<PublishReadySummary> = self
            .entries
            .iter()
            .filter_map(|entry| {
                if !matches!(entry.kind, NeuronKind::Core | NeuronKind::Concept) {
                    return None;
                }
                let hit_rate = if entry.use_count > 0 {
                    entry.hit_count as f32 / entry.use_count as f32
                } else {
                    0.0
                };
                if entry.use_count < min_use
                    || hit_rate < min_hit_rate
                    || entry.quality_score < min_quality
                {
                    return None;
                }
                Some(PublishReadySummary {
                    path: entry.neuron_path.clone(),
                    kind: entry.kind.clone(),
                    use_count: entry.use_count,
                    hit_rate,
                    quality_score: entry.quality_score,
                })
            })
            .collect();
        result.sort_by(|a, b| {
            b.use_count
                .cmp(&a.use_count)
                .then_with(|| b.hit_rate.total_cmp(&a.hit_rate))
                .then_with(|| b.quality_score.total_cmp(&a.quality_score))
                .then_with(|| a.path.cmp(&b.path))
        });
        if limit > 0 {
            result.truncate(limit);
        }
        result
    }

    /// Return the first `lines` lines of a neuron file for quick preview.
    /// Returns `None` if the file does not exist or cannot be read.
    pub fn peek_neuron(&self, path: &Path, lines: usize) -> Option<String> {
        let content = std::fs::read_to_string(path).ok()?;
        let preview: String = content.lines().take(lines).collect::<Vec<_>>().join("\n");
        Some(preview)
    }

    /// List only `@person`-scoped modules (convention: module starts with `@`).
    pub fn list_persons(&self) -> Vec<ModuleSummary> {
        self.list_modules()
            .into_iter()
            .filter(|m| m.is_person_scope)
            .collect()
    }
}
