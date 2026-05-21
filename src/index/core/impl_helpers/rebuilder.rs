use super::*;

impl NeuronIndex {
    /// Rebuild all derived structures — public entry point for `cortyx prune`.
    ///
    /// Prune evicts entries individually then calls this once to reconstruct
    /// path_index, adjacency, df_cache, etc. in a single O(n) pass.
    pub fn rebuild_derived_pub(&mut self) {
        // Force full rebuild: prune may have removed existing entries, so the
        // incremental delta path (which only handles appends) is not safe here.
        self.persistence.pending_append_count = 0;
        self.persistence
            .has_pending_updates
            .store(true, Ordering::Release);
        // S4-delta: prune removes entries — invalidate the delta baseline and force full save.
        self.persistence.delta_base.store(0, Ordering::Relaxed);
        self.persistence.delta_dirty.store(true, Ordering::Relaxed);
        self.rebuild_derived();
    }

    /// Rebuild all derived structures in a single O(n) pass.
    ///
    /// Previously five separate passes (path_index, parent_index, adjacency, df_cache,
    /// module_index); merged to reduce cache pressure and wall-clock time ~5×.
    pub(in crate::index) fn rebuild_derived(&mut self) {
        // S7: Incremental delta — skip the full clear+rebuild when only new entries were
        // appended (no updates).  This reduces the hot path (mining a new file into an
        // existing index) from O(N+n) to O(n) for the HashMap phase.
        if self.persistence.pending_append_count > 0
            && !self.persistence.has_pending_updates.load(Ordering::Acquire)
            && self.retrieval.idf_n > 0
        {
            self.rebuild_derived_delta();
            return;
        }

        self.retrieval.path_index.clear();
        self.retrieval.parent_index.clear();
        self.retrieval.adjacency.clear();
        self.retrieval.df_cache.clear();
        self.retrieval.posting_list.clear();
        self.retrieval.module_index.clear();
        self.retrieval.session_index.clear(); // R21 T6
                                              // Full rebuild reassigns all path_index positions, so usize keys in
                                              // co_return_counts would point to wrong entries. Reset to avoid
                                              // spurious Hebbian synapse formation.
        if let Ok(mut counts) = self.feedback.co_return_counts.lock() {
            counts.clear();
        }
        self.retrieval.idf_n = 0;

        let mut non_agg_total_terms = 0usize;
        let mut verbatim_total_terms = 0usize;
        let mut verbatim_count = 0usize;

        for (i, entry) in self.retrieval.entries.iter().enumerate() {
            // path_index
            self.retrieval
                .path_index
                .insert(entry.neuron_path.clone(), i);

            // parent_index
            if let Some(p) = &entry.parent {
                self.retrieval
                    .parent_index
                    .entry(p.clone())
                    .or_default()
                    .push(i);
            }

            // adjacency (forward + reverse edges)
            for syn in &entry.synapses {
                self.retrieval
                    .adjacency
                    .entry(entry.neuron_path.clone())
                    .or_default()
                    .push(syn.clone());

                self.retrieval
                    .adjacency
                    .entry(syn.target.clone())
                    .or_default()
                    .push(Synapse {
                        target: entry.neuron_path.clone(),
                        edge_type: syn.edge_type.inverse(),
                        weight: SynapseWeight::new(syn.weight.get() * 0.7),
                        reason: format!("← {}", syn.reason),
                        learned_weight: crate::types::SynapseWeight::ZERO,
                        traversal_count: 0,
                        last_co_activation_day: 0,
                    });
            }

            // df_cache + posting_list.
            // IMPORTANT: Aggregate neurons (word-count summaries, dollar totals) must NOT
            // contribute to df_cache.  An _count_music.aggregate.md neuron contains "music"
            // dozens of times, inflating df("music") and crushing its IDF.  This caused a
            // 5-entry SSU regression: session 329 ("music"×18, no "streaming"/"service") lost
            // to session 309 ("service"×7) because IDF("music") collapsed while IDF("service")
            // stayed high.  Excluding Aggregate from df_cache restores the IDF calibration
            // from the e18c4e6 baseline (100% SSU) even when aggregates are mined.
            // Posting-list is still built for ALL kinds so counting_augment can find Aggregates.
            let is_aggregate = matches!(entry.kind, NeuronKind::Aggregate);
            for term in entry.term_freq.keys() {
                if !is_aggregate {
                    *self.retrieval.df_cache.entry(term.clone()).or_insert(0) += 1;
                }
                self.retrieval
                    .posting_list
                    .entry(term.clone())
                    .or_default()
                    .push(i);
            }
            if !is_aggregate {
                self.retrieval.idf_n += 1;
            }

            // module_index
            if let Some(m) = &entry.module {
                self.retrieval
                    .module_index
                    .entry(m.clone())
                    .or_default()
                    .push(i);
            }

            // R21 T6: session_index — for session-level grouping at retrieval time
            if !entry.session_id.is_empty() {
                self.retrieval
                    .session_index
                    .entry(entry.session_id.clone())
                    .or_default()
                    .push(i);
            }

            if !is_aggregate {
                non_agg_total_terms += entry.term_count;
            }
            if matches!(entry.kind, NeuronKind::Verbatim) {
                verbatim_total_terms += entry.term_count;
                verbatim_count += 1;
            }
        }

        // avg_doc_len excludes Aggregate neurons so it matches e18c4e6 calibration.
        self.retrieval.avg_doc_len = if self.retrieval.idf_n == 0 {
            0.0
        } else {
            non_agg_total_terms as f32 / self.retrieval.idf_n as f32
        };
        self.retrieval.avg_verbatim_doc_len = if verbatim_count == 0 {
            self.retrieval.avg_doc_len
        } else {
            verbatim_total_terms as f32 / verbatim_count as f32
        };

        self.rebuild_structural_centrality();
        self.resync_total_activations();
        self.build_vocab_bridge();
        self.build_morpheme_map();
        self.build_concept_clouds();
        self.apply_peer_vocab_borrowing();
        self.merge_cooccurrence_into_vocab_bridge();
        self.load_pmi_neighbors();
        self.persistence
            .structural_artifacts_dirty
            .store(true, Ordering::Relaxed);
        self.persistence.pending_append_count = 0;
        self.persistence
            .has_pending_updates
            .store(false, Ordering::Release);
    }

    /// Incremental derived-structure update for pure-append batches (S7).
    ///
    /// When only new entries were appended (no existing entries were modified), we
    /// skip clearing and rebuilding the large HashMaps from scratch.  Instead we
    /// process only the `pending_append_count` newest entries and add their
    /// contributions to the existing structures in O(n) rather than O(N+n).
    ///
    /// The bridge/cloud/neighbor builds (vocab_bridge, morpheme_map, concept_clouds,
    /// pmi_neighbors) still run over the full corpus because they are O(terms), not
    /// O(entries²), and must reflect the complete vocabulary.
    pub(in crate::index) fn rebuild_derived_delta(&mut self) {
        let new_start = self
            .retrieval
            .entries
            .len()
            .saturating_sub(self.persistence.pending_append_count);

        for (offset, entry) in self.retrieval.entries[new_start..].iter().enumerate() {
            let abs_i = new_start + offset;

            // path_index is already maintained by index_neuron(), but ensure consistency.
            self.retrieval
                .path_index
                .insert(entry.neuron_path.clone(), abs_i);

            if let Some(p) = &entry.parent {
                self.retrieval
                    .parent_index
                    .entry(p.clone())
                    .or_default()
                    .push(abs_i);
            }

            for syn in &entry.synapses {
                self.retrieval
                    .adjacency
                    .entry(entry.neuron_path.clone())
                    .or_default()
                    .push(syn.clone());
                self.retrieval
                    .adjacency
                    .entry(syn.target.clone())
                    .or_default()
                    .push(Synapse {
                        target: entry.neuron_path.clone(),
                        edge_type: syn.edge_type.inverse(),
                        weight: SynapseWeight::new(syn.weight.get() * 0.7),
                        reason: format!("← {}", syn.reason),
                        learned_weight: crate::types::SynapseWeight::ZERO,
                        traversal_count: 0,
                        last_co_activation_day: 0,
                    });
            }

            let is_aggregate = matches!(entry.kind, NeuronKind::Aggregate);
            for term in entry.term_freq.keys() {
                if !is_aggregate {
                    *self.retrieval.df_cache.entry(term.clone()).or_insert(0) += 1;
                }
                self.retrieval
                    .posting_list
                    .entry(term.clone())
                    .or_default()
                    .push(abs_i);
            }
            if !is_aggregate {
                self.retrieval.idf_n += 1;
            }

            if let Some(m) = &entry.module {
                self.retrieval
                    .module_index
                    .entry(m.clone())
                    .or_default()
                    .push(abs_i);
            }

            if !entry.session_id.is_empty() {
                self.retrieval
                    .session_index
                    .entry(entry.session_id.clone())
                    .or_default()
                    .push(abs_i);
            }
        }

        // Recompute avg_doc_len from all entries (O(n) integer addition — cheap).
        let mut non_agg_total_terms = 0usize;
        let mut verbatim_total_terms = 0usize;
        let mut verbatim_count = 0usize;
        for entry in &self.retrieval.entries {
            let is_aggregate = matches!(entry.kind, NeuronKind::Aggregate);
            if !is_aggregate {
                non_agg_total_terms += entry.term_count;
            }
            if matches!(entry.kind, NeuronKind::Verbatim) {
                verbatim_total_terms += entry.term_count;
                verbatim_count += 1;
            }
        }
        self.retrieval.avg_doc_len = if self.retrieval.idf_n == 0 {
            0.0
        } else {
            non_agg_total_terms as f32 / self.retrieval.idf_n as f32
        };
        self.retrieval.avg_verbatim_doc_len = if verbatim_count == 0 {
            self.retrieval.avg_doc_len
        } else {
            verbatim_total_terms as f32 / verbatim_count as f32
        };

        self.rebuild_structural_centrality();
        self.resync_total_activations();

        // Bridge/cloud/neighbor builds must see the full corpus.
        self.build_vocab_bridge();
        self.build_morpheme_map();
        self.build_concept_clouds();
        self.apply_peer_vocab_borrowing();
        self.merge_cooccurrence_into_vocab_bridge();
        self.load_pmi_neighbors();
        self.persistence
            .structural_artifacts_dirty
            .store(true, Ordering::Relaxed);
        self.persistence.pending_append_count = 0;
        self.persistence
            .has_pending_updates
            .store(false, Ordering::Release);
    }

    pub(in crate::index) fn set_structural_centrality(&mut self, path: &Path, value: f32) {
        if let Some(entry) = self
            .retrieval
            .entries
            .iter_mut()
            .find(|entry| entry.neuron_path == path)
        {
            entry.structural_centrality = value.clamp(0.0, 1.0);
        }
    }

    fn rebuild_structural_centrality(&mut self) {
        for entry in &mut self.retrieval.entries {
            entry.structural_centrality = 0.0;
        }

        let mut in_degree: HashMap<PathBuf, usize> = HashMap::new();
        for synapses in self.retrieval.adjacency.values() {
            for synapse in synapses {
                if matches!(synapse.edge_type, SynapseType::Imports | SynapseType::Calls) {
                    *in_degree.entry(synapse.target.clone()).or_insert(0) += 1;
                }
            }
        }

        let max_degree = in_degree.values().copied().max().unwrap_or(0);
        if max_degree == 0 {
            return;
        }

        let updates: Vec<(PathBuf, f32)> = in_degree
            .into_iter()
            .map(|(path, degree)| (path, degree as f32 / max_degree as f32))
            .collect();
        for (path, centrality) in updates {
            self.set_structural_centrality(&path, centrality);
        }
    }

    fn resync_total_activations(&self) {
        let total = self
            .retrieval
            .entries
            .iter()
            .map(|entry| entry.use_count.load(std::sync::atomic::Ordering::Relaxed) as u64)
            .sum();
        self.feedback
            .total_activations
            .store(total, std::sync::atomic::Ordering::Relaxed);
    }
}
