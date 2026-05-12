use super::*;

impl NeuronIndex {
    /// Rebuild all derived structures — public entry point for `cortyx prune`.
    ///
    /// Prune evicts entries individually then calls this once to reconstruct
    /// path_index, adjacency, df_cache, etc. in a single O(n) pass.
    pub fn rebuild_derived_pub(&mut self) {
        // Force full rebuild: prune may have removed existing entries, so the
        // incremental delta path (which only handles appends) is not safe here.
        self.pending_append_count = 0;
        self.has_pending_updates = true;
        // S4-WAL: prune removes entries — invalidate WAL baseline and force full save.
        self.wal_base.store(0, Ordering::Relaxed);
        self.needs_full_save.store(true, Ordering::Relaxed);
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
        if self.pending_append_count > 0 && !self.has_pending_updates && self.idf_n > 0 {
            self.rebuild_derived_delta();
            return;
        }

        self.path_index.clear();
        self.parent_index.clear();
        self.adjacency.clear();
        self.df_cache.clear();
        self.posting_list.clear();
        self.module_index.clear();
        self.session_index.clear(); // R21 T6
        self.idf_n = 0;

        let mut non_agg_total_terms = 0usize;
        let mut verbatim_total_terms = 0usize;
        let mut verbatim_count = 0usize;

        for (i, entry) in self.entries.iter().enumerate() {
            // path_index
            self.path_index.insert(entry.neuron_path.clone(), i);

            // parent_index
            if let Some(p) = &entry.parent {
                self.parent_index.entry(p.clone()).or_default().push(i);
            }

            // adjacency (forward + reverse edges)
            for syn in &entry.synapses {
                self.adjacency
                    .entry(entry.neuron_path.clone())
                    .or_default()
                    .push(syn.clone());

                self.adjacency
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
                    *self.df_cache.entry(term.clone()).or_insert(0) += 1;
                }
                self.posting_list.entry(term.clone()).or_default().push(i);
            }
            if !is_aggregate {
                self.idf_n += 1;
            }

            // module_index
            if let Some(m) = &entry.module {
                self.module_index.entry(m.clone()).or_default().push(i);
            }

            // R21 T6: session_index — for session-level grouping at retrieval time
            if !entry.session_id.is_empty() {
                self.session_index
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
        self.avg_doc_len = if self.idf_n == 0 {
            0.0
        } else {
            non_agg_total_terms as f32 / self.idf_n as f32
        };
        self.avg_verbatim_doc_len = if verbatim_count == 0 {
            self.avg_doc_len
        } else {
            verbatim_total_terms as f32 / verbatim_count as f32
        };

        self.build_vocab_bridge();
        self.build_morpheme_map();
        self.build_concept_clouds();
        self.apply_peer_vocab_borrowing();
        self.merge_cooccurrence_into_vocab_bridge();
        self.load_pmi_neighbors();
        self.structural_artifacts_dirty
            .store(true, Ordering::Relaxed);
        self.pending_append_count = 0;
        self.has_pending_updates = false;
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
        let new_start = self.entries.len().saturating_sub(self.pending_append_count);

        for (offset, entry) in self.entries[new_start..].iter().enumerate() {
            let abs_i = new_start + offset;

            // path_index is already maintained by index_neuron(), but ensure consistency.
            self.path_index.insert(entry.neuron_path.clone(), abs_i);

            if let Some(p) = &entry.parent {
                self.parent_index.entry(p.clone()).or_default().push(abs_i);
            }

            for syn in &entry.synapses {
                self.adjacency
                    .entry(entry.neuron_path.clone())
                    .or_default()
                    .push(syn.clone());
                self.adjacency
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
                    *self.df_cache.entry(term.clone()).or_insert(0) += 1;
                }
                self.posting_list
                    .entry(term.clone())
                    .or_default()
                    .push(abs_i);
            }
            if !is_aggregate {
                self.idf_n += 1;
            }

            if let Some(m) = &entry.module {
                self.module_index.entry(m.clone()).or_default().push(abs_i);
            }

            if !entry.session_id.is_empty() {
                self.session_index
                    .entry(entry.session_id.clone())
                    .or_default()
                    .push(abs_i);
            }
        }

        // Recompute avg_doc_len from all entries (O(n) integer addition — cheap).
        let mut non_agg_total_terms = 0usize;
        let mut verbatim_total_terms = 0usize;
        let mut verbatim_count = 0usize;
        for entry in &self.entries {
            let is_aggregate = matches!(entry.kind, NeuronKind::Aggregate);
            if !is_aggregate {
                non_agg_total_terms += entry.term_count;
            }
            if matches!(entry.kind, NeuronKind::Verbatim) {
                verbatim_total_terms += entry.term_count;
                verbatim_count += 1;
            }
        }
        self.avg_doc_len = if self.idf_n == 0 {
            0.0
        } else {
            non_agg_total_terms as f32 / self.idf_n as f32
        };
        self.avg_verbatim_doc_len = if verbatim_count == 0 {
            self.avg_doc_len
        } else {
            verbatim_total_terms as f32 / verbatim_count as f32
        };

        // Bridge/cloud/neighbor builds must see the full corpus.
        self.build_vocab_bridge();
        self.build_morpheme_map();
        self.build_concept_clouds();
        self.apply_peer_vocab_borrowing();
        self.merge_cooccurrence_into_vocab_bridge();
        self.load_pmi_neighbors();
        self.structural_artifacts_dirty
            .store(true, Ordering::Relaxed);
        self.pending_append_count = 0;
        self.has_pending_updates = false;
    }
}
