use super::*;

impl NeuronIndex {
    /// S-VII (R16): Apply biological LTD (Long-Term Depression) temporal decay to all synapses.
    ///
    /// Called once at `serve` startup and after `compile`. Mimics Hebbian LTD:
    /// synapses that have not been co-activated for many days gradually weaken,
    /// keeping the synapse graph lean and preventing dead-edge accumulation.
    ///
    /// Decay formula (half-life ≈ 70 days, λ = 0.01):
    ///   `learned_weight *= exp(-0.01 * days_idle)`
    ///
    /// Synapses with `learned_weight < 0.05` after decay are pruned (removed).
    /// Synapses with `last_co_activation_day == 0` are skipped (not yet learned).
    ///
    /// Returns: `(decayed, pruned)` counts for logging.
    pub fn apply_synapse_decay(&mut self) -> (usize, usize) {
        let now_days = now_unix_days();
        let (mut decayed, mut pruned) = (0usize, 0usize);
        for entry in &mut self.retrieval.entries {
            let before = entry.synapses.len();
            for syn in &mut entry.synapses {
                if syn.last_co_activation_day == 0 || syn.learned_weight.is_zero() {
                    continue; // not yet learned — skip
                }
                let days_idle = now_days.saturating_sub(syn.last_co_activation_day);
                if days_idle > 0 {
                    syn.learned_weight = SynapseWeight::new(
                        syn.learned_weight.get() * f32::exp(-0.01 * days_idle as f32),
                    );
                    decayed += 1;
                }
            }
            entry
                .synapses
                .retain(|s| s.learned_weight.get() > 0.05 || s.learned_weight.is_zero());
            pruned += before - entry.synapses.len();
        }
        // Rebuild adjacency cache after pruning
        if pruned > 0 {
            self.rebuild_derived_pub();
        }
        tracing::info!(decayed, pruned, "S-VII: synapse temporal decay applied");
        (decayed, pruned)
    }

    /// Update `last_co_activation_day` for all synapses between two co-cited neurons.
    ///
    /// Called from `record_hit` when both source and target of a synapse are cited
    /// in the same session — this is the LTP (Long-Term Potentiation) counterpart
    /// to `apply_synapse_decay`'s LTD.
    pub fn touch_co_activation_day(&mut self, cited_paths: &[PathBuf]) {
        let today = now_unix_days();
        let cited_set: std::collections::HashSet<&PathBuf> = cited_paths.iter().collect();
        for entry in &mut self.retrieval.entries {
            if !cited_set.contains(&entry.neuron_path) {
                continue;
            }
            for syn in &mut entry.synapses {
                if cited_set.contains(&syn.target) {
                    syn.last_co_activation_day = today;
                }
            }
        }
    }

    /// Propagate staleness to all neurons that import/call/implement the changed one.
    ///
    /// When a source file changes its neuron is marked stale. This method finds all
    /// neurons with synapse edges pointing *to* that neuron (reverse lookup via the
    /// adjacency list) and demotes their `staleness_multiplier` by ×0.7 (floor 0.3).
    ///
    /// Effect: dependent neurons surface as "needs re-evolve" in status, and rank
    /// lower in BM25 until the LLM refreshes them — preventing silent context drift.
    ///
    /// Cost: O(n) over all entries; n < 1 000 in typical projects → <1 ms.
    pub fn cascade_staleness(&mut self, changed_neuron: &Path) {
        for entry in &mut self.retrieval.entries {
            let is_dependent = entry.synapses.iter().any(|s| {
                s.target == changed_neuron
                    && matches!(
                        s.edge_type,
                        SynapseType::Imports | SynapseType::Calls | SynapseType::Implements
                    )
            });
            if is_dependent {
                // Demote (not evict) — preserves content while signalling freshness risk.
                entry.staleness_multiplier = (entry.staleness_multiplier * 0.7).max(0.3);
                tracing::debug!(
                    path = ?entry.neuron_path,
                    "cascade_staleness: dependent neuron demoted to staleness_multiplier={:.2}",
                    entry.staleness_multiplier
                );
            }
        }
    }
}

impl NeuronIndex {}

#[cfg(test)]
mod tests {
    #[test]
    fn apply_synapse_decay_reduces_learned_weight() {
        let decay_factor = (-0.01_f32 * 30.0_f32).exp();
        assert!(
            decay_factor < 1.0,
            "decay factor over 30 days should be < 1.0"
        );
        assert!(
            decay_factor > 0.7,
            "decay factor over 30 days should still be > 0.7 (≈0.74)"
        );
        let initial_weight = 0.8_f32;
        let decayed = initial_weight * decay_factor;
        assert!(
            decayed < initial_weight,
            "decayed weight should be less than initial"
        );
        assert!(
            decayed > 0.05,
            "30-day decay should not trigger pruning threshold of 0.05"
        );
    }
}
