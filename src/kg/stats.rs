use std::path::Path;

use super::entity::KgEntity;
use super::render::list_kg_paths;

/// Aggregate statistics across all KG entities in a project.
#[derive(Debug, Default, serde::Serialize)]
pub struct KgStats {
    pub entity_count: usize,
    pub total_facts: usize,
    pub active_facts: usize,
    pub ended_facts: usize,
}

/// Compute KG stats for a project.
pub fn compute_stats(project_root: &Path) -> KgStats {
    let mut stats = KgStats::default();
    for path in list_kg_paths(project_root) {
        if let Ok(entity) = KgEntity::load(&path) {
            stats.entity_count += 1;
            stats.total_facts += entity.facts.len();
            let active = entity.active_facts(None).len();
            stats.active_facts += active;
            stats.ended_facts += entity.facts.len() - active;
        }
    }
    stats
}

#[cfg(test)]
mod tests {
    use super::super::render::kg_neuron_path;
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn kg_stats_aggregates() {
        let dir = TempDir::new().unwrap();

        for name in &["entity_a", "entity_b"] {
            let path = kg_neuron_path(dir.path(), name);
            let mut e = KgEntity::load(&path).unwrap();
            e.add_fact("key", "val", None);
            e.save().unwrap();
        }
        let stats = compute_stats(dir.path());
        assert_eq!(stats.entity_count, 2);
        assert_eq!(stats.total_facts, 2);
        assert_eq!(stats.active_facts, 2);
    }
}
