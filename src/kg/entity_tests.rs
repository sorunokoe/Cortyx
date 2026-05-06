use super::super::render::kg_neuron_path;
use super::KgEntity;
use tempfile::TempDir;

#[test]
fn kg_roundtrip_save_load() {
    let dir = TempDir::new().unwrap();
    let path = kg_neuron_path(dir.path(), "project_meta");

    let mut entity = KgEntity::load(&path).unwrap();
    entity.add_fact("language", "Rust", Some("2024-01-01"));
    entity.add_fact("version", "0.2.0", Some("2024-06-01"));
    entity.save().unwrap();

    let loaded = KgEntity::load(&path).unwrap();
    assert_eq!(loaded.facts.len(), 2);
    assert_eq!(loaded.facts[0].predicate, "language");
    assert_eq!(loaded.facts[0].value, "Rust");
    assert_eq!(loaded.facts[1].value, "0.2.0");
}

#[test]
fn kg_roundtrip_preserves_escaped_pipes() {
    let dir = TempDir::new().unwrap();
    let path = kg_neuron_path(dir.path(), "project_meta");

    let mut entity = KgEntity::load(&path).unwrap();
    entity.add_fact("notes", r#"A | B \ C"#, Some("2024-01-01"));
    entity.save().unwrap();

    let loaded = KgEntity::load(&path).unwrap();
    assert_eq!(loaded.facts.len(), 1);
    assert_eq!(loaded.facts[0].predicate, "notes");
    assert_eq!(loaded.facts[0].value, r#"A | B \ C"#);
}

#[test]
fn kg_invalidate_fact() {
    let dir = TempDir::new().unwrap();
    let path = kg_neuron_path(dir.path(), "team");

    let mut entity = KgEntity::load(&path).unwrap();
    entity.add_fact("lead", "Alice", Some("2024-01-01"));
    entity.invalidate_fact("lead", "2024-06-01").unwrap();
    assert_eq!(entity.facts[0].ended, "2024-06-01");

    assert!(entity.invalidate_fact("lead", "2025-01-01").is_err());
}

#[test]
fn kg_timeline_sorted() {
    let dir = TempDir::new().unwrap();
    let path = kg_neuron_path(dir.path(), "deps");

    let mut entity = KgEntity::load(&path).unwrap();
    entity.add_fact("db", "postgres", Some("2023-01-01"));
    entity.invalidate_fact("db", "2024-01-01").unwrap();
    entity.add_fact("db", "sqlite", Some("2024-01-01"));

    let tl = entity.timeline_for("db");
    assert_eq!(tl.len(), 2);
    assert_eq!(tl[0].value, "postgres");
    assert_eq!(tl[1].value, "sqlite");
}

#[test]
fn replace_active_fact_invalidates_old_value_and_keeps_timeline() {
    let dir = TempDir::new().unwrap();
    let path = kg_neuron_path(dir.path(), "agent_reviewer");

    let mut entity = KgEntity::load(&path).unwrap();
    assert!(entity.replace_active_fact("status", "in_progress", "2026-04-17T10:00:00Z"));
    assert!(!entity.replace_active_fact("status", "in_progress", "2026-04-17T10:01:00Z"));
    assert!(entity.replace_active_fact("status", "done", "2026-04-17T10:05:00Z"));

    let active = entity.active_values_for_predicate("status", None);
    assert_eq!(active.len(), 1);
    assert_eq!(active[0].value, "done");

    let timeline = entity.timeline_for("status");
    assert_eq!(timeline.len(), 2);
    assert_eq!(timeline[0].value, "in_progress");
    assert_eq!(timeline[0].ended, "2026-04-17T10:05:00Z");
    assert_eq!(timeline[1].value, "done");
    assert_eq!(timeline[1].ended, "");
}

#[test]
fn sync_active_values_reconciles_set_membership() {
    let dir = TempDir::new().unwrap();
    let path = kg_neuron_path(dir.path(), "agent_reviewer");

    let mut entity = KgEntity::load(&path).unwrap();
    assert!(entity.sync_active_values(
        "related_entity",
        &["auth".to_string(), "routing".to_string()],
        "2026-04-17T10:00:00Z"
    ));
    assert!(entity.sync_active_values(
        "related_entity",
        &["routing".to_string(), "middleware".to_string()],
        "2026-04-17T10:05:00Z"
    ));

    let active = entity.active_values_for_predicate("related_entity", None);
    assert_eq!(active.len(), 2);
    assert!(active.iter().any(|fact| fact.value == "routing"));
    assert!(active.iter().any(|fact| fact.value == "middleware"));

    let timeline = entity.timeline_for("related_entity");
    assert_eq!(timeline.len(), 3);
    let auth = timeline.iter().find(|fact| fact.value == "auth").unwrap();
    assert_eq!(auth.ended, "2026-04-17T10:05:00Z");
}

#[test]
fn latest_active_helpers_surface_current_state() {
    let dir = TempDir::new().unwrap();
    let path = kg_neuron_path(dir.path(), "agent_reviewer");

    let mut entity = KgEntity::load(&path).unwrap();
    entity.add_fact("status", "in_progress", Some("2026-04-17T10:00:00Z"));
    entity.add_fact("status", "blocked", Some("2026-04-17T10:05:00Z"));
    entity
        .invalidate_fact("status", "2026-04-17T10:06:00Z")
        .unwrap();
    entity.add_fact("status", "done", Some("2026-04-17T10:07:00Z"));
    entity.sync_active_values(
        "related_entity",
        &["auth".to_string(), "engine".to_string(), "auth".to_string()],
        "2026-04-17T10:08:00Z",
    );

    assert_eq!(
        entity.latest_active_value("status").as_deref(),
        Some("done")
    );
    assert_eq!(
        entity.active_value_strings("related_entity"),
        vec!["auth".to_string(), "engine".to_string()]
    );
    assert_eq!(
        entity.latest_active_timestamp().as_deref(),
        Some("2026-04-17T10:08:00Z")
    );
}
