use super::*;
use crate::index::NeuronIndex;
use crate::kg::KgFact;
use crate::neuron::provenance::{
    load_provenance, provenance_content_hash, ProvenanceOperation, ProvenanceSource,
};
use crate::reasoner::{ReasonedFact, ReasonedNode, ReasoningReport};
use crate::sync_transport::{
    SyncHandoffIssue, SyncHandoffState, SyncHandoffSummary, SyncRevisionState,
};
use std::fs;

fn test_item(path: &str, rendered: &str) -> RenderedContextItem {
    RenderedContextItem {
        path: PathBuf::from(path),
        rendered: rendered.to_string(),
        fingerprint: fingerprint_rendered_context(rendered),
    }
}

fn sample_collaboration_projection() -> CollaborationStateProjection {
    let mut diary = CollaborationDiaryRecord::new(
        "reviewer",
        StructuredDiaryEntry {
            agent: Some("reviewer".to_string()),
            title: Some("Audit auth middleware".to_string()),
            status: Some("blocked".to_string()),
            goal: Some("Close the auth bypass.".to_string()),
            next_step: Some("Wait for api-owner approval.".to_string()),
            blocker: Some("Waiting on api-owner.".to_string()),
            outcome: None,
            entities: vec!["auth".to_string(), "engine".to_string()],
            depends_on: vec!["api-owner".to_string()],
            action: None,
            refined_plan: None,
        },
    );
    diary.when = Some("2026-04-17T10:04:00Z".to_string());

    let trusted_integrity =
        |fingerprint: &str| crate::neuron::provenance::ProvenanceIntegritySummary {
            trusted: true,
            score: 100,
            fingerprint: Some(fingerprint.to_string()),
            revision_count: 1,
            author_count: 1,
            authorship_present: true,
            latest_author_present: true,
            identity_verified: true,
            content_verified: true,
            chain_verified: true,
            timestamps_monotonic: true,
            issues: Vec::new(),
        };

    let sync = SyncTransportStatus {
        neuron_uuid: "uuid-1234".to_string(),
        snapshot: Some(SyncRevisionState {
            source_path: PathBuf::from("src/engine.rs"),
            module: Some("engine".to_string()),
            content_hash: "hash-snapshot".to_string(),
            edit_id: Some("edit-2".to_string()),
            parent_edit_id: Some("edit-1".to_string()),
            edited_at: Some("2026-04-17T10:05:00Z".to_string()),
            author_id: Some("agent:reviewer".to_string()),
            author_display: Some("Reviewer".to_string()),
            created_by_id: Some("agent:reviewer".to_string()),
            created_by_display: Some("Reviewer".to_string()),
            provenance_fingerprint: Some("prov-snapshot".to_string()),
            revision_count: 1,
            author_count: 1,
            summary: Some("local auth hardening".to_string()),
            integrity: trusted_integrity("prov-snapshot"),
        }),
        outgoing: Some(SyncRevisionState {
            source_path: PathBuf::from("src/engine.rs"),
            module: Some("engine".to_string()),
            content_hash: "hash-outgoing".to_string(),
            edit_id: Some("edit-2".to_string()),
            parent_edit_id: Some("edit-1".to_string()),
            edited_at: Some("2026-04-17T10:05:00Z".to_string()),
            author_id: Some("agent:reviewer".to_string()),
            author_display: Some("Reviewer".to_string()),
            created_by_id: Some("agent:reviewer".to_string()),
            created_by_display: Some("Reviewer".to_string()),
            provenance_fingerprint: Some("prov-snapshot".to_string()),
            revision_count: 1,
            author_count: 1,
            summary: Some("local auth hardening".to_string()),
            integrity: trusted_integrity("prov-snapshot"),
        }),
        incoming: Some(SyncRevisionState {
            source_path: PathBuf::from("src/engine.rs"),
            module: Some("engine".to_string()),
            content_hash: "hash-incoming".to_string(),
            edit_id: Some("edit-3".to_string()),
            parent_edit_id: Some("edit-1".to_string()),
            edited_at: Some("2026-04-17T10:06:00Z".to_string()),
            author_id: Some("agent:reviewer".to_string()),
            author_display: Some("Reviewer".to_string()),
            created_by_id: Some("agent:reviewer".to_string()),
            created_by_display: Some("Reviewer".to_string()),
            provenance_fingerprint: Some("prov-incoming".to_string()),
            revision_count: 1,
            author_count: 1,
            summary: Some("remote auth edit".to_string()),
            integrity: trusted_integrity("prov-incoming"),
        }),
        conflict_paths: vec![PathBuf::from(
            ".cortyx/sync/conflicts/uu/uuid-1234--edit-2--edit-3.json",
        )],
        outgoing_pending: true,
        incoming_pending: true,
        handoff: SyncHandoffSummary {
            state: SyncHandoffState::Conflict,
            shared_edit_id: Some("edit-1".to_string()),
            local_edit_id: Some("edit-2".to_string()),
            remote_edit_id: Some("edit-3".to_string()),
            continuity_verified: false,
            integrity_verified: true,
            score: 60,
            issues: vec![SyncHandoffIssue::ConflictRecorded],
        },
    };

    let kg_entities = vec![
        kg::KgEntity {
            entity: agent_entity_name("reviewer"),
            facts: vec![
                KgFact {
                    predicate: AGENT_ACTION_PREDICATE.to_string(),
                    value: "Investigating auth middleware.".to_string(),
                    valid_from: "2026-04-17T10:01:00Z".to_string(),
                    ended: String::new(),
                },
                KgFact {
                    predicate: AGENT_OUTCOME_PREDICATE.to_string(),
                    value: "Found a legacy bypass.".to_string(),
                    valid_from: "2026-04-17T10:02:00Z".to_string(),
                    ended: String::new(),
                },
            ],
            path: PathBuf::from(".cortyx/neurons/_kg_agent_reviewer.context.md"),
        },
        kg::KgEntity {
            entity: "auth".to_string(),
            facts: vec![KgFact {
                predicate: "owner".to_string(),
                value: "platform-team".to_string(),
                valid_from: "2026-04-17T10:03:00Z".to_string(),
                ended: String::new(),
            }],
            path: PathBuf::from(".cortyx/neurons/_kg_auth.context.md"),
        },
    ];
    let reasoning = ReasoningReport {
        nodes: vec![ReasonedNode {
            path: PathBuf::from(".cortyx/neurons/src/engine.context.md"),
            score: 0.82,
            depth: 1,
            kind: Some(NeuronKind::Core),
            module: Some("engine".to_string()),
            summary: Some("Engine auth entrypoint".to_string()),
            supporting: vec![PathBuf::from(".cortyx/neurons/src/auth.context.md")],
            strongest_step: None,
            is_seed: false,
            is_kg_entity: false,
        }],
        facts: vec![ReasonedFact::new(
            PathBuf::from(".cortyx/neurons/_kg_auth.context.md"),
            "auth".to_string(),
            "owner".to_string(),
            "platform-team".to_string(),
            0.91,
            vec![PathBuf::from(".cortyx/neurons/src/engine.context.md")],
            true,
            "2026-04-17T10:03:00Z".to_string(),
            String::new(),
        )],
        conflicts: Vec::new(),
        ..Default::default()
    };

    project_collaboration_state(&[diary], &[sync], &kg_entities, Some(&reasoning))
}

#[test]
fn select_delta_items_emits_only_new_and_changed_chunks() {
    let old_a = test_item("a.context.md", "A1");
    let old_b = test_item("b.context.md", "B1");
    let old_d = test_item("d.context.md", "D1");
    let previous = HashMap::from([
        (old_a.path.clone(), old_a.fingerprint.clone()),
        (old_b.path.clone(), old_b.fingerprint.clone()),
        (old_d.path.clone(), old_d.fingerprint.clone()),
    ]);

    let items = vec![
        test_item("a.context.md", "A1"),
        test_item("b.context.md", "B2"),
        test_item("c.context.md", "C1"),
    ];
    let delta = select_delta_items(&items, Some(&previous));

    assert_eq!(delta.unchanged, 1);
    assert_eq!(delta.removed, 1);
    assert_eq!(delta.emitted.len(), 2);
    assert_eq!(delta.emitted[0].path, PathBuf::from("b.context.md"));
    assert_eq!(delta.emitted[1].path, PathBuf::from("c.context.md"));
}

#[test]
fn select_delta_items_emits_full_set_without_snapshot() {
    let items = vec![
        test_item("a.context.md", "A1"),
        test_item("b.context.md", "B1"),
    ];
    let delta = select_delta_items(&items, None);

    assert_eq!(delta.unchanged, 0);
    assert_eq!(delta.removed, 0);
    assert_eq!(delta.emitted.len(), 2);
}

#[test]
fn select_capsule_modules_prefers_explicit_or_dominant_module() {
    let items = vec![
        (PathBuf::from("auth.context.md"), 8.0),
        (PathBuf::from("guard.context.md"), 6.5),
        (PathBuf::from("ui.context.md"), 4.0),
    ];
    let modules = HashMap::from([
        (PathBuf::from("auth.context.md"), "auth".to_string()),
        (PathBuf::from("guard.context.md"), "auth".to_string()),
        (PathBuf::from("ui.context.md"), "ui".to_string()),
    ]);

    assert_eq!(select_capsule_modules(&items, None, &modules), vec!["auth"]);
    assert_eq!(
        select_capsule_modules(&items, Some("ui"), &modules),
        vec!["ui"]
    );
    assert!(select_capsule_modules(&items, Some("@alice"), &modules).is_empty());
}

#[test]
fn select_capsule_anchor_paths_keeps_top_dynamic_neurons() {
    let items = vec![
        (PathBuf::from("auth.context.md"), 9.0),
        (PathBuf::from("guard.context.md"), 6.0),
        (PathBuf::from("session.context.md"), 4.0),
        (PathBuf::from("ui.context.md"), 7.0),
    ];
    let modules = HashMap::from([
        (PathBuf::from("auth.context.md"), "auth".to_string()),
        (PathBuf::from("guard.context.md"), "auth".to_string()),
        (PathBuf::from("session.context.md"), "auth".to_string()),
        (PathBuf::from("ui.context.md"), "ui".to_string()),
    ]);
    let active = HashSet::from(["auth".to_string()]);

    let keep = select_capsule_anchor_paths(&items, &active, &modules);
    assert!(keep.contains(&PathBuf::from("auth.context.md")));
    assert!(keep.contains(&PathBuf::from("guard.context.md")));
    assert!(!keep.contains(&PathBuf::from("session.context.md")));
    assert!(!keep.contains(&PathBuf::from("ui.context.md")));
}

#[test]
fn render_context_item_reports_summary_read_error() {
    let path = PathBuf::from("missing.context.md");
    let rendered = render_context_item(&path, 4.0, &[], &NeuronIndex::default());
    assert!(rendered.rendered.contains("read error"));
}

#[test]
fn render_context_item_strips_answer_and_query_surface_sections() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("example.context.md");
    fs::write(
        &path,
        "# Example\n\nUseful body.\n\n## answer_surface\n<!-- SECTION: answer_surface -->\n| question_pattern | answer_span | confidence |\n| --- | --- | --- |\n| role | reviewer | 0.9 |\n<!-- /SECTION -->\n\n## query_surface\n<!-- SECTION: query_surface -->\n- audit auth route\n<!-- /SECTION -->\n\n## evidence_surface\n<!-- SECTION: evidence_surface -->\n[{\"entity\":\"Alice\",\"predicate\":\"role\",\"value\":\"reviewer\"}]\n<!-- /SECTION -->\n",
    )
    .unwrap();

    let rendered = render_context_item(&path, 9.0, &[], &NeuronIndex::default());
    assert!(rendered.rendered.contains("Useful body."));
    assert!(!rendered.rendered.contains("answer_surface"));
    assert!(!rendered.rendered.contains("query_surface"));
    assert!(!rendered.rendered.contains("evidence_surface"));
    assert!(!rendered.rendered.contains("reviewer"));
    assert!(!rendered.rendered.contains("audit auth route"));
    assert!(!rendered.rendered.contains("Alice"));
}

#[test]
fn render_context_item_uses_focused_excerpt_for_large_verbatim_contexts() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("conversation.verbatim.md");
    let filler =
        "Assistant: We also talked about side topics that are not relevant right now.\n".repeat(24);
    let content = format!(
        "User: I asked about travel plans.\nAssistant: We discussed itineraries.\n{filler}User: The venue I picked was Revolution Hall in Portland.\nAssistant: Revolution Hall is a great choice for indie music.\nUser: I also booked dinner nearby.\nAssistant: Enjoy the show.\n"
    );
    fs::write(&path, content).unwrap();

    let rendered = render_context_item(
        &path,
        6.0,
        &[
            "portland".to_string(),
            "venue".to_string(),
            "indie".to_string(),
        ],
        &NeuronIndex::default(),
    );
    assert!(rendered.rendered.contains("focused"));
    assert!(rendered.rendered.contains("Revolution Hall"));
    assert!(!rendered.rendered.contains("travel plans"));
}

#[test]
fn render_context_item_prefers_key_markdown_sections_in_focused_mode() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("auth.context.md");
    let notes =
        "Additional historical notes about auth migrations and rollout details.\n".repeat(24);
    let content = format!(
        "# Auth\n\n## purpose\nKeep tokens low for auth tasks.\n\n## api\nUse `require_auth()` before accessing the session.\n\n## pitfalls\nDo not trust unsigned cookies.\n\n## notes\n{notes}\n"
    );
    fs::write(&path, content).unwrap();

    let rendered = render_context_item(
        &path,
        6.2,
        &["auth".to_string(), "fix".to_string(), "session".to_string()],
        &NeuronIndex::default(),
    );
    assert!(rendered.rendered.contains("## purpose"));
    assert!(rendered.rendered.contains("## api"));
    assert!(rendered.rendered.contains("## pitfalls"));
    assert!(!rendered.rendered.contains("## notes"));
}

#[test]
fn render_module_capsule_reports_read_error() {
    let dir = tempfile::tempdir().unwrap();
    let capsule_path = module_capsule_path(dir.path(), "auth");
    fs::create_dir_all(&capsule_path).unwrap();

    let rendered = render_module_capsule(dir.path(), "auth").unwrap();
    assert!(rendered.rendered.contains("read error"));
}

#[test]
fn render_agent_memory_summary_uses_structured_diary_fields() {
    let content = render_structured_diary_entry(
        "reviewer",
        "Investigated auth middleware coverage.",
        Some("Audit auth middleware"),
        Some("done"),
        Some("Close the auth bypass."),
        Some("Patch the legacy REST route."),
        Some("Waiting on route ownership clarification."),
        Some("Found a bypass in the legacy route."),
        &["auth".to_string(), "middleware".to_string()],
        &["router-owner".to_string()],
    );
    let summary = render_agent_memory_summary(&content, Some(1_710_000_000));
    assert!(summary.contains("Audit auth middleware"));
    assert!(summary.contains("status: done"));
    assert!(summary.contains("goal: Close the auth bypass."));
    assert!(summary.contains("blocker: Waiting on route ownership clarification."));
    assert!(summary.contains("Found a bypass in the legacy route."));
}

#[test]
fn flush_provisional_hits_blocking_clears_without_training_feedback() {
    let dir = tempfile::tempdir().unwrap();
    let neuron_path = dir.path().join("example.context.md");
    fs::write(&neuron_path, "example").unwrap();
    let meta = NeuronMeta::new_stub(&neuron_path, NeuronKind::Core);
    fs::write(
        meta_path(&neuron_path),
        serde_json::to_string(&meta).unwrap(),
    )
    .unwrap();

    let mut idx = NeuronIndex::default();
    idx.index_neuron(&neuron_path, "example context body", &meta);
    let before = idx.use_count_for(&neuron_path);
    let index = std::sync::Arc::new(tokio::sync::RwLock::new(idx));
    let provisional = std::sync::Arc::new(tokio::sync::Mutex::new(vec![neuron_path.clone()]));

    let cleared = flush_provisional_hits_blocking(&index, &provisional).unwrap();

    assert_eq!(cleared, 1);
    assert!(provisional.blocking_lock().is_empty());
    assert_eq!(index.blocking_read().use_count_for(&neuron_path), before);
}

#[test]
fn resolve_neuron_store_path_accepts_neuron_and_rejects_escape() {
    let dir = tempfile::tempdir().unwrap();
    let neuron_root = neuron_dir(dir.path());
    fs::create_dir_all(&neuron_root).unwrap();
    let neuron_path = neuron_root.join("example.context.md");
    fs::write(&neuron_path, "hello").unwrap();
    let outside = dir.path().join("outside.context.md");
    fs::write(&outside, "nope").unwrap();

    let resolved =
        resolve_neuron_store_path(&neuron_path.display().to_string(), dir.path()).unwrap();
    assert_eq!(resolved, neuron_path.canonicalize().unwrap());
    assert!(resolve_neuron_store_path(&outside.display().to_string(), dir.path()).is_err());
}

#[test]
fn build_augmented_task_includes_editor_and_error_terms() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = NeuronIndex::default();
    let source = dir.path().join("src").join("auth.rs");
    fs::create_dir_all(source.parent().unwrap()).unwrap();
    fs::write(&source, "fn auth() {}").unwrap();
    let neuron_path = core_neuron_path(&source, dir.path());
    fs::create_dir_all(neuron_path.parent().unwrap()).unwrap();
    let mut meta = NeuronMeta::new_stub(&source, NeuronKind::Core);
    meta.tokens =
        crate::neuron::estimate_context_tokens("token validation middleware refresh auth session")
            .get();
    index.index_neuron(
        &neuron_path,
        "token validation middleware refresh auth session",
        &meta,
    );

    let input = GetContextsInput {
        task: "fix auth".to_string(),
        max_tokens: None,
        module: None,
        person: None,
        kind: None,
        min_confidence: None,
        multi_hop: None,
        previous_response: None,
        open_files: Some(vec!["src/auth.rs".to_string()]),
        error_context: Some("middleware validation failed".to_string()),
        delta_mode: None,
        context_handle: None,
        capsule_mode: None,
        answer_mode: None,
        min_answer_confidence: None,
        provenance_mode: None,
        depth: None,
        temporal_bias: None,
    };

    let augmented = build_augmented_task(&index, &input);
    assert!(augmented.contains("fix auth"));
    assert!(augmented.contains("middleware"));
    assert!(augmented.contains("validation"));
}

#[test]
fn cortyx_route_auto_uses_answer_for_questions() {
    let route = derive_cortyx_route(&CortyxInput {
        intent: None,
        task: Some("What is my job?".to_string()),
        agent: None,
        person: None,
        module: None,
        kind: None,
        path: None,
        max_tokens: None,
        min_confidence: None,
        multi_hop: None,
        previous_response: None,
        delta_mode: None,
        context_handle: None,
        capsule_mode: None,
        min_answer_confidence: None,
        provenance_mode: None,
        include_timeline: None,
    })
    .unwrap();
    assert_eq!(route.kind, CortyxRouteKind::Answer);
}

#[test]
fn cortyx_route_auto_uses_agent_status_for_agent_only() {
    let route = derive_cortyx_route(&CortyxInput {
        intent: None,
        task: None,
        agent: Some("reviewer".to_string()),
        person: None,
        module: None,
        kind: None,
        path: None,
        max_tokens: None,
        min_confidence: None,
        multi_hop: None,
        previous_response: None,
        delta_mode: None,
        context_handle: None,
        capsule_mode: None,
        min_answer_confidence: None,
        provenance_mode: None,
        include_timeline: None,
    })
    .unwrap();
    assert_eq!(route.kind, CortyxRouteKind::AgentStatus);
    assert_eq!(route.agent.as_deref(), Some("reviewer"));
}

#[test]
fn cortyx_route_auto_without_inputs_uses_capability_summary() {
    let route = derive_cortyx_route(&CortyxInput {
        intent: None,
        task: None,
        agent: None,
        person: None,
        module: None,
        kind: None,
        path: None,
        max_tokens: None,
        min_confidence: None,
        multi_hop: None,
        previous_response: None,
        delta_mode: None,
        context_handle: None,
        capsule_mode: None,
        min_answer_confidence: None,
        provenance_mode: None,
        include_timeline: None,
    })
    .unwrap();
    assert_eq!(route.kind, CortyxRouteKind::Capabilities);
    assert!(route.task.is_none());
    assert!(route.agent.is_none());
}

#[test]
fn cortyx_route_auto_uses_wake_up_for_priming_request() {
    let route = derive_cortyx_route(&CortyxInput {
        intent: None,
        task: Some("Wake up the session with reviewer memory".to_string()),
        agent: Some("reviewer".to_string()),
        person: None,
        module: None,
        kind: None,
        path: None,
        max_tokens: None,
        min_confidence: None,
        multi_hop: None,
        previous_response: None,
        delta_mode: None,
        context_handle: None,
        capsule_mode: None,
        min_answer_confidence: None,
        provenance_mode: None,
        include_timeline: None,
    })
    .unwrap();
    assert_eq!(route.kind, CortyxRouteKind::WakeUp);
}

#[tokio::test]
async fn benchmark_cortyx_routes_answer_intent_to_answer_mode() {
    let dir = tempfile::tempdir().unwrap();
    let mut idx = NeuronIndex::load_or_create(dir.path()).unwrap();
    miner::mine_text(
        "I work as a pediatric nurse at the city hospital.",
        "diary",
        dir.path(),
        &mut idx,
        None,
        Some("user"),
        Some("2026-04-17T10:00:00Z"),
    )
    .unwrap();
    let server = CortyxServer::for_benchmark(dir.path().to_path_buf(), idx);
    let output = server
        .benchmark_cortyx(CortyxInput {
            intent: Some("answer".to_string()),
            task: Some("What is my job?".to_string()),
            agent: None,
            person: None,
            module: None,
            kind: None,
            path: None,
            max_tokens: Some(4000),
            min_confidence: None,
            multi_hop: None,
            previous_response: None,
            delta_mode: None,
            context_handle: None,
            capsule_mode: None,
            min_answer_confidence: None,
            provenance_mode: Some(false),
            include_timeline: None,
        })
        .await;
    assert!(output.to_ascii_lowercase().contains("pediatric nurse"));
}

#[tokio::test]
async fn benchmark_cortyx_without_inputs_returns_capability_summary() {
    let dir = tempfile::tempdir().unwrap();
    let idx = NeuronIndex::load_or_create(dir.path()).unwrap();
    let server = CortyxServer::for_benchmark(dir.path().to_path_buf(), idx);

    let output = server
        .benchmark_cortyx(CortyxInput {
            intent: None,
            task: None,
            agent: None,
            person: None,
            module: None,
            kind: None,
            path: None,
            max_tokens: None,
            min_confidence: None,
            multi_hop: None,
            previous_response: None,
            delta_mode: None,
            context_handle: None,
            capsule_mode: None,
            min_answer_confidence: None,
            provenance_mode: None,
            include_timeline: None,
        })
        .await;

    assert!(output.contains("Cortyx capability summary"));
    assert!(output.contains("Default entrypoint: cortyx(task=\"...\")"));
    assert!(output.contains("shared sync: 0 pending item(s), 0 conflict(s)"));
}

#[tokio::test]
async fn benchmark_cortyx_scopes_agent_questions_to_agent_memory() {
    let dir = tempfile::tempdir().unwrap();
    let mut idx = NeuronIndex::load_or_create(dir.path()).unwrap();
    let diary = render_structured_diary_entry(
        "reviewer",
        "Audited the legacy auth route.",
        Some("Close auth bypass"),
        Some("in_progress"),
        Some("Close the auth bypass without regressing login."),
        Some("Patch the legacy REST route after ownership is confirmed."),
        Some("Waiting on route ownership clarification."),
        Some("Confirmed the bypass only exists on the legacy REST path."),
        &["auth".to_string(), "routing".to_string()],
        &["router-owner".to_string()],
    );
    miner::mine_text(
        &diary,
        "diary",
        dir.path(),
        &mut idx,
        Some("@agent/reviewer"),
        None,
        Some("2026-04-17T10:00:00Z"),
    )
    .unwrap();
    let server = CortyxServer::for_benchmark(dir.path().to_path_buf(), idx);
    let output = server
        .benchmark_cortyx(CortyxInput {
            intent: None,
            task: Some("What is the reviewer's goal?".to_string()),
            agent: Some("reviewer".to_string()),
            person: None,
            module: None,
            kind: None,
            path: None,
            max_tokens: Some(4000),
            min_confidence: None,
            multi_hop: None,
            previous_response: None,
            delta_mode: None,
            context_handle: None,
            capsule_mode: None,
            min_answer_confidence: None,
            provenance_mode: Some(false),
            include_timeline: None,
        })
        .await;
    assert!(output
        .to_ascii_lowercase()
        .contains("close the auth bypass without regressing login"));
}

#[tokio::test]
async fn benchmark_cortyx_answer_mode_can_abstain_with_min_answer_confidence() {
    let dir = tempfile::tempdir().unwrap();
    let mut idx = NeuronIndex::load_or_create(dir.path()).unwrap();
    miner::mine_text(
        "Work has been stressful lately, and I keep thinking about my future career path.",
        "diary",
        dir.path(),
        &mut idx,
        None,
        Some("user"),
        Some("2026-04-17T10:00:00Z"),
    )
    .unwrap();
    let server = CortyxServer::for_benchmark(dir.path().to_path_buf(), idx);
    let output = server
        .benchmark_cortyx(CortyxInput {
            intent: Some("answer".to_string()),
            task: Some("What is my job?".to_string()),
            agent: None,
            person: None,
            module: None,
            kind: None,
            path: None,
            max_tokens: Some(4000),
            min_confidence: None,
            multi_hop: None,
            previous_response: None,
            delta_mode: None,
            context_handle: None,
            capsule_mode: None,
            min_answer_confidence: Some(0.6),
            provenance_mode: Some(false),
            include_timeline: None,
        })
        .await;
    assert!(output.trim().is_empty());
}

#[tokio::test]
async fn mutation_tools_record_provenance_history() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("src").join("auth.rs");
    let target_source = dir.path().join("src").join("guard.rs");
    fs::create_dir_all(source.parent().unwrap()).unwrap();
    fs::write(&source, "fn auth() -> bool { true }\n").unwrap();
    fs::write(&target_source, "fn guard() {}\n").unwrap();

    let idx = NeuronIndex::load_or_create(dir.path()).unwrap();
    let server = CortyxServer::for_benchmark(dir.path().to_path_buf(), idx);

    let initial_content =
        "# Auth\n\n## purpose\n<!-- SECTION: purpose -->\nInitial purpose.\n<!-- /SECTION -->\n";
    let evolve = server
        .evolve_context(Parameters(EvolveContextInput {
            path: "src/auth.rs".to_string(),
            content: initial_content.to_string(),
        }))
        .await;
    assert!(evolve.contains("Neuron evolved"));

    let neuron_path = core_neuron_path(&source, dir.path());
    let create_provenance = load_provenance(&neuron_path).unwrap().unwrap();
    assert_eq!(create_provenance.edit_history.len(), 1);
    let create_edit = &create_provenance.edit_history[0];
    let initial_hash = provenance_content_hash(initial_content);
    assert_eq!(create_edit.operation, ProvenanceOperation::Create);
    assert_eq!(create_edit.source, ProvenanceSource::Local);
    assert_eq!(
        create_edit.summary.as_deref(),
        Some("created neuron from src/auth.rs")
    );
    assert_eq!(
        create_edit.content_hash.as_deref(),
        Some(initial_hash.as_str())
    );
    let create_edit_id = create_edit.edit_id.clone();

    let update = server
        .evolve_section(Parameters(EvolveSectionInput {
            path: "src/auth.rs".to_string(),
            section: "purpose".to_string(),
            content: "Refined purpose.".to_string(),
        }))
        .await;
    assert!(update.contains("Section 'purpose' updated"));

    let updated_provenance = load_provenance(&neuron_path).unwrap().unwrap();
    assert_eq!(updated_provenance.edit_history.len(), 2);
    let section_edit = updated_provenance.edit_history.last().unwrap();
    assert_eq!(section_edit.operation, ProvenanceOperation::SectionUpdate);
    assert_eq!(section_edit.source, ProvenanceSource::Local);
    assert_eq!(section_edit.section.as_deref(), Some("purpose"));
    assert_eq!(
        section_edit.summary.as_deref(),
        Some("updated purpose section for src/auth.rs")
    );
    assert_eq!(
        section_edit.parent_edit_id.as_deref(),
        Some(create_edit_id.as_str())
    );

    let rollback = server
        .rollback_section(Parameters(RollbackSectionInput {
            neuron_path: neuron_path.display().to_string(),
            section: "purpose".to_string(),
        }))
        .await;
    assert!(rollback.contains("Restored section 'purpose'"));

    let rolled_back_provenance = load_provenance(&neuron_path).unwrap().unwrap();
    assert_eq!(rolled_back_provenance.edit_history.len(), 3);
    let rollback_edit = rolled_back_provenance.edit_history.last().unwrap();
    assert_eq!(rollback_edit.operation, ProvenanceOperation::Rollback);
    assert_eq!(rollback_edit.source, ProvenanceSource::Local);
    assert_eq!(rollback_edit.section.as_deref(), Some("purpose"));
    assert_eq!(
        rollback_edit.summary.as_deref(),
        Some("restored purpose from rollback shadow")
    );
    let rollback_edit_id = rollback_edit.edit_id.clone();

    let target_neuron = core_neuron_path(&target_source, dir.path());
    fs::create_dir_all(target_neuron.parent().unwrap()).unwrap();
    fs::write(&target_neuron, "# Guard\n").unwrap();

    let neuron_root = neuron_dir(dir.path());
    let source_rel = neuron_path
        .strip_prefix(&neuron_root)
        .unwrap()
        .to_string_lossy()
        .replace('\\', "/");
    let target_rel = target_neuron
        .strip_prefix(&neuron_root)
        .unwrap()
        .to_string_lossy()
        .replace('\\', "/");

    let synapse = server
        .create_synapse(Parameters(CreateSynapseInput {
            source: source_rel.clone(),
            target: target_rel.clone(),
            reason: "imports guard helpers".to_string(),
            edge_type: Some(SynapseType::Imports),
        }))
        .await;
    assert!(synapse.contains("Synapse created"));

    let synapse_provenance = load_provenance(&neuron_path).unwrap().unwrap();
    assert_eq!(synapse_provenance.edit_history.len(), 4);
    let synapse_edit = synapse_provenance.edit_history.last().unwrap();
    let synapse_summary = format!("added synapse to {target_rel}");
    let current_hash = provenance_content_hash(&fs::read_to_string(&neuron_path).unwrap());
    assert_eq!(synapse_edit.operation, ProvenanceOperation::SectionUpdate);
    assert_eq!(synapse_edit.source, ProvenanceSource::Local);
    assert_eq!(synapse_edit.section.as_deref(), Some("cross-references"));
    assert_eq!(
        synapse_edit.summary.as_deref(),
        Some(synapse_summary.as_str())
    );
    assert_eq!(
        synapse_edit.parent_edit_id.as_deref(),
        Some(rollback_edit_id.as_str())
    );
    assert_eq!(
        synapse_edit.content_hash.as_deref(),
        Some(current_hash.as_str())
    );
}

#[tokio::test]
async fn extract_from_raw_records_import_provenance() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("src").join("router.rs");
    fs::create_dir_all(source.parent().unwrap()).unwrap();
    fs::write(&source, "fn route() {}\n").unwrap();

    let idx = NeuronIndex::load_or_create(dir.path()).unwrap();
    let server = CortyxServer::for_benchmark(dir.path().to_path_buf(), idx);

    let task_pattern = "audit auth routes";
    let extracted = server
        .extract_from_raw(Parameters(ExtractFromRawInput {
            path: "src/router.rs".to_string(),
            task_pattern: task_pattern.to_string(),
            chunk: "fn route() {}".to_string(),
            why: "It shows the auth guard order.".to_string(),
        }))
        .await;
    assert!(extracted.contains("Use-case neuron created"));

    let source_rel = "src/router.rs".replace(['/', '\\'], "_");
    let task_kebab = truncate_str(&to_kebab(task_pattern), 64);
    let neuron_path = neuron_dir(dir.path()).join(format!("{source_rel}.usecase.{task_kebab}.md"));
    let content = fs::read_to_string(&neuron_path).unwrap();
    let provenance = load_provenance(&neuron_path).unwrap().unwrap();
    assert_eq!(provenance.source_path.as_deref(), Some(source.as_path()));
    assert_eq!(provenance.edit_history.len(), 1);
    let edit = &provenance.edit_history[0];
    let expected_hash = provenance_content_hash(&content);
    assert_eq!(edit.operation, ProvenanceOperation::Create);
    assert_eq!(edit.source, ProvenanceSource::Import);
    assert_eq!(
        edit.summary.as_deref(),
        Some("extracted raw chunk for pattern \"audit auth routes\"")
    );
    assert_eq!(edit.content_hash.as_deref(), Some(expected_hash.as_str()));

    let meta: NeuronMeta =
        serde_json::from_str(&fs::read_to_string(meta_path(&neuron_path)).unwrap()).unwrap();
    assert!(!meta.source_hash.is_empty());
}

#[test]
fn sync_structured_diary_to_kg_replaces_active_agent_state() {
    let dir = tempfile::tempdir().unwrap();
    let mut idx = NeuronIndex::default();
    let first = parse_structured_diary_entry(&render_structured_diary_entry(
        "reviewer",
        "Investigated auth middleware coverage.",
        Some("Audit auth middleware"),
        Some("in_progress"),
        Some("Close the auth bypass."),
        Some("Patch the legacy REST route."),
        Some("Waiting on route ownership clarification."),
        Some("Tracing the auth bypass."),
        &["auth".to_string()],
        &["router-owner".to_string()],
    ))
    .unwrap();
    sync_structured_diary_to_kg(
        dir.path(),
        &mut idx,
        "reviewer",
        &first,
        "2026-04-17T10:00:00Z",
    )
    .unwrap();

    let second = parse_structured_diary_entry(&render_structured_diary_entry(
        "reviewer",
        "Patched the legacy REST route.",
        Some("Close auth bypass"),
        Some("done"),
        Some("Close the auth bypass."),
        Some("Ship the regression tests."),
        None,
        Some("Removed the legacy auth bypass."),
        &["auth".to_string(), "routing".to_string()],
        &["qa".to_string()],
    ))
    .unwrap();
    sync_structured_diary_to_kg(
        dir.path(),
        &mut idx,
        "reviewer",
        &second,
        "2026-04-17T10:05:00Z",
    )
    .unwrap();

    let entity = kg::KgEntity::load(&kg::kg_neuron_path(
        dir.path(),
        &agent_entity_name("reviewer"),
    ))
    .unwrap();
    assert_eq!(
        latest_active_kg_value(&entity, AGENT_STATUS_PREDICATE).as_deref(),
        Some("done")
    );
    assert_eq!(
        latest_active_kg_value(&entity, AGENT_FOCUS_PREDICATE).as_deref(),
        Some("Close auth bypass")
    );
    assert_eq!(
        latest_active_kg_value(&entity, AGENT_GOAL_PREDICATE).as_deref(),
        Some("Close the auth bypass.")
    );
    assert_eq!(
        latest_active_kg_value(&entity, AGENT_NEXT_STEP_PREDICATE).as_deref(),
        Some("Ship the regression tests.")
    );
    assert_eq!(
        latest_active_kg_value(&entity, AGENT_BLOCKER_PREDICATE),
        None
    );
    let related = active_kg_values(&entity, AGENT_RELATED_ENTITY_PREDICATE);
    assert_eq!(related, vec!["auth".to_string(), "routing".to_string()]);
    let depends_on = active_kg_values(&entity, AGENT_DEPENDS_ON_PREDICATE);
    assert_eq!(depends_on, vec!["qa".to_string()]);
    let status_timeline = entity.timeline_for(AGENT_STATUS_PREDICATE);
    assert_eq!(status_timeline.len(), 2);
    assert_eq!(status_timeline[0].ended, "2026-04-17T10:05:00Z");
    let blocker_timeline = entity.timeline_for(AGENT_BLOCKER_PREDICATE);
    assert_eq!(blocker_timeline.len(), 1);
    assert_eq!(blocker_timeline[0].ended, "2026-04-17T10:05:00Z");
}

#[test]
fn render_collaboration_status_report_summarizes_team_sync_and_modules() {
    let projection = sample_collaboration_projection();

    let report = render_collaboration_status_report(&projection, None, None, true).expect("report");

    assert!(report.contains("## Collaboration Status"));
    assert!(report.contains("collaborators: 1"));
    assert!(report.contains("modules: 1"));
    assert!(report.contains("sync conflicts: 1"));
    assert!(report.contains("average trust score:"));
    assert!(report.contains("## Top collaborators"));
    assert!(report.contains("reviewer — sync_conflict"));
    assert!(report.contains("## Shared modules"));
    assert!(report.contains("engine — sync_conflict"));
    assert!(report.contains("## Collaboration timeline"));
}

#[test]
fn render_collaboration_status_report_filters_to_module() {
    let projection = sample_collaboration_projection();

    let report = render_collaboration_status_report(&projection, None, Some("engine"), true)
        .expect("module report");

    assert!(report.contains("## Collaboration Module: engine"));
    assert!(report.contains("attention: sync_conflict"));
    assert!(report.contains("collaborators: reviewer"));
    assert!(report.contains("pending sync: yes"));
    assert!(report.contains("trust score:"));
    assert!(report.contains("## Collaboration timeline"));
}

#[test]
fn render_agent_status_report_uses_temporal_kg_fallback() {
    let dir = tempfile::tempdir().unwrap();
    let mut idx = NeuronIndex::default();
    let entry = parse_structured_diary_entry(&render_structured_diary_entry(
        "reviewer",
        "Patched the legacy REST route.",
        Some("Close auth bypass"),
        Some("done"),
        Some("Close the auth bypass."),
        Some("Ship the regression tests."),
        Some("Waiting on QA sign-off."),
        Some("Removed the legacy auth bypass."),
        &["auth".to_string(), "routing".to_string()],
        &["qa".to_string()],
    ))
    .unwrap();
    sync_structured_diary_to_kg(
        dir.path(),
        &mut idx,
        "reviewer",
        &entry,
        "2026-04-17T10:05:00Z",
    )
    .unwrap();

    let report = render_agent_status_report(&idx, dir.path(), "reviewer", true).unwrap();
    assert!(report.contains("## Agent Status: reviewer"));
    assert!(report.contains("attention: blocked"));
    assert!(report.contains("focus: Close auth bypass"));
    assert!(report.contains("status: done"));
    assert!(report.contains("goal: Close the auth bypass."));
    assert!(report.contains("next step: Ship the regression tests."));
    assert!(report.contains("blocker: Waiting on QA sign-off."));
    assert!(report.contains("depends on: qa"));
    assert!(report.contains("Removed the legacy auth bypass."));
    assert!(report.contains("pending sync: no"));
    assert!(report.contains("## Supporting facts"));
    assert!(report.contains("## Collaboration timeline"));
}
