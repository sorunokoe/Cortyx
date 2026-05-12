//! Unit and integration tests for NeuronIndex.
//! Extracted from mod.rs to keep the main file focused.

use super::*;
use crate::neuron::{NeuronKind, NeuronMeta, NeuronStatus, Synapse, SynapseType};
use crate::types::{SynapseWeight, TermFrequency};
use tempfile::TempDir;

fn make_index(dir: &TempDir) -> NeuronIndex {
    NeuronIndex::load_or_create(dir.path()).unwrap()
}

fn index_verbatim_neuron(
    idx: &mut NeuronIndex,
    dir: &TempDir,
    file_name: &str,
    content: &str,
) -> PathBuf {
    let path = dir.path().join(".cortyx").join("neurons").join(file_name);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, content).unwrap();
    let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Verbatim);
    idx.index_neuron(&path, content, &meta);
    idx.rebuild_derived();
    path
}

fn read_answer_text(idx: &NeuronIndex, task: &str) -> String {
    let path = idx
        .derived_answer_path_for_task(task)
        .expect("expected synthetic answer");
    std::fs::read_to_string(path).unwrap()
}

#[test]
fn get_contexts_returns_empty_for_no_match() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let p = ndir.join("foo.context.md");
    std::fs::write(&p, "completely unrelated content xyz").unwrap();
    let mut idx = make_index(&dir);
    let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    idx.index_neuron(&p, "completely unrelated content xyz", &meta);
    idx.rebuild_derived();
    let result = idx.get_contexts("authentication oauth jwt", 4096, None, None);
    assert!(result.is_empty() || !result.contains(&p));
}

#[test]
fn get_contexts_respects_token_budget() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);
    for i in 0..20 {
        let p = ndir.join(format!("mod_{i:02}.context.md"));
        let content = format!("auth token login validate {} {}", "word ".repeat(200), i);
        std::fs::write(&p, &content).unwrap();
        let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
        idx.index_neuron(&p, &content, &meta);
    }
    idx.rebuild_derived();
    let result = idx.get_contexts("auth token login", 500, None, None);
    let total_tokens: usize = result
        .iter()
        .filter_map(|p| idx.entry_by_path(p))
        .map(|e| e.tokens)
        .sum();
    assert!(
        total_tokens <= 500,
        "should respect token budget: {total_tokens}"
    );
}

#[test]
fn get_contexts_module_filter() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);

    let auth_p = ndir.join("auth.context.md");
    let ui_p = ndir.join("ui.context.md");
    std::fs::write(&auth_p, "auth token login validate session").unwrap();
    std::fs::write(&ui_p, "auth login button render component").unwrap();

    let mut auth_meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    auth_meta.module = Some("auth".to_string());
    let mut ui_meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    ui_meta.module = Some("ui".to_string());

    idx.index_neuron(&auth_p, "auth token login validate session", &auth_meta);
    idx.index_neuron(&ui_p, "auth login button render component", &ui_meta);
    idx.rebuild_derived();

    // With module filter: only auth module
    let filtered = idx.get_contexts("auth login", 4096, Some("auth"), None);
    assert!(filtered.contains(&auth_p));
    assert!(
        !filtered.contains(&ui_p),
        "module filter should exclude ui module"
    );
}

#[test]
fn save_writes_module_capsule_for_named_module() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);

    let auth_p = ndir.join("auth.context.md");
    let guard_p = ndir.join("guard.context.md");
    let db_p = ndir.join("db.context.md");

    let auth_content = "# Auth\n\n## purpose\nHandles login and session validation.\n\n## pitfalls\nRotate refresh tokens after every use.\n";
    let guard_content =
        "# Guard\n\n## purpose\nProtects private routes and rejects anonymous requests.\n\n## pitfalls\nRequire identity before accessing private handlers.\n";
    let db_content =
        "# DB\n\n## purpose\nPersists user records and token state.\n\n## pitfalls\nKeep writes transactional.\n";

    std::fs::write(&auth_p, auth_content).unwrap();
    std::fs::write(&guard_p, guard_content).unwrap();
    std::fs::write(&db_p, db_content).unwrap();

    let mut auth_meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    auth_meta.module = Some("auth".to_string());
    auth_meta.synapses = vec![Synapse {
        target: db_p.clone(),
        edge_type: SynapseType::Calls,
        weight: crate::types::SynapseWeight::new(0.8),
        reason: "loads user token state".to_string(),
        learned_weight: SynapseWeight::ZERO,
        traversal_count: 0,
        last_co_activation_day: 0,
    }];
    let mut guard_meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    guard_meta.module = Some("auth".to_string());
    let mut db_meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    db_meta.module = Some("db".to_string());

    idx.index_neuron(&auth_p, auth_content, &auth_meta);
    idx.index_neuron(&guard_p, guard_content, &guard_meta);
    idx.index_neuron(&db_p, db_content, &db_meta);
    idx.rebuild_derived();
    idx.save().unwrap();

    let capsule = std::fs::read_to_string(module_capsule_path(dir.path(), "auth")).unwrap();
    assert!(capsule.contains("# Module capsule: auth"));
    assert!(capsule.contains("## module purpose"));
    assert!(capsule.contains("Handles login and session validation."));
    assert!(capsule.contains("## key apis / invariants"));
    assert!(capsule.contains("## critical pitfalls"));
    assert!(capsule.contains("## dominant dependencies"));
    assert!(capsule.contains("`db` (1 cross-module edges)"));
}

// ── Typed synapse traversal ───────────────────────────────────────────────

#[test]
fn synapse_traversal_pulls_related_neuron() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("engine.rs"),
        "pub fn engine() { route_intent(); }",
    )
    .unwrap();
    std::fs::write(dir.path().join("ui.rs"), "pub fn render() {}").unwrap();

    let mut idx = NeuronIndex::load_or_create(dir.path()).unwrap();
    idx.compile().unwrap();

    let engine_neuron = crate::neuron::core_neuron_path(&dir.path().join("engine.rs"), dir.path());
    let ui_neuron = crate::neuron::core_neuron_path(&dir.path().join("ui.rs"), dir.path());

    let engine_content = format!(
        "Engine module. Routes user intent, synthesizes responses.\n\
         ## CROSS-REFERENCES (synapses)\n- `{}` → render pipeline [calls]",
        ui_neuron.display()
    );
    let mut engine_meta = NeuronMeta::new_stub(&dir.path().join("engine.rs"), NeuronKind::Core);
    engine_meta.synapses = vec![Synapse {
        target: ui_neuron.clone(),
        edge_type: SynapseType::Calls,
        weight: crate::types::SynapseWeight::new(0.8),
        reason: "render pipeline".to_string(),
        learned_weight: SynapseWeight::ZERO,
        traversal_count: 0,
        last_co_activation_day: 0,
    }];
    engine_meta.status = NeuronStatus::Fresh;
    std::fs::write(&engine_neuron, &engine_content).unwrap();
    idx.upsert_neuron(&engine_neuron, &engine_content, &engine_meta)
        .unwrap();

    let contexts = idx.get_contexts("route intent synthesize engine", 4096, None, None);
    assert!(
        contexts.contains(&ui_neuron) || contexts.contains(&engine_neuron),
        "Synapse traversal should pull in related neuron. Got: {contexts:?}"
    );
}

#[test]
fn typed_synapse_implements_has_high_multiplier() {
    assert!(
        SynapseType::Implements.type_multiplier() > SynapseType::SemanticRelated.type_multiplier()
    );
    assert_eq!(SynapseType::ConceptExpands.type_multiplier(), 1.0);
}

// ── Use-case activation ───────────────────────────────────────────────────

#[test]
fn use_case_neuron_activated_by_task_pattern() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);

    let core_p = ndir.join("auth_rs.context.md");
    std::fs::write(&core_p, "authentication token validation").unwrap();
    let core_meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    idx.index_neuron(&core_p, "authentication token validation", &core_meta);

    let uc_p = ndir.join("auth_rs.usecase.oauth.md");
    std::fs::write(&uc_p, "OAuth2 flow: redirect then exchange code for token").unwrap();
    let mut uc_meta = NeuronMeta::new_stub(dir.path(), NeuronKind::UseCase);
    uc_meta.task_pattern = Some("add oauth login".to_string());
    uc_meta.parent = Some(core_p.clone());
    idx.index_neuron(
        &uc_p,
        "OAuth2 flow: redirect then exchange code for token",
        &uc_meta,
    );
    idx.rebuild_derived();

    let result = idx.get_contexts("add oauth authentication login", 4096, None, None);
    assert!(result.contains(&uc_p) || result.contains(&core_p));
}

// ── Invalidation ──────────────────────────────────────────────────────────

#[test]
fn invalidate_marks_stale() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("a.rs");
    std::fs::write(&file, "fn a() {}").unwrap();
    let mut idx = make_index(&dir);
    idx.compile().unwrap();
    let neuron = crate::neuron::core_neuron_path(&file, dir.path());
    assert!(neuron.exists());
    idx.invalidate(&file).unwrap();
    // Stale-demotion: neuron remains in the index (preserves context) but is
    // demoted via staleness_multiplier so it won't win over fresh neurons.
    let entry = idx.entries.iter().find(|e| e.neuron_path == neuron);
    assert!(
        entry.is_some(),
        "neuron should still exist after invalidation"
    );
    assert_eq!(
        entry.unwrap().staleness_multiplier,
        0.5,
        "staleness_multiplier should be 0.5 after invalidation"
    );
}

// ── BM25 scoring ──────────────────────────────────────────────────────────

#[test]
fn bm25_scores_zero_for_no_match() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let p = ndir.join("x.context.md");
    std::fs::write(&p, "completely different topic here").unwrap();
    let mut idx = make_index(&dir);
    let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    idx.index_neuron(&p, "completely different topic here", &meta);
    idx.rebuild_derived();
    let entry = idx.entry_by_path(&p).unwrap();
    assert_eq!(idx.bm25_score(&tokenize("auth token login"), entry), 0.0);
}

#[test]
fn bm25_scores_higher_for_matching_terms() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);
    let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    let p1 = ndir.join("a.context.md");
    std::fs::write(&p1, "auth login token session").unwrap();
    idx.index_neuron(&p1, "auth login token session", &meta);
    let p2 = ndir.join("b.context.md");
    std::fs::write(&p2, "render button component style").unwrap();
    idx.index_neuron(&p2, "render button component style", &meta);
    idx.rebuild_derived();
    let terms = tokenize("auth token");
    let s1 = idx.bm25_score(&terms, idx.entry_by_path(&p1).unwrap());
    let s2 = idx.bm25_score(&terms, idx.entry_by_path(&p2).unwrap());
    assert!(s1 > s2, "auth neuron should score higher for auth query");
}

#[test]
fn bm25_idf_is_non_negative() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);
    let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    // Same term in every entry → IDF should floor at 0
    for i in 0..5 {
        let p = ndir.join(format!("{i}.context.md"));
        std::fs::write(&p, "common term here").unwrap();
        idx.index_neuron(&p, "common term here", &meta);
    }
    idx.rebuild_derived();
    for entry in &idx.entries {
        let score = idx.bm25_score(&tokenize("common"), entry);
        assert!(score >= 0.0, "BM25 score must not be negative");
    }
}

// ── Overlap score ─────────────────────────────────────────────────────────

#[test]
fn overlap_score_perfect_match() {
    let q = tokenize("add dark mode");
    let p = tokenize("add dark mode");
    assert!((simple_overlap_score(&q, &p) - 1.0).abs() < 0.001);
}

#[test]
fn overlap_score_no_match() {
    let q = tokenize("auth token");
    let p = tokenize("render button");
    assert_eq!(simple_overlap_score(&q, &p), 0.0);
}

#[test]
fn overlap_score_empty_pattern() {
    let q = tokenize("auth");
    assert_eq!(simple_overlap_score(&q, &[]), 0.0);
}

// ── Tokenizer ─────────────────────────────────────────────────────────────

#[test]
fn tokenize_basic() {
    let terms = tokenize("add dark mode to SwiftUI view");
    assert!(terms.contains(&"add".to_string()));
    assert!(terms.contains(&"dark".to_string()));
    assert!(terms.contains(&"swiftui".to_string()));
    assert!(terms.contains(&"view".to_string()));
}

#[test]
fn tokenize_filters_short_terms() {
    let terms = tokenize("a b add");
    assert!(!terms.contains(&"a".to_string()));
    assert!(!terms.contains(&"b".to_string()));
    assert!(terms.contains(&"add".to_string()));
}

#[test]
fn tokenize_lowercases() {
    let terms = tokenize("AuthService");
    assert!(terms.contains(&"authservice".to_string()));
}

#[test]
fn tokenize_preserves_underscores() {
    let terms = tokenize("snake_case_name");
    assert!(terms.contains(&"snake_case_name".to_string()));
}

#[test]
fn tokenize_empty_string() {
    assert!(tokenize("").is_empty());
}

// ── Retrieval accuracy ────────────────────────────────────────────────────

/// Verifies that BM25 retrieval returns the correct neuron for each of 10
/// distinct queries against 10 distinct content-rich neurons.
///
/// This exercises the full activation pipeline (Phase 1 only — no synapses)
/// and ensures that keyword specificity drives correct ranking.
#[test]
fn get_contexts_retrieval_accuracy_10q() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);

    // Each neuron has a unique keyword cluster — e.g. "authentication" only in auth neuron
    let neurons = [
        (
            "auth.context.md",
            "authentication token validation session jwt bearer",
        ),
        (
            "ui.context.md",
            "render component dark mode swiftui colorscheme view",
        ),
        (
            "db.context.md",
            "database migration schema sql transaction commit",
        ),
        (
            "cache.context.md",
            "cache invalidation evict stale ttl expiry redis",
        ),
        (
            "api.context.md",
            "rest api endpoint http request response route handler",
        ),
        (
            "crypto.context.md",
            "encryption decryption aes rsa signing certificate key",
        ),
        (
            "queue.context.md",
            "queue task worker job priority scheduling async",
        ),
        (
            "logger.context.md",
            "logging tracing span event diagnostic telemetry",
        ),
        (
            "config.context.md",
            "configuration environment variable toml yaml dotenv",
        ),
        (
            "deploy.context.md",
            "deployment docker kubernetes helm release pipeline",
        ),
    ];
    let queries_and_expected: [(&str, &str); 10] = [
        ("jwt bearer authentication", "auth.context.md"),
        ("dark mode colorscheme swiftui", "ui.context.md"),
        ("sql transaction schema migration", "db.context.md"),
        ("cache ttl evict stale", "cache.context.md"),
        ("http rest api endpoint route", "api.context.md"),
        ("aes rsa encryption certificate", "crypto.context.md"),
        ("job worker queue scheduling", "queue.context.md"),
        ("logging span telemetry diagnostic", "logger.context.md"),
        (
            "environment variable dotenv configuration",
            "config.context.md",
        ),
        ("docker kubernetes deployment helm", "deploy.context.md"),
    ];

    for (name, content) in &neurons {
        let p = ndir.join(name);
        std::fs::write(&p, content).unwrap();
        let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
        idx.index_neuron(&p, content, &meta);
    }
    idx.rebuild_derived();

    let mut correct = 0;
    for (query, expected_file) in &queries_and_expected {
        let results = idx.get_contexts(query, 4096, None, None);
        let expected_path = ndir.join(expected_file);
        if results.contains(&expected_path) {
            correct += 1;
        } else {
            eprintln!("[accuracy] MISS: query={query:?} expected={expected_file} got={results:?}");
        }
    }
    assert_eq!(
        correct, 10,
        "BM25 accuracy: {correct}/10 correct (expected 10/10)"
    );
}

/// Activation latency: `get_contexts` over 100 neurons must complete in <50ms p95.
///
/// This verifies the README benchmark target "≤50ms p95, 100 neurons" is met
/// with the pure in-memory BM25 engine (no disk I/O in the hot path).
#[test]
fn get_contexts_latency_p95_100_neurons() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);

    // Build a 100-neuron index with realistic content sizes (~400 chars each).
    for i in 0..100 {
        let p = ndir.join(format!("neuron_{i:03}.context.md"));
        let content = format!(
            "## Module {i}\nHandles subsystem_{i} operations including routing, \
             caching, pipeline_{i} filter validation authentication token session \
             database migration schema endpoint handler deployment configuration \
             environment worker queue scheduling logging tracing telemetry encryption."
        );
        std::fs::write(&p, &content).unwrap();
        let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
        idx.index_neuron(&p, &content, &meta);
    }
    idx.rebuild_derived();

    // Warm up: one call to populate CPU caches
    let _ = idx.get_contexts("routing pipeline authentication token", 4096, None, None);

    // Measure p95 over 20 trials
    let trials = 20;
    let mut latencies_ms: Vec<u128> = (0..trials)
        .map(|_| {
            let t = std::time::Instant::now();
            let _ = idx.get_contexts("routing pipeline authentication token", 4096, None, None);
            t.elapsed().as_millis()
        })
        .collect();
    latencies_ms.sort_unstable();
    let p95 = latencies_ms[(trials as f64 * 0.95) as usize - 1];

    assert!(
        p95 < 50,
        "get_contexts p95 latency must be <50ms over 100 neurons; got {p95}ms"
    );
}

/// Ensures that relative synapse paths written into neuron markdown
/// (e.g. from `cortyx_evolve_context`) are resolved to absolute paths
/// in the adjacency graph, so traversal works correctly.
#[test]
fn relative_synapse_targets_resolved_in_adjacency() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);

    let source_p = ndir.join("engine.context.md");
    let target_p = ndir.join("ui.context.md");

    std::fs::write(&source_p, "engine routing intent").unwrap();
    std::fs::write(&target_p, "ui rendering components").unwrap();

    // Source neuron has a RELATIVE synapse target (as parse_synapses_from_content returns)
    let mut source_meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    source_meta.synapses = vec![Synapse {
        target: PathBuf::from("ui.context.md"), // relative!
        edge_type: SynapseType::Calls,
        weight: crate::types::SynapseWeight::new(0.9),
        reason: "calls render".to_string(),
        learned_weight: SynapseWeight::ZERO,
        traversal_count: 0,
        last_co_activation_day: 0,
    }];
    let target_meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);

    idx.index_neuron(&source_p, "engine routing intent", &source_meta);
    idx.index_neuron(&target_p, "ui rendering components", &target_meta);
    idx.rebuild_derived();

    // The adjacency entry for source_p should point to the ABSOLUTE target path
    let adj = idx
        .adjacency
        .get(&source_p)
        .expect("source must be in adjacency");
    let target_syn = adj.iter().find(|s| s.target == target_p);
    assert!(
        target_syn.is_some(),
        "Relative synapse 'ui.context.md' should be resolved to absolute {}: adjacency={adj:?}",
        target_p.display()
    );
}

// ── Mine + retrieve ───────────────────────────────────────────────────────

/// Verifies the conversation mining → retrieval pipeline end-to-end:
/// mine text containing unique keywords, then get_contexts should return it.
#[test]
fn mined_neuron_is_retrievable_by_keyword() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);

    // Mine a conversation turn with a specific keyword cluster
    crate::miner::mine_text(
        "The hydrazine valve regulates fuel injection in rocket propulsion systems.",
        "test_chat",
        dir.path(),
        &mut idx,
        None,
        Some("assistant"),
        None,
    )
    .unwrap();

    // The unique keyword "hydrazine" should retrieve the mined neuron
    let results = idx.get_contexts("hydrazine valve rocket propulsion", 4096, None, None);
    assert!(
        !results.is_empty(),
        "Mined neuron should be retrievable by its keywords"
    );

    let found = results.iter().any(|p| {
        std::fs::read_to_string(p)
            .map(|c| c.contains("hydrazine"))
            .unwrap_or(false)
    });
    assert!(found, "Retrieved neuron should contain 'hydrazine'");
}

/// Mine + module filter: mined neuron tagged with module X should only
/// appear when querying with that module filter, not unfiltered in other modules.
#[test]
fn mined_neuron_module_filter_works() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);

    crate::miner::mine_text(
        "Photosynthesis converts sunlight into glucose via chlorophyll.",
        "bio_chat",
        dir.path(),
        &mut idx,
        Some("biology"),
        Some("assistant"),
        None,
    )
    .unwrap();

    // Module-filtered query should find it
    let with_module = idx.get_contexts(
        "photosynthesis sunlight glucose",
        4096,
        Some("biology"),
        None,
    );
    assert!(
        !with_module.is_empty(),
        "Module-filtered query should find mined neuron"
    );

    // Module filter for a different module should NOT find it
    let wrong_module = idx.get_contexts(
        "photosynthesis sunlight glucose",
        4096,
        Some("physics"),
        None,
    );
    assert!(
        wrong_module.is_empty(),
        "Wrong module filter should not return neuron tagged 'biology'"
    );
}

// ── Feedback loop (hit_multiplier + quarantine) ───────────────────────────

#[test]
fn hit_multiplier_reward_grows_with_citations() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);

    let p = ndir.join("auth.context.md");
    std::fs::write(&p, "authentication token session login").unwrap();
    let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    idx.index_neuron(&p, "authentication token session login", &meta);
    idx.rebuild_derived();

    let terms = tokenize("auth login");

    // Cold-start: use_count=0 → multiplier=1.0 (neutral)
    let cold_score = idx.bm25_score(&terms, idx.entry_by_path(&p).unwrap());

    // Simulate MIN_SAMPLE_SIZE activations with 100% citation rate
    if let Some(&i) = idx.path_index.get(&p) {
        idx.entries[i].use_count = MIN_SAMPLE_SIZE;
        idx.entries[i].hit_count = MIN_SAMPLE_SIZE;
    }
    let hot_score = idx.bm25_score(&terms, idx.entry_by_path(&p).unwrap());

    assert!(
        hot_score > cold_score,
        "Fully-cited neuron should score higher than cold-start (hot={hot_score:.3}, cold={cold_score:.3})"
    );
    // Max multiplier is 1.5 so the hot score should be exactly 1.5× cold
    assert!(
        (hot_score / cold_score - 1.5).abs() < 0.01,
        "100% hit rate should give 1.5× boost"
    );
}

#[test]
fn auto_quarantine_fires_after_threshold() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);

    let p = ndir.join("noisy.context.md");
    std::fs::write(&p, "generic boilerplate content").unwrap();
    let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    idx.index_neuron(&p, "generic boilerplate content", &meta);
    idx.rebuild_derived();

    // Adaptive CI (S4): QUARANTINE_MIN_SAMPLES = 5. Below this threshold
    // (use_count 0–4), adaptive_quarantine_params returns None — no action.
    if let Some(&i) = idx.path_index.get(&p) {
        idx.entries[i].use_count = QUARANTINE_MIN_SAMPLES - 2; // = 3
        idx.entries[i].hit_count = 0;
    }
    idx.record_activation(&[p.clone()]); // → use_count = 4 (still below threshold)
    let mult_early = idx
        .path_index
        .get(&p)
        .map(|&i| idx.entries[i].staleness_multiplier)
        .unwrap_or(1.0);
    assert_eq!(
        mult_early, 1.0,
        "Should NOT quarantine below QUARANTINE_MIN_SAMPLES (4 < 5)"
    );

    // At use_count = 5 (after record_activation increments to 6), z=1.0 tier fires.
    // Wilson lower bound for 0/6 at z=1.0 = 0.0 < adaptive threshold 0.02 → quarantine.
    if let Some(&i) = idx.path_index.get(&p) {
        idx.entries[i].use_count = QUARANTINE_MIN_SAMPLES; // = 5
        idx.entries[i].hit_count = 0;
    }
    idx.record_activation(&[p.clone()]); // → use_count = 6, fires adaptive z=1.0
    let mult = idx
        .path_index
        .get(&p)
        .map(|&i| idx.entries[i].staleness_multiplier)
        .unwrap_or(1.0);
    assert_eq!(
        mult, 0.3,
        "Should quarantine at QUARANTINE_MIN_SAMPLES with 0% hit rate"
    );
}

#[test]
fn quarantine_is_reversible_when_citation_rate_recovers() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);

    let p = ndir.join("recovered.context.md");
    std::fs::write(&p, "generic boilerplate content").unwrap();
    let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    idx.index_neuron(&p, "generic boilerplate content", &meta);
    idx.rebuild_derived();

    // Manually quarantine the neuron, then simulate recovery: 20 uses, 10 hits.
    // Wilson lower bound for 10/20 at z=1.645 (90% CI) ≈ 0.31 > QUARANTINE_RECOVERY_THRESHOLD (0.15).
    // Use hardcoded values (not QUARANTINE_MIN_SAMPLES) so the hit/use ratio is valid.
    if let Some(&i) = idx.path_index.get(&p) {
        idx.entries[i].staleness_multiplier = 0.3;
        idx.entries[i].use_count = 20;
        idx.entries[i].hit_count = 10;
    }
    idx.record_activation(&[p.clone()]);
    let mult = idx
        .path_index
        .get(&p)
        .map(|&i| idx.entries[i].staleness_multiplier)
        .unwrap_or(0.0);
    assert!(
        mult > 0.3,
        "Quarantine should lift when citation rate recovers (mult={mult})"
    );
}

#[test]
fn wilson_lower_bound_correctness() {
    // 0/20 → lower bound = 0.0 (no hits, fully quarantinable)
    assert!(wilson_lower_bound(0, 20) < 0.01);
    // 10/20 → lower bound ≈ 0.299 (well above recovery threshold of 0.15)
    assert!(wilson_lower_bound(10, 20) > 0.25);
    // 1/20 → lower bound near 0 but small positive
    assert!(wilson_lower_bound(1, 20) < 0.10);
    // Edge: 0 total → 0.0
    assert_eq!(wilson_lower_bound(0, 0), 0.0);
}

// ── S1: AST Signature Hash ─────────────────────────────────────────────────

#[test]
fn sig_hash_changes_on_function_rename() {
    let before = crate::ast_extractor::extract_signatures("src/auth.rs", "pub fn validate() {}");
    let after = crate::ast_extractor::extract_signatures("src/auth.rs", "pub fn authenticate() {}");
    let h1 = crate::ast_extractor::compute_sig_hash(&before);
    let h2 = crate::ast_extractor::compute_sig_hash(&after);
    assert_ne!(h1, h2, "sig_hash must change when a function is renamed");
}

#[test]
fn sig_hash_stable_on_whitespace_and_comments() {
    let base = crate::ast_extractor::extract_signatures("src/auth.rs", "pub fn validate() {}");
    let tweaked = crate::ast_extractor::extract_signatures(
        "src/auth.rs",
        "/// New doc comment\npub fn validate() {\n    // added comment\n}",
    );
    let h1 = crate::ast_extractor::compute_sig_hash(&base);
    let h2 = crate::ast_extractor::compute_sig_hash(&tweaked);
    assert_eq!(
        h1, h2,
        "sig_hash must be stable across whitespace/doc-comment edits"
    );
}

#[test]
fn sig_hash_stable_on_function_reorder() {
    let a = crate::ast_extractor::extract_signatures(
        "src/auth.rs",
        "pub fn validate() {}\npub fn refresh() {}",
    );
    let b = crate::ast_extractor::extract_signatures(
        "src/auth.rs",
        "pub fn refresh() {}\npub fn validate() {}",
    );
    let h1 = crate::ast_extractor::compute_sig_hash(&a);
    let h2 = crate::ast_extractor::compute_sig_hash(&b);
    assert_eq!(h1, h2, "sig_hash must be stable across function reordering");
}

// ── S3: Lazy Sub-Neuron Splitting ─────────────────────────────────────────

#[test]
fn sub_neuron_path_format_is_correct() {
    use crate::neuron::sub_neuron_path;
    use std::path::Path;
    let core = Path::new(".cortyx/neurons/src/engine_rs.context.md");
    let sub = sub_neuron_path(core, "validate_user");
    let name = sub.file_name().unwrap().to_string_lossy();
    assert_eq!(name, "engine_rs.fn-validate_user.context.md");
    assert_eq!(sub.parent(), core.parent());
}

#[test]
fn sub_neuron_path_sanitizes_special_chars() {
    use crate::neuron::sub_neuron_path;
    use std::path::Path;
    let core = Path::new(".cortyx/neurons/src/engine_rs.context.md");
    let sub = sub_neuron_path(core, "fn with spaces!");
    let name = sub.file_name().unwrap().to_string_lossy();
    // spaces and ! should be replaced with _
    assert!(name.starts_with("engine_rs.fn-"));
    assert!(!name.contains(' '));
    assert!(!name.contains('!'));
}

#[test]
fn sub_neuron_content_contains_function_name() {
    use crate::neuron::stub_function_neuron;
    let content = stub_function_neuron("validate_user", "src/auth.rs", "2026-01-01T00:00:00Z");
    assert!(
        content.contains("validate_user"),
        "stub must mention the function name"
    );
    assert!(
        content.contains("SECTION: purpose"),
        "stub must have purpose section"
    );
    assert!(
        content.contains("SECTION: api"),
        "stub must have api section"
    );
}

#[test]
fn split_threshold_files_produce_sub_neurons() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let src_dir = root.join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    std::fs::create_dir_all(root.join(".cortyx").join("neurons").join("src")).unwrap();

    // Write a Rust file with 7 public functions (above SUBNEURON_SPLIT_THRESHOLD=6)
    let mut src = String::new();
    for i in 0..7 {
        src.push_str(&format!("pub fn function_{i}() {{ }}\n"));
    }
    std::fs::write(src_dir.join("big_module.rs"), &src).unwrap();

    let git_confidence = std::collections::HashMap::new();
    let abs = src_dir.join("big_module.rs");
    let results = process_source_file(&abs, root, &git_confidence);

    // First result is the Core; subsequent are UseCase sub-neurons
    assert!(
        results.len() >= 2,
        "should produce Core + sub-neurons for 7-function file"
    );
    let core = &results[0];
    assert_eq!(core.meta.kind, crate::neuron::NeuronKind::Core);
    let subs: Vec<_> = results.iter().skip(1).collect();
    assert!(!subs.is_empty(), "should have at least one sub-neuron");
    assert!(subs
        .iter()
        .all(|s| s.meta.kind == crate::neuron::NeuronKind::UseCase));
    assert!(subs
        .iter()
        .all(|s| s.meta.parent.as_deref() == Some(core.neuron_path.as_path())));
}

#[test]
fn small_files_produce_no_sub_neurons() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let src_dir = root.join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    std::fs::create_dir_all(root.join(".cortyx").join("neurons").join("src")).unwrap();

    // Write a small Rust file with 2 public functions (below threshold)
    let src = "pub fn a() {}\npub fn b() {}\n";
    std::fs::write(src_dir.join("small.rs"), src).unwrap();

    let git_confidence = std::collections::HashMap::new();
    let abs = src_dir.join("small.rs");
    let results = process_source_file(&abs, root, &git_confidence);

    assert_eq!(
        results.len(),
        1,
        "small file should produce only a Core neuron"
    );
    assert_eq!(results[0].meta.kind, crate::neuron::NeuronKind::Core);
}

// ── R11-S1: Section-Level Staleness ──────────────────────────────────────

/// Verifies that `update_neuron_header` patches only the three header comment
/// lines and leaves all other content (section bodies, cross-refs) intact.
#[test]
fn update_neuron_header_patches_only_header_lines() {
    use crate::neuron::update_neuron_header;
    let content = "\
<!-- AUTO-GENERATED CONTEXT — DO NOT EDIT MANUALLY -->\n\
<!-- source: src/engine.rs -->\n\
<!-- hash: aabbccdd11223344 -->\n\
<!-- last-updated: 2024-01-01T00:00:00Z -->\n\
<!-- status: stub -->\n\
\n\
<!-- SECTION: purpose -->\n\
This module drives the core loop.\n\
<!-- /SECTION -->\n\
<!-- SECTION: api -->\n\
pub fn run()\n\
<!-- /SECTION -->\n";

    let updated = update_neuron_header(content, "deadbeef12345678", "2025-06-01T12:00:00Z");

    assert!(
        updated.contains("<!-- hash: deadbeef12345678 -->"),
        "hash line must be updated"
    );
    assert!(
        updated.contains("<!-- last-updated: 2025-06-01T12:00:00Z -->"),
        "date must be updated"
    );
    assert!(
        updated.contains("<!-- status: stale -->"),
        "status must be set to stale"
    );
    assert!(!updated.contains("aabbccdd"), "old hash must not appear");
    assert!(
        updated.contains("This module drives the core loop."),
        "purpose body must be preserved"
    );
    assert!(
        updated.contains("pub fn run()"),
        "api body must be preserved"
    );
}

/// When a source file's sig_hash changes (real API change) but the neuron already
/// exists, `process_source_file` should update only the `api` section and preserve
/// the `purpose` section content written by a previous LLM call.
#[test]
fn s1_api_section_update_preserves_purpose_on_sig_hash_change() {
    use crate::neuron::replace_section;
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let src_dir = root.join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    std::fs::create_dir_all(root.join(".cortyx").join("neurons").join("src")).unwrap();

    // Write an initial source file and compile it to get a neuron stub
    let src_v1 = "pub fn alpha() {}\n";
    std::fs::write(src_dir.join("mod.rs"), src_v1).unwrap();
    let git_confidence = std::collections::HashMap::new();
    let abs = src_dir.join("mod.rs");
    let v1 = process_source_file(&abs, root, &git_confidence);
    assert_eq!(v1.len(), 1, "v1 should produce one Core");
    let neuron_path = v1[0].neuron_path.clone();

    // Simulate LLM evolution: write a purpose section into the neuron
    let with_purpose = replace_section(
        &v1[0].content,
        "purpose",
        "Alpha drives the main processing loop.",
    );
    std::fs::write(&neuron_path, &with_purpose).unwrap();

    // Now change the source file API (rename function → sig_hash changes)
    let src_v2 = "pub fn beta() {}\n";
    std::fs::write(&abs, src_v2).unwrap();
    let v2 = process_source_file(&abs, root, &git_confidence);

    // S1: should return a compiled file with api updated but purpose preserved
    assert_eq!(v2.len(), 1, "v2 should still produce one Core");
    let new_content = std::fs::read_to_string(&neuron_path).unwrap();
    assert!(
        new_content.contains("beta"),
        "new api section should contain updated function name"
    );
    assert!(
        new_content.contains("Alpha drives the main processing loop."),
        "LLM-curated purpose section must survive a sig_hash change"
    );
    assert!(
        new_content.contains("<!-- status: stale -->"),
        "status should be stale after api change"
    );
}

// ── R11-S4: Adaptive CI Quarantine ───────────────────────────────────────

/// Verifies that `adaptive_quarantine_params` returns the correct (z, threshold) tier
/// and None below the cold-start threshold.
#[test]
fn adaptive_quarantine_params_tier_boundaries() {
    assert!(adaptive_quarantine_params(0).is_none(), "0 samples → None");
    assert!(adaptive_quarantine_params(4).is_none(), "4 samples → None");
    let (z5, t5) = adaptive_quarantine_params(5).unwrap();
    assert!((z5 - 1.0).abs() < 0.01, "5 samples → z=1.0");
    assert!((t5 - 0.02).abs() < 0.001, "5 samples → threshold=0.02");
    let (z19, _) = adaptive_quarantine_params(19).unwrap();
    assert!((z19 - 1.0).abs() < 0.01, "19 samples → still z=1.0 tier");
    let (z20, t20) = adaptive_quarantine_params(20).unwrap();
    assert!((z20 - 1.645).abs() < 0.01, "20 samples → z=1.645");
    assert!((t20 - 0.05).abs() < 0.001, "20 samples → threshold=0.05");
    let (z100, t100) = adaptive_quarantine_params(100).unwrap();
    assert!((z100 - 1.96).abs() < 0.01, "100+ samples → z=1.96");
    assert!((t100 - 0.08).abs() < 0.001, "100+ samples → threshold=0.08");
}

/// Early quarantine at 5+ samples with 0% hit rate (z=1.0 tier).
#[test]
fn adaptive_ci_quarantines_early_for_zero_hit_rate() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);

    let p = ndir.join("noise.context.md");
    std::fs::write(&p, "noise boilerplate low quality").unwrap();
    let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    idx.index_neuron(&p, "noise boilerplate low quality", &meta);
    idx.rebuild_derived();

    // 9 activations, 0 hits → z=1.0 tier, lb(0,10)=0.0 < 0.02 → should quarantine
    if let Some(&i) = idx.path_index.get(&p) {
        idx.entries[i].use_count = 9;
        idx.entries[i].hit_count = 0;
    }
    idx.record_activation(&[p.clone()]); // → use_count=10
    let mult = idx
        .path_index
        .get(&p)
        .map(|&i| idx.entries[i].staleness_multiplier)
        .unwrap_or(1.0);
    assert_eq!(
        mult, 0.3,
        "10 activations with 0 hits should quarantine at z=1.0 tier"
    );
}

/// A neuron with moderate hit rate at medium count should NOT be quarantined
/// (90% CI is too wide to conclude bad quality).
#[test]
fn adaptive_ci_does_not_quarantine_moderate_hit_rate() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);

    let p = ndir.join("moderate.context.md");
    std::fs::write(&p, "good content useful context").unwrap();
    let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    idx.index_neuron(&p, "good content useful context", &meta);
    idx.rebuild_derived();

    // 5 hits out of 20 total → 25% hit rate; lb at z=1.645 is well above 0.05
    if let Some(&i) = idx.path_index.get(&p) {
        idx.entries[i].use_count = 19;
        idx.entries[i].hit_count = 5;
    }
    idx.record_activation(&[p.clone()]); // → use_count=20
    let mult = idx
        .path_index
        .get(&p)
        .map(|&i| idx.entries[i].staleness_multiplier)
        .unwrap_or(1.0);
    assert_eq!(
        mult, 1.0,
        "25% hit rate at 20 samples should not be quarantined"
    );
}

// ── R12-S1: Concept Cloud ─────────────────────────────────────────────────

#[test]
fn concept_cloud_populated_from_structural_neighbours() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);

    let caller = ndir.join("caller.context.md");
    let callee = ndir.join("callee.context.md");
    std::fs::write(&caller, "calls validate_user auth check").unwrap();
    std::fs::write(&callee, "validate_user password hash bcrypt").unwrap();

    let mut meta_caller = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    let meta_callee = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    meta_caller.synapses.push(crate::neuron::Synapse {
        target: callee.clone(),
        edge_type: crate::neuron::SynapseType::Calls,
        weight: crate::types::SynapseWeight::new(0.8),
        reason: "calls validate_user".to_string(),
        learned_weight: SynapseWeight::ZERO,
        traversal_count: 0,
        last_co_activation_day: 0,
    });

    idx.index_neuron(&caller, "calls validate_user auth check", &meta_caller);
    idx.index_neuron(&callee, "validate_user password hash bcrypt", &meta_callee);
    idx.rebuild_derived();

    // caller's concept cloud should contain callee terms
    let caller_idx = *idx.path_index.get(&caller).unwrap();
    let cloud = &idx.entries[caller_idx].concept_cloud;
    assert!(
        cloud
            .iter()
            .any(|t| t == "bcrypt" || t == "password" || t == "validate_user"),
        "caller concept cloud should contain callee terms; got: {cloud:?}"
    );
}

#[test]
fn concept_cloud_enables_retrieval_via_graph() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);

    // "engine.rs" calls "hashing.rs", which owns the word "bcrypt".
    // A query for "bcrypt" should find engine via concept cloud even though
    // "bcrypt" does not appear in engine's own vocabulary.
    let engine = ndir.join("engine.context.md");
    let hashing = ndir.join("hashing.context.md");
    std::fs::write(&engine, "core engine dispatch orchestrate").unwrap();
    std::fs::write(&hashing, "bcrypt password hash rounds salt").unwrap();

    let mut meta_engine = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    let meta_hashing = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    meta_engine.synapses.push(crate::neuron::Synapse {
        target: hashing.clone(),
        edge_type: crate::neuron::SynapseType::Calls,
        weight: crate::types::SynapseWeight::new(0.8),
        reason: "calls hash function".to_string(),
        learned_weight: SynapseWeight::ZERO,
        traversal_count: 0,
        last_co_activation_day: 0,
    });

    idx.index_neuron(&engine, "core engine dispatch orchestrate", &meta_engine);
    idx.index_neuron(&hashing, "bcrypt password hash rounds salt", &meta_hashing);
    idx.rebuild_derived();

    // "bcrypt" is in hashing's vocab → engine's concept cloud → engine is reachable
    let engine_idx = *idx.path_index.get(&engine).unwrap();
    assert!(
        idx.entries[engine_idx]
            .concept_cloud
            .contains(&"bcrypt".to_string()),
        "engine concept cloud must contain 'bcrypt' from hashing neighbour"
    );

    // Now query for "bcrypt" — vocab bridge won't match (no module synonym),
    // but concept cloud should surface engine as a candidate.
    let results = idx.get_contexts("bcrypt", 4096, None, None);
    let found_engine = results
        .iter()
        .any(|s| s.to_string_lossy().contains("engine"));
    let found_hashing = results.iter().any(|s| {
        let p = s.to_string_lossy();
        p.contains("hashing") || p.contains("bcrypt")
    });
    assert!(
        found_hashing || found_engine,
        "concept cloud retrieval must surface at least one relevant neuron; got {results:?}"
    );
}

#[test]
fn concept_cloud_excludes_semantic_related_edges() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);

    let a = ndir.join("a.context.md");
    let b = ndir.join("b.context.md");
    std::fs::write(&a, "alpha beta gamma").unwrap();
    std::fs::write(&b, "exclusive_term_xyz zeta").unwrap();

    let mut meta_a = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    let meta_b = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    // Only SemanticRelated edge — should NOT contribute to concept cloud
    meta_a.synapses.push(crate::neuron::Synapse {
        target: b.clone(),
        edge_type: crate::neuron::SynapseType::SemanticRelated,
        weight: crate::types::SynapseWeight::new(0.5),
        reason: "related".to_string(),
        learned_weight: SynapseWeight::ZERO,
        traversal_count: 0,
        last_co_activation_day: 0,
    });

    idx.index_neuron(&a, "alpha beta gamma", &meta_a);
    idx.index_neuron(&b, "exclusive_term_xyz zeta", &meta_b);
    idx.rebuild_derived();

    let a_idx = *idx.path_index.get(&a).unwrap();
    assert!(
        !idx.entries[a_idx]
            .concept_cloud
            .contains(&"exclusive_term_xyz".to_string()),
        "SemanticRelated edges must not populate concept cloud (already handled by vocab bridge)"
    );
}

// ── S-II (R16): LSH SimHash ───────────────────────────────────────────────

#[test]
fn simhash_same_terms_identical_fingerprint() {
    // Identical content should always yield the same fingerprint (deterministic)
    let mut tf1: std::collections::HashMap<String, TermFrequency> =
        std::collections::HashMap::new();
    tf1.insert("auth".to_string(), TermFrequency::new(1.0));
    tf1.insert("token".to_string(), TermFrequency::new(2.0));
    let fp1 = simhash_with_seed(&tf1, LSH_SEEDS[0]);
    let fp2 = simhash_with_seed(&tf1, LSH_SEEDS[0]);
    assert_eq!(fp1, fp2, "same terms → same fingerprint (deterministic)");
    // Highly divergent content should produce different fingerprints with overwhelming probability
    let mut tf_other: std::collections::HashMap<String, TermFrequency> =
        std::collections::HashMap::new();
    tf_other.insert("xyzzy".to_string(), TermFrequency::new(100.0));
    tf_other.insert("quux".to_string(), TermFrequency::new(100.0));
    tf_other.insert("plonk".to_string(), TermFrequency::new(100.0));
    tf_other.insert("zork".to_string(), TermFrequency::new(100.0));
    let fp_other = simhash_with_seed(&tf_other, LSH_SEEDS[0]);
    assert_ne!(
        fp1, fp_other,
        "very different terms should produce different fingerprints"
    );
}

#[test]
fn simhash_identical_content_identical_fingerprint() {
    let mut tf: std::collections::HashMap<String, TermFrequency> = std::collections::HashMap::new();
    tf.insert("validate".to_string(), TermFrequency::new(1.5));
    tf.insert("password".to_string(), TermFrequency::new(3.0));
    let fp1 = simhash_with_seed(&tf, LSH_SEEDS[0]);
    let fp2 = simhash_with_seed(&tf, LSH_SEEDS[0]);
    assert_eq!(fp1, fp2, "same terms → same fingerprint (deterministic)");
}

#[test]
fn hamming_distance_self_is_zero() {
    let fp = 0xdeadbeefcafe1234u64;
    assert_eq!(hamming_distance(fp, fp), 0);
}

#[test]
fn hamming_distance_complement_is_64() {
    assert_eq!(hamming_distance(0u64, !0u64), 64);
}

#[test]
fn lsh_fingerprint_stored_in_entry() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);
    let neuron = ndir.join("auth.context.md");
    std::fs::write(&neuron, "auth token validate jwt bearer").unwrap();
    let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    idx.index_neuron(&neuron, "auth token validate jwt bearer", &meta);
    let entry_idx = *idx.path_index.get(&neuron).unwrap();
    assert!(
        idx.entries[entry_idx]
            .lsh_fingerprints
            .iter()
            .any(|&fp| fp != 0),
        "non-empty term set should produce non-zero 256-bit SimHash"
    );
}

// ── S-III (R16): Self-Quality Score ──────────────────────────────────────

#[test]
fn quality_score_defaults_to_one_when_no_source() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);
    let neuron = ndir.join("concept.context.md");
    std::fs::write(&neuron, "some concept terms here").unwrap();
    // Concept kind → no source file → quality_score defaults to 1.0
    let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Concept);
    idx.index_neuron(&neuron, "some concept terms here", &meta);
    let entry_idx = *idx.path_index.get(&neuron).unwrap();
    assert!(
        (idx.entries[entry_idx].quality_score - 1.0).abs() < 1e-6,
        "Concept neuron should have quality_score=1.0 (no source file)"
    );
}

#[test]
fn low_quality_count_counts_below_threshold() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);
    // All Concept neurons → quality_score = 1.0 → none below threshold
    for i in 0..3 {
        let p = ndir.join(format!("n{i}.context.md"));
        std::fs::write(&p, "terms").unwrap();
        let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Concept);
        idx.index_neuron(&p, "terms", &meta);
    }
    assert_eq!(
        idx.low_quality_count(),
        0,
        "no low-quality neurons expected"
    );
}

#[test]
fn publish_ready_candidates_filter_for_shareable_quality() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);

    let strong = ndir.join("strong.context.md");
    std::fs::write(&strong, "auth token validation middleware").unwrap();
    let strong_meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Concept);
    idx.index_neuron(&strong, "auth token validation middleware", &strong_meta);
    let strong_idx = *idx.path_index.get(&strong).unwrap();
    idx.entries[strong_idx].use_count = 12;
    idx.entries[strong_idx].hit_count = 9;

    let weak_hit = ndir.join("weak-hit.context.md");
    std::fs::write(&weak_hit, "routing fallback legacy handler").unwrap();
    let weak_hit_meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Concept);
    idx.index_neuron(&weak_hit, "routing fallback legacy handler", &weak_hit_meta);
    let weak_hit_idx = *idx.path_index.get(&weak_hit).unwrap();
    idx.entries[weak_hit_idx].use_count = 12;
    idx.entries[weak_hit_idx].hit_count = 2;

    let verbatim = ndir.join("verbatim.context.md");
    std::fs::write(&verbatim, "I fixed the auth bug today").unwrap();
    let verbatim_meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Verbatim);
    idx.index_neuron(&verbatim, "I fixed the auth bug today", &verbatim_meta);
    let verbatim_idx = *idx.path_index.get(&verbatim).unwrap();
    idx.entries[verbatim_idx].use_count = 25;
    idx.entries[verbatim_idx].hit_count = 25;

    let ready = idx.publish_ready_candidates(10, 0.5, 0.6, 10);

    assert_eq!(ready.len(), 1);
    assert_eq!(ready[0].path, strong);
    assert_eq!(ready[0].kind, NeuronKind::Concept);
    assert_eq!(ready[0].use_count, 12);
    assert!(ready[0].hit_rate >= 0.75);
    assert!(ready[0].quality_score >= 1.0);
}
