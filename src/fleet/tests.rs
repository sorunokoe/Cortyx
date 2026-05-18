use super::*;

use std::ffi::OsString;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

static HOME_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

struct TestHomeGuard {
    old_home: Option<OsString>,
    path: PathBuf,
}

impl TestHomeGuard {
    fn new(name: &str) -> Self {
        let path = std::env::current_dir()
            .unwrap()
            .join("target")
            .join("fleet-tests")
            .join(format!(
                "{}-{}-{}",
                name,
                std::process::id(),
                TEST_COUNTER.fetch_add(1, Ordering::Relaxed)
            ));
        if path.exists() {
            fs::remove_dir_all(&path).unwrap();
        }
        fs::create_dir_all(&path).unwrap();

        let old_home = std::env::var_os("HOME");
        std::env::set_var("HOME", &path);
        Self { old_home, path }
    }
}

impl Drop for TestHomeGuard {
    fn drop(&mut self) {
        if let Some(old_home) = &self.old_home {
            std::env::set_var("HOME", old_home);
        } else {
            std::env::remove_var("HOME");
        }
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn with_test_home<T>(name: &str, test: impl FnOnce(PathBuf) -> T) -> T {
    let _guard = HOME_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap();
    let guard = TestHomeGuard::new(name);
    test(guard.path.clone())
}

#[test]
fn fleet_registry_path_returns_expected_dir() {
    let path = fleet_registry_path().unwrap();
    assert!(path.ends_with(PathBuf::from(".cortyx").join("fleet").join("nodes.json")));
}

#[test]
fn load_empty_registry_when_absent() {
    with_test_home("load-empty", |_| {
        let registry = load_registry().unwrap();
        assert_eq!(registry, FleetRegistry::default());
    });
}

#[test]
fn save_and_load_registry_roundtrip() {
    with_test_home("roundtrip", |_| {
        let registry = FleetRegistry {
            version: FLEET_REGISTRY_VERSION,
            nodes: vec![FleetNode {
                id: FleetNodeId::new("node-1234"),
                path: PathBuf::from("/fleet/ignored"),
                alias: "alpha".to_string(),
                modules: vec!["core".to_string()],
                last_registered: "2026-01-01T00:00:00Z".to_string(),
                git_url: None,
                last_fetched: None,
            }],
        };

        save_registry(&registry).unwrap();
        let loaded = load_registry().unwrap();
        assert_eq!(loaded, registry);
    });
}

#[test]
fn register_and_deregister_node() {
    with_test_home("register-node", |home| {
        let project = home.join("workspace").join("alpha-project");
        fs::create_dir_all(&project).unwrap();

        let node = register_node(&project, Some("alpha".to_string())).unwrap();
        let registry = load_registry().unwrap();
        assert!(registry
            .nodes
            .iter()
            .any(|entry| entry.id == node.id && entry.alias == "alpha"));

        let removed = deregister_node("alpha").unwrap();
        assert!(removed);
        let registry = load_registry().unwrap();
        assert!(registry.nodes.is_empty());
    });
}

#[test]
fn rrf_merge_local_only_when_no_fleet() {
    let merged = rrf_merge("local context", 5.0, Vec::new(), 0.7, 0.3);
    assert_eq!(merged, "local context");
}

#[test]
fn rrf_merge_appends_fleet_context() {
    let merged = rrf_merge(
        "local context",
        5.0,
        vec![FleetQueryResult {
            node_id: FleetNodeId::new("node-1"),
            node_alias: "alpha".to_string(),
            contexts: "fleet context".to_string(),
            top_score: 2.5,
        }],
        0.7,
        0.3,
    );

    let local_pos = merged.find("local context").unwrap();
    let fleet_pos = merged.find("fleet context").unwrap();
    assert!(merged.contains("Fleet context from 1 additional nodes"));
    assert!(merged.contains("From fleet node: alpha (score: 2.50)"));
    assert!(local_pos < fleet_pos);
}

#[test]
fn fleet_low_confidence_threshold_is_correct_value() {
    assert_eq!(FLEET_LOW_CONFIDENCE_THRESHOLD, 4.0);
}

#[test]
fn fleet_node_id_display() {
    assert_eq!(FleetNodeId::new("abc").to_string(), "abc");
}

// ── C7: Dynamic fleet weight tests ────────────────────────────────────────────

#[test]
fn dynamic_fleet_weight_zero_score_gives_minimum() {
    assert!((dynamic_fleet_weight(0.0) - 0.10).abs() < 0.001);
}

#[test]
fn dynamic_fleet_weight_at_midpoint_gives_baseline() {
    // score = 4.0 (LOW_CONFIDENCE midpoint) → weight ≈ 0.30 (the prior)
    let w = dynamic_fleet_weight(4.0);
    assert!(
        (w - 0.30).abs() < 0.01,
        "expected ~0.30 at midpoint, got {w}"
    );
}

#[test]
fn dynamic_fleet_weight_high_score_amplified() {
    // score = 8.0 → weight approaching 0.50
    let w = dynamic_fleet_weight(8.0);
    assert!(
        w >= 0.40,
        "high-quality fleet result should have weight ≥ 0.40, got {w}"
    );
    assert!(w <= 0.50);
}

#[test]
fn dynamic_fleet_weight_clamped_in_range() {
    for score in [0.0, 1.0, 4.0, 8.0, 12.0, 100.0] {
        let w = dynamic_fleet_weight(score);
        assert!(
            (0.10..=0.50).contains(&w),
            "weight {w} out of [0.10, 0.50] for score {score}"
        );
    }
}

#[test]
fn dynamic_fleet_weight_monotone_increasing() {
    let scores = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0];
    let weights: Vec<f32> = scores.iter().map(|&s| dynamic_fleet_weight(s)).collect();
    for i in 1..weights.len() {
        assert!(
            weights[i] >= weights[i - 1],
            "weight should be non-decreasing: w({})={} < w({})={}",
            scores[i],
            weights[i],
            scores[i - 1],
            weights[i - 1]
        );
    }
}
