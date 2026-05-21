//! Smoke test for the `cortyx proof-certificate` CLI surface.

use serde_json::json;
use std::fs;
use std::path::Path;
use tempfile::TempDir;

mod common;
use common::run;

#[test]
fn proof_certificate_cli_smoke() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let out = run(&["proof-certificate"], repo_root);
    assert!(
        out.status.success(),
        "proof-certificate failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains(&format!(
        "Cortyx v{} Proof Certificate",
        env!("CARGO_PKG_VERSION")
    )));
    for label in [
        "Retrieval (R@5):",
        "Latency (p95):",
        "Hybrid latency:",
        "Token savings:",
        "Binary size:",
        "Verification:",
        "Frozen mode:",
    ] {
        assert!(
            stdout.contains(label),
            "missing {label} in output:\n{stdout}"
        );
    }
}

fn write_registry(dir: &TempDir, hybrid_latency: &str) {
    let registry = json!({
        "benchmarks": [
            { "id": "lme-500-official", "current_result": "484/500 = 96.8%" },
            { "id": "activation-latency-p95", "current_result": "~22ms p95" },
            { "id": "scale-2k-activation", "current_result": hybrid_latency },
            { "id": "bm25-token-savings-estimate", "current_result": "≥70% savings on 100-file project" },
            { "id": "binary-size-release", "current_result": "~30MB release binary" }
        ]
    });
    let benchmarks_dir = dir.path().join("benchmarks");
    fs::create_dir_all(&benchmarks_dir).unwrap();
    fs::write(
        benchmarks_dir.join("registry.json"),
        serde_json::to_vec_pretty(&registry).unwrap(),
    )
    .unwrap();
}

#[test]
fn proof_certificate_validate_passes_with_measured_registry() {
    let dir = TempDir::new().unwrap();
    write_registry(&dir, "~81ms p95");

    let out = run(&["proof-certificate", "--validate"], dir.path());
    assert!(
        out.status.success(),
        "proof-certificate --validate unexpectedly failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn proof_certificate_validate_fails_on_fallback_metrics() {
    let dir = TempDir::new().unwrap();
    write_registry(&dir, "pending — run bench_scale_2k_activation");

    let out = run(&["proof-certificate", "--validate"], dir.path());
    assert_eq!(out.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&out.stderr)
            .contains("Hybrid latency is still using fallback 80ms"),
        "expected fallback warning, got stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}
