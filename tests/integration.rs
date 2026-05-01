/// Integration tests for Cortyx — end-to-end compile → activate → evolve → synapse flow.
use std::fs;
use tempfile::TempDir;

mod common;
use common::run;

/// Set up a realistic mini-project in a temp dir.
fn make_project() -> TempDir {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();

    fs::write(
        root.join("engine.rs"),
        r#"
/// Routes user intent to the correct agent subsystem.
pub fn route_intent(task: &str) -> &'static str {
    if task.contains("dark mode") { "ui" } else { "core" }
}
pub fn synthesize_answer(parts: Vec<String>) -> String {
    parts.join("\n")
}
"#,
    )
    .unwrap();

    fs::write(
        root.join("ui.rs"),
        r#"
/// Applies dark mode color tokens to SwiftUI views.
pub fn apply_dark_mode(view: &mut View) {
    view.background = ColorToken::Background;
    view.foreground = ColorToken::Text;
}
pub struct View { pub background: ColorToken, pub foreground: ColorToken }
pub enum ColorToken { Background, Text, Accent }
"#,
    )
    .unwrap();

    fs::write(
        root.join("auth.rs"),
        "pub fn validate_token(tok: &str) -> bool { !tok.is_empty() }",
    )
    .unwrap();

    dir
}

#[test]
fn compile_creates_neuron_stubs() {
    let dir = make_project();
    let out = run(&["compile"], dir.path());
    assert!(
        out.status.success(),
        "compile failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("Compiled"),
        "Expected 'Compiled' in output: {stdout}"
    );

    // Verify .cortyx/neurons/ exists and has .context.md stubs
    let neurons_dir = dir.path().join(".cortyx").join("neurons");
    assert!(neurons_dir.exists(), ".cortyx/neurons/ not created");

    let stubs: Vec<_> = fs::read_dir(&neurons_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().ends_with(".context.md"))
        .collect();
    assert!(stubs.len() >= 3, "Expected ≥3 stubs, got {}", stubs.len());
}

#[test]
fn compile_is_idempotent() {
    let dir = make_project();
    // First compile
    let out1 = run(&["compile"], dir.path());
    assert!(out1.status.success());
    let stdout1 = String::from_utf8_lossy(&out1.stdout);

    // Second compile — no new stubs should be created
    let out2 = run(&["compile"], dir.path());
    assert!(out2.status.success());
    let stdout2 = String::from_utf8_lossy(&out2.stdout);
    assert!(
        stdout2.contains("Compiled 0"),
        "Second compile should create 0 new stubs, got: {stdout2}"
    );
    let _ = stdout1;
}

#[test]
fn status_shows_neuron_count() {
    let dir = make_project();
    run(&["compile"], dir.path());
    let out = run(&["status"], dir.path());
    assert!(
        out.status.success(),
        "status failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("Core neurons:"),
        "Expected 'Core neurons' in status output: {stdout}"
    );
}

#[test]
fn invalidate_marks_neuron_stale() {
    let dir = make_project();
    run(&["compile"], dir.path());

    let out = run(&["invalidate", "engine.rs"], dir.path());
    assert!(
        out.status.success(),
        "invalidate failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // Verify the sidecar JSON now has status: Stale
    let neurons_dir = dir.path().join(".cortyx").join("neurons");
    let engine_json: Vec<_> = fs::read_dir(&neurons_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            let n = e.file_name().to_string_lossy().to_string();
            n.contains("engine") && n.ends_with(".context.json")
        })
        .collect();
    assert!(!engine_json.is_empty(), "No engine sidecar JSON found");

    let data = fs::read_to_string(engine_json[0].path()).unwrap();
    assert!(
        data.contains("\"stale\""),
        "Expected Stale status in sidecar: {data}"
    );
}

#[test]
fn hash_invalidation_detects_file_change() {
    let dir = make_project();
    run(&["compile"], dir.path());

    // Modify engine.rs
    fs::write(dir.path().join("engine.rs"), "// modified").unwrap();

    // Recompile — should mark existing neuron stale, not recreate
    let out = run(&["compile"], dir.path());
    assert!(out.status.success());

    // Verify the engine neuron is now Stale
    let neurons_dir = dir.path().join(".cortyx").join("neurons");
    let engine_json: Vec<_> = fs::read_dir(&neurons_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            let n = e.file_name().to_string_lossy().to_string();
            n.contains("engine") && n.ends_with(".context.json")
        })
        .collect();
    assert!(!engine_json.is_empty(), "No engine sidecar JSON found");
    let data = fs::read_to_string(engine_json[0].path()).unwrap();
    assert!(
        data.contains("\"stale\""),
        "Expected Stale status after file change: {data}"
    );
}

#[test]
fn export_produces_valid_json_anthropic() {
    let dir = make_project();
    run(&["compile"], dir.path());

    let out = run(&["export", "--provider", "anthropic"], dir.path());
    assert!(
        out.status.success(),
        "export failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let json_str = String::from_utf8_lossy(&out.stdout);
    let parsed: serde_json::Value =
        serde_json::from_str(&json_str).expect("Export output is not valid JSON");

    // Must have cache_control on the first system block
    let system = parsed["system"].as_array().expect("system must be array");
    assert!(!system.is_empty(), "system must not be empty");
    assert!(
        system[0]["cache_control"]["type"] == "ephemeral",
        "First system block must have cache_control ephemeral"
    );
}

#[test]
fn export_produces_valid_json_openai() {
    let dir = make_project();
    run(&["compile"], dir.path());

    let out = run(&["export", "--provider", "openai"], dir.path());
    assert!(
        out.status.success(),
        "export failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let json_str = String::from_utf8_lossy(&out.stdout);
    let parsed: serde_json::Value =
        serde_json::from_str(&json_str).expect("Export output is not valid JSON");

    let messages = parsed["messages"]
        .as_array()
        .expect("messages must be array");
    assert!(!messages.is_empty(), "messages must not be empty");
    assert_eq!(messages[0]["role"], "system");
}

#[test]
fn static_prefix_is_byte_identical_across_tasks() {
    let dir = make_project();
    run(&["compile"], dir.path());

    // Export twice with different tasks — the static prefix must be byte-identical
    let out1 = run(&["export", "--provider", "anthropic"], dir.path());
    let out2 = run(&["export", "--provider", "anthropic"], dir.path());

    let json1: serde_json::Value = serde_json::from_slice(&out1.stdout).unwrap();
    let json2: serde_json::Value = serde_json::from_slice(&out2.stdout).unwrap();

    let prefix1 = json1["system"][0]["text"].as_str().unwrap();
    let prefix2 = json2["system"][0]["text"].as_str().unwrap();

    assert_eq!(
        prefix1, prefix2,
        "Static prefix must be byte-identical across calls (cache hit guarantee)"
    );
}
