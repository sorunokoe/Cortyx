use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::Value;
use tempfile::TempDir;

mod common;
use common::{cortyx_bin, run, run_with_home};

fn make_project() -> TempDir {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();

    fs::write(
        root.join("engine.rs"),
        r#"
pub fn route_intent(task: &str) -> &'static str {
    if task.contains("dark mode") { "ui" } else { "core" }
}
"#,
    )
    .unwrap();

    fs::write(
        root.join("ui.rs"),
        r#"
pub fn apply_dark_mode() -> &'static str {
    "dark mode"
}
"#,
    )
    .unwrap();

    dir
}

fn parse_ux_proof(output: &str) -> Value {
    let raw = output
        .lines()
        .find_map(|line| line.split_once("ux-proof: ").map(|(_, raw)| raw))
        .expect("output should include ux-proof JSON");
    serde_json::from_str(raw).expect("ux-proof should be valid JSON")
}

fn wait_until(timeout: Duration, mut predicate: impl FnMut() -> bool) {
    let started = Instant::now();
    while started.elapsed() < timeout {
        if predicate() {
            return;
        }
        thread::sleep(Duration::from_millis(50));
    }
    assert!(predicate(), "condition was not met within {:?}", timeout);
}

fn inject_answer_surface(root: &Path) {
    let neuron = root.join(".cortyx/neurons/chat_0001_assistant.verbatim.md");
    let mut content = fs::read_to_string(&neuron).expect("assistant verbatim neuron");
    if !content.contains("## answer_surface") {
        content.push_str(
            "\n\n## answer_surface\n\
             <!-- SECTION: answer_surface -->\n\
             | question_pattern | answer_span | confidence |\n\
             | --- | --- | --- |\n\
             | job occupation profession work career role | pediatric nurse | 0.92 |\n\
             <!-- /SECTION -->\n",
        );
        fs::write(neuron, content).unwrap();
    }
}

#[test]
fn route_banner_emits_machine_readable_ttfc_and_context_outcome() {
    let dir = make_project();
    let out = run(&["route", "--task", "trace the route flow"], dir.path());
    assert!(
        out.status.success(),
        "route failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let stderr = String::from_utf8_lossy(&out.stderr);
    let proof = parse_ux_proof(&stderr);
    assert_eq!(proof["mode"].as_str(), Some("context"));
    assert_eq!(proof["ttfc"]["triggered"].as_bool(), Some(true));
    assert!(proof["ttfc"]["compiled_neurons"].as_u64().unwrap_or(0) >= 2);
    assert_eq!(proof["recovery"]["watch"].as_str(), Some("cortyx watch"));
    assert_eq!(
        proof["entrypoint"]["terminal_route"].as_str(),
        Some("cortyx route --task \"trace the auth flow\"")
    );

    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("CORTYX CONTEXT"));
    assert!(stdout.contains("engine_rs.context.md"));
}

#[test]
fn route_auto_proves_capabilities_and_answer_outcomes_from_one_entrypoint() {
    let dir = make_project();

    let capabilities = run(&["route"], dir.path());
    assert!(
        capabilities.status.success(),
        "route summary failed: {}",
        String::from_utf8_lossy(&capabilities.stderr)
    );
    let capabilities_stderr = String::from_utf8_lossy(&capabilities.stderr);
    let capabilities_proof = parse_ux_proof(&capabilities_stderr);
    assert_eq!(capabilities_proof["mode"].as_str(), Some("capabilities"));
    assert!(String::from_utf8_lossy(&capabilities.stdout).contains("Cortyx capability summary"));

    fs::write(
        dir.path().join("chat.md"),
        "## Human\nWhat is my job?\n\n## Assistant\npediatric nurse\n",
    )
    .unwrap();
    let mined = run(&["mine", "chat.md"], dir.path());
    assert!(
        mined.status.success(),
        "mine failed: {}",
        String::from_utf8_lossy(&mined.stderr)
    );
    inject_answer_surface(dir.path());

    let answer = run(&["route", "--task", "What is my job?"], dir.path());
    assert!(
        answer.status.success(),
        "answer route failed: {}",
        String::from_utf8_lossy(&answer.stderr)
    );

    let answer_stderr = String::from_utf8_lossy(&answer.stderr);
    let answer_proof = parse_ux_proof(&answer_stderr);
    assert_eq!(answer_proof["mode"].as_str(), Some("answer"));
    assert_eq!(
        String::from_utf8_lossy(&answer.stdout).trim(),
        "pediatric nurse"
    );
}

#[test]
fn install_summary_emits_machine_readable_onboarding_proof() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    fs::create_dir_all(&home).unwrap();

    let out = run_with_home(&["install", "--global"], dir.path(), &home);
    assert!(
        out.status.success(),
        "install failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let stdout = String::from_utf8_lossy(&out.stdout);
    let proof = parse_ux_proof(&stdout);
    assert_eq!(proof["registered"].as_u64(), Some(6));
    assert_eq!(proof["created"].as_u64(), Some(6));
    assert_eq!(proof["counts"]["terminal_steps"].as_u64(), Some(3));
    assert_eq!(proof["counts"]["in_tool_steps"].as_u64(), Some(2));
    assert_eq!(
        proof["terminal_quickstart"][1].as_str(),
        Some("cortyx watch")
    );
    assert_eq!(proof["in_tool_quickstart"][0].as_str(), Some("cortyx()"));
}

#[test]
fn export_includes_machine_readable_ux_proof_metadata() {
    let dir = make_project();
    let out = run(&["compile"], dir.path());
    assert!(
        out.status.success(),
        "compile failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let out = run(&["export", "--provider", "anthropic"], dir.path());
    assert!(
        out.status.success(),
        "export failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let json: Value = serde_json::from_slice(&out.stdout).unwrap();
    let proof = &json["_cortyx_meta"]["ux_proof"];
    let outcomes = proof["one_entrypoint"]["outcomes"]
        .as_array()
        .expect("route outcomes should be an array");

    assert_eq!(proof["onboarding"]["terminal_steps"].as_u64(), Some(3));
    assert_eq!(proof["onboarding"]["in_tool_steps"].as_u64(), Some(2));
    assert_eq!(
        proof["recovery"]["incremental_compile"].as_str(),
        Some("cortyx compile --incremental")
    );
    assert!(outcomes
        .iter()
        .any(|value| value.as_str() == Some("context")));
    assert!(outcomes
        .iter()
        .any(|value| value.as_str() == Some("answer")));
}

#[test]
fn watch_startup_emits_bootstrap_proof_and_recovery_paths() {
    let dir = make_project();
    let engine_neuron = dir.path().join(".cortyx/neurons/engine_rs.context.md");

    let mut child = Command::new(cortyx_bin())
        .args(["watch"])
        .env("CORTYX_NO_DOWNLOAD", "1")
        .current_dir(dir.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("watch should start");

    wait_until(Duration::from_secs(10), || engine_neuron.exists());
    thread::sleep(Duration::from_millis(300));

    child.kill().ok();
    let output = child.wait_with_output().expect("watch output");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let proof = parse_ux_proof(&stdout);
    assert_eq!(proof["bootstrap"]["triggered"].as_bool(), Some(true));
    assert!(proof["bootstrap"]["compiled_neurons"].as_u64().unwrap_or(0) >= 2);
    assert_eq!(proof["recovery"]["doctor"].as_str(), Some("cortyx doctor"));
    assert_eq!(
        proof["recovery"]["incremental_compile"].as_str(),
        Some("cortyx compile --incremental")
    );
}

#[test]
fn incremental_recovery_keeps_route_results_current() {
    let dir = make_project();
    let engine_source = dir.path().join("engine.rs");
    let dirty_file = dir.path().join(".cortyx/dirty.json");

    let compile = run(&["compile"], dir.path());
    assert!(
        compile.status.success(),
        "compile failed: {}",
        String::from_utf8_lossy(&compile.stderr)
    );

    fs::write(
        &engine_source,
        r#"
pub fn route_intent(task: &str) -> &'static str {
    if task.contains("dark mode") { "ui" } else { "core" }
}

pub fn fallback_theme() -> &'static str {
    "fallback"
}
"#,
    )
    .unwrap();
    fs::write(
        &dirty_file,
        serde_json::to_string(&vec![engine_source.clone()]).unwrap(),
    )
    .unwrap();

    let incremental = run(&["compile", "--incremental"], dir.path());
    assert!(
        incremental.status.success(),
        "incremental compile failed: {}",
        String::from_utf8_lossy(&incremental.stderr)
    );
    let incremental_stdout = String::from_utf8_lossy(&incremental.stdout);
    assert!(incremental_stdout.contains("Incremental compile"));
    assert!(incremental_stdout.contains("neurons updated"));
    assert!(
        !dirty_file.exists(),
        "dirty queue should be cleared after recovery"
    );

    let doctor = run(&["doctor", "--json"], dir.path());
    assert!(
        doctor.status.success(),
        "doctor failed: {}",
        String::from_utf8_lossy(&doctor.stderr)
    );
    let doctor_json: Value = serde_json::from_slice(&doctor.stdout).unwrap();
    assert_eq!(doctor_json["index_valid"].as_bool(), Some(true));
    assert_eq!(doctor_json["errors"].as_u64(), Some(0));

    let route = run(
        &[
            "route",
            "--task",
            "fallback_theme route_intent engine.rs fallback_theme route_intent",
        ],
        dir.path(),
    );
    assert!(
        route.status.success(),
        "route after watch failed: {}",
        String::from_utf8_lossy(&route.stderr)
    );
    let route_stderr = String::from_utf8_lossy(&route.stderr);
    let route_proof = parse_ux_proof(&route_stderr);
    assert_eq!(route_proof["mode"].as_str(), Some("context"));
    assert!(String::from_utf8_lossy(&route.stdout).contains("CORTYX CONTEXT"));
}
