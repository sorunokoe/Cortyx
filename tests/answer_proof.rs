use serde_json::Value;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn report_dir() -> PathBuf {
    let dir = repo_root().join("target").join("answer-proof-tests");
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn blocking_reasons(report: &Value) -> Vec<String> {
    report["proof"]["blocking_reasons"]
        .as_array()
        .expect("blocking_reasons should be an array")
        .iter()
        .map(|value| {
            value
                .as_str()
                .expect("blocking reason should be a string")
                .to_string()
        })
        .collect()
}

fn run_eval(script: &str, args: &[&str], output_name: &str) -> Value {
    let root = repo_root();
    let output_path = report_dir().join(output_name);
    if output_path.exists() {
        fs::remove_file(&output_path).unwrap();
    }

    let status = Command::new("python3")
        .current_dir(&root)
        .env("CORTYX_BIN", env!("CARGO_BIN_EXE_cortyx"))
        .arg(script)
        .arg("--output")
        .arg(output_path.to_str().expect("output path must be utf-8"))
        .args(args)
        .status()
        .expect("evaluation script should run");
    assert!(status.success(), "{script} should exit successfully");

    serde_json::from_str(
        &fs::read_to_string(&output_path).expect("evaluation report should be written"),
    )
    .expect("evaluation report should be valid json")
}

#[test]
#[ignore = "slow answer proof lane; run `scripts/test-full-proof.sh`"]
fn eval_lme_emits_structured_answer_proof_report() {
    let report = run_eval(
        "scripts/eval_lme.py",
        &[
            "--fixture",
            "tests/fixtures/longmemeval_500.json",
            "--answer-mode",
            "--max-per-category",
            "1",
            "--timeout-secs",
            "120",
        ],
        "lme_answer_report.json",
    );

    assert_eq!(
        report["benchmark"],
        Value::String("longmemeval-500".to_string())
    );
    assert_eq!(report["mode"], Value::String("answer".to_string()));
    assert_eq!(report["fixture"]["entries_total"], Value::from(500));
    assert_eq!(report["fixture"]["entries_evaluated"], Value::from(7));
    assert_eq!(
        report["fixture"]["official_public_release"],
        Value::Bool(true)
    );
    assert!(report["overall"]["macro_f1"].is_number());
    assert!(report["results"]
        .as_object()
        .is_some_and(|results| !results.is_empty()));
    assert_eq!(
        report["selection"]["categories"][0],
        Value::String("absent".to_string())
    );
    assert_eq!(
        report["selection"]["category_counts_total"]["absent"],
        Value::from(30)
    );
    assert!(report["diagnostics"]["infra_failures"]
        .as_array()
        .expect("infra_failures should be an array")
        .is_empty());
    assert_eq!(report["proof"]["public_surface_ready"], Value::Bool(true));
    assert_eq!(report["proof"]["comparator_ready"], Value::Bool(false));

    assert_eq!(
        report["public_surface"]["surface"],
        Value::String("official_qa_accuracy".to_string())
    );
    assert_eq!(report["public_surface"]["same_surface"], Value::Bool(true));
    assert_eq!(report["public_surface"]["entries_exported"], Value::from(7));
    let hypotheses_path = report["public_surface"]["hypotheses_path"]
        .as_str()
        .expect("public hypotheses path should be a string");
    assert!(
        repo_root().join(hypotheses_path).exists(),
        "public hypotheses artifact should exist"
    );

    let blockers = blocking_reasons(&report);
    assert!(blockers
        .iter()
        .any(|reason| reason.contains("partial run evaluated 7/500")));
    assert!(!blockers
        .iter()
        .any(|reason| reason.contains("retrieval-mode R@5")));
    assert!(!blockers
        .iter()
        .any(|reason| reason.contains("question_id coverage")));
}

#[test]
#[ignore = "slow answer proof lane; run `scripts/test-full-proof.sh`"]
fn eval_locomo_emits_structured_answer_proof_report() {
    let fixture = "tests/fixtures/locomo_sample.json";
    if !repo_root().join(fixture).exists() {
        eprintln!("skipping: fixture {fixture} not present (large file, not in CI); run locally");
        return;
    }
    let report = run_eval(
        "scripts/eval_locomo.py",
        &[
            "--fixture",
            fixture,
            "--answer-mode",
            "--max-per-question-type",
            "1",
            "--timeout-secs",
            "120",
        ],
        "locomo_answer_report.json",
    );

    assert_eq!(report["benchmark"], Value::String("locomo".to_string()));
    assert_eq!(report["mode"], Value::String("answer".to_string()));
    assert_eq!(report["fixture"]["entries_total"], Value::from(200));
    assert_eq!(report["fixture"]["entries_evaluated"], Value::from(4));
    assert_eq!(report["fixture"]["sample_fixture"], Value::Bool(true));
    assert_eq!(report["cases"]["count"], Value::from(4));
    assert_eq!(
        report["cases"]["rows"]
            .as_array()
            .expect("rows should be an array")
            .len(),
        4
    );
    assert!(report["overall"]["macro_f1"].is_number());
    assert!(report["results"]
        .as_object()
        .is_some_and(|results| !results.is_empty()));
    assert!(report["diagnostics"]["infra_failures"]
        .as_array()
        .expect("infra_failures should be an array")
        .is_empty());
    assert_eq!(report["proof"]["comparator_ready"], Value::Bool(false));

    let blockers = blocking_reasons(&report);
    assert!(blockers
        .iter()
        .any(|reason| reason.contains("partial run evaluated 4/200")));
    assert!(blockers
        .iter()
        .any(|reason| reason.contains("sample slice")));
}

#[test]
#[ignore = "slow answer proof lane; run `scripts/test-full-proof.sh`"]
fn eval_locomo_accepts_raw_public_fixture_and_surfaces_full_metadata() {
    let fixture = "tests/fixtures/locomo10.json";
    if !repo_root().join(fixture).exists() {
        eprintln!("skipping: fixture {fixture} not present (large file, not in CI); run locally");
        return;
    }
    let report = run_eval(
        "scripts/eval_locomo.py",
        &[
            "--fixture",
            fixture,
            "--answer-mode",
            "--max-per-question-type",
            "1",
            "--timeout-secs",
            "120",
        ],
        "locomo_raw_answer_report.json",
    );

    assert_eq!(report["benchmark"], Value::String("locomo".to_string()));
    assert_eq!(report["mode"], Value::String("answer".to_string()));
    assert_eq!(report["fixture"]["entries_total"], Value::from(1540));
    assert_eq!(report["fixture"]["entries_evaluated"], Value::from(4));
    assert_eq!(report["fixture"]["sample_fixture"], Value::Bool(false));
    assert_eq!(
        report["fixture"]["source_format"],
        Value::String("raw_public_release".to_string())
    );
    assert_eq!(
        report["fixture"]["official_public_release"],
        Value::Bool(true)
    );
    assert_eq!(report["cases"]["count"], Value::from(4));
    assert_eq!(
        report["selection"]["question_type_counts_total"]["single_hop"],
        Value::from(841)
    );
    assert_eq!(
        report["selection"]["question_type_counts_total"]["multi_hop"],
        Value::from(282)
    );
    assert_eq!(
        report["selection"]["question_type_counts_total"]["temporal"],
        Value::from(321)
    );
    assert_eq!(
        report["selection"]["question_type_counts_total"]["open_qa"],
        Value::from(96)
    );
    assert!(report["diagnostics"]["infra_failures"]
        .as_array()
        .expect("infra_failures should be an array")
        .is_empty());
    assert_eq!(report["proof"]["comparator_ready"], Value::Bool(false));

    let blockers = blocking_reasons(&report);
    assert!(blockers
        .iter()
        .any(|reason| reason.contains("partial run evaluated 4/1540")));
    assert!(!blockers
        .iter()
        .any(|reason| reason.contains("sample slice")));
}

#[test]
fn checked_in_locomo_full_answer_proof_is_comparator_ready() {
    let report_path = repo_root()
        .join("tests")
        .join("fixtures")
        .join("locomo_answer_full_report.json");
    let report: Value = serde_json::from_str(
        &fs::read_to_string(report_path).expect("checked-in LoCoMo proof report should exist"),
    )
    .expect("checked-in LoCoMo proof report should be valid json");

    assert_eq!(report["benchmark"], Value::String("locomo".to_string()));
    assert_eq!(report["mode"], Value::String("answer".to_string()));
    assert_eq!(report["fixture"]["entries_total"], Value::from(1540));
    assert_eq!(report["fixture"]["entries_evaluated"], Value::from(1540));
    assert_eq!(report["fixture"]["sample_fixture"], Value::Bool(false));
    assert_eq!(
        report["fixture"]["official_public_release"],
        Value::Bool(true)
    );
    assert_eq!(report["selection"]["full_run"], Value::Bool(true));
    assert_eq!(report["cases"]["count"], Value::from(1540));
    assert!(report["diagnostics"]["infra_failures"]
        .as_array()
        .expect("infra_failures should be an array")
        .is_empty());
    assert_eq!(report["proof"]["reproducible"], Value::Bool(true));
    assert_eq!(report["proof"]["comparator_ready"], Value::Bool(true));
    assert!(blocking_reasons(&report).is_empty());
}

#[test]
fn checked_in_lme_full_answer_proof_is_comparator_ready() {
    let report_path = repo_root()
        .join("tests")
        .join("fixtures")
        .join("longmemeval_answer_full_report.json");
    let report: Value = serde_json::from_str(
        &fs::read_to_string(report_path).expect("checked-in LME proof report should exist"),
    )
    .expect("checked-in LME proof report should be valid json");

    assert_eq!(
        report["benchmark"],
        Value::String("longmemeval-500".to_string())
    );
    assert_eq!(report["mode"], Value::String("answer".to_string()));
    assert_eq!(report["fixture"]["entries_total"], Value::from(500));
    assert_eq!(report["fixture"]["entries_evaluated"], Value::from(500));
    assert_eq!(report["fixture"]["sample_fixture"], Value::Bool(false));
    assert_eq!(
        report["fixture"]["official_public_release"],
        Value::Bool(true)
    );
    assert_eq!(report["selection"]["full_run"], Value::Bool(true));
    assert_eq!(report["cases"]["count"], Value::from(500));
    assert_eq!(
        report["public_surface"]["surface"],
        Value::String("official_qa_accuracy".to_string())
    );
    assert_eq!(report["public_surface"]["same_surface"], Value::Bool(true));
    assert_eq!(
        report["public_surface"]["entries_exported"],
        Value::from(500)
    );
    assert_eq!(
        report["public_surface"]["question_id_coverage"],
        Value::from(500)
    );
    assert_eq!(
        report["public_surface"]["question_type_coverage"],
        Value::from(500)
    );

    let hypotheses_path = report["public_surface"]["hypotheses_path"]
        .as_str()
        .expect("public hypotheses path should be a string");
    let hypotheses = fs::read_to_string(repo_root().join(hypotheses_path))
        .expect("checked-in public hypotheses artifact should exist");
    assert_eq!(hypotheses.lines().count(), 500);

    assert!(report["diagnostics"]["infra_failures"]
        .as_array()
        .expect("infra_failures should be an array")
        .is_empty());
    assert_eq!(report["proof"]["reproducible"], Value::Bool(true));
    assert_eq!(report["proof"]["public_surface_ready"], Value::Bool(true));
    assert_eq!(report["proof"]["comparator_ready"], Value::Bool(true));
    assert!(blocking_reasons(&report).is_empty());
}
