use std::fs;
use std::path::PathBuf;

use serde_json::Value;

mod common;
use common::run;

fn assert_success(output: &std::process::Output, label: &str) {
    assert!(
        output.status.success(),
        "{label} failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn fixture_entry(question_id: &str) -> (String, String, String, String) {
    let fixture_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/longmemeval_500.json");
    let entries: Vec<Value> =
        serde_json::from_str(&fs::read_to_string(fixture_path).unwrap()).unwrap();
    let entry = entries
        .into_iter()
        .find(|entry| entry.get("question_id").and_then(Value::as_str) == Some(question_id))
        .unwrap_or_else(|| panic!("fixture entry not found: {question_id}"));
    (
        entry["neuron_filename"].as_str().unwrap().to_string(),
        entry["neuron_source_content"].as_str().unwrap().to_string(),
        entry["question"].as_str().unwrap().to_string(),
        entry["expected_answer"].as_str().unwrap().to_string(),
    )
}

fn mine_and_answer_fixture(question_id: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let (file_name, content, question, _) = fixture_entry(question_id);
    fs::write(dir.path().join(&file_name), content).unwrap();

    let mined = run(&["mine", &file_name], dir.path());
    assert_success(&mined, "mine");

    let answered = run(
        &[
            "get-contexts",
            "--task",
            &question,
            "--kind",
            "conversation",
            "--answer-mode",
        ],
        dir.path(),
    );
    assert_success(&answered, "get-contexts");
    String::from_utf8_lossy(&answered.stdout).trim().to_string()
}

fn contains_number_or_word(answer: &str, expected_digit: &str, expected_word: &str) -> bool {
    let lower = answer.to_ascii_lowercase();
    answer.contains(expected_digit) || lower.contains(expected_word)
}

fn assert_in_order(answer: &str, parts: &[&str]) {
    let lower = answer.to_ascii_lowercase();
    let mut last = 0usize;
    for (index, part) in parts.iter().enumerate() {
        let pos = lower
            .find(&part.to_ascii_lowercase())
            .unwrap_or_else(|| panic!("missing {part} in answer: {answer}"));
        if index > 0 {
            assert!(pos > last, "unexpected ordering: {answer}");
        }
        last = pos;
    }
}

#[test]
fn answer_mode_solves_public_temporal_duration_fixture() {
    let answer = mine_and_answer_fixture("08f4fc43");
    assert!(
        answer.contains("30 days") || answer.contains("31 days"),
        "unexpected answer: {answer}"
    );
}

#[test]
fn answer_mode_solves_public_temporal_event_count_fixture() {
    let answer = mine_and_answer_fixture("a3838d2b");
    assert!(
        contains_number_or_word(&answer, "4", "four"),
        "unexpected answer: {answer}"
    );
}

#[test]
fn answer_mode_solves_public_temporal_employment_gap_fixture() {
    let answer = mine_and_answer_fixture("gpt4_93159ced");
    assert_eq!(answer, "4 years and 9 months");
}

#[test]
fn answer_mode_solves_public_temporal_booking_lead_time_fixture() {
    let answer = mine_and_answer_fixture("982b5123");
    assert!(
        contains_number_or_word(&answer, "5", "five")
            && answer.to_ascii_lowercase().contains("month"),
        "unexpected answer: {answer}"
    );
}

#[test]
fn answer_mode_solves_public_temporal_binary_choice_fixture() {
    let answer = mine_and_answer_fixture("gpt4_2487a7cb");
    assert!(
        answer
            .to_ascii_lowercase()
            .contains("data analysis using python"),
        "unexpected answer: {answer}"
    );
}

#[test]
fn answer_mode_orders_public_temporal_trip_sequence_fixture() {
    let answer = mine_and_answer_fixture("gpt4_7f6b06db");
    let lower = answer.to_ascii_lowercase();
    let muir = lower.find("muir woods").unwrap_or(usize::MAX);
    let big_sur = lower.find("big sur").unwrap_or(usize::MAX);
    let yosemite = lower.find("yosemite").unwrap_or(usize::MAX);
    assert!(
        muir < big_sur && big_sur < yosemite,
        "unexpected ordering: {answer}"
    );
}

#[test]
fn answer_mode_solves_public_temporal_event_gap_fixture() {
    let answer = mine_and_answer_fixture("0bb5a684");
    assert!(
        answer.contains("7 days") || answer.contains("8 days"),
        "unexpected answer: {answer}"
    );
}

#[test]
fn answer_mode_orders_public_temporal_coupon_sequence_fixture() {
    let answer = mine_and_answer_fixture("gpt4_18c2b244");
    assert_in_order(&answer, &["luvs diapers", "amazon gift card", "shoprite"]);
}

#[test]
fn answer_mode_solves_public_temporal_holi_gap_fixture_2a1811e2() {
    let answer = mine_and_answer_fixture("2a1811e2");
    assert!(
        answer.contains("21 days") || answer.contains("22 days"),
        "unexpected answer: {answer}"
    );
}

#[test]
fn answer_mode_solves_public_temporal_black_friday_gap_fixture_c8090214() {
    let answer = mine_and_answer_fixture("c8090214");
    assert!(
        answer.contains("7 days") || answer.contains("8 days"),
        "unexpected answer: {answer}"
    );
}

#[test]
fn answer_mode_solves_public_temporal_rachel_house_gap_fixture_2c63a862() {
    let answer = mine_and_answer_fixture("2c63a862");
    assert!(
        answer.contains("14 days") || answer.contains("15 days"),
        "unexpected answer: {answer}"
    );
}

#[test]
fn answer_mode_solves_public_temporal_shoelace_gap_fixture_dcfa8644() {
    let answer = mine_and_answer_fixture("dcfa8644");
    assert!(
        answer.contains("14 days") || answer.contains("15 days"),
        "unexpected answer: {answer}"
    );
}
