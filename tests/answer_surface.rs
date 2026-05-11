use std::fs;

use tempfile::TempDir;

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

fn mine_and_answer(dir: &TempDir, file_name: &str, content: &str, task: &str) -> String {
    fs::write(dir.path().join(file_name), content).unwrap();

    let mined = run(&["mine", file_name], dir.path());
    assert_success(&mined, "mine");

    let answered = run(
        &[
            "get-contexts",
            "--task",
            task,
            "--kind",
            "conversation",
            "--answer-mode",
        ],
        dir.path(),
    );
    assert_success(&answered, "get-contexts");
    String::from_utf8_lossy(&answered.stdout).trim().to_string()
}

#[test]
fn answer_mode_recovers_named_speaker_research_topic() {
    let dir = tempfile::tempdir().unwrap();
    let answer = mine_and_answer(
        &dir,
        "caroline_research.txt",
        "[Session 2 — 3:00 pm on 14 May, 2023]\n\
         Melanie: What have you been researching lately?\n\
         Caroline: I've been researching adoption agencies because it's been a dream to have a family.\n",
        "What did Caroline research?",
    );
    assert!(
        answer.to_ascii_lowercase().contains("adoption agencies"),
        "unexpected answer: {answer}"
    );
}

#[test]
fn answer_mode_resolves_relative_event_date_from_session_header() {
    let dir = tempfile::tempdir().unwrap();
    let answer = mine_and_answer(
        &dir,
        "caroline_dates.txt",
        "[Session 1 — 1:56 pm on 8 May, 2023]\n\
         Caroline: I went to a LGBTQ support group yesterday and it was so powerful.\n\
         Melanie: Wow, that's cool, Caroline!\n",
        "When did Caroline go to the LGBTQ support group?",
    );
    assert!(answer.contains("7 May 2023"), "unexpected answer: {answer}");
}

#[test]
fn answer_mode_prefers_latest_personal_best_time() {
    let dir = tempfile::tempdir().unwrap();
    let answer = mine_and_answer(
        &dir,
        "fitness.txt",
        "User: I recently set a personal best time in a charity 5K run with a time of 27:12.\n\
         Assistant: Congratulations on the new PR!\n\
         User: I'm training for another charity 5K run, and I'm hoping to beat my personal best time of 25:50 this time around.\n",
        "What was my personal best time in the charity 5K run?",
    );
    assert!(answer.contains("25:50"), "unexpected answer: {answer}");
}
