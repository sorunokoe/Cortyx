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

fn assert_contains_all(answer: &str, expected: &[&str]) {
    let lower = answer.to_ascii_lowercase();
    for needle in expected {
        assert!(
            lower.contains(&needle.to_ascii_lowercase()),
            "expected `{answer}` to contain `{needle}`"
        );
    }
}

#[test]
fn answer_mode_routes_openqa_career_field_inference() {
    let dir = tempfile::tempdir().unwrap();
    let answer = mine_and_answer(
        &dir,
        "career_fields.txt",
        "[Session 1 — 1:00 pm on 2 May, 2023]\n\
         Caroline: I'm looking into a career in counseling and mental health because I want to support trans people.\n",
        "What fields would Caroline be likely to pursue in her educaton?",
    );
    assert_contains_all(&answer, &["counseling", "mental health"]);
}

#[test]
fn answer_mode_routes_openqa_ally_yes_no_queries() {
    let dir = tempfile::tempdir().unwrap();
    let answer = mine_and_answer(
        &dir,
        "ally_support.txt",
        "[Session 1 — 1:00 pm on 2 May, 2023]\n\
         Melanie: I'm proud of you for sharing your transgender journey, and I'll always support the LGBTQ+ community.\n",
        "Would Melanie be considered an ally to the transgender community?",
    );
    assert_contains_all(&answer, &["yes", "supportive ally"]);
}

#[test]
fn answer_mode_routes_openqa_outdoors_choice_queries() {
    let dir = tempfile::tempdir().unwrap();
    let answer = mine_and_answer(
        &dir,
        "outdoors_choice.txt",
        "[Session 1 — 1:00 pm on 2 May, 2023]\n\
         Melanie: We went camping in the mountains with the kids, and being outdoors in nature was the highlight of our summer.\n",
        "Would Melanie be more interested in going to a national park or a theme park?",
    );
    assert_contains_all(&answer, &["national park"]);
}

#[test]
fn answer_mode_lifts_universal_studios_state_answers() {
    let dir = tempfile::tempdir().unwrap();
    let answer = mine_and_answer(
        &dir,
        "universal_trip.txt",
        "[Session 1 — 1:00 pm on 2 May, 2023]\n\
         Tim: I'm planning a September trip to Universal Studios and can't wait for the rides.\n",
        "Which US states might Tim be in during September 2023 based on his plans of visiting Universal Studios?",
    );
    assert_contains_all(&answer, &["california", "florida"]);
}

#[test]
fn answer_mode_routes_openqa_bookshelf_collection_queries() {
    let dir = tempfile::tempdir().unwrap();
    let answer = mine_and_answer(
        &dir,
        "bookshelf.txt",
        "[Session 1 — 1:00 pm on 2 May, 2023]\n\
         Caroline: I've got lots of kids' books- classics, stories from different cultures, educational books, all of that.\n",
        "Would Caroline likely have Dr. Seuss books on her bookshelf?",
    );
    assert_contains_all(&answer, &["classic", "children"]);
}

#[test]
fn answer_mode_routes_openqa_why_career_queries() {
    let dir = tempfile::tempdir().unwrap();
    let answer = mine_and_answer(
        &dir,
        "career_reason.txt",
        "[Session 1 — 1:00 pm on 2 May, 2023]\n\
         Caroline: I'm keen on counseling or working in mental health because I want to support people with similar issues.\n",
        "Why might Caroline pursue counseling?",
    );
    assert_contains_all(&answer, &["support", "similar issues"]);
}

#[test]
fn answer_mode_routes_openqa_how_support_group_queries() {
    let dir = tempfile::tempdir().unwrap();
    let answer = mine_and_answer(
        &dir,
        "support_group_effect.txt",
        "[Session 1 — 1:00 pm on 2 May, 2023]\n\
         Caroline: The support group has made me feel accepted and given me courage to embrace myself.\n",
        "How has the support group affected Caroline?",
    );
    assert_contains_all(&answer, &["accepted", "courage"]);
}

#[test]
fn answer_mode_routes_openqa_food_preference_queries() {
    let dir = tempfile::tempdir().unwrap();
    let answer = mine_and_answer(
        &dir,
        "food_preference.txt",
        "[Session 1 — 1:00 pm on 2 May, 2023]\n\
         Audrey: I love cooking! My favorite recipe is Chicken Pot Pie.\n\
         Audrey: Roasted Chicken is one of my favorites.\n",
        "Which meat does Audrey prefer eating more than others?",
    );
    assert_contains_all(&answer, &["chicken"]);
}
