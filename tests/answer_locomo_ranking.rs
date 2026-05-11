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

fn assert_not_contains(answer: &str, unexpected: &[&str]) {
    let lower = answer.to_ascii_lowercase();
    for needle in unexpected {
        assert!(
            !lower.contains(&needle.to_ascii_lowercase()),
            "expected `{answer}` to omit `{needle}`"
        );
    }
}

#[test]
fn answer_mode_answers_move_origin_from_supporting_bridge_facts() {
    let dir = tempfile::tempdir().unwrap();
    let answer = mine_and_answer(
        &dir,
        "move_origin.txt",
        "[Session 1 — 1:00 pm on 2 May, 2023]\n\
         Caroline: This necklace was a gift from my grandma in my home country, Sweden.\n\
         Caroline: It always reminds me of where I come from.\n\
         \n\
         [Session 2 — 1:00 pm on 9 June, 2023]\n\
         Caroline: I've known these friends for 4 years, since I moved from my home country.\n",
        "Where did Caroline move from 4 years ago?",
    );
    assert_contains_all(&answer, &["sweden"]);
}

#[test]
fn answer_mode_composes_camping_locations_when_multiple_locations_share_one_path() {
    let dir = tempfile::tempdir().unwrap();
    let answer = mine_and_answer(
        &dir,
        "camping_locations.txt",
        "[Session 1 — 1:00 pm on 2 May, 2023]\n\
         Melanie: I painted the beach at sunset last week.\n\
         Melanie: Last weekend we went camping in the mountains and it was amazing.\n\
         Melanie: A few weeks later we camped at the beach too and the kids loved it.\n\
         \n\
         [Session 2 — 1:00 pm on 18 May, 2023]\n\
         Melanie: We even went on another camping trip in the forest once the weather cooled down.\n",
        "Where has Melanie camped?",
    );
    assert_contains_all(&answer, &["mountains", "beach", "forest"]);
}

#[test]
fn answer_mode_excludes_future_lgbtq_events_from_participation_lists() {
    let dir = tempfile::tempdir().unwrap();
    let answer = mine_and_answer(
        &dir,
        "lgbtq_events.txt",
        "[Session 1 — 1:00 pm on 2 May, 2023]\n\
         Caroline: I went to a LGBTQ support group yesterday and it was so powerful.\n\
         \n\
         [Session 2 — 1:00 pm on 10 May, 2023]\n\
         Caroline: I wanted to tell you about my school event last week. I talked about my transgender journey and encouraged students to get involved in the LGBTQ community.\n\
         \n\
         [Session 3 — 1:00 pm on 18 May, 2023]\n\
         Caroline: Last week I went to a LGBTQ pride parade and it made me feel like I belonged.\n\
         \n\
         [Session 4 — 1:00 pm on 25 May, 2023]\n\
         Caroline: Next month I'm having an LGBTQ art show with my paintings and I can't wait.\n",
        "What LGBTQ+ events has Caroline participated in?",
    );
    assert_contains_all(&answer, &["support group", "school speech", "pride parade"]);
    assert_not_contains(&answer, &["art show"]);
}

#[test]
fn answer_mode_keeps_religiosity_answer_with_social_support_distractors() {
    let dir = tempfile::tempdir().unwrap();
    let answer = mine_and_answer(
        &dir,
        "religion_with_distractors.txt",
        "[Session 1 — 1:00 pm on 2 May, 2023]\n\
         Caroline: I went to a LGBTQ support group yesterday and it was so powerful.\n\
         Caroline: The transgender stories were so inspiring! I was so happy and thankful for all the support.\n\
         \n\
         [Session 2 — 1:00 pm on 10 May, 2023]\n\
         Caroline: This necklace stands for love, faith, and strength.\n\
         Caroline: Thanks! It was made for a local church and shows time changing our lives.\n",
        "Would Caroline be considered religious?",
    );
    assert_contains_all(&answer, &["religious"]);
}

#[test]
fn answer_mode_formats_lgbtq_membership_queries_as_supportive_ally() {
    let dir = tempfile::tempdir().unwrap();
    let answer = mine_and_answer(
        &dir,
        "ally_membership.txt",
        "[Session 1 — 1:00 pm on 2 May, 2023]\n\
         Melanie: It's great to see the love and support for the LGBTQ+ community. I'm proud of you for sharing your transgender journey.\n",
        "Would Melanie be considered a member of the LGBTQ+ community?",
    );
    assert_contains_all(&answer, &["likely no", "supportive ally"]);
}
