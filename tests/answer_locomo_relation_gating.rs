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

fn mine_file(dir: &TempDir, file_name: &str, content: &str) {
    fs::write(dir.path().join(file_name), content).unwrap();
    let mined = run(&["mine", file_name], dir.path());
    assert_success(&mined, "mine");
}

fn answer_task(dir: &TempDir, task: &str) -> String {
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

fn mine_and_answer(dir: &TempDir, file_name: &str, content: &str, task: &str) -> String {
    mine_file(dir, file_name, content);
    answer_task(dir, task)
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

fn assert_count(answer: &str, expected: usize) {
    let lower = answer.to_ascii_lowercase();
    let numeric = expected.to_string();
    let word = match expected {
        0 => "zero",
        1 => "one",
        2 => "two",
        3 => "three",
        4 => "four",
        5 => "five",
        6 => "six",
        7 => "seven",
        8 => "eight",
        9 => "nine",
        10 => "ten",
        _ => "",
    };
    assert!(
        lower.contains(&numeric) || (!word.is_empty() && lower.contains(word)),
        "expected `{answer}` to contain count `{expected}`"
    );
}

#[test]
fn answer_mode_counts_lgbtq_events_without_ally_leakage() {
    let dir = tempfile::tempdir().unwrap();
    let answer = mine_and_answer(
        &dir,
        "event_count.txt",
        "[Session 1 — 1:00 pm on 2 May, 2023]\n\
         Caroline: I went to a LGBTQ support group yesterday and it was so powerful.\n\
         \n\
         [Session 2 — 1:00 pm on 10 May, 2023]\n\
         Melanie: I'm proud of you for sharing your transgender journey, and I'll always support the LGBTQ+ community.\n\
         \n\
         [Session 3 — 1:00 pm on 18 May, 2023]\n\
         Caroline: I wanted to tell you about my school event last week. I talked about my transgender journey and encouraged students to get involved in the LGBTQ community.\n\
         \n\
         [Session 4 — 1:00 pm on 25 May, 2023]\n\
         Caroline: Last week I went to a LGBTQ pride parade and it made me feel like I belonged.\n",
        "How many LGBTQ+ events has Caroline participated in?",
    );
    assert_count(&answer, 3);
    assert_not_contains(&answer, &["ally"]);
}

#[test]
fn answer_mode_keeps_child_help_events_separate_from_community_events() {
    let dir = tempfile::tempdir().unwrap();
    let answer = mine_and_answer(
        &dir,
        "child_help_events.txt",
        "[Session 1 — 1:00 pm on 2 May, 2023]\n\
         Caroline: I went to a LGBTQ support group yesterday and it was so powerful.\n\
         \n\
         [Session 2 — 1:00 pm on 10 May, 2023]\n\
         Caroline: I wanted to tell you about my school event last week. I talked about my transgender journey and encouraged students to get involved in the LGBTQ community.\n\
         \n\
         [Session 3 — 1:00 pm on 18 May, 2023]\n\
         Caroline: I joined a mentoring program for trans youth last weekend and it has been incredibly meaningful.\n\
         \n\
         [Session 4 — 1:00 pm on 25 May, 2023]\n\
         Caroline: Last week I went to a LGBTQ pride parade and it made me feel like I belonged.\n",
        "What events has Caroline participated in to help children?",
    );
    assert_contains_all(&answer, &["school speech", "mentoring program"]);
    assert_not_contains(&answer, &["support group", "pride parade"]);
}

#[test]
fn answer_mode_filters_book_titles_from_publishers() {
    let dir = tempfile::tempdir().unwrap();
    let answer = mine_and_answer(
        &dir,
        "books_and_publishers.txt",
        "[Session 1 — 1:00 pm on 2 May, 2023]\n\
         Melanie: I loved reading \"Charlotte's Web\" as a kid.\n\
         \n\
         [Session 2 — 1:00 pm on 10 May, 2023]\n\
         Caroline: Where is that illustrated edition from?\n\
         Melanie: It's from MinaLima.\n\
         \n\
         [Session 3 — 1:00 pm on 18 May, 2023]\n\
         Melanie: This book, \"Nothing is Impossible\", reminds me to always pursue my dreams.\n",
        "What books has Melanie read?",
    );
    assert_contains_all(&answer, &["charlotte's web", "nothing is impossible"]);
    assert_not_contains(&answer, &["minalima"]);
}

#[test]
fn answer_mode_keeps_self_care_activities_out_of_family_answers() {
    let dir = tempfile::tempdir().unwrap();
    mine_file(
        &dir,
        "activity_subtypes.txt",
        "[Session 1 — 1:00 pm on 2 May, 2023]\n\
         Melanie: Running is my favorite way to destress after a tough week.\n\
         Melanie: I signed up for a pottery class as self-care, and reading before bed is so calming.\n\
         \n\
         [Session 2 — 1:00 pm on 10 May, 2023]\n\
         Melanie: Yesterday I took the kids to the museum and they loved the dinosaur exhibit.\n\
         Melanie: We went camping with my family and even went on a hike together.\n",
    );

    let answer = answer_task(&dir, "What activities has Melanie done with her family?");
    assert_contains_all(&answer, &["museum", "camping", "hiking"]);
    assert_not_contains(&answer, &["running", "pottery", "reading"]);
}

#[test]
fn answer_mode_keeps_family_activities_out_of_self_care_answers() {
    let dir = tempfile::tempdir().unwrap();
    mine_file(
        &dir,
        "activity_subtypes.txt",
        "[Session 1 — 1:00 pm on 2 May, 2023]\n\
         Melanie: Running is my favorite way to destress after a tough week.\n\
         Melanie: I signed up for a pottery class as self-care, and reading before bed is so calming.\n\
         \n\
         [Session 2 — 1:00 pm on 10 May, 2023]\n\
         Melanie: Yesterday I took the kids to the museum and they loved the dinosaur exhibit.\n\
         Melanie: We went camping with my family and even went on a hike together.\n",
    );

    let answer = answer_task(&dir, "What does Melanie do to destress?");
    assert_contains_all(&answer, &["running", "pottery", "reading"]);
    assert_not_contains(&answer, &["museum", "camping", "hiking"]);
}

#[test]
fn answer_mode_keeps_move_origin_answer_out_of_painting_lists() {
    let dir = tempfile::tempdir().unwrap();
    let answer = mine_and_answer(
        &dir,
        "move_origin_with_painting.txt",
        "[Session 1 — 1:00 pm on 2 May, 2023]\n\
         Caroline: This necklace was a gift from my grandma in my home country, Sweden.\n\
         Melanie: I painted the beach at sunset last week.\n\
         \n\
         [Session 2 — 1:00 pm on 10 May, 2023]\n\
         Caroline: I've known these friends for 4 years, since I moved from my home country.\n",
        "Where did Caroline move from 4 years ago?",
    );
    assert_contains_all(&answer, &["sweden"]);
    assert_not_contains(&answer, &["beach", "sunset"]);
}
