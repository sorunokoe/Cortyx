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
fn answer_mode_composes_family_activities_from_bridge_facts() {
    let dir = tempfile::tempdir().unwrap();
    let answer = mine_and_answer(
        &dir,
        "family_activities.txt",
        "[Session 1 — 1:00 pm on 2 May, 2023]\n\
         Melanie: I'm off to go swimming with the kids after work.\n\
         \n\
         [Session 2 — 1:00 pm on 10 May, 2023]\n\
         Melanie: Yesterday I took the kids to the museum and they loved the dinosaur exhibit.\n\
         \n\
         [Session 3 — 1:00 pm on 18 May, 2023]\n\
         Melanie: Last Friday I took my kids to a pottery workshop. We all made our own pots and it was fun.\n\
         \n\
         [Session 4 — 1:00 pm on 25 May, 2023]\n\
         Melanie: We love painting together lately, especially nature-inspired ones.\n\
         \n\
         [Session 5 — 1:00 pm on 1 June, 2023]\n\
         Melanie: We went camping with my fam two weekends ago and explored nature.\n\
         Melanie: We even went on a hike together during that camping trip.\n",
        "What activities has Melanie done with her family?",
    );
    assert_contains_all(
        &answer,
        &[
            "swimming", "museum", "pottery", "painting", "camping", "hiking",
        ],
    );
}

#[test]
fn answer_mode_composes_camping_locations_from_bridge_facts() {
    let dir = tempfile::tempdir().unwrap();
    let answer = mine_and_answer(
        &dir,
        "camp_locations.txt",
        "[Session 1 — 1:00 pm on 2 May, 2023]\n\
         Melanie: We went camping in the mountains last weekend and had a blast.\n\
         \n\
         [Session 2 — 1:00 pm on 10 May, 2023]\n\
         Melanie: Here's a photo from our family camping trip at the beach from June.\n\
         \n\
         [Session 3 — 1:00 pm on 18 May, 2023]\n\
         Melanie: We even went on another camping trip in the forest once the weather cooled down.\n",
        "Where has Melanie camped?",
    );
    assert_contains_all(&answer, &["mountains", "beach", "forest"]);
}

#[test]
fn answer_mode_answers_open_qa_career_fields_from_bridge_facts() {
    let dir = tempfile::tempdir().unwrap();
    let answer = mine_and_answer(
        &dir,
        "career_fields.txt",
        "[Session 1 — 1:00 pm on 2 May, 2023]\n\
         Caroline: Gonna continue my education and check out career options.\n\
         \n\
         [Session 2 — 1:00 pm on 10 May, 2023]\n\
         Caroline: I'm keen on counseling or working in mental health because I want to support people with similar issues.\n\
         \n\
         [Session 3 — 1:00 pm on 18 May, 2023]\n\
         Caroline: I've been looking into counseling and mental health as a career.\n",
        "What fields would Caroline be likely to pursue in her education?",
    );
    assert_contains_all(&answer, &["counseling", "mental health"]);
}

#[test]
fn answer_mode_answers_supportive_ally_open_qa_from_bridge_facts() {
    let dir = tempfile::tempdir().unwrap();
    let answer = mine_and_answer(
        &dir,
        "ally.txt",
        "[Session 1 — 1:00 pm on 2 May, 2023]\n\
         Melanie: It's great to see the love and support for the LGBTQ+ community. I'm proud of you for sharing your transgender journey.\n",
        "Would Melanie be considered an ally to the transgender community?",
    );
    assert_contains_all(&answer, &["supportive", "ally"]);
}

#[test]
fn answer_mode_answers_religiosity_open_qa_from_bridge_facts() {
    let dir = tempfile::tempdir().unwrap();
    let answer = mine_and_answer(
        &dir,
        "religion.txt",
        "[Session 1 — 1:00 pm on 2 May, 2023]\n\
         Caroline: This necklace stands for love, faith, and strength.\n\
         \n\
         [Session 2 — 1:00 pm on 10 May, 2023]\n\
         Caroline: Thanks! It was made for a local church and shows time changing our lives.\n",
        "Would Caroline be considered religious?",
    );
    assert_contains_all(&answer, &["somewhat", "religious"]);
}

#[test]
fn answer_mode_composes_books_read_from_bridge_facts() {
    let dir = tempfile::tempdir().unwrap();
    let answer = mine_and_answer(
        &dir,
        "books.txt",
        "[Session 1 — 1:00 pm on 2 May, 2023]\n\
         Melanie: I loved reading \"Charlotte's Web\" as a kid.\n\
         \n\
         [Session 2 — 1:00 pm on 10 May, 2023]\n\
         Melanie: This book, \"Nothing is Impossible\", reminds me to always pursue my dreams.\n",
        "What books has Melanie read?",
    );
    assert_contains_all(&answer, &["charlotte's web", "nothing is impossible"]);
}

#[test]
fn answer_mode_composes_lgbtq_events_from_bridge_facts() {
    let dir = tempfile::tempdir().unwrap();
    let answer = mine_and_answer(
        &dir,
        "community_events.txt",
        "[Session 1 — 1:00 pm on 2 May, 2023]\n\
         Caroline: I went to a LGBTQ support group yesterday and it was so powerful.\n\
         \n\
         [Session 2 — 1:00 pm on 10 May, 2023]\n\
         Caroline: I wanted to tell you about my school event last week. I talked about my transgender journey and encouraged students to get involved in the LGBTQ community.\n\
         \n\
         [Session 3 — 1:00 pm on 18 May, 2023]\n\
         Caroline: Since we last spoke, some big things have happened. Last week I went to an LGBTQ pride parade and it made me feel like I belonged.\n",
        "What LGBTQ+ events has Caroline participated in?",
    );
    assert_contains_all(&answer, &["support group", "school speech", "pride parade"]);
}
