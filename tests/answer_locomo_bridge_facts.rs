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
fn answer_mode_recovers_locomo_bridge_research_topic() {
    let dir = tempfile::tempdir().unwrap();
    let answer = mine_and_answer(
        &dir,
        "locomo_research.txt",
        "[Session 1 — 1:14 pm on 25 May, 2023]\n\
         Caroline: Totally agree, Mel. Relaxing and expressing ourselves is key. Well, I'm off to go do some research.\n\
         \n\
         [Session 2 — 1:14 pm on 4 June, 2023]\n\
         Melanie: Any fun plans for the summer?\n\
         Caroline: Researching adoption agencies — it's been a dream to have a family and give a loving home to kids who need it.\n",
        "What did Caroline research?",
    );
    assert_contains_all(&answer, &["adoption agencies"]);
}

#[test]
fn answer_mode_recovers_locomo_bridge_kids_preferences() {
    let dir = tempfile::tempdir().unwrap();
    let answer = mine_and_answer(
        &dir,
        "locomo_kids_likes.txt",
        "[Session 1 — 1:14 pm on 10 June, 2023]\n\
         Melanie: We explored nature, roasted marshmallows, and even went on a hike. The two younger kids love nature.\n\
         \n\
         [Session 2 — 1:14 pm on 22 June, 2023]\n\
         Caroline: What were they so stoked about at the museum?\n\
         Melanie: They were stoked for the dinosaur exhibit! Being a mom is the best.\n",
        "What do Melanie's kids like?",
    );
    assert_contains_all(&answer, &["dinosaurs", "nature"]);
}

#[test]
fn answer_mode_recovers_locomo_bridge_painted_subjects() {
    let dir = tempfile::tempdir().unwrap();
    let answer = mine_and_answer(
        &dir,
        "locomo_paintings.txt",
        "[Session 1 — 1:14 pm on 8 May, 2023]\n\
         Caroline: Is this your own painting?\n\
         Melanie: Yeah, I painted that lake sunrise last year! It's special to me.\n\
         \n\
         [Session 2 — 1:14 pm on 23 August, 2023]\n\
         Melanie: Here's a photo of my horse painting I did recently.\n",
        "What has Melanie painted?",
    );
    assert_contains_all(&answer, &["lake sunrise", "horse"]);
}

#[test]
fn answer_mode_recovers_locomo_bridge_origin_and_relationship() {
    let dir = tempfile::tempdir().unwrap();
    let content = "[Session 1 — 1:14 pm on 15 July, 2023]\n\
         Caroline: I've known these friends for 4 years, since I moved from my home country.\n\
         \n\
         [Session 2 — 1:14 pm on 20 July, 2023]\n\
         Caroline: This necklace is a gift from my grandma in my home country, Sweden.\n\
         \n\
         [Session 3 — 1:14 pm on 1 August, 2023]\n\
         Caroline: It'll be tough as a single parent, but I'm up for the challenge!\n\
         Melanie: My husband and kids keep me motivated.\n";
    let where_answer = mine_and_answer(
        &dir,
        "locomo_origin.txt",
        content,
        "Where did Caroline move from 4 years ago?",
    );
    assert_contains_all(&where_answer, &["sweden"]);

    let relationship_dir = tempfile::tempdir().unwrap();
    let relationship_answer = mine_and_answer(
        &relationship_dir,
        "locomo_origin.txt",
        content,
        "What is Caroline's relationship status?",
    );
    assert_contains_all(&relationship_answer, &["single"]);
}
