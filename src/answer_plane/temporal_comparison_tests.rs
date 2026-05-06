use super::*;
use std::fs;

fn write_temporal_evidence_files(
    files: &[(&str, &str, f32)],
) -> (tempfile::TempDir, Vec<EvidenceItem>) {
    let dir = tempfile::tempdir().unwrap();
    let mut evidence = Vec::new();
    for (name, content, score) in files {
        let path = dir.path().join(name);
        fs::write(&path, content).unwrap();
        evidence.push(EvidenceItem {
            path,
            score: *score,
            metadata: None,
            snippet: String::new(),
        });
    }
    (dir, evidence)
}

#[test]
fn select_answer_prefers_arrival_date_over_preorder_date_for_got_first_query() {
    let (_dir, evidence) = write_temporal_evidence_files(&[(
        "session_chunk.verbatim.md",
        "[Session 8 - 10:11 am on 15 March, 2023]\n\
User: I'm planning a trip to Hawaii and need adapters for my new laptop, Dell XPS 13, and my new smartphone, Samsung Galaxy S22. By the way, I pre-ordered the laptop on January 28th, and it finally arrived on February 25th after a delay from the original expected arrival date of February 11th.\n\
User: I also got a new Samsung Galaxy S22 on February 20th and have been comparing camera settings.\n",
        10.0,
    )]);
    let answer = select_answer(
        "Which device did I get first, the Samsung Galaxy S22 or the Dell XPS 13?",
        &evidence,
        None,
    )
    .unwrap();
    assert_eq!(answer, "Samsung Galaxy S22");
}
