use super::*;
use std::fs;

fn make_evidence(content: &str) -> (tempfile::TempDir, Vec<EvidenceItem>) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session.md");
    fs::write(&path, content).unwrap();
    (
        dir,
        vec![EvidenceItem {
            path,
            score: 7.0,
            metadata: None,
            snippet: String::new(),
        }],
    )
}

#[test]
fn select_answer_surface_formats_supportive_ally_for_openqa_yes_no() {
    let (_dir, evidence) = make_evidence(
        "## answer_surface\n<!-- SECTION: answer_surface -->\n\
         | question_pattern | answer_span | confidence |\n\
         | --- | --- | --- |\n\
         | melanie ally lgbtq community transgender support | supportive ally | 0.93 |\n\
         <!-- /SECTION -->\n",
    );
    let answer = select_answer(
        "Would Melanie be considered an ally to the transgender community?",
        &evidence,
        None,
    )
    .unwrap();
    assert_eq!(answer, "Yes, supportive ally");
}

#[test]
fn select_answer_surface_requires_anchor_overlap_for_typed_openqa() {
    let (_dir, evidence) = make_evidence(
        "## answer_surface\n<!-- SECTION: answer_surface -->\n\
         | question_pattern | answer_span | confidence |\n\
         | --- | --- | --- |\n\
         | melanie camping beach mountains nature outdoors | national park | 0.93 |\n\
         <!-- /SECTION -->\n",
    );
    let answer = select_answer_surface("Would Caroline be considered religious?", &evidence);
    assert!(answer.is_none());
}

#[test]
fn select_answer_typed_openqa_abstains_without_surface_support() {
    let (_dir, evidence) = make_evidence(
        "Caroline: That's amazing and I'm so proud of you.\nMelanie: We should catch up soon.\n",
    );
    let answer = select_answer("Would Caroline be considered religious?", &evidence, None);
    assert!(answer.is_none());
}
