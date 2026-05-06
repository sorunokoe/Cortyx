use super::*;
use crate::index::NeuronIndex;
use std::fs;

fn write_validator_evidence(content: &str) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session.md");
    fs::write(&path, content).unwrap();
    (dir, path)
}

fn single_evidence(path: std::path::PathBuf) -> Vec<EvidenceItem> {
    vec![EvidenceItem {
        path,
        score: 7.0,
        metadata: None,
        snippet: String::new(),
    }]
}

#[test]
fn render_answer_output_decision_marks_absent_queries_unsupported() {
    let (_dir, path) = write_validator_evidence(
        "We spent the afternoon at the park and talked about the weather.\n",
    );
    let idx = NeuronIndex::default();
    let result =
        render_answer_output_decision(&idx, "What is my job?", &[(path, 7.0)], false, None);
    assert_eq!(result, Err(AnswerAbstentionReason::Unsupported));
}

#[test]
fn select_answer_surface_abstains_on_absent_named_entity_mismatch() {
    let (_dir, path) = write_validator_evidence(
        "# Session facts\n\n## answer_surface\n<!-- SECTION: answer_surface -->\n| question_pattern | answer_span | confidence |\n| --- | --- | --- |\n| job occupation profession work career role | software engineer specifically a | 0.92 |\n<!-- /SECTION -->\n",
    );
    let answer = select_answer(
        "How long have I been working before I started my current job at Google?",
        &single_evidence(path),
        None,
    );
    assert!(answer.is_none());
}

#[test]
fn render_answer_output_decision_rejects_garbage_preference_answer_as_unsupported() {
    let (_dir, path) = write_validator_evidence(
        "Here are some popular meditation apps and guided meditation resources that can help you relax before bed: **Meditation Apps:** 1\n",
    );
    let idx = NeuronIndex::default();
    let result = render_answer_output_decision(
        &idx,
        "Can you suggest some accessories that would complement my current photography setup?",
        &[(path, 7.0)],
        false,
        None,
    );
    assert_eq!(result, Err(AnswerAbstentionReason::Unsupported));
}

#[test]
fn render_answer_output_decision_treats_conflicting_surface_rows_as_unsupported() {
    let (_dir, path) = write_validator_evidence(
        "# Session facts\n\n## answer_surface\n<!-- SECTION: answer_surface -->\n| question_pattern | answer_span | confidence |\n| --- | --- | --- |\n| john kind online group join | service-focused online group | 0.92 |\n| john kind online group join | neighborhood mentoring online group | 0.91 |\n<!-- /SECTION -->\n",
    );
    let idx = NeuronIndex::default();
    let result = render_answer_output_decision(
        &idx,
        "What kind of online group did John join?",
        &[(path, 7.0)],
        false,
        None,
    );
    assert_eq!(result, Err(AnswerAbstentionReason::Unsupported));
}

#[test]
fn render_answer_output_decision_rejects_typed_openqa_shape_leak_as_unsupported() {
    let (_dir, path) = write_validator_evidence(
        "Caroline: I still love hiking every weekend and exploring new trails.\n",
    );
    let idx = NeuronIndex::default();
    let result = render_answer_output_decision(
        &idx,
        "What personality traits might Melanie say Caroline has?",
        &[(path, 7.0)],
        false,
        None,
    );
    assert_eq!(result, Err(AnswerAbstentionReason::Unsupported));
}

#[test]
fn render_answer_output_decision_rejects_yes_no_surface_shape_leak_as_unsupported() {
    let (_dir, path) = write_validator_evidence(
        "# Session facts\n\n## answer_surface\n<!-- SECTION: answer_surface -->\n| question_pattern | answer_span | confidence |\n| --- | --- | --- |\n| move back home country soon | Sweden | 0.93 |\n<!-- /SECTION -->\n",
    );
    let idx = NeuronIndex::default();
    let result = render_answer_output_decision(
        &idx,
        "Would Caroline want to move back to her home country soon?",
        &[(path, 7.0)],
        false,
        None,
    );
    assert_eq!(result, Err(AnswerAbstentionReason::Unsupported));
}

#[test]
fn render_answer_output_decision_uses_low_form_confidence_gate_only_after_supported_answer_exists()
{
    let (_dir, path) = write_validator_evidence(
        "Maria: What kind of online group did you join?\nJohn: I joined a service-focused online group last week and it has been inspiring.\n",
    );
    let idx = NeuronIndex::default();
    let result = render_answer_output_decision(
        &idx,
        "What kind of online group did John join?",
        &[(path, 7.0)],
        false,
        Some(0.95),
    );
    assert_eq!(result, Err(AnswerAbstentionReason::LowFormConfidence));
}

#[test]
fn select_answer_recovers_typed_openqa_education_field_answer() {
    let (_dir, path) = write_validator_evidence(
        "Speaker A: Caroline\n\
Speaker B: Melanie\n\
\n\
[Session 1 — 1:56 pm on 8 May, 2023]\n\
Melanie: That's really cool. You've got guts. What now?\n\
Caroline: Gonna continue my edu and check out career options, which is pretty exciting!\n\
Melanie: Wow, Caroline! What kinda jobs are you thinkin' of? Anything that stands out?\n\
Caroline: I'm keen on counseling or working in mental health - I'd love to support those with similar issues.\n\
\n\
[Session 2 — 10:37 am on 27 June, 2023]\n\
Melanie: What motivated you to pursue counseling?\n\
Caroline: Thanks, Melanie. My own journey and the support I got made a huge difference. Now I want to help people go through it too.\n",
    );
    let answer = select_answer(
        "What fields would Caroline be likely to pursue in her educaton?",
        &single_evidence(path),
        None,
    )
    .unwrap();
    let lower = answer.to_ascii_lowercase();
    assert!(lower.contains("counseling"));
    assert!(lower.contains("mental health"));
    assert!(!lower.contains("help people go through it too"));
}
