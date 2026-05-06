use super::*;
use crate::index::NeuronIndex;
use crate::neuron::{neuron_dir, NeuronMeta};
use std::{fs, path::PathBuf};

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

fn write_indexed_temporal_sessions(
    files: &[(&str, &str, &str)],
) -> (tempfile::TempDir, NeuronIndex, Vec<PathBuf>) {
    let dir = tempfile::tempdir().unwrap();
    let ndir = neuron_dir(dir.path());
    fs::create_dir_all(&ndir).unwrap();
    let mut idx = NeuronIndex::load_or_create(dir.path()).unwrap();
    let mut paths = Vec::new();
    for (name, content, timestamp) in files {
        let path = ndir.join(name);
        fs::write(&path, content).unwrap();
        let meta = NeuronMeta::new_verbatim_chunk(
            &path,
            Some("User".to_string()),
            content,
            Some((*timestamp).to_string()),
            None,
        );
        idx.index_neuron(&path, content, &meta);
        paths.push(path);
    }
    idx.rebuild_derived_pub();
    (dir, idx, paths)
}

#[test]
fn select_answer_uses_latest_grounded_current_anchor_for_elapsed_query() {
    let (_dir, evidence) = write_temporal_evidence_files(&[
        (
            "session_0001_chunk.verbatim.md",
            "[Session 1 - 9:00 am on 1 May, 2023]\n\
User: I attended a networking event today and met a lot of founders.\n",
            9.0,
        ),
        (
            "session_0002_chunk.verbatim.md",
            "[Session 2 - 9:00 am on 27 May, 2023]\n\
User: Today I'm planning my schedule for next week and catching up on chores.\n",
            8.5,
        ),
    ]);
    let answer = select_answer(
        "How many days ago did I attend a networking event?",
        &evidence,
        None,
    )
    .unwrap();
    assert_eq!(answer, "26 days ago");
}

#[test]
fn select_answer_rejects_elapsed_query_without_grounded_current_anchor() {
    let (_dir, evidence) = write_temporal_evidence_files(&[(
        "session_chunk.verbatim.md",
        "User: I attended a networking event today and met a lot of founders.\n",
        9.0,
    )]);
    let answer = select_answer(
        "How many days ago did I attend a networking event?",
        &evidence,
        None,
    );
    assert!(answer.is_none(), "unexpected answer: {answer:?}");
}

#[test]
fn render_answer_output_pulls_recent_current_anchor_from_index_for_elapsed_query() {
    let (_dir, idx, paths) = write_indexed_temporal_sessions(&[
        (
            "session_0001_chunk.verbatim.md",
            "[Session 1 - 9:00 am on 1 May, 2023]\n\
User: I attended a networking event today and met a lot of founders.\n",
            "2023-05-01T09:00:00Z",
        ),
        (
            "session_0002_chunk.verbatim.md",
            "[Session 2 - 9:00 am on 27 May, 2023]\n\
User: Today I'm planning my schedule for next week and catching up on chores.\n",
            "2023-05-27T09:00:00Z",
        ),
    ]);
    let answer = render_answer_output(
        &idx,
        "How many days ago did I attend a networking event?",
        &[(paths[0].clone(), 9.0)],
        false,
        None,
    )
    .unwrap();
    assert_eq!(answer.trim(), "26 days ago");
}

#[test]
fn elapsed_queries_defer_precomputed_answers() {
    assert!(should_defer_precomputed_answer(
        "How many days ago did I attend a networking event?",
        Path::new("_answer_temporal_current_anchor.md"),
    ));
}
