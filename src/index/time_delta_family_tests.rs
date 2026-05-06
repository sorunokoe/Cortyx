use super::time_delta_extractors::format_minutes_delta;
use super::*;
use crate::neuron::{NeuronKind, NeuronMeta};
use tempfile::TempDir;

fn make_index(dir: &TempDir) -> NeuronIndex {
    NeuronIndex::load_or_create(dir.path()).unwrap()
}

fn index_verbatim_neuron(idx: &mut NeuronIndex, dir: &TempDir, file_name: &str, content: &str) {
    let path = dir.path().join(".cortyx").join("neurons").join(file_name);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, content).unwrap();
    let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Verbatim);
    idx.index_neuron(&path, content, &meta);
    idx.rebuild_derived();
}

fn read_answer_text(idx: &NeuronIndex, task: &str) -> String {
    let path = idx
        .derived_answer_path_for_task(task)
        .expect("expected synthetic answer");
    std::fs::read_to_string(path).unwrap()
}

#[test]
fn format_minutes_delta_renders_singular_and_plural() {
    assert_eq!(format_minutes_delta(1), "1 minute");
    assert_eq!(format_minutes_delta(30), "30 minutes");
}

#[test]
fn synthetic_wakeup_delta_compares_friday_and_weekday_times() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "wakeups.conv.md",
        "User: On Fridays, I like to get a head start, so I wake up at 6:00 AM.\n\
         User: I usually do them right after waking up at 6:30 AM on weekdays.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How much earlier do I wake up on Fridays compared to other weekdays?",
    );
    assert!(answer.contains("Answer: 30 minutes"));
}

#[test]
fn synthetic_performance_delta_compares_current_and_previous_run_times() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "runs.conv.md",
        "User: I've done a 5K run last year, but it took me 45 minutes to complete.\n\
         User: I just got back into running and recently finished a 5K in 35 minutes.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How much faster did I finish the 5K run compared to my previous year's time?",
    );
    assert!(answer.contains("Answer: 10 minutes"));
}

#[test]
fn synthetic_performance_delta_falls_back_across_sessions() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "runs_previous.conv.md",
        "User: I've done a 5K run last year, but it took me 45 minutes to complete.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "runs_current.conv.md",
        "User: I just got back into running and recently finished a 5K in 35 minutes.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How much faster did I finish the 5K run compared to my previous year's time?",
    );
    assert!(answer.contains("Answer: 10 minutes"));
}

#[test]
fn synthetic_performance_delta_skips_non_improving_pair_before_direct_scan() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "runs_previous.conv.md",
        "User: I've done a 5K run last year, but it took me 30 minutes to complete.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "runs_current.conv.md",
        "User: I just got back into running and recently finished a 5K in 35 minutes.\n",
    );

    let raw_only_path = dir
        .path()
        .join(".cortyx")
        .join("neurons")
        .join("runs_direct_conv_0000_chunk.verbatim.md");
    std::fs::create_dir_all(raw_only_path.parent().unwrap()).unwrap();
    std::fs::write(
        &raw_only_path,
        "User: I've done a 5K run last year, but it took me 45 minutes to complete.\n\
         User: I just got back into running and recently finished a 5K in 35 minutes.\n",
    )
    .unwrap();

    let answer = read_answer_text(
        &idx,
        "How much faster did I finish the 5K run compared to my previous year's time?",
    );
    assert!(answer.contains("Answer: 10 minutes"), "{answer}");
}
