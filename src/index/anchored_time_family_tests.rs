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
fn synthetic_anchored_time_answers_bedtime_before_appointment() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "bedtime.conv.md",
        "User: I didn't get to bed until 2 AM last Wednesday, which made Thursday morning a struggle.\n\
         User: I had a doctor's appointment at 10 AM last Thursday, and that's when I got the results.\n",
    );
    let answer = read_answer_text(
        &idx,
        "What time did I go to bed on the day before I had a doctor's appointment?",
    );
    assert!(answer.contains("Answer: 2 AM"), "{answer}");
}

#[test]
fn synthetic_anchored_time_answers_clinic_arrival() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "clinic.conv.md",
        "User: I left home at 7 AM on Monday for my doctor's appointment.\n\
         User: It took me two hours to get to the clinic last time, so I'd like to find something closer.\n",
    );
    let answer = read_answer_text(&idx, "What time did I reach the clinic on Monday?");
    assert!(answer.contains("Answer: 9:00 AM"), "{answer}");
}
