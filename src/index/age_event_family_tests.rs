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
        .synthetic_age_event_answer(task, &task.to_ascii_lowercase())
        .expect("expected age-event answer");
    std::fs::read_to_string(path).unwrap()
}

#[test]
fn synthetic_age_event_answer_handles_grandma_age_difference() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "grandma.conv.md",
        "User: My grandma, who turned 75 recently, is still incredibly sharp.\n\
         User: I just turned 32 last month and it got me thinking about ageism.\n",
    );

    let answer = read_answer_text(&idx, "How many years older is my grandma than me?");
    assert!(answer.contains("Answer: 43"), "{answer}");
}

#[test]
fn synthetic_age_event_answer_handles_age_when_alex_was_born() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "alex.conv.md",
        "User: It's crazy that my intern Alex is just 21 and I'm already mentoring him.\n\
         User: I just turned 32 last month, so it feels like a big responsibility.\n",
    );

    let answer = read_answer_text(&idx, "How old was I when Alex was born?");
    assert!(answer.contains("Answer: 11"), "{answer}");
}

#[test]
fn synthetic_age_event_answer_handles_age_when_rachel_gets_married() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "rachel.conv.md",
        "User: My friend Rachel's getting married next year, and it's got me thinking about my own life goals.\n\
         Assistant: Current age: you're 32, so we'll use that as our starting point.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How many years will I be when my friend Rachel gets married?",
    );
    assert!(answer.contains("Answer: 33"), "{answer}");
}

#[test]
fn synthetic_age_event_answer_handles_rachel_marriage_abstention() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "my_wedding.verbatim.md",
        "User: I'm getting married soon and I'm looking for some wedding venue ideas.\n",
    );

    let answer = read_answer_text(&idx, "How old will Rachel be when I get married?");
    assert!(
        answer.contains("The information provided is not enough."),
        "{answer}"
    );
    assert!(answer.contains("Rachel"), "{answer}");
}
