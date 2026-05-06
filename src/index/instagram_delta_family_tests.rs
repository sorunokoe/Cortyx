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
fn synthetic_instagram_delta_answers_split_window_growth() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "growth.conv.md",
        "User: I started the year with 250 followers on Instagram, by the way.\n\
         User: After two weeks of posting regularly, I had around 350 followers on Instagram.\n",
    );
    let answer = read_answer_text(
        &idx,
        "What was the approximate increase in Instagram followers I experienced in two weeks?",
    );
    assert!(answer.contains("Answer: 100"), "{answer}");
}

#[test]
fn synthetic_instagram_delta_answers_direct_from_to_growth() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "growth_summary.conv.md",
        "- My Instagram account grew from 500 followers to 600 followers over two weeks.\n",
    );
    let answer = read_answer_text(
        &idx,
        "What was the approximate increase in Instagram followers I experienced in two weeks?",
    );
    assert!(answer.contains("Answer: 100"), "{answer}");
}

#[test]
fn synthetic_instagram_delta_ignores_large_wrong_window_noise() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "target.conv.md",
        "User: I started the year with 250 followers on Instagram.\n\
         User: After two weeks of posting regularly, I had around 350 followers on Instagram.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "noise.conv.md",
        "User: My Instagram account grew from 1000 followers to 100000 followers over six months.\n",
    );
    let answer = read_answer_text(
        &idx,
        "What was the approximate increase in Instagram followers I experienced in two weeks?",
    );
    assert!(answer.contains("Answer: 100"), "{answer}");
}
