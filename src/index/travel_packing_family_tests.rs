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

#[test]
fn synthetic_packed_shoes_percentage_renders_percent_answer() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "trip.conv.md",
        "User: For my last trip, I packed 5 pairs of shoes because I couldn't decide what to bring.\n\
         User: I ended up only wearing 2 pairs of shoes on the trip.\n",
    );

    let task = "What percentage of packed shoes did I wear on my last trip?";
    let path = idx
        .synthetic_travel_packing_answer(task, &task.to_ascii_lowercase())
        .expect("expected packed-shoes answer");
    let answer = std::fs::read_to_string(path).unwrap();
    assert!(answer.contains("Answer: 40%"), "{answer}");
}
