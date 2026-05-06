use super::*;
use crate::neuron::{NeuronKind, NeuronMeta};
use tempfile::TempDir;

fn make_index(dir: &TempDir) -> NeuronIndex {
    NeuronIndex::load_or_create(dir.path()).unwrap()
}

fn read_answer_text(idx: &NeuronIndex, task: &str) -> String {
    let path = idx
        .derived_answer_path_for_task(task)
        .expect("expected synthetic answer");
    std::fs::read_to_string(path).unwrap()
}

fn write_raw_neuron(dir: &TempDir, file_name: &str, content: &str) {
    let path = dir.path().join(".cortyx").join("neurons").join(file_name);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, content).unwrap();
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
fn synthetic_academic_gpa_average_prefers_grouped_raw_conversation() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "major.conv.md",
        "User: I recently completed my Master's degree in Data Science from Illinois.\n",
    );
    write_raw_neuron(
        &dir,
        "education_conv_0000_chunk.verbatim.md",
        "User: I recently completed my Master's degree in Data Science, where I maintained a GPA of 3.8 out of 4.0.\n",
    );
    write_raw_neuron(
        &dir,
        "education_conv_0001_chunk.verbatim.md",
        "User: As we discussed earlier, I graduated in Computer Science with an overall percentage of 83%, equivalent to a GPA of 3.86 out of 4.0.\n",
    );

    let answer = read_answer_text(
        &idx,
        "What is the average GPA of my undergraduate and graduate studies?",
    );
    assert!(answer.contains("Answer: 3.83"), "{answer}");
}

#[test]
fn synthetic_family_age_average_answers_across_relatives() {
    let dir = TempDir::new().unwrap();
    let idx = make_index(&dir);
    write_raw_neuron(
        &dir,
        "family_conv_0000_chunk.verbatim.md",
        "User: I just turned 32 on February 12th, so I'm feeling motivated to take care of myself now.\n",
    );
    write_raw_neuron(
        &dir,
        "family_conv_0001_chunk.verbatim.md",
        "User: My parents are getting older too - my mom is 55 and my dad is 58, so I'm trying to set a good example for them as well.\n",
    );
    write_raw_neuron(
        &dir,
        "family_conv_0002_chunk.verbatim.md",
        "User: My grandma is 75 and my grandpa is 78, and seeing them slow down has made me think about my own future.\n",
    );

    let answer = read_answer_text(
        &idx,
        "What is the average age of me, my parents, and my grandparents?",
    );
    assert!(answer.contains("Answer: 59.6"), "{answer}");
}
