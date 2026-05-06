use super::*;
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

#[test]
fn synthetic_online_community_hobbies_answer_grounded_pair() {
    let dir = TempDir::new().unwrap();
    let idx = make_index(&dir);
    write_raw_neuron(
        &dir,
        "lme_0397_397_conv_0000_chunk.verbatim.md",
        "User: I'm looking for some photography inspiration and tips.\n",
    );
    write_raw_neuron(
        &dir,
        "lme_0397_397_conv_0001_chunk.verbatim.md",
        "User: I've been really enjoying editing my photos in Lightroom - the online communities I've joined have been super helpful in learning new techniques and getting feedback on my work.\n",
    );
    write_raw_neuron(
        &dir,
        "lme_0451_451_conv_0000_chunk.verbatim.md",
        "User: I'm looking for some recipe inspiration.\n",
    );
    write_raw_neuron(
        &dir,
        "lme_0451_451_conv_0001_chunk.verbatim.md",
        "User: I've already joined a few online communities related to cooking, which led me to engage in discussions about recipe techniques and share my thoughts on food-related posts.\n",
    );
    write_raw_neuron(
        &dir,
        "lme_0999_999_conv_0000_chunk.verbatim.md",
        "User: I enjoy hiking and wanted recommendations for online communities where I could ask for trail advice.\n",
    );

    let answer = read_answer_text(
        &idx,
        "What are the two hobbies that led me to join online communities?",
    );
    assert!(answer.contains("photography and cooking"), "{answer}");
}
