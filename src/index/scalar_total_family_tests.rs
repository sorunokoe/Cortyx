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
fn synthetic_scalar_total_answers_sibling_count() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "siblings.conv.md",
        "User: I have a brother, which might be influencing my social circle dynamics.\n\
         User: I come from a family with 3 sisters, so I've always had a strong female presence in my life.\n",
    );
    let answer = read_answer_text(&idx, "What is the total number of siblings I have?");
    assert!(answer.contains("Answer: 4"), "{answer}");
}

#[test]
fn synthetic_scalar_total_prefers_grounded_sibling_totals() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "siblings-primary.conv.md",
        "User: I have a brother, which might be influencing my social circle dynamics.\n\
         User: I come from a family with 3 sisters, so I've always had a strong female presence in my life.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "siblings-distractor.conv.md",
        "User: I picked up some unique gifts for my siblings at the holiday market.\n\
         User: I have a brother who likes hiking.\n",
    );
    let answer = read_answer_text(&idx, "What is the total number of siblings I have?");
    assert!(answer.contains("Answer: 4"), "{answer}");
}

#[test]
fn synthetic_scalar_total_answers_platform_peak_views_total() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "views.conv.md",
        "User: My video of Luna chasing a laser pointer on TikTok has 1,456 views.\n\
         User: My tutorial on social media analytics on YouTube has been doing well, with 542 views.\n\
         User: I also gained 7 new followers on Twitter.\n",
    );
    let answer = read_answer_text(
        &idx,
        "What is the total number of views on my most popular videos on YouTube and TikTok?",
    );
    assert!(answer.contains("Answer: 1,998"), "{answer}");
}

#[test]
fn synthetic_scalar_total_answers_ready_and_commute_duration() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "commute.conv.md",
        "User: My daily commute to work takes about 30 minutes.\n\
         User: I wake up at 6:30 AM and it takes me about an hour to get ready, which includes a 20-minute meditation session and a 30-minute workout.\n",
    );
    let answer = read_answer_text(
        &idx,
        "What is the total time it takes I to get ready and commute to work?",
    );
    assert!(answer.contains("Answer: an hour and a half"), "{answer}");
}

#[test]
fn synthetic_scalar_total_prefers_same_session_duration_bundle() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "commute-primary.conv.md",
        "User: My daily commute to work takes about 30 minutes, so I want to make the most of that time.\n\
         User: I wake up at 6:30 AM and it takes me about an hour to get ready, which includes a 20-minute meditation session, a 30-minute workout, and a quick breakfast.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "commute-distractor.conv.md",
        "User: I need to fit in a 45-minute workout, meditate for 30 minutes, and get ready for work.\n\
         User: I'm trying to make the most of my morning commute.\n",
    );
    let answer = read_answer_text(
        &idx,
        "What is the total time it takes I to get ready and commute to work?",
    );
    assert!(answer.contains("Answer: an hour and a half"), "{answer}");
}
