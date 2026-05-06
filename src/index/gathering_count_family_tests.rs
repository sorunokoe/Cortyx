use super::gathering_count_extractors::parse_dinner_party_count_query;
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
fn parses_dinner_party_count_query() {
    assert!(parse_dinner_party_count_query(
        "How many dinner parties have I attended in the past month?",
        "how many dinner parties have i attended in the past month?",
    )
    .is_some());
}

#[test]
fn synthetic_dinner_party_count_answers_distinct_recent_hosts() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "dinner.conv.md",
        "User: I love these ideas! I'm definitely going to consider the Global Street Food theme. By the way, I've also had a great experience with a BBQ theme, like the one we had at Mike's place two weeks ago, where we watched a football game together.\n\
         User: That's great! I think I'll have a mix of grilled and non-grilled dishes to cater to different tastes. By the way, I've also had experience with dinner parties that are more low-key, like the ones we had at Alex's place yesterday, where we had a potluck and tried out different cuisines from around the world, and also at Mike's place, where we had a BBQ and watched a football game together.\n\
         User: I'm looking for some Italian recipe ideas for a dinner party I'm hosting soon. I attended a lovely Italian feast at Sarah's place last week, and it inspired me to try out some new dishes.\n\
         User: That's a great list of recipes! I've had a lovely experience at Sarah's place recently, where we played board games until late into the night after the Italian feast.\n",
    );

    let task = "How many dinner parties have I attended in the past month?";
    idx.synthetic_dinner_party_count_answer(task, &task.to_ascii_lowercase())
        .expect("expected dinner party count answer");
    let answer = read_answer_text(&idx, task);
    assert!(answer.contains("Answer: three"), "{answer}");
}
