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
fn synthetic_metric_delta_answers_mpg_drop() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "mpg.conv.md",
        "User: My car was getting 30 miles per gallon in the city a few months ago, so I'm hoping to get back to that.\n\
         User: I've been getting around 28 miles per gallon in the city lately, so I want to make sure the new filter helps with fuel efficiency.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How much more miles per gallon was my car getting a few months ago compared to now?",
    );
    assert!(answer.contains("Answer: 2"), "{answer}");
}

#[test]
fn synthetic_goal_money_delta_answers_overrun() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "goal.conv.md",
        "User: I recently participated in a charity cycling event where I initially aimed to raise $200 in donations for the local children's hospital.\n\
         User: I recently participated in a charity cycling event and raised $250 in donations, which was a great experience.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How much more money did I raise than my initial goal in the charity cycling event?",
    );
    assert!(answer.contains("Answer: $50"), "{answer}");
}

#[test]
fn synthetic_anchored_money_delta_answers_preapproval_gap() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "house.conv.md",
        "User: I recently got pre-approved for a mortgage and the lender said I can borrow up to $350,000.\n\
         User: The final sale price was $325,000, which I think is a great deal.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How much more was the pre-approval amount than the final sale price of the house?",
    );
    assert!(answer.contains("Answer: $25,000"), "{answer}");
}

#[test]
fn synthetic_anchored_money_delta_prefers_raw_same_conversation_pair() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "offer.conv.md",
        "User: I've already got pre-approved for a mortgage. The amount was $350,000.\n\
         User: I was thinking of offering around $320,000. The asking price is $335,000.\n",
    );

    let raw_chunk_zero = dir
        .path()
        .join(".cortyx")
        .join("neurons")
        .join("house_conv_0000_chunk.verbatim.md");
    let raw_chunk_one = dir
        .path()
        .join(".cortyx")
        .join("neurons")
        .join("house_conv_0001_chunk.verbatim.md");
    std::fs::create_dir_all(raw_chunk_zero.parent().unwrap()).unwrap();
    std::fs::write(
        &raw_chunk_zero,
        "User: I recently got pre-approved for a mortgage and the lender said I can borrow up to $350,000.\n",
    )
    .unwrap();
    std::fs::write(
        &raw_chunk_one,
        "User: The final sale price was $325,000, which I think is a great deal.\n",
    )
    .unwrap();

    let answer = read_answer_text(
        &idx,
        "How much more was the pre-approval amount than the final sale price of the house?",
    );
    assert!(answer.contains("Answer: $25,000"), "{answer}");
}
