use super::*;
use crate::neuron::{NeuronKind, NeuronMeta};
use std::fs;

fn direct_fixture_answer(file_name: &str, content: &str, question: &str) -> Option<String> {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(".cortyx").join("neurons").join(file_name);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(&path, content).unwrap();
    let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Verbatim);
    let mut idx = NeuronIndex::load_or_create(dir.path()).unwrap();
    idx.index_neuron(&path, content, &meta);
    idx.rebuild_derived();
    let answer_path = idx.derived_answer_path_for_task(question)?;
    Some(fs::read_to_string(answer_path).unwrap())
}

#[test]
fn direct_elapsed_gap_answers_ukulele_fixture_shape() {
    let answer = direct_fixture_answer(
        "ukulele-gap.conv.md",
        "[Session 1 - 1 February, 2023]\n\
         User: I'm still not sure about the keyboards, but I was thinking about my ukulele lessons. I just started taking ukulele lessons with my friend Rachel today and it's been really fun so far.\n\
         [Session 2 - 25 February, 2023]\n\
         User: I think I'll try out the Fender Acoustic 40 and Fishman Loudbox Mini. By the way, I just got back from Joe's shop. I decided to take my Taylor GS Mini to the guitar tech for servicing today - the action's been a bit high and it's been causing some discomfort in my left hand.\n",
        "How many days had passed since I started taking ukulele lessons when I decided to take my acoustic guitar to the guitar tech for servicing?",
    )
    .expect("expected direct precomputed answer");
    assert!(
        answer.contains("Answer: 24 days") || answer.contains("Answer: 25 days"),
        "{answer}"
    );
}

#[test]
fn direct_elapsed_gap_answers_flu_to_tenth_jog_fixture_shape() {
    let answer = direct_fixture_answer(
        "jog-gap.conv.md",
        "[Session 1 - 19 January, 2023]\n\
         User: I'm feeling much better now that I finally recovered from the flu today, and I was thinking about getting back into my exercise routine.\n\
         [Session 2 - 10 April, 2023]\n\
         User: I went on my 10th jog outdoors today, and it feels great to be back in shape after a harsh winter.\n",
        "How many weeks had passed since I recovered from the flu when I went on my 10th jog outdoors?",
    )
    .expect("expected direct precomputed answer");
    assert!(answer.contains("Answer: 11"), "{answer}");
}

#[test]
fn direct_elapsed_gap_answers_sculpting_fixture_shape() {
    let answer = direct_fixture_answer(
        "sculpting-gap.conv.md",
        "[Session 1 - 11 February, 2023]\n\
         User: I just started taking sculpting classes at a local art studio today, every Saturday morning from 10 am to 1 pm, and it's been a great experience so far.\n\
         [Session 2 - 4 March, 2023]\n\
         User: I actually got my own set of sculpting tools today, including a modeling tool set, a wire cutter, and a sculpting mat.\n",
        "How many weeks have I been taking sculpting classes when I invested in my own set of sculpting tools?",
    )
    .expect("expected direct precomputed answer");
    assert!(answer.contains("Answer: 3"), "{answer}");
}

#[test]
fn direct_from_now_answers_business_fixture_shape() {
    let answer = direct_fixture_answer(
        "business-gap.conv.md",
        "[Session 1 - 10 February, 2023]\n\
         User: I just launched my website and created a business plan outline, so I want to make sure my social media aligns with my overall business strategy.\n\
         [Session 2 - 1 March, 2023]\n\
         User: I just signed a contract with my first client today, and I want to make sure I'm covering all my bases for future projects.\n",
        "As of 25 March, 2023, How many days ago did I launch my website when I signed a contract with my first client?",
    )
    .expect("expected direct precomputed answer");
    assert!(
        answer.contains("Answer: 19 days ago") || answer.contains("Answer: 20 days"),
        "{answer}"
    );
}
