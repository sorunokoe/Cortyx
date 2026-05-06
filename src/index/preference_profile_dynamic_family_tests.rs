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
fn synthetic_preference_profile_answers_destination_revisit_query() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0224_224.conv.md",
        "User: During my previous visit to Denver, where I had a great time meeting Brandon Flowers after The Killers' concert, I realized how much I love the city's music scene. Are there any good music venues or festivals in Denver that I shouldn't miss?\n",
    );

    let answer = read_answer_text(
        &idx,
        "I'm planning a trip to Denver soon. Any suggestions on what to do there?",
    );
    assert!(answer.contains("Denver"));
    assert!(answer.contains("Brandon Flowers"));
    assert!(answer.contains("music scene"));
    assert!(answer.contains("generic tourist recommendations"));
}

#[test]
fn synthetic_preference_profile_answers_documentary_taste_query() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0225_225.conv.md",
        "User: I've been watching a lot of documentaries lately, especially on Netflix. Can you recommend some more documentary series similar to \"Our Planet\", \"Free Solo\", and \"Tiger King\", which I just finished?\n",
    );

    let answer = read_answer_text(
        &idx,
        "I've got some free time tonight, any documentary recommendations?",
    );
    assert!(answer.contains("Our Planet"));
    assert!(answer.contains("Free Solo"));
    assert!(answer.contains("Tiger King"));
    assert!(answer.contains("similar in style and theme"));
}

#[test]
fn synthetic_preference_profile_answers_phone_accessory_query() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0227_227.conv.md",
        "User: I'm looking for a new screen protector to replace the cracked one on my iPhone 13 Pro.\nUser: I'm also looking for a phone wallet case for my iPhone 13 Pro.\nUser: I'm also interested in getting a portable power bank to charge my phone on-the-go.\nUser: I think I'll consider getting a wireless charging power bank so I can place my iPhone 13 Pro on it.\n",
    );

    let answer = read_answer_text(
        &idx,
        "Can you suggest some useful accessories for my phone?",
    );
    assert!(answer.contains("iPhone 13 Pro"));
    assert!(answer.contains("screen protectors"));
    assert!(answer.contains("wallet cases"));
    assert!(answer.contains("power banks"));
}

#[test]
fn synthetic_preference_profile_still_ignores_unrelated_tokyo_transit_query() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0888_tokyo.conv.md",
        "User: I'm planning a Tokyo trip and want to be confident getting around from Shinjuku.\nUser: I just got a Suica card and downloaded the TripIt app, but I'm still nervous about the transit side of the trip.\n",
    );

    let task = "Can you suggest the best way for me to get to the tour meeting point in Tokyo?";
    assert!(idx
        .synthetic_preference_profile_answer(task, &task.to_lowercase())
        .is_none());
}

#[test]
fn synthetic_preference_profile_does_not_hijack_named_recall_queries() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0224_224.conv.md",
        "User: During my previous visit to Denver, where I had a great time meeting Brandon Flowers after The Killers' concert, I realized how much I love the city's music scene. Are there any good music venues or festivals in Denver that I shouldn't miss?\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "amsterdam-hostel.conv.md",
        "Assistant: 1. Stayokay Amsterdam Vondelpark: Located near Vondelpark.\n\
         2. International Budget Hostel: This hostel is situated near the famous Red Light District and offers affordable dormitory-style rooms.\n",
    );

    let answer = read_answer_text(
        &idx,
        "I'm planning my trip to Amsterdam again and I was wondering, what was the name of that hostel near the Red Light District that you recommended last time?",
    );
    assert!(answer.contains("International Budget Hostel"), "{answer}");
    assert!(
        !answer.contains("generic tourist recommendations"),
        "{answer}"
    );
}
