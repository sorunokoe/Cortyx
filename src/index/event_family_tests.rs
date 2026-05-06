use super::count_support::SignatureDetail;
use super::event_extractors::{
    extract_rollercoaster_event_quantities, extract_wedding_attendance_details,
};
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
fn wedding_extractor_keeps_named_couples() {
    let details = extract_wedding_attendance_details(
        "User: My friend Emily finally got to tie the knot with her partner Sarah, and it was amazing.",
    );
    assert_eq!(
        details,
        vec![SignatureDetail::new("emily", "Emily and Sarah")]
    );
}

#[test]
fn rollercoaster_extractor_counts_named_lists_and_repeat_counts() {
    let listed = extract_rollercoaster_event_quantities(
        "User: I rode the Mako, Kraken, and Manta rollercoasters all in one night at SeaWorld in July.",
        "user: i rode the mako, kraken, and manta rollercoasters all in one night at seaworld in july.",
        (7, 10),
    );
    assert_eq!(listed[0].1, 3);

    let repeated = extract_rollercoaster_event_quantities(
        "User: I rode Space Mountain: Ghost Galaxy three times at Disneyland on September 24th.",
        "user: i rode space mountain: ghost galaxy three times at disneyland on september 24th.",
        (7, 10),
    );
    assert_eq!(repeated[0].1, 3);
}

#[test]
fn synthetic_attended_wedding_count_answers_named_events() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "weddings.conv.md",
        "User: My cousin Rachel's wedding at the vineyard was just perfect.\n\
         User: My friend Emily finally got to tie the knot with her partner Sarah, and it was beautiful.\n\
         User: I just got back from a friend's wedding last weekend, and the bride, Jen, looked stunning in her dress, and her husband, Tom, was clearly smitten.\n",
    );

    let answer = read_answer_text(&idx, "How many weddings have I attended in this year?");
    assert!(answer.contains("Answer: I attended three weddings."));
    assert!(answer.contains("The couples were Rachel, Emily and Sarah, and Jen and Tom."));
}

#[test]
fn synthetic_rollercoaster_ride_count_sums_attended_event_quantities() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "rollercoasters.conv.md",
        "User: I rode the Mako, Kraken, and Manta rollercoasters all in one night at SeaWorld San Diego in July.\n\
         User: I rode Space Mountain: Ghost Galaxy three times at Disneyland on September 24th during Mickey's Halloween Party.\n\
         User: I rode the Xcelerator rollercoaster at Knott's Berry Farm on October 8th and it's still one of my favorite thrill rides.\n\
         User: I rode the Revenge of the Mummy rollercoaster three times in a row at Universal Studios Hollywood on October 15th, and it was such a thrill!\n",
    );

    let answer = read_answer_text(
        &idx,
        "How many times did I ride rollercoasters across all the events I attended from July to October?",
    );
    assert!(answer.contains("Answer: 10"));
}

#[test]
fn synthetic_education_completion_age_delta_matches_same_persona_age() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "marketing_degree.conv.md",
        "- I've been working in digital marketing for a while now.\n\
         - I have a Bachelor's degree in Business Administration with a concentration in Marketing from the University of California, Berkeley, which I completed at the age of 25.\n\
         - I'm considering transitioning into a more specialized role, such as a Content Marketing Strategist.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "marketing_current_age.conv.md",
        "- I'm interested in pursuing a master's degree in marketing, and my career goal is to move into a leadership role.\n\
         - Since I'm interested in marketing, I was wondering if you could provide some insights on the current trends in digital marketing.\n\
         - By the way, I'm currently 32 years old, so I want to make sure I'm using products that are suitable for my skin at this stage.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How many years older am I than when I graduated from college?",
    );
    assert!(answer.contains("Answer: 7"));
}
