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
fn recalls_video_title_and_link() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "video.conv.md",
        "User: any youtube video i can share with them?\n\
         Assistant: 1. \"How to Sit Properly at a Desk to Avoid Back Pain\" by the Mayo Clinic: <https://www.youtube.com/watch?v=UfOvNlX9Hh0>\n",
    );
    let answer = read_answer_text(
        &idx,
        "I wanted to follow up on our previous conversation about YouTube videos for workplace posture. Can you remind me of the Mayo Clinic video you recommended?",
    );
    assert!(
        answer.contains("How to Sit Properly at a Desk to Avoid Back Pain"),
        "{answer}"
    );
    assert!(
        answer.contains("https://www.youtube.com/watch?v=UfOvNlX9Hh0"),
        "{answer}"
    );
}

#[test]
fn recalls_website_from_assistant_resource_list() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "website.conv.md",
        "Assistant: 2. Mindful.org: This website includes guided imagery exercises that you can use for free, such as \"The Mountain Meditation\" and \"The Body Scan Meditation.\"\n",
    );
    let answer = read_answer_text(
        &idx,
        "I wanted to follow up on our previous conversation about mindfulness techniques. You mentioned some great resources for guided imagery exercises, can you remind me of the website that had free exercises like 'The Mountain Meditation' and 'The Body Scan Meditation'?",
    );
    assert!(answer.contains("Mindful.org"), "{answer}");
}

#[test]
fn prefers_domain_like_website_label_over_generic_heading() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "musictheory.conv.md",
        "Assistant: 1. MusicTheory.net: This website offers free lessons and exercises on music theory, covering topics such as rhythm, chords, and scales.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "distractor.conv.md",
        "Assistant: **Local Music Blogs or Websites**: This website has broad recommendations for learning music theory online.\n",
    );
    let answer = read_answer_text(
        &idx,
        "I'm going back to our previous conversation about music theory. You mentioned some online resources for learning music theory. Can you remind me of the website you recommended for free lessons and exercises?",
    );
    assert!(answer.contains("MusicTheory.net"), "{answer}");
}

#[test]
fn recalls_state_example_from_assistant_line() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "state.conv.md",
        "Assistant: For example, Pennsylvania requires fracking companies to monitor groundwater quality at nearby wells before drilling and for a certain period after drilling is complete.\n",
    );
    let answer = read_answer_text(
        &idx,
        "I wanted to follow up on our previous conversation about fracking in the Marcellus Shale region. You mentioned that some states require fracking companies to monitor groundwater quality at nearby wells before drilling and for a certain period after drilling is complete. Can you remind me which state you mentioned as an example that has this requirement?",
    );
    assert!(answer.contains("Pennsylvania"), "{answer}");
}

#[test]
fn recalls_specific_list_from_assistant_line() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "languages.conv.md",
        "Assistant: Learn a back-end programming language, such as Ruby, Python, or PHP. You'll need to build server-side applications.\n",
    );
    let answer = read_answer_text(
        &idx,
        "I wanted to follow up on our previous conversation about front-end and back-end development. Can you remind me of the specific back-end programming languages you recommended I learn?",
    );
    assert!(answer.contains("Ruby, Python, or PHP"), "{answer}");
}

#[test]
fn recalls_duration_from_assistant_line() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "duration.conv.md",
        "Assistant: Apply tomato juice mixed with lemon juice on your under-eye area and wash off after 10 minutes with cold water.\n",
    );
    let answer = read_answer_text(
        &idx,
        "I wanted to follow up on our previous conversation about natural remedies for dark circles under the eyes. You mentioned applying tomato juice mixed with lemon juice, how long did you say I should leave it on for?",
    );
    assert!(answer.contains("10 minutes"), "{answer}");
}
