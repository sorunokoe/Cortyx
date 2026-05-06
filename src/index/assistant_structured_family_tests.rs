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
fn recalls_two_factor_methods_from_assistant_line() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "twofactor.conv.md",
        "Assistant: 3. Two-factor authentication: Requiring two-factor authentication, such as biometric authentication or one-time passwords (OTP), enhances security by ensuring that only authorized users can access sensitive data.\n",
    );
    let answer = read_answer_text(
        &idx,
        "I was thinking about our previous conversation about data privacy and security. You mentioned that companies use two-factor authentication to enhance security. Can you remind me what kind of two-factor authentication methods you were referring to?",
    );
    assert!(
        answer.contains("biometric authentication or one-time passwords (OTP)"),
        "{answer}"
    );
}

#[test]
fn recalls_described_brand_from_assistant_list_line() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "brand.conv.md",
        "Assistant: 5. Veja - This French brand produces eco-friendly sneakers using organic cotton, recycled plastic bottles, and wild rubber sourced from the Amazon rainforest.\n",
    );
    let answer = read_answer_text(
        &idx,
        "I was going through our previous conversation about high-end fashion brands, and I was wondering if you could remind me of the brand that uses wild rubber sourced from the Amazon rainforest?",
    );
    assert!(answer.contains("Answer: Veja"), "{answer}");
}

#[test]
fn recalls_described_hiking_trail_from_assistant_sentence() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "trail.conv.md",
        "Assistant: Certainly! One of the best hiking trails with breathtaking views in the Natural Park of the Moncayo mountain in Aragón is the GR-90. This trail takes you through the park's most stunning landscapes and offers panoramic views of the surrounding mountainside.\n",
    );
    let answer = read_answer_text(
        &idx,
        "I'm planning to go back to the Natural Park of Moncayo mountain in Aragón and I was wondering, what was the name of that hiking trail you recommended that takes you through the park's most stunning landscapes and offers panoramic views of the surrounding mountainside?",
    );
    assert!(answer.contains("GR-90"), "{answer}");
    assert!(answer.contains("trail"), "{answer}");
}

#[test]
fn recalls_mentioned_cartoon_from_assistant_example() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "cartoon.conv.md",
        "Assistant: One example is the popular Soviet cartoon, \"Nu, pogodi!\" which mocked Western culture and portrayed the Soviet Union as superior.\n",
    );
    let answer = read_answer_text(
        &idx,
        "I was going through our previous conversation about political propaganda and humor, and I was wondering if you could remind me of that Soviet cartoon you mentioned that mocked Western culture?",
    );
    assert!(answer.contains("Nu, pogodi!"), "{answer}");
}

#[test]
fn recalls_final_name_after_user_acceptance() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "fissionator.conv.md",
        "Assistant: As for a name, how about \"Radialisk\"? It's a play on words between \"radial\" and \"hydralisk\".\n\
         User: Hmm, not sure about a name referencing other games. Any other name ideas?\n\
         Assistant: Sure, here are some potential one-word names for the Radiation Amplified:\n\
         1. Radik\n\
         2. Irradon\n\
         3. Nucleus\n\
         4. Fissionator\n\
         User: Fissionator is a REALLY cool one, especially if it's given a more clunky, mechanical-looking design.\n\
         Assistant: That's a great idea! The protective gear could also add an extra layer of difficulty to defeating the Fissionator.\n\
         User: I love the idea of the protective suit being melted into the Fissionators host! It adds to the horror of the design.\n\
         Assistant: Yes, exactly! It adds a layer of intrigue to the Fissionator's design and backstory.\n",
    );
    let answer = read_answer_text(
        &idx,
        "I was thinking back to our previous conversation about the Radiation Amplified zombie, and I was wondering if you remembered what we finally decided to name it?",
    );
    assert!(answer.contains("Fissionator."), "{answer}");
    assert!(!answer.contains("Radialisk"), "{answer}");
}
