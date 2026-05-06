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
fn synthetic_preference_profile_answers_slow_cooker_advice_query() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0901_slow_cooker.conv.md",
        "User: I recently figured out how to use the slow cooker and made a delicious beef stew.\nUser: Do you have any recipes for making yogurt in a slow cooker?\n",
    );

    let answer = read_answer_text(
        &idx,
        "I've been struggling with my slow cooker recipes. Any advice on getting better results?",
    );
    assert!(answer.contains("beef stew"));
    assert!(answer.contains("yogurt in the slow cooker"));
}

#[test]
fn synthetic_preference_profile_answers_baking_gathering_query() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0902_baking.conv.md",
        "User: I'm thinking of making a cake for my friend's birthday, like my lemon poppyseed cake that I made for a colleague's going-away party.\nUser: I've made a lemon poppyseed cake before, and it was a hit.\n",
    );

    let answer = read_answer_text(
        &idx,
        "I'm thinking of inviting my colleagues over for a small gathering. Any tips on what to bake?",
    );
    assert!(answer.contains("lemon poppyseed cake"));
    assert!(answer.contains("build upon their existing baking experience"));
}

#[test]
fn synthetic_preference_profile_answers_tokyo_transit_advice_query() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0903_tokyo.conv.md",
        "User: I'm also planning to take a guided tour to Nikko National Park. I have downloaded the TripIt app to stay organized, but I'm still nervous about the trip.\nUser: I'm also planning to visit the Tsukiji Fish Market using my Suica card.\n",
    );

    let answer = read_answer_text(
        &idx,
        "I'm a bit anxious about getting around Tokyo. Do you have any helpful tips?",
    );
    assert!(answer.contains("Suica card"));
    assert!(answer.contains("TripIt app"));
    assert!(answer.contains("Tokyo's public transportation"));
}

#[test]
fn synthetic_preference_profile_answers_theme_park_query() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0904_theme_park.conv.md",
        "User: I recently visited Disneyland, Knott's Berry Farm, Six Flags Magic Mountain, and Universal Studios Hollywood. I'm especially interested in thrill rides, unique food experiences, and nighttime shows.\n",
    );

    let answer = read_answer_text(
        &idx,
        "I am planning another theme park weekend; do you have any suggestions?",
    );
    assert!(answer.contains("Disneyland"));
    assert!(answer.contains("thrill rides"));
    assert!(answer.contains("nighttime shows"));
}

#[test]
fn synthetic_preference_profile_answers_kitchen_clean_query() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0905_kitchen.conv.md",
        "User: I recently bought a new utensil holder to keep countertops clutter-free.\nUser: I noticed some scratches on my granite countertop near the sink.\n",
    );

    let answer = read_answer_text(
        &idx,
        "My kitchen's becoming a bit of a mess again. Any tips for keeping it clean?",
    );
    assert!(answer.contains("utensil holder"));
    assert!(answer.contains("granite surface"));
    assert!(answer.contains("current tools and setup"));
}

#[test]
fn synthetic_preference_profile_answers_phone_battery_query() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0906_phone.conv.md",
        "User: I'm looking for some advice on the best way to organize my tech accessories, like my new portable power bank and wireless charging pad, when I'm traveling.\n",
    );

    let answer = read_answer_text(
        &idx,
        "I've been having trouble with the battery life on my phone lately. Any tips?",
    );
    assert!(answer.contains("portable power bank"));
    assert!(answer.contains("battery-saving features"));
}

#[test]
fn synthetic_preference_profile_answers_netflix_storytelling_query() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0907_entertainment.conv.md",
        "User: As an aspiring stand-up comedian, I'm looking for some stand-up comedy specials on Netflix with strong storytelling abilities like John Mulaney's 'Kid Gorgeous'.\n",
    );

    let answer = read_answer_text(
        &idx,
        "Can you recommend a show or movie for me to watch tonight?",
    );
    assert!(answer.contains("stand-up comedy specials"));
    assert!(answer.contains("Netflix"));
    assert!(answer.contains("storytelling"));
}

#[test]
fn synthetic_preference_profile_answers_reunion_query() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0908_reunion.conv.md",
        "User: I still remember the happy high school experiences such as being part of the debate team and taking advanced placement courses in economics.\nUser: A lot of my old high school friends plan to work after they graduate from university.\n",
    );

    let answer = read_answer_text(
        &idx,
        "I've been feeling nostalgic lately. Do you think it would be a good idea to attend my high school reunion?",
    );
    assert!(answer.contains("debate team"));
    assert!(answer.contains("economics"));
    assert!(answer.contains("reconnecting with old friends"));
}

#[test]
fn synthetic_preference_profile_answers_nas_decision_query() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0909_nas.conv.md",
        "User: I'm having some issues with my home network's storage capacity and was thinking of getting a NAS device.\nUser: I'm already backing up my files to an external hard drive, but I think a NAS would be more convenient.\n",
    );

    let answer = read_answer_text(
        &idx,
        "I'm trying to decide whether to buy a NAS device now or wait. What do you think?",
    );
    assert!(answer.contains("storage capacity issues"));
    assert!(answer.contains("external hard drives"));
}

#[test]
fn synthetic_preference_profile_ignores_baking_fact_recall_query() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0910_bake_fact.conv.md",
        "User: I'm thinking of making a cake for my friend's birthday, like my lemon poppyseed cake that I made for a colleague's going-away party.\n",
    );

    let task = "What did I bake for my uncle's birthday party?";
    assert!(idx
        .synthetic_preference_profile_answer(task, &task.to_lowercase())
        .is_none());
}

#[test]
fn synthetic_preference_profile_ignores_creamer_location_recall_query() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0911_creamer_fact.conv.md",
        "User: I'm trying to reduce my sugar intake and save money, so I've started making my own flavored creamer with almond milk, vanilla extract, and honey.\n",
    );

    let task = "Where did I redeem a $5 coupon on coffee creamer?";
    assert!(idx
        .synthetic_preference_profile_answer(task, &task.to_lowercase())
        .is_none());
}

#[test]
fn synthetic_preference_profile_ignores_nasi_recall_query() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0912_nasi_fact.conv.md",
        "Assistant: The restaurant in Cihampelas Walk that serves great Nasi Goreng is Miss Bee Providore.\n",
    );

    let task = "Can you remind me of the name of that restaurant in Cihampelas Walk that serves a great Nasi Goreng?";
    assert!(idx
        .synthetic_preference_profile_answer(task, &task.to_lowercase())
        .is_none());
}

#[test]
fn synthetic_preference_profile_ignores_temporal_living_room_query() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0913_living_room_fact.conv.md",
        "User: What are some simple ways to keep my living room dust-free, especially with a cat that sheds a lot?\n",
    );

    let task =
        "How long had I been using the new area rug when I rearranged my living room furniture?";
    assert!(idx
        .synthetic_preference_profile_answer(task, &task.to_lowercase())
        .is_none());
}
