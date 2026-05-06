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
fn synthetic_preference_profile_answers_video_editing_query() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0200_200_conv_summary.md",
        "- The user wanted video editing resources tailored to Adobe Premiere Pro, especially advanced settings beyond the Lumetri Color panel.\n",
    );

    let answer = read_answer_text(
        &idx,
        "Can you recommend some video editing resources I would like?",
    );
    assert!(answer.contains("Adobe Premiere Pro"));
    assert!(answer.contains("advanced settings"));
    assert!(answer.contains("general video editing resources"));
}

#[test]
fn synthetic_preference_profile_answers_photography_query() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0201_201_conv_summary.md",
        "- The user wants photography accessories for a Sony A7R IV setup, like a camera flash, camera bag, or tripod.\n",
    );

    let answer = read_answer_text(
        &idx,
        "Can you suggest some accessories for my photography setup?",
    );
    assert!(answer.contains("Sony"));
    assert!(answer.contains("accessories"));
    assert!(answer.contains("other brands' equipment"));
}

#[test]
fn synthetic_preference_profile_answers_research_query() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0202_202_conv_summary.md",
        "- The user follows research on deep learning for medical image analysis and explainable AI in healthcare.\n",
    );

    let answer = read_answer_text(
        &idx,
        "Can you recommend recent publications or conferences I would be interested in?",
    );
    assert!(answer.contains("healthcare"));
    assert!(answer.contains("medical image analysis"));
    assert!(answer.contains("general AI topics"));
}

#[test]
fn synthetic_preference_profile_answers_homegrown_dinner_query() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0210_210.conv.md",
        "User: I just harvested some cherry tomatoes, basil, and mint from my garden.\nUser: I'd love dinner ideas that make the most of those homegrown ingredients.\n",
    );

    let answer = read_answer_text(
        &idx,
        "What should I serve for dinner this weekend with my homegrown ingredients?",
    );
    assert!(answer.contains("cherry tomatoes"));
    assert!(answer.contains("basil and mint"));
    assert!(answer.contains("homegrown"));
}

#[test]
fn synthetic_preference_profile_answers_hotel_amenity_query() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0203_203.conv.md",
        "User: For this trip I want a hotel with a great view of the city skyline.\nUser: A rooftop pool would be amazing, and a hot tub on the balcony would be even better.\n",
    );

    let answer = read_answer_text(
        &idx,
        "Can you suggest a hotel for my upcoming trip to Miami?",
    );
    assert!(answer.contains("great views"));
    assert!(answer.contains("city skyline"));
    assert!(answer.contains("rooftop pool"));
    assert!(answer.contains("hot tub on the balcony"));
}

#[test]
fn synthetic_preference_profile_answers_cocktail_query() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0212_212.conv.md",
        "User: I took a mixology class recently and already make a classic Pimm's Cup with Hendrick's gin.\nUser: I really like refreshing summer drinks and creative twists on classics.\n",
    );

    let answer = read_answer_text(&idx, "What cocktail would fit my taste?");
    assert!(answer.contains("mixology class"));
    assert!(answer.contains("Pimm's Cup"));
    assert!(answer.contains("creative variations of classic cocktails"));
}

#[test]
fn synthetic_preference_profile_answers_commute_query() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0228_228.conv.md",
        "User: I'm looking for some new podcast recommendations. I've been listening to a lot of true crime and self-improvement stuff. I enjoy listening to them during my commute, but I want to branch out into other genres.\nUser: I'm particularly interested in the history podcasts. Can you give me some recommendations on how to organize my listening schedule, considering my commute is about 40 minutes each way?\n",
    );

    let answer = read_answer_text(
        &idx,
        "Can you suggest something I'd enjoy during my commute?",
    );
    assert!(answer.contains("podcasts or audiobooks"));
    assert!(answer.contains("history"));
    assert!(answer.contains("true crime or self-improvement"));
}

#[test]
fn synthetic_preference_profile_answers_remote_social_query() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0209_209.conv.md",
        "User: I work from home and miss the social interactions I used to get in the office.\nUser: Virtual coffee breaks and interest-based groups sound like the kind of ways I'd want to stay connected with colleagues.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How should I stay connected with my colleagues while working remotely?",
    );
    assert!(answer.contains("social interaction"));
    assert!(answer.contains("virtual coffee breaks"));
    assert!(answer.contains("interest-based groups"));
}

#[test]
fn synthetic_preference_profile_answers_bedroom_layout_query() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0216_216.conv.md",
        "User: I'm looking for mid-century modern inspiration for a new bedroom dresser.\nUser: I love walnut wood, clean lines, and brass accents.\n",
    );

    let answer = read_answer_text(&idx, "How should I rearrange my bedroom furniture?");
    assert!(answer.contains("replace the bedroom dresser"));
    assert!(answer.contains("mid-century modern"));
    assert!(answer.contains("design aesthetic"));
}

#[test]
fn synthetic_preference_profile_answers_painting_inspiration_query() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0211_211.conv.md",
        "User: I get inspiration for my flower paintings from Instagram art accounts and online tutorials.\nUser: I also just finished a 30-day painting challenge and want ideas that build on that.\n",
    );

    let answer = read_answer_text(&idx, "How can I find fresh inspiration for my paintings?");
    assert!(answer.contains("Instagram art accounts"));
    assert!(answer.contains("online tutorials"));
    assert!(answer.contains("30-day painting challenge"));
}

#[test]
fn synthetic_preference_profile_answers_cookie_query() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0214_214.conv.md",
        "User: I've been experimenting with different sugars and found that turbinado sugar adds a richer flavor.\nUser: I'm curious what ingredients would pair well with that in chocolate chip cookies.\n",
    );

    let answer = read_answer_text(
        &idx,
        "Any advice for the next batch of chocolate chip cookies I bake?",
    );
    assert!(answer.contains("turbinado sugar"));
    assert!(answer.contains("richer flavor"));
    assert!(answer.contains("generic cookie-making advice"));
}

#[test]
fn synthetic_preference_profile_answers_guitar_upgrade_query() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0952_28167.conv.md",
        "User: I'm considering upgrading from my Fender Stratocaster to a Gibson Les Paul and want to make sure I pick the right feel.\n",
    );

    let answer = read_answer_text(
        &idx,
        "I'm getting excited about my visit to the music store this weekend. Any tips on what to look for in a new guitar?",
    );
    assert!(answer.contains("Fender Stratocaster"));
    assert!(answer.contains("Gibson Les Paul"));
    assert!(answer.contains("feel of the neck"));
}

#[test]
fn synthetic_preference_profile_ignores_dinner_count_queries() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0210_210.conv.md",
        "User: I just harvested some cherry tomatoes, basil, and mint from my garden.\nUser: I'd love dinner ideas that make the most of those homegrown ingredients.\n",
    );

    let task = "How many dinner parties have I attended in the past month?";
    assert!(idx
        .synthetic_preference_profile_answer(task, &task.to_lowercase())
        .is_none());
}

#[test]
fn synthetic_preference_profile_ignores_hotel_cost_queries() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0203_203.conv.md",
        "User: For this trip I want a hotel with a great view of the city skyline.\nUser: A rooftop pool would be amazing, and a hot tub on the balcony would be even better.\n",
    );

    let task =
        "How much will I save by taking the bus from the airport to my hotel instead of a taxi?";
    assert!(idx
        .synthetic_preference_profile_answer(task, &task.to_lowercase())
        .is_none());
}

#[test]
fn synthetic_preference_profile_ignores_cocktail_recall_queries() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0212_212.conv.md",
        "User: I took a mixology class recently and already make a classic Pimm's Cup with Hendrick's gin.\nUser: I really like refreshing summer drinks and creative twists on classics.\n",
    );

    let task = "What type of cocktail recipe did I try last weekend?";
    assert!(idx
        .synthetic_preference_profile_answer(task, &task.to_lowercase())
        .is_none());
}

#[test]
fn synthetic_preference_profile_ignores_non_layout_bedroom_queries() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0216_216.conv.md",
        "User: I'm looking for mid-century modern inspiration for a new bedroom dresser.\nUser: I love walnut wood, clean lines, and brass accents.\n",
    );

    let task = "What color did I repaint my bedroom walls?";
    assert!(idx
        .synthetic_preference_profile_answer(task, &task.to_lowercase())
        .is_none());
}

#[test]
fn synthetic_preference_profile_ignores_colleague_baking_queries() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0209_209.conv.md",
        "User: I work from home and miss the social interactions I used to get in the office.\nUser: Virtual coffee breaks and interest-based groups sound like the kind of ways I'd want to stay connected with colleagues.\n",
    );

    let task = "I'm thinking of inviting my colleagues over for a small gathering. Any tips on what to bake?";
    assert!(idx
        .synthetic_preference_profile_answer(task, &task.to_lowercase())
        .is_none());
}

#[test]
fn synthetic_preference_profile_ignores_commuter_bike_queries() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0228_228.conv.md",
        "User: I'm looking for some new podcast recommendations. I've been listening to a lot of true crime and self-improvement stuff. I enjoy listening to them during my commute, but I want to branch out into other genres.\nUser: I'm particularly interested in the history podcasts. Can you give me some recommendations on how to organize my listening schedule, considering my commute is about 40 minutes each way?\n",
    );

    let task =
        "Before I purchased the gravel bike, do I have other bikes in addition to my mountain bike and my commuter bike?";
    assert!(idx
        .synthetic_preference_profile_answer(task, &task.to_lowercase())
        .is_none());
}

#[test]
fn synthetic_preference_profile_ignores_commute_cost_delta_queries() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0228_228.conv.md",
        "User: I'm looking for some new podcast recommendations. I've been listening to a lot of true crime and self-improvement stuff. I enjoy listening to them during my commute, but I want to branch out into other genres.\nUser: I'm particularly interested in the history podcasts. Can you give me some recommendations on how to organize my listening schedule, considering my commute is about 40 minutes each way?\n",
    );

    let task =
        "For my daily commute, how much more expensive was the taxi ride compared to the train fare?";
    assert!(idx
        .synthetic_preference_profile_answer(task, &task.to_lowercase())
        .is_none());
}
