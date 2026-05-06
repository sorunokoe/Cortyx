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
fn recalls_recommended_name_from_assistant_line() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "restaurant.conv.md",
        "User: Can you recommend some authentic Italian restaurants in Rome?\n\
         Assistant: Roscioli is a great option.\n\
         User: Which one would you say is the best for a romantic dinner?\n\
         Assistant: For a romantic dinner, I would recommend Roscioli. It has a cozy and intimate atmosphere.\n",
    );
    let answer = read_answer_text(
        &idx,
        "Can you remind me of the name of the romantic Italian restaurant in Rome you recommended for dinner?",
    );
    assert!(answer.contains("Answer: Roscioli"), "{answer}");
}

#[test]
fn recalls_sectioned_bullet_list_items() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "refinery.conv.md",
        "User: give me the refining processes that go on at 3 CITGO Refineries\n\
         Assistant: 1. Lake Charles Refinery:\n\
         * Atmospheric distillation: first step.\n\
         * Fluid catalytic cracking (FCC): breaks heavy fractions.\n\
         * Alkylation: creates high-octane components.\n\
         * Hydrotreating: removes impurities.\n\
         2. Lemont Refinery:\n\
         * Delayed coking: another process.\n",
    );
    let answer = read_answer_text(
        &idx,
        "I remember you told me about the refining processes at CITGO's three refineries earlier. Can you remind me what kind of processes are used at the Lake Charles Refinery?",
    );
    assert!(
        answer.contains("Atmospheric distillation, Fluid catalytic cracking (FCC), Alkylation, and Hydrotreating.")
            || answer.contains("Atmospheric distillation, fluid catalytic cracking (FCC), alkylation, and hydrotreating."),
        "{answer}"
    );
}

#[test]
fn recalls_user_example_title() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "netflix.conv.md",
        "User: I want to be able to have access to all seasons for old shows. I will give you an example, \"doc martin\" show went down after a while, and now we have access only to the last season.\n",
    );
    let answer = read_answer_text(
        &idx,
        "I wanted to check back on our previous conversation about Netflix. I mentioned that I wanted to be able to access all seasons of old shows. Do you remember what show I used as an example, the one that only had the last season available?",
    );
    assert!(answer.contains("Answer: Doc Martin"), "{answer}");
}

#[test]
fn recalls_ordinal_list_item_from_assistant_list() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "parameters.conv.md",
        "User: Give me 100 prompt parameters.\n\
         Assistant: 26. Soliloquy (e.g., dramatic, introspective, etc.)\n\
         27. Sound effects (e.g., ambient, diegetic, non-diegetic, etc.)\n\
         28. Music (e.g., genre, tempo, etc.)\n",
    );
    let answer = read_answer_text(
        &idx,
        "I remember you provided a list of 100 prompt parameters that I can specify to influence your output. Can you remind me what was the 27th parameter on that list?",
    );
    assert!(answer.contains("Sound effects"), "{answer}");
}

#[test]
fn recalls_numeric_metric_value_from_assistant_summary() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "review.conv.md",
        "Assistant: The experiments showed an average improvement in framerate of approximately 20% when using the Hardware-Aware Modular Training (HAMT) agent.\n",
    );
    let answer = read_answer_text(
        &idx,
        "Can you remind me what was the average improvement in framerate when using the Hardware-Aware Modular Training (HAMT) agent in the 'To Adapt or Not to Adapt? Real-Time Adaptation for Semantic Segmentation' submission?",
    );
    assert!(answer.contains("approximately 20%"), "{answer}");
}

#[test]
fn recalls_ordinal_bottle_from_assistant_list() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "bottles.conv.md",
        "Assistant: To make the widest variety of gin-based cocktails, I would recommend purchasing the following five bottles:\n\
         1. Sweet Vermouth: fortified wine.\n\
         2. Dry Vermouth: fortified wine.\n\
         3. Campari: bitter aperitif.\n\
         4. Elderflower Liqueur: floral liqueur.\n\
         5. Absinthe: strong herbal liqueur.\n",
    );
    let answer = read_answer_text(
        &idx,
        "I'm looking back at our previous conversation about building a cocktail bar. You recommended five bottles to make the widest variety of gin-based cocktails. Can you remind me what the fifth bottle was?",
    );
    assert!(answer.contains("Absinthe"), "{answer}");
}

#[test]
fn recalls_descriptor_matched_company_over_numeric_salary_distractor() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "chaudhary.conv.md",
        "Assistant: 1. Jaipur Rugs: Jaipur Rugs is a private company that employs over 40,000 people in the rug-manufacturing industry.\n\
         2. Dabur: Dabur is a major consumer goods company.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "salary.conv.md",
        "Assistant: + Entry-level (0-2 years of experience): $40,000 - $60,000 per year\n",
    );
    let answer = read_answer_text(
        &idx,
        "I wanted to follow up on our previous conversation about private sector businesses in Chaudhary. Can you remind me of the company that employs over 40,000 people in the rug-manufacturing industry?",
    );
    assert!(answer.contains("Jaipur Rugs"), "{answer}");
}

#[test]
fn recalls_role_person_from_adjacent_bullet_context() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "fusion-bullets.conv.md",
        "Assistant: * Lawrence Livermore National Laboratory (LLNL)\n\
         * Dr. Arati Prabhakar\n\
         * President's Chief Advisor for Science and Technology\n\
         * Director of the White House Office of Science and Technology Policy\n",
    );
    let answer = read_answer_text(
        &idx,
        "I wanted to follow up on our previous conversation about the fusion breakthrough at Lawrence Livermore National Laboratory. Can you remind me who is the President's Chief Advisor for Science and Technology mentioned in the article?",
    );
    assert!(answer.contains("Arati Prabhakar"), "{answer}");
}

#[test]
fn recalls_descriptor_matched_singer_songwriter_example() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "catalonia.conv.md",
        "Assistant: For example, the Spanish-Catalan singer/songwriter Manolo García has spoken publicly about his support for unity between Catalonia and Spain.\n\
         Assistant: Likewise, the acclaimed novelist Javier Cercas has taken a more complex view of the political situation in Catalonia.\n",
    );
    let answer = read_answer_text(
        &idx,
        "I was going through our previous conversation about the impact of the political climate in Catalonia on its literature and music. Can you remind me of the example you gave of a Spanish-Catalan singer-songwriter who supports unity between Catalonia and Spain?",
    );
    assert!(answer.contains("Manolo García"), "{answer}");
}

#[test]
fn recalls_host_stadium_count_when_query_asks_how_many_times() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "chiefs-jaguars.conv.md",
        "Assistant: As of my knowledge cut-off in 2021, The Kansas City Chiefs and the Jacksonville Jaguars have played each other 23 times since their first meeting in 1995.\n\
         Assistant: Of the 23 games played between the Kansas City Chiefs and the Jacksonville Jaguars, 12 games were played at Arrowhead Stadium in Kansas City, Missouri.\n",
    );
    let answer = read_answer_text(
        &idx,
        "I was looking back at our previous chat and I wanted to confirm, how many times did the Chiefs play the Jaguars at Arrowhead Stadium?",
    );
    assert!(
        answer.contains("The Chiefs played the Jaguars 12 times at Arrowhead Stadium."),
        "{answer}"
    );
}

#[test]
fn recalls_role_person_from_fixture_style_entity_list() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "fusion-entities.conv.md",
        "User: predict the entities of this article one sentence by a time:\n\
         Assistant: * First Time\n\
         * Researchers\n\
         * Fusion\n\
         * Energy\n\
         * Lawrence Livermore National Laboratory (LLNL)\n\
         * Scientific Breakthrough\n\
         * Dr. Arati Prabhakar\n\
         * President's Chief Advisor for Science and Technology\n\
         * Director of the White House Office of Science and Technology Policy\n",
    );
    let answer = read_answer_text(
        &idx,
        "I wanted to follow up on our previous conversation about the fusion breakthrough at Lawrence Livermore National Laboratory. Can you remind me who is the President's Chief Advisor for Science and Technology mentioned in the article?",
    );
    assert!(answer.contains("Arati Prabhakar"), "{answer}");
}

#[test]
fn recalls_singer_songwriter_from_fixture_style_context() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "catalonia-fixture.conv.md",
        "User: How has the political climate of Catalonia influenced the themes and messages in Catalan literature and music?\n\
         Assistant: in literature, many authors have explored themes of identity, nationalism, and the struggle for independence.\n\
         Assistant: similarly, in music, many catalan artists have been inspired to create songs that promote catalan culture and identity.\n\
         User: Do you think there are any Catalan writers or musicians who take a different stance on the political situation in Catalonia?\n\
         Assistant: for example, the spanish-catalan singer/songwriter manolo garcía has spoken publicly about his support for unity between catalonia and spain.\n\
         Assistant: likewise, the acclaimed novelist javier cercas has taken a more complex view of the political situation in catalonia.\n",
    );
    let answer = read_answer_text(
        &idx,
        "I was going through our previous conversation about the impact of the political climate in Catalonia on its literature and music. Can you remind me of the example you gave of a Spanish-Catalan singer-songwriter who supports unity between Catalonia and Spain?",
    );
    assert!(answer.contains("Manolo García"), "{answer}");
}

#[test]
fn recalls_host_stadium_count_from_fixture_style_followup_context() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "chiefs-jaguars-fixture.conv.md",
        "User: how many times have the chiefs played the jaguars?\n\
         Assistant: As of my knowledge cut-off in 2021, The Kansas City Chiefs and the Jacksonville Jaguars have played each other 23 times since their first meeting in 1995.\n\
         User: of those 23 games how many were played in kansas?\n\
         Assistant: Of the 23 games played between the Kansas City Chiefs and the Jacksonville Jaguars, 12 games were played at Arrowhead Stadium in Kansas City, Missouri, which is the home stadium of the Kansas City Chiefs.\n",
    );
    let answer = read_answer_text(
        &idx,
        "I was looking back at our previous chat and I wanted to confirm, how many times did the Chiefs play the Jaguars at Arrowhead Stadium?",
    );
    assert!(
        answer.contains("The Chiefs played the Jaguars 12 times at Arrowhead Stadium."),
        "{answer}"
    );
}
