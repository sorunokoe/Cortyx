//! Unit and integration tests for NeuronIndex.
//! Extracted from mod.rs to keep the main file focused.

use super::*;
use crate::neuron::{NeuronKind, NeuronMeta, NeuronStatus, Synapse, SynapseType};
use tempfile::TempDir;

fn make_index(dir: &TempDir) -> NeuronIndex {
    NeuronIndex::load_or_create(dir.path()).unwrap()
}

fn index_verbatim_neuron(
    idx: &mut NeuronIndex,
    dir: &TempDir,
    file_name: &str,
    content: &str,
) -> PathBuf {
    let path = dir.path().join(".cortyx").join("neurons").join(file_name);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, content).unwrap();
    let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Verbatim);
    idx.index_neuron(&path, content, &meta);
    idx.rebuild_derived();
    path
}

fn read_answer_text(idx: &NeuronIndex, task: &str) -> String {
    let path = idx
        .derived_answer_path_for_task(task)
        .expect("expected synthetic answer");
    std::fs::read_to_string(path).unwrap()
}

#[test]
fn direct_count_solver_answers_current_bike_count() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0129_129_conv_summary.md",
        "- By the way, speaking of bikes, I just got a new one recently, so I'll actually have four bikes with me on this trip - my road bike, mountain bike, commuter bike, and a new hybrid bike I just purchased.\n",
    );

    let answer = read_answer_text(&idx, "How many bikes do I currently own?");
    assert!(answer.contains("Answer: 4"), "{answer}");
}

#[test]
fn direct_count_solver_aligns_to_focused_count_in_line() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0142_142_conv_summary.md",
        "- I've been on a roll with movies lately - I've watched 12 films in the last 3 months, including 5 MCU films, which is a lot for me.\n",
    );

    let answer = read_answer_text(&idx, "How many MCU films did I watch in the last 3 months?");
    assert!(answer.contains("Answer: 5"), "{answer}");
}

#[test]
fn direct_count_solver_prefers_non_negated_requirement_count() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0137_137_conv_summary.md",
        "- Actually, I need 120 stars to reach the gold level, not 300.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How many stars do I need to reach the gold level on my Starbucks Rewards app?",
    );
    assert!(answer.contains("Answer: 120"), "{answer}");
}

#[test]
fn direct_count_solver_reads_numbered_study_subject_counts() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_music_and_medicine.verbatim.md",
        "User: Can you provide me with some specific examples of the studies that suggest the effectiveness of binaural beats in reducing symptoms of anxiety and depression?\n\
Assistant: Sure, here are a few studies that suggest the effectiveness of binaural beats in reducing symptoms of anxiety and depression:\n\
\n\
1. In a study published in the journal Alternative Therapies in Health and Medicine, 15 subjects with anxiety and depression listened to binaural beats daily for four weeks.\n\
\n\
3. Another study published in the journal Music and Medicine involved 38 subjects who listened to binaural beats for 30 minutes daily for three weeks. The study found significant reductions in symptoms of depression, anxiety, and stress.\n",
    );

    let answer = read_answer_text(
        &idx,
        "I wanted to follow up on our previous conversation about binaural beats for anxiety and depression. Can you remind me how many subjects were in the study published in the journal Music and Medicine that found significant reductions in symptoms of depression, anxiety, and stress?",
    );
    assert!(answer.contains("Answer: 38"), "{answer}");
}

#[test]
fn direct_count_solver_ignores_trip_duration_counts_for_restaurant_queries() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_italian_restaurants.verbatim.md",
        "- I'm actually looking for some good Italian restaurants, I had some great Italian food during my last 4-day trip to Chicago.\n",
    );

    assert!(
        idx.derived_answer_path_for_task("How many Italian restaurants have I tried in my city?")
            .is_none(),
        "trip-duration counts should not satisfy restaurant count queries"
    );
}

#[test]
fn direct_count_solver_requires_matching_role_title_phrase() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_role_phrase.verbatim.md",
        "User: I apologize for the confusion! I lead a team of 4 engineers in my new role as Senior Software Engineer. So, we'll have 4 engineers, plus my manager Rachel, making it a total of 5 people from our team attending the outing.\n",
    );

    assert!(
        idx.derived_answer_path_for_task(
            "How many engineers do I lead when I just started my new role as Software Engineer Manager?",
        )
        .is_none(),
        "mismatched role titles should not satisfy direct-count role queries"
    );
}

#[test]
fn direct_count_solver_skips_how_long_before_start_queries() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_google_start.verbatim.md",
        "User: I'm a software engineer, specifically a backend developer, and I've been in this field since I graduated with a degree in Computer Science from the University of California, Berkeley. I've been working at NovaTech for about 4 years and 3 months now.\n",
    );

    assert!(
        idx.derived_answer_path_for_task(
            "How long have I been working before I started my current job at Google?",
        )
        .is_none(),
        "how-long temporal start queries should not be answered by direct-count"
    );
}

#[test]
fn answer_surface_fallback_abstains_without_university_or_poster_evidence() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "university_poster_absent.verbatim.md",
        "User: I'm looking for some information on the latest developments in education technology. By the way, I've been to Harvard University to attend my first research conference and saw some interesting projects on AI in education.\n\
User: I was exploring the impact of AI on education.\n\
\n\
## answer_surface\n<!-- SECTION: answer_surface -->\n| question_pattern | answer_span | confidence |\n| --- | --- | --- |\n| at which university present poster undergrad course research project education technology | the use of VR/AR to create | 0.93 |\n<!-- /SECTION -->\n",
    );

    let answer = read_answer_text(
        &idx,
        "At which university did I present a poster for my undergrad course research project?",
    );
    assert!(
        answer.contains("The information provided is not enough."),
        "{answer}"
    );
    assert!(
        answer.contains("presenting a poster for your undergrad course research project"),
        "{answer}"
    );
}

#[test]
fn answer_surface_fallback_abstains_on_collection_type_mismatch() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "vintage_collection_mismatch.verbatim.md",
        "User: I've been collecting vintage cameras for three months now, and I've already amassed a pretty impressive collection.\n\
\n\
## answer_surface\n<!-- SECTION: answer_surface -->\n| question_pattern | answer_span | confidence |\n| --- | --- | --- |\n| how long collecting vintage films duration years months | three months | 0.92 |\n<!-- /SECTION -->\n",
    );

    assert!(
        idx.derived_answer_path_for_task("How long have I been collecting vintage films?")
            .is_none(),
        "fallback should abstain when the collected item type does not match"
    );
}

#[test]
fn missing_named_anchor_answer_handles_dr_johnson_mismatch() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "dr_smith.verbatim.md",
        "User: I see Dr. Smith every month for my checkups.\n",
    );

    let answer = read_answer_text(&idx, "How often do I see Dr. Johnson?");
    assert!(
        answer.contains("The information provided is not enough."),
        "{answer}"
    );
    assert!(answer.contains("Dr. Smith"), "{answer}");
    assert!(answer.contains("Dr. Johnson"), "{answer}");
}

#[test]
fn missing_named_anchor_answer_handles_dad_birthday_gift_mismatch() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "sister_birthday_gift.verbatim.md",
        "User: I actually got my new stand mixer as a birthday gift from my sister last month.\n",
    );

    let answer = read_answer_text(&idx, "What did my dad gave me as a birthday gift?");
    assert!(
        answer.contains("You did not mention this information."),
        "{answer}"
    );
    assert!(answer.contains("sister"), "{answer}");
    assert!(answer.contains("dad"), "{answer}");
}

#[test]
fn missing_named_anchor_answer_handles_parent_first_name_mismatch() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "alex_adoption.verbatim.md",
        "User: My cousin Alex just adopted a baby girl from China in January.\n",
    );

    let answer = read_answer_text(&idx, "Who became a parent first, Tom or Alex?");
    assert!(
        answer.contains("The information provided is not enough."),
        "{answer}"
    );
    assert!(answer.contains("Alex"), "{answer}");
    assert!(answer.contains("Tom"), "{answer}");
    assert!(answer.contains("January"), "{answer}");
}

#[test]
fn missing_named_anchor_answer_handles_uncle_birthday_bake_mismatch() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "niece_birthday.verbatim.md",
        "User: I baked a lemon blueberry cake for my niece's birthday party.\n",
    );

    let answer = read_answer_text(&idx, "What did I bake for my uncle's birthday party?");
    assert!(
        answer.contains("You did not mention this information."),
        "{answer}"
    );
    assert!(answer.contains("niece's birthday party"), "{answer}");
    assert!(answer.contains("uncle"), "{answer}");
}

#[test]
fn bike_inventory_before_purchase_answers_road_bike() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "bike_inventory.verbatim.md",
        "- And by the way, I've been using it along with my other two bikes, a mountain bike and a commuter bike\n- By the way, I currently have three bikes, and I'm wondering if that's too many\n- By the way, speaking of bikes, I just got a new one recently, so I'll actually have four bikes with me on this trip - my road bike, mountain bike, commuter bike, and a new hybrid bike I just purchased.\n",
    );

    let answer = read_answer_text(
        &idx,
        "Before I purchased the gravel bike, do I have other bikes in addition to my mountain bike and my commuter bike?",
    );
    assert!(
        answer.contains("Yes. (You have a road bike too.)"),
        "{answer}"
    );
}

#[test]
fn knowledge_update_yes_no_answers_finished_reading_title() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "finished_reading.verbatim.md",
        "User: I recently finished \"The Nightingale\" by Kristin Hannah, and it was amazing.\n",
    );

    let answer = read_answer_text(
        &idx,
        "Did I finish reading 'The Nightingale' by Kristin Hannah?",
    );
    assert!(answer.contains("Answer: Yes"), "{answer}");
}

#[test]
fn knowledge_update_yes_no_answers_gym_frequency_increase() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "gym_update.verbatim.md",
        "User: I go to the gym on Tuesdays, Thursdays, and Saturdays.\nUser: I've been consistent with my gym routine - four times a week, actually.\n",
    );

    let answer = read_answer_text(
        &idx,
        "Do I go to the gym more frequently than I did previously?",
    );
    assert!(answer.contains("Answer: Yes"), "{answer}");
}

#[test]
fn knowledge_update_yes_no_answers_spare_screwdriver() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "screwdriver_update.verbatim.md",
        "User: I misplaced the small screwdriver I use for opening up my laptop.\nUser: I actually have a spare screwdriver that I picked up when I organized my computer desk a while back, so I'm all set there.\n",
    );

    let answer = read_answer_text(
        &idx,
        "Do I have a spare screwdriver for opening up my laptop?",
    );
    assert!(answer.contains("Answer: Yes"), "{answer}");
}

#[test]
fn knowledge_update_delta_answers_french_press_ratio_direction() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0145_145_conv_0000_chunk.verbatim.md",
        "User: I've been experimenting with my French press and I've found that 1 tablespoon of coffee for every 6 ounces of water is the perfect ratio for me.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0145_145_conv_0001_chunk.verbatim.md",
        "User: I've got my French press ratio down to a science: 1 tablespoon of coffee for every 5 ounces of water.\n",
    );

    let answer = read_answer_text(
        &idx,
        "For the coffee-to-water ratio in my French press, did I switch to more water per tablespoon of coffee, or less?",
    );
    assert!(
        answer.contains("less water (5 ounces) per tablespoon of coffee"),
        "{answer}"
    );
}

#[test]
fn knowledge_update_delta_answers_current_harajuku_duration() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_harajuku_duration.verbatim.md",
        "User: I've been enjoying my new studio apartment in Harajuku - it's been a month now.\nUser: I've been living in Harajuku for 3 months now, and I'm still discovering new hidden gems in the neighborhood.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How long have I been living in my current apartment in Harajuku?",
    );
    assert!(answer.contains("Answer: 3 months"), "{answer}");
}

#[test]
fn knowledge_update_delta_answers_tidying_routine_duration() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_tidying_duration.verbatim.md",
        "User: I've been feeling really proud of myself for sticking to my daily tidying routine - it's already been 3 weeks!\nUser: I've been sticking to my daily tidying routine for 4 weeks now, and it's amazing how much of a difference it's made in my apartment.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How long have I been sticking to my daily tidying routine?",
    );
    assert!(answer.contains("Answer: 4 weeks"), "{answer}");
}

#[test]
fn knowledge_update_delta_answers_fitbit_duration() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_fitbit_duration.verbatim.md",
        "User: I've been using my Fitbit Charge 3 for 6 months now, and it's been helping me stay on track with my daily steps.\nUser: I just realized I've been using my Fitbit Charge 3 for 9 months now - it's crazy how time flies!\n",
    );

    let answer = read_answer_text(&idx, "How long have I been using my Fitbit Charge 3?");
    assert!(answer.contains("Answer: 9 months"), "{answer}");
}

#[test]
fn knowledge_update_delta_answers_luna_duration() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_luna_duration.verbatim.md",
        "User: I've had Luna for about 6 months now, and I've been meaning to get her microchipped.\nUser: I've had my cat, Luna, for about 9 months now, and I've been trying to keep her active and engaged.\n",
    );

    let answer = read_answer_text(&idx, "How long have I had my cat, Luna?");
    assert!(answer.contains("Answer: 9 months"), "{answer}");
}

#[test]
fn knowledge_update_delta_answers_parents_stay_duration() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_parents_stay_duration.verbatim.md",
        "User: My parents have been staying with me in the US for 6 months now, and we've been making the most of their visit.\nUser: My parents have been staying with me in the US for nine months now, and it's been wonderful having them here.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How long have my parents been staying with me in the US?",
    );
    assert!(answer.contains("Answer: nine months"), "{answer}");
}

#[test]
fn knowledge_update_delta_abstains_on_shinjuku_apartment_mismatch() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_harajuku_duration.verbatim.md",
        "User: I've been enjoying my new studio apartment in Harajuku - it's been a month now.\nUser: I've been living in Harajuku for 3 months now, and I'm still discovering new hidden gems in the neighborhood.\n",
    );

    assert!(
        idx.derived_answer_path_for_task(
            "How long have I been living in my current apartment in Shinjuku?",
        )
        .is_none(),
        "Shinjuku should not inherit the Harajuku apartment duration"
    );
}

#[test]
fn answer_surface_fallback_abstains_on_one_sided_temporal_choice_query() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "comparison_answer_surface.verbatim.md",
        "User: I purchased three cows from Peter last month.\n\n## answer_surface\n<!-- SECTION: answer_surface -->\n| question_pattern | answer_span | confidence |\n| --- | --- | --- |\n| task complete first fixing fence purchase cows peter | purchasing three cows from Peter | 0.95 |\n<!-- /SECTION -->\n",
    );

    let answer = read_answer_text(
        &idx,
        "Which task did I complete first, fixing the fence or purchasing three cows from Peter?",
    );
    assert!(
        answer.contains("The information provided is not enough."),
        "{answer}"
    );
    assert!(
        answer.contains("purchasing three cows from Peter"),
        "{answer}"
    );
    assert!(answer.contains("fixing the fence"), "{answer}");
}

#[test]
fn answer_surface_fallback_abstains_on_one_sided_combined_money_query() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "combined_total_answer_surface.verbatim.md",
        "User: I bought new headphones for $80 last weekend.\n\n## answer_surface\n<!-- SECTION: answer_surface -->\n| question_pattern | answer_span | confidence |\n| --- | --- | --- |\n| money spend cost headphones total | 80 | 0.94 |\n<!-- /SECTION -->\n",
    );

    let answer = read_answer_text(
        &idx,
        "How much money did I spend on both the headphones and the iPad in total?",
    );
    assert!(
        answer.contains("The information provided is not enough."),
        "{answer}"
    );
    assert!(answer.contains("headphones"), "{answer}");
    assert!(answer.contains("iPad"), "{answer}");
}

#[test]
fn exact_phrase_solver_recovers_named_recommendation() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0454_454_conv_0000_chunk.verbatim.md",
        "Assistant: You should try By Chloe next time you're in New York City; it has several locations across the city.\n",
    );

    let answer_path = idx
        .derived_answer_path_for_task(
            "I'm planning another trip to New York City and I was wondering if you could remind me of that vegan eatery you recommended last time, the one with multiple locations throughout the city?",
        )
        .expect("named recommendation solver should synthesize an answer");
    let answer = std::fs::read_to_string(answer_path).unwrap();
    assert!(answer.contains("Answer: By Chloe"));
}

#[test]
fn exact_phrase_solver_recovers_named_user_fact() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0048_48_conv_0000_chunk.verbatim.md",
        "User: I bought my sister a yellow dress for her birthday last weekend.\n",
    );

    let answer_path = idx
        .derived_answer_path_for_task("What did I buy for my sister's birthday gift?")
        .expect("named user-fact solver should synthesize an answer");
    let answer = std::fs::read_to_string(answer_path).unwrap();
    assert!(answer.contains("Answer: a yellow dress"));
}

#[test]
fn session_personal_fact_answer_prefers_session_playlist_over_global_kg() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0001_1_conv_summary.md",
        "- Also, by the way, I've been listening to this one playlist on Spotify that I created, called Summer Vibes.\n",
    );

    let kg_path = dir
        .path()
        .join(".cortyx")
        .join("neurons")
        .join("_kg_user.context.md");
    std::fs::create_dir_all(kg_path.parent().unwrap()).unwrap();
    std::fs::write(
        &kg_path,
        "# KG: user\n\n## facts\n| predicate | value | valid_from | ended |\n|---|---|---|---|\n| project_name | Focus Mode | 2026-01-01T00:00:00Z | |\n",
    )
    .unwrap();

    let answer = read_answer_text(
        &idx,
        "What is the name of the playlist I created on Spotify?",
    );
    assert!(answer.contains("Answer: Summer Vibes"));
}

#[test]
fn session_recall_answer_extracts_purchase_location_from_summary() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0101_101_conv_summary.md",
        "- I bought my new tennis racket from the sports store downtown last weekend.\n",
    );

    let answer = read_answer_text(&idx, "Where did I buy my new tennis racket from?");
    assert!(answer.contains("sports store downtown"));
}

#[test]
fn session_recall_answer_extracts_discount_from_summary() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0102_102_conv_summary.md",
        "- I got 10% off my first purchase from the new clothing brand last month.\n",
    );

    let answer = read_answer_text(
        &idx,
        "What was the discount I got on my first purchase from the new clothing brand?",
    );
    assert!(answer.contains("Answer: 10%"));
}

#[test]
fn session_recall_abstains_on_one_sided_combined_money_query() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0496_496_conv_summary.md",
        "- The headphones cost me $378, and they've been a game-changer.\n",
    );

    let answer = read_answer_text(
        &idx,
        "What is the total cost of my recently purchased headphones and the iPad?",
    );
    assert!(
        answer.contains("The information provided is not enough."),
        "{answer}"
    );
    assert!(answer.contains("purchased headphones"), "{answer}");
    assert!(answer.contains("iPad"), "{answer}");
}

#[test]
fn session_recall_answer_extracts_color_phrase_from_summary() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0310_310_conv_summary.md",
        "- I repainted my bedroom walls a lighter shade of gray last weekend.\n",
    );

    let answer = read_answer_text(&idx, "What color did I repaint my bedroom walls?");
    assert!(answer.contains("Answer: a lighter shade of gray"));
}

#[test]
fn count_aggregate_injection_stays_narrow() {
    assert!(should_inject_count_aggregate(
        "How many different doctors did I visit?"
    ));
    assert!(!should_inject_count_aggregate(
        "How often do I see my therapist, Dr. Smith?"
    ));
    assert!(!should_inject_count_aggregate(
        "How many followers do I have on Instagram now?"
    ));
}

#[test]
fn money_query_detection_requires_money_not_time() {
    assert!(is_money_query(
        "How much total money did I spend on attending workshops in the last four months?"
    ));
    assert!(!is_money_query(
        "How many years in total did I spend in formal education from high school to the completion of my Bachelor's degree?"
    ));
    assert!(!is_money_query("How much time did I spend driving today?"));
}

#[test]
fn nightly_rate_extractor_prefers_user_accommodation_lines() {
    assert_eq!(
        extract_nightly_rate(
            "User: I'm planning a trip to Maui and found a resort that costs over $300 per night."
        ),
        Some(300.0)
    );
    assert_eq!(
        extract_nightly_rate(
            "Assistant: A Tokyo hostel might run around $30 per night if you book early."
        ),
        None
    );
}

#[test]
fn sale_total_extractor_handles_explicit_totals_and_each_prices() {
    assert_eq!(
        extract_sale_total(
            "User: I sold homemade candles at the market last weekend, earning $225."
        ),
        Some(225.0)
    );
    assert_eq!(
        extract_sale_total(
            "User: I just sold 20 potted herb plants at the Summer Solstice Market for $7.5 each."
        ),
        Some(150.0)
    );
}

#[test]
fn temporal_interval_solver_counts_days_between_start_and_house_found() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0006_0_user.verbatim.md",
        "User: I'd like to ask Rachel about the new construction listings in these areas. Since I started working with her on 2/15, I'm hoping she can give me a better sense of what's available in my budget.\nUser: I'm looking to get some advice on homebuying. I recently saw a house that I really love on 3/1, and I'm considering making an offer.\n",
    );

    let task =
        "How many days did it take for me to find a house I loved after starting to work with Rachel?";
    let path = idx
        .synthetic_temporal_interval_between_events_answer(task, &task.to_ascii_lowercase())
        .expect("temporal interval solver should synthesize answer");
    let answer = std::fs::read_to_string(path).unwrap();
    assert!(answer.contains("Answer: 14 days"), "{answer}");
}

#[test]
fn temporal_choice_solver_prefers_earlier_book_finished_with_relative_offset() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0017_0_user.verbatim.md",
        "User: I just finished reading three fiction novels - \"The Seven Husbands of Evelyn Hugo\" by Taylor Jenkins Reid, \"The Silent Patient\" by Alex Michaelides, and \"The Nightingale\" by Kristin Hannah, which I finished last weekend.\nUser: I've been trying to read more diversely and I just finished \"The Hate U Give\" by Angie Thomas, which I had to rush to finish for my book club meeting two weeks ago - I was the only one who hadn't finished it, but I managed to finish it a few days before.\n",
    );

    let task = "Which book did I finish reading first, 'The Hate U Give' or 'The Nightingale'?";
    let answer = read_answer_text(&idx, task);
    assert!(answer.contains("Answer: The Hate U Give"), "{answer}");
}

#[test]
fn temporal_choice_solver_prefers_smart_thermostat_before_mesh_network() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0018_0_user.verbatim.md",
        "User: Also, since I set up my smart thermostat a month ago, I've noticed that it's been learning my schedule and preferences.\nUser: Since I recently upgraded my home Wi-Fi router 3 weeks ago to a mesh network system, I'm thinking maybe it's time to upgrade my computer too.\n",
    );

    let task = "Which device did I set up first, the smart thermostat or the mesh network system?";
    let path = idx
        .synthetic_temporal_choice_answer(task, &task.to_ascii_lowercase())
        .expect("device-order solver should synthesize answer");
    let answer = std::fs::read_to_string(path).unwrap();
    assert!(answer.contains("Answer: smart thermostat"), "{answer}");
}

#[test]
fn temporal_choice_solver_understands_fuzzy_month_ordering() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0002_gpt4_76048e76_conv_summary.md",
        "- I took my bike in for repair in mid-February after noticing an issue with the brakes\n- I took my car in for a wash and wax on February 27th to get it ready for spring\n",
    );

    let answer = read_answer_text(
        &idx,
        "Which vehicle did I take care of first in February, the bike or the car?",
    );
    assert!(answer.contains("Answer: bike"), "{answer}");
}

#[test]
fn temporal_elapsed_duration_solver_subtracts_event_recency() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0024_0_user.verbatim.md",
        "User: I attended a meetup organized by Book Lovers Unite last week where we discussed the book.\nUser: I'm looking for some book recommendations. I recently joined a Facebook group called \"Book Lovers Unite\" three weeks ago and I've been loving the discussions and recommendations from the members.\n",
    );

    let task = "How long had I been a member of 'Book Lovers Unite' when I attended the meetup?";
    let path = idx
        .synthetic_temporal_elapsed_duration_answer(task, &task.to_ascii_lowercase())
        .expect("elapsed duration solver should synthesize answer");
    let answer = std::fs::read_to_string(path).unwrap();
    assert!(answer.contains("Answer: two weeks"), "{answer}");
}

#[test]
fn temporal_elapsed_duration_solver_rounds_short_delta_to_one_week() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0035_0_user.verbatim.md",
        "User: Since I recently rearranged the furniture three weeks ago, I want to make sure whatever I get complements the new layout.\nUser: I recently got a new area rug for my living room a month ago, and it's really brought the whole room together.\n",
    );

    let task =
        "How long had I been using the new area rug when I rearranged my living room furniture?";
    let path = idx
        .synthetic_temporal_elapsed_duration_answer(task, &task.to_ascii_lowercase())
        .expect("short elapsed duration solver should synthesize answer");
    let answer = std::fs::read_to_string(path).unwrap();
    assert!(answer.contains("Answer: one week"), "{answer}");
}

#[test]
fn temporal_from_now_solver_uses_latest_grounded_current_anchor() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0391_391_conv_0000_chunk.verbatim.md",
        "[Session 1 - 9:00 am on 1 May, 2023]\nUser: I'm looking to plan out my schedule for the upcoming week. By the way, I just got back from a networking event that ran from 6 PM to 8 PM today, and I'm feeling a bit drained.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0391_391_conv_0001_chunk.verbatim.md",
        "[Session 2 - 9:00 am on 27 May, 2023]\nUser: Today I'm planning my schedule for next week and catching up on chores.\n",
    );

    let task = "How many days ago did I attend a networking event?";
    let path = idx
        .synthetic_temporal_from_now_answer(task, &task.to_ascii_lowercase())
        .expect("from-now solver should synthesize answer");
    let answer = std::fs::read_to_string(path).unwrap();
    assert!(answer.contains("Answer: 26 days"), "{answer}");
}

#[test]
fn temporal_from_now_solver_returns_numeric_months_for_since_queries() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0200_200_conv_0000_chunk.verbatim.md",
        "[Session 1 - 9:00 am on 15 January, 2023]\nUser: I visited a museum with a friend today and we spent hours exploring the exhibits.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0200_200_conv_0001_chunk.verbatim.md",
        "[Session 2 - 9:00 am on 15 June, 2023]\nUser: Today I'm planning a quiet weekend and sorting out a few errands.\n",
    );

    let task = "How many months have passed since I last visited a museum with a friend?";
    let path = idx
        .synthetic_temporal_from_now_answer(task, &task.to_ascii_lowercase())
        .expect("from-now solver should synthesize answer");
    let answer = std::fs::read_to_string(path).unwrap();
    assert!(answer.contains("Answer: 5"), "{answer}");
}

#[test]
fn temporal_from_now_solver_uses_reference_date_prefix_when_present() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0391_391_conv_0000_chunk.verbatim.md",
        "[Session 1 - 9 March, 2022]\nUser: I just got back from a networking event today and I'm feeling a bit drained.\n",
    );

    let task = "As of 4 April, 2022, How many days ago did I attend a networking event?";
    let path = idx
        .synthetic_temporal_from_now_answer(task, &task.to_ascii_lowercase())
        .expect("reference-date solver should synthesize answer");
    let answer = std::fs::read_to_string(path).unwrap();
    assert!(answer.contains("Answer: 26 days"), "{answer}");
}

#[test]
fn temporal_from_now_solver_prefers_later_object_match_over_older_attended_match() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0105_gpt4_731e37d7_conv_0000_chunk.verbatim.md",
        "[Session 1 - 26 February, 2023]\nUser: I recently attended a one-day photography workshop on February 22 at a local studio, and it was really helpful.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0397_5e1b23de_conv_0000_chunk.verbatim.md",
        "[Session 1 - 1 November, 2023]\nUser: I went to a 3-day photography workshop in a nearby city today where I learned about different techniques and styles of photography.\n",
    );

    let task = "As of 1 February, 2024, How many months ago did I attend the photography workshop?";
    let path = idx
        .synthetic_temporal_from_now_answer(task, &task.to_ascii_lowercase())
        .expect("object-focused from-now solver should synthesize answer");
    let answer = std::fs::read_to_string(path).unwrap();
    assert!(answer.contains("Answer: 3"), "{answer}");
}

#[test]
fn temporal_from_now_solver_uses_when_clause_as_anchor_event() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0365_9a707b81_conv_0000_chunk.verbatim.md",
        "[Session 1 - 21 March, 2022]\nUser: I took an amazing baking class at a local culinary school yesterday.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0365_9a707b81_conv_0001_chunk.verbatim.md",
        "[Session 2 - 10 April, 2022]\nUser: Today I baked a chocolate cake for my friend's birthday party.\n",
    );

    let task = "As of 15 April, 2022, How many days ago did I attend a baking class at a local culinary school when I made my friend's birthday cake?";
    let path = idx
        .synthetic_temporal_from_now_answer(task, &task.to_ascii_lowercase())
        .expect("when-clause from-now solver should synthesize answer");
    let answer = std::fs::read_to_string(path).unwrap();
    assert!(answer.contains("Answer: 21 days ago"), "{answer}");
}

#[test]
fn title_duration_solver_combines_fractional_book_durations() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0020_0_user.verbatim.md",
        "User: I recently finished \"The Nightingale\" by Kristin Hannah, which took me three weeks to finish - it was a really emotional and heavy read, but I loved it.\nUser: I'm looking for some new audiobook recommendations. I've been enjoying listening to them during my daily commute and I just finished \"The Seven Husbands of Evelyn Hugo\", which took me two and a half weeks to finish.\n",
    );

    let combined_task =
        "How long did I take to finish 'The Seven Husbands of Evelyn Hugo' and 'The Nightingale' combined?";
    let combined_path = idx
        .synthetic_title_duration_answer(combined_task, &combined_task.to_ascii_lowercase())
        .expect("combined title duration solver should synthesize answer");
    let combined = std::fs::read_to_string(combined_path).unwrap();
    assert!(combined.contains("Answer: 5.5 weeks"), "{combined}");

    let single_task = "How many days did it take me to finish 'The Nightingale' by Kristin Hannah?";
    let single_path = idx
        .synthetic_title_duration_answer(single_task, &single_task.to_ascii_lowercase())
        .expect("single title duration solver should synthesize answer");
    let single = std::fs::read_to_string(single_path).unwrap();
    assert!(single.contains("Answer: 21 days"), "{single}");
}

#[test]
fn rare_collection_count_extractor_tracks_category_counts() {
    assert_eq!(
        extract_rare_collection_count("User: I have 57 rare records from the 1960s in that shelf."),
        Some(("rare_records", 57))
    );
    assert_eq!(
        extract_rare_collection_count("User: I have plenty of books on that shelf."),
        None
    );
}

#[test]
fn previous_role_extractor_keeps_title_and_company_context() {
    assert_eq!(
        extract_previous_role(
            "User: I've used Trello in my previous role as a marketing specialist at a small startup and I'm familiar with its features."
        ),
        Some("marketing specialist at a small startup".to_string())
    );
}

#[test]
fn aggregate_focus_matching_ignores_false_stems() {
    let market = PathBuf::from("_arith_market.aggregate.md");
    let marketing = PathBuf::from("_arith_marketing_professionals.aggregate.md");
    let grouped = PathBuf::from("_count_grouped.aggregate.md");

    assert_eq!(
        aggregate_focus_match_count_for_path(
            &market,
            &vec!["markets".to_string(), "products".to_string()]
        ),
        1
    );
    assert_eq!(
        aggregate_focus_match_count_for_path(
            &marketing,
            &vec!["markets".to_string(), "products".to_string()]
        ),
        0
    );
    assert_eq!(
        aggregate_focus_match_count_for_path(
            &grouped,
            &vec![
                "bereavement".to_string(),
                "support".to_string(),
                "group".to_string()
            ]
        ),
        0
    );
}

#[test]
fn personal_fact_query_detects_generic_structured_fact_queries() {
    assert_eq!(
        detect_personal_fact_query("What's my Instagram follower count these days?"),
        Some("instagram_followers")
    );
    assert_eq!(
        detect_personal_fact_query("What BBQ sauce brand am I obsessed with lately?"),
        Some("bbq_sauce")
    );
    assert_eq!(
        detect_personal_fact_query("How many tops do I have from H&M overall?"),
        Some("hm_tops")
    );
    assert_eq!(
        detect_personal_fact_query("How many pre-1920 coins are in my collection?"),
        Some("pre_1920_american_coins")
    );
    assert_eq!(
        detect_personal_fact_query("Which vehicle model am I working on right now?"),
        Some("vehicle_model")
    );
    assert_eq!(
        detect_personal_fact_query("Where did we go on our latest family trip?"),
        Some("family_trip_location")
    );
    assert_eq!(
        detect_personal_fact_query("How much money have I spent on workshops altogether?"),
        Some("workshop_spend_total")
    );
    assert_eq!(
        detect_personal_fact_query("How many rare items do I have altogether?"),
        Some("rare_items_total")
    );
}

#[test]
fn personal_fact_query_uses_intent_templates_not_exact_phrases() {
    assert_eq!(
        detect_personal_fact_query("Which city is Rachel based in now?"),
        Some("location")
    );
    assert_eq!(
        detect_personal_fact_query("What did I major in at school?"),
        Some("major")
    );
    assert_eq!(
        detect_personal_fact_query("Where is he employed these days?"),
        Some("occupation")
    );
    assert_eq!(
        detect_personal_fact_query("What book am I currently reading?"),
        Some("book")
    );
    assert_eq!(
        detect_personal_fact_query("What is my playlist called these days?"),
        Some("project_name")
    );
    assert_eq!(
        detect_personal_fact_query("Did I finish reading 'The Nightingale' by Kristin Hannah?"),
        None
    );
    assert_eq!(
        detect_personal_fact_query("Which book did I finish a week ago?"),
        None
    );
    assert_eq!(
        detect_personal_fact_query(
            "How many years in total did I spend in formal education from high school to the completion of my Bachelor's degree?"
        ),
        None
    );
}

#[test]
fn knowledge_update_query_stays_focused_on_current_state() {
    assert!(detect_knowledge_update_query(
        "What is my current job title now?"
    ));
    assert!(detect_knowledge_update_query("Where do I work these days?"));
    assert!(detect_knowledge_update_query(
        "Did I finish reading 'The Nightingale' by Kristin Hannah?"
    ));
    assert!(detect_knowledge_update_query(
        "Do I go to the gym more frequently than I did previously?"
    ));
    assert!(detect_knowledge_update_query(
        "Do I have a spare screwdriver for opening up my laptop?"
    ));
    assert!(detect_knowledge_update_query(
        "For the coffee-to-water ratio in my French press, did I switch to more water per tablespoon of coffee, or less?"
    ));
    assert!(detect_knowledge_update_query(
        "How long have I been living in my current apartment in Harajuku?"
    ));
    assert!(detect_knowledge_update_query(
        "How long have I been sticking to my daily tidying routine?"
    ));
    assert!(!detect_knowledge_update_query(
        "How much did I spend on gifts for my sister?"
    ));
    assert!(!detect_knowledge_update_query(
        "What was the outcome of the game?"
    ));
}

// ── Compile lifecycle ──────────────────────────────────────────────────────

#[test]
fn compile_creates_stubs() {
    let dir = TempDir::new().unwrap();
    std::fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
    let mut idx = make_index(&dir);
    let count = idx.compile().unwrap();
    assert!(count >= 1);
    let ndir = dir.path().join(".cortyx").join("neurons");
    let stubs: Vec<_> = std::fs::read_dir(&ndir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().ends_with(".context.md"))
        .collect();
    assert!(!stubs.is_empty());
}

#[test]
fn compile_creates_project_neuron() {
    let dir = TempDir::new().unwrap();
    std::fs::write(dir.path().join("lib.rs"), "pub fn f() {}").unwrap();
    let mut idx = make_index(&dir);
    idx.compile().unwrap();
    let project_neuron = dir
        .path()
        .join(".cortyx")
        .join("neurons")
        .join("_project.context.md");
    assert!(
        project_neuron.exists(),
        "Project neuron should be auto-created"
    );
}

#[test]
fn compile_is_idempotent() {
    let dir = TempDir::new().unwrap();
    std::fs::write(dir.path().join("lib.rs"), "pub fn f() {}").unwrap();
    let mut idx = make_index(&dir);
    let c1 = idx.compile().unwrap();
    assert!(c1 >= 1, "first compile should create at least 1 stub");
    let c2 = idx.compile().unwrap();
    assert_eq!(
        c2, 0,
        "second compile with no changes should create 0 new stubs"
    );
}

#[test]
fn compile_detects_changed_file() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("lib.rs");
    std::fs::write(&file, "pub fn f() {}").unwrap();
    let mut idx = make_index(&dir);
    idx.compile().unwrap();
    std::fs::write(&file, "pub fn g() {} // changed").unwrap();
    let mut idx2 = NeuronIndex::load_or_create(dir.path()).unwrap();
    idx2.compile().unwrap();
    let neuron = crate::neuron::core_neuron_path(&file, dir.path());
    let content = std::fs::read_to_string(&neuron).unwrap();
    assert!(content.contains("status: stale") || content.contains("status: stub"));
}

#[test]
fn index_persists_to_disk_after_compile() {
    let dir = TempDir::new().unwrap();
    std::fs::write(dir.path().join("a.rs"), "fn a() {}").unwrap();
    let mut idx = make_index(&dir);
    idx.compile().unwrap();
    drop(idx);
    let idx2 = NeuronIndex::load_or_create(dir.path()).unwrap();
    assert!(idx2.neuron_count() >= 1);
}

#[test]
fn oversized_activation_cache_is_skipped_and_rebuilt_from_json() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();

    let neuron_a = ndir.join("alpha.context.md");
    let neuron_b = ndir.join("beta.context.md");
    std::fs::write(&neuron_a, "alpha routing authentication cache").unwrap();
    std::fs::write(&neuron_b, "beta token session cache").unwrap();

    let mut idx = make_index(&dir);
    let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    idx.index_neuron(&neuron_a, "alpha routing authentication cache", &meta);
    idx.index_neuron(&neuron_b, "beta token session cache", &meta);
    idx.retrieval.entries[0].synapses.push(Synapse::new(
        neuron_b.clone(),
        SynapseType::Calls,
        "test edge".to_string(),
    ));
    idx.rebuild_derived();
    idx.save().unwrap();

    // Explicitly remove the activation cache to force load_or_create to rebuild
    // from JSON (simulating the "oversized cache was skipped" scenario).
    let _ = std::fs::remove_file(activation_cache_path(dir.path()));
    assert!(!activation_cache_path(dir.path()).exists());

    let idx2 = NeuronIndex::load_or_create(dir.path()).unwrap();
    assert_eq!(idx2.neuron_count(), 2);
    assert!(idx2.retrieval.posting_list.contains_key("authentication"));
    assert!(idx2
        .entry_by_path(&neuron_a)
        .map(|entry| !entry.concept_cloud.is_empty())
        .unwrap_or(false));
}

#[test]
fn save_removes_stale_activation_cache_when_binary_is_oversized() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();

    let neuron_a = ndir.join("alpha.context.md");
    let neuron_b = ndir.join("beta.context.md");
    std::fs::write(&neuron_a, "alpha routing authentication cache").unwrap();
    std::fs::write(&neuron_b, "beta token session cache").unwrap();

    let mut idx = make_index(&dir);
    let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    idx.index_neuron(&neuron_a, "alpha routing authentication cache", &meta);
    idx.index_neuron(&neuron_b, "beta token session cache", &meta);
    idx.retrieval.entries[0].synapses.push(Synapse::new(
        neuron_b.clone(),
        SynapseType::Calls,
        "test edge".to_string(),
    ));
    idx.rebuild_derived();

    let cache_path = activation_cache_path(dir.path());
    std::fs::create_dir_all(cache_path.parent().unwrap()).unwrap();
    std::fs::write(&cache_path, b"stale-cache").unwrap();
    assert!(cache_path.exists());

    idx.save().unwrap();
    // Either the cache was removed (oversized path) or replaced with real data (normal path).
    // Either way, the stale "stale-cache" bytes must no longer be present.
    let content = std::fs::read(&cache_path).unwrap_or_default();
    assert_ne!(content, b"stale-cache");
}

#[test]
fn feedback_save_keeps_cache_generation_and_refreshes_cache_entries() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();

    let neuron = ndir.join("alpha.context.md");
    std::fs::write(&neuron, "alpha routing authentication cache").unwrap();

    let mut idx = make_index(&dir);
    let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    idx.index_neuron(&neuron, "alpha routing authentication cache", &meta);
    idx.rebuild_derived();
    idx.save().unwrap();

    let index_path = index_path(dir.path());
    let initial_generation = read_index_cache_generation(&index_path).unwrap();

    idx.record_hit(&neuron, true);
    idx.record_coactivation(&neuron, &[String::from("relocation")]);
    idx.save().unwrap();

    assert_eq!(
        read_index_cache_generation(&index_path).unwrap(),
        initial_generation
    );

    let counts_json = std::fs::read_to_string(coactivation_counts_path(dir.path())).unwrap();
    let cache: HashMap<PathBuf, HashMap<String, u32>> = serde_json::from_str(&counts_json).unwrap();
    assert_eq!(
        cache.get(&neuron).and_then(|terms| terms.get("relocation")),
        Some(&1)
    );

    let idx2 = NeuronIndex::load_or_create(dir.path()).unwrap();
    let reloaded = idx2.entry_by_path(&neuron).unwrap();
    assert_eq!(reloaded.hit_count, 1);
    assert_eq!(
        reloaded
            .use_count
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );
    assert_eq!(
        idx2.feedback
            .coactivation_counts
            .get(&neuron)
            .and_then(|terms| terms.get("relocation")),
        Some(&1)
    );
}

// ── upsert ────────────────────────────────────────────────────────────────

#[test]
fn upsert_neuron_persists_to_disk() {
    let dir = TempDir::new().unwrap();
    std::fs::write(dir.path().join("a.rs"), "fn a() {}").unwrap();
    let mut idx = make_index(&dir);
    idx.compile().unwrap();

    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let np = ndir.join("test.context.md");
    let content = "Cache invalidation pattern. Evicts stale entries on hash change.";
    std::fs::write(&np, content).unwrap();
    let mut meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    meta.status = NeuronStatus::Fresh;
    idx.upsert_neuron(&np, content, &meta).unwrap();
    drop(idx);

    let idx2 = NeuronIndex::load_or_create(dir.path()).unwrap();
    assert!(idx2.retrieval.entries.iter().any(|e| e.neuron_path == np));
}

// ── WAL crash recovery ────────────────────────────────────────────────────

#[test]
fn wal_replays_pending_entries_after_simulated_crash() {
    // Simulate: neurons were indexed and WAL written, but process crashed before
    // full index.json write. On reload, WAL entries should be recovered.
    //
    // Note: WAL-append mode activates only when delta_len <= delta_base/4 (i.e.,
    // the WAL is small relative to the base index). We need ≥5 base entries so that
    // adding 1 new entry stays under the 25% compaction threshold.
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();

    // Step 1: build an index with 6 neurons — delta_base=6, threshold=1,
    // so adding 1 more entry (delta_len=1 ≤ 1) stays in WAL-append mode.
    let mut idx = make_index(&dir);
    for i in 0..6u8 {
        let np = ndir.join(format!("n{i}.context.md"));
        let content = format!("neuron {i} authentication token");
        std::fs::write(&np, &content).unwrap();
        let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
        idx.index_neuron(&np, &content, &meta);
    }
    idx.rebuild_derived();
    idx.save().unwrap();
    assert_eq!(idx.retrieval.entries.len(), 6);

    // Step 2: add one new neuron — should use WAL-append mode, not full save.
    let neuron_new = ndir.join("new_entry.context.md");
    std::fs::write(&neuron_new, "cache eviction strategy").unwrap();
    let meta_new = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    idx.index_neuron(&neuron_new, "cache eviction strategy", &meta_new);
    idx.rebuild_derived();
    idx.save().unwrap();
    assert_eq!(idx.retrieval.entries.len(), 7);

    let cortyx_dir = dir.path().join(".cortyx");
    let wal_file = cortyx_dir.join("index.wal");
    assert!(wal_file.exists(), "WAL should exist after WAL-append save");

    // Step 3: simulate crash — restore index.json to the 6-entry base state.
    // In a real crash, index.json = last full save (6 entries), WAL = pending (1 entry).
    let index_json = cortyx_dir.join("index.json");
    let _ = std::fs::remove_file(activation_cache_path(dir.path()));
    let base_entry = &idx.retrieval.entries[0];
    let base_json = serde_json::to_string_pretty(&serde_json::json!({
        "version": 9u32,
        "cache_generation": 1u64,
        "entries": &idx.retrieval.entries[..6],
    }))
    .unwrap();
    let _ = base_entry; // suppress unused warning
    std::fs::write(&index_json, &base_json).unwrap();
    std::fs::write(
        cortyx_dir.join("index.checksum"),
        crc32fast::hash(base_json.as_bytes()).to_le_bytes(),
    )
    .unwrap();
    assert!(
        wal_file.exists(),
        "WAL must still be present after truncating index.json"
    );

    // Step 4: reload — WAL replay must recover the 7th entry.
    let idx2 = NeuronIndex::load_or_create(dir.path()).unwrap();
    assert_eq!(
        idx2.retrieval.entries.len(),
        7,
        "WAL replay must recover the pending entry"
    );
    assert!(
        idx2.retrieval
            .entries
            .iter()
            .any(|e| e.neuron_path == neuron_new),
        "new_entry neuron must be recovered from WAL"
    );
}

#[test]
fn wal_with_corrupt_checksum_is_discarded_gracefully() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();

    let neuron_a = ndir.join("alpha.context.md");
    std::fs::write(&neuron_a, "authentication token").unwrap();
    let mut idx = make_index(&dir);
    let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    idx.index_neuron(&neuron_a, "authentication token", &meta);
    idx.rebuild_derived();
    idx.save().unwrap();
    drop(idx);

    // Write a corrupt WAL file (invalid checksum — simulates a partial write).
    let wal_path = dir.path().join(".cortyx").join("index.wal");
    let garbage = b"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxthis is not a valid wal payload";
    std::fs::write(&wal_path, garbage).unwrap();

    // Loading must succeed without panicking; corrupt WAL is silently discarded.
    let idx2 = NeuronIndex::load_or_create(dir.path()).unwrap();
    assert_eq!(
        idx2.retrieval.entries.len(),
        1,
        "should load from index.json after discarding corrupt WAL"
    );
}

#[test]
fn get_contexts_returns_sorted_paths() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();

    let mut idx = make_index(&dir);
    for (name, content) in [
        ("z.context.md", "authentication login"),
        ("a.context.md", "auth login token"),
    ] {
        let p = ndir.join(name);
        std::fs::write(&p, content).unwrap();
        let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
        idx.index_neuron(&p, content, &meta);
    }
    idx.rebuild_derived();

    let result = idx.get_contexts("auth login", 4096, None, None);
    assert!(!result.is_empty());
    let sorted = {
        let mut r = result.clone();
        r.sort();
        r
    };
    assert_eq!(result, sorted, "output must be lexicographically sorted");
}
