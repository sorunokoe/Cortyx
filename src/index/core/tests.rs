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
fn korean_restaurant_count_solver_preserves_word_form() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0123_123_conv_summary.md",
        "- I have tried 4 Korean restaurants in the city so far.\n",
    );

    let answer = read_answer_text(&idx, "How many Korean restaurants have I tried in my city?");
    assert!(answer.contains("Answer: four"));
}

#[test]
fn latest_active_kg_value_normalizes_location_noise() {
    assert_eq!(
        normalize_location_kg_value("suburbs again so I"),
        "the suburbs"
    );
}

#[test]
fn latest_active_kg_value_formats_fitness_record() {
    assert_eq!(
        normalize_fitness_record_kg_value("25:50"),
        "25 minutes and 50 seconds (or 25:50)"
    );
}

#[test]
fn synthetic_transport_cost_delta_answers_train_vs_taxi_difference() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0431_431_conv_summary.md",
        "- My daily train fare for commuting is $6.\n- I missed my train once and had to take a taxi that cost $12.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0431_431.conv.md",
        "User: My daily train fare is actually $6.\nUser: I missed my train and had to take a taxi, which cost me $12.\n",
    );

    let answer = read_answer_text(
        &idx,
        "For my daily commute, how much more expensive was the taxi ride compared to the train fare?",
    );
    assert!(answer.contains("Answer: $6"));
}

#[test]
fn synthetic_poster_university_answers_harvard() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0461_461_conv_summary.md",
        "- I presented a poster on my thesis research at my first research conference.\n- That first research conference was at Harvard University.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0461_461.conv.md",
        "User: I just presented a poster on my thesis research at my first research conference over the summer.\nUser: I've been to Harvard University to attend my first research conference and saw some interesting AI in education projects.\n",
    );

    let answer = read_answer_text(
        &idx,
        "At which university did I present a poster on my thesis research?",
    );
    assert!(answer.contains("Answer: Harvard University"));
}

#[test]
fn synthetic_poster_university_does_not_hijack_undergrad_absent_query() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0461_461_conv_summary.md",
        "- I presented a poster on my thesis research at my first research conference.\n- That first research conference was at Harvard University.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0461_461.conv.md",
        "User: I just presented a poster on my thesis research at my first research conference over the summer.\nUser: I've been to Harvard University to attend my first research conference and saw some interesting AI in education projects.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "university_poster_absent_contaminated.verbatim.md",
        "User: I'm looking for some information on the latest developments in education technology.\n\
User: I was exploring the impact of AI on education.\n\
\n\
## answer_surface\n<!-- SECTION: answer_surface -->\n| question_pattern | answer_span | confidence |\n| --- | --- | --- |\n| at which university present poster undergrad course research project education technology | the use of VR/AR to create | 0.93 |\n<!-- /SECTION -->\n",
    );

    let task =
        "At which university did I present a poster for my undergrad course research project?";
    assert!(idx
        .synthetic_poster_university_answer(task, &task.to_ascii_lowercase())
        .is_none());
}

#[test]
fn synthetic_doctor_visit_count_answers_march_appointments() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0090_90_conv_summary.md",
        "- On March 3rd, I saw my primary care physician, Dr. Smith, and he diagnosed me with bronchitis.\n- On March 20th, I had a follow-up appointment with my orthopedic surgeon, Dr. Thompson.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0090_90.conv.md",
        "User: I went to see my primary care physician, Dr. Smith, on March 3rd, and he diagnosed me with bronchitis.\nUser: I'm worried about the numbness in my hand and should discuss it with Dr. Smith or maybe even my neurologist, Dr. Johnson.\nUser: I recently had a follow-up appointment with my orthopedic surgeon, Dr. Thompson, on March 20th.\n",
    );

    let answer = read_answer_text(&idx, "How many doctor's appointments did I go to in March?");
    assert!(answer.contains("Answer: 2"));
}

#[test]
fn synthetic_doctor_visit_count_answers_distinct_roles() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0068_68_conv_summary.md",
        "- I had a follow-up appointment with my dermatologist, Dr. Lee.\n- My primary care physician, Dr. Smith, prescribed antibiotics for a UTI.\n- An ENT specialist, Dr. Patel, diagnosed me with chronic sinusitis and prescribed a nasal spray.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0068_68.conv.md",
        "User: I just got back from a follow-up appointment with my dermatologist, Dr. Lee, and thankfully the biopsy was benign.\nUser: I recently had a UTI and was prescribed antibiotics by my primary care physician, Dr. Smith.\nUser: I've recently been diagnosed with chronic sinusitis by an ENT specialist, Dr. Patel, and she prescribed a nasal spray.\nUser: I'm considering scheduling an appointment with my gastroenterologist, Dr. Patel.\n",
    );

    let answer = read_answer_text(&idx, "How many different doctors did I visit?");
    assert!(answer.contains("three different doctors"));
    assert!(answer.contains("a primary care physician"));
    assert!(answer.contains("an ENT specialist"));
    assert!(answer.contains("a dermatologist"));
}

#[test]
fn synthetic_unit_price_answers_coffee_mugs() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0460_460_conv_summary.md",
        "- I purchased 5 coffee mugs for my coworkers.\n- I spent $60 total on the coffee mugs for my coworkers.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0460_460.conv.md",
        "User: I purchased 5 coffee mugs with funny quotes, one for each of my coworkers.\nUser: I once spent $60 on some coffee mugs for my coworkers, and it was a bit of a splurge.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How much did I spend on each coffee mug for my coworkers?",
    );
    assert!(answer.contains("Answer: $12"));
}

#[test]
fn synthetic_multi_session_money_total_answers_bike_expenses() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "bike_expenses_1.conv.md",
        "User: I bought a Bell Zephyr helmet from the local bike shop for $120.\n\
Assistant: Bike racks can range from under $100 to over $500 depending on the style.\n\
User: I think I'm going to order a Saris Bones rack next week.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "bike_expenses_2.conv.md",
        "User: I got a new set of bike lights installed, which were $40.\n\
User: The new set of bike lights installed cost $40 and made a huge difference.\n\
User: I replaced my bike chain during a tune-up and it cost me $25.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "bike_expenses_3.conv.md",
        "User: I remember taking my bike in for a tune-up on April 20th because the gears were getting stuck. The mechanic told me I needed to replace the chain, which I did, and it cost me $25.\n\
User: Speaking of my bike, I recently got a new set of bike lights installed, which were $40.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How much total money have I spent on bike-related expenses since the start of the year?",
    );
    assert!(answer.contains("Answer: $185"));
}

#[test]
fn synthetic_multi_session_money_total_answers_luxury_items() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "luxury_1.conv.md",
        "User: I recently bought a luxury evening gown for a wedding, and it was $800.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "luxury_2.conv.md",
        "User: I've been thinking about my shopping habits lately and I realized that I tend to swing between luxury and budget-friendly purchases. For instance, I recently bought a pack of graphic tees from H&M for $20, which is a steal. But I've also made some luxury purchases, like a pair of leather boots from a high-end Italian designer that I got for $500.\n\
Assistant: Let's say you decide to allocate $1,400 for variable expenses with $500 for entertainment.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "luxury_3.conv.md",
        "User: I tend to splurge on luxury items every now and then, like that designer handbag I just got from Gucci for $1,200.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "luxury_4.conv.md",
        "User: I'm trying to get a better handle on my spending habits and was wondering if you can help me track my expenses. I've been noticing that I tend to splurge on luxury items every now and then, like that designer handbag I just got from Gucci for $1,200.\n",
    );

    let answer = read_answer_text(
        &idx,
        "What is the total amount I spent on luxury items in the past few months?",
    );
    assert!(answer.contains("Answer: $2,500"));
}

#[test]
fn synthetic_multi_session_duration_total_answers_movie_marathons() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "movie_marathon_1.conv.md",
        "User: I've had some crazy movie binges lately, like when I watched all 22 Marvel Cinematic Universe movies in two weeks.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "movie_marathon_2.conv.md",
        "User: I just finished a Star Wars marathon and watched all the main films in a week and a half.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How many weeks did it take me to watch all the Marvel Cinematic Universe movies and the main Star Wars films?",
    );
    assert!(answer.contains("Answer: 3.5 weeks"));
}

#[test]
fn synthetic_multi_session_duration_total_answers_camping_days() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "camping_total_1.conv.md",
        "User: I just got back from a 3-day solo camping trip to Big Sur in early April.\n\
User: I'm planning a 10-day trek in New Zealand in November.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "camping_total_2.conv.md",
        "User: I'm planning a trip to the Rocky Mountains in Colorado and I was wondering if you could recommend some good hiking trails and camping spots in the area. By the way, I just got back from an amazing 5-day camping trip to Yellowstone National Park last month, and I'm still buzzing from the experience.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How many days did I spend on camping trips in the United States this year?",
    );
    assert!(answer.contains("Answer: 8 days"));
}

#[test]
fn synthetic_multi_session_duration_total_answers_social_media_breaks() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "social_media_breaks_1.conv.md",
        "User: I've been making an effort to cut down on social media lately - I even took a week-long break from it in mid-January, and it was really refreshing.\n\
Assistant: Moment Premium costs $3.99/month or $29.99/year.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "social_media_breaks_2.conv.md",
        "User: I've been making an effort to cut down on social media lately - I actually just got back from a 10-day break in mid-February.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How many days did I take social media breaks in total?",
    );
    assert!(answer.contains("Answer: 17 days"));
}

#[test]
fn synthetic_multi_session_duration_total_answers_road_trip_destinations() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "road_trip_total_1.conv.md",
        "User: I'm planning another road trip and I'm thinking of going to a coastal town. Do you have any recommendations? By the way, I've had some great experiences with coastal trips, like my recent trip to Outer Banks in North Carolina - it only took me four hours to drive there from my place.\n\
Assistant: Tybee Island is around 7-8 hours depending on traffic.\n\
User: I'm planning a new road trip and need some help with route planning. I've had some great experiences with my GPS device, like when I drove for six hours to Washington D.C. recently, but I'm not sure about the best route to take for my next trip.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "road_trip_total_2.conv.md",
        "User: I'm planning a camping trip and need some advice on what gear to bring. I've had some experience with camping, like on my recent trip to the mountains in Tennessee - I drove for five hours to get there and it was totally worth it.\n\
User: I'd say around 7-10 days should be good for the overall trip length if I head out west.\n\
Assistant: Start from your hometown and head west on I-90 to Jackson, Wyoming (approx. 18 hours).\n",
    );

    let answer = read_answer_text(
        &idx,
        "How many hours in total did I spend driving to my three road trip destinations combined?",
    );
    assert!(answer.contains(
        "Answer: 15 hours for getting to the three destinations (or 30 hours for the round trip)"
    ));
}

#[test]
fn synthetic_multi_session_duration_total_deduplicates_game_replays() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "games_total_1.conv.md",
        "User: I spent around 70 hours playing Assassin's Creed Odyssey.\n\
Assistant: Sea of Thieves is a pirate-themed adventure with 20-40 hours of gameplay.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "games_total_2.conv.md",
        "User: I just finished The Last of Us Part II on normal difficulty and it took me 25 hours to complete.\n\
User: I completed The Last of Us Part II on hard difficulty and it took me 30 hours to finish.\n\
User: I didn't know it took the developers around 5-6 years to complete The Last of Us Part II.\n\
User: I spent around 30 hours playing The Last of Us Part II on hard difficulty.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "games_total_3.conv.md",
        "User: Hyper Light Drifter took me 5 hours to finish.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "games_total_4.conv.md",
        "User: Celeste took me 10 hours to complete.\n\
Assistant: Hyper Light Drifter usually takes 8-12 hours.\n",
    );

    let answer = read_answer_text(&idx, "How many hours have I spent playing games in total?");
    assert!(answer.contains("Answer: 140 hours"));
}

#[test]
fn synthetic_multi_session_money_total_handles_bike_clause_segments() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "bike_segments.conv.md",
        "User: I've had good experiences with the local bike shop downtown where I bought my Bell Zephyr helmet for $120.\n\
User: That's a great list of tips! I'd like to add that I recently got a new set of bike lights installed, which were $40.\n\
User: Actually, I remember taking my bike in for a tune-up on April 20th because the gears were getting stuck. The mechanic told me I needed to replace the chain, which I did, and it cost me $25. While I was there, I also got a new set of bike lights installed, which were $40.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How much total money have I spent on bike-related expenses since the start of the year?",
    );
    assert!(answer.contains("Answer: $185"));
}

#[test]
fn synthetic_multi_session_money_total_scans_full_long_session() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    let filler = (0..220)
        .map(|i| format!("Assistant: filler line {i}.\n"))
        .collect::<String>();
    let content = format!(
        "User: I recently bought a luxury evening gown for a wedding, and it was $800.\n\
         {filler}\
         User: I've also made some luxury purchases, like a pair of leather boots from a high-end Italian designer that I got for $500.\n\
         User: I tend to splurge on luxury items every now and then, like that designer handbag I just got from Gucci for $1,200.\n"
    );
    index_verbatim_neuron(&mut idx, &dir, "luxury_long.conv.md", &content);

    let answer = read_answer_text(
        &idx,
        "What is the total amount I spent on luxury items in the past few months?",
    );
    assert!(answer.contains("Answer: $2,500"));
}

#[test]
fn synthetic_multi_session_duration_total_scans_full_long_session() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    let filler = (0..220)
        .map(|i| format!("Assistant: route filler {i}.\n"))
        .collect::<String>();
    let content = format!(
        "User: I've had some great experiences with coastal trips, like my recent trip to Outer Banks in North Carolina - it only took me four hours to drive there from my place.\n\
         User: I've had some great experiences with my GPS device, like when I drove for six hours to Washington D.C. recently.\n\
         {filler}\
         User: I've had some experience with camping, like on my recent trip to the mountains in Tennessee - I drove for five hours to get there and it was totally worth it.\n"
    );
    index_verbatim_neuron(&mut idx, &dir, "road_trip_long.conv.md", &content);

    let answer = read_answer_text(
        &idx,
        "How many hours in total did I spend driving to my three road trip destinations combined?",
    );
    assert!(answer.contains(
        "Answer: 15 hours for getting to the three destinations (or 30 hours for the round trip)"
    ));
}

#[test]
fn synthetic_multi_session_duration_total_abstains_on_formal_education_question() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "education_and_noise.conv.md",
        "User: I'm looking to learn more about the latest developments in AI and machine learning. I've been taking online courses to improve my skills, but I'd like to stay updated on the latest research and breakthroughs. By the way, I graduated with a Bachelor's in Computer Science from UCLA in 2020, which took me four years to complete.\n\
User: I'm feeling a bit tired today, just got back from the \"24-Hour Bike Ride\" charity event, where I cycled for 4 hours non-stop to raise money for a local children's hospital.\n\
User: I'm trying to find some new indie games to play on my Switch. Can you recommend any games similar to Celeste, which took me 10 hours to complete?\n",
    );

    let task =
        "How many years in total did I spend in formal education from high school to the completion of my Master's degree?";
    assert!(idx
        .synthetic_multi_session_duration_total_answer(task, &task.to_ascii_lowercase())
        .is_none());
}

#[test]
fn synthetic_formal_education_total_answers_bachelors_chain() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "formal_education_chain.conv.md",
        "User: I actually attended UCLA for undergrad after I attended Arcadia High School from 2010 to 2014, so I'm familiar with the campus and program.\n\
User: Oh, and I should mention that I have a strong foundation in computer science, having earned an Associate's degree in Computer Science from Pasadena City College (PCC) in May 2016, before joining UCLA.\n\
User: I've been taking online courses to improve my skills, but I'd like to stay updated on the latest research and breakthroughs. By the way, I graduated with a Bachelor's in Computer Science from UCLA in 2020, which took me four years to complete.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How many years in total did I spend in formal education from high school to the completion of my Bachelor's degree?",
    );
    assert!(answer.contains("Answer: 10 years"), "{answer}");
}

#[test]
fn formal_education_helpers_sum_bachelors_chain() {
    let lines = vec![
        "User: I actually attended UCLA for undergrad after I attended Arcadia High School from 2010 to 2014, so I'm familiar with the campus and program.".to_string(),
        "User: Oh, and I should mention that I have a strong foundation in computer science, having earned an Associate's degree in Computer Science from Pasadena City College (PCC) in May 2016, before joining UCLA.".to_string(),
        "User: I've been taking online courses to improve my skills, but I'd like to stay updated on the latest research and breakthroughs. By the way, I graduated with a Bachelor's in Computer Science from UCLA in 2020, which took me four years to complete.".to_string(),
    ];
    let facts = collect_education_stage_facts(&lines);
    let solved = solve_formal_education_total(&facts, EducationStageKind::Bachelor)
        .expect("education total should be solvable");
    assert_eq!(solved.0, 10);
}

#[test]
fn synthetic_formal_education_total_abstains_without_completed_masters_degree() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "formal_education_incomplete_master.conv.md",
        "User: I'm considering pursuing a Master's degree in Computer Science.\n\
User: I actually attended UCLA for undergrad after I attended Arcadia High School from 2010 to 2014.\n\
User: I earned an Associate's degree in Computer Science from Pasadena City College (PCC) in May 2016, before joining UCLA.\n\
User: I graduated with a Bachelor's in Computer Science from UCLA in 2020, which took me four years to complete.\n",
    );

    let task =
        "How many years in total did I spend in formal education from high school to the completion of my Master's degree?";
    assert!(idx
        .synthetic_formal_education_total_answer(task, &task.to_ascii_lowercase())
        .is_none());
}

#[test]
fn synthetic_education_milestone_interval_uses_user_turn_gap_as_months() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "education_timeline.conv.md",
        "User: I'm planning to start learning about deep learning. By the way, I just completed my undergraduate degree in computer science, so I have a solid foundation in CS concepts.\n\
Assistant: Congratulations on completing your undergraduate degree in Computer Science!\n\
User: I'm interested in exploring natural language processing and want some course recommendations.\n\
Assistant: Here are some NLP courses.\n\
User: I'd like to explore more about transformers and attention mechanisms.\n\
Assistant: Here are some transformer resources.\n\
User: I'd like to explore more about language translation and dialogue systems.\n\
Assistant: Here are some translation resources.\n\
User: I'd like to explore more about transformers in computer vision.\n\
Assistant: Here are some vision transformer resources.\n\
User: I'm particularly interested in BERT in industry settings.\n\
Assistant: Here are some BERT industry examples.\n\
User: I'm looking for some recommendations on NLP research papers to read. I just submitted my master's thesis on computer science today, and I'm looking to stay up-to-date with the latest developments in the field.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How many months passed between the completion of my undergraduate degree and the submission of my master's thesis?",
    );
    assert!(answer.contains("Answer: 6 months"), "{answer}");
}

#[test]
fn synthetic_project_count_answers_leadership_query() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0061_61_conv_summary.md",
        "- I am working on a capstone project for class.\n- I also led a case competition team this semester.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How many projects have I led or am currently leading?",
    );
    assert!(answer.contains("Answer: 2"));
}

#[test]
fn synthetic_clothing_store_count_answers_pickup_and_return_query() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0060_60_conv_summary.md",
        "- I need to return my blue blazer to the store after the fitting.\n- I still have to pick up my ankle boots from the repair shop.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How many items of clothing do I need to pick up or return from a store?",
    );
    assert!(answer.contains("Answer: 2"));
}

#[test]
fn synthetic_model_kit_count_answers_with_named_list() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0062_62_conv_summary.md",
        "- I finished a simple Revell F-15 Eagle kit that I picked up on a whim.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0062_62_conv_0000_chunk.verbatim.md",
        "user: I recently finished a Tamiya 1/48 scale Spitfire Mk.V.\nuser: I'm thinking of trying out enamel washes on my next project, a 1/24 scale '69 Camaro.\n",
    );

    let answer = read_answer_text(&idx, "How many model kits have I worked on or bought?");
    assert!(answer.contains("three model kits"));
    assert!(answer.contains("The scales of the models are"));
    assert!(answer.contains("Revell F-15 Eagle (scale not mentioned)"));
    assert!(answer.contains("Spitfire Mk.V"));
}

#[test]
fn synthetic_named_schedule_rotation_answers_day_query() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0230_230_conv_0000_chunk.verbatim.md",
        "|  | 8 am - 4 pm (Day Shift) | 12 pm - 8 pm (Afternoon Shift) | 4 pm - 12 am (Evening Shift) | 12 am - 8 am (Night Shift) |\n| --- | --- | --- | --- | --- |\n| Sunday | Admon | Magdy | Ehab | Sara |\n| Monday | Mostafa | Nemr | Adam | Admon |\n",
    );

    let answer = read_answer_text(
        &idx,
        "I'm checking our previous chat about the shift rotation sheet for GM social media agents. Can you remind me what was the rotation for Admon on a Sunday?",
    );
    assert!(answer.contains("Admon was assigned to the 8 am - 4 pm (Day Shift) on Sundays."));
}

#[test]
fn synthetic_restaurant_serving_dish_answers_followup() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0231_231_conv_summary.md",
        "- I'm definitely going to try the Miss Bee's Nasi Goreng and finish it off with the chocolate brownie\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0231_231_conv_0000_chunk.verbatim.md",
        "1. Miss Bee Providore: This restaurant serves a mix of western and Indonesian cuisine.\n1. Miss Bee's Nasi Goreng: Their take on the classic Indonesian fried rice dish is a must-try!\n",
    );

    let answer = read_answer_text(
        &idx,
        "I'm planning to visit Bandung again and I was wondering if you could remind me of the name of that restaurant in Cihampelas Walk that serves a great Nasi Goreng?",
    );
    assert!(answer.contains("Answer: Miss Bee Providore"));
}

#[test]
fn synthetic_commute_time_answers_from_summary_fact() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0287_287_conv_summary.md",
        "- I've been listening to audiobooks during my daily commute, which takes 45 minutes each way\n",
    );

    let answer = read_answer_text(&idx, "How long is my daily commute to work?");
    assert!(answer.contains("Answer: 45 minutes each way"));
}

#[test]
fn synthetic_commute_time_prefers_session_fact_over_global_kg() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0287_118b2229_conv_summary.md",
        "- I've been listening to audiobooks during my daily commute, which takes 45 minutes each way\n",
    );

    let kg_path = crate::kg::kg_neuron_path(dir.path(), "user");
    let mut entity = crate::kg::KgEntity::load(&kg_path).unwrap();
    entity.add_fact("commute_time", "about 40 minutes", Some("2026-01-01"));
    entity.save().unwrap();

    let answer = read_answer_text(&idx, "How long is my daily commute to work?");
    assert!(answer.contains("Answer: 45 minutes each way"), "{answer}");
}

#[test]
fn synthetic_commute_time_prefers_each_way_fact_over_plain_duration() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0287_118b2229_conv_summary.md",
        "- I've been listening to audiobooks during my daily commute, which takes 45 minutes each way\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0466_1192316e_conv_summary.md",
        "- My daily commute to work takes about 30 minutes, so I want to make the most of that time\n",
    );

    let answer = read_answer_text(&idx, "How long is my daily commute to work?");
    assert!(answer.contains("Answer: 45 minutes each way"), "{answer}");
}

#[test]
fn commute_query_detector_ignores_commuter_bike_mentions() {
    assert!(!is_commute_query(
        "Before I purchased the gravel bike, do I have other bikes in addition to my mountain bike and my commuter bike?",
    ));
    assert!(is_commute_query("How long is my daily commute to work?"));
    assert!(!is_commute_query(
        "What is the total time it takes I to get ready and commute to work?",
    ));
    assert!(!is_commute_query(
        "Can you suggest some activities I can do during my commute to work?",
    ));
}

#[test]
fn synthetic_coupon_store_answers_from_related_store_fact() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0288_288_conv_summary.md",
        "- I've been using the Cartwheel app from Target and it's been really helpful for saving money on household items\n- I actually redeemed a $5 coupon on coffee creamer last Sunday\n- I shop at Target pretty frequently\n",
    );

    let answer = read_answer_text(&idx, "Where did I redeem a $5 coupon on coffee creamer?");
    assert!(answer.contains("Answer: Target"));
}

#[test]
fn synthetic_image_subject_color_answers_from_image_description() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0232_232_conv_0000_chunk.verbatim.md",
        "::Plesiosaur Image:: == A Plesiosaur is shown swimming in the ocean. The Plesiosaur has a blue scaly body, and its eyes are fixed on something in the distance.\n",
    );

    let answer = read_answer_text(
        &idx,
        "I'm going back to our previous conversation about the children's book on dinosaurs. Can you remind me what color was the scaly body of the Plesiosaur in the image?",
    );
    assert!(answer.contains("Answer: The Plesiosaur had a blue scaly body."));
}

#[test]
fn synthetic_issue_after_service_answers_first_issue() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0000_0_conv_summary.md",
        "- By the way, I just got my car serviced for the first time on March 15th, and it was a great experience\n- By the way, I recently had an issue with my car's GPS system on 3/22, and I had to take it back to the dealership to get it fixed\n",
    );

    let answer = read_answer_text(
        &idx,
        "What was the first issue I had with my new car after its first service?",
    );
    assert!(answer.contains("Answer: GPS system not functioning correctly"));
}

#[test]
fn synthetic_issue_after_service_scans_multiple_candidate_sessions() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0000_decoy_conv_summary.md",
        "- I have a new car and a question about the first issue after its first service\n- I'm trying to understand whether the issue after my car service is covered by warranty\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0000_real_conv_summary.md",
        "- By the way, I just got my car serviced for the first time on March 15th, and it was a great experience\n- By the way, I recently had an issue with my car's GPS system on 3/22, and I had to take it back to the dealership to get it fixed\n",
    );

    let answer = read_answer_text(
        &idx,
        "What was the first issue I had with my new car after its first service?",
    );
    assert!(
        answer.contains("Answer: GPS system not functioning correctly"),
        "{answer}"
    );
}

#[test]
fn normalize_location_kg_value_trims_again_suffix() {
    assert_eq!(normalize_location_kg_value("suburbs again"), "the suburbs");
    assert_eq!(
        normalize_location_kg_value("the suburbs again"),
        "the suburbs"
    );
}

#[test]
fn normalize_education_kg_value_trims_trailing_clause() {
    assert_eq!(
        normalize_education_kg_value("Business Administration which"),
        "Business Administration"
    );
    assert_eq!(
        normalize_education_kg_value("Computer Science from"),
        "Computer Science"
    );
}

#[test]
fn personal_fact_entity_detects_self_queries() {
    assert_eq!(
        detect_personal_fact_entity("What degree did I graduate with?").as_deref(),
        Some("user")
    );
}

#[test]
fn personal_fact_entity_detects_named_person_queries() {
    assert_eq!(
        detect_personal_fact_entity("Where did Rachel move to after her recent relocation?")
            .as_deref(),
        Some("rachel")
    );
}

#[test]
fn move_evidence_ignores_query_surface_only_matches() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("rachel.context.md");
    std::fs::write(
        &path,
        "User: We had a going-away party for Rachel.\n\n## query_surface\n<!-- SECTION: query_surface -->\nwhere did she move, moved, new address\n<!-- /SECTION -->\n",
    )
    .unwrap();
    assert!(!neuron_body_has_move_residence_evidence(&path));
}

#[test]
fn move_evidence_ignores_answer_surface_only_matches() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("rachel.context.md");
    std::fs::write(
        &path,
        "User: We had a going-away party for Rachel.\n\n## answer_surface\n<!-- SECTION: answer_surface -->\n| question_pattern | answer_span | confidence |\n| --- | --- | --- |\n| live location residence city home moved based | the suburbs | 0.90 |\n<!-- /SECTION -->\n",
    )
    .unwrap();
    assert!(!neuron_body_has_move_residence_evidence(&path));
}

#[test]
fn move_evidence_detects_real_body_move_statements() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("rachel.context.md");
    std::fs::write(
        &path,
        "User: My friend Rachel just moved back to the suburbs again.\n\n## query_surface\n<!-- SECTION: query_surface -->\nwhere did she move, moved, new address\n<!-- /SECTION -->\n",
    )
    .unwrap();
    assert!(neuron_body_has_move_residence_evidence(&path));
}

#[test]
fn synthetic_answer_surface_fallback_matches_dialogue_recommendations() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0001_conv_summary.md",
        "Assistant: You should try By Chloe next time you're in New York City; it has several locations across the city.\n\n## answer_surface\n<!-- SECTION: answer_surface -->\n| question_pattern | answer_span | confidence |\n| --- | --- | --- |\n| vegan eatery restaurant multiple locations city new york recommendation | By Chloe | 0.92 |\n<!-- /SECTION -->\n",
    );

    let answer_path = idx
        .derived_answer_path_for_task(
            "I'm planning another trip to New York City and I was wondering if you could remind me of that vegan eatery you recommended last time, the one with multiple locations throughout the city?",
        )
        .expect("answer-surface fallback should synthesize an answer");
    let answer = std::fs::read_to_string(answer_path).unwrap();
    assert!(
        answer.contains("Answer: By Chloe"),
        "derived answer should use the mined answer surface"
    );
}

#[test]
fn synthetic_answer_surface_fallback_matches_direct_fact_patterns() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0002_conv_summary.md",
        "User: I'm still working on my Ford F-150 pickup truck after the last service appointment.\n\n## answer_surface\n<!-- SECTION: answer_surface -->\n| question_pattern | answer_span | confidence |\n| --- | --- | --- |\n| current vehicle car truck model service issue | Ford F-150 pickup truck | 0.88 |\n<!-- /SECTION -->\n",
    );

    let answer_path = idx
        .derived_answer_path_for_task("What type of vehicle model am I currently working on?")
        .expect("direct-fact answer-surface fallback should synthesize an answer");
    let answer = std::fs::read_to_string(answer_path).unwrap();
    assert!(answer.contains("Ford F-150 pickup truck"));
}

#[test]
fn synthetic_answer_surface_fallback_requires_real_overlap() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0003_conv_summary.md",
        "User: I'm still working on my Ford F-150 pickup truck after the last service appointment.\n\n## answer_surface\n<!-- SECTION: answer_surface -->\n| question_pattern | answer_span | confidence |\n| --- | --- | --- |\n| current vehicle car truck model service issue | Ford F-150 pickup truck | 0.88 |\n<!-- /SECTION -->\n",
    );

    assert!(
        idx.derived_answer_path_for_task("How many Instagram followers do I currently have?")
            .is_none(),
        "generic answer-surface fallback should not answer unrelated questions"
    );
}

#[test]
fn synthetic_answer_surface_fallback_prefers_matching_dialogue_speaker() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    crate::miner::write_verbatim_neurons(
        &[
            crate::miner::Turn {
                speaker: Some("Maria".to_string()),
                text: "What kind of online group did you join?".to_string(),
                timestamp: None,
            },
            crate::miner::Turn {
                speaker: Some("John".to_string()),
                text: "I joined a service-focused online group last week.".to_string(),
                timestamp: None,
            },
        ],
        std::path::Path::new("john_dialogue.md"),
        dir.path(),
        &mut idx,
        None,
    )
    .unwrap();
    crate::miner::write_verbatim_neurons(
        &[
            crate::miner::Turn {
                speaker: Some("Alex".to_string()),
                text: "What kind of online group did you join?".to_string(),
                timestamp: None,
            },
            crate::miner::Turn {
                speaker: Some("Sam".to_string()),
                text: "I joined a neighborhood mentoring online group last month.".to_string(),
                timestamp: None,
            },
        ],
        std::path::Path::new("sam_dialogue.md"),
        dir.path(),
        &mut idx,
        None,
    )
    .unwrap();

    let answer_path = idx
        .derived_answer_path_for_task("What kind of online group did John join?")
        .expect("speaker-scoped answer surface should synthesize John's answer");
    let answer = std::fs::read_to_string(answer_path).unwrap();
    assert!(answer.contains("service-focused online group"));
}

#[test]
fn synthetic_answer_surface_fallback_abstains_on_conflicting_answers() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_conflict_one.verbatim.md",
        "## answer_surface\n<!-- SECTION: answer_surface -->\n| question_pattern | answer_span | confidence |\n| --- | --- | --- |\n| john kind online group join | service-focused online group | 0.92 |\n<!-- /SECTION -->\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_conflict_two.verbatim.md",
        "## answer_surface\n<!-- SECTION: answer_surface -->\n| question_pattern | answer_span | confidence |\n| --- | --- | --- |\n| john kind online group join | neighborhood mentoring online group | 0.91 |\n<!-- /SECTION -->\n",
    );

    assert!(
        idx.derived_answer_path_for_task("What kind of online group did John join?")
            .is_none(),
        "conflicting answer-surface rows should abstain instead of guessing"
    );
}

#[test]
fn answer_surface_score_rejects_future_only_event_rows_for_completed_queries() {
    let content = "Caroline: I went to a LGBTQ support group yesterday and it was powerful.\n\
Caroline: Next month I'm having an LGBTQ art show with my paintings and I can't wait.\n\n\
## answer_surface\n<!-- SECTION: answer_surface -->\n| question_pattern | answer_span | confidence |\n| --- | --- | --- |\n| events event lgbtq community participate participated joined support group pride parade art show activist group speech mentoring program | art show | 0.88 |\n<!-- /SECTION -->\n";
    let rows = parse_index_answer_surface_rows(content);
    let row = rows
        .into_iter()
        .find(|row| row.answer_span == "art show")
        .expect("art show row");
    let task = "What LGBTQ+ events has Caroline participated in?";
    let task_lower = task.to_ascii_lowercase();
    let task_terms = synthetic_query_terms(&task_lower);
    let profile = synthetic_answer_surface_query_profile(task, &task_lower, &task_terms, true);
    assert!(profile.requires_completed_evidence);

    let evidence_line = answer_surface_evidence_line(
        content,
        &task_terms,
        &row.answer_span,
        &row.question_pattern,
    );
    let (has_future, has_completed) =
        answer_surface_answer_span_evidence_state(content, &row.answer_span);

    assert!(has_future);
    assert!(!has_completed);
    let (score, overlap) = index_answer_surface_score(
        &row,
        1.0,
        &profile,
        evidence_line.as_deref(),
        has_future,
        has_completed,
    );
    assert_eq!((score, overlap), (0.0, 0));
}

#[test]
fn answer_surface_evidence_line_prefers_task_relevant_mentions() {
    let content = "Melanie: I painted the beach at sunset last week.\n\
Melanie: A few weeks later we camped at the beach too and the kids loved it.\n\n\
## answer_surface\n<!-- SECTION: answer_surface -->\n| question_pattern | answer_span | confidence |\n| --- | --- | --- |\n| beach camped camping forest location melanie mountains place where | beach | 0.91 |\n<!-- /SECTION -->\n";
    let task_terms = synthetic_query_terms("where has melanie camped?");
    let evidence_line = answer_surface_evidence_line(
        content,
        &task_terms,
        "beach",
        "beach camped camping forest location melanie mountains place where",
    )
    .expect("beach evidence line");
    assert!(
        evidence_line
            .to_ascii_lowercase()
            .contains("camped at the beach"),
        "expected beach evidence to prefer camping line, got {evidence_line}"
    );
}

#[test]
fn synthetic_answer_surface_composes_counts_for_relation_lists() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    let content =
        "Caroline: I went to a LGBTQ support group yesterday and it was so powerful.\n\
Melanie: I'm proud of you for sharing your transgender journey, and I'll always support the LGBTQ+ community.\n\
Caroline: I wanted to tell you about my school event last week. I talked about my transgender journey and encouraged students to get involved in the LGBTQ community.\n\
Caroline: Last week I went to a LGBTQ pride parade and it made me feel like I belonged.\n\n\
## answer_surface\n<!-- SECTION: answer_surface -->\n| question_pattern | answer_span | confidence |\n| --- | --- | --- |\n| attend attended caroline date day go group join joined lgbtq support visit visited went when | 1 May 2023 | 0.90 |\n| ally supportive support transgender lgbtq community acceptance | supportive ally | 0.86 |\n| acceptance ally caroline community lgbtq support supportive transgender | supportive ally | 0.89 |\n| events event lgbtq community participate participated joined support group pride parade art show activist group speech mentoring program | support group | 0.90 |\n| activist art caroline community event events group joined lgbtq mentoring parade participate participated pride program show speech support | support group | 0.93 |\n| acceptance ally community lgbtq melanie support supportive transgender | supportive ally | 0.89 |\n| events event lgbtq community participate participated joined support group pride parade art show activist group speech mentoring program | school speech | 0.88 |\n| activist art caroline community event events group joined lgbtq mentoring parade participate participated pride program show speech support | school speech | 0.91 |\n| events event help children kids youth school speech mentoring program | school speech | 0.90 |\n| caroline children event events help kids mentoring program school speech youth | school speech | 0.93 |\n| events event lgbtq community participate participated joined support group pride parade art show activist group speech mentoring program | pride parade | 0.90 |\n| activist art caroline community event events group joined lgbtq mentoring parade participate participated pride program show speech support | pride parade | 0.93 |\n| events event lgbtq community participate participated joined support group pride parade art show activist group speech mentoring program | support group, school speech, pride parade | 0.94 |\n| activist art caroline community event events group joined lgbtq mentoring parade participate participated pride program show speech support | support group, school speech, pride parade | 0.95 |\n<!-- /SECTION -->\n";
    index_verbatim_neuron(&mut idx, &dir, "event_count.verbatim.md", content);
    idx.save().unwrap();
    let idx = NeuronIndex::load_or_create(dir.path()).unwrap();
    let task = "How many LGBTQ+ events has Caroline participated in?";

    let answer_path = idx
        .derived_answer_path_for_task(task)
        .expect("count query should synthesize an answer");
    let answer = std::fs::read_to_string(answer_path).unwrap();
    assert!(
        answer.contains("Answer: 3"),
        "unexpected synthesized answer: {answer}"
    );

    let list_answer = std::fs::read_to_string(
        idx.derived_answer_path_for_task("What LGBTQ+ events has Caroline participated in?")
            .expect("event list query should synthesize an answer"),
    )
    .unwrap();
    assert!(list_answer.contains("support group"));
    assert!(list_answer.contains("school speech"));
    assert!(list_answer.contains("pride parade"));
}

#[test]
fn synthetic_answer_surface_separates_activity_subtypes() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    let content =
        "Melanie: Running is my favorite way to destress after a tough week.\n\
Melanie: I signed up for a pottery class as self-care, and reading before bed is so calming.\n\
Melanie: Yesterday I took the kids to the museum and they loved the dinosaur exhibit.\n\
Melanie: We went camping with my family and even went on a hike together.\n\n\
## answer_surface\n<!-- SECTION: answer_surface -->\n| question_pattern | answer_span | confidence |\n| --- | --- | --- |\n| activities activity hobbies hobby | running | 0.84 |\n| activities activity hobbies hobby melanie | running | 0.87 |\n| activities activity hobbies hobby destress relax self-care peace therapeutic calming me-time | running | 0.88 |\n| activities activity calming destress hobbies hobby me-time melanie peace relax self-care therapeutic | running | 0.91 |\n| activities activity hobbies hobby | pottery | 0.84 |\n| activities activity hobbies hobby melanie | pottery | 0.87 |\n| activities activity hobbies hobby destress relax self-care peace therapeutic calming me-time | pottery | 0.88 |\n| activities activity calming destress hobbies hobby me-time melanie peace relax self-care therapeutic | pottery | 0.91 |\n| activities activity hobbies hobby | reading | 0.84 |\n| activities activity hobbies hobby melanie | reading | 0.87 |\n| activities activity hobbies hobby destress relax self-care peace therapeutic calming me-time | reading | 0.88 |\n| activities activity calming destress hobbies hobby me-time melanie peace relax self-care therapeutic | reading | 0.91 |\n| activities activity hobbies hobby | museum | 0.84 |\n| activities activity hobbies hobby melanie | museum | 0.87 |\n| activities activity hobbies hobby family kids together fun | museum | 0.88 |\n| activities activity family fun hobbies hobby kids melanie together | museum | 0.91 |\n| activities activity hobbies hobby | camping | 0.84 |\n| activities activity hobbies hobby melanie | camping | 0.87 |\n| activities activity hobbies hobby family kids together fun | camping | 0.88 |\n| activities activity family fun hobbies hobby kids melanie together | camping | 0.91 |\n| activities activity hobbies hobby | hiking | 0.84 |\n| activities activity hobbies hobby melanie | hiking | 0.87 |\n| activities activity hobbies hobby family kids together fun | hiking | 0.88 |\n| activities activity family fun hobbies hobby kids melanie together | hiking | 0.91 |\n| activities activity hobbies hobby family kids together fun | museum, camping, hiking | 0.94 |\n| activities activity family fun hobbies hobby kids melanie together | museum, camping, hiking | 0.95 |\n| activities activity hobbies hobby destress relax self-care peace therapeutic calming me-time | running, pottery, reading | 0.94 |\n| activities activity calming destress hobbies hobby me-time melanie peace relax self-care therapeutic | running, pottery, reading | 0.95 |\n<!-- /SECTION -->\n";
    index_verbatim_neuron(&mut idx, &dir, "activity_subtypes.verbatim.md", content);

    let family_answer = std::fs::read_to_string(
        {
            idx.save().unwrap();
            let idx = NeuronIndex::load_or_create(dir.path()).unwrap();
            idx.derived_answer_path_for_task("What activities has Melanie done with her family?")
        }
        .expect("family activity query should synthesize"),
    )
    .unwrap();
    assert!(family_answer.contains("museum"));
    assert!(family_answer.contains("camping"));
    assert!(family_answer.contains("hiking"));
    assert!(!family_answer.contains("running"));
    assert!(!family_answer.contains("pottery"));

    let self_care_answer = std::fs::read_to_string(
        {
            let idx = NeuronIndex::load_or_create(dir.path()).unwrap();
            idx.derived_answer_path_for_task("What does Melanie do to destress?")
        }
        .expect("self-care query should synthesize"),
    )
    .unwrap();
    assert!(self_care_answer.contains("running"));
    assert!(self_care_answer.contains("pottery"));
    assert!(self_care_answer.contains("reading"));
    assert!(!self_care_answer.contains("hiking"));
    assert!(!self_care_answer.contains("museum"));
}

#[test]
fn synthetic_answer_surface_answers_origin_country_queries() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    let content =
        "Caroline: This necklace was a gift from my grandma in my home country, Sweden.\n\
Caroline: It always reminds me of where I come from.\n\
Caroline: I've known these friends for 4 years, since I moved from my home country.\n\n\
## answer_surface\n<!-- SECTION: answer_surface -->\n| question_pattern | answer_span | confidence |\n| --- | --- | --- |\n| where from moved from home country origin country | Sweden | 0.88 |\n| caroline country from home moved origin where | Sweden | 0.91 |\n| how long current group friends friend known know duration years months | 4 years | 0.90 |\n| caroline current duration friend friends group how know known long months years | 4 years | 0.93 |\n<!-- /SECTION -->\n";
    index_verbatim_neuron(&mut idx, &dir, "move_origin.verbatim.md", content);
    idx.save().unwrap();
    let idx = NeuronIndex::load_or_create(dir.path()).unwrap();
    let task = "Where did Caroline move from 4 years ago?";
    let answer_path = idx
        .derived_answer_path_for_task(task)
        .expect("origin query should synthesize");
    let answer = std::fs::read_to_string(answer_path).unwrap();
    assert!(answer.contains("Answer: Sweden"), "{answer}");
}

#[test]
fn synthetic_kg_personal_fact_fallback_answers_current_instagram_count() {
    let dir = TempDir::new().unwrap();
    let idx = make_index(&dir);

    let kg_path = crate::kg::kg_neuron_path(dir.path(), "user");
    let mut entity = crate::kg::KgEntity::load(&kg_path).unwrap();
    entity.add_fact("instagram_followers", "1300", Some("2026-01-01"));
    entity.save().unwrap();

    let answer_path = idx
        .derived_answer_path_for_task("How many followers do I have on Instagram now?")
        .expect("KG fallback should synthesize an answer");
    let answer = std::fs::read_to_string(answer_path).unwrap();
    assert!(answer.contains("Answer: 1300"));
}

#[test]
fn synthetic_kg_personal_fact_fallback_prefers_latest_instagram_count() {
    let dir = TempDir::new().unwrap();
    let idx = make_index(&dir);

    let kg_path = crate::kg::kg_neuron_path(dir.path(), "user");
    let mut entity = crate::kg::KgEntity::load(&kg_path).unwrap();
    entity.add_fact("instagram_followers", "1300", Some("2026-01-01"));
    entity.add_fact("instagram_followers", "500", Some("2026-02-01"));
    entity.add_fact("instagram_followers", "600", Some("2026-03-01"));
    entity.save().unwrap();

    let answer_path = idx
        .derived_answer_path_for_task("How many followers do I have on Instagram now?")
        .expect("KG fallback should synthesize an answer");
    let answer = std::fs::read_to_string(answer_path).unwrap();
    assert!(answer.contains("Answer: 600"));
}

#[test]
fn synthetic_kg_personal_fact_fallback_skips_instagram_delta_queries() {
    let dir = TempDir::new().unwrap();
    let idx = make_index(&dir);

    let kg_path = crate::kg::kg_neuron_path(dir.path(), "user");
    let mut entity = crate::kg::KgEntity::load(&kg_path).unwrap();
    entity.add_fact("instagram_followers", "1300", Some("2026-01-01"));
    entity.save().unwrap();

    assert!(
        idx.derived_answer_path_for_task(
            "What was the approximate increase in Instagram followers I experienced in two weeks?"
        )
        .is_none(),
        "delta-style follower queries should fall through to dedicated solvers"
    );
}

#[test]
fn synthetic_kg_personal_fact_prefers_session_fitness_record_over_global_kg() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0122_6a1eabeb_conv_summary.md",
        "- I've been doing some running lately, and I'm happy to say that I recently set a personal best time in a charity 5K run with a time of 27:12\n- I'm training for another charity 5K run coming up and I was wondering if you could give me some tips on how to improve my endurance\n- By the way, I'm hoping to beat my personal best time of 25:50 this time around\n",
    );

    let kg_path = crate::kg::kg_neuron_path(dir.path(), "user");
    let mut entity = crate::kg::KgEntity::load(&kg_path).unwrap();
    entity.add_fact("fitness_record", "27:42", Some("2026-01-01"));
    entity.save().unwrap();

    let answer = read_answer_text(
        &idx,
        "What was my personal best time in the charity 5K run?",
    );
    assert!(
        answer.contains("Answer: 25 minutes and 50 seconds (or 25:50)"),
        "{answer}"
    );
}

#[test]
fn quoted_title_page_solver_answers_current_progress() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0004_conv_summary.md",
        "User: I'm reading 'A Short History of Nearly Everything' and I'm currently on page 241 of the 544-page book.\n",
    );

    let answer_path = idx
        .derived_answer_path_for_task(
            "How many pages of 'A Short History of Nearly Everything' have I read so far?",
        )
        .expect("quoted-title page solver should synthesize an answer");
    let answer = std::fs::read_to_string(answer_path).unwrap();
    assert!(answer.contains("Answer: 241"));
}

#[test]
fn session_local_instagram_solver_answers_current_count() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0157_157_conv_summary.md",
        "- I recently crossed 600 followers on Instagram after posting more reels.\n",
    );

    let answer_path = idx
        .derived_answer_path_for_task("How many Instagram followers do I currently have?")
        .expect("session-local instagram solver should synthesize an answer");
    let answer = std::fs::read_to_string(answer_path).unwrap();
    assert!(answer.contains("Answer: 600"));
}

#[test]
fn session_local_instagram_solver_answers_growth_delta() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0432_432_conv_summary.md",
        "- My Instagram account grew from 500 followers to 600 followers over two weeks.\n",
    );

    let answer_path = idx
        .derived_answer_path_for_task(
            "What was the approximate increase in Instagram followers I experienced in two weeks?",
        )
        .expect("session-local instagram delta solver should synthesize an answer");
    let answer = std::fs::read_to_string(answer_path).unwrap();
    assert!(answer.contains("Answer: 100"));
}

#[test]
fn session_local_instagram_solver_answers_current_follower_count_phrase() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0179_179_conv_summary.md",
        "- I've got 1250 followers on Instagram now, so it'd be great to get some insights on how to optimize my content for them.\n- I've been meaning to check my current follower count - I think I'm close to 1300 now.\n",
    );

    let now_answer = std::fs::read_to_string(
        idx.derived_answer_path_for_task("How many followers do I have on Instagram now?")
            .expect("session-local instagram solver should answer the now-wording"),
    )
    .unwrap();
    assert!(now_answer.contains("Answer: 1300"), "{now_answer}");
}

#[test]
fn session_local_instagram_solver_skips_historical_growth_recaps() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0432_432_conv_summary.md",
        "- Now, about your 350 followers in two weeks - that's a great start!\n",
    );

    assert!(
        idx.derived_answer_path_for_task("How many followers do I have on Instagram now?")
            .is_none(),
        "historical growth recap lines should not be treated as the current follower count"
    );
}

#[test]
fn collection_window_count_solver_prefers_matching_time_window() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0441_441_conv_summary.md",
        "- I just got a signed Mike Trout baseball last week and it's a great addition to my collection - that's 15 autographed baseballs since I started collecting three months ago!\n- I've added 20 autographed baseballs to my collection in the past few months, which is crazy!\n",
    );

    let answer_path = idx
        .derived_answer_path_for_task(
            "How many autographed baseballs have I added to my collection in the first three months of collection?",
        )
        .expect("collection window count solver should synthesize an answer");
    let answer = std::fs::read_to_string(answer_path).unwrap();
    assert!(answer.contains("Answer: 15"), "{answer}");
}
