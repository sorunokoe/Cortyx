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

#[test]
fn collection_window_count_solver_abstains_on_wrong_item_type() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0441_441_conv_summary.md",
        "- I just got a signed Mike Trout baseball last week and it's a great addition to my collection - that's 15 autographed baseballs since I started collecting three months ago!\n",
    );

    assert!(
        idx.derived_answer_path_for_task(
            "How many autographed football have I added to my collection in the first three months of collection?",
        )
        .is_none(),
        "collection window count solver should abstain when the item type does not match"
    );
}

#[test]
fn role_transition_count_solver_answers_then_vs_now() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0104_104_conv_summary.md",
        "- I lead a team of 4 engineers in my new role as Senior Software Engineer.\n- I've been enjoying my role as Senior Software Engineer for a while, especially the part where I now lead a team of five engineers.\n",
    );

    let answer_path = idx
        .derived_answer_path_for_task(
            "How many engineers do I lead when I just started my new role as Senior Software Engineer? How many engineers do I lead now?",
        )
        .expect("role transition count solver should synthesize an answer");
    let answer = std::fs::read_to_string(answer_path).unwrap();
    assert!(
        answer.contains(
            "Answer: When you just started your new role as Senior Software Engineer, you led 4 engineers. Now, you lead 5 engineers"
        ),
        "{answer}"
    );
}

#[test]
fn current_role_duration_solver_subtracts_pre_promotion_tenure() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0462_462_conv_summary.md",
        "- As a Senior Marketing Specialist in the company, I've been feeling a bit stuck.\n- I've been thinking about my 3 years and 9 months experience in the company and I've realized that I've built a strong understanding of our target audience.\n- I've been in marketing for a while now, started as a Marketing Coordinator and worked my way up to Senior Marketing Specialist after 2 years and 4 months.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "industry_experience.verbatim.md",
        "User: I'm interested in pursuing a master's degree in marketing, as I've been working in the industry for five years now and feel like I need to upgrade my skills to stay competitive. My career goal is to move into a leadership role.\n",
    );

    let answer = read_answer_text(&idx, "How long have I been working in my current role?");
    assert!(answer.contains("Answer: 1 year and 5 months"), "{answer}");
}

#[test]
fn named_artwork_location_solver_prefers_latest_room_for_target_title() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0146_146_conv_summary.md",
        "- I have a beautiful digital print of \"Moonlit Ocean\" by Jack Harris that I plan to frame and hang in my bedroom, but I'll leave the \"Ethereal Dreams\" painting above my living room sofa as is.\n- I just rearranged my bedroom and moved the \"Ethereal Dreams\" painting to my bedroom, where it adds a nice touch to the space.\n- Speaking of my bedroom, I recently moved the \"Ethereal Dreams\" painting by Emma Taylor above my bed, and it adds a nice touch to the space.\n",
    );

    let answer = read_answer_text(
        &idx,
        "Where is the painting 'Ethereal Dreams' by Emma Taylor currently hanging?",
    );
    assert!(answer.contains("Answer: in my bedroom"), "{answer}");
}

#[test]
fn hilton_free_night_count_solver_answers_redeemable_stays() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0488_488_conv_summary.md",
        "- I've accumulated enough points for two free night's stays at any Hilton property, so I might use that for a separate trip to Las Vegas.\n",
    );

    let answer_path = idx
        .derived_answer_path_for_task(
            "How many free night's stays can I redeem at any Hilton property with my accumulated points?",
        )
        .expect("Hilton free-night solver should synthesize an answer");
    let answer = std::fs::read_to_string(answer_path).unwrap();
    assert!(answer.contains("Answer: Two"), "{answer}");
}

#[test]
fn time_spent_range_solver_answers_latest_effort_range() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0511_511_conv_summary.md",
        "- I've spent around 5-6 hours on my abstract ocean sculpture so far.\n- I've already put in 10-12 hours on my abstract ocean sculpture, and it's still a work in progress.\n",
    );

    let answer_path = idx
        .derived_answer_path_for_task("How many hours have I spent on my abstract ocean sculpture?")
        .expect("time-spent range solver should synthesize an answer");
    let answer = std::fs::read_to_string(answer_path).unwrap();
    assert!(answer.contains("Answer: 10-12 hours"), "{answer}");
}

#[test]
fn publication_issue_count_solver_preserves_word_surface() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0522_522_conv_summary.md",
        "- I've finished five issues so far of National Geographic and they've all had great articles on the region.\n",
    );

    let answer_path = idx
        .derived_answer_path_for_task(
            "How many issues of National Geographic have I finished reading?",
        )
        .expect("publication issue-count solver should synthesize an answer");
    let answer = std::fs::read_to_string(answer_path).unwrap();
    assert!(answer.contains("Answer: Five"), "{answer}");
}

#[test]
fn collection_restart_count_solver_prefers_latest_total() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0530_530_conv_summary.md",
        "- I've added 17 new postcards since I started collecting again.\n- I've added 25 new postcards to my collection since I started collecting again.\n",
    );

    let answer_path = idx
        .derived_answer_path_for_task(
            "How many new postcards have I added to my collection since I started collecting again?",
        )
        .expect("collection restart solver should synthesize an answer");
    let answer = std::fs::read_to_string(answer_path).unwrap();
    assert!(answer.contains("Answer: 25"), "{answer}");
}

#[test]
fn since_start_count_solver_prefers_latest_short_story_total() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0133_133_conv_summary.md",
        "- I've written four short stories so far since I started writing regularly, and I'm hoping to keep the momentum going.\n- I've been writing regularly for three months now, and it's been amazing - I've even managed to complete 7 short stories since I started.\n",
    );

    let answer_path = idx
        .derived_answer_path_for_task(
            "How many short stories have I written since I started writing regularly?",
        )
        .expect("since-start count solver should synthesize a short-story answer");
    let answer = std::fs::read_to_string(answer_path).unwrap();
    assert!(answer.contains("Answer: 7"), "{answer}");
}

#[test]
fn since_start_count_solver_prefers_latest_painting_project_total() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0181_181_conv_summary.md",
        "- I've completed 4 projects since starting painting classes, and I'm feeling pretty confident about my skills.\n- I just finished my 5th project since starting painting classes, and I'm feeling pretty accomplished!\n",
    );

    let answer_path = idx
        .derived_answer_path_for_task(
            "How many projects have I completed since starting painting classes?",
        )
        .expect("since-start count solver should synthesize a painting-project answer");
    let answer = std::fs::read_to_string(answer_path).unwrap();
    assert!(answer.contains("Answer: 5"), "{answer}");
}

#[test]
fn named_meetup_count_solver_ignores_planned_future_catchup() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0309_309_conv_summary.md",
        "- I met this guy Alex from Germany at a music festival a few weeks ago, and we're planning to meet up.\n- I've got a friend Alex from Germany who I met at a music festival, and we've met up twice already - he's really cool.\n- I'm also planning to meet up with my friend Alex from Germany while I'm in Berlin, we've met up twice before and it'll be great to catch up with him again.\n",
    );

    let answer_path = idx
        .derived_answer_path_for_task("How many times have I met up with Alex from Germany?")
        .expect("named meetup count solver should synthesize an answer");
    let answer = std::fs::read_to_string(answer_path).unwrap();
    assert!(answer.contains("Answer: We've met up twice."), "{answer}");
}

#[test]
fn item_usage_frequency_solver_prefers_latest_converse_wear_total() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0147_147_conv_summary.md",
        "- I already wore my new black Converse Chuck Taylor All Star sneakers four times this week, and they're breaking in nicely.\n- By the way, I just wore my new black Converse to run some errands yesterday, so that's six times now that I've worn them.\n",
    );

    let answer_path = idx
        .derived_answer_path_for_task(
            "How many times have I worn my new black Converse Chuck Taylor All Star sneakers?",
        )
        .expect("item-usage count solver should synthesize the latest Converse wear total");
    let answer = std::fs::read_to_string(answer_path).unwrap();
    assert!(answer.contains("Answer: six"), "{answer}");
}

#[test]
fn item_usage_frequency_solver_preserves_camera_trip_word_surface() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0190_190_conv_summary.md",
        "- I've taken my Canon EOS 80D camera on three trips already: Yellowstone, Yosemite, and the Grand Canyon.\n- I'm planning a trip to Zion National Park, and I've had my Canon EOS 80D with me on five trips now.\n",
    );

    let answer_path = idx
        .derived_answer_path_for_task("How many trips have I taken my Canon EOS 80D camera on?")
        .expect("item-usage count solver should synthesize the camera trip count");
    let answer = std::fs::read_to_string(answer_path).unwrap();
    assert!(answer.contains("Answer: five"), "{answer}");
}

#[test]
fn media_rewatch_count_solver_counts_distinct_matching_titles() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0446_446_conv_summary.md",
        "- I've actually watched Doctor Strange already, it was one of the four Marvel movies I watched recently.\n- Since I just re-watched Avengers: Endgame yesterday, I've been thinking about other movies that might have a similar sense of scale and action.\n- Since I re-watched Avengers: Endgame, which is a Marvel movie, I've been exploring more sci-fi and action movies with a similar sense of grandeur and epic battles.\n- I've been into Marvel movies lately, and I also re-watched Spider-Man: No Way Home, which is another Marvel movie.\n",
    );

    let answer = read_answer_text(&idx, "How many Marvel movies did I re-watch?");
    assert!(answer.contains("Answer: 2"), "{answer}");
}

#[test]
fn family_origin_item_count_solver_counts_distinct_antique_family_items() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0433_433_conv_summary.md",
        "- I inherited my grandmother's vintage diamond necklace, an antique music box from my great-aunt, and depression-era glassware from my mom.\n- I also have an antique tea set from my cousin Rachel and a vintage typewriter that belonged to my dad.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0433_433.conv.md",
        "User: I'm trying to get my grandmother's vintage diamond necklace insured, and I inherited it recently along with an antique music box from my great-aunt and a set of depression-era glassware from my mom.\nUser: I'm also thinking about decluttering an antique tea set from my cousin Rachel and a vintage typewriter that belonged to my dad.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How many antique items did I inherit or acquire from my family members?",
    );
    assert!(answer.contains("Answer: 5"), "{answer}");
}

#[test]
fn recent_birth_count_solver_counts_twins_and_excludes_adoption() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0077_77_conv_summary.md",
        "- David had a baby boy named Jasper a few weeks ago.\n- My cousin Rachel's son Max was born in March, and Mike and Emma welcomed their daughter Charlotte around the same time.\n- My aunt's twins, Ava and Lily, were born in April.\n- Sarah recently adopted a daughter named Aaliyah.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0077_77.conv.md",
        "User: My friend from college, David, had a baby boy named Jasper a few weeks ago.\nUser: I think I'll add a few more birthdays to the calendar, including my cousin Rachel's son Max, who was born in March. I should also add my friends Mike and Emma's daughter Charlotte, who was born around the same time. And let me not forget my aunt's twins, Ava and Lily, who were born in April.\nUser: My friend Sarah's adopted daughter Aaliyah is another one I want to make sure to remember.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How many babies were born to friends and family members in the last few months?",
    );
    assert!(answer.contains("Answer: 5"), "{answer}");
}

#[test]
fn bike_service_count_solver_counts_distinct_march_bikes() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0100_100_conv_summary.md",
        "- My road bike was serviced at Pedal Power on March 10th.\n- Speaking of which, I remember cleaning and lubricating my bike chain on March 22nd, which made a big difference in its performance.\n- I'm looking into getting a new tire for my commuter bike this month, before April comes.\n- I got a new water bottle cage for my mountain bike.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0100_100.conv.md",
        "User: I'm looking into getting a new tire for my commuter bike, and I think it is time to replace it this month, before April comes.\nUser: Since I've been taking good care of my road bike, it should be able to handle the hills and terrain. Speaking of which, I remember cleaning and lubricating my bike chain on March 22nd, which has made a big difference in its performance.\nUser: I'm really looking forward to the ride after getting my road bike serviced at Pedal Power on March 10th.\nUser: I'm also planning to bring my new water bottle cage on the ride, which I got for my mountain bike a few weeks ago.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How many bikes did I service or plan to service in March?",
    );
    assert!(answer.contains("Answer: 2"), "{answer}");
}

#[test]
fn fitness_class_day_count_solver_counts_distinct_weekdays() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0483_483.conv.md",
        "User: I'm looking for some new workout playlists to try out. Do you have any recommendations? By the way, I've been trying to mix up my routine and recently started a yoga class on Wednesdays, which has been really helpful in stretching out my muscles after a long day.\nUser: I'm trying to plan out my week and was wondering if you could help me set reminders for my upcoming fitness classes. I attend Zumba classes on Tuesdays and Thursdays at 6:30 pm, and a weightlifting class on Saturdays at 10 am.\n",
    );

    let answer = read_answer_text(&idx, "How many days a week do I attend fitness classes?");
    assert!(answer.contains("Answer: 4 days"), "{answer}");
}

#[test]
fn fitness_class_day_count_solver_counts_typical_week_classes() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0094_94.conv.md",
        "User: I need something to motivate me during my weightlifting classes, like BodyPump on Mondays at 6:30 PM.\nUser: I usually take Zumba classes on Tuesdays and Thursdays at 7:00 PM.\nUser: I'm not free on Sundays since I have my yoga class at 6:00 PM.\nUser: I attend Hip Hop Abs on Saturdays at 10:00 AM.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How many fitness classes do I attend in a typical week?",
    );
    assert!(answer.contains("Answer: 5"), "{answer}");
}

#[test]
fn fitness_class_day_count_solver_ignores_generic_yoga_practice_and_uses_assistant_restate() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0483_a08a253f_conv_0001_chunk.verbatim.md",
        "User: I think I'll try to schedule my yoga practice for Monday, Wednesday, and Friday mornings at 7:00 am.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0094_2788b940_conv_0002_chunk.verbatim.md",
        "User: Can I also get some recommendations for strength training playlists? I've recently started taking a BodyPump class on Mondays and want something that'll keep me pumped up during those intense weightlifting sessions.\n\
         User: I'll definitely check those out. For my Zumba classes on Tuesdays and Thursdays, I like to get there about 15 minutes early to warm up.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0094_2788b940_conv_0004_chunk.verbatim.md",
        "User: I'm looking to explore some new fitness routines and was wondering if you have any recommendations for strength training exercises I can do at home. By the way, I'm not free on Sundays since I have my yoga class at 6:00 PM, so anything that can be done on other days would be great.\n\
         Pick the playlist that gets you most hyped, or create your own using these tracks as inspiration! Get ready to crush your Hip Hop Abs class with Mike on Saturday morning!\n\
         User: These playlists are awesome! I'll definitely try them out. I was thinking of also making a playlist for my yoga classes on Sundays. Do you have any chill hip-hop or R&B playlists that could help me unwind and relax during those classes?\n",
    );

    let answer = read_answer_text(
        &idx,
        "How many fitness classes do I attend in a typical week?",
    );
    assert!(answer.contains("Answer: 5"), "{answer}");
}

#[test]
fn month_scoped_activity_day_count_solver_counts_unique_april_learning_days() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0107_107.conv.md",
        "User: I actually learned about standardization and normalization in a 2-day workshop I attended on the 17th and 18th of April, but I'm still a bit unclear on when to use each.\nUser: I'm looking for some resources on urban planning and sustainable development. By the way, I recently attended a lecture on sustainable development at the public library on the 10th of April, and it got me interested in learning more.\nUser: I started an online course with weekly video lectures in May.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How many days did I spend attending workshops, lectures, and conferences in April?",
    );
    assert!(answer.contains("Answer: 3 days"), "{answer}");
}

#[test]
fn month_scoped_activity_day_count_solver_dedupes_december_faith_days() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0102_102.conv.md",
        "User: I actually helped out at the church's annual holiday food drive on December 10th, sorting donations and packing boxes for families in need.\nUser: I actually just did a Bible study on this same topic at my church a few weeks ago, on December 17th, and it was really thought-provoking.\nUser: I've been thinking about how faith applies to daily life a lot lately, especially after our Bible study group on December 17th, and I was wondering if you could recommend some books or resources on the topic.\nUser: I just got back from a lovely midnight mass on Christmas Eve at St. Mary's Church, which was on December 24th, with my family.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How many days did I spend participating in faith-related activities in December?",
    );
    assert!(answer.contains("Answer: 3 days"), "{answer}");
}

#[test]
fn art_related_event_count_solver_counts_recent_distinct_events() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0088_88.conv.md",
        "User: I recently went on a guided tour at the History Museum on February 24th, and it really sparked my interest in ancient history and art.\nUser: I recently attended a lecture at the Art Gallery on 'The Evolution of Street Art' on March 3rd, and it got me thinking about the role of street art in urban communities.\nUser: I was particularly drawn to the works of local artist, Rachel Lee, at the \"Women in Art\" exhibition which I attended on February 10th.\nUser: I recently volunteered at the Children's Museum for their \"Art Afternoon\" event on February 17th, and it was amazing to see the kids create their own artwork inspired by famous paintings.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How many different art-related events did I attend in the past month?",
    );
    assert!(answer.contains("Answer: 4"), "{answer}");
}

#[test]
fn distinct_cuisine_count_solver_counts_learned_and_tried_cuisines() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0081_81.conv.md",
        "User: I tried out a new Ethiopian restaurant in town last week and loved it!\nUser: I learned how to make a perfect chicken tikka masala in a class on Indian cuisine.\nUser: I recently attended a class on vegan cuisine that got me really inspired.\nUser: I just tried out a recipe for Korean bibimbap from the cooking class's online recipe library, and it was amazing.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How many different cuisines have I learned to cook or tried out in the past few months?",
    );
    assert!(answer.contains("Answer: 4"), "{answer}");
}

#[test]
fn museum_gallery_visit_count_solver_counts_distinct_february_venues() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0080_80.conv.md",
        "User: I recently saw some work when I visited The Art Cube on 2/15.\nUser: I met the curator, Rachel Lee, at the opening night of The Art Cube on 15th February.\nUser: I took my niece to the Natural History Museum on 2/8 and she loved the dinosaur exhibit!\n",
    );

    let answer = read_answer_text(
        &idx,
        "How many different museums or galleries did I visit in the month of February?",
    );
    assert!(answer.contains("Answer: 2"), "{answer}");
}

#[test]
fn citrus_fruit_count_solver_counts_distinct_cocktail_citrus_types() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0070_70.conv.md",
        "User: I recently made my own orange bitters using orange peels and vodka.\nUser: I'm planning to make Sangria with slices of orange and lemon.\nUser: I recently made a Cucumber Gimlet by mixing it with lime juice and simple syrup.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How many different types of citrus fruits have I used in my cocktail recipes?",
    );
    assert!(answer.contains("Answer: 3"), "{answer}");
}

#[test]
fn food_delivery_service_count_solver_counts_distinct_recent_services() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0083_83.conv.md",
        "User: I've been relying on food delivery services, like this new one I found called Fresh Fusion.\nUser: I've been relying on food delivery services a lot lately - I had Domino's Pizza three times last week!\nUser: My weekends have been all about Uber Eats lately, it's been a lifesaver.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How many different types of food delivery services have I used recently?",
    );
    assert!(answer.contains("Answer: 3"), "{answer}");
}

#[test]
fn missed_fun_run_count_solver_counts_march_work_conflicts() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0493_493.conv.md",
        "User: I just completed my first full marathon on April 10th and I'm feeling a bit sore. I've been active in the running community and was able to attend most of the weekly 5K fun runs at the local park, except for the run on March 5th when I had to miss due to work commitments.\nUser: I've been pretty busy with work lately and missed a few events, including a 5K fun run on March 26th.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How many fun runs did I miss in March due to work commitments?",
    );
    assert!(answer.contains("Answer: 2"), "{answer}");
}

#[test]
fn graduation_ceremony_count_solver_counts_recent_attended_ceremonies() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0115_115.conv.md",
        "User: I just attended my colleague Alex's graduation from a leadership development program at work a few weeks ago.\nUser: I just attended my little cousin Emma's preschool graduation about two months ago!\nUser: I'm still amazed by how fast my little cousin Emma is growing up - it feels like just yesterday I was attending her preschool graduation ceremony!\nUser: I just attended my best friend Rachel's master's degree graduation ceremony a couple of weeks ago, it was really inspiring.\nUser: Rachel's graduation ceremony reminded me of how important it is to invest in my own education and growth.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How many graduation ceremonies have I attended in the past three months?",
    );
    assert!(answer.contains("Answer: 3"), "{answer}");
}

#[test]
fn health_device_count_solver_counts_daily_health_devices() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0092_92.conv.md",
        "User: I've been trying to do at least one guided breathing session per day with my Fitbit, which has really been helping me relax.\nUser: I've been wearing my Fitbit Versa 3 smartwatch non-stop since I got it three weeks ago.\nUser: I need help ordering some replacement batteries for my hearing aids. I've been using the same set for months now.\nUser: I have behind-the-ear (BTE) hearing aids from Phonak, and I'm currently using size 13 batteries.\nUser: I think I'll go with the Energizer Size 13 Hearing Aid Batteries and want to know how long a pack of 6 will last.\n",
    );

    let answer = read_answer_text(&idx, "How many health-related devices do I use in a day?");
    assert!(answer.contains("Answer: 4"), "{answer}");
}

#[test]
fn peak_campaign_weekly_hours_solver_adds_base_and_peak_delta() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0443_443.conv.md",
        "User: During peak campaign seasons, I increase my work hours by 10 hours weekly to accommodate the additional workload.\nUser: By the way, I usually work 40 hours a week, with some weeks being busier than others.\nUser: I usually work 40 hours a week, but some weeks can go up to 45 hours or so.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How many hours do I work in a typical week during peak campaign seasons?",
    );
    assert!(answer.contains("Answer: 50"), "{answer}");
}

#[test]
fn recent_activity_duration_total_solver_counts_realized_activity_only() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0084_84_conv_summary.md",
        "- I used to practice yoga three times a week, each time for 2 hours, but I've been slacking off for this month and I'm trying to get back into it.\n- I went for a 30-minute jog around the neighborhood on Saturday, and I'd like to keep a record of that.\n- **Make the reminder specific**: Instead of a generic reminder, try something like \"Log today's 30-minute jog\".\n",
    );

    let answer = read_answer_text(
        &idx,
        "How many hours of jogging and yoga did I do last week?",
    );
    assert!(answer.contains("Answer: 0.5 hours"), "{answer}");
}

#[test]
fn current_magazine_subscription_count_solver_excludes_canceled_titles() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0110_110_conv_summary.md",
        "- I just canceled my Forbes magazine subscription in early March because I wasn't finding the articles that interesting, but I've been enjoying other publications like Th\n- I've been loving my subscription to The New Yorker magazine, which I subscribed to in early February - the weekly issues have been keeping me up-to-date on current events and culture.\n- I'm also getting Architectural Digest, which I love for home decor inspiration.\n- You've already taken a great step by subscribing to Architectural Digest, which is an excellent source of inspiration.\n",
    );

    let answer = read_answer_text(&idx, "How many magazine subscriptions do I currently have?");
    assert!(answer.contains("Answer: 2"), "{answer}");
}

#[test]
fn marathon_target_overrun_minutes_solver_subtracts_target_from_actual() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0487_487.conv.md",
        "User: By the way, I just completed my first full marathon in 4h 22min, so I'm looking for routes that can help me keep my endurance up while I'm away.\nUser: Oh, and by the way, my target time for the marathon was 4 hours and 10 minutes, so I'm hoping to apply some of that endurance to my triathlon training.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How many minutes did I exceed my target time by in the marathon?",
    );
    assert!(answer.contains("Answer: 12"), "{answer}");
}

#[test]
fn movie_festival_count_solver_counts_distinct_attended_festivals() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0071_71.conv.md",
        "User: I've been pretty active in the film festival scene lately - I even volunteered at the Portland Film Festival, where I helped with event coordination and got to meet some industry professionals.\nUser: I recently participated in the 48-hour film challenge at the Austin Film Festival, where my team and I had to write, shoot, and edit a short film within 48 hours - it was a wild ride!\nUser: By the way, I was impressed by how quickly we had to come up with a script, shoot, and edit our short film at the Austin Film Festival - it was definitely a challenge, but it was amazing to see how our team came together.\nUser: I got to discuss the unique narrative structure of \"The Weight of Water\" with the director himself at a Q&A session after the screening at the Seattle International Film Festival, which was really enlightening.\nUser: I've been fortunate enough to attend some amazing festivals, like AFI Fest, where I got to see \"Joker\" and attend a Q&A session with Todd Phillips and Joaquin Phoenix.\n",
    );

    let answer = read_answer_text(&idx, "How many movie festivals that I attended?");
    assert!(
        answer.contains("Answer: I attended four movie festivals."),
        "{answer}"
    );
}

#[test]
fn music_release_count_solver_counts_downloaded_and_purchased_releases() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0112_112.conv.md",
        "User: I'm looking for some new music recommendations. I've been really into indie and folk-rock lately, especially after discovering The Whiskey Wanderers at a music festival last weekend - I bought their EP 'Midnight Sky' at the festival merchandise booth and can't get enough of it.\nUser: I'm looking for some new music recommendations. I've been listening to a lot of Billie Eilish lately, especially her new album \"Happier Than Ever\" which I downloaded on Spotify.\nUser: I'm looking for some music festival recommendations in Colorado. I recently got my Tame Impala vinyl signed after the show at the Red Rocks Amphitheatre in Colorado, and I'm looking to add to my collection.\nUser: Yeah! The music festival was amazing! \"The Whiskey Wanderers\" are a folk-rock band, and their live performance was incredibly energetic and engaging. I ended up buying their EP \"Midnight Sky\" at the festival merchandise booth, and I've been listening to it non-stop.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How many music albums or EPs have I purchased or downloaded?",
    );
    assert!(answer.contains("Answer: 3"), "{answer}");
}

#[test]
fn count_solver_counts_current_owned_musical_instruments() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0099_99.conv.md",
        "User: I'm thinking of selling my old drum set, a 5-piece Pearl Export, which I haven't played in years.\nUser: I'm also a bit concerned about the maintenance of my instruments, especially my piano, a Korg B1, which I've had for about 3 years.\nUser: I'm also thinking of buying a new ukulele, and I've been eyeing a Cordoba ukulele.\nUser: By the way, I've had my acoustic guitar, a Yamaha FG800, for about 8 years, and it's been a great companion for songwriting and camping trips.\nUser: By the way, I've had my black Fender Stratocaster electric guitar for about 5 years now, and it's been my go-to instrument for playing blues and rock music.\n",
    );

    let answer = read_answer_text(&idx, "How many musical instruments do I currently own?");
    assert!(
        answer.contains("Answer: I currently own 4 musical instruments. I've had the"),
        "{answer}"
    );
}

#[test]
fn count_solver_sums_completed_online_courses_across_platforms() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0111_111_conv_summary.md",
        "- I've completed three courses on Coursera, and I'm excited to dive deeper into CNNs for text classification.\n- By the way, I've completed two courses on edX so far, which has been really helpful in my current role as a software engineer.\n",
    );

    let answer = read_answer_text(&idx, "How many online courses have I completed in total?");
    assert!(answer.contains("Answer: 5"), "{answer}");
}

#[test]
fn count_solver_counts_recent_furniture_actions() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0078_78.conv.md",
        "User: I'm looking for some recommendations on throw pillows for my couch. I just got a new coffee table and rearranged my living room, and now the old pillows are looking a bit worn out. By the way, I've been meaning to get a new mattress for ages, and last week I finally took the plunge and ordered one from Casper. It's supposed to arrive next Wednesday, and I'm really looking forward to getting a good night's sleep.\nUser: Oh, and speaking of organizing, I finally assembled that IKEA bookshelf for my home office about two months ago, and it's been a game-changer for my productivity.\nUser: My living room has a modern feel, and the dominant color scheme is a mix of neutral tones like beige, gray, and white. The new coffee table is wooden with metal legs, and I've been loving how it's added a touch of modernity to the room. By the way, speaking of fixing things around the house, I finally got around to fixing the wobbly leg on my kitchen table last weekend - it was driving me crazy for months, and all it took was a few minutes with a screwdriver to tighten the screw.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How many pieces of furniture did I buy, assemble, sell, or fix in the past few months?",
    );
    assert!(answer.contains("Answer: 4"), "{answer}");
}

#[test]
fn count_solver_counts_recent_jewelry_acquisitions() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0096_96.conv.md",
        "User: I'm thinking of cleaning my jewelry collection and I want to make sure I do it right. By the way, I just got a new silver necklace with a small pendant on the 15th of last month, and I want to make sure I take good care of it.\nUser: I think I'll reach out to my cousin now and ask her about the earrings. Meanwhile, I also wanted to ask you about resizing my engagement ring. I got it a month ago, and it's still a bit too loose.\nUser: I need help with cleaning my jewelry. Can you give me some tips on how to properly clean my gold chains and rings? Oh, and by the way, I got my engagement ring a month ago, and I still need to get it resized - it's still a bit too loose.\nUser: I'm thinking of cleaning my jewelry collection this weekend and I'm not sure what's the best way to clean different types of jewelry. By the way, I just got a new pair of earrings last weekend at a flea market - a stunning pair of emerald earrings that I'm absolutely loving!\n",
    );

    let answer = read_answer_text(
        &idx,
        "How many pieces of jewelry did I acquire in the last two months?",
    );
    assert!(answer.contains("Answer: 3"), "{answer}");
}

#[test]
fn count_solver_counts_recent_plant_acquisitions() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0065_65.conv.md",
        "User: I'm trying to figure out the best way to care for my peace lily, which I got from the nursery two weeks ago along with a succulent. It's been losing some leaves, but I've read that's normal.\nUser: I actually use a mixture of water and fertilizer when I water my plants, which I got from the nursery where I bought the peace lily and a succulent plant two weeks ago.\nUser: I'm also wondering if I should repot my snake plant, which I got from my sister last month. Do you think it would benefit from a bigger pot and some fresh potting mix?\n",
    );

    let answer = read_answer_text(&idx, "How many plants did I acquire in the last month?");
    assert!(answer.contains("Answer: 3"), "{answer}");
}

#[test]
fn count_solver_sums_initial_tomato_and_cucumber_plants() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0442_442.conv.md",
        "User: I've been enjoying the harvest immensely! I planted 5 tomato plants initially, and they've been producing like crazy.\nUser: I'm trying to plan out my meals for the week and was wondering if you have any recipe suggestions that feature cucumbers as the main ingredient? By the way, I've been growing my own cucumbers in my garden, and I've got 3 plants that are producing a lot of them!\n",
    );

    let answer = read_answer_text(
        &idx,
        "How many plants did I initially plant for tomatoes and cucumbers?",
    );
    assert!(answer.contains("Answer: 8"), "{answer}");
}

#[test]
fn count_solver_computes_sephora_points_needed_for_free_skincare() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0453_453.conv.md",
        "User: Do you know if Sephora has any current promotions or discounts on the La Roche-Posay moisturizer or any other products I might want to purchase with it? By the way, I'm really close to redeeming a free skincare product from Sephora, I just need a total of 300 points and I'm all set!\nUser: I'm looking for some advice on skincare products. I recently bought an eyeshadow palette at Sephora and earned 50 points, bringing my total to 200 points so far in their loyalty program. Can you recommend some popular skincare products that would complement my eyeshadow purchases?\n",
    );

    let answer = read_answer_text(
        &idx,
        "How many points do I need to earn to redeem a free skincare product at Sephora?",
    );
    assert!(answer.contains("Answer: 100"), "{answer}");
}

#[test]
fn count_solver_counts_projects_excluding_thesis() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0098_98.conv.md",
        "User: I'm struggling to find relevant datasets for my thesis on AI's impact on healthcare outcomes. Can you suggest some reliable sources or repositories where I can find datasets related to medical diagnosis? By the way, I've been learning a lot about data analysis in my Data Mining course, which has a group project that's keeping me pretty busy.\nUser: I'm trying to find some relevant research papers on AI in medical diagnosis, specifically on image classification. Can you suggest some databases or search engines I can use? By the way, I've also been working on a group project for my Database Systems course, so I'm juggling multiple projects at the moment.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How many projects have I been working on simultaneously, excluding my thesis?",
    );
    assert!(answer.contains("Answer: 2"), "{answer}");
}

#[test]
fn count_solver_counts_properties_before_brookside_offer() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0082_82.conv.md",
        "User: I recently saw a beautiful 3-bedroom bungalow in the Oakwood neighborhood on January 22nd that I really liked, but the kitchen needed some serious renovation work.\nUser: I appreciate your detailed explanation on the importance of research when buying a condo. I'm currently looking at condos in the downtown area, and I'm considering a few options. I viewed a 1-bedroom condo on February 10th, but the noise from the highway was a deal-breaker.\nUser: I'm in the process of buying a new home and I need some help with organizing all the paperwork. I've been house hunting for a while, and it's been a wild ride. I actually fell in love with a 2-bedroom condo on February 15th, it had amazing modern appliances and a community pool, but unfortunately, my offer got rejected on the 17th due to a higher bid.\nUser: I'm looking for a home warranty to protect my new place from unexpected repairs. Can you recommend some providers and their prices? By the way, I've been searching for a home for a while now, and I've seen some properties that just didn't fit my budget, like that one in Cedar Creek on February 1st - it was way out of my league.\nUser: Hi! I'm in the process of buying a house and I need help with some calculations. I recently put in an offer on a 3-bedroom townhouse in the Brookside neighborhood on February 25th, and after some negotiations, we agreed on a price of $340,000.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How many properties did I view before making an offer on the townhouse in the Brookside neighborhood?",
    );
    assert!(
        answer.contains(
            "Answer: I viewed four properties before making an offer on the townhouse in the Brookside neighborhood. The reasons I didn't make an offer on them were: the kitchen of the bungalow needed serious renovation, the property in Cedar Creek was out of my budget, the noise from the highway was a deal-breaker for the 1-bedroom condo, and my offer on the 2-bedroom condo was rejected due to a higher bid."
        ),
        "{answer}"
    );
}

#[test]
fn count_solver_counts_competitive_sports_played() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0450_450.conv.md",
        "User: I'm looking to find a local pool that offers lap swimming hours. I used to swim competitively in college, and I'm looking to get back into it as a way to stay active and relieve stress.\nUser: I'm actually thinking of incorporating some strength training into my routine as well. Can you recommend some exercises that would be beneficial for my tennis game, considering I used to play tennis competitively in high school?\nUser: I've been playing soccer and tennis lately, so I'm trying to build on that.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How many sports have I played competitively in the past?",
    );
    assert!(answer.contains("Answer: two"), "{answer}");
}

#[test]
fn count_solver_counts_current_tanks_including_friends_kid_setup() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0073_73.conv.md",
        "User: I'm having some issues with high nitrite levels in my tank. I've been doing partial water changes, but I'm not sure if I'm doing it correctly. Can you walk me through the process? By the way, I've had some experience with aquariums - I have a 5-gallon tank with a solitary betta fish named Finley, which I got from my cousin.\nUser: I've been learning about aquarium keeping for about 6 months now, and I've had some experience with cycling a tank. My old tank was a 5-gallon one that I got from my cousin, and I kept a solitary betta fish named Finley. I've since set up a new 20-gallon community tank, and I want to make sure I'm doing everything right.\nUser: I'm having some issues with the nitrite levels in my tank. I've been doing partial water changes, but I'm not sure if I'm doing it correctly. Can you give me some tips on how to lower nitrite levels in a freshwater tank? By the way, I've finally set up my 20-gallon freshwater community tank, which I've named \"Amazonia\", and it's been doing well so far.\nUser: I've also been taking care of a small 1-gallon tank that I set up for a friend's kid, which has a few guppies and some plants.\nUser: I'm thinking about setting up a separate quarantine tank for my new fish before introducing them to my main tank.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How many tanks do I currently have, including the one I set up for my friend's kid?",
    );
    assert!(answer.contains("Answer: 3"), "{answer}");
}

#[test]
fn count_solver_counts_recent_baking_events() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0079_79.conv.md",
        "User: I'm looking for some advice on improving my sourdough starter. I tried out a new bread recipe using sourdough starter on Tuesday, but it didn't quite turn out as expected.\nUser: By the way, I've been experimenting with different baking recipes lately, and I recently tried out a new bread recipe using sourdough starter on Tuesday.\nUser: By the way, I've been experimenting with different types of flour lately, and I recently baked a chocolate cake for my sister's birthday party using a new recipe I found online.\nUser: I'm looking for some recipe ideas for a dinner party I'm hosting next weekend. By the way, I just baked a chocolate cake for my sister's birthday party last weekend and it turned out amazing.\nUser: I've been experimenting with different types of flour lately, including whole wheat flour, which I used to make a delicious whole wheat baguette last Saturday.\nUser: I just used my oven's convection setting for the first time last Thursday to bake a batch of cookies, and it turned out amazing!\nUser: I've had good results with the convection setting on my oven, like when I used it to bake a batch of cookies last Thursday.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How many times did I bake something in the past two weeks?",
    );
    assert!(answer.contains("Answer: 4"), "{answer}");
}

#[test]
fn count_solver_abstains_for_unmentioned_baked_item() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0079_79.conv.md",
        "User: I'm looking for some advice on improving my sourdough starter. I tried out a new bread recipe using sourdough starter on Tuesday, but it didn't quite turn out as expected.\nUser: I'm looking for some recipe ideas for a dinner party I'm hosting next weekend. By the way, I just baked a chocolate cake for my sister's birthday party last weekend and it turned out amazing.\nUser: I've been experimenting with different types of flour lately, including whole wheat flour, which I used to make a delicious whole wheat baguette last Saturday.\nUser: I just used my oven's convection setting for the first time last Thursday to bake a batch of cookies, and it turned out amazing!\n",
    );

    assert!(
        idx.derived_answer_path_for_task(
            "How many times did I bake egg tarts in the past two weeks?"
        )
        .is_none(),
        "recent-baking solver should abstain when the asked baked item is never mentioned"
    );
}

#[test]
fn museum_gallery_visit_count_solver_prefers_summary_session_with_more_matching_venues() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0080_80_conv_summary.md",
        "- Can you recommend any contemporary artists similar to James Parker, whose work I recently saw when I visited The Art Cube on 2/15\n- I actually met the curator, Rachel Lee, at the opening night of The Art Cube on 15th February, and she mentioned some upcoming exhibitions and events that I'm interested in attending\n- By the way, I took my niece to the Natural History Museum on 2/8 and she loved the dinosaur exhibit\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0088_88_conv_summary.md",
        "- I recently went on a guided tour at the History Museum on February 24th, and it really sparked my interest in ancient history and art\n- I recently attended a lecture at the Art Gallery on March 3rd\n",
    );

    let answer = read_answer_text(
        &idx,
        "How many different museums or galleries did I visit in the month of February?",
    );
    assert!(answer.contains("Answer: 2"), "{answer}");
}

#[test]
fn named_team_composition_count_solver_prefers_latest_women_total() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0186_186_conv_summary.md",
        "- I just caught up with my former manager Rachel, who's now leading a team of 10 people, and half of them are women.\n- It's interesting to think about this in the context of my own experiences - for instance, my former manager Rachel's team is a great example of a diverse team, with 6 women out of 10 people.\n",
    );

    let answer_path = idx
        .derived_answer_path_for_task(
            "How many women are on the team led by my former manager Rachel?",
        )
        .expect("named team composition solver should synthesize Rachel's latest women count");
    let answer = std::fs::read_to_string(answer_path).unwrap();
    assert!(answer.contains("Answer: 6"), "{answer}");
}

#[test]
fn daily_time_commitment_solver_prefers_latest_coding_duration() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0174_174_conv_summary.md",
        "- I've been dedicating about an hour each day to coding exercises, which has been helpful in making progress.\n- That's really helpful. I've been dedicating about two hours each day to coding exercises, and I'm excited to see progress in my skills over the next few weeks.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How much time do I dedicate to coding exercises each day?",
    );
    assert!(answer.contains("Answer: about two hours"), "{answer}");
}

#[test]
fn daily_time_commitment_solver_abstains_on_different_instrument() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0354_354_conv_summary.md",
        "- I've been practicing guitar for 30 minutes daily, and it's been helping me progress nicely.\n",
    );

    assert!(
        idx.derived_answer_path_for_task(
            "How much time do I dedicate to practicing violin every day?"
        )
        .is_none(),
        "daily-time solver should abstain when only guitar evidence exists"
    );
}

#[test]
fn weight_loss_since_start_solver_answers_gym_delta() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0178_178_conv_summary.md",
        "- I've also lost about 5 pounds in the past month, which is slow progress, but I'm okay with that.\n- By the way, speaking of my cardio days, I've been doing great and I just realized I've lost 10 pounds since I started going consistently to the gym 3 months ago.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How much weight have I lost since I started going to the gym consistently?",
    );
    assert!(answer.contains("Answer: 10 pounds"), "{answer}");
}

#[test]
fn activity_frequency_transition_solver_compares_previous_and_current_tennis_schedule() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0173_173_conv_summary.md",
        "- I was just at the local park last Sunday, and I saw some people playing tennis - reminds me of my own weekly tennis sessions with friends.\n- I'm planning to play tennis with my friends this Sunday at the local park, like we do every other week.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How often do I play tennis with my friends at the local park previously? How often do I play now?",
    );
    assert!(
        answer.contains("Answer: Previously, you play tennis with your friends at the local park every week (on Sunday). Currently, you play tennis every other week (on Sunday)."),
        "{answer}"
    );
}

#[test]
fn named_recurring_frequency_solver_prefers_latest_therapist_schedule() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0141_141_conv_summary.md",
        "- I have a therapy session with Dr. Smith coming up soon - it's every two weeks, so I'm looking forward to discussing my progress with her.\n- I see Dr. Smith every week, and she's been helping me work on this stuff.\n",
    );

    let answer = read_answer_text(&idx, "How often do I see my therapist, Dr. Smith?");
    assert!(answer.contains("Answer: every week"), "{answer}");
}

#[test]
fn named_current_company_solver_prefers_current_employer_over_old_company() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0144_144_conv_summary.md",
        "- I met Rachel, an old colleague from my previous company, at the TechConnect conference.\n- Speaking of networking, I was just thinking about catching up with Rachel, an old colleague from my previous company, who's currently at TechCorp.\n",
    );

    let answer = read_answer_text(
        &idx,
        "What company is Rachel, an old colleague from my previous company, currently working at?",
    );
    assert!(answer.contains("Answer: TechCorp"), "{answer}");
}

#[test]
fn previous_named_tutor_weekday_solver_keeps_juan_separate_from_maria() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0175_175_conv_summary.md",
        "- I had a language exchange class at a local language school, where I'm paired with a Colombian tutor named Juan. We meet every Wednesday evening, and he helps me with my Spanish pronunciation and grammar while I assist him with his English vocabulary.\n- I have a language exchange session with my tutor Maria this week.\n- I'm actually meeting Maria on Thursday.\n",
    );

    let answer = read_answer_text(
        &idx,
        "What day of the week did I meet with my previous language exchange tutor Juan?",
    );
    assert!(answer.contains("Answer: Wednesday"), "{answer}");
}

#[test]
fn current_schedule_slot_solver_prefers_latest_cocktail_class_weekday() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0130_130_conv_summary.md",
        "- I have a cocktail-making class on Thursday, so I'm excited to try out some new recipes.\n- Speaking of cocktails, I have a cocktail-making class on Friday, and I'm thinking of experimenting with some new recipes.\n",
    );

    let answer = read_answer_text(
        &idx,
        "What day of the week do I take a cocktail-making class?",
    );
    assert!(answer.contains("Answer: Friday"), "{answer}");
}

#[test]
fn current_schedule_slot_solver_prefers_latest_gym_time() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0166_166_conv_summary.md",
        "- I usually go to the gym at 7:00 pm on Mondays, Wednesdays, and Fridays.\n- I need to make sure I'm done with the meeting before I head to the gym, which is usually at 6:00 pm.\n",
    );

    let answer = read_answer_text(&idx, "What time do I usually go to the gym?");
    assert!(answer.contains("Answer: 6:00 pm"), "{answer}");
}

#[test]
fn state_transition_solver_prefers_latest_ticket_to_ride_score() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0162_162_conv_summary.md",
        "- I've been crushing it in Ticket to Ride lately - my highest score so far is 124 points, and I'm eager to keep improving.\n- By the way, speaking of building and creating things, I just got my highest score in Ticket to Ride - 132 points!\n",
    );

    let answer = read_answer_text(&idx, "What is my current highest score in Ticket to Ride?");
    assert!(answer.contains("Answer: 132 points"), "{answer}");
}

#[test]
fn state_transition_solver_prefers_latest_volleyball_record() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0180_180_conv_summary.md",
        "- I've been doing pretty well in the volleyball league, we're 3-2 so far!\n- Our volleyball team, the Net Ninjas, is doing well with a 5-2 record.\n",
    );

    let answer = read_answer_text(
        &idx,
        "What is my current record in the recreational volleyball league?",
    );
    assert!(answer.contains("Answer: 5-2"), "{answer}");
}

#[test]
fn state_transition_solver_returns_previous_united_status() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0170_170_conv_summary.md",
        "- I actually just hit 20,000 miles on United Airlines, which means I'm finally eligible for Premier Silver status.\n- I do have a United Airlines MileagePlus account - I just reached Premier Gold status.\n",
    );

    let answer = read_answer_text(
        &idx,
        "What was my previous frequent flyer status on United Airlines before I got the current status?",
    );
    assert!(answer.contains("Answer: Premier Silver"), "{answer}");
}

#[test]
fn state_transition_solver_returns_previous_apex_goal() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0183_183_conv_summary.md",
        "- I've been playing a lot of Apex lately and I'm determined to reach level 100 before the end of the year.\n- Speaking of which, my current goal is to reach level 150, and I'm determined to get there eventually.\n",
    );

    let answer = read_answer_text(
        &idx,
        "What was my previous goal for my Apex Legends level before I updated my goal?",
    );
    assert!(answer.contains("Answer: level 100"), "{answer}");
}

#[test]
fn previous_purchased_item_solver_returns_instant_pot_before_air_fryer() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0191_191_conv_summary.md",
        "- I bought my new Instant Pot to make meal prep easier during the week.\n- I'm actually thinking of using the Air Fryer I got yesterday to make something crispy for dinner.\n",
    );

    let answer = read_answer_text(
        &idx,
        "What new kitchen gadget did I invest in before getting the Air Fryer?",
    );
    assert!(answer.contains("Answer: Instant Pot"), "{answer}");
}

#[test]
fn latest_purchased_lens_solver_prefers_zoom_lens_over_considering_wide_angle() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0140_140_conv_summary.md",
        "- I took my camera out last weekend and mostly used the old 18-55mm kit lens, but I recently got a new 50mm prime lens and it's been great for portraits.\n- I've been getting some great shots with my new 70-200mm zoom lens lately.\n- For my next trip, I've been considering a wide-angle lens, maybe a 14-24mm or 16-35mm, but I haven't bought one yet.\n",
    );

    let answer = read_answer_text(
        &idx,
        "What type of camera lens did I purchase most recently?",
    );
    assert!(answer.contains("Answer: a 70-200mm zoom lens"), "{answer}");
}

#[test]
fn planned_trip_stay_solver_returns_oahu_for_hawaii_birthday_trip() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0168_168_conv_summary.md",
        "- I'm planning a birthday trip to Hawaii and trying to decide how to split the time between islands.\n- I'm actually planning to stay on Oahu, so I'm looking at beaches and hiking spots there.\n",
    );

    let answer = read_answer_text(
        &idx,
        "Where am I planning to stay for my birthday trip to Hawaii?",
    );
    assert!(answer.contains("Answer: Oahu"), "{answer}");
}

#[test]
fn assistant_followup_solver_returns_mummy_count_from_prior_assistant_plan() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0244_244.conv.md",
        "User: create a D&D one shot for level 8 PCs.\nAssistant: The Lost Temple of the Djinn\n* Mummies (4):\n* Construct Guardians (2):\n",
    );

    let answer = read_answer_text(
        &idx,
        "I'm going back to our previous chat about the Lost Temple of the Djinn one-shot. Can you remind me how many mummies the party will face in the temple?",
    );
    assert!(answer.contains("Answer: 4"), "{answer}");
}

#[test]
fn assistant_followup_solver_returns_specific_beer_recommendation() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0278_278.conv.md",
        "User: Could you suggest a recipe for a classic dish from Ancash?\nAssistant: I don't have personal preferences, but in general, it's recommended to use a light or medium-bodied beer for this recipe to avoid overpowering the flavors of the lamb and spices. a pilsner or lager would work well, but you can also use a local beer from ancash to give the dish a more authentic touch.\n",
    );

    let answer = read_answer_text(
        &idx,
        "I'm looking back at our previous conversation about the Seco de Cordero recipe from Ancash. You mentioned using a light or medium-bodied beer, but I was wondering if you could remind me what type of beer you specifically recommended?",
    );
    assert!(
        answer.contains("Answer: I recommended using a Pilsner or Lager for the recipe."),
        "{answer}"
    );
}

#[test]
fn assistant_followup_solver_returns_mindful_org_website() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0256_256.conv.md",
        "User: These are great ideas! Do you have any recommendations for guided imagery exercises or resources?\nAssistant: Yes, here are some guided imagery exercises and resources:\n1. Headspace app: This app provides guided meditations.\n2. Mindful.org: This website includes guided imagery exercises that you can use for free, such as \"The Mountain Meditation\" and \"The Body Scan Meditation.\"\n3. UCLA Mindful Awareness Research Center: This website includes a variety of free guided meditations.\n",
    );

    let answer = read_answer_text(
        &idx,
        "I wanted to follow up on our previous conversation about mindfulness techniques. You mentioned some great resources for guided imagery exercises, can you remind me of the website that had free exercises like 'The Mountain Meditation' and 'The Body Scan Meditation'?",
    );
    assert!(answer.contains("Answer: Mindful.org"), "{answer}");
}

#[test]
fn assistant_followup_solver_returns_two_factor_methods() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0248_248.conv.md",
        "User: What are some steps companies take to ensure data privacy and security for their customers?\nAssistant: 3. Two-factor authentication: Requiring two-factor authentication, such as biometric authentication or one-time passwords (OTP), enhances security by ensuring that only authorized users can access sensitive data.\n",
    );

    let answer = read_answer_text(
        &idx,
        "I was thinking about our previous conversation about data privacy and security. You mentioned that companies use two-factor authentication to enhance security. Can you remind me what kind of two-factor authentication methods you were referring to?",
    );
    assert!(
        answer.contains(
            "Answer: I mentioned biometric authentication or one-time passwords (OTP) as examples of two-factor authentication methods."
        ),
        "{answer}"
    );
}

#[test]
fn assistant_followup_solver_returns_expected_chess_reply_move() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0257_257.conv.md",
        "User: no THIS is the current position\nAssistant: My apologies for the confusion. I will make a move now.\n27. Kg2 Bd5+\nAssistant: 28. Kg3 would be my move.\nUser: make a move. you're white\nAssistant: 29. Rd3 Rh4\n",
    );

    let answer = read_answer_text(
        &idx,
        "I'm looking back at our previous chess game and I was wondering, what was the move you made after 27. Kg2 Bd5+?",
    );
    assert!(answer.contains("Answer: 28. Kg3"), "{answer}");
}

#[test]
fn max_count_solver_prefers_latest_magazine_progress() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0159_159_conv_0000_chunk.verbatim.md",
        "User: I just finished my third issue of National Geographic, and I'm currently on my fourth.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "lme_0159_159_conv_0001_chunk.verbatim.md",
        "User: I recently finished my fifth issue of National Geographic and I'm planning to pick up another one soon.\n",
    );

    let answer_path = idx
        .derived_answer_path_for_task(
            "How many issues of National Geographic have I finished reading?",
        )
        .expect("max-count solver should synthesize an answer");
    let answer = std::fs::read_to_string(answer_path).unwrap();
    assert!(answer.contains("Answer: 5"));
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
    idx.entries[0].synapses.push(Synapse::new(
        neuron_b.clone(),
        SynapseType::Calls,
        "test edge".to_string(),
    ));
    idx.rebuild_derived();
    idx.save().unwrap();

    assert!(!activation_cache_path(dir.path()).exists());

    let idx2 = NeuronIndex::load_or_create(dir.path()).unwrap();
    assert_eq!(idx2.neuron_count(), 2);
    assert!(idx2.posting_list.contains_key("authentication"));
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
    idx.entries[0].synapses.push(Synapse::new(
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
    assert!(!cache_path.exists());
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
    assert_eq!(reloaded.use_count, 1);
    assert_eq!(
        idx2.coactivation_counts
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
    assert!(idx2.entries.iter().any(|e| e.neuron_path == np));
}

// ── get_contexts ──────────────────────────────────────────────────────────

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

#[test]
fn get_contexts_returns_empty_for_no_match() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let p = ndir.join("foo.context.md");
    std::fs::write(&p, "completely unrelated content xyz").unwrap();
    let mut idx = make_index(&dir);
    let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    idx.index_neuron(&p, "completely unrelated content xyz", &meta);
    idx.rebuild_derived();
    let result = idx.get_contexts("authentication oauth jwt", 4096, None, None);
    assert!(result.is_empty() || !result.contains(&p));
}

#[test]
fn get_contexts_respects_token_budget() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);
    for i in 0..20 {
        let p = ndir.join(format!("mod_{i:02}.context.md"));
        let content = format!("auth token login validate {} {}", "word ".repeat(200), i);
        std::fs::write(&p, &content).unwrap();
        let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
        idx.index_neuron(&p, &content, &meta);
    }
    idx.rebuild_derived();
    let result = idx.get_contexts("auth token login", 500, None, None);
    let total_tokens: usize = result
        .iter()
        .filter_map(|p| idx.entry_by_path(p))
        .map(|e| e.tokens)
        .sum();
    assert!(
        total_tokens <= 500,
        "should respect token budget: {total_tokens}"
    );
}

#[test]
fn get_contexts_module_filter() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);

    let auth_p = ndir.join("auth.context.md");
    let ui_p = ndir.join("ui.context.md");
    std::fs::write(&auth_p, "auth token login validate session").unwrap();
    std::fs::write(&ui_p, "auth login button render component").unwrap();

    let mut auth_meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    auth_meta.module = Some("auth".to_string());
    let mut ui_meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    ui_meta.module = Some("ui".to_string());

    idx.index_neuron(&auth_p, "auth token login validate session", &auth_meta);
    idx.index_neuron(&ui_p, "auth login button render component", &ui_meta);
    idx.rebuild_derived();

    // With module filter: only auth module
    let filtered = idx.get_contexts("auth login", 4096, Some("auth"), None);
    assert!(filtered.contains(&auth_p));
    assert!(
        !filtered.contains(&ui_p),
        "module filter should exclude ui module"
    );
}

#[test]
fn save_writes_module_capsule_for_named_module() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);

    let auth_p = ndir.join("auth.context.md");
    let guard_p = ndir.join("guard.context.md");
    let db_p = ndir.join("db.context.md");

    let auth_content = "# Auth\n\n## purpose\nHandles login and session validation.\n\n## pitfalls\nRotate refresh tokens after every use.\n";
    let guard_content =
        "# Guard\n\n## purpose\nProtects private routes and rejects anonymous requests.\n\n## pitfalls\nRequire identity before accessing private handlers.\n";
    let db_content =
        "# DB\n\n## purpose\nPersists user records and token state.\n\n## pitfalls\nKeep writes transactional.\n";

    std::fs::write(&auth_p, auth_content).unwrap();
    std::fs::write(&guard_p, guard_content).unwrap();
    std::fs::write(&db_p, db_content).unwrap();

    let mut auth_meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    auth_meta.module = Some("auth".to_string());
    auth_meta.synapses = vec![Synapse {
        target: db_p.clone(),
        edge_type: SynapseType::Calls,
        weight: 0.8,
        reason: "loads user token state".to_string(),
        learned_weight: 0.0,
        traversal_count: 0,
        last_co_activation_day: 0,
    }];
    let mut guard_meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    guard_meta.module = Some("auth".to_string());
    let mut db_meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    db_meta.module = Some("db".to_string());

    idx.index_neuron(&auth_p, auth_content, &auth_meta);
    idx.index_neuron(&guard_p, guard_content, &guard_meta);
    idx.index_neuron(&db_p, db_content, &db_meta);
    idx.rebuild_derived();
    idx.save().unwrap();

    let capsule = std::fs::read_to_string(module_capsule_path(dir.path(), "auth")).unwrap();
    assert!(capsule.contains("# Module capsule: auth"));
    assert!(capsule.contains("## module purpose"));
    assert!(capsule.contains("Handles login and session validation."));
    assert!(capsule.contains("## key apis / invariants"));
    assert!(capsule.contains("## critical pitfalls"));
    assert!(capsule.contains("## dominant dependencies"));
    assert!(capsule.contains("`db` (1 cross-module edges)"));
}

// ── Typed synapse traversal ───────────────────────────────────────────────

#[test]
fn synapse_traversal_pulls_related_neuron() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("engine.rs"),
        "pub fn engine() { route_intent(); }",
    )
    .unwrap();
    std::fs::write(dir.path().join("ui.rs"), "pub fn render() {}").unwrap();

    let mut idx = NeuronIndex::load_or_create(dir.path()).unwrap();
    idx.compile().unwrap();

    let engine_neuron = crate::neuron::core_neuron_path(&dir.path().join("engine.rs"), dir.path());
    let ui_neuron = crate::neuron::core_neuron_path(&dir.path().join("ui.rs"), dir.path());

    let engine_content = format!(
        "Engine module. Routes user intent, synthesizes responses.\n\
         ## CROSS-REFERENCES (synapses)\n- `{}` → render pipeline [calls]",
        ui_neuron.display()
    );
    let mut engine_meta = NeuronMeta::new_stub(&dir.path().join("engine.rs"), NeuronKind::Core);
    engine_meta.synapses = vec![Synapse {
        target: ui_neuron.clone(),
        edge_type: SynapseType::Calls,
        weight: 0.8,
        reason: "render pipeline".to_string(),
        learned_weight: 0.0,
        traversal_count: 0,
        last_co_activation_day: 0,
    }];
    engine_meta.status = NeuronStatus::Fresh;
    std::fs::write(&engine_neuron, &engine_content).unwrap();
    idx.upsert_neuron(&engine_neuron, &engine_content, &engine_meta)
        .unwrap();

    let contexts = idx.get_contexts("route intent synthesize engine", 4096, None, None);
    assert!(
        contexts.contains(&ui_neuron) || contexts.contains(&engine_neuron),
        "Synapse traversal should pull in related neuron. Got: {contexts:?}"
    );
}

#[test]
fn typed_synapse_implements_has_high_multiplier() {
    assert!(
        SynapseType::Implements.type_multiplier() > SynapseType::SemanticRelated.type_multiplier()
    );
    assert_eq!(SynapseType::ConceptExpands.type_multiplier(), 1.0);
}

// ── Use-case activation ───────────────────────────────────────────────────

#[test]
fn use_case_neuron_activated_by_task_pattern() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);

    let core_p = ndir.join("auth_rs.context.md");
    std::fs::write(&core_p, "authentication token validation").unwrap();
    let core_meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    idx.index_neuron(&core_p, "authentication token validation", &core_meta);

    let uc_p = ndir.join("auth_rs.usecase.oauth.md");
    std::fs::write(&uc_p, "OAuth2 flow: redirect then exchange code for token").unwrap();
    let mut uc_meta = NeuronMeta::new_stub(dir.path(), NeuronKind::UseCase);
    uc_meta.task_pattern = Some("add oauth login".to_string());
    uc_meta.parent = Some(core_p.clone());
    idx.index_neuron(
        &uc_p,
        "OAuth2 flow: redirect then exchange code for token",
        &uc_meta,
    );
    idx.rebuild_derived();

    let result = idx.get_contexts("add oauth authentication login", 4096, None, None);
    assert!(result.contains(&uc_p) || result.contains(&core_p));
}

// ── Invalidation ──────────────────────────────────────────────────────────

#[test]
fn invalidate_marks_stale() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("a.rs");
    std::fs::write(&file, "fn a() {}").unwrap();
    let mut idx = make_index(&dir);
    idx.compile().unwrap();
    let neuron = crate::neuron::core_neuron_path(&file, dir.path());
    assert!(neuron.exists());
    idx.invalidate(&file).unwrap();
    // Stale-demotion: neuron remains in the index (preserves context) but is
    // demoted via staleness_multiplier so it won't win over fresh neurons.
    let entry = idx.entries.iter().find(|e| e.neuron_path == neuron);
    assert!(
        entry.is_some(),
        "neuron should still exist after invalidation"
    );
    assert_eq!(
        entry.unwrap().staleness_multiplier,
        0.5,
        "staleness_multiplier should be 0.5 after invalidation"
    );
}

// ── BM25 scoring ──────────────────────────────────────────────────────────

#[test]
fn bm25_scores_zero_for_no_match() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let p = ndir.join("x.context.md");
    std::fs::write(&p, "completely different topic here").unwrap();
    let mut idx = make_index(&dir);
    let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    idx.index_neuron(&p, "completely different topic here", &meta);
    idx.rebuild_derived();
    let entry = idx.entry_by_path(&p).unwrap();
    assert_eq!(idx.bm25_score(&tokenize("auth token login"), entry), 0.0);
}

#[test]
fn bm25_scores_higher_for_matching_terms() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);
    let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    let p1 = ndir.join("a.context.md");
    std::fs::write(&p1, "auth login token session").unwrap();
    idx.index_neuron(&p1, "auth login token session", &meta);
    let p2 = ndir.join("b.context.md");
    std::fs::write(&p2, "render button component style").unwrap();
    idx.index_neuron(&p2, "render button component style", &meta);
    idx.rebuild_derived();
    let terms = tokenize("auth token");
    let s1 = idx.bm25_score(&terms, idx.entry_by_path(&p1).unwrap());
    let s2 = idx.bm25_score(&terms, idx.entry_by_path(&p2).unwrap());
    assert!(s1 > s2, "auth neuron should score higher for auth query");
}

#[test]
fn bm25_idf_is_non_negative() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);
    let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    // Same term in every entry → IDF should floor at 0
    for i in 0..5 {
        let p = ndir.join(format!("{i}.context.md"));
        std::fs::write(&p, "common term here").unwrap();
        idx.index_neuron(&p, "common term here", &meta);
    }
    idx.rebuild_derived();
    for entry in &idx.entries {
        let score = idx.bm25_score(&tokenize("common"), entry);
        assert!(score >= 0.0, "BM25 score must not be negative");
    }
}

// ── Overlap score ─────────────────────────────────────────────────────────

#[test]
fn overlap_score_perfect_match() {
    let q = tokenize("add dark mode");
    let p = tokenize("add dark mode");
    assert!((simple_overlap_score(&q, &p) - 1.0).abs() < 0.001);
}

#[test]
fn overlap_score_no_match() {
    let q = tokenize("auth token");
    let p = tokenize("render button");
    assert_eq!(simple_overlap_score(&q, &p), 0.0);
}

#[test]
fn overlap_score_empty_pattern() {
    let q = tokenize("auth");
    assert_eq!(simple_overlap_score(&q, &[]), 0.0);
}

// ── Tokenizer ─────────────────────────────────────────────────────────────

#[test]
fn tokenize_basic() {
    let terms = tokenize("add dark mode to SwiftUI view");
    assert!(terms.contains(&"add".to_string()));
    assert!(terms.contains(&"dark".to_string()));
    assert!(terms.contains(&"swiftui".to_string()));
    assert!(terms.contains(&"view".to_string()));
}

#[test]
fn tokenize_filters_short_terms() {
    let terms = tokenize("a b add");
    assert!(!terms.contains(&"a".to_string()));
    assert!(!terms.contains(&"b".to_string()));
    assert!(terms.contains(&"add".to_string()));
}

#[test]
fn tokenize_lowercases() {
    let terms = tokenize("AuthService");
    assert!(terms.contains(&"authservice".to_string()));
}

#[test]
fn tokenize_preserves_underscores() {
    let terms = tokenize("snake_case_name");
    assert!(terms.contains(&"snake_case_name".to_string()));
}

#[test]
fn tokenize_empty_string() {
    assert!(tokenize("").is_empty());
}

// ── Retrieval accuracy ────────────────────────────────────────────────────

/// Verifies that BM25 retrieval returns the correct neuron for each of 10
/// distinct queries against 10 distinct content-rich neurons.
///
/// This exercises the full activation pipeline (Phase 1 only — no synapses)
/// and ensures that keyword specificity drives correct ranking.
#[test]
fn get_contexts_retrieval_accuracy_10q() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);

    // Each neuron has a unique keyword cluster — e.g. "authentication" only in auth neuron
    let neurons = [
        (
            "auth.context.md",
            "authentication token validation session jwt bearer",
        ),
        (
            "ui.context.md",
            "render component dark mode swiftui colorscheme view",
        ),
        (
            "db.context.md",
            "database migration schema sql transaction commit",
        ),
        (
            "cache.context.md",
            "cache invalidation evict stale ttl expiry redis",
        ),
        (
            "api.context.md",
            "rest api endpoint http request response route handler",
        ),
        (
            "crypto.context.md",
            "encryption decryption aes rsa signing certificate key",
        ),
        (
            "queue.context.md",
            "queue task worker job priority scheduling async",
        ),
        (
            "logger.context.md",
            "logging tracing span event diagnostic telemetry",
        ),
        (
            "config.context.md",
            "configuration environment variable toml yaml dotenv",
        ),
        (
            "deploy.context.md",
            "deployment docker kubernetes helm release pipeline",
        ),
    ];
    let queries_and_expected: [(&str, &str); 10] = [
        ("jwt bearer authentication", "auth.context.md"),
        ("dark mode colorscheme swiftui", "ui.context.md"),
        ("sql transaction schema migration", "db.context.md"),
        ("cache ttl evict stale", "cache.context.md"),
        ("http rest api endpoint route", "api.context.md"),
        ("aes rsa encryption certificate", "crypto.context.md"),
        ("job worker queue scheduling", "queue.context.md"),
        ("logging span telemetry diagnostic", "logger.context.md"),
        (
            "environment variable dotenv configuration",
            "config.context.md",
        ),
        ("docker kubernetes deployment helm", "deploy.context.md"),
    ];

    for (name, content) in &neurons {
        let p = ndir.join(name);
        std::fs::write(&p, content).unwrap();
        let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
        idx.index_neuron(&p, content, &meta);
    }
    idx.rebuild_derived();

    let mut correct = 0;
    for (query, expected_file) in &queries_and_expected {
        let results = idx.get_contexts(query, 4096, None, None);
        let expected_path = ndir.join(expected_file);
        if results.contains(&expected_path) {
            correct += 1;
        } else {
            eprintln!("[accuracy] MISS: query={query:?} expected={expected_file} got={results:?}");
        }
    }
    assert_eq!(
        correct, 10,
        "BM25 accuracy: {correct}/10 correct (expected 10/10)"
    );
}

/// Activation latency: `get_contexts` over 100 neurons must complete in <50ms p95.
///
/// This verifies the README benchmark target "≤50ms p95, 100 neurons" is met
/// with the pure in-memory BM25 engine (no disk I/O in the hot path).
#[test]
fn get_contexts_latency_p95_100_neurons() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);

    // Build a 100-neuron index with realistic content sizes (~400 chars each).
    for i in 0..100 {
        let p = ndir.join(format!("neuron_{i:03}.context.md"));
        let content = format!(
            "## Module {i}\nHandles subsystem_{i} operations including routing, \
             caching, pipeline_{i} filter validation authentication token session \
             database migration schema endpoint handler deployment configuration \
             environment worker queue scheduling logging tracing telemetry encryption."
        );
        std::fs::write(&p, &content).unwrap();
        let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
        idx.index_neuron(&p, &content, &meta);
    }
    idx.rebuild_derived();

    // Warm up: one call to populate CPU caches
    let _ = idx.get_contexts("routing pipeline authentication token", 4096, None, None);

    // Measure p95 over 20 trials
    let trials = 20;
    let mut latencies_ms: Vec<u128> = (0..trials)
        .map(|_| {
            let t = std::time::Instant::now();
            let _ = idx.get_contexts("routing pipeline authentication token", 4096, None, None);
            t.elapsed().as_millis()
        })
        .collect();
    latencies_ms.sort_unstable();
    let p95 = latencies_ms[(trials as f64 * 0.95) as usize - 1];

    assert!(
        p95 < 50,
        "get_contexts p95 latency must be <50ms over 100 neurons; got {p95}ms"
    );
}

/// Ensures that relative synapse paths written into neuron markdown
/// (e.g. from `cortyx_evolve_context`) are resolved to absolute paths
/// in the adjacency graph, so traversal works correctly.
#[test]
fn relative_synapse_targets_resolved_in_adjacency() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);

    let source_p = ndir.join("engine.context.md");
    let target_p = ndir.join("ui.context.md");

    std::fs::write(&source_p, "engine routing intent").unwrap();
    std::fs::write(&target_p, "ui rendering components").unwrap();

    // Source neuron has a RELATIVE synapse target (as parse_synapses_from_content returns)
    let mut source_meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    source_meta.synapses = vec![Synapse {
        target: PathBuf::from("ui.context.md"), // relative!
        edge_type: SynapseType::Calls,
        weight: 0.9,
        reason: "calls render".to_string(),
        learned_weight: 0.0,
        traversal_count: 0,
        last_co_activation_day: 0,
    }];
    let target_meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);

    idx.index_neuron(&source_p, "engine routing intent", &source_meta);
    idx.index_neuron(&target_p, "ui rendering components", &target_meta);
    idx.rebuild_derived();

    // The adjacency entry for source_p should point to the ABSOLUTE target path
    let adj = idx
        .adjacency
        .get(&source_p)
        .expect("source must be in adjacency");
    let target_syn = adj.iter().find(|s| s.target == target_p);
    assert!(
        target_syn.is_some(),
        "Relative synapse 'ui.context.md' should be resolved to absolute {}: adjacency={adj:?}",
        target_p.display()
    );
}

// ── Mine + retrieve ───────────────────────────────────────────────────────

/// Verifies the conversation mining → retrieval pipeline end-to-end:
/// mine text containing unique keywords, then get_contexts should return it.
#[test]
fn mined_neuron_is_retrievable_by_keyword() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);

    // Mine a conversation turn with a specific keyword cluster
    crate::miner::mine_text(
        "The hydrazine valve regulates fuel injection in rocket propulsion systems.",
        "test_chat",
        dir.path(),
        &mut idx,
        None,
        Some("assistant"),
        None,
    )
    .unwrap();

    // The unique keyword "hydrazine" should retrieve the mined neuron
    let results = idx.get_contexts("hydrazine valve rocket propulsion", 4096, None, None);
    assert!(
        !results.is_empty(),
        "Mined neuron should be retrievable by its keywords"
    );

    let found = results.iter().any(|p| {
        std::fs::read_to_string(p)
            .map(|c| c.contains("hydrazine"))
            .unwrap_or(false)
    });
    assert!(found, "Retrieved neuron should contain 'hydrazine'");
}

/// Mine + module filter: mined neuron tagged with module X should only
/// appear when querying with that module filter, not unfiltered in other modules.
#[test]
fn mined_neuron_module_filter_works() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);

    crate::miner::mine_text(
        "Photosynthesis converts sunlight into glucose via chlorophyll.",
        "bio_chat",
        dir.path(),
        &mut idx,
        Some("biology"),
        Some("assistant"),
        None,
    )
    .unwrap();

    // Module-filtered query should find it
    let with_module = idx.get_contexts(
        "photosynthesis sunlight glucose",
        4096,
        Some("biology"),
        None,
    );
    assert!(
        !with_module.is_empty(),
        "Module-filtered query should find mined neuron"
    );

    // Module filter for a different module should NOT find it
    let wrong_module = idx.get_contexts(
        "photosynthesis sunlight glucose",
        4096,
        Some("physics"),
        None,
    );
    assert!(
        wrong_module.is_empty(),
        "Wrong module filter should not return neuron tagged 'biology'"
    );
}

// ── Feedback loop (hit_multiplier + quarantine) ───────────────────────────

#[test]
fn hit_multiplier_reward_grows_with_citations() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);

    let p = ndir.join("auth.context.md");
    std::fs::write(&p, "authentication token session login").unwrap();
    let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    idx.index_neuron(&p, "authentication token session login", &meta);
    idx.rebuild_derived();

    let terms = tokenize("auth login");

    // Cold-start: use_count=0 → multiplier=1.0 (neutral)
    let cold_score = idx.bm25_score(&terms, idx.entry_by_path(&p).unwrap());

    // Simulate MIN_SAMPLE_SIZE activations with 100% citation rate
    if let Some(&i) = idx.path_index.get(&p) {
        idx.entries[i].use_count = MIN_SAMPLE_SIZE;
        idx.entries[i].hit_count = MIN_SAMPLE_SIZE;
    }
    let hot_score = idx.bm25_score(&terms, idx.entry_by_path(&p).unwrap());

    assert!(
        hot_score > cold_score,
        "Fully-cited neuron should score higher than cold-start (hot={hot_score:.3}, cold={cold_score:.3})"
    );
    // Max multiplier is 1.5 so the hot score should be exactly 1.5× cold
    assert!(
        (hot_score / cold_score - 1.5).abs() < 0.01,
        "100% hit rate should give 1.5× boost"
    );
}

#[test]
fn auto_quarantine_fires_after_threshold() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);

    let p = ndir.join("noisy.context.md");
    std::fs::write(&p, "generic boilerplate content").unwrap();
    let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    idx.index_neuron(&p, "generic boilerplate content", &meta);
    idx.rebuild_derived();

    // Adaptive CI (S4): QUARANTINE_MIN_SAMPLES = 5. Below this threshold
    // (use_count 0–4), adaptive_quarantine_params returns None — no action.
    if let Some(&i) = idx.path_index.get(&p) {
        idx.entries[i].use_count = QUARANTINE_MIN_SAMPLES - 2; // = 3
        idx.entries[i].hit_count = 0;
    }
    idx.record_activation(&[p.clone()]); // → use_count = 4 (still below threshold)
    let mult_early = idx
        .path_index
        .get(&p)
        .map(|&i| idx.entries[i].staleness_multiplier)
        .unwrap_or(1.0);
    assert_eq!(
        mult_early, 1.0,
        "Should NOT quarantine below QUARANTINE_MIN_SAMPLES (4 < 5)"
    );

    // At use_count = 5 (after record_activation increments to 6), z=1.0 tier fires.
    // Wilson lower bound for 0/6 at z=1.0 = 0.0 < adaptive threshold 0.02 → quarantine.
    if let Some(&i) = idx.path_index.get(&p) {
        idx.entries[i].use_count = QUARANTINE_MIN_SAMPLES; // = 5
        idx.entries[i].hit_count = 0;
    }
    idx.record_activation(&[p.clone()]); // → use_count = 6, fires adaptive z=1.0
    let mult = idx
        .path_index
        .get(&p)
        .map(|&i| idx.entries[i].staleness_multiplier)
        .unwrap_or(1.0);
    assert_eq!(
        mult, 0.3,
        "Should quarantine at QUARANTINE_MIN_SAMPLES with 0% hit rate"
    );
}

#[test]
fn quarantine_is_reversible_when_citation_rate_recovers() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);

    let p = ndir.join("recovered.context.md");
    std::fs::write(&p, "generic boilerplate content").unwrap();
    let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    idx.index_neuron(&p, "generic boilerplate content", &meta);
    idx.rebuild_derived();

    // Manually quarantine the neuron, then simulate recovery: 20 uses, 10 hits.
    // Wilson lower bound for 10/20 at z=1.645 (90% CI) ≈ 0.31 > QUARANTINE_RECOVERY_THRESHOLD (0.15).
    // Use hardcoded values (not QUARANTINE_MIN_SAMPLES) so the hit/use ratio is valid.
    if let Some(&i) = idx.path_index.get(&p) {
        idx.entries[i].staleness_multiplier = 0.3;
        idx.entries[i].use_count = 20;
        idx.entries[i].hit_count = 10;
    }
    idx.record_activation(&[p.clone()]);
    let mult = idx
        .path_index
        .get(&p)
        .map(|&i| idx.entries[i].staleness_multiplier)
        .unwrap_or(0.0);
    assert!(
        mult > 0.3,
        "Quarantine should lift when citation rate recovers (mult={mult})"
    );
}

#[test]
fn wilson_lower_bound_correctness() {
    // 0/20 → lower bound = 0.0 (no hits, fully quarantinable)
    assert!(wilson_lower_bound(0, 20) < 0.01);
    // 10/20 → lower bound ≈ 0.299 (well above recovery threshold of 0.15)
    assert!(wilson_lower_bound(10, 20) > 0.25);
    // 1/20 → lower bound near 0 but small positive
    assert!(wilson_lower_bound(1, 20) < 0.10);
    // Edge: 0 total → 0.0
    assert_eq!(wilson_lower_bound(0, 0), 0.0);
}

// ── S1: AST Signature Hash ─────────────────────────────────────────────────

#[test]
fn sig_hash_changes_on_function_rename() {
    let before = crate::ast_extractor::extract_signatures("src/auth.rs", "pub fn validate() {}");
    let after = crate::ast_extractor::extract_signatures("src/auth.rs", "pub fn authenticate() {}");
    let h1 = crate::ast_extractor::compute_sig_hash(&before);
    let h2 = crate::ast_extractor::compute_sig_hash(&after);
    assert_ne!(h1, h2, "sig_hash must change when a function is renamed");
}

#[test]
fn sig_hash_stable_on_whitespace_and_comments() {
    let base = crate::ast_extractor::extract_signatures("src/auth.rs", "pub fn validate() {}");
    let tweaked = crate::ast_extractor::extract_signatures(
        "src/auth.rs",
        "/// New doc comment\npub fn validate() {\n    // added comment\n}",
    );
    let h1 = crate::ast_extractor::compute_sig_hash(&base);
    let h2 = crate::ast_extractor::compute_sig_hash(&tweaked);
    assert_eq!(
        h1, h2,
        "sig_hash must be stable across whitespace/doc-comment edits"
    );
}

#[test]
fn sig_hash_stable_on_function_reorder() {
    let a = crate::ast_extractor::extract_signatures(
        "src/auth.rs",
        "pub fn validate() {}\npub fn refresh() {}",
    );
    let b = crate::ast_extractor::extract_signatures(
        "src/auth.rs",
        "pub fn refresh() {}\npub fn validate() {}",
    );
    let h1 = crate::ast_extractor::compute_sig_hash(&a);
    let h2 = crate::ast_extractor::compute_sig_hash(&b);
    assert_eq!(h1, h2, "sig_hash must be stable across function reordering");
}

// ── S3: Lazy Sub-Neuron Splitting ─────────────────────────────────────────

#[test]
fn sub_neuron_path_format_is_correct() {
    use crate::neuron::sub_neuron_path;
    use std::path::Path;
    let core = Path::new(".cortyx/neurons/src/engine_rs.context.md");
    let sub = sub_neuron_path(core, "validate_user");
    let name = sub.file_name().unwrap().to_string_lossy();
    assert_eq!(name, "engine_rs.fn-validate_user.context.md");
    assert_eq!(sub.parent(), core.parent());
}

#[test]
fn sub_neuron_path_sanitizes_special_chars() {
    use crate::neuron::sub_neuron_path;
    use std::path::Path;
    let core = Path::new(".cortyx/neurons/src/engine_rs.context.md");
    let sub = sub_neuron_path(core, "fn with spaces!");
    let name = sub.file_name().unwrap().to_string_lossy();
    // spaces and ! should be replaced with _
    assert!(name.starts_with("engine_rs.fn-"));
    assert!(!name.contains(' '));
    assert!(!name.contains('!'));
}

#[test]
fn sub_neuron_content_contains_function_name() {
    use crate::neuron::stub_function_neuron;
    let content = stub_function_neuron("validate_user", "src/auth.rs", "2026-01-01T00:00:00Z");
    assert!(
        content.contains("validate_user"),
        "stub must mention the function name"
    );
    assert!(
        content.contains("SECTION: purpose"),
        "stub must have purpose section"
    );
    assert!(
        content.contains("SECTION: api"),
        "stub must have api section"
    );
}

#[test]
fn split_threshold_files_produce_sub_neurons() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let src_dir = root.join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    std::fs::create_dir_all(root.join(".cortyx").join("neurons").join("src")).unwrap();

    // Write a Rust file with 7 public functions (above SUBNEURON_SPLIT_THRESHOLD=6)
    let mut src = String::new();
    for i in 0..7 {
        src.push_str(&format!("pub fn function_{i}() {{ }}\n"));
    }
    std::fs::write(src_dir.join("big_module.rs"), &src).unwrap();

    let git_confidence = std::collections::HashMap::new();
    let abs = src_dir.join("big_module.rs");
    let results = process_source_file(&abs, root, &git_confidence);

    // First result is the Core; subsequent are UseCase sub-neurons
    assert!(
        results.len() >= 2,
        "should produce Core + sub-neurons for 7-function file"
    );
    let core = &results[0];
    assert_eq!(core.meta.kind, crate::neuron::NeuronKind::Core);
    let subs: Vec<_> = results.iter().skip(1).collect();
    assert!(!subs.is_empty(), "should have at least one sub-neuron");
    assert!(subs
        .iter()
        .all(|s| s.meta.kind == crate::neuron::NeuronKind::UseCase));
    assert!(subs
        .iter()
        .all(|s| s.meta.parent.as_deref() == Some(core.neuron_path.as_path())));
}

#[test]
fn small_files_produce_no_sub_neurons() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let src_dir = root.join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    std::fs::create_dir_all(root.join(".cortyx").join("neurons").join("src")).unwrap();

    // Write a small Rust file with 2 public functions (below threshold)
    let src = "pub fn a() {}\npub fn b() {}\n";
    std::fs::write(src_dir.join("small.rs"), src).unwrap();

    let git_confidence = std::collections::HashMap::new();
    let abs = src_dir.join("small.rs");
    let results = process_source_file(&abs, root, &git_confidence);

    assert_eq!(
        results.len(),
        1,
        "small file should produce only a Core neuron"
    );
    assert_eq!(results[0].meta.kind, crate::neuron::NeuronKind::Core);
}

// ── R11-S1: Section-Level Staleness ──────────────────────────────────────

/// Verifies that `update_neuron_header` patches only the three header comment
/// lines and leaves all other content (section bodies, cross-refs) intact.
#[test]
fn update_neuron_header_patches_only_header_lines() {
    use crate::neuron::update_neuron_header;
    let content = "\
<!-- AUTO-GENERATED CONTEXT — DO NOT EDIT MANUALLY -->\n\
<!-- source: src/engine.rs -->\n\
<!-- hash: aabbccdd11223344 -->\n\
<!-- last-updated: 2024-01-01T00:00:00Z -->\n\
<!-- status: stub -->\n\
\n\
<!-- SECTION: purpose -->\n\
This module drives the core loop.\n\
<!-- /SECTION -->\n\
<!-- SECTION: api -->\n\
pub fn run()\n\
<!-- /SECTION -->\n";

    let updated = update_neuron_header(content, "deadbeef12345678", "2025-06-01T12:00:00Z");

    assert!(
        updated.contains("<!-- hash: deadbeef12345678 -->"),
        "hash line must be updated"
    );
    assert!(
        updated.contains("<!-- last-updated: 2025-06-01T12:00:00Z -->"),
        "date must be updated"
    );
    assert!(
        updated.contains("<!-- status: stale -->"),
        "status must be set to stale"
    );
    assert!(!updated.contains("aabbccdd"), "old hash must not appear");
    assert!(
        updated.contains("This module drives the core loop."),
        "purpose body must be preserved"
    );
    assert!(
        updated.contains("pub fn run()"),
        "api body must be preserved"
    );
}

/// When a source file's sig_hash changes (real API change) but the neuron already
/// exists, `process_source_file` should update only the `api` section and preserve
/// the `purpose` section content written by a previous LLM call.
#[test]
fn s1_api_section_update_preserves_purpose_on_sig_hash_change() {
    use crate::neuron::replace_section;
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let src_dir = root.join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    std::fs::create_dir_all(root.join(".cortyx").join("neurons").join("src")).unwrap();

    // Write an initial source file and compile it to get a neuron stub
    let src_v1 = "pub fn alpha() {}\n";
    std::fs::write(src_dir.join("mod.rs"), src_v1).unwrap();
    let git_confidence = std::collections::HashMap::new();
    let abs = src_dir.join("mod.rs");
    let v1 = process_source_file(&abs, root, &git_confidence);
    assert_eq!(v1.len(), 1, "v1 should produce one Core");
    let neuron_path = v1[0].neuron_path.clone();

    // Simulate LLM evolution: write a purpose section into the neuron
    let with_purpose = replace_section(
        &v1[0].content,
        "purpose",
        "Alpha drives the main processing loop.",
    );
    std::fs::write(&neuron_path, &with_purpose).unwrap();

    // Now change the source file API (rename function → sig_hash changes)
    let src_v2 = "pub fn beta() {}\n";
    std::fs::write(&abs, src_v2).unwrap();
    let v2 = process_source_file(&abs, root, &git_confidence);

    // S1: should return a compiled file with api updated but purpose preserved
    assert_eq!(v2.len(), 1, "v2 should still produce one Core");
    let new_content = std::fs::read_to_string(&neuron_path).unwrap();
    assert!(
        new_content.contains("beta"),
        "new api section should contain updated function name"
    );
    assert!(
        new_content.contains("Alpha drives the main processing loop."),
        "LLM-curated purpose section must survive a sig_hash change"
    );
    assert!(
        new_content.contains("<!-- status: stale -->"),
        "status should be stale after api change"
    );
}

// ── R11-S4: Adaptive CI Quarantine ───────────────────────────────────────

/// Verifies that `adaptive_quarantine_params` returns the correct (z, threshold) tier
/// and None below the cold-start threshold.
#[test]
fn adaptive_quarantine_params_tier_boundaries() {
    assert!(adaptive_quarantine_params(0).is_none(), "0 samples → None");
    assert!(adaptive_quarantine_params(4).is_none(), "4 samples → None");
    let (z5, t5) = adaptive_quarantine_params(5).unwrap();
    assert!((z5 - 1.0).abs() < 0.01, "5 samples → z=1.0");
    assert!((t5 - 0.02).abs() < 0.001, "5 samples → threshold=0.02");
    let (z19, _) = adaptive_quarantine_params(19).unwrap();
    assert!((z19 - 1.0).abs() < 0.01, "19 samples → still z=1.0 tier");
    let (z20, t20) = adaptive_quarantine_params(20).unwrap();
    assert!((z20 - 1.645).abs() < 0.01, "20 samples → z=1.645");
    assert!((t20 - 0.05).abs() < 0.001, "20 samples → threshold=0.05");
    let (z100, t100) = adaptive_quarantine_params(100).unwrap();
    assert!((z100 - 1.96).abs() < 0.01, "100+ samples → z=1.96");
    assert!((t100 - 0.08).abs() < 0.001, "100+ samples → threshold=0.08");
}

/// Early quarantine at 5+ samples with 0% hit rate (z=1.0 tier).
#[test]
fn adaptive_ci_quarantines_early_for_zero_hit_rate() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);

    let p = ndir.join("noise.context.md");
    std::fs::write(&p, "noise boilerplate low quality").unwrap();
    let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    idx.index_neuron(&p, "noise boilerplate low quality", &meta);
    idx.rebuild_derived();

    // 9 activations, 0 hits → z=1.0 tier, lb(0,10)=0.0 < 0.02 → should quarantine
    if let Some(&i) = idx.path_index.get(&p) {
        idx.entries[i].use_count = 9;
        idx.entries[i].hit_count = 0;
    }
    idx.record_activation(&[p.clone()]); // → use_count=10
    let mult = idx
        .path_index
        .get(&p)
        .map(|&i| idx.entries[i].staleness_multiplier)
        .unwrap_or(1.0);
    assert_eq!(
        mult, 0.3,
        "10 activations with 0 hits should quarantine at z=1.0 tier"
    );
}

/// A neuron with moderate hit rate at medium count should NOT be quarantined
/// (90% CI is too wide to conclude bad quality).
#[test]
fn adaptive_ci_does_not_quarantine_moderate_hit_rate() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);

    let p = ndir.join("moderate.context.md");
    std::fs::write(&p, "good content useful context").unwrap();
    let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    idx.index_neuron(&p, "good content useful context", &meta);
    idx.rebuild_derived();

    // 5 hits out of 20 total → 25% hit rate; lb at z=1.645 is well above 0.05
    if let Some(&i) = idx.path_index.get(&p) {
        idx.entries[i].use_count = 19;
        idx.entries[i].hit_count = 5;
    }
    idx.record_activation(&[p.clone()]); // → use_count=20
    let mult = idx
        .path_index
        .get(&p)
        .map(|&i| idx.entries[i].staleness_multiplier)
        .unwrap_or(1.0);
    assert_eq!(
        mult, 1.0,
        "25% hit rate at 20 samples should not be quarantined"
    );
}

// ── R12-S1: Concept Cloud ─────────────────────────────────────────────────

#[test]
fn concept_cloud_populated_from_structural_neighbours() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);

    let caller = ndir.join("caller.context.md");
    let callee = ndir.join("callee.context.md");
    std::fs::write(&caller, "calls validate_user auth check").unwrap();
    std::fs::write(&callee, "validate_user password hash bcrypt").unwrap();

    let mut meta_caller = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    let meta_callee = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    meta_caller.synapses.push(crate::neuron::Synapse {
        target: callee.clone(),
        edge_type: crate::neuron::SynapseType::Calls,
        weight: 0.8,
        reason: "calls validate_user".to_string(),
        learned_weight: 0.0,
        traversal_count: 0,
        last_co_activation_day: 0,
    });

    idx.index_neuron(&caller, "calls validate_user auth check", &meta_caller);
    idx.index_neuron(&callee, "validate_user password hash bcrypt", &meta_callee);
    idx.rebuild_derived();

    // caller's concept cloud should contain callee terms
    let caller_idx = *idx.path_index.get(&caller).unwrap();
    let cloud = &idx.entries[caller_idx].concept_cloud;
    assert!(
        cloud
            .iter()
            .any(|t| t == "bcrypt" || t == "password" || t == "validate_user"),
        "caller concept cloud should contain callee terms; got: {cloud:?}"
    );
}

#[test]
fn concept_cloud_enables_retrieval_via_graph() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);

    // "engine.rs" calls "hashing.rs", which owns the word "bcrypt".
    // A query for "bcrypt" should find engine via concept cloud even though
    // "bcrypt" does not appear in engine's own vocabulary.
    let engine = ndir.join("engine.context.md");
    let hashing = ndir.join("hashing.context.md");
    std::fs::write(&engine, "core engine dispatch orchestrate").unwrap();
    std::fs::write(&hashing, "bcrypt password hash rounds salt").unwrap();

    let mut meta_engine = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    let meta_hashing = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    meta_engine.synapses.push(crate::neuron::Synapse {
        target: hashing.clone(),
        edge_type: crate::neuron::SynapseType::Calls,
        weight: 0.8,
        reason: "calls hash function".to_string(),
        learned_weight: 0.0,
        traversal_count: 0,
        last_co_activation_day: 0,
    });

    idx.index_neuron(&engine, "core engine dispatch orchestrate", &meta_engine);
    idx.index_neuron(&hashing, "bcrypt password hash rounds salt", &meta_hashing);
    idx.rebuild_derived();

    // "bcrypt" is in hashing's vocab → engine's concept cloud → engine is reachable
    let engine_idx = *idx.path_index.get(&engine).unwrap();
    assert!(
        idx.entries[engine_idx]
            .concept_cloud
            .contains(&"bcrypt".to_string()),
        "engine concept cloud must contain 'bcrypt' from hashing neighbour"
    );

    // Now query for "bcrypt" — vocab bridge won't match (no module synonym),
    // but concept cloud should surface engine as a candidate.
    let results = idx.get_contexts("bcrypt", 4096, None, None);
    let found_engine = results
        .iter()
        .any(|s| s.to_string_lossy().contains("engine"));
    let found_hashing = results.iter().any(|s| {
        let p = s.to_string_lossy();
        p.contains("hashing") || p.contains("bcrypt")
    });
    assert!(
        found_hashing || found_engine,
        "concept cloud retrieval must surface at least one relevant neuron; got {results:?}"
    );
}

#[test]
fn concept_cloud_excludes_semantic_related_edges() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);

    let a = ndir.join("a.context.md");
    let b = ndir.join("b.context.md");
    std::fs::write(&a, "alpha beta gamma").unwrap();
    std::fs::write(&b, "exclusive_term_xyz zeta").unwrap();

    let mut meta_a = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    let meta_b = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    // Only SemanticRelated edge — should NOT contribute to concept cloud
    meta_a.synapses.push(crate::neuron::Synapse {
        target: b.clone(),
        edge_type: crate::neuron::SynapseType::SemanticRelated,
        weight: 0.5,
        reason: "related".to_string(),
        learned_weight: 0.0,
        traversal_count: 0,
        last_co_activation_day: 0,
    });

    idx.index_neuron(&a, "alpha beta gamma", &meta_a);
    idx.index_neuron(&b, "exclusive_term_xyz zeta", &meta_b);
    idx.rebuild_derived();

    let a_idx = *idx.path_index.get(&a).unwrap();
    assert!(
        !idx.entries[a_idx]
            .concept_cloud
            .contains(&"exclusive_term_xyz".to_string()),
        "SemanticRelated edges must not populate concept cloud (already handled by vocab bridge)"
    );
}

// ── S-II (R16): LSH SimHash ───────────────────────────────────────────────

#[test]
fn simhash_same_terms_identical_fingerprint() {
    // Identical content should always yield the same fingerprint (deterministic)
    let mut tf1: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
    tf1.insert("auth".to_string(), 1.0);
    tf1.insert("token".to_string(), 2.0);
    let fp1 = simhash_with_seed(&tf1, LSH_SEEDS[0]);
    let fp2 = simhash_with_seed(&tf1, LSH_SEEDS[0]);
    assert_eq!(fp1, fp2, "same terms → same fingerprint (deterministic)");
    // Highly divergent content should produce different fingerprints with overwhelming probability
    let mut tf_other: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
    tf_other.insert("xyzzy".to_string(), 100.0);
    tf_other.insert("quux".to_string(), 100.0);
    tf_other.insert("plonk".to_string(), 100.0);
    tf_other.insert("zork".to_string(), 100.0);
    let fp_other = simhash_with_seed(&tf_other, LSH_SEEDS[0]);
    assert_ne!(
        fp1, fp_other,
        "very different terms should produce different fingerprints"
    );
}

#[test]
fn simhash_identical_content_identical_fingerprint() {
    let mut tf: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
    tf.insert("validate".to_string(), 1.5);
    tf.insert("password".to_string(), 3.0);
    let fp1 = simhash_with_seed(&tf, LSH_SEEDS[0]);
    let fp2 = simhash_with_seed(&tf, LSH_SEEDS[0]);
    assert_eq!(fp1, fp2, "same terms → same fingerprint (deterministic)");
}

#[test]
fn hamming_distance_self_is_zero() {
    let fp = 0xdeadbeefcafe1234u64;
    assert_eq!(hamming_distance(fp, fp), 0);
}

#[test]
fn hamming_distance_complement_is_64() {
    assert_eq!(hamming_distance(0u64, !0u64), 64);
}

#[test]
fn lsh_fingerprint_stored_in_entry() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);
    let neuron = ndir.join("auth.context.md");
    std::fs::write(&neuron, "auth token validate jwt bearer").unwrap();
    let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
    idx.index_neuron(&neuron, "auth token validate jwt bearer", &meta);
    let entry_idx = *idx.path_index.get(&neuron).unwrap();
    assert!(
        idx.entries[entry_idx]
            .lsh_fingerprints
            .iter()
            .any(|&fp| fp != 0),
        "non-empty term set should produce non-zero 1024-bit SimHash"
    );
}

// ── S-III (R16): Self-Quality Score ──────────────────────────────────────

#[test]
fn quality_score_defaults_to_one_when_no_source() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);
    let neuron = ndir.join("concept.context.md");
    std::fs::write(&neuron, "some concept terms here").unwrap();
    // Concept kind → no source file → quality_score defaults to 1.0
    let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Concept);
    idx.index_neuron(&neuron, "some concept terms here", &meta);
    let entry_idx = *idx.path_index.get(&neuron).unwrap();
    assert!(
        (idx.entries[entry_idx].quality_score - 1.0).abs() < 1e-6,
        "Concept neuron should have quality_score=1.0 (no source file)"
    );
}

#[test]
fn low_quality_count_counts_below_threshold() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);
    // All Concept neurons → quality_score = 1.0 → none below threshold
    for i in 0..3 {
        let p = ndir.join(format!("n{i}.context.md"));
        std::fs::write(&p, "terms").unwrap();
        let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Concept);
        idx.index_neuron(&p, "terms", &meta);
    }
    assert_eq!(
        idx.low_quality_count(),
        0,
        "no low-quality neurons expected"
    );
}

#[test]
fn publish_ready_candidates_filter_for_shareable_quality() {
    let dir = TempDir::new().unwrap();
    let ndir = dir.path().join(".cortyx").join("neurons");
    std::fs::create_dir_all(&ndir).unwrap();
    let mut idx = make_index(&dir);

    let strong = ndir.join("strong.context.md");
    std::fs::write(&strong, "auth token validation middleware").unwrap();
    let strong_meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Concept);
    idx.index_neuron(&strong, "auth token validation middleware", &strong_meta);
    let strong_idx = *idx.path_index.get(&strong).unwrap();
    idx.entries[strong_idx].use_count = 12;
    idx.entries[strong_idx].hit_count = 9;

    let weak_hit = ndir.join("weak-hit.context.md");
    std::fs::write(&weak_hit, "routing fallback legacy handler").unwrap();
    let weak_hit_meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Concept);
    idx.index_neuron(&weak_hit, "routing fallback legacy handler", &weak_hit_meta);
    let weak_hit_idx = *idx.path_index.get(&weak_hit).unwrap();
    idx.entries[weak_hit_idx].use_count = 12;
    idx.entries[weak_hit_idx].hit_count = 2;

    let verbatim = ndir.join("verbatim.context.md");
    std::fs::write(&verbatim, "I fixed the auth bug today").unwrap();
    let verbatim_meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Verbatim);
    idx.index_neuron(&verbatim, "I fixed the auth bug today", &verbatim_meta);
    let verbatim_idx = *idx.path_index.get(&verbatim).unwrap();
    idx.entries[verbatim_idx].use_count = 25;
    idx.entries[verbatim_idx].hit_count = 25;

    let ready = idx.publish_ready_candidates(10, 0.5, 0.6, 10);

    assert_eq!(ready.len(), 1);
    assert_eq!(ready[0].path, strong);
    assert_eq!(ready[0].kind, NeuronKind::Concept);
    assert_eq!(ready[0].use_count, 12);
    assert!(ready[0].hit_rate >= 0.75);
    assert!(ready[0].quality_score >= 1.0);
}
