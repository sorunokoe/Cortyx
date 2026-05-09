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
