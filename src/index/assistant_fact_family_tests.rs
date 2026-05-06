use super::assistant_fact_extractors::{parse_assistant_fact_query, AssistantFactQuery};
use super::assistant_fact_query_support::assistant_fact_required_terms;
use super::assistant_fact_support::extract_entity_candidate;
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
        .synthetic_assistant_fact_recall_answer(task, &task.to_ascii_lowercase())
        .or_else(|| idx.derived_answer_path_for_task(task))
        .expect("expected synthetic answer");
    std::fs::read_to_string(path).unwrap()
}

#[test]
fn assistant_fact_query_uses_who_descriptor_clause_for_required_terms() {
    let task = "I was going through our previous conversation about the impact of the political climate in Catalonia on its literature and music. Can you remind me of the example you gave of a Spanish-Catalan singer-songwriter who supports unity between Catalonia and Spain?";
    let required_terms = assistant_fact_required_terms(&task.to_ascii_lowercase());
    assert!(required_terms.iter().any(|term| term == "unity"));
    assert!(required_terms.iter().any(|term| term == "spain"));
}

#[test]
fn assistant_fact_query_extends_money_allocation_with_topic_terms() {
    let task = "I'm looking back at our previous chat about the DHL Wellness Retreats campaign. Can you remind me how much was allocated for influencer marketing in the campaign plan?";
    let query = parse_assistant_fact_query(task, &task.to_ascii_lowercase()).unwrap();
    let (required_terms, topic_terms) = match query {
        AssistantFactQuery::Value(query) => (query.required_terms, query.topic_terms),
        other => panic!("unexpected query shape: {other:?}"),
    };
    assert!(required_terms.iter().any(|term| term == "influencer"));
    assert!(required_terms.iter().any(|term| term == "wellness"));
    assert!(required_terms.iter().any(|term| term == "retreats"));
    assert!(topic_terms.iter().any(|term| term == "wellness"));
    assert!(topic_terms.iter().any(|term| term == "retreats"));
    assert!(topic_terms.iter().any(|term| term == "campaign"));
    assert!(!topic_terms.iter().any(|term| term == "allocated"));
    assert!(!topic_terms.iter().any(|term| term == "influencer"));
}

#[test]
fn assistant_fact_query_extends_year_terms_with_subject_clause() {
    let task = "I'm looking back at our previous conversation about the Bajimaya v Reward Homes Pty Ltd case. Can you remind me what year the construction of the house began?";
    let query = parse_assistant_fact_query(task, &task.to_ascii_lowercase()).unwrap();
    let required_terms = match query {
        AssistantFactQuery::Value(query) => query.required_terms,
        other => panic!("unexpected query shape: {other:?}"),
    };
    assert!(required_terms.iter().any(|term| term == "construction"));
    assert!(required_terms.iter().any(|term| term == "house"));
    assert!(required_terms.iter().any(|term| term == "bajimaya"));
}

#[test]
fn assistant_fact_query_uses_named_subject_clause_for_required_terms() {
    let task = "I'm planning my trip to Amsterdam again and I was wondering, what was the name of that hostel near the Red Light District that you recommended last time?";
    let required_terms = assistant_fact_required_terms(&task.to_ascii_lowercase());
    assert!(required_terms.iter().any(|term| term == "hostel"));
    assert!(required_terms.iter().any(|term| term == "light"));
    assert!(required_terms.iter().any(|term| term == "district"));
}

#[test]
fn recalls_named_thing_from_dash_label() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "dessert.conv.md",
        "Assistant: The Sugar Factory - A sweet shop located at Icon Park that offers giant milkshakes.\n",
    );
    let answer = read_answer_text(
        &idx,
        "I'm planning to revisit Orlando. I was wondering if you could remind me of that unique dessert shop with the giant milkshakes we talked about last time?",
    );
    assert!(
        answer.contains("The Sugar Factory at Icon Park"),
        "{answer}"
    );
}

#[test]
fn recalls_named_thing_from_dessert_list_instead_of_intro_phrase() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "dessert-list.conv.md",
        "Assistant: Absolutely! Here are some fun dessert spots that your family might enjoy after dinner:\n\
         1. The Sugar Factory - A sweet shop located at Icon Park that offers giant milkshakes.\n\
         2. Wondermade - A gourmet marshmallow shop just north of Orlando.\n",
    );
    let answer = read_answer_text(
        &idx,
        "I'm planning to revisit Orlando. I was wondering if you could remind me of that unique dessert shop with the giant milkshakes we talked about last time?",
    );
    assert!(
        answer.contains("The Sugar Factory at Icon Park"),
        "{answer}"
    );
    assert!(!answer.contains("Absolutely Here"), "{answer}");
}

#[test]
fn recalls_named_thing_from_vatican_food_recommendation_instead_of_generic_phrase() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "vatican-food.conv.md",
        "Assistant: I do not have personal experiences or preferences. however, many visitors suggest that the sistine chapel is a must-see.\n\
         Assistant: There are many great food options near the Vatican. One popular option is Pizzarium. There's also Roscioli, a famous deli that serves the best cured meats, cheeses, and traditional Roman cuisine.\n",
    );
    let answer = read_answer_text(
        &idx,
        "I'm planning to visit the Vatican again and I was wondering if you could remind me of the name of that famous deli near the Vatican that serves the best cured meats and cheeses?",
    );
    assert!(answer.contains("Roscioli"), "{answer}");
    assert!(!answer.contains("Do Not"), "{answer}");
}

#[test]
fn recalls_phone_number_from_contact_block() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "speyer.conv.md",
        "Assistant: Certainly! You can contact the tourism board of Speyer using the following details:\n\
         Speyer Tourismus Marketing GmbH\n\
         Phone: +49 (0) 62 32 / 14 23 - 0\n",
    );
    let answer = read_answer_text(
        &idx,
        "I'm planning my trip to Speyer again and I wanted to confirm, what's the phone number of the Speyer tourism board that you provided me earlier?",
    );
    assert!(answer.contains("+49 (0) 62 32 / 14 23 - 0"), "{answer}");
}

#[test]
fn recalls_other_list_options_excluding_prompted_term() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "alternatives.conv.md",
        "Assistant: 1. Sexual fixations - first option.\n\
         2. Problematic sexual behaviors - second option.\n\
         3. Sexual impulsivity - third option.\n\
         4. Compulsive sexuality - fourth option.\n",
    );
    let answer = read_answer_text(
        &idx,
        "In our previous chat, you suggested 'sexual compulsions' and a few other options for alternative terms for certain behaviors. Can you remind me what the other four options were?",
    );
    assert!(answer.contains("Sexual fixations"), "{answer}");
    assert!(answer.contains("Problematic sexual behaviors"), "{answer}");
    assert!(answer.contains("Sexual impulsivity"), "{answer}");
    assert!(answer.contains("Compulsive sexuality"), "{answer}");
}

#[test]
fn recalls_quote_from_assistant_prose() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "borges.conv.md",
        "Assistant: Borges notes, \"The Library is a sphere whose exact center is any one of its hexagons and whose circumference is inaccessible.\" (Borges, 1941)\n",
    );
    let answer = read_answer_text(
        &idx,
        "I was going through our previous conversation about The Library of Babel, and I wanted to confirm - what did Borges say about the center and circumference of the Library?",
    );
    assert!(
        answer.contains("exact center is any one of its hexagons"),
        "{answer}"
    );
}

#[test]
fn recalls_person_from_role_line() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "fusion.conv.md",
        "Assistant: Today's milestone shows what we can do with perseverance, said Dr. Arati Prabhakar, the President's Chief Advisor for Science and Technology and Director of the White House Office of Science and Technology Policy.\n",
    );
    let answer = read_answer_text(
        &idx,
        "I wanted to follow up on our previous conversation about the fusion breakthrough at Lawrence Livermore National Laboratory. Can you remind me who is the President's Chief Advisor for Science and Technology mentioned in the article?",
    );
    assert!(answer.contains("Arati Prabhakar"), "{answer}");
}

#[test]
fn recalls_budget_from_labeled_line() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "budget.conv.md",
        "Assistant: Budget:\n\
         * Influencer marketing: $2,000\n",
    );
    let answer = read_answer_text(
        &idx,
        "I'm looking back at our previous chat about the DHL Wellness Retreats campaign. Can you remind me how much was allocated for influencer marketing in the campaign plan?",
    );
    assert!(answer.contains("$2,000"), "{answer}");
}

#[test]
fn recalls_wearing_description_from_script() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "script.conv.md",
        "Assistant: Andy, a man in his 40s with a loud, deep voice and poor personal hygiene, is standing in front of the department's whiteboard. He wears an untidy, stained white shirt.\n",
    );
    let answer = read_answer_text(
        &idx,
        "I was going through our previous chat and I was wondering, what was Andy wearing in the script you wrote for the comedy movie scene?",
    );
    assert!(
        answer.contains("Andy was wearing an untidy, stained white shirt"),
        "{answer}"
    );
}

#[test]
fn recalls_implemented_algorithm_from_comparison_line() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "algorithm.conv.md",
        "Assistant: 1. 6S (Second Simulation of a Satellite Signal in the Solar Spectrum) algorithm is a radiative transfer model. It is implemented in the SIAC_GEE tool.\n",
    );
    let answer = read_answer_text(
        &idx,
        "I was going through our previous conversation about atmospheric correction methods, and I wanted to confirm - you mentioned that 6S, MAJA, and Sen2Cor are all algorithms for atmospheric correction of remote sensing images. Can you remind me which one is implemented in the SIAC_GEE tool?",
    );
    assert!(
        answer.contains("6S algorithm is implemented in the SIAC_GEE tool"),
        "{answer}"
    );
}

#[test]
fn recalls_objective_list_from_structured_section() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "objectives.conv.md",
        "Assistant: Objectives:\n\
         1. To identify molecular subtypes of endometrial cancer using a combination of genomic and transcriptomic approaches.\n\
         2. To investigate the clinical and biological significance of the identified molecular subtypes.\n\
         3. To develop biomarkers for the early detection and prognosis of endometrial cancer.\n",
    );
    let answer = read_answer_text(
        &idx,
        "I'm going back to our previous conversation about the grant aim page on molecular subtypes and endometrial cancer. Can you remind me what were the three objectives we outlined for the project?",
    );
    assert!(
        answer.contains("To identify molecular subtypes of endometrial cancer"),
        "{answer}"
    );
    assert!(
        answer.contains("To investigate the clinical and biological significance"),
        "{answer}"
    );
    assert!(answer.contains("To develop biomarkers"), "{answer}");
}

#[test]
fn recalls_ratio_value_from_assistant_line() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "ratio.conv.md",
        "Assistant: It should be diluted with a carrier oil such as coconut oil, jojoba oil or almond oil in a 1:10 ratio before application.\n",
    );
    let answer = read_answer_text(
        &idx,
        "I remember you told me to dilute tea tree oil with a carrier oil before applying it to my skin. Can you remind me what the recommended ratio is?",
    );
    assert!(answer.contains("1:10"), "{answer}");
}

#[test]
fn assistant_fact_query_does_not_trigger_on_bare_earlier_user_history() {
    let task =
        "How many largemouth bass did I catch with Alex on the earlier fishing trip to Lake Michigan before the 7/22 trip?";
    assert!(
        parse_assistant_fact_query(task, &task.to_ascii_lowercase()).is_none(),
        "bare earlier user-history questions should not route to assistant fact recall"
    );
}

#[test]
fn assistant_fact_recall_prefers_descriptor_rich_company_line() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    let task = "I wanted to follow up on our previous conversation about private sector businesses in Chaudhary. Can you remind me of the company that employs over 40,000 people in the rug-manufacturing industry?";
    let query = match parse_assistant_fact_query(task, &task.to_ascii_lowercase()).unwrap() {
        AssistantFactQuery::Entity(query) => query,
        other => panic!("unexpected query shape: {other:?}"),
    };
    let line = "1. Jaipur Rugs: Jaipur Rugs is a private company that employs over 40,000 people in the rug-manufacturing industry.";
    assert_eq!(
        extract_entity_candidate(&query, line, &line.to_ascii_lowercase()).as_deref(),
        Some("Jaipur Rugs")
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "chaudhary.conv.md",
        "Assistant: Yes, here are a few examples of private sector businesses that have been particularly impactful in reducing poverty and unemployment in Chaudhary:\n\
         1. Jaipur Rugs: Jaipur Rugs is a private company that employs over 40,000 people in the rug-manufacturing industry.\n\
         2. Dabur: Dabur is a consumer goods company.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "salary.conv.md",
        "Assistant: + Entry-level (0-2 years of experience): $40,000 - $60,000 per year\n",
    );
    let answer = read_answer_text(&idx, task);
    assert!(answer.contains("Jaipur Rugs"), "{answer}");
}

#[test]
fn assistant_fact_recall_prefers_matching_powwow_item_over_adjacent_list_context() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "powwow.conv.md",
        "Assistant: There are many traditional games that are played during Native American powwows. Some of the most popular include:\n\
         6. Double Ball - This is a fast-paced game that involves catching and throwing a small ball with two sticks. The goal is to score points by hitting a goal post with the ball.\n\
         7. Hoop Dance - This traditional dance involves intricate movements with multiple hoops, and is often performed by skilled dancers at powwows.\n",
    );
    let answer = read_answer_text(
        &idx,
        "I was looking back at our previous conversation about Native American powwows and I was wondering, which traditional game did you say was often performed by skilled dancers at powwows?",
    );
    assert!(answer.contains("Hoop Dance"), "{answer}");
    assert!(!answer.contains("Double Ball"), "{answer}");
}

#[test]
fn assistant_fact_recall_prefers_company_subject_after_structured_label() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    let task = "I was looking back at our previous conversation about environmentally responsible supply chain practices, and I was wondering if you could remind me of the company you mentioned that's doing a great job with sustainability?";
    let query = match parse_assistant_fact_query(task, &task.to_ascii_lowercase()).unwrap() {
        AssistantFactQuery::Entity(query) => query,
        other => panic!("unexpected query shape: {other:?}"),
    };
    let line = "1. Sustainable sourcing: Patagonia uses organic cotton, recycled polyester, and other sustainable materials in their products.";
    assert_eq!(
        extract_entity_candidate(&query, line, &line.to_ascii_lowercase()).as_deref(),
        Some("Patagonia")
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "supply.conv.md",
        "Assistant: Patagonia, an outdoor clothing and gear company, is known for its commitment to sustainability and environmental responsibility throughout its supply chain.\n\
         1. Sustainable sourcing: Patagonia uses organic cotton, recycled polyester, and other sustainable materials in their products.\n\
         User: It's great to see companies like Patagonia taking the initiative to be environmentally responsible.\n\
         Assistant: Hopefully, more companies will follow the lead of Patagonia and make a sincere commitment to sustainability in their supply chain practices.\n",
    );
    let answer = read_answer_text(&idx, task);
    assert!(answer.contains("Answer: Patagonia"), "{answer}");
    assert!(!answer.contains("Answer: Sustainable sourcing"), "{answer}");
}

#[test]
fn assistant_fact_recall_returns_instagram_handle_for_named_designer() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "rings.conv.md",
        "Assistant: 1. Jessica Poole (@jessica_poole_jewellery): Jessica is a UK-based jewelry designer who creates stunning, unique engagement rings using a combination of traditional and contemporary techniques. She has a passion for working with unusual gemstones and creates rings that are both modern and timeless.\n",
    );
    let answer = read_answer_text(
        &idx,
        "I was looking back at our previous conversation about buying unique engagement rings directly from designers. Can you remind me of the Instagram handle of the UK-based designer who works with unusual gemstones?",
    );
    assert!(answer.contains("@jessica_poole_jewellery"), "{answer}");
}

#[test]
fn assistant_fact_recall_prefers_campaign_budget_over_unrelated_marketing_amounts() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "dhl.conv.md",
        "Assistant: Sure, here's a detailed influencer marketing campaign plan for the DHL Wellness Retreats that starts on May 1st:\n\
         Budget:\n\
         * Influencer marketing: $2,000\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "ugc.conv.md",
        "Assistant: 1. **Content creation incentives**: Offer rewards or discounts to encourage users to create UGC (e.g., $10 - $100 per submission)\n",
    );
    let answer = read_answer_text(
        &idx,
        "I'm looking back at our previous chat about the DHL Wellness Retreats campaign. Can you remind me how much was allocated for influencer marketing in the campaign plan?",
    );
    assert!(answer.contains("$2,000"), "{answer}");
}

#[test]
fn assistant_fact_recall_prefers_named_hostel_over_later_activity_list() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "amsterdam.conv.md",
        "Assistant: 1. Stayokay Amsterdam Vondelpark: Located in the heart of Amsterdam.\n\
         2. International Budget Hostel: This hostel is situated near the famous Red Light District and offers affordable dormitory-style rooms.\n\
         Assistant: Sure, there are many fun things to do in Amsterdam.\n\
         1. Visit the Van Gogh Museum: This museum houses the largest collection of Vincent Van Gogh's artwork in the world. 2. Explore the Anne Frank House. 3. Visit the Red Light District.\n",
    );
    let answer = read_answer_text(
        &idx,
        "I'm planning my trip to Amsterdam again and I was wondering, what was the name of that hostel near the Red Light District that you recommended last time?",
    );
    assert!(answer.contains("International Budget Hostel"), "{answer}");
    assert!(!answer.contains("Visit the Van Gogh Museum"), "{answer}");
}

#[test]
fn assistant_fact_recall_requires_multiple_topic_terms_for_campaign_allocations() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "dhl.conv.md",
        "Assistant: Sure, here's a detailed influencer marketing campaign plan for the DHL Wellness Retreats that starts on May 1st:\n\
         1. Influencer Outreach\n\
         * Research and identify relevant wellness and lifestyle influencers with a significant following and engagement on Instagram and blogs.\n\
         * Reach out to these influencers with an offer to attend the retreats for free in exchange for social media posts and blog reviews.\n\
         1. Influencer Partnerships\n\
         * Develop long-term partnerships with select influencers to promote the DHL Wellness Retreats on an ongoing basis.\n\
         Budget:\n\
         * Influencer marketing: $2,000\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "product-launch.conv.md",
        "Assistant: Here's a detailed influencer marketing campaign plan for the new product launch.\n\
         Budget allocation:\n\
         * Influencer marketing campaign incentives: $10 - $100 per submission\n",
    );
    let answer = read_answer_text(
        &idx,
        "I'm looking back at our previous chat about the DHL Wellness Retreats campaign. Can you remind me how much was allocated for influencer marketing in the campaign plan?",
    );
    assert!(answer.contains("$2,000"), "{answer}");
}

#[test]
fn assistant_fact_recall_prefers_background_year_over_case_citation_year() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "bajimaya.conv.md",
        "User: Bajimaya v Reward Homes Pty Ltd - Case Summary\n\
         The case of Bajimaya v Reward Homes Pty Ltd [2021] NSWCATAP 297 highlights the importance of understanding your rights as a homeowner in the construction process.\n\
         Background:\n\
         The background of the case involves the construction of a new home in New South Wales, Australia, by the plaintiff, Mr. Bajimaya, and the defendant, Reward Homes Pty Ltd. The construction of the house began in 2014, and the contract was signed between the parties in 2015.\n\
         Assistant: Acknowledged. I understand the case summary of Bajimaya v Reward Homes Pty Ltd.\n",
    );
    let answer = read_answer_text(
        &idx,
        "I'm looking back at our previous conversation about the Bajimaya v Reward Homes Pty Ltd case. Can you remind me what year the construction of the house began?",
    );
    assert!(answer.contains("2014"), "{answer}");
}
