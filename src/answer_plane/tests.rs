//! Core answer plane tests.

use super::*;
use crate::index::NeuronIndex;
use crate::kg;
use crate::neuron::{neuron_dir, NeuronKind, NeuronMeta, Synapse, SynapseType};
use std::fs;

fn reviewer_status_evidence() -> (tempfile::TempDir, Vec<EvidenceItem>) {
    let dir = tempfile::tempdir().unwrap();
    let path = kg::kg_neuron_path(dir.path(), "agent_reviewer");
    let mut entity = kg::KgEntity::load(&path).unwrap();
    assert!(entity.replace_active_fact("status", "in_progress", "2026-04-17T10:00:00Z"));
    assert!(entity.replace_active_fact("status", "blocked", "2026-04-17T10:03:00Z"));
    assert!(entity.replace_active_fact("status", "done", "2026-04-17T10:05:00Z"));
    entity.save().unwrap();
    (
        dir,
        vec![EvidenceItem {
            path,
            score: 7.0,
            metadata: None,
            snippet: String::new(),
        }],
    )
}

fn linked_kg_reasoning_index() -> (tempfile::TempDir, NeuronIndex, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let ndir = neuron_dir(dir.path());
    fs::create_dir_all(&ndir).unwrap();

    let seed_path = ndir.join("john_profile.context.md");
    fs::write(
        &seed_path,
        "# Profile\n\n## purpose\nLinks to John's structured profile.\n",
    )
    .unwrap();

    let kg_path = kg::kg_neuron_path(dir.path(), "john");
    let mut entity = kg::KgEntity::load(&kg_path).unwrap();
    assert!(entity.replace_active_fact("occupation", "product designer", "2026-04-17T10:05:00Z"));
    entity.save().unwrap();

    let mut meta = NeuronMeta::new_stub(&seed_path, NeuronKind::Core);
    let mut synapse = Synapse::new(
        kg_path,
        SynapseType::ConceptExpands,
        "expands to John's structured facts".to_string(),
    );
    synapse.weight = crate::types::SynapseWeight::new(1.0);
    meta.synapses.push(synapse);

    let mut idx = NeuronIndex::load_or_create(dir.path()).unwrap();
    let content = fs::read_to_string(&seed_path).unwrap();
    idx.index_neuron(&seed_path, &content, &meta);
    idx.rebuild_derived_pub();
    (dir, idx, seed_path)
}

#[test]
fn extract_derived_answer_prefers_answer_line() {
    let content = "# Derived answer\n\nQuestion: test\nAnswer: 42\n";
    assert_eq!(extract_derived_answer(content).as_deref(), Some("42"));
}

#[test]
fn extract_derived_answer_preserves_long_answers() {
    let content = "# Derived answer\n\nQuestion: test\nAnswer: The user would prefer responses that suggest resources specifically tailored to Adobe Premiere Pro, especially those that delve into its advanced settings. They might not prefer general video editing resources or resources related to other video editing software.\n";
    let answer = extract_derived_answer(content).expect("derived answer should parse");
    assert!(answer.contains("other video editing software."));
}

#[test]
fn explicit_abstention_detector_matches_standard_absent_messages() {
    assert!(derived_answer_is_explicit_abstention(
        "The information provided is not enough. You did not mention that you have a 30-gallon tank."
    ));
    assert!(!derived_answer_is_explicit_abstention("38 subjects"));
}

#[test]
fn render_answer_output_uses_derived_answer_verbatim() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("_answer_test.md");
    fs::write(
        &path,
        "# Derived answer\n\nQuestion: q\nAnswer: mental health\n",
    )
    .unwrap();
    let idx = NeuronIndex::default();
    let output = render_answer_output(
        &idx,
        "What did it raise awareness for?",
        &[(path, 8.0)],
        false,
        None,
    )
    .unwrap();
    assert_eq!(output.trim(), "mental health");
}

#[test]
fn render_answer_output_uses_answer_surface_span() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session.md");
    fs::write(
        &path,
        "# Session facts\n\n## answer_surface\n<!-- SECTION: answer_surface -->\n| question_pattern | answer_span | confidence |\n| --- | --- | --- |\n| job occupation profession work career role | pediatric nurse | 0.92 |\n| live location residence city home moved based | Portland | 0.88 |\n<!-- /SECTION -->\n",
    )
    .unwrap();
    let idx = NeuronIndex::default();
    let output =
        render_answer_output(&idx, "What is my job?", &[(path, 7.0)], false, None).unwrap();
    assert_eq!(output.trim(), "pediatric nurse");
}

#[test]
fn render_answer_output_abstains_for_pages_left_without_current_page_fact() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session.md");
    fs::write(
        &path,
        "User: I've been reading \"Sapiens\" at a pace of 10-20 pages a week.\nAssistant: Since you've been reading \"Sapiens\" at a pace of 10-20 pages a week, let's assume you read 15 pages per week.\n",
    )
    .unwrap();
    let idx = NeuronIndex::default();
    let output = render_answer_output(
        &idx,
        "How many pages do I have left to read in 'Sapiens'?",
        &[(path, 8.0)],
        false,
        None,
    );
    assert!(output.is_none());
}

#[test]
fn render_answer_output_falls_back_to_best_overlap_line() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session.txt");
    fs::write(
        &path,
        "Melanie: I painted a lake sunrise last year.\nCaroline: I want to capture unity and strength from the LGBTQ center visit.\n",
    )
    .unwrap();
    let idx = NeuronIndex::default();
    let output = render_answer_output(
        &idx,
        "What inspired Caroline's painting for the art show?",
        &[(path, 6.0)],
        false,
        None,
    )
    .unwrap();
    assert!(output.to_lowercase().contains("capture unity and strength"));
}

#[test]
fn render_answer_output_abstains_without_supported_answer() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session.txt");
    fs::write(
        &path,
        "We spent the afternoon at the park and talked about the weather.\n",
    )
    .unwrap();
    let idx = NeuronIndex::default();
    let output = render_answer_output(&idx, "What is my job?", &[(path, 7.0)], false, Some(0.3));
    assert!(output.is_none());
}

#[test]
fn render_answer_output_abstains_without_supported_answer_by_default() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session.txt");
    fs::write(
        &path,
        "We spent the afternoon at the park and talked about the weather.\n",
    )
    .unwrap();
    let idx = NeuronIndex::default();
    let output = render_answer_output(&idx, "What is my job?", &[(path, 7.0)], false, None);
    assert!(output.is_none());
}

#[test]
fn compact_answer_extracts_prepositional_object() {
    let task = "What did the charity race raise awareness for?";
    let line = "That charity race sounds great, Mel! Making a difference & raising awareness for mental health is super rewarding - I'm really proud of you for taking part!";
    let output = compact_answer(task, line, &salient_query_terms(task)).unwrap();
    assert_eq!(output, "mental health");
}

#[test]
fn compact_answer_extracts_started_object() {
    let task = "What is Jon working on opening?";
    let line = "Jon: I'm starting a dance studio 'cause I'm passionate about dancing and it'd be great to share it with others.";
    let output = compact_answer(task, line, &salient_query_terms(task)).unwrap();
    assert_eq!(output, "dance studio");
}

#[test]
fn extract_after_preposition_requires_standalone_task_token_and_stays_unicode_safe() {
    let task = "What music events has John attended?";
    let line = "I'm sure you're very good at this - unfortunately, I can’t share my love for him with you, my fingers are too big.";
    let result = std::panic::catch_unwind(|| {
        extract_after_preposition(
            task,
            line,
            &task.to_ascii_lowercase(),
            &salient_query_terms(task),
        )
    });
    assert!(result.is_ok(), "unicode punctuation should not panic");
    assert!(
        result.unwrap().is_none(),
        "embedded 'at' inside 'what' should not trigger"
    );
}

#[test]
fn comparison_which_query_is_not_treated_as_enumerative() {
    assert!(!is_enumerative_query(
        "Which event did I attend first, the workshop or the webinar?"
    ));
}

#[test]
fn select_comparison_answer_prefers_earlier_dated_option() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session.txt");
    fs::write(
        &path,
        "[Session 1 — 1:56 pm on 8 May, 2023]\nI participated in a webinar on \"Data Analysis using Python\" two months ago.\nI attended the workshop on \"Effective Time Management\" last Saturday.\n",
    )
    .unwrap();
    let evidence = vec![EvidenceItem {
        path,
        score: 6.0,
        metadata: None,
        snippet: String::new(),
    }];
    let answer = select_comparison_answer(
        "Which event did I attend first, the 'Effective Time Management' workshop or the 'Data Analysis using Python' webinar?",
        &evidence,
    )
    .unwrap();
    assert_eq!(answer, "'Data Analysis using Python' webinar");
}

#[test]
fn select_answer_abstains_on_temporal_location_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session.txt");
    fs::write(
        &path,
        "[Session 1 — 1:56 pm on 7 June, 2023]\nI booked an Airbnb in San Francisco for my best friend's wedding.\n",
    )
    .unwrap();
    let evidence = vec![EvidenceItem {
        path,
        score: 6.0,
        metadata: None,
        snippet: String::new(),
    }];
    let answer = select_answer("When did I book the Airbnb in Sacramento?", &evidence, None);
    assert!(answer.is_none());
}

#[test]
fn answer_form_gate_requires_institution_answer_for_university_query() {
    let task =
        "At which university did I present a poster for my undergrad course research project?";
    assert!(answer_meets_form_gate(task, "Harvard University", None));
    assert!(!answer_meets_form_gate(
        task,
        "the use of VR/AR to create",
        None
    ));
}

#[test]
fn select_answer_abstains_on_university_topic_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session.txt");
    fs::write(
        &path,
        "[Session 1 — 1:56 pm on 7 June, 2023]\nI just presented a poster on my thesis research at my first research conference over the summer.\nI've been to Harvard University to attend my first research conference and saw some interesting AI in education projects.\nResearchers are investigating the use of VR/AR to create interactive simulations, 3D models, and virtual field trips.\n\n## answer_surface\n<!-- SECTION: answer_surface -->\n| question_pattern | answer_span | confidence |\n| --- | --- | --- |\n| at which university present poster undergrad course research project education technology | the use of VR/AR to create | 0.93 |\n<!-- /SECTION -->\n",
    )
    .unwrap();
    let evidence = vec![EvidenceItem {
        path,
        score: 6.0,
        metadata: None,
        snippet: String::new(),
    }];
    let answer = select_answer(
        "At which university did I present a poster for my undergrad course research project?",
        &evidence,
        None,
    );
    assert!(answer.is_none());
}

#[test]
fn select_answer_abstains_on_missing_binary_choice_option() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session.txt");
    fs::write(
        &path,
        "[Session 1 — 1:56 pm on 8 May, 2023]\nI fixed that broken fence on the east side of my property three weeks ago.\n",
    )
    .unwrap();
    let evidence = vec![EvidenceItem {
        path,
        score: 6.0,
        metadata: None,
        snippet: String::new(),
    }];
    let answer = select_answer(
        "Which task did I complete first, fixing the fence or purchasing three cows from Peter?",
        &evidence,
        None,
    );
    assert!(answer.is_none());
}

#[test]
fn select_answer_uses_question_answer_turn_pair() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session.txt");
    fs::write(
        &path,
        "Melanie: Woah, Caroline, it sounds like you're doing some impressive work. What motivated you to pursue counseling?\nCaroline: I've been blessed with loads of love and support throughout this journey, and I want to pass it on to others.\n",
    )
    .unwrap();
    let evidence = vec![EvidenceItem {
        path,
        score: 6.0,
        metadata: None,
        snippet: String::new(),
    }];
    let answer = select_answer(
        "What motivated Caroline to pursue counseling?",
        &evidence,
        None,
    )
    .unwrap();
    assert!(answer.to_lowercase().contains("love and support"));
}

#[test]
fn select_answer_uses_question_answer_turn_pair_for_direct_question() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session.txt");
    fs::write(
        &path,
        "Maria: What kind of online group did you join?\nJohn: I joined a service-focused online group last week and it has been inspiring.\n",
    )
    .unwrap();
    let evidence = vec![EvidenceItem {
        path,
        score: 6.0,
        metadata: None,
        snippet: String::new(),
    }];
    let answer =
        select_answer("What kind of online group did John join?", &evidence, None).unwrap();
    assert!(answer
        .to_lowercase()
        .contains("service-focused online group"));
}

#[test]
fn select_answer_prefers_named_speaker_relation_turn() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session.txt");
    fs::write(
        &path,
        "Melanie: We all helped at the fundraiser.\nJohn: I worked with a local shelter to raise awareness and funds for victims of domestic abuse.\n",
    )
    .unwrap();
    let evidence = vec![EvidenceItem {
        path,
        score: 7.0,
        metadata: None,
        snippet: String::new(),
    }];
    let answer = select_answer(
        "Who did John work with to raise awareness and funds for victims of domestic abuse?",
        &evidence,
        None,
    )
    .unwrap();
    assert!(answer.to_lowercase().contains("local shelter"));
}

#[test]
fn select_answer_prefers_current_kg_relation_value_over_conflicting_surface() {
    let dir = tempfile::tempdir().unwrap();
    let stale = dir.path().join("stale_surface.md");
    fs::write(
        &stale,
        "# Session facts\n\n## answer_surface\n<!-- SECTION: answer_surface -->\n| question_pattern | answer_span | confidence |\n| --- | --- | --- |\n| john job occupation profession work career role | sales associate | 0.93 |\n<!-- /SECTION -->\n",
    )
    .unwrap();

    let kg_path = kg::kg_neuron_path(dir.path(), "john");
    let mut entity = kg::KgEntity::load(&kg_path).unwrap();
    assert!(entity.replace_active_fact("occupation", "sales associate", "2026-04-17T10:00:00Z"));
    assert!(entity.replace_active_fact("occupation", "product designer", "2026-04-17T10:05:00Z"));
    entity.save().unwrap();

    let evidence = vec![
        EvidenceItem {
            path: stale,
            score: 8.0,
            metadata: None,
            snippet: String::new(),
        },
        EvidenceItem {
            path: kg_path,
            score: 6.4,
            metadata: None,
            snippet: String::new(),
        },
    ];

    let answer = select_answer("What is John's job?", &evidence, None).unwrap();
    assert_eq!(answer, "product designer");
}

#[test]
fn select_answer_surface_abstains_on_conflicting_rows() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session.md");
    fs::write(
        &path,
        "# Session facts\n\n## answer_surface\n<!-- SECTION: answer_surface -->\n| question_pattern | answer_span | confidence |\n| --- | --- | --- |\n| john kind online group join | service-focused online group | 0.92 |\n| john kind online group join | neighborhood mentoring online group | 0.91 |\n<!-- /SECTION -->\n",
    )
    .unwrap();
    let evidence = vec![EvidenceItem {
        path,
        score: 7.0,
        metadata: None,
        snippet: String::new(),
    }];
    let answer = select_answer_surface("What kind of online group did John join?", &evidence);
    assert!(answer.is_none());
}

#[test]
fn select_answer_suppresses_conflicting_relation_candidates_without_kg() {
    let dir = tempfile::tempdir().unwrap();
    let first = dir.path().join("group_one.txt");
    let second = dir.path().join("group_two.txt");
    fs::write(
        &first,
        "Maria: What kind of online group did you join?\nJohn: I joined a service-focused online group last week and it has been inspiring.\n",
    )
    .unwrap();
    fs::write(
        &second,
        "Alex: What kind of online group did you join?\nJohn: I joined a neighborhood mentoring online group last month and it has been energizing.\n",
    )
    .unwrap();

    let evidence = vec![
        EvidenceItem {
            path: first,
            score: 7.0,
            metadata: None,
            snippet: String::new(),
        },
        EvidenceItem {
            path: second,
            score: 6.9,
            metadata: None,
            snippet: String::new(),
        },
    ];

    let answer = select_answer("What kind of online group did John join?", &evidence, None);
    assert!(answer.is_none());
}

#[test]
fn render_answer_output_uses_reasoned_kg_evidence_beyond_selected_paths() {
    let (_dir, idx, seed_path) = linked_kg_reasoning_index();
    let output = render_answer_output(
        &idx,
        "What is John's job?",
        &[(seed_path, 7.0)],
        false,
        None,
    )
    .unwrap();
    assert_eq!(output.trim(), "product designer");
}

#[test]
fn render_provenance_output_surfaces_graph_reasoning_summary() {
    let (_dir, idx, seed_path) = linked_kg_reasoning_index();
    let output = render_provenance_output(&idx, &[(seed_path, 7.0)]).unwrap();
    assert!(output.contains("<!-- CORTYX GRAPH REASONING -->"));
    assert!(output.contains("fact john.occupation = product designer"));
}

#[test]
fn render_answer_output_surfaces_unreadable_evidence() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("missing.txt");
    fs::write(&path, "temporary").unwrap();
    fs::remove_file(&path).unwrap();
    let idx = NeuronIndex::default();
    let output =
        render_answer_output(&idx, "What happened here?", &[(path, 6.0)], false, None).unwrap();
    assert!(output.contains("read error"));
}

#[test]
fn select_answer_prefers_structured_diary_outcome() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("diary.txt");
    fs::write(
        &path,
        crate::agent_memory::render_structured_diary_entry(
            "reviewer",
            "Investigated auth middleware coverage across the login path.",
            Some("Audit auth middleware"),
            Some("done"),
            None,
            None,
            None,
            Some("Found a legacy bypass in the old REST route."),
            &["auth".to_string(), "middleware".to_string()],
            &[],
        ),
    )
    .unwrap();
    let evidence = vec![EvidenceItem {
        path,
        score: 7.0,
        metadata: None,
        snippet: String::new(),
    }];
    let answer = select_answer(
        "What did the reviewer find about auth middleware?",
        &evidence,
        None,
    )
    .unwrap();
    assert!(answer.to_lowercase().contains("legacy bypass"));
}

#[test]
fn select_answer_prefers_structured_diary_next_step_and_blocker() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("diary.txt");
    fs::write(
        &path,
        crate::agent_memory::render_structured_diary_entry(
            "reviewer",
            "Audited the legacy auth route.",
            Some("Close auth bypass"),
            Some("blocked"),
            Some("Close the auth bypass without regressing login."),
            Some("Patch the legacy REST route after ownership is confirmed."),
            Some("Waiting on route ownership clarification."),
            Some("Confirmed the bypass only exists on the legacy REST path."),
            &["auth".to_string(), "routing".to_string()],
            &["router-owner".to_string()],
        ),
    )
    .unwrap();
    let evidence = vec![EvidenceItem {
        path,
        score: 7.0,
        metadata: None,
        snippet: String::new(),
    }];
    let blocker = select_answer("What is the reviewer blocked on?", &evidence, None).unwrap();
    assert!(blocker.to_lowercase().contains("ownership clarification"));
    let next = select_answer("What is the reviewer's next step?", &evidence, None).unwrap();
    assert!(next.to_lowercase().contains("patch the legacy rest route"));
}

#[test]
fn select_answer_composes_structured_diary_multihop_fields() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("diary.txt");
    fs::write(
        &path,
        crate::agent_memory::render_structured_diary_entry(
            "reviewer",
            "Audited the legacy auth route.",
            Some("Close auth bypass"),
            Some("blocked"),
            Some("Close the auth bypass without regressing login."),
            Some("Patch the legacy REST route after ownership is confirmed."),
            Some("Waiting on route ownership clarification."),
            Some("Confirmed the bypass only exists on the legacy REST path."),
            &["auth".to_string(), "routing".to_string()],
            &["router-owner".to_string()],
        ),
    )
    .unwrap();
    let evidence = vec![EvidenceItem {
        path,
        score: 7.0,
        metadata: None,
        snippet: String::new(),
    }];
    let answer = select_answer(
        "What is the reviewer's blocker and next step?",
        &evidence,
        None,
    )
    .unwrap();
    let lower = answer.to_lowercase();
    assert!(lower.contains("blocker:"));
    assert!(lower.contains("ownership clarification"));
    assert!(lower.contains("next step:"));
    assert!(lower.contains("patch the legacy rest route"));
}

#[test]
fn select_answer_composes_list_answers_across_evidence_items() {
    let dir = tempfile::tempdir().unwrap();
    let first = dir.path().join("group_one.txt");
    let second = dir.path().join("group_two.txt");
    fs::write(
        &first,
        "Maria: What kind of online group did you join?\nJohn: I joined a service-focused online group last week.\n",
    )
    .unwrap();
    fs::write(
        &second,
        "Alex: Have you joined anything else recently?\nJohn: I joined a volunteer study group.\n",
    )
    .unwrap();
    let evidence = vec![
        EvidenceItem {
            path: first,
            score: 7.0,
            metadata: None,
            snippet: String::new(),
        },
        EvidenceItem {
            path: second,
            score: 6.8,
            metadata: None,
            snippet: String::new(),
        },
    ];
    let answer = select_answer("What groups did John join?", &evidence, None).unwrap();
    let lower = answer.to_lowercase();
    assert!(lower.contains("service-focused online group"));
    assert!(lower.contains("volunteer study group"));
}

#[test]
fn select_answer_solves_temporal_duration_question() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session.txt");
    fs::write(
        &path,
        "[Session 1 — 1:56 pm on 17 January, 2023]\nI attended the workshop on \"Effective Communication in the Workplace\" on January 10th.\nI had the team meeting I was preparing for on January 17th.\n",
    )
    .unwrap();
    let evidence = vec![EvidenceItem {
        path,
        score: 7.0,
        metadata: None,
        snippet: String::new(),
    }];
    let answer = select_answer(
        "How many days before the team meeting I was preparing for did I attend the workshop on 'Effective Communication in the Workplace'?",
        &evidence,
        None,
    )
    .unwrap();
    assert_eq!(answer, "7 days");
}

#[test]
fn select_answer_prefers_current_kg_temporal_state() {
    let (_dir, evidence) = reviewer_status_evidence();
    let answer = select_answer("What is the reviewer's current status?", &evidence, None).unwrap();
    assert_eq!(answer, "done");
}

#[test]
fn select_answer_resolves_kg_temporal_state_as_of_timestamp() {
    let (_dir, evidence) = reviewer_status_evidence();
    let answer = select_answer(
        "What was the reviewer's status as of 2026-04-17T10:04:00Z?",
        &evidence,
        None,
    )
    .unwrap();
    assert_eq!(answer, "blocked");
}

#[test]
fn select_answer_resolves_when_kg_state_changed_to_target_value() {
    let (_dir, evidence) = reviewer_status_evidence();
    let answer = select_answer(
        "When did the reviewer's status change to blocked?",
        &evidence,
        None,
    )
    .unwrap();
    assert_eq!(answer, "2026-04-17T10:03:00Z");
}

#[test]
fn select_answer_resolves_when_kg_state_last_changed() {
    let (_dir, evidence) = reviewer_status_evidence();
    let answer = select_answer(
        "When did the reviewer's status last change?",
        &evidence,
        None,
    )
    .unwrap();
    assert_eq!(answer, "2026-04-17T10:05:00Z");
}
