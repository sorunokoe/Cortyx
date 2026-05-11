use cortyx::answer_plane::render_answer_output;
use cortyx::index::NeuronIndex;
use std::fs;

fn render_answer(
    task: &str,
    evidence_files: &[(&str, &str, f32)],
    min_answer_confidence: Option<f32>,
) -> Option<String> {
    let dir = tempfile::tempdir().unwrap();
    let mut evidence = Vec::new();
    for (name, content, score) in evidence_files {
        let path = dir.path().join(name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&path, content).unwrap();
        evidence.push((path, *score));
    }
    render_answer_output(
        &NeuronIndex::default(),
        task,
        &evidence,
        false,
        min_answer_confidence,
    )
    .map(|answer| answer.trim().to_string())
}

#[test]
fn research_query_prefers_subject_matched_topical_turn_pair() {
    let conversation = "\
Speaker A: Caroline\n\
Speaker B: Melanie\n\
\n\
[Session 1 — 1:14 pm on 25 May, 2023]\n\
Melanie: Any fun plans for the summer?\n\
Caroline: Researching adoption agencies — it's been a dream to have a family and give a loving home to kids who need it.\n\
Melanie: Wow, Caroline! That's awesome! Taking in kids in need - you're so kind.\n\
\n\
[Session 2 — 10:31 am on 13 October, 2023]\n\
Melanie: Thanks, Caroline! Appreciate your help. Got any tips for getting started on it?\n\
Caroline: Yep! Do your research and find an adoption agency or lawyer.\n\
Melanie: Thanks for the tip, Caroline. Doing research and readying myself emotionally makes sense. I'll do that.\n";

    let answer = render_answer(
        "What did Caroline research?",
        &[("session.txt", conversation, 6.0)],
        None,
    )
    .unwrap();
    assert_eq!(answer, "adoption agencies");
}

#[test]
fn temporal_query_resolves_relative_session_dates() {
    let conversation = "\
Speaker A: Caroline\n\
Speaker B: Melanie\n\
\n\
[Session 1 — 1:56 pm on 8 May, 2023]\n\
Caroline: I went to a LGBTQ support group yesterday and it was so powerful.\n\
Melanie: Wow, that's amazing. Did you hear any inspiring stories?\n";

    let answer = render_answer(
        "When did Caroline go to the LGBTQ support group?",
        &[("session.txt", conversation, 6.0)],
        None,
    )
    .unwrap();
    assert_eq!(answer, "7 May 2023");
}

#[test]
fn education_query_prefers_extractable_field_answer_over_reasoning_distractor() {
    let conversation = "\
Speaker A: Caroline\n\
Speaker B: Melanie\n\
\n\
[Session 1 — 1:56 pm on 8 May, 2023]\n\
Melanie: That's really cool. You've got guts. What now?\n\
Caroline: Gonna continue my edu and check out career options, which is pretty exciting!\n\
Melanie: Wow, Caroline! What kinda jobs are you thinkin' of? Anything that stands out?\n\
Caroline: I'm keen on counseling or working in mental health - I'd love to support those with similar issues.\n\
\n\
[Session 2 — 10:37 am on 27 June, 2023]\n\
Melanie: What motivated you to pursue counseling?\n\
Caroline: Thanks, Melanie. My own journey and the support I got made a huge difference. Now I want to help people go through it too.\n";

    let answer = render_answer(
        "What fields would Caroline be likely to pursue in her educaton?",
        &[("session.txt", conversation, 6.0)],
        None,
    )
    .unwrap();
    let lower = answer.to_ascii_lowercase();
    assert!(lower.contains("counseling"));
    assert!(lower.contains("mental health"));
    assert!(!lower.contains("help people go through it too"));
}

#[test]
fn relation_query_abstains_on_conflicting_raw_dialogue_answers() {
    let answer = render_answer(
        "What kind of online group did John join?",
        &[
            (
                "group_one.txt",
                "Maria: What kind of online group did you join?\nJohn: I joined a service-focused online group last week and it has been inspiring.\n",
                7.0,
            ),
            (
                "group_two.txt",
                "Alex: What kind of online group did you join?\nJohn: I joined a neighborhood mentoring online group last month and it has been energizing.\n",
                6.9,
            ),
        ],
        None,
    );
    assert!(answer.is_none());
}
