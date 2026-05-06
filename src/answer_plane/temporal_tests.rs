use super::*;
use crate::index::NeuronIndex;
use crate::miner::mine_file;
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

fn write_temporal_evidence(content: &str) -> (tempfile::TempDir, Vec<EvidenceItem>) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session.txt");
    fs::write(&path, content).unwrap();
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

fn write_temporal_evidence_files(
    files: &[(&str, &str, f32)],
) -> (tempfile::TempDir, Vec<EvidenceItem>) {
    let dir = tempfile::tempdir().unwrap();
    let mut evidence = Vec::new();
    for (name, content, score) in files {
        let path = dir.path().join(name);
        fs::write(&path, content).unwrap();
        evidence.push(EvidenceItem {
            path,
            score: *score,
            metadata: None,
            snippet: String::new(),
        });
    }
    (dir, evidence)
}

fn fixture_entry(question_id: &str) -> (String, String, String) {
    let fixture_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/longmemeval_500.json");
    let entries: Vec<Value> =
        serde_json::from_str(&fs::read_to_string(fixture_path).unwrap()).unwrap();
    let entry = entries
        .into_iter()
        .find(|entry| entry.get("question_id").and_then(Value::as_str) == Some(question_id))
        .unwrap_or_else(|| panic!("fixture entry not found: {question_id}"));
    (
        entry["neuron_filename"].as_str().unwrap().to_string(),
        entry["neuron_source_content"].as_str().unwrap().to_string(),
        entry["question"].as_str().unwrap().to_string(),
    )
}

fn select_answer_for_fixture(question_id: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let (file_name, content, question) = fixture_entry(question_id);
    let path = dir.path().join(file_name);
    fs::write(&path, content).unwrap();
    select_answer(
        &question,
        &[EvidenceItem {
            path,
            score: 11.5,
            metadata: None,
            snippet: String::new(),
        }],
        None,
    )
    .unwrap()
}

fn mine_and_collect_fixture(
    question_id: &str,
) -> (
    tempfile::TempDir,
    NeuronIndex,
    String,
    Vec<(PathBuf, f32)>,
    Vec<EvidenceItem>,
) {
    let dir = tempfile::tempdir().unwrap();
    let (file_name, content, question) = fixture_entry(question_id);
    let path = dir.path().join(&file_name);
    fs::write(&path, content).unwrap();

    let mut idx = NeuronIndex::load_or_create(dir.path()).unwrap();
    mine_file(&path, dir.path(), &mut idx, None).unwrap();
    let idx = NeuronIndex::load_or_create(dir.path()).unwrap();
    let (included, _) = idx.get_contexts_with_scores_and_overflow(
        &question,
        3200,
        None,
        Some("conversation"),
        None,
        false,
    );
    let (evidence, _) = collect_evidence_with_reasoning(&idx, &question, &included);
    (dir, idx, question, included, evidence)
}

#[test]
fn select_answer_prefers_user_temporal_candidates_for_device_order() {
    let (_dir, evidence) = write_temporal_evidence(
        "User: I'm having some issues with my desktop computer, and I've been trying to reduce my energy bills and set up a new smart thermostat, which has been helpful.\n\
Assistant: That's great to hear about your smart thermostat.\n\
User: Also, since I set up my smart thermostat a month ago, I've noticed that it's been learning my schedule and preferences.\n\
Assistant: A new mesh network system can really help with connectivity.\n\
User: Since I've recently upgraded my home Wi-Fi router 3 weeks ago to a mesh network system, I'm thinking I should prioritize a desktop computer that can take full advantage of it.\n",
    );
    let answer = select_answer(
        "Which device did I set up first, the smart thermostat or the mesh network system?",
        &evidence,
        None,
    )
    .unwrap();
    assert_eq!(answer, "smart thermostat");
}

#[test]
fn select_answer_counts_temporal_events_before_anchor() {
    let (_dir, evidence) = write_temporal_evidence(
        "User: I volunteered at the Food Pantry 5K on June 2nd and had a great time.\n\
Assistant: That sounds like a meaningful event.\n\
User: I joined the Coastal Cleanup event on July 14th and loved helping out.\n\
Assistant: Coastal cleanup days make a real difference.\n\
User: I participated in the School Supply Drive walk on August 20th and raised money for local students.\n\
Assistant: That's a wonderful cause.\n\
User: I helped with the Harvest Relief run on September 7th and met so many great people.\n\
Assistant: Charity runs can be incredibly energizing.\n\
User: I just ran 5 kilometers in the \"Run for the Cure\" event on October 15th and raised money for breast cancer research.\n\
Assistant: It's fantastic that you participated in the Run for the Cure event.\n\
User: I'm thinking of registering for a cycling event next month.\n",
    );
    let answer = select_answer(
        "How many charity events did I participate in before the 'Run for the Cure' event?",
        &evidence,
        None,
    )
    .unwrap();
    assert_eq!(answer, "4");
}

#[test]
fn select_answer_counts_mixed_charity_event_phrasings_before_anchor() {
    let (_dir, evidence) = write_temporal_evidence(
        "User: I just ran 5 kilometers in the \"Run for the Cure\" event on October 15th and raised money for breast cancer research.\n\
Assistant: That sounds like a powerful experience.\n\
User: I just participated in the \"Dance for a Cause\" event on May 1st, which was a blast.\n\
Assistant: Dance marathons can be so energizing.\n\
User: I volunteered at the Walk for Wildlife event in June, where we raised awareness and funds for conservation efforts.\n\
Assistant: Wildlife charity events can make a real impact.\n\
User: I attended a charity golf tournament on July 17th and had a great time playing with colleagues.\n\
Assistant: Charity tournaments are a fun way to give back.\n\
User: I volunteered at the \"Food for Thought\" charity gala on September 25th, which was a great experience.\n",
    );
    let answer = select_answer(
        "How many charity events did I participate in before the 'Run for the Cure' event?",
        &evidence,
        None,
    )
    .unwrap();
    assert_eq!(answer, "4");
}

#[test]
fn select_answer_resolves_elapsed_months_from_booking_in_advance() {
    let (_dir, evidence) = write_temporal_evidence(
        "User: I'm planning a trip to San Francisco for next month, and I've been to SF before, exactly two months ago, for my best friend's wedding.\n\
Assistant: San Francisco is always a fun trip.\n\
User: I'm planning another trip to San Francisco and was wondering where to stay. I've had a great experience with Airbnb in the past, like when I stayed in Haight-Ashbury for my best friend's wedding and had to book three months in advance.\n\
Assistant: Haight-Ashbury can be a great area to stay in.\n",
    );
    let answer = select_answer(
        "How many months ago did I book the Airbnb in San Francisco?",
        &evidence,
        None,
    )
    .unwrap();
    assert_eq!(answer, "five months ago");
}

#[test]
fn select_answer_subtracts_current_job_tenure_from_total_experience() {
    let (_dir, evidence) = write_temporal_evidence(
        "User: I've been working professionally for 9 years and I'm looking for better ways to stay organized.\n\
Assistant: Digital tools can definitely help with that.\n\
User: I'm a software engineer, and I've been working at NovaTech for about 4 years and 3 months now.\n\
Assistant: That kind of tenure gives you a lot of context about NovaTech's stack.\n",
    );
    let answer = select_answer(
        "How long have I been working before I started my current job at NovaTech?",
        &evidence,
        None,
    )
    .unwrap();
    assert_eq!(answer, "4 years and 9 months");
}

#[test]
fn mined_temporal_public_fixture_charity_count_survives_mine_and_render() {
    let (_dir, idx, question, included, evidence) = mine_and_collect_fixture("a3838d2b");
    let selected = select_answer(&question, &evidence, None);
    let rendered = render_answer_output_decision(&idx, &question, &included, false, None);
    let included_names = included
        .iter()
        .map(|(path, _)| path.file_name().unwrap().to_string_lossy().to_string())
        .collect::<Vec<_>>();
    let evidence_names = evidence
        .iter()
        .map(|item| item.path.file_name().unwrap().to_string_lossy().to_string())
        .collect::<Vec<_>>();

    assert_eq!(
        selected.as_deref(),
        Some("4"),
        "selected={selected:?}; included={included_names:?}; evidence={evidence_names:?}"
    );
    assert!(
        matches!(rendered, Ok(ref answer) if answer.trim() == "4"),
        "rendered={rendered:?}; included={included_names:?}; evidence={evidence_names:?}"
    );
}

#[test]
fn mined_temporal_public_fixture_employment_gap_survives_mine_and_render() {
    let (_dir, idx, question, included, evidence) = mine_and_collect_fixture("gpt4_93159ced");
    let selected = select_answer(&question, &evidence, None);
    let rendered = render_answer_output_decision(&idx, &question, &included, false, None);
    let included_names = included
        .iter()
        .map(|(path, _)| path.file_name().unwrap().to_string_lossy().to_string())
        .collect::<Vec<_>>();
    let evidence_names = evidence
        .iter()
        .map(|item| item.path.file_name().unwrap().to_string_lossy().to_string())
        .collect::<Vec<_>>();

    assert_eq!(
        selected.as_deref(),
        Some("4 years and 9 months"),
        "selected={selected:?}; included={included_names:?}; evidence={evidence_names:?}"
    );
    assert!(
        matches!(rendered, Ok(ref answer) if answer.trim() == "4 years and 9 months"),
        "rendered={rendered:?}; included={included_names:?}; evidence={evidence_names:?}"
    );
}

#[test]
fn mined_temporal_public_fixture_binary_choice_survives_mine_and_render() {
    let (_dir, idx, question, included, evidence) = mine_and_collect_fixture("gpt4_2487a7cb");
    let selected = select_answer(&question, &evidence, None);
    let precomputed = idx.derived_answer_path_for_task(&question);
    let should_defer = precomputed
        .as_ref()
        .map(|path| should_defer_precomputed_answer(&question, path))
        .unwrap_or(false);
    let base_candidate = select_answer_internal(&question, &evidence, None, true);
    let base_answer = validate_selected_answer(&question, base_candidate.clone(), None);
    let rendered = render_answer_output_decision(&idx, &question, &included, false, None);
    let included_names = included
        .iter()
        .map(|(path, _)| path.file_name().unwrap().to_string_lossy().to_string())
        .collect::<Vec<_>>();
    let evidence_names = evidence
        .iter()
        .map(|item| item.path.file_name().unwrap().to_string_lossy().to_string())
        .collect::<Vec<_>>();

    assert_eq!(
        selected.as_deref(),
        Some("'Data Analysis using Python' webinar"),
        "selected={selected:?}; precomputed={precomputed:?}; should_defer={should_defer}; base_candidate={base_candidate:?}; base_answer={base_answer:?}; included={included_names:?}; evidence={evidence_names:?}"
    );
    assert!(
        matches!(
            rendered,
            Ok(ref answer) if answer.trim() == "'Data Analysis using Python' webinar"
        ),
        "rendered={rendered:?}; precomputed={precomputed:?}; should_defer={should_defer}; base_candidate={base_candidate:?}; base_answer={base_answer:?}; included={included_names:?}; evidence={evidence_names:?}"
    );
}

#[test]
fn mined_temporal_public_fixture_booking_lead_time_survives_mine_and_render() {
    let (_dir, idx, question, included, evidence) = mine_and_collect_fixture("982b5123");
    let selected = select_answer(&question, &evidence, None);
    let precomputed = idx.derived_answer_path_for_task(&question);
    let should_defer = precomputed
        .as_ref()
        .map(|path| should_defer_precomputed_answer(&question, path))
        .unwrap_or(false);
    let base_candidate = select_answer_internal(&question, &evidence, None, true);
    let base_answer = validate_selected_answer(&question, base_candidate.clone(), None);
    let rendered = render_answer_output_decision(&idx, &question, &included, false, None);
    let included_names = included
        .iter()
        .map(|(path, _)| path.file_name().unwrap().to_string_lossy().to_string())
        .collect::<Vec<_>>();
    let evidence_names = evidence
        .iter()
        .map(|item| item.path.file_name().unwrap().to_string_lossy().to_string())
        .collect::<Vec<_>>();

    assert_eq!(
        selected.as_deref(),
        Some("five months ago"),
        "selected={selected:?}; precomputed={precomputed:?}; should_defer={should_defer}; base_candidate={base_candidate:?}; base_answer={base_answer:?}; included={included_names:?}; evidence={evidence_names:?}"
    );
    assert!(
        matches!(rendered, Ok(ref answer) if answer.trim() == "five months ago"),
        "rendered={rendered:?}; precomputed={precomputed:?}; should_defer={should_defer}; base_candidate={base_candidate:?}; base_answer={base_answer:?}; included={included_names:?}; evidence={evidence_names:?}"
    );
}

#[test]
fn select_answer_reads_total_experience_from_summary_style_evidence() {
    let (_dir, evidence) = write_temporal_evidence_files(&[
        (
            "chunk.verbatim.md",
            "user: I'm a software engineer, specifically a backend developer, and I've been in this field since I graduated with a degree in Computer Science from the University of California, Berkeley. I've been working at NovaTech for about 4 years and 3 months now.\n\
assistant: A Cal Bear, nice! As a backend developer with 4+ years of experience, you've got a solid foundation in software development.\n\
## answer_surface\n\
<!-- SECTION: answer_surface -->\n\
| question_pattern | answer_span | confidence |\n\
| --- | --- | --- |\n\
| job occupation profession work career role | software engineer specifically a | 0.92 |\n\
<!-- /SECTION -->\n",
            11.5,
        ),
        (
            "summary.md",
            "# Session facts\n\
## facts\n\
- I've been working professionally for 9 years and I'm currently using a physical notebook to jot down notes and reminders, but I'm not sure if that's the most efficient way\n\
- I've been working at NovaTech for about 4 years and 3 months now\n\
- As a backend developer with 4+ years of experience, you'll appreciate the power of both New Relic and Datadog\n",
            11.7,
        ),
    ]);
    let direct = select_temporal_employment_duration_answer(
        "How long have I been working before I started my current job at NovaTech?",
        &evidence,
    );
    assert_eq!(direct.as_deref(), Some("4 years and 9 months"));
    let answer = select_answer(
        "How long have I been working before I started my current job at NovaTech?",
        &evidence,
        None,
    )
    .unwrap();
    assert_eq!(answer, "4 years and 9 months");
}

#[test]
fn select_answer_uses_temporal_window_for_last_month_item_query() {
    let (_dir, evidence) = write_temporal_evidence(
        "User: I'm glad I finally got around to cleaning my white Adidas sneakers last month, which I'd been meaning to do for weeks.\n\
Assistant: It's great that you finally got around to cleaning your white Adidas sneakers.\n\
User: I've been wearing my new Vans Old Skool sneakers almost every day since I got them last weekend.\n",
    );
    let answer = select_answer(
        "Which pair of shoes did I clean last month?",
        &evidence,
        None,
    )
    .unwrap();
    let lower = answer.to_ascii_lowercase();
    assert!(lower.contains("white"));
    assert!(lower.contains("adidas"));
    assert!(lower.contains("sneakers"));
}

#[test]
fn select_answer_orders_three_explicit_temporal_events_by_sequence() {
    let (_dir, evidence) = write_temporal_evidence_files(&[
        (
            "case_0000_chunk.verbatim.md",
            "I just helped my friend prepare the nursery today.",
            8.0,
        ),
        (
            "case_0001_chunk.verbatim.md",
            "I just helped my cousin pick out some stuff for her baby shower today.",
            7.8,
        ),
        (
            "case_0002_chunk.verbatim.md",
            "I just ordered a customized phone case for my friend's birthday today.",
            7.6,
        ),
    ]);
    let answer = select_answer(
        "Which three events happened in the order from first to last: the day I helped my friend prepare the nursery, the day I helped my cousin pick out stuff for her baby shower, and the day I ordered a customized phone case for my friend's birthday?",
        &evidence,
        None,
    )
    .unwrap();
    let lower = answer.to_ascii_lowercase();
    let nursery = lower.find("prepare the nursery").unwrap();
    let shower = lower.find("baby shower").unwrap();
    let phone_case = lower.find("phone case").unwrap();
    assert!(
        nursery < shower && shower < phone_case,
        "unexpected answer: {answer}"
    );
}

#[test]
fn select_answer_resolves_self_anchored_temporal_gap_from_named_holiday() {
    let (_dir, evidence) = write_temporal_evidence(
        "User: By the way, I attended the annual Holiday Market at the local mall a week before Black Friday and found some unique handmade jewelry.\n\
Assistant: That sounds like a fun trip to the Holiday Market.\n\
User: I got my iPhone 13 Pro at a discounted price of $800 from Best Buy on Black Friday, which was a great deal.\n",
    );
    let answer = select_answer(
        "How many days before I bought the iPhone 13 Pro did I attend the Holiday Market?",
        &evidence,
        None,
    )
    .unwrap();
    assert!(
        answer.contains("7 days") || answer.contains("8 days"),
        "unexpected answer: {answer}"
    );
}

#[test]
fn select_answer_rejects_temporal_gap_without_calendar_grounding() {
    let (_dir, evidence) = write_temporal_evidence_files(&[(
        "summary.md",
        "User: I bought my Adidas running shoes recently and want to take good care of them.\n\
User: I realized one of the shoelaces on my old Converse sneakers had broken, so I had to replace it.\n\
## answer_surface\n\
<!-- SECTION: answer_surface -->\n\
| question_pattern | answer_span | confidence |\n\
| --- | --- | --- |\n\
| shoe care leather conditioning | Leather Conditioning: 1 | 0.96 |\n\
<!-- /SECTION -->\n",
        11.0,
    )]);
    let answer = select_answer(
        "How many days had passed since I bought my Adidas running shoes when I realized one of the shoelaces on my old Converse sneakers had broken?",
        &evidence,
        None,
    );
    assert!(answer.is_none(), "unexpected answer: {answer:?}");
}

#[test]
fn select_answer_solves_temporal_fixture_2a1811e2() {
    let answer = select_answer_for_fixture("2a1811e2");
    assert!(
        answer.contains("21 days") || answer.contains("22 days"),
        "unexpected answer: {answer}"
    );
}

#[test]
fn select_answer_solves_temporal_fixture_c8090214() {
    let answer = select_answer_for_fixture("c8090214");
    assert!(
        answer.contains("7 days") || answer.contains("8 days"),
        "unexpected answer: {answer}"
    );
}

#[test]
fn select_answer_solves_temporal_fixture_0bb5a684() {
    let answer = select_answer_for_fixture("0bb5a684");
    assert!(
        answer.contains("7 days") || answer.contains("8 days"),
        "unexpected answer: {answer}"
    );
}

#[test]
fn select_answer_solves_temporal_fixture_2c63a862() {
    let answer = select_answer_for_fixture("2c63a862");
    assert!(
        answer.contains("14 days") || answer.contains("15 days"),
        "unexpected answer: {answer}"
    );
}

#[test]
fn select_answer_solves_temporal_fixture_dcfa8644() {
    let answer = select_answer_for_fixture("dcfa8644");
    assert!(
        answer.contains("14 days") || answer.contains("15 days"),
        "unexpected answer: {answer}"
    );
}
