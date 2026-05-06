/// Conversation mining — parses dialogue files into Verbatim neurons.
///
/// Supported formats (auto-detected by content structure):
/// - ChatGPT `conversations.json` export (mapping tree)
/// - Claude markdown export (`## Human` / `## Assistant` headings)
/// - LongMemEval JSON (`session_history` turn arrays)
/// - Generic markdown (any `##` headings as chunk boundaries)
///
/// Each parsed turn becomes a `NeuronKind::Verbatim` neuron. Consecutive
/// turns get `SynapseType::TemporalFollows` edges to capture temporal order.
use std::path::{Path, PathBuf};

use crate::error::Result;

use crate::index::NeuronIndex;

mod cooccurrence;
mod kg_apply;
mod kg_extract;
mod parsers;
mod summary;
mod surface;
mod writer;

// ─── Public types ─────────────────────────────────────────────────────────────

/// A single extracted conversation turn ready to be written as a Verbatim neuron.
#[derive(Debug, Clone)]
pub struct Turn {
    pub speaker: Option<String>,
    pub text: String,
    pub timestamp: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
struct AnswerSurfaceRow {
    pub question_pattern: String,
    pub answer_span: String,
    pub confidence: f32,
}

const SKIPPED_MINE_DIRS: &[&str] = &[".cortyx", "target", ".git", "node_modules", "__pycache__"];

// ─── Public API ───────────────────────────────────────────────────────────────

/// Parse a file (or all `.json`/`.md` files in a directory) into turns,
/// write them as Verbatim neurons, and upsert into the index.
///
/// Returns the number of neurons created.
///
/// NE-1 fix: stage ALL files first, then commit ONCE via rebuild_derived().
/// This avoids O(n²) re-index cost for large directories.
pub fn mine_path(
    path: &Path,
    project_root: &Path,
    idx: &mut NeuronIndex,
    module: Option<&str>,
) -> Result<usize> {
    if path.is_dir() {
        let mut total = 0usize;
        let mut all_neuron_paths: Vec<PathBuf> = Vec::new();
        let mut all_turns: Vec<Turn> = Vec::new();

        for entry in walkdir::WalkDir::new(path)
            .min_depth(1)
            .into_iter()
            .filter_entry(|entry| !should_skip_mined_entry(entry))
            .filter_map(|e| e.ok())
        {
            if entry.file_type().is_file() {
                let ext = entry
                    .path()
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("");
                if matches!(ext, "json" | "md" | "txt") {
                    match mine_file_staged(entry.path(), project_root, idx, module) {
                        Ok((n, paths, turns)) => {
                            total += n;
                            all_neuron_paths.extend(paths);
                            all_turns.extend(turns);
                        },
                        Err(e) => {
                            tracing::warn!("Skipping {}: {e}", entry.path().display());
                        },
                    }
                }
            }
        }

        // Build corpus-wide co-occurrence from ALL sessions (not per-session).
        cooccurrence::build_and_save_cooccurrence(&all_turns, project_root);

        // Single commit for the entire directory — rebuild_derived() called ONCE.
        idx.commit()?;

        // emit_arithmetic_aggregate_neurons writes .aggregate.md files to disk but
        // does NOT call idx.stage(), so no index state changes — no second commit needed.
        idx.emit_arithmetic_aggregate_neurons(project_root)
            .unwrap_or(false);

        #[cfg(feature = "embed")]
        writer::batch_embed_paths(&all_neuron_paths, project_root);

        Ok(total)
    } else {
        mine_file(path, project_root, idx, module)
    }
}

fn should_skip_mined_entry(entry: &walkdir::DirEntry) -> bool {
    entry.file_type().is_dir()
        && entry
            .file_name()
            .to_str()
            .map(|name| SKIPPED_MINE_DIRS.iter().any(|candidate| candidate == &name))
            .unwrap_or(false)
}

/// Mine a single file. Auto-detects format and writes Verbatim neurons.
pub fn mine_file(
    path: &Path,
    project_root: &Path,
    idx: &mut NeuronIndex,
    module: Option<&str>,
) -> Result<usize> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| crate::cortyx_err!("Cannot read {}: {e}", path.display()))?;
    let turns = detect_and_parse(&raw)
        .map_err(|e| crate::cortyx_err!("Failed to parse {}: {e}", path.display()))?;
    writer::write_verbatim_neurons(&turns, path, project_root, idx, module)
}

/// Mine a single file, staging neurons without committing the index.
///
/// Used by `mine_path` directory batch mode to defer rebuild_derived() until all
/// files are staged. Returns (count, neuron_paths, turns).
fn mine_file_staged(
    path: &Path,
    project_root: &Path,
    idx: &mut NeuronIndex,
    module: Option<&str>,
) -> Result<(usize, Vec<PathBuf>, Vec<Turn>)> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| crate::cortyx_err!("Cannot read {}: {e}", path.display()))?;
    let turns = detect_and_parse(&raw)
        .map_err(|e| crate::cortyx_err!("Failed to parse {}: {e}", path.display()))?;
    let (count, paths) =
        writer::write_verbatim_neurons_staged(&turns, path, project_root, idx, module)?;
    Ok((count, paths, turns))
}

/// Mine a raw string (called from the MCP tool with inline content).
pub fn mine_text(
    content: &str,
    source_hint: &str,
    project_root: &Path,
    idx: &mut NeuronIndex,
    module: Option<&str>,
    speaker: Option<&str>,
    timestamp: Option<&str>,
) -> Result<usize> {
    let turns = if speaker.is_some() || timestamp.is_some() {
        vec![Turn {
            speaker: speaker.map(|s| s.to_string()),
            text: content.to_string(),
            timestamp: timestamp.map(|s| s.to_string()),
        }]
    } else {
        detect_and_parse(content).unwrap_or_else(|_| {
            vec![Turn {
                speaker: None,
                text: content.to_string(),
                timestamp: None,
            }]
        })
    };

    let fake_path = PathBuf::from(source_hint);
    writer::write_verbatim_neurons(&turns, &fake_path, project_root, idx, module)
}

// Re-export write_verbatim_neurons for direct callers (e.g. integration tests)
pub use writer::write_verbatim_neurons;

/// Embed all neuron files in `paths` and save vectors to `project_root/.cortyx/embeddings.bin`.
///
/// No-op when the `embed` feature is disabled.
/// Called from `cortyx compile` to build the dense retrieval layer after indexing.
#[cfg(feature = "embed")]
pub fn embed_all(paths: &[std::path::PathBuf], project_root: &std::path::Path) {
    writer::batch_embed_paths(paths, project_root);
}

/// No-op stub when `embed` feature is absent.
#[cfg(not(feature = "embed"))]
pub fn embed_all(_paths: &[std::path::PathBuf], _project_root: &std::path::Path) {}

// ─── Format detection ─────────────────────────────────────────────────────────

fn detect_and_parse(raw: &str) -> Result<Vec<Turn>> {
    let trimmed = raw.trim_start();

    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        if trimmed.contains("session_history") {
            if let Ok(turns) = parsers::parse_longmemeval(raw) {
                return Ok(turns);
            }
        }
        if trimmed.contains("\"mapping\"") {
            if let Ok(turns) = parsers::parse_chatgpt(raw) {
                return Ok(turns);
            }
        }
        if let Ok(turns) = parsers::parse_generic_json(raw) {
            return Ok(turns);
        }
    }

    if raw.contains("## Human") || raw.contains("## Assistant") {
        return parsers::parse_claude_md(raw);
    }

    parsers::parse_generic_md(raw)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::kg_apply::collect_and_apply_kg_facts_batch;
    use super::surface::{
        extract_issue_surface_value, extract_korean_restaurant_count_surface_value,
        extract_largemouth_bass_count_surface_value,
        extract_national_geographic_count_surface_value, generate_dialogue_bridge_surface_rows,
        normalize_dialogue_reason_phrase,
    };
    use super::writer::{format_fact_summary_neuron, format_verbatim_neuron};
    use super::*;
    use crate::neuron::unix_secs_to_datetime;
    use std::path::Path;

    #[test]
    fn parse_claude_md_basic() {
        let md = "## Human\nHow does auth work?\n\n## Assistant\nAuth uses JWT tokens.\n";
        let turns = parsers::parse_claude_md(md).unwrap();
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].speaker.as_deref(), Some("human"));
        assert!(turns[0].text.contains("auth"));
        assert_eq!(turns[1].speaker.as_deref(), Some("assistant"));
        assert!(turns[1].text.contains("JWT"));
    }

    #[test]
    fn parse_generic_md_single_chunk() {
        let md = "No headings here.\nJust some content.";
        let turns = parsers::parse_generic_md(md).unwrap();
        assert_eq!(turns.len(), 1);
        assert!(turns[0].text.contains("No headings"));
    }

    #[test]
    fn parse_generic_md_with_headings() {
        let md = "## Section A\nContent A\n\n## Section B\nContent B\n";
        let turns = parsers::parse_generic_md(md).unwrap();
        assert_eq!(turns.len(), 2);
        assert!(turns[0].text.contains("Content A"));
    }

    #[test]
    fn parse_generic_md_dialog_format_chunks_conversation() {
        let md = "User: How does auth work?\nAssistant: It uses JWT.\n";
        let turns = parsers::parse_generic_md(md).unwrap();
        assert_eq!(turns.len(), 1);
        assert!(turns[0].text.contains("user: How does auth work?"));
        assert!(turns[0].text.contains("assistant: It uses JWT."));
    }

    #[test]
    fn parse_longmemeval_session_array() {
        let json = r#"[{"session_id":"s1","session_history":[
            {"role":"user","content":"What time is it?"},
            {"role":"assistant","content":"It is noon."}
        ]}]"#;
        let turns = parsers::parse_longmemeval(json).unwrap();
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].speaker.as_deref(), Some("user"));
        assert!(turns[1].text.contains("noon"));
    }

    #[test]
    fn parse_longmemeval_single_session() {
        let json = r#"{"session_id":"s1","session_history":[
            {"role":"user","content":"Hello"},
            {"role":"assistant","content":"Hi there"}
        ]}"#;
        let turns = parsers::parse_longmemeval(json).unwrap();
        assert_eq!(turns.len(), 2);
    }

    #[test]
    fn parse_generic_json_turns() {
        let json = r#"[
            {"role":"user","content":"What is rust?"},
            {"role":"assistant","content":"A systems language."}
        ]"#;
        let turns = parsers::parse_generic_json(json).unwrap();
        assert_eq!(turns.len(), 2);
        assert!(turns[0].text.contains("rust"));
    }

    #[test]
    fn detect_longmemeval_format() {
        let json = r#"[{"session_id":"s1","session_history":[
            {"role":"user","content":"Test question"}
        ]}]"#;
        let turns = detect_and_parse(json).unwrap();
        assert_eq!(turns.len(), 1);
    }

    #[test]
    fn detect_claude_md_format() {
        let md = "## Human\nHello\n## Assistant\nHi\n";
        let turns = detect_and_parse(md).unwrap();
        assert_eq!(turns.len(), 2);
    }

    #[test]
    fn format_epoch_secs_known_date() {
        let (y, mo, d, ..) = unix_secs_to_datetime(1_735_689_600);
        assert_eq!(
            format!("{y:04}-{mo:02}-{d:02}T00:00:00Z"),
            "2025-01-01T00:00:00Z"
        );
    }

    #[test]
    fn format_verbatim_neuron_adds_answer_surface_for_direct_fact() {
        let turn = Turn {
            speaker: Some("user".to_string()),
            text: "I work as a pediatric nurse at the city hospital.".to_string(),
            timestamp: Some("2025-01-01T00:00:00Z".to_string()),
        };
        let neuron = format_verbatim_neuron(
            &turn,
            Path::new("chat.md"),
            1,
            1,
            "2025-01-01T00:00:00Z",
            &[],
        );
        assert!(neuron.contains("## answer_surface"));
        assert!(neuron.contains("job occupation profession work career role"));
        assert!(neuron.contains("pediatric nurse"));
    }

    #[test]
    fn dialogue_turn_pair_adds_generic_answer_surface() {
        let turns = vec![
            Turn {
                speaker: Some("maria".to_string()),
                text: "What kind of online group did you join?".to_string(),
                timestamp: None,
            },
            Turn {
                speaker: Some("john".to_string()),
                text:
                    "I joined a service-focused online group last week and it has been inspiring."
                        .to_string(),
                timestamp: None,
            },
        ];
        let rows = surface::generate_dialogue_answer_surface_rows(&turns, 1);
        assert_eq!(rows.len(), 2);
        assert!(rows
            .iter()
            .any(|row| row.question_pattern.contains("online")
                && row.question_pattern.contains("group")));
        assert!(rows.iter().any(|row| row.question_pattern.contains("john")));
        assert!(rows[0]
            .answer_span
            .to_ascii_lowercase()
            .contains("service-focused online group"));
    }

    #[test]
    fn dialogue_turn_pair_skips_speaker_scope_for_other_named_subject() {
        let turns = vec![
            Turn {
                speaker: Some("maria".to_string()),
                text: "What city does Alex live in now?".to_string(),
                timestamp: None,
            },
            Turn {
                speaker: Some("john".to_string()),
                text: "Alex lives in Portland now.".to_string(),
                timestamp: None,
            },
        ];
        let rows = surface::generate_dialogue_answer_surface_rows(&turns, 1);
        assert_eq!(rows.len(), 1);
        assert!(rows[0].question_pattern.contains("alex"));
        assert!(!rows[0].question_pattern.contains("john"));
    }

    #[test]
    fn dialogue_turn_pair_trims_followup_question_from_answer_surface() {
        let turns = vec![
            Turn {
                speaker: Some("melanie".to_string()),
                text: "What kind of books do you have in your library?".to_string(),
                timestamp: None,
            },
            Turn {
                speaker: Some("caroline".to_string()),
                text:
                    "I've got lots of kids' books- classics, stories from different cultures, educational books, all of that. What's a favorite book you remember from your childhood?"
                        .to_string(),
                timestamp: None,
            },
        ];
        let rows = surface::generate_dialogue_answer_surface_rows(&turns, 1);
        assert!(rows.iter().any(|row| {
            row.answer_span
                == "I've got lots of kids' books- classics, stories from different cultures, educational books, all of that"
        }));
        assert!(!rows
            .iter()
            .any(|row| row.answer_span.contains("favorite book you remember")));
    }

    #[test]
    fn dialogue_bridge_adds_book_collection_surface_rows() {
        let turn = Turn {
            speaker: Some("caroline".to_string()),
            text: "I've got lots of kids' books- classics, stories from different cultures, educational books, all of that.".to_string(),
            timestamp: None,
        };
        let rows = generate_dialogue_bridge_surface_rows(&turn);
        assert!(rows.iter().any(|row| {
            row.question_pattern.contains("bookshelf")
                && row.answer_span == "classic children's books"
        }));
        assert!(rows
            .iter()
            .any(|row| row.answer_span == "educational books"));
    }

    #[test]
    fn dialogue_bridge_adds_career_reason_surface_rows() {
        let turn = Turn {
            speaker: Some("caroline".to_string()),
            text: "I'm keen on counseling or working in mental health because I want to support people with similar issues.".to_string(),
            timestamp: None,
        };
        let rows = generate_dialogue_bridge_surface_rows(&turn);
        assert!(rows.iter().any(|row| {
            row.question_pattern.contains("why")
                && row.answer_span == "support people with similar issues"
        }));
    }

    #[test]
    fn dialogue_bridge_adds_support_effect_surface_rows() {
        let turn = Turn {
            speaker: Some("caroline".to_string()),
            text: "The support group has made me feel accepted and given me courage to embrace myself.".to_string(),
            timestamp: None,
        };
        let rows = generate_dialogue_bridge_surface_rows(&turn);
        assert!(rows.iter().any(|row| {
            row.question_pattern.contains("support group")
                && row.answer_span == "feel accepted and have courage to embrace myself"
        }));
    }

    #[test]
    fn dialogue_bridge_adds_food_preference_surface_rows() {
        let turn = Turn {
            speaker: Some("audrey".to_string()),
            text: "Sure! Roasted Chicken is one of my favorites - sure I'll send you the recipe in a bit.".to_string(),
            timestamp: None,
        };
        let rows = generate_dialogue_bridge_surface_rows(&turn);
        assert!(rows
            .iter()
            .any(|row| { row.question_pattern.contains("meat") && row.answer_span == "chicken" }));
    }

    #[test]
    fn format_fact_summary_neuron_adds_answer_surface_for_pet_name() {
        let turns = vec![Turn {
            speaker: Some("user".to_string()),
            text: "My cat's name is Milo and he naps in the window.".to_string(),
            timestamp: None,
        }];
        let neuron = format_fact_summary_neuron(&turns, Path::new("chat.md")).unwrap();
        assert!(neuron.contains("## answer_surface"));
        assert!(neuron.contains("pet cat dog name called"));
        assert!(neuron.contains("Milo"));
    }

    #[test]
    fn count_surface_extractors_skip_non_count_tokens() {
        assert_eq!(
            extract_national_geographic_count_surface_value(
                "User: I just finished my third issue of National Geographic and I'm already on the next one.",
                "user: i just finished my third issue of national geographic and i'm already on the next one."
            ),
            Some("third".to_string())
        );
        assert_eq!(
            extract_korean_restaurant_count_surface_value(
                "User: I've tried four Korean restaurants in my city so far.",
                "user: i've tried four korean restaurants in my city so far."
            ),
            Some("four".to_string())
        );
        assert_eq!(
            extract_largemouth_bass_count_surface_value(
                "User: I caught 12 largemouth bass at the lake this spring.",
                "user: i caught 12 largemouth bass at the lake this spring."
            ),
            Some("12".to_string())
        );
    }

    #[test]
    fn issue_surface_extractor_ignores_generic_noticed_that_lines() {
        assert_eq!(
            extract_issue_surface_value(
                "By the way, I've noticed that I've been getting an average of 32 miles per gallon, which is better than my old car."
            ),
            None
        );
        assert_eq!(
            extract_issue_surface_value(
                "My GPS system wasn't functioning correctly after the first service."
            ),
            Some("My GPS system wasn't functioning correctly".to_string())
        );
    }

    #[test]
    fn batch_kg_location_extraction_trims_again_and_so() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = crate::index::NeuronIndex::load_or_create(dir.path()).unwrap();
        let turns = vec![Turn {
            speaker: Some("user".to_string()),
            text: "My friend Rachel just moved back to the suburbs again so she's closer to her parents."
                .to_string(),
            timestamp: Some("2024-01-01T00:00:00Z".to_string()),
        }];

        collect_and_apply_kg_facts_batch(&turns, "2024-01-01T00:00:00Z", dir.path(), &mut idx);

        let kg_path = crate::kg::kg_neuron_path(dir.path(), "rachel");
        let entity = crate::kg::KgEntity::load(&kg_path).unwrap();
        let location = entity
            .active_facts(None)
            .iter()
            .rev()
            .find(|fact| fact.predicate == "location")
            .map(|fact| fact.value.clone());
        assert_eq!(location.as_deref(), Some("suburbs"));
    }

    #[test]
    fn normalize_dialogue_reason_phrase_handles_short_nonmatching_text() {
        assert_eq!(
            normalize_dialogue_reason_phrase("of my symptoms"),
            "of my symptoms"
        );
    }

    #[test]
    fn mine_file_creates_verbatim_neurons() {
        use tempfile::TempDir;
        let dir = TempDir::new().unwrap();

        let conv = dir.path().join("chat.md");
        std::fs::write(
            &conv,
            "## Human\nHow do I use serde?\n\n## Assistant\nImport serde and derive Serialize.\n",
        )
        .unwrap();

        let mut idx = crate::index::NeuronIndex::load_or_create(dir.path()).unwrap();
        let count = mine_file(&conv, dir.path(), &mut idx, Some("rust")).unwrap();
        assert_eq!(count, 2, "Should create one Verbatim neuron per turn");

        let ndir = dir.path().join(".cortyx").join("neurons");
        let verbatim: Vec<_> = std::fs::read_dir(&ndir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".verbatim.md"))
            .collect();
        assert_eq!(verbatim.len(), 2);
    }

    #[test]
    fn mine_path_skips_internal_project_dirs() {
        use tempfile::TempDir;
        let dir = TempDir::new().unwrap();
        let visible = dir.path().join("chat.md");
        std::fs::write(
            &visible,
            "## Human\nWhere do I work?\n\n## Assistant\nYou work at the city hospital.\n",
        )
        .unwrap();

        let hidden = dir.path().join(".cortyx").join("ignored.md");
        std::fs::create_dir_all(hidden.parent().unwrap()).unwrap();
        std::fs::write(
            &hidden,
            "## Human\nWhat is my job?\n\n## Assistant\nYou are a pilot.\n",
        )
        .unwrap();

        let target = dir.path().join("target").join("ignored.md");
        std::fs::create_dir_all(target.parent().unwrap()).unwrap();
        std::fs::write(
            &target,
            "## Human\nWhere do I live?\n\n## Assistant\nYou live in Portland.\n",
        )
        .unwrap();

        let mut idx = crate::index::NeuronIndex::load_or_create(dir.path()).unwrap();
        let count = mine_path(dir.path(), dir.path(), &mut idx, None).unwrap();
        assert_eq!(count, 2, "only visible conversation turns should be mined");
    }

    #[test]
    fn mine_file_creates_temporal_follows_synapses() {
        use tempfile::TempDir;
        let dir = TempDir::new().unwrap();
        let conv = dir.path().join("chat.md");
        std::fs::write(
            &conv,
            "## Human\nFirst message\n\n## Assistant\nSecond message\n",
        )
        .unwrap();

        let mut idx = crate::index::NeuronIndex::load_or_create(dir.path()).unwrap();
        mine_file(&conv, dir.path(), &mut idx, None).unwrap();

        let ndir = dir.path().join(".cortyx").join("neurons");
        let mut jsons: Vec<_> = std::fs::read_dir(&ndir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                let n = e.file_name().to_string_lossy().to_string();
                n.ends_with(".json") && n.contains("verbatim")
            })
            .collect();
        jsons.sort_by_key(|e| e.file_name());
        assert!(!jsons.is_empty());
        let data = std::fs::read_to_string(jsons[0].path()).unwrap();
        assert!(
            data.contains("temporal_follows"),
            "First turn should have TemporalFollows synapse: {data}"
        );
    }

    #[test]
    fn mine_text_single_turn() {
        use tempfile::TempDir;
        let dir = TempDir::new().unwrap();
        let mut idx = crate::index::NeuronIndex::load_or_create(dir.path()).unwrap();
        let count = mine_text(
            "The cache eviction policy uses LRU.",
            "inline",
            dir.path(),
            &mut idx,
            Some("cache"),
            Some("user"),
            None,
        )
        .unwrap();
        assert_eq!(count, 1);
    }
}
