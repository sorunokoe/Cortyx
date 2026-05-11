use crate::index::ContextMetadata;
use std::collections::HashSet;
use std::path::PathBuf;

pub(super) const QUESTION_STOPWORDS: &[&str] = &[
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "for",
    "with",
    "from",
    "that",
    "this",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "how",
    "did",
    "does",
    "do",
    "is",
    "was",
    "are",
    "were",
    "am",
    "would",
    "could",
    "should",
    "can",
    "will",
    "may",
    "might",
    "to",
    "of",
    "in",
    "on",
    "at",
    "by",
    "it",
    "its",
    "my",
    "your",
    "our",
    "their",
    "his",
    "her",
    "have",
    "has",
    "had",
    "been",
    "being",
    "after",
    "before",
    "later",
    "earlier",
    "about",
    "into",
    "over",
    "under",
    "through",
    "likely",
    "probably",
    "possibly",
    "potentially",
    "considered",
    "still",
    "more",
    "most",
    "less",
    "least",
];

pub(super) const PREPOSITION_HINTS: &[&str] = &[
    "for", "about", "with", "from", "to", "in", "at", "on", "after", "before", "during",
];

pub(super) const TAIL_BOUNDARIES: &[&str] = &[
    ".",
    "!",
    "?",
    ";",
    ",",
    " - ",
    " — ",
    " because ",
    " but ",
    " so ",
    " while ",
    " although ",
    " though ",
    " which ",
    " that ",
    " who ",
    " when ",
    " lately ",
    " recently ",
    " yesterday ",
    " today ",
    " tomorrow ",
    " last ",
    " next ",
    " 'cause ",
    " cause ",
];

pub(super) const COPULA_BOUNDARIES: &[&str] = &[
    " is ",
    " was ",
    " are ",
    " were ",
    " feels ",
    " felt ",
    " sounds ",
    " sounded ",
    " seems ",
    " seemed ",
    " looks ",
    " looked ",
];

pub(super) const ENTITY_STOPWORDS: &[&str] = &[
    "what",
    "who",
    "when",
    "where",
    "why",
    "how",
    "which",
    "did",
    "does",
    "do",
    "the",
    "a",
    "an",
    "on",
    "in",
    "at",
    "for",
    "of",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
];

pub(super) const MULTIHOP_BUNDLE_FIELDS: &[&str] = &[
    "next step",
    "next action",
    "dependencies",
    "dependency",
    "blocker",
    "status",
    "outcome",
    "result",
    "action",
    "title",
    "focus",
    "goal",
    "entities",
    "entity",
    "location",
    "residence",
    "city",
    "home",
    "job",
    "occupation",
    "career",
    "role",
];

#[derive(Debug, Clone)]
pub(crate) struct EvidenceItem {
    pub(crate) path: PathBuf,
    pub(crate) score: f32,
    pub(crate) metadata: Option<ContextMetadata>,
    pub(crate) snippet: String,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ReasoningEnhancement {
    pub(crate) supplemental_evidence: Vec<EvidenceItem>,
    pub(crate) summary_lines: Vec<String>,
    /// Traversal chains rendered as "seed → hop1 → hop2 (score X.XX)".
    pub(crate) chain_lines: Vec<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct CandidateLine {
    pub(crate) path: PathBuf,
    pub(crate) text: String,
    pub(crate) weight: f32,
    pub(crate) retrieval_score: f32,
    pub(crate) support_overlap: usize,
    pub(crate) anchor_overlap: usize,
    pub(crate) specific_anchor_overlap: usize,
}

#[derive(Debug, Clone)]
pub(super) struct AnswerSurfaceRow {
    pub(super) question_pattern: String,
    pub(super) answer_span: String,
    pub(super) confidence: f32,
}

#[derive(Debug, Clone)]
pub(crate) struct DialogueTurn {
    pub(crate) speaker: Option<String>,
    pub(crate) text: String,
    pub(crate) session_date: Option<(i32, u32, u32)>,
}

#[derive(Debug, Clone)]
pub(crate) struct TemporalCandidate {
    pub(crate) text: String,
    pub(crate) base_date: Option<(i32, u32, u32)>,
    pub(crate) retrieval_score: f32,
    pub(crate) user_authored: bool,
    pub(crate) ordinal: usize,
    pub(crate) sequence_rank: Option<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TemporalDirection {
    Earlier,
    Later,
}

#[derive(Debug, Clone)]
pub(crate) struct ChoiceOption {
    pub(crate) display: String,
    pub(crate) tokens: Vec<String>,
}

#[derive(Debug, Clone)]
pub(crate) enum TemporalStateQuery {
    CurrentValue,
    AsOfValue { as_of: String },
    LastChange { target_value: Option<ChoiceOption> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TemporalAnswerPoint {
    Day { year: i32, month: u32, day: u32 },
    Month { year: i32, month: u32 },
    Year { year: i32 },
}

#[derive(Debug, Clone)]
pub(crate) enum TemporalGapEndpoint {
    Event(ChoiceOption),
    CurrentMoment,
}

#[derive(Debug, Clone)]
pub(crate) enum TemporalGapAnswerStyle {
    FixedUnit { unit: String },
    NaturalLanguage,
}

#[derive(Debug, Clone)]
pub(crate) struct TemporalGapQuery {
    pub(crate) start: ChoiceOption,
    pub(crate) end: TemporalGapEndpoint,
    pub(crate) answer_style: TemporalGapAnswerStyle,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CalendarGroundedRank {
    pub(crate) ordinal: usize,
    pub(crate) rank: i32,
    pub(crate) overlap: usize,
    pub(crate) score: f32,
}

#[derive(Debug, Clone)]
pub(super) struct RelationKgSupport {
    pub(super) values: Vec<String>,
}

#[derive(Debug, Clone)]
pub(super) struct RelationCandidateBucket {
    pub(super) best_candidate: CandidateLine,
    pub(super) best_single_weight: f32,
    pub(super) total_weight: f32,
    pub(super) max_retrieval_score: f32,
    pub(super) max_support_overlap: usize,
    pub(super) max_anchor_overlap: usize,
    pub(super) max_specific_anchor_overlap: usize,
    pub(super) paths: HashSet<PathBuf>,
    pub(super) hits: usize,
}

#[derive(Debug, Clone)]
pub(super) struct AnswerSurfaceBucket {
    pub(super) answer_span: String,
    pub(super) best_score: f32,
    pub(super) total_score: f32,
    pub(super) best_confidence: f32,
    pub(super) max_overlap: usize,
    pub(super) max_anchor_overlap: usize,
    pub(super) paths: HashSet<PathBuf>,
    pub(super) hits: usize,
}

#[derive(Debug, Clone)]
pub(super) enum RelationResolution {
    Answer(String),
    Suppress,
}
