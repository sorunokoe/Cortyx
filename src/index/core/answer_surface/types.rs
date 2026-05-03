//! Answer surface types for synthetic answer generation.
//!
//! These types support the answer_mode feature which generates answers
//! from indexed neuron content without changing the core retrieval path.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::PathBuf;

// ─── Answer Surface Row & Bucket ──────────────────────────────────────────────

#[derive(Clone, Debug)]
pub(crate) struct IndexAnswerSurfaceRow {
    pub question_pattern: String,
    pub answer_span: String,
    pub confidence: f32,
}

#[derive(Clone, Debug)]
pub(crate) struct IndexAnswerSurfaceBucket {
    pub answer_span: String,
    pub best_score: f32,
    pub total_score: f32,
    pub max_overlap: usize,
    pub paths: HashSet<PathBuf>,
    pub hits: usize,
    pub evidence: Vec<String>,
    pub relation_families: HashSet<SyntheticAnswerSurfaceRelationFamily>,
}

// ─── Answer Surface Enums ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SyntheticAnswerSurfaceExpectedType {
    Generic,
    ListItem,
    Date,
    Duration,
    Count,
    Person,
    Location,
    NameLike,
    Status,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SyntheticAnswerSurfaceRouteKind {
    Default,
    YesNo,
    Choice,
    LocationLift,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SyntheticAnswerSurfaceLocationTarget {
    State,
    Country,
    NationalPark,
}

/// Classifies what kind of personal-knowledge relationship a question targets.
///
/// This set is intentionally domain-specific: Cortyx ships as a personal knowledge
/// graph tool and these families are example routing categories for a single-user PKG.
/// Third-party deployments can extend the answer plane by adding additional families
/// here and wiring routing rules in `synthetic_answer_surface_query_relation_families`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum SyntheticAnswerSurfaceRelationFamily {
    Research,
    Career,
    Activity,
    FamilyActivity,
    SelfCareActivity,
    Book,
    CampLocation,
    FriendGroupDuration,
    SupportNetwork,
    KidsPreference,
    PaintSubject,
    Origin,
    Identity,
    Ally,
    Religion,
    Relationship,
    CommunityEvent,
    ChildHelpEvent,
}

// ─── Query Profile ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub(crate) struct SyntheticAnswerSurfaceChoiceOption {
    pub display: String,
    pub term_keys: HashSet<String>,
    pub affinity_term_keys: HashSet<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct SyntheticAnswerSurfaceQueryProfile {
    pub task_term_keys: HashSet<String>,
    pub subject_term_keys: HashSet<String>,
    pub anchor_term_keys: HashSet<String>,
    pub relation_term_keys: HashSet<String>,
    pub expected_type: SyntheticAnswerSurfaceExpectedType,
    pub route_kind: SyntheticAnswerSurfaceRouteKind,
    pub choice_options: Vec<SyntheticAnswerSurfaceChoiceOption>,
    pub location_target: Option<SyntheticAnswerSurfaceLocationTarget>,
    pub requires_strict_anchor_overlap: bool,
    pub requires_completed_evidence: bool,
    pub strict_relation_family_match: bool,
    pub relation_families: HashSet<SyntheticAnswerSurfaceRelationFamily>,
    pub allows_count_projection_from_lists: bool,
}
