//! Query type definitions for temporal anchor extraction.

#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::index) enum TemporalAnchorQuery {
    ElapsedBeforeEvent(ElapsedBeforeEventQuery),
    Interval(TemporalIntervalQuery),
    ElapsedGap(TemporalElapsedGapQuery),
    RelativeRecall(RelativeTemporalRecallQuery),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::index) struct ElapsedBeforeEventQuery {
    pub(in crate::index) subject_phrase: String,
    pub(in crate::index) event_phrase: String,
    pub(in crate::index) required_terms: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::index) struct TemporalIntervalQuery {
    pub(in crate::index) start_phrase: String,
    pub(in crate::index) end_phrase: String,
    pub(in crate::index) required_terms: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::index) struct TemporalElapsedGapQuery {
    pub(in crate::index) start_phrase: String,
    pub(in crate::index) end_phrase: String,
    pub(in crate::index) unit: String,
    pub(in crate::index) required_terms: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::index) enum RelativeTemporalRecallAnswerKind {
    BookTitle,
    SourcePerson,
    DirectObject,
    EventClause,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::index) struct RelativeTemporalRecallQuery {
    pub(in crate::index) target_day: i32,
    pub(in crate::index) prompt_body: String,
    pub(in crate::index) focus_terms: Vec<String>,
    pub(in crate::index) answer_kind: RelativeTemporalRecallAnswerKind,
}
