use super::super::*;
use crate::types::TermFrequency;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::{Arc, Mutex};

#[cfg(feature = "embed")]
use crate::embedder::EmbeddingStore;

/// Borrowed view over the retrieval portion of `NeuronIndex`.
pub struct RetrievalStateView<'a> {
    pub entries: &'a [BM25Entry],
    pub adjacency: &'a HashMap<PathBuf, Vec<Synapse>>,
    pub path_index: &'a HashMap<PathBuf, usize>,
    pub parent_index: &'a HashMap<PathBuf, Vec<usize>>,
    pub df_cache: &'a HashMap<String, usize>,
    pub posting_list: &'a HashMap<String, Vec<usize>>,
    pub avg_doc_len: f32,
    pub avg_verbatim_doc_len: f32,
    pub module_index: &'a HashMap<String, Vec<usize>>,
    pub vocab_bridge: &'a HashMap<String, HashSet<String>>,
    pub morpheme_map: &'a HashMap<String, Vec<String>>,
    pub session_index: &'a HashMap<String, Vec<usize>>,
    pub pmi_neighbors: &'a HashMap<String, Vec<String>>,
    #[cfg(feature = "embed")]
    pub embeddings: &'a EmbeddingStore,
    pub idf_n: usize,
}

/// Borrowed view over feedback / learning state.
pub struct FeedbackStateView<'a> {
    pub coactivation_counts: &'a HashMap<PathBuf, HashMap<String, u32>>,
    pub co_return_counts: &'a Mutex<HashMap<(usize, usize), u32>>,
    pub session_utilization: &'a [[usize; 2]],
}

/// Borrowed view over persistence state.
pub struct PersistenceStateView<'a> {
    pub project_root: &'a PathBuf,
    pub pending_append_count: usize,
    pub has_pending_updates: &'a AtomicBool,
    pub delta_base: &'a AtomicUsize,
    pub delta_dirty: &'a AtomicBool,
    pub structural_artifacts_dirty: &'a AtomicBool,
    pub dirty_sidecars: &'a Mutex<HashSet<PathBuf>>,
}

/// Borrowed view over watcher state.
pub struct WatcherStateView<'a> {
    pub dirty_set: &'a Arc<Mutex<HashSet<PathBuf>>>,
}

/// Snapshot of feedback state for a single query.
pub struct FeedbackSnapshot<'a> {
    pub coactivation_counts: &'a HashMap<PathBuf, HashMap<String, u32>>,
    pub co_return_counts: &'a Mutex<HashMap<(usize, usize), u32>>,
    pub session_utilization: &'a [[usize; 2]],
}

/// Immutable snapshot of index state for a single query.
pub struct QueryContext<'a> {
    pub task: &'a str,
    pub task_lower: String,
    pub terms: Vec<String>,
    pub seed_scoring_terms: Vec<String>,
    pub active_scoring_terms: Vec<String>,
    pub ranking_terms: Vec<String>,
    pub seed_ranking_terms: Vec<String>,
    pub bridge_ranking_terms: Vec<String>,
    pub bridge_scoring_terms: Option<Vec<String>>,
    pub module_filter: Option<&'a str>,
    pub kind_filter: Option<&'a str>,
    pub kind_lower: Option<String>,
    pub max_tokens: usize,
    pub session_id: Option<&'a str>,
    pub is_counting: bool,
    pub is_knowledge_update: bool,
    pub force_tfidf: bool,
    pub explicit_current_state_query: bool,
    pub named_person_move_query: bool,
    pub raw_counting_focus_terms: Vec<String>,
    pub raw_knowledge_focus_terms: Vec<String>,
    pub idf_n: usize,
    pub avg_doc_len: f32,
    pub avg_verbatim_doc_len: f32,
    pub seed_candidate_ids: HashSet<usize>,
    pub bridge_candidate_ids: HashSet<usize>,
    pub concept_cloud_candidate_ids: HashSet<usize>,
    pub module_set: Option<HashSet<usize>>,
    pub counting_augment: Vec<usize>,
    pub kg_router_path: Option<PathBuf>,
    pub entries: &'a [BM25Entry],
    pub posting_list: &'a HashMap<String, Vec<usize>>,
    pub adjacency: &'a HashMap<PathBuf, Vec<Synapse>>,
    pub path_index: &'a HashMap<PathBuf, usize>,
    pub parent_index: &'a HashMap<PathBuf, Vec<usize>>,
    pub module_index: &'a HashMap<String, Vec<usize>>,
    pub df_cache: &'a HashMap<String, usize>,
    pub vocab_bridge: &'a HashMap<String, HashSet<String>>,
    pub morpheme_map: &'a HashMap<String, Vec<String>>,
    pub pmi_neighbors: &'a HashMap<String, Vec<String>>,
    pub session_index: &'a HashMap<String, Vec<usize>>,
    pub project_root: &'a Path,
    pub feedback: FeedbackSnapshot<'a>,
    #[cfg(feature = "embed")]
    pub embeddings: Option<&'a EmbeddingStore>,
}

impl<'a> QueryContext<'a> {
    pub fn entry(&self, idx: usize) -> &BM25Entry {
        &self.entries[idx]
    }

    pub fn kind_matches(&self, entry: &BM25Entry) -> bool {
        match self.kind_lower.as_deref() {
            Some("conversation") => matches!(entry.kind, NeuronKind::Verbatim),
            Some("code") => matches!(entry.kind, NeuronKind::Core | NeuronKind::Project),
            _ => matches!(
                entry.kind,
                NeuronKind::Core | NeuronKind::Project | NeuronKind::Verbatim
            ),
        }
    }

    pub fn module_matches(&self, idx: usize) -> bool {
        self.module_set
            .as_ref()
            .is_none_or(|module_set| module_set.contains(&idx))
    }

    pub fn score_entry_with_terms(&self, terms: &[String], entry: &BM25Entry) -> f32 {
        let n = self.idf_n.max(1) as f32;
        let avg = self.avg_doc_len.max(1.0);
        let dl = entry.term_count as f32;
        let len_norm = 1.0 - BM25_B + BM25_B * (dl / avg);
        let k1 = if matches!(entry.kind, NeuronKind::Verbatim) {
            1.5
        } else {
            BM25_K1
        };

        let raw: f32 = terms
            .iter()
            .map(|term| {
                let tf = entry
                    .term_freq
                    .get(term)
                    .map(|value| value.get())
                    .unwrap_or(0.0);
                if tf == 0.0 {
                    return 0.0;
                }
                let df = self.df_cache.get(term).copied().unwrap_or(1) as f32;
                let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln().max(0.0);
                const BM25_DELTA: f32 = 0.5;
                idf * (BM25_DELTA + (tf * (k1 + 1.0)) / (tf + k1 * len_norm))
            })
            .sum();

        let hit_multiplier = if entry.use_count < MIN_SAMPLE_SIZE {
            1.0
        } else {
            let hit_rate = entry.hit_count as f32 / entry.use_count as f32;
            (1.0 + hit_rate).min(1.5)
        };

        raw * entry.confidence_score
            * hit_multiplier
            * entry.staleness_multiplier
            * if entry.quality_score < 0.4 { 0.7 } else { 1.0 }
    }

    pub fn score_index_with_terms(&self, terms: &[String], idx: usize) -> f32 {
        self.score_entry_with_terms(terms, self.entry(idx))
    }

    pub fn score_index(&self, idx: usize) -> f32 {
        self.score_index_with_terms(&self.ranking_terms, idx)
    }
}

#[cfg(test)]
pub struct QueryContextFixture {
    pub entries: Vec<BM25Entry>,
    pub posting_list: HashMap<String, Vec<usize>>,
    pub adjacency: HashMap<PathBuf, Vec<Synapse>>,
    pub path_index: HashMap<PathBuf, usize>,
    pub parent_index: HashMap<PathBuf, Vec<usize>>,
    pub module_index: HashMap<String, Vec<usize>>,
    pub df_cache: HashMap<String, usize>,
    pub vocab_bridge: HashMap<String, HashSet<String>>,
    pub morpheme_map: HashMap<String, Vec<String>>,
    pub pmi_neighbors: HashMap<String, Vec<String>>,
    pub session_index: HashMap<String, Vec<usize>>,
    pub coactivation_counts: HashMap<PathBuf, HashMap<String, u32>>,
    pub co_return_counts: Mutex<HashMap<(usize, usize), u32>>,
    pub session_utilization: Vec<[usize; 2]>,
    pub project_root: PathBuf,
}

#[cfg(test)]
impl QueryContextFixture {
    pub fn new(entries: Vec<BM25Entry>) -> Self {
        let path_index = entries
            .iter()
            .enumerate()
            .map(|(idx, entry)| (entry.neuron_path.clone(), idx))
            .collect();
        let mut fixture = Self {
            entries,
            posting_list: HashMap::new(),
            adjacency: HashMap::new(),
            path_index,
            parent_index: HashMap::new(),
            module_index: HashMap::new(),
            df_cache: HashMap::new(),
            vocab_bridge: HashMap::new(),
            morpheme_map: HashMap::new(),
            pmi_neighbors: HashMap::new(),
            session_index: HashMap::new(),
            coactivation_counts: HashMap::new(),
            co_return_counts: Mutex::new(HashMap::new()),
            session_utilization: Vec::new(),
            project_root: PathBuf::from("."),
        };
        for entry in &fixture.entries {
            for term in entry.term_freq.keys() {
                *fixture.df_cache.entry(term.clone()).or_insert(0) += 1;
            }
        }
        fixture
    }

    pub fn ctx<'a>(&'a self, task: &'a str) -> QueryContext<'a> {
        let avg_doc_len = if self.entries.is_empty() {
            1.0
        } else {
            self.entries
                .iter()
                .map(|entry| entry.term_count)
                .sum::<usize>() as f32
                / self.entries.len() as f32
        };
        QueryContext {
            task,
            task_lower: task.to_ascii_lowercase(),
            terms: Vec::new(),
            seed_scoring_terms: Vec::new(),
            active_scoring_terms: Vec::new(),
            ranking_terms: Vec::new(),
            seed_ranking_terms: Vec::new(),
            bridge_ranking_terms: Vec::new(),
            bridge_scoring_terms: None,
            module_filter: None,
            kind_filter: None,
            kind_lower: None,
            max_tokens: 1024,
            session_id: None,
            is_counting: false,
            is_knowledge_update: false,
            force_tfidf: false,
            explicit_current_state_query: false,
            named_person_move_query: false,
            raw_counting_focus_terms: Vec::new(),
            raw_knowledge_focus_terms: Vec::new(),
            idf_n: self.entries.len().max(1),
            avg_doc_len,
            avg_verbatim_doc_len: avg_doc_len,
            seed_candidate_ids: HashSet::new(),
            bridge_candidate_ids: HashSet::new(),
            concept_cloud_candidate_ids: HashSet::new(),
            module_set: None,
            counting_augment: Vec::new(),
            kg_router_path: None,
            entries: &self.entries,
            posting_list: &self.posting_list,
            adjacency: &self.adjacency,
            path_index: &self.path_index,
            parent_index: &self.parent_index,
            module_index: &self.module_index,
            df_cache: &self.df_cache,
            vocab_bridge: &self.vocab_bridge,
            morpheme_map: &self.morpheme_map,
            pmi_neighbors: &self.pmi_neighbors,
            session_index: &self.session_index,
            project_root: self.project_root.as_path(),
            feedback: FeedbackSnapshot {
                coactivation_counts: &self.coactivation_counts,
                co_return_counts: &self.co_return_counts,
                session_utilization: &self.session_utilization,
            },
            #[cfg(feature = "embed")]
            embeddings: None,
        }
    }
}

#[cfg(test)]
pub fn test_entry(path: &str, kind: NeuronKind, terms: &[(&str, f32)]) -> BM25Entry {
    let term_freq = terms
        .iter()
        .map(|(term, tf)| ((*term).to_string(), TermFrequency::new(*tf)))
        .collect::<HashMap<_, _>>();
    BM25Entry {
        neuron_path: PathBuf::from(path),
        kind,
        term_count: terms.len().max(1),
        tokens: 32,
        term_freq,
        confidence_score: 1.0,
        staleness_multiplier: 1.0,
        quality_score: 1.0,
        ..Default::default()
    }
}
