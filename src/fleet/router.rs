//! Lazy routing for supplementary fleet context queries.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock, RwLock};
use std::time::{Duration, Instant};

use crate::error::Result;
use crate::index::NeuronIndex;
use crate::neuron::estimate_tokens;

use super::{FleetQueryResult, FleetRegistry};

pub const FLEET_LOW_CONFIDENCE_THRESHOLD: f32 = 4.0;
pub const FLEET_QUERY_TIMEOUT_MS: u64 = 200;

fn fleet_index_cache() -> &'static RwLock<HashMap<PathBuf, Arc<NeuronIndex>>> {
    static CACHE: OnceLock<RwLock<HashMap<PathBuf, Arc<NeuronIndex>>>> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

pub(crate) fn cache_fleet_index(node_path: &Path, index: Arc<NeuronIndex>) {
    match fleet_index_cache().write() {
        Ok(mut cache) => {
            cache.insert(node_path.to_path_buf(), index);
        },
        Err(err) => tracing::warn!("Failed to lock fleet index cache for write: {err}"),
    }
}

pub(crate) fn invalidate_fleet_index(node_path: &Path) {
    match fleet_index_cache().write() {
        Ok(mut cache) => {
            cache.remove(node_path);
        },
        Err(err) => tracing::warn!("Failed to lock fleet index cache for invalidation: {err}"),
    }
}

fn load_fleet_index(node_path: &Path) -> Result<Arc<NeuronIndex>> {
    if let Ok(cache) = fleet_index_cache().read() {
        if let Some(index) = cache.get(node_path) {
            return Ok(Arc::clone(index));
        }
    } else {
        tracing::warn!("Failed to lock fleet index cache for read");
    }

    let index = Arc::new(NeuronIndex::load_or_create(node_path)?);
    cache_fleet_index(node_path, Arc::clone(&index));
    Ok(index)
}

/// Route a low-confidence query across matching fleet nodes.
pub async fn route_fleet_query(
    task: &str,
    module_filter: Option<&str>,
    max_tokens: usize,
    registry: &FleetRegistry,
) -> Vec<FleetQueryResult> {
    let matching_nodes: Vec<_> = registry
        .nodes
        .iter()
        .filter(|node| {
            module_filter
                .is_none_or(|module| node.modules.iter().any(|candidate| candidate == module))
        })
        .cloned()
        .collect();

    if matching_nodes.is_empty() {
        return Vec::new();
    }

    let mut join_set = tokio::task::JoinSet::new();
    for node in matching_nodes {
        let task = task.to_string();
        let module_filter = module_filter.map(str::to_string);
        join_set
            .spawn_blocking(move || query_node(node, &task, module_filter.as_deref(), max_tokens));
    }

    let deadline = Instant::now() + Duration::from_millis(FLEET_QUERY_TIMEOUT_MS);
    let mut results = Vec::new();

    while !join_set.is_empty() {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            tracing::warn!("Fleet query timed out before all nodes completed");
            break;
        }

        match tokio::time::timeout(remaining, join_set.join_next()).await {
            Ok(Some(Ok(Some(result)))) => results.push(result),
            Ok(Some(Ok(None))) => {},
            Ok(Some(Err(err))) => {
                tracing::warn!("Fleet worker task failed: {err}");
            },
            Ok(None) => break,
            Err(_) => {
                tracing::warn!("Fleet query timed out before all nodes completed");
                break;
            },
        }
    }

    results.sort_by(|left, right| {
        right
            .top_score
            .total_cmp(&left.top_score)
            .then_with(|| left.node_alias.cmp(&right.node_alias))
    });
    results
}

fn query_node(
    node: super::FleetNode,
    task: &str,
    module_filter: Option<&str>,
    max_tokens: usize,
) -> Option<FleetQueryResult> {
    let index_path = node.path.join(".cortyx").join("index.json");
    if !index_path.exists() {
        tracing::warn!(
            path = %node.path.display(),
            alias = %node.alias,
            index_path = %index_path.display(),
            "Skipping fleet node without registered index"
        );
        return None;
    }

    let index = match load_fleet_index(&node.path) {
        Ok(index) => index,
        Err(err) => {
            tracing::warn!(
                path = %node.path.display(),
                alias = %node.alias,
                "Failed to load fleet node index: {err}"
            );
            return None;
        },
    };

    if index.neuron_count() == 0 {
        return None;
    }

    let top_score = index.peek_max_bm25_score(task);
    let (mut neurons, _overflow) = index.get_contexts_with_scores_and_overflow(
        task,
        max_tokens,
        module_filter,
        None,
        None,
        false,
    );
    neurons.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });

    let mut contexts = Vec::new();
    let mut used_tokens = 0usize;
    for (path, _bm25_score) in neurons.into_iter().take(3) {
        let content = match fs::read_to_string(&path) {
            Ok(content) => content,
            Err(err) => {
                tracing::warn!(
                    path = %path.display(),
                    alias = %node.alias,
                    "Failed to read fleet neuron content: {err}"
                );
                continue;
            },
        };
        let token_count = estimate_tokens(&content).get();
        if !contexts.is_empty() && used_tokens.saturating_add(token_count) > max_tokens {
            break;
        }
        used_tokens = used_tokens.saturating_add(token_count);
        contexts.push(content);
        if used_tokens >= max_tokens {
            break;
        }
    }

    if contexts.is_empty() {
        return None;
    }

    Some(FleetQueryResult {
        node_id: node.id,
        node_alias: node.alias,
        contexts: contexts.join("\n\n---\n\n"),
        top_score,
    })
}
