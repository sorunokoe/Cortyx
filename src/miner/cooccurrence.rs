use std::path::Path;

use super::Turn;

/// R17 Sol2: Self-Building Co-occurrence Ontology (Firth Principle).
///
/// Builds a term co-occurrence graph from session turns:
/// - Same-turn co-occurrence: weight +3
/// - Adjacent-turn co-occurrence: weight +1
///
/// Saves top-N clusters (terms with weight ≥2) to `.cortyx/cooccurrence.json`.
/// NeuronIndex loads this in `rebuild_derived()` and merges it into `vocab_bridge`.
pub(super) fn build_and_save_cooccurrence(turns: &[Turn], project_root: &Path) {
    const STOPS: &[&str] = &[
        "the", "and", "but", "for", "are", "was", "has", "had", "she", "her", "his", "him", "they",
        "them", "our", "you", "your", "this", "that", "with", "from", "have", "will", "been",
        "just", "when", "what", "where", "then", "than", "also", "very", "well", "even", "most",
        "some", "many", "long", "good", "back", "into", "over", "down", "more", "such", "both",
        "got", "get", "did", "its", "all", "can", "not", "out", "now", "new", "like", "know",
        "make", "said", "see", "too", "here", "yes", "one", "two", "day", "use", "how", "him",
        "lot", "used", "since", "today",
    ];
    let tokenise = |text: &str| -> Vec<String> {
        text.split(|c: char| !c.is_alphanumeric())
            .filter_map(|w| {
                let lower = w.to_lowercase();
                if lower.len() >= 3 && !STOPS.contains(&lower.as_str()) {
                    Some(lower)
                } else {
                    None
                }
            })
            .collect()
    };

    let mut cooccur: std::collections::HashMap<(String, String), u32> =
        std::collections::HashMap::new();
    let turn_tokens: Vec<Vec<String>> = turns.iter().map(|t| tokenise(&t.text)).collect();

    for (i, tokens) in turn_tokens.iter().enumerate() {
        for a in 0..tokens.len() {
            for b in (a + 1)..tokens.len().min(a + 8) {
                if tokens[a] == tokens[b] {
                    continue;
                }
                let key = if tokens[a] < tokens[b] {
                    (tokens[a].clone(), tokens[b].clone())
                } else {
                    (tokens[b].clone(), tokens[a].clone())
                };
                *cooccur.entry(key).or_insert(0) += 3;
            }
        }
        if i + 1 < turn_tokens.len() {
            for ta in tokens.iter().take(5) {
                for tb in turn_tokens[i + 1].iter().take(5) {
                    if ta == tb {
                        continue;
                    }
                    let key = if ta < tb {
                        (ta.clone(), tb.clone())
                    } else {
                        (tb.clone(), ta.clone())
                    };
                    *cooccur.entry(key).or_insert(0) += 1;
                }
            }
        }
    }

    let mut weighted_clusters: std::collections::HashMap<String, Vec<(u32, String)>> =
        std::collections::HashMap::new();
    for ((a, b), weight) in &cooccur {
        if *weight < 2 {
            continue;
        }
        weighted_clusters
            .entry(a.clone())
            .or_default()
            .push((*weight, b.clone()));
        weighted_clusters
            .entry(b.clone())
            .or_default()
            .push((*weight, a.clone()));
    }
    let mut clusters: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    for (term, mut weighted_neighbors) in weighted_clusters {
        weighted_neighbors.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        weighted_neighbors.dedup_by(|a, b| a.1 == b.1);
        let neighbors: Vec<String> = weighted_neighbors
            .into_iter()
            .take(10)
            .map(|(_, n)| n)
            .collect();
        if !neighbors.is_empty() {
            clusters.insert(term, neighbors);
        }
    }

    let cortyx_dir = project_root.join(".cortyx");
    let out_path = cortyx_dir.join("cooccurrence.json");
    if let Ok(json) = serde_json::to_string(&clusters) {
        match std::fs::create_dir_all(&cortyx_dir) {
            Ok(()) => match std::fs::write(&out_path, json) {
                Ok(()) => tracing::debug!(
                    terms = clusters.len(),
                    path = %out_path.display(),
                    "R17 Sol2: co-occurrence ontology saved"
                ),
                Err(e) => tracing::warn!(
                    path = %out_path.display(),
                    "Failed to write co-occurrence ontology: {e}"
                ),
            },
            Err(e) => tracing::warn!(
                path = %cortyx_dir.display(),
                "Failed to create co-occurrence directory: {e}"
            ),
        }
    }
}
