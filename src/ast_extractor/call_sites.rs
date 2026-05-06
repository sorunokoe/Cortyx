/// A detected call-site edge: this file calls `callee_fn` which is defined in `callee_file`.
#[derive(Debug, Clone)]
pub struct CallEdge {
    /// Relative path of the file that *defines* the callee function.
    pub callee_file: std::path::PathBuf,
    /// Name of the called function (for logging / debugging).
    #[allow(dead_code)]
    pub callee_fn: String,
}

/// Scan `content` (a source file at `source_rel`) for calls to public functions
/// defined in *other* files of the project.
///
/// `vocab` maps `function_name → relative_source_path` built from all neurons'
/// `AstSummary.functions` during the compile pass.  Only calls to functions that
/// appear in the vocabulary are emitted — external / stdlib calls are ignored.
///
/// One `CallEdge` per unique callee file is returned (duplicates collapsed).
pub fn extract_call_sites(
    source_rel: &str,
    content: &str,
    vocab: &std::collections::HashMap<String, std::path::PathBuf>,
) -> Vec<CallEdge> {
    if vocab.is_empty() {
        return vec![];
    }

    let self_fns: std::collections::HashSet<String> = {
        let summary = super::extract_signatures(source_rel, content);
        summary.functions.into_iter().collect()
    };

    let mut seen_files: std::collections::HashSet<std::path::PathBuf> =
        std::collections::HashSet::new();
    let mut edges: Vec<CallEdge> = Vec::new();

    for (fn_name, callee_path) in vocab {
        if self_fns.contains(fn_name) {
            continue;
        }
        if callee_path == std::path::Path::new(source_rel) {
            continue;
        }
        if seen_files.contains(callee_path.as_path()) {
            continue;
        }
        let needle = format!("{fn_name}(");
        if content.contains(&needle) {
            seen_files.insert(callee_path.clone());
            edges.push(CallEdge {
                callee_file: callee_path.clone(),
                callee_fn: fn_name.clone(),
            });
        }
    }

    edges
}
