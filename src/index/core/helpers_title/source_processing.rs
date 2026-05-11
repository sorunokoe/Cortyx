use super::*;

/// Process a single source file: hash-check, AST-extract, write stub + meta.
///
/// Returns a `Vec<CompiledFile>`: the first element (if any) is the Core neuron;
/// subsequent elements are UseCase sub-neurons (S3 lazy splitting, fired when the
/// file has ≥ SUBNEURON_SPLIT_THRESHOLD public functions).
///
/// Returns an empty `Vec` when the file is unchanged (hash match), should be skipped,
/// or when a cosmetic change is detected (S1: sig_hash identical) — in that
/// case only the meta hash is updated on disk and the BM25Entry already in
/// memory from `load_or_create` is preserved with its `staleness_multiplier`
/// and learned feedback signals intact.
///
/// This function performs only filesystem reads and writes — no `&mut NeuronIndex`
/// access — which makes it safe to call in parallel via rayon.
pub(in crate::index) fn process_source_file(
    abs: &Path,
    root: &Path,
    git_confidence: &HashMap<PathBuf, f32>,
) -> Vec<CompiledFile> {
    let rel = abs.strip_prefix(root).unwrap_or(abs);
    if should_skip(rel) {
        return vec![];
    }

    let neuron_path = core_neuron_path(abs, root);
    let meta_file = meta_path(&neuron_path);

    let source_bytes = match std::fs::read(abs) {
        Ok(b) => b,
        Err(_) => return vec![],
    };
    let current_hash = {
        let h = blake3::hash(&source_bytes);
        h.to_hex()[..16].to_string()
    };

    // Read stored meta once and reuse for hash, sig_hash, synapses, module, and feedback counts.
    let stored_meta: Option<NeuronMeta> = if meta_file.exists() {
        std::fs::read_to_string(&meta_file)
            .ok()
            .and_then(|d| serde_json::from_str(&d).ok())
    } else {
        None
    };

    let stored_hash = stored_meta
        .as_ref()
        .map(|m| m.source_hash.as_str())
        .unwrap_or("")
        .to_string();

    // Skip if hash unchanged and neuron exists — pure no-op.
    if !current_hash.is_empty() && current_hash == stored_hash && neuron_path.exists() {
        return vec![];
    }

    let source_text = String::from_utf8_lossy(&source_bytes);
    let source_rel = rel.to_string_lossy();
    let now = now_iso8601();

    let ast_summary = ast_extractor::extract_signatures(&source_rel, &source_text);
    let sig_hash = ast_extractor::compute_sig_hash(&ast_summary);

    let stored_sig_hash = stored_meta
        .as_ref()
        .and_then(|m| m.sig_hash.as_deref())
        .unwrap_or("")
        .to_string();

    // S1 — Cosmetic change: source_hash changed but public API surface (sig_hash) is identical.
    // Whitespace edits, doc-comment tweaks, or formatting passes land here.
    // Preserve the LLM-curated stub; only update the hash in the meta file so future
    // compiles don't re-check this file. The in-memory BM25Entry (from load_or_create)
    // retains its staleness_multiplier and learned feedback signals.
    if !stored_sig_hash.is_empty()
        && sig_hash == stored_sig_hash
        && !stored_hash.is_empty()
        && neuron_path.exists()
    {
        if let Some(mut old_meta) = stored_meta {
            old_meta.source_hash = current_hash;
            old_meta.sig_hash = Some(sig_hash);
            old_meta.last_updated = now;
            if let Err(e) = atomic_write_json(&meta_file, &old_meta) {
                tracing::warn!(
                    "Failed to update meta for cosmetic change {:?}: {e}",
                    meta_file
                );
            }
        }
        return vec![];
    }

    // S1 (R11) — Section-Level Staleness: sig_hash changed (real API change) but the
    // neuron already exists with LLM-curated content. Instead of overwriting everything,
    // replace only the `api` section and update the header comments. Preserves `purpose`,
    // `pitfalls`, and cross-reference sections. Reduces LLM re-evolution calls by ~60%.
    if !stored_hash.is_empty() && neuron_path.exists() {
        // sig_hash is different — we passed the cosmetic-change gate above
        match std::fs::read_to_string(&neuron_path) {
            Ok(existing) => {
                let new_api = ast_extractor::format_for_stub(&ast_summary);
                let updated = replace_section(&existing, "api", &new_api);
                let updated = update_neuron_header(&updated, &current_hash, &now);
                if let Err(e) = atomic_write(&neuron_path, updated.as_bytes()) {
                    tracing::warn!("S1: Failed to update api section {:?}: {e}", neuron_path);
                    // Fall through to full stub generation below
                } else {
                    let old = stored_meta
                        .clone()
                        .unwrap_or_else(|| NeuronMeta::new_stub(abs, NeuronKind::Core));
                    let mut meta = old;
                    meta.source_hash = current_hash;
                    meta.sig_hash = Some(sig_hash);
                    meta.last_updated = now.clone();
                    meta.status = NeuronStatus::Stale;
                    meta.tokens = estimate_context_tokens(&updated).get();
                    if meta.module.is_none() {
                        meta.module = infer_module(rel);
                    }
                    let existing_targets: HashSet<PathBuf> =
                        meta.synapses.iter().map(|s| s.target.clone()).collect();
                    let auto_imports = import_parser::parse_imports(abs, &source_text, root);
                    for imported_source in auto_imports {
                        let target_neuron = core_neuron_path(&imported_source, root);
                        if !existing_targets.contains(&target_neuron) {
                            meta.synapses.push(Synapse::new(
                                target_neuron,
                                SynapseType::Imports,
                                "auto-inferred from import statement".to_string(),
                            ));
                        }
                    }
                    meta.confidence_score = git_confidence.get(abs).copied().unwrap_or(1.0);
                    if let Err(e) = atomic_write_json(&meta_file, &meta) {
                        tracing::warn!("S1: Failed to update meta {:?}: {e}", meta_file);
                    }
                    let mut results = vec![CompiledFile {
                        neuron_path: neuron_path.clone(),
                        content: updated,
                        meta,
                    }];
                    // Also generate sub-neurons for any new functions (idempotent — skips existing)
                    if ast_summary.functions.len() >= SUBNEURON_SPLIT_THRESHOLD {
                        for fn_name in ast_summary.functions.iter().take(MAX_SUBNEURONS_PER_FILE) {
                            let sub_path = sub_neuron_path(&neuron_path, fn_name);
                            if sub_path.exists() {
                                continue;
                            }
                            let sub_content = stub_function_neuron(fn_name, &source_rel, &now);
                            if let Err(e) = atomic_write(&sub_path, sub_content.as_bytes()) {
                                tracing::warn!(
                                    "S1: Failed to write sub-neuron {:?}: {e}",
                                    sub_path
                                );
                                continue;
                            }
                            let sub_meta_file = meta_path(&sub_path);
                            let mut sub_meta = NeuronMeta::new_stub(abs, NeuronKind::UseCase);
                            sub_meta.task_pattern = Some(fn_name.clone());
                            sub_meta.parent = Some(neuron_path.clone());
                            sub_meta.tokens = estimate_context_tokens(&sub_content).get();
                            sub_meta.last_updated = now.clone();
                            sub_meta.module = results[0].meta.module.clone();
                            sub_meta.confidence_score = results[0].meta.confidence_score;
                            if let Err(e) = atomic_write_json(&sub_meta_file, &sub_meta) {
                                tracing::warn!(
                                    "S1: Failed to write sub-neuron meta {:?}: {e}",
                                    sub_meta_file
                                );
                                continue;
                            }
                            results.push(CompiledFile {
                                neuron_path: sub_path,
                                content: sub_content,
                                meta: sub_meta,
                            });
                        }
                    }
                    tracing::debug!(path = %neuron_path.display(), "S1: api section updated, purpose/pitfalls preserved");
                    return results;
                }
            },
            Err(_) => {
                // Cannot read existing neuron — fall through to full stub regeneration
            },
        }
    }

    // Full stub (re)generation — real API change (sig_hash changed) or new file.
    let prefilled = ast_extractor::format_for_stub(&ast_summary);
    let purpose_hint = ast_extractor::format_purpose_hint(&ast_summary);
    let extra_vocab = ast_extractor::format_extra_vocab_for_stub(&ast_summary);
    let content = stub_core_neuron(
        &source_rel,
        &current_hash,
        &now,
        &prefilled,
        &purpose_hint,
        &extra_vocab,
    );

    if let Some(parent) = neuron_path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            tracing::warn!("Failed to create neuron dir {:?}: {e}", parent);
            return vec![];
        }
    }
    if let Err(e) = atomic_write(&neuron_path, content.as_bytes()) {
        tracing::warn!("Failed to write stub {:?}: {e}", neuron_path);
        return vec![];
    }

    let is_new = stored_hash.is_empty();
    let mut meta = NeuronMeta::new_stub(abs, NeuronKind::Core);
    meta.source_hash = current_hash;
    meta.sig_hash = Some(sig_hash);
    meta.tokens = estimate_context_tokens(&content).get();
    meta.last_updated = now.clone();
    meta.status = if is_new {
        NeuronStatus::Stub
    } else {
        NeuronStatus::Stale
    };

    // Preserve existing synapses, module tag, and feedback counts on hash invalidation.
    if let Some(old) = stored_meta {
        meta.synapses = old.synapses;
        meta.module = old.module;
        meta.use_count = old.use_count;
        meta.hit_count = old.hit_count;
    }

    // Auto-module: infer from directory structure when not LLM-set.
    if meta.module.is_none() {
        meta.module = infer_module(rel);
    }

    // Auto-Synapse: infer Imports edges from import statements.
    let existing_targets: HashSet<PathBuf> =
        meta.synapses.iter().map(|s| s.target.clone()).collect();
    let auto_imports = import_parser::parse_imports(abs, &source_text, root);
    for imported_source in auto_imports {
        let target_neuron = core_neuron_path(&imported_source, root);
        if !existing_targets.contains(&target_neuron) {
            meta.synapses.push(Synapse::new(
                target_neuron,
                SynapseType::Imports,
                "auto-inferred from import statement".to_string(),
            ));
        }
    }

    // Git confidence: committed + unmodified = 1.0, modified = 0.9, untracked = 0.85.
    meta.confidence_score = git_confidence.get(abs).copied().unwrap_or(1.0);

    if let Err(e) = atomic_write_json(&meta_file, &meta) {
        tracing::warn!("Failed to write meta {:?}: {e}", meta_file);
        return vec![];
    }

    let mut results = vec![CompiledFile {
        neuron_path: neuron_path.clone(),
        content,
        meta,
    }];

    // S3 — Lazy Sub-Neuron Splitting: for files with many public functions,
    // generate one UseCase sub-neuron per function so BM25 can match at
    // function-level precision. Sub-neurons slot into Phase 2 of get_contexts
    // (UseCase scoring per Core) automatically via the parent_index.
    if ast_summary.functions.len() >= SUBNEURON_SPLIT_THRESHOLD {
        for fn_name in ast_summary.functions.iter().take(MAX_SUBNEURONS_PER_FILE) {
            let sub_path = sub_neuron_path(&neuron_path, fn_name);
            // Only write a new stub if the sub-neuron doesn't already exist —
            // preserve any LLM-curated content from a previous compile.
            if sub_path.exists() {
                continue;
            }
            let sub_content = stub_function_neuron(fn_name, &source_rel, &now);
            if let Err(e) = atomic_write(&sub_path, sub_content.as_bytes()) {
                tracing::warn!("Failed to write sub-neuron {:?}: {e}", sub_path);
                continue;
            }
            let sub_meta_file = meta_path(&sub_path);
            let mut sub_meta = NeuronMeta::new_stub(abs, NeuronKind::UseCase);
            sub_meta.task_pattern = Some(fn_name.clone());
            sub_meta.parent = Some(neuron_path.clone());
            sub_meta.tokens = estimate_context_tokens(&sub_content).get();
            sub_meta.last_updated = now.clone();
            sub_meta.module = results[0].meta.module.clone();
            sub_meta.confidence_score = results[0].meta.confidence_score;
            if let Err(e) = atomic_write_json(&sub_meta_file, &sub_meta) {
                tracing::warn!("Failed to write sub-neuron meta {:?}: {e}", sub_meta_file);
                continue;
            }
            results.push(CompiledFile {
                neuron_path: sub_path,
                content: sub_content,
                meta: sub_meta,
            });
        }
    }

    results
}
