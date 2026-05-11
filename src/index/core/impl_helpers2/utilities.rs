use super::*;

impl NeuronIndex {

    /// S-XI (R16): Detect renamed/moved source files and carry over accumulated signal.
    ///
    /// After a full compile, scans for neurons whose source file no longer exists.
    /// For each such "orphaned" neuron, checks whether any newly-indexed neuron has
    /// a matching BLAKE3 content hash (from the old neuron file). If so, transfers
    /// use_count, hit_count, learned synapse weights, and UUID to the new entry.
    ///
    /// This makes rename-refactoring non-destructive: LLM quality feedback and graph
    /// weights survive `git mv` or manual renames.
    pub(in crate::index) fn apply_rename_detection(&mut self, root: &Path) {
        let ndir = neuron_dir(root);

        // Build: old_neuron_hash → (old_entry_index, meta) for neurons whose SOURCE is gone
        let mut orphaned: Vec<(String, usize)> = Vec::new(); // (neuron_content_hash, entry_idx)
        for (i, entry) in self.entries.iter().enumerate() {
            let source = &entry.source_files.first().cloned();
            let gone = source.as_ref().is_some_and(|s| !s.exists());
            if !gone {
                continue;
            }
            // Hash the neuron file itself (the .context.md) to match against new file
            if let Ok(bytes) = std::fs::read(&entry.neuron_path) {
                let h = blake3::hash(&bytes).to_hex()[..16].to_string();
                orphaned.push((h, i));
            }
        }

        if orphaned.is_empty() {
            return;
        }

        // Build: neuron_content_hash → new_entry_index for all current neurons
        let mut hash_to_new: HashMap<String, usize> = HashMap::new();
        for (i, entry) in self.entries.iter().enumerate() {
            if let Ok(bytes) = std::fs::read(&entry.neuron_path) {
                let h = blake3::hash(&bytes).to_hex()[..16].to_string();
                hash_to_new.insert(h, i);
            }
        }

        // Carry over signal from orphaned → matched new entry
        let mut transfers = 0usize;
        for (old_hash, old_idx) in &orphaned {
            if let Some(&new_idx) = hash_to_new.get(old_hash.as_str()) {
                if old_idx == &new_idx {
                    continue;
                } // same entry, skip
                  // Transfer accumulated signal (requires split borrow)
                let (use_count, hit_count, synapses) = {
                    let old = &self.entries[*old_idx];
                    (old.use_count, old.hit_count, old.synapses.clone())
                };
                {
                    let new_entry = &mut self.entries[new_idx];
                    // Only carry over if the new entry hasn't yet accumulated its own signal
                    if new_entry.use_count == 0 {
                        new_entry.use_count = use_count;
                        new_entry.hit_count = hit_count;
                        // Only merge synapses that don't already exist
                        for syn in synapses {
                            if !new_entry.synapses.iter().any(|s| s.target == syn.target) {
                                new_entry.synapses.push(syn);
                            }
                        }
                        transfers += 1;
                        tracing::info!(
                            "S-XI: transferred signal from orphaned entry[{}] → entry[{}] (rename detected)",
                            old_idx, new_idx
                        );
                    }
                }
                // Also update sidecar UUID: load new meta, set UUID from old meta if available
                let old_neuron_path = self.entries[*old_idx].neuron_path.clone();
                let new_neuron_path = self.entries[new_idx].neuron_path.clone();
                let old_meta_path = meta_path(&old_neuron_path);
                let new_meta_path = meta_path(&new_neuron_path);
                if old_meta_path.exists() && new_meta_path.exists() {
                    if let Ok(old_meta_str) = std::fs::read_to_string(&old_meta_path) {
                        if let Ok(old_meta) = serde_json::from_str::<NeuronMeta>(&old_meta_str) {
                            if let Some(old_uuid) = &old_meta.uuid {
                                if let Ok(new_meta_str) = std::fs::read_to_string(&new_meta_path) {
                                    if let Ok(mut new_meta) =
                                        serde_json::from_str::<NeuronMeta>(&new_meta_str)
                                    {
                                        if new_meta.uuid.is_none() {
                                            new_meta.uuid = Some(old_uuid.clone());
                                            if let Err(e) =
                                                atomic_write_json(&new_meta_path, &new_meta)
                                            {
                                                tracing::warn!(
                                                    "Failed to persist renamed neuron UUID for {}: {e}",
                                                    new_meta_path.display()
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                // Remove orphaned entry from sidecar (if it exists) so it doesn't re-appear
                let orphan_meta = ndir.join(
                    old_neuron_path
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .as_ref()
                        .replace(".context.md", ".context.json"),
                );
                if let Err(e) = std::fs::remove_file(&orphan_meta) {
                    if e.kind() != std::io::ErrorKind::NotFound {
                        tracing::warn!(
                            "Failed to remove orphaned renamed sidecar {}: {e}",
                            orphan_meta.display()
                        );
                    }
                }
            }
        }

        if transfers > 0 {
            tracing::info!("S-XI: rename detection transferred signal for {transfers} neuron(s)");
        }
    }

    pub(in crate::index) fn write_synthetic_answer(
        &self,
        slug: &str,
        task: &str,
        answer: &str,
        evidence: &[String],
    ) -> Option<PathBuf> {
        let path = neuron_dir(&self.project_root).join(format!("_answer_{slug}.md"));
        let mut content = format!("# Derived answer\n\nQuestion: {task}\nAnswer: {answer}\n");
        if !evidence.is_empty() {
            content.push_str("\n## evidence\n");
            for line in evidence.iter().take(3) {
                content.push_str("- ");
                content.push_str(line.trim());
                content.push('\n');
            }
        }
        atomic_write(&path, content.as_bytes()).ok()?;
        Some(path)
    }


    /// Trim a sorted list of paths to fit within `max_tokens`.
    pub(in crate::index) fn trim_to_token_budget(
        &self,
        paths: Vec<PathBuf>,
        max_tokens: usize,
    ) -> Vec<PathBuf> {
        let mut result = Vec::new();
        let mut used = 0usize;
        for path in paths {
            let tokens = self.entry_by_path(&path).map(|e| e.tokens).unwrap_or(200);
            if used + tokens <= max_tokens || result.is_empty() {
                used += tokens;
                result.push(path);
            }
        }
        result
    }


    /// Auto-create the Project neuron if it doesn't exist yet.
    pub(in crate::index) fn ensure_project_neuron(&mut self, root: &Path) -> Result<()> {
        let project_name = root
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "project".to_string());
        let project_neuron = neuron_dir(root).join("_project.context.md");

        if !project_neuron.exists() {
            let now = now_iso8601();
            let content = stub_project_neuron(&project_name, &now);
            atomic_write(&project_neuron, content.as_bytes())?;
            let mut meta = NeuronMeta::new_stub(root, NeuronKind::Project);
            meta.tokens = estimate_context_tokens(&content).get();
            meta.last_updated = now;
            atomic_write_json(&meta_path(&project_neuron), &meta)?;
            self.index_neuron(&project_neuron, &content, &meta);
        }
        Ok(())
    }


    /// S5 (R15 NE4): Generate wake-up context neurons at compile time.
    ///
    /// Creates two Concept neurons from project metadata:
    /// - `_identity.context.md` (~50 tok): project name, version, authors, repo URL, description
    /// - `_critical_facts.context.md` (~120 tok): conventions, architecture highlights, key decisions
    ///
    /// Both are standard Concept neurons — BM25-indexed, evolvable, git-tracked.
    /// They are only loaded when `cortyx_wake_up` is explicitly called (P16 Partial Action —
    /// preserves Cortyx's token efficiency advantage; zero overhead when not requested).
    ///
    /// Sources: `git config`, `Cargo.toml`/`package.json`, README first 500 chars,
    /// `CONTRIBUTING.md`/`AGENTS.md` if present (first 400 chars each).
    pub(in crate::index) fn ensure_wake_up_neurons(
        &mut self,
        root: &Path,
        ndir: &Path,
    ) -> Result<()> {
        let identity_path = ndir.join("_identity.context.md");
        let critical_path = ndir.join("_critical_facts.context.md");

        // Gather project metadata from manifest files.
        let project_name = root
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "project".to_string());

        let (pkg_name, pkg_version, pkg_authors, pkg_description, pkg_repo) =
            extract_manifest_metadata(root);

        let name = if !pkg_name.is_empty() {
            pkg_name
        } else {
            project_name.clone()
        };

        // _identity.context.md — generate if absent
        if !identity_path.exists() {
            let git_author = run_git_cmd(root, &["config", "user.name"]).unwrap_or_default();
            let git_email = run_git_cmd(root, &["config", "user.email"]).unwrap_or_default();

            let readme_intro =
                read_file_head(root, &["README.md", "README.rst", "README.txt"], 300);

            let content = format!(
                "# Identity: {name}\n\n\
                 ## purpose\n\
                 Project identity card — loaded via `cortyx_wake_up` to prime LLM session context.\n\n\
                 ## api\n\
                 | Field | Value |\n\
                 |---|---|\n\
                 | Project | {name} |\n\
                 | Version | {pkg_version} |\n\
                 | Authors | {authors} |\n\
                 | Repository | {pkg_repo} |\n\
                 | Description | {pkg_description} |\n\
                 | Git author | {git_author} <{git_email}> |\n\n\
                 ## context\n\
                 {readme_intro}\n\n\
                 ## pitfalls\n\
                 _Evolve this section with key project conventions and gotchas._\n",
                authors = if !pkg_authors.is_empty() { pkg_authors.clone() } else { git_author.clone() },
            );
            atomic_write(&identity_path, content.as_bytes())?;
            let mut meta = NeuronMeta::new_stub(root, NeuronKind::Concept);
            meta.tokens = estimate_context_tokens(&content).get();
            meta.module = Some("@wake_up".to_string());
            meta.last_updated = now_iso8601();
            atomic_write_json(&meta_path(&identity_path), &meta)?;
            self.index_neuron(&identity_path, &content, &meta);
            tracing::info!("S5: generated _identity.context.md for '{name}'");
        }

        // _critical_facts.context.md — generate if absent
        if !critical_path.exists() {
            let contributing = read_file_head(
                root,
                &["CONTRIBUTING.md", "AGENTS.md", "CONTRIBUTING.rst"],
                400,
            );
            let conventions = if !contributing.is_empty() {
                contributing
            } else {
                // Fallback: extract a "conventions" or "architecture" section from README
                read_readme_section(root, &["convention", "architecture", "structure", "design"])
                    .unwrap_or_else(|| "_Evolve with team conventions, coding standards, and architectural decisions._".to_string())
            };

            let content = format!(
                "# Critical Facts: {name}\n\n\
                 ## purpose\n\
                 Key conventions, architecture decisions, and team context — loaded via \
                 `cortyx_wake_up` for session priming.\n\n\
                 ## api\n\
                 | Field | Value |\n\
                 |---|---|\n\
                 | Stack | {name} v{pkg_version} |\n\
                 | Repo | {pkg_repo} |\n\n\
                 ## context\n\
                 {conventions}\n\n\
                 ## pitfalls\n\
                 _Evolve this section after each sprint retro or architectural change._\n",
            );
            atomic_write(&critical_path, content.as_bytes())?;
            let mut meta = NeuronMeta::new_stub(root, NeuronKind::Concept);
            meta.tokens = estimate_context_tokens(&content).get();
            meta.module = Some("@wake_up".to_string());
            meta.last_updated = now_iso8601();
            atomic_write_json(&meta_path(&critical_path), &meta)?;
            self.index_neuron(&critical_path, &content, &meta);
            tracing::info!("S5: generated _critical_facts.context.md for '{name}'");
        }

        Ok(())
    }
}
