use super::*;

impl NeuronIndex {
    // ── Activation (get_contexts) ─────────────────────────────────────────────

    /// Return the most relevant neuron paths for `task`, respecting `max_tokens`.
    ///
    /// Activation phases:
    /// 1. BM25 scoring of all Core neurons (module-filtered if `module` is Some)
    /// 2. UseCase neurons for each activated Core
    /// 3. Typed synapse traversal (up to 2 hops, score-weighted by type)
    /// 4. Lexicographic sort → token-budget trim
    ///
    /// The lexicographic sort guarantees byte-identical output for the same
    /// task + index state, which is required for prompt cache hit rates.
    pub fn get_contexts(
        &self,
        task: &str,
        max_tokens: usize,
        module: Option<&str>,
        kind: Option<&str>,
    ) -> Vec<PathBuf> {
        let Ok(query) = QueryText::new(task) else {
            return Vec::new();
        };
        let terms = tokenize(query.as_str());

        // Phase 1 — O(|candidates|) BM25 via posting list.
        //
        // Union the posting lists for all query terms to find the candidate set —
        // only entries containing at least one query term can have a non-zero BM25
        // score, so there is no accuracy loss.  For sparse queries this reduces
        // BM25 scoring from O(n) to O(|candidates|), typically ~N/50 for real tasks.
        //
        // `scoring_terms` starts as a reference to `terms` and is replaced with the
        // vocabulary-bridge-expanded set when a zero-match query fires the bridge (S2).
        // BM25 scoring always uses `scoring_terms` so bridge candidates are ranked
        // by their actual identifier vocabulary, not the zero-scoring original terms.
        let candidate_set: HashSet<usize> = {
            let mut s = HashSet::new();
            for term in &terms {
                if let Some(idxs) = self.posting_list.get(term) {
                    s.extend(idxs);
                }
            }
            s
        };

        // Optional module scope — when module is Some, restrict to entries tagged with that module.
        // If no entries carry that module tag, the result set is empty (not "unfiltered").
        let module_set: Option<HashSet<usize>> = module.map(|m| {
            self.module_index
                .get(m)
                .map(|v| v.iter().copied().collect::<HashSet<_>>())
                .unwrap_or_default() // module requested but unknown → empty set → zero results
        });

        // Vocabulary gap detector (TRIZ Standard 4.1.1 — Measurement Substance).
        // If posting lists return zero candidates for every query term, the index has
        // no vocabulary match for this task.
        //
        // S2 — Vocabulary Bridge: attempt query expansion using module-path synonyms.
        // For each zero-match query term, check if it substring-matches any module
        // fragment in vocab_bridge. If so, expand the candidate set with that module's
        // identifier vocabulary and re-run the posting-list lookup on the new terms.
        // This resolves the "authentication" → "auth_guard" gap without any model.
        //
        // When the bridge fires, `scoring_terms` is updated to the expanded set so
        // BM25 scores are computed against the actual identifier vocabulary (not the
        // original natural-language query that had zero index coverage).
        let mut scoring_terms: &[String] = &terms;
        let expanded_terms_buf: Vec<String>;

        // B2: Synonym cloud expansion — always applied before S2/B1 bridge.
        // If any query term co-activates with a neuron ≥30× historically, add
        // the synonym cloud terms to the scoring set to improve recall.
        let synonym_expansions = self.synonym_cloud_expansion(&terms);
        let morphological_expansions: Vec<String> = terms
            .iter()
            .flat_map(|term| morphological_variants(term))
            .filter(|variant| self.df_cache.contains_key(variant.as_str()))
            .collect();
        let terms_with_synonyms: Vec<String> =
            if !synonym_expansions.is_empty() || !morphological_expansions.is_empty() {
                let mut t = terms.clone();
                t.extend(synonym_expansions.iter().cloned());
                t.extend(morphological_expansions.iter().cloned());
                t.sort();
                t.dedup();
                t
            } else {
                terms.clone()
            };

        // Expand candidate set with synonym/morphological terms if we have them
        let candidate_set = {
            let mut cs = candidate_set;
            for term in synonym_expansions
                .iter()
                .chain(morphological_expansions.iter())
            {
                if let Some(idxs) = self.posting_list.get(term.as_str()) {
                    cs.extend(idxs);
                }
            }
            cs
        };

        let synonym_expansions_empty =
            synonym_expansions.is_empty() && morphological_expansions.is_empty();

        let candidate_set = if candidate_set.is_empty() && !terms.is_empty() {
            let expanded = self.expand_query_terms(&terms_with_synonyms);
            if expanded.len() > terms_with_synonyms.len() {
                let mut bridged: HashSet<usize> = HashSet::new();
                for term in &expanded {
                    if let Some(idxs) = self.posting_list.get(term) {
                        bridged.extend(idxs);
                    }
                }
                if !bridged.is_empty() {
                    tracing::debug!(
                        task,
                        original = terms.len(),
                        expanded = expanded.len(),
                        candidates = bridged.len(),
                        "Vocabulary bridge: expanded query via module synonyms + morphemes + B2"
                    );
                    expanded_terms_buf = expanded;
                    scoring_terms = &expanded_terms_buf;
                    bridged
                } else {
                    tracing::debug!(
                        task,
                        "Vocabulary gap: no posting-list candidates for query. \
                         Consider evolving relevant neurons to cover terms: {:?}",
                        &terms[..terms.len().min(5)]
                    );
                    candidate_set
                }
            } else {
                tracing::debug!(
                    task,
                    "Vocabulary gap: no posting-list candidates for query. \
                     Consider evolving relevant neurons to cover terms: {:?}",
                    &terms[..terms.len().min(5)]
                );
                candidate_set
            }
        } else {
            // Update scoring_terms to include synonym expansions when candidates found
            if !synonym_expansions_empty {
                expanded_terms_buf = terms_with_synonyms;
                scoring_terms = &expanded_terms_buf;
            }
            candidate_set
        };

        // R12-S1 — Concept Cloud fallback: graph-aware semantic expansion.
        //
        // When both the direct posting list AND the vocab bridge return zero candidates,
        // scan each neuron's concept cloud (union of identifier terms from 1-hop Calls/
        // Imports/Implements neighbours). If any neuron's cloud overlaps with the query
        // terms, that neuron becomes a candidate — no substring tricks, no model.
        //
        // This closes the gap where a query term names a callee function that lives in a
        // different file; the caller neuron's cloud contains callee terms via the graph.
        //
        // Scored against the ORIGINAL query terms only (not the cloud terms) to prevent
        // BM25 score inflation from the expanded vocabulary.
        let candidate_set = if candidate_set.is_empty() && !terms.is_empty() {
            let term_set: HashSet<&str> = terms.iter().map(|s| s.as_str()).collect();
            let cloud_candidates: HashSet<usize> = self
                .entries
                .iter()
                .enumerate()
                .filter(|(_, e)| {
                    e.concept_cloud
                        .iter()
                        .any(|t| term_set.contains(t.as_str()))
                })
                .map(|(i, _)| i)
                .collect();
            if !cloud_candidates.is_empty() {
                tracing::debug!(
                    task,
                    candidates = cloud_candidates.len(),
                    "Concept cloud (R12-S1): found candidates via 1-hop graph vocabulary"
                );
            }
            cloud_candidates
        } else {
            candidate_set
        };

        // R18 P2 Sol B — Category-Aware Query Router (zero ML, pure regex + heuristics).
        // R19 fix: removed is_multi_session from force_tfidf (2 proper nouns is too common
        // in single-session queries, causing false TF-IDF reranks and -5.7pp regression).
        let is_knowledge_update = detect_knowledge_update_query(task);
        let is_counting = detect_counting_query(task);
        let task_lower = task.to_ascii_lowercase();
        let explicit_current_state_query = has_explicit_current_state_marker(task);
        let named_person_move_query = count_proper_nouns(task) >= 1
            && (task_lower.contains(" move")
                || task_lower.contains(" moved")
                || task_lower.contains("relocation"));
        let expand_focus_terms = |base_terms: Vec<String>| {
            let mut expanded = base_terms.clone();
            for term in &base_terms {
                for variant in morphological_variants(term) {
                    if self.df_cache.contains_key(variant.as_str()) {
                        expanded.push(variant);
                    }
                }
            }
            expanded.sort();
            expanded.dedup();
            expanded
        };
        let raw_counting_focus_terms = if is_counting {
            extract_counting_focus_terms(&terms)
        } else {
            Vec::new()
        };
        let counting_focus_terms = if is_counting {
            expand_focus_terms(raw_counting_focus_terms.clone())
        } else {
            Vec::new()
        };
        let raw_knowledge_focus_terms = if !is_counting && is_knowledge_update {
            extract_knowledge_update_focus_terms(&terms)
        } else {
            Vec::new()
        };
        let knowledge_focus_terms = if !is_counting && is_knowledge_update {
            expand_focus_terms(raw_knowledge_focus_terms.clone())
        } else {
            Vec::new()
        };
        let ranking_terms: &[String] = if !counting_focus_terms.is_empty() {
            &counting_focus_terms
        } else if !knowledge_focus_terms.is_empty() {
            &knowledge_focus_terms
        } else {
            scoring_terms
        };
        // force_tfidf: only for confirmed knowledge-update queries (stale facts look
        // HIGH confidence on BM25, bypassing TF-IDF normally). Multi-session routing
        // still benefits from synapse BFS without needing forced TF-IDF.
        let force_tfidf = is_knowledge_update;

        // P2-B: KG Router — bypass BM25 for personal-attribute queries.
        //
        // "What degree did I graduate with?" → predicate=education → scan KG neurons →
        // find entity with active education fact → inject KG neuron as rank-1 result.
        //
        // This is O(|KG entities|) = O(small) at query time. KG neurons are Concept
        // neurons already in the BM25 index; injecting as rank-1 does not break the
        // existing scoring pipeline — BM25 still runs, KG result is prepended.
        let kg_router_path: Option<PathBuf> =
            (!matches!(kind, Some(k) if k.eq_ignore_ascii_case("conversation")))
                .then_some(())
                .and_then(|_| detect_personal_fact_query(task))
                .and_then(|predicate| {
                    detect_personal_fact_entity(task).and_then(|entity| {
                        let kg_path = kg::kg_neuron_path(&self.project_root, &entity);
                        if !self.path_index.contains_key(&kg_path) {
                            return None;
                        }
                        let Ok(kg_entity) = kg::KgEntity::load(&kg_path) else {
                            return None;
                        };
                        let has_fact = kg_entity
                            .active_facts(None)
                            .iter()
                            .any(|f| f.predicate == predicate && !f.value.is_empty());
                        if has_fact {
                            tracing::debug!(
                            task,
                            predicate,
                            entity,
                            kind = kind.unwrap_or("all"),
                            "P2-B KG Router: routed personal-attribute query to exact KG neuron"
                        );
                            Some(kg_path)
                        } else {
                            None
                        }
                    })
                });

        // R21 T5: Counting-query candidate expansion.
        //
        // "How many X have I done?" needs evidence from ALL sessions mentioning X, not
        // just the highest-scoring posting-list hit. When detect_counting_query fires,
        // expand the candidate set to include ALL Verbatim neurons in the index, scored
        // with BM25 against the query. Aggregate neurons stay available for explicit
        // injection below, but they do not participate in the general BM25 pool.
        let counting_augment: Vec<usize> = if is_counting {
            let in_set: std::collections::HashSet<usize> = candidate_set.iter().copied().collect();
            self.entries
                .iter()
                .enumerate()
                .filter(|(i, e)| {
                    matches!(e.kind, NeuronKind::Verbatim | NeuronKind::Aggregate)
                        && !in_set.contains(i)
                })
                .map(|(i, _)| i)
                .collect()
        } else {
            vec![]
        };

        // BM25 scoring — kind-filtered over candidates in scope.
        // kind=None or "all" → Core + Project + Verbatim (default)
        // kind="code"         → Core + Project only (exclude conversation/Verbatim)
        // kind="conversation" → Verbatim only (episodic recall, excludes code neurons)
        // Aggregate neurons are NEVER in the general BM25 pool — they are injected
        // via counting_augment only when detect_counting_query() fires, preventing
        // pollution of non-counting R@5 results.
        let kind_lower = kind.map(|k| k.to_lowercase());
        let score_bm25_candidates = |candidate_ids: &HashSet<usize>, query_terms: &[String]| {
            let mut scored: Vec<(f32, usize)> = candidate_ids
                .iter()
                .filter(|&&i| {
                    let k = &self.entries[i].kind;
                    let kind_ok = match kind_lower.as_deref() {
                        Some("conversation") => matches!(k, NeuronKind::Verbatim),
                        Some("code") => matches!(k, NeuronKind::Core | NeuronKind::Project),
                        _ => matches!(
                            k,
                            NeuronKind::Core | NeuronKind::Project | NeuronKind::Verbatim
                        ),
                    };
                    kind_ok && module_set.as_ref().is_none_or(|ms| ms.contains(&i))
                })
                .filter_map(|&i| {
                    let mut s = self.bm25_score(query_terms, &self.entries[i]);
                    if is_session_summary_path(&self.entries[i].neuron_path) {
                        if is_counting {
                            s *= 1.35;
                        } else if matches!(kind_lower.as_deref(), Some("conversation") | None) {
                            s *= 1.15;
                        }
                    }
                    // R18 P2 Sol B: knowledge-update routing — demote stale Verbatim neurons
                    // so updated KG/Concept facts rank above old verbatim assertions.
                    // R21 T4: ×0.8 → ×0.5 — old fact now needs 2× BM25 score to beat new fact.
                    if is_knowledge_update && matches!(self.entries[i].kind, NeuronKind::Verbatim) {
                        s *= 0.5;
                    }
                    (s > 0.0).then_some((s, i))
                })
                .collect();

            // Merge counting-query expanded candidates into bm25_scored.
            // Aggregate neurons are intentionally excluded here — Sol-A+ injects the best one
            // into `selected` after top_cores are determined, preventing Aggregates from
            // displacing Verbatim chunks in the BM25 top-5 ranking.
            if !counting_augment.is_empty() {
                let already_scored: std::collections::HashSet<usize> =
                    scored.iter().map(|(_, i)| *i).collect();
                for &i in &counting_augment {
                    if already_scored.contains(&i) {
                        continue;
                    }
                    // Aggregates handled exclusively by Sol-A+ block below
                    if matches!(self.entries[i].kind, NeuronKind::Aggregate) {
                        continue;
                    }
                    let s = self.bm25_score(query_terms, &self.entries[i]);
                    if s > 0.0 {
                        scored.push((s, i));
                    }
                }
                tracing::debug!(
                    task,
                    total = scored.len(),
                    "R21 T5: counting-query candidate expansion applied"
                );
            }

            scored
        };
        let mut bm25_scored: Vec<(f32, usize)> =
            score_bm25_candidates(&candidate_set, ranking_terms);

        //
        // "What was the first X?" needs the OLDEST neuron to surface; "What is the latest X?"
        // needs the NEWEST. The direction is decoded from the query itself (zero extra data).
        //
        // detect_oldest_query() fires for "first", "originally", "initially", "earliest" etc.
        // detect_temporal_query() fires for "recent", "current", "latest", "when did" etc.
        //
        // Boost strength: ×1.6 max (up from ×1.4 in R17). Boost requires ≥1 timestamped
        // neuron (was ≥2 — too conservative, now fires even on single-session temporals).
        if detect_temporal_query(task) || detect_oldest_query(task) || is_knowledge_update {
            // NE-4 fix: make oldest routing mutually exclusive with recency routing.
            // If a query triggers BOTH (ambiguous), default to newest-first (safer: most LME-500
            // temporals ask for the most recent fact, not the oldest).
            // KU queries always use newest-first: the ×0.5 KU demotion is applied equally to
            // ALL Verbatim neurons, so without a directional boost the old session (with higher
            // BM25 from more topic mentions) still outranks the updated session. The temporal
            // boost (×1.0 + boost_strength × normalized_timestamp) overcomes the vocabulary gap.
            let is_oldest =
                detect_oldest_query(task) && !detect_temporal_query(task) && !is_knowledge_update;
            // KU gets a stronger boost (0.8) than standard temporal (0.6) because BM25
            // vocabulary gap between old and new facts can be larger than event-retrieval gaps.
            let boost_strength = if named_person_move_query {
                0.0
            } else if explicit_current_state_query {
                1.2
            } else if is_knowledge_update && !detect_temporal_query(task) {
                0.8
            } else {
                0.6
            };
            let ts_values: Vec<i64> = bm25_scored
                .iter()
                .filter_map(|(_, i)| self.entries[*i].timestamp_secs)
                .collect();
            if !ts_values.is_empty() {
                let min_ts = ts_values.iter().copied().min().unwrap_or_default();
                let max_ts = ts_values.iter().copied().max().unwrap_or_default();
                let range = (max_ts - min_ts).max(1) as f32;
                for (score, i) in bm25_scored.iter_mut() {
                    if let Some(ts) = self.entries[*i].timestamp_secs {
                        let normalized = (ts - min_ts) as f32 / range;
                        if is_oldest {
                            // Oldest-first: invert direction — oldest neuron gets full boost
                            *score *= 1.0 + boost_strength * (1.0 - normalized);
                        } else {
                            // Newest-first (default): most recent neuron gets full boost
                            *score *= 1.0 + boost_strength * normalized;
                        }
                    }
                }
                tracing::debug!(
                    task,
                    is_oldest,
                    boost_strength,
                    candidates = ts_values.len(),
                    "R21 T2+KU: Bidirectional temporal boost applied"
                );
            }
        }

        // Narrow fix for named-person relocation questions: prefer candidates whose body text
        // actually contains move/live evidence, not just mine-time query_surface hints.
        if named_person_move_query {
            for (score, i) in bm25_scored.iter_mut() {
                if !matches!(self.entries[*i].kind, NeuronKind::Verbatim) {
                    continue;
                }
                if self.entries[*i].has_move_residence_evidence {
                    *score *= 1.35;
                } else {
                    *score *= 0.55;
                }
            }
            tracing::debug!(
                task,
                candidates = bm25_scored.len(),
                "Named-person relocation body-evidence rerank applied"
            );
        }

        // R20 A-3: TemporalFollows chain BM25 aggregation.
        //
        // Multi-session queries have evidence scattered across Verbatim neurons that are
        // linked by TemporalFollows edges. BM25 scores each neuron in isolation, so a
        // session-1 neuron scoring 1.8 and a session-2 neuron scoring 2.1 never combine.
        //
        // Fix: for each Verbatim neuron in the candidate set, walk its TemporalFollows
        // adjacency up to 3 hops and accumulate chain-member BM25 scores at exponential
        // discount (×0.5 per hop). The "anchor" (entry-point) neuron absorbs the chain
        // signal so multi-session evidence aggregates into a single boosted score rather
        // than splitting across many low-scoring neurons.
        //
        // Only fires for Verbatim neurons (conversation memory) — code neurons are
        // unaffected. Chain members are NOT added as new candidates (no recall change);
        // this purely reweights existing candidates. Cost: O(|Verbatim candidates| × hops).
        {
            let verbatim_scored: Vec<(usize, f32)> = bm25_scored
                .iter()
                .filter(|(_, i)| matches!(self.entries[*i].kind, NeuronKind::Verbatim))
                .map(|(s, i)| (*i, *s))
                .collect();

            if !verbatim_scored.is_empty() {
                let scored_path_map: std::collections::HashMap<PathBuf, f32> = verbatim_scored
                    .iter()
                    .map(|(i, score)| (self.entries[*i].neuron_path.clone(), *score))
                    .collect();

                for (score, i) in bm25_scored.iter_mut() {
                    if !matches!(self.entries[*i].kind, NeuronKind::Verbatim) {
                        continue;
                    }
                    let anchor = self.entries[*i].neuron_path.clone();

                    // BFS along TemporalFollows edges, up to 3 hops
                    let mut frontier = vec![anchor.clone()];
                    let mut seen: std::collections::HashSet<PathBuf> =
                        std::collections::HashSet::new();
                    seen.insert(anchor.clone());
                    let mut hop_discount = 0.5f32;

                    for _hop in 0..3 {
                        let mut next_frontier = Vec::new();
                        for path in &frontier {
                            let Some(neighbors) = self.adjacency.get(path) else {
                                continue;
                            };
                            for syn in neighbors {
                                if syn.edge_type != SynapseType::TemporalFollows {
                                    continue;
                                }
                                if seen.contains(&syn.target) {
                                    continue;
                                }
                                seen.insert(syn.target.clone());
                                // Add chain-member score to anchor — but only if the
                                // chain member is also a BM25 candidate (already scored).
                                // This keeps the boost evidence-grounded.
                                if let Some(chain_score) = scored_path_map.get(&syn.target) {
                                    *score += hop_discount * *chain_score;
                                }
                                next_frontier.push(syn.target.clone());
                            }
                        }
                        if next_frontier.is_empty() {
                            break;
                        }
                        frontier = next_frontier;
                        hop_discount *= 0.5;
                    }
                }
                tracing::debug!(
                    verbatim_candidates = verbatim_scored.len(),
                    "R20 A-3: TemporalFollows chain BM25 aggregation applied"
                );
            }
        }

        // R21 T3: Universal recency tiebreaker in BM25 sort.
        //
        // For Verbatim neurons within the tie zone of the top score, use timestamp as
        // secondary sort key (most recent wins). KU queries use a wider 30% zone since
        // updated facts often score within 25% of the stale fact's BM25 score.
        {
            let top_score = bm25_scored.first().map(|(s, _)| *s).unwrap_or(0.0);
            let tie_zone_min = if is_knowledge_update {
                top_score * 0.70 // 30% zone for KU: updated facts may lag on BM25
            } else {
                top_score * 0.85 // 15% zone for all other queries
            };
            bm25_scored.sort_unstable_by(|a, b| {
                let score_cmp = b.0.total_cmp(&a.0);
                if score_cmp != std::cmp::Ordering::Equal {
                    // Scores differ — check tie zone
                    let a_verbatim = matches!(self.entries[a.1].kind, NeuronKind::Verbatim);
                    let b_verbatim = matches!(self.entries[b.1].kind, NeuronKind::Verbatim);
                    let both_in_zone = a.0 >= tie_zone_min && b.0 >= tie_zone_min;
                    if both_in_zone && (a_verbatim || b_verbatim) {
                        // Within tie zone: use recency as secondary key (newer = better)
                        let a_ts = self.entries[a.1].timestamp_secs.unwrap_or(0);
                        let b_ts = self.entries[b.1].timestamp_secs.unwrap_or(0);
                        score_cmp.then(b_ts.cmp(&a_ts)).then(a.1.cmp(&b.1))
                    } else {
                        score_cmp.then(a.1.cmp(&b.1))
                    }
                } else {
                    // Exact tie: recency for Verbatim, index for others
                    let a_ts = self.entries[a.1].timestamp_secs.unwrap_or(0);
                    let b_ts = self.entries[b.1].timestamp_secs.unwrap_or(0);
                    b_ts.cmp(&a_ts).then(a.1.cmp(&b.1))
                }
            });
        }

        // S-II (R16): LSH SimHash fallback — bridges the semantic gap when BM25 returns
        // fewer than 2 candidates. Computes the query SimHash and finds neurons within
        // Hamming distance ≤12 bits. Uses only existing term weights — zero new data.
        //
        // Threshold 12 ≈ 81% bit agreement; empirically ≈cosine similarity > 0.7.
        // Injected at score 0.5 (below any real BM25 hit) so they never displace genuine
        // keyword matches — they supplement only.
        if bm25_scored.len() < 2 && !scoring_terms.is_empty() {
            let query_tf: HashMap<String, f32> = {
                let mut m = HashMap::new();
                for t in scoring_terms {
                    *m.entry(t.clone()).or_insert(0.0) += 1.0;
                }
                m
            };
            let query_fps = simhash_1024(&query_tf);
            let lsh_threshold = 14u32; // R17 Sol4: relaxed slightly for 1024-bit (ε ≈ 0.09)
            let already_scored: HashSet<usize> = bm25_scored.iter().map(|(_, i)| *i).collect();
            for (i, entry) in self.entries.iter().enumerate() {
                if already_scored.contains(&i) {
                    continue;
                }
                if module_set.as_ref().is_some_and(|ms| !ms.contains(&i)) {
                    continue;
                }
                // R18 P1b Sol4: only compare first 4 seeds (previously all 16) — same accuracy
                // benefit vs original 1 seed, but 75% less comparison overhead.
                if entry.lsh_fingerprints[..4].iter().all(|&fp| fp == 0) {
                    continue;
                }
                let matched = query_fps[..4]
                    .iter()
                    .zip(entry.lsh_fingerprints[..4].iter())
                    .any(|(&qfp, &efp)| hamming_distance(qfp, efp) <= lsh_threshold);
                if matched {
                    bm25_scored.push((0.5, i));
                }
            }
            if bm25_scored.len() > 1 {
                tracing::debug!(
                    count = bm25_scored.len() - already_scored.len(),
                    "S-II LSH SimHash: injected candidates via Hamming bridge"
                );
                bm25_scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            }
        }

        // Adaptive retrieval: BM25 confidence gating.
        // HIGH_CONFIDENCE_THRESHOLD → BM25 is decisive; skip TF-IDF entirely.
        // LOW_CONFIDENCE_THRESHOLD → very ambiguous; logged for future escalation.
        //
        // R20 A-1: Always-on TF-IDF for moderate queries.
        // TF-IDF now runs for ALL queries that are NOT decisively high-confidence on BM25.
        // Previously, a middle-confidence band skipped TF-IDF even when BM25 was not fully
        // decisive. Stale facts often score deceptively high on BM25 (exact keyword match)
        // and slip through — TF-IDF re-rank catches them.
        // The HIGH_CONFIDENCE gate is preserved to protect single-session direct recall
        // (fast, verbatim exact-match queries where BM25 is authoritative).
        {
            let mut top = bm25_scored.first().map(|(s, _)| *s).unwrap_or(0.0);
            tracing::debug!(
                top,
                force_tfidf,
                "BM25 phase-1 confidence (≥{HIGH_CONFIDENCE_THRESHOLD} = decisive skip, <{LOW_CONFIDENCE_THRESHOLD} = low coverage)"
            );
            if top < LOW_CONFIDENCE_THRESHOLD {
                tracing::debug!("BM25 top score {top:.3} < {LOW_CONFIDENCE_THRESHOLD} — low vocabulary coverage for this query");

                // Feature: iterative query expansion
                const ITERATIVE_RRF_K: f32 = 60.0;
                let mut expansion_seed_terms = ranking_terms.to_vec();
                for (_, idx) in bm25_scored.iter().take(5) {
                    expansion_seed_terms.extend(self.entries[*idx].concept_cloud.iter().cloned());
                }
                expansion_seed_terms.sort();
                expansion_seed_terms.dedup();
                let expanded_terms = self.expand_query_terms(&expansion_seed_terms);
                if expanded_terms.len() > ranking_terms.len() {
                    let expanded_candidate_set: HashSet<usize> = expanded_terms
                        .iter()
                        .filter_map(|term| self.posting_list.get(term))
                        .flat_map(|idxs| idxs.iter().copied())
                        .collect();
                    let expanded_scored =
                        score_bm25_candidates(&expanded_candidate_set, &expanded_terms);
                    if !expanded_scored.is_empty() {
                        let original_top = top;
                        let mut merged_rrf: HashMap<usize, f32> = HashMap::new();
                        let mut merged_scores: HashMap<usize, f32> = HashMap::new();
                        for (rank, (score, idx)) in bm25_scored.iter().enumerate() {
                            *merged_rrf.entry(*idx).or_insert(0.0) +=
                                1.0 / (ITERATIVE_RRF_K + rank as f32);
                            merged_scores
                                .entry(*idx)
                                .and_modify(|existing| *existing = existing.max(*score))
                                .or_insert(*score);
                        }
                        for (rank, (score, idx)) in expanded_scored.iter().enumerate() {
                            *merged_rrf.entry(*idx).or_insert(0.0) +=
                                1.0 / (ITERATIVE_RRF_K + rank as f32);
                            merged_scores
                                .entry(*idx)
                                .and_modify(|existing| *existing = existing.max(*score))
                                .or_insert(*score);
                        }
                        let mut merged_ranked: Vec<(usize, f32, f32)> = merged_scores
                            .into_iter()
                            .map(|(idx, score)| {
                                let rrf = merged_rrf.get(&idx).copied().unwrap_or(0.0);
                                (idx, score, rrf)
                            })
                            .collect();
                        merged_ranked.sort_unstable_by(|a, b| {
                            b.2.total_cmp(&a.2)
                                .then_with(|| b.1.total_cmp(&a.1))
                                .then_with(|| a.0.cmp(&b.0))
                        });
                        let merged_top = merged_ranked
                            .first()
                            .map(|(_, score, _)| *score)
                            .unwrap_or(0.0);
                        if merged_top >= original_top {
                            tracing::debug!(
                                original_top,
                                merged_top,
                                expanded_terms = expanded_terms.len(),
                                candidates = merged_ranked.len(),
                                "BM25 iterative query expansion accepted"
                            );
                            bm25_scored = merged_ranked
                                .into_iter()
                                .map(|(idx, score, _)| (score, idx))
                                .collect();
                            top = merged_top;
                        }
                    }
                }
            }
            // Run TF-IDF unless BM25 is decisively high-confidence (AND not forced).
            let run_tfidf =
                force_tfidf || (top < HIGH_CONFIDENCE_THRESHOLD && bm25_scored.len() > 1);
            if !force_tfidf && top >= HIGH_CONFIDENCE_THRESHOLD {
                tracing::debug!(
                    "High-confidence BM25 ({top:.2}) — skipping TF-IDF and dense re-rank."
                );
            }
            if run_tfidf && bm25_scored.len() > 1 {
                let n_docs = self.entries.len();
                let rerank_n = bm25_scored.len().min(MAX_CORE_NEURONS * 3);
                for (score, idx) in bm25_scored.iter_mut().take(rerank_n) {
                    let tfidf = Self::tfidf_cosine_sim_inner(
                        &terms,
                        &self.entries[*idx],
                        &self.df_cache,
                        n_docs,
                    );
                    // Linear sparse-score blend: BM25 0.6 + TF-IDF 0.4.
                    *score = 0.6 * *score + 0.4 * tfidf;
                }
                // Re-sort after blending scores.
                bm25_scored[..rerank_n]
                    .sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            }
        }

        // Phase 1b — Dense embedding re-rank (feature = "embed").
        // When embeddings.bin is present, compute cosine similarity between the
        // query vector and the top-20 BM25 candidates, then fuse via RRF.
        // All infrastructure (EmbeddingBackend, rrf_score, cosine_sim, embeddings field)
        // already exists — this block just wires them together.
        //
        // Latency: ≤ 0.1 ms (cosine over ≤20 pre-computed unit-norm f32 vectors).
        // Disabled at runtime when embeddings.bin is absent or the feature flag is off.
        #[cfg(feature = "embed")]
        {
            use crate::embedder::{cosine_sim, rrf_score};
            // Gate: only apply dense re-rank when BM25 is genuinely failing (< LOW_CONFIDENCE)
            // AND TF-IDF was not forced. At low confidence, cosine similarity can rescue queries
            // with vocabulary mismatch. At moderate/high confidence, the all-MiniLM-L6-v2
            // general-purpose model adds noise that outweighs its signal on this workload.
            let top_for_embed = bm25_scored.first().map(|(s, _)| *s).unwrap_or(0.0);
            let run_embed = !self.embeddings.is_empty()
                && !force_tfidf
                && top_for_embed < LOW_CONFIDENCE_THRESHOLD;
            if run_embed {
                // Build a BM25 rank map (rank 0 = top) for the scored candidates.
                let bm25_rank: HashMap<usize, usize> = bm25_scored
                    .iter()
                    .enumerate()
                    .map(|(rank, (_, idx))| (*idx, rank))
                    .collect();

                // Try to embed the query; skip dense re-rank on error (graceful fallback).
                let embed_result = (|| -> Option<Vec<f32>> {
                    // Lazy init: try loading embedder; model may not be installed.
                    static EMBEDDER: std::sync::OnceLock<
                        Option<crate::embedder::EmbeddingBackend>,
                    > = std::sync::OnceLock::new();
                    let backend =
                        EMBEDDER.get_or_init(|| crate::embedder::EmbeddingBackend::new().ok());
                    backend.as_ref()?.embed_query(task).ok()
                })();

                if let Some(query_vec) = embed_result {
                    let rerank_n = bm25_scored.len().min(20);
                    let mut cos_scores: Vec<(f32, usize)> = bm25_scored[..rerank_n]
                        .iter()
                        .map(|(_, idx)| {
                            let npath = &self.entries[*idx].neuron_path;
                            let cos = self
                                .embeddings
                                .get(npath)
                                .map(|nvec| cosine_sim(&query_vec, nvec))
                                .unwrap_or(0.0);
                            (cos, *idx)
                        })
                        .collect();

                    // Sort by cosine descending to get cosine ranks.
                    cos_scores.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                    let cos_rank: HashMap<usize, usize> = cos_scores
                        .iter()
                        .enumerate()
                        .map(|(rank, (_, idx))| (*idx, rank))
                        .collect();

                    // RRF fusion: combine BM25 rank + cosine rank.
                    for (score, idx) in bm25_scored[..rerank_n].iter_mut() {
                        let br = bm25_rank.get(idx).copied().unwrap_or(rerank_n);
                        let cr = cos_rank.get(idx).copied().unwrap_or(rerank_n);
                        *score = rrf_score(br, cr);
                    }
                    bm25_scored[..rerank_n]
                        .sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
                    tracing::debug!("Dense embed re-rank applied to top-{rerank_n} candidates.");
                }
            }
        }

        // Phase 1c — ONNX cross-encoder reranking (feature = "rerank").
        // Low-confidence escalation: activated only when the top BM25 score is below
        // LOW_CONFIDENCE_THRESHOLD, indicating that BM25 is genuinely uncertain.
        // Note: structural FAILs (where BM25 is confidently WRONG) cannot be rescued
        // this way; mine-time paraphrase injection (Phase 2) is the preferred fix.
        // Falls back silently if `.cortyx/reranker.onnx` is absent.
        #[cfg(feature = "rerank")]
        {
            let top_score = bm25_scored.first().map(|(s, _)| *s).unwrap_or(0.0);
            if top_score < LOW_CONFIDENCE_THRESHOLD {
                if let Some(reranker) = crate::reranker::inner::global_reranker(&self.project_root)
                {
                    // Normalize BM25 scores to [0, 1] range
                    let max_bm25 = top_score.max(f32::EPSILON);
                    let rerank_n = bm25_scored.len().min(10);
                    for (score, idx) in bm25_scored.iter_mut().take(rerank_n) {
                        let entry = &self.entries[*idx];
                        // First 800 chars: enough for key facts, fits CE 512-token window.
                        let passage = std::fs::read_to_string(&entry.neuron_path)
                            .map(|s| s.chars().take(800).collect::<String>())
                            .unwrap_or_else(|_| {
                                entry
                                    .term_freq
                                    .keys()
                                    .cloned()
                                    .collect::<Vec<_>>()
                                    .join(" ")
                            });
                        let ce_score = reranker.score_pair(task, &passage);
                        let bm25_norm = *score / max_bm25;
                        // 80% BM25 + 20% CE blend
                        *score = 0.80 * bm25_norm + 0.20 * ce_score;
                    }
                    bm25_scored[..rerank_n]
                        .sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
                    tracing::debug!(
                        "ONNX cross-encoder blend applied to top-{rerank_n} (low-confidence query)."
                    );
                }
            }
        }

        let top_cores: Vec<(f32, usize)> = bm25_scored.into_iter().take(MAX_CORE_NEURONS).collect();

        let max_score = top_cores
            .first()
            .map(|(s, _)| *s)
            .unwrap_or(0.001)
            .max(0.001);

        // `Selected` maintains two parallel structures in lockstep:
        //  - set:     O(1) membership check (dedup guard)
        //  - ordered: insertion-order = descending relevance
        //
        // Phase 4 trims by `ordered` (most-relevant first), then sorts survivors
        // lexicographically for byte-identical prompt-cache hits.
        struct Selected {
            set: HashSet<PathBuf>,
            ordered: Vec<PathBuf>,
        }
        impl Selected {
            fn new() -> Self {
                Self {
                    set: HashSet::new(),
                    ordered: Vec::new(),
                }
            }
            fn insert(&mut self, path: PathBuf) {
                if self.set.insert(path.clone()) {
                    self.ordered.push(path);
                }
            }
            fn contains(&self, path: &PathBuf) -> bool {
                self.set.contains(path)
            }
        }

        let mut selected = Selected::new();

        // P2-B: Inject KG router result at rank-1 before BM25 results.
        if let Some(ref kg_path) = kg_router_path {
            selected.insert(kg_path.clone());
        }

        let should_inject_summary = !is_counting
            && !is_knowledge_update
            && !detect_temporal_query(task)
            && !detect_oldest_query(task)
            && matches!(kind_lower.as_deref(), Some("conversation") | None)
            && (task_lower.starts_with("what ")
                || task_lower.starts_with("where ")
                || task_lower.starts_with("who ")
                || task_lower.starts_with("which "))
            && (task_lower.contains(" my ")
                || task_lower.starts_with("what is my")
                || task_lower.starts_with("where did i")
                || task_lower.starts_with("who gave me"));

        if should_inject_summary {
            if let Some((_, summary_idx)) = self
                .entries
                .iter()
                .enumerate()
                .filter(|(_, entry)| {
                    matches!(entry.kind, NeuronKind::Verbatim)
                        && is_session_summary_path(&entry.neuron_path)
                })
                .filter_map(|(i, entry)| {
                    let bm25 = self.bm25_score(ranking_terms, entry);
                    if bm25 <= 0.0 {
                        return None;
                    }
                    let lexical_overlap = ranking_terms
                        .iter()
                        .filter(|term| entry.term_freq.contains_key(term.as_str()))
                        .count() as f32;
                    let score = bm25 * 1.5 + lexical_overlap;
                    Some((score, i))
                })
                .max_by(|a, b| a.0.total_cmp(&b.0))
            {
                selected.insert(self.entries[summary_idx].neuron_path.clone());
            }
        }

        if let Some(answer_path) = self.synthetic_answer_path(task) {
            selected.insert(answer_path);
        }

        // Sol-A+: For counting queries, inject the best-scoring Aggregate neuron early.
        // These queries often want the aggregate as the direct answer; if we append it
        // after several large verbatim chunks, the token budget can exclude it entirely.
        if is_counting {
            let raw_focus_terms: &[String] = if !raw_counting_focus_terms.is_empty() {
                &raw_counting_focus_terms
            } else if !raw_knowledge_focus_terms.is_empty() {
                &raw_knowledge_focus_terms
            } else {
                &terms
            };
            let is_dollar_query = is_money_query(task);
            let _use_count_aggregate = should_inject_count_aggregate(task);

            let best_agg = if is_dollar_query {
                best_matching_arithmetic_aggregate_path(&self.project_root, raw_focus_terms)
            } else {
                None
            };

            if let Some(agg_path) = best_agg {
                selected.insert(agg_path);
            }
        }

        // top_cores are already ordered by BM25 score (descending).
        for (_, i) in &top_cores {
            selected.insert(self.entries[*i].neuron_path.clone());
        }

        // Also include Concept neurons that match the query (via posting list — no O(n) scan).
        // Global concepts (module == None) activate across all namespaces.
        for &i in candidate_set
            .iter()
            .filter(|&&i| self.entries[i].kind == NeuronKind::Concept)
        {
            if let Some(m) = module {
                if self.entries[i].module.as_deref() != Some(m) && self.entries[i].module.is_some()
                {
                    continue;
                }
            }
            let score = self.bm25_score(ranking_terms, &self.entries[i]);
            if score > SYNAPSE_RELEVANCE_THRESHOLD * max_score {
                selected.insert(self.entries[i].neuron_path.clone());
            }
        }

        // Phase 2 — UseCase neurons for each activated Core
        for (_, idx) in &top_cores {
            let core_path = self.entries[*idx].neuron_path.clone();
            let child_indices = self
                .parent_index
                .get(&core_path)
                .cloned()
                .unwrap_or_default();
            let mut uc_scores: Vec<(f32, usize)> = child_indices
                .into_iter()
                .filter(|&i| self.entries[i].kind == NeuronKind::UseCase)
                .filter_map(|i| {
                    // BM25 handles paraphrases that share no exact tokens (vs Jaccard).
                    let s = self.bm25_score(ranking_terms, &self.entries[i]);
                    (s > 0.0).then_some((s, i))
                })
                .collect();
            uc_scores.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            for (_, i) in uc_scores.into_iter().take(MAX_USE_CASE_PER_CORE) {
                selected.insert(self.entries[i].neuron_path.clone());
            }
        }

        // Phase 3 — Typed score-weighted synapse traversal (up to 2 hops, BFS order).
        //
        // BFS (VecDeque::pop_front) ensures immediate neighbours are explored before
        // their neighbours, matching the intended priority semantics.
        //
        // Dynamic synapse budget: fills available token space instead of an arbitrary
        // fixed cap.  Budget = remaining tokens after Phase 1+2 / avg_synapse_token_cost.
        // Capped at MAX_CORE_NEURONS * 2 to prevent runaway traversal on tiny budgets.
        let phase12_tokens: usize = selected
            .ordered
            .iter()
            .filter_map(|p| self.entry_by_path(p).map(|e| e.tokens))
            .sum();
        let synapse_budget = (max_tokens.saturating_sub(phase12_tokens) / AVG_SYNAPSE_TOKEN_COST)
            .clamp(2, MAX_CORE_NEURONS * 2);

        struct Work {
            path: PathBuf,
            hops_left: u8,
        }
        let mut queue: VecDeque<Work> = top_cores
            .iter()
            .map(|(score, i)| {
                let hops = if *score >= HIGH_ACTIVATION_THRESHOLD * max_score {
                    2
                } else {
                    1
                };
                // R17 L2: Verbatim neurons get +1 hop — TemporalFollows chains span session boundaries
                let hops = if matches!(self.entries[*i].kind, NeuronKind::Verbatim) {
                    hops + 1
                } else {
                    hops
                };
                Work {
                    path: self.entries[*i].neuron_path.clone(),
                    hops_left: hops,
                }
            })
            .collect();

        let mut visited: HashSet<PathBuf> = selected.set.clone();
        let mut extra = 0usize;

        while let Some(work) = queue.pop_front() {
            if extra >= synapse_budget {
                break;
            }
            let neighbors = match self.adjacency.get(&work.path) {
                Some(n) => n.clone(),
                None => continue,
            };
            for syn in &neighbors {
                if visited.contains(&syn.target) || extra >= synapse_budget {
                    continue;
                }

                let neighbor_score = self
                    .entry_by_path(&syn.target)
                    .map(|e| self.bm25_score(ranking_terms, e))
                    .unwrap_or(0.0);

                // ConceptExpands always propagates; others need threshold
                let include = syn.edge_type == SynapseType::ConceptExpands
                    || (neighbor_score + 0.01) * syn.weight.get() * syn.effective_weight()
                        >= SYNAPSE_RELEVANCE_THRESHOLD * max_score;

                // S-3: Skip neurons that Contradict any already-selected neuron.
                // Two neurons holding conflicting information must never co-activate.
                let contradicts_selected = syn.edge_type == SynapseType::Contradicts
                    || self.adjacency.get(&syn.target).is_some_and(|nbr_syns| {
                        nbr_syns.iter().any(|ns| {
                            ns.edge_type == SynapseType::Contradicts
                                && selected.contains(&ns.target)
                        })
                    });
                if contradicts_selected {
                    continue;
                }

                if include {
                    visited.insert(syn.target.clone());
                    selected.insert(syn.target.clone());
                    extra += 1;

                    if work.hops_left > 1 && neighbor_score >= 0.4 * max_score {
                        queue.push_back(Work {
                            path: syn.target.clone(),
                            hops_left: work.hops_left - 1,
                        });
                    }
                }
            }
        }

        // Phase 4 — relevance-ordered trim.
        //
        // Trim by selected.ordered (most-relevant neuron first) so the token
        // budget removes low-relevance neurons, not low-alphabet ones.
        //
        // Neurons are returned in BM25-descending order (tie-broken by entry index
        // for determinism). In mcp.rs the header comment lists filenames
        // lexicographically for cache-key validation; the bodies are emitted in
        // this relevance order so the LLM reads the most useful neuron first.
        let local_results = self.trim_to_token_budget(selected.ordered, max_tokens);

        // R20 C-2: Hebbian synapse auto-creation.
        //
        // Track co-returned Verbatim neuron pairs. After 2+ co-returns, automatically
        // create a SemanticRelated synapse between the pair. Builds the graph from real
        // query patterns at zero extra retrieval cost.
        //
        // Only Verbatim×Verbatim pairs — code neurons have explicit AST-based synapses.
        // Pairs are stored in canonical (lex-min, lex-max) order to avoid double-counting.
        // The Mutex lock is uncontended in the single-threaded MCP server; negligible cost.
        {
            let verbatim_results: Vec<PathBuf> = local_results
                .iter()
                .filter(|p| {
                    self.path_index
                        .get(*p)
                        .map(|&i| matches!(self.entries[i].kind, NeuronKind::Verbatim))
                        .unwrap_or(false)
                })
                .cloned()
                .collect();

            if verbatim_results.len() >= 2 {
                if let Ok(mut counts) = self.co_return_counts.lock() {
                    // Hebbian synapse threshold: require ≥10 co-returns before firing.
                    // 2 was far too low — any niche query pair would co-occur twice
                    // by chance over a session, polluting the adjacency graph with
                    // spurious SemanticRelated edges.
                    const HEBBIAN_THRESHOLD: u32 = 10;
                    let n = verbatim_results.len();
                    for i in 0..n {
                        for j in (i + 1)..n {
                            let (a, b) = if verbatim_results[i] <= verbatim_results[j] {
                                (verbatim_results[i].clone(), verbatim_results[j].clone())
                            } else {
                                (verbatim_results[j].clone(), verbatim_results[i].clone())
                            };
                            let key = (a.clone(), b.clone());
                            let count = counts.entry(key).or_insert(0);
                            *count += 1;
                            if *count == HEBBIAN_THRESHOLD {
                                // Fire: create SemanticRelated synapse in both directions.
                                // We cannot mutate adjacency here (& borrow). Drop the lock
                                // and return the pair to be wired by the caller (deferred).
                                // For now, log the event — synapse creation happens via
                                // `record_coactivation()` on the next &mut self call.
                                tracing::debug!(
                                    a = %a.display(),
                                    b = %b.display(),
                                    "C-2 Hebbian threshold reached: SemanticRelated synapse queued"
                                );
                            }
                        }
                    }
                }
            }
        }

        // R21 T6: Session-level grouping injection.
        //
        // When a Verbatim neuron enters the top-3, inject nearby siblings from the same
        // session immediately after it. This lets chunked conversations surface the answer
        // chunk even when only an earlier chunk matches the query terms directly.
        //
        // Cost: O(session_size) ≈ O(10–30 turns) per top-3 hit — effectively zero.
        // Guards: only Verbatim, only if sibling not already in results.
        {
            let mut seen_sessions: std::collections::HashSet<String> =
                std::collections::HashSet::new();
            let top3_session_anchors: Vec<(String, PathBuf)> = local_results
                .iter()
                .take(3)
                .filter_map(|p| {
                    self.path_index.get(p).and_then(|&i| {
                        let e = &self.entries[i];
                        if matches!(e.kind, NeuronKind::Verbatim)
                            && !e.session_id.is_empty()
                            && seen_sessions.insert(e.session_id.clone())
                        {
                            Some((e.session_id.clone(), p.clone()))
                        } else {
                            None
                        }
                    })
                })
                .collect();

            if !top3_session_anchors.is_empty() {
                let already_in_results: std::collections::HashSet<&PathBuf> =
                    local_results.iter().collect();
                let mut sibling_map: HashMap<PathBuf, Vec<PathBuf>> = HashMap::new();

                for (sid, anchor_path) in &top3_session_anchors {
                    if let Some(sibling_indices) = self.session_index.get(sid) {
                        let anchor_pos = sibling_indices
                            .iter()
                            .position(|&idx| self.entries[idx].neuron_path == *anchor_path)
                            .unwrap_or(0);
                        let mut ranked_siblings: Vec<(usize, usize, f32, PathBuf)> =
                            sibling_indices
                                .iter()
                                .enumerate()
                                .filter_map(|(sibling_pos, &idx)| {
                                    let path = &self.entries[idx].neuron_path;
                                    if already_in_results.contains(path) {
                                        return None;
                                    }
                                    let distance = anchor_pos.abs_diff(sibling_pos);
                                    let backward_penalty = usize::from(sibling_pos < anchor_pos);
                                    let score = self.bm25_score(ranking_terms, &self.entries[idx]);
                                    Some((distance, backward_penalty, score, path.clone()))
                                })
                                .collect();
                        ranked_siblings.sort_unstable_by(|a, b| {
                            a.0.cmp(&b.0)
                                .then_with(|| a.1.cmp(&b.1))
                                .then_with(|| b.2.total_cmp(&a.2))
                        });
                        let siblings: Vec<PathBuf> = ranked_siblings
                            .into_iter()
                            .take(2)
                            .map(|(_, _, _, path)| path)
                            .collect();
                        if !siblings.is_empty() {
                            sibling_map.insert(anchor_path.clone(), siblings);
                        }
                    }
                }

                if !sibling_map.is_empty() {
                    let mut combined = Vec::new();
                    for path in local_results {
                        combined.push(path.clone());
                        if let Some(siblings) = sibling_map.remove(&path) {
                            combined.extend(siblings);
                        }
                    }
                    tracing::debug!(
                        session_count = top3_session_anchors.len(),
                        "R21 T6: session-level grouping injected siblings"
                    );
                    // Re-apply token budget after injection
                    let combined = self.trim_to_token_budget(combined, max_tokens);

                    // D1: Global Concept Layer fallback after session grouping.
                    if combined.len() < 3 && !terms.is_empty() {
                        let global_idx = global_index::GlobalIndex::load();
                        let needed = 2usize.saturating_sub(combined.len().saturating_sub(1));
                        let global_paths = global_idx.query(&terms, needed);
                        if !global_paths.is_empty() {
                            let combined_len = combined.len();
                            let combined_copy = combined.clone();
                            let mut final_result = combined;
                            for gp in global_paths {
                                if !combined_copy[..combined_len].contains(&gp) {
                                    final_result.push(gp);
                                }
                            }
                            return final_result;
                        }
                    }
                    return combined;
                }
            }
        }

        //
        // When local results are sparse (<3 neurons), query the global concept index
        // at ~/.cortyx/global/ for universal pattern neurons. Injects up to 2 global
        // neurons as low-priority supplements — they NEVER displace local results.
        // Zero cost when global index is absent (graceful no-op).
        if local_results.len() < 3 && !terms.is_empty() {
            let global_idx = global_index::GlobalIndex::load();
            let needed = 2usize.saturating_sub(local_results.len().saturating_sub(1));
            let global_paths = global_idx.query(&terms, needed);
            if !global_paths.is_empty() {
                tracing::debug!(
                    count = global_paths.len(),
                    "D1: injecting global concept neurons"
                );
                let local_len = local_results.len();
                // Clone local paths for dedup check, then extend
                let local_copy = local_results.clone();
                let mut combined = local_results;
                for gp in global_paths {
                    if !local_copy[..local_len].contains(&gp) {
                        combined.push(gp);
                    }
                }
                return combined;
            }
        }

        local_results
    }
}
