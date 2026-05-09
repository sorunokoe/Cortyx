// This file is a submodule of `crate::index::core`.
// It contains `impl NeuronIndex` methods extracted from helpers.rs.
// All visibility is relative to `crate::index` (the parent of `core`).
use super::*;

impl NeuronIndex {
    /// Return the token count for a neuron path (for F2 budget tracking).
    pub fn tokens_for(&self, path: &Path) -> usize {
        self.entry_by_path(path).map(|e| e.tokens).unwrap_or(0)
    }

    /// S-III (R16): Count neurons with quality_score below the curation threshold.
    ///
    /// Used by `cortyx status` to surface "needs curation" count.
    pub fn low_quality_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.quality_score < 0.4)
            .count()
    }

    /// Return the number of distinct terms indexed for a neuron.
    ///
    /// Used by S-VIII auto-mine to compute code-block ∩ neuron term overlap ratio.
    pub fn term_count_for(&self, path: &Path) -> usize {
        self.entry_by_path(path)
            .map(|e| e.term_freq.len())
            .unwrap_or(0)
    }

    /// S-I (R16): Return the pre-computed Tier-1 summary for a neuron.
    ///
    /// Returns `None` if the neuron is not indexed or has no summary.
    pub fn summary_for(&self, path: &Path) -> Option<&str> {
        self.entry_by_path(path)
            .filter(|e| !e.summary.is_empty())
            .map(|e| e.summary.as_str())
    }

    pub fn module_for(&self, path: &Path) -> Option<&str> {
        self.entry_by_path(path)
            .and_then(|entry| entry.module.as_deref())
    }

    /// Build a bounded, read-only reasoning report around already-selected evidence paths.
    ///
    /// This intentionally operates after retrieval: callers provide the selected evidence
    /// seeds and the reasoner only explores a small adjacency neighborhood rooted at those
    /// seeds, leaving the BM25 hot path unchanged.
    pub fn reason_over_paths(
        &self,
        seeds: &[(PathBuf, f32)],
        options: TraversalOptions,
    ) -> ReasoningReport {
        let seeds: Vec<(PathBuf, f32)> = seeds
            .iter()
            .filter(|(_, score)| *score > 0.0)
            .map(|(path, score)| (path.clone(), *score))
            .collect();
        if seeds.is_empty() {
            return ReasoningReport::default();
        }

        let mut included = HashSet::new();
        let mut queue = VecDeque::new();
        for (path, _) in &seeds {
            if included.insert(path.clone()) {
                queue.push_back((path.clone(), 0_u8));
            }
        }

        while let Some((path, depth)) = queue.pop_front() {
            if depth >= options.max_hops {
                continue;
            }

            let Some(neighbors) = self.adjacency.get(&path) else {
                continue;
            };
            for synapse in neighbors {
                if included.insert(synapse.target.clone()) {
                    queue.push_back((synapse.target.clone(), depth + 1));
                }
            }
        }

        let neurons = included
            .iter()
            .filter_map(|path| self.entry_by_path(path).map(reasoner_neuron_from_entry))
            .collect::<Vec<_>>();
        let kg_entities = included
            .iter()
            .filter(|path| looks_like_kg_neuron_path(path))
            .filter_map(|path| kg::KgEntity::load(path).ok())
            .collect::<Vec<_>>();

        if neurons.is_empty() && kg_entities.is_empty() {
            return ReasoningReport::default();
        }

        GraphReasoner::new(neurons, kg_entities).trace(
            &seeds
                .into_iter()
                .map(|(path, score)| ReasonerSeed::new(path, score))
                .collect::<Vec<_>>(),
            options,
        )
    }

    pub fn context_metadata_for(&self, path: &Path) -> Option<ContextMetadata> {
        self.entry_by_path(path).map(|entry| {
            let hit_rate = if entry.use_count == 0 {
                0.0
            } else {
                entry.hit_count as f32 / entry.use_count as f32
            };
            ContextMetadata {
                kind: entry.kind.clone(),
                module: entry.module.clone(),
                summary: entry.summary.clone(),
                timestamp_secs: entry.timestamp_secs,
                tokens: entry.tokens,
                use_count: entry.use_count,
                hit_count: entry.hit_count,
                hit_rate,
            }
        })
    }

    pub fn derived_answer_path_for_task(&self, task: &str) -> Option<PathBuf> {
        let query = QueryText::new(task).ok()?;
        self.synthetic_answer_path(query.as_str())
    }

    /// S-I (R16): Like `get_contexts_with_overflow` but returns BM25 scores for tiered emission.
    ///
    /// Returns:
    /// - `full`: `(path, bm25_score)` for neurons within budget
    /// - `overflow`: `(path, headline)` for budget-overflow neurons
    ///
    /// Tier mapping (by score):
    /// - `score ≥ 5.0` → Tier 2 (full body) — caller reads the file
    /// - `1.5 ≤ score < 5.0` → Tier 1 (summary only) — caller uses `summary_for()`
    /// - `score < 1.5` → Tier 0 (headline only, same as overflow) — already in overflow set
    pub fn get_contexts_with_scores_and_overflow(
        &self,
        task: &str,
        max_tokens: usize,
        module: Option<&str>,
        kind: Option<&str>,
        min_confidence: Option<f32>,
        multi_hop: bool,
    ) -> (Vec<(PathBuf, f32)>, Vec<(PathBuf, String)>) {
        let Ok(query) = QueryText::new(task) else {
            return (Vec::new(), Vec::new());
        };
        // Delegation: run the full pipeline then re-score the results for tier assignment.
        let (full_paths, overflow) = self.get_contexts_with_overflow(
            task,
            max_tokens,
            module,
            kind,
            min_confidence,
            multi_hop,
        );
        let terms = tokenize(query.as_str());
        let full_with_scores: Vec<(PathBuf, f32)> = full_paths
            .into_iter()
            .map(|path| {
                let score = self
                    .entry_by_path(&path)
                    .map(|e| self.bm25_score(&terms, e))
                    .unwrap_or(0.0);
                (path, score)
            })
            .collect();
        (full_with_scores, overflow)
    }

    /// CountNeuron (TRIZ NE-5): Pre-aggregate cross-session occurrence counts at mine time.
    ///
    /// Scans all `NeuronKind::Verbatim` entries, groups them by `session_id`, and builds
    /// a `term → distinct_sessions` map.  For terms appearing in ≥3 distinct sessions it
    /// emits a `NeuronKind::Aggregate` neuron that answers "how many times did I X?" in
    /// O(1) — the count is written in BOTH numeral and word form so keyword matching hits.
    ///
    /// Call this after `idx.commit()` and call `idx.commit()` once more if it returns
    /// `true` (at least one aggregate neuron was staged).
    pub fn emit_aggregate_neurons(&mut self, project_root: &Path) -> Result<bool> {
        use crate::neuron::NeuronStatus;
        use std::collections::hash_map::Entry;

        // Common words that would produce useless aggregate neurons
        const AGG_STOP: &[&str] = &[
            "that",
            "this",
            "with",
            "from",
            "have",
            "will",
            "what",
            "when",
            "where",
            "which",
            "there",
            "their",
            "them",
            "they",
            "then",
            "been",
            "were",
            "some",
            "just",
            "also",
            "about",
            "into",
            "more",
            "than",
            "your",
            "here",
            "very",
            "well",
            "over",
            "back",
            "down",
            "would",
            "could",
            "should",
            "might",
            "does",
            "didn",
            "wasn",
            "aren",
            "isn",
            "hasn",
            "like",
            "want",
            "need",
            "think",
            "know",
            "said",
            "told",
            "went",
            "make",
            "made",
            "take",
            "took",
            "come",
            "came",
            "went",
            "going",
            "really",
            "still",
            "even",
            "already",
            "always",
            "never",
            "every",
            "after",
            "before",
            "during",
            "while",
            "other",
            "another",
            "both",
            "first",
            "last",
            "next",
            "same",
            "such",
            "much",
            "many",
            "most",
            "because",
            "since",
            "through",
            "between",
            "under",
            "again",
            "help",
            "time",
            "year",
            "week",
            "month",
            "today",
            "yesterday",
            "tomorrow",
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ];
        let agg_stop: HashSet<&str> = AGG_STOP.iter().copied().collect();

        // Gather (term, session_id) pairs from every Verbatim entry.
        let mut term_sessions: HashMap<String, HashSet<String>> = HashMap::new();
        let mut term_snippets: HashMap<String, Vec<(String, String)>> = HashMap::new();

        // Collect entries data without borrowing self mutably (for peek_neuron)
        let entries_snapshot: Vec<(NeuronKind, String, PathBuf, Vec<String>)> = self
            .entries
            .iter()
            .filter(|e| matches!(e.kind, NeuronKind::Verbatim) && !e.session_id.is_empty())
            .map(|e| {
                (
                    e.kind.clone(),
                    e.session_id.clone(),
                    e.neuron_path.clone(),
                    e.term_freq.keys().cloned().collect(),
                )
            })
            .collect();

        for (_, sid, neuron_path, terms) in &entries_snapshot {
            let content_snippet = std::fs::read_to_string(neuron_path)
                .unwrap_or_default()
                .lines()
                .filter(|l| !l.starts_with('#'))
                .take(1)
                .next()
                .unwrap_or("")
                .chars()
                .take(100)
                .collect::<String>();

            for term in terms {
                // Only count-worthy terms: ≥4 chars, letters only (no numbers/punct)
                if term.len() < 4 {
                    continue;
                }
                if !term.chars().all(|c| c.is_ascii_alphabetic()) {
                    continue;
                }
                if agg_stop.contains(term.as_str()) {
                    continue;
                }

                term_sessions
                    .entry(term.clone())
                    .or_default()
                    .insert(sid.clone());

                if let Entry::Occupied(mut e) = term_snippets.entry(term.clone()) {
                    // Limit snippets per term to avoid huge files
                    if e.get().len() < 10 && !e.get().iter().any(|(s, _)| s == sid) {
                        e.get_mut().push((sid.clone(), content_snippet.clone()));
                    }
                } else {
                    term_snippets
                        .insert(term.clone(), vec![(sid.clone(), content_snippet.clone())]);
                }
            }
        }

        let ndir = neuron_dir(project_root);
        let mut staged = 0usize;

        for (term, sessions) in &term_sessions {
            let count = sessions.len();
            if count < 3 {
                continue;
            }

            let slug: String = term.chars().take(48).collect();
            let fname = format!("_count_{slug}.aggregate.md");
            let neuron_path = ndir.join(&fname);

            let word = num_to_word(count);
            let count_str = if word.is_empty() {
                format!("{count}")
            } else {
                format!("{count} ({word})")
            };

            // Snippets section
            let snippets = term_snippets.get(term).cloned().unwrap_or_default();
            let snippet_lines: String = snippets
                .iter()
                .map(|(sid, snip)| format!("- {sid}: {snip}\n"))
                .collect();

            let query_surface = format!(
                "how many {term}\n\
                 count of {term}\n\
                 number of {term}\n\
                 how many different {term}\n\
                 total {term}\n"
            );

            let content = format!(
                "# _count_{slug}\n\
                 \n\
                 ## purpose\n\
                 Aggregate count: \"{term}\" mentioned in {count_str} sessions.\n\
                 \n\
                 ## count\n\
                 {count_str} sessions\n\
                 \n\
                 ## entity\n\
                 {term}\n\
                 \n\
                 ## query_surface\n\
                 <!-- SECTION: query_surface -->\n\
                 {query_surface}\
                 <!-- /SECTION -->\n\
                 \n\
                 ## sessions\n\
                 {snippet_lines}\
                 \n\
                 ## total\n\
                 Mentioned {count_str} times across {count_str} sessions. Count: {count} ({}).\n",
                num_to_word(count)
            );

            // Write the file
            if let Err(e) = atomic_write(&neuron_path, content.as_bytes()) {
                eprintln!("[emit_aggregate] failed to write {fname}: {e}");
                continue;
            }

            // Build meta for Aggregate neuron
            let mut meta = NeuronMeta::new_stub(project_root, NeuronKind::Aggregate);
            meta.status = NeuronStatus::Fresh;
            meta.tokens = estimate_context_tokens(&content).get();

            self.stage(&neuron_path, &content, &meta);
            staged += 1;
        }

        Ok(staged > 0)
    }

    /// TRIZ Sol-A: Pre-compute arithmetic aggregates (dollar/numeric sums) at mine time.
    ///
    /// Scans all Verbatim neurons, extracts dollar amounts, groups by entity slug,
    /// and emits offline aggregate files that answer "how much total did X spend?" in O(1).
    /// These files are kept out of the hot BM25 index and are injected directly by path
    /// for money queries, preserving recall without bloating startup latency.
    /// Emit arithmetic aggregate neurons grouped by TOPIC TERM.
    ///
    /// For each term appearing in ≥2 sessions where that term co-occurs with a dollar amount
    /// on the same line, compute the total dollars and emit `_arith_{term}.aggregate.md`.
    ///
    /// This enables Sol-A+ to inject the correct sum for queries like
    /// "how much total have I spent on bike-related expenses?" → finds _arith_bike.aggregate.md
    /// containing "Total: $185".
    pub fn emit_arithmetic_aggregate_neurons(&mut self, project_root: &Path) -> Result<bool> {
        fn parse_dollar(s: &str) -> Option<i64> {
            let cleaned: String = s
                .chars()
                .filter(|c| c.is_ascii_digit() || *c == '.')
                .collect();
            let val: f64 = cleaned.parse().ok()?;
            if val > 10_000_000.0 {
                return None;
            }
            Some((val * 100.0).round() as i64)
        }

        fn extract_dollars_on_line(line: &str) -> Vec<i64> {
            let mut results = Vec::new();
            let bytes = line.as_bytes();
            let mut i = 0;
            while i < bytes.len() {
                if bytes[i] == b'$' {
                    let start = i + 1;
                    let mut j = start;
                    while j < bytes.len()
                        && (bytes[j].is_ascii_digit() || bytes[j] == b',' || bytes[j] == b'.')
                    {
                        j += 1;
                    }
                    if j > start {
                        let num_str = &line[start..j];
                        if let Some(cents) = parse_dollar(num_str) {
                            if cents > 0 {
                                results.push(cents);
                            }
                        }
                    }
                    i = j;
                } else {
                    i += 1;
                }
            }
            results
        }

        fn is_grounded_user_money_line(lower: &str) -> bool {
            if !lower.trim_start().starts_with("user:") {
                return false;
            }

            ![
                "budget",
                "under $",
                "over $",
                "around $",
                "approximately $",
                "approx $",
                "starting at $",
                "start at $",
                "ranges from $",
                "range from $",
                "between $",
                "if you book",
                "fare is around",
                "might run around",
                "could cost",
                "would cost",
                "would be around",
            ]
            .iter()
            .any(|marker| lower.contains(marker))
        }

        fn clean_alpha_token(token: &str, agg_stop: &HashSet<&str>) -> Option<String> {
            let cleaned: String = token
                .chars()
                .filter(|c| c.is_ascii_alphabetic())
                .collect::<String>()
                .to_ascii_lowercase();
            if cleaned.len() < 4 || agg_stop.contains(cleaned.as_str()) {
                return None;
            }
            Some(cleaned)
        }

        fn trim_boundary_terms(words: &mut Vec<String>) {
            const BOUNDARY_STOP: &[&str] = &[
                "spent",
                "spend",
                "pay",
                "paid",
                "buy",
                "bought",
                "purchase",
                "purchased",
                "cost",
                "costs",
                "costing",
                "using",
                "used",
                "redeemed",
                "redeem",
                "bill",
                "bills",
                "fare",
                "fares",
                "ticket",
                "tickets",
                "coupon",
                "coupons",
                "amount",
                "total",
                "money",
                "dollars",
            ];

            while words
                .first()
                .is_some_and(|w| BOUNDARY_STOP.contains(&w.as_str()))
            {
                words.remove(0);
            }
            while words
                .last()
                .is_some_and(|w| BOUNDARY_STOP.contains(&w.as_str()))
            {
                words.pop();
            }
        }

        fn add_phrase_aliases(words: &[String], out: &mut std::collections::BTreeSet<String>) {
            if words.is_empty() {
                return;
            }

            let max_len = words.len().min(3);
            for start in 0..words.len() {
                for len in 1..=max_len.min(words.len() - start) {
                    out.insert(words[start..start + len].join(" "));
                }
            }
        }

        fn extract_topic_candidates(
            line: &str,
            agg_stop: &HashSet<&str>,
        ) -> std::collections::BTreeSet<String> {
            let raw_tokens: Vec<String> = line
                .split_whitespace()
                .map(|token| {
                    token
                        .trim_matches(|c: char| {
                            !c.is_ascii_alphanumeric() && c != '$' && c != '-' && c != '_'
                        })
                        .to_ascii_lowercase()
                })
                .filter(|token| !token.is_empty())
                .collect();

            let dollar_indices: Vec<usize> = raw_tokens
                .iter()
                .enumerate()
                .filter(|(_, token)| token.starts_with('$'))
                .map(|(i, _)| i)
                .collect();

            let mut candidates = std::collections::BTreeSet::new();
            if dollar_indices.is_empty() {
                return candidates;
            }

            for &idx in &dollar_indices {
                if let Some(anchor) = raw_tokens.get(idx + 1) {
                    if matches!(anchor.as_str(), "on" | "for" | "at" | "toward" | "towards") {
                        let words: Vec<String> = raw_tokens
                            .iter()
                            .skip(idx + 2)
                            .take(5)
                            .filter_map(|token| clean_alpha_token(token, agg_stop))
                            .collect();
                        add_phrase_aliases(&words, &mut candidates);
                    }
                }

                let mut before: Vec<String> = raw_tokens
                    .iter()
                    .take(idx)
                    .rev()
                    .take(6)
                    .filter_map(|token| clean_alpha_token(token, agg_stop))
                    .collect();
                before.reverse();
                trim_boundary_terms(&mut before);
                add_phrase_aliases(&before, &mut candidates);

                let mut after: Vec<String> = raw_tokens
                    .iter()
                    .skip(idx + 1)
                    .take(6)
                    .filter_map(|token| clean_alpha_token(token, agg_stop))
                    .collect();
                trim_boundary_terms(&mut after);
                add_phrase_aliases(&after, &mut candidates);
            }

            for (i, token) in raw_tokens.iter().enumerate() {
                if matches!(token.as_str(), "on" | "for" | "at" | "toward" | "towards") {
                    let words: Vec<String> = raw_tokens
                        .iter()
                        .skip(i + 1)
                        .take(5)
                        .filter_map(|raw| clean_alpha_token(raw, agg_stop))
                        .collect();
                    add_phrase_aliases(&words, &mut candidates);
                }
            }

            candidates
        }

        fn cents_to_dollars(cents: i64) -> String {
            let dollars = cents / 100;
            let rem = cents % 100;
            if rem == 0 {
                format!("${dollars}")
            } else {
                format!("${dollars}.{rem:02}")
            }
        }

        fn dollars_to_words(cents: i64) -> String {
            let dollars = cents / 100;
            match dollars {
                0 => "zero dollars".to_string(),
                1 => "one dollar".to_string(),
                2..=20 => format!("{} dollars", num_to_word(dollars as usize)),
                21..=99 => {
                    let tens = dollars / 10;
                    let ones = dollars % 10;
                    let tw = match tens {
                        2 => "twenty",
                        3 => "thirty",
                        4 => "forty",
                        5 => "fifty",
                        6 => "sixty",
                        7 => "seventy",
                        8 => "eighty",
                        9 => "ninety",
                        _ => "",
                    };
                    if ones == 0 {
                        format!("{tw} dollars")
                    } else {
                        format!("{tw}-{} dollars", num_to_word(ones as usize))
                    }
                },
                100..=999 => format!("{} hundred dollars", num_to_word((dollars / 100) as usize)),
                1000..=99999 => format!("{} thousand dollars", dollars / 1000),
                _ => format!("{dollars} dollars"),
            }
        }

        // Same stop words as emit_aggregate_neurons
        const AGG_STOP: &[&str] = &[
            "that",
            "this",
            "with",
            "from",
            "have",
            "will",
            "what",
            "when",
            "where",
            "which",
            "there",
            "their",
            "them",
            "they",
            "then",
            "been",
            "were",
            "some",
            "just",
            "also",
            "about",
            "into",
            "more",
            "than",
            "your",
            "here",
            "very",
            "well",
            "over",
            "back",
            "down",
            "would",
            "could",
            "should",
            "might",
            "does",
            "didn",
            "wasn",
            "aren",
            "isn",
            "hasn",
            "like",
            "want",
            "need",
            "think",
            "know",
            "said",
            "told",
            "went",
            "make",
            "made",
            "take",
            "took",
            "come",
            "came",
            "going",
            "really",
            "still",
            "even",
            "already",
            "always",
            "never",
            "every",
            "after",
            "before",
            "during",
            "while",
            "other",
            "another",
            "both",
            "first",
            "last",
            "next",
            "same",
            "such",
            "much",
            "many",
            "most",
            "because",
            "since",
            "through",
            "between",
            "under",
            "again",
            "help",
            "time",
            "year",
            "week",
            "month",
            "today",
            "yesterday",
            "tomorrow",
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
            "cost",
            "price",
            "paid",
            "spend",
            "spent",
            "total",
            "amount",
            "dollars",
        ];
        let agg_stop: HashSet<&str> = AGG_STOP.iter().copied().collect();

        // Build: topic phrase → [(session_id, dollars_on_supporting_lines)]
        let mut topic_session_dollars: HashMap<String, Vec<(String, Vec<i64>)>> = HashMap::new();
        let mut topic_aliases: HashMap<String, std::collections::BTreeSet<String>> = HashMap::new();
        let mut topic_snippets: HashMap<String, Vec<(String, String)>> = HashMap::new();
        let mut topic_seen_snippets: HashMap<String, HashSet<String>> = HashMap::new();

        let entries_snapshot: Vec<(String, PathBuf)> = self
            .entries
            .iter()
            .filter(|e| {
                matches!(e.kind, NeuronKind::Verbatim)
                    && !e.session_id.is_empty()
                    && !is_session_summary_path(&e.neuron_path)
            })
            .map(|e| (e.session_id.clone(), e.neuron_path.clone()))
            .collect();

        for (sid, neuron_path) in &entries_snapshot {
            let content = match std::fs::read_to_string(neuron_path) {
                Ok(c) => c,
                Err(_) => continue,
            };
            for line in content.lines() {
                let trimmed = line.trim();
                let lower = trimmed.to_ascii_lowercase();
                if !is_grounded_user_money_line(&lower) {
                    continue;
                }

                let body = trimmed.strip_prefix("User:").unwrap_or(trimmed).trim();
                let dollars = extract_dollars_on_line(body);
                if dollars.is_empty() {
                    continue;
                }

                let snippet: String = body.chars().take(120).collect();
                for topic in extract_topic_candidates(body, &agg_stop) {
                    let seen_key = format!("{sid}\n{snippet}");
                    if !topic_seen_snippets
                        .entry(topic.clone())
                        .or_default()
                        .insert(seen_key)
                    {
                        continue;
                    }
                    let entry = topic_session_dollars.entry(topic.clone()).or_default();
                    if let Some(se) = entry.iter_mut().find(|(s, _)| s == sid) {
                        se.1.extend_from_slice(&dollars);
                    } else {
                        entry.push((sid.clone(), dollars.clone()));
                    }

                    let aliases = topic_aliases.entry(topic.clone()).or_default();
                    aliases.insert(topic.clone());
                    for word in topic.split_whitespace() {
                        aliases.insert(word.to_string());
                    }

                    let snippets = topic_snippets.entry(topic).or_default();
                    if snippets.len() < 10
                        && !snippets
                            .iter()
                            .any(|(s, snip)| s == sid && snip == &snippet)
                    {
                        snippets.push((sid.clone(), snippet.clone()));
                    }
                }
            }
        }

        let ndir = neuron_dir(project_root);
        let mut staged = 0usize;

        for (topic, session_entries) in &topic_session_dollars {
            let sessions_with_dollars: Vec<_> = session_entries
                .iter()
                .filter(|(_, amounts)| !amounts.is_empty())
                .collect();
            let has_multi_amount_session = sessions_with_dollars
                .iter()
                .any(|(_, amounts)| amounts.len() >= 2);
            if sessions_with_dollars.len() < 2 && !has_multi_amount_session {
                continue;
            }

            let total_cents: i64 = sessions_with_dollars
                .iter()
                .flat_map(|(_, amounts)| amounts.iter().copied())
                .sum();
            if total_cents <= 0 {
                continue;
            }

            let total_str = cents_to_dollars(total_cents);
            let total_words = dollars_to_words(total_cents);
            let total_dollars = total_cents / 100;
            let session_count = sessions_with_dollars.len();
            let count_str = if session_count <= 20 {
                format!("{session_count} ({})", num_to_word(session_count))
            } else {
                format!("{session_count}")
            };

            let breakdown: String = sessions_with_dollars
                .iter()
                .map(|(sid, amounts)| {
                    let st: i64 = amounts.iter().sum();
                    format!(
                        "- {sid}: {} ({})\n",
                        cents_to_dollars(st),
                        amounts
                            .iter()
                            .map(|c| cents_to_dollars(*c))
                            .collect::<Vec<_>>()
                            .join(" + ")
                    )
                })
                .collect();
            let evidence_lines: String = topic_snippets
                .get(topic)
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .map(|(sid, snip)| format!("- {sid}: {snip}\n"))
                .collect();
            let aliases = topic_aliases
                .get(topic)
                .map(|aliases| aliases.iter().cloned().collect::<Vec<_>>())
                .unwrap_or_default();
            let alias_line = aliases.join(", ");
            let query_surface = format!(
                "how much did i spend on {topic}\n\
                 how much have i spent on {topic}\n\
                 what was the total for {topic}\n\
                 what is the total for {topic}\n\
                 total amount for {topic}\n\
                 total spent on {topic}\n\
                 how much money for {topic}\n\
                 {alias_line}\n"
            );
            let slug: String = topic
                .chars()
                .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
                .take(48)
                .collect();
            let content = format!(
                "# _arith_{slug}\n\
                 \n\
                 ## purpose\n\
                 Arithmetic aggregate: total dollar amount for topic \"{topic}\" across {count_str} sessions.\n\
                 \n\
                 ## topic\n\
                 {topic}\n\
                 Aliases: {alias_line}\n\
                 \n\
                 ## query_surface\n\
                 <!-- SECTION: query_surface -->\n\
                 {query_surface}\
                 <!-- /SECTION -->\n\
                 \n\
                 ## sum\n\
                 {total_str} ({total_words})\n\
                 \n\
                 ## breakdown\n\
                 {breakdown}\
                 \n\
                 ## evidence\n\
                 {evidence_lines}\
                 \n\
                 ## total\n\
                 Total: {total_str} across {count_str} sessions.\n\
                 Amount: {total_dollars}. Sum: {total_dollars}. Total dollars: {total_dollars}.\n\
                 In words: {total_words}.\n",
            );

            let fname = format!("_arith_{slug}.aggregate.md");
            let neuron_path = ndir.join(&fname);
            if let Err(e) = atomic_write(&neuron_path, content.as_bytes()) {
                eprintln!("[emit_arithmetic_aggregate] failed to write {fname}: {e}");
                continue;
            }
            staged += 1;
        }

        Ok(staged > 0)
    }

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
            let gone = source.as_ref().map_or(false, |s| !s.exists());
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

    ///
    /// For each file path in `open_files`, looks up the corresponding neuron entry
    /// and returns the top-N most frequent terms as soft expansion tokens.
    /// These are injected into the task string with a weight comment so BM25
    /// treats them at reduced significance relative to the direct task query.
    ///
    /// Lookup is O(k) where k = |open_files| — all data is already in the index.
    /// Returns a deduplicated list of terms (sorted by frequency descending).
    pub fn soft_terms_for_editor_context(
        &self,
        open_files: &[String],
        max_terms_per_file: usize,
    ) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();

        for file_path in open_files {
            // Match the open file path to an indexed neuron (suffix or substring match).
            let entry = self.entries.iter().find(|e| {
                let ep = e.neuron_path.to_string_lossy();
                ep.ends_with(file_path.as_str()) || ep.contains(file_path.as_str())
            });

            if let Some(e) = entry {
                // Sort by term frequency descending, take top-N
                let mut term_freq_sorted: Vec<(&String, f32)> =
                    e.term_freq.iter().map(|(t, f)| (t, *f)).collect();
                term_freq_sorted
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                for (term, _freq) in term_freq_sorted.iter().take(max_terms_per_file) {
                    if term.len() >= 3 && seen.insert((*term).clone()) {
                        result.push((*term).clone());
                    }
                }
            }
        }
        result
    }

    /// S-VII (R16): Apply biological LTD (Long-Term Depression) temporal decay to all synapses.
    ///
    /// Called once at `serve` startup and after `compile`. Mimics Hebbian LTD:
    /// synapses that have not been co-activated for many days gradually weaken,
    /// keeping the synapse graph lean and preventing dead-edge accumulation.
    ///
    /// Decay formula (half-life ≈ 70 days, λ = 0.01):
    ///   `learned_weight *= exp(-0.01 * days_idle)`
    ///
    /// Synapses with `learned_weight < 0.05` after decay are pruned (removed).
    /// Synapses with `last_co_activation_day == 0` are skipped (not yet learned).
    ///
    /// Returns: `(decayed, pruned)` counts for logging.
    pub fn apply_synapse_decay(&mut self) -> (usize, usize) {
        let now_days = now_unix_days();
        let (mut decayed, mut pruned) = (0usize, 0usize);
        for entry in &mut self.entries {
            let before = entry.synapses.len();
            for syn in &mut entry.synapses {
                if syn.last_co_activation_day == 0 || syn.learned_weight <= 0.0 {
                    continue; // not yet learned — skip
                }
                let days_idle = now_days.saturating_sub(syn.last_co_activation_day);
                if days_idle > 0 {
                    syn.learned_weight *= f32::exp(-0.01 * days_idle as f32);
                    decayed += 1;
                }
            }
            entry
                .synapses
                .retain(|s| s.learned_weight > 0.05 || s.learned_weight <= 0.0);
            pruned += before - entry.synapses.len();
        }
        // Rebuild adjacency cache after pruning
        if pruned > 0 {
            self.rebuild_derived_pub();
        }
        tracing::info!(decayed, pruned, "S-VII: synapse temporal decay applied");
        (decayed, pruned)
    }

    /// Update `last_co_activation_day` for all synapses between two co-cited neurons.
    ///
    /// Called from `record_hit` when both source and target of a synapse are cited
    /// in the same session — this is the LTP (Long-Term Potentiation) counterpart
    /// to `apply_synapse_decay`'s LTD.
    pub fn touch_co_activation_day(&mut self, cited_paths: &[PathBuf]) {
        let today = now_unix_days();
        let cited_set: std::collections::HashSet<&PathBuf> = cited_paths.iter().collect();
        for entry in &mut self.entries {
            if !cited_set.contains(&entry.neuron_path) {
                continue;
            }
            for syn in &mut entry.synapses {
                if cited_set.contains(&syn.target) {
                    syn.last_co_activation_day = today;
                }
            }
        }
    }

    /// Find all `Contradicts` edges between any pair of activated neurons.
    ///
    /// Used by `get_contexts` to append a warning block when conflicting neurons
    /// are simultaneously activated — alerting the LLM to verify which is current.
    ///
    /// Performance: O(n²) over the activated set. For typical n=5, this is 10 lookups
    /// into the adjacency HashMap — effectively O(1) at runtime.
    ///
    /// Returns: `(path_a, path_b, reason)` for each contradicting pair found.
    pub fn find_contradictions(&self, activated: &[PathBuf]) -> Vec<(PathBuf, PathBuf, String)> {
        let mut pairs = Vec::new();
        for i in 0..activated.len() {
            if let Some(syns) = self.adjacency.get(&activated[i]) {
                for syn in syns {
                    if syn.edge_type == SynapseType::Contradicts {
                        // Only report each pair once (i < j by index in activated)
                        if let Some(j) = activated[i + 1..].iter().position(|p| *p == syn.target) {
                            let j_abs = i + 1 + j;
                            pairs.push((
                                activated[i].clone(),
                                activated[j_abs].clone(),
                                syn.reason.trim_start_matches("← ").to_string(),
                            ));
                        }
                    }
                }
            }
        }
        pairs
    }

    /// Scan all neurons (or a single neuron if `path` is given) for `Contradicts` edges.
    ///
    /// Used by `cortyx_check_consistency` — a proactive scan before task execution.
    /// Returns all contradiction pairs in the index (or pairs involving `path`).
    pub fn all_contradictions(
        &self,
        path_filter: Option<&Path>,
    ) -> Vec<(PathBuf, PathBuf, String)> {
        let mut seen: std::collections::HashSet<(PathBuf, PathBuf)> = Default::default();
        let mut pairs = Vec::new();
        for (src, syns) in &self.adjacency {
            if let Some(pf) = path_filter {
                if src != pf {
                    continue;
                }
            }
            for syn in syns {
                if syn.edge_type != SynapseType::Contradicts {
                    continue;
                }
                let a = src.min(&syn.target).clone();
                let b = src.max(&syn.target).clone();
                if seen.insert((a.clone(), b.clone())) {
                    pairs.push((a, b, syn.reason.trim_start_matches("← ").to_string()));
                }
            }
        }
        pairs
    }

    /// Load neuron body text for semantic consistency checks.
    ///
    /// When `path_filter` is given, returns only that neuron's body (for single-neuron
    /// scans). Without a filter, returns up to `limit` neuron bodies ordered by hit-rate
    /// descending so the most-used neurons are checked first.
    ///
    /// Used by `cortyx_check_consistency` to feed PureReason's semantic contradiction
    /// detector with raw neuron text.
    pub fn neuron_bodies_for_consistency(
        &self,
        path_filter: Option<&Path>,
        limit: usize,
    ) -> Option<Vec<String>> {
        if let Some(pf) = path_filter {
            let body = std::fs::read_to_string(pf).ok()?;
            return Some(vec![body]);
        }
        let mut entries: Vec<&BM25Entry> = self.entries.iter().collect();
        entries.sort_by(|a, b| {
            b.hit_count
                .partial_cmp(&a.hit_count)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let bodies: Vec<String> = entries
            .into_iter()
            .take(limit)
            .filter_map(|e| std::fs::read_to_string(&e.neuron_path).ok())
            .collect();
        Some(bodies)
    }

    /// Propagate staleness to all neurons that import/call/implement the changed one.
    ///
    /// When a source file changes its neuron is marked stale. This method finds all
    /// neurons with synapse edges pointing *to* that neuron (reverse lookup via the
    /// adjacency list) and demotes their `staleness_multiplier` by ×0.7 (floor 0.3).
    ///
    /// Effect: dependent neurons surface as "needs re-evolve" in status, and rank
    /// lower in BM25 until the LLM refreshes them — preventing silent context drift.
    ///
    /// Cost: O(n) over all entries; n < 1 000 in typical projects → <1 ms.
    pub fn cascade_staleness(&mut self, changed_neuron: &Path) {
        for entry in &mut self.entries {
            let is_dependent = entry.synapses.iter().any(|s| {
                s.target == changed_neuron
                    && matches!(
                        s.edge_type,
                        SynapseType::Imports | SynapseType::Calls | SynapseType::Implements
                    )
            });
            if is_dependent {
                // Demote (not evict) — preserves content while signalling freshness risk.
                entry.staleness_multiplier = (entry.staleness_multiplier * 0.7).max(0.3);
                tracing::debug!(
                    path = ?entry.neuron_path,
                    "cascade_staleness: dependent neuron demoted to staleness_multiplier={:.2}",
                    entry.staleness_multiplier
                );
            }
        }
    }
}

impl NeuronIndex {
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
