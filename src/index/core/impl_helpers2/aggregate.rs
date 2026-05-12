use super::*;

impl NeuronIndex {
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
}
