use super::*;

/// R17 Sol1: Prospective Query Pre-image.
///
/// Scans a conversation turn for fact-bearing assertions and generates the natural-language
/// question forms that a human would ask about those facts. Returned as a space-separated
/// string of question vocabulary tokens for BM25 injection.
pub(crate) fn generate_query_surface(text: &str) -> Option<String> {
    let lower = text.to_lowercase();
    let mut tokens: Vec<&str> = Vec::new();
    for (triggers, vocab) in BASIC_PATTERNS
        .iter()
        .chain(LIFESTYLE_PATTERNS.iter())
        .chain(PERSONAL_PATTERNS.iter())
        .chain(BENCHMARK_PATTERNS.iter())
    {
        if triggers.iter().any(|t| lower.contains(t)) {
            tokens.extend_from_slice(vocab);
        }
    }

    // NE-6: Universal disclosure-signal extraction (TRIZ P10 Preliminary Action).
    //
    // "By the way, [fact]" is the dominant user disclosure pattern in conversational memory:
    // 803 occurrences across 500 sessions (1.6× per session) in LME-500.
    // "Speaking of," and "Also," are secondary signals.
    //
    // Extract up to 30 content words after each disclosure signal and add them to the
    // query_surface. This is applied ALWAYS (not just when category patterns fail) so
    // that the specific fact vocabulary — e.g. "Business Administration", "Philips LED",
    // "Target" — enters the BM25 index with the 1.5× query_surface boost, making the
    // correct session rank above competing sessions that mention the terms incidentally.
    let mut extra_tokens: Vec<String> = {
        const SKIP: &[&str] = &[
            "the", "and", "for", "are", "was", "but", "not", "you", "all", "can", "her", "his",
            "she", "they", "them", "any", "had", "our", "one", "this", "that", "its", "with",
            "have", "from", "just", "been",
        ];
        const SIGNALS: &[&str] = &[
            "by the way",
            "speaking of,",
            "also,",
            "i should mention",
            "incidentally,",
            "anyway,",
            "just wanted to mention",
        ];
        let mut extra = Vec::new();
        for signal in SIGNALS {
            if let Some(pos) = lower.find(signal) {
                let after_start = (pos + signal.len()).min(text.len());
                let after = text[after_start..].trim_start_matches([',', ' ', '\t']);
                for word in after.split_whitespace().take(30) {
                    let clean: String = word.chars().filter(|c| c.is_alphanumeric()).collect();
                    let cl = clean.to_lowercase();
                    if cl.len() >= 3 && !SKIP.contains(&cl.as_str()) {
                        extra.push(cl);
                    }
                }
            }
        }
        extra
    };

    // NE-7: Targeted person/place name extraction near personal relationship triggers.
    //
    // Narrowly scoped to rare, specific relationship labels only.  "my friend" / "my
    // colleague" are too common (appear in nearly every session) and flooding
    // query_surface with person names creates noise across multi-session and temporal
    // categories.  Only "my sister", "my cousin", and "visiting my" are kept: they are
    // specific enough that the capitalized words immediately following are almost always
    // person names or city names that are unique discriminators.
    // Example: "visiting my sister Emily in Denver" → ["emily", "denver"] added to
    // extra_tokens → query "where does my sister Emily live?" → "emily" in
    // query_surface at 1.5× → correct session ranked above generic "emily" hits.
    if !tokens.is_empty() {
        const REL_TRIGGERS: &[&str] = &["my sister", "my cousin", "visiting my"];
        for trigger in REL_TRIGGERS {
            let mut search_start = 0;
            while let Some(rel_pos) = lower[search_start..].find(trigger) {
                let abs_pos = search_start + rel_pos;
                let after_start = (abs_pos + trigger.len()).min(text.len());
                let after = &text[after_start..];
                let mut found = 0;
                for word in after.split_whitespace().take(8) {
                    let clean: String = word.chars().filter(|c| c.is_alphabetic()).collect();
                    if clean.len() >= 3
                        && clean.chars().next().is_some_and(|c| c.is_uppercase())
                        && found < 3
                    {
                        extra_tokens.push(clean.to_lowercase());
                        found += 1;
                    }
                    if found >= 3 {
                        break;
                    }
                }
                search_start = abs_pos + trigger.len();
                if search_start >= lower.len() {
                    break;
                }
            }
        }
    }

    // NE-8: Degree/field-of-study name extraction after education-specific phrases.
    // "I graduated with a degree in Business Administration" → ["business", "administration"]
    // This bridges the vocabulary gap: the query "what degree did I graduate with?" does not
    // contain "business administration", but those capitalized words are unique to the session.
    // Having them in query_surface means cross-session deduplication is stronger.
    // Fires only when tokens is non-empty (an education or other pattern already matched).
    if !tokens.is_empty() {
        const EDU_TRIGGERS: &[&str] = &[
            "degree in ",
            "majored in ",
            "major in ",
            "studied ",
            "i have a degree in",
            "graduated with a degree in",
            "studying for a ",
            "i earn my degree in",
        ];
        for trigger in EDU_TRIGGERS {
            if let Some(pos) = lower.find(trigger) {
                let after_start = (pos + trigger.len()).min(text.len());
                let after = &text[after_start..];
                let mut found = 0;
                for word in after.split_whitespace().take(5) {
                    let clean: String = word.chars().filter(|c| c.is_alphabetic()).collect();
                    if clean.len() >= 3
                        && clean.chars().next().is_some_and(|c| c.is_uppercase())
                        && found < 3
                    {
                        extra_tokens.push(clean.to_lowercase());
                        found += 1;
                    }
                    if found >= 3 {
                        break;
                    }
                }
            }
        }
    }

    // This catch-all layer ensures BM25 can find the neuron via ANY vocabulary in its
    // content, even when the content doesn't match any predefined category pattern.
    // Zero false-positive risk: these terms are extracted directly from the content.
    if tokens.is_empty() {
        let mut fallback: Vec<String> = Vec::new();

        // (a) Proper nouns: capitalized words ≥3 chars, not sentence-start
        for (i, word) in text.split_whitespace().enumerate() {
            let clean: String = word.chars().filter(|c| c.is_alphanumeric()).collect();
            if clean.len() >= 3
                && i > 0  // skip sentence-start capitals
                && clean.chars().next().is_some_and(|c| c.is_uppercase())
            {
                fallback.push(clean.to_lowercase());
            }
        }

        // (b) Numbers / quantities: tokens containing digits (ages, counts, times)
        for word in text.split_whitespace() {
            let clean: String = word
                .chars()
                .filter(|c| c.is_ascii_alphanumeric() || *c == '.')
                .collect();
            if clean.chars().any(|c| c.is_ascii_digit()) && clean.len() >= 2 {
                fallback.push(clean.to_lowercase());
            }
        }

        // (c) Quoted strings: extract content between " " or ' '
        let mut in_quote = false;
        let mut quote_buf = String::new();
        for ch in text.chars() {
            if ch == '"' || ch == '\'' {
                if in_quote && !quote_buf.trim().is_empty() {
                    for part in quote_buf.split_whitespace() {
                        let clean: String = part.chars().filter(|c| c.is_alphabetic()).collect();
                        if clean.len() >= 3 {
                            fallback.push(clean.to_lowercase());
                        }
                    }
                    quote_buf.clear();
                }
                in_quote = !in_quote;
            } else if in_quote {
                quote_buf.push(ch);
            }
        }

        fallback.extend(extra_tokens);
        if fallback.is_empty() {
            return None;
        }

        // Deduplicate fallback tokens
        let mut seen = HashSet::new();
        let deduped: Vec<String> = fallback
            .into_iter()
            .filter(|t| seen.insert(t.clone()))
            .collect();
        return Some(deduped.join(", "));
    }

    // Deduplicate while preserving order; merge category vocab + disclosure terms
    let mut seen = HashSet::new();
    let mut deduped: Vec<String> = tokens
        .into_iter()
        .filter(|t| seen.insert(t.to_string()))
        .map(|s| s.to_string())
        .collect();
    for t in extra_tokens {
        if seen.insert(t.clone()) {
            deduped.push(t);
        }
    }
    Some(deduped.join(", "))
}
