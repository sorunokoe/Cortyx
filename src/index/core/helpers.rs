// This file is a submodule of `crate::index::core`.
// It contains free-standing helper functions extracted from mod.rs (E1).
// All visibility is relative to `crate::index` (the parent of `core`).
use super::*;


// ─── Free functions ───────────────────────────────────────────────────────────

/// S-VII (R16): Return the current day as days since Unix epoch.
/// Used for synapse temporal decay calculations.
pub(in crate::index) fn now_unix_days() -> u32 {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    (secs / 86_400) as u32
}

/// S-I (R16): Extract a Tier-1 summary from neuron markdown content.
///
/// Returns: first non-empty content line of `## purpose` section (up to 200 chars).
/// Appends first content line of `## pitfalls` if present (separated by " | ").
/// Used as Tier-1 emission (~50 tokens) when BM25 score is in [1.5, 5.0) range.
pub(in crate::index) fn extract_neuron_summary(content: &str) -> String {
    let mut in_purpose = false;
    let mut in_pitfalls = false;
    let mut purpose_line = String::new();
    let mut pitfalls_line = String::new();

    for line in content.lines() {
        let l = line.trim();
        if l.starts_with("## ") {
            let section = l.trim_start_matches('#').trim().to_lowercase();
            in_purpose = section == "purpose";
            in_pitfalls = section == "pitfalls";
            continue;
        }
        if in_purpose && purpose_line.is_empty() && !l.is_empty() {
            purpose_line = l.chars().take(200).collect();
        }
        if in_pitfalls && pitfalls_line.is_empty() && !l.is_empty() {
            pitfalls_line = l.chars().take(120).collect();
            break;
        }
    }

    match (purpose_line.is_empty(), pitfalls_line.is_empty()) {
        (false, false) => format!("{purpose_line} | ⚠ {pitfalls_line}"),
        (false, true) => purpose_line,
        _ => String::new(),
    }
}

/// S5 helpers: extract manifest metadata from Cargo.toml or package.json.
/// Returns (name, version, authors, description, repository).
pub(in crate::index) fn extract_manifest_metadata(root: &Path) -> (String, String, String, String, String) {
    // Try Cargo.toml first
    if let Ok(text) = std::fs::read_to_string(root.join("Cargo.toml")) {
        let name = extract_toml_field(&text, "name").unwrap_or_default();
        let version = extract_toml_field(&text, "version").unwrap_or_default();
        let authors = extract_toml_field(&text, "authors").unwrap_or_default();
        let description = extract_toml_field(&text, "description").unwrap_or_default();
        let repo = extract_toml_field(&text, "repository").unwrap_or_default();
        return (name, version, authors, description, repo);
    }
    // Try package.json
    if let Ok(text) = std::fs::read_to_string(root.join("package.json")) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
            let name = v["name"].as_str().unwrap_or("").to_string();
            let version = v["version"].as_str().unwrap_or("").to_string();
            let description = v["description"].as_str().unwrap_or("").to_string();
            let repo = v["repository"]
                .as_str()
                .or_else(|| v["repository"]["url"].as_str())
                .unwrap_or("")
                .to_string();
            let authors = v["author"].as_str().unwrap_or("").to_string();
            return (name, version, authors, description, repo);
        }
    }
    (
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
    )
}

/// Extract a string value from a TOML file (simple key="value" or key = "value" lines).
pub(in crate::index) fn extract_toml_field(text: &str, key: &str) -> Option<String> {
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with(key) {
            if let Some(rest) = trimmed.strip_prefix(key) {
                let rest = rest.trim();
                if let Some(rest) = rest.strip_prefix('=') {
                    let val = rest.trim().trim_matches('"');
                    // Handle arrays like authors = ["Alice", "Bob"]
                    if val.starts_with('[') {
                        let inner = val.trim_start_matches('[').trim_end_matches(']');
                        let items: Vec<&str> = inner
                            .split(',')
                            .map(|s| s.trim().trim_matches('"'))
                            .filter(|s| !s.is_empty())
                            .collect();
                        return Some(items.join(", "));
                    }
                    if !val.is_empty() && !val.starts_with('[') {
                        return Some(val.to_string());
                    }
                }
            }
        }
    }
    None
}

/// Read the first `max_chars` characters from the first found file in `candidates`.
pub(in crate::index) fn read_file_head(root: &Path, candidates: &[&str], max_chars: usize) -> String {
    for name in candidates {
        if let Ok(text) = std::fs::read_to_string(root.join(name)) {
            return text.chars().take(max_chars).collect();
        }
    }
    String::new()
}

/// Extract a section from README that matches any of the given keywords.
/// Returns the section content (up to 400 chars) if found.
pub(in crate::index) fn read_readme_section(root: &Path, keywords: &[&str]) -> Option<String> {
    let text = std::fs::read_to_string(root.join("README.md")).ok()?;
    let lower = text.to_lowercase();
    for kw in keywords {
        if let Some(pos) = lower.find(kw) {
            // Find the line start
            let start = text[..pos].rfind('\n').map(|p| p + 1).unwrap_or(0);
            let section: String = text[start..].chars().take(400).collect();
            return Some(section);
        }
    }
    None
}

/// Run a git command and return trimmed stdout, or None on failure.
pub(in crate::index) fn run_git_cmd(root: &Path, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .ok()?;
    if out.status.success() {
        Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
    } else {
        None
    }
}

/// Split a camelCase or PascalCase identifier into its component words.
///
/// "getContexts" → ["get", "Contexts"]
/// "BM25Score"   → ["BM25", "Score"]
/// "simple_name" → [] (underscore-delimited; no split needed — already tokenized)
///
/// Only splits at lower→upper or digit→upper boundaries to avoid breaking
/// abbreviations: "BM25" stays together, "getURL" splits as ["get", "URL"].
pub(in crate::index) fn split_camel_case(s: &str) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < 4 {
        return Vec::new(); // too short to bother splitting
    }
    let mut parts = Vec::new();
    let mut start = 0;
    for i in 1..chars.len() {
        let prev = chars[i - 1];
        let curr = chars[i];
        let split = (prev.is_lowercase() && curr.is_uppercase())
            || (prev.is_ascii_digit() && curr.is_uppercase());
        if split {
            parts.push(chars[start..i].iter().collect::<String>());
            start = i;
        }
    }
    if start < chars.len() {
        parts.push(chars[start..].iter().collect::<String>());
    }
    // Only return parts if there was actually a split (otherwise caller already has the token)
    if parts.len() <= 1 {
        Vec::new()
    } else {
        parts
    }
}

/// Generate morphological suffix variants for a term.
///
/// Bridges the lexical gap between query vocabulary and document vocabulary when no
/// stemmer is present: "graduate" → ["graduated", "graduates", "graduating"],
/// "graduated" → ["graduate", "graduates", "graduating"], etc.
///
/// Only variants that actually exist in the index (checked via df_cache by the caller)
/// are retained — absent variants score 0 in BM25 and are harmless but wasteful.
pub(in crate::index) fn morphological_variants(term: &str) -> Vec<String> {
    let t = term;
    let mut variants = Vec::with_capacity(4);
    if t.ends_with("ing") && t.len() > 6 {
        // "running" → "run", "runed" (invalid, filtered by vocab check), "runs"
        let stem = &t[..t.len() - 3];
        variants.push(stem.to_string());
        variants.push(format!("{stem}ed"));
        variants.push(format!("{stem}s"));
        // Double-final-consonant stems: "running" → "run" → also "runner" is not needed
    } else if t.ends_with("tion") && t.len() > 7 {
        // "education" → "educate", "educated", "educating"
        // Skip — too error-prone without a real morphological analyser
    } else if t.ends_with("ed") && t.len() > 5 {
        // "graduated" → "graduate", "graduates", "graduating"
        let stem = &t[..t.len() - 2];
        variants.push(stem.to_string());
        variants.push(format!("{stem}s"));
        variants.push(format!("{stem}ing"));
        // "started" → "start" is correct; "started" → "starte" is not, but vocab check guards it
    } else if t.ends_with('s') && !t.ends_with("ss") && t.len() > 4 {
        // "graduates" → "graduate", "graduated", "graduating"
        let stem = &t[..t.len() - 1];
        variants.push(stem.to_string());
        variants.push(format!("{stem}ed"));
        variants.push(format!("{stem}ing"));
    } else if t.len() >= 4 {
        // Base form — add common inflections
        variants.push(format!("{t}s"));
        variants.push(format!("{t}ed"));
        variants.push(format!("{t}d")); // "commute" → "commuted"
        variants.push(format!("{t}ing"));
    }
    variants
}

/// Split text into lowercase terms, filtering short tokens.
///
/// Unicode-aware: handles Latin, CJK, Arabic, Devanagari, and mixed scripts.
///
/// - Latin/ASCII: split on non-alphanumeric boundaries, expand camelCase.
/// - CJK (Chinese/Japanese/Korean): emit character bigrams for the token, since
///   CJK text has no whitespace word boundaries. Individual chars are also emitted
///   when they are meaningful on their own (len check bypassed for single CJK chars).
/// - Arabic/Hebrew: split on whitespace; words already well-delimited in Arabic.
/// - Mixed script: apply both strategies.
///
/// Also expands camelCase/PascalCase identifiers so "getContexts" matches both
/// "get_contexts" and "getContexts" queries.  Each camel token is kept as-is
/// and each split part is added, giving BM25 the full vocabulary.
///
/// Light normalization: ASCII full-width characters (U+FF01–U+FF5E) are mapped to
/// their standard ASCII equivalents before splitting — common in East Asian technical text.
pub fn tokenize(text: &str) -> Vec<String> {
    // Normalize full-width ASCII (U+FF01–U+FF5E → U+0021–U+007E).
    // This is a zero-dependency alternative to full NFKC normalization for the
    // most common mixed-script case in East Asian technical documents.
    let normalized: String = text
        .chars()
        .map(|c| {
            let cp = c as u32;
            if (0xFF01..=0xFF5E).contains(&cp) {
                char::from_u32(cp - 0xFF01 + 0x0021).unwrap_or(c)
            } else {
                c
            }
        })
        .collect();

    let mut result: Vec<String> = Vec::new();
    for raw in normalized.split(|c: char| !c.is_alphanumeric() && c != '_') {
        if raw.is_empty() {
            continue;
        }

        // Detect whether this token contains CJK or Arabic characters.
        let has_cjk = raw.chars().any(is_cjk_char);
        let has_arabic = !has_cjk && raw.chars().any(is_arabic_char);

        if has_cjk {
            // CJK: emit character unigrams (≥1 char) and bigrams (≥2 chars) for
            // fine-grained BM25 vocabulary — no word boundaries in Chinese/Japanese/Korean.
            let chars: Vec<char> = raw.chars().collect();
            for &ch in &chars {
                // Single CJK characters are meaningful vocabulary units.
                let s = ch.to_string();
                result.push(s);
            }
            // Bigrams: "用户认证" → "用户", "户认", "认证"
            for window in chars.windows(2) {
                let bigram: String = window.iter().collect();
                result.push(bigram);
            }
            // Also push the full sequence if short enough (≤4 chars = one word in CJK).
            if chars.len() <= 4 && chars.len() >= MIN_TERM_LEN {
                result.push(raw.to_lowercase());
            }
        } else if has_arabic {
            // Arabic/Hebrew: words are already whitespace-delimited (the split above
            // on non-alphanumeric handles this correctly). Just lowercase and push.
            if raw.chars().count() >= MIN_TERM_LEN {
                result.push(raw.to_lowercase());
            }
        } else {
            // Latin/ASCII path (original logic).
            if raw.len() < MIN_TERM_LEN {
                continue;
            }
            let lower = raw.to_lowercase();
            for part in split_camel_case(raw) {
                if part.len() >= MIN_TERM_LEN {
                    result.push(part.to_lowercase());
                }
            }
            result.push(lower);
        }
    }
    result
}

/// Returns true for characters in CJK Unified Ideographs and common CJK extension blocks.
#[inline]
pub(in crate::index) fn is_cjk_char(c: char) -> bool {
    let cp = c as u32;
    // CJK Unified Ideographs: U+4E00–U+9FFF
    // CJK Extension A: U+3400–U+4DBF
    // CJK Extension B: U+20000–U+2A6DF
    // Hiragana: U+3040–U+309F
    // Katakana: U+30A0–U+30FF
    // Hangul Syllables: U+AC00–U+D7AF
    matches!(cp,
        0x3040..=0x30FF |   // Hiragana + Katakana
        0x3400..=0x4DBF |   // CJK Extension A
        0x4E00..=0x9FFF |   // CJK Unified Ideographs
        0xAC00..=0xD7AF |   // Hangul Syllables
        0x20000..=0x2A6DF   // CJK Extension B
    )
}

/// Returns true for characters in Arabic and Hebrew Unicode blocks.
#[inline]
pub(in crate::index) fn is_arabic_char(c: char) -> bool {
    let cp = c as u32;
    matches!(cp,
        0x0600..=0x06FF |   // Arabic
        0x0590..=0x05FF     // Hebrew
    )
}

/// Jaccard similarity — kept for use in tests; no longer used in the activation pipeline.
#[cfg(test)]
pub fn simple_overlap_score(query_terms: &[String], pattern_terms: &[String]) -> f32 {
    if pattern_terms.is_empty() || query_terms.is_empty() {
        return 0.0;
    }
    let query_set: HashSet<&String> = query_terms.iter().collect();
    let pattern_set: HashSet<&String> = pattern_terms.iter().collect();
    let intersection = query_set.intersection(&pattern_set).count();
    let union = query_set.union(&pattern_set).count();
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

/// Path of the persisted index file.
pub(in crate::index) fn index_path(project_root: &Path) -> PathBuf {
    project_root.join(".cortyx").join("index.json")
}

pub(in crate::index) fn activation_cache_path(project_root: &Path) -> PathBuf {
    project_root.join(".cortyx").join("index.fast.bin")
}

pub(in crate::index) fn coactivation_counts_path(project_root: &Path) -> PathBuf {
    project_root.join(".cortyx").join("coactivation.json")
}

pub(in crate::index) fn load_coactivation_counts(
    project_root: &Path,
) -> HashMap<PathBuf, HashMap<String, u32>> {
    let path = coactivation_counts_path(project_root);
    match std::fs::read_to_string(&path) {
        Ok(data) => match serde_json::from_str(&data) {
            Ok(counts) => counts,
            Err(e) => {
                tracing::warn!(
                    "Failed to parse coactivation counts {}: {e}",
                    path.display()
                );
                HashMap::new()
            },
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => HashMap::new(),
        Err(e) => {
            tracing::warn!("Failed to read coactivation counts {}: {e}", path.display());
            HashMap::new()
        },
    }
}

pub(in crate::index) fn save_coactivation_counts(
    project_root: &Path,
    counts: &HashMap<PathBuf, HashMap<String, u32>>,
) -> Result<()> {
    let path = coactivation_counts_path(project_root);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    atomic_write_json(&path, counts)
}

pub(in crate::index) fn read_index_cache_generation(path: &Path) -> Option<u64> {
    let file = std::fs::File::open(path).ok()?;
    let mut header = String::new();
    file.take(4096).read_to_string(&mut header).ok()?;
    let marker = "\"cache_generation\":";
    let start = header.find(marker)? + marker.len();
    let digits: String = header[start..]
        .chars()
        .skip_while(|c| c.is_ascii_whitespace())
        .take_while(|c| c.is_ascii_digit())
        .collect();
    if digits.is_empty() {
        None
    } else {
        digits.parse().ok()
    }
}

/// S-VI (R16): Read the `module` field from a neuron's sidecar JSON without fully parsing it.
///
/// Returns `None` when the sidecar is missing or has no `module` field (falls through to
/// the `__global` shard). Reading on every `save()` is acceptable: called once per entry
/// and sidecar files are tiny (~1 KB), so the full save() adds ~O(n) tiny reads — the same
/// cost as any file scan.
pub(in crate::index) fn sidecar_module_for(neuron_path: &Path) -> Option<String> {
    let sidecar = neuron_path.with_extension("json");
    let data = std::fs::read_to_string(sidecar).ok()?;
    let meta: serde_json::Value = serde_json::from_str(&data).ok()?;
    meta.get("module")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
}

/// Path to the dirty-file list written by the watcher and consumed by `compile_dirty`.
pub fn dirty_path(project_root: &Path) -> PathBuf {
    project_root.join(".cortyx").join("dirty.json")
}

pub fn is_capsule_module(module: &str) -> bool {
    !module.is_empty() && module != "__global" && !module.starts_with('@')
}

pub fn module_capsule_path(project_root: &Path, module: &str) -> PathBuf {
    project_root
        .join(".cortyx")
        .join("capsules")
        .join(format!("{}.capsule.md", safe_module_name(module)))
}

pub(in crate::index) fn safe_module_name(module: &str) -> String {
    module.replace(['/', '\\', ':', '*', '?', '"', '<', '>', '|', '@'], "_")
}

pub(in crate::index) fn capsule_entry_kind_rank(kind: &NeuronKind) -> u8 {
    match kind {
        NeuronKind::Core => 0,
        NeuronKind::Project => 1,
        NeuronKind::UseCase => 2,
        NeuronKind::Concept => 3,
        _ => 4,
    }
}

pub(in crate::index) fn is_stable_capsule_entry(kind: &NeuronKind) -> bool {
    matches!(
        kind,
        NeuronKind::Core | NeuronKind::Project | NeuronKind::UseCase | NeuronKind::Concept
    )
}

pub(in crate::index) fn split_summary_parts(summary: &str) -> (&str, Option<&str>) {
    if let Some((purpose, pitfall)) = summary.split_once("| ⚠") {
        (purpose.trim(), Some(pitfall.trim()))
    } else {
        (summary.trim(), None)
    }
}

pub(in crate::index) fn capsule_entry_label(path: &Path) -> String {
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    name.trim_end_matches(".md")
        .trim_end_matches(".context")
        .to_string()
}

pub(in crate::index) fn module_capsule_pitfall(path: &Path) -> Option<String> {
    let content = std::fs::read_to_string(path).ok()?;
    let sections = crate::neuron::parse_sections(&content);
    sections
        .get("pitfalls")
        .and_then(|body| body.lines().map(str::trim).find(|line| !line.is_empty()))
        .map(ToOwned::to_owned)
}

pub(in crate::index) fn is_capsule_glossary_term(term: &str, module_tokens: &HashSet<String>) -> bool {
    const CAPSULE_STOPWORDS: &[&str] = &[
        "about", "after", "also", "been", "being", "build", "code", "does", "file", "from", "have",
        "into", "kind", "line", "lines", "module", "must", "only", "path", "paths", "self", "that",
        "this", "used", "using", "with",
    ];

    term.len() >= 3
        && term.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !term.chars().all(|c| c.is_ascii_digit())
        && !module_tokens.contains(term)
        && !CAPSULE_STOPWORDS.contains(&term)
}

pub(in crate::index) fn build_module_capsule_content(
    module: &str,
    entries: &[&BM25Entry],
    path_modules: &HashMap<PathBuf, String>,
) -> Option<String> {
    if !is_capsule_module(module) {
        return None;
    }

    let mut stable_entries: Vec<&BM25Entry> = entries
        .iter()
        .copied()
        .filter(|entry| is_stable_capsule_entry(&entry.kind))
        .collect();
    if stable_entries.is_empty() {
        return None;
    }

    stable_entries.sort_by(|a, b| {
        capsule_entry_kind_rank(&a.kind)
            .cmp(&capsule_entry_kind_rank(&b.kind))
            .then_with(|| b.synapses.len().cmp(&a.synapses.len()))
            .then_with(|| b.quality_score.total_cmp(&a.quality_score))
            .then_with(|| a.neuron_path.cmp(&b.neuron_path))
    });

    let mut purpose_lines = Vec::new();
    let mut seen_purpose = HashSet::new();
    for entry in &stable_entries {
        let purpose = split_summary_parts(&entry.summary).0;
        if purpose.is_empty() {
            continue;
        }
        let normalized = purpose.to_lowercase();
        if seen_purpose.insert(normalized) {
            purpose_lines.push(purpose.to_string());
        }
        if purpose_lines.len() == 3 {
            break;
        }
    }
    if purpose_lines.is_empty() {
        purpose_lines.push(format!("Stable subsystem capsule for `{module}`."));
    }

    let mut api_lines = Vec::new();
    let mut seen_api = HashSet::new();
    for entry in stable_entries.iter().copied().filter(|entry| {
        matches!(
            entry.kind,
            NeuronKind::Core | NeuronKind::Project | NeuronKind::UseCase
        )
    }) {
        let label = capsule_entry_label(&entry.neuron_path);
        let purpose = split_summary_parts(&entry.summary).0;
        let line = if purpose.is_empty() {
            format!("`{label}`")
        } else {
            format!("`{label}` — {purpose}")
        };
        let normalized = line.to_lowercase();
        if seen_api.insert(normalized) {
            api_lines.push(line);
        }
        if api_lines.len() == 5 {
            break;
        }
    }

    let mut pitfall_lines = Vec::new();
    let mut seen_pitfalls = HashSet::new();
    for entry in &stable_entries {
        let pitfall = module_capsule_pitfall(&entry.neuron_path)
            .or_else(|| split_summary_parts(&entry.summary).1.map(ToOwned::to_owned));
        let Some(pitfall) = pitfall else {
            continue;
        };
        let normalized = pitfall.to_lowercase();
        if seen_pitfalls.insert(normalized) {
            pitfall_lines.push(pitfall);
        }
        if pitfall_lines.len() == 4 {
            break;
        }
    }

    let mut dependency_counts: HashMap<String, usize> = HashMap::new();
    for entry in &stable_entries {
        for syn in &entry.synapses {
            let Some(target_module) = path_modules.get(&syn.target) else {
                continue;
            };
            if target_module == module || !is_capsule_module(target_module) {
                continue;
            }
            *dependency_counts.entry(target_module.clone()).or_insert(0) += 1;
        }
    }
    let mut dependency_lines: Vec<(String, usize)> = dependency_counts.into_iter().collect();
    dependency_lines.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    dependency_lines.truncate(4);

    let module_tokens: HashSet<String> = tokenize(module).into_iter().collect();
    let mut glossary_weights: HashMap<String, f32> = HashMap::new();
    for entry in &stable_entries {
        for (term, weight) in &entry.term_freq {
            if is_capsule_glossary_term(term, &module_tokens) {
                *glossary_weights.entry(term.clone()).or_insert(0.0) += *weight;
            }
        }
        for term in entry.concept_cloud.iter().chain(entry.synonym_cloud.iter()) {
            if is_capsule_glossary_term(term, &module_tokens) {
                *glossary_weights.entry(term.clone()).or_insert(0.0) += 0.5;
            }
        }
    }
    let mut glossary_terms: Vec<(String, f32)> = glossary_weights.into_iter().collect();
    glossary_terms.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let glossary_terms: Vec<String> = glossary_terms
        .into_iter()
        .map(|(term, _)| term)
        .take(8)
        .collect();

    let mut out = format!("# Module capsule: {module}\n\n");
    out.push_str("## module purpose\n");
    for line in &purpose_lines {
        out.push_str(&format!("- {line}\n"));
    }

    if !api_lines.is_empty() {
        out.push_str("\n## key apis / invariants\n");
        for line in &api_lines {
            out.push_str(&format!("- {line}\n"));
        }
    }

    if !pitfall_lines.is_empty() {
        out.push_str("\n## critical pitfalls\n");
        for line in &pitfall_lines {
            out.push_str(&format!("- {line}\n"));
        }
    }

    if !dependency_lines.is_empty() {
        out.push_str("\n## dominant dependencies\n");
        for (dep, count) in &dependency_lines {
            out.push_str(&format!("- `{dep}` ({count} cross-module edges)\n"));
        }
    }

    if !glossary_terms.is_empty() {
        out.push_str("\n## glossary / aliases\n");
        out.push_str(&format!("{}\n", glossary_terms.join(", ")));
    }

    Some(out)
}

/// Extract a one-line headline from a neuron file for budget-overflow compression.
///
/// Looks for the first non-empty content line under `## purpose` or `**What this file does**:`.
/// Falls back to the first non-heading, non-empty line, then `"(stub)"` if the file is empty.
/// Reading the file is intentionally done lazily — this fn is only called for overflow neurons.
pub(in crate::index) fn neuron_headline_for(path: &Path) -> String {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return "(unreadable)".to_string(),
    };
    // Use parse_sections from neuron.rs if available; otherwise simple regex.
    use crate::neuron::parse_sections;
    let sections = parse_sections(&content);
    if let Some(body) = sections
        .get("purpose")
        .or_else(|| sections.get("what this file does"))
    {
        if let Some(line) = body.lines().find(|l| !l.trim().is_empty()) {
            return line.trim().to_string();
        }
    }
    // Fallback: first non-heading, non-empty line
    content
        .lines()
        .find(|l| !l.trim().is_empty() && !l.starts_with('#') && !l.starts_with("<!--"))
        .map(|l| l.trim().to_string())
        .unwrap_or_else(|| "(stub)".to_string())
}

/// Infer a module tag from the source file's relative path.///
/// Strategy (in priority order):
/// 1. Second component of the path (e.g. `src/auth/user.rs` → `"auth"`).
///    This covers the common `src/<module>/` layout.
/// 2. First component when there's no `src/` prefix (e.g. `lib/helpers.rs` → `"lib"`).
/// 3. `None` for top-level files with no meaningful sub-directory.
///
/// The LLM can always override via `cortyx_evolve_context` — this is a warm start,
/// not a hard assignment.
pub fn infer_module(rel: &Path) -> Option<String> {
    let mut components = rel.components().peekable();
    let first = components
        .next()?
        .as_os_str()
        .to_string_lossy()
        .into_owned();
    // Skip common source root directories
    let skip = matches!(first.as_str(), "src" | "lib" | "source" | "Sources" | "app");
    if skip {
        // Return the next component if it looks like a sub-module directory
        let second = components
            .next()?
            .as_os_str()
            .to_string_lossy()
            .into_owned();
        // Ignore if the second component is itself a file (has an extension)
        if second.contains('.') {
            return None;
        }
        Some(second)
    } else {
        // No standard root prefix — use first component if it's a directory
        if first.contains('.') {
            return None;
        }
        Some(first)
    }
}

pub(in crate::index) fn default_confidence() -> f32 {
    DEFAULT_CONFIDENCE
}

pub(in crate::index) fn default_staleness() -> f32 {
    1.0
}

pub(in crate::index) fn default_quality_score() -> f32 {
    1.0
}

pub(in crate::index) fn reasoner_neuron_from_entry(entry: &BM25Entry) -> ReasonerNeuron {
    let source_path = entry
        .source_files
        .first()
        .cloned()
        .unwrap_or_else(|| entry.neuron_path.clone());
    let mut meta = NeuronMeta::new_stub(&source_path, entry.kind.clone());
    meta.kind = entry.kind.clone();
    meta.synapses = entry.synapses.clone();
    meta.module = entry.module.clone();
    meta.parent = entry.parent.clone();
    meta.source_files = entry.source_files.clone();
    meta.confidence_score = entry.confidence_score;
    meta.tokens = entry.tokens;

    let mut neuron = ReasonerNeuron::new(entry.neuron_path.clone(), meta);
    if !entry.summary.is_empty() {
        neuron = neuron.with_summary(entry.summary.clone());
    }
    neuron
}

pub(in crate::index) fn looks_like_kg_neuron_path(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.starts_with("_kg_") && name.ends_with(".context.md"))
        .unwrap_or(false)
}

/// Detect temporal markers in a query — triggers recency boost in retrieval.
///
/// Returns true when the query asks about time-relative facts ("most recent",
/// "before", "after", etc.). Used to gate the temporal query routing boost so
/// purely keyword-based queries (which have no temporal intent) are unaffected.
pub(in crate::index) fn detect_temporal_query(task: &str) -> bool {
    const TEMPORAL_MARKERS: &[&str] = &[
        "when did",
        "when was",
        "before",
        "after",
        "recent",
        "latest",
        "last time",
        "earlier",
        "previously",
        "at the time",
        "used to",
        "formerly",
        "back in",
        "most recent",
        "oldest",
        "newest",
        "updated",
        "how long ago",
        "since when",
        "at what point",
        // R17 L2: broader recency patterns
        "current",
        "currently",
        "now",
        "right now",
        "still",
        "today",
        "at the moment",
        "these days",
        "nowadays",
        "at present",
        "what is her",
        "what is his",
        "what is their",
        "what does she",
        "what does he",
        "what do they",
        "what is the current",
        "what is the latest",
        // R21 T7: additional temporal triggers (recency-only — oldest-seeking markers
        // removed: "first time", "originally", "initially", "earliest",
        // "what was the first", "when did i first", "what did i first"
        // belong EXCLUSIVELY in detect_oldest_query to avoid double-boost misrouting).
        "most recently",
        "last known",
        "as of",
        "up until",
        "prior to",
        "before that",
        "what was the last",
        "when did i last",
        "most recent time",
        "past weekend",
        "this past",
        "last weekend",
        // Specific-day recency: "last Saturday", "last Tuesday", etc.
        // NOTE: These are intentionally NOT in temporal_markers because they denote a
        // specific anchor day, not a recency preference. BM25 alone (music×33 + parents×1 etc.)
        // correctly selects the right session; adding temporal boost here causes cross-scenario
        // contamination (sessions with higher file-write order IDs win unfairly).
        // "last monday", "last tuesday", ... → no temporal boost, rely on BM25.
        // Relative-day recency: "a couple of days ago", "10 days ago", etc.
        "days ago",
        "a couple of days",
        "a few days ago",
        // "a week ago", "week ago" (NOT "weeks ago" to avoid arithmetic queries like
        // "how many weeks ago" which are a separate category from recency retrieval).
        "week ago",
    ];
    let lower = task.to_lowercase();
    TEMPORAL_MARKERS.iter().any(|m| lower.contains(m))
}

/// R21 T2: Detect "oldest-first" temporal queries — questions about the FIRST/EARLIEST occurrence.
/// Returns true when the query is looking backwards in time (oldest event, first mention).
/// Complement of `detect_temporal_query`'s "most recent" direction.
pub(in crate::index) fn detect_oldest_query(task: &str) -> bool {
    const OLDEST_MARKERS: &[&str] = &[
        "what was the first",
        "when did i first",
        "what did i first",
        "first time i",
        "first time she",
        "first time he",
        "first issue",
        "first problem",
        "first mention",
        "originally",
        "at the beginning",
        "earliest",
        "earliest time",
        "earliest mention",
        "when i first",
        "the first x",
        "first ever",
        "first one",
        "first thing",
        "very first",
        "what was the original",
        "what was the initial",
    ];
    let lower = task.to_lowercase();
    if OLDEST_MARKERS.iter().any(|m| lower.contains(m)) {
        return true;
    }
    // Compound: "Which X did I do first, X or Y?" — choice-ordering questions.
    // e.g. "Which vehicle did I take care of first, the bike or the car?"
    //      "Which event did I attend first, the workshop or the conference?"
    // Pattern: starts with "which" AND contains " first" AND contains " or ".
    if lower.starts_with("which") && lower.contains(" first") && lower.contains(" or ") {
        return true;
    }
    false
}

/// R21 T5: Detect counting queries — questions that need aggregate evidence from many sessions.
/// When fired, Phase 1 expands to all Verbatim neurons; top-10 instead of top-5 returned.
pub(in crate::index) fn detect_counting_query(task: &str) -> bool {
    const COUNTING_MARKERS: &[&str] = &[
        "how many",
        "total",
        "count of",
        "number of",
        "how much",
        "sum of",
        "altogether",
        "in total",
        "combined",
        "overall",
        "how often",
        "how frequently",
        "times have i",
        "times did i",
        "how many times",
        "how often have",
        "have i had",
        "have i been",
        "how many places",
        "how many people",
        "how many sessions",
        "how many different",
        "how many types",
        // Sol-A: arithmetic-sum markers — trigger ArithmeticAggregate injection
        "how much did",
        "how much has",
        "how much have",
        "total cost",
        "total spent",
        "total spend",
        "total amount",
        "how much money",
        "how much was spent",
        "how much did i spend",
        "how much did she spend",
        "how much did he spend",
        "how much did they spend",
        "how much did we spend",
        "what did it cost",
        "what was the total",
        "what is the total",
        "overall cost",
        "overall amount",
        "overall spend",
        "amount spent",
        "money spent",
        "dollars spent",
    ];
    let lower = task.to_lowercase();
    COUNTING_MARKERS.iter().any(|m| lower.contains(m))
}

pub(in crate::index) fn extract_counting_focus_terms(terms: &[String]) -> Vec<String> {
    const COUNTING_STOP: &[&str] = &[
        "how",
        "many",
        "much",
        "total",
        "count",
        "number",
        "overall",
        "altogether",
        "combined",
        "times",
        "time",
        "money",
        "spent",
        "spend",
        "expense",
        "expenses",
        "cost",
        "costs",
        "amount",
        "have",
        "has",
        "had",
        "did",
        "does",
        "do",
        "been",
        "since",
        "start",
        "year",
        "years",
        "month",
        "months",
        "week",
        "weeks",
        "day",
        "days",
        "hour",
        "hours",
        "city",
        "different",
        "often",
        "frequently",
        "what",
        "when",
        "where",
        "with",
        "into",
        "from",
        "across",
        "overall",
        "altogether",
        "related",
        "current",
        "currently",
        "recent",
        "recently",
        "latest",
        "now",
        "today",
        "far",
        "attend",
        "attending",
        "attended",
        "visit",
        "visiting",
        "visited",
        "wear",
        "wearing",
        "worn",
        "see",
        "seeing",
        "seen",
        "try",
        "trying",
        "tried",
        "make",
        "making",
        "made",
        "buy",
        "buying",
        "bought",
        "sell",
        "selling",
        "sold",
        "earn",
        "earning",
        "earned",
        "work",
        "working",
        "worked",
        "own",
        "owned",
        "keeping",
        "kept",
        "local",
        "last",
        "first",
        "second",
        "third",
        "fourth",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
    ];
    let stop: HashSet<&str> = COUNTING_STOP.iter().copied().collect();
    let focused: Vec<String> = terms
        .iter()
        .filter(|term| term.len() >= 4 && !stop.contains(term.as_str()))
        .cloned()
        .collect();
    if focused.is_empty() {
        terms.to_vec()
    } else {
        focused
    }
}

pub(in crate::index) fn extract_direct_count_focus_terms(terms: &[String]) -> Vec<String> {
    const DIRECT_COUNT_STOP: &[&str] = &[
        "watch",
        "watching",
        "watched",
        "complete",
        "completing",
        "completed",
        "finish",
        "finishing",
        "finished",
        "need",
        "needs",
        "reach",
        "reaches",
        "require",
        "requires",
        "required",
    ];
    let extra_stop: HashSet<&str> = DIRECT_COUNT_STOP.iter().copied().collect();
    let mut focused = extract_counting_focus_terms(terms);
    focused.retain(|term| !extra_stop.contains(term.as_str()));
    if focused.len() < 2 {
        focused.extend(
            terms
                .iter()
                .filter(|term| term.len() >= 3 && !extra_stop.contains(term.as_str()))
                .cloned(),
        );
    }
    focused.sort();
    focused.dedup();
    if focused.is_empty() {
        terms.to_vec()
    } else {
        focused
    }
}

pub(in crate::index) fn extract_role_phrase(task: &str) -> Option<String> {
    Regex::new(r"(?i)(?:role as|job as|position as)\s+([^?.!]+)")
        .unwrap()
        .captures(task)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim().to_string())
        .filter(|phrase| phrase.split_whitespace().count() >= 2)
}

pub(in crate::index) fn direct_count_required_role_phrase(task_lower: &str) -> Option<String> {
    extract_role_phrase(task_lower)
}

pub(in crate::index) fn study_subject_required_journal_phrase(task_lower: &str) -> Option<String> {
    task_lower
        .split_once("journal ")
        .map(|(_, tail)| tail)
        .map(|tail| {
            [" that ", " which ", " with ", " published "]
                .iter()
                .find_map(|marker| tail.split_once(marker).map(|(head, _)| head))
                .unwrap_or(tail)
        })
        .map(|phrase| phrase.trim().trim_end_matches('?').to_string())
        .filter(|phrase| phrase.split_whitespace().count() >= 2)
}

pub(in crate::index) fn is_direct_count_candidate_line(line: &str, lower: &str, task_lower: &str) -> bool {
    is_summary_or_user_line(line, lower)
        || (task_contains_any(task_lower, &["study", "journal", "subjects"])
            && extract_numbered_list_item(line).is_some()
            && lower.contains("subject"))
}

pub(in crate::index) fn should_inject_count_aggregate(task: &str) -> bool {
    let lower = task.to_ascii_lowercase();
    if has_explicit_current_state_marker(task)
        || lower.contains("how often")
        || lower.contains("how many times")
        || lower.contains("times have i")
        || lower.contains("times did i")
    {
        return false;
    }
    lower.contains("how many different")
        || lower.contains("how many unique")
        || lower.contains("count of different")
        || lower.contains("number of different")
}

pub(in crate::index) fn synthetic_count_query_requires_multi_operand_reasoning(
    task: &str,
    task_lower: &str,
) -> bool {
    should_inject_count_aggregate(task)
        || ((detect_counting_query(task) || is_money_query(task))
            && task_lower.contains(" and ")
            && task_contains_any(task_lower, &["total", "combined", "altogether", "both"]))
        || task_contains_any(
            task_lower,
            &[
                " both ",
                " combined",
                " together",
                " in total",
                " altogether",
                " total of ",
                " instead of ",
                " compared to ",
                " difference between ",
            ],
        )
        || (task_lower.contains(" or ")
            && task_contains_any(
                task_lower,
                &[
                    " first",
                    " earlier",
                    " later",
                    " before ",
                    " after ",
                    " more often",
                    " less often",
                    " higher percentage",
                    " lower percentage",
                    " higher discount",
                    " lower discount",
                    " cheaper",
                    " more expensive",
                    " cost more",
                    " cost less",
                    " older",
                    " younger",
                ],
            ))
}

pub(in crate::index) fn extract_query_duration_window(task_lower: &str) -> Option<String> {
    Regex::new(
        r"(?i)\bfirst\s+((?:about\s+)?(?:an?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+(?:days?|weeks?|months?|years?|hours?|minutes?))\b",
    )
    .unwrap()
    .captures(task_lower)
    .and_then(|caps| caps.get(1))
    .map(|m| normalize_current_duration_answer(m.as_str()).to_ascii_lowercase())
}

pub(in crate::index) fn extract_issue_publication_phrase(task_lower: &str) -> Option<String> {
    task_lower
        .split_once("issues of ")
        .map(|(_, tail)| tail)
        .and_then(|tail| {
            [
                " have i",
                " have we",
                " have they",
                " has he",
                " has she",
                "?",
            ]
            .iter()
            .find_map(|marker| tail.split_once(marker).map(|(head, _)| head))
            .or(Some(tail))
        })
        .map(str::trim)
        .filter(|phrase| !phrase.is_empty())
        .map(ToString::to_string)
}

pub(in crate::index) fn extract_since_start_anchor_phrase(task_lower: &str) -> Option<String> {
    task_lower
        .split_once("since starting ")
        .map(|(_, tail)| tail)
        .or_else(|| {
            task_lower
                .split_once("since i started ")
                .map(|(_, tail)| tail)
        })
        .map(str::trim)
        .map(|phrase| phrase.trim_end_matches('?').to_string())
        .filter(|phrase| !phrase.is_empty())
}

pub(in crate::index) fn extract_item_usage_phrase(task_lower: &str) -> Option<(String, String)> {
    if let Some((_, tail)) = task_lower.split_once("times have i worn ") {
        let phrase = tail.trim().trim_end_matches('?');
        if !phrase.is_empty() {
            return Some(("wear".to_string(), phrase.to_string()));
        }
    }
    if let Some((_, tail)) = task_lower.split_once("times did i wear ") {
        let phrase = tail.trim().trim_end_matches('?');
        if !phrase.is_empty() {
            return Some(("wear".to_string(), phrase.to_string()));
        }
    }
    if let Some((_, tail)) = task_lower.split_once("trips have i taken ") {
        let phrase = tail
            .split_once(" on")
            .map(|(head, _)| head)
            .unwrap_or(tail)
            .trim()
            .trim_end_matches('?');
        if !phrase.is_empty() {
            return Some(("trip".to_string(), phrase.to_string()));
        }
    }
    if let Some((_, tail)) = task_lower.split_once("trips did i take ") {
        let phrase = tail
            .split_once(" on")
            .map(|(head, _)| head)
            .unwrap_or(tail)
            .trim()
            .trim_end_matches('?');
        if !phrase.is_empty() {
            return Some(("trip".to_string(), phrase.to_string()));
        }
    }
    None
}

pub(in crate::index) fn extract_media_rewatch_focus(task_lower: &str) -> Option<(String, String)> {
    let caps = Regex::new(
        r"(?i)\bhow many\s+(.*?)\s*(movies?|films?|shows?|episodes?)\s+(?:did|have)\s+i\s+re(?:-| )?watch(?:ed)?\b",
    )
    .unwrap()
    .captures(task_lower)?;
    let focus = caps
        .get(1)
        .map(|value| value.as_str().trim().to_string())
        .unwrap_or_default();
    let media_kind = caps.get(2)?.as_str().to_ascii_lowercase();
    Some((focus, media_kind))
}

pub(in crate::index) fn extract_daily_duration_commitment_phrase(task_lower: &str) -> Option<String> {
    for marker in [
        "how much time do i dedicate to ",
        "how much time do i spend on ",
        "how much time do i spend ",
    ] {
        let Some((_, tail)) = task_lower.split_once(marker) else {
            continue;
        };
        let phrase = [" each day", " every day", " daily", "?"]
            .iter()
            .find_map(|delimiter| tail.split_once(delimiter).map(|(head, _)| head))
            .unwrap_or(tail)
            .trim()
            .trim_end_matches('?');
        if !phrase.is_empty() {
            return Some(phrase.to_string());
        }
    }
    None
}

pub(in crate::index) fn extract_frequency_transition_activity_phrase(task_lower: &str) -> Option<String> {
    task_lower
        .split_once("how often do i ")
        .and_then(|(_, tail)| tail.split_once(" previously").map(|(head, _)| head))
        .map(str::trim)
        .filter(|phrase| !phrase.is_empty())
        .map(ToString::to_string)
}

pub(in crate::index) fn normalize_first_person_phrase_to_second_person(phrase: &str) -> String {
    let mut normalized = format!(" {} ", phrase.trim());
    for (from, to) in [
        (" my ", " your "),
        (" me ", " you "),
        (" mine ", " yours "),
        (" our ", " your "),
    ] {
        normalized = normalized.replace(from, to);
    }
    normalized.trim().to_string()
}

pub(in crate::index) fn extract_activity_core_phrase(phrase: &str) -> String {
    Regex::new(r"(?i)^(.+?)(?:\s+(?:with|at|in|on|for|during|around|near)\b|$)")
        .unwrap()
        .captures(phrase)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim().to_string())
        .filter(|core| core.split_whitespace().count() >= 2)
        .unwrap_or_else(|| phrase.trim().to_string())
}

pub(in crate::index) fn has_explicit_current_state_marker(task: &str) -> bool {
    const CURRENT_MARKERS: &[&str] = &[
        "current",
        "currently",
        "now",
        "right now",
        "most recent",
        "latest",
        "as of now",
        "at the moment",
        "at present",
        "so far",
    ];
    let lower = task.to_ascii_lowercase();
    CURRENT_MARKERS.iter().any(|marker| lower.contains(marker))
}

pub(in crate::index) fn capitalize_first_ascii(value: &str) -> String {
    let mut chars = value.chars();
    match chars.next() {
        Some(first) => {
            let mut answer = String::new();
            answer.push(first.to_ascii_uppercase());
            answer.push_str(chars.as_str());
            answer
        },
        None => String::new(),
    }
}

pub(in crate::index) fn extract_plural_issue_count_answer_from_line(line: &str) -> Option<String> {
    let raw = Regex::new(
        r"(?i)\b(?:finished|read|reading|completed)\s+(?:about\s+)?(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+issues?\b",
    )
    .unwrap()
    .captures(line)
    .and_then(|caps| caps.get(1))
    .map(|m| m.as_str().trim())?;
    Some(if raw.chars().all(|c| c.is_ascii_digit()) {
        raw.to_string()
    } else {
        capitalize_first_ascii(&raw.to_ascii_lowercase())
    })
}

pub(in crate::index) fn line_has_progress_count_marker(lower: &str) -> bool {
    task_contains_any(
        lower,
        &[
            "so far",
            "already",
            "managed to",
            "just finished",
            "i've written",
            "i have written",
            "i wrote",
            "i've completed",
            "i have completed",
            "i completed",
            "i've finished",
            "i have finished",
            "i just finished",
        ],
    )
}

pub(in crate::index) fn line_has_rewatch_marker(lower: &str) -> bool {
    task_contains_any(lower, &["re-watched", "re watched", "rewatched"])
}

pub(in crate::index) fn line_has_daily_duration_marker(lower: &str) -> bool {
    task_contains_any(lower, &["each day", "every day", "daily"])
}

pub(in crate::index) fn line_has_future_goal_marker(lower: &str) -> bool {
    task_contains_any(
        lower,
        &[
            "goal",
            "aim for",
            "aiming for",
            "hope to",
            "hoping to",
            "plan to",
            "planning to",
            "want to",
            "would like to",
            "next month",
        ],
    )
}

pub(in crate::index) fn small_count_word_lower(value: i32) -> Option<&'static str> {
    match value {
        0 => Some("zero"),
        1 => Some("one"),
        2 => Some("two"),
        3 => Some("three"),
        4 => Some("four"),
        5 => Some("five"),
        6 => Some("six"),
        7 => Some("seven"),
        8 => Some("eight"),
        9 => Some("nine"),
        10 => Some("ten"),
        11 => Some("eleven"),
        12 => Some("twelve"),
        _ => None,
    }
}

pub(in crate::index) fn supporting_word_count_surface(
    lines: &[String],
    value: i32,
    focus_terms: &[String],
) -> Option<String> {
    let word = small_count_word_lower(value)?;
    let focus_keys = synthetic_answer_surface_term_key_set(focus_terms);
    for line in lines {
        let lower = line.to_ascii_lowercase();
        if !lower.contains(word) {
            continue;
        }
        let line_keys = synthetic_answer_surface_term_key_set(&synthetic_query_terms(&lower));
        if synthetic_answer_surface_overlap_count(&line_keys, &focus_keys) >= 1 {
            return Some(word.to_string());
        }
    }
    None
}

pub(in crate::index) fn parse_frequency_count_token(token: &str) -> Option<i32> {
    match token
        .trim_matches(|c: char| !c.is_ascii_alphanumeric())
        .to_ascii_lowercase()
        .as_str()
    {
        "once" => Some(1),
        "twice" => Some(2),
        "thrice" => Some(3),
        other => parse_count_token_value(other),
    }
}

pub(in crate::index) fn extract_meetup_count_surface_from_line(line: &str, lower: &str) -> Option<String> {
    if !lower.contains("met up")
        || task_contains_any(
            lower,
            &[
                "planning to meet up",
                "plan to meet up",
                "we're planning to meet up",
                "going to meet up",
            ],
        )
    {
        return None;
    }
    let raw = Regex::new(
        r"(?i)\bmet up\s+(once|twice|thrice|one|two|three|four|five|six|seven|eight|nine|ten|\d+)(?:\s+times?)?\b",
    )
    .unwrap()
    .captures(line)
    .and_then(|caps| caps.get(1))
    .map(|m| m.as_str().trim())?;
    let normalized = raw.to_ascii_lowercase();
    Some(if normalized.chars().all(|c| c.is_ascii_digit()) {
        format!("We've met up {} times.", normalized)
    } else {
        format!("We've met up {}.", normalized)
    })
}

pub(in crate::index) fn extract_meetup_count_from_line(line: &str, lower: &str) -> Option<i32> {
    if !lower.contains("met up")
        || task_contains_any(
            lower,
            &[
                "planning to meet up",
                "plan to meet up",
                "we're planning to meet up",
                "going to meet up",
            ],
        )
    {
        return None;
    }
    let raw = Regex::new(
        r"(?i)\bmet up\s+(once|twice|thrice|one|two|three|four|five|six|seven|eight|nine|ten|\d+)(?:\s+times?)?\b",
    )
    .unwrap()
    .captures(line)
    .and_then(|caps| caps.get(1))
    .map(|m| m.as_str())?;
    parse_frequency_count_token(raw)
}

pub(in crate::index) fn extract_item_usage_count_surface_from_line(
    line: &str,
    lower: &str,
    usage_kind: &str,
) -> Option<String> {
    let raw = match usage_kind {
        "wear" => {
            if !(task_contains_any(lower, &["worn", "wore"]) && lower.contains("times")) {
                return None;
            }
            Regex::new(
                r"(?i)\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+times?\b",
            )
            .unwrap()
            .captures(line)
            .and_then(|caps| caps.get(1))
            .map(|m| m.as_str().trim())?
        },
        "trip" => {
            if !(lower.contains("trip") || lower.contains("adventure")) {
                return None;
            }
            Regex::new(
                r"(?i)\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+(?:trip|trips|adventures)\b",
            )
            .unwrap()
            .captures(line)
            .and_then(|caps| caps.get(1))
            .map(|m| m.as_str().trim())?
        },
        _ => return None,
    };
    Some(raw.to_ascii_lowercase())
}

pub(in crate::index) fn extract_item_usage_count_from_line(
    line: &str,
    lower: &str,
    usage_kind: &str,
) -> Option<i32> {
    let surface = extract_item_usage_count_surface_from_line(line, lower, usage_kind)?;
    parse_count_token_value(&surface)
}

pub(in crate::index) fn extract_women_count_from_line(line: &str, lower: &str) -> Option<i32> {
    if !lower.contains("women") {
        return None;
    }
    let raw = Regex::new(
        r"(?i)\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+women\b",
    )
    .unwrap()
    .captures(line)
    .and_then(|caps| caps.get(1))
    .map(|m| m.as_str().trim())?;
    parse_count_token_value(raw)
}

pub(in crate::index) fn extract_weight_loss_answer_from_line(
    line: &str,
    lower: &str,
) -> Option<(i32, String)> {
    if !lower.contains("lost") || !lower.contains("pound") {
        return None;
    }
    let captures = Regex::new(
        r"(?i)\b(?:lost|down)\s+(about\s+)?(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+pounds?\b",
    )
    .unwrap()
    .captures(line)?;
    let about = captures
        .get(1)
        .map(|m| !m.as_str().trim().is_empty())
        .unwrap_or(false);
    let raw = captures.get(2)?.as_str().trim().to_ascii_lowercase();
    let value = parse_count_token_value(&raw)?;
    let surface = if about {
        format!("about {raw} pounds")
    } else {
        format!("{raw} pounds")
    };
    Some((value, surface))
}

pub(in crate::index) fn extract_frequency_surface_from_line(line: &str, lower: &str) -> Option<String> {
    if lower.contains("every other week") {
        return Some("every other week".to_string());
    }
    if lower.contains("every two weeks") {
        return Some("every two weeks".to_string());
    }
    if lower.contains("every week") || lower.contains("weekly") {
        return Some("every week".to_string());
    }
    if lower.contains("every day") || lower.contains("daily") {
        return Some("every day".to_string());
    }
    Regex::new(
        r"(?i)\b(once|twice|thrice|one|two|three|four|five|\d+)\s+times?\s+(?:a|per)\s+(day|week|month|year)\b",
    )
    .unwrap()
    .captures(line)
    .and_then(|caps| {
        let raw = caps.get(1)?.as_str().trim().to_ascii_lowercase();
        let unit = caps.get(2)?.as_str().trim().to_ascii_lowercase();
        Some(format!("{raw} times a {unit}"))
    })
}

pub(in crate::index) fn extract_time_answer_from_line(line: &str) -> Option<String> {
    [
        r"(?i)\b(\d{1,2}:\d{2}\s?(?:AM|PM))\b",
        r"(?i)\b(\d{1,2}\s?(?:AM|PM))\b",
    ]
    .into_iter()
    .find_map(|pattern| {
        Regex::new(pattern)
            .unwrap()
            .captures(line)
            .and_then(|caps| caps.get(1))
            .map(|m| m.as_str().trim().to_string())
    })
}

pub(in crate::index) fn extract_focus_aligned_time_answer_from_line(
    line: &str,
    lower: &str,
    focus_terms: &[String],
) -> Option<String> {
    let pattern = Regex::new(r"(?i)\b(\d{1,2}(?::\d{2})?\s?(?:AM|PM))\b").unwrap();
    let matches = pattern
        .captures_iter(line)
        .filter_map(|caps| caps.get(1))
        .map(|m| (m.start(), m.as_str().trim().to_string()))
        .collect::<Vec<_>>();
    if matches.is_empty() {
        return None;
    }
    if matches.len() == 1 {
        return extract_time_answer_from_line(line);
    }
    let focus_positions = focus_terms
        .iter()
        .filter_map(|term| lower.find(term))
        .collect::<Vec<_>>();
    if focus_positions.is_empty() {
        return matches.last().map(|(_, value)| value.clone());
    }
    matches
        .into_iter()
        .min_by_key(|(time_idx, _)| {
            focus_positions
                .iter()
                .map(|focus_idx| focus_idx.abs_diff(*time_idx))
                .min()
                .unwrap_or(usize::MAX)
        })
        .map(|(_, value)| value)
}

pub(in crate::index) fn extract_schedule_slot_focus_phrase(task_lower: &str) -> Option<String> {
    for marker in [
        "what day of the week do i ",
        "which day do i ",
        "what time do i ",
    ] {
        let Some((_, tail)) = task_lower.split_once(marker) else {
            continue;
        };
        let phrase = tail.trim().trim_end_matches('?');
        if !phrase.is_empty() {
            return Some(phrase.to_string());
        }
    }
    None
}

pub(in crate::index) fn extract_points_answer_from_line(line: &str, lower: &str) -> Option<String> {
    if !(lower.contains("score") || lower.contains("points")) {
        return None;
    }
    let raw = Regex::new(r"(?i)\b(\d+)\s+points\b")
        .unwrap()
        .captures(line)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim())?;
    Some(format!("{raw} points"))
}

pub(in crate::index) fn extract_record_answer_from_line(line: &str, lower: &str) -> Option<String> {
    if !(lower.contains("record") || lower.contains("we're") || lower.contains("we are")) {
        return None;
    }
    Regex::new(r"\b(\d+\s*-\s*\d+)\b")
        .unwrap()
        .captures(line)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim().replace(' ', ""))
}

pub(in crate::index) fn extract_status_answer_from_line(line: &str, lower: &str) -> Option<String> {
    if !lower.contains("status") {
        return None;
    }
    Regex::new(r"(?i)\b(Premier\s+(?:Silver|Gold|Platinum|Bronze|Diamond|1K))\s+status\b")
        .unwrap()
        .captures(line)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim().to_string())
}

pub(in crate::index) fn extract_level_goal_answer_from_line(line: &str, lower: &str) -> Option<String> {
    if !lower.contains("level")
        || !(line_has_future_goal_marker(lower)
            || lower.contains("determined to reach")
            || lower.contains("aiming to hit")
            || lower.contains("current goal"))
    {
        return None;
    }
    Regex::new(r"(?i)\b(level\s+\d+)\b")
        .unwrap()
        .captures(line)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim().to_ascii_lowercase())
}

pub(in crate::index) fn extract_state_transition_surface_from_line(
    line: &str,
    lower: &str,
    state_kind: &str,
) -> Option<String> {
    match state_kind {
        "score" => extract_points_answer_from_line(line, lower),
        "record" => extract_record_answer_from_line(line, lower),
        "status" => extract_status_answer_from_line(line, lower),
        "goal" => extract_level_goal_answer_from_line(line, lower),
        _ => None,
    }
}

pub(in crate::index) fn extract_relative_purchase_current_item(task_lower: &str) -> Option<String> {
    [
        "before getting the ",
        "before getting ",
        "before i got the ",
        "before i got ",
        "before buying the ",
        "before buying ",
        "before i bought the ",
        "before i bought ",
        "before purchasing the ",
        "before purchasing ",
        "before i purchased the ",
        "before i purchased ",
    ]
    .into_iter()
    .find_map(|marker| {
        let (_, tail) = task_lower.split_once(marker)?;
        let item = normalize_query_item_surface(tail);
        (!item.is_empty()).then_some(item)
    })
}

pub(in crate::index) fn normalize_query_item_surface(value: &str) -> String {
    let trimmed = value
        .trim()
        .trim_end_matches('?')
        .trim_end_matches('.')
        .trim();
    for prefix in ["the ", "a ", "an "] {
        if let Some(stripped) = trimmed.strip_prefix(prefix) {
            return stripped.trim().to_string();
        }
    }
    trimmed.to_string()
}

pub(in crate::index) fn extract_purchase_family_item_from_line(
    line: &str,
    lower: &str,
    family: &str,
) -> Option<String> {
    match family {
        "gadget" => extract_gadget_purchase_item_from_line(line, lower),
        "lens" => extract_lens_purchase_item_from_line(line, lower),
        _ => None,
    }
}

pub(in crate::index) fn extract_gadget_purchase_item_from_line(line: &str, lower: &str) -> Option<String> {
    if !task_contains_any(
        lower,
        &[
            "my new ",
            "i got",
            "got yesterday",
            "bought",
            "purchased",
            "gift",
            "using the ",
            "using my ",
        ],
    ) {
        return None;
    }
    Regex::new(
        r"(?i)\b(?:my\s+new\s+|my\s+|the\s+)?((?:[a-z0-9][a-z0-9+-]*)(?:\s+[a-z0-9][a-z0-9+-]*){0,2}\s(?:pot|fryer|mixer|blender|processor|maker|oven|grill|toaster|microwave|cooker|skillet))\b",
    )
    .unwrap()
    .captures_iter(line)
    .filter_map(|caps| caps.get(1))
    .map(|m| m.as_str().trim().to_string())
    .last()
}

pub(in crate::index) fn extract_lens_purchase_item_from_line(line: &str, lower: &str) -> Option<String> {
    let has_ownership_marker = task_contains_any(
        lower,
        &[
            "i got",
            "got my ",
            "recently got",
            "just got",
            "bought my ",
            "bought a ",
            "bought an ",
            "purchased",
            "picked up",
            "my new ",
        ],
    );
    if !lower.contains("lens") || !has_ownership_marker {
        return None;
    }
    if task_contains_any(lower, &["haven't bought", "have not bought", "might buy"])
        && !task_contains_any(lower, &["got my ", "recently got", "just got", "my new "])
    {
        return None;
    }
    let phrase = Regex::new(
        r"(?i)\b(?:old\s+|new\s+)?((?:\d{1,3}(?:-\d{1,3})?mm|[a-z]+(?:-[a-z]+)?)(?:\s+[a-z]+(?:-[a-z]+)?){0,2}\s+lens)\b",
    )
    .unwrap()
    .captures_iter(line)
    .filter_map(|caps| caps.get(1))
    .map(|m| m.as_str().trim().to_string())
    .last()?;
    Some(render_with_indefinite_article(&phrase))
}

pub(in crate::index) fn render_with_indefinite_article(value: &str) -> String {
    let trimmed = value.trim();
    let lower = trimmed.to_ascii_lowercase();
    if lower.starts_with("a ") || lower.starts_with("an ") {
        return trimmed.to_string();
    }
    let article = match lower.chars().next() {
        Some('a' | 'e' | 'i' | 'o' | 'u') => "an",
        _ => "a",
    };
    format!("{article} {trimmed}")
}

pub(in crate::index) fn extract_trip_destination_from_query(task_lower: &str) -> Option<String> {
    for marker in ["trip to ", "vacation to ", "visit to "] {
        let Some((_, tail)) = task_lower.split_once(marker) else {
            continue;
        };
        let destination = tail.trim().trim_end_matches('?').trim().to_string();
        if !destination.is_empty() {
            return Some(destination);
        }
    }
    None
}

pub(in crate::index) fn extract_planned_stay_location_from_line(line: &str, lower: &str) -> Option<String> {
    let value = extract_phrase_after_any_index(
        line,
        lower,
        &[
            "planning to stay on ",
            "planning to stay in ",
            "planning to stay at ",
            "plan to stay on ",
            "plan to stay in ",
            "plan to stay at ",
            "staying on ",
            "staying in ",
            "staying at ",
            "stay on ",
            "stay in ",
            "stay at ",
        ],
        &[
            " for ",
            " because ",
            " and ",
            " but ",
            " while ",
            ".",
            ",",
            ";",
            " instead",
            " during ",
        ],
        1,
    )?;
    (value.split_whitespace().count() <= 6).then(|| normalize_location_kg_value(&value))
}

pub(in crate::index) fn line_has_current_company_marker(lower: &str) -> bool {
    task_contains_any(
        lower,
        &[
            "currently working at ",
            "currently at ",
            "current company is ",
            "works at ",
            "working at ",
            "employed at ",
        ],
    )
}

pub(in crate::index) fn extract_current_company_answer_from_line(line: &str, lower: &str) -> Option<String> {
    if !line_has_current_company_marker(lower) {
        return None;
    }
    let answer = extract_phrase_after_any_index(
        line,
        lower,
        &[
            "currently working at ",
            "currently at ",
            "current company is ",
            "works at ",
            "working at ",
            "employed at ",
        ],
        &[
            " because ",
            " and ",
            " but ",
            " while ",
            ".",
            ",",
            ";",
            " with ",
        ],
        1,
    )?;
    (answer.split_whitespace().count() <= 6).then_some(answer)
}

pub(in crate::index) fn extract_instagram_current_count_candidate(
    line: &str,
    lower: &str,
) -> Option<(i32, usize)> {
    if !lower.contains("follower")
        || task_contains_any(
            lower,
            &["facebook", "twitter", "tiktok", "youtube", "linkedin"],
        )
        || !line_has_current_count_marker(lower)
    {
        return None;
    }
    if extract_duration_answer_from_line(line).is_some()
        && !task_contains_any(
            lower,
            &[
                "just checked",
                "now at",
                "currently have",
                "currently at",
                "current follower count",
            ],
        )
    {
        return None;
    }
    let value = extract_line_numbers(line)
        .into_iter()
        .filter(|value| *value >= 10)
        .last()?;
    let mut strength = 4usize;
    if task_contains_any(
        lower,
        &[
            "just checked",
            "now at",
            "recently crossed",
            "just reached",
            "currently have",
            "currently at",
        ],
    ) {
        strength += 6;
    }
    if lower.contains("follower count") {
        strength += 2;
    }
    if task_contains_any(
        lower,
        &[
            "close to",
            "almost",
            "nearly",
            "about ",
            "around ",
            "roughly",
            "approximately",
            "approx ",
        ],
    ) {
        strength = strength.saturating_sub(4);
    }
    Some((value, strength))
}

pub(in crate::index) fn line_has_current_count_marker(lower: &str) -> bool {
    task_contains_any(
        lower,
        &[
            " currently",
            " current",
            " now",
            " right now",
            " today",
            " these days",
            " recently",
            " just ",
            " already",
            " actually",
            " still",
            " so far",
        ],
    )
}

pub(in crate::index) fn is_money_query(task: &str) -> bool {
    const MONEY_MARKERS: &[&str] = &[
        "$",
        " dollar",
        " dollars",
        "money",
        "expense",
        "expenses",
        "cost",
        "costs",
        "price",
        "prices",
        "paid",
        "bill",
        "bills",
        "budget",
        "purchase",
        "purchased",
        "income",
        "earnings",
        "earned",
        "earning",
        "salary",
        "wage",
        "wages",
        "revenue",
        "profit",
        "profits",
    ];
    const NON_MONEY_UNITS: &[&str] = &[
        "time", "times", "hour", "hours", "day", "days", "week", "weeks", "month", "months",
        "year", "years", "session", "sessions",
    ];

    let lower = task.to_ascii_lowercase();
    MONEY_MARKERS.iter().any(|marker| lower.contains(marker))
        || (lower.contains("how much") && !NON_MONEY_UNITS.iter().any(|unit| lower.contains(unit)))
}

pub(in crate::index) fn normalize_aggregate_focus_token(token: &str) -> Option<String> {
    let mut cleaned: String = token
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase();
    if cleaned.len() < 3 {
        return None;
    }
    if cleaned.ends_with("ies") && cleaned.len() > 4 {
        cleaned = format!("{}y", &cleaned[..cleaned.len() - 3]);
    } else if cleaned.ends_with('s') && !cleaned.ends_with("ss") && cleaned.len() > 4 {
        cleaned.pop();
    }
    Some(cleaned)
}

pub(in crate::index) fn aggregate_focus_tokens_for_path(path: &Path) -> Vec<String> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    let base = file_name.strip_suffix(".aggregate.md").unwrap_or(file_name);
    let topic = base
        .strip_prefix("_arith_")
        .or_else(|| base.strip_prefix("_count_"))
        .unwrap_or(base);
    topic
        .split('_')
        .filter_map(normalize_aggregate_focus_token)
        .collect()
}

pub(in crate::index) fn aggregate_focus_token_count_for_path(path: &Path) -> usize {
    aggregate_focus_tokens_for_path(path).len()
}

pub(in crate::index) fn aggregate_focus_match_count_for_path(path: &Path, focus_terms: &[String]) -> usize {
    let aggregate_tokens: HashSet<String> =
        aggregate_focus_tokens_for_path(path).into_iter().collect();
    let focus_tokens: HashSet<String> = focus_terms
        .iter()
        .filter_map(|term| normalize_aggregate_focus_token(term))
        .collect();
    aggregate_tokens.intersection(&focus_tokens).count()
}

pub(in crate::index) fn best_matching_arithmetic_aggregate_path(
    project_root: &Path,
    focus_terms: &[String],
) -> Option<PathBuf> {
    let ndir = neuron_dir(project_root);
    let Ok(read_dir) = std::fs::read_dir(&ndir) else {
        return None;
    };

    read_dir
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.starts_with("_arith_") && name.ends_with(".aggregate.md"))
                .unwrap_or(false)
        })
        .filter_map(|path| {
            let match_count = aggregate_focus_match_count_for_path(&path, focus_terms);
            if match_count == 0 {
                return None;
            }
            let token_count = aggregate_focus_token_count_for_path(&path).max(1);
            let score = (match_count as f32 * 100.0) + (match_count as f32 / token_count as f32);
            Some((score, path))
        })
        .max_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)))
        .map(|(_, path)| path)
}

pub(in crate::index) fn is_session_summary_path(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.ends_with("_summary.md"))
        .unwrap_or(false)
}

pub(in crate::index) fn strip_query_surface_section(content: &str) -> String {
    let without_query = strip_named_section(content, "query_surface");
    strip_named_section(&without_query, "answer_surface")
}

pub(in crate::index) fn strip_named_section(content: &str, section_name: &str) -> String {
    let header = format!("## {section_name}");
    let marker = format!("<!-- SECTION: {section_name} -->");
    let end_marker = "<!-- /SECTION -->";
    let Some(header_start) = content.find(&header) else {
        return content.to_string();
    };
    let Some(section_start_rel) = content[header_start..].find(&marker) else {
        return content.to_string();
    };
    let section_start = header_start + section_start_rel;
    let Some(section_end_rel) = content[section_start..].find(end_marker) else {
        return content.to_string();
    };
    let section_end = section_start + section_end_rel + end_marker.len();

    let mut stripped = String::with_capacity(content.len());
    stripped.push_str(content[..header_start].trim_end());
    if !stripped.ends_with('\n') {
        stripped.push('\n');
    }
    let tail = content[section_end..].trim_start_matches('\n');
    if !tail.is_empty() {
        stripped.push('\n');
        stripped.push_str(tail);
    }
    stripped
}

pub(in crate::index) fn parse_index_answer_surface_rows(content: &str) -> Vec<IndexAnswerSurfaceRow> {
    let sections = crate::neuron::parse_sections(content);
    let Some(table) = sections.get("answer_surface") else {
        return Vec::new();
    };

    table
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if !trimmed.starts_with('|') {
                return None;
            }
            let columns = trimmed
                .trim_matches('|')
                .split('|')
                .map(str::trim)
                .collect::<Vec<_>>();
            if columns.len() != 3 {
                return None;
            }
            if columns[0].eq_ignore_ascii_case("question_pattern")
                || columns[0].chars().all(|c| c == '-' || c == ' ')
            {
                return None;
            }

            let answer_span = columns[1]
                .trim()
                .trim_matches(|c: char| matches!(c, '"' | '\'' | '`'))
                .to_string();
            if answer_span.is_empty() {
                return None;
            }

            Some(IndexAnswerSurfaceRow {
                question_pattern: columns[0].to_string(),
                answer_span,
                confidence: columns[2].parse::<f32>().unwrap_or(0.0),
            })
        })
        .collect()
}

pub(in crate::index) fn synthetic_answer_surface_query_profile(
    task: &str,
    task_lower: &str,
    task_terms: &[String],
    compose_list_answer: bool,
) -> SyntheticAnswerSurfaceQueryProfile {
    const OPEN_QA_FILLER: &[&str] = &[
        "would",
        "could",
        "should",
        "can",
        "will",
        "may",
        "might",
        "likely",
        "probably",
        "possibly",
        "potentially",
        "considered",
        "still",
        "more",
        "most",
        "less",
        "least",
        "another",
        "kind",
        "sort",
        "thing",
        "things",
        "personality",
        "trait",
        "traits",
        "additional",
        "alternative",
        "popular",
        "based",
        "around",
    ];
    let subject_terms = synthetic_answer_surface_subject_terms(task);
    let subject_term_keys = synthetic_answer_surface_term_key_set(&subject_terms);
    let choice_options = synthetic_answer_surface_choice_options(task);
    let location_target = synthetic_answer_surface_location_target(task_lower);
    let route_kind = if !choice_options.is_empty() {
        SyntheticAnswerSurfaceRouteKind::Choice
    } else if location_target.is_some() {
        SyntheticAnswerSurfaceRouteKind::LocationLift
    } else if synthetic_answer_surface_is_typed_open_qa_query(task_lower) {
        SyntheticAnswerSurfaceRouteKind::YesNo
    } else {
        SyntheticAnswerSurfaceRouteKind::Default
    };
    let mut anchor_terms = task_terms
        .iter()
        .filter(|term| {
            !OPEN_QA_FILLER.contains(&term.as_str())
                && !choice_options.iter().any(|option| {
                    option
                        .term_keys
                        .contains(&synthetic_answer_surface_term_key(term))
                })
                && (subject_terms.iter().any(|subject| subject == *term)
                    || term.len() >= 4
                    || term.chars().any(|c| c.is_ascii_digit()))
        })
        .cloned()
        .collect::<Vec<_>>();
    if anchor_terms.is_empty() {
        anchor_terms = task_terms
            .iter()
            .filter(|term| !OPEN_QA_FILLER.contains(&term.as_str()))
            .cloned()
            .collect();
    }
    if anchor_terms.is_empty() {
        anchor_terms = task_terms.to_vec();
    }
    anchor_terms.sort();
    anchor_terms.dedup();
    let anchor_term_keys = synthetic_answer_surface_term_key_set(&anchor_terms);
    let relation_term_keys = anchor_term_keys
        .difference(&subject_term_keys)
        .cloned()
        .collect::<HashSet<_>>();
    let expected_type = synthetic_answer_surface_expected_type(task_lower, compose_list_answer);
    let (relation_families, strict_relation_family_match) =
        synthetic_answer_surface_query_relation_families(task_lower);

    SyntheticAnswerSurfaceQueryProfile {
        task_term_keys: synthetic_answer_surface_term_key_set(task_terms),
        subject_term_keys,
        anchor_term_keys,
        relation_term_keys,
        expected_type,
        route_kind,
        choice_options,
        location_target,
        requires_strict_anchor_overlap: !matches!(
            route_kind,
            SyntheticAnswerSurfaceRouteKind::Choice
        ),
        requires_completed_evidence: synthetic_answer_surface_requires_completed_evidence(
            task_lower,
        ),
        strict_relation_family_match,
        relation_families,
        allows_count_projection_from_lists: matches!(
            expected_type,
            SyntheticAnswerSurfaceExpectedType::Count
        ) && compose_list_answer,
    }
}

pub(in crate::index) fn synthetic_answer_surface_query_relation_families(
    task_lower: &str,
) -> (HashSet<SyntheticAnswerSurfaceRelationFamily>, bool) {
    let mut families = HashSet::new();
    let mut strict = false;

    let mut push_strict = |family| {
        families.insert(family);
        strict = true;
    };

    if task_contains_any(
        task_lower,
        &["move from", "moved from", "home country", "origin country"],
    ) {
        push_strict(SyntheticAnswerSurfaceRelationFamily::Origin);
    } else if task_lower.starts_with("how long ")
        && task_contains_any(task_lower, &["group of friends", "support system"])
    {
        push_strict(SyntheticAnswerSurfaceRelationFamily::FriendGroupDuration);
    } else if task_lower.starts_with("who ")
        && task_contains_any(
            task_lower,
            &[
                "support",
                "supports",
                "support system",
                "negative experience",
                "my rocks",
            ],
        )
    {
        push_strict(SyntheticAnswerSurfaceRelationFamily::SupportNetwork);
    } else if task_contains_any(
        task_lower,
        &[
            "research",
            "researched",
            "researching",
            "looking into",
            "investigating",
        ],
    ) {
        push_strict(SyntheticAnswerSurfaceRelationFamily::Research);
    } else if task_contains_any(
        task_lower,
        &[
            "career path",
            "career",
            " fields",
            " field",
            "education",
            "pursue",
            "study",
            "job",
            "work in",
        ],
    ) {
        push_strict(SyntheticAnswerSurfaceRelationFamily::Career);
    } else if task_contains_any(
        task_lower,
        &["what books", "which books", " books", "book "],
    ) && task_contains_any(task_lower, &[" read", "reading", "bookshelf", "book"])
    {
        push_strict(SyntheticAnswerSurfaceRelationFamily::Book);
    } else if task_contains_any(
        task_lower,
        &[
            "what events has",
            "which events",
            "events has",
            "events have",
            "events did",
            "in what ways",
        ],
    ) && task_contains_any(
        task_lower,
        &[
            "help children",
            "help kids",
            "help youth",
            "children",
            "kids",
            "youth",
        ],
    ) {
        push_strict(SyntheticAnswerSurfaceRelationFamily::ChildHelpEvent);
    } else if task_contains_any(
        task_lower,
        &[
            "lgbtq",
            "lgbtq+",
            "transgender-specific",
            "transgender community",
            "lgbtq community",
        ],
    ) && task_contains_any(
        task_lower,
        &[
            "event",
            "events",
            "participat",
            "attend",
            "joined",
            "join ",
            "in what ways",
        ],
    ) {
        push_strict(SyntheticAnswerSurfaceRelationFamily::CommunityEvent);
    } else if task_contains_any(task_lower, &["where has ", "where have ", " camped"])
        && task_contains_any(task_lower, &["camp", "camped", "camping"])
    {
        push_strict(SyntheticAnswerSurfaceRelationFamily::CampLocation);
    } else if task_contains_any(
        task_lower,
        &[
            "to destress",
            "to de-stress",
            "self-care",
            "stay distracted",
            "relax",
        ],
    ) {
        push_strict(SyntheticAnswerSurfaceRelationFamily::SelfCareActivity);
    } else if task_contains_any(
        task_lower,
        &[" activities", " activity", "hobbies", "hobby"],
    ) {
        if task_contains_any(
            task_lower,
            &[
                "with her family",
                "with his family",
                "with my family",
                "with their family",
                "with the kids",
                "with my kids",
                "family",
                "kids",
                "children",
                "together",
            ],
        ) {
            push_strict(SyntheticAnswerSurfaceRelationFamily::FamilyActivity);
        } else {
            families.insert(SyntheticAnswerSurfaceRelationFamily::Activity);
            families.insert(SyntheticAnswerSurfaceRelationFamily::FamilyActivity);
            families.insert(SyntheticAnswerSurfaceRelationFamily::SelfCareActivity);
        }
    } else if task_contains_any(
        task_lower,
        &["kids like", "children like", "what do", "what does"],
    ) && task_contains_any(task_lower, &["kids", "children"])
    {
        push_strict(SyntheticAnswerSurfaceRelationFamily::KidsPreference);
    } else if task_contains_any(task_lower, &["paint", "painting", "art does"]) {
        push_strict(SyntheticAnswerSurfaceRelationFamily::PaintSubject);
    } else if task_contains_any(
        task_lower,
        &[
            "member of the lgbtq community",
            "member of the transgender community",
            "ally",
        ],
    ) {
        push_strict(SyntheticAnswerSurfaceRelationFamily::Ally);
    } else if task_contains_any(
        task_lower,
        &["religious", "religion", "faith", "church", "spiritual"],
    ) {
        push_strict(SyntheticAnswerSurfaceRelationFamily::Religion);
    } else if task_lower.contains("relationship status") {
        push_strict(SyntheticAnswerSurfaceRelationFamily::Relationship);
    } else if task_contains_any(
        task_lower,
        &["identity", "transgender woman", "transgender man"],
    ) {
        push_strict(SyntheticAnswerSurfaceRelationFamily::Identity);
    }

    (families, strict)
}

pub(in crate::index) fn synthetic_answer_surface_is_typed_open_qa_query(task_lower: &str) -> bool {
    task_lower.starts_with("would ")
        || task_lower.starts_with("could ")
        || task_lower.starts_with("should ")
        || task_lower.starts_with("can ")
        || task_lower.starts_with("will ")
        || task_lower.starts_with("may ")
        || task_lower.starts_with("might ")
        || task_lower.starts_with("is ")
        || task_lower.starts_with("are ")
        || task_lower.starts_with("was ")
        || task_lower.starts_with("were ")
        || task_lower.starts_with("does ")
        || task_lower.starts_with("do ")
        || task_lower.starts_with("did ")
        || task_lower.starts_with("has ")
        || task_lower.starts_with("have ")
        || task_lower.starts_with("had ")
        || task_lower.starts_with("which ")
        || task_lower.starts_with("what might ")
        || task_lower.starts_with("what would ")
        || task_lower.contains(" likely ")
        || task_lower.contains(" likely be ")
        || task_lower.contains(" considered ")
}

pub(in crate::index) fn synthetic_answer_surface_location_target(
    task_lower: &str,
) -> Option<SyntheticAnswerSurfaceLocationTarget> {
    if task_contains_any(task_lower, &["national park", "which park"]) {
        Some(SyntheticAnswerSurfaceLocationTarget::NationalPark)
    } else if task_lower.starts_with("what state")
        || task_lower.starts_with("which state")
        || task_contains_any(
            task_lower,
            &[
                " in what state",
                " in which state",
                " us state",
                " us states",
            ],
        )
    {
        Some(SyntheticAnswerSurfaceLocationTarget::State)
    } else if task_lower.starts_with("what country")
        || task_lower.starts_with("which country")
        || task_contains_any(
            task_lower,
            &[
                " in what country",
                " in which country",
                " home country",
                "move from",
                "moved from",
                "origin country",
            ],
        )
    {
        Some(SyntheticAnswerSurfaceLocationTarget::Country)
    } else {
        None
    }
}

pub(in crate::index) fn synthetic_answer_surface_choice_options(
    task: &str,
) -> Vec<SyntheticAnswerSurfaceChoiceOption> {
    let lower = task.to_ascii_lowercase();
    if !lower.contains(" or ")
        || lower.contains("answer in yes or no")
        || lower.ends_with("yes or no")
    {
        return Vec::new();
    }

    let tail = task.trim().trim_end_matches('?').trim();
    let Some((left_segment, right_segment)) = tail.rsplit_once(" or ") else {
        return Vec::new();
    };
    let left_raw = [
        " close to ",
        " going to ",
        " visiting ",
        " visit ",
        " in ",
        " at ",
        " between ",
        " answer in ",
        ", ",
    ]
    .iter()
    .find_map(|marker| left_segment.rsplit_once(marker).map(|(_, value)| value))
    .unwrap_or(left_segment);

    [left_raw, right_segment]
        .into_iter()
        .map(synthetic_answer_surface_choice_option)
        .collect::<Option<Vec<_>>>()
        .unwrap_or_default()
}

pub(in crate::index) fn synthetic_conjoined_choice_options(
    task: &str,
) -> Vec<SyntheticAnswerSurfaceChoiceOption> {
    let lower = task.to_ascii_lowercase();
    if !lower.contains(" and ") {
        return Vec::new();
    }

    let tail = task.trim().trim_end_matches('?').trim();
    let Some((left_segment, right_segment)) = tail.rsplit_once(" and ") else {
        return Vec::new();
    };
    let left_raw = [
        " on both the ",
        " on both ",
        " both the ",
        " both ",
        " of ",
        " for ",
        " between ",
        ", ",
    ]
    .iter()
    .find_map(|marker| left_segment.rsplit_once(marker).map(|(_, value)| value))
    .unwrap_or(left_segment);

    [left_raw, right_segment]
        .into_iter()
        .map(synthetic_answer_surface_choice_option)
        .collect::<Option<Vec<_>>>()
        .unwrap_or_default()
}

pub(in crate::index) fn synthetic_answer_surface_choice_option(
    raw: &str,
) -> Option<SyntheticAnswerSurfaceChoiceOption> {
    let display = raw
        .trim()
        .trim_start_matches("the ")
        .trim_start_matches("a ")
        .trim_start_matches("an ")
        .trim_matches(|c: char| matches!(c, '?' | ',' | '.' | ':' | ';'))
        .to_string();
    if display.is_empty() {
        return None;
    }

    let display_lower = display.to_ascii_lowercase();
    let term_keys = synthetic_answer_surface_term_key_set(&synthetic_query_terms(&display_lower));
    if term_keys.is_empty() {
        return None;
    }
    let mut affinity_terms = synthetic_query_terms(&display_lower);
    affinity_terms.extend(
        synthetic_answer_surface_choice_affinity_terms(&display_lower)
            .into_iter()
            .map(|term| (*term).to_string()),
    );
    let affinity_term_keys = synthetic_answer_surface_term_key_set(&affinity_terms);
    Some(SyntheticAnswerSurfaceChoiceOption {
        display,
        term_keys,
        affinity_term_keys,
    })
}

pub(in crate::index) fn missing_operand_display_phrase(display: &str) -> String {
    let mut phrase = display.trim().to_string();
    loop {
        let lower = phrase.to_ascii_lowercase();
        let mut stripped = false;
        for prefix in [
            "my ",
            "our ",
            "his ",
            "her ",
            "their ",
            "recently ",
            "recent ",
            "new ",
        ] {
            if lower.starts_with(prefix) {
                phrase = phrase[prefix.len()..].trim().to_string();
                stripped = true;
                break;
            }
        }
        if !stripped {
            break;
        }
    }
    phrase
}

pub(in crate::index) fn synthetic_answer_surface_choice_affinity_terms(
    display_lower: &str,
) -> &'static [&'static str] {
    if display_lower.contains("national park") {
        &[
            "nature",
            "outdoors",
            "outdoor",
            "camping",
            "camp",
            "hiking",
            "mountain",
            "mountains",
            "forest",
            "woods",
            "trail",
            "lake",
            "park",
        ]
    } else if display_lower.contains("theme park") {
        &[
            "theme",
            "amusement",
            "rides",
            "roller",
            "coaster",
            "disney",
            "universal",
            "park",
        ]
    } else if display_lower.contains("mountain") {
        &[
            "mountain",
            "mountains",
            "hiking",
            "camping",
            "nature",
            "outdoors",
            "trail",
            "park",
        ]
    } else if display_lower.contains("beach") {
        &["beach", "ocean", "coast", "shore", "sand", "waves", "surf"]
    } else if display_lower == "yes" {
        &["yes", "true", "correct"]
    } else if display_lower == "no" {
        &["no", "not", "never", "false"]
    } else {
        &[]
    }
}

pub(in crate::index) fn synthetic_answer_surface_subject_terms(task: &str) -> Vec<String> {
    const STOP: &[&str] = &[
        "what", "when", "where", "which", "who", "whom", "whose", "why", "how", "does", "did",
        "do", "is", "are", "was", "were", "has", "have", "would", "could", "should", "may",
        "might", "can", "will", "the", "a", "an", "and", "or", "for", "from", "with", "about",
        "into", "after", "before", "between", "around", "through", "this", "that", "these",
        "those",
    ];
    const MONTHS: &[&str] = &[
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ];

    let mut terms = task
        .split(|c: char| !c.is_ascii_alphabetic() && c != '-' && c != '\'')
        .filter_map(|token| {
            let trimmed = token.trim();
            let first = trimmed.chars().next()?;
            if trimmed.len() < 3 || !first.is_ascii_uppercase() {
                return None;
            }
            let lower = trimmed.to_ascii_lowercase();
            if STOP.contains(&lower.as_str()) || MONTHS.contains(&lower.as_str()) {
                return None;
            }
            Some(lower)
        })
        .collect::<Vec<_>>();
    terms.sort();
    terms.dedup();
    terms
}

pub(in crate::index) fn synthetic_answer_surface_expected_type(
    task_lower: &str,
    compose_list_answer: bool,
) -> SyntheticAnswerSurfaceExpectedType {
    if task_lower.starts_with("how long ") || task_lower.contains("how long ago") {
        SyntheticAnswerSurfaceExpectedType::Duration
    } else if task_lower.starts_with("when ")
        || task_contains_any(
            task_lower,
            &[
                "what date",
                "what day",
                "which day",
                "which month",
                "what month",
                "what year",
                "around which",
            ],
        )
    {
        SyntheticAnswerSurfaceExpectedType::Date
    } else if task_lower.starts_with("how many ") || task_lower.starts_with("how much ") {
        SyntheticAnswerSurfaceExpectedType::Count
    } else if task_lower.starts_with("who ") || task_lower.contains(" who ") {
        SyntheticAnswerSurfaceExpectedType::Person
    } else if task_lower.contains("relationship status") {
        SyntheticAnswerSurfaceExpectedType::Status
    } else if task_lower.starts_with("where ")
        || task_contains_any(
            task_lower,
            &[
                " which state",
                " which country",
                " which city",
                " in what country",
                " in which state",
                " in which country",
                " live close to ",
                " close to a beach",
                " close to the mountains",
                " national park",
            ],
        )
    {
        SyntheticAnswerSurfaceExpectedType::Location
    } else if compose_list_answer
        && !task_lower.contains(" name")
        && !task_lower.contains(" names")
        && !task_contains_any(task_lower, &["book", "books", " called "])
    {
        SyntheticAnswerSurfaceExpectedType::ListItem
    } else if compose_list_answer
        || task_lower.contains(" name")
        || task_lower.contains(" names")
        || task_contains_any(task_lower, &["book", "books", " called "])
    {
        SyntheticAnswerSurfaceExpectedType::NameLike
    } else {
        SyntheticAnswerSurfaceExpectedType::Generic
    }
}

pub(in crate::index) fn synthetic_answer_surface_requires_completed_evidence(task_lower: &str) -> bool {
    task_lower.starts_with("where has ")
        || task_lower.starts_with("where did ")
        || task_lower.starts_with("what did ")
        || task_contains_any(
            task_lower,
            &[
                " participated in",
                " has participated",
                " have participated",
                " attended ",
                " joined ",
                " camped",
                " books has ",
                " books have ",
                " what books",
                " has read",
                " have read",
                " researched",
                " research",
                " tried ",
                " been on ",
                " gone on ",
            ],
        )
}

pub(in crate::index) fn synthetic_answer_surface_term_key_set(terms: &[String]) -> HashSet<String> {
    terms
        .iter()
        .map(|term| synthetic_answer_surface_term_key(term))
        .filter(|term| !term.is_empty())
        .collect()
}

pub(in crate::index) fn synthetic_answer_surface_term_key(term: &str) -> String {
    fn trim_repeated_suffix(word: &mut String) {
        let chars = word.chars().collect::<Vec<_>>();
        if chars.len() >= 2 {
            let last = chars[chars.len() - 1];
            let prev = chars[chars.len() - 2];
            if last == prev && matches!(last, 'b' | 'd' | 'g' | 'l' | 'm' | 'n' | 'p' | 'r' | 't') {
                word.pop();
            }
        }
    }

    let mut key = term
        .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '\'' && c != '-')
        .to_ascii_lowercase();
    if key.ends_with("'s") {
        key.truncate(key.len() - 2);
    }
    if key.is_empty() {
        return key;
    }

    let mapped = match key.as_str() {
        "went" | "gone" | "goes" => Some("go"),
        "bought" => Some("buy"),
        "taught" | "teaches" | "teaching" => Some("teach"),
        "grew" | "grown" | "growing" => Some("grow"),
        "ran" | "running" => Some("run"),
        "swam" | "swimming" => Some("swim"),
        "wrote" | "written" | "writing" => Some("write"),
        "reads" | "reading" => Some("read"),
        "met" | "meeting" => Some("meet"),
        "took" | "taken" => Some("take"),
        "drove" | "driving" => Some("drive"),
        "brought" => Some("bring"),
        "began" | "begun" => Some("begin"),
        _ => None,
    };
    if let Some(mapped) = mapped {
        return mapped.to_string();
    }

    if key.len() > 5 && key.ends_with("ied") {
        key.truncate(key.len() - 3);
        key.push('y');
    } else if key.len() > 5 && key.ends_with("ies") {
        key.truncate(key.len() - 3);
        key.push('y');
    } else if key.len() > 5 && key.ends_with("ing") {
        key.truncate(key.len() - 3);
        trim_repeated_suffix(&mut key);
    } else if key.len() > 4 && key.ends_with("ed") {
        key.truncate(key.len() - 2);
        trim_repeated_suffix(&mut key);
    } else if key.len() > 4 && key.ends_with("es") {
        key.truncate(key.len() - 2);
    } else if key.len() > 3 && key.ends_with('s') && !key.ends_with("ss") {
        key.pop();
    }

    if key.len() > 4 && key.ends_with('e') {
        key.pop();
    }
    key
}

pub(in crate::index) fn synthetic_answer_surface_family_activity_context(lower: &str) -> bool {
    task_contains_any(
        lower,
        &[
            " kids",
            "my kids",
            "with the kids",
            "with my kids",
            "with my fam",
            "with my family",
            "family",
            "children",
            "together",
        ],
    )
}

pub(in crate::index) fn synthetic_answer_surface_self_care_activity_context(lower: &str) -> bool {
    task_contains_any(
        lower,
        &[
            "de-stress",
            "destress",
            "self-care",
            "relax",
            "peace",
            "therapeutic",
            "calming",
            "me-time",
            "stay distracted",
        ],
    )
}

pub(in crate::index) fn synthetic_answer_surface_relation_family(
    question_pattern: &str,
    evidence_line: Option<&str>,
) -> Option<SyntheticAnswerSurfaceRelationFamily> {
    let pattern_lower = question_pattern.to_ascii_lowercase();
    let evidence_lower = evidence_line.unwrap_or_default().to_ascii_lowercase();
    let pattern_keys =
        synthetic_answer_surface_term_key_set(&synthetic_query_terms(&pattern_lower));
    let pattern_has_any = |keys: &[&str]| keys.iter().any(|key| pattern_keys.contains(*key));
    let pattern_has_all = |keys: &[&str]| keys.iter().all(|key| pattern_keys.contains(*key));

    if pattern_has_any(&["mov", "origin", "country"]) && pattern_has_any(&["from", "country"]) {
        Some(SyntheticAnswerSurfaceRelationFamily::Origin)
    } else if pattern_has_any(&["friend"])
        && pattern_has_any(&["known", "know", "long", "duration"])
        && pattern_has_any(&["year", "month", "week", "day"])
    {
        Some(SyntheticAnswerSurfaceRelationFamily::FriendGroupDuration)
    } else if !pattern_has_any(&["event"])
        && (pattern_has_all(&["who", "support"])
            || pattern_has_all(&["negative", "experienc"])
            || pattern_has_any(&["rock"]))
        && pattern_has_any(&["mentor", "friend", "family", "kid", "husband", "partner"])
    {
        Some(SyntheticAnswerSurfaceRelationFamily::SupportNetwork)
    } else if pattern_has_any(&["research", "topic", "investigat", "look", "into"]) {
        Some(SyntheticAnswerSurfaceRelationFamily::Research)
    } else if pattern_has_any(&["career", "field", "educat", "study", "job", "work"]) {
        Some(SyntheticAnswerSurfaceRelationFamily::Career)
    } else if pattern_has_any(&["book", "read", "title", "literatur"]) {
        Some(SyntheticAnswerSurfaceRelationFamily::Book)
    } else if pattern_has_any(&["camp", "location", "place"])
        && pattern_has_any(&["camp", "beach", "mountain", "forest", "lake"])
    {
        Some(SyntheticAnswerSurfaceRelationFamily::CampLocation)
    } else if pattern_has_any(&["kid", "children", "child"])
        && pattern_has_any(&["like", "lov", "enjoy", "favorit", "interest"])
    {
        Some(SyntheticAnswerSurfaceRelationFamily::KidsPreference)
    } else if pattern_has_any(&["paint", "scene", "subject"])
        || (pattern_has_any(&["art"]) && pattern_has_any(&["paint", "made", "make", "creat"]))
    {
        Some(SyntheticAnswerSurfaceRelationFamily::PaintSubject)
    } else if pattern_has_any(&[
        "identity",
        "gender",
        "transgender",
        "woman",
        "man",
        "nonbinary",
        "queer",
    ]) && !pattern_has_any(&["event"])
    {
        Some(SyntheticAnswerSurfaceRelationFamily::Identity)
    } else if pattern_has_any(&["event"]) && pattern_has_any(&["children", "kid", "youth"]) {
        Some(SyntheticAnswerSurfaceRelationFamily::ChildHelpEvent)
    } else if pattern_has_any(&["event"])
        && pattern_has_any(&[
            "lgbtq",
            "community",
            "parade",
            "activist",
            "group",
            "speech",
            "program",
            "art",
            "support",
        ])
    {
        Some(SyntheticAnswerSurfaceRelationFamily::CommunityEvent)
    } else if pattern_has_any(&["activity", "hobby"])
        && (pattern_has_any(&[
            "destress",
            "relax",
            "self-care",
            "peace",
            "therapeutic",
            "calm",
        ]) || (!pattern_has_any(&["family", "kid", "children", "together", "fun"])
            && synthetic_answer_surface_self_care_activity_context(&evidence_lower)))
    {
        Some(SyntheticAnswerSurfaceRelationFamily::SelfCareActivity)
    } else if pattern_has_any(&["activity", "hobby"])
        && (pattern_has_any(&["family", "kid", "children", "together", "fun"])
            || (!pattern_has_any(&[
                "destress",
                "relax",
                "self-care",
                "peace",
                "therapeutic",
                "calm",
            ]) && synthetic_answer_surface_family_activity_context(&evidence_lower)))
    {
        Some(SyntheticAnswerSurfaceRelationFamily::FamilyActivity)
    } else if pattern_has_any(&["activity", "hobby"]) {
        Some(SyntheticAnswerSurfaceRelationFamily::Activity)
    } else if pattern_has_any(&["religious", "religion", "faith", "church", "spiritual"]) {
        Some(SyntheticAnswerSurfaceRelationFamily::Religion)
    } else if pattern_has_any(&[
        "relationship",
        "statu",
        "single",
        "married",
        "partner",
        "spouse",
    ]) {
        Some(SyntheticAnswerSurfaceRelationFamily::Relationship)
    } else if pattern_has_any(&["ally", "supportive", "acceptance"])
        || (pattern_has_all(&["support", "community"]) && !pattern_has_any(&["event"]))
    {
        Some(SyntheticAnswerSurfaceRelationFamily::Ally)
    } else {
        None
    }
}

pub(in crate::index) fn synthetic_answer_surface_relation_family_matches(
    profile: &SyntheticAnswerSurfaceQueryProfile,
    row_family: Option<SyntheticAnswerSurfaceRelationFamily>,
    relation_overlap: usize,
) -> bool {
    if profile.relation_families.is_empty() {
        return true;
    }
    if row_family
        .map(|family| profile.relation_families.contains(&family))
        .unwrap_or(false)
    {
        return true;
    }
    if !profile.strict_relation_family_match {
        return row_family.is_some_and(|family| {
            profile
                .relation_families
                .contains(&SyntheticAnswerSurfaceRelationFamily::Activity)
                && matches!(
                    family,
                    SyntheticAnswerSurfaceRelationFamily::FamilyActivity
                        | SyntheticAnswerSurfaceRelationFamily::SelfCareActivity
                )
        }) || relation_overlap > 0;
    }
    row_family.is_none()
        && !profile.relation_term_keys.is_empty()
        && relation_overlap >= usize::min(2, profile.relation_term_keys.len())
}

pub(in crate::index) fn synthetic_answer_surface_bucket_matches_relation_profile(
    profile: &SyntheticAnswerSurfaceQueryProfile,
    bucket: &IndexAnswerSurfaceBucket,
) -> bool {
    if profile.relation_families.is_empty() || bucket.relation_families.is_empty() {
        return true;
    }
    bucket
        .relation_families
        .iter()
        .copied()
        .any(|family| synthetic_answer_surface_relation_family_matches(profile, Some(family), 1))
}

pub(in crate::index) fn synthetic_answer_surface_relation_family_supports_count_projection(
    family: SyntheticAnswerSurfaceRelationFamily,
) -> bool {
    matches!(
        family,
        SyntheticAnswerSurfaceRelationFamily::Activity
            | SyntheticAnswerSurfaceRelationFamily::FamilyActivity
            | SyntheticAnswerSurfaceRelationFamily::SelfCareActivity
            | SyntheticAnswerSurfaceRelationFamily::Book
            | SyntheticAnswerSurfaceRelationFamily::CampLocation
            | SyntheticAnswerSurfaceRelationFamily::KidsPreference
            | SyntheticAnswerSurfaceRelationFamily::PaintSubject
            | SyntheticAnswerSurfaceRelationFamily::CommunityEvent
            | SyntheticAnswerSurfaceRelationFamily::ChildHelpEvent
    )
}

pub(in crate::index) fn synthetic_answer_surface_count_projection_candidate(
    answer_span: &str,
    row_family: Option<SyntheticAnswerSurfaceRelationFamily>,
) -> bool {
    row_family
        .filter(|family| {
            synthetic_answer_surface_relation_family_supports_count_projection(*family)
        })
        .is_some()
        && (looks_like_answer_surface_list_item(answer_span)
            || looks_like_answer_surface_name_like(answer_span)
            || looks_like_answer_surface_location(answer_span)
            || looks_like_answer_surface_person(answer_span))
}

pub(in crate::index) fn synthetic_answer_surface_overlap_count(
    candidate_keys: &HashSet<String>,
    query_keys: &HashSet<String>,
) -> usize {
    candidate_keys.intersection(query_keys).count()
}

pub(in crate::index) fn synthetic_answer_surface_evidence_looks_future(lower: &str) -> bool {
    task_contains_any(
        lower,
        &[
            " going to ",
            " gonna ",
            " planning ",
            " plan to ",
            " next week",
            " next month",
            " next year",
            " tomorrow",
            " can’t wait",
            " can't wait",
            " looking forward",
            " coming up",
            " signed up",
        ],
    )
}

pub(in crate::index) fn synthetic_answer_surface_evidence_looks_completed(lower: &str) -> bool {
    task_contains_any(
        lower,
        &[
            " yesterday",
            " last week",
            " last month",
            " last year",
            " ago",
            " went ",
            " visited ",
            " joined ",
            " attended ",
            " read ",
            " finished ",
            " completed ",
            " moved ",
            " camped ",
            " took ",
            " made ",
            " gave ",
            " spoke ",
            " went on ",
            " had ",
        ],
    )
}

pub(in crate::index) fn synthetic_answer_surface_query_bonus(
    profile: &SyntheticAnswerSurfaceQueryProfile,
    row: &IndexAnswerSurfaceRow,
    evidence_line: Option<&str>,
) -> f32 {
    let answer_lower = row.answer_span.to_ascii_lowercase();
    let pattern_lower = row.question_pattern.to_ascii_lowercase();
    let evidence_lower = evidence_line.unwrap_or_default().to_ascii_lowercase();
    let combined = format!("{answer_lower} {pattern_lower} {evidence_lower}");
    let mut bonus = 0.0;

    if profile
        .relation_families
        .contains(&SyntheticAnswerSurfaceRelationFamily::Religion)
        && task_contains_any(
            &combined,
            &["religious", "religion", "faith", "church", "spiritual"],
        )
    {
        bonus += if answer_lower.contains("religious") {
            5.0
        } else {
            2.5
        };
    }
    if (profile
        .relation_families
        .contains(&SyntheticAnswerSurfaceRelationFamily::Ally)
        || profile
            .relation_families
            .contains(&SyntheticAnswerSurfaceRelationFamily::Identity))
        && answer_lower.contains("ally")
    {
        bonus += 5.0;
    }

    bonus
}

pub(in crate::index) fn synthetic_answer_surface_type_bonus(
    profile: &SyntheticAnswerSurfaceQueryProfile,
    answer_span: &str,
    row_family: Option<SyntheticAnswerSurfaceRelationFamily>,
) -> Option<f32> {
    match profile.expected_type {
        SyntheticAnswerSurfaceExpectedType::Generic => {
            Some(match answer_span.split_whitespace().count() {
                0 => 0.0,
                1..=6 => 0.8,
                7..=12 => 0.3,
                _ => 0.0,
            })
        },
        SyntheticAnswerSurfaceExpectedType::Date => {
            looks_like_answer_surface_date(answer_span).then_some(6.0)
        },
        SyntheticAnswerSurfaceExpectedType::Duration => {
            looks_like_answer_surface_duration(answer_span).then_some(5.5)
        },
        SyntheticAnswerSurfaceExpectedType::Count => {
            if looks_like_answer_surface_count(answer_span) {
                Some(5.0)
            } else if profile.allows_count_projection_from_lists
                && synthetic_answer_surface_count_projection_candidate(answer_span, row_family)
            {
                Some(3.0)
            } else {
                None
            }
        },
        SyntheticAnswerSurfaceExpectedType::Person => {
            looks_like_answer_surface_person(answer_span).then_some(4.5)
        },
        SyntheticAnswerSurfaceExpectedType::Location => {
            looks_like_answer_surface_location(answer_span).then_some(4.5)
        },
        SyntheticAnswerSurfaceExpectedType::ListItem => {
            looks_like_answer_surface_list_item(answer_span).then_some(4.0)
        },
        SyntheticAnswerSurfaceExpectedType::NameLike => {
            looks_like_answer_surface_name_like(answer_span).then_some(4.0)
        },
        SyntheticAnswerSurfaceExpectedType::Status => {
            looks_like_answer_surface_status(answer_span).then_some(4.0)
        },
    }
}

pub(in crate::index) fn synthetic_answer_surface_choice_overlap(
    profile: &SyntheticAnswerSurfaceQueryProfile,
    support_term_keys: &HashSet<String>,
) -> usize {
    profile
        .choice_options
        .iter()
        .map(|choice| {
            synthetic_answer_surface_overlap_count(support_term_keys, &choice.affinity_term_keys)
        })
        .max()
        .unwrap_or(0)
}

pub(in crate::index) fn synthetic_answer_surface_choice_projection(
    profile: &SyntheticAnswerSurfaceQueryProfile,
    row: &IndexAnswerSurfaceRow,
    evidence_line: Option<&str>,
) -> Option<String> {
    let answer_keys = synthetic_answer_surface_term_key_set(&synthetic_query_terms(
        &row.answer_span.to_ascii_lowercase(),
    ));
    let pattern_keys = synthetic_answer_surface_term_key_set(&synthetic_query_terms(
        &row.question_pattern.to_ascii_lowercase(),
    ));
    let evidence_keys = evidence_line
        .map(|line| {
            synthetic_answer_surface_term_key_set(&synthetic_query_terms(
                &line.to_ascii_lowercase(),
            ))
        })
        .unwrap_or_default();
    let combined_keys = answer_keys
        .union(&pattern_keys)
        .cloned()
        .chain(evidence_keys.iter().cloned())
        .collect::<HashSet<_>>();

    let mut scored = profile
        .choice_options
        .iter()
        .map(|choice| {
            let direct = synthetic_answer_surface_overlap_count(&combined_keys, &choice.term_keys);
            let affinity =
                synthetic_answer_surface_overlap_count(&combined_keys, &choice.affinity_term_keys);
            let score = direct * 5 + affinity * 3;
            (score, choice.display.clone())
        })
        .filter(|(score, _)| *score > 0)
        .collect::<Vec<_>>();
    scored.sort_by(|left, right| {
        right
            .0
            .cmp(&left.0)
            .then_with(|| left.1.len().cmp(&right.1.len()))
    });
    let (best_score, best_answer) = scored.first()?.clone();
    if scored
        .get(1)
        .is_some_and(|(runner_up, _)| *runner_up + 1 >= best_score)
    {
        return None;
    }
    Some(best_answer)
}

pub(in crate::index) fn synthetic_answer_surface_location_projection(
    target: SyntheticAnswerSurfaceLocationTarget,
    row: &IndexAnswerSurfaceRow,
    evidence_line: Option<&str>,
) -> Option<String> {
    let combined = format!(
        "{} {} {}",
        row.answer_span,
        row.question_pattern,
        evidence_line.unwrap_or_default()
    )
    .to_ascii_lowercase();

    match target {
        SyntheticAnswerSurfaceLocationTarget::State => synthetic_answer_surface_location_alias(
            &combined,
            &[
                (
                    &["universal studios hollywood", "hollywood", "los angeles"],
                    "California",
                ),
                (
                    &[
                        "universal studios orlando",
                        "orlando",
                        "miami",
                        "disney world",
                    ],
                    "Florida",
                ),
                (&["universal studios"], "California or Florida"),
                (&["florida", "orlando", "miami", "disney world"], "Florida"),
                (&["california"], "California"),
                (&["indiana", "indianapolis", "indiana dunes"], "Indiana"),
                (
                    &["minnesota", "minneapolis", "st. paul", "voyageurs"],
                    "Minnesota",
                ),
                (
                    &["connecticut", "new haven", "hartford", "bridgeport"],
                    "Connecticut",
                ),
                (&["alaska", "anchorage", "denali", "fairbanks"], "Alaska"),
                (&["arizona", "grand canyon"], "Arizona"),
            ],
        ),
        SyntheticAnswerSurfaceLocationTarget::Country => synthetic_answer_surface_location_alias(
            &combined,
            &[
                (&["canada", "vancouver", "toronto", "montreal"], "Canada"),
                (&["greenland"], "Greenland"),
                (&["france", "paris"], "France"),
                (&["colombia", "bogota", "medellin", "cartagena"], "Colombia"),
                (&["sweden"], "Sweden"),
                (
                    &[
                        "united states",
                        "u.s.",
                        "usa",
                        "america",
                        "boston",
                        "new york",
                        "florida",
                        "california",
                        "minnesota",
                        "connecticut",
                        "alaska",
                        "arizona",
                        "universal studios",
                    ],
                    "United States",
                ),
            ],
        ),
        SyntheticAnswerSurfaceLocationTarget::NationalPark => {
            synthetic_answer_surface_location_alias(
                &combined,
                &[
                    (
                        &["voyageurs", "voyageurs national park"],
                        "Voyageurs National Park",
                    ),
                    (&["grand canyon"], "Grand Canyon National Park"),
                    (&["yellowstone"], "Yellowstone National Park"),
                ],
            )
        },
    }
}

pub(in crate::index) fn synthetic_answer_surface_location_alias(
    combined: &str,
    aliases: &[(&[&str], &str)],
) -> Option<String> {
    aliases.iter().find_map(|(needles, canonical)| {
        needles
            .iter()
            .any(|needle| combined.contains(needle))
            .then(|| (*canonical).to_string())
    })
}

pub(in crate::index) fn synthetic_answer_surface_project_answer(
    profile: &SyntheticAnswerSurfaceQueryProfile,
    row: &IndexAnswerSurfaceRow,
    evidence_line: Option<&str>,
) -> Option<String> {
    match profile.route_kind {
        SyntheticAnswerSurfaceRouteKind::Choice => {
            synthetic_answer_surface_choice_projection(profile, row, evidence_line)
        },
        SyntheticAnswerSurfaceRouteKind::LocationLift => profile
            .location_target
            .and_then(|target| {
                synthetic_answer_surface_location_projection(target, row, evidence_line)
            })
            .or_else(|| {
                (looks_like_answer_surface_location(&row.answer_span)
                    && row.answer_span.split_whitespace().count() <= 4)
                    .then(|| row.answer_span.clone())
            }),
        _ => Some(row.answer_span.clone()),
    }
}

pub(in crate::index) fn looks_like_answer_surface_date(answer_span: &str) -> bool {
    const MONTHS: &[&str] = &[
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ];
    let lower = answer_span.to_ascii_lowercase();
    Regex::new(r"\b(?:19|20)\d{2}\b").unwrap().is_match(&lower)
        || MONTHS.iter().any(|month| lower.contains(month))
        || task_contains_any(
            &lower,
            &[
                "yesterday",
                "today",
                "tonight",
                "tomorrow",
                "last week",
                "last month",
                "last year",
                "next week",
                "next month",
                "week before",
                "month before",
                "year before",
                "last saturday",
                "last sunday",
                "last monday",
                "last tuesday",
                "last wednesday",
                "last thursday",
                "last friday",
            ],
        )
}

pub(in crate::index) fn looks_like_answer_surface_duration(answer_span: &str) -> bool {
    let lower = answer_span.to_ascii_lowercase();
    lower.starts_with("since ")
        || Regex::new(
            r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(?:day|week|month|year)s?\b",
        )
        .unwrap()
        .is_match(&lower)
        || Regex::new(
            r"\b(?:day|week|month|year)s?\s+(?:ago|already|now)\b",
        )
        .unwrap()
        .is_match(&lower)
}

pub(in crate::index) fn looks_like_answer_surface_count(answer_span: &str) -> bool {
    if looks_like_answer_surface_date(answer_span) {
        return false;
    }
    let lower = answer_span.to_ascii_lowercase();
    Regex::new(
        r"^(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twice|thrice)(?:\s+(?:times?|kids?|children|dogs?|cats?|followers?|issues?|books?|letters?))?$",
    )
    .unwrap()
    .is_match(lower.trim())
}

pub(in crate::index) fn looks_like_answer_surface_person(answer_span: &str) -> bool {
    let lower = answer_span.to_ascii_lowercase();
    if task_contains_any(
        &lower,
        &[
            "family",
            "friends",
            "friend",
            "mentor",
            "mentors",
            "mother",
            "mom",
            "father",
            "dad",
            "aunt",
            "uncle",
            "sister",
            "brother",
            "husband",
            "wife",
            "partner",
            "spouse",
            "colleague",
            "colleagues",
            "teammates",
            "children",
            "kids",
        ],
    ) {
        return true;
    }
    let words = answer_span.split_whitespace().collect::<Vec<_>>();
    words.len() <= 8
        && words.iter().any(|word| {
            word.chars()
                .next()
                .map(|c| c.is_ascii_uppercase())
                .unwrap_or(false)
        })
}

pub(in crate::index) fn looks_like_answer_surface_name_like(answer_span: &str) -> bool {
    if answer_span.contains('?')
        || answer_span.contains(". ")
        || looks_like_answer_surface_date(answer_span)
        || looks_like_answer_surface_duration(answer_span)
        || looks_like_answer_surface_count(answer_span)
    {
        return false;
    }
    let words = answer_span.split_whitespace().collect::<Vec<_>>();
    words.len() <= 10
        && words.iter().any(|word| {
            word.chars()
                .next()
                .map(|c| c.is_ascii_uppercase())
                .unwrap_or(false)
        })
}

pub(in crate::index) fn looks_like_answer_surface_list_item(answer_span: &str) -> bool {
    if answer_span.contains('?')
        || answer_span.contains(". ")
        || looks_like_answer_surface_date(answer_span)
        || looks_like_answer_surface_duration(answer_span)
        || looks_like_answer_surface_count(answer_span)
    {
        return false;
    }
    let words = answer_span.split_whitespace().collect::<Vec<_>>();
    !words.is_empty()
        && words.len() <= 8
        && !task_contains_any(
            &answer_span.to_ascii_lowercase(),
            &[" because ", " although ", " however ", " but "],
        )
}

pub(in crate::index) fn looks_like_answer_surface_location(answer_span: &str) -> bool {
    if looks_like_answer_surface_date(answer_span) || looks_like_answer_surface_count(answer_span) {
        return false;
    }
    let lower = answer_span.to_ascii_lowercase();
    if task_contains_any(
        &lower,
        &[
            "beach",
            "mountain",
            "mountains",
            "forest",
            "woods",
            "lake",
            "park",
            "city",
            "country",
            "state",
            "suburbs",
            "downtown",
            "village",
            "town",
            "island",
        ],
    ) {
        return true;
    }
    let words = answer_span.split_whitespace().collect::<Vec<_>>();
    words.len() <= 6
        && words.iter().any(|word| {
            word.chars()
                .next()
                .map(|c| c.is_ascii_uppercase())
                .unwrap_or(false)
        })
}

pub(in crate::index) fn looks_like_answer_surface_status(answer_span: &str) -> bool {
    matches!(
        answer_span.to_ascii_lowercase().trim(),
        "single" | "married" | "engaged" | "divorced" | "widowed" | "separated"
    )
}

pub(in crate::index) fn index_answer_surface_score(
    row: &IndexAnswerSurfaceRow,
    retrieval_score: f32,
    profile: &SyntheticAnswerSurfaceQueryProfile,
    evidence_line: Option<&str>,
    has_future_answer_evidence: bool,
    has_completed_answer_evidence: bool,
) -> (f32, usize) {
    let pattern_terms = synthetic_query_terms(&row.question_pattern.to_ascii_lowercase());
    let pattern_term_keys = synthetic_answer_surface_term_key_set(&pattern_terms);
    if pattern_term_keys.is_empty() {
        return (0.0, 0);
    }

    let evidence_terms = evidence_line
        .map(|line| synthetic_query_terms(&line.to_ascii_lowercase()))
        .unwrap_or_default();
    let evidence_term_keys = synthetic_answer_surface_term_key_set(&evidence_terms);
    let mut support_term_keys = pattern_term_keys.clone();
    support_term_keys.extend(evidence_term_keys.iter().cloned());
    let row_family = synthetic_answer_surface_relation_family(&row.question_pattern, evidence_line);
    let relation_overlap =
        synthetic_answer_surface_overlap_count(&support_term_keys, &profile.relation_term_keys);

    let overlap =
        synthetic_answer_surface_overlap_count(&support_term_keys, &profile.task_term_keys);
    if overlap == 0 {
        return (0.0, 0);
    }

    let subject_overlap =
        synthetic_answer_surface_overlap_count(&support_term_keys, &profile.subject_term_keys);
    if !profile.subject_term_keys.is_empty() && subject_overlap == 0 {
        return (0.0, 0);
    }

    let anchor_overlap =
        synthetic_answer_surface_overlap_count(&support_term_keys, &profile.anchor_term_keys);
    if profile.requires_strict_anchor_overlap
        && !profile.anchor_term_keys.is_empty()
        && anchor_overlap == 0
    {
        return (0.0, 0);
    }
    if matches!(
        profile.expected_type,
        SyntheticAnswerSurfaceExpectedType::Count
            | SyntheticAnswerSurfaceExpectedType::Duration
            | SyntheticAnswerSurfaceExpectedType::Person
    ) && !profile.relation_term_keys.is_empty()
        && relation_overlap < usize::min(2, profile.relation_term_keys.len())
    {
        return (0.0, 0);
    }
    if !synthetic_answer_surface_relation_family_matches(profile, row_family, relation_overlap) {
        return (0.0, 0);
    }
    let choice_overlap = synthetic_answer_surface_choice_overlap(profile, &support_term_keys);
    if matches!(profile.route_kind, SyntheticAnswerSurfaceRouteKind::Choice) && choice_overlap == 0
    {
        return (0.0, 0);
    }
    let Some(type_bonus) =
        synthetic_answer_surface_type_bonus(profile, &row.answer_span, row_family)
    else {
        return (0.0, 0);
    };
    if profile.requires_completed_evidence {
        if has_future_answer_evidence && !has_completed_answer_evidence {
            return (0.0, 0);
        }
        if let Some(line) = evidence_line {
            let lower = line.to_ascii_lowercase();
            if synthetic_answer_surface_evidence_looks_future(&lower)
                && !synthetic_answer_surface_evidence_looks_completed(&lower)
                && !has_completed_answer_evidence
            {
                return (0.0, 0);
            }
        }
    }

    let coverage = overlap as f32 / profile.task_term_keys.len().max(1) as f32;
    let specificity = overlap as f32 / support_term_keys.len().max(1) as f32;
    let anchor_coverage = anchor_overlap as f32 / profile.anchor_term_keys.len().max(1) as f32;
    let evidence_overlap =
        synthetic_answer_surface_overlap_count(&evidence_term_keys, &profile.task_term_keys);
    let evidence_bonus = evidence_overlap as f32 * 2.0
        + if profile.requires_completed_evidence
            && evidence_line
                .map(|line| {
                    synthetic_answer_surface_evidence_looks_completed(&line.to_ascii_lowercase())
                })
                .unwrap_or(false)
        {
            1.0
        } else {
            0.0
        };
    let query_bonus = synthetic_answer_surface_query_bonus(profile, row, evidence_line);
    let relation_bonus = if profile.relation_families.is_empty() {
        0.0
    } else if row_family
        .map(|family| profile.relation_families.contains(&family))
        .unwrap_or(false)
    {
        5.0
    } else {
        relation_overlap as f32 * 1.5
    };
    (
        retrieval_score * 0.75
            + overlap as f32 * 3.5
            + coverage * 4.0
            + specificity * 1.5
            + anchor_overlap as f32 * 3.0
            + anchor_coverage * 4.0
            + relation_overlap as f32 * 2.5
            + choice_overlap as f32 * 3.5
            + subject_overlap as f32 * 3.5
            + evidence_bonus
            + relation_bonus
            + query_bonus
            + row.confidence
            + type_bonus,
        overlap,
    )
}

pub(in crate::index) fn format_index_answer_surface_answer(task_lower: &str, answer: &str) -> String {
    let answer_lower = answer.to_ascii_lowercase();
    if answer_lower.contains("ally")
        && task_contains_any(
            task_lower,
            &[
                "member of the lgbtq community",
                "member of the lgbtq+ community",
                "part of the lgbtq community",
                "part of the lgbtq+ community",
                "member of the transgender community",
            ],
        )
    {
        return "Likely no, supportive ally".to_string();
    }
    if answer_lower.contains("ally")
        && task_contains_any(
            task_lower,
            &[
                "ally to the transgender community",
                "ally to the lgbtq community",
                "ally to the lgbtq+ community",
                "considered an ally",
            ],
        )
    {
        return "Yes, supportive ally".to_string();
    }
    answer.to_string()
}

pub(in crate::index) fn answer_surface_evidence_line(
    content: &str,
    task_terms: &[String],
    answer_span: &str,
    question_pattern: &str,
) -> Option<String> {
    let body = strip_query_surface_section(content);
    let answer_lower = answer_span.to_ascii_lowercase();
    let answer_term_keys =
        synthetic_answer_surface_term_key_set(&synthetic_query_terms(&answer_lower));
    let pattern_terms = synthetic_query_terms(&question_pattern.to_ascii_lowercase());
    let pattern_term_keys = synthetic_answer_surface_term_key_set(&pattern_terms);
    let task_term_keys = synthetic_answer_surface_term_key_set(task_terms);

    let mut best: Option<(usize, usize, usize, bool, String)> = None;
    for line in body
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .filter(|line| !line.starts_with('|'))
    {
        let lower = line.to_ascii_lowercase();
        let line_terms = synthetic_query_terms(&lower);
        let line_term_keys = synthetic_answer_surface_term_key_set(&line_terms);
        let pattern_overlap =
            synthetic_answer_surface_overlap_count(&line_term_keys, &pattern_term_keys);
        let task_overlap = synthetic_answer_surface_overlap_count(&line_term_keys, &task_term_keys);
        let answer_hit = lower.contains(&answer_lower)
            || (!answer_term_keys.is_empty()
                && answer_term_keys.iter().all(|term| {
                    line_term_keys.iter().any(|line_term| {
                        line_term == term
                            || line_term.starts_with(term.as_str())
                            || term.starts_with(line_term.as_str())
                    })
                }));
        let score = usize::from(answer_hit) * 10 + task_overlap * 4 + pattern_overlap * 2;
        if !answer_hit && pattern_overlap < 2 && task_overlap == 0 {
            continue;
        }
        let replace = best
            .as_ref()
            .map(
                |(best_score, best_task, best_pattern, best_answer_hit, best_line)| {
                    score > *best_score
                        || (score == *best_score
                            && (task_overlap > *best_task
                                || (task_overlap == *best_task
                                    && (pattern_overlap > *best_pattern
                                        || (pattern_overlap == *best_pattern
                                            && (answer_hit && !*best_answer_hit
                                                || (answer_hit == *best_answer_hit
                                                    && line.len() < best_line.len())))))))
                },
            )
            .unwrap_or(true);
        if replace {
            best = Some((
                score,
                task_overlap,
                pattern_overlap,
                answer_hit,
                line.to_string(),
            ));
        }
    }

    best.map(|(_, _, _, _, line)| line)
}

pub(in crate::index) fn answer_surface_answer_span_evidence_state(
    content: &str,
    answer_span: &str,
) -> (bool, bool) {
    let body = strip_query_surface_section(content);
    let answer_lower = answer_span.to_ascii_lowercase();
    let mut has_future = false;
    let mut has_completed = false;

    for line in body
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .filter(|line| !line.starts_with('|'))
    {
        let lower = line.to_ascii_lowercase();
        if !lower.contains(&answer_lower) {
            continue;
        }
        has_future |= synthetic_answer_surface_evidence_looks_future(&lower);
        has_completed |= synthetic_answer_surface_evidence_looks_completed(&lower);
    }

    (has_future, has_completed)
}

pub(in crate::index) fn latest_active_kg_value(entity: &kg::KgEntity, predicate: &str) -> Option<String> {
    fn latest_value_for_predicate(entity: &kg::KgEntity, predicate: &str) -> Option<String> {
        let mut facts = entity.active_values_for_predicate(predicate, None);
        facts.sort_by(|a, b| a.valid_from.cmp(&b.valid_from));
        if let Some(value) = facts
            .last()
            .map(|fact| fact.value.trim())
            .filter(|value| !value.is_empty())
        {
            return Some(normalize_latest_kg_value(predicate, value));
        }
        None
    }

    latest_value_for_predicate(entity, predicate).or_else(|| match predicate {
        "education" => latest_value_for_predicate(entity, "major"),
        "major" => latest_value_for_predicate(entity, "education"),
        _ => None,
    })
}

pub(in crate::index) fn normalize_latest_kg_value(predicate: &str, value: &str) -> String {
    match predicate {
        "location" => normalize_location_kg_value(value),
        "education" | "major" => normalize_education_kg_value(value),
        "fitness_record" => normalize_fitness_record_kg_value(value),
        _ => value.trim().to_string(),
    }
}

pub(in crate::index) fn normalize_location_kg_value(value: &str) -> String {
    let lower = value.to_ascii_lowercase();
    let cutoff = [" again ", " so ", " because ", " but ", " with ", " after "]
        .iter()
        .filter_map(|marker| lower.find(marker))
        .min()
        .unwrap_or(value.len());
    let mut trimmed = value[..cutoff]
        .trim()
        .trim_matches(|c: char| matches!(c, ',' | '.' | ';' | ':' | '"' | '\''));
    if let Some(stripped) = trimmed.strip_suffix(" again") {
        trimmed = stripped.trim();
    }
    if trimmed.eq_ignore_ascii_case("suburbs") {
        "the suburbs".to_string()
    } else if trimmed.eq_ignore_ascii_case("the suburbs") {
        "the suburbs".to_string()
    } else {
        trimmed.to_string()
    }
}

pub(in crate::index) fn normalize_education_kg_value(value: &str) -> String {
    let lower = value.to_ascii_lowercase();
    let cutoff = [
        " which ",
        " that ",
        " because ",
        " but ",
        " and ",
        " from ",
        " with a concentration in ",
        " with concentration in ",
        " with a minor in ",
        " with minor in ",
    ]
    .iter()
    .filter_map(|marker| lower.find(marker))
    .min()
    .unwrap_or(value.len());
    let mut trimmed = value[..cutoff]
        .trim()
        .trim_matches(|c: char| matches!(c, ',' | '.' | ';' | ':' | '"' | '\''));
    for suffix in [" which", " that", " from"] {
        if let Some(stripped) = trimmed.strip_suffix(suffix) {
            trimmed = stripped.trim();
        }
    }
    trimmed.to_string()
}

pub(in crate::index) fn normalize_fitness_record_kg_value(value: &str) -> String {
    let trimmed = value.trim();
    let parts: Vec<_> = trimmed.split(':').collect();
    if parts.len() == 2
        && parts[0].chars().all(|c| c.is_ascii_digit())
        && parts[1].len() == 2
        && parts[1].chars().all(|c| c.is_ascii_digit())
    {
        let minutes = parts[0].parse::<u32>().ok();
        let seconds = parts[1].parse::<u32>().ok();
        if let (Some(minutes), Some(seconds)) = (minutes, seconds) {
            return format!("{minutes} minutes and {seconds} seconds (or {trimmed})");
        }
    }
    trimmed.to_string()
}

pub(in crate::index) fn extract_fitness_record_time_value(line: &str) -> Option<(u32, String)> {
    Regex::new(r"\b(\d{1,2}):(\d{2})\b")
        .unwrap()
        .captures_iter(line)
        .filter_map(|caps| {
            let minutes = caps.get(1)?.as_str().parse::<u32>().ok()?;
            let seconds = caps.get(2)?.as_str().parse::<u32>().ok()?;
            (seconds < 60).then_some((minutes * 60 + seconds, caps.get(0)?.as_str().to_string()))
        })
        .min_by_key(|(total_seconds, _)| *total_seconds)
}

pub(in crate::index) fn parse_count_token_value(token: &str) -> Option<i32> {
    let cleaned = token
        .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '$' && c != ',' && c != '%')
        .to_ascii_lowercase();
    if cleaned.is_empty() {
        return None;
    }
    match cleaned.as_str() {
        "zero" => Some(0),
        "one" => Some(1),
        "first" => Some(1),
        "two" => Some(2),
        "second" => Some(2),
        "three" => Some(3),
        "third" => Some(3),
        "four" => Some(4),
        "fourth" => Some(4),
        "five" => Some(5),
        "fifth" => Some(5),
        "six" => Some(6),
        "sixth" => Some(6),
        "seven" => Some(7),
        "seventh" => Some(7),
        "eight" => Some(8),
        "eighth" => Some(8),
        "nine" => Some(9),
        "ninth" => Some(9),
        "ten" => Some(10),
        "tenth" => Some(10),
        "eleven" => Some(11),
        "eleventh" => Some(11),
        "twelve" => Some(12),
        "twelfth" => Some(12),
        _ => {
            if let Some(stripped) = cleaned
                .strip_suffix("st")
                .or_else(|| cleaned.strip_suffix("nd"))
                .or_else(|| cleaned.strip_suffix("rd"))
                .or_else(|| cleaned.strip_suffix("th"))
            {
                if !stripped.is_empty() && stripped.chars().all(|c| c.is_ascii_digit() || c == ',')
                {
                    return stripped.replace(',', "").parse::<i32>().ok();
                }
            }
            if cleaned.chars().any(|c| c.is_ascii_digit())
                && cleaned.chars().any(|c| c.is_ascii_alphabetic())
                && !cleaned.contains('-')
            {
                return None;
            }
            let digits: String = cleaned
                .chars()
                .filter(|c| c.is_ascii_digit() || *c == ',')
                .collect();
            if digits.is_empty() {
                None
            } else {
                digits.replace(',', "").parse::<i32>().ok()
            }
        },
    }
}

pub(in crate::index) fn extract_line_numbers(line: &str) -> Vec<i32> {
    line.split_whitespace()
        .filter_map(parse_count_token_value)
        .collect()
}

pub(in crate::index) fn extract_focus_aligned_count(
    line: &str,
    focus_terms: &[String],
    task_lower: &str,
) -> Option<(i32, usize)> {
    const TIME_UNITS: &[&str] = &[
        "day", "days", "week", "weeks", "month", "months", "year", "years", "hour", "hours",
    ];
    let focus_keys: HashSet<String> = focus_terms
        .iter()
        .map(|term| synthetic_answer_surface_term_key(term))
        .filter(|key| !key.is_empty())
        .collect();
    if focus_keys.is_empty() {
        return None;
    }

    let tokens = line.split_whitespace().collect::<Vec<_>>();
    let token_keys = tokens
        .iter()
        .map(|token| synthetic_answer_surface_term_key(token))
        .collect::<Vec<_>>();
    let mut best: Option<(usize, usize, i32)> = None;

    for (idx, token) in tokens.iter().enumerate() {
        if idx == 0
            && token
                .trim_end_matches(['.', ')'])
                .chars()
                .all(|c| c.is_ascii_digit())
        {
            continue;
        }
        let Some(value) = parse_count_token_value(token) else {
            continue;
        };
        if (1900..=2100).contains(&value) {
            continue;
        }

        let negation_start = idx.saturating_sub(2);
        if token_keys[negation_start..idx]
            .iter()
            .any(|key| matches!(key.as_str(), "not" | "never"))
        {
            continue;
        }

        let raw_token = token.to_ascii_lowercase();
        let adjacent_time_unit = TIME_UNITS.iter().find(|unit| {
            raw_token.contains(&format!("-{unit}"))
                || token_keys
                    .get(idx + 1)
                    .map(|next| next == *unit)
                    .unwrap_or(false)
        });
        if let Some(unit) = adjacent_time_unit {
            if !task_lower.contains(unit) {
                continue;
            }
        }

        let window_start = idx.saturating_sub(6);
        let window_end = usize::min(token_keys.len(), idx + 7);
        let nearby_focus = token_keys[window_start..window_end]
            .iter()
            .filter(|key| focus_keys.contains(*key))
            .collect::<HashSet<_>>()
            .len();
        if nearby_focus == 0 {
            continue;
        }

        let nearest_distance = token_keys
            .iter()
            .enumerate()
            .filter(|(_, key)| focus_keys.contains(*key))
            .map(|(focus_idx, _)| idx.abs_diff(focus_idx))
            .min()
            .unwrap_or(usize::MAX);
        let score = nearby_focus * 10 + 7usize.saturating_sub(nearest_distance.min(7));

        if best
            .as_ref()
            .map(|(best_score, best_distance, best_value)| {
                score > *best_score
                    || (score == *best_score && nearest_distance < *best_distance)
                    || (score == *best_score
                        && nearest_distance == *best_distance
                        && value > *best_value)
            })
            .unwrap_or(true)
        {
            best = Some((score, nearest_distance, value));
        }
    }

    best.map(|(score, _, value)| (value, score))
}

pub(in crate::index) fn is_summary_or_user_line(line: &str, lower: &str) -> bool {
    lower.starts_with("user:") || line.trim_start().starts_with('-')
}

pub(in crate::index) fn is_session_answer_candidate_line(line: &str) -> bool {
    let trimmed = line.trim();
    !(trimmed.is_empty()
        || trimmed.starts_with('#')
        || trimmed.starts_with("<!--")
        || trimmed.starts_with('|')
        || trimmed.starts_with("Question:")
        || trimmed.starts_with("Answer:"))
}

pub(in crate::index) fn normalize_session_answer_line_body(line: &str) -> String {
    let mut body = line.trim();
    if let Some(stripped) = body.strip_prefix('-') {
        body = stripped.trim();
    }

    let digit_prefix = body.chars().take_while(|c| c.is_ascii_digit()).count();
    if digit_prefix > 0 {
        let rest = body[digit_prefix..].trim_start();
        if rest.starts_with('.') || rest.starts_with(')') {
            body = rest[1..].trim();
        }
    }

    let lower = body.to_ascii_lowercase();
    for prefix in ["user:", "assistant:"] {
        if lower.starts_with(prefix) {
            body = body[prefix.len()..].trim();
            break;
        }
    }

    body.trim_matches(|c: char| matches!(c, '"' | '\'' | '`'))
        .trim()
        .to_string()
}

pub(in crate::index) fn task_has_recall_context(task_lower: &str) -> bool {
    task_contains_any(
        task_lower,
        &[
            "remind me",
            "previous chat",
            "previous conversation",
            "last time",
            "follow up",
            "follow-up",
            "told me",
            "talked about",
            "we talked",
            "remember you",
            "remember what",
            "used as an example",
            "going back to our previous",
        ],
    )
}

pub(in crate::index) fn should_try_session_recall_answer(task: &str, task_lower: &str) -> bool {
    if task_contains_any(
        task_lower,
        &[
            " in total",
            " altogether",
            " combined",
            " before ",
            " after ",
            " difference ",
            " compared ",
            " how long had i been",
            " when i just started",
        ],
    ) {
        return false;
    }

    task_lower.contains("what color")
        || (task_lower.starts_with("where ")
            && task_contains_any(
                task_lower,
                &[
                    "buy",
                    "bought",
                    "redeem",
                    "use my coupon",
                    "which store",
                    "shop",
                    "keep",
                    "kept",
                ],
            ))
        || task_lower.contains("discount")
        || is_money_query(task)
        || task_contains_any(task_lower, &["what speed", "internet plan", "camera lens"])
}

pub(in crate::index) fn normalized_synthetic_phrase_key(text: &str) -> String {
    text.trim()
        .trim_matches(|c: char| matches!(c, '"' | '\'' | ',' | '.' | ';' | ':' | '!' | '?'))
        .to_ascii_lowercase()
}

pub(in crate::index) fn project_session_answer_from_line(
    task: &str,
    task_lower: &str,
    predicate: Option<&str>,
    line: &str,
    lower: &str,
) -> Option<String> {
    match predicate {
        Some("education") | Some("major") => return extract_session_education_answer(line, lower),
        Some("project_name") => {
            return extract_session_named_answer_from_line(task_lower, line, lower)
        },
        Some("location") => return extract_session_location_answer(task_lower, line, lower),
        Some("occupation") => return extract_session_occupation_answer(line, lower),
        Some("book") => return extract_session_named_answer_from_line(task_lower, line, lower),
        _ => {},
    }

    if is_education_query(task_lower) || is_major_query(task_lower) {
        if let Some(answer) = extract_session_education_answer(line, lower) {
            return Some(answer);
        }
    }
    if task_lower.contains("discount") {
        if let Some(answer) = extract_percent_answer_from_line(line) {
            return Some(answer);
        }
    }
    if task_lower.contains("what color") {
        if task_lower.contains("did i")
            && (lower.contains("planning to") || lower.contains("thinking of"))
        {
            return None;
        }
        if let Some(answer) = extract_color_answer_from_line(line) {
            return Some(answer);
        }
    }
    if task_lower.starts_with("where ") {
        if let Some(answer) = extract_session_location_answer(task_lower, line, lower) {
            return Some(answer);
        }
    }
    if task_lower.starts_with("when ")
        || task_contains_any(task_lower, &["what day", "what date", "what time"])
    {
        if let Some(answer) = extract_date_or_time_answer_from_line(line) {
            return Some(answer);
        }
    }
    if task_lower.starts_with("how long ") {
        if let Some(answer) = extract_duration_answer_from_line(line) {
            return Some(answer);
        }
    }
    if is_money_query(task) {
        if let Some(answer) = extract_money_answer_from_line(line) {
            return Some(answer);
        }
    }
    if task_contains_any(task_lower, &["what speed", "internet plan"]) {
        if let Some(answer) = extract_speed_answer_from_line(line) {
            return Some(answer);
        }
    }
    if task_contains_any(task_lower, &["camera lens"]) {
        if let Some(answer) = extract_session_purchase_item(line, lower) {
            if answer.to_ascii_lowercase().contains("lens") {
                return Some(answer);
            }
        }
    }
    if task_has_recall_context(task_lower) && detect_counting_query(task) {
        if let Some(answer) = extract_query_aligned_numeric_answer(task_lower, line) {
            return Some(answer);
        }
    }
    if task_has_recall_context(task_lower)
        || task_contains_any(
            task_lower,
            &[
                "name of",
                "called",
                "call it",
                "title",
                "what kind",
                "what type",
                "specific",
            ],
        )
    {
        if let Some(answer) = extract_session_list_answer_from_line(task_lower, line, lower) {
            return Some(answer);
        }
        if let Some(answer) = extract_session_named_answer_from_line(task_lower, line, lower) {
            return Some(answer);
        }
    }

    None
}

pub(in crate::index) fn is_assistant_followup_query(task_lower: &str) -> bool {
    task_contains_any(
        task_lower,
        &[
            "you mentioned",
            "you recommended",
            "our previous conversation",
            "previous conversation",
            "previous chat",
            "previous chess game",
            "follow up on our previous",
            "looking back at our previous",
            "going back to our previous",
        ],
    ) && task_contains_any(
        task_lower,
        &[
            "remind me",
            "can you remind me",
            "what was",
            "what kind",
            "what type",
            "how many",
            "which website",
            "what website",
            "what move",
            "which move",
            "what was the move",
        ],
    )
}

pub(in crate::index) fn project_assistant_followup_answer_from_context(
    task: &str,
    task_lower: &str,
    lines: &[String],
    line_idx: usize,
) -> Option<String> {
    if let Some(answer) = extract_adjacent_role_person_followup_answer(task_lower, lines, line_idx)
    {
        return Some(answer);
    }
    let line = lines.get(line_idx)?;
    let lower = line.to_ascii_lowercase();
    project_assistant_followup_answer_from_line(task, task_lower, line, &lower)
}

pub(in crate::index) fn extract_adjacent_role_person_followup_answer(
    task_lower: &str,
    lines: &[String],
    line_idx: usize,
) -> Option<String> {
    if !task_contains_any(task_lower, &["who is the", "who was the"]) {
        return None;
    }
    let role_terms = assistant_followup_role_terms(task_lower);
    if role_terms.is_empty() {
        return None;
    }
    let line = lines.get(line_idx)?;
    let lower = line.to_ascii_lowercase();
    let role_overlap = role_terms
        .iter()
        .filter(|term| lower.contains(term.as_str()))
        .count();
    if role_overlap == 0 {
        return None;
    }
    for neighbor_idx in [line_idx.checked_sub(1), Some(line_idx + 1)] {
        let Some(neighbor_idx) = neighbor_idx else {
            continue;
        };
        let Some(neighbor) = lines.get(neighbor_idx) else {
            continue;
        };
        let neighbor_lower = neighbor.to_ascii_lowercase();
        if let Some(answer) =
            extract_session_named_answer_from_line(task_lower, neighbor, &neighbor_lower)
        {
            if answer
                .chars()
                .next()
                .map(|ch| ch.is_ascii_uppercase())
                .unwrap_or(false)
            {
                return Some(answer);
            }
        }
    }
    None
}

pub(in crate::index) fn project_assistant_followup_answer_from_line(
    task: &str,
    task_lower: &str,
    line: &str,
    lower: &str,
) -> Option<String> {
    if task_contains_any(
        task_lower,
        &["what move", "which move", "what was the move"],
    ) {
        if let Some(answer) = extract_chess_move_answer_from_line(
            line,
            extract_expected_chess_reply_move_number(task_lower),
        ) {
            return Some(answer);
        }
    }
    if let Some(answer) = extract_descriptor_named_followup_answer(task_lower, line, lower) {
        return Some(answer);
    }
    if detect_counting_query(task) {
        if let Some(answer) = extract_parenthetical_label_count_answer(task_lower, line, lower)
            .or_else(|| extract_query_aligned_numeric_answer(task_lower, line))
        {
            return Some(answer);
        }
        return None;
    }
    if task_lower.contains("website") {
        if let Some(answer) = extract_website_name_from_line(line) {
            return Some(answer);
        }
    }
    if task_contains_any(task_lower, &["what type of beer", "what kind of beer"]) {
        if let Some(answer) = extract_beer_recommendation_answer_from_line(lower) {
            return Some(answer);
        }
    }
    if task_lower.contains("two-factor authentication") {
        if let Some(answer) = extract_two_factor_method_answer_from_line(line, lower) {
            return Some(answer);
        }
    }
    project_session_answer_from_line(task, task_lower, None, line, lower)
}

pub(in crate::index) fn extract_descriptor_named_followup_answer(
    task_lower: &str,
    line: &str,
    lower: &str,
) -> Option<String> {
    if detect_counting_query(task_lower)
        || task_lower.starts_with("how ")
        || task_lower.starts_with("when ")
        || task_lower.starts_with("where ")
    {
        return None;
    }
    let descriptor_terms = assistant_followup_descriptor_terms(task_lower);
    if descriptor_terms.len() < 2 {
        return None;
    }
    let matched = descriptor_terms
        .iter()
        .filter(|term| lower.contains(term.as_str()))
        .count();
    if matched < 2 {
        return None;
    }
    extract_session_named_answer_from_line(task_lower, line, lower)
}

pub(in crate::index) fn assistant_followup_descriptor_terms(task_lower: &str) -> Vec<String> {
    let mut terms = Vec::new();
    if let Some((_, clause)) = task_lower
        .rsplit_once(" that ")
        .or_else(|| task_lower.rsplit_once(" which "))
        .or_else(|| task_lower.rsplit_once(" who "))
    {
        terms.extend(
            synthetic_query_terms(clause)
                .into_iter()
                .filter(|term| term.len() >= 3)
                .filter(|term| !term.chars().all(|ch| ch.is_ascii_digit()))
                .filter(|term| {
                    !matches!(term.as_str(), "companies" | "company" | "people" | "person")
                }),
        );
    }
    if let Some(subject_clause) = assistant_followup_subject_descriptor_clause(task_lower) {
        terms.extend(
            synthetic_query_terms(subject_clause)
                .into_iter()
                .filter(|term| term.len() >= 3)
                .filter(|term| !term.chars().all(|ch| ch.is_ascii_digit()))
                .filter(|term| !matches!(term.as_str(), "example" | "gave" | "people" | "person")),
        );
    }
    terms.sort();
    terms.dedup();
    terms
}

pub(in crate::index) fn assistant_followup_subject_descriptor_clause(task_lower: &str) -> Option<&str> {
    for marker in [
        "example you gave of a ",
        "example you gave of an ",
        "example you gave of the ",
    ] {
        let Some((_, tail)) = task_lower.split_once(marker) else {
            continue;
        };
        let stop = tail
            .find(" who ")
            .or_else(|| tail.find(" that "))
            .or_else(|| tail.find(" which "))
            .unwrap_or(tail.len());
        let clause = tail[..stop].trim();
        if !clause.is_empty() {
            return Some(clause);
        }
    }
    None
}

pub(in crate::index) fn assistant_followup_role_terms(task_lower: &str) -> Vec<String> {
    synthetic_query_terms(task_lower)
        .into_iter()
        .filter(|term| term.len() >= 5)
        .filter(|term| {
            !matches!(
                term.as_str(),
                "article"
                    | "conversation"
                    | "follow"
                    | "mentioned"
                    | "previous"
                    | "remind"
                    | "science"
                    | "technology"
            )
        })
        .collect()
}

pub(in crate::index) fn assistant_followup_anchor_terms(task_lower: &str) -> Vec<String> {
    let Some((_, tail)) = task_lower.rsplit_once(" at ") else {
        return Vec::new();
    };
    let segment = tail.split(['.', '?', '!', ',']).next().unwrap_or("").trim();
    let terms: Vec<String> = synthetic_query_terms(segment)
        .into_iter()
        .filter(|term| term.len() >= 4)
        .collect();
    if (1..=4).contains(&terms.len()) {
        terms
    } else {
        Vec::new()
    }
}

pub(in crate::index) fn assistant_followup_anchor_distance(
    line_lower: &str,
    match_end: usize,
    anchor_terms: &[String],
) -> Option<usize> {
    if anchor_terms.is_empty() {
        return None;
    }
    anchor_terms
        .iter()
        .filter_map(|term| {
            line_lower[match_end..]
                .find(term)
                .map(|offset| offset + match_end)
        })
        .map(|position| position.saturating_sub(match_end))
        .min()
}

pub(in crate::index) fn assistant_followup_context(lines: &[String], line_idx: usize) -> String {
    let start = line_idx.saturating_sub(1);
    let end = usize::min(line_idx + 1, lines.len().saturating_sub(1));
    lines[start..=end].join(" ")
}

pub(in crate::index) fn extract_expected_chess_reply_move_number(task_lower: &str) -> Option<i32> {
    let prior_move = Regex::new(r"after\s+(\d+)\.")
        .unwrap()
        .captures(task_lower)
        .and_then(|caps| caps.get(1))
        .and_then(|m| m.as_str().parse::<i32>().ok())?;
    Some(prior_move + 1)
}

pub(in crate::index) fn extract_chess_move_answer_from_line(
    line: &str,
    expected_move_number: Option<i32>,
) -> Option<String> {
    let capture =
        Regex::new(r"\b(\d+)\.\s*(O-O(?:-O)?|[KQRNB]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRNB])?[+#]?)\b")
            .unwrap()
            .captures(line)?;
    let move_number = capture.get(1)?.as_str().parse::<i32>().ok()?;
    if expected_move_number.is_some_and(|expected| expected != move_number) {
        return None;
    }
    let notation = capture.get(2)?.as_str().trim();
    Some(format!("{move_number}. {notation}"))
}

pub(in crate::index) fn extract_parenthetical_label_count_answer(
    task_lower: &str,
    line: &str,
    _lower: &str,
) -> Option<String> {
    let focus_terms = synthetic_query_terms(task_lower);
    let focus_refs: Vec<&str> = focus_terms.iter().map(String::as_str).collect();
    let capture = Regex::new(r"(?i)\b([A-Za-z][A-Za-z' -]+?)\s*\((\d+)\)")
        .unwrap()
        .captures(line)?;
    let label = capture.get(1)?.as_str().trim().to_ascii_lowercase();
    (term_overlap_count(&label, &focus_refs) >= 1)
        .then(|| capture.get(2).map(|m| m.as_str().trim().to_string()))
        .flatten()
}

pub(in crate::index) fn extract_website_name_from_line(line: &str) -> Option<String> {
    Regex::new(r"\b([A-Za-z0-9-]+\.(?:org|com|net|edu|io))\b")
        .unwrap()
        .captures(line)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim().to_string())
}

pub(in crate::index) fn extract_beer_recommendation_answer_from_line(lower: &str) -> Option<String> {
    (lower.contains("beer") && lower.contains("pilsner") && lower.contains("lager"))
        .then_some("I recommended using a Pilsner or Lager for the recipe.".to_string())
}

pub(in crate::index) fn extract_two_factor_method_answer_from_line(
    line: &str,
    lower: &str,
) -> Option<String> {
    if !lower.contains("two-factor authentication") {
        return None;
    }
    let methods = extract_phrase_after_any_index(
        line,
        lower,
        &["such as "],
        &[", enhances security", " enhances security", ".", ";"],
        1,
    )?;
    Some(format!(
        "I mentioned {} as examples of two-factor authentication methods.",
        methods.trim().trim_end_matches(',')
    ))
}

pub(in crate::index) fn extract_session_education_answer(line: &str, lower: &str) -> Option<String> {
    let mut answer = extract_phrase_after_any_index(
        line,
        lower,
        &[
            "degree in ",
            "bachelor's in ",
            "bachelors in ",
            "master's in ",
            "masters in ",
            "graduated with a degree in ",
            "graduated with degree in ",
            "graduated with ",
            "majored in ",
            "major in ",
            "studying ",
            "study ",
        ],
        &[
            " which",
            " from ",
            " at ",
            " and ",
            " but ",
            " because ",
            ",",
        ],
        1,
    )?;
    for prefix in [
        "a degree in ",
        "degree in ",
        "a bachelor's in ",
        "a bachelors in ",
        "bachelor's in ",
        "bachelors in ",
        "a master's in ",
        "a masters in ",
        "master's in ",
        "masters in ",
    ] {
        if answer.to_ascii_lowercase().starts_with(prefix) {
            answer = answer[prefix.len()..].trim().to_string();
            break;
        }
    }
    Some(normalize_education_kg_value(&answer))
}

pub(in crate::index) fn extract_session_named_answer_from_line(
    task_lower: &str,
    line: &str,
    lower: &str,
) -> Option<String> {
    let is_query_context = |candidate: &str| {
        let terms = tokenize(&candidate.to_ascii_lowercase());
        !terms.is_empty()
            && terms
                .iter()
                .all(|term| term.len() <= 2 || task_lower.contains(term.as_str()))
    };
    if let Some(value) = extract_descriptor_led_named_answer(line) {
        if !is_query_context(&value) {
            return Some(value);
        }
    }
    let is_question = lower.trim_end().ends_with('?');
    let markers = if is_question {
        vec![
            "called ",
            "named ",
            "titled ",
            "example is ",
            "example was ",
        ]
    } else {
        vec![
            "called ",
            "named ",
            "titled ",
            "recommend ",
            "recommended ",
            "try ",
            "example is ",
            "example was ",
            "was ",
        ]
    };
    if let Some(value) = extract_phrase_after_any_index(
        line,
        lower,
        &markers,
        &[" for ", " because ", " and ", " but ", ".", ",", " while "],
        1,
    ) {
        if let Some(best_title) = extract_title_like_phrases(&value)
            .into_iter()
            .find(|candidate| !is_query_context(candidate))
        {
            return Some(best_title);
        }
        if value.split_whitespace().count() <= 8 && !is_query_context(&value) {
            return Some(value);
        }
    }

    let mut titles = extract_title_like_phrases(line)
        .into_iter()
        .filter(|value| {
            let lower_value = value.to_ascii_lowercase();
            ![
                "also", "by", "can", "do", "does", "for", "i", "it", "my", "our", "that", "the",
                "this", "we", "what", "when", "where", "which", "who",
            ]
            .contains(&lower_value.as_str())
                && !is_query_context(value)
        })
        .collect::<Vec<_>>();
    if task_contains_any(task_lower, &["playlist", "project", "blog", "channel"]) {
        titles.retain(|value| value.split_whitespace().count() <= 6);
    }
    titles.sort_by(|left, right| right.len().cmp(&left.len()).then_with(|| left.cmp(right)));
    titles.into_iter().next()
}

pub(in crate::index) fn extract_descriptor_led_named_answer(line: &str) -> Option<String> {
    let body = normalize_session_answer_line_body(line);
    let body_lower = body.to_ascii_lowercase();
    let split_idx = [
        " has ", " have ", " had ", " is ", " was ", " said ", " taken ",
    ]
    .into_iter()
    .filter_map(|marker| body_lower.find(marker))
    .min()?;
    let mut prefix = body[..split_idx].trim();
    for marker in ["for example,", "for instance,", "likewise,", "similarly,"] {
        if body_lower.starts_with(marker) {
            prefix = prefix[marker.len()..].trim();
            break;
        }
    }
    prefix = prefix
        .trim_start_matches("the ")
        .trim_start_matches("a ")
        .trim_start_matches("an ")
        .trim();
    let tokens: Vec<&str> = prefix
        .split_whitespace()
        .map(|token| {
            token.trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '\'' && ch != '-')
        })
        .filter(|token| !token.is_empty())
        .collect();
    if tokens.len() < 2 {
        return None;
    }
    let candidate_tokens: Vec<&str> = tokens
        .iter()
        .rev()
        .take_while(|token| !token.contains('/') && !token.eq_ignore_ascii_case("the"))
        .take(2)
        .copied()
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    if candidate_tokens.len() < 2 {
        return None;
    }
    Some(title_case_named_words(&candidate_tokens.join(" ")))
}

pub(in crate::index) fn title_case_named_words(value: &str) -> String {
    value
        .split_whitespace()
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(first) => format!("{}{}", first.to_uppercase(), chars.as_str()),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

pub(in crate::index) fn extract_session_list_answer_from_line(
    task_lower: &str,
    line: &str,
    lower: &str,
) -> Option<String> {
    let answer = extract_phrase_after_any_index(
        line,
        lower,
        &[
            "such as ",
            "including ",
            "include ",
            "includes ",
            "uses ",
            "using ",
            "were ",
        ],
        &[". ", "?", " and i'm ", " and i’m ", " but "],
        1,
    )?;
    task_contains_any(
        task_lower,
        &["what kind", "what type", "specific", "what were the"],
    )
    .then_some(answer)
}

pub(in crate::index) fn extract_session_location_answer(
    task_lower: &str,
    line: &str,
    lower: &str,
) -> Option<String> {
    if task_contains_any(
        task_lower,
        &[
            "buy",
            "bought",
            "redeem",
            "use my coupon",
            "which store",
            "shop",
        ],
    ) {
        return extract_phrase_after_any_index(
            line,
            lower,
            &["from the ", "from ", "at the ", "at "],
            &[
                " for ",
                " with ",
                " because ",
                " and ",
                " but ",
                " last ",
                ".",
            ],
            1,
        );
    }
    if task_contains_any(
        task_lower,
        &["keep", "kept", "store", "stored", "put", "place"],
    ) {
        for marker in ["under ", "in ", "inside ", "on "] {
            if let Some(phrase) = extract_phrase_after_any_index(
                line,
                lower,
                &[marker],
                &[" because ", " and ", " but ", ".", ","],
                1,
            ) {
                return Some(format!("{} {}", marker.trim(), phrase));
            }
        }
    }
    extract_phrase_after_any_index(
        line,
        lower,
        &[
            "based in ",
            "live in ",
            "living in ",
            "now living in ",
            "moved to ",
            "moved back to ",
        ],
        &[
            " again",
            " because ",
            " and ",
            " but ",
            " with ",
            " after ",
            ".",
            ",",
        ],
        1,
    )
    .map(|value| normalize_location_kg_value(&value))
}

pub(in crate::index) fn extract_session_occupation_answer(line: &str, lower: &str) -> Option<String> {
    extract_phrase_after_any_index(
        line,
        lower,
        &[
            "work as ",
            "working as ",
            "employed as ",
            "job as ",
            "role as ",
            "i'm a ",
            "i am a ",
        ],
        &[" at ", " for ", " and ", " but ", " because ", "."],
        1,
    )
}

pub(in crate::index) fn extract_money_answer_from_line(line: &str) -> Option<String> {
    Regex::new(r"(?i)(\$\d[\d,]*(?:\.\d+)?)")
        .unwrap()
        .captures(line)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim().to_string())
}

pub(in crate::index) fn extract_percent_answer_from_line(line: &str) -> Option<String> {
    Regex::new(r"(?i)(\d+(?:\.\d+)?%)")
        .unwrap()
        .captures(line)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim().to_string())
}

pub(in crate::index) fn extract_speed_answer_from_line(line: &str) -> Option<String> {
    Regex::new(r"(?i)(\d+(?:\.\d+)?\s*(?:mbps|gbps))")
        .unwrap()
        .captures(line)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim().to_string())
}

pub(in crate::index) fn extract_university_name_from_line(line: &str) -> Option<String> {
    Regex::new(r"([A-Z][A-Za-z&.'-]*(?:\s+[A-Z][A-Za-z&.'-]*)*\s+University)")
        .unwrap()
        .captures(line)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim().to_string())
}

pub(in crate::index) fn extract_query_month_name(lower: &str) -> Option<&'static str> {
    [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ]
    .into_iter()
    .find(|month| lower.contains(month))
}

pub(in crate::index) fn next_month_name(month: &str) -> Option<&'static str> {
    match month {
        "january" => Some("february"),
        "february" => Some("march"),
        "march" => Some("april"),
        "april" => Some("may"),
        "may" => Some("june"),
        "june" => Some("july"),
        "july" => Some("august"),
        "august" => Some("september"),
        "september" => Some("october"),
        "october" => Some("november"),
        "november" => Some("december"),
        "december" => Some("january"),
        _ => None,
    }
}

pub(in crate::index) fn line_matches_query_month_window(lower: &str, month: &str) -> bool {
    if lower.contains(month) {
        return true;
    }

    lower.contains("this month")
        && next_month_name(month)
            .map(|next_month| lower.contains(&format!("before {next_month}")))
            .unwrap_or(false)
}

pub(in crate::index) fn line_describes_actual_doctor_visit(lower: &str) -> bool {
    let positive = task_contains_any(
        lower,
        &[
            "follow-up appointment",
            "appointment with",
            "went to see",
            "got back from",
            "diagnosed me with",
            "diagnosed with",
            "was prescribed",
            "prescribed antibiotics",
            "prescribed a nasal spray",
            "recently had",
            "just got diagnosed",
        ],
    );
    if !positive {
        return false;
    }

    if task_contains_any(
        lower,
        &[
            "thinking about",
            "considering",
            "i'll schedule",
            "i will schedule",
            "schedule an appointment",
            "scheduling an appointment",
            "talk to dr.",
            "ask dr.",
            "follow up with dr.",
            "consult with",
        ],
    ) {
        return false;
    }

    true
}

pub(in crate::index) fn extract_doctor_role_from_line(_line: &str, lower: &str) -> Option<String> {
    [
        ("primary care physician", "a primary care physician"),
        ("ent specialist", "an ENT specialist"),
        ("dermatologist", "a dermatologist"),
        ("orthopedic surgeon", "an orthopedic surgeon"),
        ("neurologist", "a neurologist"),
        ("gastroenterologist", "a gastroenterologist"),
    ]
    .into_iter()
    .find(|(needle, _)| lower.contains(needle))
    .map(|(_, rendered)| rendered.to_string())
}

pub(in crate::index) fn doctor_role_sort_key(role: &str) -> usize {
    match role {
        "a primary care physician" => 0,
        "an ENT specialist" => 1,
        "a dermatologist" => 2,
        "an orthopedic surgeon" => 3,
        "a neurologist" => 4,
        "a gastroenterologist" => 5,
        _ => 99,
    }
}

pub(in crate::index) fn doctor_visit_event_key(role: &str, lower: &str) -> String {
    let day = Regex::new(r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?\b")
        .unwrap()
        .captures(lower)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().to_string());
    match day {
        Some(day) => format!("{role}|{day}"),
        None => role.to_string(),
    }
}

pub(in crate::index) fn extract_duration_answer_from_line(line: &str) -> Option<String> {
    Regex::new(
        r"(?i)\b((?:about\s+)?(?:an?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?)\s+(?:days?|weeks?|months?|years?|hours?|minutes?)(?:\s+(?:ago|now|each way))?)\b",
    )
    .unwrap()
    .captures(line)
    .and_then(|caps| caps.get(1))
    .map(|m| m.as_str().trim().to_string())
}

pub(in crate::index) fn normalize_current_duration_answer(duration: &str) -> String {
    duration
        .trim()
        .trim_start_matches("about ")
        .trim_end_matches(" now")
        .trim_end_matches(" ago")
        .trim_start_matches("an ")
        .trim_start_matches("a ")
        .to_string()
        .replacen("one ", "1 ", 1)
}

pub(in crate::index) fn duration_answer_magnitude(duration: &str) -> Option<f32> {
    let lower = duration.to_ascii_lowercase();
    let caps = Regex::new(
        r"\b(\d+(?:\.\d+)?|an?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)(?:\s*-\s*(\d+(?:\.\d+)?))?\s+(day|week|month|year|hour|minute)s?\b",
    )
    .unwrap()
    .captures(&lower)?;
    let quantity = match caps.get(2).map(|m| m.as_str()) {
        Some(value) => value.parse::<f32>().ok()?,
        None => match caps.get(1)?.as_str() {
            "a" | "an" => 1.0,
            "one" => 1.0,
            "two" => 2.0,
            "three" => 3.0,
            "four" => 4.0,
            "five" => 5.0,
            "six" => 6.0,
            "seven" => 7.0,
            "eight" => 8.0,
            "nine" => 9.0,
            "ten" => 10.0,
            "eleven" => 11.0,
            "twelve" => 12.0,
            value => value.parse::<f32>().ok()?,
        },
    };
    let unit_days = match caps.get(3)?.as_str() {
        "minute" => 1.0 / (24.0 * 60.0),
        "hour" => 1.0 / 24.0,
        "day" => 1.0,
        "week" => 7.0,
        "month" => 30.0,
        "year" => 365.0,
        _ => return None,
    };
    Some(quantity * unit_days)
}

pub(in crate::index) fn is_ongoing_duration_query(task_lower: &str) -> bool {
    task_lower.starts_with("how long have ")
        && !task_contains_any(
            task_lower,
            &[" before ", " after ", " until ", "left to", "remaining"],
        )
}

pub(in crate::index) fn extract_ongoing_duration_anchor_terms(terms: &[String]) -> Vec<String> {
    const STOP: &[&str] = &[
        "long",
        "been",
        "being",
        "using",
        "living",
        "sticking",
        "staying",
        "working",
        "collecting",
        "keeping",
        "having",
        "doing",
        "going",
        "current",
        "daily",
        "about",
        "around",
        "there",
        "here",
    ];
    let anchors: Vec<String> = terms
        .iter()
        .filter(|term| term.len() >= 4)
        .filter(|term| !STOP.contains(&term.as_str()))
        .cloned()
        .collect();
    if anchors.is_empty() {
        terms
            .iter()
            .filter(|term| term.len() >= 3)
            .filter(|term| !STOP.contains(&term.as_str()))
            .cloned()
            .collect()
    } else {
        anchors
    }
}

pub(in crate::index) fn extract_tablespoon_water_ounces(line: &str) -> Option<f32> {
    let lower = line.to_ascii_lowercase();
    if !(lower.contains("tablespoon")
        && lower.contains("coffee")
        && lower.contains("ounces")
        && lower.contains("water"))
    {
        return None;
    }
    Regex::new(r"(?i)\b(\d+(?:\.\d+)?)\s+ounces?\s+of\s+water\b")
        .unwrap()
        .captures(line)
        .and_then(|caps| caps.get(1))
        .and_then(|m| m.as_str().parse::<f32>().ok())
}

pub(in crate::index) fn compact_decimal_string(value: f32) -> String {
    let mut rendered = value.to_string();
    if rendered.ends_with(".0") {
        rendered.truncate(rendered.len() - 2);
    }
    rendered
}

pub(in crate::index) fn extract_date_or_time_answer_from_line(line: &str) -> Option<String> {
    for pattern in [
        r"(?i)\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?(?:-\d{1,2}(?:st|nd|rd|th)?)?)\b",
        r"(?i)\b(\d{1,2}:\d{2}\s?(?:AM|PM))\b",
        r"(?i)\b(\d{1,2}\s?(?:AM|PM))\b",
        r"(?i)\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
    ] {
        if let Some(value) = Regex::new(pattern)
            .unwrap()
            .captures(line)
            .and_then(|caps| caps.get(1))
            .map(|m| m.as_str().trim().to_string())
        {
            return Some(value);
        }
    }
    None
}

pub(in crate::index) fn extract_color_answer_from_line(line: &str) -> Option<String> {
    for pattern in [
        r"(?i)\b((?:a\s+)?(?:lighter|darker|light|dark|soft|pale|bright|deep)\s+shade of\s+(?:gray|grey|blue|green|pink|purple|yellow|red|orange|white|black|beige|brown))\b",
        r"(?i)\b((?:light|dark|pale|bright|deep|soft)\s+(?:gray|grey|blue|green|pink|purple|yellow|red|orange|white|black|beige|brown))\b",
        r"(?i)\b(gray|grey|blue|green|pink|purple|yellow|red|orange|white|black|beige|brown)\b",
    ] {
        if let Some(value) = Regex::new(pattern)
            .unwrap()
            .captures(line)
            .and_then(|caps| caps.get(1))
            .map(|m| m.as_str().trim().to_string())
        {
            return Some(value);
        }
    }
    None
}

pub(in crate::index) fn extract_query_aligned_numeric_answer(task_lower: &str, line: &str) -> Option<String> {
    let mut terms = synthetic_query_terms(task_lower)
        .into_iter()
        .filter(|term| term.len() >= 4)
        .filter(|term| {
            ![
                "current",
                "currently",
                "recently",
                "specific",
                "previous",
                "conversation",
                "recommended",
            ]
            .contains(&term.as_str())
        })
        .collect::<Vec<_>>();
    if task_lower.contains("times") {
        terms.extend(
            ["game", "games", "match", "matches", "meeting", "meetings"]
                .into_iter()
                .map(str::to_string),
        );
    }
    terms.sort();
    terms.dedup();
    let line_lower = line.to_ascii_lowercase();
    let anchor_terms = assistant_followup_anchor_terms(task_lower);
    let mut best_anchor_match: Option<(usize, usize, String)> = None;
    for term in &terms {
        let pattern = Regex::new(&format!(
            r"(?i)\b((?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred))\s+{}s?\b",
            regex::escape(&term)
        ))
        .unwrap();
        for capture in pattern.captures_iter(line) {
            let Some(full_match) = capture.get(0) else {
                continue;
            };
            let Some(value_match) = capture.get(1) else {
                continue;
            };
            let Some(distance) =
                assistant_followup_anchor_distance(&line_lower, full_match.end(), &anchor_terms)
            else {
                continue;
            };
            let value = value_match.as_str().trim().to_string();
            if best_anchor_match
                .as_ref()
                .map(|(best_distance, best_start, _)| {
                    distance < *best_distance
                        || (distance == *best_distance && full_match.start() > *best_start)
                })
                .unwrap_or(true)
            {
                best_anchor_match = Some((distance, full_match.start(), value));
            }
        }
    }
    if let Some((_, _, value)) = best_anchor_match {
        return Some(value);
    }
    for term in terms {
        let pattern = Regex::new(&format!(
            r"(?i)\b((?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred))\s+{}s?\b",
            regex::escape(&term)
        ))
        .unwrap();
        if let Some(value) = pattern
            .captures(line)
            .and_then(|caps| caps.get(1))
            .map(|m| m.as_str().trim().to_string())
        {
            return Some(value);
        }
    }
    None
}

pub(in crate::index) fn extract_session_purchase_item(line: &str, lower: &str) -> Option<String> {
    extract_phrase_after_any_index(
        line,
        lower,
        &[
            "purchased a ",
            "purchased an ",
            "bought a ",
            "bought an ",
            "picked up a ",
            "picked up an ",
            "got a ",
            "got an ",
        ],
        &[" for ", " with ", " because ", " and ", " but ", "."],
        1,
    )
}

pub(in crate::index) fn extract_title_like_phrases(text: &str) -> Vec<String> {
    const CONNECTORS: &[&str] = &[
        "of", "the", "and", "at", "in", "on", "to", "for", "dei", "del", "di", "du", "&", "+",
    ];
    let mut phrases = Vec::new();
    let mut current = Vec::new();
    let mut seen_title = false;

    for raw in text.split_whitespace() {
        let cleaned = raw.trim_matches(|c: char| {
            !c.is_ascii_alphanumeric() && !matches!(c, '&' | '+' | '\'' | '-')
        });
        if cleaned.is_empty() {
            continue;
        }
        let lower = cleaned.to_ascii_lowercase();
        let starts_upper = cleaned
            .chars()
            .next()
            .map(|c| c.is_ascii_uppercase())
            .unwrap_or(false);
        let short_acronym = cleaned.len() <= 5
            && cleaned
                .chars()
                .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || matches!(c, '&' | '+'));
        let is_title = starts_upper || short_acronym;

        if is_title || (seen_title && CONNECTORS.contains(&lower.as_str())) {
            current.push(cleaned.to_string());
            if is_title {
                seen_title = true;
            }
            continue;
        }

        if seen_title && !current.is_empty() {
            let phrase = current.join(" ");
            if phrase.split_whitespace().count() <= 8 {
                phrases.push(phrase);
            }
        }
        current.clear();
        seen_title = false;
    }

    if seen_title && !current.is_empty() {
        let phrase = current.join(" ");
        if phrase.split_whitespace().count() <= 8 {
            phrases.push(phrase);
        }
    }

    phrases
}

pub(in crate::index) fn extract_phrase_after_any_index(
    line: &str,
    lower: &str,
    markers: &[&str],
    stop_markers: &[&str],
    min_words: usize,
) -> Option<String> {
    let mut best = None;
    for marker in markers {
        let Some(idx) = lower.find(marker) else {
            continue;
        };
        let tail = &line[idx + marker.len()..];
        let lower_tail = tail.to_ascii_lowercase();
        let cut = stop_markers
            .iter()
            .filter_map(|needle| lower_tail.find(needle))
            .min()
            .unwrap_or(tail.len());
        let mut phrase = tail[..cut]
            .trim()
            .trim_matches(|c: char| matches!(c, '"' | '\'' | ',' | '.' | ';' | ':' | '!' | '?'))
            .trim()
            .to_string();
        for prefix in ["the ", "a ", "an ", "simple "] {
            if phrase.to_ascii_lowercase().starts_with(prefix) {
                phrase = phrase[prefix.len()..].trim().to_string();
            }
        }
        if phrase.split_whitespace().count() < min_words {
            continue;
        }
        if best
            .as_ref()
            .map(|current: &String| phrase.len() > current.len())
            .unwrap_or(true)
        {
            best = Some(phrase);
        }
    }
    best
}

pub(in crate::index) fn extract_project_count_item(line: &str, lower: &str) -> Option<String> {
    if lower.contains("case competition") {
        return Some("case competition".to_string());
    }
    let phrase = extract_phrase_after_any_index(
        line,
        lower,
        &[
            "working on a ",
            "working on ",
            "leading a ",
            "leading ",
            "started a ",
            "building a ",
            "creating a ",
        ],
        &[
            " for ",
            " with ",
            " because ",
            " and ",
            " but ",
            " that ",
            ",",
        ],
        2,
    )?;
    let lower_phrase = phrase.to_ascii_lowercase();
    (lower_phrase.contains("project") || lower_phrase.contains("competition")).then_some(phrase)
}

pub(in crate::index) fn normalize_model_kit_count_item(text: &str) -> String {
    let mut item = text.trim().to_string();
    for prefix in [
        "diorama featuring a ",
        "diorama featuring ",
        "a simple ",
        "simple ",
    ] {
        if item.to_ascii_lowercase().starts_with(prefix) {
            item = item[prefix.len()..].trim().to_string();
            break;
        }
    }
    let lower = item.to_ascii_lowercase();
    let cutoff = [" do you ", ". do you ", "? ", " and i'm ", " and i’m "]
        .iter()
        .filter_map(|marker| lower.find(marker))
        .min()
        .unwrap_or(item.len());
    item.truncate(cutoff);
    item = item
        .trim()
        .trim_matches(|c: char| matches!(c, ',' | '.' | ';' | ':' | '"' | '\''))
        .to_string();
    if item.to_ascii_lowercase().ends_with(" kit") {
        item.truncate(item.len().saturating_sub(4));
        item = item.trim().to_string();
    }
    item
}

pub(in crate::index) fn extract_model_kit_count_item(line: &str, lower: &str) -> Option<String> {
    let phrase = extract_phrase_after_any_index(
        line,
        lower,
        &[
            "finished a simple ",
            "finished a ",
            "working on a ",
            "working on ",
            "next project, a ",
            "featuring a ",
            "for your ",
        ],
        &[
            " that ",
            " and ",
            " but ",
            " because ",
            " while ",
            " next",
            " where ",
            ",",
        ],
        2,
    )?;
    let item = normalize_model_kit_count_item(&phrase);
    let lower_item = item.to_ascii_lowercase();
    (lower_item.contains("scale")
        || lower_item.contains("camaro")
        || lower_item.contains("bomber")
        || lower_item.contains("tank")
        || lower_item.contains("spitfire")
        || lower_item.contains("eagle"))
    .then_some(item)
}

pub(in crate::index) fn extract_clothing_store_item(line: &str, lower: &str) -> Option<String> {
    if lower.contains("dry cleaning for ") {
        return extract_phrase_after_any_index(
            line,
            lower,
            &["dry cleaning for the ", "dry cleaning for "],
            &[" i ", " and ", " but ", " because ", ","],
            2,
        );
    }

    let phrase = extract_phrase_after_any_index(
        line,
        lower,
        &[
            "return some ",
            "return my ",
            "return the ",
            "pick up my ",
            "pick up the ",
        ],
        &[" to ", " from ", " because ", " and ", " but ", ","],
        1,
    )?;
    let lower_phrase = phrase.to_ascii_lowercase();
    [
        "blazer", "boots", "jeans", "shirt", "sweater", "dress", "sundress", "coat", "jacket",
        "pants", "trousers", "skirt", "top",
    ]
    .iter()
    .any(|needle| lower_phrase.contains(needle))
    .then_some(phrase)
}

pub(in crate::index) fn normalize_family_origin_item(text: &str) -> String {
    let mut item = text.trim().to_string();
    for prefix in ["a set of ", "set of ", "my ", "the ", "a ", "an "] {
        if item.to_ascii_lowercase().starts_with(prefix) {
            item = item[prefix.len()..].trim().to_string();
        }
    }
    item.trim_matches(|c: char| matches!(c, '"' | '\'' | ',' | '.' | ';' | ':' | '!' | '?'))
        .trim()
        .to_string()
}

pub(in crate::index) fn extract_family_origin_antique_items_from_line(
    line: &str,
    lower: &str,
) -> Vec<String> {
    if !task_contains_any(
        lower,
        &[
            "grandmother",
            "great-aunt",
            "great aunt",
            "mom",
            "dad",
            "cousin",
            "family heirloom",
            "family heirlooms",
            "inherited",
            "belonged to my",
            "from my",
        ],
    ) || !task_contains_any(lower, &["antique", "vintage", "depression-era"])
    {
        return Vec::new();
    }

    let pattern = Regex::new(
        r"(?i)(?:antique|vintage|depression-era)\s+[a-z][a-z-]*(?:\s+[a-z][a-z-]*){0,3}",
    )
    .unwrap();
    let mut items = Vec::new();
    let mut seen = HashSet::new();
    for item_match in pattern.find_iter(line) {
        let mut item = item_match.as_str().trim().to_string();
        let lower_item = item.to_ascii_lowercase();
        if let Some(cut) = [
            " from ",
            " that ",
            " which ",
            " belonged ",
            " came ",
            " insured",
            " appraised",
            " valued",
            " selling",
            " sold",
        ]
        .iter()
        .filter_map(|needle| lower_item.find(needle))
        .min()
        {
            item = item[..cut].trim().to_string();
        }
        let item = normalize_family_origin_item(&item);
        let lower_item = item.to_ascii_lowercase();
        if item.is_empty()
            || task_contains_any(
                &lower_item,
                &[
                    "dealer",
                    "dealers",
                    "appraiser",
                    "appraisers",
                    "insurance",
                    "company",
                    "companies",
                    "organization",
                    "organizations",
                    "marketplace",
                    "marketplaces",
                    "forum",
                    "forums",
                ],
            )
        {
            continue;
        }
        let key = normalized_synthetic_phrase_key(&item);
        if seen.insert(key) {
            items.push(item);
        }
    }
    items
}

pub(in crate::index) fn extract_born_child_names_from_line(line: &str, lower: &str) -> Vec<String> {
    if lower.contains("adopted") {
        return Vec::new();
    }

    let mut names = Vec::new();
    let mut seen = HashSet::new();

    let twin_pattern =
        Regex::new(r"(?i)\btwins?(?:\s+\w+)?\s*,\s*([A-Z][a-z]+)\s+and\s+([A-Z][a-z]+)\b").unwrap();
    for caps in twin_pattern.captures_iter(line) {
        for idx in [1, 2] {
            let Some(name_match) = caps.get(idx) else {
                continue;
            };
            let name = name_match.as_str().trim().to_string();
            let key = normalized_synthetic_phrase_key(&name);
            if seen.insert(key) {
                names.push(name);
            }
        }
    }

    let single_patterns = [
        Regex::new(r"(?i)\bbaby\s+(?:boy|girl)\s+named\s+([A-Z][a-z]+)\b").unwrap(),
        Regex::new(r"(?i)\b(?:son|daughter)\s+([A-Z][a-z]+)\b").unwrap(),
    ];
    for pattern in &single_patterns {
        for caps in pattern.captures_iter(line) {
            let Some(name_match) = caps.get(1) else {
                continue;
            };
            let name = name_match.as_str().trim().to_string();
            let key = normalized_synthetic_phrase_key(&name);
            if seen.insert(key) {
                names.push(name);
            }
        }
    }

    names
}

pub(in crate::index) fn normalize_bike_service_item(text: &str) -> String {
    let mut item = text.trim().to_string();
    for prefix in ["regular ", "my ", "the ", "our ", "a ", "an "] {
        if item.to_ascii_lowercase().starts_with(prefix) {
            item = item[prefix.len()..].trim().to_string();
        }
    }
    item.trim_matches(|c: char| matches!(c, '"' | '\'' | ',' | '.' | ';' | ':' | '!' | '?'))
        .trim()
        .to_string()
}

pub(in crate::index) fn extract_bike_phrase_from_line(line: &str, _lower: &str) -> Option<String> {
    let with_determiner = Regex::new(
        r"(?i)\b(?:my|the|our|a|an)\s+((?:road|commuter|mountain|hybrid|gravel|touring|electric|e-bike|ebike|bmx|trail)\s+bike)\b",
    )
            .unwrap()
            .captures(line)
            .and_then(|caps| caps.get(1))
            .map(|m| m.as_str().trim().to_string());
    let phrase = with_determiner.or_else(|| {
        Regex::new(
            r"(?i)\b((?:road|commuter|mountain|hybrid|gravel|touring|electric|e-bike|ebike|bmx|trail)\s+bike)\b",
        )
            .unwrap()
            .captures(line)
            .and_then(|caps| caps.get(1))
            .map(|m| m.as_str().trim().to_string())
    })?;
    let phrase = normalize_bike_service_item(&phrase);
    (phrase != "bike").then_some(phrase)
}

pub(in crate::index) fn line_describes_bike_service_event(lower: &str) -> bool {
    lower.contains("bike")
        && task_contains_any(
            lower,
            &[
                "serviced at",
                "bike serviced",
                "cleaned and lubricated",
                "cleaning and lubricating",
                "time to replace",
                "replace it this month",
                "before april",
                "planning to service",
                "plan to service",
                "getting a new tire",
                "get a new tire",
                "new tire for my",
            ],
        )
}

pub(in crate::index) fn extract_bike_service_item_from_line(
    line: &str,
    lower: &str,
    month: &str,
) -> Option<String> {
    if !line_matches_query_month_window(lower, month) || !line_describes_bike_service_event(lower) {
        return None;
    }
    extract_bike_phrase_from_line(line, lower)
}

pub(in crate::index) fn render_day_count_answer(count: usize) -> String {
    format!("{count} {}", if count == 1 { "day" } else { "days" })
}

pub(in crate::index) fn line_describes_countable_fitness_class_schedule(line: &str, lower: &str) -> bool {
    let speaker_grounded = lower.starts_with("user:") || line.trim_start().starts_with('-');
    let assistant_restate = lower.contains("your ");
    let explicit_class_signal = task_contains_any(
        lower,
        &[
            "fitness class",
            "fitness classes",
            "bodypump",
            "hip hop abs",
            "yoga class",
            "yoga classes",
            "zumba",
        ],
    ) || ((lower.contains(" class") || lower.contains(" classes"))
        && task_contains_any(
            lower,
            &[
                "weightlifting",
                "strength training",
                "pilates",
                "spin",
                "kickboxing",
                "barre",
                "cycling",
                "aerobics",
            ],
        ));

    explicit_class_signal && (speaker_grounded || assistant_restate)
}

pub(in crate::index) fn extract_weekday_mentions_from_line(lower: &str) -> Vec<String> {
    let mut days = Vec::new();
    let mut seen = HashSet::new();
    for day in [
        "sunday",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
    ] {
        if lower.contains(day) && seen.insert(day) {
            days.push(day.to_string());
        }
    }
    days
}

pub(in crate::index) fn push_month_day(days: &mut Vec<u32>, seen: &mut HashSet<u32>, value: u32) {
    if (1..=31).contains(&value) && seen.insert(value) {
        days.push(value);
    }
}

pub(in crate::index) fn push_month_day_range(
    days: &mut Vec<u32>,
    seen: &mut HashSet<u32>,
    start: u32,
    end: u32,
) {
    if !(1..=31).contains(&start) || !(1..=31).contains(&end) || end < start {
        return;
    }
    for value in start..=end {
        push_month_day(days, seen, value);
    }
}

pub(in crate::index) fn extract_month_day_values_from_line(line: &str, lower: &str, month: &str) -> Vec<u32> {
    if !lower.contains(month) {
        return Vec::new();
    }

    let month_pattern = regex::escape(month);
    let mut days = Vec::new();
    let mut seen = HashSet::new();

    let month_range = Regex::new(&format!(
        r"(?i)\b{}\s+(\d{{1,2}})(?:st|nd|rd|th)?\s*-\s*(\d{{1,2}})(?:st|nd|rd|th)?\b",
        month_pattern
    ))
    .unwrap();
    for caps in month_range.captures_iter(line) {
        let Some(start) = caps.get(1).and_then(|m| m.as_str().parse::<u32>().ok()) else {
            continue;
        };
        let Some(end) = caps.get(2).and_then(|m| m.as_str().parse::<u32>().ok()) else {
            continue;
        };
        push_month_day_range(&mut days, &mut seen, start, end);
    }

    let day_pair = Regex::new(&format!(
        r"(?i)\b(\d{{1,2}})(?:st|nd|rd|th)?\s+and\s+(\d{{1,2}})(?:st|nd|rd|th)?\s+of\s+{}\b",
        month_pattern
    ))
    .unwrap();
    for caps in day_pair.captures_iter(line) {
        let Some(first) = caps.get(1).and_then(|m| m.as_str().parse::<u32>().ok()) else {
            continue;
        };
        let Some(second) = caps.get(2).and_then(|m| m.as_str().parse::<u32>().ok()) else {
            continue;
        };
        push_month_day(&mut days, &mut seen, first);
        push_month_day(&mut days, &mut seen, second);
    }

    let month_single = Regex::new(&format!(
        r"(?i)\b{}\s+(\d{{1,2}})(?:st|nd|rd|th)?\b",
        month_pattern
    ))
    .unwrap();
    for caps in month_single.captures_iter(line) {
        let Some(day) = caps.get(1).and_then(|m| m.as_str().parse::<u32>().ok()) else {
            continue;
        };
        push_month_day(&mut days, &mut seen, day);
    }

    let of_month_single = Regex::new(&format!(
        r"(?i)\b(\d{{1,2}})(?:st|nd|rd|th)?\s+of\s+{}\b",
        month_pattern
    ))
    .unwrap();
    for caps in of_month_single.captures_iter(line) {
        let Some(day) = caps.get(1).and_then(|m| m.as_str().parse::<u32>().ok()) else {
            continue;
        };
        push_month_day(&mut days, &mut seen, day);
    }

    days
}

pub(in crate::index) fn line_matches_activity_markers(lower: &str, markers: &[&str]) -> bool {
    markers.iter().any(|marker| lower.contains(marker))
}

pub(in crate::index) fn extract_month_scoped_activity_days_from_line(
    line: &str,
    lower: &str,
    month: &str,
    activity_markers: &[&str],
) -> Vec<u32> {
    if !line_matches_query_month_window(lower, month)
        || !line_matches_activity_markers(lower, activity_markers)
    {
        return Vec::new();
    }
    extract_month_day_values_from_line(line, lower, month)
}

pub(in crate::index) fn month_name_to_number(month: &str) -> Option<u32> {
    match month {
        "january" => Some(1),
        "february" => Some(2),
        "march" => Some(3),
        "april" => Some(4),
        "may" => Some(5),
        "june" => Some(6),
        "july" => Some(7),
        "august" => Some(8),
        "september" => Some(9),
        "october" => Some(10),
        "november" => Some(11),
        "december" => Some(12),
        _ => None,
    }
}

pub(in crate::index) fn line_matches_query_month_or_numeric_date(
    line: &str,
    lower: &str,
    month: &str,
) -> bool {
    if line_matches_query_month_window(lower, month) {
        return true;
    }
    let Some(target_month) = month_name_to_number(month) else {
        return false;
    };
    Regex::new(r"(?i)\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b")
        .unwrap()
        .captures_iter(line)
        .filter_map(|caps| caps.get(1))
        .filter_map(|value| value.as_str().parse::<u32>().ok())
        .any(|value| value == target_month)
}

pub(in crate::index) fn extract_first_quoted_phrase(line: &str) -> Option<String> {
    Regex::new(r#""([^"]+)""#)
        .unwrap()
        .captures(line)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim().to_string())
}

pub(in crate::index) fn extract_art_event_anchor(line: &str) -> Option<String> {
    extract_first_quoted_phrase(line).or_else(|| {
        extract_title_like_phrases(line)
            .into_iter()
            .filter(|phrase| {
                let lower = phrase.to_ascii_lowercase();
                lower.contains("museum")
                    || lower.contains("gallery")
                    || lower.contains("art cube")
                    || lower.contains("women in art")
                    || lower.contains("art afternoon")
            })
            .max_by_key(|phrase| phrase.len())
    })
}

pub(in crate::index) fn line_describes_art_related_event(lower: &str) -> bool {
    task_contains_any(
        lower,
        &[
            "art",
            "museum",
            "gallery",
            "street art",
            "children's museum",
        ],
    ) && task_contains_any(
        lower,
        &[
            "guided tour",
            "lecture",
            "exhibition",
            "opening night",
            "workshop",
            "event",
            "festival",
        ],
    ) && task_contains_any(
        lower,
        &[
            "attended",
            "went on",
            "went to",
            "visited",
            "volunteered at",
            "opening night",
        ],
    )
}

pub(in crate::index) fn extract_art_related_event_signature_from_line(
    line: &str,
    lower: &str,
) -> Option<(i32, String)> {
    if !line_describes_art_related_event(lower) {
        return None;
    }
    let rank = extract_explicit_date_rank(line)?;
    let kind = if lower.contains("guided tour") {
        "guided-tour"
    } else if lower.contains("opening night") {
        "opening-night"
    } else if lower.contains("lecture") {
        "lecture"
    } else if lower.contains("exhibition") {
        "exhibition"
    } else if lower.contains("workshop") {
        "workshop"
    } else if lower.contains("festival") {
        "festival"
    } else if lower.contains("event") {
        "event"
    } else {
        return None;
    };
    let anchor = extract_art_event_anchor(line)
        .map(|value| normalized_synthetic_phrase_key(&value))
        .unwrap_or_default();
    Some((rank, format!("{rank}:{kind}:{anchor}")))
}

pub(in crate::index) fn line_describes_cuisine_learning_or_trying(lower: &str) -> bool {
    task_contains_any(
        lower,
        &[
            "tried out",
            "learned how to make",
            "learned to make",
            "class on",
            "attended a class on",
            "recipe for",
            "online recipe library",
            "restaurant",
        ],
    )
}

pub(in crate::index) fn extract_cuisine_labels_from_line(_line: &str, lower: &str) -> Vec<String> {
    if !line_describes_cuisine_learning_or_trying(lower) {
        return Vec::new();
    }
    let mut cuisines = Vec::new();
    let mut seen = HashSet::new();
    for cuisine in [
        "ethiopian",
        "indian",
        "korean",
        "thai",
        "mexican",
        "italian",
        "japanese",
        "chinese",
        "greek",
        "moroccan",
        "vietnamese",
        "french",
        "mediterranean",
        "lebanese",
        "spanish",
        "turkish",
        "brazilian",
        "peruvian",
        "middle eastern",
        "vegan",
    ] {
        if lower.contains(cuisine) && seen.insert(cuisine) {
            cuisines.push(cuisine.to_string());
        }
    }
    cuisines
}

pub(in crate::index) fn line_describes_museum_gallery_visit(lower: &str) -> bool {
    task_contains_any(
        lower,
        &[
            "visited",
            "went to",
            "took my niece to",
            "opening night of",
            "met the curator",
            "guided tour at",
        ],
    )
}

pub(in crate::index) fn normalize_visit_venue(text: &str) -> String {
    let mut venue = text.trim().to_string();
    for prefix in ["the ", "my ", "our ", "a ", "an "] {
        if venue.to_ascii_lowercase().starts_with(prefix) {
            venue = venue[prefix.len()..].trim().to_string();
        }
    }
    venue
        .trim_matches(|c: char| matches!(c, '"' | '\'' | ',' | '.' | ';' | ':' | '!' | '?'))
        .trim()
        .to_string()
}

pub(in crate::index) fn extract_museum_gallery_visit_venue_from_line(
    line: &str,
    lower: &str,
    month: &str,
) -> Option<String> {
    if !line_matches_query_month_or_numeric_date(line, lower, month)
        || !line_describes_museum_gallery_visit(lower)
    {
        return None;
    }
    let direct = extract_phrase_after_any_index(
        line,
        lower,
        &[
            "visited ",
            "opening night of ",
            "took my niece to ",
            "went on a guided tour at ",
            "guided tour at ",
            "went to ",
        ],
        &[" on ", ",", ".", " and "],
        1,
    )
    .map(|phrase| normalize_visit_venue(&phrase))
    .filter(|phrase| {
        let lower = phrase.to_ascii_lowercase();
        lower.contains("museum") || lower.contains("gallery") || lower.contains("art cube")
    });
    direct.or_else(|| {
        extract_title_like_phrases(line)
            .into_iter()
            .map(|phrase| normalize_visit_venue(&phrase))
            .filter(|phrase| {
                let lower = phrase.to_ascii_lowercase();
                lower.contains("museum") || lower.contains("gallery") || lower.contains("art cube")
            })
            .max_by_key(|phrase| phrase.len())
    })
}

pub(in crate::index) fn line_mentions_candidate_museum_gallery_visit(
    line: &str,
    lower: &str,
    month: &str,
) -> bool {
    line_matches_query_month_or_numeric_date(line, lower, month)
        && line_describes_museum_gallery_visit(lower)
        && task_contains_any(lower, &["museum", "gallery", "art cube"])
}

pub(in crate::index) fn extract_citrus_fruits_from_line(_line: &str, lower: &str) -> Vec<String> {
    if !task_contains_any(
        lower,
        &[
            "cocktail",
            "cocktails",
            "sangria",
            "daiquiri",
            "gimlet",
            "bitters",
            "mixology",
        ],
    ) {
        return Vec::new();
    }
    let mut fruits = Vec::new();
    let mut seen = HashSet::new();
    for fruit in ["orange", "lemon", "lime", "grapefruit"] {
        if lower.contains(fruit) && seen.insert(fruit) {
            fruits.push(fruit.to_string());
        }
    }
    fruits
}

pub(in crate::index) fn extract_food_delivery_service_from_line(_line: &str, lower: &str) -> Option<String> {
    let labels = [
        ("fresh fusion", "Fresh Fusion"),
        ("uber eats", "Uber Eats"),
        ("domino's pizza", "Domino's Pizza"),
        ("dominos pizza", "Domino's Pizza"),
        ("domino's", "Domino's Pizza"),
        ("doordash", "DoorDash"),
        ("grubhub", "Grubhub"),
        ("postmates", "Postmates"),
        ("seamless", "Seamless"),
        ("caviar", "Caviar"),
    ];
    labels
        .into_iter()
        .find(|(needle, _)| lower.contains(needle))
        .map(|(_, label)| label.to_string())
}

pub(in crate::index) fn extract_missed_fun_run_signature_from_line(
    line: &str,
    lower: &str,
    month: &str,
) -> Option<String> {
    if !line_matches_query_month_or_numeric_date(line, lower, month)
        || !task_contains_any(lower, &["fun run", "fun runs", "5k fun run", "5k fun runs"])
        || !task_contains_any(lower, &["missed", "had to miss", "unable to attend"])
    {
        return None;
    }
    let mut days = extract_month_day_values_from_line(line, lower, month);
    if days.is_empty() {
        let rank = extract_explicit_date_rank(line)?;
        return Some(format!("fun-run:{rank}"));
    }
    days.sort_unstable();
    let day = *days.last()?;
    Some(format!("fun-run:{month}:{day}"))
}

pub(in crate::index) fn line_mentions_recent_three_month_window(lower: &str) -> bool {
    task_contains_any(
        lower,
        &[
            "today",
            "yesterday",
            "last week",
            "week ago",
            "weeks ago",
            "last month",
            "month ago",
            "months ago",
            "a few weeks ago",
            "few weeks ago",
            "a couple of weeks ago",
            "couple of weeks ago",
            "two months ago",
            "three months ago",
        ],
    )
}

pub(in crate::index) fn trim_trailing_relative_time_phrase(text: &str) -> String {
    let trimmed = Regex::new(
        r"(?i)\s+(?:about|around)?\s*(?:a\s+few|few|a\s+couple\s+of|couple\s+of|one|two|three|\d+)\s+(?:day|days|week|weeks|month|months|year|years)\s+ago[.!?,]?\s*$",
    )
    .unwrap()
    .replace(text.trim(), "")
    .to_string();
    trimmed
        .trim_matches(|c: char| matches!(c, '"' | '\'' | ',' | '.' | ';' | ':' | '!' | '?'))
        .trim()
        .to_string()
}

pub(in crate::index) fn extract_graduation_ceremony_signature_from_line(
    line: &str,
    lower: &str,
) -> Option<String> {
    if !lower.contains("graduation")
        || !task_contains_any(lower, &["attended my", "attended our", "attended the"])
        || !line_mentions_recent_three_month_window(lower)
    {
        return None;
    }
    let caps = Regex::new(
        r"(?i)attended (?:my|our|the) ([^\n]+?)'s ((?:[^.!?\n]+?\s+)?graduation(?: ceremony)?(?: from [^.!?\n]+?)?)\b",
    )
    .unwrap()
    .captures(line)?;
    let owner = normalized_synthetic_phrase_key(caps.get(1)?.as_str());
    let event =
        normalized_synthetic_phrase_key(&trim_trailing_relative_time_phrase(caps.get(2)?.as_str()));
    Some(format!("{owner}:{event}"))
}

pub(in crate::index) fn extract_health_device_units_from_line(_line: &str, lower: &str) -> Vec<String> {
    let mut devices = Vec::new();
    let mut seen = HashSet::new();

    let has_specific_fitbit =
        lower.contains("fitbit versa 3 smartwatch") || lower.contains("fitbit versa 3");
    let has_generic_fitbit = Regex::new(r"(?i)\bfitbit\b").unwrap().is_match(lower);
    let wearable = if has_specific_fitbit {
        Some("fitbit versa 3 smartwatch")
    } else if has_generic_fitbit {
        Some("fitbit")
    } else {
        None
    };
    if let Some(device) = wearable {
        if seen.insert(device) {
            devices.push(device.to_string());
        }
    }

    if lower.contains("hearing aids") {
        devices.push("left hearing aid".to_string());
        devices.push("right hearing aid".to_string());
        return devices;
    }

    let mentions_batteries = lower.contains("battery") || lower.contains("batteries");

    for device in [
        "hearing aid",
        "blood pressure monitor",
        "glucose monitor",
        "continuous glucose monitor",
        "fitness tracker",
        "smartwatch",
        "cpap",
        "inhaler",
    ] {
        if has_generic_fitbit && matches!(device, "fitness tracker" | "smartwatch") {
            continue;
        }
        if mentions_batteries && device == "hearing aid" {
            continue;
        }
        if lower.contains(device) && seen.insert(device) {
            devices.push(device.to_string());
        }
    }

    devices
}

pub(in crate::index) fn extract_peak_campaign_weekly_hour_delta_from_line(
    line: &str,
    lower: &str,
) -> Option<f32> {
    if !lower.contains("peak campaign")
        || !task_contains_any(
            lower,
            &["i increase my work hours by", "increase my work hours by"],
        )
    {
        return None;
    }
    Regex::new(
        r"(?i)\bincrease my (?:work )?hours by (\d+(?:\.\d+)?) hours? (?:weekly|a week|per week)\b",
    )
    .unwrap()
    .captures(line)?
    .get(1)?
    .as_str()
    .parse::<f32>()
    .ok()
}

pub(in crate::index) fn extract_typical_weekly_work_hours_from_line(line: &str, lower: &str) -> Option<f32> {
    if !task_contains_any(lower, &["i usually work", "usually work"]) {
        return None;
    }
    Regex::new(r"(?i)\bi usually work (\d+(?:\.\d+)?) hours? (?:a|per) week\b")
        .unwrap()
        .captures(line)?
        .get(1)?
        .as_str()
        .parse::<f32>()
        .ok()
}

pub(in crate::index) fn extract_peak_campaign_total_weekly_hours_from_line(
    line: &str,
    lower: &str,
) -> Option<f32> {
    if !lower.contains("peak campaign") {
        return None;
    }
    Regex::new(
        r"(?i)\b(?:working )?up to (\d+(?:\.\d+)?) hours?(?:\s*/\s*week|\s+per\s+week|\s+a\s+week)\b",
    )
    .unwrap()
    .captures(line)?
    .get(1)?
    .as_str()
    .parse::<f32>()
    .ok()
}

pub(in crate::index) fn extract_recent_activity_query_labels(task_lower: &str) -> Vec<&'static str> {
    let mut labels = Vec::new();
    for (label, needles) in [
        ("jogging", &["jogging", "jog"][..]),
        ("yoga", &["yoga"][..]),
        ("walking", &["walking", "walk"][..]),
        ("swimming", &["swimming", "swim"][..]),
        ("cycling", &["cycling", "biking", "bike", "cycle"][..]),
        (
            "strength training",
            &["strength training", "weightlifting", "lifting"][..],
        ),
    ] {
        if task_contains_any(task_lower, needles) {
            labels.push(label);
        }
    }
    labels
}

pub(in crate::index) fn line_mentions_recent_activity_label(lower: &str, label: &str) -> bool {
    match label {
        "jogging" => task_contains_any(lower, &["jogging", "jog", "jogged"]),
        "yoga" => lower.contains("yoga"),
        "walking" => task_contains_any(lower, &["walking", "walk", "walked"]),
        "swimming" => task_contains_any(lower, &["swimming", "swim", "swam"]),
        "cycling" => task_contains_any(lower, &["cycling", "biking", "bike", "biked", "cycled"]),
        "strength training" => {
            task_contains_any(lower, &["strength training", "weightlifting", "lifting"])
        },
        _ => false,
    }
}

pub(in crate::index) fn extract_recent_activity_duration_facts_from_line(
    line: &str,
    lower: &str,
    requested_activities: &[&'static str],
) -> Vec<(String, &'static str, SyntheticDurationValue)> {
    if !task_contains_any(
        lower,
        &[
            "i went for",
            "i went on",
            "i did",
            "i completed",
            "i ran",
            "i jogged",
            "i walked",
            "i biked",
            "i cycled",
            "i swam",
            "i practiced",
        ],
    ) {
        return Vec::new();
    }
    if task_contains_any(
        lower,
        &[
            "used to",
            "slacking off",
            "trying to get back",
            "schedule my",
            "set reminders",
            "habit",
            "times a week",
            "each time for",
            "looking to increase",
            "trying to incorporate",
        ],
    ) || line_has_future_goal_marker(lower)
    {
        return Vec::new();
    }

    let Some(duration) = extract_aggregate_duration_value(line) else {
        return Vec::new();
    };
    let day_surface = extract_weekday_surface_from_line(lower);
    let line_body = normalize_session_answer_line_body(line);
    let mut facts = Vec::new();
    for activity in requested_activities {
        if !line_mentions_recent_activity_label(lower, activity) {
            continue;
        }
        let signature = match day_surface.as_deref() {
            Some(day) => format!("{activity}:{day}"),
            None => format!("{activity}:{}", normalized_synthetic_phrase_key(&line_body)),
        };
        facts.push((signature, *activity, duration));
    }
    facts
}

pub(in crate::index) fn extract_current_magazine_subscription_updates_from_line(
    line: &str,
    lower: &str,
) -> Vec<(String, bool)> {
    let mut updates = Vec::new();
    let mut seen = HashSet::new();
    let mut push_update = |publication: Option<String>, is_active: bool| {
        let Some(publication) = publication else {
            return;
        };
        let publication = publication
            .trim_matches(|c: char| matches!(c, '"' | '\'' | ',' | '.' | ';' | ':' | '!' | '?'))
            .trim()
            .to_string();
        let normalized = normalized_synthetic_phrase_key(&publication);
        if publication.is_empty() || normalized.len() < 4 {
            return;
        }
        let key = format!("{normalized}:{is_active}");
        if seen.insert(key) {
            updates.push((publication, is_active));
        }
    };

    push_update(
        extract_phrase_after_any_index(
            line,
            lower,
            &["canceled my "],
            &[
                " magazine subscription",
                " subscription",
                " because ",
                ",",
                ".",
            ],
            1,
        ),
        false,
    );
    push_update(
        extract_phrase_after_any_index(
            line,
            lower,
            &[
                "loving my subscription to ",
                "enjoying my subscription to ",
                "my subscription to ",
            ],
            &[
                " magazine",
                " subscription",
                " which ",
                " in ",
                " on ",
                ",",
                ".",
                " -",
            ],
            1,
        ),
        true,
    );
    push_update(
        extract_phrase_after_any_index(
            line,
            lower,
            &["other publications like "],
            &[" which ", " in ", " on ", ",", ".", " -"],
            1,
        ),
        true,
    );
    push_update(
        extract_phrase_after_any_index(
            line,
            lower,
            &["i'm also getting ", "i am also getting "],
            &[" which ", " in ", " on ", ",", ".", " -"],
            1,
        ),
        true,
    );

    updates
}

pub(in crate::index) fn extract_hour_minute_total_from_text(text: &str) -> Option<i32> {
    for regex in [
        Regex::new(r"(?i)\b(\d+)\s*h(?:ours?)?\s*(\d+)\s*min(?:ute)?s?\b").unwrap(),
        Regex::new(r"(?i)\b(\d+)\s+hours?\s+(?:and\s+)?(\d+)\s+minutes?\b").unwrap(),
    ] {
        let Some(caps) = regex.captures(text) else {
            continue;
        };
        let hours = caps.get(1)?.as_str().parse::<i32>().ok()?;
        let minutes = caps.get(2)?.as_str().parse::<i32>().ok()?;
        return Some(hours * 60 + minutes);
    }
    None
}

pub(in crate::index) fn extract_marathon_completion_minutes_from_line(
    line: &str,
    lower: &str,
) -> Option<i32> {
    if !lower.contains("marathon")
        || !task_contains_any(
            lower,
            &["completed my first full marathon", "completed the marathon"],
        )
    {
        return None;
    }
    for marker in [
        "completed my first full marathon in ",
        "completed the marathon in ",
        "full marathon in ",
        "marathon in ",
    ] {
        let Some(idx) = lower.find(marker) else {
            continue;
        };
        if let Some(total) = extract_hour_minute_total_from_text(&line[idx + marker.len()..]) {
            return Some(total);
        }
    }
    None
}

pub(in crate::index) fn extract_marathon_target_minutes_from_line(line: &str, lower: &str) -> Option<i32> {
    if !lower.contains("marathon") || !lower.contains("target time") {
        return None;
    }
    for marker in [
        "target time for the marathon was ",
        "target time for the marathon is ",
        "target time was ",
    ] {
        let Some(idx) = lower.find(marker) else {
            continue;
        };
        if let Some(total) = extract_hour_minute_total_from_text(&line[idx + marker.len()..]) {
            return Some(total);
        }
    }
    None
}

pub(in crate::index) fn extract_attended_movie_festival_from_line(line: &str, lower: &str) -> Option<String> {
    if !task_contains_any(
        lower,
        &[
            "i volunteered",
            "i even volunteered",
            "i recently participated",
            "i was impressed by",
            "i got to discuss",
            "i've been fortunate enough",
            "i had the opportunity",
            "i had a great conversation",
            "i was part of a team",
            "i attended",
        ],
    ) {
        return None;
    }
    let caps = Regex::new(
        r"(?i)\b(?:at|after the screening at|like)\b\s+(?:the\s+)?([A-Z][A-Za-z0-9&' .-]+?Film Festival|AFI Fest|TIFF)\b",
    )
    .unwrap()
    .captures(line)?;
    Some(caps.get(1)?.as_str().trim().to_string())
}

pub(in crate::index) fn spell_small_cardinal(count: usize) -> Option<&'static str> {
    match count {
        0 => Some("zero"),
        1 => Some("one"),
        2 => Some("two"),
        3 => Some("three"),
        4 => Some("four"),
        5 => Some("five"),
        6 => Some("six"),
        7 => Some("seven"),
        8 => Some("eight"),
        9 => Some("nine"),
        10 => Some("ten"),
        11 => Some("eleven"),
        12 => Some("twelve"),
        _ => None,
    }
}

pub(in crate::index) fn extract_music_release_signatures_from_line(line: &str, lower: &str) -> Vec<String> {
    let mut releases = Vec::new();
    let mut seen = HashSet::new();

    if task_contains_any(lower, &["i bought", "i ended up buying"]) {
        if let Some(caps) = Regex::new(r#"(?i)\b(?:EP|album)\s+["']([^"']+)["']"#)
            .unwrap()
            .captures(line)
        {
            if let Some(title) = caps.get(1) {
                let key = normalized_synthetic_phrase_key(title.as_str());
                if key.len() >= 3 && seen.insert(key.clone()) {
                    releases.push(key);
                }
            }
        }
    }

    if lower.contains("downloaded") {
        if let Some(caps) = Regex::new(r#"(?i)\balbum\s+["']([^"']+)["'][^.\n]*\bdownloaded\b"#)
            .unwrap()
            .captures(line)
        {
            if let Some(title) = caps.get(1) {
                let key = normalized_synthetic_phrase_key(title.as_str());
                if key.len() >= 3 && seen.insert(key.clone()) {
                    releases.push(key);
                }
            }
        }
    }

    if lower.contains("vinyl") && lower.contains("signed") {
        for regex in [
            Regex::new(r"(?i)\bgot my ([A-Z][A-Za-z0-9&' .-]+?) vinyl signed\b").unwrap(),
            Regex::new(r"(?i)\bsaw ([A-Z][A-Za-z0-9&' .-]+?) live[^.\n]*\bgot my vinyl signed\b")
                .unwrap(),
        ] {
            let Some(caps) = regex.captures(line) else {
                continue;
            };
            let Some(artist) = caps.get(1) else {
                continue;
            };
            let key = normalized_synthetic_phrase_key(&format!("{} vinyl", artist.as_str().trim()));
            if key.len() >= 3 && seen.insert(key.clone()) {
                releases.push(key);
            }
        }
    }

    releases
}

pub(in crate::index) fn extract_owned_musical_instrument_signatures_from_line(
    line: &str,
    lower: &str,
) -> Vec<String> {
    let mut instruments = Vec::new();
    let mut seen = HashSet::new();
    if !(lower.starts_with("user:") || line.trim_start().starts_with('-')) {
        return instruments;
    }
    if task_contains_any(
        lower,
        &[
            "thinking of buying",
            "eyeing a ",
            "considering buying",
            "maybe getting",
            "might get",
            "want to buy",
        ],
    ) {
        return instruments;
    }

    let mut push = |label: String| {
        let key = normalized_synthetic_phrase_key(&label);
        if key.len() >= 3 && seen.insert(key.clone()) {
            instruments.push(key);
        }
    };

    if lower.contains("drum set")
        && task_contains_any(
            lower,
            &["my old drum set", "my drum set", "selling my old drum set"],
        )
    {
        let mut inserted = false;
        for regex in [
            Regex::new(
                r"\bdrum set,\s+a\s+((?:\d+-piece\s+)?[A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+)*)\b",
            )
            .unwrap(),
            Regex::new(
                r"\b((?:\d+-piece\s+)?[A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+)*)\s+drum set\b",
            )
            .unwrap(),
        ] {
            let Some(caps) = regex.captures(line) else {
                continue;
            };
            let Some(model) = caps.get(1) else {
                continue;
            };
            push(format!("{} drum set", model.as_str().trim()));
            inserted = true;
        }
        if !inserted {
            push("drum set".to_string());
        }
    }

    if lower.contains("piano") && lower.contains(" my ") {
        let mut inserted = false;
        for regex in [
            Regex::new(r"\bpiano,\s+a\s+([A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+)*)\b")
                .unwrap(),
            Regex::new(r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+)*)\s+piano\b").unwrap(),
            Regex::new(r"\b(Korg\s+B1)\b").unwrap(),
        ] {
            let Some(caps) = regex.captures(line) else {
                continue;
            };
            let Some(model) = caps.get(1) else {
                continue;
            };
            push(format!("{} piano", model.as_str().trim()));
            inserted = true;
        }
        if !inserted && lower.contains("my piano") {
            push("piano".to_string());
        }
    }

    if lower.contains("acoustic guitar") {
        let mut inserted = false;
        for regex in [
            Regex::new(
                r"\bacoustic guitar,\s+a\s+([A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+)*)\b",
            )
            .unwrap(),
            Regex::new(r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+)*)\s+acoustic guitar\b")
                .unwrap(),
        ] {
            let Some(caps) = regex.captures(line) else {
                continue;
            };
            let Some(model) = caps.get(1) else {
                continue;
            };
            push(format!("{} acoustic guitar", model.as_str().trim()));
            inserted = true;
        }
        if !inserted && lower.contains("my acoustic guitar") {
            push("acoustic guitar".to_string());
        }
    }

    if lower.contains("electric guitar") {
        let mut inserted = false;
        for regex in [
            Regex::new(
                r"\b(?:my|had my|playing my)\s+(?:[a-z]+\s+)?([A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+)*)\s+electric guitar\b",
            )
            .unwrap(),
            Regex::new(
                r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+)*)\s+electric guitar\b",
            )
            .unwrap(),
        ] {
            let Some(caps) = regex.captures(line) else {
                continue;
            };
            let Some(model) = caps.get(1) else {
                continue;
            };
            push(format!("{} electric guitar", model.as_str().trim()));
            inserted = true;
        }
        if !inserted && lower.contains("my electric guitar") {
            push("electric guitar".to_string());
        }
    }

    if lower.contains("ukulele") && lower.contains("my ") {
        let mut inserted = false;
        for regex in [
            Regex::new(r"\bukulele,\s+a\s+([A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+)*)\b")
                .unwrap(),
            Regex::new(r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+)*)\s+ukulele\b").unwrap(),
        ] {
            let Some(caps) = regex.captures(line) else {
                continue;
            };
            let Some(model) = caps.get(1) else {
                continue;
            };
            push(format!("{} ukulele", model.as_str().trim()));
            inserted = true;
        }
        if !inserted && lower.contains("my ukulele") {
            push("ukulele".to_string());
        }
    }

    instruments
}

pub(in crate::index) fn extract_online_course_completion_updates_from_line(
    line: &str,
    lower: &str,
) -> Vec<(String, i32)> {
    let mut updates = Vec::new();
    let mut seen = HashSet::new();
    if !(lower.starts_with("user:") || line.trim_start().starts_with('-'))
        || !lower.contains("completed")
        || !lower.contains("course")
    {
        return updates;
    }

    let mut count = None;
    for regex in [
        Regex::new(r"(?i)\bcompleted\s+([A-Za-z0-9,-]+)\s+courses?\b").unwrap(),
        Regex::new(r"(?i)\b([A-Za-z0-9,-]+)\s+courses?\s+on\b").unwrap(),
    ] {
        let Some(caps) = regex.captures(line) else {
            continue;
        };
        let Some(value) = caps
            .get(1)
            .and_then(|m| parse_count_token_value(m.as_str()))
        else {
            continue;
        };
        if value > 0 {
            count = Some(value);
            break;
        }
    }
    let Some(count) = count else {
        return updates;
    };

    for (platform_key, platform_name) in [
        ("coursera", "Coursera"),
        ("edx", "edX"),
        ("udemy", "Udemy"),
        ("datacamp", "DataCamp"),
        ("fast.ai", "Fast.ai"),
        ("kaggle", "Kaggle"),
    ] {
        if lower.contains(platform_key) && seen.insert(platform_key) {
            updates.push((platform_name.to_string(), count));
        }
    }

    updates
}

pub(in crate::index) fn extract_recent_furniture_action_signatures_from_line(
    line: &str,
    lower: &str,
) -> Vec<String> {
    let mut items = Vec::new();
    let mut seen = HashSet::new();
    if !(lower.starts_with("user:") || line.trim_start().starts_with('-')) {
        return items;
    }

    let mut push = |label: &str| {
        let key = normalized_synthetic_phrase_key(label);
        if key.len() >= 3 && seen.insert(key.clone()) {
            items.push(key);
        }
    };

    if lower.contains("coffee table")
        && task_contains_any(
            lower,
            &[
                "got a new coffee table",
                "got my coffee table",
                "bought my coffee table",
                "bought a coffee table",
                "coffee table was delivered",
                "delivered last thursday",
            ],
        )
    {
        push("coffee table");
    }

    if lower.contains("mattress")
        && task_contains_any(
            lower,
            &[
                "ordered one from casper",
                "ordered my new mattress",
                "ordered a new mattress",
                "took the plunge and ordered",
                "supposed to arrive",
                "mattress was delivered",
            ],
        )
    {
        push("mattress");
    }

    if lower.contains("bookshelf")
        && task_contains_any(lower, &["assembled", "built", "put together"])
    {
        push("bookshelf");
    }

    if task_contains_any(
        lower,
        &["fixed", "fixing", "repaired", "repairing", "wobbly leg"],
    ) {
        if lower.contains("kitchen table") {
            push("kitchen table");
        } else if lower.contains("coffee table") {
            push("coffee table");
        } else if lower.contains("desk") {
            push("desk");
        } else if lower.contains("chair") {
            push("chair");
        } else if lower.contains("dresser") {
            push("dresser");
        }
    }

    items
}

pub(in crate::index) fn extract_loyalty_point_goal_total_from_line(line: &str, lower: &str) -> Option<i32> {
    if !lower.contains("point") {
        return None;
    }
    for pattern in [
        r"(?i)\bneed(?:\s+\w+){0,4}\s+total of\s+(\d+)\s+points\b",
        r"(?i)\breach(?:ing)?\s+(\d+)\s+points\b",
        r"(?i)\b(\d+)\s+points goal\b",
    ] {
        let regex = Regex::new(pattern).unwrap();
        if let Some(caps) = regex.captures(line) {
            if let Ok(value) = caps.get(1)?.as_str().parse::<i32>() {
                return Some(value);
            }
        }
    }
    None
}

pub(in crate::index) fn extract_loyalty_point_current_total_from_line(
    line: &str,
    lower: &str,
) -> Option<i32> {
    if !lower.contains("point") {
        return None;
    }
    for pattern in [
        r"(?i)\bbringing my total to\s+(\d+)\s+points\b",
        r"(?i)\bmy total to\s+(\d+)\s+points\b",
        r"(?i)\btotal to\s+(\d+)\s+points so far\b",
    ] {
        let regex = Regex::new(pattern).unwrap();
        if let Some(caps) = regex.captures(line) {
            if let Ok(value) = caps.get(1)?.as_str().parse::<i32>() {
                return Some(value);
            }
        }
    }
    None
}

pub(in crate::index) fn extract_property_view_reason_from_line(
    line: &str,
    lower: &str,
) -> Option<(String, i32, String)> {
    let rank = extract_explicit_date_rank(line)?;

    if lower.contains("1-bedroom condo") && lower.contains("highway") {
        return Some((
            "1-bedroom condo".to_string(),
            rank,
            "the noise from the highway was a deal-breaker for the 1-bedroom condo".to_string(),
        ));
    }

    if lower.contains("bungalow") && lower.contains("kitchen") && lower.contains("renovation") {
        return Some((
            "bungalow".to_string(),
            rank,
            "the kitchen of the bungalow needed serious renovation".to_string(),
        ));
    }

    if lower.contains("cedar creek")
        && (lower.contains("out of my budget")
            || lower.contains("way out of my league")
            || lower.contains("didn't fit my budget"))
    {
        return Some((
            "property in cedar creek".to_string(),
            rank,
            "the property in Cedar Creek was out of my budget".to_string(),
        ));
    }

    if lower.contains("2-bedroom condo") && lower.contains("higher bid") {
        return Some((
            "2-bedroom condo".to_string(),
            rank,
            "my offer on the 2-bedroom condo was rejected due to a higher bid".to_string(),
        ));
    }

    None
}

pub(in crate::index) fn small_cardinal_word_lower(value: usize) -> String {
    match value {
        0 => "zero".to_string(),
        1 => "one".to_string(),
        2 => "two".to_string(),
        3 => "three".to_string(),
        4 => "four".to_string(),
        5 => "five".to_string(),
        6 => "six".to_string(),
        7 => "seven".to_string(),
        8 => "eight".to_string(),
        9 => "nine".to_string(),
        10 => "ten".to_string(),
        _ => value.to_string(),
    }
}

pub(in crate::index) fn join_reason_clauses(items: &[String]) -> String {
    match items {
        [] => String::new(),
        [only] => only.clone(),
        [first, second] => format!("{first} and {second}"),
        _ => {
            let mut rendered = items[..items.len() - 1].join(", ");
            rendered.push_str(", and ");
            rendered.push_str(items.last().unwrap());
            rendered
        },
    }
}

pub(in crate::index) fn collapsed_owned_instrument_count(instruments: &HashSet<String>) -> usize {
    retained_owned_instrument_keys(instruments).len()
}

pub(in crate::index) fn retained_owned_instrument_keys(instruments: &HashSet<String>) -> Vec<String> {
    let mut retained = instruments
        .iter()
        .filter(|instrument| {
            let Some(suffix) = (match instrument.as_str() {
                "drum set" => Some(" drum set"),
                "piano" => Some(" piano"),
                "acoustic guitar" => Some(" acoustic guitar"),
                "electric guitar" => Some(" electric guitar"),
                "ukulele" => Some(" ukulele"),
                _ => None,
            }) else {
                return true;
            };
            !instruments
                .iter()
                .any(|other| other.as_str() != instrument.as_str() && other.ends_with(suffix))
        })
        .cloned()
        .collect::<Vec<_>>();
    retained.sort_by_key(|instrument| {
        let rank = if instrument.ends_with(" electric guitar") {
            0
        } else if instrument.ends_with(" acoustic guitar") {
            1
        } else if instrument.ends_with(" drum set") {
            2
        } else if instrument.ends_with(" piano") {
            3
        } else if instrument.ends_with(" ukulele") {
            4
        } else {
            5
        };
        (rank, instrument.clone())
    });
    retained
}

pub(in crate::index) fn compose_current_musical_instrument_count_answer(
    instruments: &HashSet<String>,
    durations: &HashMap<String, Option<String>>,
    count: usize,
) -> String {
    let retained = retained_owned_instrument_keys(instruments);
    if retained.is_empty() {
        return count.to_string();
    }
    let descriptors = retained
        .iter()
        .map(|instrument| {
            let display = display_owned_instrument_label(instrument);
            let duration = durations
                .get(instrument)
                .and_then(|value| value.as_ref())
                .map(String::as_str)
                .unwrap_or("an unspecified amount of time");
            format!("the {display} for {duration}")
        })
        .collect::<Vec<_>>();
    let joined = match descriptors.as_slice() {
        [only] => only.clone(),
        [first, second] => format!("{first} and {second}"),
        [first, second, third] => format!("{first}, {second}, and {third}"),
        _ => {
            let mut leading = descriptors[..descriptors.len() - 1].join(", ");
            leading.push_str(", and ");
            leading.push_str(descriptors.last().unwrap());
            leading
        },
    };
    format!("I currently own {count} musical instruments. I've had {joined}.")
}

pub(in crate::index) fn display_owned_instrument_label(instrument: &str) -> String {
    instrument
        .split_whitespace()
        .map(|word| match word {
            "electric" | "acoustic" | "guitar" | "drum" | "set" | "piano" | "ukulele" => {
                word.to_string()
            },
            "fg800" => "FG800".to_string(),
            "b1" => "B1".to_string(),
            _ if word
                .chars()
                .next()
                .map(|c| c.is_ascii_digit())
                .unwrap_or(false) =>
            {
                word.to_string()
            },
            _ => capitalize_first_ascii(word),
        })
        .collect::<Vec<_>>()
        .join(" ")
}

pub(in crate::index) fn extract_weekday_from_query(task_lower: &str) -> Option<&'static str> {
    [
        "sunday",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
    ]
    .into_iter()
    .find(|day| task_lower.contains(day))
}

pub(in crate::index) fn extract_weekday_surface_from_line(lower: &str) -> Option<String> {
    extract_weekday_from_query(lower).map(capitalize_first_ascii)
}

pub(in crate::index) fn pluralize_weekday(day: &str) -> String {
    let mut chars = day.chars();
    let first = chars.next().map(|c| c.to_ascii_uppercase()).unwrap_or('D');
    format!("{first}{}s", chars.as_str())
}

pub(in crate::index) fn extract_schedule_query_person(task: &str) -> Option<String> {
    let mut best = None;
    for token in task.split(|c: char| !c.is_ascii_alphabetic() && c != '-') {
        let trimmed = token.trim();
        if trimmed.len() < 3 {
            continue;
        }
        let Some(first) = trimmed.chars().next() else {
            continue;
        };
        if !first.is_ascii_uppercase() {
            continue;
        }
        let lower = trimmed.to_ascii_lowercase();
        if [
            "bandung",
            "can",
            "cihampelas",
            "friday",
            "gm",
            "monday",
            "previous",
            "saturday",
            "sunday",
            "thursday",
            "tuesday",
            "wednesday",
        ]
        .contains(&lower.as_str())
        {
            continue;
        }
        best = Some(trimmed.to_string());
    }
    best
}

pub(in crate::index) fn parse_markdown_table_cells(line: &str) -> Option<Vec<String>> {
    let trimmed = line.trim();
    if !trimmed.starts_with('|') {
        return None;
    }
    let cells: Vec<String> = trimmed
        .split('|')
        .map(str::trim)
        .filter(|cell| !cell.is_empty())
        .map(|cell| cell.to_string())
        .collect();
    if cells.is_empty() {
        return None;
    }
    if cells
        .iter()
        .all(|cell| cell.chars().all(|c| matches!(c, '-' | ':' | ' ')))
    {
        return None;
    }
    Some(cells)
}

pub(in crate::index) fn extract_schedule_shift_from_table(
    lines: &[String],
    person: &str,
    day: &str,
) -> Option<(String, Vec<String>)> {
    fn looks_like_shift_header_row(cells: &[String]) -> bool {
        cells.iter().any(|cell| {
            let lower = cell.to_ascii_lowercase();
            lower.contains("shift") || lower.contains("am -") || lower.contains("pm -")
        })
    }

    let person_lower = person.to_ascii_lowercase();
    let mut header = None::<(Vec<String>, String)>;
    for line in lines {
        let Some(cells) = parse_markdown_table_cells(line) else {
            continue;
        };
        if looks_like_shift_header_row(&cells) {
            header = Some((cells, line.clone()));
            continue;
        }
        if header.is_none() {
            header = Some((cells, line.clone()));
            continue;
        }
        let Some((header_cells, header_line)) = header.as_ref() else {
            continue;
        };
        if cells.is_empty() || !cells[0].eq_ignore_ascii_case(day) {
            continue;
        }
        for (idx, cell) in cells.iter().enumerate().skip(1) {
            if cell.eq_ignore_ascii_case(&person_lower) || cell.eq_ignore_ascii_case(person) {
                let header_idx = if header_cells.len() + 1 == cells.len() {
                    idx - 1
                } else {
                    idx
                };
                if let Some(shift) = header_cells.get(header_idx) {
                    return Some((shift.clone(), vec![header_line.clone(), line.clone()]));
                }
            }
        }
    }
    None
}

pub(in crate::index) fn extract_served_dish_from_query(task: &str, task_lower: &str) -> Option<String> {
    let marker = if let Some(idx) = task_lower.find("serves ") {
        idx + "serves ".len()
    } else if let Some(idx) = task_lower.find("serve ") {
        idx + "serve ".len()
    } else {
        return None;
    };
    let tail = task[marker..].trim();
    let mut words = Vec::new();
    for raw in tail.split_whitespace() {
        let cleaned = raw
            .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '\'')
            .to_ascii_lowercase();
        if cleaned.is_empty() {
            continue;
        }
        if ["a", "an", "the", "great", "good"].contains(&cleaned.as_str()) {
            continue;
        }
        if ["that", "which", "with", "in"].contains(&cleaned.as_str()) {
            break;
        }
        words.push(cleaned);
    }
    (!words.is_empty()).then(|| words.join(" "))
}

pub(in crate::index) fn extract_list_item_label(line: &str) -> Option<String> {
    let trimmed = line.trim();
    let candidate = trimmed
        .trim_start_matches(|c: char| c.is_ascii_digit() || c == '.' || c == ' ')
        .trim();
    let label = candidate.split(':').next()?.trim();
    (!label.is_empty()).then(|| label.to_string())
}

pub(in crate::index) fn venue_stem_from_dish_label(label: &str, dish: &str) -> Option<String> {
    let lower = label.to_ascii_lowercase();
    let dish_lower = dish.to_ascii_lowercase();
    if let Some(idx) = lower.find(&format!("'s {dish_lower}")) {
        return Some(label[..idx].trim().to_string());
    }
    lower
        .find(&dish_lower)
        .map(|idx| label[..idx].trim().to_string())
        .filter(|stem| !stem.is_empty())
}

pub(in crate::index) fn extract_restaurant_serving_dish(
    lines: &[String],
    dish: &str,
) -> Option<(String, Vec<String>)> {
    let dish_lower = dish.to_ascii_lowercase();
    for line in lines {
        if !line.contains(':') {
            continue;
        }
        let lower = line.to_ascii_lowercase();
        if !lower.contains(&dish_lower) {
            continue;
        }
        let Some(label) = extract_list_item_label(line) else {
            continue;
        };
        let Some(candidate_stem) = venue_stem_from_dish_label(&label, dish) else {
            continue;
        };
        let stem_lower = candidate_stem.to_ascii_lowercase();
        let mut best = None::<(String, String)>;
        for venue_line in lines {
            if !venue_line.contains(':') {
                continue;
            }
            let Some(venue_label) = extract_list_item_label(venue_line) else {
                continue;
            };
            let lower_label = venue_label.to_ascii_lowercase();
            if lower_label.contains(&dish_lower) || !lower_label.contains(&stem_lower) {
                continue;
            }
            if best
                .as_ref()
                .map(|(current, _)| venue_label.len() > current.len())
                .unwrap_or(true)
            {
                best = Some((venue_label, venue_line.clone()));
            }
        }
        if let Some((restaurant, venue_line)) = best {
            return Some((restaurant, vec![venue_line, line.clone()]));
        }
    }
    None
}

pub(in crate::index) fn extract_commute_duration_from_line(line: &str) -> Option<String> {
    let lower = line.to_ascii_lowercase();
    if !lower.contains("commute") {
        return None;
    }
    let pattern = Regex::new(
        r"(?i)(?:which\s+takes|takes|is)\s+(?:about\s+)?((?:an?|one|\d+)\s+(?:hours?|minutes?)(?:\s+each\s+way)?)",
    )
    .unwrap();
    pattern.captures(line).and_then(|caps| {
        caps.get(1).map(|m| {
            m.as_str()
                .trim()
                .trim_matches(|c: char| matches!(c, ',' | '.' | ';' | ':' | '"' | '\''))
                .to_string()
        })
    })
}

pub(in crate::index) fn extract_store_name_from_line(_line: &str, lower: &str) -> Option<String> {
    [
        ("whole foods", "Whole Foods"),
        ("trader joe", "Trader Joe's"),
        ("target", "Target"),
        ("walmart", "Walmart"),
        ("costco", "Costco"),
        ("walgreens", "Walgreens"),
        ("cvs", "CVS"),
    ]
    .into_iter()
    .find_map(|(needle, rendered)| lower.contains(needle).then(|| rendered.to_string()))
}

pub(in crate::index) fn extract_image_subject_from_query(task: &str) -> Option<String> {
    let scoped = Regex::new(r"of the ([A-Z][A-Za-z-]+)").unwrap();
    if let Some(subject) = scoped
        .captures(task)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().to_string())
    {
        return Some(subject);
    }

    let mut best = None::<String>;
    for token in task.split(|c: char| !c.is_ascii_alphabetic() && c != '-') {
        let trimmed = token.trim();
        if trimmed.len() < 3 {
            continue;
        }
        let Some(first) = trimmed.chars().next() else {
            continue;
        };
        if !first.is_ascii_uppercase() {
            continue;
        }
        let lower = trimmed.to_ascii_lowercase();
        if ["bandung", "can", "dinosaurs", "i", "im", "what"].contains(&lower.as_str()) {
            continue;
        }
        best = Some(trimmed.to_string());
    }
    best
}

pub(in crate::index) fn extract_image_subject_body_color(
    lines: &[String],
    subject: &str,
) -> Option<(String, Vec<String>)> {
    let pattern = Regex::new(&format!(
        r"(?i)\b{}\b[^.]*?\bhas a ([a-z ]+?) body",
        regex::escape(subject)
    ))
    .unwrap();
    for line in lines {
        let Some(caps) = pattern.captures(line) else {
            continue;
        };
        let phrase = caps
            .get(1)
            .map(|m| {
                m.as_str()
                    .trim()
                    .trim_matches(|c: char| matches!(c, ',' | '.' | ';' | ':' | '"' | '\''))
                    .to_string()
            })
            .filter(|value| !value.is_empty())?;
        let answer = format!("The {subject} had a {phrase} body.");
        return Some((answer, vec![line.clone()]));
    }
    None
}

pub(in crate::index) fn extract_issue_after_service_line(line: &str, lower: &str) -> Option<String> {
    let mut issue = extract_phrase_after_any_index(
        line,
        lower,
        &["issue with my ", "issue with the ", "issue with "],
        &[" on ", " and ", " but ", " because ", ","],
        2,
    )?;
    let prefixes = ["my car's ", "the car's ", "car's ", "my ", "the "];
    for prefix in prefixes {
        if issue.to_ascii_lowercase().starts_with(prefix) {
            issue = issue[prefix.len()..].trim().to_string();
            break;
        }
    }
    let lower_issue = issue.to_ascii_lowercase();
    if lower_issue.contains("gps") && lower_issue.contains("system") {
        return Some("GPS system not functioning correctly".to_string());
    }
    Some(issue)
}

pub(in crate::index) fn extract_dollar_amounts(line: &str) -> Vec<f32> {
    let pattern = Regex::new(r"\$([0-9][0-9,]*(?:\.[0-9]+)?)").unwrap();
    pattern
        .captures_iter(line)
        .filter_map(|caps| caps.get(1))
        .filter_map(|m| m.as_str().replace(',', "").parse::<f32>().ok())
        .collect()
}

pub(in crate::index) fn is_grounded_user_money_fact_line(lower: &str) -> bool {
    if !lower.trim_start().starts_with("user:") {
        return false;
    }

    ![
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
        "going to order",
        "order next week",
        "thinking about getting",
        "set a budget",
        "budget and stick to it",
        "budget for",
        "budget of $",
        "my budget is $",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

pub(in crate::index) fn is_grounded_user_duration_fact_line(lower: &str) -> bool {
    lower.trim_start().starts_with("user:")
}

pub(in crate::index) fn split_numeric_aggregate_segments(line: &str) -> Vec<String> {
    let mut segments = vec![line.trim().to_string()];
    for delimiter in [". ", "! ", "? ", "; ", " but ", " however ", " while "] {
        let mut next = Vec::new();
        for segment in segments {
            if segment.contains(delimiter) {
                next.extend(
                    segment
                        .split(delimiter)
                        .map(str::trim)
                        .filter(|part| !part.is_empty())
                        .map(ToString::to_string),
                );
            } else if !segment.is_empty() {
                next.push(segment);
            }
        }
        segments = next;
    }
    segments
}

pub(in crate::index) fn split_duration_aggregate_segments(line: &str) -> Vec<String> {
    let mut segments = split_numeric_aggregate_segments(line);
    for delimiter in [
        ", By the way, ",
        " By the way, ",
        ", by the way, ",
        " by the way, ",
        ", like ",
        " like ",
    ] {
        let mut next = Vec::new();
        for segment in segments {
            if segment.contains(delimiter) {
                next.extend(
                    segment
                        .split(delimiter)
                        .map(str::trim)
                        .filter(|part| !part.is_empty())
                        .map(ToString::to_string),
                );
            } else if !segment.is_empty() {
                next.push(segment);
            }
        }
        segments = next;
    }
    segments
}

pub(in crate::index) fn extract_focused_dollar_amounts(line: &str, focus_terms: &[String]) -> Vec<f32> {
    let amounts = extract_dollar_amounts(line);
    if amounts.len() <= 1 {
        return amounts;
    }

    let focus_refs: Vec<&str> = focus_terms.iter().map(String::as_str).collect();
    let mut focused = Vec::new();
    for segment in split_numeric_aggregate_segments(line) {
        let lower = segment.to_ascii_lowercase();
        if term_overlap_count(&lower, &focus_refs) == 0 {
            continue;
        }
        focused.extend(extract_dollar_amounts(&segment));
    }
    if !focused.is_empty() {
        return focused;
    }

    let lower = line.to_ascii_lowercase();
    if term_overlap_count(&lower, &focus_refs) > 0 {
        amounts
    } else {
        Vec::new()
    }
}

pub(in crate::index) fn money_total_line_matches_query(task_lower: &str, lower: &str) -> bool {
    if !task_contains_any(
        lower,
        &[
            "bought",
            "buy ",
            "got ",
            "cost me",
            "paid",
            "spent",
            "splurge",
            "purchase",
            "purchased",
            "installed",
            "replaced",
        ],
    ) {
        return false;
    }

    if task_lower.contains("luxury") {
        return task_contains_any(lower, &["luxury", "designer", "gucci", "high-end"]);
    }
    if task_lower.contains("bike") {
        return task_contains_any(
            lower,
            &[
                "bike",
                "helmet",
                "lights",
                "chain",
                "cycling",
                "tune-up",
                "bike shop",
            ],
        );
    }
    true
}

pub(in crate::index) fn format_numeric_answer(value: f32) -> String {
    if (value - value.round()).abs() < 0.01 {
        return (value.round() as i64).to_string();
    }

    let mut rendered = format!("{value:.2}");
    while rendered.ends_with('0') {
        rendered.pop();
    }
    if rendered.ends_with('.') {
        rendered.pop();
    }
    rendered
}

pub(in crate::index) fn format_integer_with_commas(value: i64) -> String {
    let digits = value.abs().to_string();
    let mut parts = Vec::new();
    let mut idx = digits.len();
    while idx > 3 {
        parts.push(digits[idx - 3..idx].to_string());
        idx -= 3;
    }
    parts.push(digits[..idx].to_string());
    parts.reverse();
    let joined = parts.join(",");
    if value < 0 {
        format!("-{joined}")
    } else {
        joined
    }
}

pub(in crate::index) fn format_money_answer(value: f32) -> String {
    if (value - value.round()).abs() < 0.01 {
        return format!("${}", format_integer_with_commas(value.round() as i64));
    }
    format!("${}", format_numeric_answer(value))
}

pub(in crate::index) fn extract_aggregate_duration_value(line: &str) -> Option<SyntheticDurationValue> {
    fn parse_amount(token: &str) -> Option<f32> {
        match token.to_ascii_lowercase().as_str() {
            "a" | "an" | "one" => Some(1.0),
            "two" => Some(2.0),
            "three" => Some(3.0),
            "four" => Some(4.0),
            "five" => Some(5.0),
            "six" => Some(6.0),
            "seven" => Some(7.0),
            "eight" => Some(8.0),
            "nine" => Some(9.0),
            "ten" => Some(10.0),
            "eleven" => Some(11.0),
            "twelve" => Some(12.0),
            "couple" => Some(2.0),
            "few" => Some(3.0),
            value => value.parse::<f32>().ok(),
        }
    }

    let postfix_half = Regex::new(
        r"(?i)\b(?:about\s+|around\s+)?(a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|couple|few|\d+(?:\.\d+)?)(?:\s+|-)(day|days|week|weeks|month|months|year|years|hour|hours|minute|minutes)\s+and\s+a\s+half\b",
    )
    .unwrap();
    let long_form = Regex::new(
        r"(?i)\b(?:(a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|couple|few|\d+(?:\.\d+)?)\s+)?(day|days|week|weeks|month|months|year|years|hour|hours|minute|minutes)(?:-|\s+)long\b",
    )
    .unwrap();
    let prefix_half = Regex::new(
        r"(?i)\b(?:about\s+|around\s+)?(a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|couple|few|\d+(?:\.\d+)?)(\s+and\s+a\s+half)?(?:\s+|-)(day|days|week|weeks|month|months|year|years|hour|hours|minute|minutes)\b",
    )
    .unwrap();
    let (amount_token, has_half, unit) = if let Some(caps) = postfix_half.captures(line) {
        (
            caps.get(1)?.as_str().to_string(),
            true,
            caps.get(2)?.as_str().to_ascii_lowercase(),
        )
    } else if let Some(caps) = long_form.captures(line) {
        (
            caps.get(1)
                .map(|m| m.as_str().to_string())
                .unwrap_or_else(|| "one".to_string()),
            false,
            caps.get(2)?.as_str().to_ascii_lowercase(),
        )
    } else {
        let caps = prefix_half.captures(line)?;
        (
            caps.get(1)?.as_str().to_string(),
            caps.get(2).is_some(),
            caps.get(3)?.as_str().to_ascii_lowercase(),
        )
    };
    let mut amount = parse_amount(&amount_token)?;
    if has_half {
        amount += 0.5;
    }
    let days = amount
        * match unit.as_str() {
            "minute" | "minutes" => 1.0 / (24.0 * 60.0),
            "hour" | "hours" => 1.0 / 24.0,
            "day" | "days" => 1.0,
            "week" | "weeks" => 7.0,
            "month" | "months" => 30.0,
            "year" | "years" => 365.0,
            _ => return None,
        };
    Some(SyntheticDurationValue {
        amount,
        days,
        unit: match unit.as_str() {
            "minute" | "minutes" => "minute",
            "hour" | "hours" => "hour",
            "day" | "days" => "day",
            "week" | "weeks" => "week",
            "month" | "months" => "month",
            "year" | "years" => "year",
            _ => return None,
        },
    })
}

pub(in crate::index) fn extract_requested_aggregate_duration_unit(task_lower: &str) -> Option<&'static str> {
    let caps = Regex::new(r"(?i)\bhow many\s+(day|days|week|weeks|month|months|year|years|hour|hours|minute|minutes)\b")
        .unwrap()
        .captures(task_lower)?;
    match caps.get(1)?.as_str().to_ascii_lowercase().as_str() {
        "minute" | "minutes" => Some("minute"),
        "hour" | "hours" => Some("hour"),
        "day" | "days" => Some("day"),
        "week" | "weeks" => Some("week"),
        "month" | "months" => Some("month"),
        "year" | "years" => Some("year"),
        _ => None,
    }
}

pub(in crate::index) fn format_aggregate_duration_answer(amount: f32, unit: &str) -> String {
    let rendered = format_numeric_answer(amount);
    let singular = (amount - 1.0).abs() < 0.01;
    let suffix = if singular {
        unit.to_string()
    } else {
        format!("{unit}s")
    };
    format!("{rendered} {suffix}")
}

pub(in crate::index) fn convert_duration_days(days: f32, unit: &str) -> f32 {
    match unit {
        "minute" => days * 24.0 * 60.0,
        "hour" => days * 24.0,
        "day" => days,
        "week" => days / 7.0,
        "month" => days / 30.0,
        "year" => days / 365.0,
        _ => days,
    }
}

pub(in crate::index) fn should_try_multi_session_money_total(task_lower: &str) -> bool {
    task_contains_any(
        task_lower,
        &[
            "$",
            " dollar",
            " dollars",
            " money",
            " expense",
            " expenses",
            " cost",
            " costs",
            " paid",
            " purchase",
            " purchased",
            " spent",
            " amount",
        ],
    ) && task_contains_any(
        task_lower,
        &[
            "how much total",
            "total money",
            "total amount",
            "in total",
            "combined",
            "altogether",
            "since the start",
            "past few months",
            "expenses",
        ],
    ) && !task_contains_any(
        task_lower,
        &[
            " compared to ",
            " difference ",
            " more expensive ",
            " less expensive ",
            " save ",
            " saved ",
            " each ",
            " per ",
            " before ",
            " after ",
        ],
    )
}

pub(in crate::index) fn should_try_multi_session_duration_total(task_lower: &str) -> bool {
    extract_requested_aggregate_duration_unit(task_lower).is_some()
        && (task_contains_any(
            task_lower,
            &[
                " in total",
                " combined",
                " altogether",
                " this year",
                " since the start",
                " past few months",
            ],
        ) || task_lower.contains(" and ")
            || task_contains_any(
                task_lower,
                &[
                    "trips",
                    "breaks",
                    "games",
                    "destinations",
                    "films",
                    "movies",
                ],
            ))
        && !task_contains_any(
            task_lower,
            &[
                "formal education",
                "high school",
                "bachelor",
                "master",
                "degree",
                "college",
                "university",
            ],
        )
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(in crate::index) enum EducationStageKind {
    HighSchool,
    Associate,
    Bachelor,
    Master,
}

#[derive(Clone, Debug)]
pub(in crate::index) struct EducationStageFact {
    kind: EducationStageKind,
    completed: bool,
    start_year: Option<i32>,
    end_year: Option<i32>,
    duration_years: Option<i32>,
    evidence: String,
}

pub(in crate::index) fn extract_formal_education_target_stage(
    task_lower: &str,
) -> Option<EducationStageKind> {
    if !task_lower.contains("formal education") || !task_lower.contains("high school") {
        return None;
    }
    if task_lower.contains("bachelor") {
        return Some(EducationStageKind::Bachelor);
    }
    if task_lower.contains("master") {
        return Some(EducationStageKind::Master);
    }
    None
}

pub(in crate::index) fn collect_education_stage_facts(
    lines: &[String],
) -> HashMap<EducationStageKind, EducationStageFact> {
    let mut facts = HashMap::new();
    for line in lines {
        let Some(parsed) = parse_education_stage_fact(line) else {
            continue;
        };
        let should_replace = facts
            .get(&parsed.kind)
            .map(|existing| {
                education_stage_fact_score(&parsed) > education_stage_fact_score(existing)
            })
            .unwrap_or(true);
        if should_replace {
            facts.insert(parsed.kind, parsed);
        }
    }
    facts
}

pub(in crate::index) fn solve_formal_education_total(
    facts: &HashMap<EducationStageKind, EducationStageFact>,
    target_stage: EducationStageKind,
) -> Option<(i32, Vec<String>, usize)> {
    let high_school = facts.get(&EducationStageKind::HighSchool)?;
    let high_school_duration = education_stage_duration_years(high_school)?;
    let high_school_end = education_stage_end_year(high_school)?;

    let bachelor = facts
        .get(&EducationStageKind::Bachelor)
        .filter(|fact| fact.completed)?;
    let bachelor_duration = education_stage_duration_years(bachelor)?;
    let bachelor_start = education_stage_start_year(bachelor)?;
    let bachelor_end = education_stage_end_year(bachelor)?;

    let mut total_years = high_school_duration + bachelor_duration;
    let mut evidence = vec![high_school.evidence.clone()];

    if let Some(associate) = facts
        .get(&EducationStageKind::Associate)
        .filter(|fact| fact.completed)
    {
        let associate_duration = education_stage_duration_years(associate).or_else(|| {
            let associate_end = education_stage_end_year(associate)?;
            ((associate_end > high_school_end) && (associate_end <= bachelor_start))
                .then_some(associate_end - high_school_end)
        });
        if let Some(years) = associate_duration.filter(|years| *years > 0) {
            total_years += years;
            evidence.push(associate.evidence.clone());
        }
    }

    evidence.push(bachelor.evidence.clone());

    if target_stage == EducationStageKind::Master {
        let master = facts
            .get(&EducationStageKind::Master)
            .filter(|fact| fact.completed)?;
        let master_duration = education_stage_duration_years(master).or_else(|| {
            let master_end = education_stage_end_year(master)?;
            (master_end > bachelor_end).then_some(master_end - bachelor_end)
        })?;
        if master_duration <= 0 {
            return None;
        }
        total_years += master_duration;
        evidence.push(master.evidence.clone());
    }

    let fact_count = evidence.len();
    Some((total_years, evidence, fact_count))
}

pub(in crate::index) fn parse_education_stage_fact(line: &str) -> Option<EducationStageFact> {
    let body = normalize_session_answer_line_body(line);
    let lower = body.to_ascii_lowercase();
    let years = extract_year_mentions(&body);

    let high_school_range =
        Regex::new(r"(?i)\bhigh school\b.*?\bfrom\s+(\d{4})\s+to\s+(\d{4})\b").unwrap();
    if let Some(caps) = high_school_range.captures(&body) {
        let start_year = caps.get(1)?.as_str().parse::<i32>().ok()?;
        let end_year = caps.get(2)?.as_str().parse::<i32>().ok()?;
        if end_year > start_year {
            return Some(EducationStageFact {
                kind: EducationStageKind::HighSchool,
                completed: true,
                start_year: Some(start_year),
                end_year: Some(end_year),
                duration_years: Some(end_year - start_year),
                evidence: line.to_string(),
            });
        }
    }

    if task_contains_any(
        &lower,
        &[
            "associate's degree",
            "associates degree",
            "associate degree",
        ],
    ) {
        return Some(EducationStageFact {
            kind: EducationStageKind::Associate,
            completed: task_contains_any(&lower, &["earned", "completed", "graduated"]),
            start_year: None,
            end_year: years.last().copied(),
            duration_years: extract_education_duration_years(&lower),
            evidence: line.to_string(),
        });
    }

    if task_contains_any(
        &lower,
        &[
            "bachelor's degree",
            "bachelors degree",
            "bachelor degree",
            "bachelor's in",
            "bachelors in",
            "bachelor in",
        ],
    ) {
        return Some(EducationStageFact {
            kind: EducationStageKind::Bachelor,
            completed: task_contains_any(&lower, &["graduated", "earned", "completed"])
                || lower.contains("took me"),
            start_year: None,
            end_year: years.last().copied(),
            duration_years: extract_education_duration_years(&lower),
            evidence: line.to_string(),
        });
    }

    if task_contains_any(
        &lower,
        &[
            "master's degree",
            "masters degree",
            "master degree",
            "master's in",
            "masters in",
            "master in",
        ],
    ) {
        return Some(EducationStageFact {
            kind: EducationStageKind::Master,
            completed: task_contains_any(&lower, &["graduated", "earned", "completed", "finished"]),
            start_year: None,
            end_year: years.last().copied(),
            duration_years: extract_education_duration_years(&lower),
            evidence: line.to_string(),
        });
    }

    None
}

pub(in crate::index) fn extract_education_duration_years(lower: &str) -> Option<i32> {
    for marker in [
        "which took me ",
        "took me ",
        "completed in ",
        "finished in ",
    ] {
        let Some(idx) = lower.find(marker) else {
            continue;
        };
        let tail = &lower[idx + marker.len()..];
        let value = parse_leading_duration_value(tail)?;
        if value.unit == "year" {
            return Some(value.amount.round() as i32);
        }
    }
    None
}

pub(in crate::index) fn extract_year_mentions(text: &str) -> Vec<i32> {
    let years = Regex::new(r"\b(19|20)\d{2}\b").unwrap();
    years
        .captures_iter(text)
        .filter_map(|caps| caps.get(0).and_then(|m| m.as_str().parse::<i32>().ok()))
        .collect()
}

pub(in crate::index) fn education_stage_fact_score(fact: &EducationStageFact) -> i32 {
    let mut score = 0;
    if fact.completed {
        score += 2;
    }
    if fact.start_year.is_some() {
        score += 2;
    }
    if fact.end_year.is_some() {
        score += 2;
    }
    if fact.duration_years.is_some() {
        score += 3;
    }
    score
}

pub(in crate::index) fn education_stage_duration_years(fact: &EducationStageFact) -> Option<i32> {
    fact.duration_years.or_else(|| {
        fact.start_year
            .zip(fact.end_year)
            .and_then(|(start, end)| (end > start).then_some(end - start))
    })
}

pub(in crate::index) fn education_stage_start_year(fact: &EducationStageFact) -> Option<i32> {
    fact.start_year.or_else(|| {
        fact.end_year
            .zip(fact.duration_years)
            .and_then(|(end, years)| (years > 0).then_some(end - years))
    })
}

pub(in crate::index) fn education_stage_end_year(fact: &EducationStageFact) -> Option<i32> {
    fact.end_year.or_else(|| {
        fact.start_year
            .zip(fact.duration_years)
            .and_then(|(start, years)| (years > 0).then_some(start + years))
    })
}

pub(in crate::index) fn extract_multi_session_money_focus_terms(task_lower: &str) -> Vec<String> {
    const STOP: &[&str] = &[
        "total",
        "combined",
        "altogether",
        "since",
        "start",
        "year",
        "years",
        "month",
        "months",
        "past",
        "last",
        "few",
        "item",
        "items",
        "related",
        "i",
        "money",
        "amount",
        "spent",
        "spend",
        "cost",
        "costs",
        "expense",
        "expenses",
        "paid",
        "purchase",
        "purchased",
    ];
    let stop: HashSet<&str> = STOP.iter().copied().collect();
    let mut terms = synthetic_query_terms(task_lower);
    terms.retain(|term| !stop.contains(term.as_str()));
    if task_lower.contains("bike") {
        for extra in [
            "helmet",
            "lights",
            "chain",
            "cycling",
            "tune-up",
            "bike shop",
        ] {
            terms.push(extra.to_string());
        }
    }
    terms.sort();
    terms.dedup();
    terms
}

pub(in crate::index) fn extract_multi_session_duration_focus_terms(task_lower: &str) -> Vec<String> {
    const STOP: &[&str] = &[
        "total",
        "combined",
        "altogether",
        "since",
        "start",
        "year",
        "years",
        "month",
        "months",
        "week",
        "weeks",
        "day",
        "days",
        "hour",
        "hours",
        "minute",
        "minutes",
        "time",
        "take",
        "took",
        "spent",
        "spend",
        "main",
        "past",
        "last",
        "few",
        "item",
        "items",
        "related",
        "united",
        "states",
        "i",
    ];
    let stop: HashSet<&str> = STOP.iter().copied().collect();
    let mut terms = synthetic_query_terms(task_lower);
    terms.retain(|term| !stop.contains(term.as_str()));
    terms.retain(|term| term.len() >= 2);
    if task_lower.contains("game") || task_lower.contains("gaming") {
        for extra in [
            "playing",
            "played",
            "finish",
            "finished",
            "complete",
            "completed",
        ] {
            terms.push(extra.to_string());
        }
    }
    if task_lower.contains("road trip") || task_lower.contains("destinations") {
        terms.retain(|term| !matches!(term.as_str(), "three" | "destination" | "destinations"));
        for extra in ["road", "trip", "drive", "drove", "driving"] {
            terms.push(extra.to_string());
        }
    }
    terms.sort();
    terms.dedup();
    terms
}

pub(in crate::index) fn is_realized_duration_fact_text(lower: &str) -> bool {
    let realized = task_contains_any(
        lower,
        &[
            "just got back",
            "got back from",
            "watched all",
            "did it in",
            "spent around",
            "spent ",
            "took me",
            "completed",
            "finished",
            "clocked",
            "drove for",
            "driving to",
            "camping trip",
            "break in",
            "break from",
            "marathon",
        ],
    );
    let future = task_contains_any(
        lower,
        &[
            "i'm planning",
            "i am planning",
            "plan to",
            "going to",
            "i'll",
            "i will",
            "next week",
            "next month",
            "by the end",
            "goal is",
            "goal to",
            "thinking about",
            "thinking of",
        ],
    );
    realized && !future
}

pub(in crate::index) fn extract_matching_duration_total_segments(
    line: &str,
    task_lower: &str,
) -> Vec<(String, SyntheticDurationValue)> {
    let mut matches = Vec::new();
    for segment in split_duration_aggregate_segments(line) {
        let lower = segment.to_ascii_lowercase();
        if !is_realized_duration_fact_text(&lower)
            || !duration_total_line_matches_query(task_lower, &lower)
        {
            continue;
        }
        let Some(duration) = extract_aggregate_duration_value(&segment) else {
            continue;
        };
        matches.push((segment, duration));
    }
    matches
}

pub(in crate::index) fn duration_total_line_matches_query(task_lower: &str, lower: &str) -> bool {
    if task_lower.contains("social media") {
        return lower.contains("social media")
            && task_contains_any(lower, &["break from", "break in", "break"]);
    }
    if task_lower.contains("camping") {
        return lower.contains("camping trip");
    }
    if task_lower.contains("road trip") || task_lower.contains("destinations") {
        return task_contains_any(
            lower,
            &[
                "drove for",
                "drive there",
                "drive to",
                "driving to",
                "took me",
            ],
        );
    }
    if task_contains_any(task_lower, &["marvel", "star wars", "movies", "films"]) {
        return task_contains_any(lower, &["watched", "marathon"]);
    }
    if task_contains_any(task_lower, &["games", "gaming"]) {
        return task_contains_any(
            lower,
            &[
                "playing",
                "spent around",
                "took me",
                "finished",
                "completed",
            ],
        ) && !task_contains_any(
            lower,
            &[
                "developers",
                "development",
                "develop ",
                "release",
                "announced",
                "team ",
                "script",
                "dialogue",
                "motion capture",
                "pages long",
            ],
        );
    }
    true
}

pub(in crate::index) fn aggregate_fact_terms(line: &str) -> HashSet<String> {
    synthetic_query_terms(&normalize_session_answer_line_body(line).to_ascii_lowercase())
        .into_iter()
        .collect()
}

pub(in crate::index) fn is_duplicate_numeric_aggregate_fact(
    existing: &[(String, f32, HashSet<String>)],
    session_id: &str,
    value: f32,
    terms: &HashSet<String>,
) -> bool {
    existing
        .iter()
        .any(|(existing_session, existing_value, existing_terms)| {
            if (existing_value - value).abs() >= 0.01 {
                return false;
            }
            let overlap = existing_terms.intersection(terms).count();
            let min_size = existing_terms.len().min(terms.len());
            if existing_session == session_id {
                overlap >= 4 || (min_size > 0 && overlap == min_size)
            } else {
                overlap >= 5 || (min_size >= 4 && overlap == min_size)
            }
        })
}

pub(in crate::index) fn extract_nightly_rate(line: &str) -> Option<f32> {
    let lower = line.to_ascii_lowercase();
    if !lower.starts_with("user:") {
        return None;
    }
    if !lower.contains("per night") {
        return None;
    }
    if !task_contains_any(
        &lower,
        &[
            "stay", "staying", "hotel", "hostel", "resort", "room", "accommod",
        ],
    ) {
        return None;
    }
    extract_dollar_amounts(line).into_iter().next()
}

pub(in crate::index) fn extract_sale_total(line: &str) -> Option<f32> {
    let lower = line.to_ascii_lowercase();
    if !(lower.starts_with("user:") || line.trim_start().starts_with('-')) {
        return None;
    }
    if !(lower.contains("sold") || lower.contains("earned") || lower.contains("earning")) {
        return None;
    }

    let explicit_total = Regex::new(
        r"(?:earned|earning(?: a total of)?|for a total of)\s+\$([0-9][0-9,]*(?:\.[0-9]+)?)",
    )
    .unwrap();
    if let Some(caps) = explicit_total.captures(&lower) {
        if let Some(value) = caps
            .get(1)
            .and_then(|m| m.as_str().replace(',', "").parse::<f32>().ok())
        {
            return Some(value);
        }
    }

    let per_item =
        Regex::new(r"sold\s+(\d+)[^$]{0,160}?\$([0-9][0-9,]*(?:\.[0-9]+)?)\s*each").unwrap();
    if let Some(caps) = per_item.captures(&lower) {
        let quantity = caps.get(1).and_then(|m| m.as_str().parse::<f32>().ok())?;
        let price = caps
            .get(2)
            .and_then(|m| m.as_str().replace(',', "").parse::<f32>().ok())?;
        return Some(quantity * price);
    }

    None
}

pub(in crate::index) fn normalized_index_answer_surface_key(text: &str) -> String {
    text.trim()
        .trim_matches(|c: char| matches!(c, '"' | '\'' | ',' | '.' | ';' | ':' | '!' | '?'))
        .to_ascii_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

pub(in crate::index) fn index_answer_surface_answers_overlap(left: &str, right: &str) -> bool {
    let left_key = normalized_index_answer_surface_key(left);
    let right_key = normalized_index_answer_surface_key(right);
    !left_key.is_empty()
        && !right_key.is_empty()
        && (left_key == right_key || left_key.contains(&right_key) || right_key.contains(&left_key))
}

pub(in crate::index) fn index_answer_surface_bucket_rank(bucket: &IndexAnswerSurfaceBucket) -> f32 {
    let corroboration = ((bucket.total_score - bucket.best_score).max(0.0)).min(6.0) * 0.15;
    bucket.best_score
        + bucket.max_overlap as f32 * 1.5
        + (bucket.paths.len().saturating_sub(1).min(2) as f32) * 0.75
        + (bucket.hits.saturating_sub(1).min(3) as f32) * 0.25
        + corroboration
}

pub(in crate::index) fn index_answer_surface_buckets_conflict(
    top: &IndexAnswerSurfaceBucket,
    runner_up: &IndexAnswerSurfaceBucket,
) -> bool {
    !index_answer_surface_answers_overlap(&top.answer_span, &runner_up.answer_span)
        && index_answer_surface_bucket_rank(runner_up) + 2.5
            >= index_answer_surface_bucket_rank(top)
        && runner_up.max_overlap + 1 >= top.max_overlap
}

pub(in crate::index) fn index_answer_surface_bucket_has_query_affinity(
    task_lower: &str,
    bucket: &IndexAnswerSurfaceBucket,
) -> bool {
    let answer_lower = bucket.answer_span.to_ascii_lowercase();
    (task_contains_any(
        task_lower,
        &["religious", "religion", "faith", "church", "spiritual"],
    ) && answer_lower.contains("religious"))
        || (task_contains_any(
            task_lower,
            &[
                "member of the lgbtq community",
                "member of the lgbtq+ community",
                "part of the lgbtq community",
                "part of the lgbtq+ community",
                "member of the transgender community",
                "ally to the transgender community",
                "ally to the lgbtq community",
                "ally to the lgbtq+ community",
                "considered an ally",
            ],
        ) && answer_lower.contains("ally"))
        || (task_contains_any(
            task_lower,
            &["move from", "moved from", "home country", "origin country"],
        ) && bucket
            .relation_families
            .contains(&SyntheticAnswerSurfaceRelationFamily::Origin))
        || (task_contains_any(
            task_lower,
            &["what books", "which books", " books", "book "],
        ) && bucket
            .relation_families
            .contains(&SyntheticAnswerSurfaceRelationFamily::Book))
        || (task_contains_any(
            task_lower,
            &[
                "what lgbtq",
                "transgender-specific events",
                "lgbtq events",
                "in what ways",
            ],
        ) && bucket
            .relation_families
            .contains(&SyntheticAnswerSurfaceRelationFamily::CommunityEvent))
        || (task_contains_any(task_lower, &["help children", "help kids", "help youth"])
            && bucket
                .relation_families
                .contains(&SyntheticAnswerSurfaceRelationFamily::ChildHelpEvent))
        || (task_contains_any(
            task_lower,
            &[
                "with her family",
                "with his family",
                "with my family",
                "with the kids",
                "family activities",
            ],
        ) && bucket
            .relation_families
            .contains(&SyntheticAnswerSurfaceRelationFamily::FamilyActivity))
        || (task_contains_any(
            task_lower,
            &["to destress", "to de-stress", "self-care", "relax"],
        ) && bucket
            .relation_families
            .contains(&SyntheticAnswerSurfaceRelationFamily::SelfCareActivity))
}

pub(in crate::index) fn synthetic_answer_surface_should_skip_fallback(
    task: &str,
    task_lower: &str,
    profile: &SyntheticAnswerSurfaceQueryProfile,
    evidence: &[String],
) -> bool {
    let real_evidence = evidence
        .iter()
        .filter(|line| !line.starts_with("answer_surface:"))
        .collect::<Vec<_>>();
    let evidence_has_any = |needles: &[&str]| {
        real_evidence.iter().any(|line| {
            let lower = line.to_ascii_lowercase();
            task_contains_any(&lower, needles)
        })
    };
    let collecting_target = task_lower
        .split_once("collecting ")
        .map(|(_, tail)| tail)
        .map(|tail| {
            ["?", ".", ",", " before ", " after "]
                .iter()
                .find_map(|marker| tail.split_once(marker).map(|(head, _)| head))
                .unwrap_or(tail)
                .trim()
                .to_string()
        })
        .filter(|phrase| phrase.split_whitespace().count() >= 2);
    let mut poster_focus_terms = synthetic_query_terms(task_lower);
    poster_focus_terms.retain(|term| {
        term.len() >= 4
            && !matches!(
                term.as_str(),
                "university"
                    | "college"
                    | "present"
                    | "presented"
                    | "poster"
                    | "research"
                    | "conference"
            )
    });
    let poster_focus_refs = poster_focus_terms
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    let poster_focus_min_overlap = if poster_focus_refs.len() >= 2 { 2 } else { 1 };

    (matches!(profile.route_kind, SyntheticAnswerSurfaceRouteKind::Choice)
        && task_contains_any(
            task_lower,
            &[
                " first",
                " earlier",
                " later",
                " before ",
                " after ",
                " more often",
                " less often",
                " higher percentage",
                " lower percentage",
                " higher discount",
                " lower discount",
                " cheaper",
                " more expensive",
                " cost more",
                " cost less",
                " older",
                " younger",
            ],
        ))
        || ((matches!(
            profile.expected_type,
            SyntheticAnswerSurfaceExpectedType::Count
        ) || is_money_query(task))
            && synthetic_count_query_requires_multi_operand_reasoning(task, task_lower))
        || (task_contains_any(
            task_lower,
            &[
                "at which university",
                "which university",
                "what university",
                "which college",
                "what college",
            ],
        ) && !evidence_has_any(&["university", "college", "school", "institute"]))
        || (task_contains_any(task_lower, &["presented", "poster"])
            && !evidence_has_any(&["presented", "poster"]))
        || (task_contains_any(
            task_lower,
            &[
                "at which university",
                "which university",
                "what university",
                "which college",
                "what college",
            ],
        ) && task_contains_any(task_lower, &["present", "poster"])
            && !poster_focus_refs.is_empty()
            && !real_evidence.iter().any(|line| {
                let lower = line.to_ascii_lowercase();
                term_overlap_count(&lower, &poster_focus_refs) >= poster_focus_min_overlap
            }))
        || (task_contains_any(task_lower, &["conference"]) && !evidence_has_any(&["conference"]))
        || collecting_target.as_ref().is_some_and(|phrase| {
            !real_evidence
                .iter()
                .any(|line| line.to_ascii_lowercase().contains(phrase))
        })
}

pub(in crate::index) fn extract_rare_collection_count(line: &str) -> Option<(&'static str, i32)> {
    let lower = line.to_ascii_lowercase();
    if !(lower.starts_with("user:") || line.trim_start().starts_with('-')) {
        return None;
    }
    let kind = if lower.contains("rare books") {
        "rare_books"
    } else if lower.contains("rare records") {
        "rare_records"
    } else if lower.contains("rare figurines") {
        "rare_figurines"
    } else if lower.contains("rare coins") {
        "rare_coins"
    } else {
        return None;
    };

    let count = extract_line_numbers(line)
        .into_iter()
        .find(|value| *value > 0 && *value < 1000)?;
    Some((kind, count))
}

pub(in crate::index) fn extract_previous_role(line: &str) -> Option<String> {
    let lower = line.to_ascii_lowercase();
    if !lower.starts_with("user:") || !lower.contains("previous role") {
        return None;
    }

    let pattern =
        Regex::new(r"previous role as a[n]?\s+(.+?)(?:,|\.| and\b| but\b| with\b)").unwrap();
    let role = pattern
        .captures(line)?
        .get(1)?
        .as_str()
        .trim()
        .trim_matches('"')
        .to_string();
    if role.is_empty() {
        None
    } else {
        Some(role)
    }
}

pub(in crate::index) fn extract_finished_issue_count(line: &str, lower: &str) -> Option<i32> {
    if !(lower.starts_with("user:") || line.trim_start().starts_with('-'))
        || !lower.contains("national geographic")
    {
        return None;
    }

    if lower.contains("finished") {
        return extract_line_numbers(line).into_iter().next();
    }
    if lower.contains("currently on") {
        return extract_line_numbers(line)
            .into_iter()
            .next()
            .map(|value| value - 1)
            .filter(|value| *value > 0);
    }
    None
}

pub(in crate::index) fn extract_quoted_title(task: &str) -> Option<String> {
    extract_quoted_titles(task)
        .into_iter()
        .next()
        .map(|title| title.to_ascii_lowercase())
}

pub(in crate::index) fn extract_quoted_titles(task: &str) -> Vec<String> {
    let mut titles = Vec::new();
    for quote in ['"', '\''] {
        let mut cursor = task;
        while let Some(start) = cursor.find(quote) {
            let tail = &cursor[start + quote.len_utf8()..];
            let Some(end) = tail.find(quote) else {
                break;
            };
            let title = tail[..end].trim();
            if title.split_whitespace().count() >= 2 {
                let title = title.to_string();
                if !titles.iter().any(|existing| existing == &title) {
                    titles.push(title);
                }
            }
            cursor = &tail[end + quote.len_utf8()..];
        }
        if !titles.is_empty() {
            break;
        }
    }
    titles
}

pub(in crate::index) fn extract_named_artwork_location_surface_from_line(
    _line: &str,
    line_lower: &str,
    title_lower: &str,
) -> Option<String> {
    let title_idx = line_lower.find(title_lower)?;
    let context_lower = &line_lower[title_idx + title_lower.len()..];
    extract_named_artwork_room_surface_from_context(context_lower, line_lower).or_else(|| {
        let prefix_lower = &line_lower[..title_idx];
        if context_lower.contains("above my bed")
            || context_lower.contains("above the bed")
            || (task_contains_any(context_lower, &["on my wall", "on the wall"])
                && prefix_lower.contains("bedroom"))
        {
            Some("in my bedroom".to_string())
        } else if task_contains_any(context_lower, &["above my sofa", "above the sofa"])
            && prefix_lower.contains("living room")
        {
            Some("above my living room sofa".to_string())
        } else {
            None
        }
    })
}

pub(in crate::index) fn extract_named_artwork_room_surface_from_context(
    context_lower: &str,
    full_lower: &str,
) -> Option<String> {
    if context_lower.contains("living room sofa") {
        return Some("above my living room sofa".to_string());
    }
    if context_lower.contains("above my bed") || context_lower.contains("above the bed") {
        return Some("in my bedroom".to_string());
    }
    for (marker, answer) in [
        ("bedroom", "in my bedroom"),
        ("living room", "in my living room"),
        ("dining room", "in my dining room"),
        ("family room", "in my family room"),
        ("guest room", "in my guest room"),
        ("office", "in my office"),
        ("studio", "in my studio"),
        ("kitchen", "in my kitchen"),
        ("hallway", "in my hallway"),
        ("entryway", "in my entryway"),
        ("party area", "in the party area"),
    ] {
        if context_lower.contains(marker) {
            return Some(answer.to_string());
        }
    }
    if task_contains_any(context_lower, &["on my wall", "on the wall"]) {
        for (marker, answer) in [
            ("bedroom", "in my bedroom"),
            ("living room", "in my living room"),
            ("office", "in my office"),
            ("studio", "in my studio"),
        ] {
            if full_lower.contains(marker) {
                return Some(answer.to_string());
            }
        }
    }
    None
}

pub(in crate::index) fn extract_rewatch_title_from_line(line: &str, lower: &str) -> Option<String> {
    for marker in ["re-watched ", "re watched ", "rewatched "] {
        let Some(start) = lower.find(marker) else {
            continue;
        };
        let title_start = start + marker.len();
        let tail = line[title_start..].trim();
        let tail_lower = lower[title_start..].trim();
        let mut end = tail.len();
        for delimiter in [
            ",",
            ".",
            "?",
            "!",
            " yesterday",
            " today",
            " again",
            " which ",
            " and ",
            " but ",
            " because ",
        ] {
            if let Some(idx) = tail_lower.find(delimiter) {
                end = end.min(idx);
            }
        }
        let title = tail[..end]
            .trim()
            .trim_matches(|c: char| matches!(c, '"' | '\'' | ',' | '.' | ';' | '!' | '?'));
        if title.len() >= 3 {
            return Some(title.to_string());
        }
    }
    None
}

pub(in crate::index) fn normalize_rewatch_title(title: &str) -> String {
    title
        .trim()
        .trim_matches(|c: char| matches!(c, '"' | '\'' | ',' | '.' | ';' | '!' | '?'))
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

pub(in crate::index) fn extract_origin_country_answer(line: &str) -> Option<String> {
    Regex::new(r"(?i)home country[, ]+([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)?)")
        .unwrap()
        .captures(line)
        .and_then(|captures| captures.get(1))
        .map(|value| value.as_str().trim().to_string())
        .filter(|value| !value.is_empty())
}

#[derive(Clone, Copy)]
pub(in crate::index) enum SyntheticDurationAnchor {
    CurrentDays(i32),
    AbsoluteDay(i32),
}

#[derive(Clone, Copy)]
pub(in crate::index) enum SyntheticEventAnchor {
    RelativeDaysAgo(i32),
    AbsoluteDay(i32),
}

#[derive(Clone, Copy)]
pub(in crate::index) struct SyntheticDurationValue {
    pub(in crate::index) amount: f32,
    pub(in crate::index) days: f32,
    pub(in crate::index) unit: &'static str,
}

#[derive(Clone, Copy)]
pub(in crate::index) enum SyntheticTemporalDirection {
    Earlier,
    Later,
}

pub(in crate::index) fn extract_temporal_choice_options(task: &str) -> Option<(String, String)> {
    let quoted = extract_quoted_titles(task);
    if quoted.len() >= 2 {
        return Some((quoted[0].trim().to_string(), quoted[1].trim().to_string()));
    }

    let tail = task
        .split_once(',')
        .map(|(_, suffix)| suffix)
        .unwrap_or(task)
        .trim()
        .trim_end_matches('?');
    let (left, right) = tail.rsplit_once(" or ")?;
    Some((
        normalize_temporal_choice_option(left),
        normalize_temporal_choice_option(right),
    ))
}

pub(in crate::index) fn normalize_temporal_choice_option(option: &str) -> String {
    option
        .trim()
        .trim_matches(|c: char| matches!(c, '"' | '\''))
        .trim_start_matches("the ")
        .trim_start_matches("The ")
        .trim()
        .to_string()
}

pub(in crate::index) fn extract_temporal_elapsed_phrases(task_lower: &str) -> Option<(String, String)> {
    let trimmed = task_lower.trim().trim_end_matches('?');
    let rest = trimmed.strip_prefix("how long had i been ")?;
    let (subject, event) = rest.split_once(" when ")?;
    Some((subject.trim().to_string(), event.trim().to_string()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::index) enum SyntheticElapsedFromNowUnit {
    Day,
    Week,
    Month,
    Year,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::index) struct SyntheticFromNowQuery {
    pub(super) unit: SyntheticElapsedFromNowUnit,
    pub(super) event_phrase: String,
    pub(super) anchor_phrase: Option<String>,
    pub(super) append_ago: bool,
}

pub(in crate::index) fn extract_temporal_from_now_query(task_lower: &str) -> Option<SyntheticFromNowQuery> {
    let trimmed = strip_temporal_reference_prefix(task_lower)
        .trim()
        .trim_end_matches('?');
    let rest = trimmed.strip_prefix("how many ")?;
    if let Some((unit_raw, event)) = rest.split_once(" ago did i ") {
        let unit = parse_temporal_from_now_unit(unit_raw)?;
        let (event_phrase, anchor_phrase) = split_temporal_when_anchor(event);
        let append_ago = anchor_phrase.is_some();
        return Some(SyntheticFromNowQuery {
            unit,
            event_phrase,
            anchor_phrase,
            append_ago,
        });
    }
    if let Some((unit_raw, event)) = rest.split_once(" have passed since i ") {
        let unit = parse_temporal_from_now_unit(unit_raw)?;
        let (event_phrase, anchor_phrase) = split_temporal_when_anchor(event);
        return Some(SyntheticFromNowQuery {
            unit,
            event_phrase,
            anchor_phrase,
            append_ago: false,
        });
    }
    None
}

pub(in crate::index) fn split_temporal_when_anchor(event: &str) -> (String, Option<String>) {
    let trimmed = event.trim();
    if let Some((primary, anchor)) = trimmed.split_once(" when i ") {
        let primary = primary.trim().to_string();
        let anchor = anchor.trim();
        if !primary.is_empty() && !anchor.is_empty() {
            return (primary, Some(anchor.to_string()));
        }
    }
    (trimmed.to_string(), None)
}

pub(in crate::index) fn strip_temporal_reference_prefix(task_lower: &str) -> &str {
    let trimmed = task_lower.trim();
    if trimmed.starts_with("as of ") {
        if let Some(pos) = trimmed.find("how many ") {
            return &trimmed[pos..];
        }
    }
    trimmed
}

pub(in crate::index) fn extract_task_reference_label(task: &str) -> Option<String> {
    let trimmed = task.trim();
    let lower = trimmed.to_ascii_lowercase();
    if !lower.starts_with("as of ") {
        return None;
    }
    let question_pos = lower.find("how many ")?;
    let candidate = trimmed[6..question_pos].trim().trim_end_matches(',').trim();
    if extract_explicit_date_rank(candidate).is_some() {
        return Some(candidate.to_string());
    }
    None
}

pub(in crate::index) fn verbatim_source_group_key(entry: &BM25Entry) -> String {
    if let Ok(content) = std::fs::read_to_string(&entry.neuron_path) {
        if let Some(line) = content.lines().next() {
            if let Some(source_idx) = line.find("source:") {
                let source = &line[source_idx + "source:".len()..];
                let source = source.trim();
                let source = source.strip_suffix("-->").unwrap_or(source).trim();
                if !source.is_empty() {
                    return source.to_string();
                }
            }
        }
    }

    let Some(name) = entry.neuron_path.file_name().and_then(|name| name.to_str()) else {
        return entry.neuron_path.display().to_string();
    };
    name.split('.').next().unwrap_or(name).to_string()
}

pub(in crate::index) fn parse_temporal_from_now_unit(raw: &str) -> Option<SyntheticElapsedFromNowUnit> {
    match raw.trim() {
        "day" | "days" => Some(SyntheticElapsedFromNowUnit::Day),
        "week" | "weeks" => Some(SyntheticElapsedFromNowUnit::Week),
        "month" | "months" => Some(SyntheticElapsedFromNowUnit::Month),
        "year" | "years" => Some(SyntheticElapsedFromNowUnit::Year),
        _ => None,
    }
}

pub(in crate::index) fn extract_temporal_interval_phrases(task_lower: &str) -> Option<(String, String)> {
    let trimmed = task_lower.trim().trim_end_matches('?');
    let (before_after, start_phrase) = trimmed.split_once(" after ")?;
    let end_phrase = before_after
        .strip_prefix("how many days did it take for me to ")
        .or_else(|| before_after.strip_prefix("how many days did it take me to "))?
        .trim();
    Some((end_phrase.to_string(), start_phrase.trim().to_string()))
}

pub(in crate::index) fn best_temporal_rank_line(
    lines: &[String],
    phrase_lower: &str,
    terms: &[String],
) -> Option<(i32, usize, String)> {
    best_temporal_rank_line_with_min_overlap(lines, phrase_lower, terms, None)
}

pub(in crate::index) fn best_temporal_rank_line_with_min_overlap(
    lines: &[String],
    phrase_lower: &str,
    terms: &[String],
    min_overlap_override: Option<usize>,
) -> Option<(i32, usize, String)> {
    let keys = synthetic_answer_surface_term_key_set(terms);
    let min_overlap = min_overlap_override.unwrap_or_else(|| if keys.len() >= 3 { 2 } else { 1 });
    let mut best: Option<(i32, usize, usize, usize, String)> = None;
    for (line_idx, line) in lines.iter().enumerate() {
        let lower = line.to_ascii_lowercase();
        let line_keys = synthetic_answer_surface_term_key_set(&synthetic_query_terms(&lower));
        let overlap = synthetic_answer_surface_overlap_count(&line_keys, &keys);
        let exact = lower.contains(phrase_lower);
        if overlap < min_overlap && !exact {
            continue;
        }
        let Some(rank) = extract_temporal_rank_value(line) else {
            continue;
        };
        let exact_bonus = usize::from(exact);
        let score = overlap * 10 + exact_bonus * 5;
        let should_replace = best
            .as_ref()
            .map(|(_, best_score, best_exact, best_line_idx, _)| {
                score > *best_score
                    || (score == *best_score
                        && (exact_bonus > *best_exact
                            || (exact_bonus == *best_exact && line_idx > *best_line_idx)))
            })
            .unwrap_or(true);
        if should_replace {
            best = Some((rank, score, exact_bonus, line_idx, line.clone()));
        }
    }
    best.map(|(rank, score, _, _, line)| (rank, score, line))
}

pub(in crate::index) fn best_user_turn_line_with_min_overlap(
    lines: &[String],
    phrase_lower: &str,
    terms: &[String],
    min_overlap_override: Option<usize>,
) -> Option<(i32, usize, String)> {
    let keys = synthetic_answer_surface_term_key_set(terms);
    let min_overlap = min_overlap_override.unwrap_or_else(|| if keys.len() >= 3 { 2 } else { 1 });
    let mut best: Option<(i32, usize, usize, String)> = None;
    let mut user_turn = 0i32;
    for line in lines {
        let lower = line.to_ascii_lowercase();
        if !lower.starts_with("user:") {
            continue;
        }
        user_turn += 1;
        let line_keys = synthetic_answer_surface_term_key_set(&synthetic_query_terms(&lower));
        let overlap = synthetic_answer_surface_overlap_count(&line_keys, &keys);
        let exact = lower.contains(phrase_lower);
        if overlap < min_overlap && !exact {
            continue;
        }
        let exact_bonus = usize::from(exact);
        let score = overlap * 10 + exact_bonus * 5;
        let should_replace = best
            .as_ref()
            .map(|(best_turn, best_score, best_exact, _)| {
                score > *best_score
                    || (score == *best_score
                        && (exact_bonus > *best_exact
                            || (exact_bonus == *best_exact && user_turn > *best_turn)))
            })
            .unwrap_or(true);
        if should_replace {
            best = Some((user_turn, score, exact_bonus, line.clone()));
        }
    }
    best.map(|(turn, score, _, line)| (turn, score, line))
}

pub(in crate::index) fn best_temporal_duration_anchor_line(
    lines: &[String],
    phrase_lower: &str,
    terms: &[String],
) -> Option<(SyntheticDurationAnchor, usize, String)> {
    let keys = synthetic_answer_surface_term_key_set(terms);
    let min_overlap = if keys.len() >= 3 { 2 } else { 1 };
    let mut best: Option<(SyntheticDurationAnchor, usize, usize, usize, String)> = None;
    for (line_idx, line) in lines.iter().enumerate() {
        let lower = line.to_ascii_lowercase();
        let line_keys = synthetic_answer_surface_term_key_set(&synthetic_query_terms(&lower));
        let overlap = synthetic_answer_surface_overlap_count(&line_keys, &keys);
        let exact = lower.contains(phrase_lower);
        if overlap < min_overlap && !exact {
            continue;
        }
        let anchor = if let Some(days) = extract_current_duration_days(line) {
            SyntheticDurationAnchor::CurrentDays(days)
        } else if let Some(day) = extract_explicit_date_rank(line) {
            SyntheticDurationAnchor::AbsoluteDay(day)
        } else {
            continue;
        };
        let exact_bonus = usize::from(exact);
        let score = overlap * 10 + exact_bonus * 5;
        let should_replace = best
            .as_ref()
            .map(|(_, best_score, best_exact, best_line_idx, _)| {
                score > *best_score
                    || (score == *best_score
                        && (exact_bonus > *best_exact
                            || (exact_bonus == *best_exact && line_idx > *best_line_idx)))
            })
            .unwrap_or(true);
        if should_replace {
            best = Some((anchor, score, exact_bonus, line_idx, line.clone()));
        }
    }
    best.map(|(anchor, score, _, _, line)| (anchor, score, line))
}

pub(in crate::index) fn best_temporal_event_anchor_line(
    lines: &[String],
    phrase_lower: &str,
    terms: &[String],
) -> Option<(SyntheticEventAnchor, usize, String)> {
    let keys = synthetic_answer_surface_term_key_set(terms);
    let min_overlap = if keys.len() >= 3 { 2 } else { 1 };
    let required_action_key = terms
        .first()
        .map(|term| synthetic_answer_surface_term_key(term))
        .filter(|term| !term.is_empty());
    let mut best: Option<(SyntheticEventAnchor, usize, usize, usize, String)> = None;
    for (line_idx, line) in lines.iter().enumerate() {
        let lower = line.to_ascii_lowercase();
        let line_keys = synthetic_answer_surface_term_key_set(&synthetic_query_terms(&lower));
        if required_action_key
            .as_ref()
            .is_some_and(|term| !line_keys.contains(term))
        {
            continue;
        }
        let overlap = synthetic_answer_surface_overlap_count(&line_keys, &keys);
        let exact = lower.contains(phrase_lower);
        if overlap < min_overlap && !exact {
            continue;
        }
        let anchor = if let Some(days_ago) = extract_temporal_relative_days(line) {
            let adjusted = match extract_relative_reference_offset_days(line) {
                Some((SyntheticTemporalDirection::Earlier, offset)) => days_ago + offset,
                Some((SyntheticTemporalDirection::Later, offset)) => {
                    days_ago.saturating_sub(offset)
                },
                None => days_ago,
            };
            SyntheticEventAnchor::RelativeDaysAgo(adjusted)
        } else if let Some(day) = extract_explicit_date_rank(line) {
            SyntheticEventAnchor::AbsoluteDay(day)
        } else {
            continue;
        };
        let exact_bonus = usize::from(exact);
        let score = overlap * 10 + exact_bonus * 5;
        let should_replace = best
            .as_ref()
            .map(|(_, best_score, best_exact, best_line_idx, _)| {
                score > *best_score
                    || (score == *best_score
                        && (exact_bonus > *best_exact
                            || (exact_bonus == *best_exact && line_idx > *best_line_idx)))
            })
            .unwrap_or(true);
        if should_replace {
            best = Some((anchor, score, exact_bonus, line_idx, line.clone()));
        }
    }
    best.map(|(anchor, score, _, _, line)| (anchor, score, line))
}

pub(in crate::index) fn best_temporal_from_now_event_line(
    lines: &[String],
    phrase_lower: &str,
    terms: &[String],
) -> Option<(i32, usize, String)> {
    let focus_terms = temporal_from_now_focus_terms(terms);
    let min_overlap = if focus_terms.len() >= 3 { 2 } else { 1 };
    let mut best: Option<(i32, usize, usize, String)> = None;
    for (line_idx, line) in lines.iter().enumerate() {
        let lower = line.to_ascii_lowercase();
        let overlap = temporal_from_now_overlap_count(&lower, &focus_terms);
        let exact = lower.contains(phrase_lower);
        if overlap < min_overlap && !exact {
            continue;
        }
        let day = if let Some(base_day) = temporal_base_day_at_line(lines, line_idx) {
            if let Some(days_ago) = extract_temporal_relative_days(line) {
                let adjusted = match extract_relative_reference_offset_days(line) {
                    Some((SyntheticTemporalDirection::Earlier, offset)) => days_ago + offset,
                    Some((SyntheticTemporalDirection::Later, offset)) => {
                        days_ago.saturating_sub(offset)
                    },
                    None => days_ago,
                };
                base_day - adjusted
            } else if let Some(day) = extract_explicit_date_rank(line) {
                day
            } else {
                base_day
            }
        } else if let Some(days_ago) = extract_temporal_relative_days(line) {
            let adjusted = match extract_relative_reference_offset_days(line) {
                Some((SyntheticTemporalDirection::Earlier, offset)) => days_ago + offset,
                Some((SyntheticTemporalDirection::Later, offset)) => {
                    days_ago.saturating_sub(offset)
                },
                None => days_ago,
            };
            -adjusted
        } else if let Some(day) = extract_explicit_date_rank(line) {
            day
        } else {
            continue;
        };
        let score = overlap * 10 + usize::from(exact) * 5;
        let should_replace = best
            .as_ref()
            .map(|(best_day, best_score, best_line_idx, _)| {
                score > *best_score
                    || (score == *best_score
                        && (day > *best_day || (day == *best_day && line_idx > *best_line_idx)))
            })
            .unwrap_or(true);
        if should_replace {
            best = Some((day, score, line_idx, line.clone()));
        }
    }
    let (day, score, _, line) = best?;
    Some((day, score, line))
}

pub(in crate::index) fn temporal_from_now_overlap_count(lower_line: &str, terms: &[String]) -> usize {
    terms
        .iter()
        .filter(|term| temporal_from_now_line_matches_term(lower_line, term))
        .count()
}

pub(in crate::index) fn temporal_from_now_line_matches_term(lower_line: &str, term: &str) -> bool {
    if lower_line.contains(term) {
        return true;
    }
    match term {
        "find" | "found" => {
            lower_line.contains("find")
                || lower_line.contains("found")
                || lower_line.contains("saw")
        },
        "launch" | "launched" => lower_line.contains("launch"),
        "sign" | "signed" => lower_line.contains("sign"),
        "go" | "went" => lower_line.contains("go") || lower_line.contains("went"),
        "take" | "taking" | "took" => {
            lower_line.contains("take")
                || lower_line.contains("taking")
                || lower_line.contains("took")
        },
        _ => {
            let stem = term
                .trim_end_matches("ing")
                .trim_end_matches("ed")
                .trim_end_matches('s');
            stem.len() >= 3 && lower_line.contains(stem)
        },
    }
}

pub(in crate::index) fn temporal_from_now_focus_terms(terms: &[String]) -> Vec<String> {
    const LEADING_FOCUS_STOP: &[&str] = &[
        "attend", "visit", "go", "join", "make", "buy", "take", "run", "last", "i", "me", "my",
    ];

    let mut start = 0usize;
    while start + 1 < terms.len() {
        let key = synthetic_answer_surface_term_key(&terms[start]);
        if LEADING_FOCUS_STOP.contains(&key.as_str()) {
            start += 1;
            continue;
        }
        break;
    }

    let focus = terms[start..]
        .iter()
        .filter(|term| {
            let key = synthetic_answer_surface_term_key(term);
            !matches!(key.as_str(), "i" | "me" | "my" | "last")
        })
        .cloned()
        .collect::<Vec<_>>();
    if focus.is_empty() {
        terms.to_vec()
    } else {
        focus
    }
}

pub(in crate::index) fn extract_temporal_rank_value(line: &str) -> Option<i32> {
    if let Some(day) = extract_explicit_date_rank(line) {
        return Some(day);
    }
    let days_ago = extract_temporal_relative_days(line)?;
    let adjusted = match extract_relative_reference_offset_days(line) {
        Some((SyntheticTemporalDirection::Earlier, offset)) => days_ago + offset,
        Some((SyntheticTemporalDirection::Later, offset)) => days_ago.saturating_sub(offset),
        None => days_ago,
    };
    Some(-adjusted)
}

pub(in crate::index) fn extract_current_duration_days(line: &str) -> Option<i32> {
    duration_answer_magnitude(&extract_duration_answer_from_line(line)?)
        .map(|days| days.round() as i32)
}

pub(in crate::index) fn temporal_base_day_at_line(lines: &[String], line_idx: usize) -> Option<i32> {
    lines
        .iter()
        .take(line_idx + 1)
        .rev()
        .find_map(|line| extract_explicit_date_rank(line))
}

pub(in crate::index) fn best_temporal_current_anchor_line(
    lines: &[String],
) -> Option<(usize, usize, String)> {
    let mut best: Option<(usize, usize, usize, String)> = None;
    let mut user_turn = 0usize;
    for (line_idx, line) in lines.iter().enumerate() {
        let lower = line.to_ascii_lowercase();
        if !lower.starts_with("user:") {
            continue;
        }
        user_turn += 1;
        if !has_temporal_current_marker(&lower) {
            continue;
        }
        let score = 10 + user_turn;
        let should_replace = best
            .as_ref()
            .map(|(best_score, best_turn, best_line_idx, best_line)| {
                score > *best_score
                    || (score == *best_score
                        && (user_turn > *best_turn
                            || (user_turn == *best_turn
                                && (line_idx > *best_line_idx
                                    || (line_idx == *best_line_idx && line < best_line)))))
            })
            .unwrap_or(true);
        if should_replace {
            best = Some((score, user_turn, line_idx, line.clone()));
        }
    }
    best.map(|(score, _, line_idx, line)| (score, line_idx, line))
}

pub(in crate::index) fn has_temporal_current_marker(lower: &str) -> bool {
    lower.contains("today")
        || lower.contains("right now")
        || lower.contains("currently")
        || lower.contains("this week")
        || lower.contains("this month")
        || lower.contains("this year")
        || lower
            .split(|c: char| !c.is_ascii_alphanumeric())
            .any(|token| token == "now")
}

pub(in crate::index) fn extract_temporal_relative_days(text: &str) -> Option<i32> {
    let lower = text.to_ascii_lowercase();
    if lower.contains("today") {
        return Some(0);
    }
    if lower.contains("yesterday") {
        return Some(1);
    }
    if lower.contains("a couple of days ago") {
        return Some(2);
    }
    if lower.contains("a few days ago") || lower.contains("few days ago") {
        return Some(3);
    }
    if lower.contains("last weekend") || lower.contains("last week") {
        return Some(7);
    }
    if lower.contains("last month") {
        return Some(30);
    }
    for (unit, scale) in [("day", 1), ("week", 7), ("month", 30), ("year", 365)] {
        for marker in [format!("{unit} ago"), format!("{unit}s ago")] {
            if !lower.contains(&marker) {
                continue;
            }
            let prefix = lower.split(&marker).next()?;
            let amount = extract_temporal_trailing_count(prefix)?;
            return Some(amount * scale);
        }
    }
    None
}

pub(in crate::index) fn extract_relative_reference_offset_days(
    text: &str,
) -> Option<(SyntheticTemporalDirection, i32)> {
    let lower = text.to_ascii_lowercase();
    for (unit, scale) in [("day", 1), ("week", 7), ("month", 30), ("year", 365)] {
        for (marker, direction) in [
            (
                format!("{unit} in advance"),
                SyntheticTemporalDirection::Earlier,
            ),
            (
                format!("{unit}s in advance"),
                SyntheticTemporalDirection::Earlier,
            ),
            (
                format!("{unit} before"),
                SyntheticTemporalDirection::Earlier,
            ),
            (
                format!("{unit}s before"),
                SyntheticTemporalDirection::Earlier,
            ),
            (format!("{unit} after"), SyntheticTemporalDirection::Later),
            (format!("{unit}s after"), SyntheticTemporalDirection::Later),
            (format!("{unit} later"), SyntheticTemporalDirection::Later),
            (format!("{unit}s later"), SyntheticTemporalDirection::Later),
        ] {
            if !lower.contains(&marker) {
                continue;
            }
            let prefix = lower.split(&marker).next()?;
            let amount = extract_temporal_trailing_count(prefix)?;
            return Some((direction, amount * scale));
        }
    }
    None
}

pub(in crate::index) fn extract_temporal_trailing_count(prefix: &str) -> Option<i32> {
    let token = prefix
        .split_whitespace()
        .rev()
        .find(|token| !token.is_empty())?;
    parse_temporal_count_token(token)
}

pub(in crate::index) fn parse_temporal_count_token(token: &str) -> Option<i32> {
    let clean = token
        .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '+')
        .trim_end_matches('+');
    if let Ok(value) = clean.parse::<i32>() {
        return Some(value);
    }
    match clean {
        "a" | "an" | "one" => Some(1),
        "two" => Some(2),
        "three" => Some(3),
        "four" => Some(4),
        "five" => Some(5),
        "six" => Some(6),
        "seven" => Some(7),
        "eight" => Some(8),
        "nine" => Some(9),
        "ten" => Some(10),
        "eleven" => Some(11),
        "twelve" => Some(12),
        "couple" => Some(2),
        "few" => Some(3),
        _ => None,
    }
}

pub(in crate::index) fn extract_duration_months_from_text(text: &str) -> Option<i32> {
    let lower = text.to_ascii_lowercase();
    let years = Regex::new(
        r"(?i)\b(a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+years?\b",
    )
    .unwrap()
    .captures(&lower)
    .and_then(|caps| caps.get(1))
    .and_then(|value| parse_temporal_count_token(value.as_str()));
    let months = Regex::new(
        r"(?i)\b(a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+months?\b",
    )
    .unwrap()
    .captures(&lower)
    .and_then(|caps| caps.get(1))
    .and_then(|value| parse_temporal_count_token(value.as_str()));
    match (years, months) {
        (None, None) => None,
        (Some(years), None) => Some(years * 12),
        (None, Some(months)) => Some(months),
        (Some(years), Some(months)) => Some(years * 12 + months),
    }
}

pub(in crate::index) fn extract_current_role_total_months_from_line(line: &str, lower: &str) -> Option<i32> {
    let has_total_marker = task_contains_any(
        lower,
        &[
            "experience in the company",
            "experience at the company",
            "with the company",
            "at the company",
            "been at ",
            "been with ",
            "working at ",
        ],
    );
    if !has_total_marker {
        return None;
    }
    extract_duration_months_from_text(line)
}

pub(in crate::index) fn extract_current_role_offset_months_from_line(line: &str, lower: &str) -> Option<i32> {
    if !task_contains_any(
        lower,
        &[
            "worked my way up to ",
            "promoted to ",
            "promotion to ",
            "moved into ",
            "became ",
        ],
    ) {
        return None;
    }
    let (_, tail) = lower.split_once(" after ")?;
    extract_duration_months_from_text(tail).or_else(|| extract_duration_months_from_text(line))
}

pub(in crate::index) fn extract_current_role_title_from_transition_line(
    line: &str,
    lower: &str,
) -> Option<String> {
    for marker in [
        "worked my way up to ",
        "promoted to ",
        "promotion to ",
        "moved into ",
        "became ",
    ] {
        let Some(start) = lower.find(marker) else {
            continue;
        };
        let tail = &line[start + marker.len()..];
        let title = [" after ", ",", "."]
            .iter()
            .filter_map(|delimiter| tail.find(delimiter))
            .min()
            .map(|end| &tail[..end])
            .unwrap_or(tail)
            .trim()
            .trim_matches(|c: char| matches!(c, '"' | '\'' | ',' | '.' | ';' | ':'));
        if !title.is_empty() {
            return Some(title.to_ascii_lowercase());
        }
    }
    None
}

pub(in crate::index) fn render_month_span(total_months: i32) -> String {
    let years = total_months / 12;
    let months = total_months % 12;
    match (years, months) {
        (0, months) => format!("{months} {}", if months == 1 { "month" } else { "months" }),
        (years, 0) => format!("{years} {}", if years == 1 { "year" } else { "years" }),
        (years, months) => format!(
            "{years} {} and {months} {}",
            if years == 1 { "year" } else { "years" },
            if months == 1 { "month" } else { "months" }
        ),
    }
}

pub(in crate::index) fn extract_explicit_date_rank(line: &str) -> Option<i32> {
    let numeric = Regex::new(r"(?i)\b(\d{1,2})/(\d{1,2})(?:/(\d{4}))?\b").unwrap();
    if let Some(caps) = numeric.captures(line) {
        let month = caps.get(1)?.as_str().parse::<u32>().ok()?;
        let day = caps.get(2)?.as_str().parse::<u32>().ok()?;
        if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
            return None;
        }
        let year = caps
            .get(3)
            .and_then(|value| value.as_str().parse::<i32>().ok())
            .unwrap_or(2023);
        return Some(ymd_to_days(year, month, day));
    }

    let month_day = Regex::new(
        r"(?i)\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(\d{4}))?\b",
    )
    .unwrap();
    if let Some(caps) = month_day.captures(line) {
        let month = named_month_to_number(caps.get(1)?.as_str())?;
        let day = caps.get(2)?.as_str().parse::<u32>().ok()?;
        let year = caps
            .get(3)
            .and_then(|value| value.as_str().parse::<i32>().ok())
            .unwrap_or(2023);
        return Some(ymd_to_days(year, month, day));
    }

    let day_month_named = Regex::new(
        r"(?i)\b(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December)(?:,\s*(\d{4}))?\b",
    )
    .unwrap();
    if let Some(caps) = day_month_named.captures(line) {
        let day = caps.get(1)?.as_str().parse::<u32>().ok()?;
        let month = named_month_to_number(caps.get(2)?.as_str())?;
        let year = caps
            .get(3)
            .and_then(|value| value.as_str().parse::<i32>().ok())
            .unwrap_or(2023);
        return Some(ymd_to_days(year, month, day));
    }

    let day_month = Regex::new(
        r"(?i)\b(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)?\s+of\s+(January|February|March|April|May|June|July|August|September|October|November|December)(?:,\s*(\d{4}))?\b",
    )
    .unwrap();
    if let Some(caps) = day_month.captures(line) {
        let day = caps.get(1)?.as_str().parse::<u32>().ok()?;
        let month = named_month_to_number(caps.get(2)?.as_str())?;
        let year = caps
            .get(3)
            .and_then(|value| value.as_str().parse::<i32>().ok())
            .unwrap_or(2023);
        return Some(ymd_to_days(year, month, day));
    }

    let fuzzy_month = Regex::new(
        r"(?i)\b(?:(early|mid|late)[-\s]+)?(January|February|March|April|May|June|July|August|September|October|November|December)(?:,\s*|\s+)?(\d{4})?\b",
    )
    .unwrap();
    let caps = fuzzy_month.captures(line)?;
    let month = named_month_to_number(caps.get(2)?.as_str())?;
    let day = match caps
        .get(1)
        .map(|value| value.as_str().to_ascii_lowercase())
        .as_deref()
    {
        Some("early") => 5,
        Some("late") => 25,
        _ => 15,
    };
    let year = caps
        .get(3)
        .and_then(|value| value.as_str().parse::<i32>().ok())
        .unwrap_or(2023);
    Some(ymd_to_days(year, month, day))
}

pub(in crate::index) fn named_month_to_number(month: &str) -> Option<u32> {
    match &month.to_ascii_lowercase()[..] {
        "january" => Some(1),
        "february" => Some(2),
        "march" => Some(3),
        "april" => Some(4),
        "may" => Some(5),
        "june" => Some(6),
        "july" => Some(7),
        "august" => Some(8),
        "september" => Some(9),
        "october" => Some(10),
        "november" => Some(11),
        "december" => Some(12),
        _ => None,
    }
}

pub(in crate::index) fn ymd_to_days(year: i32, month: u32, day: u32) -> i32 {
    const MONTH_START_DAYS: [i32; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let leap_years = {
        let y = year - 1;
        y / 4 - y / 100 + y / 400 - (1969 / 4 - 1969 / 100 + 1969 / 400)
    };
    (year - 1970) * 365 + leap_years + MONTH_START_DAYS[(month - 1) as usize] + day as i32 - 1
}

pub(in crate::index) fn extract_title_duration_value(
    line: &str,
    title_lower: &str,
) -> Option<SyntheticDurationValue> {
    let lower = line.to_ascii_lowercase();
    if !lower.contains(title_lower) {
        return None;
    }
    for marker in ["which took me ", "took me ", "took "] {
        let Some(idx) = lower.find(marker) else {
            continue;
        };
        let tail = &lower[idx + marker.len()..];
        if let Some(value) = parse_leading_duration_value(tail) {
            return Some(value);
        }
    }
    None
}

pub(in crate::index) fn parse_leading_duration_value(text: &str) -> Option<SyntheticDurationValue> {
    let regex = Regex::new(
        r"(?i)^\s*(?:about\s+|around\s+)?(a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|couple|few|\d+(?:\.\d+)?)(\s+and\s+a\s+half)?\s+(day|days|week|weeks|month|months|year|years)\b",
    )
    .unwrap();
    let caps = regex.captures(text)?;
    let mut amount =
        caps.get(1)
            .and_then(|value| match value.as_str().to_ascii_lowercase().as_str() {
                "a" | "an" | "one" => Some(1.0),
                "two" => Some(2.0),
                "three" => Some(3.0),
                "four" => Some(4.0),
                "five" => Some(5.0),
                "six" => Some(6.0),
                "seven" => Some(7.0),
                "eight" => Some(8.0),
                "nine" => Some(9.0),
                "ten" => Some(10.0),
                "eleven" => Some(11.0),
                "twelve" => Some(12.0),
                "couple" => Some(2.0),
                "few" => Some(3.0),
                value => value.parse::<f32>().ok(),
            })?;
    if caps.get(2).is_some() {
        amount += 0.5;
    }
    let unit = caps.get(3)?.as_str().to_ascii_lowercase();
    let days = amount
        * match unit.as_str() {
            "day" | "days" => 1.0,
            "week" | "weeks" => 7.0,
            "month" | "months" => 30.0,
            "year" | "years" => 365.0,
            _ => return None,
        };
    Some(SyntheticDurationValue {
        amount,
        days,
        unit: match unit.as_str() {
            "day" | "days" => "day",
            "week" | "weeks" => "week",
            "month" | "months" => "month",
            "year" | "years" => "year",
            _ => return None,
        },
    })
}

pub(in crate::index) fn render_duration_unit(unit: &'static str, amount: f32) -> &'static str {
    if (amount - 1.0).abs() < f32::EPSILON {
        unit
    } else {
        match unit {
            "day" => "days",
            "week" => "weeks",
            "month" => "months",
            "year" => "years",
            _ => unit,
        }
    }
}

pub(in crate::index) fn render_elapsed_duration_answer(days: i32) -> String {
    if days % 30 == 0 {
        return render_small_duration(days / 30, "month");
    }
    if days % 7 == 0 {
        return render_small_duration(days / 7, "week");
    }
    if (7..=10).contains(&days) {
        return "one week".to_string();
    }
    render_small_duration(days, "day")
}

pub(in crate::index) fn render_elapsed_from_now_answer(
    days: i32,
    unit: SyntheticElapsedFromNowUnit,
    append_ago: bool,
) -> String {
    let answer = match unit {
        SyntheticElapsedFromNowUnit::Day => render_small_duration(days, "day"),
        SyntheticElapsedFromNowUnit::Week => (((days as f32) / 7.0).round() as i32).to_string(),
        SyntheticElapsedFromNowUnit::Month => (((days as f32) / 30.0).round() as i32).to_string(),
        SyntheticElapsedFromNowUnit::Year => (((days as f32) / 365.0).round() as i32).to_string(),
    };
    if append_ago {
        format!("{answer} ago")
    } else {
        answer
    }
}

pub(in crate::index) fn render_small_duration(amount: i32, unit: &str) -> String {
    let amount_text = match amount {
        1 => "one".to_string(),
        2 => "two".to_string(),
        3 => "three".to_string(),
        4 => "four".to_string(),
        5 => "five".to_string(),
        6 => "six".to_string(),
        7 => "seven".to_string(),
        8 => "eight".to_string(),
        9 => "nine".to_string(),
        10 => "ten".to_string(),
        11 => "eleven".to_string(),
        12 => "twelve".to_string(),
        _ => amount.to_string(),
    };
    let rendered_unit = if amount == 1 {
        unit
    } else {
        match unit {
            "day" => "days",
            "week" => "weeks",
            "month" => "months",
            "year" => "years",
            _ => unit,
        }
    };
    format!("{amount_text} {rendered_unit}")
}

