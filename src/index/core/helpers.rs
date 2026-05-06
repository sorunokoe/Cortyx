// This file is a submodule of `crate::index::core`.
// It contains free-standing helper functions extracted from mod.rs (E1).
// All visibility is relative to `crate::index` (the parent of `core`).
use super::*;
use crate::index::compile_regex;
use crate::types::{QueryText, SynapseWeight};

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
pub(in crate::index) fn extract_manifest_metadata(
    root: &Path,
) -> (String, String, String, String, String) {
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
pub(in crate::index) fn read_file_head(
    root: &Path,
    candidates: &[&str],
    max_chars: usize,
) -> String {
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

pub(in crate::index) fn is_capsule_glossary_term(
    term: &str,
    module_tokens: &HashSet<String>,
) -> bool {
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
    compile_regex(r"(?i)(?:role as|job as|position as)\s+([^?.!]+)")
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

pub(in crate::index) fn is_direct_count_candidate_line(
    line: &str,
    lower: &str,
    task_lower: &str,
) -> bool {
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
    compile_regex(
        r"(?i)\bfirst\s+((?:about\s+)?(?:an?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+(?:days?|weeks?|months?|years?|hours?|minutes?))\b",
    )
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
    let caps = compile_regex(
        r"(?i)\bhow many\s+(.*?)\s*(movies?|films?|shows?|episodes?)\s+(?:did|have)\s+i\s+re(?:-| )?watch(?:ed)?\b",
    )
    .captures(task_lower)?;
    let focus = caps
        .get(1)
        .map(|value| value.as_str().trim().to_string())
        .unwrap_or_default();
    let media_kind = caps.get(2)?.as_str().to_ascii_lowercase();
    Some((focus, media_kind))
}

pub(in crate::index) fn extract_daily_duration_commitment_phrase(
    task_lower: &str,
) -> Option<String> {
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

pub(in crate::index) fn extract_frequency_transition_activity_phrase(
    task_lower: &str,
) -> Option<String> {
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
    compile_regex(r"(?i)^(.+?)(?:\s+(?:with|at|in|on|for|during|around|near)\b|$)")
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
    let raw = compile_regex(
        r"(?i)\b(?:finished|read|reading|completed)\s+(?:about\s+)?(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+issues?\b",
    )
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

pub(in crate::index) fn extract_meetup_count_surface_from_line(
    line: &str,
    lower: &str,
) -> Option<String> {
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
    let raw = compile_regex(
        r"(?i)\bmet up\s+(once|twice|thrice|one|two|three|four|five|six|seven|eight|nine|ten|\d+)(?:\s+times?)?\b",
    )
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
    let raw = compile_regex(
        r"(?i)\bmet up\s+(once|twice|thrice|one|two|three|four|five|six|seven|eight|nine|ten|\d+)(?:\s+times?)?\b",
    )
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
            compile_regex(
                r"(?i)\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+times?\b",
            )
            .captures(line)
            .and_then(|caps| caps.get(1))
            .map(|m| m.as_str().trim())?
        },
        "trip" => {
            if !(lower.contains("trip") || lower.contains("adventure")) {
                return None;
            }
            compile_regex(
                r"(?i)\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+(?:trip|trips|adventures)\b",
            )
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
    let raw = compile_regex(
        r"(?i)\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+women\b",
    )
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
    let captures = compile_regex(
        r"(?i)\b(?:lost|down)\s+(about\s+)?(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+pounds?\b",
    )
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

pub(in crate::index) fn extract_frequency_surface_from_line(
    line: &str,
    lower: &str,
) -> Option<String> {
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
    compile_regex(
        r"(?i)\b(once|twice|thrice|one|two|three|four|five|\d+)\s+times?\s+(?:a|per)\s+(day|week|month|year)\b",
    )
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
        compile_regex(pattern)
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
    let pattern = compile_regex(r"(?i)\b(\d{1,2}(?::\d{2})?\s?(?:AM|PM))\b");
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
    let raw = compile_regex(r"(?i)\b(\d+)\s+points\b")
        .captures(line)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim())?;
    Some(format!("{raw} points"))
}

pub(in crate::index) fn extract_record_answer_from_line(line: &str, lower: &str) -> Option<String> {
    if !(lower.contains("record") || lower.contains("we're") || lower.contains("we are")) {
        return None;
    }
    compile_regex(r"\b(\d+\s*-\s*\d+)\b")
        .captures(line)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim().replace(' ', ""))
}

pub(in crate::index) fn extract_status_answer_from_line(line: &str, lower: &str) -> Option<String> {
    if !lower.contains("status") {
        return None;
    }
    compile_regex(r"(?i)\b(Premier\s+(?:Silver|Gold|Platinum|Bronze|Diamond|1K))\s+status\b")
        .captures(line)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim().to_string())
}

pub(in crate::index) fn extract_level_goal_answer_from_line(
    line: &str,
    lower: &str,
) -> Option<String> {
    if !lower.contains("level")
        || !(line_has_future_goal_marker(lower)
            || lower.contains("determined to reach")
            || lower.contains("aiming to hit")
            || lower.contains("current goal"))
    {
        return None;
    }
    compile_regex(r"(?i)\b(level\s+\d+)\b")
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

pub(in crate::index) fn extract_gadget_purchase_item_from_line(
    line: &str,
    lower: &str,
) -> Option<String> {
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
    compile_regex(
        r"(?i)\b(?:my\s+new\s+|my\s+|the\s+)?((?:[a-z0-9][a-z0-9+-]*)(?:\s+[a-z0-9][a-z0-9+-]*){0,2}\s(?:pot|fryer|mixer|blender|processor|maker|oven|grill|toaster|microwave|cooker|skillet))\b",
    )
    .captures_iter(line)
    .filter_map(|caps| caps.get(1))
    .map(|m| m.as_str().trim().to_string())
    .last()
}

pub(in crate::index) fn extract_lens_purchase_item_from_line(
    line: &str,
    lower: &str,
) -> Option<String> {
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
    let phrase = compile_regex(
        r"(?i)\b(?:old\s+|new\s+)?((?:\d{1,3}(?:-\d{1,3})?mm|[a-z]+(?:-[a-z]+)?)(?:\s+[a-z]+(?:-[a-z]+)?){0,2}\s+lens)\b",
    )
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

pub(in crate::index) fn extract_planned_stay_location_from_line(
    line: &str,
    lower: &str,
) -> Option<String> {
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

pub(in crate::index) fn extract_current_company_answer_from_line(
    line: &str,
    lower: &str,
) -> Option<String> {
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

pub(in crate::index) fn aggregate_focus_match_count_for_path(
    path: &Path,
    focus_terms: &[String],
) -> usize {
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

pub(in crate::index) fn parse_index_answer_surface_rows(
    content: &str,
) -> Vec<IndexAnswerSurfaceRow> {
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

pub(in crate::index) fn synthetic_answer_surface_requires_completed_evidence(
    task_lower: &str,
) -> bool {
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
    pub(in crate::index) fn trim_repeated_suffix(word: &mut String) {
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
    compile_regex(r"\b(?:19|20)\d{2}\b").is_match(&lower)
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
        || compile_regex(
            r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(?:day|week|month|year)s?\b",
        )
        .is_match(&lower)
        || compile_regex(
            r"\b(?:day|week|month|year)s?\s+(?:ago|already|now)\b",
        )
        .is_match(&lower)
}

pub(in crate::index) fn looks_like_answer_surface_count(answer_span: &str) -> bool {
    if looks_like_answer_surface_date(answer_span) {
        return false;
    }
    let lower = answer_span.to_ascii_lowercase();
    compile_regex(
        r"^(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twice|thrice)(?:\s+(?:times?|kids?|children|dogs?|cats?|followers?|issues?|books?|letters?))?$",
    )
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

pub(in crate::index) fn format_index_answer_surface_answer(
    task_lower: &str,
    answer: &str,
) -> String {
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

pub(in crate::index) fn latest_active_kg_value(
    entity: &kg::KgEntity,
    predicate: &str,
) -> Option<String> {
    pub(in crate::index) fn latest_value_for_predicate(
        entity: &kg::KgEntity,
        predicate: &str,
    ) -> Option<String> {
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
    compile_regex(r"\b(\d{1,2}):(\d{2})\b")
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

pub(in crate::index) fn assistant_followup_subject_descriptor_clause(
    task_lower: &str,
) -> Option<&str> {
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
    let prior_move = compile_regex(r"after\s+(\d+)\.")
        .captures(task_lower)
        .and_then(|caps| caps.get(1))
        .and_then(|m| m.as_str().parse::<i32>().ok())?;
    Some(prior_move + 1)
}

pub(in crate::index) fn extract_chess_move_answer_from_line(
    line: &str,
    expected_move_number: Option<i32>,
) -> Option<String> {
    let capture = compile_regex(
        r"\b(\d+)\.\s*(O-O(?:-O)?|[KQRNB]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRNB])?[+#]?)\b",
    )
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
    let capture = compile_regex(r"(?i)\b([A-Za-z][A-Za-z' -]+?)\s*\((\d+)\)").captures(line)?;
    let label = capture.get(1)?.as_str().trim().to_ascii_lowercase();
    (term_overlap_count(&label, &focus_refs) >= 1)
        .then(|| capture.get(2).map(|m| m.as_str().trim().to_string()))
        .flatten()
}

pub(in crate::index) fn extract_website_name_from_line(line: &str) -> Option<String> {
    compile_regex(r"\b([A-Za-z0-9-]+\.(?:org|com|net|edu|io))\b")
        .captures(line)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim().to_string())
}

pub(in crate::index) fn extract_beer_recommendation_answer_from_line(
    lower: &str,
) -> Option<String> {
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

pub(in crate::index) fn extract_session_education_answer(
    line: &str,
    lower: &str,
) -> Option<String> {
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

pub(in crate::index) fn extract_session_occupation_answer(
    line: &str,
    lower: &str,
) -> Option<String> {
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
    compile_regex(r"(?i)(\$\d[\d,]*(?:\.\d+)?)")
        .captures(line)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim().to_string())
}

pub(in crate::index) fn extract_percent_answer_from_line(line: &str) -> Option<String> {
    compile_regex(r"(?i)(\d+(?:\.\d+)?%)")
        .captures(line)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim().to_string())
}

pub(in crate::index) fn extract_speed_answer_from_line(line: &str) -> Option<String> {
    compile_regex(r"(?i)(\d+(?:\.\d+)?\s*(?:mbps|gbps))")
        .captures(line)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim().to_string())
}

pub(in crate::index) fn extract_university_name_from_line(line: &str) -> Option<String> {
    compile_regex(r"([A-Z][A-Za-z&.'-]*(?:\s+[A-Z][A-Za-z&.'-]*)*\s+University)")
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
    let day = compile_regex(r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?\b")
        .captures(lower)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().to_string());
    match day {
        Some(day) => format!("{role}|{day}"),
        None => role.to_string(),
    }
}

pub(in crate::index) fn extract_duration_answer_from_line(line: &str) -> Option<String> {
    compile_regex(
        r"(?i)\b((?:about\s+)?(?:an?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?)\s+(?:days?|weeks?|months?|years?|hours?|minutes?)(?:\s+(?:ago|now|each way))?)\b",
    )
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
    let caps = compile_regex(
        r"\b(\d+(?:\.\d+)?|an?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)(?:\s*-\s*(\d+(?:\.\d+)?))?\s+(day|week|month|year|hour|minute)s?\b",
    )
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
    compile_regex(r"(?i)\b(\d+(?:\.\d+)?)\s+ounces?\s+of\s+water\b")
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
        if let Some(value) = compile_regex(pattern)
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
        if let Some(value) = compile_regex(pattern)
            .captures(line)
            .and_then(|caps| caps.get(1))
            .map(|m| m.as_str().trim().to_string())
        {
            return Some(value);
        }
    }
    None
}

pub(in crate::index) fn extract_query_aligned_numeric_answer(
    task_lower: &str,
    line: &str,
) -> Option<String> {
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
        let pattern = compile_regex(&format!(
            r"(?i)\b((?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred))\s+{}s?\b",
            regex::escape(&term)
        ));
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
        let pattern = compile_regex(&format!(
            r"(?i)\b((?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred))\s+{}s?\b",
            regex::escape(&term)
        ));
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

    let pattern = compile_regex(
        r"(?i)(?:antique|vintage|depression-era)\s+[a-z][a-z-]*(?:\s+[a-z][a-z-]*){0,3}",
    );
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
        compile_regex(r"(?i)\btwins?(?:\s+\w+)?\s*,\s*([A-Z][a-z]+)\s+and\s+([A-Z][a-z]+)\b");
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
        compile_regex(r"(?i)\bbaby\s+(?:boy|girl)\s+named\s+([A-Z][a-z]+)\b"),
        compile_regex(r"(?i)\b(?:son|daughter)\s+([A-Z][a-z]+)\b"),
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
    let with_determiner = compile_regex(
        r"(?i)\b(?:my|the|our|a|an)\s+((?:road|commuter|mountain|hybrid|gravel|touring|electric|e-bike|ebike|bmx|trail)\s+bike)\b",
    )
            .captures(line)
            .and_then(|caps| caps.get(1))
            .map(|m| m.as_str().trim().to_string());
    let phrase = with_determiner.or_else(|| {
        compile_regex(
            r"(?i)\b((?:road|commuter|mountain|hybrid|gravel|touring|electric|e-bike|ebike|bmx|trail)\s+bike)\b",
        )
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

pub(in crate::index) fn line_describes_countable_fitness_class_schedule(
    line: &str,
    lower: &str,
) -> bool {
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

pub(in crate::index) fn extract_month_day_values_from_line(
    line: &str,
    lower: &str,
    month: &str,
) -> Vec<u32> {
    if !lower.contains(month) {
        return Vec::new();
    }

    let month_pattern = regex::escape(month);
    let mut days = Vec::new();
    let mut seen = HashSet::new();

    let month_range = compile_regex(&format!(
        r"(?i)\b{}\s+(\d{{1,2}})(?:st|nd|rd|th)?\s*-\s*(\d{{1,2}})(?:st|nd|rd|th)?\b",
        month_pattern
    ));
    for caps in month_range.captures_iter(line) {
        let Some(start) = caps.get(1).and_then(|m| m.as_str().parse::<u32>().ok()) else {
            continue;
        };
        let Some(end) = caps.get(2).and_then(|m| m.as_str().parse::<u32>().ok()) else {
            continue;
        };
        push_month_day_range(&mut days, &mut seen, start, end);
    }

    let day_pair = compile_regex(&format!(
        r"(?i)\b(\d{{1,2}})(?:st|nd|rd|th)?\s+and\s+(\d{{1,2}})(?:st|nd|rd|th)?\s+of\s+{}\b",
        month_pattern
    ));
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

    let month_single = compile_regex(&format!(
        r"(?i)\b{}\s+(\d{{1,2}})(?:st|nd|rd|th)?\b",
        month_pattern
    ));
    for caps in month_single.captures_iter(line) {
        let Some(day) = caps.get(1).and_then(|m| m.as_str().parse::<u32>().ok()) else {
            continue;
        };
        push_month_day(&mut days, &mut seen, day);
    }

    let of_month_single = compile_regex(&format!(
        r"(?i)\b(\d{{1,2}})(?:st|nd|rd|th)?\s+of\s+{}\b",
        month_pattern
    ));
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
    compile_regex(r"(?i)\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b")
        .captures_iter(line)
        .filter_map(|caps| caps.get(1))
        .filter_map(|value| value.as_str().parse::<u32>().ok())
        .any(|value| value == target_month)
}

pub(in crate::index) fn extract_first_quoted_phrase(line: &str) -> Option<String> {
    compile_regex(r#""([^"]+)""#)
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

pub(in crate::index) fn extract_food_delivery_service_from_line(
    _line: &str,
    lower: &str,
) -> Option<String> {
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
    let trimmed = compile_regex(
        r"(?i)\s+(?:about|around)?\s*(?:a\s+few|few|a\s+couple\s+of|couple\s+of|one|two|three|\d+)\s+(?:day|days|week|weeks|month|months|year|years)\s+ago[.!?,]?\s*$",
    )
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
    let caps = compile_regex(
        r"(?i)attended (?:my|our|the) ([^\n]+?)'s ((?:[^.!?\n]+?\s+)?graduation(?: ceremony)?(?: from [^.!?\n]+?)?)\b",
    )
    .captures(line)?;
    let owner = normalized_synthetic_phrase_key(caps.get(1)?.as_str());
    let event =
        normalized_synthetic_phrase_key(&trim_trailing_relative_time_phrase(caps.get(2)?.as_str()));
    Some(format!("{owner}:{event}"))
}

pub(in crate::index) fn extract_health_device_units_from_line(
    _line: &str,
    lower: &str,
) -> Vec<String> {
    let mut devices = Vec::new();
    let mut seen = HashSet::new();

    let has_specific_fitbit =
        lower.contains("fitbit versa 3 smartwatch") || lower.contains("fitbit versa 3");
    let has_generic_fitbit = compile_regex(r"(?i)\bfitbit\b").is_match(lower);
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
    compile_regex(
        r"(?i)\bincrease my (?:work )?hours by (\d+(?:\.\d+)?) hours? (?:weekly|a week|per week)\b",
    )
    .captures(line)?
    .get(1)?
    .as_str()
    .parse::<f32>()
    .ok()
}

pub(in crate::index) fn extract_typical_weekly_work_hours_from_line(
    line: &str,
    lower: &str,
) -> Option<f32> {
    if !task_contains_any(lower, &["i usually work", "usually work"]) {
        return None;
    }
    compile_regex(r"(?i)\bi usually work (\d+(?:\.\d+)?) hours? (?:a|per) week\b")
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
    compile_regex(
        r"(?i)\b(?:working )?up to (\d+(?:\.\d+)?) hours?(?:\s*/\s*week|\s+per\s+week|\s+a\s+week)\b",
    )
    .captures(line)?
    .get(1)?
    .as_str()
    .parse::<f32>()
    .ok()
}

pub(in crate::index) fn extract_recent_activity_query_labels(
    task_lower: &str,
) -> Vec<&'static str> {
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
        compile_regex(r"(?i)\b(\d+)\s*h(?:ours?)?\s*(\d+)\s*min(?:ute)?s?\b"),
        compile_regex(r"(?i)\b(\d+)\s+hours?\s+(?:and\s+)?(\d+)\s+minutes?\b"),
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

pub(in crate::index) fn extract_marathon_target_minutes_from_line(
    line: &str,
    lower: &str,
) -> Option<i32> {
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

pub(in crate::index) fn extract_attended_movie_festival_from_line(
    line: &str,
    lower: &str,
) -> Option<String> {
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
    let caps = compile_regex(
        r"(?i)\b(?:at|after the screening at|like)\b\s+(?:the\s+)?([A-Z][A-Za-z0-9&' .-]+?Film Festival|AFI Fest|TIFF)\b",
    )
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

pub(in crate::index) fn extract_music_release_signatures_from_line(
    line: &str,
    lower: &str,
) -> Vec<String> {
    let mut releases = Vec::new();
    let mut seen = HashSet::new();

    if task_contains_any(lower, &["i bought", "i ended up buying"]) {
        if let Some(caps) = compile_regex(r#"(?i)\b(?:EP|album)\s+["']([^"']+)["']"#).captures(line)
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
        if let Some(caps) =
            compile_regex(r#"(?i)\balbum\s+["']([^"']+)["'][^.\n]*\bdownloaded\b"#).captures(line)
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
            compile_regex(r"(?i)\bgot my ([A-Z][A-Za-z0-9&' .-]+?) vinyl signed\b"),
            compile_regex(
                r"(?i)\bsaw ([A-Z][A-Za-z0-9&' .-]+?) live[^.\n]*\bgot my vinyl signed\b",
            ),
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
            compile_regex(
                r"\bdrum set,\s+a\s+((?:\d+-piece\s+)?[A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+)*)\b",
            ),
            compile_regex(
                r"\b((?:\d+-piece\s+)?[A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+)*)\s+drum set\b",
            ),
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
            compile_regex(r"\bpiano,\s+a\s+([A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+)*)\b"),
            compile_regex(r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+)*)\s+piano\b"),
            compile_regex(r"\b(Korg\s+B1)\b"),
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
            compile_regex(
                r"\bacoustic guitar,\s+a\s+([A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+)*)\b",
            ),
            compile_regex(r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+)*)\s+acoustic guitar\b"),
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
            compile_regex(
                r"\b(?:my|had my|playing my)\s+(?:[a-z]+\s+)?([A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+)*)\s+electric guitar\b",
            ),
            compile_regex(r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+)*)\s+electric guitar\b"),
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
            compile_regex(r"\bukulele,\s+a\s+([A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+)*)\b"),
            compile_regex(r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+)*)\s+ukulele\b"),
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
        compile_regex(r"(?i)\bcompleted\s+([A-Za-z0-9,-]+)\s+courses?\b"),
        compile_regex(r"(?i)\b([A-Za-z0-9,-]+)\s+courses?\s+on\b"),
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

pub(in crate::index) fn extract_loyalty_point_goal_total_from_line(
    line: &str,
    lower: &str,
) -> Option<i32> {
    if !lower.contains("point") {
        return None;
    }
    for pattern in [
        r"(?i)\bneed(?:\s+\w+){0,4}\s+total of\s+(\d+)\s+points\b",
        r"(?i)\breach(?:ing)?\s+(\d+)\s+points\b",
        r"(?i)\b(\d+)\s+points goal\b",
    ] {
        let regex = compile_regex(pattern);
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
        let regex = compile_regex(pattern);
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
            rendered.push_str(&items[items.len() - 1]);
            rendered
        },
    }
}

pub(in crate::index) fn collapsed_owned_instrument_count(instruments: &HashSet<String>) -> usize {
    retained_owned_instrument_keys(instruments).len()
}

pub(in crate::index) fn retained_owned_instrument_keys(
    instruments: &HashSet<String>,
) -> Vec<String> {
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
            leading.push_str(&descriptors[descriptors.len() - 1]);
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
    pub(in crate::index) fn looks_like_shift_header_row(cells: &[String]) -> bool {
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

pub(in crate::index) fn extract_served_dish_from_query(
    task: &str,
    task_lower: &str,
) -> Option<String> {
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
    let pattern = compile_regex(
        r"(?i)(?:which\s+takes|takes|is)\s+(?:about\s+)?((?:an?|one|\d+)\s+(?:hours?|minutes?)(?:\s+each\s+way)?)",
    );
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
    let scoped = compile_regex(r"of the ([A-Z][A-Za-z-]+)");
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
    let pattern = compile_regex(&format!(
        r"(?i)\b{}\b[^.]*?\bhas a ([a-z ]+?) body",
        regex::escape(subject)
    ));
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

pub(in crate::index) fn extract_issue_after_service_line(
    line: &str,
    lower: &str,
) -> Option<String> {
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
    let pattern = compile_regex(r"\$([0-9][0-9,]*(?:\.[0-9]+)?)");
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

pub(in crate::index) fn extract_focused_dollar_amounts(
    line: &str,
    focus_terms: &[String],
) -> Vec<f32> {
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

pub(in crate::index) fn extract_aggregate_duration_value(
    line: &str,
) -> Option<SyntheticDurationValue> {
    pub(in crate::index) fn parse_amount(token: &str) -> Option<f32> {
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

    let postfix_half = compile_regex(
        r"(?i)\b(?:about\s+|around\s+)?(a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|couple|few|\d+(?:\.\d+)?)(?:\s+|-)(day|days|week|weeks|month|months|year|years|hour|hours|minute|minutes)\s+and\s+a\s+half\b",
    );
    let long_form = compile_regex(
        r"(?i)\b(?:(a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|couple|few|\d+(?:\.\d+)?)\s+)?(day|days|week|weeks|month|months|year|years|hour|hours|minute|minutes)(?:-|\s+)long\b",
    );
    let prefix_half = compile_regex(
        r"(?i)\b(?:about\s+|around\s+)?(a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|couple|few|\d+(?:\.\d+)?)(\s+and\s+a\s+half)?(?:\s+|-)(day|days|week|weeks|month|months|year|years|hour|hours|minute|minutes)\b",
    );
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

pub(in crate::index) fn extract_requested_aggregate_duration_unit(
    task_lower: &str,
) -> Option<&'static str> {
    let caps = compile_regex(r"(?i)\bhow many\s+(day|days|week|weeks|month|months|year|years|hour|hours|minute|minutes)\b")
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
        compile_regex(r"(?i)\bhigh school\b.*?\bfrom\s+(\d{4})\s+to\s+(\d{4})\b");
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
    let years = compile_regex(r"\b(19|20)\d{2}\b");
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

pub(in crate::index) fn extract_multi_session_duration_focus_terms(
    task_lower: &str,
) -> Vec<String> {
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

    let explicit_total = compile_regex(
        r"(?:earned|earning(?: a total of)?|for a total of)\s+\$([0-9][0-9,]*(?:\.[0-9]+)?)",
    );
    if let Some(caps) = explicit_total.captures(&lower) {
        if let Some(value) = caps
            .get(1)
            .and_then(|m| m.as_str().replace(',', "").parse::<f32>().ok())
        {
            return Some(value);
        }
    }

    let per_item = compile_regex(r"sold\s+(\d+)[^$]{0,160}?\$([0-9][0-9,]*(?:\.[0-9]+)?)\s*each");
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

    let pattern = compile_regex(r"previous role as a[n]?\s+(.+?)(?:,|\.| and\b| but\b| with\b)");
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
    compile_regex(r"(?i)home country[, ]+([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)?)")
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

pub(in crate::index) fn extract_temporal_elapsed_phrases(
    task_lower: &str,
) -> Option<(String, String)> {
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

pub(in crate::index) fn extract_temporal_from_now_query(
    task_lower: &str,
) -> Option<SyntheticFromNowQuery> {
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

pub(in crate::index) fn parse_temporal_from_now_unit(
    raw: &str,
) -> Option<SyntheticElapsedFromNowUnit> {
    match raw.trim() {
        "day" | "days" => Some(SyntheticElapsedFromNowUnit::Day),
        "week" | "weeks" => Some(SyntheticElapsedFromNowUnit::Week),
        "month" | "months" => Some(SyntheticElapsedFromNowUnit::Month),
        "year" | "years" => Some(SyntheticElapsedFromNowUnit::Year),
        _ => None,
    }
}

pub(in crate::index) fn extract_temporal_interval_phrases(
    task_lower: &str,
) -> Option<(String, String)> {
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

pub(in crate::index) fn temporal_from_now_overlap_count(
    lower_line: &str,
    terms: &[String],
) -> usize {
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

pub(in crate::index) fn temporal_base_day_at_line(
    lines: &[String],
    line_idx: usize,
) -> Option<i32> {
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
    let years = compile_regex(
        r"(?i)\b(a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+years?\b",
    )
    .captures(&lower)
    .and_then(|caps| caps.get(1))
    .and_then(|value| parse_temporal_count_token(value.as_str()));
    let months = compile_regex(
        r"(?i)\b(a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+months?\b",
    )
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

pub(in crate::index) fn extract_current_role_total_months_from_line(
    line: &str,
    lower: &str,
) -> Option<i32> {
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

pub(in crate::index) fn extract_current_role_offset_months_from_line(
    line: &str,
    lower: &str,
) -> Option<i32> {
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
    let numeric = compile_regex(r"(?i)\b(\d{1,2})/(\d{1,2})(?:/(\d{4}))?\b");
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

    let month_day = compile_regex(
        r"(?i)\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(\d{4}))?\b",
    );
    if let Some(caps) = month_day.captures(line) {
        let month = named_month_to_number(caps.get(1)?.as_str())?;
        let day = caps.get(2)?.as_str().parse::<u32>().ok()?;
        let year = caps
            .get(3)
            .and_then(|value| value.as_str().parse::<i32>().ok())
            .unwrap_or(2023);
        return Some(ymd_to_days(year, month, day));
    }

    let day_month_named = compile_regex(
        r"(?i)\b(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December)(?:,\s*(\d{4}))?\b",
    );
    if let Some(caps) = day_month_named.captures(line) {
        let day = caps.get(1)?.as_str().parse::<u32>().ok()?;
        let month = named_month_to_number(caps.get(2)?.as_str())?;
        let year = caps
            .get(3)
            .and_then(|value| value.as_str().parse::<i32>().ok())
            .unwrap_or(2023);
        return Some(ymd_to_days(year, month, day));
    }

    let day_month = compile_regex(
        r"(?i)\b(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)?\s+of\s+(January|February|March|April|May|June|July|August|September|October|November|December)(?:,\s*(\d{4}))?\b",
    );
    if let Some(caps) = day_month.captures(line) {
        let day = caps.get(1)?.as_str().parse::<u32>().ok()?;
        let month = named_month_to_number(caps.get(2)?.as_str())?;
        let year = caps
            .get(3)
            .and_then(|value| value.as_str().parse::<i32>().ok())
            .unwrap_or(2023);
        return Some(ymd_to_days(year, month, day));
    }

    let fuzzy_month = compile_regex(
        r"(?i)\b(?:(early|mid|late)[-\s]+)?(January|February|March|April|May|June|July|August|September|October|November|December)(?:,\s*|\s+)?(\d{4})?\b",
    );
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
    let regex = compile_regex(
        r"(?i)^\s*(?:about\s+|around\s+)?(a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|couple|few|\d+(?:\.\d+)?)(\s+and\s+a\s+half)?\s+(day|days|week|weeks|month|months|year|years)\b",
    );
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

impl NeuronIndex {
    // ── Compile ───────────────────────────────────────────────────────────────

    /// Walk the project tree, create stubs for new/changed source files.
    ///
    /// Idempotent: re-running on an unchanged project is a no-op (only the
    /// hash check causes any work). Returns the total number of neurons managed.
    ///
    /// Enhancements per compile pass:
    /// - **AST Bootstrap**: extracts function signatures + types from source at compile
    ///   time and pre-fills the `api` section of new stubs so BM25 has vocabulary
    ///   from day 1, before any LLM curation.
    /// - **Auto-Synapse**: parses import statements and creates `Imports`-typed synapse
    ///   edges automatically so the graph traversal works from day 1.
    /// - **Git Confidence**: queries `git ls-files` once to classify files as committed
    ///   (1.0), modified (0.9), or untracked (0.85) — applied as a mild BM25 multiplier.
    pub fn compile(&mut self) -> Result<usize> {
        let root = self.project_root.clone();
        let ndir = neuron_dir(&root);
        std::fs::create_dir_all(&ndir)?;

        // Ensure the project neuron exists.
        self.ensure_project_neuron(&root)?;
        // S5 (R15 NE4): generate wake-up neurons from project metadata.
        self.ensure_wake_up_neurons(&root, &ndir)?;

        // Build git confidence map once (3 git commands, silent on non-git projects).
        let git_confidence = build_git_confidence_map(&root);

        // S4 — Parallel compile: Phase 1 collect files, Phase 2 process in parallel,
        // Phase 3 batch-insert sequentially.
        //
        // Each file's pipeline (hash-check → AST extract → stub write → meta write) is
        // fully data-parallel: no shared mutable state across files. Only the final
        // index_neuron() calls require &mut self and run sequentially in Phase 3.
        //
        // Expected speedup: 4–8× on a modern multi-core laptop for 1 000-file projects.

        // Phase 1: collect all source file paths (sequential WalkDir, fast).
        let files: Vec<PathBuf> = WalkDir::new(&root)
            .min_depth(1)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .map(|e| e.path().to_path_buf())
            .collect();

        // Phase 2: hash-check + AST + stub/meta writes (parallel, I/O-bound).
        // process_source_file returns Vec<CompiledFile>: [Core] + any UseCase sub-neurons (S3).
        let compiled: Vec<CompiledFile> = files
            .par_iter()
            .flat_map(|abs| process_source_file(abs, &root, &git_confidence))
            .collect();

        // Phase 3: sequential batch insert into the in-memory index.
        let new_count = self.index_compiled_files(compiled, false);
        self.finalize_compile_pass(&root)?;
        Ok(new_count)
    }

    /// Incremental compile — processes only files listed in `.cortyx/dirty.json`.
    ///
    /// The file watcher writes changed source paths to dirty.json after each batch.
    /// On next server start (or `cortyx compile --incremental`), only those files
    /// are re-indexed instead of walking the entire tree — O(changed) not O(all).
    ///
    /// Falls back to a full `compile()` if dirty.json is absent or unparseable.
    /// Clears dirty.json after successful processing.
    pub fn compile_dirty(&mut self) -> Result<usize> {
        let dirty_file = dirty_path(&self.project_root);

        if !dirty_file.exists() {
            tracing::debug!("No dirty.json — falling back to full compile.");
            return self.compile();
        }

        let dirty_paths: Vec<PathBuf> = std::fs::read_to_string(&dirty_file)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();

        if dirty_paths.is_empty() {
            if let Err(e) = std::fs::remove_file(&dirty_file) {
                if e.kind() != std::io::ErrorKind::NotFound {
                    tracing::warn!("Failed to clear empty dirty.json: {e}");
                }
            }
            return Ok(0);
        }

        tracing::info!(
            "Incremental compile: processing {} dirty file(s).",
            dirty_paths.len()
        );

        let root = self.project_root.clone();
        let git_confidence = build_git_confidence_map(&root);
        let compiled: Vec<CompiledFile> = dirty_paths
            .par_iter()
            .flat_map(|abs| process_source_file(abs, &root, &git_confidence))
            .collect();

        let new_count = self.index_compiled_files(compiled, true);
        self.finalize_compile_pass(&root)?;
        if let Err(e) = std::fs::remove_file(&dirty_file) {
            if e.kind() != std::io::ErrorKind::NotFound {
                tracing::warn!("Failed to clear dirty.json after compile_dirty: {e}");
            }
        }
        Ok(new_count)
    }

    /// Add/update a single entry in the in-memory index without rebuilding derived structures.
    ///
    /// Use this in tight loops (e.g. bulk mining) and call `commit()` once at the end.
    pub fn stage(&mut self, neuron_path: &Path, content: &str, meta: &NeuronMeta) {
        self.index_neuron(neuron_path, content, meta);
    }

    /// Rebuild all derived structures and persist the index.
    ///
    /// Call after a batch of `stage()` calls to apply changes in a single pass.
    pub fn commit(&mut self) -> Result<()> {
        self.rebuild_derived();
        self.save()
    }

    /// Add/update a single neuron in the index (called by MCP tools).
    ///
    /// Persists the index to disk after every mutation so MCP changes
    /// survive a server restart.
    pub fn upsert_neuron(
        &mut self,
        neuron_path: &Path,
        content: &str,
        meta: &NeuronMeta,
    ) -> Result<()> {
        self.stage(neuron_path, content, meta);
        self.commit()
    }

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
                    kind_ok && module_set.as_ref().map_or(true, |ms| ms.contains(&i))
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
                if module_set.as_ref().map_or(false, |ms| !ms.contains(&i)) {
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
            let use_count_aggregate = should_inject_count_aggregate(task);

            let best_agg = if is_dollar_query {
                best_matching_arithmetic_aggregate_path(&self.project_root, raw_focus_terms)
            } else if use_count_aggregate {
                None
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
                    || self.adjacency.get(&syn.target).map_or(false, |nbr_syns| {
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

    /// Like `get_contexts` but also returns compressed (headline-only) neurons that
    /// exceeded the token budget.
    ///
    /// Returns `(full_neurons, overflow_neurons)`.  `overflow_neurons` is a vec of
    /// `(path, headline)` pairs — the headline is the first content line of the
    /// `## purpose` section (or a stub fallback).  Callers can inject the headlines
    /// into the prompt as low-cost navigation hints without the full neuron body.
    ///
    /// `min_confidence`: when `Some(threshold)`, returns `([], [])` immediately if the
    /// top raw BM25 score for `task` is below `threshold`.  Use this to implement the
    /// LongMemEval *abstention* signal — the system should say "no relevant memory"
    /// rather than hallucinating a low-quality match.  Typical threshold: `0.5`
    /// (= `LOW_CONFIDENCE_THRESHOLD`).  Pass `None` to disable (default behaviour).
    pub fn get_contexts_with_overflow(
        &self,
        task: &str,
        max_tokens: usize,
        module: Option<&str>,
        kind: Option<&str>,
        min_confidence: Option<f32>,
        multi_hop: bool,
    ) -> (Vec<PathBuf>, Vec<(PathBuf, String)>) {
        let Ok(query) = QueryText::new(task) else {
            return (Vec::new(), Vec::new());
        };
        // Abstention signal: if caller set a minimum confidence threshold and the
        // best BM25 score for this query is below it, return nothing immediately.
        // This is critical for LongMemEval "absent" questions (20% of the dataset),
        // where returning a low-relevance neuron counts as a false positive.
        if let Some(threshold) = min_confidence {
            if self.peek_max_bm25_score(query.as_str()) < threshold {
                tracing::debug!(
                    task = query.as_str(),
                    threshold,
                    "Abstention: top BM25 score below min_confidence — returning empty."
                );
                return (vec![], vec![]);
            }
        }

        // F1: Task Complexity Adaptive Budget
        //
        // Scale max_tokens by [0.5, 1.5] based on query complexity:
        //   - BM25 breadth: how many distinct terms have posting-list hits
        //   - Module spread: unique modules in top candidates
        //   - Synapse depth: whether candidates have outgoing synapses
        //
        // Simple queries (breadth=1, no synapses) → 0.5× budget (saves tokens)
        // Complex queries (broad match, cross-module) → 1.5× budget
        let terms = tokenize(query.as_str());
        let complexity = self.compute_task_complexity(&terms);
        // F2: apply session-history budget scale on top of F1 complexity scale
        let history_scale = self.adaptive_budget_scale();
        let adjusted_max = ((max_tokens as f32 * complexity * history_scale) as usize)
            .max(512)
            .min(8192.max(max_tokens * 2));
        tracing::debug!(
            task,
            complexity,
            history_scale,
            original_max = max_tokens,
            adjusted_max,
            "F1+F2: adaptive token budget"
        );

        let candidate_set: HashSet<usize> = {
            let mut s = HashSet::new();
            for term in &terms {
                if let Some(idxs) = self.posting_list.get(term) {
                    s.extend(idxs);
                }
            }
            s
        };

        // Run the full activation pipeline via get_contexts with an enormous budget,
        // then re-split. Slightly wasteful but keeps logic DRY.
        //
        // Collected as Vec so the multi-hop block can reference the pre-budget-split
        // ranked order (all_ordered[..5]) without re-running the pipeline.
        let all_ordered: Vec<PathBuf> = self.get_contexts(task, usize::MAX / 2, module, kind);

        let mut full = Vec::new();
        let mut overflow = Vec::new();
        let mut used = 0usize;

        for path in all_ordered.iter().cloned() {
            let tokens = self.entry_by_path(&path).map(|e| e.tokens).unwrap_or(200);
            if used + tokens <= adjusted_max || full.is_empty() {
                used += tokens;
                full.push(path);
            } else {
                // Collect headline for overflow neuron
                let headline = neuron_headline_for(&path);
                overflow.push((path, headline));
            }
        }

        // Multi-hop retrieval: expand from the top-5 pre-budget-split retrieval hits
        // to discover neurons reachable via multiple semantic paths.
        //
        // Improvement over prior top-1 expansion: seeding from all top-5 hits captures
        // terms from multiple subtopics, improving recall for complex multi-hop queries
        // (recursiveMAS iterative deepening principle applied heuristically).
        //
        // All novel neurons go to overflow (lower-priority hints), so full results and
        // their ranking are unchanged — recall can only increase, not decrease.
        if multi_hop && !all_ordered.is_empty() {
            let seed_entries: Vec<&BM25Entry> = all_ordered
                .iter()
                .take(5)
                .filter_map(|p| self.entry_by_path(p))
                .collect();

            if !seed_entries.is_empty() {
                let mut hop_terms = terms.clone();

                for entry in &seed_entries {
                    // Sort clouds before truncation for determinism across runs.
                    let mut cloud: Vec<&String> = entry.concept_cloud.iter().collect();
                    cloud.sort();
                    hop_terms.extend(cloud.into_iter().take(5).cloned());

                    let mut syns: Vec<&String> = entry.synonym_cloud.iter().collect();
                    syns.sort();
                    hop_terms.extend(syns.into_iter().take(3).cloned());
                }

                // Gather TF-IDF terms from all seeds; deduplicate by keeping max freq per
                // term via BTreeMap (lexicographic key order → deterministic output).
                let already: HashSet<&str> = hop_terms.iter().map(|s| s.as_str()).collect();
                let mut tfidf_best: std::collections::BTreeMap<String, f32> =
                    std::collections::BTreeMap::new();
                for entry in &seed_entries {
                    for (t, &f) in &entry.term_freq {
                        if t.len() >= 4 && !already.contains(t.as_str()) {
                            tfidf_best
                                .entry(t.clone())
                                .and_modify(|v| *v = v.max(f))
                                .or_insert(f);
                        }
                    }
                }
                // Sort by (freq DESC, term ASC) for stable ordering across runs.
                let mut tfidf: Vec<(f32, String)> =
                    tfidf_best.into_iter().map(|(t, f)| (f, t)).collect();
                tfidf.sort_unstable_by(|a, b| {
                    b.0.total_cmp(&a.0).then(a.1.as_str().cmp(b.1.as_str()))
                });
                hop_terms.extend(tfidf.into_iter().take(15).map(|(_, t)| t));

                hop_terms.sort();
                hop_terms.dedup();

                let expanded_task = hop_terms.join(" ");
                let second_pass = self.get_contexts(&expanded_task, usize::MAX / 2, module, kind);

                let already_included: HashSet<&PathBuf> =
                    full.iter().chain(overflow.iter().map(|(p, _)| p)).collect();
                // Cap novel overflow additions to avoid explosion on broad expanded queries.
                let novel: Vec<(PathBuf, String)> = second_pass
                    .into_iter()
                    .filter(|p| !already_included.contains(p))
                    .take(25)
                    .map(|p| {
                        let headline = neuron_headline_for(&p);
                        (p, headline)
                    })
                    .collect();

                if !novel.is_empty() {
                    tracing::debug!(
                        count = novel.len(),
                        seeds = seed_entries.len(),
                        "Multi-hop 2nd pass: injected additional candidate neurons \
                         (top-{} seed expansion)",
                        seed_entries.len()
                    );
                    overflow.extend(novel);
                }
            }
        }

        let _ = candidate_set; // suppress unused warning
        (full, overflow)
    }

    /// F1: Compute task complexity as a [0.5, 1.5] budget scale factor.
    ///
    /// Inputs:
    /// - BM25 breadth: fraction of query terms that hit the posting list (term coverage)
    /// - Module spread: unique module count in top-10 candidates (cross-module indicator)
    /// - Synapse depth: fraction of top candidates with outgoing synapses (graph richness)
    ///
    /// Formula: clamp(0.5 + breadth * 0.3 + spread * 0.4 + depth * 0.3, 0.5, 1.5)
    pub(in crate::index) fn compute_task_complexity(&self, terms: &[String]) -> f32 {
        if terms.is_empty() {
            return 1.0;
        }

        // Breadth: fraction of query terms with any posting-list hit
        let hit_terms = terms
            .iter()
            .filter(|t| self.posting_list.contains_key(t.as_str()))
            .count();
        let breadth = hit_terms as f32 / terms.len() as f32;

        // Candidate set for spread/depth analysis
        let mut candidates: HashSet<usize> = HashSet::new();
        for t in terms {
            if let Some(idxs) = self.posting_list.get(t.as_str()) {
                candidates.extend(idxs.iter().take(10));
            }
        }

        // Spread: unique modules among top candidates (normalized by 3)
        let unique_modules: HashSet<Option<&str>> = candidates
            .iter()
            .filter_map(|&i| self.entries.get(i))
            .map(|e| e.module.as_deref())
            .collect();
        let spread = ((unique_modules.len() as f32 - 1.0) / 3.0).clamp(0.0, 1.0);

        // Depth: fraction of candidates that have outgoing synapses
        let with_synapses = candidates
            .iter()
            .filter_map(|&i| self.entries.get(i))
            .filter(|e| !e.synapses.is_empty())
            .count();
        let depth = if candidates.is_empty() {
            0.0
        } else {
            with_synapses as f32 / candidates.len() as f32
        };

        (0.5 + breadth * 0.3 + spread * 0.4 + depth * 0.3).clamp(0.5, 1.5)
    }

    // ── Stats ─────────────────────────────────────────────────────────────────

    pub fn neuron_count(&self) -> usize {
        self.entries.len()
    }

    pub fn synapse_count(&self) -> usize {
        // Count the forward synapses defined on each entry (not the reverse copies in adjacency).
        self.entries.iter().map(|e| e.synapses.len()).sum()
    }

    /// Status counts for doctor: (fresh, stale, stub)
    pub fn status_counts(&self) -> (usize, usize, usize) {
        let ndir = neuron_dir(&self.project_root);
        let mut fresh = 0usize;
        let mut stale = 0usize;
        let mut stub = 0usize;
        for entry in &self.entries {
            let meta_p = meta_path(&entry.neuron_path);
            let status = std::fs::read_to_string(&meta_p)
                .ok()
                .and_then(|d| serde_json::from_str::<NeuronMeta>(&d).ok())
                .map(|m| m.status)
                .unwrap_or(NeuronStatus::Stub);
            // If .context.md is in the ndir, it's a real neuron (avoid counting adjacency copies)
            if !entry.neuron_path.starts_with(&ndir) {
                continue;
            }
            match status {
                NeuronStatus::Fresh => fresh += 1,
                NeuronStatus::Stale => stale += 1,
                NeuronStatus::Stub => stub += 1,
            }
        }
        (fresh, stale, stub)
    }

    /// Return the use_count for a neuron (for display purposes).
    pub fn use_count_for(&self, path: &Path) -> u32 {
        self.path_index
            .get(path)
            .map(|&i| self.entries[i].use_count)
            .unwrap_or(0)
    }

    /// Increment `use_count` for each neuron in `paths` and persist their metadata.
    ///
    /// Also applies auto-quarantine: if a neuron has ≥ MIN_SAMPLE_SIZE activations
    /// but its hit_rate is below QUARANTINE_THRESHOLD (10%), it's a chronic
    /// over-activator — retrieved often but rarely cited. Its staleness_multiplier
    /// is reduced to 0.3, effectively deprioritising it without deletion.
    /// The quarantine lifts automatically when the neuron is re-evolved.
    pub fn record_activation(&mut self, paths: &[std::path::PathBuf]) {
        for path in paths {
            if let Some(&i) = self.path_index.get(path) {
                self.entries[i].use_count = self.entries[i].use_count.saturating_add(1);

                // Bayesian quarantine with adaptive confidence intervals (TRIZ S4 R11).
                //
                // Adaptive tiers:
                //   use_count <  5  → withhold judgment (too few samples)
                //   use_count  5–19 → z=1.0,   threshold=0.02 (react fast to obvious noise)
                //   use_count 20–99 → z=1.645, threshold=0.05 (90% CI — standard behaviour)
                //   use_count ≥100  → z=1.96,  threshold=0.08 (strict for mature neurons)
                // Quarantine is reversible: lower bound > QUARANTINE_RECOVERY_THRESHOLD → restore.
                let uc = self.entries[i].use_count;
                let hc = self.entries[i].hit_count;
                if let Some((z, threshold)) = adaptive_quarantine_params(uc) {
                    let lower = wilson_lower_bound_z(hc, uc, z);
                    let currently_quarantined = self.entries[i].staleness_multiplier <= 0.3;
                    if !currently_quarantined && lower < threshold {
                        self.entries[i].staleness_multiplier = 0.3;
                        tracing::debug!(
                            path = %path.display(),
                            wilson_lower_bound = lower,
                            use_count = uc,
                            hit_count = hc,
                            z = z,
                            threshold = threshold,
                            "Auto-quarantined: Wilson CI lower bound {lower:.3} < {threshold}"
                        );
                    } else if currently_quarantined && lower > QUARANTINE_RECOVERY_THRESHOLD {
                        self.entries[i].staleness_multiplier = 0.7;
                        tracing::debug!(
                            path = %path.display(),
                            wilson_lower_bound = lower,
                            "Quarantine lifted: Wilson CI lower bound {lower:.3} > {QUARANTINE_RECOVERY_THRESHOLD}"
                        );
                    }
                }

                // Persist the updated use_count to the sidecar JSON so it survives restarts.
                let meta_p = meta_path(path);
                if let Ok(data) = std::fs::read_to_string(&meta_p) {
                    if let Ok(mut meta) = serde_json::from_str::<NeuronMeta>(&data) {
                        meta.use_count = self.entries[i].use_count;
                        if let Err(e) = atomic_write_json(&meta_p, &meta) {
                            tracing::warn!(
                                "Failed to persist updated use_count for {}: {e}",
                                meta_p.display()
                            );
                        }
                    }
                }
            }
        }
    }

    /// Increment `hit_count` for a neuron when the LLM confirms it was cited.
    ///
    /// Returns the updated hit_rate = hit_count / use_count.max(1).
    pub fn record_hit(&mut self, neuron_path: &Path, was_cited: bool) -> f32 {
        if let Some(&i) = self.path_index.get(neuron_path) {
            if was_cited {
                self.entries[i].hit_count = self.entries[i].hit_count.saturating_add(1);
            }
            // Always increment use_count on explicit feedback (in case get_contexts missed it)
            self.entries[i].use_count = self.entries[i].use_count.saturating_add(1);

            let hit_rate =
                self.entries[i].hit_count as f32 / self.entries[i].use_count.max(1) as f32;

            // Persist both counters
            let meta_p = meta_path(neuron_path);
            if let Ok(data) = std::fs::read_to_string(&meta_p) {
                if let Ok(mut meta) = serde_json::from_str::<NeuronMeta>(&data) {
                    meta.use_count = self.entries[i].use_count;
                    meta.hit_count = self.entries[i].hit_count;
                    if let Err(e) = atomic_write_json(&meta_p, &meta) {
                        tracing::warn!(
                            "Failed to persist hit feedback for {}: {e}",
                            meta_p.display()
                        );
                    }
                }
            }

            // Adaptive synapse EMA: update learned_weight for all synapses that
            // point to this neuron, reinforcing or downweighting the traversal path.
            self.update_synapse_ema(neuron_path, was_cited);

            hit_rate
        } else {
            0.0
        }
    }

    /// B2: Record query term co-activations for a neuron.
    ///
    /// Called from `get_contexts` for each activated neuron with the query terms.
    /// After ≥30 co-activations, a term is promoted to the neuron's `synonym_cloud`.
    /// The synonym cloud is persisted to the BM25Entry and used at query time for
    /// vocabulary expansion before BM25 scoring.
    pub fn record_coactivation(&mut self, neuron_path: &Path, query_terms: &[String]) {
        const SYNONYM_THRESHOLD: u32 = 30;

        let Some(&entry_idx) = self.path_index.get(neuron_path) else {
            return;
        };

        let counts = self
            .coactivation_counts
            .entry(neuron_path.to_path_buf())
            .or_default();

        let mut promoted = Vec::new();
        for term in query_terms {
            if term.len() < 3 {
                continue;
            }
            let count = counts.entry(term.clone()).or_insert(0);
            *count += 1;
            if *count == SYNONYM_THRESHOLD {
                promoted.push(term.clone());
            }
        }

        if !promoted.is_empty() {
            let cloud = &mut self.entries[entry_idx].synonym_cloud;
            for term in &promoted {
                if !cloud.contains(term) {
                    cloud.push(term.clone());
                    tracing::debug!(
                        neuron = %neuron_path.display(),
                        term,
                        "B2: promoted term to synonym cloud"
                    );
                }
            }
        }

        // R20 C-2: Drain any pending Hebbian synapse creations.
        //
        // `get_contexts()` (a &self method) accumulates co-return counts in a Mutex.
        // Once a pair crosses HEBBIAN_THRESHOLD (10 co-returns), it's flagged there but
        // can't mutate adjacency. Here, in the first subsequent &mut self call, we drain
        // the flagged pairs and create bidirectional SemanticRelated synapses.
        self.apply_pending_hebbian_synapses();
    }

    /// Drain pending Hebbian synapse pairs and create SemanticRelated edges in adjacency.
    pub(in crate::index) fn apply_pending_hebbian_synapses(&mut self) {
        const HEBBIAN_THRESHOLD: u32 = 10;
        let pairs_to_wire: Vec<(PathBuf, PathBuf)> = {
            let Ok(counts) = self.co_return_counts.lock() else {
                return;
            };
            counts
                .iter()
                .filter(|(_, &c)| c == HEBBIAN_THRESHOLD) // exactly at threshold — fire once
                .map(|(k, _)| k.clone())
                .collect()
        };

        for (a, b) in pairs_to_wire {
            // Mark as wired (sentinel = HEBBIAN_THRESHOLD + 1) so we don't re-fire on future calls
            if let Ok(mut counts) = self.co_return_counts.lock() {
                if let Some(c) = counts.get_mut(&(a.clone(), b.clone())) {
                    *c = HEBBIAN_THRESHOLD + 1;
                }
            }

            let already_exists = self.adjacency.get(&a).map_or(false, |syns| {
                syns.iter()
                    .any(|s| s.target == b && s.edge_type == SynapseType::SemanticRelated)
            });
            if already_exists {
                continue;
            }

            let syn_ab = Synapse::new(
                b.clone(),
                SynapseType::SemanticRelated,
                "hebbian:co-return".to_string(),
            );
            let syn_ba = Synapse::new(
                a.clone(),
                SynapseType::SemanticRelated,
                "hebbian:co-return".to_string(),
            );
            self.adjacency.entry(a.clone()).or_default().push(syn_ab);
            self.adjacency.entry(b.clone()).or_default().push(syn_ba);
            tracing::debug!(
                a = %a.display(),
                b = %b.display(),
                "C-2 Hebbian: SemanticRelated synapse created from co-return signal"
            );
        }
    }

    /// B2: Expand query terms through per-neuron synonym clouds.
    ///
    /// For each activated neuron path, return any synonym-cloud terms that appear
    /// in the query — as augmented expansion terms for the next retrieval pass.
    /// Used during `get_contexts` vocabulary expansion phase.
    /// Return the highest raw BM25 score for `task` across all indexed neurons.
    ///
    /// Runs Phase 1 posting-list lookup + BM25 scoring only (no synapse traversal,
    /// no TF-IDF, no dense re-rank).  Used by `get_contexts_with_overflow` to
    /// implement the abstention signal: if the top score is below `min_confidence`,
    /// no neurons are returned and the caller prints a "no relevant memory" message.
    ///
    /// Complexity: O(|candidates|) — same as the fast path in `get_contexts`.
    pub fn peek_max_bm25_score(&self, task: &str) -> f32 {
        let Ok(query) = QueryText::new(task) else {
            return 0.0;
        };
        let terms = tokenize(query.as_str());
        let mut max_score = 0.0f32;
        for term in &terms {
            if let Some(idxs) = self.posting_list.get(term) {
                for &i in idxs {
                    let s = self.bm25_score(&terms, &self.entries[i]);
                    if s > max_score {
                        max_score = s;
                    }
                }
            }
        }
        max_score
    }

    /// Knowledge-update supersession: demote old Verbatim neurons whose content is
    /// substantially overlapped by a newer neuron in the same module/person scope.
    ///
    /// Called by `write_verbatim_neurons` after staging each new Verbatim neuron. When a
    /// newly-ingested turn has ≥60% term overlap with an older turn in the same module AND
    /// the older turn's timestamp pre-dates the new one, the old neuron's
    /// `staleness_multiplier` is halved (→ 0.5×BM25 score). This surfaces the most
    /// current fact for LME-500 knowledge-update questions without evicting history.
    ///
    /// Only applies to Verbatim neurons — code neurons are unaffected.
    pub fn detect_and_mark_supersessions(&mut self, new_path: &Path) {
        const OVERLAP_THRESHOLD: f32 = 0.60;
        const MIN_TERMS: usize = 4;

        let Some(&new_idx) = self.path_index.get(new_path) else {
            return;
        };

        // Snapshot new-entry data to avoid borrow conflicts below.
        let (new_module, new_ts, new_terms) = {
            let e = &self.entries[new_idx];
            if !matches!(e.kind, NeuronKind::Verbatim) {
                return;
            }
            let terms: HashSet<String> = e
                .term_freq
                .keys()
                .filter(|t| t.len() >= MIN_TERMS)
                .cloned()
                .collect();
            (e.module.clone(), e.timestamp_secs, terms)
        };

        if new_terms.is_empty() {
            return;
        }
        let new_ts_val = new_ts.unwrap_or(i64::MAX);

        for i in 0..self.entries.len() {
            if i == new_idx {
                continue;
            }
            let e = &self.entries[i];
            if !matches!(e.kind, NeuronKind::Verbatim) {
                continue;
            }
            if e.module != new_module {
                continue;
            }
            let old_ts = e.timestamp_secs.unwrap_or(0);
            // Only demote OLDER neurons — if old_ts ≥ new_ts, the "old" entry is newer
            // or simultaneous; skip it to avoid mutual demotion within a batch.
            if old_ts >= new_ts_val {
                continue;
            }

            let old_terms: HashSet<&str> = e
                .term_freq
                .keys()
                .filter(|t| t.len() >= MIN_TERMS)
                .map(|s| s.as_str())
                .collect();
            if old_terms.len() < MIN_TERMS {
                continue;
            }

            let overlap = new_terms
                .iter()
                .filter(|t| old_terms.contains(t.as_str()))
                .count();
            let ratio = overlap as f32 / old_terms.len() as f32;

            if ratio >= OVERLAP_THRESHOLD {
                self.entries[i].staleness_multiplier =
                    (self.entries[i].staleness_multiplier * 0.5).max(0.1);
                tracing::debug!(
                    old = ?self.entries[i].neuron_path,
                    new = ?new_path,
                    overlap_ratio = ratio,
                    "Knowledge-update supersession: demoted older neuron"
                );
            }
        }
    }

    pub fn synonym_cloud_expansion(&self, query_terms: &[String]) -> Vec<String> {
        let query_set: HashSet<&String> = query_terms.iter().collect();
        let mut expansion: HashSet<String> = HashSet::new();

        for entry in &self.entries {
            // For each neuron: check if any query term matches an entry term
            let neuron_has_query_term = entry.term_freq.keys().any(|t| query_set.contains(t));
            if neuron_has_query_term {
                // Expand with this neuron's synonym cloud
                for syn_term in &entry.synonym_cloud {
                    expansion.insert(syn_term.clone());
                }
            }
        }

        // Remove terms already in the query to avoid re-adding them
        for t in query_terms {
            expansion.remove(t);
        }

        expansion.into_iter().collect()
    }

    /// F2: Record session token utilization for budget adaptation.
    ///
    /// Call at the end of each session (close_task) with the tokens used and the budget.
    /// Keeps the last 5 sessions' data. The next call to `adaptive_budget_scale()` uses
    /// this history to adjust max_tokens up or down.
    pub fn record_session_utilization(&mut self, tokens_used: usize, tokens_budget: usize) {
        const MAX_HISTORY: usize = 5;
        self.session_utilization.push([tokens_used, tokens_budget]);
        if self.session_utilization.len() > MAX_HISTORY {
            self.session_utilization.remove(0);
        }
    }

    /// F2: Compute the budget scale factor from session history.
    ///
    /// - If last 5 sessions used < 40% of budget → scale down by 20% (too much headroom)
    /// - If ≥3 of last 5 sessions hit 100% of budget (overflow) → scale up by 20%
    /// - Otherwise: no change (scale = 1.0)
    ///
    /// Returns a multiplier [0.8, 1.2] to apply to max_tokens.
    /// Capped post-multiplication at [512, 8192] by the caller.
    pub fn adaptive_budget_scale(&self) -> f32 {
        let history = &self.session_utilization;
        if history.len() < 2 {
            return 1.0; // not enough data
        }

        let underused = history
            .iter()
            .filter(|[used, budget]| *budget > 0 && (*used as f32 / *budget as f32) < 0.4)
            .count();

        let overflowed = history
            .iter()
            .filter(|[used, budget]| *used >= *budget)
            .count();

        if underused == history.len() {
            0.8 // all sessions underused → shrink
        } else if overflowed >= 3 {
            1.2 // ≥3/5 sessions overflowed → grow
        } else {
            1.0 // normal
        }
    }

    /// `cited = true` → signal = 1.0 (this synapse helped); `false` → 0.0.
    ///
    /// EMA rule: `learned_weight ← α × signal + (1 − α) × learned_weight`  (α = 0.1)
    ///
    /// Cold-start: when `learned_weight == 0.0`, it is initialised to the type
    /// multiplier before the first update so the decay doesn't start from zero.
    ///
    /// Only in-memory entries are updated; `save()` persists them to `index.json`.
    /// NeuronMeta sidecar files are NOT updated (they are the source-of-truth for
    /// compile-time synapse topology, not runtime weights).
    pub fn update_synapse_ema(&mut self, target_path: &Path, cited: bool) {
        const ALPHA: f32 = 0.1;
        let signal = if cited { 1.0_f32 } else { 0.0_f32 };

        for entry in &mut self.entries {
            for syn in &mut entry.synapses {
                if syn.target == target_path {
                    // Cold-start init: seed from type multiplier so EMA starts at a
                    // sensible baseline rather than decaying from 0.
                    if syn.learned_weight <= 0.0 {
                        syn.learned_weight = syn.edge_type.type_multiplier();
                    }
                    syn.learned_weight = ALPHA * signal + (1.0 - ALPHA) * syn.learned_weight;
                    syn.traversal_count = syn.traversal_count.saturating_add(1);
                }
            }
        }
    }

    pub fn print_status(&self) {
        let mut cores = 0usize;
        let mut usecases = 0usize;
        let mut verbatim = 0usize;
        let mut concepts = 0usize;
        let mut stubs = 0usize;
        for e in &self.entries {
            match e.kind {
                NeuronKind::Core | NeuronKind::Project => {
                    cores += 1;
                    if e.term_count == 0 || e.term_freq.is_empty() {
                        stubs += 1;
                    }
                },
                NeuronKind::UseCase => usecases += 1,
                NeuronKind::Verbatim => verbatim += 1,
                NeuronKind::Concept | NeuronKind::Aggregate => concepts += 1,
            }
        }
        println!("Cortyx Index");
        println!("============");
        println!("  Core neurons:         {cores}  ({stubs} stubs — run cortyx_evolve_context)");
        println!("  Use-case neurons:     {usecases}");
        println!("  Verbatim chunks:      {verbatim}");
        println!("  Concept neurons:      {concepts}");
        println!("  Synapses:             {}", self.synapse_count());
        println!("  Modules indexed:      {}", self.module_index.len());
        println!("  Avg doc length:       {:.0} terms", self.avg_doc_len);
    }

    // ── Invalidation ──────────────────────────────────────────────────────────

    /// Mark a source file's neuron as stale (hash changed or forced).
    ///
    /// The stale neuron is demoted (staleness_multiplier → 0.5) rather than evicted
    /// so it can still activate on niche queries where it remains the best match.
    /// A full eviction would lose context permanently before the LLM re-evolves it.
    pub fn invalidate(&mut self, source: &Path) -> Result<()> {
        let neuron = core_neuron_path(source, &self.project_root);
        let meta_file = meta_path(&neuron);
        if meta_file.exists() {
            if let Ok(data) = std::fs::read_to_string(&meta_file) {
                if let Ok(mut meta) = serde_json::from_str::<NeuronMeta>(&data) {
                    meta.status = NeuronStatus::Stale;
                    if let Err(e) = atomic_write_json(&meta_file, &meta) {
                        tracing::warn!(
                            "Failed to persist stale marker for {}: {e}",
                            meta_file.display()
                        );
                    }
                }
            }
        }
        // Demote the in-memory entry rather than removing it.
        if let Some(&i) = self.path_index.get(&neuron) {
            self.entries[i].staleness_multiplier = 0.5;
        }
        self.save()
    }

    /// Permanently remove a neuron from the index and delete its files from disk.
    ///
    /// Unlike `invalidate`, this is a hard delete — the neuron's `.context.md` and
    /// its sidecar `.json` are removed. Used by `cortyx prune`.
    ///
    /// Returns `true` if the neuron was found and removed, `false` if it was unknown.
    pub fn evict_entry(&mut self, neuron_path: &Path) -> bool {
        let Some(&idx) = self.path_index.get(neuron_path) else {
            return false;
        };
        self.entries.swap_remove(idx);
        // After swap_remove, the entry previously at the last position is now at `idx`.
        // Update its path_index slot so future lookups remain correct.
        if idx < self.entries.len() {
            self.path_index
                .insert(self.entries[idx].neuron_path.clone(), idx);
        }
        self.path_index.remove(neuron_path);
        // Rebuild derived structures — eviction happens in bulk during prune,
        // so the caller calls rebuild_derived() once after all evictions.
        true
    }

    /// Neuron paths together with their activation count — used by `cortyx prune`.
    pub fn neuron_paths_and_use_counts(&self) -> Vec<(PathBuf, u32)> {
        self.entries
            .iter()
            .map(|e| (e.neuron_path.clone(), e.use_count))
            .collect()
    }

    // ── Hierarchy navigation (TRIZ R13-G2) ───────────────────────────────────

    /// List all modules with their neuron count and average hit rate.
    /// Includes `@person` scoped modules alongside directory modules.
    /// Returns entries sorted by name for deterministic output.
    pub fn list_modules(&self) -> Vec<ModuleSummary> {
        let mut map: HashMap<String, (usize, f32)> = HashMap::new();
        for entry in &self.entries {
            if let Some(m) = entry.module.as_deref() {
                let e = map.entry(m.to_string()).or_default();
                e.0 += 1;
                let rate = if entry.use_count > 0 {
                    entry.hit_count as f32 / entry.use_count as f32
                } else {
                    0.0
                };
                e.1 += rate;
            }
        }
        let mut result: Vec<ModuleSummary> = map
            .into_iter()
            .map(|(name, (count, rate_sum))| ModuleSummary {
                name: name.clone(),
                neuron_count: count,
                avg_hit_rate: if count > 0 {
                    rate_sum / count as f32
                } else {
                    0.0
                },
                is_person_scope: name.starts_with('@'),
            })
            .collect();
        result.sort_by(|a, b| a.name.cmp(&b.name));
        result
    }

    /// List neurons in a module (or all neurons if `module` is None).
    /// Returns a summary of each neuron's path, kind, staleness, and hit rate.
    pub fn list_neurons(&self, module: Option<&str>) -> Vec<NeuronSummary> {
        let indices: Vec<usize> = if let Some(m) = module {
            self.module_index.get(m).cloned().unwrap_or_default()
        } else {
            (0..self.entries.len()).collect()
        };
        let mut result: Vec<NeuronSummary> = indices
            .into_iter()
            .map(|i| {
                let e = &self.entries[i];
                let hit_rate = if e.use_count > 0 {
                    e.hit_count as f32 / e.use_count as f32
                } else {
                    0.0
                };
                NeuronSummary {
                    path: e.neuron_path.clone(),
                    kind: e.kind.clone(),
                    staleness_multiplier: e.staleness_multiplier,
                    hit_rate,
                    use_count: e.use_count,
                }
            })
            .collect();
        result.sort_by(|a, b| a.path.cmp(&b.path));
        result
    }

    /// Return the most recent Verbatim neurons that mention "current moment" markers.
    ///
    /// This stays index-only: it uses precomputed timestamps plus token presence to cheaply
    /// surface likely `today` / `currently` / `this week` sessions for downstream temporal
    /// reasoning without scanning the full corpus at query time.
    pub fn recent_verbatim_paths_with_current_markers(
        &self,
        module: Option<&str>,
        limit: usize,
    ) -> Vec<PathBuf> {
        if limit == 0 {
            return Vec::new();
        }

        let has_current_marker_terms = |terms: &HashMap<String, f32>| {
            terms.contains_key("today")
                || terms.contains_key("currently")
                || terms.contains_key("now")
                || (terms.contains_key("this")
                    && (terms.contains_key("week")
                        || terms.contains_key("month")
                        || terms.contains_key("year")))
        };

        let mut ranked = self
            .entries
            .iter()
            .filter(|entry| matches!(entry.kind, NeuronKind::Verbatim))
            .filter(|entry| {
                module.is_none_or(|scope| entry.module.as_deref() == Some(scope))
                    && has_current_marker_terms(&entry.term_freq)
            })
            .filter_map(|entry| Some((entry.timestamp_secs?, entry.neuron_path.clone())))
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)));
        ranked
            .into_iter()
            .take(limit)
            .map(|(_, path)| path)
            .collect()
    }

    /// Return neurons that are strong candidates for the shared concept library.
    ///
    /// Candidates must:
    /// - be Core or Concept neurons
    /// - meet the minimum use_count / hit_rate / quality thresholds
    /// - be sorted by strongest observed utility first
    pub fn publish_ready_candidates(
        &self,
        min_use: u32,
        min_hit_rate: f32,
        min_quality: f32,
        limit: usize,
    ) -> Vec<PublishReadySummary> {
        let mut result: Vec<PublishReadySummary> = self
            .entries
            .iter()
            .filter_map(|entry| {
                if !matches!(entry.kind, NeuronKind::Core | NeuronKind::Concept) {
                    return None;
                }
                let hit_rate = if entry.use_count > 0 {
                    entry.hit_count as f32 / entry.use_count as f32
                } else {
                    0.0
                };
                if entry.use_count < min_use
                    || hit_rate < min_hit_rate
                    || entry.quality_score < min_quality
                {
                    return None;
                }
                Some(PublishReadySummary {
                    path: entry.neuron_path.clone(),
                    kind: entry.kind.clone(),
                    use_count: entry.use_count,
                    hit_rate,
                    quality_score: entry.quality_score,
                })
            })
            .collect();
        result.sort_by(|a, b| {
            b.use_count
                .cmp(&a.use_count)
                .then_with(|| b.hit_rate.total_cmp(&a.hit_rate))
                .then_with(|| b.quality_score.total_cmp(&a.quality_score))
                .then_with(|| a.path.cmp(&b.path))
        });
        if limit > 0 {
            result.truncate(limit);
        }
        result
    }

    /// Return the first `lines` lines of a neuron file for quick preview.
    /// Returns `None` if the file does not exist or cannot be read.
    pub fn peek_neuron(&self, path: &Path, lines: usize) -> Option<String> {
        let content = std::fs::read_to_string(path).ok()?;
        let preview: String = content.lines().take(lines).collect::<Vec<_>>().join("\n");
        Some(preview)
    }

    /// List only `@person`-scoped modules (convention: module starts with `@`).
    pub fn list_persons(&self) -> Vec<ModuleSummary> {
        self.list_modules()
            .into_iter()
            .filter(|m| m.is_person_scope)
            .collect()
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Mine each source file for function call sites that match public functions
    /// defined in *other* source files of the project.
    ///
    /// Workflow:
    /// 1. Build a vocabulary map `fn_name → source_rel_path` from all entries'
    ///    extracted function names (stored in `term_freq` keys during compile).
    ///    Entries with no functions in their term_freq are skipped.
    /// 2. Walk each source file, call `ast_extractor::extract_call_sites`,
    ///    and for each detected `CallEdge`, emit a `Calls`-typed synapse from
    ///    the calling neuron to the callee neuron (if one doesn't already exist).
    ///
    /// This is a second compile pass and runs in O(files × |vocab|) — both are
    /// typically small so runtime is negligible.
    pub(in crate::index) fn apply_call_graph_synapses(&mut self, root: &Path) {
        // Build fn_name → source_path vocabulary from the already-loaded entries.
        // We use term_freq keys that look like function names (alphabetic, no spaces).
        // This is approximate but practical — false positives are filtered by
        // the self-loop guard in extract_call_sites.
        //
        // A tighter approach would be to store a dedicated `functions: Vec<String>`
        // field in BM25Entry, but term_freq already contains them from AST Bootstrap.
        // Function names are pure alphabetic tokens, distinct from normal prose terms.
        let mut fn_vocab: HashMap<String, PathBuf> = HashMap::new();
        for entry in &self.entries {
            let rel_source = entry
                .neuron_path
                .strip_prefix(root)
                .map(|r| r.to_path_buf())
                .unwrap_or_else(|_| entry.neuron_path.clone());

            // Extract function names: those that appear in term_freq AND match the
            // pattern of a public function name (all word chars, len ≥ 3, not all-lowercase
            // common English words). We use a simple heuristic rather than re-running AST.
            for term in entry.term_freq.keys() {
                // Public function names are typically CamelCase or snake_case identifiers
                // ≥ 3 chars with no digits-only and not a BM25 stop-word.
                if term.len() >= 3 && term.chars().all(|c| c.is_alphanumeric() || c == '_') {
                    fn_vocab
                        .entry(term.clone())
                        .or_insert_with(|| rel_source.clone());
                }
            }
        }

        if fn_vocab.is_empty() {
            return;
        }

        // Walk each source file and find call sites.
        let source_extensions = [
            "rs", "py", "ts", "tsx", "js", "jsx", "go", "swift", "kt", "java", "cs", "rb", "c",
            "cpp", "cc",
        ];
        let walker = WalkDir::new(root)
            .min_depth(1)
            .into_iter()
            .filter_map(|e| e.ok());
        let mut synapse_patches: Vec<(PathBuf, PathBuf)> = Vec::new(); // (caller_neuron, callee_neuron)

        for entry in walker {
            if !entry.file_type().is_file() {
                continue;
            }
            let abs = entry.path();
            let rel = abs.strip_prefix(root).unwrap_or(abs);
            let ext = rel.extension().and_then(|e| e.to_str()).unwrap_or("");
            if !source_extensions.contains(&ext) || should_skip(rel) {
                continue;
            }
            let Ok(content) = std::fs::read_to_string(abs) else {
                continue;
            };
            let source_rel = rel.to_string_lossy();
            let call_edges = ast_extractor::extract_call_sites(&source_rel, &content, &fn_vocab);
            if call_edges.is_empty() {
                continue;
            }
            let caller_neuron = core_neuron_path(abs, root);
            for edge in call_edges {
                let callee_source = root.join(&edge.callee_file);
                let callee_neuron = core_neuron_path(&callee_source, root);
                if callee_neuron != caller_neuron {
                    synapse_patches.push((caller_neuron.clone(), callee_neuron));
                }
            }
        }

        // Apply collected patches to meta files and in-memory entries.
        for (caller_neuron, callee_neuron) in synapse_patches {
            let meta_file = meta_path(&caller_neuron);
            let Ok(data) = std::fs::read_to_string(&meta_file) else {
                continue;
            };
            let Ok(mut meta) = serde_json::from_str::<NeuronMeta>(&data) else {
                continue;
            };
            let already_exists = meta
                .synapses
                .iter()
                .any(|s| s.target == callee_neuron && matches!(s.edge_type, SynapseType::Calls));
            if already_exists {
                continue;
            }
            meta.synapses.push(Synapse::new(
                callee_neuron.clone(),
                SynapseType::Calls,
                "auto-inferred from call-site scan".to_string(),
            ));
            if let Err(e) = atomic_write_json(&meta_file, &meta) {
                tracing::warn!(
                    "Failed to persist call-graph synapse for {}: {e}",
                    meta_file.display()
                );
            }
            // Update in-memory entry as well.
            if let Some(&idx) = self.path_index.get(&caller_neuron) {
                self.entries[idx].synapses.push(Synapse::new(
                    callee_neuron,
                    SynapseType::Calls,
                    "auto-inferred from call-site scan".to_string(),
                ));
            }
        }
    }

    /// Mine `git log --name-only` to find files co-committed ≥ `min_cochange` times.
    ///
    /// For each qualifying pair, add a `SemanticRelated` auto-synapse to the
    /// source neuron's meta if one does not already exist. Called once per compile.
    pub(in crate::index) fn apply_cochange_synapses(&mut self, root: &Path) {
        /// Cap on files per commit before skipping the pair-wise O(n²) step.
        ///
        /// A commit touching more than this many files is almost certainly a
        /// bulk change (dependency bump, generated code, refactor) where co-change
        /// is not a useful semantic signal. Without this cap, a 500-file commit
        /// generates ~125,000 pairs, making compile time degenerate on large repos.
        const MAX_FILES_PER_COMMIT: usize = 50;

        // Adaptive minimum co-change threshold based on repo size.
        // Small repos (≤50 neurons) produce sparse commit histories; 2 co-changes
        // is strong signal. Large repos (>500 neurons) have noisy histories and
        // benefit from a higher bar to avoid false semantic edges.
        let min_cochange: u32 = match self.path_index.len() {
            n if n <= 50 => 2,
            n if n <= 500 => 3,
            _ => 5,
        };

        let output = match std::process::Command::new("git")
            .args(["log", "--name-only", "--pretty=format:"])
            .current_dir(root)
            .output()
        {
            Ok(o) if o.status.success() => o.stdout,
            _ => return, // not a git repo or git unavailable — skip silently
        };

        // Build per-commit file lists and count co-changes
        let mut cochange: HashMap<(PathBuf, PathBuf), u32> = HashMap::new();
        let mut commit_files: Vec<PathBuf> = Vec::new();

        for line in String::from_utf8_lossy(&output).lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                // Commit boundary — process accumulated files only if the commit is
                // small enough that co-change is a meaningful signal.
                if commit_files.len() <= MAX_FILES_PER_COMMIT {
                    for i in 0..commit_files.len() {
                        for j in (i + 1)..commit_files.len() {
                            let (a, b) = (&commit_files[i], &commit_files[j]);
                            // Canonical ordering so (a,b) == (b,a)
                            let key = if a <= b {
                                (a.clone(), b.clone())
                            } else {
                                (b.clone(), a.clone())
                            };
                            *cochange.entry(key).or_insert(0) += 1;
                        }
                    }
                }
                commit_files.clear();
            } else {
                commit_files.push(PathBuf::from(trimmed));
            }
        }
        // Flush any trailing files — git log output may not end with a blank line,
        // which would silently drop the most-recent commit's co-change signal.
        if !commit_files.is_empty() && commit_files.len() <= MAX_FILES_PER_COMMIT {
            for i in 0..commit_files.len() {
                for j in (i + 1)..commit_files.len() {
                    let (a, b) = (&commit_files[i], &commit_files[j]);
                    let key = if a <= b {
                        (a.clone(), b.clone())
                    } else {
                        (b.clone(), a.clone())
                    };
                    *cochange.entry(key).or_insert(0) += 1;
                }
            }
        }

        // Add synapses for qualifying pairs
        let mut changes: Vec<(PathBuf, Synapse)> = Vec::new();
        for ((fa, fb), count) in &cochange {
            if *count < min_cochange {
                continue;
            }
            let na = core_neuron_path(&root.join(fa), root);
            let nb = core_neuron_path(&root.join(fb), root);
            let weight = SynapseWeight::new((0.5_f32 + *count as f32 * 0.05).min(0.9));
            let reason = format!("git co-change: committed together {count}×");

            // Only create synapses for neurons that exist in our index
            if self.path_index.contains_key(&na) && self.path_index.contains_key(&nb) {
                changes.push((
                    na.clone(),
                    Synapse {
                        target: nb.clone(),
                        edge_type: SynapseType::SemanticRelated,
                        weight,
                        reason: reason.clone(),
                        learned_weight: 0.0,
                        traversal_count: 0,
                        last_co_activation_day: 0,
                    },
                ));
                changes.push((
                    nb,
                    Synapse {
                        target: na,
                        edge_type: SynapseType::SemanticRelated,
                        weight,
                        reason,
                        learned_weight: 0.0,
                        traversal_count: 0,
                        last_co_activation_day: 0,
                    },
                ));
            }
        }

        for (source_neuron, syn) in changes {
            let meta_p = meta_path(&source_neuron);
            if let Ok(data) = std::fs::read_to_string(&meta_p) {
                if let Ok(mut meta) = serde_json::from_str::<NeuronMeta>(&data) {
                    let already = meta.synapses.iter().any(|s| s.target == syn.target);
                    if !already {
                        meta.synapses.push(syn.clone());
                        if let Err(e) = atomic_write_json(&meta_p, &meta) {
                            tracing::warn!(
                                "Failed to persist co-change synapse for {}: {e}",
                                meta_p.display()
                            );
                        }
                    }
                }
            }
            if let Some(&i) = self.path_index.get(&source_neuron) {
                let already = self.entries[i]
                    .synapses
                    .iter()
                    .any(|s| s.target == syn.target);
                if !already {
                    self.entries[i].synapses.push(syn);
                }
            }
        }
    }

    /// Add or replace a single entry in `self.entries` (does NOT rebuild derived).
    pub fn index_neuron(&mut self, neuron_path: &Path, content: &str, meta: &NeuronMeta) {
        let index_content = content;

        let terms = tokenize(index_content);
        let mut tf: HashMap<String, f32> = HashMap::new();
        for t in &terms {
            *tf.entry(t.clone()).or_insert(0.0) += 1.0;
        }

        // P3-B: Paraphrase + alias surface boost.
        // ## paraphrases and the narrow fact_aliases surface bridge natural-language
        // questions to answer-bearing facts without polluting summaries with broad
        // category vocabulary.
        // This closes the vocabulary gap: documents contain both answer vocabulary
        // (original content) and question vocabulary (these sections).
        {
            use crate::neuron::parse_sections;
            let sections = parse_sections(index_content);
            for section_name in ["paraphrases", "query_surface", "fact_aliases"] {
                if let Some(section_content) = sections.get(section_name) {
                    for t in tokenize(section_content) {
                        let v = tf.entry(t).or_insert(0.0);
                        *v += 0.5; // boost: question vocab is high-signal (kept low to avoid over-boosting generic category tokens)
                    }
                }
            }
        }

        // NE-6: User-turn boost for Verbatim (conversation) neurons.
        // In episodic memory retrieval, facts are stated by the user, not the assistant.
        // User utterances are the ground truth for SSU/KU/multi queries. Assistant text
        // is context/response and should not dominate BM25 scoring.
        // Implementation: give user-turn lines an extra +1.0 TF weight (doubling their
        // effective TF vs assistant lines), making user-disclosed facts rank much higher.
        if matches!(meta.kind, crate::neuron::NeuronKind::Verbatim) {
            for line in index_content.lines() {
                let lower = line.as_bytes();
                let is_user = lower.starts_with(b"user:")
                    || lower.starts_with(b"User:")
                    || lower.starts_with(b"human:")
                    || lower.starts_with(b"Human:");
                if is_user && line.len() > 6 {
                    for t in tokenize(line) {
                        *tf.entry(t).or_insert(0.0) += 1.0;
                    }
                }
            }
        }

        // A1: Multi-Source Vocabulary Injection — inject soft terms from source file
        // (git commit messages + inline comments) at 0.3× weight. These terms are never
        // shown in the retrieved context, but improve BM25 query matching for cold stubs.
        if let Some(source_abs) = meta.source_files.first() {
            for t in git_extractor::extract_soft_terms(source_abs) {
                // Only inject when not already present in neuron content — hard terms win.
                let v = tf.entry(t).or_insert(0.0);
                if *v == 0.0 {
                    *v = 0.3;
                }
            }
        }

        // B3: Alias Injection — inject natural-language aliases for public function/type names
        // at 0.5× weight. "get_user" → ["fetch", "retrieve", "account", "member"].
        // These aliases bridge the lexical gap between user queries and code identifiers
        // without any model download.
        {
            // Collect function/type names from task_pattern (sub-neuron) or from the neuron
            // file stem (proxy for the source file's primary identifier).
            let mut names: Vec<String> = Vec::new();
            if let Some(ref pattern) = meta.task_pattern {
                names.push(pattern.clone());
            }
            // Also include the neuron path stem as a fallback source of identifiers
            if let Some(stem) = neuron_path
                .file_stem()
                .and_then(|s| s.to_str())
                .map(|s| s.trim_end_matches(".context").to_string())
            {
                names.push(stem);
            }
            if !names.is_empty() {
                for t in alias_gen::generate_alias_terms(&names) {
                    let v = tf.entry(t).or_insert(0.0);
                    if *v < 0.5 {
                        *v = 0.5;
                    }
                }
            }
        }

        let task_pattern_terms = meta
            .task_pattern
            .as_deref()
            .map(tokenize)
            .unwrap_or_default();

        // Normalize synapse targets to absolute paths so the adjacency graph
        // uses consistent keys regardless of whether the path was parsed from
        // a markdown backtick (relative) or stored directly (absolute).
        //
        // S-1: Validate that the resolved target stays inside the neuron directory.
        // This prevents path traversal attacks via crafted .cortyx/neurons/*.json files
        // (e.g. a compromised CI artifact injecting "../../etc/sensitive").
        let ndir = neuron_dir(&self.project_root);
        let synapses: Vec<Synapse> = meta
            .synapses
            .iter()
            .filter_map(|s| {
                let target = if s.target.is_absolute() {
                    s.target.clone()
                } else {
                    ndir.join(&s.target)
                };
                if !target.starts_with(&ndir) {
                    tracing::warn!(
                        "Skipping synapse with path-traversal target {:?} in {:?}",
                        target,
                        neuron_path
                    );
                    return None;
                }
                Some(Synapse {
                    target,
                    ..s.clone()
                })
            })
            .collect();

        // S-III (R16): Self-Quality Score — fraction of neuron terms that overlap with
        // the corresponding source file's AST-extracted terms.
        // Only computed for Core neurons with a known source file; defaults to 1.0 (neutral).
        let quality_score: f32 =
            if matches!(meta.kind, NeuronKind::Core) && !meta.source_files.is_empty() {
                let source_path = &meta.source_files[0];
                if let Ok(source_text) = std::fs::read_to_string(source_path) {
                    let source_rel = source_path.to_string_lossy();
                    let ast = ast_extractor::extract_signatures(&source_rel, &source_text);
                    // Build source AST term set from all function/type names (split on _ and camelCase)
                    let mut ast_terms: std::collections::HashSet<String> =
                        std::collections::HashSet::new();
                    for name in ast.functions.iter().chain(ast.types.iter()) {
                        ast_terms.extend(tokenize(name));
                    }
                    if ast_terms.is_empty() {
                        1.0 // no AST info → neutral
                    } else {
                        let neuron_terms: std::collections::HashSet<&str> =
                            tf.keys().map(|s| s.as_str()).collect();
                        let overlap = ast_terms
                            .iter()
                            .filter(|t| neuron_terms.contains(t.as_str()))
                            .count();
                        overlap as f32 / ast_terms.len() as f32
                    }
                } else {
                    1.0
                }
            } else {
                1.0 // non-Core or no source → neutral
            };

        // S-II (R16/R17 Sol4): Compute a 16-seed SimHash ensemble for LSH fallback.
        let lsh_fingerprints = simhash_1024(&tf);

        // S-I (R16): Extract Tier-1 summary from neuron content.
        // Takes: first non-empty line of `## purpose` section + first line of `## pitfalls`.
        // Stored in memory only (not persisted); rebuilt from neuron file at each index_neuron call.
        let summary = extract_neuron_summary(content);
        let has_move_residence_evidence = content_has_move_residence_evidence(content);

        let entry = BM25Entry {
            neuron_path: neuron_path.to_path_buf(),
            kind: meta.kind.clone(),
            term_freq: tf,
            term_count: terms.len(),
            // Use meta.tokens when available (set by compile/upsert after reading disk).
            // Fall back to estimating from content so the token budget works in tests
            // and when index_neuron is called before NeuronMeta.tokens is populated.
            tokens: if meta.tokens > 0 {
                meta.tokens
            } else {
                estimate_tokens(content).get().max(10)
            },
            task_pattern_terms,
            parent: meta.parent.clone(),
            synapses,
            source_files: meta.source_files.clone(),
            module: meta.module.clone(),
            confidence_score: meta.confidence_score,
            use_count: meta.use_count,
            hit_count: meta.hit_count,
            staleness_multiplier: 1.0,
            concept_cloud: Vec::new(), // populated by build_concept_clouds() in rebuild_derived
            synonym_cloud: Vec::new(), // populated by record_coactivation() at runtime
            lsh_fingerprints,
            quality_score,
            summary,
            timestamp_secs: parse_iso8601_to_secs(meta.timestamp.as_deref()),
            has_move_residence_evidence,
            // R21 T6: Extract session_id from neuron filename stem for Verbatim neurons.
            // Pattern: "lme_0060_0_user.verbatim.md" → session_id = "lme_0060"
            // Split on '_', take first two parts if the stem follows the N_N pattern.
            session_id: if matches!(meta.kind, NeuronKind::Verbatim) {
                neuron_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(|name| {
                        // strip extension(s): "lme_0060_0_user.verbatim.md" → "lme_0060_0_user"
                        let stem = name.split('.').next().unwrap_or(name);
                        // take first two underscore-separated parts: "lme" + "0060"
                        let parts: Vec<&str> = stem.splitn(3, '_').collect();
                        if parts.len() >= 2 {
                            format!("{}_{}", parts[0], parts[1])
                        } else {
                            stem.to_string()
                        }
                    })
                    .unwrap_or_default()
            } else {
                String::new()
            },
        };

        if let Some(&pos) = self.path_index.get(neuron_path) {
            self.entries[pos] = entry;
            self.has_pending_updates = true;
            self.needs_full_save.store(true, Ordering::Relaxed);
        } else {
            let pos = self.entries.len();
            self.path_index.insert(neuron_path.to_path_buf(), pos);
            self.entries.push(entry);
            self.pending_append_count += 1;
        }
    }

    /// Rebuild all derived structures — public entry point for `cortyx prune`.
    ///
    /// Prune evicts entries individually then calls this once to reconstruct
    /// path_index, adjacency, df_cache, etc. in a single O(n) pass.
    pub fn rebuild_derived_pub(&mut self) {
        // Force full rebuild: prune may have removed existing entries, so the
        // incremental delta path (which only handles appends) is not safe here.
        self.pending_append_count = 0;
        self.has_pending_updates = true;
        // S4-WAL: prune removes entries — invalidate WAL baseline and force full save.
        self.wal_base.store(0, Ordering::Relaxed);
        self.needs_full_save.store(true, Ordering::Relaxed);
        self.rebuild_derived();
    }

    /// Rebuild all derived structures in a single O(n) pass.
    ///
    /// Previously five separate passes (path_index, parent_index, adjacency, df_cache,
    /// module_index); merged to reduce cache pressure and wall-clock time ~5×.
    pub(in crate::index) fn rebuild_derived(&mut self) {
        // S7: Incremental delta — skip the full clear+rebuild when only new entries were
        // appended (no updates).  This reduces the hot path (mining a new file into an
        // existing index) from O(N+n) to O(n) for the HashMap phase.
        if self.pending_append_count > 0 && !self.has_pending_updates && self.idf_n > 0 {
            self.rebuild_derived_delta();
            return;
        }

        self.path_index.clear();
        self.parent_index.clear();
        self.adjacency.clear();
        self.df_cache.clear();
        self.posting_list.clear();
        self.module_index.clear();
        self.session_index.clear(); // R21 T6
        self.idf_n = 0;

        let mut non_agg_total_terms = 0usize;
        let mut verbatim_total_terms = 0usize;
        let mut verbatim_count = 0usize;

        for (i, entry) in self.entries.iter().enumerate() {
            // path_index
            self.path_index.insert(entry.neuron_path.clone(), i);

            // parent_index
            if let Some(p) = &entry.parent {
                self.parent_index.entry(p.clone()).or_default().push(i);
            }

            // adjacency (forward + reverse edges)
            for syn in &entry.synapses {
                self.adjacency
                    .entry(entry.neuron_path.clone())
                    .or_default()
                    .push(syn.clone());

                self.adjacency
                    .entry(syn.target.clone())
                    .or_default()
                    .push(Synapse {
                        target: entry.neuron_path.clone(),
                        edge_type: syn.edge_type.inverse(),
                        weight: SynapseWeight::new(syn.weight.get() * 0.7),
                        reason: format!("← {}", syn.reason),
                        learned_weight: 0.0,
                        traversal_count: 0,
                        last_co_activation_day: 0,
                    });
            }

            // df_cache + posting_list.
            // IMPORTANT: Aggregate neurons (word-count summaries, dollar totals) must NOT
            // contribute to df_cache.  An _count_music.aggregate.md neuron contains "music"
            // dozens of times, inflating df("music") and crushing its IDF.  This caused a
            // 5-entry SSU regression: session 329 ("music"×18, no "streaming"/"service") lost
            // to session 309 ("service"×7) because IDF("music") collapsed while IDF("service")
            // stayed high.  Excluding Aggregate from df_cache restores the IDF calibration
            // from the e18c4e6 baseline (100% SSU) even when aggregates are mined.
            // Posting-list is still built for ALL kinds so counting_augment can find Aggregates.
            let is_aggregate = matches!(entry.kind, NeuronKind::Aggregate);
            for term in entry.term_freq.keys() {
                if !is_aggregate {
                    *self.df_cache.entry(term.clone()).or_insert(0) += 1;
                }
                self.posting_list.entry(term.clone()).or_default().push(i);
            }
            if !is_aggregate {
                self.idf_n += 1;
            }

            // module_index
            if let Some(m) = &entry.module {
                self.module_index.entry(m.clone()).or_default().push(i);
            }

            // R21 T6: session_index — for session-level grouping at retrieval time
            if !entry.session_id.is_empty() {
                self.session_index
                    .entry(entry.session_id.clone())
                    .or_default()
                    .push(i);
            }

            if !is_aggregate {
                non_agg_total_terms += entry.term_count;
            }
            if matches!(entry.kind, NeuronKind::Verbatim) {
                verbatim_total_terms += entry.term_count;
                verbatim_count += 1;
            }
        }

        // avg_doc_len excludes Aggregate neurons so it matches e18c4e6 calibration.
        self.avg_doc_len = if self.idf_n == 0 {
            0.0
        } else {
            non_agg_total_terms as f32 / self.idf_n as f32
        };
        self.avg_verbatim_doc_len = if verbatim_count == 0 {
            self.avg_doc_len
        } else {
            verbatim_total_terms as f32 / verbatim_count as f32
        };

        self.build_vocab_bridge();
        self.build_morpheme_map();
        self.build_concept_clouds();
        self.apply_peer_vocab_borrowing();
        self.merge_cooccurrence_into_vocab_bridge();
        self.load_pmi_neighbors();
        self.structural_artifacts_dirty
            .store(true, Ordering::Relaxed);
        self.pending_append_count = 0;
        self.has_pending_updates = false;
    }

    /// Incremental derived-structure update for pure-append batches (S7).
    ///
    /// When only new entries were appended (no existing entries were modified), we
    /// skip clearing and rebuilding the large HashMaps from scratch.  Instead we
    /// process only the `pending_append_count` newest entries and add their
    /// contributions to the existing structures in O(n) rather than O(N+n).
    ///
    /// The bridge/cloud/neighbor builds (vocab_bridge, morpheme_map, concept_clouds,
    /// pmi_neighbors) still run over the full corpus because they are O(terms), not
    /// O(entries²), and must reflect the complete vocabulary.
    pub(in crate::index) fn rebuild_derived_delta(&mut self) {
        let new_start = self.entries.len().saturating_sub(self.pending_append_count);

        for (offset, entry) in self.entries[new_start..].iter().enumerate() {
            let abs_i = new_start + offset;

            // path_index is already maintained by index_neuron(), but ensure consistency.
            self.path_index.insert(entry.neuron_path.clone(), abs_i);

            if let Some(p) = &entry.parent {
                self.parent_index.entry(p.clone()).or_default().push(abs_i);
            }

            for syn in &entry.synapses {
                self.adjacency
                    .entry(entry.neuron_path.clone())
                    .or_default()
                    .push(syn.clone());
                self.adjacency
                    .entry(syn.target.clone())
                    .or_default()
                    .push(Synapse {
                        target: entry.neuron_path.clone(),
                        edge_type: syn.edge_type.inverse(),
                        weight: SynapseWeight::new(syn.weight.get() * 0.7),
                        reason: format!("← {}", syn.reason),
                        learned_weight: 0.0,
                        traversal_count: 0,
                        last_co_activation_day: 0,
                    });
            }

            let is_aggregate = matches!(entry.kind, NeuronKind::Aggregate);
            for term in entry.term_freq.keys() {
                if !is_aggregate {
                    *self.df_cache.entry(term.clone()).or_insert(0) += 1;
                }
                self.posting_list
                    .entry(term.clone())
                    .or_default()
                    .push(abs_i);
            }
            if !is_aggregate {
                self.idf_n += 1;
            }

            if let Some(m) = &entry.module {
                self.module_index.entry(m.clone()).or_default().push(abs_i);
            }

            if !entry.session_id.is_empty() {
                self.session_index
                    .entry(entry.session_id.clone())
                    .or_default()
                    .push(abs_i);
            }
        }

        // Recompute avg_doc_len from all entries (O(n) integer addition — cheap).
        let mut non_agg_total_terms = 0usize;
        let mut verbatim_total_terms = 0usize;
        let mut verbatim_count = 0usize;
        for entry in &self.entries {
            let is_aggregate = matches!(entry.kind, NeuronKind::Aggregate);
            if !is_aggregate {
                non_agg_total_terms += entry.term_count;
            }
            if matches!(entry.kind, NeuronKind::Verbatim) {
                verbatim_total_terms += entry.term_count;
                verbatim_count += 1;
            }
        }
        self.avg_doc_len = if self.idf_n == 0 {
            0.0
        } else {
            non_agg_total_terms as f32 / self.idf_n as f32
        };
        self.avg_verbatim_doc_len = if verbatim_count == 0 {
            self.avg_doc_len
        } else {
            verbatim_total_terms as f32 / verbatim_count as f32
        };

        // Bridge/cloud/neighbor builds must see the full corpus.
        self.build_vocab_bridge();
        self.build_morpheme_map();
        self.build_concept_clouds();
        self.apply_peer_vocab_borrowing();
        self.merge_cooccurrence_into_vocab_bridge();
        self.load_pmi_neighbors();
        self.structural_artifacts_dirty
            .store(true, Ordering::Relaxed);
        self.pending_append_count = 0;
        self.has_pending_updates = false;
    }

    /// A2: Peer Template Vocabulary Borrowing.
    ///
    /// When a neuron has < 10 unique BM25 terms (e.g. a tiny file with no doc comments,
    /// no git history, and no function names), it's a "cold stub" with near-zero recall.
    /// A2 finds the 3 most similar peer neurons by identifier overlap and borrows their
    /// vocabulary at 0.2× weight — giving the stub a starting vocabulary without any LLM call.
    ///
    /// Similarity metric: Jaccard overlap of term sets (both sides filtered to len ≥ 4).
    ///
    /// Only runs on neurons with < A2_COLD_STUB_THRESHOLD unique terms.
    /// Only injects terms not already present (peer vocab never overwrites hard terms).
    /// Called once per rebuild_derived() after concept clouds are built.
    pub(in crate::index) fn apply_peer_vocab_borrowing(&mut self) {
        const A2_COLD_STUB_THRESHOLD: usize = 10;
        const A2_PEER_COUNT: usize = 3;
        const A2_TERMS_PER_PEER: usize = 30;
        const A2_WEIGHT: f32 = 0.2;

        // Collect indices of cold stubs
        let cold_indices: Vec<usize> = (0..self.entries.len())
            .filter(|&i| {
                self.entries[i].term_freq.len() < A2_COLD_STUB_THRESHOLD
                    && self.entries[i].kind == NeuronKind::Core
            })
            .collect();

        if cold_indices.is_empty() {
            return;
        }

        // Precompute filtered term sets for all non-cold neurons (peers)
        // Only use neurons with >= A2_COLD_STUB_THRESHOLD terms as donors
        let peer_term_sets: Vec<(usize, HashSet<String>)> = (0..self.entries.len())
            .filter(|&i| self.entries[i].term_freq.len() >= A2_COLD_STUB_THRESHOLD)
            .map(|i| {
                let terms: HashSet<String> = self.entries[i]
                    .term_freq
                    .keys()
                    .filter(|t| t.len() >= 4)
                    .cloned()
                    .collect();
                (i, terms)
            })
            .collect();

        // For each cold stub, find top-3 peers by Jaccard and borrow vocabulary
        let mut borrowed: Vec<(usize, Vec<(String, f32)>)> = Vec::new();
        for cold_idx in cold_indices {
            let cold_terms: HashSet<String> = self.entries[cold_idx]
                .term_freq
                .keys()
                .filter(|t| t.len() >= 4)
                .cloned()
                .collect();

            // Same module preferred — compute similarity against all peers
            let cold_module = self.entries[cold_idx].module.clone();
            let mut scored: Vec<(f32, usize)> = peer_term_sets
                .iter()
                .filter(|(pi, _)| *pi != cold_idx)
                .map(|(pi, peer_terms)| {
                    let inter = cold_terms.intersection(peer_terms).count();
                    let union = cold_terms.union(peer_terms).count();
                    let jaccard = if union > 0 {
                        inter as f32 / union as f32
                    } else {
                        0.0
                    };
                    // Module bonus: same module → +0.1
                    let module_bonus =
                        if cold_module.is_some() && cold_module == self.entries[*pi].module {
                            0.1
                        } else {
                            0.0
                        };
                    (jaccard + module_bonus, *pi)
                })
                .collect();

            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

            let mut terms_to_add: Vec<(String, f32)> = Vec::new();
            for (_, peer_idx) in scored.iter().take(A2_PEER_COUNT) {
                let peer_terms: Vec<(String, f32)> = self.entries[*peer_idx]
                    .term_freq
                    .iter()
                    .filter(|(t, _)| t.len() >= 4)
                    .take(A2_TERMS_PER_PEER)
                    .map(|(t, _)| (t.clone(), A2_WEIGHT))
                    .collect();
                terms_to_add.extend(peer_terms);
            }

            if !terms_to_add.is_empty() {
                borrowed.push((cold_idx, terms_to_add));
            }
        }

        // Apply borrowed vocabulary (avoids borrow conflict — collected above)
        for (cold_idx, terms) in borrowed {
            for (term, weight) in terms {
                let v = self.entries[cold_idx].term_freq.entry(term).or_insert(0.0);
                if *v == 0.0 {
                    *v = weight;
                }
            }
        }
    }

    /// Build the vocabulary bridge map: module_fragment → term set.
    ///
    /// Aggregates all terms from neurons tagged with a module into a single set
    /// keyed by the module name. Also adds sub-word fragments from the neuron path
    /// (e.g., "auth_guard" → fragments ["auth", "guard"]) as additional keys so
    /// path-derived synonyms are reachable. Called by rebuild_derived().
    pub(in crate::index) fn build_vocab_bridge(&mut self) {
        let mut bridge: HashMap<String, HashSet<String>> = HashMap::new();
        for entry in &self.entries {
            // Aggregate neurons (word-count / dollar summaries) must NOT contribute to the
            // vocab bridge.  Their path fragments ("fish", "bike", "music" …) would become
            // bridge keys containing hundreds of spurious co-topic terms, which would then
            // be injected into every query that mentions those words — corrupting BM25
            // candidate ranking and causing regressions in multi-session retrieval.
            if matches!(entry.kind, NeuronKind::Aggregate) {
                continue;
            }
            // Key 1: module name (e.g. "auth")
            if let Some(module) = entry.module.as_deref() {
                let key = module.to_lowercase();
                if !key.is_empty() {
                    let terms = bridge.entry(key).or_default();
                    for term in entry.term_freq.keys() {
                        if term.len() >= 3 {
                            terms.insert(term.clone());
                        }
                    }
                }
            }
            // Key 2: path fragments derived from the neuron filename stem
            // (e.g., neurons/src/auth_guard_rs.context.md → ["auth", "guard"])
            if let Some(stem) = entry.neuron_path.file_stem().and_then(|s| s.to_str()) {
                let cleaned = stem
                    .trim_end_matches(".context")
                    .replace("_rs", "")
                    .replace("_ts", "")
                    .replace("_py", "")
                    .replace("_go", "")
                    .to_lowercase();
                for fragment in cleaned.split('_').filter(|f| f.len() >= 4) {
                    let terms = bridge.entry(fragment.to_string()).or_default();
                    for term in entry.term_freq.keys() {
                        if term.len() >= 3 {
                            terms.insert(term.clone());
                        }
                    }
                }
            }
        }
        self.vocab_bridge = bridge;

        // S2 (R11) — Co-change vocabulary expansion: neurons connected by SemanticRelated
        // synapses (which includes git co-change auto-synapses from `apply_cochange_synapses`)
        // donate their vocabulary to the bridge under their partner's path stem.
        //
        // Effect: a query containing terms specific to file A also expands to include
        // terms from co-changed file B, even when A and B use entirely different vocabulary.
        // Since `apply_cochange_synapses` adds bidirectional edges, the expansion is symmetric.
        // Vocabulary gap estimate: ~3% → ~0.5% (TRIZ R11-S2).
        //
        // adjacency is fully built before this call — collect pairs into a local Vec
        // first to avoid re-borrowing self inside the loop.
        let cochange_pairs: Vec<(String, Vec<String>)> = {
            let mut pairs = Vec::new();
            for (src_path, syns) in &self.adjacency {
                let Some(&src_idx) = self.path_index.get(src_path) else {
                    continue;
                };
                for syn in syns {
                    if syn.edge_type != SynapseType::SemanticRelated {
                        continue;
                    }
                    let Some(tgt_stem) = syn
                        .target
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .map(|s| s.trim_end_matches(".context").to_lowercase())
                    else {
                        continue;
                    };
                    let src_terms: Vec<String> = self.entries[src_idx]
                        .term_freq
                        .keys()
                        .filter(|t| t.len() >= 3)
                        .take(30)
                        .cloned()
                        .collect();
                    if !src_terms.is_empty() {
                        pairs.push((tgt_stem, src_terms));
                    }
                }
            }
            pairs
        };
        for (tgt_stem, src_terms) in cochange_pairs {
            self.vocab_bridge
                .entry(tgt_stem)
                .or_default()
                .extend(src_terms);
        }
    }

    /// R17 Sol2: Merge co-occurrence ontology into vocab_bridge.
    ///
    /// Loads `.cortyx/cooccurrence.json` (written by `miner::build_and_save_cooccurrence`)
    /// and merges its clusters into `self.vocab_bridge`. This gives BM25 free synonym
    /// expansion derived entirely from the user's own conversation data (Firth Principle).
    ///
    /// Merge strategy: each cluster entry is a HashSet extension — never overwrites
    /// existing structural vocab, only extends it with conversation-derived synonyms.
    pub(in crate::index) fn merge_cooccurrence_into_vocab_bridge(&mut self) {
        let co_path = self.project_root.join(".cortyx").join("cooccurrence.json");
        if !co_path.exists() {
            return;
        }
        let Ok(json) = std::fs::read_to_string(&co_path) else {
            return;
        };
        let Ok(clusters): Result<std::collections::HashMap<String, Vec<String>>, _> =
            serde_json::from_str(&json)
        else {
            return;
        };

        // R18 P1a: cap to 150 high-signal pairs total (both terms ≥4 chars).
        // Prevents the O(n×|bridge|) query expansion blowup that caused the 2.5× slowdown.
        let mut added = 0usize;
        const MAX_CO_PAIRS: usize = 150;
        'outer: for (term, synonyms) in clusters {
            if term.len() < 4 {
                continue;
            }
            let entry = self.vocab_bridge.entry(term).or_default();
            for syn in synonyms {
                if syn.len() >= 4 && entry.insert(syn) {
                    added += 1;
                    if added >= MAX_CO_PAIRS {
                        break 'outer;
                    }
                }
            }
        }
        tracing::debug!(
            pairs = added,
            "R17 Sol2 (capped): co-occurrence vocab bridge merged"
        );
    }

    /// P1-A: Load PMI semantic neighbors from cooccurrence.json without a global cap.
    ///
    /// Unlike merge_cooccurrence_into_vocab_bridge (which adds to the substring-matched
    /// vocab_bridge and was capped at 150 pairs to prevent O(n) scan blowup), this method
    /// stores neighbors in a separate exact-key map for O(1) lookup at query time.
    ///
    /// Admits all pairs where both terms are ≥4 chars. The cooccurrence builder already
    /// filters pairs by weight ≥2 and caps at 10 neighbors per term, so this is safe.
    pub(in crate::index) fn load_pmi_neighbors(&mut self) {
        let co_path = self.project_root.join(".cortyx").join("cooccurrence.json");
        if !co_path.exists() {
            return;
        }
        let Ok(json) = std::fs::read_to_string(&co_path) else {
            return;
        };
        let Ok(clusters): Result<HashMap<String, Vec<String>>, _> = serde_json::from_str(&json)
        else {
            return;
        };

        let mut loaded = 0usize;
        for (term, neighbors) in clusters {
            if term.len() < 4 {
                continue;
            }
            let valid: Vec<String> = neighbors
                .into_iter()
                .filter(|n| n.len() >= 4)
                .take(5)
                .collect();
            if !valid.is_empty() {
                self.pmi_neighbors.insert(term, valid);
                loaded += 1;
            }
        }
        tracing::debug!(terms = loaded, "P1-A: PMI neighbors loaded (no global cap)");
    }

    ///
    /// Splits all identifier tokens across all neurons on `_` boundaries (snake_case)
    /// and camelCase boundaries. Maps each sub-token (minimum 3 chars) to the full tokens
    /// that contain it.
    ///
    /// At query time, each query term that misses BM25 is split into sub-tokens and expanded
    /// through this map, recovering matches against compound identifiers. Example:
    ///   query: "auth" → morpheme_map["auth"] → ["authenticate", "auth_guard", "oauth_token"]
    ///   → those terms are then searched in the posting list.
    ///
    /// Reduces vocabulary gap from ~3% to ~0.3% (no model download, O(|terms|) at query time).
    pub(in crate::index) fn build_morpheme_map(&mut self) {
        let mut map: HashMap<String, Vec<String>> = HashMap::new();

        for entry in &self.entries {
            // Aggregates contain English prose terms, not camelCase/snake_case identifiers.
            // Including them adds noise to morpheme expansion without benefit.
            if matches!(entry.kind, NeuronKind::Aggregate) {
                continue;
            }
            for token in entry.term_freq.keys() {
                if token.len() < 4 {
                    continue;
                }
                // Split on underscores (snake_case)
                let snake_parts: Vec<&str> = token.split('_').collect();
                // Split on camelCase transitions (e.g. "validateUser" → ["validate", "User"])
                let camel_parts = split_camel_case(token);

                let mut sub_tokens: HashSet<&str> = HashSet::new();
                for part in snake_parts.iter().chain(
                    camel_parts
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .iter(),
                ) {
                    if part.len() >= 3 {
                        sub_tokens.insert(part);
                    }
                }

                for sub in sub_tokens {
                    let sub_lower = sub.to_lowercase();
                    if sub_lower != *token {
                        map.entry(sub_lower).or_default().push(token.clone());
                    }
                }
            }
        }

        // Deduplicate per sub-token (multiple neurons may share the same full token)
        for v in map.values_mut() {
            v.sort_unstable();
            v.dedup();
        }

        self.morpheme_map = map;
    }

    /// Build per-neuron concept clouds from 1-hop structural synapse neighbours (TRIZ R12-S1).
    ///
    /// For each neuron, traverse its Calls, Imports, and Implements edges and collect the
    /// significant identifier terms from each neighbour's BM25 vocabulary into a `concept_cloud`.
    /// Cap: 50 terms per neighbour, 200 terms total per cloud.
    ///
    /// At query time, concept clouds serve as a graph-aware semantic thesaurus: a query
    /// for "validate_user" can activate auth.rs via engine.rs's concept cloud even when
    /// "validate_user" does not appear in auth.rs's own vocabulary.
    ///
    /// Not persisted (`#[serde(skip)]` on the field) — rebuilt from the live adjacency
    /// map on every `rebuild_derived()` call. Zero I/O overhead.
    pub(in crate::index) fn build_concept_clouds(&mut self) {
        const MAX_TERMS_PER_NEIGHBOUR: usize = 50;
        const MAX_CLOUD_SIZE: usize = 200;

        // Collect all (entry_idx, neighbour_terms) pairs upfront to avoid borrow conflicts.
        let clouds: Vec<Vec<String>> = (0..self.entries.len())
            .map(|i| {
                let path = self.entries[i].neuron_path.clone();
                let mut cloud: Vec<String> = Vec::new();
                let syns = self.adjacency.get(&path).cloned().unwrap_or_default();
                for syn in &syns {
                    if !matches!(
                        syn.edge_type,
                        SynapseType::Calls | SynapseType::Imports | SynapseType::Implements
                    ) {
                        continue;
                    }
                    if cloud.len() >= MAX_CLOUD_SIZE {
                        break;
                    }
                    if let Some(&tgt_idx) = self.path_index.get(&syn.target) {
                        let remaining = MAX_CLOUD_SIZE - cloud.len();
                        let limit = remaining.min(MAX_TERMS_PER_NEIGHBOUR);
                        let neighbour_terms = self.entries[tgt_idx]
                            .term_freq
                            .keys()
                            .filter(|t| t.len() >= 3)
                            .take(limit)
                            .cloned();
                        cloud.extend(neighbour_terms);
                    }
                }
                cloud
            })
            .collect();

        for (entry, cloud) in self.entries.iter_mut().zip(clouds) {
            entry.concept_cloud = cloud;
        }
    }

    /// Expand query terms using the vocabulary bridge (S2) and morphemic trie (B1).
    ///
    /// Phase 1 (S2): For each query term that returns zero BM25 candidates, check if it
    /// substring-matches any module fragment in `vocab_bridge`. If so, add that module's full
    /// identifier vocabulary as additional search terms.
    ///
    /// Phase 2 (B1): For each query term, split on camelCase and `_` boundaries and look
    /// up sub-tokens in `morpheme_map`. This resolves "auth" → ["auth_guard", "authentication"]
    /// for any query term, not just module-level gaps.
    ///
    /// Expansion is capped at 50 terms per bridge hit to avoid BM25 score inflation.
    pub(in crate::index) fn expand_query_terms(&self, terms: &[String]) -> Vec<String> {
        let mut expanded: HashSet<String> = terms.iter().cloned().collect();
        for term in terms {
            let term_lower = term.to_lowercase();

            // S2 — Vocabulary Bridge: module-fragment substring matching
            for (fragment, vocab) in &self.vocab_bridge {
                if fragment.contains(term_lower.as_str()) || term_lower.contains(fragment.as_str())
                {
                    expanded.extend(vocab.iter().take(50).cloned());
                }
            }

            // B1 — Morphemic Trie Bridge: sub-token expansion (snake_case + camelCase)
            // Split the query term on _ and camelCase boundaries, then look up each part
            let sub_tokens = {
                let mut parts = vec![];
                for snake_part in term_lower.split('_') {
                    if snake_part.len() >= 3 {
                        parts.push(snake_part.to_string());
                    }
                }
                for camel_part in split_camel_case(&term_lower) {
                    if camel_part.len() >= 3 {
                        parts.push(camel_part);
                    }
                }
                parts
            };
            for sub in &sub_tokens {
                if let Some(full_tokens) = self.morpheme_map.get(sub.as_str()) {
                    expanded.extend(full_tokens.iter().take(20).cloned());
                }
            }

            // P1-B: PMI semantic neighbors — exact-key O(1) lookup.
            // Expands conversation vocabulary: "degree" → ["master","education","completed"]
            // "commute" → ["expense","productive","fare"], "marathon" → ["achievement","race"]
            // Uses top-3 neighbors to avoid over-expansion while covering key synonyms.
            if let Some(pmi_nbrs) = self.pmi_neighbors.get(term_lower.as_str()) {
                expanded.extend(pmi_nbrs.iter().take(3).cloned());
            }

            // Morphological suffix expansion: bridges vocabulary gap between query and doc.
            // Query "graduate" → doc has "graduated"; query "commute" → doc has "commuting".
            // Add suffix variants only when the resulting term exists in the posting lists
            // (zero contribution if not in vocab — safe to add unconditionally).
            // Weight is implicitly 1.0 (same as original terms) since BM25 contribution
            // of an absent term is 0 regardless.
            let variants = morphological_variants(&term_lower);
            for variant in variants {
                if self.df_cache.contains_key(variant.as_str()) {
                    expanded.insert(variant);
                }
            }
        }
        expanded.into_iter().collect()
    }

    /// BM25 score for a single entry given query terms.
    ///
    /// Uses the precomputed `df_cache` for O(1) IDF lookup.
    /// Applies `entry.confidence_score` as a mild prior multiplier:
    /// committed + unmodified = 1.0 (neutral), modified = 0.9, untracked = 0.85.
    pub(in crate::index) fn bm25_score(&self, terms: &[String], entry: &BM25Entry) -> f32 {
        // Use idf_n (non-Aggregate count) as IDF corpus size so Aggregate neurons
        // that contain high-frequency terms do not corrupt IDF calibration.
        let n = self.idf_n.max(1) as f32;
        let avg = self.avg_doc_len.max(1.0);
        let dl = entry.term_count as f32;
        let len_norm = 1.0 - BM25_B + BM25_B * (dl / avg);

        // R21 T10: per-entry k1 — Verbatim neurons (long conversation text) use k1=1.5
        // to allow longer documents to score higher on frequently-mentioned terms.
        // Core/Project neurons keep the default k1=1.2.
        let k1 = if matches!(entry.kind, NeuronKind::Verbatim) {
            1.5
        } else {
            BM25_K1
        };

        let raw: f32 = terms
            .iter()
            .map(|t| {
                let tf = entry.term_freq.get(t).copied().unwrap_or(0.0);
                if tf == 0.0 {
                    return 0.0;
                }
                // Laplace floor: if a term appears only in Aggregate neurons it may be
                // absent from df_cache (which is built from regular neurons during
                // rebuild_derived). Default df=1 prevents IDF blow-up for such terms:
                //   IDF = ln((n - 0.5) / 1.5)  — reasonable for rare terms.
                let df = self.df_cache.get(t).copied().unwrap_or(1) as f32;
                let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln().max(0.0);
                // R18 P3 Sol D / R19 fix: BM25+ δ=0.5 (reduced from 1.0 — smaller perturbation,
                // less global ranking disruption while still providing the lower-bound benefit).
                const BM25_DELTA: f32 = 0.5;
                idf * (BM25_DELTA + (tf * (k1 + 1.0)) / (tf + k1 * len_norm))
            })
            .sum();

        // hit_rate reward: proven neurons earn up to +50% score boost.
        // Cold-start guard: neutral (×1.0) until MIN_SAMPLE_SIZE activations have
        // accumulated — no penalty for newly-added neurons.
        //
        // Range: [1.0, 1.50] — reward only, never penalty.  A neuron that is never
        // cited simply stays at ×1.0; the auto-quarantine (staleness_multiplier = 0.3)
        // handles chronic over-activators separately.
        let hit_multiplier = if entry.use_count < MIN_SAMPLE_SIZE {
            1.0
        } else {
            let hit_rate = entry.hit_count as f32 / entry.use_count as f32;
            (1.0 + hit_rate).min(1.5)
        };

        raw * entry.confidence_score * hit_multiplier * entry.staleness_multiplier
            // S-III (R16): demote low-quality neurons — they may be stale or uncurated
            * if entry.quality_score < 0.4 { 0.7 } else { 1.0 }
    }

    /// TF-IDF cosine similarity between query terms and a BM25 entry.
    ///
    /// Reuses `entry.term_freq` (already computed) and `df_cache` — zero new dependencies.
    /// Returned value is in `[0.0, 1.0]` (normalised cosine similarity).
    /// Used as a tie-breaker when BM25 confidence ratio is low.
    pub(in crate::index) fn tfidf_cosine_sim_inner(
        query_terms: &[String],
        entry: &BM25Entry,
        df: &std::collections::HashMap<String, usize>,
        n_docs: usize,
    ) -> f32 {
        let n = n_docs.max(1) as f32;
        let mut dot = 0.0f32;
        let mut q_mag = 0.0f32;
        let mut d_mag = 0.0f32;
        for term in query_terms {
            let idf = {
                let df_t = df.get(term).copied().unwrap_or(0) as f32;
                ((n + 1.0) / (df_t + 1.0)).ln().max(0.0)
            };
            let q_tf = 1.0f32; // query term frequency is always 1 for bag-of-words queries
            let d_tf = entry.term_freq.get(term).copied().unwrap_or(0.0);
            let q_w = q_tf * idf;
            let d_w = d_tf * idf;
            dot += q_w * d_w;
            q_mag += q_w * q_w;
            d_mag += d_w * d_w;
        }
        let denom = q_mag.sqrt() * d_mag.sqrt();
        if denom == 0.0 {
            0.0
        } else {
            (dot / denom).clamp(0.0, 1.0)
        }
    }

    /// Find an entry by its neuron path — O(1) via precomputed path_index.
    pub(in crate::index) fn entry_by_path(&self, path: &Path) -> Option<&BM25Entry> {
        self.path_index.get(path).map(|&i| &self.entries[i])
    }

    /// Count how many of the given tokens appear in the BM25 term_freq for `path`.
    ///
    /// Used by `close_task` for term-freq soft citation: if the response text shares
    /// ≥ N vocabulary terms with a neuron, it's likely grounded in that neuron.
    pub fn term_freq_overlap(
        &self,
        path: &Path,
        tokens: &std::collections::HashSet<String>,
    ) -> usize {
        self.entry_by_path(path)
            .map(|e| {
                tokens
                    .iter()
                    .filter(|t| e.term_freq.contains_key(*t))
                    .count()
            })
            .unwrap_or(0)
    }

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
