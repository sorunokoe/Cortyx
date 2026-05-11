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
        0x0590..=0x06FF     // Hebrew
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
