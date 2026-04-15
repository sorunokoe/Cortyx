use anyhow::{Result, bail};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ─── Neuron kinds ─────────────────────────────────────────────────────────────

/// The role a neuron plays in the knowledge graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum NeuronKind {
    #[default]
    /// Per-file AI-curated context stub — the primary neuron type.
    Core,
    /// Proven task-specific chunk extracted from a raw source.
    UseCase,
    /// Raw conversation turn mined verbatim (no LLM curation needed).
    Verbatim,
    /// Cross-file synthesized concept (e.g. "JWT auth flow").
    Concept,
    /// One per project — top-level overview neuron.
    Project,
    /// Mine-time cross-session aggregate: pre-computed count and context snippets
    /// for entities/topics mentioned in ≥3 distinct sessions. Answers "how many
    /// times did I X?" queries in O(1) without runtime graph traversal.
    Aggregate,
}

/// Lifecycle state of a neuron.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum NeuronStatus {
    #[default]
    /// Just created — content is a TODO placeholder, not yet useful for retrieval.
    Stub,
    /// Content has been set by the LLM — ready for activation.
    Fresh,
    /// Source file changed — content may be outdated; re-evolve recommended.
    Stale,
}

// ─── Typed synapse edge ───────────────────────────────────────────────────────

/// The semantic type of a connection between two neurons.
///
/// Each type has an associated relevance multiplier applied during graph
/// traversal — structural edges (Imports, Implements) carry more weight
/// than loose semantic associations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SynapseType {
    #[default]
    /// General content similarity — weakest traversal signal (×0.50)
    SemanticRelated,
    /// A imports / depends on B (×0.80)
    Imports,
    /// A calls functions defined in B (×0.70)
    Calls,
    /// A implements an interface / trait from B (×0.90)
    Implements,
    /// B is the concrete implementation of A's interface (×0.80, reverse of Implements)
    ImplementedBy,
    /// B defines functions that A calls (×0.65, reverse of Calls)
    CalledBy,
    /// A and B hold conflicting information — excluded from co-activation (×0.40)
    Contradicts,
    /// B is the next session / event after A (×0.60)
    TemporalFollows,
    /// B's knowledge was derived from A (×0.70)
    Derived,
    /// Concept neuron → its constituent source files (×1.00, always propagates)
    ConceptExpands,
}

impl SynapseType {
    /// Weight multiplier applied during graph traversal.
    pub fn type_multiplier(&self) -> f32 {
        match self {
            Self::SemanticRelated => 0.50,
            Self::Imports        => 0.80,
            Self::Calls          => 0.70,
            Self::Implements     => 0.90,
            Self::ImplementedBy  => 0.80,
            Self::CalledBy       => 0.65,
            Self::Contradicts    => 0.40,
            Self::TemporalFollows => 0.60,
            Self::Derived        => 0.70,
            Self::ConceptExpands => 1.00,
        }
    }

    /// Return the semantic inverse of this edge type for reverse graph construction.
    pub fn inverse(&self) -> SynapseType {
        match self {
            Self::Implements     => Self::ImplementedBy,
            Self::ImplementedBy  => Self::Implements,
            Self::Calls          => Self::CalledBy,
            Self::CalledBy       => Self::Calls,
            Self::Contradicts    => Self::Contradicts, // symmetric
            // All others collapse to SemanticRelated when reversed
            _ => Self::SemanticRelated,
        }
    }
}

/// A directed, typed, weighted edge in the neuron knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Synapse {
    /// Target neuron path (absolute or relative, must exist in the index).
    pub target: PathBuf,
    /// Semantic type — controls traversal multiplier and directionality.
    pub edge_type: SynapseType,
    /// Relevance weight in [0, 1]. Starts at 0.5; can be set manually via create_synapse.
    pub weight: f32,
    /// Human-readable reason written by the LLM.
    pub reason: String,
    /// Learned traversal weight — starts at `edge_type.type_multiplier()` and updates
    /// via EMA (α = 0.1) from citation signals in `record_hit`. After 10+ traversals,
    /// this weight encodes the actual helpfulness of this specific synapse edge.
    ///
    /// `#[serde(default)]` ensures backward compatibility: old index.json files that
    /// lack this field will deserialize to 0.0, then `effective_weight()` falls back
    /// to `type_multiplier()` so behaviour is identical before any learning occurs.
    #[serde(default)]
    pub learned_weight: f32,
    /// Number of times this synapse was evaluated (target cited or not) — used to
    /// decide when the learned_weight has enough signal to trust.
    #[serde(default)]
    pub traversal_count: u32,
    /// Unix day (days since epoch) of the last co-activation of source + target.
    /// Used by S-VII synapse temporal decay: synapses idle for many days decay toward 0.
    /// `#[serde(default)]` → existing index files default to 0 (treats as never co-activated,
    /// but decay only fires after first co-activation so this is safely conservative).
    #[serde(default)]
    pub last_co_activation_day: u32,
}

impl Synapse {
    pub fn new(target: PathBuf, edge_type: SynapseType, reason: String) -> Self {
        Self {
            target,
            edge_type,
            weight: 0.5,
            reason,
            learned_weight: 0.0, // 0.0 → effective_weight() returns type_multiplier()
            traversal_count: 0,
            last_co_activation_day: 0,
        }
    }

    /// Effective traversal weight, blending the static type multiplier with the
    /// learned weight once enough signal has accumulated.
    ///
    /// Cold-start (traversal_count < 10 or learned_weight == 0.0):
    ///   returns `type_multiplier()` — identical to old behaviour.
    /// Warm (traversal_count ≥ 10):
    ///   blends 50% static + 50% learned, clamped to [0.1, 1.0].
    ///
    /// The blend prevents over-fitting: a synapse that helped in 1 out of 3
    /// traversals doesn't get immediately downweighted to 0.33.
    pub fn effective_weight(&self) -> f32 {
        let base = self.edge_type.type_multiplier();
        if self.traversal_count < 10 || self.learned_weight <= 0.0 {
            return base;
        }
        // Blend: more learned weight trust as evidence accumulates (cap at 50%).
        let blend = 0.5_f32.min(self.traversal_count as f32 / 100.0);
        ((1.0 - blend) * base + blend * self.learned_weight).clamp(0.1, 1.0)
    }
}

// ─── Neuron metadata (sidecar JSON) ──────────────────────────────────────────

/// Sidecar metadata stored beside every `.context.md` neuron.
///
/// Persisted as `<stem>.context.json` adjacent to the Markdown file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronMeta {
    /// Absolute path of the original source file this neuron describes.
    pub source_path: PathBuf,
    pub kind: NeuronKind,
    pub status: NeuronStatus,
    pub source_hash: String,
    /// BLAKE3 hash of the AST signature string (sorted function/type names only).
    /// Compared against `source_hash` to distinguish cosmetic edits (whitespace, doc-comments)
    /// from semantic changes (new/renamed/removed public API). When `source_hash` changes but
    /// `sig_hash` does not, the LLM-curated stub is preserved and only the meta hash is updated.
    #[serde(default)]
    pub sig_hash: Option<String>,
    pub tokens: usize,
    pub last_updated: String,
    pub use_count: u32,
    /// Number of times the LLM confirmed this neuron was actually cited (via cortyx_record_hit).
    /// Used alongside use_count to compute hit_rate = hit_count / use_count.max(1).
    #[serde(default)]
    pub hit_count: u32,
    /// Typed edges to related neurons.
    pub synapses: Vec<Synapse>,
    /// Task pattern phrase (UseCase neurons only).
    pub task_pattern: Option<String>,
    /// Parent Core neuron (UseCase neurons only).
    pub parent: Option<PathBuf>,
    /// Optional project/module tag — used for namespace filtering.
    pub module: Option<String>,
    /// Source files synthesized by this Concept neuron (Concept kind only).
    pub source_files: Vec<PathBuf>,
    /// Speaker label (Verbatim neurons from conversation mining).
    pub speaker: Option<String>,
    /// ISO 8601 timestamp (Verbatim neurons).
    pub timestamp: Option<String>,
    /// Git-derived confidence score (1.0 = committed + unmodified, 0.85 = untracked/WIP).
    /// Applied as a mild BM25 multiplier. Defaults to 1.0 (neutral) on non-git projects.
    /// Clamped to [0.0, 1.0] on deserialization to prevent hand-edited JSON from
    /// silently zeroing or infinitely amplifying neuron scores.
    #[serde(default = "default_confidence", deserialize_with = "deserialize_confidence")]
    pub confidence_score: f32,
    /// E2 (TRIZ R14): One shadow copy per section — stored before any evolve_* call.
    /// Key "_full" = full neuron body before evolve_context.
    /// Key "purpose", "api", etc. = previous section body before evolve_section.
    /// Recovered via `cortyx rollback-section <path> <section>` or the MCP tool.
    /// ~200 bytes overhead per neuron; only the most recent shadow is kept.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub shadow_sections: HashMap<String, String>,
    /// S-XI (R16): Stable UUID — rename-resilient identifier.
    ///
    /// Generated once at neuron creation as a BLAKE3-derived 32-hex-char ID from
    /// the source path + creation timestamp. Persisted in sidecar JSON; survives
    /// file renames. Used at compile time to carry over learned weights + synapse
    /// links when a source file is moved.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uuid: Option<String>,
}

/// Default confidence score for neurons in non-git projects or committed + unmodified files.
pub const DEFAULT_CONFIDENCE: f32 = 1.0;

fn default_confidence() -> f32 {
    DEFAULT_CONFIDENCE
}

fn deserialize_confidence<'de, D: serde::Deserializer<'de>>(d: D) -> Result<f32, D::Error> {
    let v = f32::deserialize(d)?;
    Ok(v.clamp(0.0, 1.0))
}

impl NeuronMeta {
    pub fn new_stub(source: &Path, kind: NeuronKind) -> Self {
        Self {
            source_path: source.to_path_buf(),
            kind,
            status: NeuronStatus::Stub,
            source_hash: String::new(),
            sig_hash: None,
            tokens: 0,
            last_updated: now_iso8601(),
            use_count: 0,
            hit_count: 0,
            synapses: Vec::new(),
            task_pattern: None,
            parent: None,
            module: None,
            source_files: Vec::new(),
            speaker: None,
            timestamp: None,
            confidence_score: DEFAULT_CONFIDENCE,
            shadow_sections: HashMap::new(),
            uuid: Some(generate_neuron_uuid(source)),
        }
    }

    /// Create metadata for a Verbatim neuron (raw conversation chunk).
    ///
    /// Verbatim neurons store the full text and need no LLM curation.
    pub fn new_verbatim_chunk(
        neuron_path: &Path,
        speaker: Option<String>,
        text: &str,
        timestamp: Option<String>,
        module: Option<String>,
    ) -> Self {
        Self {
            source_path: neuron_path.to_path_buf(),
            kind: NeuronKind::Verbatim,
            status: NeuronStatus::Fresh,
            source_hash: String::new(),
            sig_hash: None,
            tokens: estimate_tokens(text),
            last_updated: timestamp.clone().unwrap_or_default(),
            use_count: 0,
            hit_count: 0,
            synapses: Vec::new(),
            task_pattern: None,
            parent: None,
            module,
            source_files: Vec::new(),
            speaker,
            timestamp,
            confidence_score: 1.0,
            shadow_sections: HashMap::new(),
            uuid: Some(generate_neuron_uuid(neuron_path)),
        }
    }
}

// ─── Path helpers ─────────────────────────────────────────────────────────────

/// Root of the neuron store inside a project.
///
/// Example: `/my/project/.cortyx/neurons/`
pub fn neuron_dir(project_root: &Path) -> PathBuf {
    project_root.join(".cortyx").join("neurons")
}

/// Map a source file to its Core neuron path.
///
/// Preserves the directory structure under `.cortyx/neurons/` to prevent
/// flat-file collisions (e.g. `src/engine.rs` and root `src_engine.rs`).
/// Only dots in the filename are replaced with `_` (to avoid ambiguity with
/// the `.context.md` extension).
///
/// Example: `src/engine.rs` → `.cortyx/neurons/src/engine_rs.context.md`
pub fn core_neuron_path(source: &Path, project_root: &Path) -> PathBuf {
    let rel = source.strip_prefix(project_root).unwrap_or(source);
    let parent = rel.parent().unwrap_or(Path::new(""));
    let stem = rel
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .replace('.', "_");
    neuron_dir(project_root).join(parent).join(format!("{stem}.context.md"))
}

/// Map a Core neuron path + function name to its UseCase sub-neuron path.
///
/// Sub-neurons live alongside the Core, prefixed with `fn-`.
///
/// Example: `.cortyx/neurons/src/engine_rs.context.md` + `"validate_user"` →
///          `.cortyx/neurons/src/engine_rs.fn-validate_user.context.md`
pub fn sub_neuron_path(core_path: &Path, fn_name: &str) -> PathBuf {
    let safe_name: String = fn_name
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' })
        .collect();
    let dir = core_path.parent().unwrap_or(Path::new("."));
    let core_stem = core_path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .strip_suffix(".context")
        .map(|s| s.to_owned())
        .unwrap_or_else(|| core_path.file_stem().unwrap_or_default().to_string_lossy().into_owned());
    dir.join(format!("{core_stem}.fn-{safe_name}.context.md"))
}

/// Map a `.context.md` path to its sidecar `.context.json` path.
///
/// Example: `neurons/engine_rs.context.md` → `neurons/engine_rs.context.json`
pub fn meta_path(neuron_md: &Path) -> PathBuf {
    let name = neuron_md.file_name().unwrap_or_default().to_string_lossy();
    let json_name = name
        .strip_suffix(".md")
        .map(|s| format!("{s}.json"))
        .unwrap_or_else(|| format!("{name}.json"));
    neuron_md.parent().unwrap_or(Path::new(".")).join(json_name)
}

// ─── File scanner filter ──────────────────────────────────────────────────────

/// Returns `true` for paths that should not get neurons.
///
/// Expects a **relative** path from the project root. Calling with an
/// absolute path risks false positives (e.g. macOS tempdirs start with `.`).
pub fn should_skip(rel: &Path) -> bool {
    // Skip hidden dirs/files (dot-prefix components)
    for component in rel.components() {
        let s = component.as_os_str().to_string_lossy();
        if s.starts_with('.') {
            return true;
        }
    }

    // Skip generated / dependency directories
    const SKIP_DIRS: &[&str] = &[
        "target", "node_modules", "__pycache__", "vendor", "dist", ".next",
        "build", "out", ".venv", "venv", "env",
    ];
    for component in rel.components() {
        let s = component.as_os_str().to_string_lossy();
        if SKIP_DIRS.contains(&s.as_ref()) {
            return true;
        }
    }

    // Skip neurons themselves (avoid indexing our own output)
    let s = rel.to_string_lossy();
    if s.contains(".cortyx") || s.ends_with(".context.md") || s.ends_with(".context.json") {
        return true;
    }

    // Skip binary and generated file extensions
    const SKIP_EXT: &[&str] = &[
        "png", "jpg", "jpeg", "gif", "svg", "ico", "webp",
        "woff", "woff2", "ttf", "eot",
        "mp3", "mp4", "wav", "ogg",
        "zip", "tar", "gz", "bz2", "xz", "7z",
        "pdf", "doc", "docx", "xls", "xlsx",
        "exe", "dll", "so", "dylib", "a", "o",
        "class", "pyc", "pyo",
        "bin", "dat", "db", "sqlite", "sqlite3",
        "min.js", "min.css", "map",
    ];
    if let Some(ext) = rel.extension().map(|e| e.to_string_lossy().to_lowercase()) {
        if SKIP_EXT.iter().any(|s| ext.as_str() == *s) {
            return true;
        }
    }

    // Skip lock and log files
    const SKIP_EXACT: &[&str] = &[
        "Cargo.lock", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
        "poetry.lock", "go.sum", "composer.lock", "Gemfile.lock", "uv.lock",
    ];
    if let Some(name) = rel.file_name().map(|n| n.to_string_lossy()) {
        if SKIP_EXACT.contains(&name.as_ref()) {
            return true;
        }
        if name.ends_with(".lock") || name.ends_with(".log") {
            return true;
        }
    }

    false
}

// ─── Security ─────────────────────────────────────────────────────────────────

/// Validate that a user-supplied path is safe to use relative to the project root.
///
/// Rejects: absolute paths, `..` components, and components starting with `.`.
/// Returns the validated path as a `PathBuf`.
pub fn validate_relative_path(raw: &str) -> Result<PathBuf> {
    let path = PathBuf::from(raw);
    if path.is_absolute() {
        bail!("path must be relative, got absolute: {raw}");
    }
    for component in path.components() {
        use std::path::Component;
        match component {
            Component::Normal(s) => {
                let s = s.to_string_lossy();
                if s.starts_with('.') {
                    bail!("hidden component not allowed: {s} in {raw}");
                }
            }
            // Reject everything else: ParentDir (..), RootDir (/), CurDir (.), Prefix (C:)
            other => bail!("unsafe path component {:?} in: {raw}", other),
        }
    }
    Ok(path)
}

/// Validate a neuron-to-neuron synapse target path.
///
/// Less strict than `validate_relative_path`: allows hidden directory components (so
/// `.cortyx/neurons/...` targets are accepted), but still blocks traversal (`..`),
/// absolute paths, and Windows-style prefixes.
pub fn validate_synapse_path(raw: &str) -> Result<PathBuf> {
    let path = PathBuf::from(raw);
    if path.is_absolute() {
        bail!("synapse target must be relative, got absolute: {raw}");
    }
    for component in path.components() {
        use std::path::Component;
        match component {
            Component::Normal(_) => {}
            other => bail!("unsafe path component {:?} in synapse target: {raw}", other),
        }
    }
    Ok(path)
}

// ─── Utility ──────────────────────────────────────────────────────────────────

/// Rough token count estimate (1 token ≈ 4 bytes of UTF-8 for code/prose mix).
///
/// GPT-4/Claude tokenize English at ~4 chars/token. Using 4 avoids the 25–35%
/// overestimate that `/3` produces, which silently overflows context windows.
pub fn estimate_tokens(text: &str) -> usize {
    (text.len() / 4).max(1)
}

/// BLAKE3 hash of a file's contents, returned as a 16-char hex prefix.
/// Returns empty string on error (file may not exist yet).
pub fn hash_file(path: &Path) -> Option<String> {
    let data = std::fs::read(path).ok()?;
    let hash = blake3::hash(&data);
    Some(hash.to_hex()[..16].to_string())
}

// ─── Synapse parsing from neuron Markdown ────────────────────────────────────

/// Parse the `## CROSS-REFERENCES (synapses)` section of a neuron file.
///
/// Supports the format:
/// ```markdown
/// - `path/to/other.context.md` → reason [imports]
/// ```
/// The `[type]` suffix is optional; defaults to `SemanticRelated`.
pub fn parse_synapses_from_content(content: &str) -> Vec<Synapse> {
    let mut in_section = false;
    let mut synapses = Vec::new();

    for line in content.lines() {
        if line.contains("## CROSS-REFERENCES") || line.contains("## SYNAPSES") {
            in_section = true;
            continue;
        }
        if in_section && line.starts_with("## ") {
            break;
        }
        if !in_section {
            continue;
        }
        let trimmed = line.trim_start();
        if !trimmed.starts_with("- ") && !trimmed.starts_with("* ") {
            continue;
        }

        // Extract backtick-quoted path
        let Some(bt_start) = line.find('`') else { continue };
        let rest = &line[bt_start + 1..];
        let Some(bt_end) = rest.find('`') else { continue };
        let path_str = &rest[..bt_end];
        if path_str.is_empty() {
            continue;
        }
        // Validate: reject path traversal from LLM-authored content.
        // Uses the synapse-specific validator that allows `.cortyx/neurons/` targets.
        if validate_synapse_path(path_str).is_err() {
            tracing::warn!("Skipping unsafe synapse target in neuron content: {path_str}");
            continue;
        }
        let target = PathBuf::from(path_str);

        // Extract reason (after → or ->)
        let raw_reason = line
            .find('→')
            .or_else(|| line.find("->"))
            .map(|i| line[i + "→".len()..].trim())
            .unwrap_or("")
            .to_string();

        // Detect type from trailing [type] bracket
        let (edge_type, reason) = extract_edge_type(&raw_reason);

        synapses.push(Synapse::new(target, edge_type, reason));
    }
    synapses
}

/// Detect synapse type from optional `[type]` suffix and keywords.
fn extract_edge_type(reason: &str) -> (SynapseType, String) {
    let lower = reason.to_lowercase();

    // Check for explicit [type] suffix
    let kind = if lower.ends_with("[imports]") || lower.ends_with("[import]") {
        SynapseType::Imports
    } else if lower.ends_with("[calls]") || lower.ends_with("[call]") {
        SynapseType::Calls
    } else if lower.ends_with("[implements]") || lower.ends_with("[implement]") {
        SynapseType::Implements
    } else if lower.ends_with("[temporal]") || lower.ends_with("[follows]") {
        SynapseType::TemporalFollows
    } else if lower.ends_with("[contradicts]") || lower.ends_with("[contradict]") {
        SynapseType::Contradicts
    } else if lower.ends_with("[derived]") {
        SynapseType::Derived
    } else if lower.ends_with("[concept]") {
        SynapseType::ConceptExpands
    } else {
        // Keyword fallback
        if lower.contains("import") || lower.contains("depend") {
            SynapseType::Imports
        } else if lower.contains("calls") || lower.contains("invoke") {
            SynapseType::Calls
        } else if lower.contains("implement") {
            SynapseType::Implements
        } else {
            SynapseType::SemanticRelated
        }
    };

    // Strip the [type] bracket from the displayed reason
    let clean = if let Some(i) = reason.rfind('[') {
        reason[..i].trim().to_string()
    } else {
        reason.trim().to_string()
    };

    (kind, clean)
}

// ─── Neuron Markdown templates ────────────────────────────────────────────────

/// Core neuron stub — sections filled by the host LLM via `cortyx_evolve_section`
/// or fully rewritten with `cortyx_evolve_context`.
///
/// When `prefilled` is non-empty (AST Bootstrap), the `api` section is pre-populated
/// with extracted function signatures and types so BM25 has vocabulary from day 1.
/// When `purpose_hint` is non-empty (A3: LLM-Free Pre-Population), the purpose section
/// is filled with extracted doc comment lines — producing a Level-1 neuron.
pub fn stub_core_neuron(source_rel: &str, hash: &str, now: &str, prefilled: &str, purpose_hint: &str) -> String {
    let api_content = if prefilled.is_empty() {
        "[TODO — key functions / symbols the model should know]".to_string()
    } else {
        prefilled.to_string()
    };

    let purpose_content = if purpose_hint.is_empty() {
        format!("[TODO — call cortyx_evolve_section(\"{source_rel}\", \"purpose\", \"...\") to fill this in]")
    } else {
        // Level-1 neuron: pre-populated from doc comments (no LLM call required)
        format!("{purpose_hint}\n\n<!-- Auto-populated from doc comments — call cortyx_evolve_section to refine -->")
    };

    format!(
        r#"<!-- AUTO-GENERATED CONTEXT — DO NOT EDIT MANUALLY -->
<!-- source: {source_rel} -->
<!-- hash: {hash} -->
<!-- last-updated: {now} -->
<!-- status: stub -->

**What this file does (for the AI):**
<!-- SECTION: purpose -->
{purpose_content}
<!-- /SECTION -->

**Key functions / symbols:**
<!-- SECTION: api -->
{api_content}
<!-- /SECTION -->

**Common pitfalls:**
<!-- SECTION: pitfalls -->
[TODO]
<!-- /SECTION -->

## CROSS-REFERENCES (synapses)

[TODO — add related neuron paths here, one per line]
[Format: `path/to/other.context.md` → reason [imports|calls|implements|semantic]]
"#
    )
}

/// UseCase sub-neuron stub for a single public function (S3 lazy splitting).
///
/// Created during compile when a source file has many public functions —
/// provides function-level retrieval precision while the Core neuron retains
/// the high-level summary. BM25 vocabulary is seeded with the function name so
/// "how does {fn_name} work?" queries activate this sub-neuron directly.
pub fn stub_function_neuron(fn_name: &str, source_rel: &str, now: &str) -> String {
    format!(
        r#"<!-- AUTO-GENERATED FUNCTION NEURON — DO NOT EDIT MANUALLY -->
<!-- source: {source_rel} -->
<!-- function: {fn_name} -->
<!-- last-updated: {now} -->
<!-- status: stub -->

**Function `{fn_name}` — what it does:**
<!-- SECTION: purpose -->
[TODO — call cortyx_evolve_section to describe {fn_name}]
<!-- /SECTION -->

**Signature & parameters:**
<!-- SECTION: api -->
[TODO — describe the inputs, outputs, and error conditions of {fn_name}]
<!-- /SECTION -->

**Pitfalls & edge cases:**
<!-- SECTION: pitfalls -->
[TODO]
<!-- /SECTION -->
"#
    )
}

/// Project neuron stub — one per project, auto-created at compile time.
pub fn stub_project_neuron(project_name: &str, now: &str) -> String {
    format!(
        r#"<!-- PROJECT NEURON — fill in via cortyx_evolve_section -->
<!-- project: {project_name} -->
<!-- last-updated: {now} -->
<!-- status: stub -->

**What this project does:**
<!-- SECTION: overview -->
[TODO — high-level description of the project for the AI]
<!-- /SECTION -->

**Main entry points:**
<!-- SECTION: entry_points -->
[TODO]
<!-- /SECTION -->

**Architecture overview:**
<!-- SECTION: architecture -->
[TODO]
<!-- /SECTION -->

## CROSS-REFERENCES (synapses)

[TODO — link to the most important Core neurons in this project]
"#
    )
}

// ─── Section Protocol ─────────────────────────────────────────────────────────

/// Parse all `<!-- SECTION: name -->` … `<!-- /SECTION -->` blocks in a neuron.
///
/// Returns `section_name → body` (content between tags, whitespace-trimmed).
/// Handles unclosed sections (content captured until EOF or next open tag).
///
/// Implementation note: lines are collected into a Vec and joined with `\n`,
/// which correctly handles both `\n` and `\r\n` line endings — the previous
/// byte-offset approach added exactly +1 per line and produced wrong slices on
/// Windows-style files.
pub fn parse_sections(content: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let mut current_name: Option<String> = None;
    let mut body_lines: Vec<&str> = Vec::new();

    for line in content.lines() {
        if let Some(name) = section_open_name(line) {
            // Close any previous open section before starting a new one.
            if let Some(prev_name) = current_name.take() {
                map.insert(prev_name, body_lines.join("\n").trim().to_string());
                body_lines.clear();
            }
            current_name = Some(name.to_string());
        } else if line.contains("<!-- /SECTION -->") {
            if let Some(name) = current_name.take() {
                map.insert(name, body_lines.join("\n").trim().to_string());
                body_lines.clear();
            }
        } else if current_name.is_some() {
            body_lines.push(line);
        }
    }

    // Handle unclosed final section
    if let Some(name) = current_name {
        map.insert(name, body_lines.join("\n").trim().to_string());
    }

    map
}

/// Replace or append a named section in neuron markdown.
///
/// - If `<!-- SECTION: name -->` exists: replaces its body with `new_body`.
/// - If not found: appends the section at the end of the file.
/// The surrounding tags are always preserved; `new_body` replaces only the content.
pub fn replace_section(content: &str, name: &str, new_body: &str) -> String {
    let mut result = String::with_capacity(content.len() + new_body.len() + 64);
    let mut in_section = false;
    let mut found = false;

    for line in content.lines() {
        let trimmed = line.trim();

        if !in_section {
            if let Some(open_name) = section_open_name(line) {
                if open_name == name {
                    result.push_str(line);
                    result.push('\n');
                    result.push_str(new_body.trim_end_matches('\n'));
                    result.push('\n');
                    in_section = true;
                    found = true;
                    continue;
                }
            }
            result.push_str(line);
            result.push('\n');
        } else if trimmed.contains("<!-- /SECTION -->") {
            in_section = false;
            result.push_str(line);
            result.push('\n');
            // else: old body — skip (already replaced above)
        }
    }

    if !found {
        if !result.ends_with('\n') {
            result.push('\n');
        }
        result.push_str(&format!(
            "<!-- SECTION: {name} -->\n{}\n<!-- /SECTION -->\n",
            new_body.trim_end_matches('\n')
        ));
    }

    result
}

/// Update the fixed header comment lines of an existing neuron.
///
/// Patches `<!-- hash: … -->`, `<!-- last-updated: … -->`, and
/// `<!-- status: … -->` lines in-place, leaving all other content intact.
/// Used by the section-level staleness update (S1, TRIZ R11) so that API
/// changes overwrite only the `api` section while preserving LLM-curated
/// `purpose`, `pitfalls`, and cross-reference sections.
///
/// Always produces output with a trailing newline.
pub fn update_neuron_header(content: &str, hash: &str, now: &str) -> String {
    let mut out = content
        .lines()
        .map(|line| {
            let t = line.trim_start();
            if t.starts_with("<!-- hash:") {
                format!("<!-- hash: {hash} -->")
            } else if t.starts_with("<!-- last-updated:") {
                format!("<!-- last-updated: {now} -->")
            } else if t.starts_with("<!-- status:") {
                "<!-- status: stale -->".to_string()
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out
}

/// Detect whether a line is a section open tag; return the section name if so.
fn section_open_name(line: &str) -> Option<&str> {
    let trimmed = line.trim();
    let rest = trimmed.strip_prefix("<!-- SECTION:")?;
    let name_part = rest.split("-->").next()?.trim();
    // Handle `name | v:N` variant — take only the name portion
    let name = name_part.split('|').next()?.trim();
    if name.is_empty() { None } else { Some(name) }
}

/// Write `data` to `path` atomically via a sibling `.tmp` file then rename.
///
/// Prevents torn writes from corrupting neuron files or the index on power loss.
/// Both files live on the same filesystem so `rename` is guaranteed atomic on POSIX.
pub fn atomic_write(path: &Path, data: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, data)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Serialize `value` to pretty JSON and write it to `path` atomically.
pub fn atomic_write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    atomic_write(path, serde_json::to_string_pretty(value)?.as_bytes())
}

// ─── Time helper (no chrono dep) ─────────────────────────────────────────────

/// Current time as RFC 3339 / ISO 8601 string (UTC, second precision).
///
/// Uses Hinnant's civil calendar algorithm — no external crate required.
pub fn now_iso8601() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let (y, mo, d, h, mi, s) = unix_secs_to_datetime(secs);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

/// S-XI (R16): Generate a stable neuron UUID from source path + current nanoseconds.
///
/// Uses BLAKE3 over `{path}:{nanos}` to produce a 32-char hex string.
/// UUID format: first 8 chars — path-derived; rest — time-salted for uniqueness.
/// Called once at neuron creation; thereafter the UUID is loaded from sidecar JSON.
pub fn generate_neuron_uuid(source: &Path) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    let input = format!("{}:{nanos}", source.display());
    let hash = blake3::hash(input.as_bytes());
    // Return first 32 hex chars (128 bits) — UUID-like without dashes
    hash.to_hex()[..32].to_string()
}

/// Decompose Unix epoch seconds into `(year, month, day, hour, minute, second)`.
///
/// Uses Hinnant's civil calendar algorithm (civil_from_days).
/// Reference: <http://howardhinnant.github.io/date_algorithms.html>
pub fn unix_secs_to_datetime(secs: u64) -> (u32, u32, u32, u32, u32, u32) {
    let h = (secs / 3600) % 24;
    let mi = (secs / 60) % 60;
    let s = secs % 60;
    let (y, mo, d) = days_to_ymd((secs / 86400) as i64);
    (y as u32, mo as u32, d as u32, h as u32, mi as u32, s as u32)
}

/// Convert days since Unix epoch to (year, month, day) using Hinnant's algorithm.
pub fn days_to_ymd(z: i64) -> (i32, i32, i32) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y as i32, m as i32, d as i32)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── meta_path ─────────────────────────────────────────────────────────────

    #[test]
    fn meta_path_core_neuron() {
        let p = Path::new("/project/.cortyx/neurons/src_engine_rs.context.md");
        let m = meta_path(p);
        assert_eq!(m, Path::new("/project/.cortyx/neurons/src_engine_rs.context.json"));
    }

    #[test]
    fn meta_path_usecase_neuron() {
        let p = Path::new("/project/.cortyx/neurons/src_auth_rs.usecase.add-oauth.md");
        let m = meta_path(p);
        assert_eq!(m, Path::new("/project/.cortyx/neurons/src_auth_rs.usecase.add-oauth.json"));
    }

    #[test]
    fn meta_path_file_with_dots_in_name() {
        let p = Path::new("/neurons/foo.bar.baz.context.md");
        let m = meta_path(p);
        assert_eq!(m, Path::new("/neurons/foo.bar.baz.context.json"));
    }

    // ── core_neuron_path ──────────────────────────────────────────────────────

    #[test]
    fn core_neuron_path_basic() {
        let root = Path::new("/project");
        let source = root.join("src/engine.rs");
        let neuron = core_neuron_path(&source, root);
        assert_eq!(
            neuron,
            Path::new("/project/.cortyx/neurons/src/engine_rs.context.md")
        );
    }

    #[test]
    fn core_neuron_path_root_file() {
        let root = Path::new("/project");
        let source = root.join("main.rs");
        let neuron = core_neuron_path(&source, root);
        assert_eq!(
            neuron,
            Path::new("/project/.cortyx/neurons/main_rs.context.md")
        );
    }

    #[test]
    fn core_neuron_path_deep() {
        let root = Path::new("/project");
        let source = root.join("src/ui/components/button.swift");
        let neuron = core_neuron_path(&source, root);
        assert_eq!(
            neuron,
            Path::new("/project/.cortyx/neurons/src/ui/components/button_swift.context.md")
        );
    }

    #[test]
    fn core_neuron_path_no_collision() {
        // src/engine.rs and root src_engine.rs must produce different neuron paths.
        let root = Path::new("/project");
        let a = core_neuron_path(&root.join("src/engine.rs"), root);
        let b = core_neuron_path(&root.join("src_engine.rs"), root);
        assert_ne!(a, b, "flat-file collision: {a:?} == {b:?}");
    }

    // ── should_skip ───────────────────────────────────────────────────────────

    #[test]
    fn should_skip_target_dir() {
        assert!(should_skip(Path::new("target/debug/cortyx")));
    }

    #[test]
    fn should_skip_hidden_dir() {
        assert!(should_skip(Path::new(".git/HEAD")));
    }

    #[test]
    fn should_skip_node_modules() {
        assert!(should_skip(Path::new("node_modules/react/index.js")));
    }

    #[test]
    fn should_skip_neuron_files() {
        assert!(should_skip(Path::new(".cortyx/neurons/foo.context.md")));
    }

    #[test]
    fn should_skip_binary_extensions() {
        assert!(should_skip(Path::new("assets/logo.png")));
        assert!(should_skip(Path::new("dist/bundle.min.js")));
    }

    #[test]
    fn should_skip_lock_files() {
        assert!(should_skip(Path::new("Cargo.lock")));
        assert!(should_skip(Path::new("package-lock.json")));
        assert!(should_skip(Path::new("yarn.lock")));
    }

    #[test]
    fn should_skip_log_files() {
        assert!(should_skip(Path::new("logs/app.log")));
        assert!(should_skip(Path::new("debug.log")));
    }

    #[test]
    fn should_not_skip_source_files() {
        assert!(!should_skip(Path::new("src/main.rs")));
        assert!(!should_skip(Path::new("lib/auth.py")));
        assert!(!should_skip(Path::new("README.md")));
    }

    // ── validate_relative_path ────────────────────────────────────────────────

    #[test]
    fn validate_relative_path_ok() {
        let p = validate_relative_path("src/engine.rs").unwrap();
        assert_eq!(p, PathBuf::from("src/engine.rs"));
    }

    #[test]
    fn validate_relative_path_rejects_absolute() {
        assert!(validate_relative_path("/etc/passwd").is_err());
    }

    #[test]
    fn validate_relative_path_rejects_parent_dir() {
        assert!(validate_relative_path("../../etc/passwd").is_err());
    }

    #[test]
    fn validate_relative_path_rejects_hidden() {
        assert!(validate_relative_path(".hidden/file").is_err());
    }

    // ── SynapseType ───────────────────────────────────────────────────────────

    #[test]
    fn synapse_type_multipliers_ordered() {
        // Implements should have highest structural multiplier
        assert!(SynapseType::Implements.type_multiplier() > SynapseType::SemanticRelated.type_multiplier());
        assert!(SynapseType::ConceptExpands.type_multiplier() == 1.0);
        assert!(SynapseType::Contradicts.type_multiplier() < SynapseType::SemanticRelated.type_multiplier());
    }

    // ── parse_synapses_from_content ───────────────────────────────────────────

    #[test]
    fn parse_synapses_basic() {
        let content = "## CROSS-REFERENCES (synapses)\n\
                       - `.cortyx/neurons/auth_rs.context.md` → handles tokens [imports]\n";
        let synapses = parse_synapses_from_content(content);
        assert_eq!(synapses.len(), 1);
        assert_eq!(synapses[0].target, PathBuf::from(".cortyx/neurons/auth_rs.context.md"));
        assert_eq!(synapses[0].edge_type, SynapseType::Imports);
        assert_eq!(synapses[0].reason, "handles tokens");
    }

    #[test]
    fn parse_synapses_defaults_to_semantic() {
        let content = "## CROSS-REFERENCES (synapses)\n\
                       - `.cortyx/neurons/ui_rs.context.md` → related UI code\n";
        let synapses = parse_synapses_from_content(content);
        assert_eq!(synapses.len(), 1);
        assert_eq!(synapses[0].edge_type, SynapseType::SemanticRelated);
    }

    #[test]
    fn parse_synapses_no_section() {
        let content = "No cross references here";
        assert!(parse_synapses_from_content(content).is_empty());
    }

    #[test]
    fn parse_synapses_empty_section() {
        let content = "## CROSS-REFERENCES (synapses)\n[TODO]\n";
        assert!(parse_synapses_from_content(content).is_empty());
    }

    #[test]
    fn parse_synapses_ignores_malformed_lines() {
        let content = "## CROSS-REFERENCES (synapses)\n\
                       - no backticks here\n\
                       - `valid.context.md` → ok\n";
        let synapses = parse_synapses_from_content(content);
        assert_eq!(synapses.len(), 1);
    }

    #[test]
    fn parse_synapses_multiple_types() {
        let content = "## CROSS-REFERENCES (synapses)\n\
                       - `a.context.md` → provides types [implements]\n\
                       - `b.context.md` → next session [temporal]\n\
                       - `c.context.md` → loosely related\n";
        let synapses = parse_synapses_from_content(content);
        assert_eq!(synapses.len(), 3);
        assert_eq!(synapses[0].edge_type, SynapseType::Implements);
        assert_eq!(synapses[1].edge_type, SynapseType::TemporalFollows);
        assert_eq!(synapses[2].edge_type, SynapseType::SemanticRelated);
    }

    #[test]
    fn synapse_has_correct_defaults() {
        let s = Synapse::new(PathBuf::from("a.md"), SynapseType::Imports, "test".into());
        assert_eq!(s.weight, 0.5);
        assert_eq!(s.edge_type, SynapseType::Imports);
    }

    // ── Section Protocol ──────────────────────────────────────────────────────

    #[test]
    fn parse_sections_basic() {
        let content = "header\n<!-- SECTION: purpose -->\nsome purpose\n<!-- /SECTION -->\nfooter\n";
        let sections = parse_sections(content);
        assert_eq!(sections.get("purpose").map(String::as_str), Some("some purpose"));
    }

    #[test]
    fn parse_sections_multiple() {
        let content = "<!-- SECTION: api -->\nfn foo()\n<!-- /SECTION -->\n<!-- SECTION: pitfalls -->\nwatch out\n<!-- /SECTION -->\n";
        let sections = parse_sections(content);
        assert_eq!(sections.get("api").map(String::as_str), Some("fn foo()"));
        assert_eq!(sections.get("pitfalls").map(String::as_str), Some("watch out"));
    }

    #[test]
    fn parse_sections_empty_returns_empty_map() {
        assert!(parse_sections("no sections here").is_empty());
    }

    #[test]
    fn replace_section_existing() {
        let content = "pre\n<!-- SECTION: api -->\nold\n<!-- /SECTION -->\npost\n";
        let result = replace_section(content, "api", "new content");
        assert!(result.contains("new content"), "new: {result}");
        assert!(!result.contains("old"), "old body removed: {result}");
        assert!(result.contains("pre"), "prefix preserved");
        assert!(result.contains("post"), "suffix preserved");
        assert!(result.contains("<!-- SECTION: api -->"), "open tag preserved");
        assert!(result.contains("<!-- /SECTION -->"), "close tag preserved");
    }

    #[test]
    fn replace_section_appends_if_missing() {
        let content = "existing content\n";
        let result = replace_section(content, "new_section", "body");
        assert!(result.contains("<!-- SECTION: new_section -->"));
        assert!(result.contains("body"));
        assert!(result.contains("existing content"), "original preserved");
    }

    #[test]
    fn replace_section_round_trip() {
        // parse_sections + replace_section must produce consistent results
        let content = "<!-- SECTION: purpose -->\noriginal\n<!-- /SECTION -->\n";
        let updated = replace_section(content, "purpose", "updated");
        let sections = parse_sections(&updated);
        assert_eq!(sections.get("purpose").map(String::as_str), Some("updated"));
    }

    // ── Time helpers ──────────────────────────────────────────────────────────

    #[test]
    fn now_iso8601_format() {
        let s = now_iso8601();
        assert!(s.ends_with('Z'), "should be UTC: {s}");
        assert_eq!(s.len(), 20, "YYYY-MM-DDTHH:MM:SSZ: {s}");
    }

    #[test]
    fn days_to_ymd_known_dates() {
        // Unix epoch = 1970-01-01
        assert_eq!(days_to_ymd(0), (1970, 1, 1));
        // 2025-01-01 = day 20089
        assert_eq!(days_to_ymd(20089), (2025, 1, 1));
        // Leap year: 2000-02-29 = day 11016
        assert_eq!(days_to_ymd(11016), (2000, 2, 29));
    }
}
