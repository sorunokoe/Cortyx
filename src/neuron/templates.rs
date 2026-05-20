/// Core neuron stub — sections filled by the host LLM via `cortyx_evolve_section`
/// or fully rewritten with `cortyx_evolve_context`.
///
/// When `prefilled` is non-empty (AST Bootstrap), the `api` section is pre-populated
/// with extracted function signatures and types so BM25 has vocabulary from day 1.
/// When `purpose_hint` is non-empty (A3: LLM-Free Pre-Population), the purpose section
/// is filled with extracted doc comment lines — producing a Level-1 neuron.
#[must_use]
pub fn stub_core_neuron(
    source_rel: &str,
    hash: &str,
    now: &str,
    prefilled: &str,
    purpose_hint: &str,
    extra_vocab: &str,
) -> String {
    let api_content = if prefilled.is_empty() {
        "[TODO — key functions / symbols the model should know]".to_string()
    } else {
        prefilled.to_string()
    };

    let purpose_content = if purpose_hint.is_empty() {
        format!("[TODO — call cortyx_evolve_section(\"{source_rel}\", \"purpose\", \"...\") to fill this in]")
    } else {
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
{extra_vocab}"#
    )
}

/// UseCase sub-neuron stub for a single public function (S3 lazy splitting).
#[must_use]
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
#[must_use]
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
