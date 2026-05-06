use crate::error::Result;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static ATOMIC_WRITE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Root of the neuron store inside a project.
///
/// Example: `/my/project/.cortyx/neurons/`
pub fn neuron_dir(project_root: &Path) -> PathBuf {
    project_root.join(".cortyx").join("neurons")
}

/// Map a source file to its Core neuron path.
///
/// Preserves the directory structure under `.cortyx/neurons/` to prevent
/// flat-file collisions. Only dots in the filename are replaced with `_`.
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
    neuron_dir(project_root)
        .join(parent)
        .join(format!("{stem}.context.md"))
}

/// Map a Core neuron path + function name to its UseCase sub-neuron path.
///
/// Example: `.cortyx/neurons/src/engine_rs.context.md` + `"validate_user"` →
///          `.cortyx/neurons/src/engine_rs.fn-validate_user.context.md`
pub fn sub_neuron_path(core_path: &Path, fn_name: &str) -> PathBuf {
    let safe_name: String = fn_name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let dir = core_path.parent().unwrap_or(Path::new("."));
    let core_stem = core_path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .strip_suffix(".context")
        .map(|s| s.to_owned())
        .unwrap_or_else(|| {
            core_path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned()
        });
    dir.join(format!("{core_stem}.fn-{safe_name}.context.md"))
}

/// Map a `.context.md` path to its sidecar `.context.json` path.
///
/// Example: `neurons/engine_rs.context.md` → `neurons/engine_rs.context.json`
pub fn meta_path(neuron_md: &Path) -> PathBuf {
    sidecar_path(neuron_md, ".json")
}

pub fn sidecar_path(neuron_md: &Path, suffix: &str) -> PathBuf {
    let name = neuron_md.file_name().unwrap_or_default().to_string_lossy();
    let sidecar_name = name
        .strip_suffix(".md")
        .map(|s| format!("{s}{suffix}"))
        .unwrap_or_else(|| format!("{name}{suffix}"));
    neuron_md
        .parent()
        .unwrap_or(Path::new("."))
        .join(sidecar_name)
}

/// Write `data` to `path` atomically via a sibling `.tmp` file then rename.
///
/// Prevents torn writes from corrupting neuron files or the index on power loss.
/// Both files live on the same filesystem so `rename` is guaranteed atomic on POSIX.
pub fn atomic_write(path: &Path, data: &[u8]) -> Result<()> {
    let tmp = atomic_write_tmp_path(path);
    std::fs::write(&tmp, data)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Serialize `value` to pretty JSON and write it to `path` atomically.
pub fn atomic_write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    atomic_write(path, serde_json::to_string_pretty(value)?.as_bytes())
}

pub(super) fn atomic_write_tmp_path(path: &Path) -> PathBuf {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("tmp");
    let pid = std::process::id();
    let nonce = ATOMIC_WRITE_COUNTER.fetch_add(1, Ordering::Relaxed);
    parent.join(format!(".{file_name}.{pid}.{nonce}.tmp"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meta_path_core_neuron() {
        let p = Path::new("/project/.cortyx/neurons/src_engine_rs.context.md");
        let m = meta_path(p);
        assert_eq!(
            m,
            Path::new("/project/.cortyx/neurons/src_engine_rs.context.json")
        );
    }

    #[test]
    fn meta_path_usecase_neuron() {
        let p = Path::new("/project/.cortyx/neurons/src_auth_rs.usecase.add-oauth.md");
        let m = meta_path(p);
        assert_eq!(
            m,
            Path::new("/project/.cortyx/neurons/src_auth_rs.usecase.add-oauth.json")
        );
    }

    #[test]
    fn meta_path_file_with_dots_in_name() {
        let p = Path::new("/neurons/foo.bar.baz.context.md");
        let m = meta_path(p);
        assert_eq!(m, Path::new("/neurons/foo.bar.baz.context.json"));
    }

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
        let root = Path::new("/project");
        let a = core_neuron_path(&root.join("src/engine.rs"), root);
        let b = core_neuron_path(&root.join("src_engine.rs"), root);
        assert_ne!(a, b, "flat-file collision: {a:?} == {b:?}");
    }

    #[test]
    fn atomic_write_tmp_path_is_unique_per_call() {
        let path = Path::new("/tmp/example.context.md");
        let a = atomic_write_tmp_path(path);
        let b = atomic_write_tmp_path(path);
        assert_ne!(a, b);
        assert_eq!(a.parent(), path.parent());
        assert_eq!(b.parent(), path.parent());
    }
}
