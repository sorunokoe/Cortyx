use std::path::{Path, PathBuf};

/// Normalise an entity name to a lower-snake-case slug safe for filenames.
#[must_use]
pub fn slugify(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>()
        .split('_')
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join("_")
}

pub(super) fn entity_slug_from_path(path: &Path) -> String {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    let stem = stem.strip_suffix(".context").unwrap_or(stem);
    stem.strip_prefix("_kg_").unwrap_or(stem).to_string()
}

/// Derive the KG neuron path from a project root and entity slug.
#[must_use]
pub fn kg_neuron_path(project_root: &Path, entity: &str) -> PathBuf {
    project_root
        .join(".cortyx")
        .join("neurons")
        .join(format!("_kg_{}.context.md", slugify(entity)))
}

/// Collect all KG entity paths under a project root.
#[must_use]
pub fn list_kg_paths(project_root: &Path) -> Vec<PathBuf> {
    let ndir = project_root.join(".cortyx").join("neurons");
    let Ok(rd) = std::fs::read_dir(&ndir) else {
        return Vec::new();
    };
    rd.flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("_kg_") && n.ends_with(".context.md"))
                .unwrap_or(false)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugify_normalises() {
        assert_eq!(slugify("My Project!"), "my_project");
        assert_eq!(slugify("hello-world"), "hello_world");
        assert_eq!(slugify("camelCase"), "camelcase");
    }
}
