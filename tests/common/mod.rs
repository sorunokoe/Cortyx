/// Shared test utilities for integration and benchmark tests.
use std::path::PathBuf;
use std::sync::OnceLock;

static CORTYX_BIN: OnceLock<PathBuf> = OnceLock::new();

pub fn cortyx_bin() -> PathBuf {
    CORTYX_BIN
        .get_or_init(|| {
            std::env::var_os("CARGO_BIN_EXE_cortyx")
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    let mut path = std::env::current_exe().unwrap();
                    path.pop(); // remove test binary name
                    if path.ends_with("deps") {
                        path.pop();
                    }
                    path.push("cortyx");
                    path
                })
        })
        .clone()
}

pub fn run(args: &[&str], cwd: &std::path::Path) -> std::process::Output {
    std::process::Command::new(cortyx_bin())
        .args(args)
        .env("CORTYX_NO_DOWNLOAD", "1")
        .current_dir(cwd)
        .output()
        .expect("Failed to run cortyx binary")
}

#[allow(dead_code)]
pub fn run_with_home(
    args: &[&str],
    cwd: &std::path::Path,
    home: &std::path::Path,
) -> std::process::Output {
    std::process::Command::new(cortyx_bin())
        .args(args)
        .env("CORTYX_NO_DOWNLOAD", "1")
        .env("HOME", home)
        .current_dir(cwd)
        .output()
        .expect("Failed to run cortyx binary")
}
