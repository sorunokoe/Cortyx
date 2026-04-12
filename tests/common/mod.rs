/// Shared test utilities for integration and benchmark tests.
use std::path::PathBuf;

pub fn cortyx_bin() -> PathBuf {
    let mut path = std::env::current_exe().unwrap();
    path.pop(); // remove test binary name
    if path.ends_with("deps") {
        path.pop();
    }
    path.push("cortyx");
    path
}

pub fn run(args: &[&str], cwd: &std::path::Path) -> std::process::Output {
    std::process::Command::new(cortyx_bin())
        .args(args)
        .current_dir(cwd)
        .output()
        .expect("Failed to run cortyx binary")
}
