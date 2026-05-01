//! Core language AST extractors (Level-0 tree-sitter + Level-1 regex).
//!
//! Supports 10 major languages: Rust, Python, TypeScript/JS, Go, Swift,
//! Kotlin, Java, C#, Ruby, C/C++.

mod extractors;
mod regex;

// Re-export all extractors for parent module
pub(super) use extractors::{
    extract_c, extract_csharp, extract_go, extract_java, extract_kotlin, extract_python,
    extract_ruby, extract_rust, extract_swift, extract_typescript,
};
