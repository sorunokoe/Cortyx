use super::*;

#[test]
fn extract_rust_functions_and_types() {
    let src = r#"
/// Parse the session state.
pub fn parse_session(input: &str) -> Result<Session> { todo!() }
pub async fn load_model(cfg: Config) -> Model { todo!() }
fn not_pub() {}
struct Private;
pub struct Session { id: u64 }
pub enum Status { Active, Idle }
pub trait Runnable { fn run(&self); }
"#;
    let s = extract_signatures("src/engine.rs", src);
    assert!(
        s.functions.contains(&"parse_session".to_string()),
        "{:?}",
        s.functions
    );
    assert!(s.functions.contains(&"load_model".to_string()));
    assert!(
        !s.functions.contains(&"not_pub".to_string()),
        "private fn excluded"
    );
    assert!(s.types.contains(&"Session".to_string()));
    assert!(s.types.contains(&"Status".to_string()));
    assert!(s.types.contains(&"Runnable".to_string()));
    let formatted = format_for_stub(&s);
    assert!(
        formatted.contains("parse_session"),
        "formatted: {formatted}"
    );
    let purpose = format_purpose_hint(&s);
    assert!(
        purpose.contains("Parse the session state"),
        "doc comment: {purpose}"
    );
}

#[test]
fn extract_python_classes_and_functions() {
    let src = "class AuthService:\n    def login(self): pass\ndef logout(): pass\n";
    let s = extract_signatures("auth.py", src);
    assert!(
        s.types.contains(&"AuthService".to_string()),
        "{:?}",
        s.types
    );
    assert!(
        s.functions.contains(&"login".to_string()),
        "{:?}",
        s.functions
    );
    assert!(s.functions.contains(&"logout".to_string()));
}

#[test]
fn extract_python_docstring() {
    let src = r#""""
Authentication service for JWT tokens.
Handles login, logout, and refresh.
"""
class AuthService:
    def login(self): pass
"#;
    let s = extract_signatures("auth.py", src);
    assert!(!s.doc_lines.is_empty(), "should capture docstring");
    assert!(
        s.doc_lines.iter().any(|l| l.contains("Authentication")),
        "doc lines: {:?}",
        s.doc_lines
    );
}

#[test]
fn extract_go_doc_comments() {
    let src = "// Package server provides HTTP handling.\n// Use NewServer to initialise.\nfunc NewServer(cfg Config) *Server {}\n";
    let s = extract_signatures("server.go", src);
    assert!(!s.doc_lines.is_empty(), "should capture go doc comments");
    assert!(
        s.doc_lines.iter().any(|l| l.contains("server")),
        "doc lines: {:?}",
        s.doc_lines
    );
}

#[test]
fn extract_typescript_exports_only() {
    let src = r#"
export function fetchUser(id: string): Promise<User> {}
export const handler = async (req: Request) => {}
export interface UserProfile { name: string }
export class UserService {}
function internal() {}
const privateConst = () => {}
"#;
    let s = extract_signatures("api/user.ts", src);
    assert!(
        s.functions.contains(&"fetchUser".to_string()),
        "{:?}",
        s.functions
    );
    assert!(s.functions.contains(&"handler".to_string()));
    assert!(
        !s.functions.contains(&"internal".to_string()),
        "non-export excluded"
    );
    assert!(
        !s.functions.contains(&"privateConst".to_string()),
        "private arrow excluded"
    );
    assert!(s.types.contains(&"UserProfile".to_string()));
    assert!(s.types.contains(&"UserService".to_string()));
}

#[test]
fn extract_go_exported_only() {
    let src = "func NewServer(cfg Config) *Server {}\nfunc (s *Server) Start() error {}\nfunc internal() {}\ntype Server struct {}\n";
    let s = extract_signatures("server.go", src);
    assert!(
        s.functions.contains(&"NewServer".to_string()),
        "{:?}",
        s.functions
    );
    assert!(
        s.functions.contains(&"Start".to_string()),
        "method Start exported"
    );
    assert!(
        !s.functions.contains(&"internal".to_string()),
        "lowercase fn excluded"
    );
    assert!(s.types.contains(&"Server".to_string()));
}

#[test]
fn unknown_extension_returns_empty() {
    let s = extract_signatures("data.csv", "hello,world\n1,2\n");
    assert!(s.is_empty());
    assert_eq!(format_for_stub(&s), "");
}

#[test]
fn format_for_stub_caps_at_600_chars() {
    let long_fns: Vec<String> = (0..50)
        .map(|i| format!("very_long_function_name_{i}"))
        .collect();
    let summary = AstSummary {
        functions: long_fns,
        types: vec!["T".to_string()],
        doc_lines: Vec::new(),
        ..Default::default()
    };
    let out = format_for_stub(&summary);
    assert!(out.len() <= 601, "output length: {}", out.len());
}

#[test]
fn extract_swift_functions_and_types() {
    let src = r#"
/// Authentication service.
public class AuthService {
    public func login(user: String) -> Bool { false }
    open func logout() {}
    private func reset() {}
}
public struct Token {}
public protocol Authenticatable {}
"#;
    let s = extract_signatures("Auth.swift", src);
    assert!(
        s.functions.contains(&"login".to_string()),
        "{:?}",
        s.functions
    );
    assert!(s.functions.contains(&"logout".to_string()));
    assert!(s.types.contains(&"AuthService".to_string()));
    assert!(s.types.contains(&"Token".to_string()));
    assert!(s.types.contains(&"Authenticatable".to_string()));
    assert!(!s.doc_lines.is_empty(), "should capture doc comment");
}

#[test]
fn extract_kotlin_functions_and_types() {
    let src = "class UserService {\n    fun getUser(id: Int): User = TODO()\n    suspend fun fetchAsync(): List<User> = TODO()\n    private fun helper() {}\n}\ndata class User(val id: Int)\n";
    let s = extract_signatures("UserService.kt", src);
    assert!(
        s.functions.contains(&"getUser".to_string()),
        "{:?}",
        s.functions
    );
    assert!(s.functions.contains(&"fetchAsync".to_string()));
    assert!(s.types.contains(&"UserService".to_string()));
    assert!(s.types.contains(&"User".to_string()));
}

#[test]
fn extract_java_public_methods_and_classes() {
    let src = r#"
/**
 * Handles user authentication.
 */
public class AuthController {
    public static User authenticate(String token) { return null; }
    protected void logout(HttpRequest req) {}
    private void helper() {}
}
public interface Repository {}
"#;
    let s = extract_signatures("AuthController.java", src);
    assert!(
        s.functions.contains(&"authenticate".to_string()),
        "{:?}",
        s.functions
    );
    assert!(s.functions.contains(&"logout".to_string()));
    assert!(
        !s.functions.contains(&"helper".to_string()),
        "private excluded"
    );
    assert!(s.types.contains(&"AuthController".to_string()));
    assert!(s.types.contains(&"Repository".to_string()));
}

#[test]
fn extract_csharp_methods_and_types() {
    let src = r#"
/// <summary>Authentication service</summary>
public class AuthService {
    public static async Task<User> LoginAsync(string token) { return null; }
    protected void Logout() {}
}
public interface IRepository {}
public enum Status { Active, Idle }
"#;
    let s = extract_signatures("AuthService.cs", src);
    assert!(
        s.functions.contains(&"LoginAsync".to_string()),
        "{:?}",
        s.functions
    );
    assert!(s.types.contains(&"AuthService".to_string()));
    assert!(s.types.contains(&"IRepository".to_string()));
    assert!(!s.doc_lines.is_empty(), "should capture XML doc comment");
}

#[test]
fn extract_ruby_methods_and_classes() {
    let src = "module Auth\n  class UserService\n    def login(token)\n      true\n    end\n    def logout; end\n    def _private_helper; end\n  end\nend\n";
    let s = extract_signatures("user_service.rb", src);
    assert!(
        s.functions.contains(&"login".to_string()),
        "{:?}",
        s.functions
    );
    assert!(s.functions.contains(&"logout".to_string()));
    assert!(
        !s.functions.contains(&"_private_helper".to_string()),
        "private excluded"
    );
    assert!(s.types.contains(&"UserService".to_string()));
    assert!(s.types.contains(&"Auth".to_string()));
}

#[test]
fn extract_c_functions_and_structs() {
    let src = r#"
// Authentication utilities
struct AuthToken {
    int id;
};
typedef struct User User;
int authenticate(const char* token) {
    return 0;
}
void cleanup(AuthToken* t) {}
"#;
    let s = extract_signatures("auth.c", src);
    assert!(
        s.functions.contains(&"authenticate".to_string()),
        "{:?}",
        s.functions
    );
    assert!(s.functions.contains(&"cleanup".to_string()));
    assert!(
        s.types.contains(&"AuthToken".to_string()) || s.types.contains(&"User".to_string()),
        "{:?}",
        s.types
    );
}

#[test]
fn extract_call_sites_detects_cross_file_calls() {
    use std::collections::HashMap;
    use std::path::PathBuf;

    let caller_content = r#"
fn activate(&self) {
    let session = parse_session(input);
    let model = load_model(cfg);
}
"#;
    let mut vocab: HashMap<String, PathBuf> = HashMap::new();
    vocab.insert("parse_session".to_string(), PathBuf::from("src/engine.rs"));
    vocab.insert("load_model".to_string(), PathBuf::from("src/model.rs"));
    vocab.insert("unrelated_fn".to_string(), PathBuf::from("src/other.rs"));

    let edges = extract_call_sites("src/index.rs", caller_content, &vocab);
    let callee_files: Vec<&std::path::Path> =
        edges.iter().map(|e| e.callee_file.as_path()).collect();
    assert!(
        callee_files.contains(&std::path::Path::new("src/engine.rs")),
        "should detect parse_session call: {callee_files:?}"
    );
    assert!(
        callee_files.contains(&std::path::Path::new("src/model.rs")),
        "should detect load_model call: {callee_files:?}"
    );
    assert!(
        !callee_files.contains(&std::path::Path::new("src/other.rs")),
        "unrelated_fn not called: {callee_files:?}"
    );
}

#[test]
fn extract_call_sites_empty_vocab_returns_empty() {
    use std::collections::HashMap;
    let edges = extract_call_sites(
        "src/main.rs",
        "fn main() { do_something(); }",
        &HashMap::new(),
    );
    assert!(edges.is_empty());
}

#[test]
fn extract_call_sites_no_self_loops() {
    use std::collections::HashMap;
    use std::path::PathBuf;

    let content = "pub fn parse_session(s: &str) -> () { parse_session(s) }";
    let mut vocab: HashMap<String, PathBuf> = HashMap::new();
    vocab.insert("parse_session".to_string(), PathBuf::from("src/engine.rs"));

    let edges = extract_call_sites("src/engine.rs", content, &vocab);
    let self_loops: Vec<_> = edges
        .iter()
        .filter(|e| e.callee_file == std::path::Path::new("src/engine.rs"))
        .collect();
    assert!(
        self_loops.is_empty(),
        "no self-loops expected: {self_loops:?}"
    );
}

#[test]
fn compute_sig_hash_is_16_hex_chars() {
    let summary = extract_signatures("src/lib.rs", "pub fn foo() {}\npub struct Bar {}");
    let hash = compute_sig_hash(&summary);
    assert_eq!(hash.len(), 16, "sig_hash must be 16 hex chars");
    assert!(
        hash.chars().all(|c| c.is_ascii_hexdigit()),
        "sig_hash must be hex"
    );
}

#[test]
fn compute_sig_hash_empty_summary_is_stable() {
    let empty = extract_signatures("src/empty.unknown", "no recognizable signatures here");
    let h1 = compute_sig_hash(&empty);
    let h2 = compute_sig_hash(&empty);
    assert_eq!(h1, h2, "empty summary hash must be deterministic");
}
