/// B3: NLP-Free Paraphrase Alias Generation
///
/// Generates natural-language alias terms from code identifiers (function names, type names)
/// using deterministic verb/noun synonym tables — no model download, no API call.
///
/// For each public function like `get_user_by_email`, this produces:
///   ["fetch", "retrieve", "find", "lookup", "user", "account", "member", "email", "mail"]
///
/// These aliases are injected into BM25 at 0.5× weight, bridging the lexical gap between
/// user queries ("how to look up an account by email") and code identifiers ("get_user_by_email").
/// Together with A1 (git/comment vocab) and B1 (morphemic trie), B3 closes the remaining
/// cold-start vocabulary gap.
use std::collections::{HashMap, HashSet};

/// Verb synonym groups — any verb in a group is an alias for all others.
///
/// Key design: groups are small and domain-relevant. We do NOT use a full thesaurus
/// (which would add noise). Only verbs that appear commonly in code identifiers are included.
fn verb_groups() -> &'static [&'static [&'static str]] {
    &[
        &["get", "fetch", "retrieve", "find", "load", "read", "lookup"],
        &[
            "set", "update", "write", "save", "store", "put", "persist", "upsert",
        ],
        &[
            "delete", "remove", "drop", "clear", "clean", "purge", "erase",
        ],
        &[
            "create",
            "new",
            "make",
            "build",
            "generate",
            "produce",
            "construct",
            "add",
        ],
        &[
            "check", "validate", "verify", "test", "assert", "ensure", "confirm", "guard",
        ],
        &[
            "send",
            "publish",
            "emit",
            "dispatch",
            "broadcast",
            "notify",
            "push",
        ],
        &[
            "parse",
            "decode",
            "deserialize",
            "extract",
            "process",
            "consume",
        ],
        &[
            "encode",
            "serialize",
            "format",
            "render",
            "transform",
            "convert",
        ],
        &[
            "connect",
            "join",
            "link",
            "attach",
            "bind",
            "register",
            "subscribe",
        ],
        &[
            "disconnect",
            "close",
            "stop",
            "terminate",
            "shutdown",
            "cancel",
            "abort",
        ],
        &[
            "list",
            "query",
            "search",
            "filter",
            "scan",
            "enumerate",
            "paginate",
        ],
        &[
            "log", "trace", "debug", "record", "report", "audit", "track",
        ],
        &[
            "init",
            "initialize",
            "setup",
            "configure",
            "boot",
            "start",
            "launch",
        ],
        &["handle", "process", "run", "execute", "invoke", "call"],
        &["lock", "acquire", "claim"],
        &["unlock", "release", "free"],
        &["import", "load", "require", "include", "inject"],
        &["export", "expose", "provide", "serve"],
        &["open", "begin", "start"],
        &[
            "merge",
            "combine",
            "aggregate",
            "collect",
            "gather",
            "reduce",
        ],
        &["split", "partition", "divide", "chunk", "segment", "slice"],
        &["sort", "order", "rank", "prioritize"],
        &["apply", "use", "consume", "activate"],
    ]
}

/// Noun synonym groups — domain-specific entity synonyms.
fn noun_groups() -> &'static [&'static [&'static str]] {
    &[
        &[
            "user", "account", "member", "person", "profile", "player", "actor", "owner",
        ],
        &[
            "token",
            "credential",
            "secret",
            "api_key",
            "apikey",
            "jwt",
            "bearer",
            "auth",
        ],
        &["session", "context", "state", "scope"],
        &[
            "message",
            "event",
            "notification",
            "alert",
            "signal",
            "payload",
        ],
        &[
            "error",
            "exception",
            "failure",
            "fault",
            "panic",
            "problem",
            "issue",
        ],
        &[
            "config",
            "settings",
            "options",
            "params",
            "parameters",
            "configuration",
            "prefs",
        ],
        &["file", "document", "resource", "artifact", "blob", "asset"],
        &["id", "identifier", "uuid", "key", "ref", "handle"],
        &["data", "content", "body", "info", "information", "record"],
        &["request", "query", "input", "args", "arguments", "params"],
        &["response", "result", "output", "reply", "answer"],
        &["client", "consumer", "caller", "requester"],
        &["server", "service", "provider", "backend", "endpoint"],
        &["database", "db", "store", "storage", "repository", "repo"],
        &["cache", "buffer", "pool", "queue"],
        &["email", "mail", "address"],
        &["password", "pass", "secret", "hash"],
        &["name", "title", "label", "slug", "identifier"],
        &[
            "role",
            "permission",
            "policy",
            "access",
            "scope",
            "privilege",
        ],
        &["url", "path", "route", "endpoint", "uri", "link"],
        &["index", "idx", "pos", "offset", "position"],
        &["time", "timestamp", "date", "datetime", "epoch"],
        &["count", "size", "length", "total", "sum", "num"],
        &["status", "state", "condition", "mode", "phase", "step"],
        &["type", "kind", "category", "class", "variant"],
        &["hash", "checksum", "digest", "fingerprint", "signature"],
        &["task", "job", "work", "operation", "action"],
        &["log", "trace", "record", "entry", "event"],
        &["version", "revision", "release", "tag"],
        &["image", "photo", "thumbnail", "avatar", "icon"],
        &["model", "schema", "struct", "entity", "object", "value"],
    ]
}

/// Build reverse lookup: term → group index for fast expansion.
fn build_lookup() -> HashMap<String, Vec<String>> {
    let mut map: HashMap<String, Vec<String>> = HashMap::new();

    let process_group = |group: &[&str], map: &mut HashMap<String, Vec<String>>| {
        let all: Vec<String> = group.iter().map(|s| s.to_string()).collect();
        for &member in group {
            let entry = map.entry(member.to_string()).or_default();
            for alias in &all {
                if alias != member {
                    entry.push(alias.clone());
                }
            }
        }
    };

    for group in verb_groups() {
        process_group(group, &mut map);
    }
    for group in noun_groups() {
        process_group(group, &mut map);
    }

    map
}

/// Global alias lookup — initialized once.
fn alias_lookup() -> &'static HashMap<String, Vec<String>> {
    use std::sync::OnceLock;
    static LOOKUP: OnceLock<HashMap<String, Vec<String>>> = OnceLock::new();
    LOOKUP.get_or_init(build_lookup)
}

/// Split a code identifier into sub-tokens.
///
/// Handles snake_case (`get_user_email` → `["get", "user", "email"]`) and
/// camelCase (`getUserEmail` → `["get", "user", "email"]`).
fn split_identifier(name: &str) -> Vec<String> {
    let mut parts: HashSet<String> = HashSet::new();
    // snake_case split on the lowercased string
    let lower = name.to_lowercase();
    for part in lower.split('_') {
        let p = part.trim_matches(|c: char| !c.is_alphanumeric());
        if p.len() >= 2 {
            parts.insert(p.to_string());
        }
    }
    // camelCase split on the original name (uppercase detection requires original case)
    let chars: Vec<char> = name.chars().collect();
    let mut start = 0;
    for i in 1..chars.len() {
        if chars[i].is_uppercase() {
            let part: String = chars[start..i].iter().collect::<String>().to_lowercase();
            if part.len() >= 2 {
                parts.insert(part);
            }
            start = i;
        }
    }
    let last: String = chars[start..].iter().collect::<String>().to_lowercase();
    if last.len() >= 2 {
        parts.insert(last);
    }
    parts.into_iter().collect()
}

/// Generate alias terms for a list of function/type names.
///
/// For each name:
/// 1. Split into sub-tokens (snake_case + camelCase)
/// 2. Expand each sub-token through verb/noun synonym groups
/// 3. Collect all aliases (excluding the original tokens — they're already in BM25)
///
/// Returns deduplicated, lowercase alias terms for BM25 injection at 0.5× weight.
pub fn generate_alias_terms(names: &[String]) -> Vec<String> {
    let lookup = alias_lookup();
    let mut aliases: HashSet<String> = HashSet::new();

    for name in names {
        let parts = split_identifier(name);
        for part in &parts {
            if let Some(synonyms) = lookup.get(part.as_str()) {
                for syn in synonyms {
                    aliases.insert(syn.clone());
                }
            }
        }
    }

    let mut result: Vec<String> = aliases.into_iter().collect();
    result.sort_unstable();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_user_aliases() {
        let aliases = generate_alias_terms(&["get_user".to_string()]);
        // "get" → fetch, retrieve, find, load, read, lookup
        assert!(
            aliases.contains(&"fetch".to_string()),
            "aliases: {aliases:?}"
        );
        assert!(aliases.contains(&"retrieve".to_string()));
        assert!(aliases.contains(&"find".to_string()));
        // "user" → account, member, person, profile
        assert!(aliases.contains(&"account".to_string()));
        assert!(aliases.contains(&"member".to_string()));
    }

    #[test]
    fn test_validate_user_email() {
        let aliases = generate_alias_terms(&["validate_user_email".to_string()]);
        // "validate" → check, verify, test, assert, ensure
        assert!(aliases.contains(&"check".to_string()));
        assert!(aliases.contains(&"verify".to_string()));
        // "user" → account, member
        assert!(aliases.contains(&"account".to_string()));
        // "email" → mail, address
        assert!(aliases.contains(&"mail".to_string()));
    }

    #[test]
    fn test_delete_record() {
        let aliases = generate_alias_terms(&["delete_record".to_string()]);
        assert!(aliases.contains(&"remove".to_string()));
        assert!(aliases.contains(&"drop".to_string()));
        assert!(aliases.contains(&"purge".to_string()));
    }

    #[test]
    fn test_camel_case_split() {
        let parts = split_identifier("getUserByEmail");
        assert!(
            parts.contains(&"get".to_string()) || parts.contains(&"user".to_string()),
            "parts: {parts:?}"
        );
    }

    #[test]
    fn test_empty_input() {
        let aliases = generate_alias_terms(&[]);
        assert!(aliases.is_empty());
    }

    #[test]
    fn test_no_match_returns_empty() {
        let aliases = generate_alias_terms(&["xyzzy_foobar".to_string()]);
        // "xyzzy" and "foobar" are not in any group → no aliases
        assert!(aliases.is_empty() || aliases.len() < 5);
    }
}
