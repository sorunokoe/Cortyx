use std::collections::HashMap;

/// Maps term strings to compact `u32` IDs for cache-friendly BM25 hot paths.
/// Built in memory at index load time; never persisted to disk.
#[derive(Debug, Default)]
pub struct TermInterner {
    term_to_id: HashMap<String, u32>,
    id_to_term: Vec<String>,
}

impl TermInterner {
    pub fn new() -> Self {
        Self::default()
    }

    /// Interns a term, returning its stable `u32` ID.
    /// Terms are never evicted; IDs are stable for the lifetime of this interner.
    pub fn intern(&mut self, term: &str) -> u32 {
        if let Some(&id) = self.term_to_id.get(term) {
            return id;
        }

        let id =
            u32::try_from(self.id_to_term.len()).expect("TermInterner exceeded u32::MAX terms");
        let owned = term.to_owned();
        self.id_to_term.push(owned.clone());
        self.term_to_id.insert(owned, id);
        id
    }

    /// Looks up an already-interned term. Returns `None` if not interned.
    #[allow(dead_code)]
    pub fn get(&self, term: &str) -> Option<u32> {
        self.term_to_id.get(term).copied()
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.id_to_term.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.id_to_term.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::TermInterner;

    #[test]
    fn intern_returns_stable_ids() {
        let mut interner = TermInterner::new();
        let first = interner.intern("alpha");
        let second = interner.intern("alpha");

        assert_eq!(first, second);
    }

    #[test]
    fn different_terms_get_different_ids() {
        let mut interner = TermInterner::new();

        assert_ne!(interner.intern("alpha"), interner.intern("beta"));
    }

    #[test]
    fn get_returns_none_for_unknown() {
        let mut interner = TermInterner::new();
        interner.intern("alpha");

        assert_eq!(interner.get("unknown"), None);
    }

    #[test]
    fn len_tracks_unique_terms() {
        let mut interner = TermInterner::new();
        interner.intern("alpha");
        interner.intern("beta");
        interner.intern("alpha");

        assert_eq!(interner.len(), 2);
    }
}
