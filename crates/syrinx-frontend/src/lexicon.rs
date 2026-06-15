//! Two-tier pronunciation override table (T-01.03).
//!
//! A fixed built-in *default* lexicon is consulted before phonemization; a
//! caller-supplied *user* lexicon layers on top and wins every key collision.
//! `lookup` folds the query key to lowercase before matching either tier and
//! returns the winning replacement as an owned `String`, or `None` when neither
//! tier holds the key. Case folding applies to the key only — the stored
//! replacement value is returned byte-for-byte, never re-cased.

use std::collections::HashMap;

/// A two-tier word→pronunciation override table: a user tier over the built-in
/// default tier, with the user tier winning every collision.
pub struct Lexicon {
    user: HashMap<String, String>,
}

impl Lexicon {
    /// Build a lexicon layering `user` on top of the built-in default tier.
    pub fn with_user(user: HashMap<String, String>) -> Lexicon {
        Lexicon { user }
    }

    /// Look up `word` (case-folded on the key) in the user tier, then the default
    /// tier. Returns the winning replacement verbatim, or `None` on a total miss.
    pub fn lookup(&self, word: &str) -> Option<String> {
        let key = word.to_lowercase();
        if let Some(value) = self.user.get(&key) {
            return Some(value.clone());
        }
        default_lookup(&key)
    }
}

/// The fixed built-in default tier.
fn default_lookup(key: &str) -> Option<String> {
    match key {
        "tomato" => Some("tom-ah-to".to_string()),
        "data" => Some("day-ta".to_string()),
        _ => None,
    }
}
