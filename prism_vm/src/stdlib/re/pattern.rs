//! Compiled pattern object.
//!
//! Python `re.Pattern` equivalent - a pre-compiled regex that can be
//! reused for multiple match operations.

use super::engine::{Engine, EngineKind, EngineResult, compile_pattern};
use super::flags::RegexFlags;
use super::match_obj::Match;
use std::sync::Arc;

// =============================================================================
// Compiled Pattern
// =============================================================================

/// A compiled regular expression pattern.
///
/// This is the Rust equivalent of Python's `re.Pattern` object.
/// Compiled patterns can be reused for multiple match operations,
/// avoiding the overhead of repeated compilation.
#[derive(Debug)]
pub struct CompiledPattern {
    /// The underlying engine.
    engine: Box<dyn Engine>,
    /// Original pattern string.
    pattern: Arc<str>,
    /// Compilation flags.
    flags: RegexFlags,
}

impl CompiledPattern {
    /// Compile a pattern with optional flags.
    pub fn compile(pattern: &str, flags: RegexFlags) -> EngineResult<Self> {
        let engine = compile_pattern(pattern, flags)?;
        Ok(Self {
            engine,
            pattern: Arc::from(pattern),
            flags,
        })
    }

    /// Compile with default flags.
    pub fn compile_default(pattern: &str) -> EngineResult<Self> {
        Self::compile(pattern, RegexFlags::default())
    }

    // =========================================================================
    // Properties
    // =========================================================================

    /// Get the pattern string.
    #[inline]
    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    /// Get the flags.
    #[inline]
    pub fn flags(&self) -> RegexFlags {
        self.flags
    }

    /// Get the engine kind.
    #[inline]
    pub fn engine_kind(&self) -> EngineKind {
        self.engine.kind()
    }

    /// Get number of groups.
    #[inline]
    pub fn groups(&self) -> usize {
        self.engine.captures_len()
    }

    /// Get group names.
    pub fn groupindex(&self) -> Vec<Option<String>> {
        self.engine.group_names()
    }

    // =========================================================================
    // Matching Methods
    // =========================================================================

    /// Check if pattern matches anywhere in string.
    #[inline]
    pub fn is_match(&self, text: &str) -> bool {
        self.engine.is_match(text)
    }

    /// Try to match pattern at start of string.
    ///
    /// Returns `Some(Match)` if pattern matches at position 0.
    #[inline]
    pub fn match_(&self, text: &str) -> Option<Match> {
        self.engine.match_start(text)
    }

    /// Try to match pattern at start of string from position.
    pub fn match_at(&self, text: &str, pos: usize) -> Option<Match> {
        if pos > text.len() {
            return None;
        }
        let substr = &text[pos..];
        self.engine.match_start(substr).map(|mut m| {
            // Adjust spans for the offset
            m
        })
    }

    /// Search for pattern anywhere in string.
    ///
    /// Returns the first match found.
    #[inline]
    pub fn search(&self, text: &str) -> Option<Match> {
        self.engine.find(text)
    }

    /// Search from a specific position.
    pub fn search_at(&self, text: &str, pos: usize) -> Option<Match> {
        if pos > text.len() {
            return None;
        }
        let substr = &text[pos..];
        self.engine.find(substr)
    }

    /// Match entire string against pattern.
    pub fn fullmatch(&self, text: &str) -> Option<Match> {
        let m = self.engine.match_start(text)?;
        if m.end() == text.len() { Some(m) } else { None }
    }

    /// Find all non-overlapping matches.
    #[inline]
    pub fn findall(&self, text: &str) -> Vec<Match> {
        self.engine.find_all(text)
    }

    /// Find all matches as strings (matching Python behavior).
    ///
    /// If pattern has no groups, returns list of full matches.
    /// If pattern has one group, returns list of group contents.
    /// If pattern has multiple groups, returns list of tuples.
    pub fn findall_strings(&self, text: &str) -> Vec<Vec<String>> {
        let matches = self.engine.find_all(text);
        let num_groups = self.engine.captures_len();

        matches
            .into_iter()
            .map(|m| {
                if num_groups <= 1 {
                    // No capture groups - return full match
                    vec![m.as_str().to_string()]
                } else if num_groups == 2 {
                    // One capture group - return just that group
                    vec![m.group(1).unwrap_or("").to_string()]
                } else {
                    // Multiple groups - return all groups
                    m.groups()
                        .into_iter()
                        .map(|g| g.unwrap_or("").to_string())
                        .collect()
                }
            })
            .collect()
    }

    /// Return iterator over all matches.
    pub fn finditer<'a>(&'a self, text: &'a str) -> impl Iterator<Item = Match> + 'a {
        self.engine.find_all(text).into_iter()
    }

    // =========================================================================
    // Substitution Methods
    // =========================================================================

    /// Replace first occurrence.
    #[inline]
    pub fn sub(&self, repl: &str, text: &str) -> String {
        self.engine.replace(text, repl)
    }

    /// Replace first N occurrences (0 means all).
    pub fn sub_n(&self, repl: &str, text: &str, count: usize) -> String {
        if count == 0 {
            self.engine.replace_all(text, repl)
        } else if count == 1 {
            self.engine.replace(text, repl)
        } else {
            // Implement counting replacement
            let mut result = text.to_string();
            let mut replaced = 0;
            while replaced < count {
                let new = self.engine.replace(&result, repl);
                if new == result {
                    break;
                }
                result = new;
                replaced += 1;
            }
            result
        }
    }

    /// Replace with count returned.
    pub fn subn(&self, repl: &str, text: &str, count: usize) -> (String, usize) {
        let matches = self.engine.find_all(text);
        let num_matches = if count == 0 {
            matches.len()
        } else {
            matches.len().min(count)
        };

        let result = self.sub_n(repl, text, count);
        (result, num_matches)
    }

    // =========================================================================
    // Split Methods
    // =========================================================================

    /// Split string by pattern.
    #[inline]
    pub fn split(&self, text: &str) -> Vec<String> {
        self.engine.split(text)
    }

    /// Split with limit.
    #[inline]
    pub fn split_n(&self, text: &str, maxsplit: usize) -> Vec<String> {
        if maxsplit == 0 {
            self.engine.split(text)
        } else {
            self.engine.splitn(text, maxsplit + 1)
        }
    }
}

impl Clone for CompiledPattern {
    fn clone(&self) -> Self {
        // Re-compile the pattern (engines aren't directly clonable due to Box<dyn>)
        Self::compile(&self.pattern, self.flags).expect("pattern should be valid")
    }
}

impl std::fmt::Display for CompiledPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "re.compile({:?}, {})", self.pattern, self.flags)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile() {
        let pattern = CompiledPattern::compile_default(r"\d+").unwrap();
        assert_eq!(pattern.pattern(), r"\d+");
        assert_eq!(pattern.engine_kind(), EngineKind::Standard);
    }

    #[test]
    fn test_compile_with_flags() {
        let flags = RegexFlags::new(RegexFlags::IGNORECASE);
        let pattern = CompiledPattern::compile(r"hello", flags).unwrap();
        assert!(pattern.is_match("HELLO"));
    }

    #[test]
    fn test_match() {
        let pattern = CompiledPattern::compile_default(r"\d+").unwrap();
        assert!(pattern.match_("123abc").is_some());
        assert!(pattern.match_("abc123").is_none());
    }

    #[test]
    fn test_search() {
        let pattern = CompiledPattern::compile_default(r"\d+").unwrap();
        let m = pattern.search("abc123def").unwrap();
        assert_eq!(m.as_str(), "123");
        assert_eq!(m.start(), 3);
    }

    #[test]
    fn test_fullmatch() {
        let pattern = CompiledPattern::compile_default(r"\d+").unwrap();
        assert!(pattern.fullmatch("123").is_some());
        assert!(pattern.fullmatch("123abc").is_none());
    }

    #[test]
    fn test_findall() {
        let pattern = CompiledPattern::compile_default(r"\d+").unwrap();
        let matches = pattern.findall("a1b22c333");
        assert_eq!(matches.len(), 3);
        assert_eq!(matches[0].as_str(), "1");
        assert_eq!(matches[1].as_str(), "22");
        assert_eq!(matches[2].as_str(), "333");
    }

    #[test]
    fn test_findall_with_groups() {
        let pattern = CompiledPattern::compile_default(r"(\d+)-(\d+)").unwrap();
        let strings = pattern.findall_strings("1-2 3-4 5-6");
        assert_eq!(strings.len(), 3);
        assert_eq!(strings[0], vec!["1", "2"]);
    }

    #[test]
    fn test_sub() {
        let pattern = CompiledPattern::compile_default(r"\d+").unwrap();
        assert_eq!(pattern.sub("X", "a1b2c3"), "aXb2c3");
    }

    #[test]
    fn test_sub_n() {
        let pattern = CompiledPattern::compile_default(r"\d+").unwrap();
        assert_eq!(pattern.sub_n("X", "a1b2c3", 0), "aXbXcX");
        assert_eq!(pattern.sub_n("X", "a1b2c3", 2), "aXbXc3");
    }

    #[test]
    fn test_split() {
        let pattern = CompiledPattern::compile_default(r",\s*").unwrap();
        let parts = pattern.split("a, b,  c");
        assert_eq!(parts, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_split_n() {
        let pattern = CompiledPattern::compile_default(r",").unwrap();
        let parts = pattern.split_n("a,b,c,d", 2);
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[2], "c,d");
    }

    #[test]
    fn test_display() {
        let pattern = CompiledPattern::compile_default(r"\d+").unwrap();
        let s = pattern.to_string();
        assert!(s.contains("re.compile"));
        assert!(s.contains(r"\d+"));
    }

    #[test]
    fn test_clone() {
        let pattern = CompiledPattern::compile_default(r"\d+").unwrap();
        let cloned = pattern.clone();
        assert_eq!(cloned.pattern(), pattern.pattern());
    }

    #[test]
    fn test_groups_count() {
        let pattern = CompiledPattern::compile_default(r"(\d+)-(\d+)").unwrap();
        assert_eq!(pattern.groups(), 3); // Full match + 2 groups
    }

    #[test]
    fn test_backreference_pattern() {
        let pattern = CompiledPattern::compile_default(r"(.)\1").unwrap();
        assert_eq!(pattern.engine_kind(), EngineKind::Fancy);
        assert!(pattern.is_match("aa"));
        assert!(!pattern.is_match("ab"));
    }
}
