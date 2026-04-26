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
        self.match_range(text, pos, None)
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
        self.search_range(text, pos, None)
    }

    /// Match entire string against pattern.
    pub fn fullmatch(&self, text: &str) -> Option<Match> {
        self.fullmatch_range(text, 0, None)
    }

    /// Match at the beginning of the bounded string range.
    pub fn match_range(&self, text: &str, pos: usize, endpos: Option<usize>) -> Option<Match> {
        let (substr, offset) = bounded_text_range(text, pos, endpos)?;
        self.engine
            .match_start(substr)
            .map(|m| m.with_offset(text, offset))
    }

    /// Search within a bounded string range.
    pub fn search_range(&self, text: &str, pos: usize, endpos: Option<usize>) -> Option<Match> {
        let (substr, offset) = bounded_text_range(text, pos, endpos)?;
        self.engine
            .find(substr)
            .map(|m| m.with_offset(text, offset))
    }

    /// Fullmatch within a bounded string range.
    pub fn fullmatch_range(&self, text: &str, pos: usize, endpos: Option<usize>) -> Option<Match> {
        let (substr, offset) = bounded_text_range(text, pos, endpos)?;
        let m = self.engine.match_start(substr)?;
        (m.end() == substr.len()).then(|| m.with_offset(text, offset))
    }

    /// Find all non-overlapping matches.
    #[inline]
    pub fn findall(&self, text: &str) -> Vec<Match> {
        self.engine.find_all(text)
    }

    /// Find all matches within a bounded string range.
    pub fn findall_range(&self, text: &str, pos: usize, endpos: Option<usize>) -> Vec<Match> {
        let Some((substr, offset)) = bounded_text_range(text, pos, endpos) else {
            return Vec::new();
        };
        self.engine
            .find_all(substr)
            .into_iter()
            .map(|m| m.with_offset(text, offset))
            .collect()
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

    /// Return an iterator over matches within a bounded string range.
    pub fn finditer_range<'a>(
        &'a self,
        text: &'a str,
        pos: usize,
        endpos: Option<usize>,
    ) -> impl Iterator<Item = Match> + 'a {
        self.findall_range(text, pos, endpos).into_iter()
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

fn bounded_text_range(text: &str, pos: usize, endpos: Option<usize>) -> Option<(&str, usize)> {
    let text_len = text.len();
    let pos = pos.min(text_len);
    let end = endpos.unwrap_or(text_len).min(text_len);
    if pos > end || !text.is_char_boundary(pos) || !text.is_char_boundary(end) {
        return None;
    }
    Some((&text[pos..end], pos))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
