//! Regex engine abstraction layer.
//!
//! Provides a unified interface over:
//! - `regex` crate (fast, O(m*n) guaranteed)
//! - `fancy-regex` (backreferences, lookaround, potentially exponential)
//!
//! # Engine Selection
//!
//! The engine is automatically selected based on pattern complexity:
//! - Patterns without backreferences/lookaround use `regex` (fast path)
//! - Patterns requiring advanced features use `fancy-regex`

use super::flags::RegexFlags;
use super::match_obj::Match;
use regex::Regex as StdRegex;
use std::fmt::Debug;
use std::sync::Arc;

// =============================================================================
// Engine Error
// =============================================================================

/// Regex compilation/execution error.
#[derive(Debug, Clone)]
pub struct RegexError {
    pub kind: RegexErrorKind,
    pub message: String,
    pub pattern: Option<String>,
    pub position: Option<usize>,
}

/// Error kind for categorization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegexErrorKind {
    /// Invalid regex syntax.
    Syntax,
    /// Pattern too complex.
    Complexity,
    /// Unsupported feature.
    Unsupported,
    /// Internal engine error.
    Internal,
}

impl RegexError {
    /// Create a syntax error.
    pub fn syntax(message: impl Into<String>, pattern: Option<String>) -> Self {
        Self {
            kind: RegexErrorKind::Syntax,
            message: message.into(),
            pattern,
            position: None,
        }
    }

    /// Create a complexity error.
    #[allow(dead_code)]
    pub fn complexity(message: impl Into<String>) -> Self {
        Self {
            kind: RegexErrorKind::Complexity,
            message: message.into(),
            pattern: None,
            position: None,
        }
    }
}

impl std::fmt::Display for RegexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for RegexError {}

// =============================================================================
// Engine Kind
// =============================================================================

/// Which regex engine is in use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineKind {
    /// Standard regex crate (fast, guaranteed O(m*n)).
    Standard,
    /// Fancy-regex (supports backreferences, lookaround).
    Fancy,
}

// =============================================================================
// Engine Trait
// =============================================================================

/// Result type for engine operations.
pub type EngineResult<T> = Result<T, RegexError>;

/// Unified regex engine interface.
pub trait Engine: Send + Sync + Debug {
    /// Get the engine kind.
    fn kind(&self) -> EngineKind;

    /// Check if the pattern matches anywhere in the string.
    fn is_match(&self, text: &str) -> bool;

    /// Find the first match in the string.
    fn find(&self, text: &str) -> Option<Match>;

    /// Find all non-overlapping matches.
    fn find_all(&self, text: &str) -> Vec<Match>;

    /// Match at the beginning of the string only.
    fn match_start(&self, text: &str) -> Option<Match>;

    /// Replace first occurrence.
    fn replace(&self, text: &str, replacement: &str) -> String;

    /// Replace all occurrences.
    fn replace_all(&self, text: &str, replacement: &str) -> String;

    /// Split string by pattern (returns owned strings).
    fn split(&self, text: &str) -> Vec<String>;

    /// Split with limit (returns owned strings).
    fn splitn(&self, text: &str, limit: usize) -> Vec<String>;

    /// Get the pattern string.
    fn pattern(&self) -> &str;

    /// Get number of capture groups.
    fn captures_len(&self) -> usize;

    /// Get named capture group names.
    fn group_names(&self) -> Vec<Option<String>>;
}

// =============================================================================
// Standard Engine (regex crate)
// =============================================================================

/// Fast regex engine using the `regex` crate.
///
/// Provides guaranteed O(m*n) time complexity by avoiding
/// backreferences and lookaround assertions.
#[derive(Debug, Clone)]
pub struct StandardEngine {
    regex: StdRegex,
    pattern: Arc<str>,
}

impl StandardEngine {
    /// Compile a pattern with flags.
    pub fn compile(pattern: &str, flags: RegexFlags) -> EngineResult<Self> {
        // Build regex with flags
        let regex_pattern = apply_flags_to_pattern(pattern, flags);

        let regex = regex::RegexBuilder::new(&regex_pattern)
            .case_insensitive(flags.is_case_insensitive())
            .multi_line(flags.is_multiline())
            .dot_matches_new_line(flags.is_dotall())
            .ignore_whitespace(flags.is_verbose())
            .build()
            .map_err(|e| RegexError::syntax(e.to_string(), Some(pattern.to_string())))?;

        Ok(Self {
            regex,
            pattern: Arc::from(pattern),
        })
    }
}

impl Engine for StandardEngine {
    #[inline]
    fn kind(&self) -> EngineKind {
        EngineKind::Standard
    }

    #[inline]
    fn is_match(&self, text: &str) -> bool {
        self.regex.is_match(text)
    }

    fn find(&self, text: &str) -> Option<Match> {
        self.regex
            .captures(text)
            .map(|caps| Match::from_captures(&caps, text))
    }

    fn find_all(&self, text: &str) -> Vec<Match> {
        self.regex
            .captures_iter(text)
            .map(|caps| Match::from_captures(&caps, text))
            .collect()
    }

    fn match_start(&self, text: &str) -> Option<Match> {
        self.regex.captures(text).and_then(|caps| {
            let m = caps.get(0).unwrap();
            if m.start() == 0 {
                Some(Match::from_captures(&caps, text))
            } else {
                None
            }
        })
    }

    fn replace(&self, text: &str, replacement: &str) -> String {
        self.regex.replace(text, replacement).into_owned()
    }

    fn replace_all(&self, text: &str, replacement: &str) -> String {
        self.regex.replace_all(text, replacement).into_owned()
    }

    fn split(&self, text: &str) -> Vec<String> {
        self.regex.split(text).map(|s| s.to_string()).collect()
    }

    fn splitn(&self, text: &str, limit: usize) -> Vec<String> {
        self.regex
            .splitn(text, limit)
            .map(|s| s.to_string())
            .collect()
    }

    #[inline]
    fn pattern(&self) -> &str {
        &self.pattern
    }

    #[inline]
    fn captures_len(&self) -> usize {
        self.regex.captures_len()
    }

    fn group_names(&self) -> Vec<Option<String>> {
        self.regex
            .capture_names()
            .map(|n| n.map(|s| s.to_string()))
            .collect()
    }
}

// =============================================================================
// Fancy Engine (fancy-regex crate)
// =============================================================================

/// Extended regex engine using `fancy-regex`.
///
/// Supports backreferences and lookaround, but with potentially
/// exponential time complexity for pathological patterns.
#[derive(Debug, Clone)]
pub struct FancyEngine {
    regex: fancy_regex::Regex,
    pattern: Arc<str>,
}

impl FancyEngine {
    /// Compile a pattern with flags.
    pub fn compile(pattern: &str, flags: RegexFlags) -> EngineResult<Self> {
        let regex_pattern = apply_flags_to_pattern(pattern, flags);

        let regex = fancy_regex::Regex::new(&regex_pattern)
            .map_err(|e| RegexError::syntax(e.to_string(), Some(pattern.to_string())))?;

        Ok(Self {
            regex,
            pattern: Arc::from(pattern),
        })
    }
}

impl Engine for FancyEngine {
    #[inline]
    fn kind(&self) -> EngineKind {
        EngineKind::Fancy
    }

    #[inline]
    fn is_match(&self, text: &str) -> bool {
        self.regex.is_match(text).unwrap_or(false)
    }

    fn find(&self, text: &str) -> Option<Match> {
        self.regex
            .captures(text)
            .ok()
            .flatten()
            .map(|caps| Match::from_fancy_captures(&caps, text))
    }

    fn find_all(&self, text: &str) -> Vec<Match> {
        self.regex
            .captures_iter(text)
            .filter_map(|r| r.ok())
            .map(|caps| Match::from_fancy_captures(&caps, text))
            .collect()
    }

    fn match_start(&self, text: &str) -> Option<Match> {
        self.regex.captures(text).ok().flatten().and_then(|caps| {
            if let Some(m) = caps.get(0) {
                if m.start() == 0 {
                    return Some(Match::from_fancy_captures(&caps, text));
                }
            }
            None
        })
    }

    fn replace(&self, text: &str, replacement: &str) -> String {
        self.regex.replace(text, replacement).into_owned()
    }

    fn replace_all(&self, text: &str, replacement: &str) -> String {
        self.regex.replace_all(text, replacement).into_owned()
    }

    fn split(&self, text: &str) -> Vec<String> {
        // fancy-regex doesn't have built-in split, implement manually
        let mut result = Vec::new();
        let mut last_end = 0;

        for m in self.regex.find_iter(text).filter_map(|r| r.ok()) {
            if m.start() > last_end {
                result.push(text[last_end..m.start()].to_string());
            }
            last_end = m.end();
        }

        if last_end < text.len() {
            result.push(text[last_end..].to_string());
        } else if last_end == 0 {
            result.push(text.to_string());
        }

        result
    }

    fn splitn(&self, text: &str, limit: usize) -> Vec<String> {
        if limit == 0 {
            return vec![text.to_string()];
        }

        let mut result = Vec::with_capacity(limit);
        let mut last_end = 0;
        let mut count = 0;

        for m in self.regex.find_iter(text).filter_map(|r| r.ok()) {
            if count >= limit - 1 {
                break;
            }
            if m.start() > last_end {
                result.push(text[last_end..m.start()].to_string());
                count += 1;
            }
            last_end = m.end();
        }

        if last_end < text.len() {
            result.push(text[last_end..].to_string());
        }

        result
    }

    #[inline]
    fn pattern(&self) -> &str {
        &self.pattern
    }

    #[inline]
    fn captures_len(&self) -> usize {
        // fancy-regex doesn't expose this directly, parse from pattern
        count_capture_groups(&self.pattern)
    }

    fn group_names(&self) -> Vec<Option<String>> {
        // fancy-regex doesn't expose names directly
        Vec::new()
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Apply flags to pattern using inline modifiers.
fn apply_flags_to_pattern(pattern: &str, flags: RegexFlags) -> String {
    // Check if we need to add any inline modifiers
    let mut modifiers = String::new();

    if flags.is_case_insensitive() {
        modifiers.push('i');
    }
    if flags.is_multiline() {
        modifiers.push('m');
    }
    if flags.is_dotall() {
        modifiers.push('s');
    }
    if flags.is_verbose() {
        modifiers.push('x');
    }

    if modifiers.is_empty() {
        pattern.to_string()
    } else {
        format!("(?{}){}", modifiers, pattern)
    }
}

/// Check if pattern requires fancy-regex features.
pub fn requires_fancy_engine(pattern: &str) -> bool {
    // Check for backreferences: \1, \2, etc.
    let mut chars = pattern.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(&next) = chars.peek() {
                // Backreference
                if next.is_ascii_digit() && next != '0' {
                    return true;
                }
            }
        }
        // Lookahead/lookbehind
        if c == '(' {
            if let Some(&next) = chars.peek() {
                if next == '?' {
                    chars.next();
                    if let Some(&after) = chars.peek() {
                        // (?= positive lookahead
                        // (?! negative lookahead
                        // (?<= positive lookbehind
                        // (?<! negative lookbehind
                        if after == '=' || after == '!' {
                            return true;
                        }
                        if after == '<' {
                            chars.next();
                            if let Some(&lookbehind) = chars.peek() {
                                if lookbehind == '=' || lookbehind == '!' {
                                    return true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    false
}

/// Count capture groups in a pattern.
fn count_capture_groups(pattern: &str) -> usize {
    let mut count = 1; // Group 0 is the full match
    let mut in_escape = false;
    let mut in_class = false;

    for c in pattern.chars() {
        if in_escape {
            in_escape = false;
            continue;
        }
        if c == '\\' {
            in_escape = true;
            continue;
        }
        if c == '[' {
            in_class = true;
            continue;
        }
        if c == ']' && in_class {
            in_class = false;
            continue;
        }
        if !in_class && c == '(' {
            count += 1;
        }
    }

    count
}

/// Compile a pattern, automatically selecting the appropriate engine.
pub fn compile_pattern(pattern: &str, flags: RegexFlags) -> EngineResult<Box<dyn Engine>> {
    if requires_fancy_engine(pattern) {
        Ok(Box::new(FancyEngine::compile(pattern, flags)?))
    } else {
        Ok(Box::new(StandardEngine::compile(pattern, flags)?))
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_engine_compile() {
        let engine = StandardEngine::compile(r"\d+", RegexFlags::default()).unwrap();
        assert_eq!(engine.kind(), EngineKind::Standard);
    }

    #[test]
    fn test_standard_engine_is_match() {
        let engine = StandardEngine::compile(r"\d+", RegexFlags::default()).unwrap();
        assert!(engine.is_match("abc123def"));
        assert!(!engine.is_match("abcdef"));
    }

    #[test]
    fn test_standard_engine_find() {
        let engine = StandardEngine::compile(r"\d+", RegexFlags::default()).unwrap();
        let m = engine.find("abc123def").unwrap();
        assert_eq!(m.start(), 3);
        assert_eq!(m.end(), 6);
    }

    #[test]
    fn test_standard_engine_find_all() {
        let engine = StandardEngine::compile(r"\d+", RegexFlags::default()).unwrap();
        let matches = engine.find_all("a1b22c333");
        assert_eq!(matches.len(), 3);
    }

    #[test]
    fn test_standard_engine_match_start() {
        let engine = StandardEngine::compile(r"\d+", RegexFlags::default()).unwrap();
        assert!(engine.match_start("123abc").is_some());
        assert!(engine.match_start("abc123").is_none());
    }

    #[test]
    fn test_standard_engine_replace() {
        let engine = StandardEngine::compile(r"\d+", RegexFlags::default()).unwrap();
        assert_eq!(engine.replace("a1b2c3", "X"), "aXb2c3");
        assert_eq!(engine.replace_all("a1b2c3", "X"), "aXbXcX");
    }

    #[test]
    fn test_standard_engine_split() {
        let engine = StandardEngine::compile(r",", RegexFlags::default()).unwrap();
        let parts = engine.split("a,b,c");
        assert_eq!(parts, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_case_insensitive() {
        let flags = RegexFlags::new(RegexFlags::IGNORECASE);
        let engine = StandardEngine::compile(r"hello", flags).unwrap();
        assert!(engine.is_match("HELLO"));
        assert!(engine.is_match("Hello"));
    }

    #[test]
    fn test_multiline() {
        let flags = RegexFlags::new(RegexFlags::MULTILINE);
        let engine = StandardEngine::compile(r"^line", flags).unwrap();
        let matches = engine.find_all("line1\nline2\nline3");
        assert_eq!(matches.len(), 3);
    }

    #[test]
    fn test_dotall() {
        let flags = RegexFlags::new(RegexFlags::DOTALL);
        let engine = StandardEngine::compile(r"a.b", flags).unwrap();
        assert!(engine.is_match("a\nb"));
    }

    #[test]
    fn test_requires_fancy_backreference() {
        assert!(requires_fancy_engine(r"(.)\1"));
        assert!(!requires_fancy_engine(r"\d+"));
    }

    #[test]
    fn test_requires_fancy_lookahead() {
        assert!(requires_fancy_engine(r"foo(?=bar)"));
        assert!(requires_fancy_engine(r"foo(?!bar)"));
    }

    #[test]
    fn test_requires_fancy_lookbehind() {
        assert!(requires_fancy_engine(r"(?<=foo)bar"));
        assert!(requires_fancy_engine(r"(?<!foo)bar"));
    }

    #[test]
    fn test_fancy_engine_compile() {
        let engine = FancyEngine::compile(r"(.)\1", RegexFlags::default()).unwrap();
        assert_eq!(engine.kind(), EngineKind::Fancy);
    }

    #[test]
    fn test_fancy_engine_backreference() {
        let engine = FancyEngine::compile(r"(.)\1", RegexFlags::default()).unwrap();
        assert!(engine.is_match("aa"));
        assert!(engine.is_match("bb"));
        assert!(!engine.is_match("ab"));
    }

    #[test]
    fn test_fancy_engine_lookahead() {
        let engine = FancyEngine::compile(r"foo(?=bar)", RegexFlags::default()).unwrap();
        let m = engine.find("foobar");
        assert!(m.is_some());
        let m = m.unwrap();
        assert_eq!(m.as_str(), "foo");
    }

    #[test]
    fn test_compile_pattern_auto_select() {
        let simple = compile_pattern(r"\d+", RegexFlags::default()).unwrap();
        assert_eq!(simple.kind(), EngineKind::Standard);

        let fancy = compile_pattern(r"(.)\1", RegexFlags::default()).unwrap();
        assert_eq!(fancy.kind(), EngineKind::Fancy);
    }
}
