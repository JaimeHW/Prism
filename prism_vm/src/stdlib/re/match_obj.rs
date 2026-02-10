//! Match object implementation.
//!
//! Python `re.Match` equivalent with all standard methods:
//! - `group()`, `groups()`, `groupdict()`
//! - `start()`, `end()`, `span()`
//! - String extraction with zero-copy slices

use rustc_hash::FxHashMap;
use std::ops::Range;
use std::sync::Arc;

// =============================================================================
// Match Object
// =============================================================================

/// A regex match result.
///
/// Contains the matched string, capture groups, and position information.
#[derive(Debug, Clone)]
pub struct Match {
    /// Full matched string.
    string: Arc<str>,
    /// Span of the full match in the original string.
    full_span: Range<usize>,
    /// Capture group spans (group 0 is the full match).
    groups: Vec<Option<Range<usize>>>,
    /// Named capture groups.
    named_groups: FxHashMap<Arc<str>, usize>,
}

impl Match {
    /// Create a new match from spans.
    pub fn new(
        string: Arc<str>,
        full_span: Range<usize>,
        groups: Vec<Option<Range<usize>>>,
        named_groups: FxHashMap<Arc<str>, usize>,
    ) -> Self {
        Self {
            string,
            full_span,
            groups,
            named_groups,
        }
    }

    /// Create from `regex::Captures`.
    pub fn from_captures(caps: &regex::Captures, text: &str) -> Self {
        let string = Arc::from(text);

        // Get full match span
        let full_match = caps.get(0).unwrap();
        let full_span = full_match.start()..full_match.end();

        // Extract all group spans
        let groups: Vec<Option<Range<usize>>> =
            caps.iter().map(|m| m.map(|m| m.start()..m.end())).collect();

        // Extract named groups
        let mut named_groups = FxHashMap::default();
        for name in caps.name("") {
            // This doesn't work as expected in regex crate
            // We need to iterate over the regex's capture names
            let _ = name;
        }

        Self {
            string,
            full_span,
            groups,
            named_groups,
        }
    }

    /// Create from `regex::Captures` with regex for named groups.
    pub fn from_captures_with_regex(
        caps: &regex::Captures,
        text: &str,
        regex: &regex::Regex,
    ) -> Self {
        let string = Arc::from(text);

        // Get full match span
        let full_match = caps.get(0).unwrap();
        let full_span = full_match.start()..full_match.end();

        // Extract all group spans
        let groups: Vec<Option<Range<usize>>> =
            caps.iter().map(|m| m.map(|m| m.start()..m.end())).collect();

        // Extract named groups
        let mut named_groups = FxHashMap::default();
        for (i, name_opt) in regex.capture_names().enumerate() {
            if let Some(name) = name_opt {
                named_groups.insert(Arc::from(name), i);
            }
        }

        Self {
            string,
            full_span,
            groups,
            named_groups,
        }
    }

    /// Create from `fancy_regex::Captures`.
    pub fn from_fancy_captures(caps: &fancy_regex::Captures, text: &str) -> Self {
        let string = Arc::from(text);

        // Get full match span
        let full_match = caps.get(0).unwrap();
        let full_span = full_match.start()..full_match.end();

        // Extract all group spans
        let groups: Vec<Option<Range<usize>>> =
            caps.iter().map(|m| m.map(|m| m.start()..m.end())).collect();

        // Named groups not directly accessible in fancy-regex
        let named_groups = FxHashMap::default();

        Self {
            string,
            full_span,
            groups,
            named_groups,
        }
    }

    // =========================================================================
    // Python API Methods
    // =========================================================================

    /// Return the entire match as a string slice.
    #[inline]
    pub fn as_str(&self) -> &str {
        &self.string[self.full_span.clone()]
    }

    /// Return group `n` as a string slice.
    ///
    /// Group 0 is the entire match.
    pub fn group(&self, n: usize) -> Option<&str> {
        self.groups
            .get(n)
            .and_then(|opt| opt.as_ref().map(|span| &self.string[span.clone()]))
    }

    /// Return all groups except group 0 as a tuple (Vec).
    pub fn groups(&self) -> Vec<Option<&str>> {
        self.groups
            .iter()
            .skip(1) // Skip group 0 (full match)
            .map(|opt| opt.as_ref().map(|span| &self.string[span.clone()]))
            .collect()
    }

    /// Return all groups with default value for unmatched groups.
    pub fn groups_with_default<'a>(&'a self, default: &'a str) -> Vec<&'a str> {
        self.groups
            .iter()
            .skip(1)
            .map(|opt| {
                opt.as_ref()
                    .map(|span| &self.string[span.clone()])
                    .unwrap_or(default)
            })
            .collect()
    }

    /// Return a dict of named groups.
    pub fn groupdict(&self) -> FxHashMap<Arc<str>, Option<&str>> {
        self.named_groups
            .iter()
            .map(|(name, &idx)| {
                let value = self
                    .groups
                    .get(idx)
                    .and_then(|opt| opt.as_ref().map(|span| &self.string[span.clone()]));
                (name.clone(), value)
            })
            .collect()
    }

    /// Return the start position of the match.
    #[inline]
    pub fn start(&self) -> usize {
        self.full_span.start
    }

    /// Return the start position of a specific group.
    pub fn start_group(&self, n: usize) -> Option<usize> {
        self.groups
            .get(n)
            .and_then(|opt| opt.as_ref().map(|span| span.start))
    }

    /// Return the end position of the match.
    #[inline]
    pub fn end(&self) -> usize {
        self.full_span.end
    }

    /// Return the end position of a specific group.
    pub fn end_group(&self, n: usize) -> Option<usize> {
        self.groups
            .get(n)
            .and_then(|opt| opt.as_ref().map(|span| span.end))
    }

    /// Return the span (start, end) of the match.
    #[inline]
    pub fn span(&self) -> (usize, usize) {
        (self.full_span.start, self.full_span.end)
    }

    /// Return the span of a specific group.
    pub fn span_group(&self, n: usize) -> Option<(usize, usize)> {
        self.groups
            .get(n)
            .and_then(|opt| opt.as_ref().map(|span| (span.start, span.end)))
    }

    /// Return the original string.
    #[inline]
    pub fn string(&self) -> &str {
        &self.string
    }

    /// Return the number of groups (including group 0).
    #[inline]
    pub fn len(&self) -> usize {
        self.groups.len()
    }

    /// Check if match is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.full_span.start == self.full_span.end
    }

    /// Return the last group index.
    pub fn lastindex(&self) -> Option<usize> {
        // Return the index of the last matched group
        for (i, group) in self.groups.iter().enumerate().rev() {
            if i > 0 && group.is_some() {
                return Some(i);
            }
        }
        None
    }

    /// Return the last named group.
    pub fn lastgroup(&self) -> Option<&str> {
        if let Some(last_idx) = self.lastindex() {
            for (name, &idx) in &self.named_groups {
                if idx == last_idx {
                    return Some(name.as_ref());
                }
            }
        }
        None
    }
}

impl std::fmt::Display for Match {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "<re.Match object; span=({}, {}), match='{}'>",
            self.start(),
            self.end(),
            self.as_str()
        )
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_match() -> Match {
        let text = Arc::from("hello world");
        let groups = vec![
            Some(0..5),  // group 0: "hello"
            Some(6..11), // group 1: "world"
        ];
        let mut named_groups = FxHashMap::default();
        named_groups.insert(Arc::from("word"), 1);
        Match::new(text, 0..5, groups, named_groups)
    }

    #[test]
    fn test_as_str() {
        let m = create_test_match();
        assert_eq!(m.as_str(), "hello");
    }

    #[test]
    fn test_group() {
        let m = create_test_match();
        assert_eq!(m.group(0), Some("hello"));
        assert_eq!(m.group(1), Some("world"));
        assert_eq!(m.group(2), None);
    }

    #[test]
    fn test_groups() {
        let m = create_test_match();
        let groups = m.groups();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0], Some("world"));
    }

    #[test]
    fn test_start_end_span() {
        let m = create_test_match();
        assert_eq!(m.start(), 0);
        assert_eq!(m.end(), 5);
        assert_eq!(m.span(), (0, 5));
    }

    #[test]
    fn test_start_end_group() {
        let m = create_test_match();
        assert_eq!(m.start_group(0), Some(0));
        assert_eq!(m.end_group(0), Some(5));
        assert_eq!(m.start_group(1), Some(6));
        assert_eq!(m.end_group(1), Some(11));
    }

    #[test]
    fn test_groupdict() {
        let m = create_test_match();
        let dict = m.groupdict();
        assert_eq!(dict.get(&Arc::from("word")), Some(&Some("world")));
    }

    #[test]
    fn test_lastindex() {
        let m = create_test_match();
        assert_eq!(m.lastindex(), Some(1));
    }

    #[test]
    fn test_lastgroup() {
        let m = create_test_match();
        assert_eq!(m.lastgroup(), Some("word"));
    }

    #[test]
    fn test_display() {
        let m = create_test_match();
        let s = m.to_string();
        assert!(s.contains("span=(0, 5)"));
        assert!(s.contains("match='hello'"));
    }

    #[test]
    fn test_from_regex_captures() {
        let re = regex::Regex::new(r"(\d+)-(\d+)").unwrap();
        let caps = re.captures("123-456").unwrap();
        let m = Match::from_captures(&caps, "123-456");
        assert_eq!(m.as_str(), "123-456");
        assert_eq!(m.group(1), Some("123"));
        assert_eq!(m.group(2), Some("456"));
    }
}
