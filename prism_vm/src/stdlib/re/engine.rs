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
        let regex_pattern = prepare_pattern_for_backend(pattern, flags)?;

        let regex = regex::RegexBuilder::new(&regex_pattern)
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
            .map(|caps| Match::from_captures_with_regex(&caps, text, &self.regex))
    }

    fn find_all(&self, text: &str) -> Vec<Match> {
        self.regex
            .captures_iter(text)
            .map(|caps| Match::from_captures_with_regex(&caps, text, &self.regex))
            .collect()
    }

    fn match_start(&self, text: &str) -> Option<Match> {
        self.regex.captures(text).and_then(|caps| {
            let m = caps.get(0).unwrap();
            if m.start() == 0 {
                Some(Match::from_captures_with_regex(&caps, text, &self.regex))
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
        let regex_pattern = prepare_pattern_for_backend(pattern, flags)?;

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
            .map(|caps| Match::from_fancy_captures_with_regex(&caps, text, &self.regex))
    }

    fn find_all(&self, text: &str) -> Vec<Match> {
        self.regex
            .captures_iter(text)
            .filter_map(|r| r.ok())
            .map(|caps| Match::from_fancy_captures_with_regex(&caps, text, &self.regex))
            .collect()
    }

    fn match_start(&self, text: &str) -> Option<Match> {
        self.regex.captures(text).ok().flatten().and_then(|caps| {
            if let Some(m) = caps.get(0) {
                if m.start() == 0 {
                    return Some(Match::from_fancy_captures_with_regex(
                        &caps,
                        text,
                        &self.regex,
                    ));
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
        self.regex.captures_len()
    }

    fn group_names(&self) -> Vec<Option<String>> {
        self.regex
            .capture_names()
            .map(|name| name.map(|name| name.to_string()))
            .collect()
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Prepare a Python pattern for the backend regex engine.
pub(crate) fn prepare_pattern_for_backend(
    pattern: &str,
    flags: RegexFlags,
) -> EngineResult<String> {
    if flags.contains(RegexFlags::LOCALE) {
        return Err(RegexError {
            kind: RegexErrorKind::Unsupported,
            message: "LOCALE flag is not yet supported".to_string(),
            pattern: Some(pattern.to_string()),
            position: None,
        });
    }

    let normalized = normalize_python_pattern(pattern)?;
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
        return Ok(normalized);
    }

    Ok(format!("(?{}){}", modifiers, normalized))
}

/// Normalize Python-specific syntax to the equivalent backend syntax.
///
/// Rust's `regex` ecosystem uses `\z` for the strict end-of-string anchor while
/// CPython exposes `\Z`. Translating it here keeps the Python-visible surface
/// compatible without forcing callers to know about backend-specific syntax.
pub(crate) fn normalize_python_pattern(pattern: &str) -> EngineResult<String> {
    let mut normalized = String::with_capacity(pattern.len());
    let mut in_class = false;
    let mut class_item_count = 0usize;
    let mut offset = 0;

    while offset < pattern.len() {
        let ch = pattern[offset..]
            .chars()
            .next()
            .expect("slice should begin on a character boundary");
        match ch {
            '\\' => {
                let (translated, consumed) = normalize_escape_sequence(pattern, offset, in_class)?;
                normalized.push_str(&translated);
                offset += consumed;
            }
            '[' if !in_class => {
                in_class = true;
                class_item_count = 0;
                normalized.push(ch);
                offset += ch.len_utf8();
            }
            '[' if in_class => {
                normalized.push('\\');
                normalized.push('[');
                class_item_count += 1;
                offset += ch.len_utf8();
            }
            '{' if !in_class => {
                if let Some((translated, consumed)) = normalize_brace_sequence(&pattern[offset..]) {
                    normalized.push_str(&translated);
                    offset += consumed;
                } else {
                    normalized.push('\\');
                    normalized.push('{');
                    offset += ch.len_utf8();
                }
            }
            ']' if in_class => {
                if class_item_count == 0 {
                    normalized.push('\\');
                    normalized.push(']');
                    class_item_count += 1;
                } else {
                    in_class = false;
                    normalized.push(ch);
                }
                offset += ch.len_utf8();
            }
            '}' if !in_class => {
                normalized.push('\\');
                normalized.push('}');
                offset += ch.len_utf8();
            }
            '(' if !in_class => {
                if pattern[offset..].starts_with("(?>") {
                    normalized.push_str("(?:");
                    offset += 3;
                    continue;
                }
                if let Some((translated, consumed)) =
                    translate_inline_flag_group(&pattern[offset..], pattern)?
                {
                    normalized.push_str(&translated);
                    offset += consumed;
                } else {
                    normalized.push(ch);
                    offset += ch.len_utf8();
                }
            }
            _ => {
                normalized.push(ch);
                if in_class {
                    class_item_count += 1;
                }
                offset += ch.len_utf8();
            }
        }
    }

    Ok(normalized)
}

fn normalize_escape_sequence(
    pattern: &str,
    offset: usize,
    in_class: bool,
) -> EngineResult<(String, usize)> {
    let bytes = pattern.as_bytes();
    debug_assert_eq!(bytes[offset], b'\\');

    if offset + 1 >= bytes.len() {
        return Ok(("\\".to_string(), 1));
    }

    if let Some((value, consumed)) = parse_python_octal_escape(bytes, offset, in_class)? {
        return Ok((format!(r"\x{{{:x}}}", value), consumed));
    }

    let next = pattern[offset + 1..]
        .chars()
        .next()
        .expect("slice should begin on a character boundary");
    let mut translated = String::with_capacity(1 + next.len_utf8());
    translated.push('\\');
    if !in_class && next == 'Z' {
        translated.push('z');
    } else {
        translated.push(next);
    }

    Ok((translated, 1 + next.len_utf8()))
}

fn parse_python_octal_escape(
    bytes: &[u8],
    offset: usize,
    in_class: bool,
) -> EngineResult<Option<(u32, usize)>> {
    let first_index = offset + 1;
    if first_index >= bytes.len() || !is_ascii_octal_digit(bytes[first_index]) {
        return Ok(None);
    }

    let first = bytes[first_index];
    let mut digits = [0u8; 3];
    let digit_count = if in_class || first == b'0' {
        let mut count = 1usize;
        digits[0] = first;
        while count < 3 {
            let next_index = first_index + count;
            if next_index >= bytes.len() || !is_ascii_octal_digit(bytes[next_index]) {
                break;
            }
            digits[count] = bytes[next_index];
            count += 1;
        }
        count
    } else if first_index + 2 < bytes.len()
        && is_ascii_octal_digit(bytes[first_index + 1])
        && is_ascii_octal_digit(bytes[first_index + 2])
    {
        digits[0] = first;
        digits[1] = bytes[first_index + 1];
        digits[2] = bytes[first_index + 2];
        3
    } else {
        return Ok(None);
    };

    let value = digits[..digit_count]
        .iter()
        .fold(0u32, |acc, digit| (acc << 3) + u32::from(digit - b'0'));
    if value > 0o377 {
        let escape =
            std::str::from_utf8(&bytes[offset..first_index + digit_count]).unwrap_or("\\???");
        return Err(RegexError {
            kind: RegexErrorKind::Syntax,
            message: format!("octal escape value {escape} outside of range 0-0o377"),
            pattern: None,
            position: Some(offset),
        });
    }

    Ok(Some((value, 1 + digit_count)))
}

fn python_octal_escape_len(bytes: &[u8], offset: usize, in_class: bool) -> Option<usize> {
    let first_index = offset + 1;
    if first_index >= bytes.len() || !is_ascii_octal_digit(bytes[first_index]) {
        return None;
    }

    let first = bytes[first_index];
    if in_class || first == b'0' {
        let mut count = 1usize;
        while count < 3 {
            let next_index = first_index + count;
            if next_index >= bytes.len() || !is_ascii_octal_digit(bytes[next_index]) {
                break;
            }
            count += 1;
        }
        return Some(1 + count);
    }

    if first_index + 2 < bytes.len()
        && is_ascii_octal_digit(bytes[first_index + 1])
        && is_ascii_octal_digit(bytes[first_index + 2])
    {
        Some(4)
    } else {
        None
    }
}

#[inline]
fn is_ascii_octal_digit(byte: u8) -> bool {
    matches!(byte, b'0'..=b'7')
}

fn normalize_brace_sequence(input: &str) -> Option<(String, usize)> {
    let bytes = input.as_bytes();
    if bytes.first() != Some(&b'{') {
        return None;
    }

    let mut index = 1;
    let first_digits_start = index;
    while index < bytes.len() && bytes[index].is_ascii_digit() {
        index += 1;
    }

    if index < bytes.len() && bytes[index] == b'}' && index > first_digits_start {
        return Some((input[..=index].to_string(), index + 1));
    }

    if index >= bytes.len() || bytes[index] != b',' {
        return None;
    }

    let has_leading_bound = index > first_digits_start;
    index += 1;
    let second_digits_start = index;
    while index < bytes.len() && bytes[index].is_ascii_digit() {
        index += 1;
    }

    if index >= bytes.len() || bytes[index] != b'}' {
        return None;
    }

    if !has_leading_bound && second_digits_start == index {
        return None;
    }

    if !has_leading_bound {
        let upper = &input[second_digits_start..index];
        return Some((format!("{{0,{upper}}}"), index + 1));
    }

    Some((input[..=index].to_string(), index + 1))
}

fn translate_inline_flag_group(
    input: &str,
    original_pattern: &str,
) -> EngineResult<Option<(String, usize)>> {
    let bytes = input.as_bytes();
    if bytes.len() < 4 || bytes[0] != b'(' || bytes[1] != b'?' {
        return Ok(None);
    }

    match bytes[2] {
        b':' | b'=' | b'!' | b'#' | b'P' | b'<' => return Ok(None),
        _ => {}
    }

    let mut index = 2;
    let mut saw_flag = false;
    let mut saw_dash = false;
    let mut enabled = String::new();
    let mut disabled = String::new();

    while index < bytes.len() {
        match bytes[index] {
            b'i' | b'm' | b's' | b'x' | b'a' | b'u' | b'L' => {
                saw_flag = true;
                if saw_dash {
                    disabled.push(bytes[index] as char);
                } else {
                    enabled.push(bytes[index] as char);
                }
                index += 1;
            }
            b'-' if saw_flag && !saw_dash => {
                saw_dash = true;
                index += 1;
            }
            b':' | b')' if saw_flag => {
                let spec = translate_inline_flag_spec(&enabled, &disabled, original_pattern)?;
                let delimiter = bytes[index] as char;
                return Ok(Some((format!("(?{spec}{delimiter}"), index + 1)));
            }
            _ => return Ok(None),
        }
    }

    Ok(None)
}

fn translate_inline_flag_spec(
    enabled: &str,
    disabled: &str,
    pattern: &str,
) -> EngineResult<String> {
    let mut rust_enabled = String::new();
    let mut rust_disabled = String::new();
    let mut unicode_mode: Option<bool> = None;

    for flag in enabled.chars() {
        match flag {
            'i' | 'm' | 's' | 'x' => rust_enabled.push(flag),
            'a' => {
                if unicode_mode == Some(true) {
                    return Err(RegexError::syntax(
                        "inline flags 'a' and 'u' are mutually exclusive",
                        Some(pattern.to_string()),
                    ));
                }
                unicode_mode = Some(false);
            }
            'u' => {
                if unicode_mode == Some(false) {
                    return Err(RegexError::syntax(
                        "inline flags 'a' and 'u' are mutually exclusive",
                        Some(pattern.to_string()),
                    ));
                }
                unicode_mode = Some(true);
            }
            'L' => {
                return Err(RegexError {
                    kind: RegexErrorKind::Unsupported,
                    message: "locale-dependent inline regex flags are not yet supported"
                        .to_string(),
                    pattern: Some(pattern.to_string()),
                    position: None,
                });
            }
            _ => {}
        }
    }

    for flag in disabled.chars() {
        match flag {
            'i' | 'm' | 's' | 'x' => rust_disabled.push(flag),
            'u' => unicode_mode = Some(false),
            'a' | 'L' => {
                return Err(RegexError::syntax(
                    format!("inline flag '{flag}' cannot be disabled"),
                    Some(pattern.to_string()),
                ));
            }
            _ => {}
        }
    }

    match unicode_mode {
        Some(true) => rust_enabled.push('u'),
        Some(false) => rust_disabled.push('u'),
        None => {}
    }

    if rust_disabled.is_empty() {
        Ok(rust_enabled)
    } else if rust_enabled.is_empty() {
        Ok(format!("-{rust_disabled}"))
    } else {
        Ok(format!("{rust_enabled}-{rust_disabled}"))
    }
}

/// Check if pattern requires fancy-regex features.
pub fn requires_fancy_engine(pattern: &str) -> bool {
    let bytes = pattern.as_bytes();
    let mut in_class = false;
    let mut class_item_count = 0usize;
    let mut offset = 0usize;

    while offset < pattern.len() {
        let ch = pattern[offset..]
            .chars()
            .next()
            .expect("slice should begin on a character boundary");

        if ch == '\\' {
            if let Some(consumed) = python_octal_escape_len(bytes, offset, in_class) {
                offset += consumed;
                continue;
            }

            let next_index = offset + 1;
            if !in_class
                && next_index < pattern.len()
                && bytes[next_index].is_ascii_digit()
                && bytes[next_index] != b'0'
            {
                return true;
            }

            offset += 1;
            if offset < pattern.len() {
                let escaped = pattern[offset..]
                    .chars()
                    .next()
                    .expect("slice should begin on a character boundary");
                offset += escaped.len_utf8();
            }
            if in_class {
                class_item_count += 1;
            }
            continue;
        }

        if ch == '[' && !in_class {
            in_class = true;
            class_item_count = 0;
            offset += ch.len_utf8();
            continue;
        }

        if ch == ']' && in_class {
            if class_item_count == 0 {
                class_item_count += 1;
            } else {
                in_class = false;
            }
            offset += ch.len_utf8();
            continue;
        }

        if !in_class && ch == '(' {
            let suffix = &pattern[offset..];
            if suffix.starts_with("(?=")
                || suffix.starts_with("(?!")
                || suffix.starts_with("(?<=")
                || suffix.starts_with("(?<!")
                || suffix.starts_with("(?P=")
            {
                return true;
            }
        }

        if in_class {
            class_item_count += 1;
        }
        offset += ch.len_utf8();
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
        assert!(requires_fancy_engine(r#"(?P<quote>['"])(?P=quote)"#));
        assert!(!requires_fancy_engine(r"\111"));
        assert!(!requires_fancy_engine(r"[\041-\176]+:$"));
        assert!(!requires_fancy_engine(r"\\1"));
        assert!(!requires_fancy_engine(r"\d+"));
    }

    #[test]
    fn test_python_named_backreference_uses_fancy_engine() {
        let engine = compile_pattern(
            r#"^(?P<name>\w+)=(?P<quote>["']?)(?P<value>.*)(?P=quote)$"#,
            RegexFlags::default(),
        )
        .expect("named Python backreference should compile");
        assert_eq!(engine.kind(), EngineKind::Fancy);
        let matched = engine.match_start("NAME=\"Prism\"").expect("match");
        assert_eq!(
            matched.group(matched.group_index("name").expect("name group")),
            Some("NAME")
        );
        assert_eq!(
            matched.group(matched.group_index("value").expect("value group")),
            Some("Prism")
        );
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

    #[test]
    fn test_fancy_engine_reports_named_group_metadata() {
        let engine = compile_pattern(r"(?P<word>foo)(?!bar)", RegexFlags::default()).unwrap();
        assert_eq!(engine.kind(), EngineKind::Fancy);
        let group_names = engine.group_names();
        assert_eq!(group_names.len(), 2);
        assert_eq!(group_names[1].as_deref(), Some("word"));
    }

    #[test]
    fn test_normalize_python_pattern_converts_end_of_string_anchor() {
        assert_eq!(normalize_python_pattern(r"foo\Z").unwrap(), r"foo\z");
    }

    #[test]
    fn test_normalize_python_pattern_preserves_escaped_anchor_literal() {
        assert_eq!(normalize_python_pattern(r"foo\\Z").unwrap(), r"foo\\Z");
    }

    #[test]
    fn test_normalize_python_pattern_translates_ascii_scoped_flag_group() {
        assert_eq!(
            normalize_python_pattern(r"(?a:[_a-z][_a-z0-9]*)").unwrap(),
            r"(?-u:[_a-z][_a-z0-9]*)"
        );
    }

    #[test]
    fn test_normalize_python_pattern_escapes_literal_braces() {
        assert_eq!(
            normalize_python_pattern(r"{(?P<braced>\w+)}").unwrap(),
            r"\{(?P<braced>\w+)\}"
        );
    }

    #[test]
    fn test_normalize_python_pattern_translates_open_lower_quantifier_bound() {
        assert_eq!(normalize_python_pattern(r"a{,2}").unwrap(), r"a{0,2}");
    }

    #[test]
    fn test_normalize_python_pattern_escapes_literal_open_bracket_inside_class() {
        assert_eq!(normalize_python_pattern(r"([*?[])").unwrap(), r"([*?\[])");
    }

    #[test]
    fn test_normalize_python_pattern_translates_octal_escapes() {
        assert_eq!(
            normalize_python_pattern(r"[\041-\176]+:$").unwrap(),
            r"[\x{21}-\x{7e}]+:$"
        );
        assert_eq!(
            normalize_python_pattern(r"\0\01\018").unwrap(),
            r"\x{0}\x{1}\x{1}8"
        );
        assert_eq!(normalize_python_pattern(r"\111").unwrap(), r"\x{49}");
    }

    #[test]
    fn test_normalize_python_pattern_rejects_out_of_range_octal_escapes() {
        let outside_class = normalize_python_pattern(r"\567").unwrap_err();
        assert_eq!(
            outside_class.message,
            r"octal escape value \567 outside of range 0-0o377"
        );
        assert_eq!(outside_class.position, Some(0));

        let inside_class = normalize_python_pattern(r"[\567]").unwrap_err();
        assert_eq!(
            inside_class.message,
            r"octal escape value \567 outside of range 0-0o377"
        );
        assert_eq!(inside_class.position, Some(1));
    }

    #[test]
    fn test_prepare_pattern_for_backend_applies_ascii_and_verbose_flags() {
        let prepared = prepare_pattern_for_backend(
            r"\$(?:(?P<named>(?a:[_a-z][_a-z0-9]*)))",
            RegexFlags::new(RegexFlags::IGNORECASE | RegexFlags::VERBOSE),
        )
        .unwrap();
        assert!(prepared.starts_with("(?ix)"));
        assert!(prepared.contains(r"(?-u:[_a-z][_a-z0-9]*)"));
    }

    #[test]
    fn test_prepare_pattern_for_backend_normalizes_atomic_groups() {
        let prepared = prepare_pattern_for_backend(r"(?s:(?>.*?a).*)\Z", RegexFlags::default())
            .expect("atomic groups should normalize for backend regex engines");
        assert_eq!(prepared, r"(?s:(?:.*?a).*)\z");
    }

    #[test]
    fn test_standard_engine_compiles_string_template_identifier_pattern() {
        let pattern = r"
            \$(?:
              (?P<escaped>\$)  |
              (?P<named>(?a:[_a-z][_a-z0-9]*)) |
              {(?P<braced>(?a:[_a-z][_a-z0-9]*))} |
              (?P<invalid>)
            )
        ";
        let flags = RegexFlags::new(RegexFlags::IGNORECASE | RegexFlags::VERBOSE);
        let engine =
            StandardEngine::compile(pattern, flags).expect("template pattern should compile");
        assert!(engine.is_match("$name"));
    }

    #[test]
    fn test_standard_engine_accepts_python_end_of_string_anchor() {
        let engine =
            StandardEngine::compile(r"foo\Z", RegexFlags::default()).expect(r"\Z should compile");
        assert!(engine.is_match("foo"));
        assert!(!engine.is_match("foo\n"));
    }

    #[test]
    fn test_standard_engine_accepts_glob_magic_character_class_pattern() {
        let engine = StandardEngine::compile(r"([*?[])", RegexFlags::default())
            .expect("glob's magic-check pattern should compile");
        assert!(engine.is_match("["));
        assert!(engine.is_match("*"));
        assert!(engine.is_match("?"));
    }

    #[test]
    fn test_standard_engine_accepts_cpython_email_header_character_range() {
        let engine = StandardEngine::compile(r"[\041-\176]+:$", RegexFlags::default())
            .expect("email.header's printable ASCII range should compile");
        assert!(engine.is_match("Subject:"));
        assert!(!engine.is_match("Subject"));
    }
}
