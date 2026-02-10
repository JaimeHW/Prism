//! Public API functions matching Python's `re` module.
//!
//! These functions provide the user-facing API for regex operations.

use super::cache::global_cache;
use super::engine::RegexError;
use super::flags::RegexFlags;
use super::match_obj::Match;
use super::pattern::CompiledPattern;

// =============================================================================
// Result Type
// =============================================================================

/// Result type for regex operations.
pub type ReResult<T> = Result<T, RegexError>;

// =============================================================================
// Compile
// =============================================================================

/// Compile a regular expression pattern.
///
/// Equivalent to Python's `re.compile(pattern, flags=0)`.
///
/// # Example
/// ```ignore
/// let pattern = compile(r"\d+", 0)?;
/// let m = pattern.search("abc123")?;
/// ```
pub fn compile(pattern: &str, flags: u32) -> ReResult<CompiledPattern> {
    let flags = RegexFlags::new(flags);
    global_cache().get_or_compile(pattern, flags)
}

/// Compile with default flags.
pub fn compile_default(pattern: &str) -> ReResult<CompiledPattern> {
    compile(pattern, 0)
}

// =============================================================================
// Match
// =============================================================================

/// Try to apply the pattern at the start of the string.
///
/// Equivalent to Python's `re.match(pattern, string, flags=0)`.
///
/// Returns `None` if the pattern doesn't match at position 0.
pub fn match_(pattern: &str, string: &str, flags: u32) -> ReResult<Option<Match>> {
    let compiled = compile(pattern, flags)?;
    Ok(compiled.match_(string))
}

/// Match with default flags.
pub fn match_default(pattern: &str, string: &str) -> ReResult<Option<Match>> {
    match_(pattern, string, 0)
}

// =============================================================================
// Search
// =============================================================================

/// Scan through string looking for a match.
///
/// Equivalent to Python's `re.search(pattern, string, flags=0)`.
///
/// Returns the first match found, or `None` if no match.
pub fn search(pattern: &str, string: &str, flags: u32) -> ReResult<Option<Match>> {
    let compiled = compile(pattern, flags)?;
    Ok(compiled.search(string))
}

/// Search with default flags.
pub fn search_default(pattern: &str, string: &str) -> ReResult<Option<Match>> {
    search(pattern, string, 0)
}

// =============================================================================
// Fullmatch
// =============================================================================

/// Try to apply the pattern to all of the string.
///
/// Equivalent to Python's `re.fullmatch(pattern, string, flags=0)`.
///
/// Returns `None` if the pattern doesn't match the entire string.
pub fn fullmatch(pattern: &str, string: &str, flags: u32) -> ReResult<Option<Match>> {
    let compiled = compile(pattern, flags)?;
    Ok(compiled.fullmatch(string))
}

/// Fullmatch with default flags.
pub fn fullmatch_default(pattern: &str, string: &str) -> ReResult<Option<Match>> {
    fullmatch(pattern, string, 0)
}

// =============================================================================
// Find All
// =============================================================================

/// Find all non-overlapping matches in string.
///
/// Equivalent to Python's `re.findall(pattern, string, flags=0)`.
///
/// Returns all matches as Match objects.
pub fn findall(pattern: &str, string: &str, flags: u32) -> ReResult<Vec<Match>> {
    let compiled = compile(pattern, flags)?;
    Ok(compiled.findall(string))
}

/// Findall with default flags.
pub fn findall_default(pattern: &str, string: &str) -> ReResult<Vec<Match>> {
    findall(pattern, string, 0)
}

/// Find all non-overlapping matches as strings.
///
/// Like Python's `re.findall()`, returns:
/// - List of full matches if no groups
/// - List of group contents if one group
/// - List of tuples if multiple groups
pub fn findall_strings(pattern: &str, string: &str, flags: u32) -> ReResult<Vec<Vec<String>>> {
    let compiled = compile(pattern, flags)?;
    Ok(compiled.findall_strings(string))
}

// =============================================================================
// Find Iterator
// =============================================================================

/// Return an iterator yielding match objects for all matches.
///
/// Equivalent to Python's `re.finditer(pattern, string, flags=0)`.
pub fn finditer(pattern: &str, string: &str, flags: u32) -> ReResult<Vec<Match>> {
    let compiled = compile(pattern, flags)?;
    Ok(compiled.finditer(string).collect())
}

/// Finditer with default flags.
pub fn finditer_default(pattern: &str, string: &str) -> ReResult<Vec<Match>> {
    finditer(pattern, string, 0)
}

// =============================================================================
// Substitution
// =============================================================================

/// Replace occurrences of pattern in string.
///
/// Equivalent to Python's `re.sub(pattern, repl, string, count=0, flags=0)`.
///
/// If count is 0, all occurrences are replaced.
pub fn sub(pattern: &str, repl: &str, string: &str, count: usize, flags: u32) -> ReResult<String> {
    let compiled = compile(pattern, flags)?;
    Ok(compiled.sub_n(repl, string, count))
}

/// Sub with default count and flags.
pub fn sub_default(pattern: &str, repl: &str, string: &str) -> ReResult<String> {
    sub(pattern, repl, string, 0, 0)
}

/// Replace with replacement count returned.
///
/// Equivalent to Python's `re.subn(pattern, repl, string, count=0, flags=0)`.
///
/// Returns (new_string, number_of_substitutions).
pub fn subn(
    pattern: &str,
    repl: &str,
    string: &str,
    count: usize,
    flags: u32,
) -> ReResult<(String, usize)> {
    let compiled = compile(pattern, flags)?;
    Ok(compiled.subn(repl, string, count))
}

/// Subn with default count and flags.
pub fn subn_default(pattern: &str, repl: &str, string: &str) -> ReResult<(String, usize)> {
    subn(pattern, repl, string, 0, 0)
}

// =============================================================================
// Split
// =============================================================================

/// Split string by pattern occurrences.
///
/// Equivalent to Python's `re.split(pattern, string, maxsplit=0, flags=0)`.
///
/// If maxsplit is 0, all occurrences are used.
pub fn split(pattern: &str, string: &str, maxsplit: usize, flags: u32) -> ReResult<Vec<String>> {
    let compiled = compile(pattern, flags)?;
    Ok(compiled.split_n(string, maxsplit))
}

/// Split with default maxsplit and flags.
pub fn split_default(pattern: &str, string: &str) -> ReResult<Vec<String>> {
    split(pattern, string, 0, 0)
}

// =============================================================================
// Escape
// =============================================================================

/// Escape special characters in pattern.
///
/// Equivalent to Python's `re.escape(pattern)`.
///
/// Returns a string with all regex metacharacters escaped.
pub fn escape(pattern: &str) -> String {
    let mut result = String::with_capacity(pattern.len() * 2);
    for c in pattern.chars() {
        match c {
            '\\' | '.' | '+' | '*' | '?' | '[' | ']' | '{' | '}' | '(' | ')' | '^' | '$' | '|'
            | '#' | '&' | '-' | '~' => {
                result.push('\\');
                result.push(c);
            }
            _ => result.push(c),
        }
    }
    result
}

// =============================================================================
// Purge
// =============================================================================

/// Clear the regex cache.
///
/// Equivalent to Python's `re.purge()`.
pub fn purge() {
    super::cache::purge_global_cache();
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile() {
        let pattern = compile(r"\d+", 0).unwrap();
        assert!(pattern.is_match("123"));
    }

    #[test]
    fn test_match() {
        let m = match_default(r"\d+", "123abc").unwrap();
        assert!(m.is_some());
        assert_eq!(m.unwrap().as_str(), "123");

        let m = match_default(r"\d+", "abc123").unwrap();
        assert!(m.is_none());
    }

    #[test]
    fn test_search() {
        let m = search_default(r"\d+", "abc123def").unwrap();
        assert!(m.is_some());
        assert_eq!(m.unwrap().as_str(), "123");
    }

    #[test]
    fn test_fullmatch() {
        assert!(fullmatch_default(r"\d+", "123").unwrap().is_some());
        assert!(fullmatch_default(r"\d+", "123abc").unwrap().is_none());
    }

    #[test]
    fn test_findall() {
        let matches = findall_default(r"\d+", "a1b22c333").unwrap();
        assert_eq!(matches.len(), 3);
    }

    #[test]
    fn test_findall_strings() {
        let strings = findall_strings(r"\d+", "a1b22c333", 0).unwrap();
        assert_eq!(strings.len(), 3);
        assert_eq!(strings[0], vec!["1"]);
        assert_eq!(strings[1], vec!["22"]);
        assert_eq!(strings[2], vec!["333"]);
    }

    #[test]
    fn test_sub() {
        assert_eq!(sub_default(r"\d+", "X", "a1b2c3").unwrap(), "aXbXcX");
    }

    #[test]
    fn test_sub_count() {
        assert_eq!(sub(r"\d+", "X", "a1b2c3", 2, 0).unwrap(), "aXbXc3");
    }

    #[test]
    fn test_subn() {
        let (result, count) = subn_default(r"\d+", "X", "a1b2c3").unwrap();
        assert_eq!(result, "aXbXcX");
        assert_eq!(count, 3);
    }

    #[test]
    fn test_split() {
        let parts = split_default(r",\s*", "a, b,  c").unwrap();
        assert_eq!(parts, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_split_maxsplit() {
        let parts = split(r",", "a,b,c,d", 2, 0).unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[2], "c,d");
    }

    #[test]
    fn test_escape() {
        assert_eq!(escape(r"a.b*c?"), r"a\.b\*c\?");
        assert_eq!(escape(r"hello"), r"hello");
        assert_eq!(escape(r"[test]"), r"\[test\]");
    }

    #[test]
    fn test_flags() {
        // Case insensitive
        let m = match_(r"hello", "HELLO", RegexFlags::IGNORECASE).unwrap();
        assert!(m.is_some());

        // Without flag
        let m = match_(r"hello", "HELLO", 0).unwrap();
        assert!(m.is_none());
    }

    #[test]
    fn test_multiline() {
        let matches = findall(r"^\d+", "1\n2\n3", RegexFlags::MULTILINE).unwrap();
        assert_eq!(matches.len(), 3);
    }

    #[test]
    fn test_dotall() {
        let m = search(r"a.b", "a\nb", RegexFlags::DOTALL).unwrap();
        assert!(m.is_some());

        let m = search(r"a.b", "a\nb", 0).unwrap();
        assert!(m.is_none());
    }

    #[test]
    fn test_purge() {
        compile_default(r"test_pattern").unwrap();
        purge();
        // After purge, cache should be empty (can't directly verify, but no error)
    }

    #[test]
    fn test_error_invalid_pattern() {
        let result = compile(r"[invalid", 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_groups_in_findall() {
        let strings = findall_strings(r"(\d+)-(\d+)", "1-2 3-4 5-6", 0).unwrap();
        assert_eq!(strings.len(), 3);
        assert_eq!(strings[0], vec!["1", "2"]);
        assert_eq!(strings[1], vec!["3", "4"]);
        assert_eq!(strings[2], vec!["5", "6"]);
    }

    #[test]
    fn test_backreference() {
        // Should use fancy engine
        let m = search_default(r"(.)\1", "hello").unwrap();
        assert!(m.is_some());
        assert_eq!(m.unwrap().as_str(), "ll");
    }
}
