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
mod tests;
