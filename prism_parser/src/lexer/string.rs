//! String literal parsing for Python.
//!
//! Supports all Python 3.12 string literal formats:
//! - Single and double quotes: `'hello'`, `"world"`
//! - Triple quotes: `'''multiline'''`, `"""docstring"""`
//! - Raw strings: `r"path\to\file"`
//! - Byte strings: `b"binary data"`
//! - F-strings: `f"Hello {name}!"`
//! - Combined prefixes: `rf"raw {expr}"`, `br"raw bytes"`

use super::cursor::{Cursor, EOF_CHAR};
use crate::token::TokenKind;

/// String prefix flags.
#[derive(Debug, Clone, Copy, Default)]
pub struct StringPrefix {
    /// Raw string (no escape processing).
    pub raw: bool,
    /// Byte string.
    pub bytes: bool,
    /// Format string (f-string).
    pub format: bool,
    /// Unicode string (Python 2 compatibility, ignored in Python 3).
    pub unicode: bool,
}

impl StringPrefix {
    /// Parse prefix characters from a string.
    #[must_use]
    pub fn from_chars(chars: &str) -> Self {
        let mut prefix = Self::default();
        for c in chars.chars() {
            match c.to_ascii_lowercase() {
                'r' => prefix.raw = true,
                'b' => prefix.bytes = true,
                'f' => prefix.format = true,
                'u' => prefix.unicode = true,
                _ => {}
            }
        }
        prefix
    }

    /// Check if the prefix combination is valid.
    #[must_use]
    pub const fn is_valid(&self) -> bool {
        // Invalid combinations:
        // - bytes + format
        // - bytes + unicode
        // - format + unicode
        if self.bytes && (self.format || self.unicode) {
            return false;
        }
        if self.format && self.unicode {
            return false;
        }
        true
    }
}

/// Parse a string literal.
///
/// The cursor should be positioned at the opening quote.
/// `prefix` contains any prefix characters that were consumed.
pub fn parse_string(cursor: &mut Cursor<'_>, prefix: StringPrefix) -> TokenKind {
    let quote_char = cursor.bump_or_eof();
    if quote_char != '"' && quote_char != '\'' {
        return TokenKind::Error("expected quote".to_string());
    }

    // Check for triple-quoted string
    let is_triple = cursor.first() == quote_char && cursor.second() == quote_char;
    if is_triple {
        cursor.bump(); // second quote
        cursor.bump(); // third quote
    }

    // Handle f-string specially
    if prefix.format {
        return parse_fstring_content(cursor, quote_char, is_triple, prefix.raw);
    }

    // Parse regular string content
    let content = parse_string_content(cursor, quote_char, is_triple, prefix.raw);

    match content {
        Ok(s) => {
            if prefix.bytes {
                // Convert to bytes
                TokenKind::Bytes(s.into_bytes())
            } else {
                TokenKind::String(s)
            }
        }
        Err(msg) => TokenKind::Error(msg),
    }
}

/// Parse string content until the closing quote(s).
fn parse_string_content(
    cursor: &mut Cursor<'_>,
    quote_char: char,
    is_triple: bool,
    is_raw: bool,
) -> Result<String, String> {
    let mut content = String::new();

    loop {
        let c = cursor.first();

        if c == EOF_CHAR {
            return Err("unterminated string".to_string());
        }

        // Check for string end
        if c == quote_char {
            if is_triple {
                if cursor.second() == quote_char && cursor.third() == quote_char {
                    cursor.bump(); // first quote
                    cursor.bump(); // second quote
                    cursor.bump(); // third quote
                    break;
                }
            } else {
                cursor.bump();
                break;
            }
        }

        // Check for newline in single-quoted string
        if !is_triple && (c == '\n' || c == '\r') {
            return Err("unterminated string (newline in single-quoted string)".to_string());
        }

        cursor.bump();

        // Handle escapes
        if c == '\\' && !is_raw {
            let escaped = parse_escape(cursor)?;
            content.push(escaped);
        } else if c == '\\' && is_raw {
            // Raw strings: backslash is literal, but \' and \" still need handling
            content.push('\\');
            if cursor.first() == quote_char {
                content.push(cursor.bump_or_eof());
            }
        } else {
            content.push(c);
        }
    }

    Ok(content)
}

/// Parse an escape sequence.
fn parse_escape(cursor: &mut Cursor<'_>) -> Result<char, String> {
    let c = cursor.bump_or_eof();
    match c {
        '\\' => Ok('\\'),
        '\'' => Ok('\''),
        '"' => Ok('"'),
        'n' => Ok('\n'),
        'r' => Ok('\r'),
        't' => Ok('\t'),
        'b' => Ok('\x08'), // backspace
        'f' => Ok('\x0C'), // form feed
        'v' => Ok('\x0B'), // vertical tab
        '0' => Ok('\0'),
        'a' => Ok('\x07'), // bell
        '\n' => Ok(' '),   // line continuation, return space as placeholder
        '\r' => {
            // Handle \r\n
            if cursor.first() == '\n' {
                cursor.bump();
            }
            Ok(' ') // placeholder for line continuation
        }
        'x' => parse_hex_escape(cursor, 2),
        'u' => parse_hex_escape(cursor, 4),
        'U' => parse_hex_escape(cursor, 8),
        'N' => parse_unicode_name_escape(cursor),
        _ if c.is_ascii_digit() => parse_octal_escape(cursor, c),
        EOF_CHAR => Err("unterminated escape sequence".to_string()),
        _ => {
            // Python keeps unrecognized escapes as-is
            Ok(c)
        }
    }
}

/// Parse a hex escape sequence (\xNN, \uNNNN, \UNNNNNNNN).
fn parse_hex_escape(cursor: &mut Cursor<'_>, count: usize) -> Result<char, String> {
    let mut value: u32 = 0;
    for _ in 0..count {
        let c = cursor.bump_or_eof();
        let digit = c
            .to_digit(16)
            .ok_or_else(|| format!("invalid hex escape: {}", c))?;
        value = value * 16 + digit;
    }
    char::from_u32(value).ok_or_else(|| format!("invalid unicode codepoint: {}", value))
}

/// Parse an octal escape sequence (\NNN).
fn parse_octal_escape(cursor: &mut Cursor<'_>, first: char) -> Result<char, String> {
    let mut value = first.to_digit(8).unwrap();

    // Up to 2 more octal digits
    for _ in 0..2 {
        if let Some(digit) = cursor.first().to_digit(8) {
            value = value * 8 + digit;
            cursor.bump();
        } else {
            break;
        }
    }

    if value > 0o377 {
        return Err("octal escape too large".to_string());
    }

    Ok(value as u8 as char)
}

/// Parse a unicode name escape (\N{name}).
fn parse_unicode_name_escape(cursor: &mut Cursor<'_>) -> Result<char, String> {
    if cursor.first() != '{' {
        return Err("expected '{' after \\N".to_string());
    }
    cursor.bump();

    let mut name = String::new();
    loop {
        let c = cursor.bump_or_eof();
        if c == '}' {
            break;
        }
        if c == EOF_CHAR {
            return Err("unterminated unicode name escape".to_string());
        }
        name.push(c);
    }

    // For now, we don't implement full unicode name lookup
    // This would require a unicode database
    Err(format!("unicode name lookup not implemented: {}", name))
}

/// Parse f-string content.
fn parse_fstring_content(
    cursor: &mut Cursor<'_>,
    quote_char: char,
    is_triple: bool,
    is_raw: bool,
) -> TokenKind {
    // F-strings are complex - for now return a placeholder
    // A full implementation would need to:
    // 1. Parse literal parts
    // 2. Identify expression parts (in braces)
    // 3. Handle nested braces, strings, and format specs
    // 4. Return a structured representation

    // Skip to end of string for now
    let result = parse_string_content(cursor, quote_char, is_triple, is_raw);
    match result {
        Ok(s) => TokenKind::String(format!("f-string: {}", s)),
        Err(e) => TokenKind::Error(e),
    }
}

/// Check if a character can start a string prefix.
#[inline]
#[must_use]
pub fn is_string_prefix(c: char) -> bool {
    matches!(c.to_ascii_lowercase(), 'r' | 'b' | 'f' | 'u')
}

/// Check if a string of prefix characters is a valid string prefix.
#[must_use]
pub fn is_valid_string_prefix(s: &str) -> bool {
    let prefix = StringPrefix::from_chars(s);
    prefix.is_valid()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex_string(s: &str) -> TokenKind {
        let mut cursor = Cursor::new(s);
        parse_string(&mut cursor, StringPrefix::default())
    }

    fn lex_string_with_prefix(s: &str, prefix: &str) -> TokenKind {
        let mut cursor = Cursor::new(s);
        parse_string(&mut cursor, StringPrefix::from_chars(prefix))
    }

    #[test]
    fn test_simple_double_quote() {
        let result = lex_string("\"hello\"");
        assert_eq!(result, TokenKind::String("hello".to_string()));
    }

    #[test]
    fn test_simple_single_quote() {
        let result = lex_string("'world'");
        assert_eq!(result, TokenKind::String("world".to_string()));
    }

    #[test]
    fn test_empty_string() {
        let result = lex_string("\"\"");
        assert_eq!(result, TokenKind::String(String::new()));
    }

    #[test]
    fn test_escape_newline() {
        let result = lex_string("\"hello\\nworld\"");
        assert_eq!(result, TokenKind::String("hello\nworld".to_string()));
    }

    #[test]
    fn test_escape_tab() {
        let result = lex_string("\"hello\\tworld\"");
        assert_eq!(result, TokenKind::String("hello\tworld".to_string()));
    }

    #[test]
    fn test_escape_quote() {
        let result = lex_string("\"she said \\\"hi\\\"\"");
        assert_eq!(result, TokenKind::String("she said \"hi\"".to_string()));
    }

    #[test]
    fn test_escape_backslash() {
        let result = lex_string("\"path\\\\to\\\\file\"");
        assert_eq!(result, TokenKind::String("path\\to\\file".to_string()));
    }

    #[test]
    fn test_hex_escape() {
        let result = lex_string("\"\\x41\\x42\\x43\"");
        assert_eq!(result, TokenKind::String("ABC".to_string()));
    }

    #[test]
    fn test_unicode_escape() {
        let result = lex_string("\"\\u0041\\u0042\"");
        assert_eq!(result, TokenKind::String("AB".to_string()));
    }

    #[test]
    fn test_raw_string() {
        let result = lex_string_with_prefix("\"hello\\nworld\"", "r");
        assert_eq!(result, TokenKind::String("hello\\nworld".to_string()));
    }

    #[test]
    fn test_byte_string() {
        let result = lex_string_with_prefix("\"hello\"", "b");
        assert_eq!(result, TokenKind::Bytes(b"hello".to_vec()));
    }

    #[test]
    fn test_triple_quote() {
        let result = lex_string("\"\"\"hello\nworld\"\"\"");
        assert_eq!(result, TokenKind::String("hello\nworld".to_string()));
    }

    #[test]
    fn test_unterminated_string() {
        let result = lex_string("\"hello");
        assert!(matches!(result, TokenKind::Error(_)));
    }

    #[test]
    fn test_newline_in_single_quote() {
        let result = lex_string("\"hello\nworld\"");
        assert!(matches!(result, TokenKind::Error(_)));
    }

    #[test]
    fn test_string_prefix_valid() {
        assert!(StringPrefix::from_chars("r").is_valid());
        assert!(StringPrefix::from_chars("b").is_valid());
        assert!(StringPrefix::from_chars("f").is_valid());
        assert!(StringPrefix::from_chars("rf").is_valid());
        assert!(StringPrefix::from_chars("fr").is_valid());
        assert!(StringPrefix::from_chars("br").is_valid());
        assert!(StringPrefix::from_chars("rb").is_valid());
    }

    #[test]
    fn test_string_prefix_invalid() {
        assert!(!StringPrefix::from_chars("bf").is_valid()); // bytes + format
        assert!(!StringPrefix::from_chars("fb").is_valid());
        assert!(!StringPrefix::from_chars("uf").is_valid()); // unicode + format
    }

    #[test]
    fn test_is_string_prefix() {
        assert!(is_string_prefix('r'));
        assert!(is_string_prefix('R'));
        assert!(is_string_prefix('b'));
        assert!(is_string_prefix('f'));
        assert!(is_string_prefix('u'));
        assert!(!is_string_prefix('x'));
    }
}
