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
use prism_core::python_unicode::encode_python_code_point;

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

    if prefix.bytes {
        match parse_bytes_content(cursor, quote_char, is_triple, prefix.raw) {
            Ok(bytes) => TokenKind::Bytes(bytes),
            Err(msg) => TokenKind::Error(msg),
        }
    } else {
        match parse_string_content(cursor, quote_char, is_triple, prefix.raw) {
            Ok(s) => TokenKind::String(s),
            Err(msg) => TokenKind::Error(msg),
        }
    }
}

fn parse_bytes_content(
    cursor: &mut Cursor<'_>,
    quote_char: char,
    is_triple: bool,
    is_raw: bool,
) -> Result<Vec<u8>, String> {
    let mut content = Vec::new();
    let mut raw_backslash_run = 0usize;

    loop {
        let c = cursor.first();

        if c == EOF_CHAR {
            return Err("unterminated string".to_string());
        }

        let raw_quote_escaped = is_raw && raw_backslash_run % 2 == 1;
        if c == quote_char && !raw_quote_escaped {
            if is_triple {
                if cursor.second() == quote_char && cursor.third() == quote_char {
                    cursor.bump();
                    cursor.bump();
                    cursor.bump();
                    break;
                }
            } else {
                cursor.bump();
                break;
            }
        }

        if !is_triple && (c == '\n' || c == '\r') {
            return Err("unterminated string (newline in single-quoted string)".to_string());
        }

        cursor.bump();

        if c == '\\' && !is_raw {
            parse_byte_escape(cursor, &mut content)?;
            raw_backslash_run = 0;
        } else if c == '\\' && is_raw {
            content.push(b'\\');
            raw_backslash_run += 1;
        } else {
            push_ascii_byte(&mut content, c)?;
            raw_backslash_run = 0;
        }
    }

    Ok(content)
}

#[inline]
fn push_ascii_byte(content: &mut Vec<u8>, c: char) -> Result<(), String> {
    if c.is_ascii() {
        content.push(c as u8);
        Ok(())
    } else {
        Err("bytes can only contain ASCII literal characters".to_string())
    }
}

fn parse_byte_escape(cursor: &mut Cursor<'_>, content: &mut Vec<u8>) -> Result<(), String> {
    let c = cursor.bump_or_eof();
    match c {
        '\\' => content.push(b'\\'),
        '\'' => content.push(b'\''),
        '"' => content.push(b'"'),
        'n' => content.push(b'\n'),
        'r' => content.push(b'\r'),
        't' => content.push(b'\t'),
        'b' => content.push(b'\x08'),
        'f' => content.push(b'\x0c'),
        'v' => content.push(b'\x0b'),
        'a' => content.push(b'\x07'),
        '\n' => {}
        '\r' => {
            if cursor.first() == '\n' {
                cursor.bump();
            }
        }
        'x' => content.push(parse_byte_hex_escape(cursor, 2)?),
        'u' | 'U' | 'N' => {
            content.push(b'\\');
            content.push(c as u8);
        }
        _ if c.is_ascii_digit() => content.push(parse_byte_octal_escape(cursor, c)?),
        EOF_CHAR => return Err("unterminated escape sequence".to_string()),
        _ => {
            content.push(b'\\');
            push_ascii_byte(content, c)?;
        }
    }
    Ok(())
}

fn parse_byte_hex_escape(cursor: &mut Cursor<'_>, count: usize) -> Result<u8, String> {
    let mut value: u32 = 0;
    for _ in 0..count {
        let c = cursor.bump_or_eof();
        let digit = c
            .to_digit(16)
            .ok_or_else(|| format!("invalid hex escape: {}", c))?;
        value = value * 16 + digit;
    }
    u8::try_from(value).map_err(|_| format!("invalid byte value: {}", value))
}

fn parse_byte_octal_escape(cursor: &mut Cursor<'_>, first: char) -> Result<u8, String> {
    let mut value = first.to_digit(8).unwrap();

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

    Ok(value as u8)
}

/// Parse string content until the closing quote(s).
fn parse_string_content(
    cursor: &mut Cursor<'_>,
    quote_char: char,
    is_triple: bool,
    is_raw: bool,
) -> Result<String, String> {
    let mut content = String::new();
    let mut raw_backslash_run = 0usize;

    loop {
        let c = cursor.first();

        if c == EOF_CHAR {
            return Err("unterminated string".to_string());
        }

        // Check for string end
        let raw_quote_escaped = is_raw && raw_backslash_run % 2 == 1;
        if c == quote_char && !raw_quote_escaped {
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
            if let Some(escaped) = parse_escape(cursor)? {
                content.push(escaped);
            }
            raw_backslash_run = 0;
        } else if c == '\\' && is_raw {
            content.push('\\');
            raw_backslash_run += 1;
        } else {
            content.push(c);
            raw_backslash_run = 0;
        }
    }

    Ok(content)
}

/// Parse an escape sequence.
fn parse_escape(cursor: &mut Cursor<'_>) -> Result<Option<char>, String> {
    let c = cursor.bump_or_eof();
    match c {
        '\\' => Ok(Some('\\')),
        '\'' => Ok(Some('\'')),
        '"' => Ok(Some('"')),
        'n' => Ok(Some('\n')),
        'r' => Ok(Some('\r')),
        't' => Ok(Some('\t')),
        'b' => Ok(Some('\x08')), // backspace
        'f' => Ok(Some('\x0C')), // form feed
        'v' => Ok(Some('\x0B')), // vertical tab
        '0' => Ok(Some('\0')),
        'a' => Ok(Some('\x07')), // bell
        '\n' => Ok(None),
        '\r' => {
            // Handle \r\n
            if cursor.first() == '\n' {
                cursor.bump();
            }
            Ok(None)
        }
        'x' => parse_hex_escape(cursor, 2).map(Some),
        'u' => parse_hex_escape(cursor, 4).map(Some),
        'U' => parse_hex_escape(cursor, 8).map(Some),
        'N' => parse_unicode_name_escape(cursor).map(Some),
        _ if c.is_ascii_digit() => parse_octal_escape(cursor, c).map(Some),
        EOF_CHAR => Err("unterminated escape sequence".to_string()),
        _ => {
            // Python keeps unrecognized escapes as-is
            Ok(Some(c))
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
    encode_python_code_point(value).ok_or_else(|| format!("invalid unicode codepoint: {}", value))
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
    let result = scan_fstring_content(cursor, quote_char, is_triple, is_raw);
    match result {
        Ok(s) => TokenKind::FString(s),
        Err(e) => TokenKind::Error(e),
    }
}

fn scan_fstring_content(
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

        if c == quote_char {
            if is_triple {
                if cursor.second() == quote_char && cursor.third() == quote_char {
                    cursor.bump();
                    cursor.bump();
                    cursor.bump();
                    break;
                }
            } else {
                cursor.bump();
                break;
            }
        }

        if !is_triple && (c == '\n' || c == '\r') {
            return Err("unterminated string (newline in single-quoted string)".to_string());
        }

        if c == '{' {
            cursor.bump();
            if cursor.first() == '{' {
                cursor.bump();
                content.push('{');
                content.push('{');
                continue;
            }

            content.push('{');
            scan_fstring_expression(cursor, &mut content)?;
            continue;
        }

        if c == '}' {
            cursor.bump();
            if cursor.first() == '}' {
                cursor.bump();
                content.push('}');
                content.push('}');
                continue;
            }
            return Err("single '}' is not allowed in f-string".to_string());
        }

        cursor.bump();

        if c == '\\' && !is_raw {
            if let Some(escaped) = parse_escape(cursor)? {
                content.push(escaped);
            }
        } else if c == '\\' && is_raw {
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

fn scan_fstring_expression(cursor: &mut Cursor<'_>, content: &mut String) -> Result<(), String> {
    let mut paren_depth = 0usize;
    let mut bracket_depth = 0usize;
    let mut brace_depth = 0usize;

    loop {
        let c = cursor.first();

        if c == EOF_CHAR {
            return Err("unterminated f-string expression".to_string());
        }

        if c == '\'' || c == '"' {
            scan_nested_python_string_literal(cursor, content)?;
            continue;
        }

        cursor.bump();
        content.push(c);

        match c {
            '(' => paren_depth += 1,
            ')' => paren_depth = paren_depth.saturating_sub(1),
            '[' => bracket_depth += 1,
            ']' => bracket_depth = bracket_depth.saturating_sub(1),
            '{' => brace_depth += 1,
            '}' => {
                if paren_depth == 0 && bracket_depth == 0 && brace_depth == 0 {
                    break;
                }
                brace_depth = brace_depth.saturating_sub(1);
            }
            _ => {}
        }
    }

    Ok(())
}

fn scan_nested_python_string_literal(
    cursor: &mut Cursor<'_>,
    content: &mut String,
) -> Result<(), String> {
    let quote_char = cursor.bump_or_eof();
    if quote_char != '\'' && quote_char != '"' {
        return Err("expected quote".to_string());
    }
    content.push(quote_char);

    let is_triple = cursor.first() == quote_char && cursor.second() == quote_char;
    if is_triple {
        content.push(cursor.bump_or_eof());
        content.push(cursor.bump_or_eof());
    }

    loop {
        let c = cursor.first();
        if c == EOF_CHAR {
            return Err("unterminated string in f-string expression".to_string());
        }

        cursor.bump();
        content.push(c);

        if c == '\\' {
            let next = cursor.bump_or_eof();
            if next == EOF_CHAR {
                return Err("unterminated escape sequence in f-string expression".to_string());
            }
            content.push(next);
            continue;
        }

        if c == quote_char {
            if is_triple {
                if cursor.first() == quote_char && cursor.second() == quote_char {
                    content.push(cursor.bump_or_eof());
                    content.push(cursor.bump_or_eof());
                    break;
                }
            } else {
                break;
            }
        }
    }

    Ok(())
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
    use prism_core::python_unicode::encode_python_code_point;

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
    fn test_unicode_escape_preserves_python_surrogates_via_internal_carrier() {
        let result = lex_string("\"\\uDC80\"");
        let surrogate = encode_python_code_point(0xDC80)
            .expect("Python surrogate escape should map into carrier range");
        assert_eq!(result, TokenKind::String(surrogate.to_string()));
    }

    #[test]
    fn test_raw_string() {
        let result = lex_string_with_prefix("\"hello\\nworld\"", "r");
        assert_eq!(result, TokenKind::String("hello\\nworld".to_string()));
    }

    #[test]
    fn test_raw_string_allows_even_backslashes_before_closing_quote() {
        let result = lex_string_with_prefix("'\\\\'", "r");
        assert_eq!(result, TokenKind::String("\\\\".to_string()));
    }

    #[test]
    fn test_raw_string_rejects_odd_backslash_before_closing_quote() {
        let result = lex_string_with_prefix("'\\'", "r");
        assert!(matches!(result, TokenKind::Error(_)));
    }

    #[test]
    fn test_raw_bytes_allows_even_backslashes_before_closing_quote() {
        let result = lex_string_with_prefix("'\\\\'", "br");
        assert_eq!(result, TokenKind::Bytes(br"\\".to_vec()));
    }

    #[test]
    fn test_fstring_expression_can_contain_matching_inner_string_quotes() {
        let result = lex_string_with_prefix(
            "'Ignored error getting __notes__: {_safe_string(e, '__notes__', repr)}'",
            "f",
        );
        assert_eq!(
            result,
            TokenKind::FString(
                "Ignored error getting __notes__: {_safe_string(e, '__notes__', repr)}".to_string()
            )
        );
    }

    #[test]
    fn test_fstring_expression_can_contain_nested_braces() {
        let result = lex_string_with_prefix("\"{ {'key': value} }\"", "f");
        assert_eq!(result, TokenKind::FString("{ {'key': value} }".to_string()));
    }

    #[test]
    fn test_fstring_preserves_escaped_braces_for_parser_lowering() {
        let result = lex_string_with_prefix("\"{{name}} = {value}\"", "f");
        assert_eq!(result, TokenKind::FString("{{name}} = {value}".to_string()));
    }

    #[test]
    fn test_byte_string() {
        let result = lex_string_with_prefix("\"hello\"", "b");
        assert_eq!(result, TokenKind::Bytes(b"hello".to_vec()));
    }

    #[test]
    fn test_byte_string_hex_escape_preserves_single_byte_values() {
        let result = lex_string_with_prefix("\"\\x00A\\xff\"", "b");
        assert_eq!(result, TokenKind::Bytes(vec![0x00, b'A', 0xff]));
    }

    #[test]
    fn test_byte_string_unicode_escape_is_kept_literal() {
        let result = lex_string_with_prefix("\"\\u0041\"", "b");
        assert_eq!(result, TokenKind::Bytes(br"\u0041".to_vec()));
    }

    #[test]
    fn test_byte_string_rejects_non_ascii_literal_characters() {
        let result = lex_string_with_prefix("\"é\"", "b");
        match result {
            TokenKind::Error(message) => {
                assert!(message.contains("ASCII literal characters"));
            }
            other => panic!("expected bytes literal error, got {:?}", other),
        }
    }

    #[test]
    fn test_triple_quote() {
        let result = lex_string("\"\"\"hello\nworld\"\"\"");
        assert_eq!(result, TokenKind::String("hello\nworld".to_string()));
    }

    #[test]
    fn test_backslash_newline_continuation_is_elided() {
        let result = lex_string("\"hello\\\nworld\"");
        assert_eq!(result, TokenKind::String("helloworld".to_string()));
    }

    #[test]
    fn test_triple_quote_initial_backslash_newline_is_elided() {
        let result = lex_string("\"\"\"\\\nNAME=Fedora\"\"\"");
        assert_eq!(result, TokenKind::String("NAME=Fedora".to_string()));
    }

    #[test]
    fn test_fstring_backslash_newline_continuation_is_elided() {
        let result = lex_string_with_prefix("\"hello\\\n{name}\"", "f");
        assert_eq!(result, TokenKind::FString("hello{name}".to_string()));
    }

    #[test]
    fn test_byte_string_backslash_newline_continuation_is_elided() {
        let result = lex_string_with_prefix("\"hello\\\nworld\"", "b");
        assert_eq!(result, TokenKind::Bytes(b"helloworld".to_vec()));
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
