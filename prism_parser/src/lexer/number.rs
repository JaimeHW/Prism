//! Number literal parsing for Python.
//!
//! Supports all Python 3.12 numeric literal formats:
//! - Decimal integers: `42`, `1_000_000`
//! - Hex integers: `0xFF`, `0x_dead_beef`
//! - Octal integers: `0o755`
//! - Binary integers: `0b1010`
//! - Floating point: `3.14`, `1e10`, `2.5e-3`
//! - Complex/imaginary: `3j`, `2.5j`

use super::cursor::Cursor;
use crate::token::TokenKind;

/// Parse a number literal starting from the current position.
///
/// The cursor should be positioned at the first digit (or '.' for floats like `.5`).
pub fn parse_number(cursor: &mut Cursor<'_>, first_char: char) -> TokenKind {
    // Check for hex, octal, or binary prefix
    if first_char == '0' {
        match cursor.first() {
            'x' | 'X' => return parse_hex(cursor),
            'o' | 'O' => return parse_octal(cursor),
            'b' | 'B' => return parse_binary(cursor),
            _ => {}
        }
    }

    // Parse decimal number (may be int, float, or complex)
    parse_decimal(cursor, first_char)
}

/// Parse a hexadecimal integer.
fn parse_hex(cursor: &mut Cursor<'_>) -> TokenKind {
    cursor.bump(); // consume 'x' or 'X'

    let start = cursor.pos();
    eat_hex_digits(cursor);

    if cursor.pos() == start {
        return TokenKind::Error("invalid hex literal".to_string());
    }

    // Check for imaginary suffix
    if cursor.first() == 'j' || cursor.first() == 'J' {
        cursor.bump();
        return TokenKind::Error("complex literals cannot be hex".to_string());
    }

    let digits = cursor.slice_from(start);
    match parse_int_radix(digits, 16) {
        Ok(n) => TokenKind::Int(n),
        Err(_) => TokenKind::BigInt(format!("0x{}", digits)),
    }
}

/// Parse an octal integer.
fn parse_octal(cursor: &mut Cursor<'_>) -> TokenKind {
    cursor.bump(); // consume 'o' or 'O'

    let start = cursor.pos();
    eat_octal_digits(cursor);

    if cursor.pos() == start {
        return TokenKind::Error("invalid octal literal".to_string());
    }

    let digits = cursor.slice_from(start);
    match parse_int_radix(digits, 8) {
        Ok(n) => TokenKind::Int(n),
        Err(_) => TokenKind::BigInt(format!("0o{}", digits)),
    }
}

/// Parse a binary integer.
fn parse_binary(cursor: &mut Cursor<'_>) -> TokenKind {
    cursor.bump(); // consume 'b' or 'B'

    let start = cursor.pos();
    eat_binary_digits(cursor);

    if cursor.pos() == start {
        return TokenKind::Error("invalid binary literal".to_string());
    }

    let digits = cursor.slice_from(start);
    match parse_int_radix(digits, 2) {
        Ok(n) => TokenKind::Int(n),
        Err(_) => TokenKind::BigInt(format!("0b{}", digits)),
    }
}

/// Parse a decimal number (integer, float, or complex).
fn parse_decimal(cursor: &mut Cursor<'_>, first_char: char) -> TokenKind {
    let start = if first_char == '.' {
        cursor.pos()
    } else {
        cursor.pos().saturating_sub(1)
    };
    let mut has_dot = first_char == '.';
    let mut has_exp = false;

    // Eat integer part digits
    eat_decimal_digits(cursor);

    // Check for decimal point
    if cursor.first() == '.' && cursor.second() != '.' {
        // Not ellipsis (..)
        has_dot = true;
        cursor.bump();
        eat_decimal_digits(cursor);
    }

    // Check for exponent
    if cursor.first() == 'e' || cursor.first() == 'E' {
        has_exp = true;
        cursor.bump();

        // Optional sign
        if cursor.first() == '+' || cursor.first() == '-' {
            cursor.bump();
        }

        eat_decimal_digits(cursor);
    }

    // Check for imaginary suffix
    let is_complex = cursor.first() == 'j' || cursor.first() == 'J';
    if is_complex {
        cursor.bump();
    }

    // Get the number text (including the first character which was already consumed)
    let end = cursor.pos();
    let text = &cursor.source()[start..end];

    // Remove the 'j' suffix for parsing if complex
    let num_text: String = if is_complex {
        text[..text.len().saturating_sub(1)]
            .chars()
            .filter(|c| *c != '_')
            .collect()
    } else {
        text.chars().filter(|c| *c != '_').collect()
    };

    // Construct the result
    if is_complex {
        match num_text.parse::<f64>() {
            Ok(n) => TokenKind::Complex(n),
            Err(_) => TokenKind::Error("invalid complex literal".to_string()),
        }
    } else if has_dot || has_exp {
        // Float
        match num_text.parse::<f64>() {
            Ok(n) => TokenKind::Float(n),
            Err(_) => TokenKind::Error("invalid float literal".to_string()),
        }
    } else {
        // Integer
        match num_text.parse::<i64>() {
            Ok(n) => TokenKind::Int(n),
            Err(_) => TokenKind::BigInt(num_text),
        }
    }
}

/// Consume decimal digits and underscores.
fn eat_decimal_digits(cursor: &mut Cursor<'_>) {
    cursor.eat_while(|c| c.is_ascii_digit() || c == '_');
}

/// Consume hex digits and underscores.
fn eat_hex_digits(cursor: &mut Cursor<'_>) {
    cursor.eat_while(|c| c.is_ascii_hexdigit() || c == '_');
}

/// Consume octal digits and underscores.
fn eat_octal_digits(cursor: &mut Cursor<'_>) {
    cursor.eat_while(|c| matches!(c, '0'..='7' | '_'));
}

/// Consume binary digits and underscores.
fn eat_binary_digits(cursor: &mut Cursor<'_>) {
    cursor.eat_while(|c| c == '0' || c == '1' || c == '_');
}

/// Parse an integer from digits with underscores, given radix.
fn parse_int_radix(digits: &str, radix: u32) -> Result<i64, ()> {
    let clean: String = digits.chars().filter(|c| *c != '_').collect();
    if clean.is_empty() {
        return Err(());
    }
    i64::from_str_radix(&clean, radix).map_err(|_| ())
}

/// Check if a character can start a number.
#[inline]
#[must_use]
pub fn is_number_start(c: char, next: char) -> bool {
    c.is_ascii_digit() || (c == '.' && next.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex_number(s: &str) -> TokenKind {
        let mut cursor = Cursor::new(s);
        let first = cursor.bump().unwrap();
        parse_number(&mut cursor, first)
    }

    #[test]
    fn test_hex_lowercase() {
        let result = lex_number("0xff");
        assert_eq!(result, TokenKind::Int(255));
    }

    #[test]
    fn test_hex_uppercase() {
        let result = lex_number("0XFF");
        assert_eq!(result, TokenKind::Int(255));
    }

    #[test]
    fn test_hex_with_underscores() {
        let result = lex_number("0x_dead_beef");
        assert_eq!(result, TokenKind::Int(0xDEAD_BEEF));
    }

    #[test]
    fn test_octal() {
        let result = lex_number("0o755");
        assert_eq!(result, TokenKind::Int(0o755));
    }

    #[test]
    fn test_binary() {
        let result = lex_number("0b1010");
        assert_eq!(result, TokenKind::Int(0b1010));
    }

    #[test]
    fn test_binary_with_underscores() {
        let result = lex_number("0b1111_0000");
        assert_eq!(result, TokenKind::Int(0b1111_0000));
    }

    #[test]
    fn test_invalid_hex() {
        let result = lex_number("0x");
        assert!(matches!(result, TokenKind::Error(_)));
    }

    #[test]
    fn test_invalid_octal() {
        let result = lex_number("0o");
        assert!(matches!(result, TokenKind::Error(_)));
    }

    #[test]
    fn test_invalid_binary() {
        let result = lex_number("0b");
        assert!(matches!(result, TokenKind::Error(_)));
    }

    #[test]
    fn test_is_number_start() {
        assert!(is_number_start('0', ' '));
        assert!(is_number_start('9', ' '));
        assert!(is_number_start('.', '5'));
        assert!(!is_number_start('.', '.'));
        assert!(!is_number_start('a', '5'));
    }
}
