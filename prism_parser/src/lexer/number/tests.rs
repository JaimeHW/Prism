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
