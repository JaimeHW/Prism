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
