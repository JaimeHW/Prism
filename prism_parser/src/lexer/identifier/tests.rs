use super::*;

#[test]
fn test_is_id_start_ascii() {
    assert!(is_id_start('a'));
    assert!(is_id_start('z'));
    assert!(is_id_start('A'));
    assert!(is_id_start('Z'));
    assert!(is_id_start('_'));
    assert!(!is_id_start('0'));
    assert!(!is_id_start('9'));
    assert!(!is_id_start(' '));
}

#[test]
fn test_is_id_start_unicode() {
    assert!(is_id_start('α')); // Greek alpha
    assert!(is_id_start('日')); // CJK
    assert!(is_id_start('π')); // Greek pi
    assert!(!is_id_start('①')); // Circled digit one
}

#[test]
fn test_is_id_continue_ascii() {
    assert!(is_id_continue('a'));
    assert!(is_id_continue('0'));
    assert!(is_id_continue('9'));
    assert!(is_id_continue('_'));
    assert!(!is_id_continue(' '));
    assert!(!is_id_continue('-'));
}

#[test]
fn test_is_id_continue_unicode() {
    assert!(is_id_continue('α'));
    assert!(is_id_continue('文')); // CJK
    // Combining marks
    assert!(is_id_continue('\u{0300}')); // Combining grave accent
}

#[test]
fn test_parse_identifier_simple() {
    let mut cursor = Cursor::new("hello world");
    let first = cursor.bump().unwrap();
    let result = parse_identifier(&mut cursor, first);
    assert_eq!(result, TokenKind::Ident("hello".to_string()));
}

#[test]
fn test_parse_identifier_underscore() {
    let mut cursor = Cursor::new("_private_var = 5");
    let first = cursor.bump().unwrap();
    let result = parse_identifier(&mut cursor, first);
    assert_eq!(result, TokenKind::Ident("_private_var".to_string()));
}

#[test]
fn test_parse_identifier_unicode() {
    let mut cursor = Cursor::new("αβγ = 3.14");
    let first = cursor.bump().unwrap();
    let result = parse_identifier(&mut cursor, first);
    assert_eq!(result, TokenKind::Ident("αβγ".to_string()));
}

#[test]
fn test_parse_keyword_if() {
    let mut cursor = Cursor::new("if x:");
    let first = cursor.bump().unwrap();
    let result = parse_identifier(&mut cursor, first);
    assert_eq!(result, TokenKind::Keyword(Keyword::If));
}

#[test]
fn test_parse_keyword_def() {
    let mut cursor = Cursor::new("def func():");
    let first = cursor.bump().unwrap();
    let result = parse_identifier(&mut cursor, first);
    assert_eq!(result, TokenKind::Keyword(Keyword::Def));
}

#[test]
fn test_parse_keyword_class() {
    let mut cursor = Cursor::new("class Foo:");
    let first = cursor.bump().unwrap();
    let result = parse_identifier(&mut cursor, first);
    assert_eq!(result, TokenKind::Keyword(Keyword::Class));
}

#[test]
fn test_parse_true_false_none() {
    let mut cursor1 = Cursor::new("True");
    let first1 = cursor1.bump().unwrap();
    assert_eq!(
        parse_identifier(&mut cursor1, first1),
        TokenKind::Keyword(Keyword::True)
    );

    let mut cursor2 = Cursor::new("False");
    let first2 = cursor2.bump().unwrap();
    assert_eq!(
        parse_identifier(&mut cursor2, first2),
        TokenKind::Keyword(Keyword::False)
    );

    let mut cursor3 = Cursor::new("None");
    let first3 = cursor3.bump().unwrap();
    assert_eq!(
        parse_identifier(&mut cursor3, first3),
        TokenKind::Keyword(Keyword::None)
    );
}

#[test]
fn test_soft_keyword() {
    assert_eq!(SoftKeyword::from_str("match"), Some(SoftKeyword::Match));
    assert_eq!(SoftKeyword::from_str("case"), Some(SoftKeyword::Case));
    assert_eq!(SoftKeyword::from_str("type"), Some(SoftKeyword::Type));
    assert_eq!(SoftKeyword::from_str("_"), Some(SoftKeyword::Underscore));
    assert_eq!(SoftKeyword::from_str("other"), None);
}

#[test]
fn test_reserved_ident() {
    assert_eq!(
        ReservedIdent::from_str("__init__"),
        Some(ReservedIdent::DunderInit)
    );
    assert_eq!(
        ReservedIdent::from_str("__name__"),
        Some(ReservedIdent::DunderName)
    );
    assert_eq!(ReservedIdent::from_str("regular"), None);
}

#[test]
fn test_identifier_not_keyword() {
    // "iffy" starts with "if" but is not a keyword
    let mut cursor = Cursor::new("iffy");
    let first = cursor.bump().unwrap();
    let result = parse_identifier(&mut cursor, first);
    assert_eq!(result, TokenKind::Ident("iffy".to_string()));
}

#[test]
fn test_identifier_with_numbers() {
    let mut cursor = Cursor::new("var123");
    let first = cursor.bump().unwrap();
    let result = parse_identifier(&mut cursor, first);
    assert_eq!(result, TokenKind::Ident("var123".to_string()));
}

#[test]
fn test_dunder_name() {
    let mut cursor = Cursor::new("__init__");
    let first = cursor.bump().unwrap();
    let result = parse_identifier(&mut cursor, first);
    assert_eq!(result, TokenKind::Ident("__init__".to_string()));
}
