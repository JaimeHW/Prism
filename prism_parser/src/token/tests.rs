use super::*;

#[test]
fn test_token_creation() {
    let token = Token::new(TokenKind::Plus, Span::new(0, 1));
    assert_eq!(token.kind, TokenKind::Plus);
    assert_eq!(token.span.start, 0);
    assert_eq!(token.span.end, 1);
}

#[test]
fn test_token_is_eof() {
    let eof = Token::new(TokenKind::Eof, Span::new(100, 100));
    let plus = Token::new(TokenKind::Plus, Span::new(0, 1));

    assert!(eof.is_eof());
    assert!(!plus.is_eof());
}

#[test]
fn test_token_is_newline() {
    let newline = Token::new(TokenKind::Newline, Span::new(10, 11));
    let plus = Token::new(TokenKind::Plus, Span::new(0, 1));

    assert!(newline.is_newline());
    assert!(!plus.is_newline());
}

#[test]
fn test_keyword_from_str() {
    assert_eq!(Keyword::from_str("if"), Some(Keyword::If));
    assert_eq!(Keyword::from_str("while"), Some(Keyword::While));
    assert_eq!(Keyword::from_str("True"), Some(Keyword::True));
    assert_eq!(Keyword::from_str("False"), Some(Keyword::False));
    assert_eq!(Keyword::from_str("None"), Some(Keyword::None));
    assert_eq!(Keyword::from_str("match"), None);
    assert_eq!(Keyword::from_str("case"), None);
    assert_eq!(Keyword::from_str("type"), None);
    assert_eq!(Keyword::from_str("not_a_keyword"), None);
}

#[test]
fn test_keyword_as_str() {
    assert_eq!(Keyword::If.as_str(), "if");
    assert_eq!(Keyword::While.as_str(), "while");
    assert_eq!(Keyword::True.as_str(), "True");
}

#[test]
fn test_keyword_display() {
    assert_eq!(format!("{}", Keyword::Def), "def");
    assert_eq!(format!("{}", Keyword::Class), "class");
}

#[test]
fn test_all_keywords_roundtrip() {
    let keywords = [
        Keyword::False,
        Keyword::None,
        Keyword::True,
        Keyword::And,
        Keyword::As,
        Keyword::Assert,
        Keyword::Async,
        Keyword::Await,
        Keyword::Break,
        Keyword::Class,
        Keyword::Continue,
        Keyword::Def,
        Keyword::Del,
        Keyword::Elif,
        Keyword::Else,
        Keyword::Except,
        Keyword::Finally,
        Keyword::For,
        Keyword::From,
        Keyword::Global,
        Keyword::If,
        Keyword::Import,
        Keyword::In,
        Keyword::Is,
        Keyword::Lambda,
        Keyword::Nonlocal,
        Keyword::Not,
        Keyword::Or,
        Keyword::Pass,
        Keyword::Raise,
        Keyword::Return,
        Keyword::Try,
        Keyword::While,
        Keyword::With,
        Keyword::Yield,
    ];

    for kw in keywords {
        let s = kw.as_str();
        let parsed = Keyword::from_str(s);
        assert_eq!(parsed, Some(kw), "Roundtrip failed for {:?}", kw);
    }
}

#[test]
fn test_token_kind_is_comparison() {
    assert!(TokenKind::Less.is_comparison());
    assert!(TokenKind::Greater.is_comparison());
    assert!(TokenKind::LessEqual.is_comparison());
    assert!(TokenKind::GreaterEqual.is_comparison());
    assert!(TokenKind::EqualEqual.is_comparison());
    assert!(TokenKind::NotEqual.is_comparison());

    assert!(!TokenKind::Plus.is_comparison());
    assert!(!TokenKind::Equal.is_comparison());
}

#[test]
fn test_token_kind_is_augmented_assign() {
    assert!(TokenKind::PlusEqual.is_augmented_assign());
    assert!(TokenKind::MinusEqual.is_augmented_assign());
    assert!(TokenKind::StarEqual.is_augmented_assign());
    assert!(TokenKind::DoubleStarEqual.is_augmented_assign());

    assert!(!TokenKind::Equal.is_augmented_assign());
    assert!(!TokenKind::Plus.is_augmented_assign());
}

#[test]
fn test_token_kind_display() {
    assert_eq!(format!("{}", TokenKind::Plus), "+");
    assert_eq!(format!("{}", TokenKind::DoubleStar), "**");
    assert_eq!(format!("{}", TokenKind::ColonEqual), ":=");
    assert_eq!(format!("{}", TokenKind::Ellipsis), "...");
    assert_eq!(format!("{}", TokenKind::Arrow), "->");
}

#[test]
fn test_token_kind_literals_display() {
    assert_eq!(format!("{}", TokenKind::Int(42)), "42");
    assert_eq!(format!("{}", TokenKind::Float(3.125)), "3.125");
    assert_eq!(
        format!("{}", TokenKind::String("hello".to_string())),
        "\"hello\""
    );
}

#[test]
fn test_token_equality() {
    let t1 = Token::new(TokenKind::Plus, Span::new(0, 1));
    let t2 = Token::new(TokenKind::Plus, Span::new(0, 1));
    let t3 = Token::new(TokenKind::Plus, Span::new(1, 2));
    let t4 = Token::new(TokenKind::Minus, Span::new(0, 1));

    assert_eq!(t1, t2);
    assert_ne!(t1, t3); // Different span
    assert_ne!(t1, t4); // Different kind
}
