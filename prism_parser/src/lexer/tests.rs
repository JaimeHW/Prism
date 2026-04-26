use super::*;

fn lex(s: &str) -> Vec<TokenKind> {
    Lexer::tokenize(s).into_iter().map(|t| t.kind).collect()
}

fn lex_no_eof(s: &str) -> Vec<TokenKind> {
    lex(s)
        .into_iter()
        .filter(|k| !matches!(k, TokenKind::Eof))
        .collect()
}

#[test]
fn test_empty() {
    let tokens = lex("");
    assert_eq!(tokens, vec![TokenKind::Eof]);
}

#[test]
fn test_single_identifier() {
    let tokens = lex_no_eof("hello");
    assert_eq!(tokens, vec![TokenKind::Ident("hello".to_string())]);
}

#[test]
fn test_multiple_identifiers() {
    let tokens = lex_no_eof("foo bar baz");
    assert_eq!(
        tokens,
        vec![
            TokenKind::Ident("foo".to_string()),
            TokenKind::Ident("bar".to_string()),
            TokenKind::Ident("baz".to_string()),
        ]
    );
}

#[test]
fn test_operators() {
    let tokens = lex_no_eof("+ - * / // ** % @");
    assert_eq!(
        tokens,
        vec![
            TokenKind::Plus,
            TokenKind::Minus,
            TokenKind::Star,
            TokenKind::Slash,
            TokenKind::DoubleSlash,
            TokenKind::DoubleStar,
            TokenKind::Percent,
            TokenKind::At,
        ]
    );
}

#[test]
fn test_comparison() {
    let tokens = lex_no_eof("< > <= >= == !=");
    assert_eq!(
        tokens,
        vec![
            TokenKind::Less,
            TokenKind::Greater,
            TokenKind::LessEqual,
            TokenKind::GreaterEqual,
            TokenKind::EqualEqual,
            TokenKind::NotEqual,
        ]
    );
}

#[test]
fn test_augmented_assign() {
    let tokens = lex_no_eof("+= -= *= /=");
    assert_eq!(
        tokens,
        vec![
            TokenKind::PlusEqual,
            TokenKind::MinusEqual,
            TokenKind::StarEqual,
            TokenKind::SlashEqual,
        ]
    );
}

#[test]
fn test_brackets() {
    let tokens = lex_no_eof("( ) [ ] { }");
    assert_eq!(
        tokens,
        vec![
            TokenKind::LeftParen,
            TokenKind::RightParen,
            TokenKind::LeftBracket,
            TokenKind::RightBracket,
            TokenKind::LeftBrace,
            TokenKind::RightBrace,
        ]
    );
}

#[test]
fn test_punctuation() {
    let tokens = lex_no_eof(", : ; . ... -> :=");
    assert_eq!(
        tokens,
        vec![
            TokenKind::Comma,
            TokenKind::Colon,
            TokenKind::Semicolon,
            TokenKind::Dot,
            TokenKind::Ellipsis,
            TokenKind::Arrow,
            TokenKind::ColonEqual,
        ]
    );
}

#[test]
fn test_bitwise() {
    let tokens = lex_no_eof("& | ^ ~ << >>");
    assert_eq!(
        tokens,
        vec![
            TokenKind::Ampersand,
            TokenKind::Pipe,
            TokenKind::Caret,
            TokenKind::Tilde,
            TokenKind::LeftShift,
            TokenKind::RightShift,
        ]
    );
}

#[test]
fn test_integer() {
    let tokens = lex_no_eof("42");
    assert!(matches!(&tokens[0], TokenKind::Int(_)));
}

#[test]
fn test_hex() {
    let tokens = lex_no_eof("0xFF");
    assert_eq!(tokens, vec![TokenKind::Int(255)]);
}

#[test]
fn test_comment_skipped() {
    let tokens = lex_no_eof("x # this is a comment\ny");
    assert!(tokens.contains(&TokenKind::Ident("x".to_string())));
    assert!(tokens.contains(&TokenKind::Ident("y".to_string())));
}

#[test]
fn test_keywords() {
    use crate::token::Keyword;
    let tokens = lex_no_eof("if else while for def class return");
    assert!(tokens.contains(&TokenKind::Keyword(Keyword::If)));
    assert!(tokens.contains(&TokenKind::Keyword(Keyword::Else)));
    assert!(tokens.contains(&TokenKind::Keyword(Keyword::While)));
    assert!(tokens.contains(&TokenKind::Keyword(Keyword::For)));
    assert!(tokens.contains(&TokenKind::Keyword(Keyword::Def)));
    assert!(tokens.contains(&TokenKind::Keyword(Keyword::Class)));
    assert!(tokens.contains(&TokenKind::Keyword(Keyword::Return)));
}

#[test]
fn test_newline() {
    let tokens = lex("x\ny");
    assert!(tokens.iter().any(|t| matches!(t, TokenKind::Newline)));
}

#[test]
fn test_newline_ignored_in_brackets() {
    let tokens = lex_no_eof("(\nx\n)");
    // Should not contain NEWLINE inside brackets
    let newline_count = tokens
        .iter()
        .filter(|t| matches!(t, TokenKind::Newline))
        .count();
    assert_eq!(newline_count, 0);
}

#[test]
fn test_indent_dedent() {
    let source = "if x:\n    y\nz";
    let tokens = lex(source);
    assert!(tokens.iter().any(|t| matches!(t, TokenKind::Indent)));
    assert!(tokens.iter().any(|t| matches!(t, TokenKind::Dedent)));
}

#[test]
fn test_multiple_indent_levels() {
    let source = "if x:\n    if y:\n        z\n    w\nv";
    let tokens = lex(source);
    let indent_count = tokens
        .iter()
        .filter(|t| matches!(t, TokenKind::Indent))
        .count();
    let dedent_count = tokens
        .iter()
        .filter(|t| matches!(t, TokenKind::Dedent))
        .count();
    assert_eq!(indent_count, 2);
    assert_eq!(dedent_count, 2);
}

#[test]
fn test_line_continuation() {
    let tokens = lex_no_eof("x + \\\ny");
    // Should be: x + y (line continuation joins lines)
    assert!(tokens.contains(&TokenKind::Ident("x".to_string())));
    assert!(tokens.contains(&TokenKind::Plus));
    assert!(tokens.contains(&TokenKind::Ident("y".to_string())));
}

#[test]
fn test_arrow() {
    let tokens = lex_no_eof("def f() -> int:");
    assert!(tokens.contains(&TokenKind::Arrow));
}

#[test]
fn test_walrus() {
    let tokens = lex_no_eof("if (x := 5):");
    assert!(tokens.contains(&TokenKind::ColonEqual));
}
