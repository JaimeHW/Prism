use super::*;

#[test]
fn test_parser_creation() {
    let parser = Parser::new("x + 1");
    assert!(!parser.is_at_end());
}

#[test]
fn test_empty_module() {
    let result = parse("");
    assert!(result.is_ok());
    let module = result.unwrap();
    assert!(module.body.is_empty());
}

#[test]
fn test_precedence_ordering() {
    assert!(Precedence::Primary > Precedence::Lowest);
    assert!(Precedence::Multiplicative > Precedence::Additive);
    assert!(Precedence::Power > Precedence::Unary);
}

#[test]
fn test_precedence_next() {
    assert_eq!(Precedence::Additive.next(), Precedence::Multiplicative);
    assert_eq!(Precedence::Primary.next(), Precedence::Primary);
}

#[test]
fn test_missing_compound_colon_reports_syntax_error() {
    let err = parse("if True\n    pass\n").expect_err("missing colon should fail");
    assert!(
        err.to_string().contains("expected ':'"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_recovery_skips_orphaned_indent_after_syntax_error() {
    let err = parse("if True\n    pass\nx = 1\n").expect_err("missing colon should fail");
    assert!(
        err.to_string().contains("expected ':'"),
        "unexpected error: {err}"
    );
}
