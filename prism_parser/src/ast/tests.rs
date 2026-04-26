use super::*;

#[test]
fn test_module_creation() {
    let module = Module::new(vec![], Span::dummy());
    assert!(module.body.is_empty());
}

#[test]
fn test_stmt_creation() {
    let stmt = Stmt::new(StmtKind::Pass, Span::new(0, 4));
    assert!(matches!(stmt.kind, StmtKind::Pass));
}

#[test]
fn test_expr_creation() {
    let expr = Expr::new(ExprKind::Int(42), Span::new(0, 2));
    assert!(matches!(expr.kind, ExprKind::Int(42)));
}

#[test]
fn test_string_literal() {
    let lit = StringLiteral::new("hello".to_string());
    assert_eq!(lit.value, "hello");
    assert!(lit.unicode);
}
