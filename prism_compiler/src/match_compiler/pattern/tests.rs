use super::*;
use prism_core::Span;

fn make_pattern(kind: PatternKind) -> Pattern {
    Pattern {
        kind,
        span: Span::new(0, 0),
    }
}

#[test]
fn test_flatten_wildcard() {
    let pat = make_pattern(PatternKind::MatchAs {
        pattern: None,
        name: None,
    });
    let (flat, bindings) = flatten(&pat).unwrap();
    assert!(matches!(flat, FlatPattern::Wildcard));
    assert!(bindings.is_empty());
}

#[test]
fn test_flatten_binding() {
    let pat = make_pattern(PatternKind::MatchAs {
        pattern: None,
        name: Some("x".to_string()),
    });
    let (flat, bindings) = flatten(&pat).unwrap();
    assert!(matches!(flat, FlatPattern::Binding(ref n) if n.as_ref() == "x"));
    assert_eq!(bindings.len(), 1);
    assert_eq!(bindings[0].name.as_ref(), "x");
}

#[test]
fn test_flatten_singleton() {
    let pat = make_pattern(PatternKind::MatchSingleton(Singleton::None));
    let (flat, _) = flatten(&pat).unwrap();
    assert!(matches!(flat, FlatPattern::Singleton(Singleton::None)));
}

#[test]
fn test_flatten_literal_int() {
    let expr = Expr::new(ExprKind::Int(42), Span::new(0, 0));
    let pat = make_pattern(PatternKind::MatchValue(Box::new(expr)));
    let (flat, _) = flatten(&pat).unwrap();
    assert!(matches!(flat, FlatPattern::Literal(LiteralValue::Int(42))));
}

#[test]
fn test_flatten_runtime_value_pattern_rejects_debug_string_placeholder() {
    let expr = Expr::new(ExprKind::Name("Color.RED".to_string()), Span::new(0, 0));
    let pat = make_pattern(PatternKind::MatchValue(Box::new(expr)));
    let err = flatten(&pat).expect_err("runtime value patterns need explicit lowering");

    assert!(
        err.message
            .contains("unsupported pattern value expression for decision-tree lowering"),
        "unexpected error: {err:?}"
    );
}

#[test]
fn test_flatten_sequence_fixed() {
    let patterns = vec![
        make_pattern(PatternKind::MatchAs {
            pattern: None,
            name: Some("a".to_string()),
        }),
        make_pattern(PatternKind::MatchAs {
            pattern: None,
            name: Some("b".to_string()),
        }),
    ];
    let pat = make_pattern(PatternKind::MatchSequence(patterns));
    let (flat, bindings) = flatten(&pat).unwrap();

    if let FlatPattern::Sequence {
        min_len,
        star_index,
        patterns,
    } = flat
    {
        assert_eq!(min_len, 2);
        assert!(star_index.is_none());
        assert_eq!(patterns.len(), 2);
    } else {
        panic!("Expected Sequence pattern");
    }
    assert_eq!(bindings.len(), 2);
}

#[test]
fn test_access_path_chaining() {
    let path = AccessPath::Root.index(0).attr(Arc::from("x"));
    assert!(matches!(path, AccessPath::Attr(_, _)));
}
