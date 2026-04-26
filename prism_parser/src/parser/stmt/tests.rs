use super::*;
use crate::parser::parse;

fn parse_stmt(s: &str) -> Stmt {
    let module = parse(s).expect("parse failed");
    module.body.into_iter().next().expect("no statements")
}

#[test]
fn test_pass() {
    let stmt = parse_stmt("pass");
    assert!(matches!(stmt.kind, StmtKind::Pass));
}

#[test]
fn test_break() {
    let stmt = parse_stmt("break");
    assert!(matches!(stmt.kind, StmtKind::Break));
}

#[test]
fn test_continue() {
    let stmt = parse_stmt("continue");
    assert!(matches!(stmt.kind, StmtKind::Continue));
}

#[test]
fn test_return() {
    let stmt = parse_stmt("return 42");
    assert!(matches!(stmt.kind, StmtKind::Return(Some(_))));
}

#[test]
fn test_return_none() {
    let stmt = parse_stmt("return");
    assert!(matches!(stmt.kind, StmtKind::Return(None)));
}

#[test]
fn test_generator_statement_yield_tuple_value() {
    let stmt = parse_stmt("def probe():\n    yield 1, 2\n");
    match stmt.kind {
        StmtKind::FunctionDef { body, .. } => match &body[0].kind {
            StmtKind::Expr(expr) => match &expr.kind {
                ExprKind::Yield(Some(value)) => match &value.kind {
                    ExprKind::Tuple(elements) => assert_eq!(elements.len(), 2),
                    other => panic!("expected tuple payload, got {:?}", other),
                },
                other => panic!("expected yield expression, got {:?}", other),
            },
            other => panic!("expected expression statement, got {:?}", other),
        },
        other => panic!("expected function definition, got {:?}", other),
    }
}

#[test]
fn test_expression_stmt() {
    let stmt = parse_stmt("x + 1");
    assert!(matches!(stmt.kind, StmtKind::Expr(_)));
}

#[test]
fn test_assignment() {
    let stmt = parse_stmt("x = 1");
    assert!(matches!(stmt.kind, StmtKind::Assign { .. }));
}

#[test]
fn test_tuple_unpacking_assignment() {
    let stmt = parse_stmt("a, b = 1, 2");
    match stmt.kind {
        StmtKind::Assign { targets, value } => {
            assert_eq!(targets.len(), 1);
            match &targets[0].kind {
                ExprKind::Tuple(elts) => assert_eq!(elts.len(), 2),
                other => panic!("expected tuple target, got {:?}", other),
            }
            match &value.kind {
                ExprKind::Tuple(elts) => assert_eq!(elts.len(), 2),
                other => panic!("expected tuple value, got {:?}", other),
            }
        }
        other => panic!("expected assignment, got {:?}", other),
    }
}

#[test]
fn test_chained_assignment_with_tuple_value() {
    let stmt = parse_stmt("a = b = 1, 2");
    match stmt.kind {
        StmtKind::Assign { targets, value } => {
            assert_eq!(targets.len(), 2);
            match &value.kind {
                ExprKind::Tuple(elts) => assert_eq!(elts.len(), 2),
                other => panic!("expected tuple value, got {:?}", other),
            }
        }
        other => panic!("expected assignment, got {:?}", other),
    }
}

#[test]
fn test_aug_assign() {
    let stmt = parse_stmt("x += 1");
    assert!(matches!(
        stmt.kind,
        StmtKind::AugAssign { op: AugOp::Add, .. }
    ));
}

#[test]
fn test_ann_assign() {
    let stmt = parse_stmt("x: int = 1");
    assert!(matches!(stmt.kind, StmtKind::AnnAssign { .. }));
}

#[test]
fn test_import() {
    let stmt = parse_stmt("import os");
    assert!(matches!(stmt.kind, StmtKind::Import(_)));
}

#[test]
fn test_match_pattern_supports_adjacent_string_literals() {
    let stmt = parse_stmt("match x:\n    case \"meta\" \"class\":\n        pass\n");
    match stmt.kind {
        StmtKind::Match { cases, .. } => match &cases[0].pattern.kind {
            PatternKind::MatchValue(expr) => match &expr.kind {
                ExprKind::String(literal) => assert_eq!(literal.value, "metaclass"),
                other => panic!("expected string literal pattern, got {:?}", other),
            },
            other => panic!("expected literal pattern, got {:?}", other),
        },
        other => panic!("expected match statement, got {:?}", other),
    }
}

#[test]
fn test_from_import() {
    let stmt = parse_stmt("from os import path");
    assert!(matches!(stmt.kind, StmtKind::ImportFrom { .. }));
}

#[test]
fn test_from_import_allows_soft_keyword_module_segment() {
    let stmt = parse_stmt("from unittest.case import TestCase");
    match stmt.kind {
        StmtKind::ImportFrom { module, names, .. } => {
            assert_eq!(module.as_deref(), Some("unittest.case"));
            assert_eq!(names.len(), 1);
            assert_eq!(names[0].name, "TestCase");
        }
        other => panic!("expected import-from, got {:?}", other),
    }
}

#[test]
fn test_assignment_allows_type_identifier() {
    let stmt = parse_stmt("value = type");
    match stmt.kind {
        StmtKind::Assign { value, .. } => match value.kind {
            ExprKind::Name(name) => assert_eq!(name, "type"),
            other => panic!("expected name expression, got {:?}", other),
        },
        other => panic!("expected assignment, got {:?}", other),
    }
}

#[test]
fn test_type_alias_parses_with_soft_keyword_tokenization() {
    let stmt = parse_stmt("type Point = int");
    assert!(matches!(stmt.kind, StmtKind::TypeAlias { .. }));
}

#[test]
fn test_match_statement_parses_with_soft_keyword_tokenization() {
    let module = parse("match x:\n    case 1:\n        pass").expect("parse failed");
    assert!(matches!(module.body[0].kind, StmtKind::Match { .. }));
}

#[test]
fn test_global() {
    let stmt = parse_stmt("global x, y");
    if let StmtKind::Global(names) = stmt.kind {
        assert_eq!(names.len(), 2);
    } else {
        panic!("expected Global");
    }
}

#[test]
fn test_if() {
    let module = parse("if x:\n    pass").expect("parse failed");
    assert_eq!(module.body.len(), 1);
    assert!(matches!(module.body[0].kind, StmtKind::If { .. }));
}

#[test]
fn test_if_elif_nests_orelse_chain() {
    let module =
        parse("if x:\n    pass\nelif y:\n    pass\nelif z:\n    pass").expect("parse failed");

    let StmtKind::If { orelse, .. } = &module.body[0].kind else {
        panic!("expected top-level If");
    };
    assert_eq!(
        orelse.len(),
        1,
        "elif chain should be represented as one nested If"
    );

    let StmtKind::If {
        body: first_elif_body,
        orelse: nested_orelse,
        ..
    } = &orelse[0].kind
    else {
        panic!("expected first elif to lower into nested If");
    };
    assert_eq!(first_elif_body.len(), 1);
    assert_eq!(
        nested_orelse.len(),
        1,
        "second elif should appear in the first elif's orelse"
    );
    assert!(
        matches!(nested_orelse[0].kind, StmtKind::If { .. }),
        "second elif should remain a nested If node"
    );
}

#[test]
fn test_if_elif_else_attaches_to_innermost_nested_if() {
    let module =
        parse("if x:\n    pass\nelif y:\n    pass\nelse:\n    z = 1").expect("parse failed");

    let StmtKind::If { orelse, .. } = &module.body[0].kind else {
        panic!("expected top-level If");
    };
    assert_eq!(orelse.len(), 1, "single elif should still nest in orelse");

    let StmtKind::If {
        orelse: nested_orelse,
        ..
    } = &orelse[0].kind
    else {
        panic!("expected elif to lower into nested If");
    };
    assert_eq!(
        nested_orelse.len(),
        1,
        "else body should attach to the innermost nested If"
    );
    assert!(
        matches!(nested_orelse[0].kind, StmtKind::Assign { .. }),
        "else body should remain ordinary statements, not sibling If nodes"
    );
}

#[test]
fn test_while() {
    let module = parse("while x:\n    pass").expect("parse failed");
    assert!(matches!(module.body[0].kind, StmtKind::While { .. }));
}

#[test]
fn test_for() {
    let module = parse("for x in y:\n    pass").expect("parse failed");
    assert!(matches!(module.body[0].kind, StmtKind::For { .. }));
}

#[test]
fn test_for_accepts_implicit_tuple_iterable() {
    let module = parse("for x in False, True:\n    pass").expect("parse failed");
    let StmtKind::For { iter, .. } = &module.body[0].kind else {
        panic!("expected For");
    };

    match &iter.kind {
        ExprKind::Tuple(elements) => assert_eq!(elements.len(), 2),
        other => panic!("expected tuple iterable, got {:?}", other),
    }
}

#[test]
fn test_function_def() {
    let module = parse("def foo():\n    pass").expect("parse failed");
    assert!(matches!(module.body[0].kind, StmtKind::FunctionDef { .. }));
}

#[test]
fn test_inline_function_def() {
    let module = parse("def foo(): pass").expect("parse failed");
    let StmtKind::FunctionDef { body, .. } = &module.body[0].kind else {
        panic!("expected FunctionDef");
    };
    assert_eq!(body.len(), 1);
    assert!(matches!(body[0].kind, StmtKind::Pass));
}

#[test]
fn test_class_def() {
    let module = parse("class Foo:\n    pass").expect("parse failed");
    assert!(matches!(module.body[0].kind, StmtKind::ClassDef { .. }));
}

#[test]
fn test_inline_class_method_def() {
    let module = parse("class Foo:\n    def bar(self): pass").expect("parse failed");
    let StmtKind::ClassDef { body, .. } = &module.body[0].kind else {
        panic!("expected ClassDef");
    };
    assert_eq!(body.len(), 1);

    let StmtKind::FunctionDef { body, .. } = &body[0].kind else {
        panic!("expected FunctionDef in class body");
    };
    assert_eq!(body.len(), 1);
    assert!(matches!(body[0].kind, StmtKind::Pass));
}

#[test]
fn test_decorator_rejects_bare_tuple_and_assignment_forms() {
    for source in [
        "@x,\ndef f(): pass",
        "@x, y\ndef f(): pass",
        "@x = y\ndef f(): pass",
    ] {
        assert!(
            parse(source).is_err(),
            "decorator form should be rejected: {source:?}"
        );
    }
}

#[test]
fn test_decorator_rejects_statement_keywords() {
    for source in ["@pass\ndef f(): pass", "@import sys\ndef f(): pass"] {
        assert!(
            parse(source).is_err(),
            "statement-like decorator should be rejected: {source:?}"
        );
    }
}

#[test]
fn test_decorator_accepts_parenthesized_tuple_and_named_expression() {
    for source in ["@(x, y)\ndef f(): pass", "@(x := y)\ndef f(): pass"] {
        assert!(
            parse(source).is_ok(),
            "valid decorator expression should parse: {source:?}"
        );
    }
}

#[test]
fn test_try() {
    let module = parse("try:\n    pass\nexcept:\n    pass").expect("parse failed");
    assert!(matches!(module.body[0].kind, StmtKind::Try { .. }));
}

#[test]
fn test_with() {
    let module = parse("with x:\n    pass").expect("parse failed");
    assert!(matches!(module.body[0].kind, StmtKind::With { .. }));
}

#[test]
fn test_inline_if_suite_with_semicolons() {
    let module = parse("if x: pass; y = 1").expect("parse failed");
    let StmtKind::If { body, .. } = &module.body[0].kind else {
        panic!("expected If");
    };
    assert_eq!(body.len(), 2);
    assert!(matches!(body[0].kind, StmtKind::Pass));
    assert!(matches!(body[1].kind, StmtKind::Assign { .. }));
}

#[test]
fn test_top_level_semicolon_separates_simple_statements() {
    let module = parse("x = 1; y = 2; z = 3").expect("parse failed");
    assert_eq!(module.body.len(), 3);
    assert!(matches!(module.body[0].kind, StmtKind::Assign { .. }));
    assert!(matches!(module.body[1].kind, StmtKind::Assign { .. }));
    assert!(matches!(module.body[2].kind, StmtKind::Assign { .. }));
}

#[test]
fn test_indented_block_semicolon_separates_simple_statements() {
    let module = parse("if True:\n    x = 1; y = 2\n").expect("parse failed");
    let StmtKind::If { body, .. } = &module.body[0].kind else {
        panic!("expected If");
    };
    assert_eq!(body.len(), 2);
    assert!(matches!(body[0].kind, StmtKind::Assign { .. }));
    assert!(matches!(body[1].kind, StmtKind::Assign { .. }));
}
