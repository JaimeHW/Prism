use super::*;
use crate::parser::parse_expression;

fn parse(s: &str) -> Expr {
    parse_expression(s).expect("parse failed")
}

#[test]
fn test_integer() {
    let expr = parse("42");
    if let ExprKind::Int(n) = expr.kind {
        assert_eq!(n, 42);
    } else {
        panic!("expected Int, got {:?}", expr.kind);
    }
}

#[test]
fn test_float() {
    let expr = parse("3.125");
    if let ExprKind::Float(f) = expr.kind {
        assert!((f - 3.125).abs() < 0.001);
    } else {
        panic!("expected Float, got {:?}", expr.kind);
    }
}

#[test]
fn test_identifier() {
    let expr = parse("foo");
    assert!(matches!(expr.kind, ExprKind::Name(ref n) if n == "foo"));
}

#[test]
fn test_adjacent_string_literals_are_concatenated() {
    let expr = parse("\"meta\" \"class\"");
    match expr.kind {
        ExprKind::String(literal) => assert_eq!(literal.value, "metaclass"),
        other => panic!("expected String, got {:?}", other),
    }
}

#[test]
fn test_parenthesized_multiline_string_literals_are_concatenated() {
    let expr = parse("(\"meta\" \n \"class\")");
    match expr.kind {
        ExprKind::String(literal) => assert_eq!(literal.value, "metaclass"),
        other => panic!("expected String, got {:?}", other),
    }
}

#[test]
fn test_adjacent_bytes_literals_are_concatenated() {
    let expr = parse("b\"meta\" b\"class\"");
    match expr.kind {
        ExprKind::Bytes(literal) => assert_eq!(literal, b"metaclass"),
        other => panic!("expected Bytes, got {:?}", other),
    }
}

#[test]
fn test_bytes_literal_hex_escapes_preserve_raw_byte_values() {
    let expr = parse("b\"\\x00A\\xff\"");
    match expr.kind {
        ExprKind::Bytes(literal) => assert_eq!(literal, vec![0x00, b'A', 0xff]),
        other => panic!("expected Bytes, got {:?}", other),
    }
}

#[test]
fn test_mixed_string_and_bytes_literals_are_rejected() {
    let err = parse_expression("(\"meta\" b\"class\")").expect_err("expected syntax error");
    assert!(
        format!("{err}").contains("cannot mix bytes and nonbytes literals"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_fstring_lowers_to_joined_str() {
    let expr = parse("f\"{key} = 42\"");
    let ExprKind::JoinedStr(parts) = expr.kind else {
        panic!("expected JoinedStr");
    };

    assert_eq!(parts.len(), 2);
    assert!(matches!(
        &parts[0].kind,
        ExprKind::FormattedValue {
            value,
            conversion: -1,
            format_spec: None,
        } if matches!(value.kind, ExprKind::Name(ref name) if name == "key")
    ));
    assert!(matches!(
        parts[1].kind,
        ExprKind::String(ref literal) if literal.value == " = 42"
    ));
}

#[test]
fn test_adjacent_plain_and_formatted_strings_join_into_single_joined_str() {
    let expr = parse("\"prefix\" f\"{value}\" \"suffix\"");
    let ExprKind::JoinedStr(parts) = expr.kind else {
        panic!("expected JoinedStr");
    };

    assert_eq!(parts.len(), 3);
    assert!(matches!(
        parts[0].kind,
        ExprKind::String(ref literal) if literal.value == "prefix"
    ));
    assert!(matches!(parts[1].kind, ExprKind::FormattedValue { .. }));
    assert!(matches!(
        parts[2].kind,
        ExprKind::String(ref literal) if literal.value == "suffix"
    ));
}

#[test]
fn test_fstring_conversion_and_nested_format_spec_are_parsed() {
    let expr = parse("f\"{value!r:{width}}\"");
    let ExprKind::JoinedStr(parts) = expr.kind else {
        panic!("expected JoinedStr");
    };

    let ExprKind::FormattedValue {
        value,
        conversion,
        format_spec,
    } = &parts[0].kind
    else {
        panic!("expected formatted value");
    };

    assert!(matches!(value.kind, ExprKind::Name(ref name) if name == "value"));
    assert_eq!(*conversion, 'r' as i8);

    let spec = format_spec.as_ref().expect("expected nested format spec");
    let ExprKind::JoinedStr(spec_parts) = &spec.kind else {
        panic!("expected JoinedStr format spec");
    };
    assert!(matches!(
        &spec_parts[0].kind,
        ExprKind::FormattedValue {
            value,
            conversion: -1,
            format_spec: None,
        } if matches!(value.kind, ExprKind::Name(ref name) if name == "width")
    ));
}

#[test]
fn test_fstring_debug_syntax_lowers_to_literal_plus_repr_formatted_value() {
    let expr = parse("f\"{value=}\"");
    let ExprKind::JoinedStr(parts) = expr.kind else {
        panic!("expected JoinedStr");
    };

    assert_eq!(parts.len(), 2);
    assert!(matches!(
        parts[0].kind,
        ExprKind::String(ref literal) if literal.value == "value="
    ));
    assert!(matches!(
        &parts[1].kind,
        ExprKind::FormattedValue {
            value,
            conversion,
            format_spec: None,
        } if *conversion == 'r' as i8
            && matches!(value.kind, ExprKind::Name(ref name) if name == "value")
    ));
}

#[test]
fn test_fstring_debug_syntax_preserves_literal_spacing_and_nested_format_spec() {
    let expr = parse("f\"{value = :>{width}}\"");
    let ExprKind::JoinedStr(parts) = expr.kind else {
        panic!("expected JoinedStr");
    };

    assert_eq!(parts.len(), 2);
    assert!(matches!(
        parts[0].kind,
        ExprKind::String(ref literal) if literal.value == "value = "
    ));

    let ExprKind::FormattedValue {
        value,
        conversion,
        format_spec,
    } = &parts[1].kind
    else {
        panic!("expected formatted value");
    };

    assert!(matches!(value.kind, ExprKind::Name(ref name) if name == "value"));
    assert_eq!(*conversion, 'r' as i8);

    let spec = format_spec.as_ref().expect("expected nested format spec");
    let ExprKind::JoinedStr(spec_parts) = &spec.kind else {
        panic!("expected JoinedStr format spec");
    };
    assert!(matches!(
        &spec_parts[1].kind,
        ExprKind::FormattedValue {
            value,
            conversion: -1,
            format_spec: None,
        } if matches!(value.kind, ExprKind::Name(ref name) if name == "width")
    ));
}

#[test]
fn test_fstring_equality_expression_is_not_misparsed_as_debug_syntax() {
    let expr = parse("f\"{left == right}\"");
    let ExprKind::JoinedStr(parts) = expr.kind else {
        panic!("expected JoinedStr");
    };

    assert_eq!(parts.len(), 1);
    assert!(matches!(
        &parts[0].kind,
        ExprKind::FormattedValue {
            value,
            conversion: -1,
            format_spec: None,
        } if matches!(value.kind, ExprKind::Compare { .. })
    ));
}

#[test]
fn test_binary_add() {
    let expr = parse("1 + 2");
    assert!(matches!(expr.kind, ExprKind::BinOp { op: BinOp::Add, .. }));
}

#[test]
fn test_precedence() {
    let expr = parse("1 + 2 * 3");
    // Should parse as 1 + (2 * 3)
    if let ExprKind::BinOp { op, right, .. } = expr.kind {
        assert_eq!(op, BinOp::Add);
        assert!(matches!(
            right.kind,
            ExprKind::BinOp {
                op: BinOp::Mult,
                ..
            }
        ));
    } else {
        panic!("expected BinOp");
    }
}

#[test]
fn test_unary_minus() {
    let expr = parse("-x");
    assert!(matches!(
        expr.kind,
        ExprKind::UnaryOp {
            op: UnaryOp::USub,
            ..
        }
    ));
}

#[test]
fn test_comparison() {
    let expr = parse("a < b");
    assert!(matches!(expr.kind, ExprKind::Compare { .. }));
}

#[test]
fn test_chained_comparison() {
    let expr = parse("a < b <= c");
    if let ExprKind::Compare {
        ops, comparators, ..
    } = expr.kind
    {
        assert_eq!(ops.len(), 2);
        assert_eq!(comparators.len(), 2);
    } else {
        panic!("expected Compare");
    }
}

#[test]
fn test_bool_and() {
    let expr = parse("a and b");
    assert!(matches!(
        expr.kind,
        ExprKind::BoolOp {
            op: BoolOp::And,
            ..
        }
    ));
}

#[test]
fn test_attribute() {
    let expr = parse("x.y");
    assert!(matches!(expr.kind, ExprKind::Attribute { .. }));
}

#[test]
fn test_call() {
    let expr = parse("f(x)");
    assert!(matches!(expr.kind, ExprKind::Call { .. }));
}

#[test]
fn test_subscript() {
    let expr = parse("x[0]");
    assert!(matches!(expr.kind, ExprKind::Subscript { .. }));
}

#[test]
fn test_subscript_tuple_index() {
    let expr = parse("matrix[row, col]");

    let ExprKind::Subscript { slice, .. } = expr.kind else {
        panic!("expected Subscript");
    };

    let ExprKind::Tuple(items) = slice.kind else {
        panic!("expected tuple subscript");
    };

    assert_eq!(items.len(), 2);
    assert!(matches!(items[0].kind, ExprKind::Name(_)));
    assert!(matches!(items[1].kind, ExprKind::Name(_)));
}

#[test]
fn test_subscript_tuple_mixed_slice_and_index() {
    let expr = parse("matrix[1:3, col]");

    let ExprKind::Subscript { slice, .. } = expr.kind else {
        panic!("expected Subscript");
    };

    let ExprKind::Tuple(items) = slice.kind else {
        panic!("expected tuple subscript");
    };

    assert_eq!(items.len(), 2);
    assert!(matches!(items[0].kind, ExprKind::Slice { .. }));
    assert!(matches!(items[1].kind, ExprKind::Name(_)));
}

#[test]
fn test_list() {
    let expr = parse("[1, 2, 3]");
    if let ExprKind::List(elements) = expr.kind {
        assert_eq!(elements.len(), 3);
    } else {
        panic!("expected List");
    }
}

#[test]
fn test_tuple() {
    let expr = parse("(1, 2)");
    if let ExprKind::Tuple(elements) = expr.kind {
        assert_eq!(elements.len(), 2);
    } else {
        panic!("expected Tuple");
    }
}

#[test]
fn test_yield_value_absorbs_comma_tuple() {
    let mut parser = Parser::new("yield 1, 2");
    let expr = ExprParser::parse(&mut parser, Precedence::Lowest).expect("parse failed");

    match expr.kind {
        ExprKind::Yield(Some(value)) => match value.kind {
            ExprKind::Tuple(elements) => assert_eq!(elements.len(), 2),
            other => panic!("expected yield tuple payload, got {:?}", other),
        },
        other => panic!("expected yield expression, got {:?}", other),
    }

    assert!(parser.check(TokenKind::Eof));
}

#[test]
fn test_dict() {
    let expr = parse("{1: 2}");
    assert!(matches!(expr.kind, ExprKind::Dict { .. }));
}

#[test]
fn test_set() {
    let expr = parse("{1, 2}");
    if let ExprKind::Set(elements) = expr.kind {
        assert_eq!(elements.len(), 2);
    } else {
        panic!("expected Set");
    }
}

#[test]
fn test_conditional() {
    let expr = parse("a if b else c");
    assert!(matches!(expr.kind, ExprKind::IfExp { .. }));
}

#[test]
fn test_lambda() {
    let expr = parse("lambda x: x + 1");
    assert!(matches!(expr.kind, ExprKind::Lambda { .. }));
}

#[test]
fn test_lambda_no_params() {
    let expr = parse("lambda: 42");
    if let ExprKind::Lambda { args, .. } = expr.kind {
        assert!(args.args.is_empty());
        assert!(args.vararg.is_none());
        assert!(args.kwarg.is_none());
    } else {
        panic!("expected Lambda");
    }
}

#[test]
fn test_lambda_multiple_params() {
    let expr = parse("lambda x, y, z: x + y + z");
    if let ExprKind::Lambda { args, .. } = expr.kind {
        assert_eq!(args.args.len(), 3);
    } else {
        panic!("expected Lambda");
    }
}

#[test]
fn test_lambda_with_defaults() {
    let expr = parse("lambda x, y=10: x + y");
    if let ExprKind::Lambda { args, .. } = expr.kind {
        assert_eq!(args.args.len(), 2);
        assert_eq!(args.defaults.len(), 1);
    } else {
        panic!("expected Lambda");
    }
}

#[test]
fn test_lambda_varargs() {
    let expr = parse("lambda *args: args");
    if let ExprKind::Lambda { args, .. } = expr.kind {
        assert!(args.vararg.is_some());
        assert_eq!(args.vararg.as_ref().unwrap().arg.as_str(), "args");
    } else {
        panic!("expected Lambda");
    }
}

#[test]
fn test_lambda_kwargs() {
    let expr = parse("lambda **kwargs: kwargs");
    if let ExprKind::Lambda { args, .. } = expr.kind {
        assert!(args.kwarg.is_some());
        assert_eq!(args.kwarg.as_ref().unwrap().arg.as_str(), "kwargs");
    } else {
        panic!("expected Lambda");
    }
}

#[test]
fn test_lambda_varargs_and_kwargs() {
    let expr = parse("lambda *args, **kwargs: (args, kwargs)");
    if let ExprKind::Lambda { args, .. } = expr.kind {
        assert!(args.vararg.is_some());
        assert!(args.kwarg.is_some());
    } else {
        panic!("expected Lambda");
    }
}

#[test]
fn test_lambda_positional_and_varargs() {
    let expr = parse("lambda a, b, *args: (a, b, args)");
    if let ExprKind::Lambda { args, .. } = expr.kind {
        assert_eq!(args.args.len(), 2);
        assert!(args.vararg.is_some());
    } else {
        panic!("expected Lambda");
    }
}

#[test]
fn test_lambda_keyword_only() {
    let expr = parse("lambda *, key: key");
    if let ExprKind::Lambda { args, .. } = expr.kind {
        assert!(args.vararg.is_none()); // bare *
        assert_eq!(args.kwonlyargs.len(), 1);
        assert_eq!(args.kwonlyargs[0].arg.as_str(), "key");
    } else {
        panic!("expected Lambda");
    }
}

#[test]
fn test_lambda_keyword_only_with_default() {
    let expr = parse("lambda *, a, b=10: a + b");
    if let ExprKind::Lambda { args, .. } = expr.kind {
        assert_eq!(args.kwonlyargs.len(), 2);
        assert_eq!(args.kw_defaults.len(), 2);
        // First kwonly has no default (None), second has default (Some)
        assert!(args.kw_defaults[0].is_none());
        assert!(args.kw_defaults[1].is_some());
    } else {
        panic!("expected Lambda");
    }
}

#[test]
fn test_lambda_varargs_and_keyword_only() {
    let expr = parse("lambda *args, key: (args, key)");
    if let ExprKind::Lambda { args, .. } = expr.kind {
        assert!(args.vararg.is_some());
        assert_eq!(args.kwonlyargs.len(), 1);
    } else {
        panic!("expected Lambda");
    }
}

#[test]
fn test_lambda_full_signature() {
    let expr = parse("lambda a, b=1, *args, c, d=2, **kwargs: (a, b, args, c, d, kwargs)");
    if let ExprKind::Lambda { args, .. } = expr.kind {
        assert_eq!(args.args.len(), 2); // a, b
        assert_eq!(args.defaults.len(), 1); // b=1
        assert!(args.vararg.is_some()); // *args
        assert_eq!(args.kwonlyargs.len(), 2); // c, d
        assert_eq!(args.kw_defaults.len(), 2); // c (None), d=2 (Some)
        assert!(args.kwarg.is_some()); // **kwargs
    } else {
        panic!("expected Lambda");
    }
}

#[test]
fn test_lambda_positional_only() {
    let expr = parse("lambda a, b, /: a + b");
    if let ExprKind::Lambda { args, .. } = expr.kind {
        assert_eq!(args.posonlyargs.len(), 2);
        assert!(args.args.is_empty());
    } else {
        panic!("expected Lambda");
    }
}

#[test]
fn test_lambda_positional_only_and_regular() {
    let expr = parse("lambda a, /, b: a + b");
    if let ExprKind::Lambda { args, .. } = expr.kind {
        assert_eq!(args.posonlyargs.len(), 1);
        assert_eq!(args.args.len(), 1);
    } else {
        panic!("expected Lambda");
    }
}

#[test]
fn test_lambda_complete_signature() {
    // pos_only, /, regular, *args, kw_only, **kwargs
    let expr = parse("lambda a, /, b, *args, c, **kw: 0");
    if let ExprKind::Lambda { args, .. } = expr.kind {
        assert_eq!(args.posonlyargs.len(), 1); // a
        assert_eq!(args.args.len(), 1); // b
        assert!(args.vararg.is_some()); // *args
        assert_eq!(args.kwonlyargs.len(), 1); // c
        assert!(args.kwarg.is_some()); // **kw
    } else {
        panic!("expected Lambda");
    }
}

#[test]
fn test_power_right_assoc() {
    let expr = parse("2 ** 3 ** 4");
    // Should parse as 2 ** (3 ** 4)
    if let ExprKind::BinOp { right, .. } = expr.kind {
        assert!(matches!(right.kind, ExprKind::BinOp { op: BinOp::Pow, .. }));
    } else {
        panic!("expected BinOp");
    }
}
