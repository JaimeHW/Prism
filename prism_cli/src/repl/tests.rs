use super::*;

// =========================================================================
// Continuation Detection Tests
// =========================================================================

#[test]
fn test_needs_continuation_def() {
    assert!(needs_continuation("def foo():"));
}

#[test]
fn test_needs_continuation_class() {
    assert!(needs_continuation("class MyClass:"));
}

#[test]
fn test_needs_continuation_if() {
    assert!(needs_continuation("if x > 0:"));
}

#[test]
fn test_needs_continuation_elif() {
    assert!(needs_continuation("elif x < 0:"));
}

#[test]
fn test_needs_continuation_else() {
    assert!(needs_continuation("else:"));
}

#[test]
fn test_needs_continuation_for() {
    assert!(needs_continuation("for i in range(10):"));
}

#[test]
fn test_needs_continuation_while() {
    assert!(needs_continuation("while True:"));
}

#[test]
fn test_needs_continuation_try() {
    assert!(needs_continuation("try:"));
}

#[test]
fn test_needs_continuation_except() {
    assert!(needs_continuation("except ValueError:"));
}

#[test]
fn test_needs_continuation_finally() {
    assert!(needs_continuation("finally:"));
}

#[test]
fn test_needs_continuation_with() {
    assert!(needs_continuation("with open('f') as f:"));
}

#[test]
fn test_needs_continuation_match() {
    assert!(needs_continuation("match x:"));
}

#[test]
fn test_needs_continuation_case() {
    assert!(needs_continuation("case 1:"));
}

#[test]
fn test_needs_continuation_async() {
    assert!(needs_continuation("async def foo():"));
}

#[test]
fn test_needs_continuation_backslash() {
    assert!(needs_continuation("x = 1 + \\"));
}

#[test]
fn test_no_continuation_assignment() {
    assert!(!needs_continuation("x = 42"));
}

#[test]
fn test_no_continuation_function_call() {
    assert!(!needs_continuation("print('hello')"));
}

#[test]
fn test_no_continuation_colon_in_dict() {
    // Dict literal has `:` but doesn't start with a keyword.
    assert!(!needs_continuation("d = {'a': 1}"));
}

#[test]
fn test_no_continuation_empty() {
    assert!(!needs_continuation(""));
}

#[test]
fn test_no_continuation_comment() {
    assert!(!needs_continuation("# comment"));
}

// =========================================================================
// Value Formatting Tests
// =========================================================================

#[test]
fn test_format_value_int() {
    let v = prism_core::Value::int(42).unwrap();
    assert_eq!(format_value(&v), "42");
}

#[test]
fn test_format_value_negative_int() {
    let v = prism_core::Value::int(-7).unwrap();
    assert_eq!(format_value(&v), "-7");
}

#[test]
fn test_format_value_zero() {
    let v = prism_core::Value::int(0).unwrap();
    assert_eq!(format_value(&v), "0");
}

#[test]
fn test_format_value_float() {
    let v = prism_core::Value::float(3.14);
    assert_eq!(format_value(&v), "3.14");
}

#[test]
fn test_format_value_float_whole() {
    let v = prism_core::Value::float(42.0);
    assert_eq!(format_value(&v), "42.0");
}

#[test]
fn test_format_value_bool_true() {
    let v = prism_core::Value::bool(true);
    assert_eq!(format_value(&v), "True");
}

#[test]
fn test_format_value_bool_false() {
    let v = prism_core::Value::bool(false);
    assert_eq!(format_value(&v), "False");
}

#[test]
fn test_format_value_none() {
    let v = prism_core::Value::none();
    assert_eq!(format_value(&v), "None");
}

// =========================================================================
// REPL Execution Tests
// =========================================================================

#[test]
fn test_execute_repl_preserves_main_module_state_between_inputs() {
    let mut vm = prism_vm::VirtualMachine::new();
    let config = RuntimeConfig::from_args(&crate::args::PrismArgs::default());

    execute_repl_input("x = 41\n", &mut vm, &config);
    execute_repl_input("y = x + 1\n", &mut vm, &config);

    let main = vm
        .cached_module("__main__")
        .expect("repl should cache a persistent __main__ module");
    assert_eq!(
        main.get_attr("x").and_then(|value| value.as_int()),
        Some(41)
    );
    assert_eq!(
        main.get_attr("y").and_then(|value| value.as_int()),
        Some(42)
    );
}
