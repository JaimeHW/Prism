use super::*;

#[test]
fn test_indent_stack_new() {
    let stack = IndentStack::new();
    assert_eq!(stack.current_indent(), 0);
    assert!(stack.at_line_start());
    assert!(stack.tracking_indent());
}

#[test]
fn test_indent_increase() {
    let mut stack = IndentStack::new();
    let result = stack.process_indent(4);
    assert_eq!(result, Ok(Some(true))); // INDENT
    assert_eq!(stack.current_indent(), 4);
}

#[test]
fn test_indent_same_level() {
    let mut stack = IndentStack::new();
    stack.process_indent(4).unwrap();
    let result = stack.process_indent(4);
    assert_eq!(result, Ok(None)); // No change
}

#[test]
fn test_dedent_single() {
    let mut stack = IndentStack::new();
    stack.process_indent(4).unwrap();
    let result = stack.process_indent(0);
    assert_eq!(result, Ok(Some(false))); // DEDENT(s)
    assert_eq!(stack.pending_dedents, 1);
    assert!(stack.consume_dedent());
    assert!(!stack.consume_dedent());
}

#[test]
fn test_dedent_multiple() {
    let mut stack = IndentStack::new();
    stack.process_indent(4).unwrap();
    stack.process_indent(8).unwrap();
    stack.process_indent(12).unwrap();

    let result = stack.process_indent(0);
    assert_eq!(result, Ok(Some(false)));
    assert_eq!(stack.pending_dedents, 3);
}

#[test]
fn test_dedent_partial() {
    let mut stack = IndentStack::new();
    stack.process_indent(4).unwrap();
    stack.process_indent(8).unwrap();

    let result = stack.process_indent(4);
    assert_eq!(result, Ok(Some(false)));
    assert_eq!(stack.pending_dedents, 1);
    assert_eq!(stack.current_indent(), 4);
}

#[test]
fn test_inconsistent_dedent() {
    let mut stack = IndentStack::new();
    stack.process_indent(4).unwrap();
    let result = stack.process_indent(2);
    assert_eq!(result, Err("inconsistent dedent"));
}

#[test]
fn test_bracket_nesting_disables_indent() {
    let mut stack = IndentStack::new();
    stack.open_bracket();
    assert!(!stack.tracking_indent());

    let result = stack.process_indent(100);
    assert_eq!(result, Ok(None)); // Ignored inside brackets

    stack.close_bracket();
    assert!(stack.tracking_indent());
}

#[test]
fn test_close_all() {
    let mut stack = IndentStack::new();
    stack.process_indent(4).unwrap();
    stack.process_indent(8).unwrap();

    let count = stack.close_all();
    assert_eq!(count, 2);
    assert_eq!(stack.pending_dedents, 2);
    assert_eq!(stack.current_indent(), 0);
}

#[test]
fn test_line_start_tracking() {
    let mut stack = IndentStack::new();
    assert!(stack.at_line_start());

    stack.consumed_content();
    assert!(!stack.at_line_start());

    stack.new_line();
    assert!(stack.at_line_start());
}

#[test]
fn test_depth() {
    let mut stack = IndentStack::new();
    assert_eq!(stack.depth(), 0);

    stack.process_indent(4).unwrap();
    assert_eq!(stack.depth(), 1);

    stack.process_indent(8).unwrap();
    assert_eq!(stack.depth(), 2);
}

#[test]
fn test_nested_brackets() {
    let mut stack = IndentStack::new();
    stack.open_bracket();
    stack.open_bracket();
    assert!(!stack.tracking_indent());

    stack.close_bracket();
    assert!(!stack.tracking_indent());

    stack.close_bracket();
    assert!(stack.tracking_indent());
}

#[test]
fn test_close_bracket_underflow() {
    let mut stack = IndentStack::new();
    stack.close_bracket(); // Should not panic
    assert!(stack.tracking_indent());
}
