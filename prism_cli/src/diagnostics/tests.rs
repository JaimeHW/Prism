use super::*;

// =========================================================================
// SourceMap Construction Tests
// =========================================================================

#[test]
fn test_source_map_single_line() {
    let sm = SourceMap::new("hello", "test.py");
    assert_eq!(sm.line_count(), 1);
    assert_eq!(sm.line_text(1), Some("hello"));
}

#[test]
fn test_source_map_multiple_lines() {
    let sm = SourceMap::new("line1\nline2\nline3", "test.py");
    assert_eq!(sm.line_count(), 3);
    assert_eq!(sm.line_text(1), Some("line1"));
    assert_eq!(sm.line_text(2), Some("line2"));
    assert_eq!(sm.line_text(3), Some("line3"));
}

#[test]
fn test_source_map_trailing_newline() {
    let sm = SourceMap::new("line1\nline2\n", "test.py");
    assert_eq!(sm.line_count(), 3);
    assert_eq!(sm.line_text(1), Some("line1"));
    assert_eq!(sm.line_text(2), Some("line2"));
    assert_eq!(sm.line_text(3), Some(""));
}

#[test]
fn test_source_map_empty() {
    let sm = SourceMap::new("", "test.py");
    assert_eq!(sm.line_count(), 1);
    assert_eq!(sm.line_text(1), Some(""));
}

#[test]
fn test_source_map_crlf() {
    let sm = SourceMap::new("line1\r\nline2\r\n", "test.py");
    // \r\n splits on \n, so \r remains at end of each line.
    assert_eq!(sm.line_text(1), Some("line1"));
    assert_eq!(sm.line_text(2), Some("line2"));
}

#[test]
fn test_source_map_filename() {
    let sm = SourceMap::new("x", "/path/to/script.py");
    assert_eq!(sm.filename(), "/path/to/script.py");
}

// =========================================================================
// Position Resolution Tests
// =========================================================================

#[test]
fn test_resolve_first_char() {
    let sm = SourceMap::new("hello\nworld", "test.py");
    let pos = sm.resolve(0);
    assert_eq!(pos, SourcePosition { line: 1, column: 0 });
}

#[test]
fn test_resolve_mid_first_line() {
    let sm = SourceMap::new("hello\nworld", "test.py");
    let pos = sm.resolve(3);
    assert_eq!(pos, SourcePosition { line: 1, column: 3 });
}

#[test]
fn test_resolve_newline_char() {
    let sm = SourceMap::new("hello\nworld", "test.py");
    let pos = sm.resolve(5);
    // Offset 5 is the newline itself — end of line 1.
    assert_eq!(pos, SourcePosition { line: 1, column: 5 });
}

#[test]
fn test_resolve_second_line_start() {
    let sm = SourceMap::new("hello\nworld", "test.py");
    let pos = sm.resolve(6);
    assert_eq!(pos, SourcePosition { line: 2, column: 0 });
}

#[test]
fn test_resolve_second_line_mid() {
    let sm = SourceMap::new("hello\nworld", "test.py");
    let pos = sm.resolve(8);
    assert_eq!(pos, SourcePosition { line: 2, column: 2 });
}

#[test]
fn test_resolve_third_line() {
    let sm = SourceMap::new("a\nbb\nccc", "test.py");
    let pos = sm.resolve(5); // 'c' at index 5
    assert_eq!(pos, SourcePosition { line: 3, column: 0 });
}

#[test]
fn test_resolve_end_of_file() {
    let sm = SourceMap::new("abc", "test.py");
    let pos = sm.resolve(3);
    assert_eq!(pos, SourcePosition { line: 1, column: 3 });
}

// =========================================================================
// Line Text Tests
// =========================================================================

#[test]
fn test_line_text_out_of_bounds() {
    let sm = SourceMap::new("a\nb", "test.py");
    assert_eq!(sm.line_text(0), None);
    assert_eq!(sm.line_text(3), None);
    assert_eq!(sm.line_text(100), None);
}

#[test]
fn test_line_text_with_indentation() {
    let sm = SourceMap::new("def foo():\n    return 42", "test.py");
    assert_eq!(sm.line_text(2), Some("    return 42"));
}

// =========================================================================
// Error Rendering Tests
// =========================================================================

#[test]
fn test_render_source_error_single_char() {
    let sm = SourceMap::new("x = 1 / 0", "test.py");
    let span = Span::new(8, 9); // The '0'
    let output = render_source_error(&sm, &span, "ZeroDivisionError", "division by zero");
    assert!(output.contains("File \"test.py\", line 1"));
    assert!(output.contains("x = 1 / 0"));
    assert!(output.contains("^"));
    assert!(output.contains("ZeroDivisionError: division by zero"));
}

#[test]
fn test_render_source_error_multichar_span() {
    let sm = SourceMap::new("result = undefined_name", "test.py");
    let span = Span::new(9, 23); // 'undefined_name'
    let output = render_source_error(
        &sm,
        &span,
        "NameError",
        "name 'undefined_name' is not defined",
    );
    assert!(output.contains("File \"test.py\", line 1"));
    assert!(output.contains("result = undefined_name"));
    assert!(output.contains("~"));
    assert!(output.contains("^"));
    assert!(output.contains("NameError"));
}

#[test]
fn test_render_source_error_second_line() {
    let sm = SourceMap::new("x = 1\ny = bad", "test.py");
    let span = Span::new(10, 13); // 'bad' on line 2
    let output = render_source_error(&sm, &span, "NameError", "name 'bad' is not defined");
    assert!(output.contains("File \"test.py\", line 2"));
    assert!(output.contains("y = bad"));
}

#[test]
fn test_render_simple_error() {
    let output = render_simple_error("TypeError", "unsupported operand");
    assert_eq!(output, "TypeError: unsupported operand");
}

#[test]
fn test_render_traceback_header() {
    assert_eq!(
        render_traceback_header(),
        "Traceback (most recent call last):"
    );
}

// =========================================================================
// SourcePosition Tests
// =========================================================================

// =========================================================================
// Edge Case Tests
// =========================================================================

#[test]
fn test_render_error_zero_length_span() {
    let sm = SourceMap::new("x = 1", "test.py");
    let span = Span::new(2, 2);
    let output = render_source_error(&sm, &span, "SyntaxError", "invalid syntax");
    assert!(output.contains("^"));
    assert!(output.contains("SyntaxError: invalid syntax"));
}

#[test]
fn test_long_line_with_deep_column() {
    let source = format!("{}{}", " ".repeat(40), "error_here");
    let sm = SourceMap::new(&source, "test.py");
    let span = Span::new(40, 50);
    let output = render_source_error(&sm, &span, "SyntaxError", "bad");
    assert!(output.contains("error_here"));
}

#[test]
fn test_source_map_many_lines() {
    let lines: Vec<String> = (0..1000).map(|i| format!("line_{}", i)).collect();
    let source = lines.join("\n");
    let sm = SourceMap::new(&source, "big.py");
    assert_eq!(sm.line_count(), 1000);
    assert_eq!(sm.line_text(500), Some("line_499"));
    assert_eq!(sm.line_text(1000), Some("line_999"));
}

#[test]
fn test_resolve_consistency_across_all_lines() {
    let sm = SourceMap::new("aa\nbb\ncc", "test.py");
    // Line 1: offsets 0,1 → line 1
    assert_eq!(sm.resolve(0).line, 1);
    assert_eq!(sm.resolve(1).line, 1);
    // Line 2: offsets 3,4 → line 2
    assert_eq!(sm.resolve(3).line, 2);
    assert_eq!(sm.resolve(4).line, 2);
    // Line 3: offsets 6,7 → line 3
    assert_eq!(sm.resolve(6).line, 3);
    assert_eq!(sm.resolve(7).line, 3);
}
