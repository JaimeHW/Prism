//! Source-mapped error diagnostics with CPython-style caret display.
//!
//! Translates `Span` byte offsets into line:column positions and renders
//! error messages with source context and caret underlines, matching
//! CPython 3.12's error output format.

use prism_core::span::Span;

// =============================================================================
// Source Map
// =============================================================================

/// Pre-computed line offset table for O(log n) span-to-position lookup.
///
/// Built once per source file; subsequent lookups are binary search over
/// the line start offsets.
#[derive(Debug, Clone)]
pub struct SourceMap {
    /// Byte offsets of each line start (always starts with 0).
    line_starts: Vec<usize>,
    /// The original source text.
    source: String,
    /// Filename for display.
    filename: String,
}

/// A resolved source position (1-indexed line, 0-indexed column).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourcePosition {
    /// 1-indexed line number.
    pub line: usize,
    /// 0-indexed column (byte offset from line start).
    pub column: usize,
}

impl SourceMap {
    /// Build a source map from source text and filename.
    ///
    /// Pre-computes line start offsets in a single pass — O(n) construction,
    /// O(log n) per lookup thereafter.
    pub fn new(source: &str, filename: &str) -> Self {
        let mut line_starts = vec![0usize];
        for (i, byte) in source.bytes().enumerate() {
            if byte == b'\n' {
                line_starts.push(i + 1);
            }
        }
        Self {
            line_starts,
            source: source.to_string(),
            filename: filename.to_string(),
        }
    }

    /// Resolve a byte offset to a source position.
    ///
    /// Uses binary search over pre-computed line starts — O(log n).
    #[inline]
    pub fn resolve(&self, offset: usize) -> SourcePosition {
        let line_idx = match self.line_starts.binary_search(&offset) {
            Ok(exact) => exact,
            Err(insert) => insert.saturating_sub(1),
        };
        let col = offset.saturating_sub(self.line_starts[line_idx]);
        SourcePosition {
            line: line_idx + 1, // 1-indexed
            column: col,
        }
    }

    /// Get the source text of a given line (1-indexed).
    ///
    /// Returns the line without trailing newline.
    pub fn line_text(&self, line: usize) -> Option<&str> {
        if line == 0 || line > self.line_starts.len() {
            return None;
        }
        let start = self.line_starts[line - 1];
        let end = if line < self.line_starts.len() {
            self.line_starts[line]
        } else {
            self.source.len()
        };
        // Trim trailing \n and \r\n.
        let text = &self.source[start..end];
        Some(text.trim_end_matches('\n').trim_end_matches('\r'))
    }

    /// Get the filename.
    #[inline]
    pub fn filename(&self) -> &str {
        &self.filename
    }

    /// Get the total number of lines.
    #[inline]
    pub fn line_count(&self) -> usize {
        self.line_starts.len()
    }
}

// =============================================================================
// Error Rendering
// =============================================================================

/// Render a CPython-style error diagnostic with source context and caret.
///
/// Output format (matching CPython 3.12):
/// ```text
///   File "test.py", line 3
///     x = 1 / 0
///         ~~^~~
/// ```
pub fn render_source_error(
    source_map: &SourceMap,
    span: &Span,
    error_type: &str,
    message: &str,
) -> String {
    let mut output = String::with_capacity(256);

    let pos = source_map.resolve(span.start as usize);
    let end_pos = source_map.resolve(span.end.saturating_sub(1).max(span.start) as usize);

    // File location line.
    output.push_str(&format!(
        "  File \"{}\", line {}\n",
        source_map.filename(),
        pos.line,
    ));

    // Source line with leading indent.
    if let Some(line_text) = source_map.line_text(pos.line) {
        output.push_str(&format!("    {}\n", line_text));

        // Caret underline.
        let caret_start = pos.column;
        let caret_end = if pos.line == end_pos.line {
            end_pos.column + 1
        } else {
            line_text.len()
        };
        let caret_len = caret_end.saturating_sub(caret_start).max(1);

        output.push_str("    ");
        for _ in 0..caret_start {
            output.push(' ');
        }
        if caret_len == 1 {
            output.push('^');
        } else {
            // CPython style: tildes with caret in center.
            let mid = caret_len / 2;
            for i in 0..caret_len {
                if i == mid {
                    output.push('^');
                } else {
                    output.push('~');
                }
            }
        }
        output.push('\n');
    }

    // Error type and message.
    output.push_str(&format!("{}: {}", error_type, message));

    output
}

/// Render a simple error (no source location) in CPython format.
pub fn render_simple_error(error_type: &str, message: &str) -> String {
    format!("{}: {}", error_type, message)
}

/// Render a traceback header matching CPython's format.
pub fn render_traceback_header() -> &'static str {
    "Traceback (most recent call last):"
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
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

    #[test]
    fn test_source_position_equality() {
        let a = SourcePosition { line: 1, column: 5 };
        let b = SourcePosition { line: 1, column: 5 };
        let c = SourcePosition { line: 2, column: 5 };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_source_position_copy() {
        let a = SourcePosition { line: 1, column: 5 };
        let b = a;
        assert_eq!(a, b);
    }

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
}
