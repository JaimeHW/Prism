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
