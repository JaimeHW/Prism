//! Indentation tracking for Python's significant whitespace.
//!
//! Python uses INDENT and DEDENT tokens to represent block structure.
//! This module manages the indentation stack and generates appropriate tokens.

/// Tracks indentation levels and generates INDENT/DEDENT tokens.
#[derive(Debug, Clone)]
pub struct IndentStack {
    /// Stack of indentation levels (column numbers).
    stack: Vec<usize>,
    /// Pending DEDENT count to emit.
    pending_dedents: usize,
    /// Whether we're at the start of a logical line.
    at_line_start: bool,
    /// Current bracket/paren/brace nesting depth (disables indent tracking).
    bracket_depth: usize,
}

impl IndentStack {
    /// Create a new indentation tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            stack: vec![0], // Python always starts with indent level 0
            pending_dedents: 0,
            at_line_start: true,
            bracket_depth: 0,
        }
    }

    /// Check if we're at the start of a logical line.
    #[inline]
    #[must_use]
    pub const fn at_line_start(&self) -> bool {
        self.at_line_start
    }

    /// Mark that we've consumed content, no longer at line start.
    #[inline]
    pub fn consumed_content(&mut self) {
        self.at_line_start = false;
    }

    /// Mark the start of a new logical line.
    #[inline]
    pub fn new_line(&mut self) {
        self.at_line_start = true;
    }

    /// Get the current indentation level.
    #[inline]
    #[must_use]
    pub fn current_indent(&self) -> usize {
        *self.stack.last().unwrap_or(&0)
    }

    /// Check if there are pending DEDENT tokens to emit.
    #[inline]
    #[must_use]
    pub const fn has_pending_dedents(&self) -> bool {
        self.pending_dedents > 0
    }

    /// Consume one pending DEDENT.
    #[inline]
    pub fn consume_dedent(&mut self) -> bool {
        if self.pending_dedents > 0 {
            self.pending_dedents -= 1;
            true
        } else {
            false
        }
    }

    /// Check if indentation is currently being tracked.
    /// Indentation is ignored inside brackets/parens/braces.
    #[inline]
    #[must_use]
    pub const fn tracking_indent(&self) -> bool {
        self.bracket_depth == 0
    }

    /// Increment bracket nesting depth.
    #[inline]
    pub fn open_bracket(&mut self) {
        self.bracket_depth += 1;
    }

    /// Decrement bracket nesting depth.
    #[inline]
    pub fn close_bracket(&mut self) {
        self.bracket_depth = self.bracket_depth.saturating_sub(1);
    }

    /// Process indentation at the start of a line.
    ///
    /// Returns:
    /// - `Some(true)` if an INDENT token should be emitted
    /// - `Some(false)` if DEDENT tokens should be emitted (check `pending_dedents`)
    /// - `None` if indentation is unchanged or invalid
    ///
    /// # Errors
    /// Returns `Err` with a message if the indentation is inconsistent.
    pub fn process_indent(&mut self, indent: usize) -> Result<Option<bool>, &'static str> {
        // Don't track indent inside brackets
        if !self.tracking_indent() {
            return Ok(None);
        }

        let current = self.current_indent();

        if indent > current {
            // Indentation increased - emit INDENT
            self.stack.push(indent);
            Ok(Some(true))
        } else if indent < current {
            // Indentation decreased - emit DEDENT(s)
            let mut dedent_count = 0;
            while let Some(&level) = self.stack.last() {
                if level <= indent {
                    break;
                }
                self.stack.pop();
                dedent_count += 1;
            }

            // Verify we landed on a valid indentation level
            if self.current_indent() != indent {
                return Err("inconsistent dedent");
            }

            self.pending_dedents = dedent_count;
            Ok(Some(false))
        } else {
            // Same level - no token needed
            Ok(None)
        }
    }

    /// Generate DEDENT tokens to close all open blocks at end of file.
    #[must_use]
    pub fn close_all(&mut self) -> usize {
        // Don't count the base level 0
        let count = self.stack.len().saturating_sub(1);
        self.stack.truncate(1);
        self.pending_dedents = count;
        count
    }

    /// Get the number of open indentation levels (excluding base).
    #[must_use]
    pub fn depth(&self) -> usize {
        self.stack.len().saturating_sub(1)
    }
}

impl Default for IndentStack {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
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
}
