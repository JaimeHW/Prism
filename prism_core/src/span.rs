//! Source span tracking for error reporting.
//!
//! Spans represent byte offset ranges in source files, enabling precise
//! error messages and source mapping for debugging.

use std::fmt;
use std::ops::Range;

/// A span representing a byte offset range in source code.
///
/// Spans are half-open intervals: `[start, end)`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Span {
    /// Start byte offset (inclusive).
    pub start: u32,
    /// End byte offset (exclusive).
    pub end: u32,
}

impl Span {
    /// Create a new span from start to end.
    #[inline]
    #[must_use]
    pub const fn new(start: u32, end: u32) -> Self {
        Self { start, end }
    }

    /// Create an empty span at a position.
    #[inline]
    #[must_use]
    pub const fn empty(pos: u32) -> Self {
        Self {
            start: pos,
            end: pos,
        }
    }

    /// Create a span covering a single byte.
    #[inline]
    #[must_use]
    pub const fn single(pos: u32) -> Self {
        Self {
            start: pos,
            end: pos + 1,
        }
    }

    /// Create a dummy span for generated code.
    #[inline]
    #[must_use]
    pub const fn dummy() -> Self {
        Self { start: 0, end: 0 }
    }

    /// Check if this span is a dummy span.
    #[inline]
    #[must_use]
    pub const fn is_dummy(&self) -> bool {
        self.start == 0 && self.end == 0
    }

    /// Get the length of this span in bytes.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> u32 {
        self.end.saturating_sub(self.start)
    }

    /// Check if the span is empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.start >= self.end
    }

    /// Check if this span contains a byte offset.
    #[inline]
    #[must_use]
    pub const fn contains(&self, offset: u32) -> bool {
        offset >= self.start && offset < self.end
    }

    /// Check if this span fully contains another span.
    #[inline]
    #[must_use]
    pub const fn contains_span(&self, other: Span) -> bool {
        self.start <= other.start && self.end >= other.end
    }

    /// Check if this span overlaps with another.
    #[inline]
    #[must_use]
    pub const fn overlaps(&self, other: Span) -> bool {
        self.start < other.end && other.start < self.end
    }

    /// Merge two spans into one covering both.
    #[inline]
    #[must_use]
    pub const fn merge(self, other: Span) -> Span {
        let start = if self.start < other.start {
            self.start
        } else {
            other.start
        };
        let end = if self.end > other.end {
            self.end
        } else {
            other.end
        };
        Span { start, end }
    }

    /// Extend this span to include another.
    #[inline]
    pub fn extend(&mut self, other: Span) {
        if other.start < self.start {
            self.start = other.start;
        }
        if other.end > self.end {
            self.end = other.end;
        }
    }

    /// Shrink the span by removing bytes from start and end.
    #[inline]
    #[must_use]
    pub const fn shrink(self, start_bytes: u32, end_bytes: u32) -> Span {
        let new_start = self.start.saturating_add(start_bytes);
        let new_end = self.end.saturating_sub(end_bytes);
        if new_start >= new_end {
            Span::empty(self.start)
        } else {
            Span::new(new_start, new_end)
        }
    }

    /// Get the byte range as a `Range<usize>`.
    #[inline]
    #[must_use]
    pub const fn as_range(&self) -> Range<usize> {
        self.start as usize..self.end as usize
    }

    /// Get a subslice of source text covered by this span.
    #[inline]
    #[must_use]
    pub fn slice<'a>(&self, source: &'a str) -> &'a str {
        let start = self.start as usize;
        let end = self.end as usize;
        if start <= end && end <= source.len() {
            &source[start..end]
        } else {
            ""
        }
    }

    /// Get the (1-indexed) line and column from source.
    #[must_use]
    pub fn line_col(&self, source: &str) -> (usize, usize) {
        let offset = self.start as usize;
        if offset > source.len() {
            return (1, 1);
        }

        let prefix = &source[..offset];
        let line = prefix.chars().filter(|&c| c == '\n').count() + 1;
        let last_newline = prefix.rfind('\n').map_or(0, |pos| pos + 1);
        let col = offset - last_newline + 1;

        (line, col)
    }
}

impl fmt::Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}..{}", self.start, self.end)
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {})", self.start, self.end)
    }
}

impl From<Range<u32>> for Span {
    fn from(range: Range<u32>) -> Self {
        Self::new(range.start, range.end)
    }
}

impl From<Range<usize>> for Span {
    fn from(range: Range<usize>) -> Self {
        Self::new(range.start as u32, range.end as u32)
    }
}

impl From<Span> for Range<usize> {
    fn from(span: Span) -> Self {
        span.as_range()
    }
}

/// A value with an associated source span.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Spanned<T> {
    /// The value.
    pub value: T,
    /// The source span.
    pub span: Span,
}

impl<T> Spanned<T> {
    /// Create a new spanned value.
    #[inline]
    #[must_use]
    pub const fn new(value: T, span: Span) -> Self {
        Self { value, span }
    }

    /// Map the inner value with a function.
    #[inline]
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Spanned<U> {
        Spanned {
            value: f(self.value),
            span: self.span,
        }
    }

    /// Get a reference to the inner value.
    #[inline]
    #[must_use]
    pub const fn as_ref(&self) -> Spanned<&T> {
        Spanned {
            value: &self.value,
            span: self.span,
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for Spanned<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} @ {:?}", self.value, self.span)
    }
}

impl<T: fmt::Display> fmt::Display for Spanned<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.value, f)
    }
}
