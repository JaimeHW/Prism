//! Low-level character cursor for lexer navigation.
//!
//! The cursor provides efficient character-by-character iteration over source
//! code with position tracking and lookahead capabilities.

use prism_core::Span;

/// End-of-file sentinel character.
pub const EOF_CHAR: char = '\0';

/// A cursor over source code that tracks position and provides lookahead.
#[derive(Debug, Clone)]
pub struct Cursor<'src> {
    /// The source code being lexed.
    source: &'src str,
    /// Remaining source to process (as bytes for efficiency).
    chars: std::str::Chars<'src>,
    /// Current byte position in source.
    pos: usize,
    /// Length of original source.
    len: usize,
}

impl<'src> Cursor<'src> {
    /// Create a new cursor over the given source.
    #[inline]
    #[must_use]
    pub fn new(source: &'src str) -> Self {
        Self {
            source,
            chars: source.chars(),
            pos: 0,
            len: source.len(),
        }
    }

    /// Get the current byte position.
    #[inline]
    #[must_use]
    pub const fn pos(&self) -> usize {
        self.pos
    }

    /// Check if we've reached the end of the source.
    #[inline]
    #[must_use]
    pub fn is_eof(&self) -> bool {
        self.pos >= self.len
    }

    /// Peek at the next character without consuming it.
    #[inline]
    #[must_use]
    pub fn first(&self) -> char {
        self.chars.clone().next().unwrap_or(EOF_CHAR)
    }

    /// Peek at the character after next without consuming.
    #[inline]
    #[must_use]
    pub fn second(&self) -> char {
        let mut chars = self.chars.clone();
        chars.next();
        chars.next().unwrap_or(EOF_CHAR)
    }

    /// Peek at the third character without consuming.
    #[inline]
    #[must_use]
    pub fn third(&self) -> char {
        let mut chars = self.chars.clone();
        chars.next();
        chars.next();
        chars.next().unwrap_or(EOF_CHAR)
    }

    /// Consume and return the next character.
    #[inline]
    pub fn bump(&mut self) -> Option<char> {
        let c = self.chars.next()?;
        self.pos += c.len_utf8();
        Some(c)
    }

    /// Consume the next character, returning EOF_CHAR if at end.
    #[inline]
    pub fn bump_or_eof(&mut self) -> char {
        self.bump().unwrap_or(EOF_CHAR)
    }

    /// Consume characters while the predicate returns true.
    #[inline]
    pub fn eat_while(&mut self, mut predicate: impl FnMut(char) -> bool) {
        while predicate(self.first()) && !self.is_eof() {
            self.bump();
        }
    }

    /// Consume a specific character if it matches.
    #[inline]
    pub fn eat(&mut self, c: char) -> bool {
        if self.first() == c {
            self.bump();
            true
        } else {
            false
        }
    }

    /// Get a slice of the source from start to current position.
    #[inline]
    #[must_use]
    pub fn slice_from(&self, start: usize) -> &'src str {
        &self.source[start..self.pos]
    }

    /// Create a span from start to current position.
    #[inline]
    #[must_use]
    pub fn span_from(&self, start: usize) -> Span {
        Span::new(start as u32, self.pos as u32)
    }

    /// Get the full source.
    #[inline]
    #[must_use]
    pub const fn source(&self) -> &'src str {
        self.source
    }

    /// Get remaining source as a string slice.
    #[inline]
    #[must_use]
    pub fn remaining(&self) -> &'src str {
        &self.source[self.pos..]
    }
}

#[cfg(test)]
mod tests;
