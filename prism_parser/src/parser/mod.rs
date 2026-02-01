//! Modular Python 3.12 parser.
//!
//! This module provides a complete recursive descent parser with Pratt parsing
//! for expressions. The parser produces a complete AST.

mod expr;
mod stmt;

use crate::ast::{Expr, Module, Stmt};
use crate::lexer::Lexer;
use crate::token::{Keyword, Token, TokenKind};
use prism_core::{PrismError, PrismResult, Span};

pub use expr::ExprParser;
pub use stmt::StmtParser;

// =============================================================================
// Parser Core
// =============================================================================

/// Python 3.12 parser.
pub struct Parser<'src> {
    /// Lexer for tokenization.
    lexer: Lexer<'src>,
    /// Current token.
    current: Token,
    /// Previous token (for span tracking).
    previous: Token,
    /// Whether we've hit an error.
    had_error: bool,
    /// Whether we're in panic mode (error recovery).
    panic_mode: bool,
}

impl<'src> Parser<'src> {
    /// Create a new parser for the given source code.
    pub fn new(source: &'src str) -> Self {
        let mut lexer = Lexer::new(source);
        let first_token = lexer.next_token();
        Self {
            lexer,
            current: first_token.clone(),
            previous: first_token,
            had_error: false,
            panic_mode: false,
        }
    }

    /// Parse a module (file).
    pub fn parse_module(&mut self) -> PrismResult<Module> {
        let start = self.current.span.start;
        let mut body = Vec::new();

        // Skip leading newlines
        self.skip_newlines();

        // Parse statements until EOF
        while !self.check(TokenKind::Eof) {
            match self.parse_statement() {
                Ok(stmt) => body.push(stmt),
                Err(e) => {
                    self.had_error = true;
                    self.synchronize();
                    if self.check(TokenKind::Eof) {
                        return Err(e);
                    }
                }
            }
            self.skip_newlines();
        }

        let end = self.current.span.end;
        Ok(Module::new(body, Span::new(start, end)))
    }

    /// Parse a single statement.
    pub fn parse_statement(&mut self) -> PrismResult<Stmt> {
        StmtParser::parse(self)
    }

    /// Parse an expression.
    pub fn parse_expression(&mut self) -> PrismResult<Expr> {
        ExprParser::parse(self, Precedence::Lowest)
    }

    /// Parse an expression with minimum precedence.
    pub fn parse_expression_with_precedence(&mut self, min_prec: Precedence) -> PrismResult<Expr> {
        ExprParser::parse(self, min_prec)
    }

    // =========================================================================
    // Token Management
    // =========================================================================

    /// Get the current token.
    #[inline]
    pub fn current(&self) -> &Token {
        &self.current
    }

    /// Get the previous token.
    #[inline]
    pub fn previous(&self) -> &Token {
        &self.previous
    }

    /// Advance to the next token, returning the previous.
    pub fn advance(&mut self) -> &Token {
        self.previous = std::mem::replace(&mut self.current, self.lexer.next_token());
        &self.previous
    }

    /// Check if the current token matches the given kind.
    #[inline]
    pub fn check(&self, kind: TokenKind) -> bool {
        std::mem::discriminant(&self.current.kind) == std::mem::discriminant(&kind)
    }

    /// Check if the current token is a specific keyword.
    #[inline]
    pub fn check_keyword(&self, kw: Keyword) -> bool {
        matches!(&self.current.kind, TokenKind::Keyword(k) if *k == kw)
    }

    /// Consume the current token if it matches, otherwise return false.
    pub fn match_token(&mut self, kind: TokenKind) -> bool {
        if self.check(kind) {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Consume the current token if it's the given keyword.
    pub fn match_keyword(&mut self, kw: Keyword) -> bool {
        if self.check_keyword(kw) {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Expect and consume a specific token, or error.
    pub fn expect(&mut self, kind: TokenKind, msg: &str) -> PrismResult<&Token> {
        if self.check(kind) {
            self.advance();
            Ok(&self.previous)
        } else {
            Err(self.error_at_current(msg))
        }
    }

    /// Expect and consume a specific keyword, or error.
    pub fn expect_keyword(&mut self, kw: Keyword, msg: &str) -> PrismResult<&Token> {
        if self.check_keyword(kw) {
            self.advance();
            Ok(&self.previous)
        } else {
            Err(self.error_at_current(msg))
        }
    }

    /// Expect and consume an identifier, returning the name.
    pub fn expect_identifier(&mut self, msg: &str) -> PrismResult<String> {
        if let TokenKind::Ident(name) = &self.current.kind {
            let name = name.clone();
            self.advance();
            Ok(name)
        } else {
            Err(self.error_at_current(msg))
        }
    }

    /// Skip any newline tokens.
    pub fn skip_newlines(&mut self) {
        while self.check(TokenKind::Newline) {
            self.advance();
        }
    }

    /// Check if at end of file.
    #[inline]
    pub fn is_at_end(&self) -> bool {
        self.check(TokenKind::Eof)
    }

    // =========================================================================
    // Span Tracking
    // =========================================================================

    /// Get a span from start to current position.
    pub fn span_from(&self, start: u32) -> Span {
        Span::new(start, self.previous.span.end)
    }

    /// Get the current position for span tracking.
    pub fn start_span(&self) -> u32 {
        self.current.span.start
    }

    // =========================================================================
    // Error Handling
    // =========================================================================

    /// Create an error at the current token.
    pub fn error_at_current(&mut self, msg: &str) -> PrismError {
        self.error_at(&self.current.clone(), msg)
    }

    /// Create an error at the previous token.
    pub fn error_at_previous(&self, msg: &str) -> PrismError {
        self.error_at(&self.previous, msg)
    }

    /// Create an error at a specific token.
    fn error_at(&self, token: &Token, msg: &str) -> PrismError {
        let location = match &token.kind {
            TokenKind::Eof => "at end of file".to_string(),
            TokenKind::Error(e) => format!("lexer error: {}", e),
            _ => format!("at '{}'", token.kind),
        };
        PrismError::syntax(format!("{}: {}", location, msg), token.span)
    }

    /// Synchronize after an error by skipping to the next statement.
    fn synchronize(&mut self) {
        self.panic_mode = false;

        while !self.check(TokenKind::Eof) {
            // Stop at newlines (end of statement)
            if self.previous.kind == TokenKind::Newline {
                return;
            }

            // Stop at statement-starting keywords
            if let TokenKind::Keyword(kw) = &self.current.kind {
                match kw {
                    Keyword::Class
                    | Keyword::Def
                    | Keyword::For
                    | Keyword::If
                    | Keyword::While
                    | Keyword::Return
                    | Keyword::Import
                    | Keyword::From
                    | Keyword::Try
                    | Keyword::With
                    | Keyword::Match => return,
                    _ => {}
                }
            }

            self.advance();
        }
    }
}

// =============================================================================
// Precedence Levels
// =============================================================================

/// Expression precedence levels for Pratt parsing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Precedence {
    /// Lowest precedence (lambda, assignment expressions).
    Lowest = 0,
    /// Named expression `:=`
    NamedExpr = 1,
    /// Conditional `if-else`
    Conditional = 2,
    /// `or`
    Or = 3,
    /// `and`
    And = 4,
    /// `not`
    Not = 5,
    /// Comparisons
    Comparison = 6,
    /// `|`
    BitwiseOr = 7,
    /// `^`
    BitwiseXor = 8,
    /// `&`
    BitwiseAnd = 9,
    /// `<<`, `>>`
    Shift = 10,
    /// `+`, `-`
    Additive = 11,
    /// `*`, `@`, `/`, `//`, `%`
    Multiplicative = 12,
    /// Unary `+`, `-`, `~`
    Unary = 13,
    /// `**`
    Power = 14,
    /// `await`
    Await = 15,
    /// Attribute, subscript, call
    Primary = 16,
}

impl Precedence {
    /// Get the next higher precedence level.
    #[must_use]
    pub fn next(self) -> Self {
        match self {
            Self::Lowest => Self::NamedExpr,
            Self::NamedExpr => Self::Conditional,
            Self::Conditional => Self::Or,
            Self::Or => Self::And,
            Self::And => Self::Not,
            Self::Not => Self::Comparison,
            Self::Comparison => Self::BitwiseOr,
            Self::BitwiseOr => Self::BitwiseXor,
            Self::BitwiseXor => Self::BitwiseAnd,
            Self::BitwiseAnd => Self::Shift,
            Self::Shift => Self::Additive,
            Self::Additive => Self::Multiplicative,
            Self::Multiplicative => Self::Unary,
            Self::Unary => Self::Power,
            Self::Power => Self::Await,
            Self::Await => Self::Primary,
            Self::Primary => Self::Primary,
        }
    }
}

// =============================================================================
// Public API
// =============================================================================

/// Parse source code into a module.
pub fn parse(source: &str) -> PrismResult<Module> {
    Parser::new(source).parse_module()
}

/// Parse a single expression.
pub fn parse_expression(source: &str) -> PrismResult<Expr> {
    Parser::new(source).parse_expression()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_creation() {
        let parser = Parser::new("x + 1");
        assert!(!parser.is_at_end());
    }

    #[test]
    fn test_empty_module() {
        let result = parse("");
        assert!(result.is_ok());
        let module = result.unwrap();
        assert!(module.body.is_empty());
    }

    #[test]
    fn test_precedence_ordering() {
        assert!(Precedence::Primary > Precedence::Lowest);
        assert!(Precedence::Multiplicative > Precedence::Additive);
        assert!(Precedence::Power > Precedence::Unary);
    }

    #[test]
    fn test_precedence_next() {
        assert_eq!(Precedence::Additive.next(), Precedence::Multiplicative);
        assert_eq!(Precedence::Primary.next(), Precedence::Primary);
    }
}
