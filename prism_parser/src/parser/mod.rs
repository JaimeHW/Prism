//! Modular Python 3.12 parser.
//!
//! This module provides a complete recursive descent parser with Pratt parsing
//! for expressions. The parser produces a complete AST.

mod expr;
mod stmt;

use crate::ast::{Expr, ExprKind, Module, Stmt, StringLiteral};
use crate::lexer::Lexer;
use crate::token::{Keyword, Token, TokenKind};
use prism_core::{PrismError, PrismResult, Span};

pub use expr::ExprParser;
pub use stmt::StmtParser;

// =============================================================================
// Parser Core
// =============================================================================

/// Python 3.12 parser.
#[derive(Clone)]
pub struct Parser<'src> {
    /// Lexer for tokenization.
    lexer: Lexer<'src>,
    /// Current token.
    current: Token,
    /// Previous token (for span tracking).
    previous: Token,
    /// Whether we've hit an error.
    had_error: bool,
    /// First parse error encountered while recovering.
    first_error: Option<PrismError>,
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
            first_error: None,
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
            match StmtParser::parse_statement_sequence(self) {
                Ok(stmts) => body.extend(stmts),
                Err(e) => {
                    self.had_error = true;
                    if self.first_error.is_none() {
                        self.first_error = Some(e);
                    }
                    self.synchronize();
                }
            }
            self.skip_newlines();
        }

        if let Some(err) = self.first_error.take() {
            return Err(err);
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

    /// Parse an implicit comma tuple that continues from an already-parsed
    /// first expression.
    pub(crate) fn parse_comma_tuple_expr_until<F>(
        &mut self,
        start: u32,
        first: Expr,
        mut should_stop: F,
    ) -> PrismResult<Expr>
    where
        F: FnMut(&Parser<'src>) -> bool,
    {
        if !self.match_token(TokenKind::Comma) {
            return Ok(first);
        }

        let mut elements = vec![first];
        while !self.check(TokenKind::Newline) && !self.check(TokenKind::Eof) {
            if should_stop(self) {
                break;
            }

            elements.push(ExprParser::parse(self, Precedence::Lowest)?);
            if !self.match_token(TokenKind::Comma) {
                break;
            }
        }

        Ok(Expr::new(ExprKind::Tuple(elements), self.span_from(start)))
    }

    /// Parse an implicit comma tuple, optionally stopping before `=`.
    pub(crate) fn parse_comma_tuple_expr(
        &mut self,
        start: u32,
        first: Expr,
        stop_at_equal: bool,
    ) -> PrismResult<Expr> {
        self.parse_comma_tuple_expr_until(start, first, |parser| {
            stop_at_equal && parser.check(TokenKind::Equal)
        })
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

    /// Check if the current token is the given identifier text.
    #[inline]
    pub fn check_identifier_value(&self, value: &str) -> bool {
        matches!(&self.current.kind, TokenKind::Ident(name) if name == value)
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

    /// Consume the current token if it is the given identifier text.
    pub fn match_identifier_value(&mut self, value: &str) -> bool {
        if self.check_identifier_value(value) {
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

    /// Parse one or more adjacent plain/f-string literal tokens into an
    /// expression node.
    ///
    /// Ordinary adjacent strings collapse into a single `ExprKind::String`.
    /// If any formatted string participates in the group, the result is lowered
    /// to `ExprKind::JoinedStr` with compile-time concatenation of adjacent
    /// literal segments.
    pub(crate) fn parse_concatenated_string_expr(&mut self) -> PrismResult<Expr> {
        let start = self.start_span();
        let mut plain_literal = String::new();
        let mut joined_parts = Vec::new();
        let mut saw_fstring = false;

        loop {
            match self.current.kind.clone() {
                TokenKind::String(segment) => {
                    self.advance();
                    if saw_fstring {
                        append_joined_string_literal(&mut joined_parts, &segment);
                    } else {
                        plain_literal.push_str(&segment);
                    }
                }
                TokenKind::FString(content) => {
                    self.advance();
                    if !plain_literal.is_empty() {
                        append_joined_string_literal(&mut joined_parts, &plain_literal);
                        plain_literal.clear();
                    }
                    saw_fstring = true;
                    joined_parts.extend(parse_fstring_parts(&content)?);
                }
                TokenKind::Bytes(_) => {
                    return Err(self.error_at_current("cannot mix bytes and nonbytes literals"));
                }
                _ => break,
            }
        }

        if !saw_fstring {
            return Ok(Expr::new(
                ExprKind::String(StringLiteral::new(plain_literal)),
                self.span_from(start),
            ));
        }

        if !plain_literal.is_empty() {
            append_joined_string_literal(&mut joined_parts, &plain_literal);
        }
        if joined_parts.is_empty() {
            append_joined_string_literal(&mut joined_parts, "");
        }

        Ok(Expr::new(
            ExprKind::JoinedStr(joined_parts),
            self.span_from(start),
        ))
    }

    /// Parse one or more adjacent string literal tokens into a single literal.
    ///
    /// Python performs compile-time concatenation for adjacent string literals.
    /// This helper also rejects mixed string/bytes literal groups with the same
    /// syntax error CPython reports for that construct.
    pub(crate) fn parse_concatenated_string_literal(&mut self) -> PrismResult<StringLiteral> {
        let mut value = match self.current.kind.clone() {
            TokenKind::String(value) => value,
            _ => unreachable!("parse_concatenated_string_literal requires a string token"),
        };
        self.advance();

        loop {
            match self.current.kind.clone() {
                TokenKind::String(segment) => {
                    value.push_str(&segment);
                    self.advance();
                }
                TokenKind::Bytes(_) => {
                    return Err(self.error_at_current("cannot mix bytes and nonbytes literals"));
                }
                _ => break,
            }
        }

        Ok(StringLiteral::new(value))
    }

    /// Parse one or more adjacent bytes literal tokens into a single literal.
    ///
    /// Python performs compile-time concatenation for adjacent bytes literals.
    /// Mixed string/bytes groups are rejected as syntax errors.
    pub(crate) fn parse_concatenated_bytes_literal(&mut self) -> PrismResult<Vec<u8>> {
        let mut value = match self.current.kind.clone() {
            TokenKind::Bytes(value) => value,
            _ => unreachable!("parse_concatenated_bytes_literal requires a bytes token"),
        };
        self.advance();

        loop {
            match self.current.kind.clone() {
                TokenKind::Bytes(segment) => {
                    value.extend(segment);
                    self.advance();
                }
                TokenKind::String(_) => {
                    return Err(self.error_at_current("cannot mix bytes and nonbytes literals"));
                }
                _ => break,
            }
        }

        Ok(value)
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
            if self.previous.kind == TokenKind::Newline && self.current_can_start_statement() {
                return;
            }

            if self.current_can_start_compound_statement() {
                return;
            }

            self.advance();
        }
    }

    #[inline]
    fn current_can_start_statement(&self) -> bool {
        self.current_can_start_compound_statement()
            || matches!(
                &self.current.kind,
                TokenKind::Int(_)
                    | TokenKind::BigInt(_)
                    | TokenKind::Float(_)
                    | TokenKind::Complex(_)
                    | TokenKind::String(_)
                    | TokenKind::FString(_)
                    | TokenKind::Bytes(_)
                    | TokenKind::Ident(_)
                    | TokenKind::Minus
                    | TokenKind::Plus
                    | TokenKind::Tilde
                    | TokenKind::LeftParen
                    | TokenKind::LeftBracket
                    | TokenKind::LeftBrace
                    | TokenKind::Star
                    | TokenKind::Ellipsis
                    | TokenKind::At
                    | TokenKind::Keyword(
                        Keyword::Assert
                            | Keyword::Await
                            | Keyword::Break
                            | Keyword::Continue
                            | Keyword::Del
                            | Keyword::False
                            | Keyword::Global
                            | Keyword::Import
                            | Keyword::Lambda
                            | Keyword::None
                            | Keyword::Nonlocal
                            | Keyword::Not
                            | Keyword::Pass
                            | Keyword::Raise
                            | Keyword::Return
                            | Keyword::True
                            | Keyword::Yield
                    )
            )
    }

    #[inline]
    fn current_can_start_compound_statement(&self) -> bool {
        matches!(
            &self.current.kind,
            TokenKind::Keyword(
                Keyword::Class
                    | Keyword::Def
                    | Keyword::For
                    | Keyword::If
                    | Keyword::Try
                    | Keyword::While
                    | Keyword::With
                    | Keyword::Async
            )
        ) || self.check_identifier_value("match")
            || self.check_identifier_value("type")
    }
}

fn append_joined_string_literal(parts: &mut Vec<Expr>, segment: &str) {
    if let Some(Expr {
        kind: ExprKind::String(literal),
        ..
    }) = parts.last_mut()
    {
        literal.value.push_str(segment);
        return;
    }

    parts.push(Expr::new(
        ExprKind::String(StringLiteral::new(segment.to_string())),
        Span::dummy(),
    ));
}

fn parse_fstring_parts(content: &str) -> PrismResult<Vec<Expr>> {
    let mut parts = Vec::new();
    let mut literal = String::new();
    let mut index = 0usize;

    while index < content.len() {
        let ch = next_char(content, index)?;
        match ch {
            '{' => {
                if next_char_at(content, index + 1) == Some('{') {
                    literal.push('{');
                    index += 2;
                    continue;
                }

                if !literal.is_empty() {
                    append_joined_string_literal(&mut parts, &literal);
                    literal.clear();
                }

                let end = find_fstring_expression_end(content, index + 1)?;
                let field = &content[index + 1..end];
                parts.extend(parse_fstring_field_parts(field)?);
                index = end + 1;
            }
            '}' => {
                if next_char_at(content, index + 1) == Some('}') {
                    literal.push('}');
                    index += 2;
                    continue;
                }
                return Err(PrismError::syntax(
                    "single '}' is not allowed in f-string",
                    Span::dummy(),
                ));
            }
            _ => {
                literal.push(ch);
                index += ch.len_utf8();
            }
        }
    }

    if !literal.is_empty() {
        append_joined_string_literal(&mut parts, &literal);
    }

    Ok(parts)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SplitFStringField {
    expr_text: String,
    conversion: i8,
    format_spec: Option<String>,
    debug_literal: Option<String>,
}

fn parse_fstring_field_parts(field: &str) -> PrismResult<Vec<Expr>> {
    let SplitFStringField {
        expr_text,
        conversion,
        format_spec,
        debug_literal,
    } = split_fstring_field(field)?;
    let value = parse_embedded_fstring_expr(expr_text.trim())?;
    let format_spec = format_spec
        .map(parse_fstring_format_spec)
        .transpose()?
        .map(Box::new);

    let formatted = Expr::new(
        ExprKind::FormattedValue {
            value: Box::new(value),
            conversion,
            format_spec,
        },
        Span::dummy(),
    );

    if let Some(debug_literal) = debug_literal {
        let mut parts = Vec::with_capacity(2);
        append_joined_string_literal(&mut parts, &debug_literal);
        parts.push(formatted);
        Ok(parts)
    } else {
        Ok(vec![formatted])
    }
}

fn parse_fstring_format_spec(spec: String) -> PrismResult<Expr> {
    let mut parts = parse_fstring_parts(&spec)?;
    if parts.is_empty() {
        append_joined_string_literal(&mut parts, "");
    }
    Ok(Expr::new(ExprKind::JoinedStr(parts), Span::dummy()))
}

fn parse_embedded_fstring_expr(source: &str) -> PrismResult<Expr> {
    let mut parser = Parser::new(source);
    let expr = parser.parse_expression()?;
    parser.skip_newlines();
    if !parser.is_at_end() {
        return Err(parser.error_at_current("unexpected trailing tokens in f-string expression"));
    }
    Ok(expr)
}

fn split_fstring_field(field: &str) -> PrismResult<SplitFStringField> {
    let mut paren_depth = 0usize;
    let mut bracket_depth = 0usize;
    let mut brace_depth = 0usize;
    let mut conversion_index = None;
    let mut format_index = None;
    let mut index = 0usize;

    while index < field.len() {
        let ch = next_char(field, index)?;
        if ch == '\'' || ch == '"' {
            index = skip_python_string_literal(field, index)?;
            continue;
        }

        match ch {
            '(' => paren_depth += 1,
            ')' => paren_depth = paren_depth.saturating_sub(1),
            '[' => bracket_depth += 1,
            ']' => bracket_depth = bracket_depth.saturating_sub(1),
            '{' => brace_depth += 1,
            '}' => brace_depth = brace_depth.saturating_sub(1),
            '!' if paren_depth == 0 && bracket_depth == 0 && brace_depth == 0 => {
                conversion_index = Some(index);
                break;
            }
            ':' if paren_depth == 0 && bracket_depth == 0 && brace_depth == 0 => {
                format_index = Some(index);
                break;
            }
            _ => {}
        }

        index += ch.len_utf8();
    }

    let expr_boundary = conversion_index.or(format_index).unwrap_or(field.len());
    let debug_equals = find_fstring_debug_equals(field, expr_boundary)?;
    let expr_text_end = debug_equals.unwrap_or(expr_boundary);
    let debug_literal = debug_equals.map(|_| field[..expr_boundary].to_string());

    if let Some(conversion_start) = conversion_index {
        let expr_text = field[..expr_text_end].to_string();
        let conversion_offset = conversion_start + 1;
        let conversion = next_char(field, conversion_offset)?;
        let conversion_value = match conversion {
            's' | 'r' | 'a' => conversion as i8,
            _ => {
                return Err(PrismError::syntax(
                    format!("invalid conversion character '{conversion}' in f-string"),
                    Span::dummy(),
                ));
            }
        };

        let after_conversion = conversion_offset + conversion.len_utf8();
        let (format_spec, remainder_start) = if next_char_at(field, after_conversion) == Some(':') {
            (
                Some(field[after_conversion + 1..].to_string()),
                after_conversion,
            )
        } else {
            (None, after_conversion)
        };

        if after_conversion < field.len()
            && !matches!(next_char_at(field, after_conversion), Some(':'))
        {
            return Err(PrismError::syntax(
                "expected ':' after f-string conversion specifier",
                Span::dummy(),
            ));
        }
        let _ = remainder_start;

        return Ok(SplitFStringField {
            expr_text,
            conversion: conversion_value,
            format_spec,
            debug_literal,
        });
    }

    if format_index.is_none() {
        let mut paren_depth = 0usize;
        let mut bracket_depth = 0usize;
        let mut brace_depth = 0usize;
        let mut scan = 0usize;

        while scan < field.len() {
            let ch = next_char(field, scan)?;
            if ch == '\'' || ch == '"' {
                scan = skip_python_string_literal(field, scan)?;
                continue;
            }

            match ch {
                '(' => paren_depth += 1,
                ')' => paren_depth = paren_depth.saturating_sub(1),
                '[' => bracket_depth += 1,
                ']' => bracket_depth = bracket_depth.saturating_sub(1),
                '{' => brace_depth += 1,
                '}' => brace_depth = brace_depth.saturating_sub(1),
                ':' if paren_depth == 0 && bracket_depth == 0 && brace_depth == 0 => {
                    format_index = Some(scan);
                    break;
                }
                _ => {}
            }

            scan += ch.len_utf8();
        }
    }

    if let Some(format_start) = format_index {
        Ok(SplitFStringField {
            expr_text: field[..expr_text_end].to_string(),
            conversion: if debug_equals.is_some() {
                'r' as i8
            } else {
                -1
            },
            format_spec: Some(field[format_start + 1..].to_string()),
            debug_literal,
        })
    } else {
        Ok(SplitFStringField {
            expr_text: field[..expr_text_end].to_string(),
            conversion: if debug_equals.is_some() {
                'r' as i8
            } else {
                -1
            },
            format_spec: None,
            debug_literal,
        })
    }
}

fn find_fstring_debug_equals(field: &str, boundary: usize) -> PrismResult<Option<usize>> {
    let trimmed_boundary = field[..boundary]
        .trim_end_matches(char::is_whitespace)
        .len();
    if trimmed_boundary == 0 {
        return Ok(None);
    }

    let Some(eq_index) = previous_char_start(field, trimmed_boundary) else {
        return Ok(None);
    };
    if next_char(field, eq_index)? != '=' {
        return Ok(None);
    }
    if !is_standalone_fstring_equals(field, eq_index) {
        return Ok(None);
    }

    let mut paren_depth = 0usize;
    let mut bracket_depth = 0usize;
    let mut brace_depth = 0usize;
    let mut index = 0usize;

    while index < eq_index {
        let ch = next_char(field, index)?;
        if ch == '\'' || ch == '"' {
            index = skip_python_string_literal(field, index)?;
            continue;
        }

        match ch {
            '(' => paren_depth += 1,
            ')' => paren_depth = paren_depth.saturating_sub(1),
            '[' => bracket_depth += 1,
            ']' => bracket_depth = bracket_depth.saturating_sub(1),
            '{' => brace_depth += 1,
            '}' => brace_depth = brace_depth.saturating_sub(1),
            _ => {}
        }

        index += ch.len_utf8();
    }

    if paren_depth == 0 && bracket_depth == 0 && brace_depth == 0 {
        Ok(Some(eq_index))
    } else {
        Ok(None)
    }
}

fn previous_char_start(content: &str, end: usize) -> Option<usize> {
    content[..end].char_indices().last().map(|(index, _)| index)
}

fn is_standalone_fstring_equals(field: &str, index: usize) -> bool {
    let previous = field[..index].chars().next_back();
    let next = next_char_at(field, index + 1);

    !matches!(previous, Some('=' | '!' | '<' | '>' | ':')) && next != Some('=')
}

fn find_fstring_expression_end(content: &str, start: usize) -> PrismResult<usize> {
    let mut paren_depth = 0usize;
    let mut bracket_depth = 0usize;
    let mut brace_depth = 0usize;
    let mut index = start;

    while index < content.len() {
        let ch = next_char(content, index)?;
        if ch == '\'' || ch == '"' {
            index = skip_python_string_literal(content, index)?;
            continue;
        }

        match ch {
            '(' => paren_depth += 1,
            ')' => paren_depth = paren_depth.saturating_sub(1),
            '[' => bracket_depth += 1,
            ']' => bracket_depth = bracket_depth.saturating_sub(1),
            '{' => brace_depth += 1,
            '}' => {
                if paren_depth == 0 && bracket_depth == 0 && brace_depth == 0 {
                    return Ok(index);
                }
                brace_depth = brace_depth.saturating_sub(1);
            }
            _ => {}
        }

        index += ch.len_utf8();
    }

    Err(PrismError::syntax(
        "unterminated f-string expression",
        Span::dummy(),
    ))
}

fn skip_python_string_literal(content: &str, start: usize) -> PrismResult<usize> {
    let quote = next_char(content, start)?;
    let quote_byte = quote as u8;
    let bytes = content.as_bytes();
    let mut index = start + 1;
    let is_triple =
        bytes.get(index) == Some(&quote_byte) && bytes.get(index + 1) == Some(&quote_byte);

    if is_triple {
        index += 2;
    }

    while index < content.len() {
        let ch = next_char(content, index)?;
        if ch == '\\' {
            index += 1;
            if index >= content.len() {
                return Err(PrismError::syntax(
                    "unterminated escape sequence in f-string expression",
                    Span::dummy(),
                ));
            }
            let escaped = next_char(content, index)?;
            index += escaped.len_utf8();
            continue;
        }

        if ch == quote {
            if is_triple {
                if bytes.get(index + 1) == Some(&quote_byte)
                    && bytes.get(index + 2) == Some(&quote_byte)
                {
                    return Ok(index + 3);
                }
            } else {
                return Ok(index + 1);
            }
        }

        index += ch.len_utf8();
    }

    Err(PrismError::syntax(
        "unterminated string in f-string expression",
        Span::dummy(),
    ))
}

fn next_char(content: &str, index: usize) -> PrismResult<char> {
    content[index..]
        .chars()
        .next()
        .ok_or_else(|| PrismError::syntax("unexpected end of f-string content", Span::dummy()))
}

fn next_char_at(content: &str, index: usize) -> Option<char> {
    content.get(index..)?.chars().next()
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
