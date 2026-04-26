//! Expression parser using Pratt parsing.
//!
//! Implements operator precedence parsing for all Python 3.12 expressions.

use crate::ast::{
    Arguments, BinOp, BoolOp, CmpOp, Comprehension, Expr, ExprKind, Keyword, UnaryOp,
};
use crate::parser::{Parser, Precedence};
use crate::token::{Keyword as KW, TokenKind};
use prism_core::PrismResult;

/// Expression parser.
pub struct ExprParser;

impl ExprParser {
    /// Parse an expression with the given minimum precedence.
    pub fn parse(parser: &mut Parser<'_>, min_prec: Precedence) -> PrismResult<Expr> {
        // Parse prefix expression
        let mut left = Self::parse_prefix(parser)?;

        // Parse infix expressions while precedence allows
        while let Some(prec) = Self::infix_precedence(parser) {
            if prec < min_prec {
                break;
            }
            left = Self::parse_infix(parser, left, prec)?;
        }

        Ok(left)
    }

    /// Parse a prefix expression (literals, unary ops, atoms).
    fn parse_prefix(parser: &mut Parser<'_>) -> PrismResult<Expr> {
        let start = parser.start_span();
        let kind = parser.current().kind.clone();
        match kind {
            // === Literals ===
            TokenKind::Int(n) => {
                parser.advance();
                Ok(Expr::new(ExprKind::Int(n), parser.span_from(start)))
            }
            TokenKind::BigInt(s) => {
                parser.advance();
                Ok(Expr::new(ExprKind::BigInt(s), parser.span_from(start)))
            }
            TokenKind::Float(n) => {
                parser.advance();
                Ok(Expr::new(ExprKind::Float(n), parser.span_from(start)))
            }
            TokenKind::Complex(n) => {
                parser.advance();
                Ok(Expr::new(
                    ExprKind::Complex { real: 0.0, imag: n },
                    parser.span_from(start),
                ))
            }
            TokenKind::String(_) | TokenKind::FString(_) => parser.parse_concatenated_string_expr(),
            TokenKind::Bytes(_) => {
                let literal = parser.parse_concatenated_bytes_literal()?;
                Ok(Expr::new(ExprKind::Bytes(literal), parser.span_from(start)))
            }
            TokenKind::Keyword(KW::True) => {
                parser.advance();
                Ok(Expr::new(ExprKind::Bool(true), parser.span_from(start)))
            }
            TokenKind::Keyword(KW::False) => {
                parser.advance();
                Ok(Expr::new(ExprKind::Bool(false), parser.span_from(start)))
            }
            TokenKind::Keyword(KW::None) => {
                parser.advance();
                Ok(Expr::new(ExprKind::None, parser.span_from(start)))
            }
            TokenKind::Ellipsis => {
                parser.advance();
                Ok(Expr::new(ExprKind::Ellipsis, parser.span_from(start)))
            }

            // === Identifiers ===
            TokenKind::Ident(name) => {
                parser.advance();
                Ok(Expr::new(ExprKind::Name(name), parser.span_from(start)))
            }

            // === Unary Operators ===
            TokenKind::Minus => {
                parser.advance();
                let operand = Self::parse(parser, Precedence::Unary)?;
                Ok(Expr::new(
                    ExprKind::UnaryOp {
                        op: UnaryOp::USub,
                        operand: Box::new(operand),
                    },
                    parser.span_from(start),
                ))
            }
            TokenKind::Plus => {
                parser.advance();
                let operand = Self::parse(parser, Precedence::Unary)?;
                Ok(Expr::new(
                    ExprKind::UnaryOp {
                        op: UnaryOp::UAdd,
                        operand: Box::new(operand),
                    },
                    parser.span_from(start),
                ))
            }
            TokenKind::Tilde => {
                parser.advance();
                let operand = Self::parse(parser, Precedence::Unary)?;
                Ok(Expr::new(
                    ExprKind::UnaryOp {
                        op: UnaryOp::Invert,
                        operand: Box::new(operand),
                    },
                    parser.span_from(start),
                ))
            }
            TokenKind::Keyword(KW::Not) => {
                parser.advance();
                let operand = Self::parse(parser, Precedence::Not)?;
                Ok(Expr::new(
                    ExprKind::UnaryOp {
                        op: UnaryOp::Not,
                        operand: Box::new(operand),
                    },
                    parser.span_from(start),
                ))
            }

            // === Await ===
            TokenKind::Keyword(KW::Await) => {
                parser.advance();
                let operand = Self::parse(parser, Precedence::Await)?;
                Ok(Expr::new(
                    ExprKind::Await(Box::new(operand)),
                    parser.span_from(start),
                ))
            }

            // === Lambda ===
            TokenKind::Keyword(KW::Lambda) => Self::parse_lambda(parser, start),

            // === Grouped/Tuple/Generator ===
            TokenKind::LeftParen => Self::parse_paren_expr(parser, start),

            // === List/ListComp ===
            TokenKind::LeftBracket => Self::parse_list_expr(parser, start),

            // === Dict/Set/Comprehensions ===
            TokenKind::LeftBrace => Self::parse_brace_expr(parser, start),

            // === Starred ===
            TokenKind::Star => {
                parser.advance();
                let operand = Self::parse(parser, Precedence::Primary)?;
                Ok(Expr::new(
                    ExprKind::Starred(Box::new(operand)),
                    parser.span_from(start),
                ))
            }

            // === Yield ===
            TokenKind::Keyword(KW::Yield) => Self::parse_yield(parser, start),

            _ => Err(parser.error_at_current("expected expression")),
        }
    }

    /// Parse an infix expression.
    fn parse_infix(parser: &mut Parser<'_>, left: Expr, _prec: Precedence) -> PrismResult<Expr> {
        let start = left.span.start;

        match &parser.current().kind.clone() {
            // === Binary Operators ===
            TokenKind::Plus => {
                parser.advance();
                let right = Self::parse(parser, Precedence::Additive.next())?;
                Ok(Expr::new(
                    ExprKind::BinOp {
                        left: Box::new(left),
                        op: BinOp::Add,
                        right: Box::new(right),
                    },
                    parser.span_from(start),
                ))
            }
            TokenKind::Minus => {
                parser.advance();
                let right = Self::parse(parser, Precedence::Additive.next())?;
                Ok(Expr::new(
                    ExprKind::BinOp {
                        left: Box::new(left),
                        op: BinOp::Sub,
                        right: Box::new(right),
                    },
                    parser.span_from(start),
                ))
            }
            TokenKind::Star => {
                parser.advance();
                let right = Self::parse(parser, Precedence::Multiplicative.next())?;
                Ok(Expr::new(
                    ExprKind::BinOp {
                        left: Box::new(left),
                        op: BinOp::Mult,
                        right: Box::new(right),
                    },
                    parser.span_from(start),
                ))
            }
            TokenKind::At => {
                parser.advance();
                let right = Self::parse(parser, Precedence::Multiplicative.next())?;
                Ok(Expr::new(
                    ExprKind::BinOp {
                        left: Box::new(left),
                        op: BinOp::MatMult,
                        right: Box::new(right),
                    },
                    parser.span_from(start),
                ))
            }
            TokenKind::Slash => {
                parser.advance();
                let right = Self::parse(parser, Precedence::Multiplicative.next())?;
                Ok(Expr::new(
                    ExprKind::BinOp {
                        left: Box::new(left),
                        op: BinOp::Div,
                        right: Box::new(right),
                    },
                    parser.span_from(start),
                ))
            }
            TokenKind::DoubleSlash => {
                parser.advance();
                let right = Self::parse(parser, Precedence::Multiplicative.next())?;
                Ok(Expr::new(
                    ExprKind::BinOp {
                        left: Box::new(left),
                        op: BinOp::FloorDiv,
                        right: Box::new(right),
                    },
                    parser.span_from(start),
                ))
            }
            TokenKind::Percent => {
                parser.advance();
                let right = Self::parse(parser, Precedence::Multiplicative.next())?;
                Ok(Expr::new(
                    ExprKind::BinOp {
                        left: Box::new(left),
                        op: BinOp::Mod,
                        right: Box::new(right),
                    },
                    parser.span_from(start),
                ))
            }
            TokenKind::DoubleStar => {
                // Power is right-associative
                parser.advance();
                let right = Self::parse(parser, Precedence::Power)?;
                Ok(Expr::new(
                    ExprKind::BinOp {
                        left: Box::new(left),
                        op: BinOp::Pow,
                        right: Box::new(right),
                    },
                    parser.span_from(start),
                ))
            }
            TokenKind::LeftShift => {
                parser.advance();
                let right = Self::parse(parser, Precedence::Shift.next())?;
                Ok(Expr::new(
                    ExprKind::BinOp {
                        left: Box::new(left),
                        op: BinOp::LShift,
                        right: Box::new(right),
                    },
                    parser.span_from(start),
                ))
            }
            TokenKind::RightShift => {
                parser.advance();
                let right = Self::parse(parser, Precedence::Shift.next())?;
                Ok(Expr::new(
                    ExprKind::BinOp {
                        left: Box::new(left),
                        op: BinOp::RShift,
                        right: Box::new(right),
                    },
                    parser.span_from(start),
                ))
            }
            TokenKind::Pipe => {
                parser.advance();
                let right = Self::parse(parser, Precedence::BitwiseOr.next())?;
                Ok(Expr::new(
                    ExprKind::BinOp {
                        left: Box::new(left),
                        op: BinOp::BitOr,
                        right: Box::new(right),
                    },
                    parser.span_from(start),
                ))
            }
            TokenKind::Caret => {
                parser.advance();
                let right = Self::parse(parser, Precedence::BitwiseXor.next())?;
                Ok(Expr::new(
                    ExprKind::BinOp {
                        left: Box::new(left),
                        op: BinOp::BitXor,
                        right: Box::new(right),
                    },
                    parser.span_from(start),
                ))
            }
            TokenKind::Ampersand => {
                parser.advance();
                let right = Self::parse(parser, Precedence::BitwiseAnd.next())?;
                Ok(Expr::new(
                    ExprKind::BinOp {
                        left: Box::new(left),
                        op: BinOp::BitAnd,
                        right: Box::new(right),
                    },
                    parser.span_from(start),
                ))
            }

            // === Boolean Operators ===
            TokenKind::Keyword(KW::And) => {
                parser.advance();
                let right = Self::parse(parser, Precedence::And.next())?;
                Ok(Expr::new(
                    ExprKind::BoolOp {
                        op: BoolOp::And,
                        values: vec![left, right],
                    },
                    parser.span_from(start),
                ))
            }
            TokenKind::Keyword(KW::Or) => {
                parser.advance();
                let right = Self::parse(parser, Precedence::Or.next())?;
                Ok(Expr::new(
                    ExprKind::BoolOp {
                        op: BoolOp::Or,
                        values: vec![left, right],
                    },
                    parser.span_from(start),
                ))
            }

            // === Comparison Operators ===
            TokenKind::Less
            | TokenKind::Greater
            | TokenKind::LessEqual
            | TokenKind::GreaterEqual
            | TokenKind::EqualEqual
            | TokenKind::NotEqual
            | TokenKind::Keyword(KW::Is)
            | TokenKind::Keyword(KW::In)
            | TokenKind::Keyword(KW::Not) => Self::parse_comparison(parser, left, start),

            // === Conditional ===
            TokenKind::Keyword(KW::If) => {
                parser.advance();
                let test = Self::parse(parser, Precedence::Or)?;
                parser.expect_keyword(KW::Else, "expected 'else' in conditional")?;
                let orelse = Self::parse(parser, Precedence::Conditional)?;
                Ok(Expr::new(
                    ExprKind::IfExp {
                        test: Box::new(test),
                        body: Box::new(left),
                        orelse: Box::new(orelse),
                    },
                    parser.span_from(start),
                ))
            }

            // === Named Expression (Walrus) ===
            TokenKind::ColonEqual => {
                parser.advance();
                let value = Self::parse(parser, Precedence::NamedExpr)?;
                Ok(Expr::new(
                    ExprKind::NamedExpr {
                        target: Box::new(left),
                        value: Box::new(value),
                    },
                    parser.span_from(start),
                ))
            }

            // === Attribute Access ===
            TokenKind::Dot => {
                parser.advance();
                let attr = parser.expect_identifier("expected attribute name")?;
                Ok(Expr::new(
                    ExprKind::Attribute {
                        value: Box::new(left),
                        attr,
                    },
                    parser.span_from(start),
                ))
            }

            // === Subscript ===
            TokenKind::LeftBracket => Self::parse_subscript(parser, left, start),

            // === Call ===
            TokenKind::LeftParen => Self::parse_call(parser, left, start),

            _ => Err(parser.error_at_current("unexpected token in expression")),
        }
    }

    /// Get the precedence of the current token as an infix operator.
    fn infix_precedence(parser: &Parser<'_>) -> Option<Precedence> {
        Some(match &parser.current().kind {
            // Comparison
            TokenKind::Less
            | TokenKind::Greater
            | TokenKind::LessEqual
            | TokenKind::GreaterEqual
            | TokenKind::EqualEqual
            | TokenKind::NotEqual => Precedence::Comparison,

            TokenKind::Keyword(kw) => match kw {
                KW::Is | KW::In => Precedence::Comparison,
                KW::Not => {
                    // `not in` is comparison, but `not` alone is prefix
                    // We check for `not in` pattern
                    Precedence::Comparison
                }
                KW::And => Precedence::And,
                KW::Or => Precedence::Or,
                KW::If => Precedence::Conditional,
                _ => return None,
            },

            // Bitwise
            TokenKind::Pipe => Precedence::BitwiseOr,
            TokenKind::Caret => Precedence::BitwiseXor,
            TokenKind::Ampersand => Precedence::BitwiseAnd,

            // Shift
            TokenKind::LeftShift | TokenKind::RightShift => Precedence::Shift,

            // Additive
            TokenKind::Plus | TokenKind::Minus => Precedence::Additive,

            // Multiplicative
            TokenKind::Star
            | TokenKind::At
            | TokenKind::Slash
            | TokenKind::DoubleSlash
            | TokenKind::Percent => Precedence::Multiplicative,

            // Power
            TokenKind::DoubleStar => Precedence::Power,

            // Named expression
            TokenKind::ColonEqual => Precedence::NamedExpr,

            // Primary (call, subscript, attribute)
            TokenKind::LeftParen | TokenKind::LeftBracket | TokenKind::Dot => Precedence::Primary,

            _ => return None,
        })
    }

    // =========================================================================
    // Complex Expression Parsers
    // =========================================================================

    /// Parse a comparison chain.
    fn parse_comparison(parser: &mut Parser<'_>, left: Expr, start: u32) -> PrismResult<Expr> {
        let mut ops = Vec::new();
        let mut comparators = Vec::new();

        loop {
            let op = match &parser.current().kind {
                TokenKind::Less => CmpOp::Lt,
                TokenKind::Greater => CmpOp::Gt,
                TokenKind::LessEqual => CmpOp::LtE,
                TokenKind::GreaterEqual => CmpOp::GtE,
                TokenKind::EqualEqual => CmpOp::Eq,
                TokenKind::NotEqual => CmpOp::NotEq,
                TokenKind::Keyword(KW::Is) => {
                    parser.advance();
                    if parser.match_keyword(KW::Not) {
                        CmpOp::IsNot
                    } else {
                        CmpOp::Is
                    }
                }
                TokenKind::Keyword(KW::In) => CmpOp::In,
                TokenKind::Keyword(KW::Not) => {
                    parser.advance();
                    parser.expect_keyword(KW::In, "expected 'in' after 'not'")?;
                    CmpOp::NotIn
                }
                _ => break,
            };

            // Advance past the operator (if not already advanced for is/not)
            if !matches!(
                &parser.previous().kind,
                TokenKind::Keyword(KW::Is)
                    | TokenKind::Keyword(KW::Not)
                    | TokenKind::Keyword(KW::In)
            ) {
                parser.advance();
            }

            ops.push(op);
            let right = Self::parse(parser, Precedence::Comparison.next())?;
            comparators.push(right);
        }

        Ok(Expr::new(
            ExprKind::Compare {
                left: Box::new(left),
                ops,
                comparators,
            },
            parser.span_from(start),
        ))
    }

    /// Parse a parenthesized expression, tuple, or generator.
    fn parse_paren_expr(parser: &mut Parser<'_>, start: u32) -> PrismResult<Expr> {
        parser.advance(); // consume '('

        // Empty tuple
        if parser.match_token(TokenKind::RightParen) {
            return Ok(Expr::new(ExprKind::Tuple(vec![]), parser.span_from(start)));
        }

        // Parse first element
        let first = Self::parse(parser, Precedence::Lowest)?;

        // Check for generator expression (supports both 'for' and 'async for')
        if parser.check_keyword(KW::For) || parser.check_keyword(KW::Async) {
            let generators = Self::parse_comprehension_clauses(parser)?;
            parser.expect(TokenKind::RightParen, "expected ')'")?;
            return Ok(Expr::new(
                ExprKind::GeneratorExp {
                    elt: Box::new(first),
                    generators,
                },
                parser.span_from(start),
            ));
        }

        // Check for tuple
        if parser.match_token(TokenKind::Comma) {
            let mut elements = vec![first];
            if !parser.check(TokenKind::RightParen) {
                loop {
                    elements.push(Self::parse(parser, Precedence::Lowest)?);
                    if !parser.match_token(TokenKind::Comma) {
                        break;
                    }
                    if parser.check(TokenKind::RightParen) {
                        break;
                    }
                }
            }
            parser.expect(TokenKind::RightParen, "expected ')'")?;
            return Ok(Expr::new(
                ExprKind::Tuple(elements),
                parser.span_from(start),
            ));
        }

        // Just a parenthesized expression
        parser.expect(TokenKind::RightParen, "expected ')'")?;
        Ok(first)
    }

    /// Parse a list or list comprehension.
    fn parse_list_expr(parser: &mut Parser<'_>, start: u32) -> PrismResult<Expr> {
        parser.advance(); // consume '['

        // Empty list
        if parser.match_token(TokenKind::RightBracket) {
            return Ok(Expr::new(ExprKind::List(vec![]), parser.span_from(start)));
        }

        // Parse first element
        let first = Self::parse(parser, Precedence::Lowest)?;

        // Check for comprehension (supports both 'for' and 'async for')
        if parser.check_keyword(KW::For) || parser.check_keyword(KW::Async) {
            let generators = Self::parse_comprehension_clauses(parser)?;
            parser.expect(TokenKind::RightBracket, "expected ']'")?;
            return Ok(Expr::new(
                ExprKind::ListComp {
                    elt: Box::new(first),
                    generators,
                },
                parser.span_from(start),
            ));
        }

        // Regular list
        let mut elements = vec![first];
        while parser.match_token(TokenKind::Comma) {
            if parser.check(TokenKind::RightBracket) {
                break;
            }
            elements.push(Self::parse(parser, Precedence::Lowest)?);
        }
        parser.expect(TokenKind::RightBracket, "expected ']'")?;
        Ok(Expr::new(ExprKind::List(elements), parser.span_from(start)))
    }

    /// Parse a dict, set, or comprehension.
    fn parse_brace_expr(parser: &mut Parser<'_>, start: u32) -> PrismResult<Expr> {
        parser.advance(); // consume '{'

        // Empty dict
        if parser.match_token(TokenKind::RightBrace) {
            return Ok(Expr::new(
                ExprKind::Dict {
                    keys: vec![],
                    values: vec![],
                },
                parser.span_from(start),
            ));
        }

        // Check for **expr (dict unpacking)
        let first_is_splat = parser.match_token(TokenKind::DoubleStar);
        let first = Self::parse(parser, Precedence::Lowest)?;

        // Is this a dict (has colon)?
        if !first_is_splat && parser.match_token(TokenKind::Colon) {
            let value = Self::parse(parser, Precedence::Lowest)?;

            // Check for dict comprehension (supports both 'for' and 'async for')
            if parser.check_keyword(KW::For) || parser.check_keyword(KW::Async) {
                let generators = Self::parse_comprehension_clauses(parser)?;
                parser.expect(TokenKind::RightBrace, "expected '}'")?;
                return Ok(Expr::new(
                    ExprKind::DictComp {
                        key: Box::new(first),
                        value: Box::new(value),
                        generators,
                    },
                    parser.span_from(start),
                ));
            }

            // Regular dict
            let mut keys = vec![Some(first)];
            let mut values = vec![value];

            while parser.match_token(TokenKind::Comma) {
                if parser.check(TokenKind::RightBrace) {
                    break;
                }
                if parser.match_token(TokenKind::DoubleStar) {
                    keys.push(None);
                    values.push(Self::parse(parser, Precedence::Lowest)?);
                } else {
                    let key = Self::parse(parser, Precedence::Lowest)?;
                    parser.expect(TokenKind::Colon, "expected ':' in dict")?;
                    let val = Self::parse(parser, Precedence::Lowest)?;
                    keys.push(Some(key));
                    values.push(val);
                }
            }

            parser.expect(TokenKind::RightBrace, "expected '}'")?;
            return Ok(Expr::new(
                ExprKind::Dict { keys, values },
                parser.span_from(start),
            ));
        }

        // Dict with splat first
        if first_is_splat {
            let mut keys = vec![None];
            let mut values = vec![first];

            while parser.match_token(TokenKind::Comma) {
                if parser.check(TokenKind::RightBrace) {
                    break;
                }
                if parser.match_token(TokenKind::DoubleStar) {
                    keys.push(None);
                    values.push(Self::parse(parser, Precedence::Lowest)?);
                } else {
                    let key = Self::parse(parser, Precedence::Lowest)?;
                    parser.expect(TokenKind::Colon, "expected ':' in dict")?;
                    let val = Self::parse(parser, Precedence::Lowest)?;
                    keys.push(Some(key));
                    values.push(val);
                }
            }

            parser.expect(TokenKind::RightBrace, "expected '}'")?;
            return Ok(Expr::new(
                ExprKind::Dict { keys, values },
                parser.span_from(start),
            ));
        }

        // It's a set - check for set comprehension (supports both 'for' and 'async for')
        if parser.check_keyword(KW::For) || parser.check_keyword(KW::Async) {
            let generators = Self::parse_comprehension_clauses(parser)?;
            parser.expect(TokenKind::RightBrace, "expected '}'")?;
            return Ok(Expr::new(
                ExprKind::SetComp {
                    elt: Box::new(first),
                    generators,
                },
                parser.span_from(start),
            ));
        }

        // Regular set
        let mut elements = vec![first];
        while parser.match_token(TokenKind::Comma) {
            if parser.check(TokenKind::RightBrace) {
                break;
            }
            elements.push(Self::parse(parser, Precedence::Lowest)?);
        }
        parser.expect(TokenKind::RightBrace, "expected '}'")?;
        Ok(Expr::new(ExprKind::Set(elements), parser.span_from(start)))
    }

    /// Parse comprehension clauses (for ... if ...).
    fn parse_comprehension_clauses(parser: &mut Parser<'_>) -> PrismResult<Vec<Comprehension>> {
        let mut generators = Vec::new();

        while parser.match_keyword(KW::For) || parser.match_keyword(KW::Async) {
            let is_async = parser.previous().kind == TokenKind::Keyword(KW::Async);
            if is_async {
                parser.expect_keyword(KW::For, "expected 'for' after 'async'")?;
            }

            // Parse target - need to handle tuple targets like `k, v`
            // First parse at BitwiseOr to get the first element
            let target_start = parser.start_span();
            let first = Self::parse(parser, Precedence::BitwiseOr)?;

            // Check for comma (tuple target like `for k, v in items`)
            let target = if parser.match_token(TokenKind::Comma) {
                let mut elements = vec![first];
                // Parse remaining comma-separated targets until 'in' keyword
                while !parser.check_keyword(KW::In) {
                    elements.push(Self::parse(parser, Precedence::BitwiseOr)?);
                    if !parser.match_token(TokenKind::Comma) {
                        break;
                    }
                }
                // Create tuple from elements
                Expr::new(ExprKind::Tuple(elements), parser.span_from(target_start))
            } else {
                first
            };

            parser.expect_keyword(KW::In, "expected 'in'")?;
            let iter = Self::parse(parser, Precedence::Or)?;

            let mut ifs = Vec::new();
            while parser.match_keyword(KW::If) {
                ifs.push(Self::parse(parser, Precedence::Or)?);
            }

            generators.push(Comprehension {
                target,
                iter,
                ifs,
                is_async,
            });
        }

        Ok(generators)
    }

    /// Parse a subscript expression.
    fn parse_subscript(parser: &mut Parser<'_>, value: Expr, start: u32) -> PrismResult<Expr> {
        parser.advance(); // consume '['

        let slice = Self::parse_slice_or_index(parser)?;
        parser.expect(TokenKind::RightBracket, "expected ']'")?;

        Ok(Expr::new(
            ExprKind::Subscript {
                value: Box::new(value),
                slice: Box::new(slice),
            },
            parser.span_from(start),
        ))
    }

    /// Parse a slice or index expression.
    fn parse_slice_or_index(parser: &mut Parser<'_>) -> PrismResult<Expr> {
        let first = Self::parse_subscript_item(parser)?;

        if !parser.match_token(TokenKind::Comma) {
            return Ok(first);
        }

        let start = first.span.start;
        let mut items = vec![first];

        loop {
            if parser.check(TokenKind::RightBracket) {
                break;
            }

            items.push(Self::parse_subscript_item(parser)?);

            if !parser.match_token(TokenKind::Comma) {
                break;
            }
        }

        Ok(Expr::new(ExprKind::Tuple(items), parser.span_from(start)))
    }

    /// Parse a single subscript item, which can be either an index expression
    /// or a slice expression.
    fn parse_subscript_item(parser: &mut Parser<'_>) -> PrismResult<Expr> {
        let start = parser.start_span();

        // Check if this looks like a slice
        let lower = if parser.check(TokenKind::Colon) {
            None
        } else {
            Some(Box::new(Self::parse(parser, Precedence::Lowest)?))
        };

        // Not a slice, just an index
        if !parser.match_token(TokenKind::Colon) {
            return Ok(*lower.unwrap());
        }

        // Parse upper bound
        let upper = if parser.check(TokenKind::Colon)
            || parser.check(TokenKind::RightBracket)
            || parser.check(TokenKind::Comma)
        {
            None
        } else {
            Some(Box::new(Self::parse(parser, Precedence::Lowest)?))
        };

        // Parse step
        let step = if parser.match_token(TokenKind::Colon) {
            if parser.check(TokenKind::RightBracket) || parser.check(TokenKind::Comma) {
                None
            } else {
                Some(Box::new(Self::parse(parser, Precedence::Lowest)?))
            }
        } else {
            None
        };

        Ok(Expr::new(
            ExprKind::Slice { lower, upper, step },
            parser.span_from(start),
        ))
    }

    /// Parse a function call.
    fn parse_call(parser: &mut Parser<'_>, func: Expr, start: u32) -> PrismResult<Expr> {
        parser.advance(); // consume '('

        let mut args = Vec::new();
        let mut keywords = Vec::new();

        if !parser.check(TokenKind::RightParen) {
            loop {
                let arg_start = parser.start_span();

                // Check for **kwargs
                if parser.match_token(TokenKind::DoubleStar) {
                    let value = Self::parse(parser, Precedence::Lowest)?;
                    keywords.push(Keyword {
                        arg: None,
                        value,
                        span: parser.span_from(arg_start),
                    });
                }
                // Check for *args
                else if parser.match_token(TokenKind::Star) {
                    let value = Self::parse(parser, Precedence::Lowest)?;
                    args.push(Expr::new(
                        ExprKind::Starred(Box::new(value)),
                        parser.span_from(arg_start),
                    ));
                }
                // Regular argument or keyword argument
                else {
                    let expr = Self::parse(parser, Precedence::Lowest)?;

                    // Check for generator expression as sole argument: sum(x for x in items)
                    // This is a special Python syntax where parentheses around the genexp are optional
                    // when the genexp is the only argument to the function call.
                    if args.is_empty()
                        && keywords.is_empty()
                        && (parser.check_keyword(KW::For) || parser.check_keyword(KW::Async))
                    {
                        let generators = Self::parse_comprehension_clauses(parser)?;
                        parser.expect(TokenKind::RightParen, "expected ')'")?;

                        let genexp = Expr::new(
                            ExprKind::GeneratorExp {
                                elt: Box::new(expr),
                                generators,
                            },
                            parser.span_from(arg_start),
                        );

                        return Ok(Expr::new(
                            ExprKind::Call {
                                func: Box::new(func),
                                args: vec![genexp],
                                keywords: vec![],
                            },
                            parser.span_from(start),
                        ));
                    }

                    // Check if this is a keyword argument
                    if parser.match_token(TokenKind::Equal) {
                        if let ExprKind::Name(name) = expr.kind {
                            let value = Self::parse(parser, Precedence::Lowest)?;
                            keywords.push(Keyword {
                                arg: Some(name),
                                value,
                                span: parser.span_from(arg_start),
                            });
                        } else {
                            return Err(parser
                                .error_at_previous("keyword argument name must be identifier"));
                        }
                    } else {
                        args.push(expr);
                    }
                }

                if !parser.match_token(TokenKind::Comma) {
                    break;
                }
                if parser.check(TokenKind::RightParen) {
                    break;
                }
            }
        }

        parser.expect(TokenKind::RightParen, "expected ')'")?;

        Ok(Expr::new(
            ExprKind::Call {
                func: Box::new(func),
                args,
                keywords,
            },
            parser.span_from(start),
        ))
    }

    /// Parse a lambda expression.
    fn parse_lambda(parser: &mut Parser<'_>, start: u32) -> PrismResult<Expr> {
        parser.advance(); // consume 'lambda'

        // Parse parameters
        let args = Self::parse_lambda_params(parser)?;

        parser.expect(TokenKind::Colon, "expected ':' in lambda")?;

        let body = Self::parse(parser, Precedence::Lowest)?;

        Ok(Expr::new(
            ExprKind::Lambda {
                args: Box::new(args),
                body: Box::new(body),
            },
            parser.span_from(start),
        ))
    }

    /// Parse lambda parameters (supports full Python syntax including *args, **kwargs).
    fn parse_lambda_params(parser: &mut Parser<'_>) -> PrismResult<Arguments> {
        let mut args = Arguments::default();

        if parser.check(TokenKind::Colon) {
            return Ok(args);
        }

        // Track state for proper parameter ordering
        let mut seen_star = false; // After bare * or *args
        let mut seen_kwargs = false; // After **kwargs

        loop {
            // Check for **kwargs
            if parser.match_token(TokenKind::DoubleStar) {
                if seen_kwargs {
                    return Err(parser.error_at_current("duplicate **kwargs"));
                }
                if let TokenKind::Ident(name) = &parser.current().kind {
                    let arg_start = parser.start_span();
                    let name = name.clone();
                    parser.advance();
                    args.kwarg = Some(crate::ast::Arg {
                        arg: name,
                        annotation: None,
                        span: parser.span_from(arg_start),
                    });
                    seen_kwargs = true;
                } else {
                    return Err(parser.error_at_current("expected identifier after **"));
                }
            }
            // Check for *args or bare *
            else if parser.match_token(TokenKind::Star) {
                if seen_star {
                    return Err(parser.error_at_current("duplicate *"));
                }
                if seen_kwargs {
                    return Err(parser.error_at_current("* cannot follow **kwargs"));
                }
                seen_star = true;

                // Check if it's *args or bare *
                if let TokenKind::Ident(name) = &parser.current().kind {
                    let arg_start = parser.start_span();
                    let name = name.clone();
                    parser.advance();
                    args.vararg = Some(crate::ast::Arg {
                        arg: name,
                        annotation: None,
                        span: parser.span_from(arg_start),
                    });
                }
                // else: bare * - just marks start of keyword-only params
            }
            // Check for positional-only marker /
            else if parser.match_token(TokenKind::Slash) {
                // Move all current args to posonlyargs
                args.posonlyargs = std::mem::take(&mut args.args);
                // Defaults stay as-is, they belong to posonlyargs now
            }
            // Regular parameter
            else if let TokenKind::Ident(name) = &parser.current().kind {
                let arg_start = parser.start_span();
                let name = name.clone();
                parser.advance();

                let arg = crate::ast::Arg {
                    arg: name,
                    annotation: None,
                    span: parser.span_from(arg_start),
                };

                // Check for default value
                if parser.match_token(TokenKind::Equal) {
                    let default = Self::parse(parser, Precedence::Lowest)?;
                    if seen_star {
                        args.kw_defaults.push(Some(default));
                        args.kwonlyargs.push(arg);
                    } else {
                        args.defaults.push(default);
                        args.args.push(arg);
                    }
                } else {
                    if seen_star {
                        args.kw_defaults.push(None);
                        args.kwonlyargs.push(arg);
                    } else {
                        args.args.push(arg);
                    }
                }
            } else {
                break;
            }

            // Continue if comma, stop at colon or end
            if !parser.match_token(TokenKind::Comma) {
                break;
            }
            if parser.check(TokenKind::Colon) {
                break;
            }
        }

        Ok(args)
    }

    /// Parse a yield expression.
    fn parse_yield(parser: &mut Parser<'_>, start: u32) -> PrismResult<Expr> {
        parser.advance(); // consume 'yield'

        if parser.match_keyword(KW::From) {
            let value = Self::parse(parser, Precedence::Lowest)?;
            Ok(Expr::new(
                ExprKind::YieldFrom(Box::new(value)),
                parser.span_from(start),
            ))
        } else if parser.check(TokenKind::Newline)
            || parser.check(TokenKind::RightParen)
            || parser.check(TokenKind::Eof)
        {
            Ok(Expr::new(ExprKind::Yield(None), parser.span_from(start)))
        } else {
            let value_start = parser.start_span();
            let first = Self::parse(parser, Precedence::Lowest)?;
            let value = parser.parse_comma_tuple_expr_until(value_start, first, |parser| {
                parser.check(TokenKind::RightParen)
                    || parser.check(TokenKind::RightBracket)
                    || parser.check(TokenKind::RightBrace)
                    || parser.check(TokenKind::Colon)
                    || parser.check(TokenKind::Semicolon)
            })?;
            Ok(Expr::new(
                ExprKind::Yield(Some(Box::new(value))),
                parser.span_from(start),
            ))
        }
    }
}

#[cfg(test)]
mod tests;
