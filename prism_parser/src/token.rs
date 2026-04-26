//! Token definitions for the Python lexer.

use prism_core::Span;
use std::fmt;

/// A token produced by the lexer.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    /// The kind of token.
    pub kind: TokenKind,
    /// Source span.
    pub span: Span,
}

impl Token {
    /// Create a new token.
    #[inline]
    #[must_use]
    pub const fn new(kind: TokenKind, span: Span) -> Self {
        Self { kind, span }
    }

    /// Check if this is an end-of-file token.
    #[inline]
    #[must_use]
    pub const fn is_eof(&self) -> bool {
        matches!(self.kind, TokenKind::Eof)
    }

    /// Check if this is a newline token.
    #[inline]
    #[must_use]
    pub const fn is_newline(&self) -> bool {
        matches!(self.kind, TokenKind::Newline)
    }
}

/// Token kinds for Python lexical analysis.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Literals
    /// Integer literal.
    Int(i64),
    /// Big integer literal (stored as string for precision).
    BigInt(String),
    /// Float literal.
    Float(f64),
    /// Complex literal (imaginary part).
    Complex(f64),
    /// String literal.
    String(String),
    /// Formatted string literal content.
    FString(String),
    /// Bytes literal.
    Bytes(Vec<u8>),
    /// F-string start.
    FStringStart,
    /// F-string middle (literal part).
    FStringMiddle(String),
    /// F-string end.
    FStringEnd,

    // Identifiers and Keywords
    /// Identifier.
    Ident(String),
    /// Keyword.
    Keyword(Keyword),

    // Operators
    /// `+`
    Plus,
    /// `-`
    Minus,
    /// `*`
    Star,
    /// `**`
    DoubleStar,
    /// `/`
    Slash,
    /// `//`
    DoubleSlash,
    /// `%`
    Percent,
    /// `@`
    At,
    /// `<<`
    LeftShift,
    /// `>>`
    RightShift,
    /// `&`
    Ampersand,
    /// `|`
    Pipe,
    /// `^`
    Caret,
    /// `~`
    Tilde,
    /// `:=`
    ColonEqual,
    /// `<`
    Less,
    /// `>`
    Greater,
    /// `<=`
    LessEqual,
    /// `>=`
    GreaterEqual,
    /// `==`
    EqualEqual,
    /// `!=`
    NotEqual,

    // Delimiters
    /// `(`
    LeftParen,
    /// `)`
    RightParen,
    /// `[`
    LeftBracket,
    /// `]`
    RightBracket,
    /// `{`
    LeftBrace,
    /// `}`
    RightBrace,
    /// `,`
    Comma,
    /// `:`
    Colon,
    /// `.`
    Dot,
    /// `;`
    Semicolon,
    /// `->`
    Arrow,
    /// `=`
    Equal,
    /// `+=`
    PlusEqual,
    /// `-=`
    MinusEqual,
    /// `*=`
    StarEqual,
    /// `/=`
    SlashEqual,
    /// `//=`
    DoubleSlashEqual,
    /// `%=`
    PercentEqual,
    /// `@=`
    AtEqual,
    /// `&=`
    AmpersandEqual,
    /// `|=`
    PipeEqual,
    /// `^=`
    CaretEqual,
    /// `>>=`
    RightShiftEqual,
    /// `<<=`
    LeftShiftEqual,
    /// `**=`
    DoubleStarEqual,
    /// `...`
    Ellipsis,

    // Indentation
    /// Indentation increase.
    Indent,
    /// Indentation decrease.
    Dedent,
    /// Newline.
    Newline,

    // Special
    /// End of file.
    Eof,
    /// Error token.
    Error(String),
}

impl TokenKind {
    /// Check if this is a comparison operator.
    #[must_use]
    pub const fn is_comparison(&self) -> bool {
        matches!(
            self,
            Self::Less
                | Self::Greater
                | Self::LessEqual
                | Self::GreaterEqual
                | Self::EqualEqual
                | Self::NotEqual
        )
    }

    /// Check if this is an augmented assignment operator.
    #[must_use]
    pub const fn is_augmented_assign(&self) -> bool {
        matches!(
            self,
            Self::PlusEqual
                | Self::MinusEqual
                | Self::StarEqual
                | Self::SlashEqual
                | Self::DoubleSlashEqual
                | Self::PercentEqual
                | Self::AtEqual
                | Self::AmpersandEqual
                | Self::PipeEqual
                | Self::CaretEqual
                | Self::RightShiftEqual
                | Self::LeftShiftEqual
                | Self::DoubleStarEqual
        )
    }
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(n) => write!(f, "{}", n),
            Self::BigInt(s) => write!(f, "{}", s),
            Self::Float(n) => write!(f, "{}", n),
            Self::Complex(n) => write!(f, "{}j", n),
            Self::String(s) => write!(f, "\"{}\"", s),
            Self::FString(s) => write!(f, "f\"{}\"", s),
            Self::Bytes(b) => write!(f, "b\"{}\"", String::from_utf8_lossy(b)),
            Self::FStringStart => write!(f, "f\""),
            Self::FStringMiddle(s) => write!(f, "{}", s),
            Self::FStringEnd => write!(f, "\""),
            Self::Ident(s) => write!(f, "{}", s),
            Self::Keyword(kw) => write!(f, "{}", kw),
            Self::Plus => write!(f, "+"),
            Self::Minus => write!(f, "-"),
            Self::Star => write!(f, "*"),
            Self::DoubleStar => write!(f, "**"),
            Self::Slash => write!(f, "/"),
            Self::DoubleSlash => write!(f, "//"),
            Self::Percent => write!(f, "%"),
            Self::At => write!(f, "@"),
            Self::LeftShift => write!(f, "<<"),
            Self::RightShift => write!(f, ">>"),
            Self::Ampersand => write!(f, "&"),
            Self::Pipe => write!(f, "|"),
            Self::Caret => write!(f, "^"),
            Self::Tilde => write!(f, "~"),
            Self::ColonEqual => write!(f, ":="),
            Self::Less => write!(f, "<"),
            Self::Greater => write!(f, ">"),
            Self::LessEqual => write!(f, "<="),
            Self::GreaterEqual => write!(f, ">="),
            Self::EqualEqual => write!(f, "=="),
            Self::NotEqual => write!(f, "!="),
            Self::LeftParen => write!(f, "("),
            Self::RightParen => write!(f, ")"),
            Self::LeftBracket => write!(f, "["),
            Self::RightBracket => write!(f, "]"),
            Self::LeftBrace => write!(f, "{{"),
            Self::RightBrace => write!(f, "}}"),
            Self::Comma => write!(f, ","),
            Self::Colon => write!(f, ":"),
            Self::Dot => write!(f, "."),
            Self::Semicolon => write!(f, ";"),
            Self::Arrow => write!(f, "->"),
            Self::Equal => write!(f, "="),
            Self::PlusEqual => write!(f, "+="),
            Self::MinusEqual => write!(f, "-="),
            Self::StarEqual => write!(f, "*="),
            Self::SlashEqual => write!(f, "/="),
            Self::DoubleSlashEqual => write!(f, "//="),
            Self::PercentEqual => write!(f, "%="),
            Self::AtEqual => write!(f, "@="),
            Self::AmpersandEqual => write!(f, "&="),
            Self::PipeEqual => write!(f, "|="),
            Self::CaretEqual => write!(f, "^="),
            Self::RightShiftEqual => write!(f, ">>="),
            Self::LeftShiftEqual => write!(f, "<<="),
            Self::DoubleStarEqual => write!(f, "**="),
            Self::Ellipsis => write!(f, "..."),
            Self::Indent => write!(f, "INDENT"),
            Self::Dedent => write!(f, "DEDENT"),
            Self::Newline => write!(f, "NEWLINE"),
            Self::Eof => write!(f, "EOF"),
            Self::Error(msg) => write!(f, "ERROR({})", msg),
        }
    }
}

/// Python keywords.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Keyword {
    /// `False`
    False,
    /// `None`
    None,
    /// `True`
    True,
    /// `and`
    And,
    /// `as`
    As,
    /// `assert`
    Assert,
    /// `async`
    Async,
    /// `await`
    Await,
    /// `break`
    Break,
    /// `class`
    Class,
    /// `continue`
    Continue,
    /// `def`
    Def,
    /// `del`
    Del,
    /// `elif`
    Elif,
    /// `else`
    Else,
    /// `except`
    Except,
    /// `finally`
    Finally,
    /// `for`
    For,
    /// `from`
    From,
    /// `global`
    Global,
    /// `if`
    If,
    /// `import`
    Import,
    /// `in`
    In,
    /// `is`
    Is,
    /// `lambda`
    Lambda,
    /// `nonlocal`
    Nonlocal,
    /// `not`
    Not,
    /// `or`
    Or,
    /// `pass`
    Pass,
    /// `raise`
    Raise,
    /// `return`
    Return,
    /// `try`
    Try,
    /// `while`
    While,
    /// `with`
    With,
    /// `yield`
    Yield,
}

impl Keyword {
    /// Try to parse a keyword from a string.
    #[must_use]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "False" => Some(Self::False),
            "None" => Some(Self::None),
            "True" => Some(Self::True),
            "and" => Some(Self::And),
            "as" => Some(Self::As),
            "assert" => Some(Self::Assert),
            "async" => Some(Self::Async),
            "await" => Some(Self::Await),
            "break" => Some(Self::Break),
            "class" => Some(Self::Class),
            "continue" => Some(Self::Continue),
            "def" => Some(Self::Def),
            "del" => Some(Self::Del),
            "elif" => Some(Self::Elif),
            "else" => Some(Self::Else),
            "except" => Some(Self::Except),
            "finally" => Some(Self::Finally),
            "for" => Some(Self::For),
            "from" => Some(Self::From),
            "global" => Some(Self::Global),
            "if" => Some(Self::If),
            "import" => Some(Self::Import),
            "in" => Some(Self::In),
            "is" => Some(Self::Is),
            "lambda" => Some(Self::Lambda),
            "nonlocal" => Some(Self::Nonlocal),
            "not" => Some(Self::Not),
            "or" => Some(Self::Or),
            "pass" => Some(Self::Pass),
            "raise" => Some(Self::Raise),
            "return" => Some(Self::Return),
            "try" => Some(Self::Try),
            "while" => Some(Self::While),
            "with" => Some(Self::With),
            "yield" => Some(Self::Yield),
            _ => None,
        }
    }

    /// Get the string representation.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::False => "False",
            Self::None => "None",
            Self::True => "True",
            Self::And => "and",
            Self::As => "as",
            Self::Assert => "assert",
            Self::Async => "async",
            Self::Await => "await",
            Self::Break => "break",
            Self::Class => "class",
            Self::Continue => "continue",
            Self::Def => "def",
            Self::Del => "del",
            Self::Elif => "elif",
            Self::Else => "else",
            Self::Except => "except",
            Self::Finally => "finally",
            Self::For => "for",
            Self::From => "from",
            Self::Global => "global",
            Self::If => "if",
            Self::Import => "import",
            Self::In => "in",
            Self::Is => "is",
            Self::Lambda => "lambda",
            Self::Nonlocal => "nonlocal",
            Self::Not => "not",
            Self::Or => "or",
            Self::Pass => "pass",
            Self::Raise => "raise",
            Self::Return => "return",
            Self::Try => "try",
            Self::While => "while",
            Self::With => "with",
            Self::Yield => "yield",
        }
    }
}

impl fmt::Display for Keyword {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
