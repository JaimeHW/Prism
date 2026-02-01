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
    /// `match` (Python 3.10+)
    Match,
    /// `case` (Python 3.10+)
    Case,
    /// `type` (Python 3.12+)
    Type,
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
            "match" => Some(Self::Match),
            "case" => Some(Self::Case),
            "type" => Some(Self::Type),
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
            Self::Match => "match",
            Self::Case => "case",
            Self::Type => "type",
        }
    }
}

impl fmt::Display for Keyword {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_creation() {
        let token = Token::new(TokenKind::Plus, Span::new(0, 1));
        assert_eq!(token.kind, TokenKind::Plus);
        assert_eq!(token.span.start, 0);
        assert_eq!(token.span.end, 1);
    }

    #[test]
    fn test_token_is_eof() {
        let eof = Token::new(TokenKind::Eof, Span::new(100, 100));
        let plus = Token::new(TokenKind::Plus, Span::new(0, 1));

        assert!(eof.is_eof());
        assert!(!plus.is_eof());
    }

    #[test]
    fn test_token_is_newline() {
        let newline = Token::new(TokenKind::Newline, Span::new(10, 11));
        let plus = Token::new(TokenKind::Plus, Span::new(0, 1));

        assert!(newline.is_newline());
        assert!(!plus.is_newline());
    }

    #[test]
    fn test_keyword_from_str() {
        assert_eq!(Keyword::from_str("if"), Some(Keyword::If));
        assert_eq!(Keyword::from_str("while"), Some(Keyword::While));
        assert_eq!(Keyword::from_str("True"), Some(Keyword::True));
        assert_eq!(Keyword::from_str("False"), Some(Keyword::False));
        assert_eq!(Keyword::from_str("None"), Some(Keyword::None));
        assert_eq!(Keyword::from_str("match"), Some(Keyword::Match));
        assert_eq!(Keyword::from_str("type"), Some(Keyword::Type));
        assert_eq!(Keyword::from_str("not_a_keyword"), None);
    }

    #[test]
    fn test_keyword_as_str() {
        assert_eq!(Keyword::If.as_str(), "if");
        assert_eq!(Keyword::While.as_str(), "while");
        assert_eq!(Keyword::True.as_str(), "True");
    }

    #[test]
    fn test_keyword_display() {
        assert_eq!(format!("{}", Keyword::Def), "def");
        assert_eq!(format!("{}", Keyword::Class), "class");
    }

    #[test]
    fn test_all_keywords_roundtrip() {
        let keywords = [
            Keyword::False,
            Keyword::None,
            Keyword::True,
            Keyword::And,
            Keyword::As,
            Keyword::Assert,
            Keyword::Async,
            Keyword::Await,
            Keyword::Break,
            Keyword::Class,
            Keyword::Continue,
            Keyword::Def,
            Keyword::Del,
            Keyword::Elif,
            Keyword::Else,
            Keyword::Except,
            Keyword::Finally,
            Keyword::For,
            Keyword::From,
            Keyword::Global,
            Keyword::If,
            Keyword::Import,
            Keyword::In,
            Keyword::Is,
            Keyword::Lambda,
            Keyword::Nonlocal,
            Keyword::Not,
            Keyword::Or,
            Keyword::Pass,
            Keyword::Raise,
            Keyword::Return,
            Keyword::Try,
            Keyword::While,
            Keyword::With,
            Keyword::Yield,
            Keyword::Match,
            Keyword::Case,
            Keyword::Type,
        ];

        for kw in keywords {
            let s = kw.as_str();
            let parsed = Keyword::from_str(s);
            assert_eq!(parsed, Some(kw), "Roundtrip failed for {:?}", kw);
        }
    }

    #[test]
    fn test_token_kind_is_comparison() {
        assert!(TokenKind::Less.is_comparison());
        assert!(TokenKind::Greater.is_comparison());
        assert!(TokenKind::LessEqual.is_comparison());
        assert!(TokenKind::GreaterEqual.is_comparison());
        assert!(TokenKind::EqualEqual.is_comparison());
        assert!(TokenKind::NotEqual.is_comparison());

        assert!(!TokenKind::Plus.is_comparison());
        assert!(!TokenKind::Equal.is_comparison());
    }

    #[test]
    fn test_token_kind_is_augmented_assign() {
        assert!(TokenKind::PlusEqual.is_augmented_assign());
        assert!(TokenKind::MinusEqual.is_augmented_assign());
        assert!(TokenKind::StarEqual.is_augmented_assign());
        assert!(TokenKind::DoubleStarEqual.is_augmented_assign());

        assert!(!TokenKind::Equal.is_augmented_assign());
        assert!(!TokenKind::Plus.is_augmented_assign());
    }

    #[test]
    fn test_token_kind_display() {
        assert_eq!(format!("{}", TokenKind::Plus), "+");
        assert_eq!(format!("{}", TokenKind::DoubleStar), "**");
        assert_eq!(format!("{}", TokenKind::ColonEqual), ":=");
        assert_eq!(format!("{}", TokenKind::Ellipsis), "...");
        assert_eq!(format!("{}", TokenKind::Arrow), "->");
    }

    #[test]
    fn test_token_kind_literals_display() {
        assert_eq!(format!("{}", TokenKind::Int(42)), "42");
        assert_eq!(format!("{}", TokenKind::Float(3.14)), "3.14");
        assert_eq!(
            format!("{}", TokenKind::String("hello".to_string())),
            "\"hello\""
        );
    }

    #[test]
    fn test_token_equality() {
        let t1 = Token::new(TokenKind::Plus, Span::new(0, 1));
        let t2 = Token::new(TokenKind::Plus, Span::new(0, 1));
        let t3 = Token::new(TokenKind::Plus, Span::new(1, 2));
        let t4 = Token::new(TokenKind::Minus, Span::new(0, 1));

        assert_eq!(t1, t2);
        assert_ne!(t1, t3); // Different span
        assert_ne!(t1, t4); // Different kind
    }

    #[test]
    fn test_keyword_equality() {
        assert_eq!(Keyword::If, Keyword::If);
        assert_ne!(Keyword::If, Keyword::Else);
    }

    #[test]
    fn test_keyword_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(Keyword::If);
        set.insert(Keyword::While);
        set.insert(Keyword::If); // Duplicate

        assert_eq!(set.len(), 2);
    }
}
