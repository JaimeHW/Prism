//! Identifier and keyword handling for Python.
//!
//! Implements Python 3 identifier rules per PEP 3131:
//! - ASCII letters, digits, underscore
//! - Unicode letters and combining marks
//! - Keywords are distinguished from identifiers

use super::cursor::Cursor;
use crate::token::{Keyword, TokenKind};

/// Check if a character can start an identifier.
///
/// Python identifiers can start with:
/// - ASCII letters (a-z, A-Z)
/// - Underscore (_)
/// - Unicode letters (Lu, Ll, Lt, Lm, Lo, Nl categories)
#[inline]
#[must_use]
pub fn is_id_start(c: char) -> bool {
    // Fast path for common ASCII
    if c.is_ascii_alphabetic() || c == '_' {
        return true;
    }

    // Unicode identifier support per PEP 3131
    unicode_xid::UnicodeXID::is_xid_start(c)
}

/// Check if a character can continue an identifier.
///
/// In addition to start characters, identifiers can contain:
/// - ASCII digits (0-9)
/// - Unicode combining marks and connectors
#[inline]
#[must_use]
pub fn is_id_continue(c: char) -> bool {
    // Fast path for common ASCII
    if c.is_ascii_alphanumeric() || c == '_' {
        return true;
    }

    // Unicode identifier support per PEP 3131
    unicode_xid::UnicodeXID::is_xid_continue(c)
}

/// Parse an identifier or keyword.
///
/// The cursor should be positioned just after the first character has been consumed.
/// Returns the identifier text along with whether it's a keyword.
pub fn parse_identifier(cursor: &mut Cursor<'_>, first_char: char) -> TokenKind {
    // Calculate start position (before the first char we already consumed)
    let start = cursor.pos() - first_char.len_utf8();

    // Collect remaining identifier characters
    cursor.eat_while(is_id_continue);

    // Get the full identifier text
    let text = cursor.slice_from(start);

    // Check if it's a keyword
    if let Some(kw) = Keyword::from_str(text) {
        TokenKind::Keyword(kw)
    } else {
        TokenKind::Ident(text.to_string())
    }
}

/// Soft keywords are context-sensitive keywords introduced in Python 3.10+.
/// They are only treated as keywords in specific syntactic contexts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoftKeyword {
    /// `match` in match statements
    Match,
    /// `case` in match statements  
    Case,
    /// `type` in type alias statements (Python 3.12+)
    Type,
    /// `_` as a pattern wildcard
    Underscore,
}

impl SoftKeyword {
    /// Check if a string is a soft keyword.
    #[must_use]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "match" => Some(Self::Match),
            "case" => Some(Self::Case),
            "type" => Some(Self::Type),
            "_" => Some(Self::Underscore),
            _ => None,
        }
    }

    /// Get the string representation.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Match => "match",
            Self::Case => "case",
            Self::Type => "type",
            Self::Underscore => "_",
        }
    }
}

/// Reserved identifiers that have special meaning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReservedIdent {
    /// `__name__`
    DunderName,
    /// `__doc__`
    DunderDoc,
    /// `__file__`
    DunderFile,
    /// `__package__`
    DunderPackage,
    /// `__spec__`
    DunderSpec,
    /// `__annotations__`
    DunderAnnotations,
    /// `__dict__`
    DunderDict,
    /// `__module__`
    DunderModule,
    /// `__slots__`
    DunderSlots,
    /// `__init__`
    DunderInit,
    /// `__new__`
    DunderNew,
    /// `__del__`
    DunderDel,
    /// `__repr__`
    DunderRepr,
    /// `__str__`
    DunderStr,
    /// `__bool__`
    DunderBool,
    /// `__len__`
    DunderLen,
    /// `__iter__`
    DunderIter,
    /// `__next__`
    DunderNext,
    /// `__getitem__`
    DunderGetitem,
    /// `__setitem__`
    DunderSetitem,
    /// `__call__`
    DunderCall,
}

impl ReservedIdent {
    /// Check if a string is a reserved identifier.
    #[must_use]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "__name__" => Some(Self::DunderName),
            "__doc__" => Some(Self::DunderDoc),
            "__file__" => Some(Self::DunderFile),
            "__package__" => Some(Self::DunderPackage),
            "__spec__" => Some(Self::DunderSpec),
            "__annotations__" => Some(Self::DunderAnnotations),
            "__dict__" => Some(Self::DunderDict),
            "__module__" => Some(Self::DunderModule),
            "__slots__" => Some(Self::DunderSlots),
            "__init__" => Some(Self::DunderInit),
            "__new__" => Some(Self::DunderNew),
            "__del__" => Some(Self::DunderDel),
            "__repr__" => Some(Self::DunderRepr),
            "__str__" => Some(Self::DunderStr),
            "__bool__" => Some(Self::DunderBool),
            "__len__" => Some(Self::DunderLen),
            "__iter__" => Some(Self::DunderIter),
            "__next__" => Some(Self::DunderNext),
            "__getitem__" => Some(Self::DunderGetitem),
            "__setitem__" => Some(Self::DunderSetitem),
            "__call__" => Some(Self::DunderCall),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_id_start_ascii() {
        assert!(is_id_start('a'));
        assert!(is_id_start('z'));
        assert!(is_id_start('A'));
        assert!(is_id_start('Z'));
        assert!(is_id_start('_'));
        assert!(!is_id_start('0'));
        assert!(!is_id_start('9'));
        assert!(!is_id_start(' '));
    }

    #[test]
    fn test_is_id_start_unicode() {
        assert!(is_id_start('α')); // Greek alpha
        assert!(is_id_start('日')); // CJK
        assert!(is_id_start('π')); // Greek pi
        assert!(!is_id_start('①')); // Circled digit one
    }

    #[test]
    fn test_is_id_continue_ascii() {
        assert!(is_id_continue('a'));
        assert!(is_id_continue('0'));
        assert!(is_id_continue('9'));
        assert!(is_id_continue('_'));
        assert!(!is_id_continue(' '));
        assert!(!is_id_continue('-'));
    }

    #[test]
    fn test_is_id_continue_unicode() {
        assert!(is_id_continue('α'));
        assert!(is_id_continue('文')); // CJK
        // Combining marks
        assert!(is_id_continue('\u{0300}')); // Combining grave accent
    }

    #[test]
    fn test_parse_identifier_simple() {
        let mut cursor = Cursor::new("hello world");
        let first = cursor.bump().unwrap();
        let result = parse_identifier(&mut cursor, first);
        assert_eq!(result, TokenKind::Ident("hello".to_string()));
    }

    #[test]
    fn test_parse_identifier_underscore() {
        let mut cursor = Cursor::new("_private_var = 5");
        let first = cursor.bump().unwrap();
        let result = parse_identifier(&mut cursor, first);
        assert_eq!(result, TokenKind::Ident("_private_var".to_string()));
    }

    #[test]
    fn test_parse_identifier_unicode() {
        let mut cursor = Cursor::new("αβγ = 3.14");
        let first = cursor.bump().unwrap();
        let result = parse_identifier(&mut cursor, first);
        assert_eq!(result, TokenKind::Ident("αβγ".to_string()));
    }

    #[test]
    fn test_parse_keyword_if() {
        let mut cursor = Cursor::new("if x:");
        let first = cursor.bump().unwrap();
        let result = parse_identifier(&mut cursor, first);
        assert_eq!(result, TokenKind::Keyword(Keyword::If));
    }

    #[test]
    fn test_parse_keyword_def() {
        let mut cursor = Cursor::new("def func():");
        let first = cursor.bump().unwrap();
        let result = parse_identifier(&mut cursor, first);
        assert_eq!(result, TokenKind::Keyword(Keyword::Def));
    }

    #[test]
    fn test_parse_keyword_class() {
        let mut cursor = Cursor::new("class Foo:");
        let first = cursor.bump().unwrap();
        let result = parse_identifier(&mut cursor, first);
        assert_eq!(result, TokenKind::Keyword(Keyword::Class));
    }

    #[test]
    fn test_parse_true_false_none() {
        let mut cursor1 = Cursor::new("True");
        let first1 = cursor1.bump().unwrap();
        assert_eq!(
            parse_identifier(&mut cursor1, first1),
            TokenKind::Keyword(Keyword::True)
        );

        let mut cursor2 = Cursor::new("False");
        let first2 = cursor2.bump().unwrap();
        assert_eq!(
            parse_identifier(&mut cursor2, first2),
            TokenKind::Keyword(Keyword::False)
        );

        let mut cursor3 = Cursor::new("None");
        let first3 = cursor3.bump().unwrap();
        assert_eq!(
            parse_identifier(&mut cursor3, first3),
            TokenKind::Keyword(Keyword::None)
        );
    }

    #[test]
    fn test_soft_keyword() {
        assert_eq!(SoftKeyword::from_str("match"), Some(SoftKeyword::Match));
        assert_eq!(SoftKeyword::from_str("case"), Some(SoftKeyword::Case));
        assert_eq!(SoftKeyword::from_str("type"), Some(SoftKeyword::Type));
        assert_eq!(SoftKeyword::from_str("_"), Some(SoftKeyword::Underscore));
        assert_eq!(SoftKeyword::from_str("other"), None);
    }

    #[test]
    fn test_reserved_ident() {
        assert_eq!(
            ReservedIdent::from_str("__init__"),
            Some(ReservedIdent::DunderInit)
        );
        assert_eq!(
            ReservedIdent::from_str("__name__"),
            Some(ReservedIdent::DunderName)
        );
        assert_eq!(ReservedIdent::from_str("regular"), None);
    }

    #[test]
    fn test_identifier_not_keyword() {
        // "iffy" starts with "if" but is not a keyword
        let mut cursor = Cursor::new("iffy");
        let first = cursor.bump().unwrap();
        let result = parse_identifier(&mut cursor, first);
        assert_eq!(result, TokenKind::Ident("iffy".to_string()));
    }

    #[test]
    fn test_identifier_with_numbers() {
        let mut cursor = Cursor::new("var123");
        let first = cursor.bump().unwrap();
        let result = parse_identifier(&mut cursor, first);
        assert_eq!(result, TokenKind::Ident("var123".to_string()));
    }

    #[test]
    fn test_dunder_name() {
        let mut cursor = Cursor::new("__init__");
        let first = cursor.bump().unwrap();
        let result = parse_identifier(&mut cursor, first);
        assert_eq!(result, TokenKind::Ident("__init__".to_string()));
    }
}
