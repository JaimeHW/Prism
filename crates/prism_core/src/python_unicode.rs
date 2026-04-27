//! Shared helpers for Python-specific Unicode edge cases.
//!
//! Rust `char` and `str` require Unicode scalar values, but Python source and
//! runtime semantics also expose surrogate code points in a few well-defined
//! places, such as `"\uDC80"` source escapes and filesystem
//! `surrogateescape`-style round-trips.
//!
//! Prism carries those surrogate code points through its UTF-8-native pipeline
//! by remapping them into an internal, valid-scalar carrier range. Callers that
//! need Python-visible semantics should translate carrier scalars back to the
//! original surrogate code point before exposing them.

/// First Unicode surrogate code point.
pub const PYTHON_SURROGATE_START: u32 = 0xD800;

/// Last Unicode surrogate code point.
pub const PYTHON_SURROGATE_END: u32 = 0xDFFF;

/// First internal carrier scalar used to store Python surrogate code points.
pub const PYTHON_SURROGATE_CARRIER_BASE: u32 = 0xF0000;

/// Last internal carrier scalar used to store Python surrogate code points.
pub const PYTHON_SURROGATE_CARRIER_END: u32 =
    PYTHON_SURROGATE_CARRIER_BASE + (PYTHON_SURROGATE_END - PYTHON_SURROGATE_START);

/// Returns whether `code_point` is a Python surrogate code point.
#[inline]
#[must_use]
pub const fn is_python_surrogate(code_point: u32) -> bool {
    code_point >= PYTHON_SURROGATE_START && code_point <= PYTHON_SURROGATE_END
}

/// Returns whether `code_point` is an internal surrogate carrier scalar.
#[inline]
#[must_use]
pub const fn is_surrogate_carrier(code_point: u32) -> bool {
    code_point >= PYTHON_SURROGATE_CARRIER_BASE && code_point <= PYTHON_SURROGATE_CARRIER_END
}

/// Maps a Python surrogate code point to its internal carrier scalar.
#[inline]
#[must_use]
pub const fn surrogate_to_carrier(code_point: u32) -> Option<u32> {
    if is_python_surrogate(code_point) {
        Some(PYTHON_SURROGATE_CARRIER_BASE + (code_point - PYTHON_SURROGATE_START))
    } else {
        None
    }
}

/// Maps an internal carrier scalar back to its Python surrogate code point.
#[inline]
#[must_use]
pub const fn carrier_to_surrogate(code_point: u32) -> Option<u32> {
    if is_surrogate_carrier(code_point) {
        Some(PYTHON_SURROGATE_START + (code_point - PYTHON_SURROGATE_CARRIER_BASE))
    } else {
        None
    }
}

/// Returns the Python-visible code point for a stored scalar value.
#[inline]
#[must_use]
pub const fn logical_python_code_point(code_point: u32) -> u32 {
    match carrier_to_surrogate(code_point) {
        Some(surrogate) => surrogate,
        None => code_point,
    }
}

/// Encodes a Python source/runtime code point into a Rust `char`.
///
/// Surrogate code points are remapped into Prism's carrier range so the result
/// remains a valid Unicode scalar.
#[inline]
#[must_use]
pub fn encode_python_code_point(code_point: u32) -> Option<char> {
    let stored = match surrogate_to_carrier(code_point) {
        Some(carrier) => carrier,
        None => code_point,
    };
    char::from_u32(stored)
}

/// Return the CPython 3.12 Unicode decimal digit value for `ch`.
///
/// Python accepts all Unicode characters with the Decimal_Number property in
/// text-to-number conversions, but Rust's `char::to_digit(10)` intentionally
/// only recognizes ASCII radix digits. The ranges below are generated from
/// CPython 3.12's `Objects/unicodetype_db.h` decimal table.
#[inline]
#[must_use]
pub fn python_decimal_digit(ch: char) -> Option<u8> {
    let code_point = ch as u32;
    match ch {
        '\u{0030}'..='\u{0039}' => Some((code_point - 0x30) as u8),
        '\u{0660}'..='\u{0669}' => Some((code_point - 0x660) as u8),
        '\u{06F0}'..='\u{06F9}' => Some((code_point - 0x6F0) as u8),
        '\u{07C0}'..='\u{07C9}' => Some((code_point - 0x7C0) as u8),
        '\u{0966}'..='\u{096F}' => Some((code_point - 0x966) as u8),
        '\u{09E6}'..='\u{09EF}' => Some((code_point - 0x9E6) as u8),
        '\u{0A66}'..='\u{0A6F}' => Some((code_point - 0xA66) as u8),
        '\u{0AE6}'..='\u{0AEF}' => Some((code_point - 0xAE6) as u8),
        '\u{0B66}'..='\u{0B6F}' => Some((code_point - 0xB66) as u8),
        '\u{0BE6}'..='\u{0BEF}' => Some((code_point - 0xBE6) as u8),
        '\u{0C66}'..='\u{0C6F}' => Some((code_point - 0xC66) as u8),
        '\u{0CE6}'..='\u{0CEF}' => Some((code_point - 0xCE6) as u8),
        '\u{0D66}'..='\u{0D6F}' => Some((code_point - 0xD66) as u8),
        '\u{0DE6}'..='\u{0DEF}' => Some((code_point - 0xDE6) as u8),
        '\u{0E50}'..='\u{0E59}' => Some((code_point - 0xE50) as u8),
        '\u{0ED0}'..='\u{0ED9}' => Some((code_point - 0xED0) as u8),
        '\u{0F20}'..='\u{0F29}' => Some((code_point - 0xF20) as u8),
        '\u{1040}'..='\u{1049}' => Some((code_point - 0x1040) as u8),
        '\u{1090}'..='\u{1099}' => Some((code_point - 0x1090) as u8),
        '\u{17E0}'..='\u{17E9}' => Some((code_point - 0x17E0) as u8),
        '\u{1810}'..='\u{1819}' => Some((code_point - 0x1810) as u8),
        '\u{1946}'..='\u{194F}' => Some((code_point - 0x1946) as u8),
        '\u{19D0}'..='\u{19D9}' => Some((code_point - 0x19D0) as u8),
        '\u{1A80}'..='\u{1A89}' => Some((code_point - 0x1A80) as u8),
        '\u{1A90}'..='\u{1A99}' => Some((code_point - 0x1A90) as u8),
        '\u{1B50}'..='\u{1B59}' => Some((code_point - 0x1B50) as u8),
        '\u{1BB0}'..='\u{1BB9}' => Some((code_point - 0x1BB0) as u8),
        '\u{1C40}'..='\u{1C49}' => Some((code_point - 0x1C40) as u8),
        '\u{1C50}'..='\u{1C59}' => Some((code_point - 0x1C50) as u8),
        '\u{A620}'..='\u{A629}' => Some((code_point - 0xA620) as u8),
        '\u{A8D0}'..='\u{A8D9}' => Some((code_point - 0xA8D0) as u8),
        '\u{A900}'..='\u{A909}' => Some((code_point - 0xA900) as u8),
        '\u{A9D0}'..='\u{A9D9}' => Some((code_point - 0xA9D0) as u8),
        '\u{A9F0}'..='\u{A9F9}' => Some((code_point - 0xA9F0) as u8),
        '\u{AA50}'..='\u{AA59}' => Some((code_point - 0xAA50) as u8),
        '\u{ABF0}'..='\u{ABF9}' => Some((code_point - 0xABF0) as u8),
        '\u{FF10}'..='\u{FF19}' => Some((code_point - 0xFF10) as u8),
        '\u{104A0}'..='\u{104A9}' => Some((code_point - 0x104A0) as u8),
        '\u{10D30}'..='\u{10D39}' => Some((code_point - 0x10D30) as u8),
        '\u{11066}'..='\u{1106F}' => Some((code_point - 0x11066) as u8),
        '\u{110F0}'..='\u{110F9}' => Some((code_point - 0x110F0) as u8),
        '\u{11136}'..='\u{1113F}' => Some((code_point - 0x11136) as u8),
        '\u{111D0}'..='\u{111D9}' => Some((code_point - 0x111D0) as u8),
        '\u{112F0}'..='\u{112F9}' => Some((code_point - 0x112F0) as u8),
        '\u{11450}'..='\u{11459}' => Some((code_point - 0x11450) as u8),
        '\u{114D0}'..='\u{114D9}' => Some((code_point - 0x114D0) as u8),
        '\u{11650}'..='\u{11659}' => Some((code_point - 0x11650) as u8),
        '\u{116C0}'..='\u{116C9}' => Some((code_point - 0x116C0) as u8),
        '\u{11730}'..='\u{11739}' => Some((code_point - 0x11730) as u8),
        '\u{118E0}'..='\u{118E9}' => Some((code_point - 0x118E0) as u8),
        '\u{11950}'..='\u{11959}' => Some((code_point - 0x11950) as u8),
        '\u{11C50}'..='\u{11C59}' => Some((code_point - 0x11C50) as u8),
        '\u{11D50}'..='\u{11D59}' => Some((code_point - 0x11D50) as u8),
        '\u{11DA0}'..='\u{11DA9}' => Some((code_point - 0x11DA0) as u8),
        '\u{11F50}'..='\u{11F59}' => Some((code_point - 0x11F50) as u8),
        '\u{16A60}'..='\u{16A69}' => Some((code_point - 0x16A60) as u8),
        '\u{16AC0}'..='\u{16AC9}' => Some((code_point - 0x16AC0) as u8),
        '\u{16B50}'..='\u{16B59}' => Some((code_point - 0x16B50) as u8),
        '\u{1D7CE}'..='\u{1D7D7}' => Some((code_point - 0x1D7CE) as u8),
        '\u{1D7D8}'..='\u{1D7E1}' => Some((code_point - 0x1D7D8) as u8),
        '\u{1D7E2}'..='\u{1D7EB}' => Some((code_point - 0x1D7E2) as u8),
        '\u{1D7EC}'..='\u{1D7F5}' => Some((code_point - 0x1D7EC) as u8),
        '\u{1D7F6}'..='\u{1D7FF}' => Some((code_point - 0x1D7F6) as u8),
        '\u{1E140}'..='\u{1E149}' => Some((code_point - 0x1E140) as u8),
        '\u{1E2F0}'..='\u{1E2F9}' => Some((code_point - 0x1E2F0) as u8),
        '\u{1E4F0}'..='\u{1E4F9}' => Some((code_point - 0x1E4F0) as u8),
        '\u{1E950}'..='\u{1E959}' => Some((code_point - 0x1E950) as u8),
        '\u{1FBF0}'..='\u{1FBF9}' => Some((code_point - 0x1FBF0) as u8),
        _ => None,
    }
}

/// Returns whether a string contains any internal surrogate carrier scalars.
#[inline]
#[must_use]
pub fn contains_surrogate_carriers(input: &str) -> bool {
    input.chars().any(|ch| is_surrogate_carrier(ch as u32))
}

/// Formats a Python-visible code point using CPython-style escape rules.
#[inline]
#[must_use]
pub fn python_code_point_escape(code_point: u32) -> String {
    let logical = logical_python_code_point(code_point);
    match logical {
        0x00..=0xFF => format!("\\x{logical:02x}"),
        0x0100..=0xFFFF => format!("\\u{logical:04x}"),
        _ => format!("\\U{logical:08x}"),
    }
}

/// Formats a stored scalar using CPython-style escape rules.
#[inline]
#[must_use]
pub fn python_char_escape(ch: char) -> String {
    python_code_point_escape(ch as u32)
}
