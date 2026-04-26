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

#[cfg(test)]
mod tests;
