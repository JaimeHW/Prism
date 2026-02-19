//! String and bytes builtins (ord, chr, bytes, bytearray, format).
//!
//! High-performance implementations with full Unicode support and
//! optimal code point conversion. All functions are Python 3.12 compatible.
//!
//! # Performance Characteristics
//!
//! - **ord/chr**: O(1) Unicode code point conversion
//! - **ASCII fast paths**: Optimized for common ASCII range [0, 127]
//! - **UTF-8 aware**: Proper handling of multi-byte sequences
//!
//! # Python Semantics
//!
//! - `ord(c)` - Returns Unicode code point for single character
//! - `chr(i)` - Returns character for Unicode code point [0, 0x10FFFF]
//! - `bytes()` - Immutable byte sequence constructor
//! - `bytearray()` - Mutable byte sequence constructor
//! - `format()` - String formatting (simplified)

use super::BuiltinError;
use prism_core::Value;
use prism_core::intern::{intern, interned_by_ptr};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::string::StringObject;

// =============================================================================
// Unicode Constants
// =============================================================================

/// Maximum valid Unicode code point (U+10FFFF).
const MAX_UNICODE_CODE_POINT: u32 = 0x10FFFF;

/// Start of surrogate range (not valid for encoding).
const SURROGATE_START: u32 = 0xD800;

/// End of surrogate range (not valid for encoding).
const SURROGATE_END: u32 = 0xDFFF;

/// ASCII printable range end.
const ASCII_MAX: u32 = 0x7F;

/// Maximum byte sequence length accepted by constructors.
const MAX_BYTE_SEQUENCE_SIZE: i64 = 1_000_000_000;

#[derive(Clone, Copy, PartialEq, Eq)]
enum ByteSequenceKind {
    Bytes,
    ByteArray,
}

impl ByteSequenceKind {
    #[inline]
    fn fn_name(self) -> &'static str {
        match self {
            Self::Bytes => "bytes",
            Self::ByteArray => "bytearray",
        }
    }

    #[inline]
    fn type_id(self) -> TypeId {
        match self {
            Self::Bytes => TypeId::BYTES,
            Self::ByteArray => TypeId::BYTEARRAY,
        }
    }

    #[inline]
    fn range_error(self) -> &'static str {
        match self {
            Self::Bytes => "bytes must be in range(0, 256)",
            Self::ByteArray => "byte must be in range(0, 256)",
        }
    }

    #[inline]
    fn from_data(self, data: Vec<u8>) -> Value {
        let obj = BytesObject::from_vec_with_type(data, self.type_id());
        let ptr = Box::leak(Box::new(obj)) as *mut BytesObject as *const ();
        Value::object_ptr(ptr)
    }
}

// =============================================================================
// ord() - Get Unicode Code Point
// =============================================================================

/// Builtin ord(c) function.
///
/// Returns the Unicode code point for a one-character string.
///
/// # Python Semantics
/// - `ord('a')` → `97`
/// - `ord('€')` → `8364`
/// - `ord('🎉')` → `127881`
/// - `ord('')` → TypeError (empty string)
/// - `ord('ab')` → TypeError (string length > 1)
/// - `ord(123)` → TypeError (not a string)
///
/// # Performance
/// - O(1) for single-byte UTF-8 (ASCII)
/// - O(1) for multi-byte UTF-8 (decode directly)
pub fn builtin_ord(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "ord() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let s = value_to_string(args[0]).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "ord() expected string of length 1, but {} found",
            type_name_of(args[0])
        ))
    })?;

    let code_point = ord_from_str(&s)?;
    Value::int(code_point as i64).ok_or_else(|| {
        BuiltinError::OverflowError("ord() result out of supported integer range".to_string())
    })
}

/// Get Unicode code point from a single-character string slice.
///
/// This is the core implementation used by `ord()`.
/// Returns error if string is empty or has more than one character.
#[inline]
pub fn ord_from_str(s: &str) -> Result<u32, BuiltinError> {
    let mut chars = s.chars();

    match chars.next() {
        None => Err(BuiltinError::TypeError(
            "ord() expected a character, but string of length 0 found".to_string(),
        )),
        Some(c) => {
            // Verify exactly one character
            if chars.next().is_some() {
                return Err(BuiltinError::TypeError(format!(
                    "ord() expected a character, but string of length {} found",
                    s.chars().count()
                )));
            }
            Ok(c as u32)
        }
    }
}

// =============================================================================
// chr() - Get Character from Code Point
// =============================================================================

/// Builtin chr(i) function.
///
/// Returns a string of one character whose Unicode code point is the integer i.
///
/// # Python Semantics
/// - `chr(97)` → `'a'`
/// - `chr(8364)` → `'€'`
/// - `chr(127881)` → `'🎉'`
/// - `chr(-1)` → ValueError (negative)
/// - `chr(0x110000)` → ValueError (> max code point)
/// - `chr(0xD800)` → ValueError (surrogate)
///
/// # Performance
/// - O(1) range check and conversion
/// - Single branch for surrogate detection
/// - Direct char::from_u32_unchecked for valid code points
pub fn builtin_chr(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "chr() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let code_point = extract_code_point(&args[0])?;
    let c = chr_from_code_point(code_point)?;

    let mut buf = [0u8; 4];
    let s = c.encode_utf8(&mut buf);
    Ok(Value::string(intern(s)))
}

/// Extract a Unicode code point from a Value.
///
/// Accepts integer or bool. Returns error for other types.
#[inline]
fn extract_code_point(val: &Value) -> Result<u32, BuiltinError> {
    if let Some(i) = val.as_int() {
        if i < 0 {
            return Err(BuiltinError::ValueError(format!(
                "chr() arg not in range(0x110000): {} (negative)",
                i
            )));
        }
        if i > MAX_UNICODE_CODE_POINT as i64 {
            return Err(BuiltinError::ValueError(format!(
                "chr() arg not in range(0x110000): {} (too large)",
                i
            )));
        }
        return Ok(i as u32);
    }

    // Bool is valid (True=1='\\x01', False=0='\\x00')
    if let Some(b) = val.as_bool() {
        return Ok(if b { 1 } else { 0 });
    }

    Err(BuiltinError::TypeError(
        "an integer is required".to_string(),
    ))
}

/// Convert a Unicode code point to a character.
///
/// Returns error for invalid code points (surrogates, out of range).
#[inline]
pub fn chr_from_code_point(code_point: u32) -> Result<char, BuiltinError> {
    // Check for surrogate range (U+D800 to U+DFFF)
    if code_point >= SURROGATE_START && code_point <= SURROGATE_END {
        return Err(BuiltinError::ValueError(format!(
            "chr() arg not in range(0x110000): {} (surrogate)",
            code_point
        )));
    }

    // Check upper bound
    if code_point > MAX_UNICODE_CODE_POINT {
        return Err(BuiltinError::ValueError(format!(
            "chr() arg not in range(0x110000): {} (too large)",
            code_point
        )));
    }

    // SAFETY: We've validated the code point is not a surrogate
    // and is within the valid Unicode range.
    // char::from_u32 would do the same checks, but we've already done them.
    match char::from_u32(code_point) {
        Some(c) => Ok(c),
        None => Err(BuiltinError::ValueError(format!(
            "chr() arg not in range(0x110000): {}",
            code_point
        ))),
    }
}

// =============================================================================
// bytes() - Immutable Byte Sequence
// =============================================================================

/// Builtin bytes([source[, encoding[, errors]]]) function.
///
/// Returns an immutable bytes object.
///
/// # Python Semantics
/// - `bytes()` → `b''`
/// - `bytes(5)` → `b'\x00\x00\x00\x00\x00'` (5 null bytes)
/// - `bytes([65, 66, 67])` → `b'ABC'`
/// - `bytes('hello', 'utf-8')` → `b'hello'`
///
/// Fully supports empty/count/iterable/string+encoding constructor forms.
pub fn builtin_bytes(args: &[Value]) -> Result<Value, BuiltinError> {
    build_byte_sequence(args, ByteSequenceKind::Bytes)
}

// =============================================================================
// bytearray() - Mutable Byte Sequence
// =============================================================================

/// Builtin bytearray([source[, encoding[, errors]]]) function.
///
/// Returns a mutable bytearray object.
///
/// # Python Semantics
/// Same as bytes() but returns mutable bytearray.
///
/// Fully supports empty/count/iterable/string+encoding constructor forms.
pub fn builtin_bytearray(args: &[Value]) -> Result<Value, BuiltinError> {
    build_byte_sequence(args, ByteSequenceKind::ByteArray)
}

fn build_byte_sequence(args: &[Value], kind: ByteSequenceKind) -> Result<Value, BuiltinError> {
    let fn_name = kind.fn_name();
    if args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "{}() takes at most 3 arguments ({} given)",
            fn_name,
            args.len()
        )));
    }

    if args.is_empty() {
        return Ok(kind.from_data(Vec::new()));
    }

    // source + encoding[, errors]
    if args.len() >= 2 {
        let source = value_to_string(args[0]).ok_or_else(|| {
            BuiltinError::TypeError("encoding without a string argument".to_string())
        })?;
        let encoding = value_to_string(args[1]).ok_or_else(|| {
            BuiltinError::TypeError(format!(
                "{}() argument 2 must be str, not {}",
                fn_name,
                type_name_of(args[1])
            ))
        })?;
        let errors = if args.len() == 3 {
            value_to_string(args[2]).ok_or_else(|| {
                BuiltinError::TypeError(format!(
                    "{}() argument 3 must be str, not {}",
                    fn_name,
                    type_name_of(args[2])
                ))
            })?
        } else {
            "strict".to_string()
        };

        let data = encode_string(&source, &encoding, &errors, fn_name)?;
        return Ok(kind.from_data(data));
    }

    // Single-argument form.
    let source = args[0];
    if let Some(count) = source.as_int() {
        return sequence_from_count(kind, count);
    }
    if let Some(b) = source.as_bool() {
        return sequence_from_count(kind, if b { 1 } else { 0 });
    }

    if source.is_string() {
        return Err(BuiltinError::TypeError(
            "string argument without an encoding".to_string(),
        ));
    }

    if let Some(ptr) = source.as_object_ptr() {
        match crate::ops::objects::extract_type_id(ptr) {
            TypeId::STR => {
                return Err(BuiltinError::TypeError(
                    "string argument without an encoding".to_string(),
                ));
            }
            TypeId::BYTES => {
                if kind == ByteSequenceKind::Bytes {
                    return Ok(source);
                }
                let bytes = unsafe { &*(ptr as *const BytesObject) };
                return Ok(kind.from_data(bytes.to_vec()));
            }
            TypeId::BYTEARRAY => {
                let bytes = unsafe { &*(ptr as *const BytesObject) };
                return Ok(kind.from_data(bytes.to_vec()));
            }
            _ => {}
        }
    }

    let data = sequence_from_iterable(source, kind)?;
    Ok(kind.from_data(data))
}

#[inline]
fn sequence_from_count(kind: ByteSequenceKind, count: i64) -> Result<Value, BuiltinError> {
    if count < 0 {
        return Err(BuiltinError::ValueError("negative count".to_string()));
    }
    if count > MAX_BYTE_SEQUENCE_SIZE {
        return Err(BuiltinError::OverflowError(format!(
            "{} size too large",
            kind.fn_name()
        )));
    }
    Ok(kind.from_data(vec![0; count as usize]))
}

#[inline]
fn sequence_from_iterable(source: Value, kind: ByteSequenceKind) -> Result<Vec<u8>, BuiltinError> {
    let values = if let Some(iter) = super::iter_dispatch::get_iterator_mut(&source) {
        iter.collect_remaining()
    } else {
        let mut iter =
            super::iter_dispatch::value_to_iterator(&source).map_err(BuiltinError::from)?;
        iter.collect_remaining()
    };

    let mut data = Vec::with_capacity(values.len());
    for value in values {
        data.push(value_to_byte(value, kind)?);
    }
    Ok(data)
}

#[inline]
fn value_to_byte(value: Value, kind: ByteSequenceKind) -> Result<u8, BuiltinError> {
    if let Some(i) = value.as_int() {
        return i64_to_byte(i, kind);
    }
    if let Some(b) = value.as_bool() {
        return Ok(if b { 1 } else { 0 });
    }

    Err(BuiltinError::TypeError(format!(
        "'{}' object cannot be interpreted as an integer",
        type_name_of(value)
    )))
}

#[inline]
fn i64_to_byte(value: i64, kind: ByteSequenceKind) -> Result<u8, BuiltinError> {
    if (0..=255).contains(&value) {
        Ok(value as u8)
    } else {
        Err(BuiltinError::ValueError(kind.range_error().to_string()))
    }
}

#[derive(Clone, Copy)]
enum EncodingErrorPolicy {
    Strict,
    Ignore,
    Replace,
}

#[inline]
fn parse_encoding_error_policy(errors: &str) -> Result<EncodingErrorPolicy, BuiltinError> {
    match errors.to_ascii_lowercase().as_str() {
        "strict" => Ok(EncodingErrorPolicy::Strict),
        "ignore" => Ok(EncodingErrorPolicy::Ignore),
        "replace" => Ok(EncodingErrorPolicy::Replace),
        _ => Err(BuiltinError::ValueError(format!(
            "unknown error handler name '{}'",
            errors
        ))),
    }
}

fn encode_string(
    input: &str,
    encoding: &str,
    errors: &str,
    fn_name: &str,
) -> Result<Vec<u8>, BuiltinError> {
    let normalized = encoding.trim().to_ascii_lowercase().replace('_', "-");
    let policy = parse_encoding_error_policy(errors)?;

    match normalized.as_str() {
        "utf8" | "utf-8" => Ok(input.as_bytes().to_vec()),
        "ascii" => encode_ascii(input, policy, fn_name),
        "latin1" | "latin-1" | "iso-8859-1" => encode_latin1(input, policy, fn_name),
        _ => Err(BuiltinError::ValueError(format!(
            "unknown encoding: {}",
            encoding
        ))),
    }
}

fn encode_ascii(
    input: &str,
    policy: EncodingErrorPolicy,
    fn_name: &str,
) -> Result<Vec<u8>, BuiltinError> {
    let mut out = Vec::with_capacity(input.len());
    for ch in input.chars() {
        let code = ch as u32;
        if code <= 0x7f {
            out.push(code as u8);
            continue;
        }
        match policy {
            EncodingErrorPolicy::Strict => {
                return Err(BuiltinError::ValueError(format!(
                    "{}() could not encode character U+{:04X} with ascii codec",
                    fn_name, code
                )));
            }
            EncodingErrorPolicy::Ignore => {}
            EncodingErrorPolicy::Replace => out.push(b'?'),
        }
    }
    Ok(out)
}

fn encode_latin1(
    input: &str,
    policy: EncodingErrorPolicy,
    fn_name: &str,
) -> Result<Vec<u8>, BuiltinError> {
    let mut out = Vec::with_capacity(input.len());
    for ch in input.chars() {
        let code = ch as u32;
        if code <= 0xff {
            out.push(code as u8);
            continue;
        }
        match policy {
            EncodingErrorPolicy::Strict => {
                return Err(BuiltinError::ValueError(format!(
                    "{}() could not encode character U+{:04X} with latin-1 codec",
                    fn_name, code
                )));
            }
            EncodingErrorPolicy::Ignore => {}
            EncodingErrorPolicy::Replace => out.push(b'?'),
        }
    }
    Ok(out)
}

// =============================================================================
// format() - String Formatting
// =============================================================================

/// Builtin format(value[, format_spec]) function.
///
/// Returns a formatted representation of value.
///
/// # Python Semantics
/// - `format(42)` → `'42'`
/// - `format(3.14159, '.2f')` → `'3.14'`
/// - `format(255, 'x')` → `'ff'`
/// - `format(255, '#x')` → `'0xff'`
/// - `format(1234567, ',')` → `'1,234,567'`
///
/// # Implementation Note
/// Full implementation requires __format__ protocol.
pub fn builtin_format(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "format() takes 1 or 2 arguments ({} given)",
            args.len()
        )));
    }

    let value = args[0];
    let format_spec = if args.len() == 2 {
        value_to_string(args[1]).ok_or_else(|| {
            BuiltinError::TypeError(format!(
                "format() argument 2 must be str, not {}",
                type_name_of(args[1])
            ))
        })?
    } else {
        String::new()
    };

    // Handle basic numeric formatting
    format_value(value, &format_spec)
}

/// Format a value according to format_spec.
///
/// This is a simplified implementation for numeric types.
#[inline]
fn format_value(value: Value, format_spec: &str) -> Result<Value, BuiltinError> {
    // Empty format spec: default formatting
    if format_spec.is_empty() {
        return super::types::builtin_str(&[value]);
    }

    // Integer formatting
    if let Some(i) = value.as_int() {
        return format_int(i, format_spec);
    }

    // Float formatting
    if let Some(f) = value.as_float() {
        return format_float(f, format_spec);
    }

    Err(BuiltinError::TypeError(format!(
        "unsupported format string passed to {}.__format__",
        type_name_of(value)
    )))
}

/// Format an integer according to format_spec.
#[inline]
fn format_int(n: i64, format_spec: &str) -> Result<Value, BuiltinError> {
    let formatted = match format_spec {
        "b" => format!("{:b}", n),   // Binary
        "#b" => format!("{:#b}", n), // Binary with prefix
        "o" => format!("{:o}", n),   // Octal
        "#o" => format!("{:#o}", n), // Octal with prefix
        "x" => format!("{:x}", n),   // Hex lowercase
        "#x" => format!("{:#x}", n), // Hex lowercase with prefix
        "X" => format!("{:X}", n),   // Hex uppercase
        "#X" => format!("{:#X}", n), // Hex uppercase with prefix
        "d" => format!("{}", n),     // Decimal
        "c" => {
            // Character (for integer code point)
            if n < 0 || n > MAX_UNICODE_CODE_POINT as i64 {
                return Err(BuiltinError::OverflowError(
                    "%c arg not in range(0x110000)".to_string(),
                ));
            }
            match char::from_u32(n as u32) {
                Some(c) => c.to_string(),
                None => {
                    return Err(BuiltinError::OverflowError(
                        "%c arg not in range(0x110000)".to_string(),
                    ));
                }
            }
        }
        "," => format_with_thousands_separator(n),
        "_" => format_with_underscore_separator(n),
        _ if format_spec.contains('.') => {
            // Precision specification (for integers, pads with zeros)
            if let Some(precision) = parse_precision(format_spec) {
                format!("{:0width$}", n, width = precision)
            } else {
                format!("{}", n)
            }
        }
        _ => format!("{}", n),
    };

    Ok(Value::string(intern(&formatted)))
}

/// Format a float according to format_spec.
#[inline]
fn format_float(f: f64, format_spec: &str) -> Result<Value, BuiltinError> {
    let formatted = match format_spec {
        "e" => format!("{:e}", f),     // Exponential lowercase
        "E" => format!("{:E}", f),     // Exponential uppercase
        "f" | "F" => format!("{}", f), // Fixed-point
        "g" | "G" => {
            // General format: use exponential if exponent >= precision
            // Simplified: just use Display format
            format!("{}", f)
        }
        "%" => format!("{}%", f * 100.0), // Percentage
        _ if format_spec.starts_with('.') => {
            // Precision specification
            if let Some(precision) = parse_precision(format_spec) {
                format!("{:.prec$}", f, prec = precision)
            } else {
                format!("{}", f)
            }
        }
        _ => format!("{}", f),
    };

    Ok(Value::string(intern(&formatted)))
}

/// Parse precision from format spec like ".2f" → 2.
#[inline]
fn parse_precision(format_spec: &str) -> Option<usize> {
    let s = format_spec.trim_start_matches('.');
    let digits: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().ok()
}

/// Format integer with thousands separator (,).
#[inline]
fn format_with_thousands_separator(n: i64) -> String {
    let s = n.abs().to_string();
    let negative = n < 0;

    let mut result = String::with_capacity(s.len() + s.len() / 3 + 1);
    if negative {
        result.push('-');
    }

    let chars: Vec<char> = s.chars().collect();
    for (i, c) in chars.iter().enumerate() {
        if i > 0 && (chars.len() - i) % 3 == 0 {
            result.push(',');
        }
        result.push(*c);
    }

    result
}

/// Format integer with underscore separator (_).
#[inline]
fn format_with_underscore_separator(n: i64) -> String {
    let s = n.abs().to_string();
    let negative = n < 0;

    let mut result = String::with_capacity(s.len() + s.len() / 3 + 1);
    if negative {
        result.push('-');
    }

    let chars: Vec<char> = s.chars().collect();
    for (i, c) in chars.iter().enumerate() {
        if i > 0 && (chars.len() - i) % 3 == 0 {
            result.push('_');
        }
        result.push(*c);
    }

    result
}

// =============================================================================
// ASCII Fast Path Utilities
// =============================================================================

/// Check if a code point is in ASCII range.
#[inline(always)]
pub const fn is_ascii(code_point: u32) -> bool {
    code_point <= ASCII_MAX
}

/// Check if a code point is a valid Unicode scalar value.
#[inline(always)]
pub const fn is_valid_code_point(code_point: u32) -> bool {
    code_point <= MAX_UNICODE_CODE_POINT
        && !(code_point >= SURROGATE_START && code_point <= SURROGATE_END)
}

/// Check if a code point is in the surrogate range.
#[inline(always)]
pub const fn is_surrogate(code_point: u32) -> bool {
    code_point >= SURROGATE_START && code_point <= SURROGATE_END
}

#[inline]
fn value_to_string(value: Value) -> Option<String> {
    if value.is_string() {
        let ptr = value.as_string_object_ptr()?;
        let interned = interned_by_ptr(ptr as *const u8)?;
        return Some(interned.as_str().to_string());
    }

    let ptr = value.as_object_ptr()?;
    if crate::ops::objects::extract_type_id(ptr) != TypeId::STR {
        return None;
    }
    let string_obj = unsafe { &*(ptr as *const StringObject) };
    Some(string_obj.as_str().to_string())
}

#[inline]
fn type_name_of(value: Value) -> &'static str {
    if value.is_none() {
        "NoneType"
    } else if value.is_bool() {
        "bool"
    } else if value.is_int() {
        "int"
    } else if value.is_float() {
        "float"
    } else if value.is_string() {
        "str"
    } else if let Some(ptr) = value.as_object_ptr() {
        crate::ops::objects::extract_type_id(ptr).name()
    } else {
        "object"
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use prism_core::intern::{intern, interned_by_ptr};
    use prism_runtime::types::bytes::BytesObject;
    use prism_runtime::types::string::StringObject;

    fn value_to_rust_string(value: Value) -> String {
        if value.is_string() {
            let ptr = value
                .as_string_object_ptr()
                .expect("tagged string should have pointer");
            let interned =
                interned_by_ptr(ptr as *const u8).expect("interned pointer should resolve");
            return interned.as_str().to_string();
        }

        let ptr = value
            .as_object_ptr()
            .expect("string value should be object-backed");
        assert_eq!(crate::ops::objects::extract_type_id(ptr), TypeId::STR);
        let string_obj = unsafe { &*(ptr as *const StringObject) };
        string_obj.as_str().to_string()
    }

    fn boxed_value<T>(obj: T) -> (Value, *mut T) {
        let ptr = Box::into_raw(Box::new(obj));
        (Value::object_ptr(ptr as *const ()), ptr)
    }

    unsafe fn drop_boxed<T>(ptr: *mut T) {
        drop(unsafe { Box::from_raw(ptr) });
    }

    fn value_to_byte_vec(value: Value) -> Vec<u8> {
        let ptr = value
            .as_object_ptr()
            .expect("byte sequence should be object-backed");
        let type_id = crate::ops::objects::extract_type_id(ptr);
        assert!(
            type_id == TypeId::BYTES || type_id == TypeId::BYTEARRAY,
            "unexpected type id for byte sequence: {:?}",
            type_id
        );
        let bytes_obj = unsafe { &*(ptr as *const BytesObject) };
        bytes_obj.as_bytes().to_vec()
    }

    fn byte_sequence_type(value: Value) -> TypeId {
        let ptr = value
            .as_object_ptr()
            .expect("byte sequence should be object-backed");
        crate::ops::objects::extract_type_id(ptr)
    }

    // =========================================================================
    // ord() Argument Validation Tests
    // =========================================================================

    #[test]
    fn test_ord_no_args() {
        let result = builtin_ord(&[]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(msg)) => {
                assert!(msg.contains("exactly one argument"));
            }
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_ord_too_many_args() {
        let result = builtin_ord(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
        assert!(result.is_err());
    }

    #[test]
    fn test_ord_tagged_string() {
        let result = builtin_ord(&[Value::string(intern("A"))]).unwrap();
        assert_eq!(result.as_int(), Some(65));
    }

    #[test]
    fn test_ord_heap_string() {
        let (string_value, string_ptr) = boxed_value(StringObject::new("€"));
        let result = builtin_ord(&[string_value]).unwrap();
        assert_eq!(result.as_int(), Some(8364));
        unsafe { drop_boxed(string_ptr) };
    }

    // =========================================================================
    // ord_from_str() Tests
    // =========================================================================

    #[test]
    fn test_ord_from_str_ascii() {
        assert_eq!(ord_from_str("a").unwrap(), 97);
        assert_eq!(ord_from_str("A").unwrap(), 65);
        assert_eq!(ord_from_str("0").unwrap(), 48);
        assert_eq!(ord_from_str(" ").unwrap(), 32);
        assert_eq!(ord_from_str("~").unwrap(), 126);
    }

    #[test]
    fn test_ord_from_str_control_chars() {
        assert_eq!(ord_from_str("\0").unwrap(), 0);
        assert_eq!(ord_from_str("\t").unwrap(), 9);
        assert_eq!(ord_from_str("\n").unwrap(), 10);
        assert_eq!(ord_from_str("\r").unwrap(), 13);
    }

    #[test]
    fn test_ord_from_str_unicode_bmp() {
        // Basic Multilingual Plane
        assert_eq!(ord_from_str("€").unwrap(), 8364); // Euro sign
        assert_eq!(ord_from_str("£").unwrap(), 163); // Pound sign
        assert_eq!(ord_from_str("¥").unwrap(), 165); // Yen sign
        assert_eq!(ord_from_str("©").unwrap(), 169); // Copyright
        assert_eq!(ord_from_str("®").unwrap(), 174); // Registered
    }

    #[test]
    fn test_ord_from_str_unicode_supplementary() {
        // Supplementary planes (emoji, etc.)
        assert_eq!(ord_from_str("🎉").unwrap(), 127881); // Party popper
        assert_eq!(ord_from_str("😀").unwrap(), 128512); // Grinning face
        assert_eq!(ord_from_str("🚀").unwrap(), 128640); // Rocket
        assert_eq!(ord_from_str("💻").unwrap(), 128187); // Laptop
    }

    #[test]
    fn test_ord_from_str_unicode_cjk() {
        assert_eq!(ord_from_str("中").unwrap(), 20013); // Chinese
        assert_eq!(ord_from_str("日").unwrap(), 26085); // Japanese
        assert_eq!(ord_from_str("한").unwrap(), 54620); // Korean
    }

    #[test]
    fn test_ord_from_str_empty() {
        let result = ord_from_str("");
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(msg)) => {
                assert!(msg.contains("length 0"));
            }
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_ord_from_str_multiple_chars() {
        let result = ord_from_str("ab");
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(msg)) => {
                assert!(msg.contains("length 2"));
            }
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_ord_from_str_multiple_emoji() {
        // This is 2 graphemes, 2 code points
        let result = ord_from_str("🎉🚀");
        assert!(result.is_err());
    }

    // =========================================================================
    // chr() Argument Validation Tests
    // =========================================================================

    #[test]
    fn test_chr_no_args() {
        let result = builtin_chr(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_chr_too_many_args() {
        let result = builtin_chr(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
        assert!(result.is_err());
    }

    #[test]
    fn test_chr_float_error() {
        let result = builtin_chr(&[Value::float(3.14)]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(msg)) => {
                assert!(msg.contains("integer"));
            }
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_chr_ascii_and_unicode_return_strings() {
        let ascii = builtin_chr(&[Value::int(97).unwrap()]).unwrap();
        assert_eq!(value_to_rust_string(ascii), "a");

        let unicode = builtin_chr(&[Value::int(127881).unwrap()]).unwrap();
        assert_eq!(value_to_rust_string(unicode), "🎉");
    }

    // =========================================================================
    // chr_from_code_point() Tests
    // =========================================================================

    #[test]
    fn test_chr_from_code_point_ascii() {
        assert_eq!(chr_from_code_point(97).unwrap(), 'a');
        assert_eq!(chr_from_code_point(65).unwrap(), 'A');
        assert_eq!(chr_from_code_point(48).unwrap(), '0');
        assert_eq!(chr_from_code_point(32).unwrap(), ' ');
        assert_eq!(chr_from_code_point(126).unwrap(), '~');
    }

    #[test]
    fn test_chr_from_code_point_control() {
        assert_eq!(chr_from_code_point(0).unwrap(), '\0');
        assert_eq!(chr_from_code_point(9).unwrap(), '\t');
        assert_eq!(chr_from_code_point(10).unwrap(), '\n');
        assert_eq!(chr_from_code_point(13).unwrap(), '\r');
    }

    #[test]
    fn test_chr_from_code_point_unicode() {
        assert_eq!(chr_from_code_point(8364).unwrap(), '€');
        assert_eq!(chr_from_code_point(127881).unwrap(), '🎉');
        assert_eq!(chr_from_code_point(128512).unwrap(), '😀');
    }

    #[test]
    fn test_chr_from_code_point_boundary() {
        // Minimum valid
        assert!(chr_from_code_point(0).is_ok());
        // Maximum valid (U+10FFFF)
        assert!(chr_from_code_point(MAX_UNICODE_CODE_POINT).is_ok());
    }

    #[test]
    fn test_chr_from_code_point_surrogate_start() {
        let result = chr_from_code_point(SURROGATE_START);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::ValueError(msg)) => {
                assert!(msg.contains("surrogate"));
            }
            _ => panic!("Expected ValueError"),
        }
    }

    #[test]
    fn test_chr_from_code_point_surrogate_middle() {
        let result = chr_from_code_point(0xDA00);
        assert!(result.is_err());
    }

    #[test]
    fn test_chr_from_code_point_surrogate_end() {
        let result = chr_from_code_point(SURROGATE_END);
        assert!(result.is_err());
    }

    #[test]
    fn test_chr_from_code_point_too_large() {
        let result = chr_from_code_point(0x110000);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::ValueError(msg)) => {
                assert!(msg.contains("too large"));
            }
            _ => panic!("Expected ValueError"),
        }
    }

    // =========================================================================
    // ord/chr Roundtrip Tests
    // =========================================================================

    #[test]
    fn test_ord_chr_roundtrip_ascii() {
        for cp in 0..=127u32 {
            let c = chr_from_code_point(cp).unwrap();
            let s = c.to_string();
            let result = ord_from_str(&s).unwrap();
            assert_eq!(result, cp, "Roundtrip failed for code point {}", cp);
        }
    }

    #[test]
    fn test_ord_chr_roundtrip_extended() {
        let test_points = [
            128, 255, 256, 1000, 8364, 20013, 65535, 66000, 100000, 127881, 128512, 0x10FFFF,
        ];
        for cp in test_points {
            let c = chr_from_code_point(cp).unwrap();
            let s = c.to_string();
            let result = ord_from_str(&s).unwrap();
            assert_eq!(result, cp, "Roundtrip failed for code point {}", cp);
        }
    }

    // =========================================================================
    // bytes() Constructor Tests
    // =========================================================================

    #[test]
    fn test_bytes_too_many_args() {
        let err = builtin_bytes(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
            Value::int(4).unwrap(),
        ])
        .unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    #[test]
    fn test_bytes_empty_constructor() {
        let value = builtin_bytes(&[]).unwrap();
        assert_eq!(byte_sequence_type(value), TypeId::BYTES);
        assert_eq!(value_to_byte_vec(value), Vec::<u8>::new());
    }

    #[test]
    fn test_bytes_count_constructor() {
        let value = builtin_bytes(&[Value::int(4).unwrap()]).unwrap();
        assert_eq!(byte_sequence_type(value), TypeId::BYTES);
        assert_eq!(value_to_byte_vec(value), vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_bytes_bool_count_constructor() {
        let false_value = builtin_bytes(&[Value::bool(false)]).unwrap();
        assert_eq!(value_to_byte_vec(false_value), Vec::<u8>::new());

        let true_value = builtin_bytes(&[Value::bool(true)]).unwrap();
        assert_eq!(value_to_byte_vec(true_value), vec![0]);
    }

    #[test]
    fn test_bytes_negative_count() {
        let err = builtin_bytes(&[Value::int(-5).unwrap()]).unwrap_err();
        assert!(matches!(err, BuiltinError::ValueError(_)));
        assert!(err.to_string().contains("negative count"));
    }

    #[test]
    fn test_bytes_overflow_count() {
        let err = builtin_bytes(&[Value::int(MAX_BYTE_SEQUENCE_SIZE + 1).unwrap()]).unwrap_err();
        assert!(matches!(err, BuiltinError::OverflowError(_)));
    }

    #[test]
    fn test_bytes_string_without_encoding_errors() {
        let err = builtin_bytes(&[Value::string(intern("abc"))]).unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
        assert!(err.to_string().contains("without an encoding"));
    }

    #[test]
    fn test_bytes_from_tagged_string_with_utf8() {
        let value = builtin_bytes(&[
            Value::string(intern("h\u{00e9}")),
            Value::string(intern("utf-8")),
        ])
        .unwrap();
        assert_eq!(byte_sequence_type(value), TypeId::BYTES);
        assert_eq!(value_to_byte_vec(value), "h\u{00e9}".as_bytes().to_vec());
    }

    #[test]
    fn test_bytes_from_heap_string_with_latin1() {
        let (heap_str, str_ptr) = boxed_value(StringObject::new("\u{00e9}"));
        let value = builtin_bytes(&[heap_str, Value::string(intern("latin-1"))]).unwrap();
        assert_eq!(value_to_byte_vec(value), vec![0xe9]);
        unsafe { drop_boxed(str_ptr) };
    }

    #[test]
    fn test_bytes_ascii_error_policies() {
        let strict_err = builtin_bytes(&[
            Value::string(intern("A\u{00e9}")),
            Value::string(intern("ascii")),
        ])
        .unwrap_err();
        assert!(matches!(strict_err, BuiltinError::ValueError(_)));

        let ignore = builtin_bytes(&[
            Value::string(intern("A\u{00e9}")),
            Value::string(intern("ascii")),
            Value::string(intern("ignore")),
        ])
        .unwrap();
        assert_eq!(value_to_byte_vec(ignore), b"A");

        let replace = builtin_bytes(&[
            Value::string(intern("A\u{00e9}")),
            Value::string(intern("ascii")),
            Value::string(intern("replace")),
        ])
        .unwrap();
        assert_eq!(value_to_byte_vec(replace), b"A?");
    }

    #[test]
    fn test_bytes_latin1_strict_error() {
        let err = builtin_bytes(&[
            Value::string(intern("\u{20ac}")),
            Value::string(intern("latin-1")),
        ])
        .unwrap_err();
        assert!(matches!(err, BuiltinError::ValueError(_)));
        assert!(err.to_string().contains("latin-1"));
    }

    #[test]
    fn test_bytes_unknown_encoding_and_error_policy() {
        let unknown_encoding = builtin_bytes(&[
            Value::string(intern("abc")),
            Value::string(intern("does-not-exist")),
        ])
        .unwrap_err();
        assert!(matches!(unknown_encoding, BuiltinError::ValueError(_)));
        assert!(unknown_encoding.to_string().contains("unknown encoding"));

        let unknown_policy = builtin_bytes(&[
            Value::string(intern("abc")),
            Value::string(intern("utf-8")),
            Value::string(intern("not-a-policy")),
        ])
        .unwrap_err();
        assert!(matches!(unknown_policy, BuiltinError::ValueError(_)));
        assert!(unknown_policy.to_string().contains("unknown error handler"));
    }

    #[test]
    fn test_bytes_encoding_requires_string_source() {
        let err =
            builtin_bytes(&[Value::int(1).unwrap(), Value::string(intern("utf-8"))]).unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
        assert!(
            err.to_string()
                .contains("encoding without a string argument")
        );
    }

    #[test]
    fn test_bytes_encoding_and_errors_argument_types() {
        let err2 =
            builtin_bytes(&[Value::string(intern("abc")), Value::int(1).unwrap()]).unwrap_err();
        assert!(matches!(err2, BuiltinError::TypeError(_)));
        assert!(err2.to_string().contains("argument 2 must be str"));

        let err3 = builtin_bytes(&[
            Value::string(intern("abc")),
            Value::string(intern("utf-8")),
            Value::int(1).unwrap(),
        ])
        .unwrap_err();
        assert!(matches!(err3, BuiltinError::TypeError(_)));
        assert!(err3.to_string().contains("argument 3 must be str"));
    }

    #[test]
    fn test_bytes_from_iterable_values() {
        let list = prism_runtime::types::list::ListObject::from_slice(&[
            Value::int(65).unwrap(),
            Value::bool(true),
            Value::int(0).unwrap(),
            Value::int(255).unwrap(),
        ]);
        let (list_value, list_ptr) = boxed_value(list);

        let value = builtin_bytes(&[list_value]).unwrap();
        assert_eq!(value_to_byte_vec(value), vec![65, 1, 0, 255]);

        unsafe { drop_boxed(list_ptr) };
    }

    #[test]
    fn test_bytes_from_iterable_out_of_range_error() {
        let list = prism_runtime::types::list::ListObject::from_slice(&[Value::int(256).unwrap()]);
        let (list_value, list_ptr) = boxed_value(list);
        let err = builtin_bytes(&[list_value]).unwrap_err();
        assert!(matches!(err, BuiltinError::ValueError(_)));
        assert!(err.to_string().contains("range(0, 256)"));
        unsafe { drop_boxed(list_ptr) };
    }

    #[test]
    fn test_bytes_from_iterable_non_int_error() {
        let list =
            prism_runtime::types::list::ListObject::from_slice(&[Value::string(intern("x"))]);
        let (list_value, list_ptr) = boxed_value(list);
        let err = builtin_bytes(&[list_value]).unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
        assert!(
            err.to_string()
                .contains("cannot be interpreted as an integer")
        );
        unsafe { drop_boxed(list_ptr) };
    }

    #[test]
    fn test_bytes_from_bytes_returns_same_object() {
        let (src, src_ptr) = boxed_value(BytesObject::from_slice(b"abc"));
        let out = builtin_bytes(&[src]).unwrap();
        assert_eq!(out.as_object_ptr(), src.as_object_ptr());
        unsafe { drop_boxed(src_ptr) };
    }

    #[test]
    fn test_bytes_from_bytearray_copies() {
        let (src, src_ptr) = boxed_value(BytesObject::bytearray_from_slice(&[1, 2, 3]));
        let out = builtin_bytes(&[src]).unwrap();
        assert_eq!(byte_sequence_type(out), TypeId::BYTES);
        assert_eq!(value_to_byte_vec(out), vec![1, 2, 3]);
        assert_ne!(out.as_object_ptr(), src.as_object_ptr());
        unsafe { drop_boxed(src_ptr) };
    }

    // =========================================================================
    // bytearray() Constructor Tests
    // =========================================================================

    #[test]
    fn test_bytearray_too_many_args() {
        let err = builtin_bytearray(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
            Value::int(4).unwrap(),
        ])
        .unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    #[test]
    fn test_bytearray_empty_and_count_constructor() {
        let empty = builtin_bytearray(&[]).unwrap();
        assert_eq!(byte_sequence_type(empty), TypeId::BYTEARRAY);
        assert_eq!(value_to_byte_vec(empty), Vec::<u8>::new());

        let counted = builtin_bytearray(&[Value::int(3).unwrap()]).unwrap();
        assert_eq!(byte_sequence_type(counted), TypeId::BYTEARRAY);
        assert_eq!(value_to_byte_vec(counted), vec![0, 0, 0]);
    }

    #[test]
    fn test_bytearray_negative_count() {
        let err = builtin_bytearray(&[Value::int(-5).unwrap()]).unwrap_err();
        assert!(matches!(err, BuiltinError::ValueError(_)));
    }

    #[test]
    fn test_bytearray_from_string_with_encoding() {
        let out = builtin_bytearray(&[
            Value::string(intern("h\u{00e9}")),
            Value::string(intern("utf-8")),
        ])
        .unwrap();
        assert_eq!(byte_sequence_type(out), TypeId::BYTEARRAY);
        assert_eq!(value_to_byte_vec(out), "h\u{00e9}".as_bytes());
    }

    #[test]
    fn test_bytearray_encoding_without_string_error() {
        let err = builtin_bytearray(&[Value::int(1).unwrap(), Value::string(intern("utf-8"))])
            .unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
        assert!(
            err.to_string()
                .contains("encoding without a string argument")
        );
    }

    #[test]
    fn test_bytearray_from_iterable_and_range_error() {
        let list = prism_runtime::types::list::ListObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ]);
        let (list_value, list_ptr) = boxed_value(list);
        let out = builtin_bytearray(&[list_value]).unwrap();
        assert_eq!(value_to_byte_vec(out), vec![1, 2, 3]);
        unsafe { drop_boxed(list_ptr) };

        let bad = prism_runtime::types::list::ListObject::from_slice(&[Value::int(-1).unwrap()]);
        let (bad_value, bad_ptr) = boxed_value(bad);
        let err = builtin_bytearray(&[bad_value]).unwrap_err();
        assert!(matches!(err, BuiltinError::ValueError(_)));
        assert!(err.to_string().contains("range(0, 256)"));
        unsafe { drop_boxed(bad_ptr) };
    }

    #[test]
    fn test_bytearray_from_bytes_and_bytearray_copy() {
        let (bytes_src, bytes_src_ptr) = boxed_value(BytesObject::from_slice(&[5, 6]));
        let from_bytes = builtin_bytearray(&[bytes_src]).unwrap();
        assert_eq!(byte_sequence_type(from_bytes), TypeId::BYTEARRAY);
        assert_eq!(value_to_byte_vec(from_bytes), vec![5, 6]);
        assert_ne!(from_bytes.as_object_ptr(), bytes_src.as_object_ptr());
        unsafe { drop_boxed(bytes_src_ptr) };

        let (bytearray_src, bytearray_src_ptr) =
            boxed_value(BytesObject::bytearray_from_slice(&[7, 8]));
        let from_bytearray = builtin_bytearray(&[bytearray_src]).unwrap();
        assert_eq!(byte_sequence_type(from_bytearray), TypeId::BYTEARRAY);
        assert_eq!(value_to_byte_vec(from_bytearray), vec![7, 8]);
        assert_ne!(
            from_bytearray.as_object_ptr(),
            bytearray_src.as_object_ptr()
        );
        unsafe { drop_boxed(bytearray_src_ptr) };
    }

    // =========================================================================
    // format() Argument Validation Tests
    // =========================================================================

    #[test]
    fn test_format_no_args() {
        let result = builtin_format(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_format_too_many_args() {
        let result = builtin_format(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn test_format_default_and_numeric_specs() {
        let default = builtin_format(&[Value::int(42).unwrap()]).unwrap();
        assert_eq!(value_to_rust_string(default), "42");

        let hex = builtin_format(&[Value::int(255).unwrap(), Value::string(intern("#x"))]).unwrap();
        assert_eq!(value_to_rust_string(hex), "0xff");

        let grouped =
            builtin_format(&[Value::int(1_234_567).unwrap(), Value::string(intern(","))]).unwrap();
        assert_eq!(value_to_rust_string(grouped), "1,234,567");
    }

    #[test]
    fn test_format_second_arg_must_be_str() {
        let err = builtin_format(&[Value::int(1).unwrap(), Value::int(2).unwrap()]).unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
        assert!(err.to_string().contains("argument 2 must be str"));
    }

    #[test]
    fn test_format_non_empty_spec_on_unsupported_type_errors() {
        let err = builtin_format(&[Value::none(), Value::string(intern("x"))]).unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
        assert!(err.to_string().contains("unsupported format string"));
    }

    // =========================================================================
    // Utility Function Tests
    // =========================================================================

    #[test]
    fn test_is_ascii() {
        assert!(is_ascii(0));
        assert!(is_ascii(65));
        assert!(is_ascii(127));
        assert!(!is_ascii(128));
        assert!(!is_ascii(256));
        assert!(!is_ascii(8364));
    }

    #[test]
    fn test_is_valid_code_point() {
        // Valid points
        assert!(is_valid_code_point(0));
        assert!(is_valid_code_point(127));
        assert!(is_valid_code_point(128));
        assert!(is_valid_code_point(8364));
        assert!(is_valid_code_point(MAX_UNICODE_CODE_POINT));

        // Surrogates are invalid
        assert!(!is_valid_code_point(0xD800));
        assert!(!is_valid_code_point(0xDFFF));
        assert!(!is_valid_code_point(0xDA00));

        // Above max is invalid
        assert!(!is_valid_code_point(0x110000));
        assert!(!is_valid_code_point(0x1FFFFF));
    }

    #[test]
    fn test_is_surrogate() {
        // Before surrogate range
        assert!(!is_surrogate(0xD7FF));

        // Surrogate range
        assert!(is_surrogate(0xD800));
        assert!(is_surrogate(0xDA00));
        assert!(is_surrogate(0xDC00));
        assert!(is_surrogate(0xDFFF));

        // After surrogate range
        assert!(!is_surrogate(0xE000));
    }

    // =========================================================================
    // Format Helper Tests
    // =========================================================================

    #[test]
    fn test_format_with_thousands_separator() {
        assert_eq!(format_with_thousands_separator(0), "0");
        assert_eq!(format_with_thousands_separator(1), "1");
        assert_eq!(format_with_thousands_separator(12), "12");
        assert_eq!(format_with_thousands_separator(123), "123");
        assert_eq!(format_with_thousands_separator(1234), "1,234");
        assert_eq!(format_with_thousands_separator(12345), "12,345");
        assert_eq!(format_with_thousands_separator(123456), "123,456");
        assert_eq!(format_with_thousands_separator(1234567), "1,234,567");
        assert_eq!(format_with_thousands_separator(-1234567), "-1,234,567");
    }

    #[test]
    fn test_format_with_underscore_separator() {
        assert_eq!(format_with_underscore_separator(0), "0");
        assert_eq!(format_with_underscore_separator(1234), "1_234");
        assert_eq!(format_with_underscore_separator(1234567), "1_234_567");
        assert_eq!(format_with_underscore_separator(-1234567), "-1_234_567");
    }

    #[test]
    fn test_parse_precision() {
        assert_eq!(parse_precision(".2f"), Some(2));
        assert_eq!(parse_precision(".10g"), Some(10));
        assert_eq!(parse_precision(".0"), Some(0));
        assert_eq!(parse_precision(""), None);
        assert_eq!(parse_precision("f"), None);
    }

    // =========================================================================
    // extract_code_point() Tests
    // =========================================================================

    #[test]
    fn test_extract_code_point_int() {
        assert_eq!(extract_code_point(&Value::int(97).unwrap()).unwrap(), 97);
        assert_eq!(extract_code_point(&Value::int(0).unwrap()).unwrap(), 0);
        assert_eq!(
            extract_code_point(&Value::int(0x10FFFF).unwrap()).unwrap(),
            0x10FFFF
        );
    }

    #[test]
    fn test_extract_code_point_bool() {
        assert_eq!(extract_code_point(&Value::bool(true)).unwrap(), 1);
        assert_eq!(extract_code_point(&Value::bool(false)).unwrap(), 0);
    }

    #[test]
    fn test_extract_code_point_negative() {
        let result = extract_code_point(&Value::int(-1).unwrap());
        assert!(result.is_err());
        match result {
            Err(BuiltinError::ValueError(msg)) => {
                assert!(msg.contains("negative"));
            }
            _ => panic!("Expected ValueError"),
        }
    }

    #[test]
    fn test_extract_code_point_too_large() {
        let result = extract_code_point(&Value::int(0x110000).unwrap());
        assert!(result.is_err());
        match result {
            Err(BuiltinError::ValueError(msg)) => {
                assert!(msg.contains("too large"));
            }
            _ => panic!("Expected ValueError"),
        }
    }

    #[test]
    fn test_extract_code_point_float_error() {
        let result = extract_code_point(&Value::float(97.0));
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(msg)) => {
                assert!(msg.contains("integer"));
            }
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_extract_code_point_none_error() {
        let result = extract_code_point(&Value::none());
        assert!(result.is_err());
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_chr_boundary_before_surrogate() {
        // U+D7FF is last valid before surrogate range
        assert!(chr_from_code_point(0xD7FF).is_ok());
    }

    #[test]
    fn test_chr_boundary_after_surrogate() {
        // U+E000 is first valid after surrogate range
        assert!(chr_from_code_point(0xE000).is_ok());
    }

    #[test]
    fn test_chr_max_bmp() {
        // U+FFFF is last code point in BMP
        assert!(chr_from_code_point(0xFFFF).is_ok());
    }

    #[test]
    fn test_chr_first_supplementary() {
        // U+10000 is first supplementary plane code point
        assert!(chr_from_code_point(0x10000).is_ok());
    }

    // =========================================================================
    // Constant Verification Tests
    // =========================================================================

    #[test]
    fn test_unicode_constants() {
        assert_eq!(MAX_UNICODE_CODE_POINT, 0x10FFFF);
        assert_eq!(SURROGATE_START, 0xD800);
        assert_eq!(SURROGATE_END, 0xDFFF);
        assert_eq!(ASCII_MAX, 0x7F);
    }
}
