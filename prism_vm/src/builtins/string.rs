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
use crate::error::RuntimeError;
use crate::stdlib::exceptions::ExceptionTypeId;
use prism_core::Value;
use prism_core::intern::{intern, interned_by_ptr};
use prism_core::python_unicode::{
    PYTHON_SURROGATE_END, PYTHON_SURROGATE_START, contains_surrogate_carriers,
    encode_python_code_point, is_python_surrogate, logical_python_code_point,
    python_code_point_escape,
};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::bytes::{BytesObject, value_as_bytes_ref};
use prism_runtime::types::memoryview::value_as_memoryview_ref;
use prism_runtime::types::string::StringObject;

// =============================================================================
// Unicode Constants
// =============================================================================

/// Maximum valid Unicode code point (U+10FFFF).
const MAX_UNICODE_CODE_POINT: u32 = 0x10FFFF;

/// Start of surrogate range (not valid for encoding).
const SURROGATE_START: u32 = PYTHON_SURROGATE_START;

/// End of surrogate range (not valid for encoding).
const SURROGATE_END: u32 = PYTHON_SURROGATE_END;

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
        byte_sequence_value(self.type_id(), data)
    }
}

#[inline]
pub(crate) fn byte_sequence_value(type_id: TypeId, data: Vec<u8>) -> Value {
    let obj = BytesObject::from_vec_with_type(data, type_id);
    let ptr = Box::leak(Box::new(obj)) as *mut BytesObject as *const ();
    Value::object_ptr(ptr)
}

#[inline]
pub(crate) fn encode_text_to_data(
    input: &str,
    encoding: Option<&str>,
    errors: Option<&str>,
) -> Result<Vec<u8>, BuiltinError> {
    encode_string(
        input,
        encoding.unwrap_or("utf-8"),
        errors.unwrap_or("strict"),
    )
}

#[inline]
pub(crate) fn encode_text_to_value(
    input: &str,
    encoding: Option<&str>,
    errors: Option<&str>,
    type_id: TypeId,
) -> Result<Value, BuiltinError> {
    let data = encode_text_to_data(input, encoding, errors)?;
    Ok(byte_sequence_value(type_id, data))
}

#[inline]
pub(crate) fn decode_bytes_to_text(
    input: &[u8],
    encoding: Option<&str>,
    errors: Option<&str>,
) -> Result<String, BuiltinError> {
    decode_string(
        input,
        encoding.unwrap_or("utf-8"),
        errors.unwrap_or("strict"),
    )
}

#[inline]
pub(crate) fn decode_bytes_to_value(
    input: &[u8],
    encoding: Option<&str>,
    errors: Option<&str>,
) -> Result<Value, BuiltinError> {
    Ok(text_value(decode_bytes_to_text(input, encoding, errors)?))
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
            Ok(logical_python_code_point(c as u32))
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
    if is_python_surrogate(code_point) {
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

        return encode_text_to_value(&source, Some(&encoding), Some(&errors), kind.type_id());
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
            TypeId::MEMORYVIEW => {
                let view = value_as_memoryview_ref(source).ok_or_else(|| {
                    BuiltinError::TypeError("invalid memoryview object".to_string())
                })?;
                if view.released() {
                    return Err(BuiltinError::ValueError(
                        "operation forbidden on released memoryview object".to_string(),
                    ));
                }
                return Ok(kind.from_data(view.to_vec()));
            }
            _ => {}
        }

        if let Some(bytes) = value_as_bytes_ref(source) {
            return Ok(kind.from_data(bytes.to_vec()));
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TextCodecErrorPolicy {
    Strict,
    Ignore,
    Replace,
    BackslashReplace,
    SurrogateEscape,
    SurrogatePass,
}

#[inline]
fn parse_encoding_error_policy(errors: &str) -> Result<TextCodecErrorPolicy, BuiltinError> {
    match errors.to_ascii_lowercase().as_str() {
        "strict" => Ok(TextCodecErrorPolicy::Strict),
        "ignore" => Ok(TextCodecErrorPolicy::Ignore),
        "replace" => Ok(TextCodecErrorPolicy::Replace),
        "backslashreplace" => Ok(TextCodecErrorPolicy::BackslashReplace),
        "surrogateescape" => Ok(TextCodecErrorPolicy::SurrogateEscape),
        "surrogatepass" => Ok(TextCodecErrorPolicy::SurrogatePass),
        _ => Err(BuiltinError::ValueError(format!(
            "unknown error handler name '{}'",
            errors
        ))),
    }
}

fn encode_string(input: &str, encoding: &str, errors: &str) -> Result<Vec<u8>, BuiltinError> {
    let normalized = encoding.trim().to_ascii_lowercase().replace('_', "-");
    let policy = parse_encoding_error_policy(errors)?;

    match normalized.as_str() {
        "utf8" | "utf-8" => encode_utf8(input, policy, "utf-8"),
        "ascii" => encode_ascii(input, policy, "ascii"),
        "latin1" | "latin-1" | "iso-8859-1" => encode_latin1(input, policy, "latin-1"),
        "raw-unicode-escape" => Ok(encode_raw_unicode_escape(input)),
        _ => Err(BuiltinError::ValueError(format!(
            "unknown encoding: {}",
            encoding
        ))),
    }
}

fn decode_string(input: &[u8], encoding: &str, errors: &str) -> Result<String, BuiltinError> {
    let normalized = encoding.trim().to_ascii_lowercase().replace('_', "-");
    let policy = parse_encoding_error_policy(errors)?;

    match normalized.as_str() {
        "utf8" | "utf-8" => decode_utf8(input, policy, "utf-8"),
        "ascii" => decode_ascii(input, policy, "ascii"),
        "latin1" | "latin-1" | "iso-8859-1" => decode_latin1(input),
        "raw-unicode-escape" => decode_raw_unicode_escape(input, policy),
        _ => Err(BuiltinError::ValueError(format!(
            "unknown encoding: {}",
            encoding
        ))),
    }
}

fn encode_utf8(
    input: &str,
    policy: TextCodecErrorPolicy,
    codec_name: &'static str,
) -> Result<Vec<u8>, BuiltinError> {
    if !contains_surrogate_carriers(input) {
        return Ok(input.as_bytes().to_vec());
    }

    let mut out = Vec::with_capacity(input.len());
    for (position, ch) in input.chars().enumerate() {
        let code = logical_python_code_point(ch as u32);
        if is_python_surrogate(code) {
            match policy {
                TextCodecErrorPolicy::Strict => {
                    return Err(unicode_encode_error(
                        codec_name,
                        ch,
                        position,
                        "surrogates not allowed",
                    ));
                }
                TextCodecErrorPolicy::Ignore => continue,
                TextCodecErrorPolicy::Replace => {
                    out.push(b'?');
                    continue;
                }
                TextCodecErrorPolicy::BackslashReplace => {
                    append_backslashreplace(code, &mut out);
                    continue;
                }
                TextCodecErrorPolicy::SurrogateEscape => {
                    if let Some(byte) = surrogateescape_byte(code) {
                        out.push(byte);
                        continue;
                    }
                    return Err(unicode_encode_error(
                        codec_name,
                        ch,
                        position,
                        "surrogates not allowed",
                    ));
                }
                TextCodecErrorPolicy::SurrogatePass => {
                    append_utf8_code_point(code, &mut out);
                    continue;
                }
            }
        }

        append_utf8_code_point(code, &mut out);
    }
    Ok(out)
}

fn encode_ascii(
    input: &str,
    policy: TextCodecErrorPolicy,
    codec_name: &'static str,
) -> Result<Vec<u8>, BuiltinError> {
    let mut out = Vec::with_capacity(input.len());
    for (position, ch) in input.chars().enumerate() {
        let code = logical_python_code_point(ch as u32);
        if code <= 0x7f {
            out.push(code as u8);
            continue;
        }
        match policy {
            TextCodecErrorPolicy::Strict | TextCodecErrorPolicy::SurrogatePass => {
                return Err(unicode_encode_error(
                    codec_name,
                    ch,
                    position,
                    "ordinal not in range(128)",
                ));
            }
            TextCodecErrorPolicy::Ignore => {}
            TextCodecErrorPolicy::Replace => out.push(b'?'),
            TextCodecErrorPolicy::BackslashReplace => {
                append_backslashreplace(code, &mut out);
            }
            TextCodecErrorPolicy::SurrogateEscape => {
                if let Some(byte) = surrogateescape_byte(code) {
                    out.push(byte);
                } else {
                    return Err(unicode_encode_error(
                        codec_name,
                        ch,
                        position,
                        "ordinal not in range(128)",
                    ));
                }
            }
        }
    }
    Ok(out)
}

fn encode_latin1(
    input: &str,
    policy: TextCodecErrorPolicy,
    codec_name: &'static str,
) -> Result<Vec<u8>, BuiltinError> {
    let mut out = Vec::with_capacity(input.len());
    for (position, ch) in input.chars().enumerate() {
        let code = logical_python_code_point(ch as u32);
        if code <= 0xff {
            out.push(code as u8);
            continue;
        }
        match policy {
            TextCodecErrorPolicy::Strict | TextCodecErrorPolicy::SurrogatePass => {
                return Err(unicode_encode_error(
                    codec_name,
                    ch,
                    position,
                    "ordinal not in range(256)",
                ));
            }
            TextCodecErrorPolicy::Ignore => {}
            TextCodecErrorPolicy::Replace => out.push(b'?'),
            TextCodecErrorPolicy::BackslashReplace => {
                append_backslashreplace(code, &mut out);
            }
            TextCodecErrorPolicy::SurrogateEscape => {
                if let Some(byte) = surrogateescape_byte(code) {
                    out.push(byte);
                } else {
                    return Err(unicode_encode_error(
                        codec_name,
                        ch,
                        position,
                        "ordinal not in range(256)",
                    ));
                }
            }
        }
    }
    Ok(out)
}

fn encode_raw_unicode_escape(input: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len());
    for ch in input.chars() {
        let code = logical_python_code_point(ch as u32);
        match code {
            0x00..=0xFF => out.push(code as u8),
            0x0100..=0xFFFF => append_hex_escape(&mut out, b'u', code, 4),
            _ => append_hex_escape(&mut out, b'U', code, 8),
        }
    }
    out
}

fn decode_raw_unicode_escape(
    input: &[u8],
    policy: TextCodecErrorPolicy,
) -> Result<String, BuiltinError> {
    let mut out = String::with_capacity(input.len());
    let mut index = 0usize;

    while index < input.len() {
        let byte = input[index];
        if byte != b'\\' || index + 1 >= input.len() {
            out.push(byte as char);
            index += 1;
            continue;
        }

        let marker = input[index + 1];
        let width = match marker {
            b'u' => 4,
            b'U' => 8,
            _ => {
                out.push(byte as char);
                index += 1;
                continue;
            }
        };

        let start = index;
        let digits_start = index + 2;
        let mut digits = 0usize;
        let mut code = 0u32;
        while digits < width && digits_start + digits < input.len() {
            let Some(nibble) = hex_nibble(input[digits_start + digits]) else {
                break;
            };
            code = (code << 4) | u32::from(nibble);
            digits += 1;
        }

        if digits != width {
            let consumed = 2 + digits;
            handle_raw_unicode_escape_error(
                input,
                start,
                consumed,
                policy,
                &mut out,
                if marker == b'u' {
                    "truncated \\uXXXX escape"
                } else {
                    "truncated \\UXXXXXXXX escape"
                },
            )?;
            index += consumed;
            continue;
        }

        match encode_python_code_point(code) {
            Some(ch) => {
                out.push(ch);
                index += 2 + width;
            }
            None => {
                handle_raw_unicode_escape_error(
                    input,
                    start,
                    2 + width,
                    policy,
                    &mut out,
                    if marker == b'u' {
                        "\\uxxxx out of range"
                    } else {
                        "\\Uxxxxxxxx out of range"
                    },
                )?;
                index += 2 + width;
            }
        }
    }

    Ok(out)
}

fn handle_raw_unicode_escape_error(
    input: &[u8],
    start: usize,
    consumed: usize,
    policy: TextCodecErrorPolicy,
    out: &mut String,
    reason: &'static str,
) -> Result<(), BuiltinError> {
    match policy {
        TextCodecErrorPolicy::Ignore => Ok(()),
        TextCodecErrorPolicy::Replace => {
            out.push('\u{FFFD}');
            Ok(())
        }
        TextCodecErrorPolicy::BackslashReplace => {
            for &byte in &input[start..start + consumed] {
                append_backslashreplace_byte(byte, out);
            }
            Ok(())
        }
        TextCodecErrorPolicy::Strict
        | TextCodecErrorPolicy::SurrogateEscape
        | TextCodecErrorPolicy::SurrogatePass => Err(unicode_decode_error(
            "raw-unicode-escape",
            input[start],
            start,
            reason,
        )),
    }
}

fn decode_ascii(
    input: &[u8],
    policy: TextCodecErrorPolicy,
    codec_name: &'static str,
) -> Result<String, BuiltinError> {
    let mut out = String::with_capacity(input.len());
    for (position, &byte) in input.iter().enumerate() {
        if byte <= ASCII_MAX as u8 {
            out.push(byte as char);
            continue;
        }

        match policy {
            TextCodecErrorPolicy::Strict | TextCodecErrorPolicy::SurrogatePass => {
                return Err(unicode_decode_error(
                    codec_name,
                    byte,
                    position,
                    "ordinal not in range(128)",
                ));
            }
            TextCodecErrorPolicy::Ignore => {}
            TextCodecErrorPolicy::Replace => out.push('\u{FFFD}'),
            TextCodecErrorPolicy::BackslashReplace => append_backslashreplace_byte(byte, &mut out),
            TextCodecErrorPolicy::SurrogateEscape => {
                push_python_code_point(&mut out, 0xDC00 + u32::from(byte));
            }
        }
    }
    Ok(out)
}

#[inline]
fn decode_latin1(input: &[u8]) -> Result<String, BuiltinError> {
    let mut out = String::with_capacity(input.len());
    for &byte in input {
        out.push(byte as char);
    }
    Ok(out)
}

fn decode_utf8(
    input: &[u8],
    policy: TextCodecErrorPolicy,
    codec_name: &'static str,
) -> Result<String, BuiltinError> {
    let mut out = String::with_capacity(input.len());
    let allow_surrogates = matches!(policy, TextCodecErrorPolicy::SurrogatePass);
    let mut index = 0usize;

    while index < input.len() {
        match decode_utf8_code_point(&input[index..], allow_surrogates) {
            Ok((code_point, len)) => {
                push_python_code_point(&mut out, code_point);
                index += len;
            }
            Err(err) => {
                let error_index = index + err.byte_offset;
                let offending = input[error_index];
                match policy {
                    TextCodecErrorPolicy::Strict | TextCodecErrorPolicy::SurrogatePass => {
                        return Err(unicode_decode_error(
                            codec_name,
                            offending,
                            error_index,
                            err.reason,
                        ));
                    }
                    TextCodecErrorPolicy::Ignore => {
                        index += 1;
                    }
                    TextCodecErrorPolicy::Replace => {
                        out.push('\u{FFFD}');
                        index += 1;
                    }
                    TextCodecErrorPolicy::BackslashReplace => {
                        append_backslashreplace_byte(offending, &mut out);
                        index += 1;
                    }
                    TextCodecErrorPolicy::SurrogateEscape => {
                        push_python_code_point(&mut out, 0xDC00 + u32::from(input[index]));
                        index += 1;
                    }
                }
            }
        }
    }

    Ok(out)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Utf8DecodeSequenceError {
    byte_offset: usize,
    reason: &'static str,
}

fn decode_utf8_code_point(
    input: &[u8],
    allow_surrogates: bool,
) -> Result<(u32, usize), Utf8DecodeSequenceError> {
    let first = input[0];
    match first {
        0x00..=0x7F => Ok((u32::from(first), 1)),
        0xC2..=0xDF => {
            let second = required_continuation(input, 1)?;
            Ok(((u32::from(first & 0x1F) << 6) | u32::from(second & 0x3F), 2))
        }
        0xE0 => {
            let second = required_byte_in_range(input, 1, 0xA0..=0xBF)?;
            let third = required_continuation(input, 2)?;
            Ok((
                (u32::from(first & 0x0F) << 12)
                    | (u32::from(second & 0x3F) << 6)
                    | u32::from(third & 0x3F),
                3,
            ))
        }
        0xE1..=0xEC | 0xEE..=0xEF => {
            let second = required_continuation(input, 1)?;
            let third = required_continuation(input, 2)?;
            Ok((
                (u32::from(first & 0x0F) << 12)
                    | (u32::from(second & 0x3F) << 6)
                    | u32::from(third & 0x3F),
                3,
            ))
        }
        0xED => {
            let second = if allow_surrogates {
                required_continuation(input, 1)?
            } else {
                required_byte_in_range(input, 1, 0x80..=0x9F)?
            };
            let third = required_continuation(input, 2)?;
            Ok((
                (u32::from(first & 0x0F) << 12)
                    | (u32::from(second & 0x3F) << 6)
                    | u32::from(third & 0x3F),
                3,
            ))
        }
        0xF0 => {
            let second = required_byte_in_range(input, 1, 0x90..=0xBF)?;
            let third = required_continuation(input, 2)?;
            let fourth = required_continuation(input, 3)?;
            Ok((
                (u32::from(first & 0x07) << 18)
                    | (u32::from(second & 0x3F) << 12)
                    | (u32::from(third & 0x3F) << 6)
                    | u32::from(fourth & 0x3F),
                4,
            ))
        }
        0xF1..=0xF3 => {
            let second = required_continuation(input, 1)?;
            let third = required_continuation(input, 2)?;
            let fourth = required_continuation(input, 3)?;
            Ok((
                (u32::from(first & 0x07) << 18)
                    | (u32::from(second & 0x3F) << 12)
                    | (u32::from(third & 0x3F) << 6)
                    | u32::from(fourth & 0x3F),
                4,
            ))
        }
        0xF4 => {
            let second = required_byte_in_range(input, 1, 0x80..=0x8F)?;
            let third = required_continuation(input, 2)?;
            let fourth = required_continuation(input, 3)?;
            Ok((
                (u32::from(first & 0x07) << 18)
                    | (u32::from(second & 0x3F) << 12)
                    | (u32::from(third & 0x3F) << 6)
                    | u32::from(fourth & 0x3F),
                4,
            ))
        }
        _ => Err(Utf8DecodeSequenceError {
            byte_offset: 0,
            reason: "invalid start byte",
        }),
    }
}

#[inline]
fn required_continuation(input: &[u8], offset: usize) -> Result<u8, Utf8DecodeSequenceError> {
    required_byte_in_range(input, offset, 0x80..=0xBF)
}

#[inline]
fn required_byte_in_range(
    input: &[u8],
    offset: usize,
    range: std::ops::RangeInclusive<u8>,
) -> Result<u8, Utf8DecodeSequenceError> {
    let Some(&byte) = input.get(offset) else {
        return Err(Utf8DecodeSequenceError {
            byte_offset: input.len().saturating_sub(1),
            reason: "unexpected end of data",
        });
    };

    if range.contains(&byte) {
        Ok(byte)
    } else {
        Err(Utf8DecodeSequenceError {
            byte_offset: offset,
            reason: "invalid continuation byte",
        })
    }
}

#[inline]
fn append_utf8_code_point(code_point: u32, out: &mut Vec<u8>) {
    match code_point {
        0x0000..=0x007F => out.push(code_point as u8),
        0x0080..=0x07FF => {
            out.push(0xC0 | ((code_point >> 6) as u8));
            out.push(0x80 | ((code_point & 0x3F) as u8));
        }
        0x0800..=0xFFFF => {
            out.push(0xE0 | ((code_point >> 12) as u8));
            out.push(0x80 | (((code_point >> 6) & 0x3F) as u8));
            out.push(0x80 | ((code_point & 0x3F) as u8));
        }
        _ => {
            out.push(0xF0 | ((code_point >> 18) as u8));
            out.push(0x80 | (((code_point >> 12) & 0x3F) as u8));
            out.push(0x80 | (((code_point >> 6) & 0x3F) as u8));
            out.push(0x80 | ((code_point & 0x3F) as u8));
        }
    }
}

#[inline]
fn append_backslashreplace(code_point: u32, out: &mut Vec<u8>) {
    out.extend_from_slice(python_code_point_escape(code_point).as_bytes());
}

#[inline]
fn append_hex_escape(out: &mut Vec<u8>, marker: u8, code_point: u32, width: usize) {
    out.push(b'\\');
    out.push(marker);
    for shift in (0..width).rev() {
        let nibble = ((code_point >> (shift * 4)) & 0xF) as u8;
        out.push(hex_digit(nibble));
    }
}

#[inline]
fn hex_digit(nibble: u8) -> u8 {
    match nibble {
        0..=9 => b'0' + nibble,
        10..=15 => b'a' + (nibble - 10),
        _ => unreachable!("nibble should be in range"),
    }
}

#[inline]
fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[inline]
fn append_backslashreplace_byte(byte: u8, out: &mut String) {
    out.push('\\');
    out.push('x');
    const HEX: &[u8; 16] = b"0123456789abcdef";
    out.push(HEX[(byte >> 4) as usize] as char);
    out.push(HEX[(byte & 0x0F) as usize] as char);
}

#[inline]
fn push_python_code_point(out: &mut String, code_point: u32) {
    let ch = encode_python_code_point(code_point).expect("valid Python code point should encode");
    out.push(ch);
}

#[inline]
fn surrogateescape_byte(code_point: u32) -> Option<u8> {
    if (0xDC80..=0xDCFF).contains(&code_point) {
        Some((code_point - 0xDC00) as u8)
    } else {
        None
    }
}

#[inline]
fn text_value(text: String) -> Value {
    if text.is_ascii() {
        return Value::string(intern(&text));
    }

    let ptr =
        Box::leak(Box::new(StringObject::from_string(text))) as *mut StringObject as *const ();
    Value::object_ptr(ptr)
}

#[inline]
fn unicode_encode_error(
    codec_name: &'static str,
    ch: char,
    position: usize,
    reason: &'static str,
) -> BuiltinError {
    BuiltinError::Raised(RuntimeError::exception(
        ExceptionTypeId::UnicodeEncodeError.as_u8() as u16,
        format!(
            "'{codec_name}' codec can't encode character '{}' in position {position}: {reason}",
            python_unicode_escape(ch)
        ),
    ))
}

#[inline]
fn unicode_decode_error(
    codec_name: &'static str,
    byte: u8,
    position: usize,
    reason: &'static str,
) -> BuiltinError {
    BuiltinError::Raised(RuntimeError::exception(
        ExceptionTypeId::UnicodeDecodeError.as_u8() as u16,
        format!(
            "'{codec_name}' codec can't decode byte 0x{byte:02x} in position {position}: {reason}",
        ),
    ))
}

#[inline]
fn python_unicode_escape(ch: char) -> String {
    python_code_point_escape(ch as u32)
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

    if let Some(boolean) = value.as_bool() {
        return format_int(if boolean { 1 } else { 0 }, format_spec);
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NumericFormatSpec {
    alternate_form: bool,
    zero_pad: bool,
    width: Option<usize>,
    grouping: Option<char>,
    precision: Option<usize>,
    ty: Option<char>,
}

impl NumericFormatSpec {
    #[inline]
    fn parse(format_spec: &str) -> Result<Self, BuiltinError> {
        let mut chars = format_spec.chars().peekable();

        if matches!(chars.clone().nth(1), Some('<' | '>' | '^' | '=')) {
            chars.next();
            chars.next();
        } else if matches!(chars.peek(), Some('<' | '>' | '^' | '=')) {
            chars.next();
        }

        if matches!(chars.peek(), Some('+' | '-' | ' ')) {
            chars.next();
        }

        let alternate_form = matches!(chars.peek(), Some('#'));
        if alternate_form {
            chars.next();
        }

        let zero_pad = matches!(chars.peek(), Some('0'));
        if zero_pad {
            chars.next();
        }

        let mut width_digits = String::new();
        while matches!(chars.peek(), Some(ch) if ch.is_ascii_digit()) {
            width_digits.push(chars.next().expect("peeked digit must exist"));
        }
        let width = (!width_digits.is_empty())
            .then(|| width_digits.parse::<usize>())
            .transpose()
            .map_err(|_| {
                BuiltinError::ValueError(format!("invalid format specifier '{format_spec}'"))
            })?;

        let grouping = if matches!(chars.peek(), Some(',' | '_')) {
            chars.next()
        } else {
            None
        };

        let precision = if matches!(chars.peek(), Some('.')) {
            chars.next();
            let mut digits = String::new();
            while matches!(chars.peek(), Some(ch) if ch.is_ascii_digit()) {
                digits.push(chars.next().expect("peeked digit must exist"));
            }
            Some(digits.parse::<usize>().map_err(|_| {
                BuiltinError::ValueError(format!("invalid format specifier '{format_spec}'"))
            })?)
        } else {
            None
        };

        let ty = chars.next();
        if chars.next().is_some() {
            return Err(BuiltinError::ValueError(format!(
                "invalid format specifier '{format_spec}'"
            )));
        }

        Ok(Self {
            alternate_form,
            zero_pad,
            width,
            grouping,
            precision,
            ty,
        })
    }
}

/// Format an integer according to format_spec.
#[inline]
fn format_int(n: i64, format_spec: &str) -> Result<Value, BuiltinError> {
    let spec = NumericFormatSpec::parse(format_spec)?;
    let negative = n < 0;
    let magnitude = if negative {
        n.checked_abs().ok_or_else(|| {
            BuiltinError::OverflowError("integer absolute value overflow".to_string())
        })? as u64
    } else {
        n as u64
    };

    let (mut prefix, mut digits) = match spec.ty.unwrap_or('d') {
        'b' => (
            if spec.alternate_form { "0b" } else { "" },
            format!("{:b}", magnitude),
        ),
        'o' => (
            if spec.alternate_form { "0o" } else { "" },
            format!("{:o}", magnitude),
        ),
        'x' => (
            if spec.alternate_form { "0x" } else { "" },
            format!("{:x}", magnitude),
        ),
        'X' => (
            if spec.alternate_form { "0X" } else { "" },
            format!("{:X}", magnitude),
        ),
        'c' => {
            if n < 0 || n > MAX_UNICODE_CODE_POINT as i64 {
                return Err(BuiltinError::OverflowError(
                    "%c arg not in range(0x110000)".to_string(),
                ));
            }
            let ch = char::from_u32(n as u32).ok_or_else(|| {
                BuiltinError::OverflowError("%c arg not in range(0x110000)".to_string())
            })?;
            ("", ch.to_string())
        }
        'd' | 'n' => ("", format_decimal_digits(magnitude, spec.grouping)),
        other => {
            return Err(BuiltinError::ValueError(format!(
                "unknown format code '{other}' for object of type 'int'"
            )));
        }
    };

    if let Some(precision) = spec.precision {
        digits = format!(
            "{}{digits}",
            "0".repeat(precision.saturating_sub(digits.len()))
        );
    }

    let sign = if negative { "-" } else { "" };
    let core_len = sign.len() + prefix.len() + digits.len();
    let width = spec.width.unwrap_or(0);
    let padding_len = width.saturating_sub(core_len);
    let formatted = if spec.zero_pad && spec.precision.is_none() && spec.ty != Some('c') {
        format!("{sign}{prefix}{}{digits}", "0".repeat(padding_len))
    } else {
        let mut rendered = String::with_capacity(width.max(core_len));
        rendered.push_str(&" ".repeat(padding_len));
        rendered.push_str(sign);
        rendered.push_str(prefix);
        rendered.push_str(&digits);
        rendered
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

#[inline]
fn format_decimal_digits(n: u64, grouping: Option<char>) -> String {
    match grouping {
        Some(',') => format_unsigned_with_separator(n, ','),
        Some('_') => format_unsigned_with_separator(n, '_'),
        _ => n.to_string(),
    }
}

#[inline]
fn format_unsigned_with_separator(n: u64, separator: char) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (index, ch) in s.chars().enumerate() {
        if index > 0 && (s.len() - index) % 3 == 0 {
            result.push(separator);
        }
        result.push(ch);
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
    code_point <= MAX_UNICODE_CODE_POINT && !is_python_surrogate(code_point)
}

/// Check if a code point is in the surrogate range.
#[inline(always)]
pub const fn is_surrogate(code_point: u32) -> bool {
    is_python_surrogate(code_point)
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
mod tests;
