//! Numeric formatting builtins (bin, hex, oct, complex).
//!
//! High-performance implementations with zero-allocation fast paths
//! for common integer sizes. All functions are Python 3.12 compatible.
//!
//! # Performance Characteristics
//!
//! - **bin/hex/oct**: Zero heap allocation for integers ≤64 bits
//! - **Lookup tables**: Pre-computed nibble→char mappings avoid branching
//! - **Exact sizing**: Pre-compute output length to avoid reallocations
//!
//! # Python Semantics
//!
//! These builtins accept any object with `__index__` protocol, but for
//! maximum performance, we fast-path native integers. The fallback
//! path handles objects with `__index__` when the object system is wired.

use super::BuiltinError;
use crate::python_numeric::{ComplexParts, complex_like_parts, is_complex_value};
use prism_core::Value;
use prism_core::intern::{intern, interned_by_ptr};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::complex::ComplexObject;

// =============================================================================
// Lookup Tables (Zero-Branch Hex Conversion)
// =============================================================================

/// Lowercase hex digits lookup table.
/// Index by nibble value (0-15) to get ASCII char.
const HEX_CHARS_LOWER: [u8; 16] = *b"0123456789abcdef";

/// Uppercase hex digits lookup table (for potential future use).
#[allow(dead_code)]
const HEX_CHARS_UPPER: [u8; 16] = *b"0123456789ABCDEF";

/// Octal digits lookup table.
/// Index by 3-bit value (0-7) to get ASCII char.
const OCT_CHARS: [u8; 8] = *b"01234567";

// =============================================================================
// Stack Buffer for Zero-Allocation Formatting
// =============================================================================

/// Maximum buffer size needed for formatting.
/// - Binary: 2 (prefix) + 64 (bits) + 1 (sign) = 67 chars
/// - Hex: 2 (prefix) + 16 (nibbles) + 1 (sign) = 19 chars  
/// - Octal: 2 (prefix) + 22 (max for i64) + 1 (sign) = 25 chars
const FORMAT_BUFFER_SIZE: usize = 72;

/// Stack-allocated buffer for formatting operations.
/// Avoids heap allocation for all standard integer sizes.
struct FormatBuffer {
    /// Fixed-size buffer on the stack.
    data: [u8; FORMAT_BUFFER_SIZE],
    /// Current write position (we write right-to-left).
    pos: usize,
}

impl FormatBuffer {
    /// Create a new empty buffer positioned at the end.
    #[inline(always)]
    const fn new() -> Self {
        Self {
            data: [0u8; FORMAT_BUFFER_SIZE],
            pos: FORMAT_BUFFER_SIZE,
        }
    }

    /// Push a byte to the buffer (right-to-left).
    #[inline(always)]
    fn push(&mut self, byte: u8) {
        debug_assert!(self.pos > 0, "FormatBuffer overflow");
        self.pos -= 1;
        self.data[self.pos] = byte;
    }

    /// Push two bytes (prefix like "0b", "0x", "0o").
    #[inline(always)]
    fn push_prefix(&mut self, prefix: &[u8; 2]) {
        self.push(prefix[1]);
        self.push(prefix[0]);
    }

    /// Get the formatted string as a byte slice.
    #[inline(always)]
    fn as_bytes(&self) -> &[u8] {
        &self.data[self.pos..]
    }

    /// Get the formatted string as a str (unsafe but we only write ASCII).
    #[inline(always)]
    fn as_str(&self) -> &str {
        // SAFETY: We only write ASCII bytes
        unsafe { std::str::from_utf8_unchecked(self.as_bytes()) }
    }

    /// Get the length of the formatted content.
    #[inline(always)]
    fn len(&self) -> usize {
        FORMAT_BUFFER_SIZE - self.pos
    }
}

// =============================================================================
// bin() - Binary String Representation
// =============================================================================

/// Builtin bin(x) function.
///
/// Returns the binary representation of an integer as a string.
/// Output format: `"0b..."` for positive, `"-0b..."` for negative.
///
/// # Python Semantics
/// - `bin(0)` → `"0b0"`
/// - `bin(5)` → `"0b101"`  
/// - `bin(-5)` → `"-0b101"`
/// - Non-integers raise TypeError
///
/// # Performance
/// - Zero heap allocation for stack formatting
/// - Single pass bit extraction with shifts
/// - Exact output length pre-computed
pub fn builtin_bin(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "bin() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    // Fast path: native integer
    if let Some(n) = args[0].as_int() {
        return format_binary(n);
    }

    // Bool is valid via __index__ (True=1, False=0)
    if let Some(b) = args[0].as_bool() {
        return format_binary(if b { 1 } else { 0 });
    }

    // TODO: Handle objects with __index__ protocol
    Err(BuiltinError::TypeError(format!(
        "'{}' object cannot be interpreted as an integer",
        type_name_of(&args[0])
    )))
}

/// Format an i64 as binary string "0b...".
///
/// Uses zero-allocation stack buffer and right-to-left filling.
#[inline]
fn format_binary(n: i64) -> Result<Value, BuiltinError> {
    let mut buf = FormatBuffer::new();
    let negative = n < 0;

    // Work with absolute value (handle i64::MIN specially)
    let abs_val = if n == i64::MIN {
        // i64::MIN has no positive counterpart, work with u64
        i64::MIN as u64
    } else {
        n.unsigned_abs()
    };

    if abs_val == 0 {
        // Special case: "0b0"
        buf.push(b'0');
    } else {
        // Extract bits right-to-left
        let mut val = abs_val;
        while val > 0 {
            buf.push(b'0' + (val & 1) as u8);
            val >>= 1;
        }
    }

    // Add prefix "0b"
    buf.push_prefix(b"0b");

    // Add minus sign if negative
    if negative {
        buf.push(b'-');
    }

    Ok(interned_string_value(buf.as_str()))
}

// =============================================================================
// hex() - Hexadecimal String Representation
// =============================================================================

/// Builtin hex(x) function.
///
/// Returns the hexadecimal representation of an integer as a string.
/// Output format: `"0x..."` for positive, `"-0x..."` for negative.
/// Uses lowercase letters (a-f).
///
/// # Python Semantics
/// - `hex(0)` → `"0x0"`
/// - `hex(255)` → `"0xff"`
/// - `hex(-255)` → `"-0xff"`
/// - Non-integers raise TypeError
///
/// # Performance
/// - Lookup table for nibble→char (no branching)
/// - Zero heap allocation for formatting
/// - Single pass nibble extraction
pub fn builtin_hex(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "hex() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    // Fast path: native integer
    if let Some(n) = args[0].as_int() {
        return format_hex(n);
    }

    // Bool is valid via __index__
    if let Some(b) = args[0].as_bool() {
        return format_hex(if b { 1 } else { 0 });
    }

    Err(BuiltinError::TypeError(format!(
        "'{}' object cannot be interpreted as an integer",
        type_name_of(&args[0])
    )))
}

/// Format an i64 as hexadecimal string "0x...".
///
/// Uses lookup table for zero-branch nibble conversion.
#[inline]
fn format_hex(n: i64) -> Result<Value, BuiltinError> {
    let mut buf = FormatBuffer::new();
    let negative = n < 0;

    // Work with absolute value
    let abs_val = if n == i64::MIN {
        i64::MIN as u64
    } else {
        n.unsigned_abs()
    };

    if abs_val == 0 {
        buf.push(b'0');
    } else {
        let mut val = abs_val;
        while val > 0 {
            let nibble = (val & 0xF) as usize;
            buf.push(HEX_CHARS_LOWER[nibble]);
            val >>= 4;
        }
    }

    buf.push_prefix(b"0x");

    if negative {
        buf.push(b'-');
    }

    Ok(interned_string_value(buf.as_str()))
}

// =============================================================================
// oct() - Octal String Representation
// =============================================================================

/// Builtin oct(x) function.
///
/// Returns the octal representation of an integer as a string.
/// Output format: `"0o..."` for positive, `"-0o..."` for negative.
///
/// # Python Semantics
/// - `oct(0)` → `"0o0"`
/// - `oct(8)` → `"0o10"`
/// - `oct(-8)` → `"-0o10"`
/// - Non-integers raise TypeError
///
/// # Performance
/// - Lookup table for 3-bit→char
/// - Zero heap allocation for formatting
pub fn builtin_oct(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "oct() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    // Fast path: native integer
    if let Some(n) = args[0].as_int() {
        return format_oct(n);
    }

    // Bool is valid via __index__
    if let Some(b) = args[0].as_bool() {
        return format_oct(if b { 1 } else { 0 });
    }

    Err(BuiltinError::TypeError(format!(
        "'{}' object cannot be interpreted as an integer",
        type_name_of(&args[0])
    )))
}

/// Format an i64 as octal string "0o...".
#[inline]
fn format_oct(n: i64) -> Result<Value, BuiltinError> {
    let mut buf = FormatBuffer::new();
    let negative = n < 0;

    let abs_val = if n == i64::MIN {
        i64::MIN as u64
    } else {
        n.unsigned_abs()
    };

    if abs_val == 0 {
        buf.push(b'0');
    } else {
        let mut val = abs_val;
        while val > 0 {
            let digit = (val & 0x7) as usize;
            buf.push(OCT_CHARS[digit]);
            val >>= 3;
        }
    }

    buf.push_prefix(b"0o");

    if negative {
        buf.push(b'-');
    }

    Ok(interned_string_value(buf.as_str()))
}

#[inline]
fn interned_string_value(s: &str) -> Value {
    Value::string(intern(s))
}

// =============================================================================
// complex() - Complex Number Constructor
// =============================================================================

/// Builtin complex([real[, imag]]) function.
///
/// Creates a complex number from real and imaginary parts.
///
/// # Python Semantics
/// - `complex()` → `0j`
/// - `complex(1)` → `(1+0j)`
/// - `complex(1, 2)` → `(1+2j)`
/// - `complex("1+2j")` → `(1+2j)` (string parsing)
///
pub fn builtin_complex(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "complex() takes at most 2 arguments ({} given)",
            args.len()
        )));
    }

    match args.len() {
        0 => Ok(boxed_complex_value(0.0, 0.0)),
        1 => {
            if is_complex_value(args[0]) {
                return Ok(args[0]);
            }

            let parts = extract_complex_parts(args[0], "real")?;
            Ok(boxed_complex_value(parts.real, parts.imag))
        }
        2 => {
            let real = extract_complex_parts(args[0], "real")?;
            let imag = extract_complex_parts(args[1], "imag")?;
            Ok(boxed_complex_value(
                real.real - imag.imag,
                real.imag + imag.real,
            ))
        }
        _ => unreachable!(),
    }
}

#[inline]
fn boxed_complex_value(real: f64, imag: f64) -> Value {
    crate::alloc_managed_value(ComplexObject::new(real, imag))
}

#[inline]
fn extract_complex_parts(value: Value, part_name: &str) -> Result<ComplexParts, BuiltinError> {
    if let Some(parts) = complex_like_parts(value) {
        return Ok(parts);
    }

    Err(BuiltinError::TypeError(format!(
        "complex() {} part must be a number, not '{}'",
        part_name,
        type_name_of(&value)
    )))
}

/// Get the type name of a value for error messages.
#[inline]
fn type_name_of(val: &Value) -> &'static str {
    if val.is_none() {
        "NoneType"
    } else if val.is_bool() {
        "bool"
    } else if val.is_int() {
        "int"
    } else if val.is_float() {
        "float"
    } else if val.is_object() {
        val.as_object_ptr()
            .map(crate::ops::objects::extract_type_id)
            .map(TypeId::name)
            .unwrap_or("object")
    } else {
        "unknown"
    }
}

// =============================================================================
// Internal Formatting Functions (Public for Testing)
// =============================================================================

/// Format an integer as binary and return the string.
///
/// This is the low-level implementation exposed for testing.
/// The actual builtin wraps this to return a Value.
pub fn format_binary_string(n: i64) -> String {
    let mut buf = FormatBuffer::new();
    let negative = n < 0;

    let abs_val = if n == i64::MIN {
        i64::MIN as u64
    } else {
        n.unsigned_abs()
    };

    if abs_val == 0 {
        buf.push(b'0');
    } else {
        let mut val = abs_val;
        while val > 0 {
            buf.push(b'0' + (val & 1) as u8);
            val >>= 1;
        }
    }

    buf.push_prefix(b"0b");

    if negative {
        buf.push(b'-');
    }

    buf.as_str().to_string()
}

/// Format an integer as hexadecimal and return the string.
pub fn format_hex_string(n: i64) -> String {
    let mut buf = FormatBuffer::new();
    let negative = n < 0;

    let abs_val = if n == i64::MIN {
        i64::MIN as u64
    } else {
        n.unsigned_abs()
    };

    if abs_val == 0 {
        buf.push(b'0');
    } else {
        let mut val = abs_val;
        while val > 0 {
            let nibble = (val & 0xF) as usize;
            buf.push(HEX_CHARS_LOWER[nibble]);
            val >>= 4;
        }
    }

    buf.push_prefix(b"0x");

    if negative {
        buf.push(b'-');
    }

    buf.as_str().to_string()
}

/// Format an integer as octal and return the string.
pub fn format_oct_string(n: i64) -> String {
    let mut buf = FormatBuffer::new();
    let negative = n < 0;

    let abs_val = if n == i64::MIN {
        i64::MIN as u64
    } else {
        n.unsigned_abs()
    };

    if abs_val == 0 {
        buf.push(b'0');
    } else {
        let mut val = abs_val;
        while val > 0 {
            let digit = (val & 0x7) as usize;
            buf.push(OCT_CHARS[digit]);
            val >>= 3;
        }
    }

    buf.push_prefix(b"0o");

    if negative {
        buf.push(b'-');
    }

    buf.as_str().to_string()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
