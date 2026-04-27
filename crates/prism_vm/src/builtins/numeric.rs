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

use super::{BuiltinError, runtime_error_to_builtin_error};
use crate::VirtualMachine;
use crate::ops::calls::invoke_callable_value;
use crate::ops::method_dispatch::load_method::{BoundMethodTarget, resolve_special_method};
use crate::python_numeric::{ComplexParts, complex_like_parts, is_complex_value};
use num_bigint::BigInt;
use num_traits::{Signed, Zero};
use prism_core::Value;
use prism_core::intern::{intern, interned_by_ptr};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::complex::ComplexObject;
use prism_runtime::types::int::value_to_bigint;

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

#[derive(Clone, Copy)]
enum IntegerFormatKind {
    Binary,
    Hex,
    Octal,
}

impl IntegerFormatKind {
    #[inline]
    fn builtin_name(self) -> &'static str {
        match self {
            Self::Binary => "bin",
            Self::Hex => "hex",
            Self::Octal => "oct",
        }
    }

    #[inline]
    fn prefix(self) -> &'static str {
        match self {
            Self::Binary => "0b",
            Self::Hex => "0x",
            Self::Octal => "0o",
        }
    }

    #[inline]
    fn radix(self) -> u32 {
        match self {
            Self::Binary => 2,
            Self::Hex => 16,
            Self::Octal => 8,
        }
    }

    #[inline]
    fn format_i64(self, value: i64) -> Result<Value, BuiltinError> {
        match self {
            Self::Binary => format_binary(value),
            Self::Hex => format_hex(value),
            Self::Octal => format_oct(value),
        }
    }
}

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
    format_integer_direct(args, IntegerFormatKind::Binary)
}

/// VM-aware bin(x) that honors Python's `__index__` protocol.
pub fn builtin_bin_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    format_integer_vm(vm, args, IntegerFormatKind::Binary)
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
    format_integer_direct(args, IntegerFormatKind::Hex)
}

/// VM-aware hex(x) that honors Python's `__index__` protocol.
pub fn builtin_hex_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    format_integer_vm(vm, args, IntegerFormatKind::Hex)
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
    format_integer_direct(args, IntegerFormatKind::Octal)
}

/// VM-aware oct(x) that honors Python's `__index__` protocol.
pub fn builtin_oct_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    format_integer_vm(vm, args, IntegerFormatKind::Octal)
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

fn format_integer_direct(args: &[Value], kind: IntegerFormatKind) -> Result<Value, BuiltinError> {
    let value = unary_integer_format_arg(args, kind)?;
    format_integer_value(value, kind).ok_or_else(|| integer_format_type_error(value))
}

fn format_integer_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
    kind: IntegerFormatKind,
) -> Result<Value, BuiltinError> {
    let value = unary_integer_format_arg(args, kind)?;
    if let Some(formatted) = format_integer_value(value, kind) {
        return Ok(formatted);
    }

    let target = match resolve_special_method(value, "__index__") {
        Ok(target) => target,
        Err(err) if err.is_attribute_error() => return Err(integer_format_type_error(value)),
        Err(err) => return Err(runtime_error_to_builtin_error(err)),
    };

    let indexed = invoke_bound_method_no_args(vm, target)?;
    format_integer_value(indexed, kind).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "__index__ returned non-int (type {})",
            type_name_of(&indexed)
        ))
    })
}

#[inline]
fn unary_integer_format_arg(
    args: &[Value],
    kind: IntegerFormatKind,
) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "{}() takes exactly one argument ({} given)",
            kind.builtin_name(),
            args.len()
        )));
    }

    Ok(args[0])
}

#[inline]
fn format_integer_value(value: Value, kind: IntegerFormatKind) -> Option<Value> {
    if let Some(n) = value.as_int() {
        return Some(kind.format_i64(n).expect("i64 formatting is infallible"));
    }

    if let Some(b) = value.as_bool() {
        return Some(
            kind.format_i64(if b { 1 } else { 0 })
                .expect("bool formatting is infallible"),
        );
    }

    value_to_bigint(value).map(|integer| format_bigint(&integer, kind))
}

fn format_bigint(value: &BigInt, kind: IntegerFormatKind) -> Value {
    if value.is_zero() {
        return interned_string_value(match kind {
            IntegerFormatKind::Binary => "0b0",
            IntegerFormatKind::Hex => "0x0",
            IntegerFormatKind::Octal => "0o0",
        });
    }

    let negative = value.is_negative();
    let digits = value.abs().to_str_radix(kind.radix());
    let mut output =
        String::with_capacity(usize::from(negative) + kind.prefix().len() + digits.len());
    if negative {
        output.push('-');
    }
    output.push_str(kind.prefix());
    output.push_str(&digits);
    interned_string_value(&output)
}

#[inline]
fn integer_format_type_error(value: Value) -> BuiltinError {
    BuiltinError::TypeError(format!(
        "'{}' object cannot be interpreted as an integer",
        type_name_of(&value)
    ))
}

#[inline]
fn invoke_bound_method_no_args(
    vm: &mut VirtualMachine,
    target: BoundMethodTarget,
) -> Result<Value, BuiltinError> {
    match target.implicit_self {
        Some(implicit_self) => invoke_callable_value(vm, target.callable, &[implicit_self]),
        None => invoke_callable_value(vm, target.callable, &[]),
    }
    .map_err(runtime_error_to_builtin_error)
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
