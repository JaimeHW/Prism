//! Python value representation using NaN-boxing for high performance.
//!
//! This module implements a tagged union representation of Python values that
//! fits in a single 64-bit word. We use NaN-boxing: IEEE 754 floating point
//! NaN values have many unused bit patterns that we exploit to store other
//! types inline.
//!
//! ## NaN-Boxing Scheme
//!
//! IEEE 754 double-precision NaN: sign(1) + exponent(11, all 1s) + mantissa(52, non-zero)
//!
//! We use the following encoding (little-endian view):
//! - Floats: standard IEEE 754 encoding (unboxed)
//! - Tagged values: exponent=0x7FF (NaN), bit 51=1 (quiet NaN), bits 48-50=tag, bits 0-47=payload
//!
//! | Tag  | Type        | Payload                           |
//! |------|-------------|-----------------------------------|
//! | 0x0  | None        | unused                            |
//! | 0x1  | Bool        | 0=false, 1=true                   |
//! | 0x2  | Int (small) | 48-bit signed integer             |
//! | 0x3  | Object      | 48-bit pointer                    |
//! | 0x4  | String      | 48-bit interned string pointer    |
//! | 0x5  | Reserved    | future use                        |
//! | 0x6  | Reserved    | future use                        |
//! | 0x7  | Reserved    | future use                        |

use crate::intern::InternedString;
use std::fmt;
use std::hash::{Hash, Hasher};

/// Quiet NaN bit pattern: exponent all 1s + quiet NaN bit (bit 51)
/// NOTE: We use 0x7FF8 NOT 0x7FFC to leave bits 48-50 free for tag encoding
const QNAN: u64 = 0x7FF8_0000_0000_0000;

/// Tag bits position (bits 48-50)
const TAG_SHIFT: u64 = 48;
const TAG_MASK: u64 = 0x0007_0000_0000_0000;

/// Payload mask (bits 0-47)
const PAYLOAD_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;

/// Sign bit mask
const SIGN_BIT: u64 = 0x8000_0000_0000_0000;

/// Tag values
const TAG_NONE: u64 = 0;
const TAG_BOOL: u64 = 1;
const TAG_INT: u64 = 2;
const TAG_OBJECT: u64 = 3;
const TAG_STRING: u64 = 4;

// =============================================================================
// Public Tag Patterns (for branchless speculation)
// =============================================================================

/// Combined QNAN + TAG pattern for string values.
/// Use with `value.raw_bits() & STRING_TAG_MASK == STRING_TAG_PATTERN` for branchless checks.
pub const STRING_TAG_PATTERN: u64 = QNAN | (TAG_STRING << TAG_SHIFT);

/// Combined QNAN + TAG pattern for int values.
pub const INT_TAG_PATTERN: u64 = QNAN | (TAG_INT << TAG_SHIFT);

/// Mask for extracting the type tag portion (QNAN + tag bits).
pub const TYPE_TAG_MASK: u64 = QNAN | TAG_MASK;

/// Payload mask for extracting pointer/value from tagged values.
pub const VALUE_PAYLOAD_MASK: u64 = PAYLOAD_MASK;

/// Maximum small integer (47-bit signed)
pub const SMALL_INT_MAX: i64 = (1_i64 << 47) - 1;
/// Minimum small integer (47-bit signed)
pub const SMALL_INT_MIN: i64 = -(1_i64 << 47);

/// A Python value using NaN-boxing for efficient storage.
///
/// This type is exactly 8 bytes and can represent:
/// - Floating point numbers (unboxed)
/// - None
/// - Booleans
/// - Small integers (48-bit signed, approximately ±140 trillion)
/// - Object references (heap-allocated Python objects)
/// - Interned strings (special case for fast string operations)
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Value {
    bits: u64,
}

impl Value {
    // =========================================================================
    // Constructors
    // =========================================================================

    /// Create a None value.
    #[inline]
    #[must_use]
    pub const fn none() -> Self {
        Self {
            bits: QNAN | (TAG_NONE << TAG_SHIFT),
        }
    }

    /// Create a boolean value.
    #[inline]
    #[must_use]
    pub const fn bool(b: bool) -> Self {
        Self {
            bits: QNAN | (TAG_BOOL << TAG_SHIFT) | (b as u64),
        }
    }

    /// Create an integer value.
    ///
    /// If the integer is in the small int cache range [-5, 256], returns
    /// a pre-computed cached value. Otherwise, if it fits in 48 bits,
    /// it's stored inline. Returns None if the integer is too large.
    ///
    /// # Performance
    ///
    /// For cached integers: O(1) array lookup (< 1ns)
    /// For uncached small ints: O(1) bit manipulation
    #[inline]
    #[must_use]
    pub fn int(i: i64) -> Option<Self> {
        // Fast path: check small int cache first
        if let Some(cached) = crate::small_int_cache::SmallIntCache::get(i) {
            return Some(cached);
        }

        // Slow path: construct inline if it fits
        if i >= SMALL_INT_MIN && i <= SMALL_INT_MAX {
            let payload = (i as u64) & PAYLOAD_MASK;
            Some(Self {
                bits: QNAN | (TAG_INT << TAG_SHIFT) | payload,
            })
        } else {
            None
        }
    }

    /// Create an integer value, panicking if it doesn't fit inline.
    #[inline]
    #[must_use]
    pub const fn int_unchecked(i: i64) -> Self {
        let payload = (i as u64) & PAYLOAD_MASK;
        Self {
            bits: QNAN | (TAG_INT << TAG_SHIFT) | payload,
        }
    }

    /// Create a float value.
    #[inline]
    #[must_use]
    pub fn float(f: f64) -> Self {
        let bits = f.to_bits();
        // Check if it's a NaN that would collide with our tagged values
        // Any NaN where (bits & QNAN) == QNAN would be misidentified as tagged
        if bits & QNAN == QNAN {
            // Use a safe NaN representation: quiet NaN with payload=1, tag bits clear
            // 0x7FF0_0000_0000_0001 = exponent all 1s, mantissa = 1 (valid NaN, but doesn't collide)
            Self {
                bits: 0x7FF0_0000_0000_0001,
            }
        } else {
            Self { bits }
        }
    }

    /// Create an object reference value.
    ///
    /// # Safety
    /// The pointer must be valid and properly aligned.
    #[inline]
    #[must_use]
    pub fn object_ptr(ptr: *const ()) -> Self {
        let ptr_bits = ptr as usize as u64;
        assert!(
            ptr_bits & !PAYLOAD_MASK == 0,
            "Pointer too large for NaN-boxing"
        );
        Self {
            bits: QNAN | (TAG_OBJECT << TAG_SHIFT) | (ptr_bits & PAYLOAD_MASK),
        }
    }

    /// Create an interned string value.
    #[inline]
    #[must_use]
    pub fn string(s: InternedString) -> Self {
        // The interner owns the canonical Arc<str>, so the string data pointer
        // stays stable for the lifetime of the program.
        let ptr = s.as_str().as_ptr() as usize as u64;
        assert!(ptr & !PAYLOAD_MASK == 0, "Pointer too large for NaN-boxing");
        Self {
            bits: QNAN | (TAG_STRING << TAG_SHIFT) | (ptr & PAYLOAD_MASK),
        }
    }

    // =========================================================================
    // Type Checking
    // =========================================================================

    /// Check if this is a tagged value (not a float).
    #[inline]
    #[must_use]
    pub const fn is_tagged(&self) -> bool {
        (self.bits & QNAN) == QNAN
    }

    /// Check if this is a float.
    #[inline]
    #[must_use]
    pub const fn is_float(&self) -> bool {
        (self.bits & QNAN) != QNAN
    }

    /// Check if this is None.
    #[inline]
    #[must_use]
    pub const fn is_none(&self) -> bool {
        self.is_tagged() && self.tag() == TAG_NONE
    }

    /// Check if this is a boolean.
    #[inline]
    #[must_use]
    pub const fn is_bool(&self) -> bool {
        self.is_tagged() && self.tag() == TAG_BOOL
    }

    /// Check if this is a small integer.
    #[inline]
    #[must_use]
    pub const fn is_int(&self) -> bool {
        self.is_tagged() && self.tag() == TAG_INT
    }

    /// Check if this is an object reference.
    #[inline]
    #[must_use]
    pub const fn is_object(&self) -> bool {
        self.is_tagged() && self.tag() == TAG_OBJECT
    }

    /// Check if this is an interned string.
    #[inline]
    #[must_use]
    pub const fn is_string(&self) -> bool {
        self.is_tagged() && self.tag() == TAG_STRING
    }

    /// Get the tag (for tagged values).
    #[inline]
    const fn tag(&self) -> u64 {
        (self.bits & TAG_MASK) >> TAG_SHIFT
    }

    /// Get the payload (for tagged values).
    #[inline]
    const fn payload(&self) -> u64 {
        self.bits & PAYLOAD_MASK
    }

    // =========================================================================
    // Value Extraction
    // =========================================================================

    /// Try to extract as a boolean.
    #[inline]
    #[must_use]
    pub const fn as_bool(&self) -> Option<bool> {
        if self.is_bool() {
            Some(self.payload() != 0)
        } else {
            None
        }
    }

    /// Try to extract as a small integer.
    #[inline]
    #[must_use]
    pub const fn as_int(&self) -> Option<i64> {
        if self.is_int() {
            // Sign-extend from 48 bits
            let payload = self.payload();
            let sign_bit = payload & (1 << 47);
            if sign_bit != 0 {
                // Negative: sign-extend
                Some((payload | !PAYLOAD_MASK) as i64)
            } else {
                // Positive
                Some(payload as i64)
            }
        } else {
            None
        }
    }

    /// Try to extract as a float.
    #[inline]
    #[must_use]
    pub fn as_float(&self) -> Option<f64> {
        if self.is_float() {
            Some(f64::from_bits(self.bits))
        } else {
            None
        }
    }

    /// Try to extract as a float, coercing integers if needed.
    #[inline]
    #[must_use]
    pub fn as_float_coerce(&self) -> Option<f64> {
        if let Some(f) = self.as_float() {
            Some(f)
        } else if let Some(i) = self.as_int() {
            Some(i as f64)
        } else {
            None
        }
    }

    /// Try to extract as an object pointer.
    #[inline]
    #[must_use]
    pub const fn as_object_ptr(&self) -> Option<*const ()> {
        if self.is_object() {
            Some(self.payload() as *const ())
        } else {
            None
        }
    }

    /// Try to extract as a string object pointer.
    ///
    /// Returns a raw pointer to the string data. For interned strings,
    /// this points to the Arc's data buffer. For heap-allocated StringObjects,
    /// this points to the StringObject in the GC heap.
    ///
    /// # Performance
    ///
    /// This is a branchless const operation after the type check.
    /// The returned pointer is suitable for direct string operations.
    ///
    /// # Safety
    ///
    /// The returned pointer is valid as long as:
    /// - For interned strings: the InternedString is not dropped (they're leaked)
    /// - For heap strings: the StringObject is reachable by the GC
    #[inline]
    #[must_use]
    pub const fn as_string_object_ptr(&self) -> Option<*const ()> {
        if self.is_string() {
            Some(self.payload() as *const ())
        } else {
            None
        }
    }

    /// Get raw bits (for speculation optimizations).
    ///
    /// # Performance
    ///
    /// Enables branchless type checking in speculation code:
    /// ```ignore
    /// let is_string = (value.raw_bits() >> 48) == STRING_TAG_PATTERN;
    /// ```
    #[inline(always)]
    #[must_use]
    pub const fn raw_bits(&self) -> u64 {
        self.bits
    }

    // =========================================================================
    // Truthiness (Python bool conversion)
    // =========================================================================

    /// Python truthiness evaluation.
    ///
    /// Returns false for: None, False, 0, 0.0, empty collections
    /// For objects, this requires calling __bool__ or __len__.
    #[inline]
    #[must_use]
    pub fn is_truthy(&self) -> bool {
        if self.is_none() {
            false
        } else if let Some(b) = self.as_bool() {
            b
        } else if let Some(i) = self.as_int() {
            i != 0
        } else if let Some(f) = self.as_float() {
            f != 0.0
        } else {
            // Objects are truthy by default (need __bool__/__len__ for full impl)
            true
        }
    }

    // =========================================================================
    // Type Name
    // =========================================================================

    /// Get the Python type name.
    #[must_use]
    pub const fn type_name(&self) -> &'static str {
        if self.is_float() {
            "float"
        } else if self.is_none() {
            "NoneType"
        } else if self.is_bool() {
            "bool"
        } else if self.is_int() {
            "int"
        } else if self.is_string() {
            "str"
        } else if self.is_object() {
            "object"
        } else {
            "unknown"
        }
    }

    /// Get the raw bits (for debugging/serialization).
    #[inline]
    #[must_use]
    pub const fn to_bits(&self) -> u64 {
        self.bits
    }

    /// Create from raw bits (for deserialization).
    #[inline]
    #[must_use]
    pub const fn from_bits(bits: u64) -> Self {
        Self { bits }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        // Handle NaN first (NaN != NaN in IEEE 754, even with identical bits)
        if self.is_float() && other.is_float() {
            let a = f64::from_bits(self.bits);
            let b = f64::from_bits(other.bits);
            // NaN != NaN per IEEE 754, otherwise compare normally
            return a == b;
        }

        // Fast path: identical bits (safe now that we've handled NaN)
        if self.bits == other.bits {
            return true;
        }

        // Handle int/float coercion (1 == 1.0 in Python)
        if let (Some(i), Some(f)) = (self.as_int(), other.as_float()) {
            return (i as f64) == f;
        }
        if let (Some(f), Some(i)) = (self.as_float(), other.as_int()) {
            return f == (i as f64);
        }

        false
    }
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // For consistency with Python, int and equivalent float should hash the same
        if let Some(i) = self.as_int() {
            i.hash(state);
        } else if let Some(f) = self.as_float() {
            // Check if it's an integer value
            if f.fract() == 0.0 && f.is_finite() && f >= (i64::MIN as f64) && f <= (i64::MAX as f64)
            {
                (f as i64).hash(state);
            } else {
                self.bits.hash(state);
            }
        } else {
            self.bits.hash(state);
        }
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_none() {
            write!(f, "Value(None)")
        } else if let Some(b) = self.as_bool() {
            write!(f, "Value({})", if b { "True" } else { "False" })
        } else if let Some(i) = self.as_int() {
            write!(f, "Value({})", i)
        } else if let Some(fl) = self.as_float() {
            write!(f, "Value({:?})", fl)
        } else if self.is_object() {
            write!(f, "Value(object@{:#x})", self.payload())
        } else if self.is_string() {
            write!(f, "Value(str@{:#x})", self.payload())
        } else {
            write!(f, "Value(bits={:#018x})", self.bits)
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_none() {
            write!(f, "None")
        } else if let Some(b) = self.as_bool() {
            write!(f, "{}", if b { "True" } else { "False" })
        } else if let Some(i) = self.as_int() {
            write!(f, "{}", i)
        } else if let Some(fl) = self.as_float() {
            // Python-style float formatting
            if fl.fract() == 0.0 && fl.is_finite() {
                write!(f, "{}.0", fl as i64)
            } else {
                write!(f, "{}", fl)
            }
        } else if self.is_object() {
            write!(f, "<object at {:#x}>", self.payload())
        } else if self.is_string() {
            write!(f, "<str at {:#x}>", self.payload())
        } else {
            write!(f, "<unknown>")
        }
    }
}

impl Default for Value {
    fn default() -> Self {
        Self::none()
    }
}

impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Self::bool(b)
    }
}

impl From<f64> for Value {
    fn from(f: f64) -> Self {
        Self::float(f)
    }
}

impl From<f32> for Value {
    fn from(f: f32) -> Self {
        Self::float(f as f64)
    }
}

impl TryFrom<i64> for Value {
    type Error = ();

    fn try_from(i: i64) -> Result<Self, Self::Error> {
        Self::int(i).ok_or(())
    }
}

impl From<i32> for Value {
    fn from(i: i32) -> Self {
        Self::int_unchecked(i as i64)
    }
}

impl From<i16> for Value {
    fn from(i: i16) -> Self {
        Self::int_unchecked(i as i64)
    }
}

impl From<i8> for Value {
    fn from(i: i8) -> Self {
        Self::int_unchecked(i as i64)
    }
}

impl From<u32> for Value {
    fn from(i: u32) -> Self {
        Self::int_unchecked(i as i64)
    }
}

impl From<u16> for Value {
    fn from(i: u16) -> Self {
        Self::int_unchecked(i as i64)
    }
}

impl From<u8> for Value {
    fn from(i: u8) -> Self {
        Self::int_unchecked(i as i64)
    }
}
