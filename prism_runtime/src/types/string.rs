//! High-performance Python string object implementation.
//!
//! Features:
//! - **Small String Optimization (SSO)**: Strings ≤23 bytes stored inline (no heap)
//! - **Interned Strings**: Globally deduplicated strings for identifiers
//! - **Heap Strings**: Arc-backed for longer strings with zero-copy cloning
//! - **Cached Hash**: ObjectHeader.hash for O(1) repeated lookups
//! - **UTF-8 Native**: All strings are valid UTF-8
//! - **Static Empty String**: Zero-allocation access via `empty_string()`

use crate::object::type_obj::TypeId;
use crate::object::{HASH_NOT_COMPUTED, ObjectHeader, PyObject};
use prism_core::intern::InternedString;
use std::borrow::Cow;
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, LazyLock};

// =============================================================================
// Constants
// =============================================================================

/// Maximum inline storage for SSO (24 bytes total, 1 for discriminant/len)
pub const SSO_MAX_LEN: usize = 23;

/// Empty string constant for zero-allocation empty string creation.
const EMPTY_INLINE: InlineString = InlineString {
    len: 0,
    data: [0u8; SSO_MAX_LEN],
};

// =============================================================================
// Static Empty String Singleton
// =============================================================================

/// Thread-safe static empty string singleton.
///
/// Returns a reference to a pre-allocated empty StringObject.
/// Used for zero-allocation empty string operations like `str * 0`.
static STATIC_EMPTY_STRING: LazyLock<StringObject> = LazyLock::new(StringObject::empty);

/// Get a reference to the static empty string (zero allocation).
///
/// # Performance
///
/// This returns a static reference, avoiding any allocation.
/// Use this for operations that would return an empty string:
/// - String multiplication by zero: `s * 0`
/// - Empty concatenation results
/// - Empty slices
#[inline(always)]
pub fn empty_string() -> &'static StringObject {
    &STATIC_EMPTY_STRING
}

// =============================================================================
// Inline String Storage (SSO)
// =============================================================================

/// Inline storage for short strings.
///
/// Layout: 1 byte length + 23 bytes data = 24 bytes total.
/// This fits in 3 cache lines with ObjectHeader (16 bytes).
#[derive(Clone, Copy)]
#[repr(C)]
struct InlineString {
    /// Length in bytes (max 23).
    len: u8,
    /// UTF-8 string data.
    data: [u8; SSO_MAX_LEN],
}

impl InlineString {
    /// Create from a string slice. Panics if too long.
    #[inline]
    fn from_str(s: &str) -> Self {
        debug_assert!(s.len() <= SSO_MAX_LEN);
        let mut data = [0u8; SSO_MAX_LEN];
        let len = s.len();
        data[..len].copy_from_slice(s.as_bytes());
        Self {
            len: len as u8,
            data,
        }
    }

    /// Get as string slice.
    #[inline]
    fn as_str(&self) -> &str {
        // Safety: We only store valid UTF-8
        unsafe { std::str::from_utf8_unchecked(&self.data[..self.len as usize]) }
    }

    /// Get byte length.
    #[inline]
    fn len(&self) -> usize {
        self.len as usize
    }
}

// =============================================================================
// String Kind (Discriminated Union)
// =============================================================================

/// Internal representation of string data.
///
/// Uses a tagged union for efficient storage:
/// - `Inline`: No allocation for strings ≤23 bytes
/// - `Heap`: Arc-backed for zero-copy cloning of long strings
/// - `Interned`: Globally deduplicated, pointer-equality comparison
#[derive(Clone)]
enum StringRepr {
    /// Inline storage for short strings (≤23 bytes).
    Inline(InlineString),
    /// Heap-allocated for longer strings.
    Heap(Arc<str>),
    /// Interned string (shared globally, O(1) equality).
    Interned(InternedString),
}

impl StringRepr {
    /// Get as string slice.
    #[inline]
    fn as_str(&self) -> &str {
        match self {
            StringRepr::Inline(s) => s.as_str(),
            StringRepr::Heap(s) => s,
            StringRepr::Interned(s) => s.as_str(),
        }
    }

    /// Get byte length.
    #[inline]
    fn len(&self) -> usize {
        match self {
            StringRepr::Inline(s) => s.len(),
            StringRepr::Heap(s) => s.len(),
            StringRepr::Interned(s) => s.len(),
        }
    }

    /// Check if empty.
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// =============================================================================
// StringObject
// =============================================================================

/// Python string object.
///
/// Immutable UTF-8 string with three storage strategies:
///
/// 1. **Inline (SSO)**: Strings ≤23 bytes stored inline (no heap allocation)
/// 2. **Heap**: Longer strings use `Arc<str>` for zero-copy cloning
/// 3. **Interned**: Shared globally for identifiers, O(1) equality check
///
/// # Performance
///
/// - SSO avoids heap allocation for short strings (common case)
/// - Hash is cached in ObjectHeader for repeated lookups
/// - Interned strings use pointer equality for O(1) comparison
///
/// # Memory Layout
///
/// ```text
/// StringObject (40 bytes total):
/// ├── ObjectHeader (16 bytes): type_id, gc_flags, hash
/// └── StringRepr (24 bytes): enum with inline storage
/// ```
#[repr(C)]
pub struct StringObject {
    /// Object header with cached hash.
    pub header: ObjectHeader,
    /// String representation.
    repr: StringRepr,
}

impl StringObject {
    /// Create a new string object, choosing the optimal representation.
    ///
    /// - Strings ≤23 bytes use inline storage (no heap allocation)
    /// - Longer strings use heap allocation with Arc
    #[inline]
    pub fn new(s: &str) -> Self {
        let repr = if s.len() <= SSO_MAX_LEN {
            StringRepr::Inline(InlineString::from_str(s))
        } else {
            StringRepr::Heap(s.into())
        };
        Self {
            header: ObjectHeader::new(TypeId::STR),
            repr,
        }
    }

    /// Create from an owned String, avoiding allocation if possible.
    #[inline]
    pub fn from_string(s: String) -> Self {
        let repr = if s.len() <= SSO_MAX_LEN {
            StringRepr::Inline(InlineString::from_str(&s))
        } else {
            StringRepr::Heap(s.into())
        };
        Self {
            header: ObjectHeader::new(TypeId::STR),
            repr,
        }
    }

    /// Create from an Arc<str> (zero allocation).
    #[inline]
    pub fn from_arc(s: Arc<str>) -> Self {
        let repr = if s.len() <= SSO_MAX_LEN {
            // Copy to inline even from Arc for consistency
            StringRepr::Inline(InlineString::from_str(&s))
        } else {
            StringRepr::Heap(s)
        };
        Self {
            header: ObjectHeader::new(TypeId::STR),
            repr,
        }
    }

    /// Create from an interned string (fastest equality checks).
    #[inline]
    pub fn from_interned(s: InternedString) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::STR),
            repr: StringRepr::Interned(s),
        }
    }

    /// Create an empty string (no allocation).
    #[inline]
    pub fn empty() -> Self {
        Self {
            header: ObjectHeader::new(TypeId::STR),
            repr: StringRepr::Inline(EMPTY_INLINE),
        }
    }

    /// Get the string content as a slice.
    #[inline]
    pub fn as_str(&self) -> &str {
        self.repr.as_str()
    }

    /// Get the byte length.
    #[inline]
    pub fn len(&self) -> usize {
        self.repr.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.repr.is_empty()
    }

    /// Get the character (codepoint) count.
    ///
    /// Note: This is O(n) for non-ASCII strings as it counts UTF-8 codepoints.
    /// Uses SIMD acceleration for counting non-continuation UTF-8 bytes.
    #[inline]
    pub fn char_count(&self) -> usize {
        crate::types::simd::validation::utf8_char_count(self.as_str().as_bytes())
    }

    /// Check if this is an interned string.
    #[inline]
    pub fn is_interned(&self) -> bool {
        matches!(self.repr, StringRepr::Interned(_))
    }

    /// Check if this uses inline storage.
    #[inline]
    pub fn is_inline(&self) -> bool {
        matches!(self.repr, StringRepr::Inline(_))
    }

    /// Get the cached hash, computing if needed.
    #[inline]
    pub fn hash(&self) -> u64 {
        if self.header.hash != HASH_NOT_COMPUTED {
            return self.header.hash;
        }
        // Compute and cache (note: header.hash is not mut, we'd need interior mutability)
        // For now, compute each time if not cached
        self.compute_hash()
    }

    /// Compute string hash using FxHash algorithm.
    #[inline]
    fn compute_hash(&self) -> u64 {
        use std::hash::BuildHasher;
        let hasher = rustc_hash::FxBuildHasher::default();
        let mut h = hasher.build_hasher();
        self.as_str().hash(&mut h);
        h.finish()
    }

    // =========================================================================
    // String Operations
    // =========================================================================

    /// Concatenate two strings with maximum performance.
    ///
    /// # Performance Optimizations
    ///
    /// 1. **Identity Fast-Path (Empty LHS)**: If `self` is empty, returns clone of `other`
    /// 2. **Identity Fast-Path (Empty RHS)**: If `other` is empty, returns clone of `self`
    /// 3. **SSO Direct Copy**: Results ≤23 bytes use `ptr::copy_nonoverlapping`
    /// 4. **Pre-Sized Heap Allocation**: Heap path pre-allocates exact capacity
    ///
    /// # Example
    ///
    /// ```
    /// use prism_runtime::types::StringObject;
    /// let a = StringObject::new("hello");
    /// let b = StringObject::new(" world");
    /// let result = a.concat(&b);
    /// assert_eq!(result.as_str(), "hello world");
    /// ```
    #[inline(always)]
    pub fn concat(&self, other: &StringObject) -> StringObject {
        // FAST PATH 1: Empty lhs → return other (zero copy for common case)
        if self.is_empty() {
            return other.clone();
        }

        // FAST PATH 2: Empty rhs → return self (zero copy)
        if other.is_empty() {
            return self.clone();
        }

        let self_str = self.as_str();
        let other_str = other.as_str();
        let self_len = self_str.len();
        let other_len = other_str.len();
        let total_len = self_len + other_len;

        if total_len <= SSO_MAX_LEN {
            // SSO path: Direct copy into inline storage, no intermediate allocation
            Self::concat_sso_direct(self_str, other_str, self_len, other_len, total_len)
        } else {
            // Heap path: Pre-sized allocation
            Self::concat_heap(self_str, other_str, total_len)
        }
    }

    /// SSO concatenation using direct pointer copies.
    ///
    /// # Safety
    /// Caller must ensure `total_len <= SSO_MAX_LEN`.
    #[inline(always)]
    fn concat_sso_direct(
        a: &str,
        b: &str,
        a_len: usize,
        b_len: usize,
        total_len: usize,
    ) -> StringObject {
        debug_assert!(total_len <= SSO_MAX_LEN);

        let mut data = [0u8; SSO_MAX_LEN];

        // SAFETY: We've verified total_len <= SSO_MAX_LEN, so these copies are in-bounds.
        // Using ptr::copy_nonoverlapping is faster than slice copy for small fixed buffers.
        unsafe {
            std::ptr::copy_nonoverlapping(a.as_ptr(), data.as_mut_ptr(), a_len);
            std::ptr::copy_nonoverlapping(b.as_ptr(), data.as_mut_ptr().add(a_len), b_len);
        }

        StringObject {
            header: ObjectHeader::new(TypeId::STR),
            repr: StringRepr::Inline(InlineString {
                len: total_len as u8,
                data,
            }),
        }
    }

    /// Heap concatenation with pre-sized allocation.
    #[inline(never)] // Cold path, don't pollute instruction cache
    fn concat_heap(a: &str, b: &str, total_len: usize) -> StringObject {
        let mut result = String::with_capacity(total_len);
        result.push_str(a);
        result.push_str(b);
        StringObject::from_string(result)
    }

    /// Repeat the string n times with maximum performance.
    ///
    /// # Performance Optimizations
    ///
    /// 1. **Zero Return**: `n == 0` returns static empty string (zero allocation)
    /// 2. **Identity Return**: `n == 1` returns clone of self
    /// 3. **Single-Byte ASCII**: Single ASCII char uses optimized byte fill
    /// 4. **SSO Path**: Results ≤23 bytes use inline storage
    /// 5. **Pre-Sized Heap**: Larger results use pre-allocated String
    ///
    /// # Panics
    ///
    /// Panics if `self.len() * n` overflows `usize`.
    #[inline(always)]
    pub fn repeat(&self, n: usize) -> StringObject {
        // FAST PATH 1: n == 0 → return empty (zero allocation via static)
        if n == 0 {
            return empty_string().clone();
        }

        // FAST PATH 2: n == 1 → return clone (avoid all computation)
        if n == 1 {
            return self.clone();
        }

        let s = self.as_str();
        let s_len = s.len();

        // Check for overflow using checked multiplication
        let Some(total_len) = s_len.checked_mul(n) else {
            panic!("string repeat overflow: {} * {} overflows", s_len, n);
        };

        // FAST PATH 3: Single ASCII byte → use optimized fill
        if s_len == 1 {
            let byte = s.as_bytes()[0];
            // For single-byte strings, use memset-style optimization
            return Self::repeat_single_byte(byte, n, total_len);
        }

        if total_len <= SSO_MAX_LEN {
            // SSO path with unrolled copy
            Self::repeat_sso(s, s_len, n, total_len)
        } else {
            // Heap path
            Self::repeat_heap(s, n)
        }
    }

    /// Optimized single-byte repetition.
    ///
    /// For single-character strings, we can use a highly optimized byte fill
    /// which the compiler often vectorizes using SIMD instructions.
    #[inline(always)]
    fn repeat_single_byte(byte: u8, n: usize, total_len: usize) -> StringObject {
        if total_len <= SSO_MAX_LEN {
            // Fill SSO buffer with single byte - compiler can use SIMD
            let mut data = [byte; SSO_MAX_LEN];
            // Zero out unused portion (not strictly necessary but safe)
            for i in total_len..SSO_MAX_LEN {
                data[i] = 0;
            }
            StringObject {
                header: ObjectHeader::new(TypeId::STR),
                repr: StringRepr::Inline(InlineString {
                    len: total_len as u8,
                    data,
                }),
            }
        } else {
            // Heap path for large single-byte repetition
            // Use vec! macro which is highly optimized for fills
            let bytes = vec![byte; n];
            // SAFETY: Single ASCII byte repeated is always valid UTF-8
            let s = unsafe { String::from_utf8_unchecked(bytes) };
            StringObject::from_string(s)
        }
    }

    /// SSO repetition with direct copies.
    #[inline(always)]
    fn repeat_sso(s: &str, s_len: usize, n: usize, total_len: usize) -> StringObject {
        debug_assert!(total_len <= SSO_MAX_LEN);

        let mut data = [0u8; SSO_MAX_LEN];
        let src = s.as_bytes();

        // Copy each repetition directly using pointer arithmetic
        for i in 0..n {
            let start = i * s_len;
            // SAFETY: total_len <= SSO_MAX_LEN guarantees we're in bounds
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), data.as_mut_ptr().add(start), s_len);
            }
        }

        StringObject {
            header: ObjectHeader::new(TypeId::STR),
            repr: StringRepr::Inline(InlineString {
                len: total_len as u8,
                data,
            }),
        }
    }

    /// Heap repetition (cold path).
    #[inline(never)]
    fn repeat_heap(s: &str, n: usize) -> StringObject {
        StringObject::from_string(s.repeat(n))
    }

    /// Get a character by index (0-based, supports negative indexing).
    ///
    /// Returns None if index is out of bounds.
    pub fn get_char(&self, index: i64) -> Option<char> {
        let s = self.as_str();
        let char_count = s.chars().count() as i64;
        let idx = if index < 0 {
            (char_count + index) as usize
        } else {
            index as usize
        };
        s.chars().nth(idx)
    }

    /// Get a substring slice.
    ///
    /// Supports Python-style slicing with optional start/end/step.
    pub fn slice(&self, start: Option<i64>, end: Option<i64>) -> StringObject {
        let s = self.as_str();
        let len = s.len() as i64;

        let start = start.map(|i| self.normalize_index(i, len)).unwrap_or(0) as usize;
        let end = end
            .map(|i| self.normalize_index(i, len))
            .unwrap_or(len as i64) as usize;

        if start >= end || start >= s.len() {
            return StringObject::empty();
        }

        let end = end.min(s.len());
        StringObject::new(&s[start..end])
    }

    /// Check if string contains a substring.
    ///
    /// Uses SIMD-accelerated substring search for larger strings.
    #[inline]
    pub fn contains(&self, needle: &str) -> bool {
        crate::types::simd::search::str_contains(self.as_str(), needle)
    }

    /// Check if string starts with a prefix.
    #[inline]
    pub fn starts_with(&self, prefix: &str) -> bool {
        self.as_str().starts_with(prefix)
    }

    /// Check if string ends with a suffix.
    #[inline]
    pub fn ends_with(&self, suffix: &str) -> bool {
        self.as_str().ends_with(suffix)
    }

    /// Find the first occurrence of a substring.
    ///
    /// Uses SIMD-accelerated substring search (SSE4.2 PCMPESTRI for short patterns).
    #[inline]
    pub fn find(&self, needle: &str) -> Option<usize> {
        crate::types::simd::search::str_find(self.as_str(), needle)
    }

    /// Convert to lowercase.
    ///
    /// Uses SIMD-accelerated ASCII case conversion for pure ASCII strings,
    /// falling back to Unicode-aware conversion for non-ASCII.
    pub fn lower(&self) -> StringObject {
        let s = self.as_str();
        // Fast path: use SIMD for ASCII-only strings
        if crate::types::simd::validation::is_ascii(s.as_bytes()) {
            return StringObject::from_string(crate::types::simd::case::str_to_lowercase(s));
        }
        // Slow path: full Unicode case mapping
        StringObject::from_string(s.to_lowercase())
    }

    /// Convert to uppercase.
    ///
    /// Uses SIMD-accelerated ASCII case conversion for pure ASCII strings,
    /// falling back to Unicode-aware conversion for non-ASCII.
    pub fn upper(&self) -> StringObject {
        let s = self.as_str();
        // Fast path: use SIMD for ASCII-only strings
        if crate::types::simd::validation::is_ascii(s.as_bytes()) {
            return StringObject::from_string(crate::types::simd::case::str_to_uppercase(s));
        }
        // Slow path: full Unicode case mapping
        StringObject::from_string(s.to_uppercase())
    }

    /// Strip whitespace from both ends.
    ///
    /// Uses SIMD-accelerated ASCII whitespace detection for pure ASCII strings,
    /// falling back to Unicode whitespace for non-ASCII.
    pub fn strip(&self) -> StringObject {
        let s = self.as_str();
        // Fast path: use SIMD for ASCII
        if crate::types::simd::validation::is_ascii(s.as_bytes()) {
            return StringObject::new(crate::types::simd::whitespace::trim_str(s));
        }
        // Slow path: Unicode whitespace
        StringObject::new(s.trim())
    }

    /// Strip whitespace from the left.
    ///
    /// Uses SIMD-accelerated whitespace detection for large ASCII strings.
    pub fn lstrip(&self) -> StringObject {
        let s = self.as_str();
        // Fast path: use SIMD for ASCII
        if crate::types::simd::validation::is_ascii(s.as_bytes()) {
            return StringObject::new(crate::types::simd::whitespace::trim_start_str(s));
        }
        // Slow path: Unicode whitespace
        StringObject::new(s.trim_start())
    }

    /// Strip whitespace from the right.
    ///
    /// Uses SIMD-accelerated whitespace detection for large ASCII strings.
    pub fn rstrip(&self) -> StringObject {
        let s = self.as_str();
        // Fast path: use SIMD for ASCII
        if crate::types::simd::validation::is_ascii(s.as_bytes()) {
            return StringObject::new(crate::types::simd::whitespace::trim_end_str(s));
        }
        // Slow path: Unicode whitespace
        StringObject::new(s.trim_end())
    }

    /// Split by a separator.
    pub fn split(&self, sep: &str) -> Vec<StringObject> {
        self.as_str().split(sep).map(StringObject::new).collect()
    }

    /// Join strings with this string as separator.
    pub fn join(&self, parts: &[StringObject]) -> StringObject {
        if parts.is_empty() {
            return StringObject::empty();
        }
        if parts.len() == 1 {
            return parts[0].clone();
        }

        let sep = self.as_str();
        let total_len: usize =
            parts.iter().map(|p| p.len()).sum::<usize>() + sep.len() * (parts.len() - 1);

        let mut result = String::with_capacity(total_len);
        for (i, part) in parts.iter().enumerate() {
            if i > 0 {
                result.push_str(sep);
            }
            result.push_str(part.as_str());
        }
        StringObject::from_string(result)
    }

    /// Replace occurrences of a substring.
    pub fn replace(&self, old: &str, new: &str, count: Option<usize>) -> StringObject {
        match count {
            Some(n) => StringObject::from_string(self.as_str().replacen(old, new, n)),
            None => StringObject::from_string(self.as_str().replace(old, new)),
        }
    }

    // =========================================================================
    // Internal Helpers
    // =========================================================================

    /// Normalize a potentially negative index.
    fn normalize_index(&self, index: i64, len: i64) -> i64 {
        if index < 0 {
            (len + index).max(0)
        } else {
            index.min(len)
        }
    }
}

impl Clone for StringObject {
    fn clone(&self) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::STR),
            repr: self.repr.clone(),
        }
    }
}

impl PartialEq for StringObject {
    fn eq(&self, other: &Self) -> bool {
        // Fast path 1: Check for interned pointer equality (O(1))
        if let (StringRepr::Interned(a), StringRepr::Interned(b)) = (&self.repr, &other.repr) {
            return a == b;
        }

        // Fast path 2: Use SIMD-accelerated byte comparison
        // This is faster than std's str comparison for large strings
        crate::types::simd::equality::str_eq(self.as_str(), other.as_str())
    }
}

impl Eq for StringObject {}

impl PartialOrd for StringObject {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for StringObject {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_str().cmp(other.as_str())
    }
}

impl Hash for StringObject {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_str().hash(state);
    }
}

impl fmt::Debug for StringObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StringObject({:?})", self.as_str())
    }
}

impl fmt::Display for StringObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl AsRef<str> for StringObject {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl std::ops::Deref for StringObject {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}

impl PyObject for StringObject {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}

// =============================================================================
// Conversions
// =============================================================================

impl From<&str> for StringObject {
    fn from(s: &str) -> Self {
        StringObject::new(s)
    }
}

impl From<String> for StringObject {
    fn from(s: String) -> Self {
        StringObject::from_string(s)
    }
}

impl From<Arc<str>> for StringObject {
    fn from(s: Arc<str>) -> Self {
        StringObject::from_arc(s)
    }
}

impl From<InternedString> for StringObject {
    fn from(s: InternedString) -> Self {
        StringObject::from_interned(s)
    }
}

impl<'a> From<Cow<'a, str>> for StringObject {
    fn from(s: Cow<'a, str>) -> Self {
        match s {
            Cow::Borrowed(s) => StringObject::new(s),
            Cow::Owned(s) => StringObject::from_string(s),
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_string() {
        let s = StringObject::empty();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
        assert_eq!(s.as_str(), "");
        assert!(s.is_inline());
    }

    #[test]
    fn test_short_string_inline() {
        let s = StringObject::new("hello");
        assert_eq!(s.as_str(), "hello");
        assert_eq!(s.len(), 5);
        assert!(s.is_inline());
    }

    #[test]
    fn test_max_inline_string() {
        // Exactly 23 bytes should still be inline
        let data = "a".repeat(SSO_MAX_LEN);
        let s = StringObject::new(&data);
        assert!(s.is_inline());
        assert_eq!(s.len(), SSO_MAX_LEN);
    }

    #[test]
    fn test_heap_string() {
        // 24+ bytes should go to heap
        let data = "a".repeat(SSO_MAX_LEN + 1);
        let s = StringObject::new(&data);
        assert!(!s.is_inline());
        assert_eq!(s.len(), SSO_MAX_LEN + 1);
    }

    #[test]
    fn test_interned_string() {
        use prism_core::intern::intern;
        let interned = intern("identifier");
        let s = StringObject::from_interned(interned);
        assert!(s.is_interned());
        assert_eq!(s.as_str(), "identifier");
    }

    #[test]
    fn test_interned_equality_fast_path() {
        use prism_core::intern::intern;
        let i1 = intern("test");
        let i2 = intern("test");
        let s1 = StringObject::from_interned(i1);
        let s2 = StringObject::from_interned(i2);
        assert_eq!(s1, s2); // Uses O(1) pointer comparison
    }

    #[test]
    fn test_concat_inline() {
        let s1 = StringObject::new("hello");
        let s2 = StringObject::new(" world");
        let result = s1.concat(&s2);
        assert_eq!(result.as_str(), "hello world");
        assert!(result.is_inline()); // 11 bytes fits in SSO
    }

    #[test]
    fn test_concat_heap() {
        let s1 = StringObject::new(&"a".repeat(20));
        let s2 = StringObject::new(&"b".repeat(20));
        let result = s1.concat(&s2);
        assert_eq!(result.len(), 40);
        assert!(!result.is_inline()); // >23 bytes goes to heap
    }

    #[test]
    fn test_repeat() {
        let s = StringObject::new("ab");
        let r = s.repeat(3);
        assert_eq!(r.as_str(), "ababab");
    }

    #[test]
    fn test_repeat_zero() {
        let s = StringObject::new("hello");
        let r = s.repeat(0);
        assert!(r.is_empty());
    }

    #[test]
    fn test_get_char() {
        let s = StringObject::new("hello");
        assert_eq!(s.get_char(0), Some('h'));
        assert_eq!(s.get_char(4), Some('o'));
        assert_eq!(s.get_char(-1), Some('o'));
        assert_eq!(s.get_char(-5), Some('h'));
        assert_eq!(s.get_char(5), None);
    }

    #[test]
    fn test_slice() {
        let s = StringObject::new("hello world");
        assert_eq!(s.slice(Some(0), Some(5)).as_str(), "hello");
        assert_eq!(s.slice(Some(6), None).as_str(), "world");
        assert_eq!(s.slice(None, Some(5)).as_str(), "hello");
    }

    #[test]
    fn test_contains() {
        let s = StringObject::new("hello world");
        assert!(s.contains("world"));
        assert!(!s.contains("xyz"));
    }

    #[test]
    fn test_starts_ends_with() {
        let s = StringObject::new("hello world");
        assert!(s.starts_with("hello"));
        assert!(s.ends_with("world"));
    }

    #[test]
    fn test_case_conversion() {
        let s = StringObject::new("Hello World");
        assert_eq!(s.lower().as_str(), "hello world");
        assert_eq!(s.upper().as_str(), "HELLO WORLD");
    }

    #[test]
    fn test_strip() {
        let s = StringObject::new("  hello  ");
        assert_eq!(s.strip().as_str(), "hello");
        assert_eq!(s.lstrip().as_str(), "hello  ");
        assert_eq!(s.rstrip().as_str(), "  hello");
    }

    #[test]
    fn test_split() {
        let s = StringObject::new("a,b,c");
        let parts = s.split(",");
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].as_str(), "a");
        assert_eq!(parts[1].as_str(), "b");
        assert_eq!(parts[2].as_str(), "c");
    }

    #[test]
    fn test_join() {
        let sep = StringObject::new(", ");
        let parts: Vec<_> = ["a", "b", "c"]
            .iter()
            .map(|s| StringObject::new(s))
            .collect();
        let result = sep.join(&parts);
        assert_eq!(result.as_str(), "a, b, c");
    }

    #[test]
    fn test_replace() {
        let s = StringObject::new("hello world world");
        assert_eq!(s.replace("world", "rust", None).as_str(), "hello rust rust");
        assert_eq!(
            s.replace("world", "rust", Some(1)).as_str(),
            "hello rust world"
        );
    }

    #[test]
    fn test_unicode() {
        let s = StringObject::new("こんにちは");
        assert_eq!(s.char_count(), 5);
        assert!(s.len() > 5); // UTF-8 bytes > codepoints
    }

    #[test]
    fn test_emoji() {
        let s = StringObject::new("🦀🐍");
        assert_eq!(s.char_count(), 2);
        assert_eq!(s.get_char(0), Some('🦀'));
        assert_eq!(s.get_char(1), Some('🐍'));
    }

    #[test]
    fn test_clone() {
        let s1 = StringObject::new("test");
        let s2 = s1.clone();
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_ordering() {
        let s1 = StringObject::new("apple");
        let s2 = StringObject::new("banana");
        assert!(s1 < s2);
    }

    #[test]
    fn test_hash_consistency() {
        use std::collections::HashMap;
        let s1 = StringObject::new("key");
        let s2 = StringObject::new("key");

        let mut map = HashMap::new();
        map.insert(s1.as_str().to_string(), 42);
        assert_eq!(map.get(s2.as_str()), Some(&42));
    }

    #[test]
    fn test_from_conversions() {
        let _s1: StringObject = "test".into();
        let _s2: StringObject = String::from("test").into();
        let _s3: StringObject = Arc::<str>::from("test").into();
    }

    // =========================================================================
    // SIMD Integration Tests (Comprehensive)
    // =========================================================================

    mod simd_integration_tests {
        use super::*;

        // Helper: Generate a pattern string
        fn make_pattern_string(len: usize, ch: char) -> String {
            std::iter::repeat(ch).take(len).collect()
        }

        // Helper: Generate ASCII lowercase string
        fn make_lowercase_string(len: usize) -> String {
            (0..len)
                .map(|i| ((b'a' + (i % 26) as u8) as char))
                .collect()
        }

        // Helper: Generate ASCII uppercase string
        fn make_uppercase_string(len: usize) -> String {
            (0..len)
                .map(|i| ((b'A' + (i % 26) as u8) as char))
                .collect()
        }

        // -----------------------------------------------------------------
        // Equality Tests (SIMD str_eq integration)
        // -----------------------------------------------------------------

        #[test]
        fn test_simd_eq_empty() {
            let s1 = StringObject::new("");
            let s2 = StringObject::new("");
            assert_eq!(s1, s2);
        }

        #[test]
        fn test_simd_eq_small() {
            for len in 1..32 {
                let data = make_pattern_string(len, 'x');
                let s1 = StringObject::new(&data);
                let s2 = StringObject::new(&data);
                assert_eq!(s1, s2, "len={}", len);
            }
        }

        #[test]
        fn test_simd_eq_medium() {
            for len in [32, 64, 100, 128, 200] {
                let data = make_pattern_string(len, 'y');
                let s1 = StringObject::new(&data);
                let s2 = StringObject::new(&data);
                assert_eq!(s1, s2, "len={}", len);
            }
        }

        #[test]
        fn test_simd_eq_large() {
            for len in [256, 512, 1024, 4096, 10000] {
                let data = make_pattern_string(len, 'z');
                let s1 = StringObject::new(&data);
                let s2 = StringObject::new(&data);
                assert_eq!(s1, s2, "len={}", len);
            }
        }

        #[test]
        fn test_simd_neq_at_start() {
            let s1 = StringObject::new("hello world");
            let s2 = StringObject::new("xello world");
            assert_ne!(s1, s2);
        }

        #[test]
        fn test_simd_neq_at_end() {
            let s1 = StringObject::new("hello world");
            let s2 = StringObject::new("hello worlx");
            assert_ne!(s1, s2);
        }

        #[test]
        fn test_simd_neq_at_middle() {
            let data1: String = (0..100).map(|_| 'a').collect();
            let mut data2 = data1.clone();
            data2.replace_range(50..51, "x");
            let s1 = StringObject::new(&data1);
            let s2 = StringObject::new(&data2);
            assert_ne!(s1, s2);
        }

        #[test]
        fn test_simd_eq_unicode() {
            let s1 = StringObject::new("héllo wörld 日本語");
            let s2 = StringObject::new("héllo wörld 日本語");
            assert_eq!(s1, s2);
        }

        // -----------------------------------------------------------------
        // Contains/Find Tests (SIMD search integration)
        // -----------------------------------------------------------------

        #[test]
        fn test_simd_contains_single_byte() {
            let s = StringObject::new("hello world");
            assert!(s.contains("o"));
            assert!(!s.contains("x"));
        }

        #[test]
        fn test_simd_contains_short_pattern() {
            let s = StringObject::new("the quick brown fox jumps over the lazy dog");
            assert!(s.contains("quick"));
            assert!(s.contains("fox"));
            assert!(s.contains("lazy"));
            assert!(!s.contains("cat"));
        }

        #[test]
        fn test_simd_contains_large_haystack() {
            let haystack = make_pattern_string(10000, 'x');
            let mut haystack_with_needle = haystack.clone();
            haystack_with_needle.push_str("needle");

            let s1 = StringObject::new(&haystack);
            let s2 = StringObject::new(&haystack_with_needle);

            assert!(!s1.contains("needle"));
            assert!(s2.contains("needle"));
        }

        #[test]
        fn test_simd_find_basic() {
            let s = StringObject::new("hello world");
            assert_eq!(s.find("world"), Some(6));
            assert_eq!(s.find("xyz"), None);
        }

        #[test]
        fn test_simd_find_at_start() {
            let s = StringObject::new("hello world");
            assert_eq!(s.find("hello"), Some(0));
        }

        #[test]
        fn test_simd_find_at_end() {
            let s = StringObject::new("hello world");
            assert_eq!(s.find("world"), Some(6));
        }

        #[test]
        fn test_simd_find_empty_pattern() {
            let s = StringObject::new("hello");
            assert_eq!(s.find(""), Some(0));
        }

        #[test]
        fn test_simd_find_overlapping() {
            let s = StringObject::new("aaaa");
            assert_eq!(s.find("aa"), Some(0));
        }

        // -----------------------------------------------------------------
        // Case Conversion Tests (SIMD case integration)
        // -----------------------------------------------------------------

        #[test]
        fn test_simd_lower_ascii() {
            let s = StringObject::new("HELLO WORLD");
            assert_eq!(s.lower().as_str(), "hello world");
        }

        #[test]
        fn test_simd_lower_mixed() {
            let s = StringObject::new("HeLLo WoRLd");
            assert_eq!(s.lower().as_str(), "hello world");
        }

        #[test]
        fn test_simd_lower_with_numbers() {
            let s = StringObject::new("HELLO123WORLD");
            assert_eq!(s.lower().as_str(), "hello123world");
        }

        #[test]
        fn test_simd_lower_large() {
            let upper = make_uppercase_string(10000);
            let expected_lower = make_lowercase_string(10000);
            let s = StringObject::new(&upper);
            assert_eq!(s.lower().as_str(), expected_lower);
        }

        #[test]
        fn test_simd_lower_unicode_fallback() {
            // Non-ASCII should fall back to std's Unicode-aware lowercase
            let s = StringObject::new("HÉLLO");
            // This goes through the Unicode fallback, which handles é properly
            let lower = s.lower();
            // Note: é is already lowercase in the input (it's the uppercase that differs)
            assert!(lower.as_str().starts_with("h"));
        }

        #[test]
        fn test_simd_upper_ascii() {
            let s = StringObject::new("hello world");
            assert_eq!(s.upper().as_str(), "HELLO WORLD");
        }

        #[test]
        fn test_simd_upper_mixed() {
            let s = StringObject::new("HeLLo WoRLd");
            assert_eq!(s.upper().as_str(), "HELLO WORLD");
        }

        #[test]
        fn test_simd_upper_large() {
            let lower = make_lowercase_string(10000);
            let expected_upper = make_uppercase_string(10000);
            let s = StringObject::new(&lower);
            assert_eq!(s.upper().as_str(), expected_upper);
        }

        // -----------------------------------------------------------------
        // Whitespace Trimming Tests (SIMD whitespace integration)
        // -----------------------------------------------------------------

        #[test]
        fn test_simd_strip_basic() {
            let s = StringObject::new("  hello world  ");
            assert_eq!(s.strip().as_str(), "hello world");
        }

        #[test]
        fn test_simd_strip_tabs_newlines() {
            let s = StringObject::new("\t\nhello\r\n\t");
            assert_eq!(s.strip().as_str(), "hello");
        }

        #[test]
        fn test_simd_strip_no_whitespace() {
            let s = StringObject::new("hello");
            assert_eq!(s.strip().as_str(), "hello");
        }

        #[test]
        fn test_simd_strip_all_whitespace() {
            let s = StringObject::new("   \t\n   ");
            assert_eq!(s.strip().as_str(), "");
        }

        #[test]
        fn test_simd_strip_large() {
            let mut data = String::new();
            for _ in 0..500 {
                data.push(' ');
            }
            data.push_str("hello");
            for _ in 0..500 {
                data.push(' ');
            }
            let s = StringObject::new(&data);
            assert_eq!(s.strip().as_str(), "hello");
        }

        #[test]
        fn test_simd_lstrip_basic() {
            let s = StringObject::new("  hello  ");
            assert_eq!(s.lstrip().as_str(), "hello  ");
        }

        #[test]
        fn test_simd_rstrip_basic() {
            let s = StringObject::new("  hello  ");
            assert_eq!(s.rstrip().as_str(), "  hello");
        }

        // -----------------------------------------------------------------
        // Character Count Tests (SIMD utf8_char_count integration)
        // -----------------------------------------------------------------

        #[test]
        fn test_simd_char_count_ascii() {
            let s = StringObject::new("hello world");
            assert_eq!(s.char_count(), 11);
        }

        #[test]
        fn test_simd_char_count_unicode() {
            let s = StringObject::new("héllo wörld");
            // Each non-ASCII takes 2 bytes but is 1 character
            assert_eq!(s.char_count(), 11);
        }

        #[test]
        fn test_simd_char_count_cjk() {
            let s = StringObject::new("日本語");
            assert_eq!(s.char_count(), 3);
        }

        #[test]
        fn test_simd_char_count_emoji() {
            let s = StringObject::new("🦀🐍🎉");
            assert_eq!(s.char_count(), 3);
        }

        #[test]
        fn test_simd_char_count_large_ascii() {
            let len = 10000;
            let data = make_pattern_string(len, 'a');
            let s = StringObject::new(&data);
            assert_eq!(s.char_count(), len);
        }

        #[test]
        fn test_simd_char_count_mixed() {
            // Mix of ASCII, Latin-1 extended, CJK, and emoji
            let s = StringObject::new("Hello héllo 日本語 🦀");
            // h-e-l-l-o- -h-é-l-l-o- -日-本-語- -🦀 = 17 chars
            assert_eq!(s.char_count(), 17);
        }

        // -----------------------------------------------------------------
        // Boundary and Edge Case Tests
        // -----------------------------------------------------------------

        #[test]
        fn test_simd_operations_at_simd_boundaries() {
            // Test at 16-byte (SSE), 32-byte (AVX2), 64-byte (AVX-512) boundaries
            for len in [15, 16, 17, 31, 32, 33, 63, 64, 65] {
                let data = make_pattern_string(len, 'x');
                let s = StringObject::new(&data);

                // Equality
                let s2 = StringObject::new(&data);
                assert_eq!(s, s2, "equality at len={}", len);

                // Contains
                assert!(s.contains("x"), "contains at len={}", len);

                // Strip (add whitespace around)
                let with_ws = format!("  {}  ", data);
                let s_ws = StringObject::new(&with_ws);
                assert_eq!(s_ws.strip().as_str(), data, "strip at len={}", len);
            }
        }

        #[test]
        fn test_simd_case_preserve_non_letter() {
            // Numbers, punctuation should be preserved
            let s = StringObject::new("HELLO123!@#WORLD");
            assert_eq!(s.lower().as_str(), "hello123!@#world");

            let s2 = StringObject::new("hello456$%^world");
            assert_eq!(s2.upper().as_str(), "HELLO456$%^WORLD");
        }
    }
}
