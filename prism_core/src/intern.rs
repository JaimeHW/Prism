//! String interning for O(1) string equality and reduced memory usage.
//!
//! This module provides a high-performance string interner that stores unique
//! copies of strings and returns lightweight handles. Interned strings can be
//! compared by pointer equality, making identifier comparison extremely fast.

use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// A handle to an interned string.
///
/// `InternedString` is a thin wrapper around an `Arc<str>` that provides
/// O(1) equality comparison via pointer comparison. Two `InternedString`s
/// are equal if and only if they were interned by the same interner and
/// contain the same string content.
#[derive(Clone)]
pub struct InternedString {
    inner: Arc<str>,
}

impl InternedString {
    /// Create a new interned string (for testing/internal use).
    /// Prefer using `StringInterner::intern` for deduplication.
    #[inline]
    fn new(s: Arc<str>) -> Self {
        Self { inner: s }
    }

    /// Get the string content.
    #[inline]
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.inner
    }

    /// Get the length in bytes.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if the string is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get the pointer address for identity comparison.
    #[inline]
    fn ptr(&self) -> *const u8 {
        self.inner.as_ptr()
    }

    /// Get a clone of the underlying Arc.
    ///
    /// This increments the reference count rather than creating a new allocation,
    /// ensuring pointer stability for NaN-boxing in `Value::string()`.
    #[inline]
    pub fn get_arc(&self) -> Arc<str> {
        self.inner.clone()
    }
}

impl PartialEq for InternedString {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // Pointer comparison for O(1) equality
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl Eq for InternedString {}

impl Hash for InternedString {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash the pointer for consistency with Eq
        self.ptr().hash(state);
    }
}

impl fmt::Debug for InternedString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "InternedString({:?})", self.as_str())
    }
}

impl fmt::Display for InternedString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl AsRef<str> for InternedString {
    #[inline]
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl std::ops::Deref for InternedString {
    type Target = str;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}

impl PartialEq<str> for InternedString {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<&str> for InternedString {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl PartialEq<String> for InternedString {
    fn eq(&self, other: &String) -> bool {
        self.as_str() == other
    }
}

/// Thread-safe string interner.
///
/// The interner maintains a set of unique strings and returns handles to them.
/// Interning the same string multiple times returns the same handle, enabling
/// O(1) equality comparison.
pub struct StringInterner {
    /// Interner state protected by a read-write lock.
    maps: RwLock<InternerMaps>,
}

/// Internal interner maps.
///
/// `by_value` provides the canonical dedup map from string content to handle.
/// `by_ptr` enables O(1) lookup from the leaked data pointer used in `Value::string`.
struct InternerMaps {
    by_value: FxHashMap<Arc<str>, InternedString>,
    by_ptr: FxHashMap<usize, InternedString>,
}

impl InternerMaps {
    #[inline]
    fn new() -> Self {
        Self {
            by_value: FxHashMap::default(),
            by_ptr: FxHashMap::default(),
        }
    }

    #[inline]
    fn with_capacity(capacity: usize) -> Self {
        Self {
            by_value: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
            by_ptr: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
        }
    }
}

impl StringInterner {
    /// Create a new, empty string interner.
    #[must_use]
    pub fn new() -> Self {
        Self {
            maps: RwLock::new(InternerMaps::new()),
        }
    }

    /// Create a new interner with preallocated capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            maps: RwLock::new(InternerMaps::with_capacity(capacity)),
        }
    }

    /// Intern a string, returning a handle.
    ///
    /// If the string has been interned before, the same handle is returned.
    /// This method is thread-safe.
    pub fn intern(&self, s: &str) -> InternedString {
        // Fast path: check if already interned with read lock
        {
            let maps = self.maps.read();
            if let Some(interned) = maps.by_value.get(s) {
                return interned.clone();
            }
        }

        // Slow path: insert with write lock
        let mut maps = self.maps.write();

        // Double-check after acquiring write lock
        if let Some(interned) = maps.by_value.get(s) {
            return interned.clone();
        }

        // Create new interned string
        let arc: Arc<str> = s.into();
        let interned = InternedString::new(arc.clone());
        maps.by_value.insert(arc, interned.clone());
        maps.by_ptr
            .insert(interned.ptr() as usize, interned.clone());
        interned
    }

    /// Intern a string from an owned String.
    ///
    /// This avoids an allocation if the string is not already interned.
    pub fn intern_owned(&self, s: String) -> InternedString {
        // Fast path: check if already interned with read lock
        {
            let maps = self.maps.read();
            if let Some(interned) = maps.by_value.get(s.as_str()) {
                return interned.clone();
            }
        }

        // Slow path: insert with write lock
        let mut maps = self.maps.write();

        // Double-check after acquiring write lock
        if let Some(interned) = maps.by_value.get(s.as_str()) {
            return interned.clone();
        }

        // Create new interned string from owned String
        let arc: Arc<str> = s.into();
        let interned = InternedString::new(arc.clone());
        maps.by_value.insert(arc, interned.clone());
        maps.by_ptr
            .insert(interned.ptr() as usize, interned.clone());
        interned
    }

    /// Get an already-interned string without creating a new one.
    ///
    /// Returns `None` if the string has not been interned.
    #[must_use]
    pub fn get(&self, s: &str) -> Option<InternedString> {
        self.maps.read().by_value.get(s).cloned()
    }

    /// Get an interned string by its data pointer.
    ///
    /// This is used by NaN-boxed `Value::string` payload decoding.
    #[must_use]
    pub fn get_by_ptr(&self, ptr: *const u8) -> Option<InternedString> {
        self.maps.read().by_ptr.get(&(ptr as usize)).cloned()
    }

    /// Get interned string byte length by data pointer.
    ///
    /// Returns `None` when the pointer is not present in the interner.
    #[must_use]
    pub fn len_by_ptr(&self, ptr: *const u8) -> Option<usize> {
        self.maps
            .read()
            .by_ptr
            .get(&(ptr as usize))
            .map(InternedString::len)
    }

    /// Check if a string has been interned.
    #[must_use]
    pub fn contains(&self, s: &str) -> bool {
        self.maps.read().by_value.contains_key(s)
    }

    /// Get the number of interned strings.
    #[must_use]
    pub fn len(&self) -> usize {
        self.maps.read().by_value.len()
    }

    /// Check if the interner is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.maps.read().by_value.is_empty()
    }

    /// Clear all interned strings.
    ///
    /// Existing `InternedString` handles remain valid but will no longer
    /// be deduplicated with newly interned strings.
    pub fn clear(&self) {
        let mut maps = self.maps.write();
        maps.by_value.clear();
        maps.by_ptr.clear();
    }
}

impl Default for StringInterner {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for StringInterner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let maps = self.maps.read();
        f.debug_struct("StringInterner")
            .field("count", &maps.by_value.len())
            .finish()
    }
}

/// A global string interner for common identifiers.
///
/// This is useful for keywords and common identifiers that appear frequently.
pub static GLOBAL_INTERNER: std::sync::LazyLock<StringInterner> =
    std::sync::LazyLock::new(StringInterner::new);

/// Intern a string using the global interner.
#[inline]
pub fn intern(s: &str) -> InternedString {
    GLOBAL_INTERNER.intern(s)
}

/// Intern an owned string using the global interner.
#[inline]
pub fn intern_owned(s: String) -> InternedString {
    GLOBAL_INTERNER.intern_owned(s)
}

/// Resolve an interned string from its data pointer.
///
/// This supports NaN-boxed string payload decoding in the VM.
#[inline]
pub fn interned_by_ptr(ptr: *const u8) -> Option<InternedString> {
    GLOBAL_INTERNER.get_by_ptr(ptr)
}

/// Resolve interned string length from its data pointer.
#[inline]
pub fn interned_len_by_ptr(ptr: *const u8) -> Option<usize> {
    GLOBAL_INTERNER.len_by_ptr(ptr)
}
