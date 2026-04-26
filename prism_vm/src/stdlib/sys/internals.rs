//! Internal functions for debugging and introspection.
//!
//! Provides low-level access to interpreter internals.

use prism_core::Value;
use std::collections::HashSet;
use std::sync::RwLock;

// =============================================================================
// Reference Counting (Stub for GC-based runtime)
// =============================================================================

/// Get the reference count of an object.
///
/// In a GC-based runtime, this returns 1 for most objects.
/// The value is primarily for debugging.
#[inline]
pub fn getrefcount(_value: &Value) -> usize {
    // In a tracing GC, ref count isn't meaningful
    // Return 1 to indicate "at least one reference"
    1
}

// =============================================================================
// Object Size
// =============================================================================

/// Get the size of an object in bytes.
///
/// Returns the approximate memory usage of the value.
#[inline]
pub fn getsizeof(value: &Value) -> usize {
    // Base size of Value (8 bytes for NaN-boxed)
    let base = std::mem::size_of::<Value>();

    // For now, just return the base size
    // String content size would require accessing the string data
    if value.is_string() {
        // Placeholder - would need to access actual string length
        base + 32 // Estimate
    } else {
        base
    }
}

/// Get the size of an object in bytes with default.
#[inline]
pub fn getsizeof_default(value: &Value, default: usize) -> usize {
    if value.is_none() {
        default
    } else {
        getsizeof(value)
    }
}

// =============================================================================
// String Interning
// =============================================================================

/// Global string intern pool.
static INTERN_POOL: RwLock<Option<HashSet<String>>> = RwLock::new(None);

/// Initialize the intern pool if needed.
fn init_intern_pool() {
    let mut pool = INTERN_POOL.write().unwrap();
    if pool.is_none() {
        *pool = Some(HashSet::new());
    }
}

/// Intern a string.
///
/// Returns the interned string. Identical strings share the same allocation.
pub fn intern(s: String) -> String {
    init_intern_pool();

    let pool = INTERN_POOL.read().unwrap();
    if let Some(ref set) = *pool {
        if let Some(existing) = set.get(&s) {
            return existing.clone();
        }
    }
    drop(pool);

    // Not found, need to insert
    let mut pool = INTERN_POOL.write().unwrap();
    if let Some(ref mut set) = *pool {
        set.insert(s.clone());
    }
    s
}

/// Check if a string is interned.
pub fn is_interned(s: &str) -> bool {
    let pool = INTERN_POOL.read().unwrap();
    if let Some(ref set) = *pool {
        set.contains(s)
    } else {
        false
    }
}

/// Get the number of interned strings.
pub fn intern_count() -> usize {
    let pool = INTERN_POOL.read().unwrap();
    if let Some(ref set) = *pool {
        set.len()
    } else {
        0
    }
}

// =============================================================================
// Audit Hooks
// =============================================================================

/// Audit event for security monitoring.
#[derive(Debug, Clone)]
pub struct AuditEvent {
    /// Event name.
    pub name: String,
    /// Event arguments (simplified).
    pub args: Vec<String>,
}

impl AuditEvent {
    /// Create a new audit event.
    #[inline]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            args: Vec::new(),
        }
    }

    /// Create with arguments.
    #[inline]
    pub fn with_args(name: impl Into<String>, args: Vec<String>) -> Self {
        Self {
            name: name.into(),
            args,
        }
    }
}

// =============================================================================
// Frame Introspection
// =============================================================================

/// Get the current call stack depth.
///
/// This is a simplified implementation that tracks recursion depth.
#[derive(Debug, Default)]
pub struct CallDepth {
    depth: usize,
}

impl CallDepth {
    /// Create with zero depth.
    #[inline]
    pub fn new() -> Self {
        Self { depth: 0 }
    }

    /// Get current depth.
    #[inline]
    pub fn get(&self) -> usize {
        self.depth
    }

    /// Increment depth (entering function).
    #[inline]
    pub fn enter(&mut self) {
        self.depth += 1;
    }

    /// Decrement depth (leaving function).
    #[inline]
    pub fn leave(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }

    /// Reset to zero.
    #[inline]
    pub fn reset(&mut self) {
        self.depth = 0;
    }
}
