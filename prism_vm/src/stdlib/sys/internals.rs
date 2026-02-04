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

/// Clear the intern pool (for testing).
#[cfg(test)]
pub fn clear_intern_pool() {
    let mut pool = INTERN_POOL.write().unwrap();
    if let Some(ref mut set) = *pool {
        set.clear();
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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Global lock to serialize tests that use the shared INTERN_POOL.
    /// This prevents race conditions during parallel test execution.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    // =========================================================================
    // getrefcount Tests
    // =========================================================================

    #[test]
    fn test_getrefcount_int() {
        let value = Value::int(42).unwrap();
        let count = getrefcount(&value);
        assert_eq!(count, 1);
    }

    #[test]
    fn test_getrefcount_none() {
        let value = Value::none();
        let count = getrefcount(&value);
        assert_eq!(count, 1);
    }

    // =========================================================================
    // getsizeof Tests
    // =========================================================================

    #[test]
    fn test_getsizeof_int() {
        let value = Value::int(42).unwrap();
        let size = getsizeof(&value);
        assert!(size > 0);
    }

    #[test]
    fn test_getsizeof_none() {
        let value = Value::none();
        let size = getsizeof(&value);
        assert!(size > 0);
    }

    #[test]
    fn test_getsizeof_default_none() {
        let value = Value::none();
        let size = getsizeof_default(&value, 100);
        assert_eq!(size, 100);
    }

    #[test]
    fn test_getsizeof_default_not_none() {
        let value = Value::int(42).unwrap();
        let size = getsizeof_default(&value, 100);
        assert_ne!(size, 100);
    }

    // =========================================================================
    // String Interning Tests
    // =========================================================================

    #[test]
    fn test_intern_new_string() {
        let _guard = TEST_LOCK.lock().unwrap();
        clear_intern_pool();
        let s = intern("test_intern_new".to_string());
        assert_eq!(s, "test_intern_new");
    }

    #[test]
    fn test_intern_returns_same() {
        let _guard = TEST_LOCK.lock().unwrap();
        clear_intern_pool();
        let s1 = intern("shared".to_string());
        let s2 = intern("shared".to_string());
        // Both should be equal
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_is_interned() {
        let _guard = TEST_LOCK.lock().unwrap();
        clear_intern_pool();
        intern("is_interned_test".to_string());
        assert!(is_interned("is_interned_test"));
    }

    #[test]
    fn test_is_not_interned() {
        let _guard = TEST_LOCK.lock().unwrap();
        clear_intern_pool();
        assert!(!is_interned("never_interned_xyz"));
    }

    #[test]
    fn test_intern_count() {
        let _guard = TEST_LOCK.lock().unwrap();
        clear_intern_pool();
        let initial = intern_count();
        intern("count_test_1".to_string());
        intern("count_test_2".to_string());
        assert_eq!(intern_count(), initial + 2);
    }

    #[test]
    fn test_intern_count_no_duplicates() {
        let _guard = TEST_LOCK.lock().unwrap();
        clear_intern_pool();
        intern("no_dup".to_string());
        let count1 = intern_count();
        intern("no_dup".to_string());
        let count2 = intern_count();
        // count should not increase for duplicate
        assert_eq!(count1, count2);
    }

    // =========================================================================
    // AuditEvent Tests
    // =========================================================================

    #[test]
    fn test_audit_event_new() {
        let event = AuditEvent::new("test.event");
        assert_eq!(event.name, "test.event");
        assert!(event.args.is_empty());
    }

    #[test]
    fn test_audit_event_with_args() {
        let event = AuditEvent::with_args("import", vec!["module".to_string()]);
        assert_eq!(event.name, "import");
        assert_eq!(event.args.len(), 1);
    }

    #[test]
    fn test_audit_event_clone() {
        let event = AuditEvent::new("clone.test");
        let cloned = event.clone();
        assert_eq!(event.name, cloned.name);
    }

    // =========================================================================
    // CallDepth Tests
    // =========================================================================

    #[test]
    fn test_call_depth_new() {
        let depth = CallDepth::new();
        assert_eq!(depth.get(), 0);
    }

    #[test]
    fn test_call_depth_default() {
        let depth = CallDepth::default();
        assert_eq!(depth.get(), 0);
    }

    #[test]
    fn test_call_depth_enter() {
        let mut depth = CallDepth::new();
        depth.enter();
        assert_eq!(depth.get(), 1);
        depth.enter();
        assert_eq!(depth.get(), 2);
    }

    #[test]
    fn test_call_depth_leave() {
        let mut depth = CallDepth::new();
        depth.enter();
        depth.enter();
        depth.leave();
        assert_eq!(depth.get(), 1);
    }

    #[test]
    fn test_call_depth_leave_at_zero() {
        let mut depth = CallDepth::new();
        depth.leave(); // Should not underflow
        assert_eq!(depth.get(), 0);
    }

    #[test]
    fn test_call_depth_reset() {
        let mut depth = CallDepth::new();
        depth.enter();
        depth.enter();
        depth.enter();
        depth.reset();
        assert_eq!(depth.get(), 0);
    }

    #[test]
    fn test_call_depth_balanced() {
        let mut depth = CallDepth::new();

        // Simulate call stack
        depth.enter(); // level 1
        depth.enter(); // level 2
        depth.leave(); // back to 1
        depth.enter(); // level 2 again
        depth.enter(); // level 3
        depth.leave(); // level 2
        depth.leave(); // level 1
        depth.leave(); // level 0

        assert_eq!(depth.get(), 0);
    }
}
