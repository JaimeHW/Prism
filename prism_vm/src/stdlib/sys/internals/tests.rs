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

// =========================================================================
// CallDepth Tests
// =========================================================================

#[test]
fn test_call_depth_new() {
    let depth = CallDepth::new();
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
