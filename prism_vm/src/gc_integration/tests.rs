use super::*;
use prism_gc::NoopObjectTracer;

#[test]
fn test_managed_heap_creation() {
    let heap = ManagedHeap::new(GcConfig::default());
    // Fresh heap should have no roots registered
    assert_eq!(heap.roots().handle_count(), 0);
    assert_eq!(heap.roots().global_count(), 0);
}

#[test]
fn test_managed_heap_defaults() {
    let heap = ManagedHeap::with_defaults();
    let config = heap.config();
    assert!(config.nursery_size > 0);
}

#[test]
fn test_collect_minor_with_tracer() {
    let mut heap = ManagedHeap::with_defaults();
    let result = heap.collect_minor(&NoopObjectTracer);
    assert_eq!(result.bytes_freed, 0);
}

#[test]
fn test_collect_major_with_tracer() {
    let mut heap = ManagedHeap::with_defaults();
    let result = heap.collect_major(&NoopObjectTracer);
    assert_eq!(result.bytes_freed, 0);
}

#[test]
fn test_collect_auto_with_tracer() {
    let mut heap = ManagedHeap::with_defaults();
    let result = heap.collect_auto(&NoopObjectTracer);
    // Empty heap should do minor collection
    assert_eq!(result.bytes_freed, 0);
}

#[test]
fn test_collect_roots_only() {
    let mut heap = ManagedHeap::with_defaults();
    let minor_result = heap.collect_minor_roots_only();
    let major_result = heap.collect_major_roots_only();
    assert_eq!(minor_result.bytes_freed, 0);
    assert_eq!(major_result.bytes_freed, 0);
}

#[test]
fn test_stack_roots() {
    let mut roots = StackRoots::new();
    assert!(roots.is_empty());

    // Adding primitives doesn't add roots
    roots.add(Value::int(42).unwrap());
    roots.add(Value::bool(true));
    roots.add(Value::none());

    // Primitives are not object references
    assert_eq!(roots.len(), 0);
}

#[test]
fn test_stack_roots_clear() {
    let mut roots = StackRoots::new();
    roots.add(Value::none());
    roots.clear();
    assert!(roots.is_empty());
}

#[test]
fn test_safe_point() {
    assert!(is_safe_point());
}

#[test]
fn test_alloc_result() {
    let result: AllocResult<i32> = AllocResult::Ok(42);
    assert!(result.is_ok());
    assert_eq!(result.ok(), Some(42));

    let result: AllocResult<i32> = AllocResult::NeedsCollection;
    assert!(!result.is_ok());
    assert_eq!(result.ok(), None);
}

#[test]
fn test_should_collect_initially_false() {
    let heap = ManagedHeap::with_defaults();
    // Empty heap shouldn't need collection
    assert!(!heap.should_minor_collect());
}

#[test]
fn test_promotion_age() {
    let heap = ManagedHeap::with_defaults();
    assert_eq!(heap.promotion_age(), 2);
}
