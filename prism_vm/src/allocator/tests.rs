use super::*;
use prism_gc::config::GcConfig;
use prism_gc::trace::Tracer;

/// Simple traceable test object.
#[derive(Debug, Clone, PartialEq)]
struct TestObject {
    value: i64,
    data: [u8; 32],
}

impl TestObject {
    fn new(value: i64) -> Self {
        Self {
            value,
            data: [0; 32],
        }
    }
}

unsafe impl Trace for TestObject {
    fn trace(&self, _tracer: &mut dyn Tracer) {}
}

/// Object with nested references for trace testing.
#[derive(Debug)]
struct NestedObject {
    values: Vec<Value>,
}

unsafe impl Trace for NestedObject {
    fn trace(&self, tracer: &mut dyn Tracer) {
        for v in &self.values {
            tracer.trace_value(*v);
        }
    }
}

// -------------------------------------------------------------------------
// Basic Allocation Tests
// -------------------------------------------------------------------------

#[test]
fn test_alloc_simple() {
    let heap = GcHeap::new(GcConfig::default());
    let allocator = GcAllocator::new(&heap);

    let ptr = allocator.alloc(TestObject::new(42));
    assert!(ptr.is_some());

    let obj = unsafe { &*ptr.unwrap() };
    assert_eq!(obj.value, 42);
}

#[test]
fn test_alloc_multiple() {
    let heap = GcHeap::new(GcConfig::default());
    let allocator = GcAllocator::new(&heap);

    let ptrs: Vec<_> = (0..100)
        .map(|i| allocator.alloc(TestObject::new(i)))
        .collect();

    // All allocations should succeed
    assert!(ptrs.iter().all(|p| p.is_some()));

    // Verify each object has correct value
    for (i, ptr) in ptrs.into_iter().enumerate() {
        let obj = unsafe { &*ptr.unwrap() };
        assert_eq!(obj.value, i as i64);
    }
}

#[test]
fn test_alloc_different_types() {
    let heap = GcHeap::new(GcConfig::default());
    let allocator = GcAllocator::new(&heap);

    // Allocate different sized objects
    let small = allocator.alloc(42i64);
    let medium = allocator.alloc(TestObject::new(100));
    let large = allocator.alloc(NestedObject { values: vec![] });

    assert!(small.is_some());
    assert!(medium.is_some());
    assert!(large.is_some());
}

// -------------------------------------------------------------------------
// alloc_value Tests
// -------------------------------------------------------------------------

#[test]
fn test_alloc_value() {
    let heap = GcHeap::new(GcConfig::default());
    let allocator = GcAllocator::new(&heap);

    let val = allocator.alloc_value(TestObject::new(42));
    assert!(val.is_some());

    let val = val.unwrap();
    assert!(val.as_object_ptr().is_some());
}

#[test]
fn test_alloc_value_roundtrip() {
    let heap = GcHeap::new(GcConfig::default());
    let allocator = GcAllocator::new(&heap);

    let val = allocator.alloc_value(TestObject::new(999)).unwrap();
    let ptr = val.as_object_ptr().unwrap() as *const TestObject;
    let obj = unsafe { &*ptr };

    assert_eq!(obj.value, 999);
}

// -------------------------------------------------------------------------
// try_alloc Tests
// -------------------------------------------------------------------------

#[test]
fn test_try_alloc_success() {
    let heap = GcHeap::new(GcConfig::default());
    let allocator = GcAllocator::new(&heap);

    let result = allocator.try_alloc(TestObject::new(42));
    assert!(result.is_ok());
}

#[test]
fn test_try_alloc_result_map() {
    let heap = GcHeap::new(GcConfig::default());
    let allocator = GcAllocator::new(&heap);

    let result = allocator
        .try_alloc(TestObject::new(42))
        .map(|ptr| unsafe { (*ptr).value });

    assert_eq!(result.ok(), Some(42));
}

// -------------------------------------------------------------------------
// alloc_tenured Tests
// -------------------------------------------------------------------------

#[test]
fn test_alloc_tenured() {
    let heap = GcHeap::new(GcConfig::default());
    let allocator = GcAllocator::new(&heap);

    let ptr = allocator.alloc_tenured(TestObject::new(42));
    assert!(ptr.is_some());

    let obj = unsafe { &*ptr.unwrap() };
    assert_eq!(obj.value, 42);
}

#[test]
fn test_alloc_tenured_is_in_old_gen() {
    let heap = GcHeap::new(GcConfig::default());
    let allocator = GcAllocator::new(&heap);

    let ptr = allocator.alloc_tenured(TestObject::new(42)).unwrap();
    // Tenured allocations go to old space
    assert!(heap.is_old(ptr as *const ()));
}

#[test]
fn test_alloc_tenured_preserves_type_alignment() {
    #[repr(align(64))]
    struct AlignedObject {
        value: u64,
    }

    unsafe impl Trace for AlignedObject {
        fn trace(&self, _tracer: &mut dyn Tracer) {}
    }

    let heap = GcHeap::new(GcConfig::default());
    let allocator = GcAllocator::new(&heap);

    let ptr = allocator
        .alloc_tenured(AlignedObject { value: 42 })
        .unwrap();

    assert_eq!(ptr as usize % 64, 0);
    assert_eq!(unsafe { (*ptr).value }, 42);
    assert!(heap.is_old(ptr as *const ()));
}

// -------------------------------------------------------------------------
// alloc_raw Tests
// -------------------------------------------------------------------------

#[test]
fn test_alloc_raw() {
    let heap = GcHeap::new(GcConfig::default());
    let allocator = GcAllocator::new(&heap);

    let ptr = allocator.alloc_raw(64);
    assert!(ptr.is_some());
}

#[test]
fn test_alloc_raw_various_sizes() {
    let heap = GcHeap::new(GcConfig::default());
    let allocator = GcAllocator::new(&heap);

    for size in [8, 16, 32, 64, 128, 256, 512, 1024] {
        let ptr = allocator.alloc_raw(size);
        assert!(ptr.is_some(), "Failed to allocate {} bytes", size);
    }
}

// -------------------------------------------------------------------------
// can_alloc Tests
// -------------------------------------------------------------------------

#[test]
fn test_can_alloc_initially_true() {
    let heap = GcHeap::new(GcConfig::default());
    let allocator = GcAllocator::new(&heap);

    assert!(allocator.can_alloc(64));
}

// -------------------------------------------------------------------------
// AllocResult Tests
// -------------------------------------------------------------------------

#[test]
fn test_alloc_result_ok() {
    let result: AllocResult<i32> = AllocResult::Ok(42);
    assert!(result.is_ok());
    assert!(!result.needs_collection());
    assert_eq!(result.ok(), Some(42));
}

#[test]
fn test_alloc_result_needs_collection() {
    let result: AllocResult<i32> = AllocResult::NeedsCollection;
    assert!(!result.is_ok());
    assert!(result.needs_collection());
    assert_eq!(result.ok(), None);
}

#[test]
fn test_alloc_result_oom() {
    let result: AllocResult<i32> = AllocResult::OutOfMemory;
    assert!(!result.is_ok());
    assert!(!result.needs_collection());
    assert_eq!(result.ok(), None);
}

#[test]
fn test_alloc_result_map_ok() {
    let result: AllocResult<i32> = AllocResult::Ok(42);
    let mapped = result.map(|x| x * 2);
    assert_eq!(mapped.ok(), Some(84));
}

#[test]
fn test_alloc_result_map_error() {
    let result: AllocResult<i32> = AllocResult::NeedsCollection;
    let mapped = result.map(|x| x * 2);
    assert!(mapped.needs_collection());
}

#[test]
fn test_alloc_result_unwrap() {
    let result: AllocResult<i32> = AllocResult::Ok(42);
    assert_eq!(result.unwrap(), 42);
}

#[test]
#[should_panic(expected = "needs collection")]
fn test_alloc_result_unwrap_panic_collection() {
    let result: AllocResult<i32> = AllocResult::NeedsCollection;
    let _ = result.unwrap();
}

#[test]
#[should_panic(expected = "out of memory")]
fn test_alloc_result_unwrap_panic_oom() {
    let result: AllocResult<i32> = AllocResult::OutOfMemory;
    let _ = result.unwrap();
}

// -------------------------------------------------------------------------
// Memory Layout Tests
// -------------------------------------------------------------------------

#[test]
fn test_allocation_alignment() {
    let heap = GcHeap::new(GcConfig::default());
    let allocator = GcAllocator::new(&heap);

    // All allocations should be 8-byte aligned
    for _ in 0..100 {
        let ptr = allocator.alloc(TestObject::new(0)).unwrap();
        assert_eq!(ptr as usize % 8, 0, "Pointer not 8-byte aligned");
    }
}

#[test]
fn test_allocations_non_overlapping() {
    let heap = GcHeap::new(GcConfig::default());
    let allocator = GcAllocator::new(&heap);

    let mut ptrs: Vec<*mut TestObject> = Vec::new();

    for i in 0..50 {
        let ptr = allocator.alloc(TestObject::new(i)).unwrap();
        ptrs.push(ptr);
    }

    // Check no two allocations overlap
    let size = std::mem::size_of::<TestObject>();
    for i in 0..ptrs.len() {
        for j in (i + 1)..ptrs.len() {
            let start_i = ptrs[i] as usize;
            let end_i = start_i + size;
            let start_j = ptrs[j] as usize;
            let end_j = start_j + size;

            assert!(
                end_i <= start_j || end_j <= start_i,
                "Allocations {} and {} overlap",
                i,
                j
            );
        }
    }
}

// -------------------------------------------------------------------------
// Stats Tests
// -------------------------------------------------------------------------

#[test]
fn test_allocator_stats() {
    let heap = GcHeap::new(GcConfig::default());
    let allocator = GcAllocator::new(&heap);

    let initial = allocator
        .stats()
        .objects_allocated
        .load(std::sync::atomic::Ordering::Relaxed);

    allocator.alloc(TestObject::new(0));

    let after = allocator
        .stats()
        .objects_allocated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(after > initial, "Object allocation count should increase");
}

// -------------------------------------------------------------------------
// Edge Case Tests
// -------------------------------------------------------------------------

#[test]
fn test_alloc_zero_sized_type() {
    let heap = GcHeap::new(GcConfig::default());
    let allocator = GcAllocator::new(&heap);

    // Zero-sized type
    #[derive(Debug)]
    struct Empty;

    unsafe impl Trace for Empty {
        fn trace(&self, _tracer: &mut dyn Tracer) {}
    }

    // Should still allocate minimum size
    let ptr = allocator.alloc(Empty);
    assert!(ptr.is_some());
}

#[test]
fn test_alloc_with_values() {
    let heap = GcHeap::new(GcConfig::default());
    let allocator = GcAllocator::new(&heap);

    // Allocate object with Value references
    let obj = NestedObject {
        values: vec![
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ],
    };

    let ptr = allocator.alloc(obj).unwrap();
    let allocated = unsafe { &*ptr };

    assert_eq!(allocated.values.len(), 3);
}
