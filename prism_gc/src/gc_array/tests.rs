use super::*;

// Helper to create a GcArray for testing
fn alloc_test_array<T>(capacity: usize) -> (*mut GcArray<T>, Layout) {
    let (layout, _) = GcArray::<T>::layout_for_capacity(capacity);
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) as *mut GcArray<T> };
    unsafe {
        GcArray::init_at(NonNull::new_unchecked(ptr), capacity);
    }
    (ptr, layout)
}

fn free_test_array<T>(ptr: *mut GcArray<T>, layout: Layout) {
    unsafe {
        // Drop the array properly
        std::ptr::drop_in_place(ptr);
        std::alloc::dealloc(ptr as *mut u8, layout);
    }
}

#[test]
fn test_layout_calculation() {
    // u64 array
    let (layout, offset) = GcArray::<u64>::layout_for_capacity(10);
    assert!(layout.size() >= std::mem::size_of::<GcArray<u64>>() + 10 * 8);
    assert!(offset >= std::mem::size_of::<GcArray<u64>>());
    assert_eq!(offset % 8, 0); // Properly aligned for u64
}

#[test]
fn test_create_and_destroy() {
    let (ptr, layout) = alloc_test_array::<u64>(10);
    assert!(!ptr.is_null());

    unsafe {
        assert_eq!((*ptr).capacity(), 10);
        assert_eq!((*ptr).len(), 0);
        assert!((*ptr).is_empty());
    }

    free_test_array(ptr, layout);
}

#[test]
fn test_push_pop() {
    let (ptr, layout) = alloc_test_array::<u64>(5);

    unsafe {
        let array = &mut *ptr;

        assert!(array.push(10));
        assert!(array.push(20));
        assert!(array.push(30));
        assert_eq!(array.len(), 3);

        assert_eq!(array.pop(), Some(30));
        assert_eq!(array.pop(), Some(20));
        assert_eq!(array.len(), 1);
    }

    free_test_array(ptr, layout);
}

#[test]
fn test_capacity_limit() {
    let (ptr, layout) = alloc_test_array::<u64>(3);

    unsafe {
        let array = &mut *ptr;

        assert!(array.push(1));
        assert!(array.push(2));
        assert!(array.push(3));
        assert!(array.is_full());
        assert!(!array.push(4)); // Should fail
    }

    free_test_array(ptr, layout);
}

#[test]
fn test_get_set() {
    let (ptr, layout) = alloc_test_array::<u64>(5);

    unsafe {
        let array = &mut *ptr;

        array.push(100);
        array.push(200);
        array.push(300);

        assert_eq!(*array.get(0).unwrap(), 100);
        assert_eq!(*array.get(1).unwrap(), 200);
        assert_eq!(*array.get(2).unwrap(), 300);
        assert!(array.get(3).is_none());

        array.set(1, 999);
        assert_eq!(*array.get(1).unwrap(), 999);
    }

    free_test_array(ptr, layout);
}

#[test]
fn test_iteration() {
    let (ptr, layout) = alloc_test_array::<u64>(5);

    unsafe {
        let array = &mut *ptr;

        array.push(1);
        array.push(2);
        array.push(3);

        let sum: u64 = array.iter().sum();
        assert_eq!(sum, 6);
    }

    free_test_array(ptr, layout);
}

#[test]
fn test_slice_access() {
    let (ptr, layout) = alloc_test_array::<u64>(5);

    unsafe {
        let array = &mut *ptr;

        array.push(10);
        array.push(20);
        array.push(30);

        let slice = array.as_slice();
        assert_eq!(slice, &[10, 20, 30]);

        array.as_slice_mut()[1] = 999;
        assert_eq!(array.as_slice(), &[10, 999, 30]);
    }

    free_test_array(ptr, layout);
}

#[test]
fn test_clear() {
    let (ptr, layout) = alloc_test_array::<u64>(5);

    unsafe {
        let array = &mut *ptr;

        array.push(1);
        array.push(2);
        array.push(3);
        assert_eq!(array.len(), 3);

        array.clear();
        assert_eq!(array.len(), 0);
        assert!(array.is_empty());
    }

    free_test_array(ptr, layout);
}

#[test]
fn test_copy_from_slice() {
    let (ptr, layout) = alloc_test_array::<u64>(10);

    unsafe {
        let array = &mut *ptr;

        let source = [1u64, 2, 3, 4, 5];
        let copied = array.copy_from_slice(&source);
        assert_eq!(copied, 5);
        assert_eq!(array.as_slice(), &[1, 2, 3, 4, 5]);
    }

    free_test_array(ptr, layout);
}

#[test]
fn test_extend_from_slice() {
    let (ptr, layout) = alloc_test_array::<String>(5);

    unsafe {
        let array = &mut *ptr;

        let source = ["hello".to_string(), "world".to_string()];
        let extended = array.extend_from_slice(&source);
        assert_eq!(extended, 2);
        assert_eq!(array.len(), 2);
        assert_eq!(array.get(0).unwrap(), "hello");
        assert_eq!(array.get(1).unwrap(), "world");
    }

    free_test_array(ptr, layout);
}

#[test]
fn test_allocation_size() {
    // Check that allocation size is reasonable
    let size = GcArray::<u64>::allocation_size(100);
    assert!(size >= 100 * 8); // At least element storage
    assert!(size < 100 * 8 + 1024); // Not too much overhead
}
