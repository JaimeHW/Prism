use super::*;

#[test]
fn test_large_object_alloc() {
    let los = LargeObjectSpace::new();

    let ptr = los.alloc(16 * 1024).expect("Alloc failed");
    assert_eq!(los.usage(), 16 * 1024);
    assert_eq!(los.count(), 1);
    assert!(los.contains(ptr.as_ptr() as *const ()));
}

#[test]
fn test_large_object_sweep() {
    let los = LargeObjectSpace::new();

    let ptr1 = los.alloc(1024).expect("Alloc 1 failed");
    let ptr2 = los.alloc(2048).expect("Alloc 2 failed");

    // Mark only ptr1
    los.mark(ptr1.as_ptr() as *const ());

    // Sweep should free ptr2
    let (bytes_freed, objects_freed) = los.sweep();

    assert_eq!(bytes_freed, 2048);
    assert_eq!(objects_freed, 1);
    assert_eq!(los.count(), 1);
    assert!(los.contains(ptr1.as_ptr() as *const ()));
    assert!(!los.contains(ptr2.as_ptr() as *const ()));
}

#[test]
fn test_large_object_mark_accepts_interior_pointer() {
    let los = LargeObjectSpace::new();

    let ptr = los.alloc(1024).expect("Alloc failed");
    let interior = unsafe { ptr.as_ptr().add(32) } as *const ();

    assert!(los.mark(interior));
    let (bytes_freed, objects_freed) = los.sweep();

    assert_eq!(bytes_freed, 0);
    assert_eq!(objects_freed, 0);
    assert_eq!(los.count(), 1);
    assert!(los.contains(ptr.as_ptr() as *const ()));
}

#[test]
fn test_clear_and_sweep_all() {
    let los = LargeObjectSpace::new();

    let ptr1 = los.alloc(1024).expect("Alloc 1 failed");
    let ptr2 = los.alloc(2048).expect("Alloc 2 failed");

    // Mark both (collect pointers first to avoid deadlock)
    los.mark(ptr1.as_ptr() as *const ());
    los.mark(ptr2.as_ptr() as *const ());

    // Sweep should free nothing
    let (bytes_freed, _) = los.sweep();
    assert_eq!(bytes_freed, 0);
    assert_eq!(los.count(), 2);

    // Clear marks and sweep all
    los.clear_marks();
    let (bytes_freed, objects_freed) = los.sweep();
    assert_eq!(bytes_freed, 1024 + 2048);
    assert_eq!(objects_freed, 2);
    assert_eq!(los.count(), 0);
}

#[test]
fn test_large_object_layout_preserves_alignment() {
    let los = LargeObjectSpace::new();
    let layout = Layout::from_size_align(17 * 1024, 64).expect("valid layout");

    let ptr = los.alloc_layout(layout).expect("Alloc failed");

    assert_eq!(ptr.as_ptr() as usize % 64, 0);
    assert_eq!(los.size_of(ptr.as_ptr() as *const ()), Some(17 * 1024));
    assert_eq!(
        los.objects
            .lock()
            .get(&(ptr.as_ptr() as usize))
            .map(|obj| obj.layout.align()),
        Some(64)
    );
}

#[test]
fn test_large_object_sweep_uses_original_layout() {
    let los = LargeObjectSpace::new();
    let layout = Layout::from_size_align(17 * 1024, 64).expect("valid layout");
    let ptr = los.alloc_layout(layout).expect("Alloc failed");

    let (bytes_freed, objects_freed) = los.sweep();

    assert_eq!(bytes_freed, 17 * 1024);
    assert_eq!(objects_freed, 1);
    assert_eq!(los.count(), 0);
    assert!(!los.contains(ptr.as_ptr() as *const ()));
}

#[test]
fn test_large_object_drop_uses_original_layout() {
    let layout = Layout::from_size_align(17 * 1024, 64).expect("valid layout");
    let ptr_addr = {
        let los = LargeObjectSpace::new();
        let ptr = los.alloc_layout(layout).expect("Alloc failed");
        assert_eq!(ptr.as_ptr() as usize % 64, 0);
        ptr.as_ptr() as usize
    };

    assert_ne!(ptr_addr, 0);
}
